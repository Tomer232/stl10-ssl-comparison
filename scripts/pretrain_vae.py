"""Stage 1: pretrain the beta-VAE on STL-10's 100k unlabeled images.

    python scripts/pretrain_vae.py --config configs/vae_base.json

This is the generative half of the two-arm comparison. Lior pretrains the same trunk with
NT-Xent; we pretrain it with the ELBO. Everything else -- architecture, embedding
dimensionality, epochs, batch size, optimizer, learning-rate schedule, seed, normalization
constants -- is held fixed, so the pretraining objective is the only variable.

THE TEST SPLIT IS NOT READ HERE, AND NEITHER IS THE LABELED TRAIN SPLIT.
This script opens `unlabeled_X.bin` and nothing else. Pretraining is entirely
self-supervised; the labels do not exist as far as this file is concerned. Test is touched
exactly once, at the very end, by the final evaluation script.

WHAT TO WATCH IN THE LOG
------------------------
The per-epoch line prints the reconstruction term and the KL term separately, and the KL
is the raw, UNWEIGHTED value in nats (not beta * KL). That is the number that catches the
failure mode this project is most exposed to: posterior collapse. If the KL slides toward
zero, every posterior has become the prior, mu is the same vector for every image, every
pairwise cosine similarity is ~1, and the KNN graph built on those embeddings is noise --
label propagation would then be measuring nothing at all, while the total loss still looks
perfectly healthy because the reconstruction term dominates it. A warning is printed when
the KL drops below COLLAPSE_WARNING_NATS.

NO DATALOADER
-------------
Per CLAUDE.md the 100k uint8 images (2.76 GB) are uploaded to VRAM once and batches are
sliced on-device. With the images already resident, a DataLoader's worker processes would
only add host-to-device copies and CPU-side collation to a step that is otherwise pure
GPU. When there is not enough VRAM (or no GPU at all) the same tensor stays in host RAM
and each batch is copied across as it is drawn; the log says which path was taken.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy
import torch

# Repo root on sys.path so "python scripts/pretrain_vae.py" works from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.augment import LIGHT_AUGMENTATION_POLICY, augment_batch
from src.data import load_split, normalize_batch
from src.optim import build_optimizer, learning_rate_at_step, set_learning_rate
from src.seeding import enable_deterministic, reset_seed, seeded_generator
from src.vae import LATENT_DIM, VariationalAutoencoder, beta_at_epoch, vae_loss


DEFAULT_CONFIG_PATH = config.PROJECT_ROOT / "configs" / "vae_base.json"

# KL below this many nats (summed over the 512 latent dimensions, averaged over the batch)
# means the posterior has effectively become the prior. A healthy run sits in the tens to
# hundreds. One nat is far enough below anything informative that a warning is never a
# false alarm, and the warning is the whole reason the KL is logged separately.
COLLAPSE_WARNING_NATS = 1.0

# Bytes of VRAM to leave free after uploading the image tensor, for activations,
# parameters, optimizer state and allocator fragmentation. A batch of 256 through a
# twelve-block SE-ResNet plus a four-stage decoder at 96x96 is a few GB of activations;
# 10 GB is generous, and being wrong in this direction only costs a fallback to host
# indexing, whereas being wrong the other way is an OOM two hours in.
DEFAULT_VRAM_HEADROOM_GIGABYTES = 10.0

BYTES_PER_GIGABYTE = 1024.0 ** 3


def load_config_file(path):
    """Read a run config, ignoring keys that start with an underscore.

    JSON has no comments, and a config nobody can annotate is a config nobody trusts. The
    convention here is that "_documentation" and friends are prose for the reader and are
    dropped before anything looks at the values.
    """
    if path is None:
        return {}

    if not os.path.isfile(path):
        raise FileNotFoundError("Config file not found: %s" % path)

    with open(path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)

    return {key: value for key, value in stored.items() if not key.startswith("_")}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain the beta-VAE on STL-10's unlabeled split.",
    )

    # Every config-backed flag defaults to None rather than to a value, so that "the user
    # did not pass this" is distinguishable from "the user passed the same value the
    # config happens to hold". Resolution order is: flag > config file > src/config.py.
    # The effective default of --seed is 42, which is what configs/vae_base.json holds and
    # also what config.SEED is if no config file is given.
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="JSON config supplying defaults (default: %s). Pass an empty "
                             "string to use only src/config.py." % DEFAULT_CONFIG_PATH)

    parser.add_argument("--seed", type=int, default=None,
                        help="Master seed (default: 42).")
    parser.add_argument("--beta", type=float, default=None,
                        help="Maximum KL weight after warmup (default: 0.01).")
    parser.add_argument("--beta-warmup-epochs", type=int, default=None,
                        help="Epochs over which beta ramps linearly from 0 (default: 10).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Pretraining epochs (default: 100 -- budget parity with Lior).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Images per optimizer step (default: 256).")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Latent dimensionality (default: 512; must match Lior's "
                             "512-d embedding or the comparison is unfair).")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Linear learning-rate warmup epochs (default: 10).")
    parser.add_argument("--max-learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--data-root", default=str(config.DATA_ROOT),
                        help="Directory holding stl10_binary/.")
    parser.add_argument("--normalization-path",
                        default=str(config.DATA_ROOT / "normalization.json"),
                        help="Channel statistics written by scripts/prepare_data.py.")

    parser.add_argument("--run-name", default=None,
                        help="Prefix of runs/<run_name>_<timestamp>/ (default: vae_base).")
    parser.add_argument("--output-dir", default=None,
                        help="Write here instead of runs/<run_name>_<timestamp>/. The "
                             "directory must not already exist.")

    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N unlabeled images. For smoke tests "
                             "only -- a run with --limit set is not a real result and is "
                             "flagged as such in the checkpoint.")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, cuda:0 ... (default: cuda when available).")
    parser.add_argument("--vram-headroom-gigabytes", type=float, default=None,
                        help="VRAM to leave free after uploading the images; below this "
                             "the script falls back to host-side batch indexing.")

    return parser.parse_args()


def resolve_settings(arguments, file_config):
    """Merge command line, config file and src/config.py into one settings dict.

    The resolved dict is written into the run directory and into every checkpoint, so a
    run can always be reproduced from its own artifacts rather than from shell history.
    """

    def pick(name, file_key, fallback):
        value = getattr(arguments, name)
        if value is not None:
            return value, "command-line"
        if file_key in file_config:
            return file_config[file_key], "config-file"
        return fallback, "src/config.py"

    settings = {}
    sources = {}

    for name, file_key, fallback in (
        ("seed", "seed", config.SEED),
        ("beta", "beta", config.BETA_MAX),
        ("beta_warmup_epochs", "beta_warmup_epochs", config.BETA_WARMUP_EPOCHS),
        ("epochs", "epochs", config.EPOCHS),
        ("batch_size", "batch_size", config.BATCH_SIZE),
        ("latent_dim", "latent_dim", LATENT_DIM),
        ("warmup_epochs", "warmup_epochs", config.WARMUP_EPOCHS),
        ("max_learning_rate", "max_learning_rate", config.MAX_LEARNING_RATE),
        ("min_learning_rate", "min_learning_rate", config.MIN_LEARNING_RATE),
        ("weight_decay", "weight_decay", config.WEIGHT_DECAY),
        ("run_name", "run_name", "vae_base"),
        ("vram_headroom_gigabytes", "vram_headroom_gigabytes",
         DEFAULT_VRAM_HEADROOM_GIGABYTES),
    ):
        settings[name], sources[name] = pick(name, file_key, fallback)

    # Not config-file backed: these describe *this invocation*, not the experiment.
    settings["limit"] = arguments.limit
    settings["data_root"] = arguments.data_root
    settings["normalization_path"] = arguments.normalization_path
    settings["config_path"] = arguments.config or None

    return settings, sources


def resolve_device(requested):
    """Pick the device, and fail loudly rather than silently training on CPU.

    A 100-epoch pretrain that quietly fell back to CPU would take weeks and nobody would
    notice until the log was checked. So an explicit --device cuda with no CUDA present is
    an error, while an omitted --device is allowed to fall back (that is the laptop test
    path).
    """
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device %s was requested but torch.cuda.is_available() is False. "
                "This machine has no usable CUDA GPU; training belongs on the lab "
                "server (see the gpu-run skill)." % requested)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_normalization(path):
    """Read the constants prepare_data.py computed from the unlabeled split.

    Deliberately not recomputed here. Recomputing would take minutes at the start of every
    run and, worse, would make it possible for two runs to normalize differently -- at
    which point the embeddings from those runs are not in the same space and the beta
    ablation is comparing apples to oranges.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Normalization constants not found at %s. Run scripts/prepare_data.py first "
            "-- both arms must normalize with identical constants." % path)

    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    return record["mean"], record["standard_deviation"], record


def load_unlabeled_images(data_root, limit):
    """Return the unlabeled split as a contiguous uint8 numpy array [N, 3, 96, 96].

    Memory-mapped first, then materialised, so that --limit reads only the slice it needs
    instead of pulling 2.76 GB off disk for a two-minute smoke test. The
    ascontiguousarray is not optional: data.load_images_memory_mapped applies STL-10's
    column-major fix as a stride swap, and torch.from_numpy rejects the resulting
    non-contiguous view.
    """
    images, _ = load_split(data_root, "unlabeled", memory_map_unlabeled=True)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1, got %r" % (limit,))
        images = images[:limit]

    return numpy.ascontiguousarray(images)


def place_images(images_uint8, device, headroom_gigabytes):
    """Upload the whole uint8 tensor to VRAM if it fits; otherwise keep it in host RAM.

    Returns (tensor, description). The tensor's device decides the batching path for the
    rest of the run: when it is on the GPU a batch is an on-device gather with no PCIe
    traffic at all, which is the plan CLAUDE.md describes. When it stays on the host, each
    batch is copied across as it is drawn -- correct, just slower.
    """
    tensor = torch.from_numpy(images_uint8)
    required_bytes = tensor.numel() * tensor.element_size()
    required_gigabytes = required_bytes / BYTES_PER_GIGABYTE

    if device.type != "cuda":
        return tensor, ("host RAM (%s device); batches are indexed on the CPU and copied "
                        "per step -- %.2f GB resident"
                        % (device.type, required_gigabytes))

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    headroom_bytes = headroom_gigabytes * BYTES_PER_GIGABYTE

    if required_bytes + headroom_bytes <= free_bytes:
        # The single large PCIe transfer this run makes: 2.76 GB up front, and then no
        # host-to-device traffic at all for the rest of the 100 epochs.
        resident = tensor.to(device, non_blocking=False)
        return resident, (
            "VRAM-resident (%.2f GB uploaded, %.2f GB free of %.2f GB before upload); "
            "batches are gathered on-device, no DataLoader"
            % (required_gigabytes, free_bytes / BYTES_PER_GIGABYTE,
               total_bytes / BYTES_PER_GIGABYTE))

    return tensor, (
        "host RAM fallback: %.2f GB of images plus %.2f GB headroom exceeds %.2f GB free "
        "VRAM; batches are copied per step"
        % (required_gigabytes, headroom_gigabytes, free_bytes / BYTES_PER_GIGABYTE))


def resolve_output_directory(arguments, settings, timestamp):
    """Decide where the run writes, and refuse a directory that already holds a run.

    Validates WITHOUT creating. The directory is only made once the preflight (device,
    normalization constants, dataset) has passed, so a failed invocation does not litter
    runs/ with empty dated directories that someone later has to work out the meaning of.

    Never overwrite silently: a checkpoint quietly replaced by a rerun with different
    hyperparameters is the kind of thing that is only discovered while writing the report.
    """
    if arguments.output_dir is not None:
        run_directory = Path(arguments.output_dir)
    else:
        run_directory = config.RUNS_ROOT / ("%s_%s" % (settings["run_name"], timestamp))

    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            "Output directory %s already exists and is not empty. Pick another "
            "--output-dir or --run-name; this script never overwrites a previous run."
            % run_directory)

    return run_directory


def create_output_directory(run_directory):
    """Materialise the run directory once the preflight has passed."""
    config.ensure_directory(run_directory)
    config.ensure_directory(run_directory / "checkpoints")
    return run_directory


def save_checkpoint(path, model, settings, epoch, history, extra):
    """Write a checkpoint atomically (temp file, then rename).

    The rename matters because best.pt is rewritten many times over a multi-hour run; a
    process killed halfway through torch.save would otherwise leave a truncated file where
    the only good checkpoint used to be.

    The optimizer state is deliberately NOT saved. It would roughly triple the file (AdamW
    keeps two moments per parameter) and the lab server has ~13 GB free on a single
    volume, which is the binding constraint on this project. The cost is that a killed run
    restarts from scratch rather than resuming; the run is a few hours, the disk is not
    recoverable.

    The encoder weights are not stored separately either -- they are the `encoder.*` keys
    of `model_state`, and the parity check loads them into SEResNetEncoder with
    strict=True by stripping that prefix.
    """
    payload = {
        "model_state": model.state_dict(),
        "config": settings,
        "epoch": epoch,
        "history": history,
    }
    payload.update(extra)

    temporary_path = str(path) + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


HISTORY_COLUMNS = (
    "epoch",
    "beta",
    "total_loss",
    "reconstruction_loss",
    "kl_loss",
    "learning_rate_first_step",
    "learning_rate_last_step",
    "steps",
    "seconds",
)


def open_history_csv(path):
    """Open the per-epoch history CSV and write its header.

    Written and flushed epoch by epoch rather than dumped at the end, so that a run killed
    at epoch 80 still leaves 80 usable rows -- which is exactly when the KL curve is most
    worth having.
    """
    handle = open(path, "w", encoding="utf-8", newline="")
    handle.write(",".join(HISTORY_COLUMNS) + "\n")
    handle.flush()
    return handle


def append_history_row(handle, row):
    handle.write(",".join(repr(row[column]) if isinstance(row[column], float)
                          else str(row[column]) for column in HISTORY_COLUMNS) + "\n")
    handle.flush()


def train(model, images_tensor, settings, device, mean, standard_deviation,
          run_directory, history_handle):
    """The training loop. Returns the per-epoch history.

    One epoch is `len(images) // batch_size` steps -- the trailing partial batch is
    dropped. That mirrors Lior's `DataLoader(..., drop_last=True)`, so both arms take the
    same number of optimizer steps per epoch and the learning-rate schedule lines up step
    for step. It also keeps every BatchNorm update on a full batch.
    """
    image_count = images_tensor.shape[0]
    steps_per_epoch = image_count // settings["batch_size"]

    if steps_per_epoch < 1:
        raise ValueError(
            "batch size %d exceeds the %d available images, so an epoch would contain "
            "zero steps. Lower --batch-size or raise --limit."
            % (settings["batch_size"], image_count))

    total_steps = settings["epochs"] * steps_per_epoch
    warmup_steps = settings["warmup_epochs"] * steps_per_epoch

    optimizer = build_optimizer(model, weight_decay=settings["weight_decay"])

    # One CPU generator drives both the epoch shuffle and the augmentation draws. CPU so
    # that the stream is identical whether the run is on the laptop or the 5090 -- see
    # augment.sample_uniform, which moves each draw to the batch's device.
    generator = seeded_generator(settings["seed"])

    print("steps/epoch %d, total steps %d, warmup steps %d"
          % (steps_per_epoch, total_steps, warmup_steps), flush=True)
    print("", flush=True)

    history = []
    best_total_loss = float("inf")
    best_epoch = None
    global_step = 0

    for epoch in range(settings["epochs"]):
        epoch_started = time.time()
        model.train()

        beta = beta_at_epoch(epoch, settings["beta"], settings["beta_warmup_epochs"])

        # Shuffled on the CPU with the seeded generator, then moved: torch.randperm with
        # an explicit generator requires the generator and the output device to agree.
        permutation = torch.randperm(image_count, generator=generator)
        permutation = permutation.to(images_tensor.device)

        # Accumulated on-device and read once at the end of the epoch. A .item() per step
        # would synchronise the GPU ~390 times an epoch for numbers nobody looks at until
        # the epoch is over.
        running = torch.zeros(3, dtype=torch.float64, device=device)
        first_learning_rate = None
        last_learning_rate = None

        for step_in_epoch in range(steps_per_epoch):
            start = step_in_epoch * settings["batch_size"]
            batch_indices = permutation[start:start + settings["batch_size"]]
            batch_uint8 = images_tensor[batch_indices]

            # Normalize first, augment second: the crop resamples bilinearly and so needs
            # floats anyway, and normalizing after the crop would compute statistics over
            # resampled pixels instead of the fixed constants both arms share.
            batch = normalize_batch(batch_uint8, mean, standard_deviation, device)

            # The augmented view is ALSO the reconstruction target. Reconstructing the
            # un-augmented original from an augmented input would make this a denoising
            # autoencoder, which is a different objective from the ELBO.
            batch = augment_batch(batch, LIGHT_AUGMENTATION_POLICY, generator)

            # Set before the step, never after: applying a learning rate computed for step
            # n to step n-1 shifts the whole schedule by one and quietly breaks parity
            # with Lior's curve.
            learning_rate = learning_rate_at_step(
                global_step, total_steps, warmup_steps,
                settings["max_learning_rate"], settings["min_learning_rate"])
            set_learning_rate(optimizer, learning_rate)

            if first_learning_rate is None:
                first_learning_rate = learning_rate
            last_learning_rate = learning_rate

            reconstruction, mu, logvar = model(batch)
            total, reconstruction_term, kl_term = vae_loss(
                reconstruction, batch, mu, logvar, beta)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            running += torch.stack([
                total.detach(), reconstruction_term.detach(), kl_term.detach()
            ]).double()

            global_step += 1

        if device.type == "cuda":
            # So the reported seconds are compute time, not queue-submission time.
            torch.cuda.synchronize(device)

        mean_total, mean_reconstruction, mean_kl = (running / steps_per_epoch).tolist()
        elapsed = time.time() - epoch_started

        row = {
            "epoch": epoch,
            "beta": beta,
            "total_loss": mean_total,
            "reconstruction_loss": mean_reconstruction,
            "kl_loss": mean_kl,
            "learning_rate_first_step": first_learning_rate,
            "learning_rate_last_step": last_learning_rate,
            "steps": steps_per_epoch,
            "seconds": elapsed,
        }
        history.append(row)
        append_history_row(history_handle, row)

        print("epoch %3d/%d | loss %12.2f | recon %12.2f | KL %10.3f | beta %.6f | "
              "lr %.3e -> %.3e | %.1fs"
              % (epoch + 1, settings["epochs"], mean_total, mean_reconstruction, mean_kl,
                 beta, first_learning_rate, last_learning_rate, elapsed), flush=True)

        if not numpy.isfinite(mean_total):
            raise RuntimeError(
                "Loss became non-finite at epoch %d. Training is over; the checkpoints "
                "written so far are still on disk." % epoch)

        # The signal this whole script is instrumented for. See the module docstring.
        if beta > 0.0 and mean_kl < COLLAPSE_WARNING_NATS:
            print("    *** POSTERIOR COLLAPSE WARNING: KL = %.4f nats (< %.1f). Every "
                  "posterior is converging to the prior, so mu is losing its dependence "
                  "on the image and the KNN graph built from it will be noise. Lower "
                  "--beta or lengthen --beta-warmup-epochs. ***"
                  % (mean_kl, COLLAPSE_WARNING_NATS), flush=True)

        # Best-loss selection is restricted to epochs at full beta. During the warmup the
        # loss is beta-dependent and shrinks simply because beta is still small, so
        # comparing a warmup epoch against a post-warmup one would systematically pick an
        # early, undertrained model. If the run is shorter than the warmup (a smoke test),
        # every epoch is eligible instead, since there is nothing better to do.
        at_full_beta = epoch >= settings["beta_warmup_epochs"]
        warmup_covers_run = settings["beta_warmup_epochs"] >= settings["epochs"]

        if (at_full_beta or warmup_covers_run) and mean_total < best_total_loss:
            best_total_loss = mean_total
            best_epoch = epoch
            save_checkpoint(
                run_directory / "checkpoints" / "best.pt",
                model, settings, epoch, history,
                extra={
                    "selection": "best total loss among full-beta epochs",
                    "best_total_loss": best_total_loss,
                    "beta_at_epoch": beta,
                },
            )

    return history, best_epoch, best_total_loss


def main():
    arguments = parse_arguments()

    file_config = load_config_file(arguments.config or None)
    settings, sources = resolve_settings(arguments, file_config)

    # First thing, before any CUDA context can exist: enable_deterministic sets
    # CUBLAS_WORKSPACE_CONFIG, which torch only reads at context creation.
    reset_seed(settings["seed"])
    enable_deterministic()

    # Device first, run directory second: an unusable --device should fail before an
    # empty run directory has been created and has to be cleaned up by hand.
    device = resolve_device(arguments.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = resolve_output_directory(arguments, settings, timestamp)

    settings["device"] = str(device)
    settings["timestamp"] = timestamp
    settings["run_directory"] = str(run_directory)
    settings["augmentation_policy"] = dict(LIGHT_AUGMENTATION_POLICY)
    # A --limit run is a smoke test, not a result. Recorded in the config and in every
    # checkpoint so an undersized model can never be mistaken for the real thing.
    settings["is_smoke_test"] = settings["limit"] is not None

    print("pretrain_vae", flush=True)
    print("  run directory: %s" % run_directory, flush=True)
    print("  device:        %s" % device, flush=True)
    if device.type == "cuda":
        print("  gpu:           %s" % torch.cuda.get_device_name(device), flush=True)
    print("  config file:   %s" % (settings["config_path"] or "(none)"), flush=True)
    print("", flush=True)
    print("resolved settings (value <- source):", flush=True)
    for name in sorted(sources):
        print("  %-24s %-12s <- %s" % (name, settings[name], sources[name]), flush=True)
    if settings["is_smoke_test"]:
        print("", flush=True)
        print("  *** SMOKE TEST: --limit %d, only the first %d unlabeled images. This "
              "run's checkpoints are not a result. ***"
              % (settings["limit"], settings["limit"]), flush=True)
    print("", flush=True)

    if settings["latent_dim"] != config.ENCODER_DIM:
        # Not an error -- an ablation may legitimately want it -- but it must be loud.
        print("  *** WARNING: latent_dim %d != ENCODER_DIM %d. Lior's embedding is "
              "512-d, so this run's mu is NOT dimension-matched to his and any accuracy "
              "comparison against the SimCLR arm is confounded. ***"
              % (settings["latent_dim"], config.ENCODER_DIM), flush=True)
        print("", flush=True)

    mean, standard_deviation, normalization_record = load_normalization(
        settings["normalization_path"])
    settings["normalization"] = normalization_record
    print("normalization (from %s, computed on the %s split):"
          % (settings["normalization_path"], normalization_record.get("source_split")),
          flush=True)
    print("  mean = [%.6f, %.6f, %.6f]" % tuple(mean), flush=True)
    print("  std  = [%.6f, %.6f, %.6f]" % tuple(standard_deviation), flush=True)
    print("", flush=True)

    print("loading the unlabeled split (labels are never read -- pretraining is "
          "self-supervised, and the test split is not opened at all)...", flush=True)
    load_started = time.time()
    images_uint8 = load_unlabeled_images(settings["data_root"], settings["limit"])
    print("  %d images, %.2f GB as uint8, %.1f s"
          % (len(images_uint8), images_uint8.nbytes / BYTES_PER_GIGABYTE,
             time.time() - load_started), flush=True)

    images_tensor, placement = place_images(
        images_uint8, device, settings["vram_headroom_gigabytes"])
    print("  placement: %s" % placement, flush=True)
    settings["image_placement"] = placement

    if images_tensor.device.type == "cuda":
        # The host copy is dead weight once it is on the GPU, and 2.76 GB of it.
        del images_uint8
    print("", flush=True)

    model = VariationalAutoencoder(latent_dim=settings["latent_dim"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    encoder_parameter_count = sum(
        parameter.numel() for parameter in model.encoder.parameters())
    settings["parameter_count"] = int(parameter_count)
    settings["encoder_parameter_count"] = int(encoder_parameter_count)
    print("model: %d parameters (%d in the shared SE-ResNet trunk)"
          % (parameter_count, encoder_parameter_count), flush=True)
    print("", flush=True)

    # Created only now, at the end of the preflight: the device, the normalization
    # constants, the dataset and the model have all been resolved without raising, so this
    # run is going to start. Doing it earlier would litter runs/ with empty dated
    # directories every time an invocation failed on a bad flag.
    create_output_directory(run_directory)

    with open(run_directory / "config.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, default=str)
        handle.write("\n")

    history_handle = open_history_csv(run_directory / "history.csv")
    training_started = time.time()

    try:
        history, best_epoch, best_total_loss = train(
            model, images_tensor, settings, device, mean, standard_deviation,
            run_directory, history_handle)
    finally:
        history_handle.close()

    total_seconds = time.time() - training_started

    save_checkpoint(
        run_directory / "checkpoints" / "final.pt",
        model, settings, settings["epochs"] - 1, history,
        extra={
            "selection": "final epoch",
            "total_seconds": total_seconds,
        },
    )

    summary = {
        "run_directory": str(run_directory),
        "settings": settings,
        "total_seconds": total_seconds,
        "best_epoch": best_epoch,
        "best_total_loss": best_total_loss,
        "final_epoch": settings["epochs"] - 1,
        "final_total_loss": history[-1]["total_loss"],
        "final_reconstruction_loss": history[-1]["reconstruction_loss"],
        "final_kl_loss": history[-1]["kl_loss"],
        "posterior_collapsed": bool(
            settings["beta"] > 0.0 and history[-1]["kl_loss"] < COLLAPSE_WARNING_NATS),
    }
    with open(run_directory / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")

    print("", flush=True)
    print("done in %.1f min" % (total_seconds / 60.0), flush=True)
    print("  final  epoch %d: loss %.2f, recon %.2f, KL %.3f"
          % (settings["epochs"] - 1, history[-1]["total_loss"],
             history[-1]["reconstruction_loss"], history[-1]["kl_loss"]), flush=True)
    if best_epoch is not None:
        print("  best   epoch %d: loss %.2f" % (best_epoch, best_total_loss), flush=True)
    print("  checkpoints: %s" % (run_directory / "checkpoints"), flush=True)
    print("  history:     %s" % (run_directory / "history.csv"), flush=True)

    if summary["posterior_collapsed"]:
        print("", flush=True)
        print("  *** THE POSTERIOR COLLAPSED (final KL %.4f nats). mu carries almost no "
              "information about the image; do not build the KNN graph on these "
              "embeddings without saying so in the report. ***"
              % history[-1]["kl_loss"], flush=True)


if __name__ == "__main__":
    main()
