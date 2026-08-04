"""Cache the frozen VAE embeddings that every downstream stage reads.

    python scripts/extract_embeddings.py --checkpoint runs/vae_beta0.01/checkpoints/final.pt

Loads a pretrained VariationalAutoencoder, freezes it, and runs the UNLABELED
(100000) and LABELED TRAIN (5000) splits through the encoder + `to_mu` head to
produce one [105000, 512] float16 array. That array is the input to the KNN
graph, to label propagation, to the linear probe and to the downstream CNN, so
it is computed once here rather than re-encoded by each of them -- 105k images
through a twelve-block SE-ResNet is minutes of GPU time, and re-running it per K
in the sweep would dominate the whole experiment.

THE TEST SPLIT IS NEVER LOADED BY THIS SCRIPT. The label-propagation graph is
built over unlabeled + labeled train only (config.NUM_GRAPH_NODES), and the 8000
test images are touched exactly once at the very end, by the final evaluation
script, in both arms.

THREE THINGS THIS SCRIPT IS CAREFUL ABOUT
-----------------------------------------
1. It extracts `mu`, never a sample from the posterior. See `encode_split`.
2. It applies the DETERMINISTIC transform only -- normalize, no augmentation.
   `src.augment` is deliberately not imported here: a random crop would make the
   cached embedding of an image depend on the RNG state at extraction time, and
   the graph would then differ between runs of the sweep.
3. It writes a manifest recording which checkpoint produced the array, in which
   ORDER the two splits were concatenated, and the exact index offsets of each.
   Everything downstream converts a labeled-train index into a graph node index
   by adding that offset, so the ordering is a load-bearing part of the result
   and belongs in a file rather than in someone's memory.

Runs on CPU or CUDA; the device is a flag and nothing here hardcodes .cuda().
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy
import torch

# Repo root on sys.path so "python scripts/extract_embeddings.py" works from the
# repo root without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src import data
from src.seeding import enable_deterministic, reset_seed
from src.vae import VariationalAutoencoder


# The order is the contract. Unlabeled first, labeled train second, which makes
# the labeled train images the LAST 5000 nodes of the graph and gives them the
# contiguous offset range recorded in the manifest. Changing this order silently
# invalidates every cached embedding file and every result derived from one.
SPLIT_ORDER = ("unlabeled", "train")

# Keys a checkpoint might store the model weights under. Our own pretraining
# script writes one of these; accepting several means this script does not break
# if that convention is adjusted.
STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model_state", "model", "vae_state_dict")

# Sub-dictionaries of the checkpoint that hold run settings rather than weights.
# scripts/pretrain_vae.py saves its resolved settings under "config", and the
# normalization record is nested one level further down inside that ("config" ->
# "normalization" -> "mean"/"standard_deviation"). A flat top-level-only lookup
# therefore finds nothing in our own checkpoints, silently falls through to
# recomputing the statistics, and loses the beta value the ablation groups by.
NESTED_METADATA_KEYS = ("config", "settings", "run_config")

# Keys a checkpoint might store the normalization constants under. Reusing the
# constants the model was TRAINED with is not optional: normalizing with
# different statistics at extraction time feeds the encoder a distribution it
# never saw, and the embeddings would be quietly wrong rather than obviously so.
MEAN_KEYS = ("normalization_mean", "channel_mean", "mean")
STANDARD_DEVIATION_KEYS = (
    "normalization_standard_deviation", "normalization_std", "channel_std",
    "standard_deviation", "std",
)

# Largest finite value float16 can hold. mu is cast to fp16 for the cache, so a
# larger magnitude would become inf on disk and turn into NaN the moment
# src/knn.py normalizes the row -- which would then poison topk for every other
# row in the same chunk. A healthy encoder sits in the single digits.
FLOAT16_MAXIMUM = 65504.0

# Absolute difference above which two sets of channel statistics are treated as
# genuinely different rather than as float formatting noise. They are computed in
# float64 from the same 100k images, so a real match is exact to ~1e-15.
NORMALIZATION_MATCH_TOLERANCE = 1e-9


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract and cache frozen VAE mu embeddings (unlabeled + labeled train).")

    parser.add_argument("--checkpoint", required=True,
                        help="Path to the pretrained VAE checkpoint (.pt).")
    parser.add_argument("--data-root", default=str(config.DATA_ROOT),
                        help="Directory holding stl10_binary/ (or the stl10_binary/ dir itself).")
    parser.add_argument("--run-name", default="embeddings",
                        help="Prefix of the run directory created under runs/.")
    parser.add_argument("--output-directory", default=None,
                        help="Write here instead of runs/<run-name>_<timestamp>/. "
                             "Must not already contain the output files.")
    parser.add_argument("--output-stem", default="embeddings",
                        help="Base name of the .npy and .manifest.json pair.")

    parser.add_argument("--batch-size", type=int, default=config.FEATURE_BATCH_SIZE,
                        help="Images per forward pass (default config.FEATURE_BATCH_SIZE).")
    parser.add_argument("--device", default=None,
                        help="cuda, cuda:0, cpu. Default: cuda when available, else cpu.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Seed for reset_seed. Extraction is deterministic anyway "
                             "(mu is a function of the image), so this only pins the "
                             "environment, but it is logged for provenance.")

    parser.add_argument("--normalization-path",
                        default=str(config.DATA_ROOT / "normalization.json"),
                        help="Channel statistics written by scripts/prepare_data.py. Used "
                             "when the checkpoint carries none, and cross-checked against "
                             "the checkpoint's copy when it does.")
    parser.add_argument("--recompute-normalization", action="store_true",
                        help="Recompute the channel statistics from the unlabeled split even "
                             "if the checkpoint carries them. Use only to reproduce the "
                             "constants; the recomputed values must match the stored ones.")
    parser.add_argument("--no-memory-map", action="store_true",
                        help="Read the 2.76 GB unlabeled split fully into RAM instead of "
                             "memory-mapping it. Faster on the server, painful on a laptop.")
    parser.add_argument("--max-images-per-split", type=int, default=0,
                        help="SMOKE TEST ONLY: cap each split at this many images (0 = all). "
                             "The manifest records the reduced counts, so downstream code "
                             "still gets correct offsets, but the numbers are not results.")

    return parser.parse_args()


def resolve_device(requested):
    """Explicit flag wins; otherwise CUDA when the machine has it.

    An explicit --device cuda on a machine without CUDA is an error rather than a
    silent CPU fallback: 105k images through the SE-ResNet on Tomer's laptop
    would run for hours and produce a file nobody was waiting for. An omitted
    --device is allowed to fall back, since that is the small-subset test path.
    """
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "--device %s was requested but torch.cuda.is_available() is False. "
                "Extraction belongs on the lab server (see the gpu-run skill)." % (requested,))
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_digest(path, chunk_bytes=8 * 1024 * 1024):
    """SHA-256 of a file, read in chunks so a 300 MB checkpoint does not go into RAM.

    Recorded in the manifest because "which checkpoint produced this array" is the
    single question we will be asked when two runs disagree, and a path plus a
    timestamp is not an answer -- checkpoints get overwritten.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_object(path):
    """torch.load the checkpoint onto the CPU, tolerating both weights_only modes.

    torch >= 2.6 defaults to weights_only=True, which refuses to unpickle anything
    beyond tensors and plain containers. Our checkpoints also carry metadata
    (beta, epoch, normalization constants), so the strict load can fail on a
    perfectly good file. We try the safe mode first and fall back loudly, because
    the fallback is only acceptable for a file we produced ourselves.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch has no weights_only argument at all.
        return torch.load(path, map_location="cpu")
    except Exception as error:
        print("  weights_only=True load failed (%s); retrying with weights_only=False. "
              "Only do this for a checkpoint this project produced." % (error,), flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint_object):
    """Pull the model weights out of whatever shape the checkpoint has.

    Returns (state_dict, metadata) where metadata is the rest of the checkpoint
    with the weights removed -- beta, epoch, normalization constants and so on,
    all of which go into the manifest.
    """
    if not isinstance(checkpoint_object, dict):
        raise ValueError("Checkpoint must be a dict or a state dict, got %r"
                         % (type(checkpoint_object),))

    for key in STATE_DICT_KEYS:
        if key in checkpoint_object and isinstance(checkpoint_object[key], dict):
            metadata = {name: value for name, value in checkpoint_object.items() if name != key}
            return checkpoint_object[key], metadata

    # Otherwise assume the file IS the state dict (torch.save(model.state_dict())).
    values_are_tensors = all(isinstance(value, torch.Tensor)
                             for value in checkpoint_object.values())
    if values_are_tensors and len(checkpoint_object) > 0:
        return checkpoint_object, {}

    raise ValueError(
        "Could not find model weights in the checkpoint. Expected one of %s, or a bare "
        "state dict. Top-level keys were: %s"
        % (list(STATE_DICT_KEYS), sorted(checkpoint_object.keys())))


def strip_module_prefix(state_dict):
    """Drop the "module." prefix DataParallel/DDP adds, if it is there.

    A checkpoint saved from a wrapped model will not load into a bare
    VariationalAutoencoder, and the resulting "unexpected key" error is a very
    confusing way to learn that.
    """
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()}


def json_safe(value):
    """Convert torch/numpy scalars in checkpoint metadata to plain Python for json.dump."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() > 1 else float(value.item())
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_metadata_value(metadata, keys):
    """First of `keys` found in the checkpoint metadata, searching one level down.

    Checkpoints from scripts/pretrain_vae.py are
    `{"model_state": ..., "config": settings, "epoch": ..., "history": [...]}`,
    and the interesting values (beta, the normalization record) live inside
    `config` rather than at the top level -- with the normalization constants a
    further level down inside `config["normalization"]`. So a top-level scan is
    tried first, then each settings container, then that container's
    normalization record. Anything else returns None and the caller decides.
    """
    if not isinstance(metadata, dict):
        return None

    for key in keys:
        if metadata.get(key) is not None:
            return metadata[key]

    for container_key in NESTED_METADATA_KEYS:
        container = metadata.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in keys:
            if container.get(key) is not None:
                return container[key]
        record = container.get("normalization")
        if isinstance(record, dict):
            for key in keys:
                if record.get(key) is not None:
                    return record[key]

    return None


def as_float_list(values):
    """Normalize a mean/std to a plain list of Python floats, or None."""
    if values is None:
        return None
    converted = json_safe(values)
    if isinstance(converted, (int, float)):
        return [float(converted)]
    return [float(value) for value in converted]


def statistics_match(first, second):
    """True when two mean/std triples agree to NORMALIZATION_MATCH_TOLERANCE."""
    if first is None or second is None or len(first) != len(second):
        return False
    return all(abs(left - right) <= NORMALIZATION_MATCH_TOLERANCE
               for left, right in zip(first, second))


def load_normalization_file(path):
    """Read data/normalization.json, or return None when it is not there.

    This is the file scripts/prepare_data.py computed once from the unlabeled
    split and that both arms are supposed to share. It is the fallback rather
    than the first choice only because the checkpoint's own copy is the one the
    encoder was actually trained with, and if the two ever disagree that is the
    thing we need to know about.
    """
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_normalization(checkpoint_metadata, data_root, recompute, normalization_path):
    """Get the per-channel mean/std, preferring the ones the model was trained with.

    Per CLAUDE.md and config.NORMALIZATION_SOURCE_SPLIT the statistics come from
    the UNLABELED split only -- using labeled train or test would leak evaluation
    data into preprocessing.

    Priority, and why:
      1. the checkpoint's own copy. Feeding the encoder a distribution it was not
         trained on produces embeddings that are wrong in a way nothing downstream
         can detect;
      2. data/normalization.json, the file prepare_data.py wrote and both arms
         read;
      3. recomputing from the unlabeled split, which is minutes over 2.76 GB and
         is only the default when neither of the above exists.

    When (1) and (2) are both available they are compared, and a mismatch is
    shouted about: it means this checkpoint was trained with constants that are
    no longer the project's constants, and the two arms are then normalizing
    differently, which breaks the "same pixels, same preprocessing" claim the
    whole comparison rests on.

    Returns (mean, standard_deviation, source_string).
    """
    checkpoint_mean = as_float_list(find_metadata_value(checkpoint_metadata, MEAN_KEYS))
    checkpoint_standard_deviation = as_float_list(
        find_metadata_value(checkpoint_metadata, STANDARD_DEVIATION_KEYS))
    has_checkpoint_statistics = (checkpoint_mean is not None
                                 and checkpoint_standard_deviation is not None)

    file_record = load_normalization_file(normalization_path)
    file_mean = as_float_list(file_record.get("mean")) if file_record else None
    file_standard_deviation = (as_float_list(file_record.get("standard_deviation"))
                               if file_record else None)
    has_file_statistics = file_mean is not None and file_standard_deviation is not None

    if has_checkpoint_statistics and has_file_statistics:
        if not (statistics_match(checkpoint_mean, file_mean)
                and statistics_match(checkpoint_standard_deviation, file_standard_deviation)):
            print("  *** WARNING: the checkpoint was trained with normalization constants "
                  "that differ from %s. Using the CHECKPOINT's, because those are what the "
                  "encoder saw -- but the two arms are now preprocessing differently and "
                  "the comparison is confounded until this is resolved. ***"
                  % (normalization_path,), flush=True)
            print("      checkpoint mean %s std %s"
                  % (checkpoint_mean, checkpoint_standard_deviation), flush=True)
            print("      file       mean %s std %s"
                  % (file_mean, file_standard_deviation), flush=True)

    if has_checkpoint_statistics and not recompute:
        return (checkpoint_mean, checkpoint_standard_deviation, "checkpoint")

    if has_file_statistics and not recompute:
        return (file_mean, file_standard_deviation, "file:%s" % (normalization_path,))

    print("Computing channel statistics from the %s split (float64, chunked)..."
          % (config.NORMALIZATION_SOURCE_SPLIT,), flush=True)
    unlabeled_images, _ = data.load_split(
        data_root, config.NORMALIZATION_SOURCE_SPLIT, memory_map_unlabeled=True)
    mean, standard_deviation = data.compute_channel_statistics(
        unlabeled_images, chunk_size=config.NORMALIZATION_CHUNK_IMAGES)

    # A recompute that disagrees with what the model was trained on is exactly the
    # case --recompute-normalization exists to catch, so say so rather than
    # quietly returning the new numbers.
    if has_checkpoint_statistics and not statistics_match(checkpoint_mean, as_float_list(mean)):
        print("  *** WARNING: recomputed mean %s does not match the checkpoint's %s. The "
              "embeddings below are NOT in the space the encoder was trained in. ***"
              % (mean, checkpoint_mean), flush=True)

    return (mean, standard_deviation, "recomputed from the %s split"
            % (config.NORMALIZATION_SOURCE_SPLIT,))


def assert_mu_is_deterministic(model, sample_batch):
    """Encode the same batch twice and require bit-identical mu.

    This is the executable form of the rule in src/vae.py's class docstring:
    downstream code must consume mu, never a sample z = mu + sigma * epsilon. In
    eval mode `reparameterize` already returns mu, and we do not call it at all --
    but "we intended to" is not a guarantee, and a single stray `.train()` or a
    dropout layer added later would break the guarantee without breaking anything
    visible. Two identical encodes is a check that would fail immediately if
    sampling ever crept back in.
    """
    first_mu, _ = model.encode(sample_batch)
    second_mu, _ = model.encode(sample_batch)
    if not torch.equal(first_mu, second_mu):
        raise RuntimeError(
            "Encoding the same batch twice gave different mu. The encoder is not "
            "deterministic -- the model is probably in train mode, which would make "
            "every cached embedding a posterior SAMPLE and the KNN graph different on "
            "every run.")


def encode_split(model, images, mean, standard_deviation, device, batch_size,
                 output_array, output_offset, split_name):
    """Encode one split into `output_array[output_offset:...]` as float16.

    Deliberately NOT applying augmentation. `src.augment.augment_batch` is what
    the pretraining loop uses; here the transform is normalize-only, so the
    embedding of an image is a pure function of its pixels and the checkpoint.

    Returns (largest_absolute_mu, elapsed_seconds).
    """
    number_of_images = len(images)
    largest_absolute_mu = 0.0
    started_at = time.time()
    last_report_at = started_at

    for batch_start in range(0, number_of_images, batch_size):
        batch_end = min(batch_start + batch_size, number_of_images)

        batch = data.normalize_batch(
            images[batch_start:batch_end], mean, standard_deviation, device)

        # ==================================================================
        # mu, AND ONLY mu. `encode` returns (mu, logvar); logvar is discarded
        # and `reparameterize` is never called, so no sample from the
        # posterior can reach the cache. See src/vae.py.
        # ==================================================================
        mu, _ = model.encode(batch)

        if not bool(torch.isfinite(mu).all().item()):
            raise RuntimeError(
                "Non-finite mu in %s batch starting at %d. The checkpoint is diverged; "
                "extracting from it would poison every downstream number."
                % (split_name, batch_start))

        largest_absolute_mu = max(largest_absolute_mu, float(mu.abs().max().item()))

        # The cast below is lossless in range only while |mu| stays inside
        # float16. Past that the value becomes inf on disk, knn.l2_normalize
        # divides inf by inf, and the resulting NaN row poisons the topk of every
        # other row in its similarity chunk -- a failure that shows up as a
        # nonsense graph rather than as an exception.
        if largest_absolute_mu > FLOAT16_MAXIMUM:
            raise RuntimeError(
                "|mu| reached %.1f in %s batch starting at %d, which overflows float16 "
                "(max %.0f). The cache would hold inf. Store fp32 or investigate the "
                "checkpoint -- a healthy encoder produces single-digit mu."
                % (largest_absolute_mu, split_name, batch_start, FLOAT16_MAXIMUM))

        # float16 on disk: 105000 x 512 fp16 is ~107 MB, versus ~215 MB in fp32,
        # and src/knn.py up-casts to fp32 before the similarity search precisely
        # so the halved storage costs no precision in the top-K.
        output_array[output_offset + batch_start:output_offset + batch_end] = (
            mu.to(torch.float16).cpu().numpy())

        now = time.time()
        if now - last_report_at > 20.0 or batch_end == number_of_images:
            elapsed = now - started_at
            rate = batch_end / elapsed if elapsed > 0 else 0.0
            print("  %s %d/%d images  %.0f img/s  %.1fs elapsed"
                  % (split_name, batch_end, number_of_images, rate, elapsed), flush=True)
            last_report_at = now

    return largest_absolute_mu, time.time() - started_at


def resolve_output_paths(arguments, timestamp):
    """Decide where to write, and refuse to overwrite an existing embedding file.

    Validates WITHOUT creating anything, so an invocation that fails its preflight
    (missing checkpoint, no CUDA) does not litter runs/ with empty dated
    directories somebody later has to work out the meaning of.

    Results are expensive; silently clobbering a cached array that a sweep is
    already reading would be worse than an error.
    """
    if arguments.output_directory is not None:
        directory = Path(arguments.output_directory)
    else:
        directory = config.RUNS_ROOT / ("%s_%s" % (arguments.run_name, timestamp))

    embeddings_file = directory / (arguments.output_stem + ".npy")
    manifest_file = directory / (arguments.output_stem + ".manifest.json")
    for path in (embeddings_file, manifest_file):
        if path.exists():
            raise FileExistsError(
                "%s already exists. Refusing to overwrite -- pass a different "
                "--output-directory or --output-stem." % (path,))

    return directory, embeddings_file, manifest_file


def main():
    arguments = parse_arguments()

    # Seeding first, and enable_deterministic before any CUDA context exists --
    # it sets CUBLAS_WORKSPACE_CONFIG, which cuBLAS only reads at init.
    reset_seed(arguments.seed)
    enable_deterministic()

    # TF32 truncates the mantissa to ~10 bits. src/knn.py explains why that
    # matters: it manufactures near-ties in the cosine similarities and the
    # top-K neighbour lists then differ between machines. The embeddings feed
    # that search, so they are computed in full fp32 too.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = resolve_device(arguments.device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    started_at = time.time()

    checkpoint_path = Path(arguments.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("No checkpoint at %s" % (checkpoint_path,))

    # Preflight passed -- only now is anything created on disk.
    output_directory, embeddings_file, manifest_file = resolve_output_paths(
        arguments, timestamp)
    config.ensure_directory(output_directory)

    print("=" * 78, flush=True)
    print("Extract frozen VAE mu embeddings", flush=True)
    print("  torch          %s" % (torch.__version__,), flush=True)
    print("  device         %s" % (device,), flush=True)
    print("  checkpoint     %s" % (arguments.checkpoint,), flush=True)
    print("  data root      %s" % (arguments.data_root,), flush=True)
    print("  output         %s" % (output_directory,), flush=True)
    print("  split order    %s (this ordering is the downstream contract)"
          % (list(SPLIT_ORDER),), flush=True)
    print("  test split     NOT loaded by this script", flush=True)
    print("=" * 78, flush=True)

    print("Hashing checkpoint...", flush=True)
    checkpoint_sha256 = file_digest(checkpoint_path)
    print("  sha256 %s" % (checkpoint_sha256,), flush=True)

    checkpoint_object = load_checkpoint_object(checkpoint_path)
    state_dict, checkpoint_metadata = extract_state_dict(checkpoint_object)
    state_dict = strip_module_prefix(state_dict)

    model = VariationalAutoencoder()
    # strict=True on purpose: a silently partial load would leave randomly
    # initialised layers in the encoder, and the embeddings would look plausible
    # while meaning nothing.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()  # eval mode is what makes reparameterize return mu; we also never call it.
    print("Loaded VariationalAutoencoder (latent_dim=%d) in eval mode."
          % (model.latent_dim,), flush=True)

    mean, standard_deviation, normalization_source = resolve_normalization(
        checkpoint_metadata, arguments.data_root, arguments.recompute_normalization,
        arguments.normalization_path)
    print("Normalization (%s): mean=%s std=%s"
          % (normalization_source, mean, standard_deviation), flush=True)

    # Load both splits before allocating, so the node count in the manifest is
    # the real one rather than the config constant.
    loaded_splits = []
    for split_name in SPLIT_ORDER:
        images, labels = data.load_split(
            arguments.data_root, split_name,
            memory_map_unlabeled=not arguments.no_memory_map)
        if arguments.max_images_per_split > 0:
            images = images[:arguments.max_images_per_split]
            if labels is not None:
                labels = labels[:arguments.max_images_per_split]
        loaded_splits.append((split_name, images, labels))
        print("Loaded split %-10s %d images" % (split_name, len(images)), flush=True)

    number_of_nodes = sum(len(images) for _, images, _ in loaded_splits)
    embedding_dimension = model.latent_dim

    # One preallocated float16 array; each split writes its own contiguous slice.
    embeddings = numpy.empty((number_of_nodes, embedding_dimension), dtype=numpy.float16)
    print("Allocated embedding cache [%d, %d] float16 (%.1f MB)"
          % (number_of_nodes, embedding_dimension, embeddings.nbytes / 1e6), flush=True)

    split_records = []
    largest_absolute_mu = 0.0
    offset = 0

    with torch.inference_mode():
        # Determinism check on a handful of real images, before the long loop.
        probe_images = loaded_splits[0][1][:min(8, len(loaded_splits[0][1]))]
        probe_batch = data.normalize_batch(probe_images, mean, standard_deviation, device)
        assert_mu_is_deterministic(model, probe_batch)
        print("mu determinism check passed (two encodes of the same batch are identical).",
              flush=True)
        del probe_batch

        for split_name, images, _ in loaded_splits:
            print("Encoding %s (%d images, batch %d)..."
                  % (split_name, len(images), arguments.batch_size), flush=True)
            split_largest_mu, split_seconds = encode_split(
                model, images, mean, standard_deviation, device, arguments.batch_size,
                embeddings, offset, split_name)

            split_records.append({
                "name": split_name,
                "start": offset,
                "end": offset + len(images),
                "count": len(images),
                "seconds": split_seconds,
            })
            largest_absolute_mu = max(largest_absolute_mu, split_largest_mu)
            offset += len(images)

    print("Saving %s ..." % (embeddings_file,), flush=True)
    numpy.save(embeddings_file, embeddings)

    labeled_train_offset = None
    for record in split_records:
        if record["name"] == "train":
            labeled_train_offset = record["start"]

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/extract_embeddings.py",
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": arguments.seed,

        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
            "metadata": json_safe(checkpoint_metadata),
        },
        # Hoisted out of the metadata blob because the beta ablation groups its
        # tables by this value and should not have to guess where it lives.
        # scripts/pretrain_vae.py stores it as config["beta"], one level down,
        # which is why the nested lookup is used rather than a flat .get().
        "beta_max": json_safe(find_metadata_value(checkpoint_metadata, ("beta_max", "beta"))),

        "embeddings_file": embeddings_file.name,
        "dtype": "float16",
        "num_nodes": number_of_nodes,
        "embedding_dimension": embedding_dimension,
        "tap_point": "mu",
        "sampled_from_posterior": False,
        "augmentation": "none",
        "transform": "normalize only (deterministic)",
        "largest_absolute_mu": largest_absolute_mu,

        # THE ORDERING CONTRACT. Downstream code turns an index into the 5000
        # labeled train images into a graph node index by adding
        # labeled_train_offset; it must read that number from here rather than
        # assume 100000, which would be wrong for any subset run.
        "split_order": [record["name"] for record in split_records],
        "split_offsets": split_records,
        "labeled_train_offset": labeled_train_offset,

        "normalization": {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "source": normalization_source,
            "path": arguments.normalization_path,
            "source_split": config.NORMALIZATION_SOURCE_SPLIT,
        },
        "data_root": str(Path(arguments.data_root).resolve()),
        "batch_size": arguments.batch_size,
        "max_images_per_split": arguments.max_images_per_split,
        "is_subset": arguments.max_images_per_split > 0,
        "test_split_touched": False,
        "total_seconds": time.time() - started_at,
    }

    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("=" * 78, flush=True)
    print("Wrote %s  (%.1f MB)" % (embeddings_file, embeddings.nbytes / 1e6), flush=True)
    print("Wrote %s" % (manifest_file,), flush=True)
    print("Nodes %d = %s" % (
        number_of_nodes,
        " + ".join("%s[%d:%d]" % (record["name"], record["start"], record["end"])
                   for record in split_records)), flush=True)
    print("largest |mu| = %.4f (float16 holds up to 65504, so no overflow)"
          % (largest_absolute_mu,), flush=True)
    print("Total %.1fs" % (time.time() - started_at,), flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
