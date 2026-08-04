"""THE ONLY SCRIPT IN THIS REPO THAT READS THE STL-10 TEST SPLIT.

    Everything else -- the K sweep, the beta ablation, the C search, both modes
    of scripts/train_downstream.py -- is scored on the 200 held-out
    development-validation images of each official fold. The 8000 test images are
    read here, once, at the very end of the project, and only when this script is
    handed --confirm-test-evaluation.

WHY THE CEREMONY
----------------
The claim the report rests on is "these two latent spaces were compared under an
identical, pre-registered protocol". That claim survives exactly as long as no
hyperparameter is chosen after somebody has seen a test number. Selecting K, beta
or C on test does not merely inflate the accuracy -- it makes the whole
comparison unfalsifiable, because the winning arm is then partly the arm whose
knobs got tuned on the evaluation set. There is no way to undo it afterwards and
no honest way to report it.

So the protection is structural, not a note in the README:

  1. the script refuses to start without an explicit --confirm-test-evaluation;
  2. it refuses to start without a selection.json written BEFORE it ran, holding
     the locked K, beta and C -- there is no --k / --beta / --c flag to pass, so
     there is nothing to fiddle with here;
  3. that selection.json is validated (it must state that selection happened on
     the development split), hashed, and copied verbatim into the output
     directory, so the numbers ship with the exact configuration that produced
     them;
  4. the output directory is timestamped and never reused, so an earlier
     benchmark cannot be quietly overwritten by a second, luckier attempt.

WHAT IT MIRRORS
---------------
Lior's final protocol, exactly: for each of the 10 official STL-10 folds, fit one
classifier on ALL 1000 fold images and evaluate on ALL 8000 test images; the
headline number is the mean over the 10 folds.

TWO ROWS, ALWAYS BOTH
---------------------
  linear_probe        our hand-rolled logistic regression on the frozen VAE mu
                      embeddings. THE APPLES-TO-APPLES ROW -- Lior's downstream
                      is a linear probe too, so this is the only row where the
                      classifier is held constant and the pretraining objective
                      is the single variable. The report leads with this one.

  cnn_pseudo_labels   the PseudoLabelClassifier trained by
                      scripts/train_downstream.py --mode cnn on ~100k propagated
                      pseudo-labels, evaluated on the same 8000 test images. It
                      is CONFOUNDED by classifier capacity (a CNN on 100k noisy
                      labels vs a linear probe on 1000 clean ones), which is
                      precisely why both rows are emitted side by side: the gap
                      between them is the size of the confound.

Outputs, under runs/<name>_<timestamp>/:
  fold_metrics.csv       one row per (arm, fold): accuracy, macro-F1,
                         macro-precision, cross-entropy
  per_class_metrics.csv  one row per (arm, fold, class): precision, recall, F1,
                         support
  confusion_long.csv     long format (arm, fold, true_class, predicted_class,
                         count), plus the fold-summed matrix as fold = -1
  summary.csv/.json      mean +/- std over the 10 folds, per arm
  selection_used.json    verbatim copy of the locked selection, with its sha256
  test_embeddings.npy    the 8000 test mu vectors, when this run encoded them

WHERE THE TEST EMBEDDINGS COME FROM
-----------------------------------
scripts/extract_embeddings.py caches the 105k graph nodes (unlabeled + labeled
train) and is deliberately incapable of reading the test split. So the probe row
encodes the 8000 test images HERE, from --vae-checkpoint, which keeps the single
test read of the whole project inside the single script that announces it. The
checkpoint's sha256 is checked against the embedding cache's manifest first: the
probe is fitted on cached train features and scored on these, and features from
two different encoders would produce a number that means nothing.
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy
import torch

# Repo root on sys.path so "python scripts/final_benchmark.py" works from the
# repo root. Only the root, never src/ itself -- see the same note in
# scripts/train_downstream.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import classifier, config, data, metrics, probe, seeding
from src.vae import VariationalAutoencoder


# Same recognised spellings as scripts/train_downstream.py -- see the FILE
# CONTRACT block in that module's docstring, which is the single description of
# what the earlier stages leave on disk.
GRAPH_EMBEDDING_KEYS = ("graph_embeddings", "embeddings", "mu", "graph_mu")
TRAIN_EMBEDDING_KEYS = ("train_embeddings", "labeled_train_embeddings", "train_mu")
TEST_EMBEDDING_KEYS = ("test_embeddings", "test_mu", "embeddings_test")

# Keys selection.json MUST carry. There is no default and no command-line
# override for any of them: a missing key means the hyperparameters were not
# locked before the test split was opened, which is the one failure this script
# exists to prevent.
REQUIRED_SELECTION_KEYS = ("selected_k", "selected_beta", "selected_c",
                           "selection_split", "locked_at")

# Values of selection_split that mean "this was chosen honestly".
ALLOWED_SELECTION_SPLITS = ("development", "dev", "fold_development",
                            "development_validation")

CNN_CHECKPOINT_PATTERN = "cnn_fold*.pt"
CNN_FOLD_IN_NAME = re.compile(r"cnn_fold(\d+)\.pt$")

# Spellings of the standard-deviation field in normalization.json. The canonical
# one is "standard_deviation", written by scripts/prepare_data.py.
NORMALIZATION_STD_KEYS = ("standard_deviation", "std")

# Where the VAE weights sit inside a checkpoint. "model_state" first -- that is
# what scripts/pretrain_vae.py writes.
STATE_DICT_KEYS = ("model_state", "model_state_dict", "state_dict", "model", "vae_state_dict")


def log(message):
    """Print immediately -- the server runs this under nohup with python -u."""
    print(message, flush=True)


def print_banner():
    """Loud, unmissable, and printed before a single test byte is read."""
    log("")
    log("#" * 78)
    log("#" + " " * 76 + "#")
    log("#   FINAL BENCHMARK -- THIS RUN READS THE STL-10 TEST SPLIT (8000 IMAGES)   #")
    log("#" + " " * 76 + "#")
    log("#   The test split is touched EXACTLY ONCE per project, at the very end.    #")
    log("#   Every hyperparameter must already be locked in selection.json. If you   #")
    log("#   are still choosing K, beta or C, stop now and use the development       #")
    log("#   split -- selecting on test invalidates the entire comparison and        #")
    log("#   cannot be undone by rerunning anything.                                 #")
    log("#" + " " * 76 + "#")
    log("#" * 78)
    log("")


# ---------------------------------------------------------------------------
# Small IO helpers (deliberately duplicated from train_downstream.py -- scripts
# are thin drivers and a shared scripts/_util.py would be the start of a second,
# untested library sitting outside src/)
# ---------------------------------------------------------------------------

def create_run_directory(run_name):
    """runs/<run_name>_<timestamp>/, created fresh, never reused."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    directory = config.RUNS_ROOT / (run_name + "_" + timestamp)
    if directory.exists():
        raise SystemExit(
            "refusing to write into an existing run directory: " + str(directory))
    config.ensure_directory(directory)
    return directory


def write_csv(path, field_names, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log("wrote " + str(path))


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    log("wrote " + str(path))


def summarize(values):
    """mean / std / min / max. std is the SAMPLE std (ddof = 1).

    ddof = 1 is the project-wide convention -- see the fuller note on `summarize`
    in scripts/train_downstream.py. Every table in this project uses it, so
    spreads from the sweep, the downstream runs and this benchmark are the same
    statistic. Lior's notebook reports fold MEANS only and computes no standard
    deviation anywhere, so this column has no counterpart in his arm.
    """
    array = numpy.asarray(values, dtype=numpy.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": float(array.mean()),
        # ddof=1 needs at least two values; a single fold has no spread to report.
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sha256_of_file(path):
    """Hash the locked selection so the report can prove which config produced the numbers."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------

def load_locked_selection(path, started_at):
    """Read and validate selection.json. Every failure here is fatal, on purpose.

    `started_at` is this process's start time. The file must already exist and
    predate it: a selection written while the benchmark was running is, by
    definition, not a selection made before the test split was opened.
    """
    if not os.path.isfile(path):
        raise SystemExit(
            "no selection.json at " + str(path) + ".\n"
            "The final benchmark cannot choose hyperparameters -- that is the whole\n"
            "point of it. Run the sweep first; it writes a selection.json holding the\n"
            "K, beta and C chosen on the DEVELOPMENT split, e.g.\n"
            '  {"selected_k": 15, "selected_beta": 0.01, "selected_c": 10.0,\n'
            '   "selection_split": "development", "test_seen_during_selection": false,\n'
            '   "locked_at": "2026-08-04T12:00:00", "embeddings_run": "vae_beta0.01_seed42"}'
        )

    modified_at = os.path.getmtime(path)
    if modified_at > started_at:
        raise SystemExit(
            "selection.json was modified after this run started (" + str(path) + "). "
            "Hyperparameters must be locked BEFORE the test split is opened."
        )

    with open(path, "r", encoding="utf-8") as handle:
        selection = json.load(handle)

    missing = [key for key in REQUIRED_SELECTION_KEYS if key not in selection]
    if missing:
        raise SystemExit(
            "selection.json is missing required key(s): " + ", ".join(missing)
            + ". Without them the hyperparameters were not locked, and a test "
            "evaluation would not be interpretable."
        )

    selection_split = str(selection["selection_split"]).lower()
    if selection_split not in ALLOWED_SELECTION_SPLITS:
        raise SystemExit(
            'selection_split is "' + str(selection["selection_split"]) + '"; it must be one of '
            + ", ".join(ALLOWED_SELECTION_SPLITS) + ". If any hyperparameter was chosen "
            "on the test split, the comparison is already invalid and no rerun fixes it."
        )

    if "test_seen_during_selection" in selection:
        if bool(selection["test_seen_during_selection"]):
            raise SystemExit(
                "selection.json declares test_seen_during_selection = true. Refusing to "
                "produce a final number from hyperparameters chosen on the test split."
            )

    # Parsed rather than merely present: a free-text "locked_at" would let a
    # placeholder through, and this field is what the report cites as the lock date.
    try:
        datetime.datetime.fromisoformat(str(selection["locked_at"]))
    except ValueError:
        raise SystemExit(
            'locked_at must be an ISO-8601 timestamp (e.g. "2026-08-04T12:00:00"), got '
            + repr(selection["locked_at"])
        )

    return selection


def resolve_locked_c(selection, fold_index):
    """The C for one fold. The single scalar `selected_c` unless per-fold is opted into.

    THE SCALAR `selected_c` IS THE ONE THAT MATCHES LIOR, so it is the default and
    a stray `selected_c_per_fold` key in the selection file can no longer silently
    override it. His inner C search does run inside every fold, but the choice is
    made once: cell 16 averages each C's inner-CV scores across the ten folds and
    fits all ten with that single winner.

    Using per-fold C requires BOTH an explicit `"c_selection": "per-fold"` in the
    selection file AND a `selected_c_per_fold` list. That variant is more
    permissive than his protocol -- ten chances at a good regularization strength
    where his arm gets one -- so if it is used, the report has to say so, because
    part of any gap then belongs to the selection procedure rather than to the
    latent space.

    Both are honest in the sense that matters most (each was chosen on
    development data, never on test); they are simply not the same protocol.
    """
    if selection.get("c_selection") == "per-fold":
        per_fold = selection.get("selected_c_per_fold")
        if per_fold is None:
            raise SystemExit(
                'selection sets "c_selection": "per-fold" but has no '
                "selected_c_per_fold list to read"
            )
        if len(per_fold) != config.NUM_FOLDS:
            raise SystemExit(
                "selected_c_per_fold must hold " + str(config.NUM_FOLDS) + " values, got "
                + str(len(per_fold))
            )
        return float(per_fold[fold_index])
    return float(selection["selected_c"])


# ---------------------------------------------------------------------------
# Loading embeddings
# ---------------------------------------------------------------------------

def load_array_archive(path):
    if not os.path.isfile(path):
        raise SystemExit("no such file: " + str(path))
    if str(path).endswith(".npy"):
        return {"embeddings": numpy.load(path)}
    with numpy.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def pick_array(archive, candidate_keys, description, path):
    for key in candidate_keys:
        if key in archive:
            return archive[key]
    raise SystemExit(
        "could not find the " + description + " in " + str(path) + ". Looked for "
        + ", ".join(candidate_keys) + "; the file holds " + ", ".join(sorted(archive)) + "."
    )


def check_embedding_dimension(embeddings, what):
    """512-d or nothing -- the highest-risk fairness item in the project."""
    if embeddings.ndim != 2:
        raise SystemExit(what + " must be [N, D], got shape " + str(embeddings.shape))
    if embeddings.shape[1] != config.ENCODER_DIM:
        raise SystemExit(
            what + " is " + str(embeddings.shape[1]) + "-d but parity with Lior's arm "
            "requires " + str(config.ENCODER_DIM) + "-d -- refusing to run"
        )


def load_labeled_train_embeddings(embeddings_path):
    """Return the labeled train embeddings, [5000, D].

    The labeled rows are sliced out of the [105000, D] graph array at
    NUM_UNLABELED_IMAGES when no explicit train key exists -- the graph node
    array is unlabeled-then-labeled-train, so labeled image i is graph row
    100000 + i.
    """
    archive = load_array_archive(embeddings_path)

    for key in TRAIN_EMBEDDING_KEYS:
        if key in archive:
            train_embeddings = numpy.asarray(archive[key])
            break
    else:
        graph_embeddings = numpy.asarray(
            pick_array(archive, GRAPH_EMBEDDING_KEYS, "graph embedding matrix", embeddings_path))
        if graph_embeddings.shape[0] == config.NUM_GRAPH_NODES:
            train_embeddings = graph_embeddings[config.NUM_UNLABELED_IMAGES:]
        elif graph_embeddings.shape[0] == config.NUM_TRAIN_IMAGES:
            train_embeddings = graph_embeddings
        else:
            raise SystemExit(
                "embedding array has " + str(graph_embeddings.shape[0]) + " rows; expected "
                + str(config.NUM_GRAPH_NODES) + " or " + str(config.NUM_TRAIN_IMAGES))

    check_embedding_dimension(train_embeddings, "labeled train embeddings")
    if train_embeddings.shape[0] != config.NUM_TRAIN_IMAGES:
        raise SystemExit("labeled train embeddings must hold "
                         + str(config.NUM_TRAIN_IMAGES) + " rows, got "
                         + str(train_embeddings.shape[0]))

    return numpy.ascontiguousarray(train_embeddings)


def find_embedding_manifest(embeddings_path):
    """The manifest scripts/extract_embeddings.py wrote next to the array, or None."""
    stem_manifest = os.path.splitext(str(embeddings_path))[0] + ".manifest.json"
    if os.path.isfile(stem_manifest):
        with open(stem_manifest, "r", encoding="utf-8") as handle:
            return json.load(handle), stem_manifest
    return None, None


def assert_same_encoder(embeddings_path, checkpoint_path):
    """Refuse to mix a train embedding from one encoder with a test embedding from another.

    The probe fits on the cached labeled-train embeddings and scores the test
    embeddings encoded below. If those two came from different checkpoints, the
    classifier is being applied to features from a network it never saw -- the
    accuracy would be near chance, or worse, plausible-but-meaningless. The
    manifest records the sha256 of the checkpoint that produced the cache, so the
    check is exact rather than a naming convention.
    """
    manifest, manifest_path = find_embedding_manifest(embeddings_path)
    if manifest is None:
        log("  no manifest next to " + str(embeddings_path) + " -- cannot verify that the "
            "cached train embeddings and these test embeddings share an encoder")
        return

    recorded = manifest.get("checkpoint", {}).get("sha256")
    if not recorded:
        return

    actual = sha256_of_file(checkpoint_path)
    if recorded != actual:
        raise SystemExit(
            "encoder mismatch. " + str(manifest_path) + " says the cached embeddings came\n"
            "from a checkpoint with sha256 " + recorded + ",\nbut --vae-checkpoint "
            + str(checkpoint_path) + " hashes to " + actual + ".\n"
            "Fitting the probe on one encoder's features and scoring another's is not a "
            "result. Pass the checkpoint that produced the cache."
        )
    log("  encoder verified against " + str(manifest_path) + " (sha256 matches)")


def load_vae_state_dict(path):
    """Pull the VAE weights out of a pretraining checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        log("note: falling back to torch.load(weights_only=False) for " + str(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict):
        for key in STATE_DICT_KEYS:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def encode_test_embeddings(checkpoint_path, embeddings_path, data_root, normalization_path,
                           device, batch_size):
    """Encode the 8000 test images to mu, [8000, 512] float32.

    THIS READS THE TEST SPLIT, which is exactly why it lives here and not in
    scripts/extract_embeddings.py: that script caches the 105k graph nodes and is
    deliberately incapable of touching test, so the single test read in the whole
    project stays inside the single script that announces it.

    mu, never a posterior sample: `encode` returns (mu, logvar), `reparameterize`
    is never called, and the model is in eval mode. Normalize-only, no
    augmentation -- the same deterministic transform the cached train embeddings
    were produced with, or the probe would be scoring features from a different
    distribution than it was fitted on.
    """
    assert_same_encoder(embeddings_path, checkpoint_path)

    state_dict = load_vae_state_dict(checkpoint_path)
    model = VariationalAutoencoder()
    # strict=True: a partial load would leave randomly initialised layers in the
    # encoder, and the test features would look plausible while meaning nothing.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    mean, standard_deviation = resolve_normalization(normalization_path, data_root)
    log("  normalization mean=" + str(mean) + " std=" + str(standard_deviation))

    test_images, _ = data.load_split(data_root, "test")
    log("  encoding " + str(len(test_images)) + " test images at batch " + str(batch_size) + "...")

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(test_images), batch_size):
            end = min(start + batch_size, len(test_images))
            batch = data.normalize_batch(
                test_images[start:end], mean, standard_deviation, device)
            mu, _ = model.encode(batch)
            if not bool(torch.isfinite(mu).all().item()):
                raise SystemExit(
                    "non-finite mu in the test batch starting at " + str(start)
                    + " -- the checkpoint is diverged")
            chunks.append(mu.to(torch.float32).cpu().numpy())

    return numpy.ascontiguousarray(numpy.concatenate(chunks, axis=0))


def resolve_test_embeddings(arguments, run_directory, device):
    """The [8000, D] test features, from a cache if one exists, else encoded here."""
    if arguments.test_embeddings is not None:
        archive = load_array_archive(arguments.test_embeddings)
        test_embeddings = numpy.asarray(
            pick_array(archive, TEST_EMBEDDING_KEYS, "test embedding matrix",
                       arguments.test_embeddings))
        log("test embeddings: cached, from " + str(arguments.test_embeddings))
    else:
        archive = load_array_archive(arguments.embeddings)
        cached = None
        for key in TEST_EMBEDDING_KEYS:
            if key in archive:
                cached = numpy.asarray(archive[key])
                break

        if cached is not None:
            test_embeddings = cached
            log("test embeddings: found inside " + str(arguments.embeddings))
        elif arguments.vae_checkpoint is not None:
            log("test embeddings: encoding them here from " + str(arguments.vae_checkpoint))
            test_embeddings = encode_test_embeddings(
                arguments.vae_checkpoint, arguments.embeddings, arguments.data_root,
                arguments.normalization, device, arguments.batch_size)
            # Saved so a rerun of the report tables does not re-encode, and so the
            # numbers can be audited against the exact features that produced them.
            numpy.save(run_directory / "test_embeddings.npy",
                       test_embeddings.astype(numpy.float16))
            log("wrote " + str(run_directory / "test_embeddings.npy"))
        else:
            raise SystemExit(
                "no test embeddings.\n"
                "scripts/extract_embeddings.py caches the 105k graph nodes only -- it never\n"
                "touches the test split, by design. So the probe row needs one of:\n"
                "  --vae-checkpoint <the .pt the cached embeddings came from>  (encode here)\n"
                "  --test-embeddings <archive holding test_embeddings>         (a cache)")

    check_embedding_dimension(test_embeddings, "test embeddings")
    if test_embeddings.shape[0] != config.NUM_TEST_IMAGES:
        raise SystemExit("test embeddings must hold " + str(config.NUM_TEST_IMAGES)
                         + " rows, got " + str(test_embeddings.shape[0]))

    return numpy.ascontiguousarray(test_embeddings)


def resolve_normalization(normalization_path, data_root):
    """Unlabeled-split channel statistics, from the cache if it exists."""
    if normalization_path is not None and os.path.isfile(normalization_path):
        with open(normalization_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        for key in NORMALIZATION_STD_KEYS:
            if key in stored:
                return stored["mean"], stored[key]
        raise SystemExit(
            str(normalization_path) + " has no standard deviation under any of "
            + ", ".join(NORMALIZATION_STD_KEYS) + "; it holds " + ", ".join(sorted(stored)))

    log("computing normalization constants from the unlabeled split (float64, chunked)...")
    unlabeled_images, _ = data.load_split(data_root, "unlabeled", memory_map_unlabeled=True)
    return data.compute_channel_statistics(
        unlabeled_images, chunk_size=config.NORMALIZATION_CHUNK_IMAGES)


# ---------------------------------------------------------------------------
# Turning one confusion matrix into the report's rows
# ---------------------------------------------------------------------------

def metric_rows_from_confusion(arm, fold_index, confusion, log_loss, class_names):
    """(fold row, per-class rows, confusion long rows) off a single [C, C] table.

    Everything the report quotes is derived from the confusion matrix, which is
    the one primitive src/metrics.py is built around -- accuracy, macro-F1 and
    macro-precision are three readings of the same counts, not three algorithms.
    """
    precision, recall, f1 = metrics.per_class_precision_recall_f1(confusion)
    support = confusion.sum(dim=1)

    fold_row = {
        "arm": arm,
        "fold": fold_index,
        "accuracy": metrics.accuracy(confusion),
        "f1_macro": metrics.macro_f1(confusion),
        "precision_macro": metrics.macro_precision(confusion),
        "cross_entropy": log_loss,
        "test_images": int(confusion.sum().item()),
    }

    per_class_rows = []
    for class_id in range(confusion.shape[0]):
        per_class_rows.append({
            "arm": arm,
            "fold": fold_index,
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
            "precision": float(precision[class_id].item()),
            "recall": float(recall[class_id].item()),
            "f1": float(f1[class_id].item()),
            "support": int(support[class_id].item()),
        })

    confusion_rows = []
    counts = confusion.to(torch.int64).cpu().numpy()
    for true_class in range(counts.shape[0]):
        for predicted_class in range(counts.shape[1]):
            confusion_rows.append({
                "arm": arm,
                "fold": fold_index,
                "true_class": true_class,
                "true_class_name": class_names[true_class] if true_class < len(class_names)
                                   else str(true_class),
                "predicted_class": predicted_class,
                "predicted_class_name": class_names[predicted_class]
                                        if predicted_class < len(class_names)
                                        else str(predicted_class),
                "count": int(counts[true_class, predicted_class]),
            })

    return fold_row, per_class_rows, confusion_rows


# ---------------------------------------------------------------------------
# Row 1: the linear probe -- the apples-to-apples comparison
# ---------------------------------------------------------------------------

def evaluate_linear_probe(arguments, selection, splits, device, class_names, run_directory):
    """Fit on all 1000 fold images with the locked C, score all 8000 test images."""
    official_folds = splits["official_folds"]
    train_labels = splits["train_labels"]

    train_embeddings = load_labeled_train_embeddings(arguments.embeddings)
    test_embeddings = resolve_test_embeddings(arguments, run_directory, device)

    train_features = torch.as_tensor(train_embeddings, dtype=torch.float64, device=device)
    test_features = torch.as_tensor(test_embeddings, dtype=torch.float64, device=device)
    label_vector = torch.as_tensor(train_labels, dtype=torch.int64, device=device)

    # Only the labels are needed for the probe row -- the probe scores cached
    # embeddings, so reading the 221 MB of test pixels would be pure waste.
    test_labels = data.load_labels(
        os.path.join(data.resolve_binary_directory(arguments.data_root), "test_y.bin"))
    if len(test_labels) != config.NUM_TEST_IMAGES:
        raise SystemExit("expected " + str(config.NUM_TEST_IMAGES) + " test labels, got "
                         + str(len(test_labels)))
    test_label_vector = torch.as_tensor(test_labels, dtype=torch.int64, device=device)

    fold_rows = []
    per_class_rows = []
    confusion_rows = []
    summed_confusion = None

    for fold_index in range(config.NUM_FOLDS):
        fold_members = official_folds[fold_index]
        fold_index_tensor = torch.as_tensor(fold_members, device=device)

        fold_features = train_features[fold_index_tensor]
        fold_labels = label_vector[fold_index_tensor]
        locked_c = resolve_locked_c(selection, fold_index)

        # Standardizer fitted on the 1000 FOLD images only, then applied to test.
        # Fitting it on the test features (or on train+test together) would leak
        # the evaluation set's mean and variance into the model that scores it.
        standardizer = probe.Standardizer().fit(fold_features)
        weights, bias = probe.fit_logistic_regression(
            standardizer.transform(fold_features),
            fold_labels,
            C=locked_c,
            max_iterations=arguments.max_iterations,
            num_classes=config.NUM_CLASSES,
        )

        standardized_test = standardizer.transform(test_features)
        probabilities = probe.predict_proba(standardized_test, weights, bias)
        predictions = probabilities.argmax(dim=1)

        confusion = metrics.confusion_matrix(
            test_label_vector, predictions, config.NUM_CLASSES)
        log_loss = metrics.cross_entropy_loss(probabilities, test_label_vector)

        fold_row, fold_class_rows, fold_confusion_rows = metric_rows_from_confusion(
            "linear_probe", fold_index, confusion, log_loss, class_names)
        fold_row["locked_c"] = locked_c
        fold_row["train_images"] = int(fold_features.shape[0])

        fold_rows.append(fold_row)
        per_class_rows.extend(fold_class_rows)
        confusion_rows.extend(fold_confusion_rows)

        summed_confusion = confusion if summed_confusion is None else summed_confusion + confusion

        log("linear_probe fold %d: C=%g  test accuracy=%.4f  macro-F1=%.4f  CE=%.4f"
            % (fold_index, locked_c, fold_row["accuracy"], fold_row["f1_macro"], log_loss))

    return fold_rows, per_class_rows, confusion_rows, summed_confusion


# ---------------------------------------------------------------------------
# Row 2: the CNN on pseudo-labels -- the capacity-confounded comparison
# ---------------------------------------------------------------------------

def find_cnn_checkpoints(checkpoint_directory):
    """{fold_index: path} from cnn_fold%02d.pt, the name train_downstream.py writes."""
    found = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_directory, CNN_CHECKPOINT_PATTERN))):
        match = CNN_FOLD_IN_NAME.search(os.path.basename(path))
        if match is None:
            continue
        found[int(match.group(1))] = path
    return found


def evaluate_cnn(arguments, splits, device, class_names):
    """Score each fold's trained PseudoLabelClassifier on all 8000 test images."""
    checkpoints = find_cnn_checkpoints(arguments.cnn_checkpoints)
    if not checkpoints:
        raise SystemExit(
            "no " + CNN_CHECKPOINT_PATTERN + " under " + str(arguments.cnn_checkpoints)
            + ". That directory should be the checkpoints/ folder of a "
            "scripts/train_downstream.py --mode cnn run."
        )

    log("found CNN checkpoints for folds " + str(sorted(checkpoints)))

    test_images, test_labels = data.load_split(arguments.data_root, "test")
    normalization_mean, normalization_standard_deviation = resolve_normalization(
        arguments.normalization, arguments.data_root)

    fold_rows = []
    per_class_rows = []
    confusion_rows = []
    summed_confusion = None

    for fold_index in sorted(checkpoints):
        payload = torch.load(checkpoints[fold_index], map_location="cpu", weights_only=False)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) \
            and "model_state_dict" in payload else payload

        model = classifier.PseudoLabelClassifier(num_classes=config.NUM_CLASSES)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        probabilities = classifier.predict_probabilities(
            model, test_images, normalization_mean, normalization_standard_deviation,
            device=device, batch_size=config.FEATURE_BATCH_SIZE)
        predictions = probabilities.argmax(dim=1)

        confusion = metrics.confusion_matrix(test_labels, predictions, config.NUM_CLASSES)
        log_loss = metrics.cross_entropy_loss(probabilities, test_labels)

        fold_row, fold_class_rows, fold_confusion_rows = metric_rows_from_confusion(
            "cnn_pseudo_labels", fold_index, confusion, log_loss, class_names)
        fold_row["locked_c"] = ""      # not applicable -- kept so the CSV stays rectangular
        fold_row["train_images"] = ""  # the CNN trained on ~100k pseudo-labelled images

        fold_rows.append(fold_row)
        per_class_rows.extend(fold_class_rows)
        confusion_rows.extend(fold_confusion_rows)

        summed_confusion = confusion if summed_confusion is None else summed_confusion + confusion

        log("cnn_pseudo_labels fold %d: test accuracy=%.4f  macro-F1=%.4f  CE=%.4f"
            % (fold_index, fold_row["accuracy"], fold_row["f1_macro"], log_loss))

    return fold_rows, per_class_rows, confusion_rows, summed_confusion


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="FINAL test-set evaluation. Reads the 8000 STL-10 test images. "
                    "Requires --confirm-test-evaluation and a pre-existing selection.json.")

    parser.add_argument("--confirm-test-evaluation", action="store_true",
                        help="required. Confirms every hyperparameter is already locked.")
    parser.add_argument("--selection", default=str(config.RESULTS_ROOT / "selection.json"),
                        help="selection.json written by the sweep, BEFORE this run")

    parser.add_argument("--name", default="final_benchmark",
                        help="run name; output goes to runs/<name>_<timestamp>/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")

    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--splits", default=str(config.DATA_ROOT / "splits.npz"),
                        help="fold split .npz from data.build_and_save_splits")
    parser.add_argument("--normalization",
                        default=str(config.DATA_ROOT / "normalization.json"))

    parser.add_argument("--embeddings", required=True,
                        help="cached frozen mu embeddings (.npz) for the probe row")
    parser.add_argument("--test-embeddings", default=None,
                        help="separate archive holding the 8000 test embeddings, if they "
                             "are not in --embeddings")
    parser.add_argument("--vae-checkpoint", default=None,
                        help="the VAE .pt the cached embeddings came from. Used to encode "
                             "the 8000 test images here when no test-embedding cache "
                             "exists -- scripts/extract_embeddings.py never touches test, "
                             "so this is the normal path.")
    parser.add_argument("--batch-size", type=int, default=config.FEATURE_BATCH_SIZE,
                        help="images per forward pass when encoding the test split")
    parser.add_argument("--max-iterations", type=int, default=probe.LINEAR_MAX_ITERATIONS)

    parser.add_argument("--cnn-checkpoints", default=None,
                        help="checkpoints/ directory of a --mode cnn run; without it the "
                             "capacity-confound row cannot be emitted")

    return parser.parse_args(argv)


def main(argv=None):
    started_at = time.time()
    arguments = parse_arguments(argv)

    print_banner()

    if not arguments.confirm_test_evaluation:
        raise SystemExit(
            "refusing to run without --confirm-test-evaluation.\n"
            "This script reads the test split. It is meant to be run once, at the end of\n"
            "the project, with every hyperparameter already locked in selection.json.\n"
            "If you are still tuning, score on the development split with\n"
            "  python scripts/train_downstream.py --mode probe ..."
        )

    selection = load_locked_selection(arguments.selection, started_at)
    selection_digest = sha256_of_file(arguments.selection)

    seeding.enable_deterministic()
    seeding.reset_seed(arguments.seed)

    # Same reason as scripts/extract_embeddings.py: TF32 truncates the mantissa to
    # ~10 bits, and the test embeddings encoded below must match the cached train
    # embeddings' precision exactly or the probe is fitted and scored on features
    # from two slightly different numeric pipelines.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = resolve_device(arguments.device)
    run_directory = create_run_directory(arguments.name)

    log("locked selection   " + str(arguments.selection))
    log("  sha256           " + selection_digest)
    log("  locked_at        " + str(selection["locked_at"]))
    log("  selection split  " + str(selection["selection_split"]))
    log("  K                " + str(selection["selected_k"]))
    log("  beta             " + str(selection["selected_beta"]))
    log("  C                " + str(selection.get("selected_c_per_fold",
                                                  selection["selected_c"])))
    log("seed               " + str(arguments.seed))
    log("device             " + str(device))
    log("run directory      " + str(run_directory))
    log("")

    # The exact locked configuration ships next to the numbers it produced.
    write_json(run_directory / "selection_used.json", {
        "selection_path": str(arguments.selection),
        "selection_sha256": selection_digest,
        "selection": selection,
    })
    write_json(run_directory / "arguments.json", vars(arguments))

    if not os.path.isfile(arguments.splits):
        raise SystemExit(
            "no splits file at " + str(arguments.splits) + ". The final benchmark will not "
            "build one: the folds must be the same object every earlier stage used."
        )
    splits = data.load_splits(arguments.splits)

    try:
        class_names = data.load_class_names(arguments.data_root)
    except OSError:
        class_names = [str(index) for index in range(config.NUM_CLASSES)]

    all_fold_rows = []
    all_per_class_rows = []
    all_confusion_rows = []
    summaries = {}

    # --- row 1: the apples-to-apples linear probe ---------------------------
    log("--- linear probe on frozen VAE mu (APPLES-TO-APPLES ROW) ---")
    probe_rows, probe_class_rows, probe_confusion_rows, probe_summed = evaluate_linear_probe(
        arguments, selection, splits, device, class_names, run_directory)

    all_fold_rows.extend(probe_rows)
    all_per_class_rows.extend(probe_class_rows)
    all_confusion_rows.extend(probe_confusion_rows)
    all_confusion_rows.extend(
        metric_rows_from_confusion("linear_probe", -1, probe_summed, float("nan"),
                                   class_names)[2])

    summaries["linear_probe"] = {
        "row_label": "linear probe on frozen mu -- apples-to-apples vs Lior's SimCLR probe",
        "folds": len(probe_rows),
        "accuracy": summarize([row["accuracy"] for row in probe_rows]),
        "f1_macro": summarize([row["f1_macro"] for row in probe_rows]),
        "precision_macro": summarize([row["precision_macro"] for row in probe_rows]),
        "cross_entropy": summarize([row["cross_entropy"] for row in probe_rows]),
    }

    # --- row 2: the capacity-confounded CNN ---------------------------------
    if arguments.cnn_checkpoints is not None:
        log("")
        log("--- CNN on propagated pseudo-labels (CAPACITY-CONFOUNDED ROW) ---")
        cnn_rows, cnn_class_rows, cnn_confusion_rows, cnn_summed = evaluate_cnn(
            arguments, splits, device, class_names)

        all_fold_rows.extend(cnn_rows)
        all_per_class_rows.extend(cnn_class_rows)
        all_confusion_rows.extend(cnn_confusion_rows)
        all_confusion_rows.extend(
            metric_rows_from_confusion("cnn_pseudo_labels", -1, cnn_summed, float("nan"),
                                       class_names)[2])

        summaries["cnn_pseudo_labels"] = {
            "row_label": "CNN on ~100k pseudo-labels -- CONFOUNDED by classifier capacity; "
                         "read it beside the probe row, never instead of it",
            "folds": len(cnn_rows),
            "accuracy": summarize([row["accuracy"] for row in cnn_rows]),
            "f1_macro": summarize([row["f1_macro"] for row in cnn_rows]),
            "precision_macro": summarize([row["precision_macro"] for row in cnn_rows]),
            "cross_entropy": summarize([row["cross_entropy"] for row in cnn_rows]),
        }
    else:
        log("")
        log("!" * 78)
        log("WARNING: no --cnn-checkpoints, so only the linear-probe row was produced.")
        log("The report needs BOTH rows: without the CNN row the capacity confound")
        log("described in CLAUDE.md is invisible, and a reader cannot tell how much of")
        log("any gap comes from the latent space and how much from the classifier.")
        log("!" * 78)

    write_csv(run_directory / "fold_metrics.csv",
              ["arm", "fold", "accuracy", "f1_macro", "precision_macro", "cross_entropy",
               "locked_c", "train_images", "test_images"],
              all_fold_rows)

    write_csv(run_directory / "per_class_metrics.csv",
              ["arm", "fold", "class_id", "class_name", "precision", "recall", "f1", "support"],
              all_per_class_rows)

    write_csv(run_directory / "confusion_long.csv",
              ["arm", "fold", "true_class", "true_class_name", "predicted_class",
               "predicted_class_name", "count"],
              all_confusion_rows)

    summary_rows = []
    for arm, arm_summary in summaries.items():
        for metric_name in ("accuracy", "f1_macro", "precision_macro", "cross_entropy"):
            summary_rows.append({
                "arm": arm,
                "metric": metric_name,
                "mean": arm_summary[metric_name]["mean"],
                "std": arm_summary[metric_name]["std"],
                "min": arm_summary[metric_name]["min"],
                "max": arm_summary[metric_name]["max"],
                "folds": arm_summary[metric_name]["count"],
            })
    write_csv(run_directory / "summary.csv",
              ["arm", "metric", "mean", "std", "min", "max", "folds"], summary_rows)

    write_json(run_directory / "summary.json", {
        "protocol": "one classifier per official STL-10 fold (1000 labeled images), "
                    "evaluated on all 8000 test images, mean over 10 folds -- mirrors "
                    "Lior's SimCLR arm exactly",
        "test_split_touched": True,
        "test_images": config.NUM_TEST_IMAGES,
        "seed": arguments.seed,
        "std_convention": ("sample std (ddof = 1) over the 10 folds, used project-wide. "
                           "Lior's arm reports fold means only and computes no std, so "
                           "this column has no counterpart there"),
        "selection_sha256": selection_digest,
        "selection": selection,
        "arms": summaries,
        "cnn_row_present": arguments.cnn_checkpoints is not None,
    })

    log("")
    log("=" * 78)
    log("FINAL RESULTS -- mean +/- std over " + str(config.NUM_FOLDS) + " official folds, "
        "8000 test images each")
    for arm, arm_summary in summaries.items():
        log("  " + arm)
        log("    accuracy         %.4f +/- %.4f"
            % (arm_summary["accuracy"]["mean"], arm_summary["accuracy"]["std"]))
        log("    macro-F1         %.4f +/- %.4f"
            % (arm_summary["f1_macro"]["mean"], arm_summary["f1_macro"]["std"]))
        log("    macro-precision  %.4f +/- %.4f"
            % (arm_summary["precision_macro"]["mean"], arm_summary["precision_macro"]["std"]))
        log("    cross-entropy    %.4f +/- %.4f"
            % (arm_summary["cross_entropy"]["mean"], arm_summary["cross_entropy"]["std"]))
    log("=" * 78)
    log("")
    log("The test split has now been used. Do not tune anything and rerun this --")
    log("if a hyperparameter changes, the honest path is to say so in the report.")
    log("")
    log("done -> " + str(run_directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
