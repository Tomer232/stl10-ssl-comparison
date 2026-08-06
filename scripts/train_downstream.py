"""Downstream training for Tomer's arm, off cached embeddings and pseudo-labels.

    THIS SCRIPT NEVER TOUCHES THE TEST SPLIT.

Everything here is scored on the 200 held-out development-validation images of
each official fold. The 8000 test images are read by exactly one script in the
repo, `scripts/final_benchmark.py`, and only when it is handed
`--confirm-test-evaluation`. If you find yourself wanting a test number here,
that is the signal that model selection is about to leak -- put the knob in the
sweep instead.

Two modes, both thin drivers over `src/`:

  --mode probe
      The hand-rolled multinomial logistic regression (`src/probe.py`) on our
      own frozen VAE mu embeddings. Per official fold: search C over
      (0.1, 1, 10, 100) with 5 stratified inner splits on the 800
      development-train images, lock the winner, refit on all 800, score the 200
      development-validation images. THIS IS THE APPLES-TO-APPLES ROW against
      Lior's SimCLR arm -- his downstream is a linear probe too, so this is the
      only row where the classifier is held constant and the pretraining
      objective is the single variable. Every output file from this mode is
      labelled `linear_probe` for that reason.

  --mode cnn
      `classifier.PseudoLabelClassifier` trained on the propagated pseudo-labels
      of the 100k unlabeled images. Nodes label propagation could not reach carry
      the -1 sentinel and are dropped before the first epoch; the per-node
      confidence from `labelprop.pseudo_labels_from` can optionally weight the
      loss. This row answers a NARROWER question than the probe row -- how far
      the full pseudo-labelling pipeline gets end to end -- and it is confounded
      by classifier capacity (a CNN on 100k images vs a linear probe on 800).
      Report it beside the probe row, never instead of it.

Both modes write per-fold rows plus a mean +/- std summary.


FILE CONTRACT WITH THE EARLIER STAGES
-------------------------------------
This script consumes artifacts, it does not produce embeddings or run label
propagation. What it expects on disk:

  --embeddings   .npz (or .npy) holding the frozen mu embeddings.
                 Recognised keys, first match wins:
                   graph_embeddings | embeddings | mu | graph_mu
                     [105000, 512] -- the graph node array, unlabeled split
                     first (rows 0..99999) then the labeled train split
                     (rows 100000..104999), which is the ordering
                     config.NUM_GRAPH_NODES describes.
                   train_embeddings | labeled_train_embeddings | train_mu
                     [5000, 512] -- optional; used directly when present,
                     otherwise the labeled rows are sliced out of the graph
                     array.
                 fp16 on disk is expected and fine (107 MB); it is up-cast here.

  --pseudo-labels  .npz holding, for --mode cnn:
                   pseudo_labels  int64 [105000] or [num_folds, 105000]
                   confidence     float  same shape (optional)
                   fold_index / fold_indices  (optional) which fold each row was
                     seeded from, so a 1-D array can still say which fold it
                     belongs to.
                 A 2-D array is the honest form: label propagation is seeded
                 from ONE fold's labeled images, so its pseudo-labels are a
                 function of the fold.

                 OPTIONAL. scripts/sweep_knn_lp.py scores label propagation but
                 does not persist its output, so when this flag is omitted
                 --mode cnn propagates here instead, from --embeddings at the
                 locked --k, using the same src/knn + src/labelprop calls the
                 sweep makes. The resulting labels are written to the run
                 directory as pseudo_labels_fold%02d.npz.

  --splits       the .npz written by `data.build_and_save_splits`. Built here on
                 first use if missing, at config.SEED -- the split is part of the
                 evaluation PROTOCOL, so it is deliberately not re-drawn per
                 --seed. Varying --seed varies training randomness, never which
                 images are held out.

Outputs land in runs/<name>_<timestamp>/, which is created fresh every time so a
rerun can never silently overwrite an earlier result.
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy
import torch

# Repo root on sys.path so "python scripts/train_downstream.py" works from the
# repo root without an editable install. Inserted at the front so the checkout
# always wins over anything installed site-wide. Only the ROOT goes on the path,
# never src/ itself: with both there, "import metrics" and "from src import
# metrics" would build two distinct module objects holding two distinct copies
# of any module state.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import classifier, config, data, knn, labelprop, metrics, probe, seeding


# Keys we accept for each array, in priority order. The earlier stages are
# written by different people at different times; accepting the obvious spellings
# is cheaper than a migration script, and an unrecognised archive fails with a
# message that lists exactly what was looked for.
GRAPH_EMBEDDING_KEYS = ("graph_embeddings", "embeddings", "mu", "graph_mu")
TRAIN_EMBEDDING_KEYS = ("train_embeddings", "labeled_train_embeddings", "train_mu")
PSEUDO_LABEL_KEYS = ("pseudo_labels", "labels", "propagated_labels")
CONFIDENCE_KEYS = ("confidence", "confidences", "pseudo_label_confidence")

# Where an encoder checkpoint's state dict may be hiding inside a training
# checkpoint. A bare state dict is also accepted. "model_state" comes FIRST
# because that is the key scripts/pretrain_vae.py actually writes -- getting this
# wrong does not raise here, it hands from_pretrained_encoder the whole
# checkpoint dict (weights plus config plus history) and the failure surfaces as
# an unreadable load_state_dict error two calls later.
STATE_DICT_KEYS = ("model_state", "model_state_dict", "state_dict", "model", "vae_state_dict")

# Spellings of the standard-deviation field in normalization.json. The canonical
# one is "standard_deviation" (scripts/prepare_data.py writes it, and
# scripts/pretrain_vae.py reads it); "std" is accepted so a hand-written file
# still loads. The mean is "mean" in every writer.
NORMALIZATION_STD_KEYS = ("standard_deviation", "std")


def log(message):
    """Print immediately.

    Every run on the server goes through `nohup python -u`, and a buffered
    stdout makes a live job look hung for minutes at a time. flush=True belts
    what -u braces.
    """
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def create_run_directory(run_name):
    """runs/<run_name>_<timestamp>/, created fresh. Refuses to reuse a directory.

    Results are never overwritten silently: a rerun with the same --name lands in
    a new timestamped directory, so the earlier numbers stay on disk and can be
    diffed. The explicit exists() check is for the (unlikely) same-second
    collision, where exist_ok=True would quietly merge two runs' files.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    directory = config.RUNS_ROOT / (run_name + "_" + timestamp)
    if directory.exists():
        raise SystemExit(
            "refusing to write into an existing run directory: " + str(directory)
        )
    config.ensure_directory(directory)
    return directory


def write_csv(path, field_names, rows):
    """Write a list of dicts as CSV. newline="" is required on Windows."""
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
    """mean / std / min / max over the per-fold values, as plain floats.

    std is the SAMPLE standard deviation (ddof = 1), the project-wide convention.
    The 10 folds are a sample of the protocol's variability, not the whole
    population of it, so dividing by n-1 is the right estimator. Every table in
    this project uses ddof = 1 -- scripts/sweep_knn_lp.py, this script, and
    scripts/final_benchmark.py -- so a spread quoted from any of them is the same
    statistic and they can sit in one table.

    ONE CAVEAT THE REPORT MUST CARRY, because it is not visible in the number:
    Lior's notebook computes no standard deviation at all. It reports fold MEANS
    only (`accuracy_mean`, `f1_macro_mean`, `log_loss_mean`, cells 16 and 26). The
    "+/- std" column is ours alone; there is nothing on his side to match it
    against, and it must not be presented as a like-for-like spread. This is the
    same single-seed gap CLAUDE.md says to raise rather than paper over.
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
    """"auto" -> cuda when present, else cpu. Anything else is taken literally."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Loading the cached artifacts
# ---------------------------------------------------------------------------

def load_array_archive(path):
    """Load a .npz into a dict of arrays, or a .npy into {"embeddings": array}."""
    if not os.path.isfile(path):
        raise SystemExit("no such file: " + str(path))

    if str(path).endswith(".npy"):
        return {"embeddings": numpy.load(path)}

    with numpy.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def pick_array(archive, candidate_keys, description, path):
    """First matching key, or a SystemExit naming what was looked for."""
    for key in candidate_keys:
        if key in archive:
            return archive[key]
    raise SystemExit(
        "could not find the " + description + " in " + str(path) + ". Looked for "
        + ", ".join(candidate_keys) + "; the file holds "
        + ", ".join(sorted(archive)) + "."
    )


def resolve_labeled_train_embeddings(path):
    """Return the [5000, 512] embeddings of the labeled train split.

    Either taken from an explicit train key, or sliced out of the [105000, 512]
    graph array at NUM_UNLABELED_IMAGES. The slice offset is the whole reason
    this helper exists: the graph node array is unlabeled-then-labeled, so
    labeled train image i lives at graph row 100000 + i, and getting that offset
    wrong produces a probe trained on random unlabeled images that still runs and
    still reports a plausible-looking ~10% accuracy.
    """
    archive = load_array_archive(path)

    for key in TRAIN_EMBEDDING_KEYS:
        if key in archive:
            embeddings = numpy.asarray(archive[key])
            source = key
            break
    else:
        graph_embeddings = numpy.asarray(
            pick_array(archive, GRAPH_EMBEDDING_KEYS, "graph embedding matrix", path))
        if graph_embeddings.shape[0] == config.NUM_GRAPH_NODES:
            embeddings = graph_embeddings[config.NUM_UNLABELED_IMAGES:]
            source = "graph rows " + str(config.NUM_UNLABELED_IMAGES) + ".."
        elif graph_embeddings.shape[0] == config.NUM_TRAIN_IMAGES:
            # Already just the labeled split.
            embeddings = graph_embeddings
            source = "whole array (already 5000 rows)"
        else:
            raise SystemExit(
                "embedding array has " + str(graph_embeddings.shape[0]) + " rows; expected "
                + str(config.NUM_GRAPH_NODES) + " (graph nodes) or "
                + str(config.NUM_TRAIN_IMAGES) + " (labeled train only)"
            )

    check_embedding_dimension(embeddings)
    log("labeled train embeddings: " + str(embeddings.shape) + " from " + source)
    return numpy.ascontiguousarray(embeddings)


def resolve_graph_embeddings(path):
    """Return the full [105000, D] graph node array -- unlabeled first, then labeled train.

    Used only by the on-the-fly propagation path below, which needs every node,
    not just the labeled tail.
    """
    archive = load_array_archive(path)
    embeddings = numpy.asarray(
        pick_array(archive, GRAPH_EMBEDDING_KEYS, "graph embedding matrix", path))

    check_embedding_dimension(embeddings)
    if embeddings.shape[0] <= config.NUM_TRAIN_IMAGES:
        raise SystemExit(
            "the embedding array holds only " + str(embeddings.shape[0]) + " rows, which "
            "is the labeled train split alone. Propagation needs the whole graph "
            "(unlabeled + labeled train, " + str(config.NUM_GRAPH_NODES) + " rows)."
        )

    log("graph embeddings: " + str(embeddings.shape))
    return embeddings


def check_embedding_dimension(embeddings):
    """Refuse to proceed on anything but 512-d.

    The single highest-risk fairness item in the project: Lior's embedding is the
    512-d pooled trunk feature with the projection head discarded. A different
    width here would hand one arm a dimensionality advantage that has nothing to
    do with generative vs contrastive, and the comparison would be worthless
    while still producing a full results table. Fail loudly instead.
    """
    if embeddings.ndim != 2:
        raise SystemExit("embeddings must be [N, D], got shape " + str(embeddings.shape))
    if embeddings.shape[1] != config.ENCODER_DIM:
        raise SystemExit(
            "embedding dimension is " + str(embeddings.shape[1]) + " but fairness parity "
            "with Lior's arm requires " + str(config.ENCODER_DIM) + " -- refusing to run"
        )


def load_or_build_splits(splits_path, data_root):
    """Load the fold/development split .npz, building it once if it is missing.

    Built at config.SEED, NOT at --seed. The 10 official folds and their 800/200
    stratified development splits are part of the evaluation protocol both arms
    share; re-drawing them per training seed would mean the multi-seed loop was
    also varying which images are held out, and the seed-to-seed spread would
    then mix training noise with split noise.
    """
    if os.path.isfile(splits_path):
        return data.load_splits(splits_path)

    log("splits file not found, building it once at seed " + str(config.SEED)
        + ": " + str(splits_path))
    return data.build_and_save_splits(data_root, str(splits_path), seed=config.SEED)


def resolve_normalization(normalization_path, data_root):
    """Per-channel mean/std from the UNLABELED split, cached to JSON after the first run.

    Computing them means a float64 pass over 2.76 GB, so the result is cached.
    They come from the unlabeled split only -- using train or test images would
    leak evaluation data into preprocessing -- and both arms must use the same
    constants or the "same embedding space" claim does not hold.
    """
    if normalization_path is not None and os.path.isfile(normalization_path):
        with open(normalization_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        for key in NORMALIZATION_STD_KEYS:
            if key in stored:
                log("normalization constants loaded from " + str(normalization_path))
                return stored["mean"], stored[key]
        raise SystemExit(
            str(normalization_path) + " has no standard deviation under any of "
            + ", ".join(NORMALIZATION_STD_KEYS) + "; it holds " + ", ".join(sorted(stored))
        )

    log("computing normalization constants from the unlabeled split (float64, chunked)...")
    unlabeled_images, _ = data.load_split(data_root, "unlabeled", memory_map_unlabeled=True)
    mean, standard_deviation = data.compute_channel_statistics(
        unlabeled_images, chunk_size=config.NORMALIZATION_CHUNK_IMAGES)

    if normalization_path is not None:
        config.ensure_directory(pathlib.Path(normalization_path).resolve().parent)
        with open(normalization_path, "w", encoding="utf-8") as handle:
            # "standard_deviation", not "std": scripts/prepare_data.py owns this
            # file's schema and scripts/pretrain_vae.py reads that exact key. A
            # file written here with a different spelling would load fine in this
            # script and crash the pretraining that reads it next.
            json.dump({"mean": mean, "standard_deviation": standard_deviation,
                       "source_split": config.NORMALIZATION_SOURCE_SPLIT}, handle, indent=2)
        log("cached normalization constants to " + str(normalization_path))

    return mean, standard_deviation


def parse_fold_argument(fold_argument):
    """"all" -> every official fold; otherwise a comma-separated list of indices."""
    if fold_argument.strip().lower() == "all":
        return list(range(config.NUM_FOLDS))

    folds = []
    for piece in fold_argument.split(","):
        piece = piece.strip()
        if not piece:
            continue
        index = int(piece)
        if not 0 <= index < config.NUM_FOLDS:
            raise SystemExit("fold index out of range: " + str(index))
        folds.append(index)
    if not folds:
        raise SystemExit("--folds selected no folds")
    return folds


# ---------------------------------------------------------------------------
# Mode 1: the linear probe -- the apples-to-apples row
# ---------------------------------------------------------------------------

def run_probe_mode(arguments, run_directory, device):
    """Lior's protocol: inner C search per fold, ONE C for all folds, then refit.

    Two passes, because his selection is global (cell 16):

      pass 1  run the inner 5-fold CV over the C grid inside each fold's 800
              development-train images, and keep every fold's search rows
      ---     average each C's log loss / accuracy / macro-F1 ACROSS the folds and
              rank that aggregate -- one winning C for the whole checkpoint
      pass 2  refit every fold on its 800 with that single C and score its 200
              held-out images

    Selecting per fold instead would give our arm ten chances at a good
    regularization strength where his arm gets one, so the headline row uses the
    global C. `--c-selection per-fold` keeps the other variant available as a
    reported sensitivity check; it is not the comparable number.

    The test split is never read.
    """
    splits = load_or_build_splits(arguments.splits, arguments.data_root)
    official_folds = splits["official_folds"]
    train_labels = splits["train_labels"]
    development_splits = splits["development_splits"]

    embeddings = resolve_labeled_train_embeddings(arguments.embeddings)

    # float64 for the same reason src/probe.py works in float64: N is at most
    # 1000 rows of 512 features, so double precision is free and it removes any
    # question about fp32 rounding from numbers we quote to four decimals.
    feature_matrix = torch.as_tensor(embeddings, dtype=torch.float64, device=device)
    label_vector = torch.as_tensor(train_labels, dtype=torch.int64, device=device)

    fold_rows = []
    cross_validation_rows = []
    per_fold_search_rows = []
    per_fold_c = {}
    prepared_folds = []

    # ---- pass 1: inner C search inside every fold, no fitting yet -------------
    for fold_index in parse_fold_argument(arguments.folds):
        search_start = time.time()

        # Compose fold-relative development positions with the fold's official
        # indices to get positions into the 5000 labeled train images.
        fold_members = official_folds[fold_index]
        train_positions, validation_positions = development_splits[fold_index]

        development_train_indices = fold_members[train_positions]
        development_validation_indices = fold_members[validation_positions]

        train_index = torch.as_tensor(development_train_indices, device=device)
        validation_index = torch.as_tensor(development_validation_indices, device=device)

        development_train_features = feature_matrix[train_index]
        development_train_labels = label_vector[train_index]
        development_validation_features = feature_matrix[validation_index]
        development_validation_labels = label_vector[validation_index]

        # Inner search. Seed is --seed + fold_index, mirroring Lior's
        # random_state = SEED + fold_index, so with --seed 42 the two arms run the
        # same protocol with the same per-fold stream.
        search_rows = probe.cross_validate_c(
            development_train_features,
            development_train_labels,
            c_grid=probe.LINEAR_C_GRID,
            num_splits=probe.CROSS_VALIDATION_SPLITS,
            seed=arguments.seed + fold_index,
        )
        fold_best_c = probe.select_best_c(search_rows)
        per_fold_search_rows.append(search_rows)
        per_fold_c[fold_index] = fold_best_c

        prepared_folds.append({
            "fold": fold_index,
            "search_seconds": time.time() - search_start,
            "train_features": development_train_features,
            "train_labels": development_train_labels,
            "validation_features": development_validation_features,
            "validation_labels": development_validation_labels,
        })

        log("fold %d: inner search done, fold-local best C=%g (%.1fs)"
            % (fold_index, fold_best_c, prepared_folds[-1]["search_seconds"]))

    # ---- the selection itself, once, across all folds ------------------------
    global_c, aggregate_c_rows = probe.select_best_c_across_folds(per_fold_search_rows)
    modal_c = max(set(per_fold_c.values()), key=list(per_fold_c.values()).count)

    if arguments.c_selection == "global":
        log("")
        log("C selection: GLOBAL C=%g applied to every fold (Lior's cell-16 rule)" % global_c)
    else:
        log("")
        log("C selection: PER-FOLD (sensitivity variant, NOT the comparable row)")

    for row in aggregate_c_rows:
        cross_validation_rows.append({
            "fold": -1,  # -1 marks the across-fold aggregate, not a real fold
            "c": row["c"],
            "cv_accuracy_mean": row["accuracy_mean"],
            "cv_f1_macro_mean": row["f1_macro_mean"],
            "cv_log_loss_mean": row["log_loss_mean"],
            "selected": int(row["c"] == global_c),
        })

    # ---- pass 2: refit each fold with the selected C and score its 200 --------
    for position, prepared in enumerate(prepared_folds):
        fold_index = prepared["fold"]
        fold_start = time.time()

        development_train_features = prepared["train_features"]
        development_train_labels = prepared["train_labels"]
        development_validation_features = prepared["validation_features"]
        development_validation_labels = prepared["validation_labels"]

        selected_c = global_c if arguments.c_selection == "global" else per_fold_c[fold_index]

        for row in per_fold_search_rows[position]:
            cross_validation_rows.append({
                "fold": fold_index,
                "c": row["c"],
                "cv_accuracy_mean": row["accuracy_mean"],
                "cv_f1_macro_mean": row["f1_macro_mean"],
                "cv_log_loss_mean": row["log_loss_mean"],
                "selected": int(row["c"] == selected_c),
            })

        # Refit on all 800 with the locked C. The standardizer is fitted on the
        # 800 training rows ONLY and then applied to the 200 held-out rows --
        # fitting it on all 1000 would let the validation images' mean and
        # variance into the model that scores them.
        standardizer = probe.Standardizer().fit(development_train_features)
        weights, bias = probe.fit_logistic_regression(
            standardizer.transform(development_train_features),
            development_train_labels,
            C=selected_c,
            max_iterations=arguments.max_iterations,
            num_classes=config.NUM_CLASSES,
        )

        scores = probe.evaluate_probe(
            standardizer.transform(development_validation_features),
            development_validation_labels,
            weights, bias,
            num_classes=config.NUM_CLASSES,
        )

        # macro-precision is not in evaluate_probe's dict, so it is read off the
        # same confusion matrix here rather than recomputing predictions.
        predictions = probe.predict(
            standardizer.transform(development_validation_features), weights, bias)
        confusion = metrics.confusion_matrix(
            development_validation_labels, predictions, config.NUM_CLASSES)

        fold_rows.append({
            "arm": "linear_probe",
            "fold": fold_index,
            "selected_c": selected_c,
            # What this fold's own inner search would have picked, kept so the
            # global-vs-per-fold gap is visible in the table without a second run.
            "fold_local_c": per_fold_c[fold_index],
            "c_source": arguments.c_selection,
            "train_images": int(development_train_features.shape[0]),
            "eval_images": int(development_validation_features.shape[0]),
            "dev_accuracy": scores["accuracy"],
            "dev_f1_macro": scores["f1_macro"],
            "dev_precision_macro": metrics.macro_precision(confusion),
            "dev_log_loss": scores["log_loss"],
            "seconds": time.time() - fold_start,
        })

        log("fold %d: C=%g  dev-val accuracy=%.4f  macro-F1=%.4f  log-loss=%.4f  (%.1fs)"
            % (fold_index, selected_c, scores["accuracy"], scores["f1_macro"],
               scores["log_loss"], fold_rows[-1]["seconds"]))

    write_csv(run_directory / "fold_metrics.csv",
              ["arm", "fold", "selected_c", "fold_local_c", "c_source",
               "train_images", "eval_images",
               "dev_accuracy", "dev_f1_macro", "dev_precision_macro", "dev_log_loss",
               "seconds"],
              fold_rows)

    write_csv(run_directory / "c_search.csv",
              ["fold", "c", "cv_accuracy_mean", "cv_f1_macro_mean", "cv_log_loss_mean",
               "selected"],
              cross_validation_rows)

    selected_c_values = [row["selected_c"] for row in fold_rows]
    fold_local_c_values = [row["fold_local_c"] for row in fold_rows]

    summary = {
        "arm": "linear_probe",
        "row_label": "apples-to-apples vs Lior's SimCLR linear probe",
        "evaluated_on": "fold development-validation (200 images per fold)",
        "test_split_touched": False,
        "folds": [row["fold"] for row in fold_rows],
        "accuracy": summarize([row["dev_accuracy"] for row in fold_rows]),
        "f1_macro": summarize([row["dev_f1_macro"] for row in fold_rows]),
        "precision_macro": summarize([row["dev_precision_macro"] for row in fold_rows]),
        "log_loss": summarize([row["dev_log_loss"] for row in fold_rows]),
        "c_selection": arguments.c_selection,
        "selected_c_global": global_c,
        "selected_c_modal": modal_c,
        "selected_c_used_per_fold": selected_c_values,
        "fold_local_c": fold_local_c_values,
        "c_selection_rule": ("ONE C for all folds (default). Each fold's inner 5-fold CV over "
                             "the C grid is averaged ACROSS folds, and that aggregate is ranked "
                             "by lowest log loss, then highest accuracy, then highest macro-F1, "
                             "then smaller C -- Lior's cell 16 rule, applied the way he applies "
                             "it. fold_local_c records what each fold would have chosen alone; "
                             "--c-selection per-fold uses those instead, as a sensitivity check "
                             "that is NOT comparable to his arm."),
    }
    write_json(run_directory / "summary.json", summary)

    write_csv(run_directory / "c_search_across_folds.csv",
              ["c", "log_loss_mean", "accuracy_mean", "f1_macro_mean"],
              aggregate_c_rows)

    # A candidate C for the final benchmark's lock. Deliberately NOT called
    # selection.json: final_benchmark.py requires the sweep's own selection.json,
    # which also carries the locked K and beta. This file is an INPUT to that
    # lock, not a substitute for it -- writing it here keeps the number the
    # development split actually chose, so nobody has to retype it later.
    write_json(run_directory / "probe_selection_candidate.json", {
        # The GLOBAL C: Lior's arm fits all ten folds with a single C chosen from
        # the fold-averaged inner search, so this is the value that keeps the two
        # arms' probes regularized the same way. Always the global number here,
        # even under --c-selection per-fold, because the lock feeds the benchmark
        # row that has to be comparable to his.
        "selected_c": global_c,
        "selected_c_modal": modal_c,
        "fold_local_c": fold_local_c_values,
        "c_selection_used_in_this_run": arguments.c_selection,
        "selection_split": "development",
        "test_seen_during_selection": False,
        "source": "scripts/train_downstream.py --mode probe",
        "note": "feed this into the sweep's selection.json; it is not a selection.json",
    })

    log("")
    log("LINEAR PROBE (apples-to-apples row) -- development-validation, no test")
    log("  accuracy  %.4f +/- %.4f" % (summary["accuracy"]["mean"], summary["accuracy"]["std"]))
    log("  macro-F1  %.4f +/- %.4f" % (summary["f1_macro"]["mean"], summary["f1_macro"]["std"]))
    log("  log-loss  %.4f +/- %.4f" % (summary["log_loss"]["mean"], summary["log_loss"]["std"]))
    log("  std is the sample std (ddof=1) over the folds; Lior's arm reports no std")
    log("  C used (%s): %s" % (arguments.c_selection, str(selected_c_values)))
    log("  C global (Lior's rule -- one C for all folds): " + str(global_c))
    log("  C each fold would have picked alone:          " + str(fold_local_c_values))
    if arguments.c_selection != "global":
        log("  WARNING: per-fold C is a sensitivity variant. The row comparable to")
        log("           Lior's arm is the one produced by --c-selection global.")


# ---------------------------------------------------------------------------
# Mode 2: the CNN on propagated pseudo-labels
# ---------------------------------------------------------------------------

def load_encoder_state_dict(path):
    """Pull a state dict out of a checkpoint file, wrapper or not."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older torch has no weights_only, and our own checkpoints carry plain
        # metadata (floats, ints, lists) that the safe unpickler may refuse.
        log("note: falling back to torch.load(weights_only=False) for " + str(path))
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        for key in STATE_DICT_KEYS:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def select_fold_pseudo_labels(archive, path, fold_index, requested_fold_count):
    """Return (pseudo_labels, confidence) for one fold, both [num_unlabeled].

    Label propagation is seeded from ONE fold's labeled images, so its output is
    a function of the fold and the natural cache shape is [num_folds, N]. A 1-D
    array is accepted for the single-fold case; combining it with several
    requested folds is refused, because training the same pseudo-labels ten times
    and calling the spread a fold-to-fold standard deviation would be a lie.
    """
    pseudo_labels = numpy.asarray(
        pick_array(archive, PSEUDO_LABEL_KEYS, "pseudo-label array", path))

    confidence = None
    for key in CONFIDENCE_KEYS:
        if key in archive:
            confidence = numpy.asarray(archive[key])
            break

    if pseudo_labels.ndim == 2:
        if not 0 <= fold_index < pseudo_labels.shape[0]:
            raise SystemExit(
                "pseudo-label array holds " + str(pseudo_labels.shape[0]) + " folds, "
                "cannot take fold " + str(fold_index)
            )
        fold_labels = pseudo_labels[fold_index]
        fold_confidence = None if confidence is None else confidence[fold_index]
    elif pseudo_labels.ndim == 1:
        if requested_fold_count > 1:
            raise SystemExit(
                "the pseudo-label cache holds a single 1-D array but " + str(requested_fold_count)
                + " folds were requested. Label propagation is seeded per fold, so a "
                "fold-to-fold spread needs a [num_folds, N] array. Either rerun the "
                "sweep so it stores one row per fold, or pass --folds <one index>."
            )
        fold_labels = pseudo_labels
        fold_confidence = confidence
    else:
        raise SystemExit("pseudo_labels must be [N] or [num_folds, N], got shape "
                         + str(pseudo_labels.shape))

    # Trim the graph array down to the unlabeled split. The graph is
    # unlabeled-then-labeled-train; the labeled tail holds the seeds themselves,
    # whose "pseudo-labels" are just their clamped true labels and would add
    # nothing but a duplicate of the 800 images the graph was seeded with.
    if fold_labels.shape[0] == config.NUM_GRAPH_NODES:
        fold_labels = fold_labels[:config.NUM_UNLABELED_IMAGES]
        if fold_confidence is not None:
            fold_confidence = fold_confidence[:config.NUM_UNLABELED_IMAGES]
    elif fold_labels.shape[0] != config.NUM_UNLABELED_IMAGES:
        raise SystemExit(
            "pseudo-label row has " + str(fold_labels.shape[0]) + " entries; expected "
            + str(config.NUM_GRAPH_NODES) + " (graph nodes) or "
            + str(config.NUM_UNLABELED_IMAGES) + " (unlabeled split)"
        )

    return fold_labels.astype(numpy.int64), fold_confidence


def propagate_fold_pseudo_labels(propagation_matrix, official_folds, development_splits,
                                 train_labels, fold_index, num_nodes, arguments, device):
    """Run label propagation for one fold and return its (pseudo_labels, confidence).

    THE FALLBACK PATH, used when no --pseudo-labels cache exists. Preferring the
    cache is not laziness: the sweep already built this graph once per K, and
    rebuilding it here costs another exact 105k-node search. But the sweep does
    not persist the propagated labels, and a CNN with no pseudo-labels is not a
    pipeline, so the option has to exist.

    Every line below is a call into src/ -- knn.build_propagation_matrix built the
    matrix in the caller, and the propagation itself is labelprop.propagate. The
    identical sequence lives in scripts/sweep_knn_lp.py's evaluate_fold, and the
    two MUST agree: if the labels this trains on were produced by a different
    procedure than the one that selected K, the selected K would not be the K in
    force here.

    SEEDS ARE THE 800 DEVELOPMENT-TRAIN IMAGES ONLY, never the full 1000. The
    other 200 are this fold's evaluation set; seeding them would clamp their true
    labels into the graph and then score the CNN on images whose answers were
    handed to the propagation that generated its training data.

    Only the unlabeled rows are returned. The labeled tail holds the seeds, whose
    "pseudo-labels" are their own clamped true labels.
    """
    seed_positions, _ = development_splits[fold_index]
    seed_indices = official_folds[fold_index][seed_positions]

    # Labeled train image i is graph node labeled_train_offset + i. The offset is
    # DERIVED from the two array lengths rather than read from
    # config.NUM_UNLABELED_IMAGES, so the same expression gives the node index and
    # the trim point below and the two cannot disagree. For a full run it is
    # 100000, the value extract_embeddings.py records in its manifest.
    labeled_train_offset = num_nodes - len(train_labels)
    if labeled_train_offset <= 0:
        raise SystemExit(
            "the embedding array holds " + str(num_nodes) + " nodes but there are "
            + str(len(train_labels)) + " labeled train images -- the unlabeled split "
            "is missing from the graph")

    seed_nodes = torch.as_tensor(
        seed_indices.astype(numpy.int64) + labeled_train_offset, device=device)
    seed_labels = torch.as_tensor(
        train_labels[seed_indices].astype(numpy.int64), device=device)

    initial_F = labelprop.initial_label_matrix(
        num_nodes, config.NUM_CLASSES, seed_nodes, seed_labels, device)
    # The clamp target is exactly what initial_label_matrix wrote at the seed
    # rows, so the two cannot drift apart.
    labeled_onehot = initial_F[seed_nodes]

    # Must match whatever scripts/sweep_knn_lp.py selected K on, or the CNN is
    # trained on pseudo-labels from a different algorithm than the one the sweep
    # justified. `spread` is the default for the reason documented in
    # labelprop.spread: hard clamping converges to a nearly flat fixed point on
    # this graph (mean winning-class share 0.155 against a 0.10 floor), so the
    # pseudo-labels it produces are close to noise.
    if arguments.lp_method == "spread":
        final_F, iterations, final_delta = labelprop.spread(
            propagation_matrix, initial_F, seed_nodes, labeled_onehot,
            arguments.lp_alpha, arguments.lp_max_iterations, arguments.lp_tolerance)
    else:
        final_F, iterations, final_delta = labelprop.propagate(
            propagation_matrix, initial_F, seed_nodes, labeled_onehot,
            arguments.lp_max_iterations, arguments.lp_tolerance)

    if final_delta >= arguments.lp_tolerance:
        log("  WARNING: propagation hit the " + str(arguments.lp_max_iterations)
            + "-iteration cap with delta " + ("%.3e" % final_delta) + " -- these "
            "pseudo-labels are a snapshot of a still-moving iteration. Report it.")
    else:
        log("  propagation converged in " + str(iterations) + " iterations (delta "
            + ("%.3e" % final_delta) + ")")

    pseudo_labels, confidence = labelprop.pseudo_labels_from(final_F)

    return (pseudo_labels[:labeled_train_offset].cpu().numpy().astype(numpy.int64),
            confidence[:labeled_train_offset].cpu().numpy())


def run_cnn_mode(arguments, run_directory, device):
    """Train PseudoLabelClassifier per fold on that fold's pseudo-labels. No test."""
    splits = load_or_build_splits(arguments.splits, arguments.data_root)
    official_folds = splits["official_folds"]
    train_labels = splits["train_labels"]
    development_splits = splits["development_splits"]

    folds = parse_fold_argument(arguments.folds)
    if len(folds) > 1:
        log("WARNING: " + str(len(folds)) + " folds x " + str(arguments.epochs)
            + " epochs over ~100k images. On the 5090 that is roughly "
            + str(len(folds)) + " x 1.5-2 h. Use --folds 0 for a single-fold run.")

    # Two ways to obtain the pseudo-labels, in priority order.
    #   1. a cache written by an earlier stage -- preferred, because then this
    #      script trains on exactly the array the sweep scored;
    #   2. propagate here from the cached embeddings at the locked K. The graph
    #      depends only on the embeddings and K, never on which nodes are seeded,
    #      so it is built ONCE outside the fold loop and the ten folds differ only
    #      in the seed set handed to labelprop.propagate.
    archive = None
    propagation_matrix = None
    num_nodes = None

    if arguments.pseudo_labels is not None:
        archive = load_array_archive(arguments.pseudo_labels)
        log("pseudo-labels: cached, from " + str(arguments.pseudo_labels))
    else:
        graph_embeddings = resolve_graph_embeddings(arguments.embeddings)
        num_nodes = int(graph_embeddings.shape[0])
        log("pseudo-labels: propagating here at K=" + str(arguments.k)
            + " (no --pseudo-labels cache given)")
        graph_start = time.time()
        # Handed to knn as a TENSOR ALREADY ON `device`, not as the numpy array,
        # and this is not a style preference. knn.build_knn_graph returns its
        # output wherever the INPUT lived, so passing the numpy array would run
        # the neighbour search on the GPU and then move the graph back to the
        # host -- after which the heat-kernel weighting, the 2*N*K-entry sort in
        # symmetrize_union (10.5M edges at K=50), row_normalize and every
        # iteration of label propagation would all execute on the CPU. fp16 on
        # disk -> fp32 on the device, the same up-cast src/knn.py does
        # internally. scripts/sweep_knn_lp.py already does exactly this.
        graph_features = torch.from_numpy(
            numpy.ascontiguousarray(graph_embeddings)).to(device=device, dtype=torch.float32)
        propagation_matrix, _, _, sigma = knn.build_propagation_matrix(
            graph_features, arguments.k, sigma=arguments.sigma,
            chunk_rows=config.SIMILARITY_CHUNK_ROWS, device=device)
        del graph_features, graph_embeddings
        log("built the KNN graph in %.1fs (sigma=%.4f)" % (time.time() - graph_start, sigma))

    normalization_mean, normalization_standard_deviation = resolve_normalization(
        arguments.normalization, arguments.data_root)

    log("loading the unlabeled split (" + ("resident" if arguments.load_images_to_ram
                                           else "memory-mapped") + ")...")
    unlabeled_images, _ = data.load_split(
        arguments.data_root, "unlabeled",
        memory_map_unlabeled=not arguments.load_images_to_ram)

    # The 200 development-validation images of each fold come from the labeled
    # train split, which is small enough to hold outright (138 MB).
    labeled_images, _ = data.load_split(arguments.data_root, "train")

    encoder_state = None
    if arguments.encoder_checkpoint is not None:
        encoder_state = load_encoder_state_dict(arguments.encoder_checkpoint)
        log("encoder initialised from " + str(arguments.encoder_checkpoint))
    else:
        log("WARNING: no --encoder-checkpoint, the trunk starts from random init. "
            "That measures the pseudo-labels alone, not the VAE representation.")

    checkpoint_directory = config.ensure_directory(run_directory / "checkpoints")

    fold_rows = []

    for fold_index in folds:
        fold_start = time.time()
        log("")
        log("=== fold " + str(fold_index) + " ===")

        # Reseed per fold so a fold is reproducible on its own, rather than
        # depending on how many folds ran before it.
        seeding.reset_seed(arguments.seed + fold_index)

        if archive is not None:
            pseudo_labels, confidence = select_fold_pseudo_labels(
                archive, arguments.pseudo_labels, fold_index, len(folds))
        else:
            pseudo_labels, confidence = propagate_fold_pseudo_labels(
                propagation_matrix, official_folds, development_splits, train_labels,
                fold_index, num_nodes, arguments, device)
            # Cached next to the results so the exact labels this fold trained on
            # can be re-examined without rebuilding the graph.
            numpy.savez_compressed(
                run_directory / ("pseudo_labels_fold%02d.npz" % fold_index),
                pseudo_labels=pseudo_labels, confidence=confidence,
                k=numpy.asarray(arguments.k), fold_index=numpy.asarray(fold_index))

        unreachable = int((pseudo_labels < 0).sum())

        # Optional confidence floor. Nodes below it are demoted to the -1
        # sentinel, which train_classifier already drops -- rather than being
        # silently down-weighted to near zero while still costing a forward pass.
        if arguments.min_confidence > 0.0:
            if confidence is None:
                raise SystemExit("--min-confidence needs a confidence array in the cache")
            low_confidence = confidence < arguments.min_confidence
            pseudo_labels = numpy.where(low_confidence, -1, pseudo_labels)
            log("dropped " + str(int(low_confidence.sum())) + " nodes below confidence "
                + str(arguments.min_confidence))

        sample_weights = None
        if arguments.confidence_weighting:
            if confidence is None:
                raise SystemExit("--confidence-weighting needs a confidence array in the cache")
            sample_weights = numpy.asarray(confidence, dtype=numpy.float64)

        usable = int((pseudo_labels >= 0).sum())
        log("pseudo-labels: " + str(usable) + " usable, " + str(unreachable)
            + " unreachable (-1, dropped)")

        model = build_classifier(encoder_state, arguments.freeze_encoder)
        model.to(device)

        history_rows = []

        def epoch_callback(epoch_record, _model):
            history_rows.append(dict(epoch_record))
            if epoch_record["epoch"] % arguments.log_every == 0 \
                    or epoch_record["epoch"] == arguments.epochs - 1:
                log("  epoch %3d  loss %.4f  pseudo-label acc %.4f  lr %.2e"
                    % (epoch_record["epoch"], epoch_record["loss"],
                       epoch_record["pseudo_label_accuracy"],
                       epoch_record["last_learning_rate"]))

        classifier.train_classifier(
            model,
            unlabeled_images,
            pseudo_labels,
            normalization_mean,
            normalization_standard_deviation,
            device=device,
            sample_weights=sample_weights,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            max_learning_rate=config.MAX_LEARNING_RATE,
            min_learning_rate=config.MIN_LEARNING_RATE,
            warmup_epochs=config.WARMUP_EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            seed=arguments.seed + fold_index,
            num_classes=config.NUM_CLASSES,
            ignore_label=classifier.UNLABELED_SENTINEL,
            epoch_callback=epoch_callback,
        )

        write_csv(run_directory / ("history_fold%02d.csv" % fold_index),
                  ["epoch", "loss", "pseudo_label_accuracy",
                   "first_learning_rate", "last_learning_rate", "samples"],
                  history_rows)

        # Scored on the fold's 200 development-validation images, which hold REAL
        # labels and were never seeded into label propagation. Not the test split.
        _, validation_positions = development_splits[fold_index]
        validation_indices = official_folds[fold_index][validation_positions]

        probabilities = classifier.predict_probabilities(
            model,
            labeled_images[validation_indices],
            normalization_mean,
            normalization_standard_deviation,
            device=device,
            batch_size=config.FEATURE_BATCH_SIZE,
        )
        predictions = probabilities.argmax(dim=1)
        true_labels = train_labels[validation_indices]
        confusion = metrics.confusion_matrix(true_labels, predictions, config.NUM_CLASSES)

        checkpoint_path = checkpoint_directory / ("cnn_fold%02d.pt" % fold_index)
        torch.save({
            "fold_index": fold_index,
            "model_state_dict": model.state_dict(),
            "num_classes": config.NUM_CLASSES,
            "seed": arguments.seed + fold_index,
            "epochs": arguments.epochs,
            "confidence_weighting": bool(arguments.confidence_weighting),
            "min_confidence": float(arguments.min_confidence),
            "freeze_encoder": bool(arguments.freeze_encoder),
        }, checkpoint_path)
        log("wrote " + str(checkpoint_path))

        fold_rows.append({
            "arm": "cnn_pseudo_labels",
            "fold": fold_index,
            "training_images": usable,
            "unreachable_dropped": unreachable,
            "final_train_loss": history_rows[-1]["loss"] if history_rows else float("nan"),
            "final_pseudo_label_accuracy":
                history_rows[-1]["pseudo_label_accuracy"] if history_rows else float("nan"),
            "dev_accuracy": metrics.accuracy(confusion),
            "dev_f1_macro": metrics.macro_f1(confusion),
            "dev_precision_macro": metrics.macro_precision(confusion),
            "dev_log_loss": metrics.cross_entropy_loss(probabilities, true_labels),
            "seconds": time.time() - fold_start,
        })

        log("fold %d: dev-val accuracy=%.4f  macro-F1=%.4f  (%.1f min)"
            % (fold_index, fold_rows[-1]["dev_accuracy"], fold_rows[-1]["dev_f1_macro"],
               fold_rows[-1]["seconds"] / 60.0))

    write_csv(run_directory / "fold_metrics.csv",
              ["arm", "fold", "training_images", "unreachable_dropped",
               "final_train_loss", "final_pseudo_label_accuracy",
               "dev_accuracy", "dev_f1_macro", "dev_precision_macro", "dev_log_loss",
               "seconds"],
              fold_rows)

    summary = {
        "arm": "cnn_pseudo_labels",
        "row_label": "CNN on propagated pseudo-labels -- CONFOUNDED by classifier "
                     "capacity vs Lior's linear probe; report beside the probe row",
        "evaluated_on": "fold development-validation (200 images per fold)",
        "test_split_touched": False,
        "folds": [row["fold"] for row in fold_rows],
        "accuracy": summarize([row["dev_accuracy"] for row in fold_rows]),
        "f1_macro": summarize([row["dev_f1_macro"] for row in fold_rows]),
        "precision_macro": summarize([row["dev_precision_macro"] for row in fold_rows]),
        "log_loss": summarize([row["dev_log_loss"] for row in fold_rows]),
        "confidence_weighting": bool(arguments.confidence_weighting),
        "min_confidence": float(arguments.min_confidence),
        "checkpoint_directory": str(checkpoint_directory),
        "pseudo_label_source": (str(arguments.pseudo_labels) if archive is not None
                                else "propagated in this run from " + str(arguments.embeddings)),
        "k": arguments.k,
    }
    write_json(run_directory / "summary.json", summary)

    log("")
    log("CNN ON PSEUDO-LABELS (capacity-confounded row) -- development-validation, no test")
    log("  accuracy  %.4f +/- %.4f" % (summary["accuracy"]["mean"], summary["accuracy"]["std"]))
    log("  macro-F1  %.4f +/- %.4f" % (summary["f1_macro"]["mean"], summary["f1_macro"]["std"]))


def build_classifier(encoder_state, freeze_encoder):
    """Fresh PseudoLabelClassifier, optionally starting from a pretrained trunk."""
    if encoder_state is None:
        model = classifier.PseudoLabelClassifier(
            num_classes=config.NUM_CLASSES, freeze_encoder=freeze_encoder)
    else:
        # strict=True on purpose: a successful load is itself evidence that our
        # trunk and the checkpoint's trunk are the same network, which is the
        # trunk-parity claim the report makes.
        model = classifier.PseudoLabelClassifier.from_pretrained_encoder(
            encoder_state, num_classes=config.NUM_CLASSES,
            freeze_encoder=freeze_encoder, strict=True)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Downstream training on cached embeddings / pseudo-labels. "
                    "Scored on the development split only -- never on test.")

    parser.add_argument("--mode", choices=("probe", "cnn"), required=True,
                        help="probe: linear probe on frozen mu (apples-to-apples row). "
                             "cnn: PseudoLabelClassifier on propagated pseudo-labels.")
    parser.add_argument("--name", default=None,
                        help="run name; the output goes to runs/<name>_<timestamp>/")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="training seed. The fold/development splits are NOT "
                             "re-drawn from it -- they are protocol, fixed at config.SEED.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--folds", default="all",
                        help='"all" or a comma-separated list, e.g. "0,1,2"')

    parser.add_argument("--data-root", default=str(config.DATA_ROOT),
                        help="directory holding stl10_binary/")
    parser.add_argument("--splits", default=str(config.DATA_ROOT / "splits.npz"),
                        help="fold/development split .npz; built once if missing")
    parser.add_argument("--normalization",
                        default=str(config.DATA_ROOT / "normalization.json"),
                        help="cached unlabeled-split channel statistics (cnn mode)")

    parser.add_argument("--embeddings", default=None,
                        help="cached frozen mu embeddings (.npz/.npy) -- probe mode")
    parser.add_argument("--max-iterations", type=int, default=probe.LINEAR_MAX_ITERATIONS,
                        help="L-BFGS iterations for the probe refit")

    parser.add_argument("--c-selection", choices=("global", "per-fold"), default="global",
                        help="how the probe picks C. 'global' (default) averages every "
                             "fold's inner CV across folds and fits all ten folds with the "
                             "one winner, which is Lior's cell-16 protocol and the only "
                             "row comparable to his arm. 'per-fold' lets each fold pick "
                             "its own C -- ten chances at a good regularization strength "
                             "where his arm gets one -- and is a sensitivity check only.")

    parser.add_argument("--pseudo-labels", default=None,
                        help="cached propagated pseudo-labels (.npz) -- cnn mode. "
                             "Without it, cnn mode propagates from --embeddings at --k.")
    parser.add_argument("--k", type=int, default=None,
                        help="LOCKED K for the on-the-fly propagation in cnn mode. There is "
                             "no default on purpose: K is a selected hyperparameter and it "
                             "must come from the sweep's selection.json, not from a guess.")
    parser.add_argument("--sigma", type=float, default=None,
                        help="heat-kernel bandwidth; default is self-tuning per graph "
                             "(see knn.heat_kernel_weights)")
    parser.add_argument("--lp-max-iterations", type=int, default=config.LP_MAX_ITERATIONS,
                        help="label-propagation iteration cap")
    parser.add_argument("--lp-tolerance", type=float, default=config.LP_TOLERANCE,
                        help="convergence tolerance on max|F_next - F|")
    parser.add_argument("--lp-method", choices=("spread", "clamp"), default="spread",
                        help="spread (default): Zhou label spreading. clamp: "
                             "Zhu-Ghahramani hard clamping, whose fixed point is nearly "
                             "flat on this graph -- kept only to reproduce the negative "
                             "result. MUST match what sweep_knn_lp.py selected K on.")
    parser.add_argument("--lp-alpha", type=float, default=config.LP_ALPHA,
                        help="restart weight for --lp-method spread; ignored by clamp")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="VAE (or trunk) checkpoint to initialise the CNN's encoder")
    parser.add_argument("--confidence-weighting", action="store_true",
                        help="weight the loss by the propagation confidence")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="drop pseudo-labels below this confidence entirely")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="train only the classification head")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--load-images-to-ram", action="store_true",
                        help="hold the 2.76 GB unlabeled split resident instead of "
                             "memory-mapping it (the server default should be on)")
    parser.add_argument("--log-every", type=int, default=1,
                        help="print every Nth epoch in cnn mode")

    arguments = parser.parse_args(argv)

    if arguments.mode == "probe" and arguments.embeddings is None:
        parser.error("--mode probe requires --embeddings")
    if arguments.mode == "cnn" and arguments.pseudo_labels is None:
        # No cache, so the labels have to be propagated here -- which needs the
        # graph embeddings and the K the sweep locked.
        if arguments.embeddings is None:
            parser.error("--mode cnn requires either --pseudo-labels (a cache) or "
                         "--embeddings plus --k (propagate here)")
        if arguments.k is None:
            parser.error("--mode cnn without --pseudo-labels requires --k, the K selected "
                         "on the development split by scripts/sweep_knn_lp.py")
    if arguments.name is None:
        arguments.name = "downstream_" + arguments.mode
    # Guarded because the epoch printer takes a modulo by it.
    arguments.log_every = max(1, arguments.log_every)

    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)

    # Both calls, in this order, before anything touches the GPU:
    # enable_deterministic sets CUBLAS_WORKSPACE_CONFIG, which only has an effect
    # before the CUDA context exists.
    seeding.enable_deterministic()
    seeding.reset_seed(arguments.seed)

    device = resolve_device(arguments.device)
    run_directory = create_run_directory(arguments.name)

    log("mode           " + arguments.mode)
    log("seed           " + str(arguments.seed))
    log("device         " + str(device))
    log("run directory  " + str(run_directory))
    log("TEST SPLIT     not touched by this script")
    log("")

    write_json(run_directory / "arguments.json", vars(arguments))

    if arguments.mode == "probe":
        run_probe_mode(arguments, run_directory, device)
    else:
        run_cnn_mode(arguments, run_directory, device)

    log("")
    log("done -> " + str(run_directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
