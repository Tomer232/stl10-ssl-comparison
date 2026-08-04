"""The K sweep: build the KNN graph, propagate labels, select K on dev. Never on test.

    # K sweep for one checkpoint
    python scripts/sweep_knn_lp.py --embeddings runs/embeddings_.../embeddings.npy

    # beta x K ablation -- same code path, one embedding file per beta
    python scripts/sweep_knn_lp.py \
        --embeddings runs/beta0.001/embeddings.npy runs/beta0.01/embeddings.npy \
        --labels beta=0.001 beta=0.01

This is the heart of Tomer's arm. For every K in the grid it builds the cosine
KNN graph over the cached 105k x 512 embeddings ONCE, and then runs label
propagation ten times -- once per official STL-10 fold -- seeding from that
fold's 800 dev-train images and scoring on its 200 dev-validation images.

============================================================================
THE TEST SPLIT IS NEVER LOADED, NEVER READ, NEVER SCORED BY THIS SCRIPT.
============================================================================
Model selection happens entirely on the 200 held-out dev-validation nodes
inside each fold, exactly as in Lior's SimCLR arm. The 8000 test images are
touched once, at the very end, by the final evaluation script. The only labels
this script reads are those of the 5000-image labeled TRAIN split.

WHY THE GRAPH IS BUILT ONCE PER K AND NOT ONCE PER (K, FOLD)
------------------------------------------------------------
The graph depends only on the embeddings and K -- not on which nodes happen to
be labeled. The folds differ only in the seed set handed to the propagation, so
rebuilding the graph per fold would repeat a 105k-node brute-force search ten
times for an identical result. The per-fold cost is then just the propagation
itself: a handful of sparse matmuls on a [105000, 10] matrix.

WHAT GETS RECORDED, AND WHY IT IS MORE THAN ACCURACY
-----------------------------------------------------
CLAUDE.md requires connected-component coverage and edge purity next to
accuracy, because accuracy alone cannot distinguish a bad embedding from a
fragmented graph. A small K can look respectable on purity while shattering the
graph into islands no label can reach; a large K keeps everything connected
while bridging classes through bad edges. The three numbers together are the
argument for whichever K we pick, and when they disagree the script says so
out loud -- that disagreement is a reportable finding, not a problem to hide.

Runs on CPU (for tests, with --device cpu and a small subset) and on CUDA.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy
import torch

# Repo root on sys.path so "python scripts/sweep_knn_lp.py" works from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src import data
from src import knn
from src import labelprop
from src import metrics
from src import unionfind
from src.seeding import enable_deterministic, reset_seed


# Columns of per_fold.csv, in the order they are written. Kept explicit rather
# than derived from a dict so the CSV schema is a visible, reviewable thing.
PER_FOLD_COLUMNS = [
    "embedding_label", "beta", "k", "fold",
    "dev_accuracy", "dev_macro_f1", "dev_accuracy_reachable",
    "dev_unreachable_count", "dev_size",
    "edge_purity", "edge_purity_fold",
    "num_components", "largest_component_fraction", "component_label_coverage",
    "unreachable_count", "unreachable_fraction",
    "iterations", "final_delta", "converged", "confidence_mean",
    "sigma", "graph_seconds", "fold_seconds",
    "num_nodes", "num_seed_nodes", "embeddings_path",
]

# The columns aggregated to mean +/- std over the 10 folds. "converged" is a bool
# per fold and averages to the fraction of folds that reached the tolerance, which
# is the number that says whether a cell's accuracy is a fixed point or a snapshot
# of a still-moving iteration.
AGGREGATED_METRICS = [
    "dev_accuracy", "dev_macro_f1", "dev_accuracy_reachable", "dev_unreachable_count",
    "edge_purity", "edge_purity_fold", "component_label_coverage",
    "num_components", "largest_component_fraction",
    "unreachable_count", "unreachable_fraction",
    "iterations", "final_delta", "converged", "confidence_mean",
    "graph_seconds", "fold_seconds",
]

# Two mean metrics closer than this are treated as tied. Without it, K values that
# score identically (component coverage is 1.0 everywhere on a well-connected
# graph) would trip the "the diagnostics disagree" warning purely because max()
# has to return one of them, and a warning that fires on a tie is a warning the
# reader learns to ignore.
TIE_TOLERANCE = 1e-9

# Fewest members a fold may keep and still be split into a seed set and a dev set.
# Only ever binds on a --max-images-per-split subset run; the real folds hold 1000.
MINIMUM_FOLD_MEMBERS = 2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sweep K for the KNN graph + label propagation, selecting on dev only.")

    parser.add_argument("--embeddings", nargs="+", required=True,
                        help="One or more cached embedding .npy files written by "
                             "scripts/extract_embeddings.py. More than one puts the "
                             "script in beta-sweep mode and produces a beta x K table.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display label per embedding file (e.g. beta=0.01). "
                             "Defaults to the manifest's beta_max, else the file stem.")
    parser.add_argument("--manifests", nargs="+", default=None,
                        help="Manifest per embedding file. Defaults to "
                             "<stem>.manifest.json next to the array.")

    parser.add_argument("--data-root", default=str(config.DATA_ROOT),
                        help="Directory holding stl10_binary/ -- only the labeled train "
                             "labels and fold_indices.txt are read from it.")
    parser.add_argument("--splits", default=None,
                        help="The .npz from scripts/prepare_data.py. Defaults to "
                             "data/splits.npz when that exists -- both arms are supposed to "
                             "read the same file. Only when it does not exist are the folds "
                             "and their 800/200 dev splits derived in-process, with exactly "
                             "the same rule (seed + fold_index, stratified 80/20).")

    parser.add_argument("--k-grid", nargs="+", type=int, default=list(config.K_GRID),
                        help="K values to sweep (default config.K_GRID).")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Heat-kernel bandwidth. Default: self-tuning per graph, see "
                             "knn.heat_kernel_weights.")
    parser.add_argument("--chunk-rows", type=int, default=config.SIMILARITY_CHUNK_ROWS,
                        help="Rows of the similarity matrix per chunk.")
    parser.add_argument("--max-iterations", type=int, default=config.LP_MAX_ITERATIONS,
                        help="Label-propagation iteration cap.")
    parser.add_argument("--tolerance", type=float, default=config.LP_TOLERANCE,
                        help="Convergence tolerance on max|F_next - F|.")

    parser.add_argument("--device", default=None,
                        help="cuda, cuda:0, cpu. Default: cuda when available, else cpu.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Seed for reset_seed and for the per-fold 800/200 dev splits "
                             "(fold i uses seed + i, mirroring Lior).")
    parser.add_argument("--validation-fraction", type=float, default=0.20,
                        help="Dev-validation share inside each fold (0.20 -> 800/200).")

    parser.add_argument("--run-name", default=None,
                        help="Prefix of the run directory under runs/. Default "
                             "knn_lp_sweep, or beta_knn_lp_sweep in beta-sweep mode.")
    parser.add_argument("--output-directory", default=None,
                        help="Write here instead of runs/<run-name>_<timestamp>/.")
    parser.add_argument("--allow-nondeterministic", action="store_true",
                        help="Skip seeding.enable_deterministic(). Only for the case where a "
                             "kernel on this machine has no deterministic implementation and "
                             "torch refuses to run at all -- results stop being bit-reproducible, "
                             "and the run is flagged as such in every output file.")

    return parser.parse_args()


def resolve_device(requested):
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_splits_path(requested):
    """Prefer the splits file prepare_data.py wrote; derive them only if it is absent.

    A split recomputed in three scripts is a split that will eventually disagree in
    one of them, and the entire two-arm comparison rests on both arms scoring the
    same held-out images. So the shared .npz wins whenever it exists, and an
    explicitly requested path that is missing is an error rather than a silent
    fallback to a locally derived split that might not match.
    """
    if requested is not None:
        if not os.path.isfile(requested):
            raise FileNotFoundError(
                "--splits %s does not exist. Run scripts/prepare_data.py, or omit the flag "
                "to derive the folds in-process." % (requested,))
        return requested

    default_path = config.DATA_ROOT / "splits.npz"
    if default_path.is_file():
        return str(default_path)
    return None


def resolve_manifest_path(embeddings_path, override):
    """Find the manifest that describes an embedding array.

    The manifest is not optional metadata: it carries the split ordering and the
    labeled-train offset, and every fold index in this script is turned into a
    graph node index by adding that offset. Guessing 100000 would be wrong for any
    subset run and silently scramble the labels, so a missing manifest is a hard
    error rather than a default.
    """
    if override is not None:
        return Path(override)

    embeddings_path = Path(embeddings_path)
    stem_manifest = embeddings_path.with_suffix(".manifest.json")
    if stem_manifest.is_file():
        return stem_manifest

    directory_manifest = embeddings_path.parent / "manifest.json"
    if directory_manifest.is_file():
        return directory_manifest

    raise FileNotFoundError(
        "No manifest found for %s (looked for %s and %s). The manifest carries the split "
        "ordering and the labeled-train offset, which this script must not guess."
        % (embeddings_path, stem_manifest, directory_manifest))


def load_embedding_sets(arguments):
    """Load every embedding array plus its manifest, and check they are comparable."""
    if arguments.labels is not None and len(arguments.labels) != len(arguments.embeddings):
        raise ValueError("--labels must have one entry per --embeddings file")
    if arguments.manifests is not None and len(arguments.manifests) != len(arguments.embeddings):
        raise ValueError("--manifests must have one entry per --embeddings file")

    embedding_sets = []
    for position, embeddings_path in enumerate(arguments.embeddings):
        embeddings_path = Path(embeddings_path)
        manifest_override = arguments.manifests[position] if arguments.manifests else None
        manifest_path = resolve_manifest_path(embeddings_path, manifest_override)

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("tap_point") != "mu":
            raise ValueError(
                "%s was extracted from tap point %r, not mu. The graph must be built on the "
                "posterior mean; a sample would change the edge set between runs."
                % (manifest_path, manifest.get("tap_point")))

        array = numpy.load(embeddings_path)
        if array.shape[0] != manifest["num_nodes"]:
            raise ValueError("%s holds %d rows but its manifest says %d"
                             % (embeddings_path, array.shape[0], manifest["num_nodes"]))

        beta = manifest.get("beta_max")
        if arguments.labels is not None:
            label = arguments.labels[position]
        elif beta is not None:
            label = "beta=%g" % (float(beta),)
        else:
            label = embeddings_path.parent.name + "/" + embeddings_path.stem

        embedding_sets.append({
            "label": label,
            "beta": beta,
            "path": str(embeddings_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest": manifest,
            "array": array,
        })

    # A beta x K table only means anything if every cell was scored on the same
    # nodes in the same order. Different node counts or a different split
    # ordering would make the rows incomparable, so refuse rather than warn.
    reference = embedding_sets[0]
    for embedding_set in embedding_sets[1:]:
        for key in ("num_nodes", "split_order", "labeled_train_offset", "embedding_dimension"):
            if embedding_set["manifest"].get(key) != reference["manifest"].get(key):
                raise ValueError(
                    "Embedding sets disagree on %r: %r has %r, %r has %r. They cannot share "
                    "one table." % (key, reference["label"], reference["manifest"].get(key),
                                    embedding_set["label"], embedding_set["manifest"].get(key)))

    return embedding_sets


def load_train_labels(data_root):
    """Labels of the 5000-image labeled TRAIN split, and nothing else.

    Read straight from train_y.bin rather than through data.load_split so we do
    not pull 138 MB of pixels into RAM: this script works off the cached
    embeddings and never needs an image. The test labels are not read at all.
    """
    labels_path = os.path.join(data.resolve_binary_directory(data_root), "train_y.bin")
    return data.load_labels(labels_path)


def build_fold_definitions(arguments, train_count):
    """The 10 official folds, each split 800/200 into dev-train and dev-validation.

    Either loaded from the .npz written by data.build_and_save_splits (preferred
    when it exists, because both arms then read the same file) or derived here
    with the identical rule: fold i is split with seed + i, stratified, 80/20.

    Every index returned is an index into the 5000-image labeled train split. The
    caller adds the manifest's labeled_train_offset to reach a graph node.
    """
    if arguments.splits is not None:
        stored = data.load_splits(arguments.splits)
        official_folds = stored["official_folds"]
        train_labels = stored["train_labels"]
        development_splits = stored["development_splits"]
        source = arguments.splits
    else:
        official_folds = data.load_official_folds(arguments.data_root)
        train_labels = load_train_labels(arguments.data_root)
        development_splits = None
        source = "derived in-process (seed + fold_index, stratified 80/20)"

    fold_definitions = []
    dropped_folds = []

    for fold_index in range(len(official_folds)):
        fold_members = official_folds[fold_index]

        # Smoke-test path only: with --max-images-per-split the cached array holds
        # fewer than 5000 labeled nodes, so fold members past the end have no
        # embedding. Drop them and re-derive the dev split over what is left.
        truncated = train_count < len(train_labels)
        if truncated:
            fold_members = fold_members[fold_members < train_count]

        # A fold that lost almost all its members to that truncation cannot be
        # split into a seed set and a dev set at all, and data.stratified_split
        # raises an obscure "need at least one array to concatenate" on it. Drop
        # such folds explicitly rather than crashing: this only ever happens on a
        # subset run, which is a pipeline check and not a result.
        if len(fold_members) < MINIMUM_FOLD_MEMBERS:
            dropped_folds.append(fold_index)
            continue

        if development_splits is not None and not truncated:
            train_positions, validation_positions = development_splits[fold_index]
        else:
            train_positions, validation_positions = data.stratified_split(
                train_labels[fold_members],
                validation_fraction=arguments.validation_fraction,
                seed=arguments.seed + fold_index,
            )

        fold_definitions.append({
            "fold": fold_index,
            "seed_indices": fold_members[train_positions],
            "seed_labels": train_labels[fold_members[train_positions]],
            "dev_indices": fold_members[validation_positions],
            "dev_labels": train_labels[fold_members[validation_positions]],
        })

    if dropped_folds:
        print("  *** WARNING: folds %s were dropped -- the cached embeddings hold only %d "
              "labeled nodes, so those folds kept fewer than %d members. This is a subset "
              "run and its numbers are not results. ***"
              % (dropped_folds, train_count, MINIMUM_FOLD_MEMBERS), flush=True)

    if not fold_definitions:
        raise ValueError(
            "Every one of the %d official folds was empty after restricting them to the %d "
            "labeled nodes present in the embedding cache. Extract embeddings without "
            "--max-images-per-split, or raise the cap high enough that folds survive."
            % (len(official_folds), train_count))

    return fold_definitions, train_labels, source


def build_node_label_arrays(train_labels, num_nodes, labeled_train_offset, train_count):
    """[N] true labels with -1 on unlabeled nodes, plus the [N] labeled mask.

    Used only for the edge-purity diagnostic, which needs to know the class of
    both endpoints of an edge. -1 marks "class unknown"; metrics.edge_purity is
    given the mask and never scores an edge touching one of those nodes.
    """
    node_labels = numpy.full(num_nodes, -1, dtype=numpy.int64)
    node_labels[labeled_train_offset:labeled_train_offset + train_count] = \
        train_labels[:train_count]

    labeled_mask = numpy.zeros(num_nodes, dtype=bool)
    labeled_mask[labeled_train_offset:labeled_train_offset + train_count] = True

    return node_labels, labeled_mask


def evaluate_fold(propagation_matrix, fold, labeled_train_offset, num_nodes,
                  neighbor_indices, node_labels, component_ids, arguments, device):
    """Propagate from one fold's 800 seeds and score its 200 dev-validation nodes.

    Returns a dict of the per-(K, fold) numbers. Nothing here reads the test
    split; `fold["dev_indices"]` are held-out members of the labeled TRAIN split.
    """
    started_at = time.time()

    seed_nodes = torch.as_tensor(
        fold["seed_indices"].astype(numpy.int64) + labeled_train_offset, device=device)
    seed_labels = torch.as_tensor(fold["seed_labels"].astype(numpy.int64), device=device)
    dev_nodes = torch.as_tensor(
        fold["dev_indices"].astype(numpy.int64) + labeled_train_offset, device=device)
    dev_true_labels = torch.as_tensor(fold["dev_labels"].astype(numpy.int64), device=device)

    initial_F = labelprop.initial_label_matrix(
        num_nodes, config.NUM_CLASSES, seed_nodes, seed_labels, device)
    # The clamp target is exactly what initial_label_matrix just wrote at the seed
    # rows, so the two cannot drift apart.
    labeled_onehot = initial_F[seed_nodes]

    final_F, iterations, final_delta = labelprop.propagate(
        propagation_matrix, initial_F, seed_nodes, labeled_onehot,
        arguments.max_iterations, arguments.tolerance)

    pseudo_labels, _ = labelprop.pseudo_labels_from(final_F)

    # The graph-wide diagnostics are computed on CPU copies. F is only
    # [num_nodes, 10] floats (~4 MB), and propagation_summary uses torch.bincount,
    # which has no deterministic CUDA kernel -- under enable_deterministic() the
    # CUDA version raises rather than running. Same reason knn._row_sums falls
    # back to the host.
    summary = labelprop.propagation_summary(final_F.cpu(), pseudo_labels.cpu())

    dev_predictions = pseudo_labels[dev_nodes]

    # A dev node in a component with no seed never receives any mass and comes
    # back as -1. It cannot go into the confusion matrix (metrics refuses the
    # sentinel, correctly -- folding it into class 0 would inflate that class),
    # so it is excluded from the matrix and counted as an error afterwards.
    dev_reachable = dev_predictions >= 0
    dev_size = int(dev_predictions.numel())
    dev_reachable_count = int(dev_reachable.sum().item())

    confusion = metrics.confusion_matrix(
        dev_true_labels[dev_reachable].cpu(),
        dev_predictions[dev_reachable].cpu(),
        config.NUM_CLASSES)

    accuracy_reachable = metrics.accuracy(confusion)
    macro_f1 = metrics.macro_f1(confusion)

    # Everything is read off the one bincount confusion matrix, per build-from-0:
    # accuracy over ALL dev nodes is the reachable accuracy scaled by the share of
    # dev nodes that were reachable, i.e. unreachable nodes count as wrong.
    dev_accuracy = accuracy_reachable * (dev_reachable_count / dev_size) if dev_size else 0.0

    # Component coverage is fold-dependent: it asks what fraction of nodes sit in
    # a component holding at least one of THIS fold's 800 seeds.
    seeded_mask = numpy.zeros(num_nodes, dtype=bool)
    seeded_mask[fold["seed_indices"].astype(numpy.int64) + labeled_train_offset] = True
    coverage = metrics.component_label_coverage(component_ids, seeded_mask)

    # Edge purity restricted to this fold's own 1000 labeled members. Far fewer
    # countable edges than the all-labeled version below, hence noisier, but it
    # uses only labels this fold was allowed to see.
    fold_mask = numpy.zeros(num_nodes, dtype=bool)
    fold_mask[fold["seed_indices"].astype(numpy.int64) + labeled_train_offset] = True
    fold_mask[fold["dev_indices"].astype(numpy.int64) + labeled_train_offset] = True
    fold_edge_purity = metrics.edge_purity(neighbor_indices, node_labels, valid_mask=fold_mask)

    return {
        "dev_accuracy": dev_accuracy,
        "dev_macro_f1": macro_f1,
        "dev_accuracy_reachable": accuracy_reachable,
        "dev_unreachable_count": dev_size - dev_reachable_count,
        "dev_size": dev_size,
        "edge_purity_fold": fold_edge_purity,
        "component_label_coverage": coverage,
        "unreachable_count": summary["unreachable_count"],
        "unreachable_fraction": summary["unreachable_fraction"],
        "iterations": iterations,
        "final_delta": final_delta,
        "converged": bool(final_delta < arguments.tolerance),
        "confidence_mean": summary["confidence_mean"],
        "num_seed_nodes": int(seed_nodes.numel()),
        "fold_seconds": time.time() - started_at,
    }


def aggregate_rows(rows, group_keys, metric_names):
    """Mean and sample std (ddof=1) of each metric, grouped by `group_keys`.

    ddof=1 because the 10 folds are a sample, not the population -- the same
    convention as the mean +/- std Lior reports over his folds.
    """
    groups = {}
    order = []
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    summary_rows = []
    for key in order:
        group = groups[key]
        entry = dict(zip(group_keys, key))
        entry["num_folds"] = len(group)
        for metric_name in metric_names:
            values = numpy.array([float(row[metric_name]) for row in group],
                                 dtype=numpy.float64)
            entry[metric_name + "_mean"] = float(values.mean())
            entry[metric_name + "_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(entry)

    return summary_rows


def write_csv(path, columns, rows):
    """Tidy CSV, one row per record, columns in the given order."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary_table(summary_rows, beta_sweep_mode):
    """Print the mean +/- std table the report's K-sweep figure is drawn from."""
    header = ("%-14s %5s   %-16s %-16s %-14s %-14s %8s %7s %6s"
              % ("embedding", "K", "dev acc", "dev macroF1", "edge purity",
                 "coverage", "unreach", "iters", "conv"))
    print("", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for row in summary_rows:
        label = row["embedding_label"] if beta_sweep_mode else ""
        # "conv" is the fraction of the 10 folds that reached the tolerance before
        # the iteration cap. Anything below 1.00 means the accuracy on that row is
        # a snapshot of an iteration that was still moving.
        print("%-14s %5d   %6.4f +/-%.4f %6.4f +/-%.4f %6.4f +/-%.3f %6.4f +/-%.3f %8.0f %7.1f %6.2f"
              % (label[:14], row["k"],
                 row["dev_accuracy_mean"], row["dev_accuracy_std"],
                 row["dev_macro_f1_mean"], row["dev_macro_f1_std"],
                 row["edge_purity_mean"], row["edge_purity_std"],
                 row["component_label_coverage_mean"], row["component_label_coverage_std"],
                 row["unreachable_count_mean"], row["iterations_mean"],
                 row["converged_mean"]),
              flush=True)
    print("", flush=True)


def pivot_rows(summary_rows, metric_name, k_grid):
    """Reshape the tidy summary into one row per embedding set, one column per K.

    The tidy summary.csv is the machine-readable artifact, but the beta ablation's
    actual question -- "does the best K move as beta changes, and does the whole
    curve shift?" -- is a two-dimensional one, and reading it off a long-format
    table means holding 36 rows in your head. This is the same numbers pivoted so
    the answer is visible, and it is what the report's beta x K table is drawn
    from.
    """
    labels = []
    for row in summary_rows:
        if row["embedding_label"] not in labels:
            labels.append(row["embedding_label"])

    columns = ["embedding_label", "beta"] + ["K=%d" % (k,) for k in k_grid]
    rows = []

    for label in labels:
        entry = {"embedding_label": label, "beta": None}
        for k in k_grid:
            matches = [row for row in summary_rows
                       if row["embedding_label"] == label and row["k"] == k]
            if matches:
                entry["beta"] = matches[0]["beta"]
                entry["K=%d" % (k,)] = matches[0][metric_name + "_mean"]
            else:
                entry["K=%d" % (k,)] = float("nan")
        rows.append(entry)

    return columns, rows


def print_pivot_table(columns, rows, title):
    """Print a pivot produced by `pivot_rows`."""
    print("", flush=True)
    print(title, flush=True)
    header = "%-16s" % ("embedding",) + "".join("%10s" % (name,) for name in columns[2:])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        print("%-16s" % (str(row["embedding_label"])[:16],)
              + "".join("%10.4f" % (row[name],) for name in columns[2:]), flush=True)
    print("", flush=True)


def best_row_by(summary_rows, metric_name):
    """The summary row with the largest mean of `metric_name`.

    Ties go to whichever row comes first, and rows are appended in K order, so a
    tie resolves to the SMALLEST K. That is the right default: a smaller K is a
    sparser graph and a cheaper search, and nothing is gained by preferring the
    larger one when the metric cannot tell them apart.
    """
    return max(summary_rows, key=lambda row: row[metric_name + "_mean"])


def build_selection(summary_rows, embedding_sets, k_grid, tolerance, max_iterations):
    """Pick K by mean dev accuracy and record where the other diagnostics disagree.

    CLAUDE.md is explicit that the best-LP-accuracy K need not be the best K on
    edge purity or component coverage, and that the disagreement is worth
    reporting. So the selection rule stays simple and stated up front -- highest
    mean dev accuracy -- and any disagreement is recorded next to it rather than
    resolved by a composite score nobody could defend.
    """
    warnings = []
    per_embedding = {}

    for embedding_set in embedding_sets:
        label = embedding_set["label"]
        rows = [row for row in summary_rows if row["embedding_label"] == label]
        if not rows:
            continue

        accuracy_row = best_row_by(rows, "dev_accuracy")
        purity_row = best_row_by(rows, "edge_purity")
        coverage_row = best_row_by(rows, "component_label_coverage")

        disagreements = []

        # A disagreement counts only when the other K is genuinely better on that
        # diagnostic, not merely first past an exact tie -- see TIE_TOLERANCE.
        for metric_name, description, best_row in (
            ("edge_purity", "edge purity", purity_row),
            ("component_label_coverage", "component coverage", coverage_row),
        ):
            selected_value = accuracy_row[metric_name + "_mean"]
            best_value = best_row[metric_name + "_mean"]
            if best_row["k"] == accuracy_row["k"] or best_value <= selected_value + TIE_TOLERANCE:
                continue
            message = ("%s: best dev accuracy at K=%d (%.4f) but best %s at K=%d "
                       "(%.4f there vs %.4f at the selected K)"
                       % (label, accuracy_row["k"], accuracy_row["dev_accuracy_mean"],
                          description, best_row["k"], best_value, selected_value))
            disagreements.append(message)
            warnings.append(message)

        # A cell whose folds hit the iteration cap has not reached the harmonic
        # fixed point, so its accuracy is a snapshot of a still-moving iteration.
        # That has to be visible next to the selected K rather than buried in
        # per_fold.csv, because it changes how the number should be read.
        unconverged = [row for row in rows if row["converged_mean"] < 1.0]
        if unconverged:
            message = ("%s: label propagation did not converge in every fold at K=%s "
                       "(tolerance %g, cap %d iterations). Their accuracies are snapshots "
                       "of a still-moving iteration -- raise --max-iterations or loosen "
                       "--tolerance before quoting them."
                       % (label, [int(row["k"]) for row in unconverged],
                          tolerance, max_iterations))
            disagreements.append(message)
            warnings.append(message)

        per_embedding[label] = {
            "beta": embedding_set["beta"],
            "embeddings_path": embedding_set["path"],
            "best_k": int(accuracy_row["k"]),
            "mean_dev_accuracy": accuracy_row["dev_accuracy_mean"],
            "std_dev_accuracy": accuracy_row["dev_accuracy_std"],
            "mean_dev_macro_f1": accuracy_row["dev_macro_f1_mean"],
            "mean_edge_purity": accuracy_row["edge_purity_mean"],
            "mean_component_label_coverage": accuracy_row["component_label_coverage_mean"],
            "mean_unreachable_count": accuracy_row["unreachable_count_mean"],
            "converged_fraction_at_best_k": accuracy_row["converged_mean"],
            "mean_iterations_at_best_k": accuracy_row["iterations_mean"],
            "best_k_by_edge_purity": int(purity_row["k"]),
            "best_k_by_component_coverage": int(coverage_row["k"]),
            "disagreements": disagreements,
        }

    overall_row = best_row_by(summary_rows, "dev_accuracy")

    return {
        "selection_rule": ("highest mean dev accuracy over the 10 official folds, measured on "
                           "each fold's 200 dev-validation nodes; ties go to the smaller K"),
        "test_split_touched": False,
        "k_grid": [int(k) for k in k_grid],
        "lp_tolerance": tolerance,
        "lp_max_iterations": max_iterations,
        "per_embedding": per_embedding,
        "best_overall": {
            "embedding_label": overall_row["embedding_label"],
            "k": int(overall_row["k"]),
            "mean_dev_accuracy": overall_row["dev_accuracy_mean"],
            "std_dev_accuracy": overall_row["dev_accuracy_std"],
        },
        "warnings": warnings,
    }


def create_output_directory(arguments, beta_sweep_mode, timestamp):
    """Make the run directory, refusing to write into one that already has results."""
    if arguments.output_directory is not None:
        directory = Path(arguments.output_directory)
    else:
        default_name = "beta_knn_lp_sweep" if beta_sweep_mode else "knn_lp_sweep"
        run_name = arguments.run_name if arguments.run_name else default_name
        directory = config.RUNS_ROOT / ("%s_%s" % (run_name, timestamp))

    config.ensure_directory(directory)
    for name in ("per_fold.csv", "summary.csv", "graphs.csv", "selection.json",
                 "pivot_dev_accuracy.csv"):
        if (directory / name).exists():
            raise FileExistsError(
                "%s already exists. Refusing to overwrite an existing sweep -- pass a "
                "different --output-directory." % (directory / name,))
    return directory


def main():
    arguments = parse_arguments()

    reset_seed(arguments.seed)
    if arguments.allow_nondeterministic:
        print("WARNING: --allow-nondeterministic given; deterministic kernels are NOT enforced "
              "and this sweep is not bit-reproducible.", flush=True)
    else:
        enable_deterministic()

    # See the TF32 note in src/knn.py: a truncated mantissa manufactures near-ties
    # in the cosine similarities, which changes the top-K neighbour lists between
    # machines and therefore the graph the whole experiment rests on.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = resolve_device(arguments.device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    started_at = time.time()

    embedding_sets = load_embedding_sets(arguments)
    beta_sweep_mode = len(embedding_sets) > 1
    output_directory = create_output_directory(arguments, beta_sweep_mode, timestamp)

    reference_manifest = embedding_sets[0]["manifest"]
    num_nodes = int(reference_manifest["num_nodes"])
    labeled_train_offset = reference_manifest.get("labeled_train_offset")
    if labeled_train_offset is None:
        raise ValueError("The manifest has no labeled_train_offset; this script cannot map a "
                         "fold index onto a graph node without it.")
    labeled_train_offset = int(labeled_train_offset)

    train_count = None
    for record in reference_manifest["split_offsets"]:
        if record["name"] == "train":
            train_count = int(record["count"])
    if train_count is None:
        raise ValueError("The manifest lists no 'train' split; the graph has no labeled nodes.")

    print("=" * 96, flush=True)
    print("K sweep: KNN graph + label propagation, selected on dev only", flush=True)
    print("  torch            %s" % (torch.__version__,), flush=True)
    print("  device           %s" % (device,), flush=True)
    print("  embedding sets   %d %s" % (len(embedding_sets),
                                        [s["label"] for s in embedding_sets]), flush=True)
    print("  nodes            %d (labeled train at offset %d, count %d)"
          % (num_nodes, labeled_train_offset, train_count), flush=True)
    print("  K grid           %s" % (arguments.k_grid,), flush=True)
    print("  output           %s" % (output_directory,), flush=True)
    print("  TEST SPLIT       not loaded, not scored -- selection is dev-only", flush=True)
    print("=" * 96, flush=True)

    arguments.splits = resolve_splits_path(arguments.splits)
    fold_definitions, train_labels, split_source = build_fold_definitions(
        arguments, train_count)
    print("Folds from: %s" % (split_source,), flush=True)
    print("Fold 0: %d dev-train seeds, %d dev-validation nodes"
          % (len(fold_definitions[0]["seed_indices"]),
             len(fold_definitions[0]["dev_indices"])), flush=True)

    node_labels, labeled_mask = build_node_label_arrays(
        train_labels, num_nodes, labeled_train_offset, train_count)

    per_fold_rows = []
    graph_rows = []

    for embedding_set in embedding_sets:
        print("", flush=True)
        print("### embedding set %s  (%s)" % (embedding_set["label"], embedding_set["path"]),
              flush=True)

        # fp16 on disk -> fp32 on the compute device. Passing a device tensor
        # (rather than the numpy array) is what keeps the graph, P and F all on
        # the same device; knn.build_knn_graph returns its output wherever the
        # input embeddings live.
        features = torch.from_numpy(embedding_set["array"]).to(
            device=device, dtype=torch.float32)

        # Explicit, though build_propagation_matrix normalizes internally too --
        # l2_normalize is idempotent, and doing it here makes "the graph is built
        # on unit-norm mu, so a dot product IS a cosine" visible at the call site.
        normalized_features = knn.l2_normalize(features, device=device)
        del features

        for k in arguments.k_grid:
            if k > num_nodes - 1:
                raise ValueError("K=%d exceeds N-1=%d" % (k, num_nodes - 1))

            print("  K=%d: building graph..." % (k,), flush=True)
            graph_started_at = time.time()
            propagation_matrix, neighbor_indices, _, sigma = knn.build_propagation_matrix(
                normalized_features, k,
                sigma=arguments.sigma,
                chunk_rows=arguments.chunk_rows,
                device=device)
            graph_seconds = time.time() - graph_started_at

            # Components depend on the graph only, not on which nodes are seeded,
            # so this runs once per K rather than once per fold.
            component_started_at = time.time()
            component_ids, number_of_components = unionfind.connected_components(
                neighbor_indices, num_nodes)
            component_sizes = numpy.bincount(component_ids)
            largest_component_fraction = float(component_sizes.max()) / num_nodes
            component_seconds = time.time() - component_started_at

            # Edge purity over ALL labeled train nodes: a property of the graph,
            # not of a fold, so it is computed once and repeated on each fold's
            # row. The fold-restricted version (edge_purity_fold) is computed per
            # fold, but with only 1000 labeled members it counts far fewer edges
            # and is correspondingly noisy.
            all_labeled_edge_purity = metrics.edge_purity(
                neighbor_indices, node_labels, valid_mask=labeled_mask)

            print("    sigma=%.5f  components=%d (largest %.4f of nodes)  purity=%.4f  "
                  "graph %.1fs + components %.1fs"
                  % (sigma, number_of_components, largest_component_fraction,
                     all_labeled_edge_purity, graph_seconds, component_seconds), flush=True)

            graph_rows.append({
                "embedding_label": embedding_set["label"],
                "beta": embedding_set["beta"],
                "k": k,
                "sigma": sigma,
                "edge_purity": all_labeled_edge_purity,
                "num_components": number_of_components,
                "largest_component_fraction": largest_component_fraction,
                "graph_seconds": graph_seconds,
                "component_seconds": component_seconds,
                "num_nodes": num_nodes,
                "embeddings_path": embedding_set["path"],
            })

            for fold in fold_definitions:
                fold_result = evaluate_fold(
                    propagation_matrix, fold, labeled_train_offset, num_nodes,
                    neighbor_indices, node_labels, component_ids, arguments, device)

                row = {
                    "embedding_label": embedding_set["label"],
                    "beta": embedding_set["beta"],
                    "k": k,
                    "fold": fold["fold"],
                    "edge_purity": all_labeled_edge_purity,
                    "num_components": number_of_components,
                    "largest_component_fraction": largest_component_fraction,
                    "sigma": sigma,
                    "graph_seconds": graph_seconds,
                    "num_nodes": num_nodes,
                    "embeddings_path": embedding_set["path"],
                }
                row.update(fold_result)
                per_fold_rows.append(row)

                print("    fold %d: dev acc %.4f  macroF1 %.4f  coverage %.4f  "
                      "unreachable %d  iters %d%s  %.1fs"
                      % (fold["fold"], row["dev_accuracy"], row["dev_macro_f1"],
                         row["component_label_coverage"], row["unreachable_count"],
                         row["iterations"], "" if row["converged"] else " (NOT CONVERGED)",
                         row["fold_seconds"]), flush=True)

            # Free the graph before the next K allocates its own; at K=50 the
            # sparse P alone is a few hundred MB.
            del propagation_matrix, neighbor_indices, component_ids

        del normalized_features

    summary_rows = aggregate_rows(
        per_fold_rows, ["embedding_label", "beta", "k"], AGGREGATED_METRICS)

    print_summary_table(summary_rows, beta_sweep_mode)

    selection = build_selection(summary_rows, embedding_sets, arguments.k_grid,
                                arguments.tolerance, arguments.max_iterations)

    for message in selection["warnings"]:
        print("WARNING: %s" % (message,), flush=True)
    if not selection["warnings"]:
        print("Best K agrees on dev accuracy, edge purity and component coverage, and every "
              "fold converged.", flush=True)

    summary_columns = (["embedding_label", "beta", "k", "num_folds"]
                       + [name + suffix for name in AGGREGATED_METRICS
                          for suffix in ("_mean", "_std")])
    graph_columns = ["embedding_label", "beta", "k", "sigma", "edge_purity",
                     "num_components", "largest_component_fraction",
                     "graph_seconds", "component_seconds", "num_nodes", "embeddings_path"]

    write_csv(output_directory / "per_fold.csv", PER_FOLD_COLUMNS, per_fold_rows)
    write_csv(output_directory / "summary.csv", summary_columns, summary_rows)
    write_csv(output_directory / "graphs.csv", graph_columns, graph_rows)

    # The same numbers as summary.csv, pivoted to embedding x K. In beta-sweep
    # mode this IS the beta x K table the ablation reports; with a single
    # embedding set it is the one-row K curve.
    for metric_name, title in (
        ("dev_accuracy", "mean dev accuracy (beta x K)"),
        ("edge_purity", "mean edge purity (beta x K)"),
        ("component_label_coverage", "mean component label coverage (beta x K)"),
    ):
        pivot_columns, pivot_table = pivot_rows(summary_rows, metric_name, arguments.k_grid)
        write_csv(output_directory / ("pivot_%s.csv" % (metric_name,)),
                  pivot_columns, pivot_table)
        if beta_sweep_mode:
            print_pivot_table(pivot_columns, pivot_table, title)

    selection["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    selection["deterministic"] = not arguments.allow_nondeterministic
    with open(output_directory / "selection.json", "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    run_configuration = {
        "script": "scripts/sweep_knn_lp.py",
        "arguments": vars(arguments),
        "torch_version": torch.__version__,
        "device": str(device),
        "beta_sweep_mode": beta_sweep_mode,
        "split_source": split_source,
        "num_nodes": num_nodes,
        "labeled_train_offset": labeled_train_offset,
        "labeled_train_count": train_count,
        "embedding_sets": [
            {"label": s["label"], "beta": s["beta"], "path": s["path"],
             "manifest_path": s["manifest_path"],
             "checkpoint": s["manifest"].get("checkpoint", {}).get("path"),
             "checkpoint_sha256": s["manifest"].get("checkpoint", {}).get("sha256")}
            for s in embedding_sets
        ],
        "test_split_touched": False,
        "total_seconds": time.time() - started_at,
    }
    with open(output_directory / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_configuration, handle, indent=2)

    print("", flush=True)
    print("Selected: %s K=%d  mean dev accuracy %.4f +/- %.4f"
          % (selection["best_overall"]["embedding_label"], selection["best_overall"]["k"],
             selection["best_overall"]["mean_dev_accuracy"],
             selection["best_overall"]["std_dev_accuracy"]), flush=True)
    print("Wrote %s" % (output_directory,), flush=True)
    print("Total %.1fs" % (time.time() - started_at,), flush=True)


if __name__ == "__main__":
    main()
