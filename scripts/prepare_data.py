"""Stage 0: verify STL-10 is on disk, then derive the constants every later stage reuses.

Run this once per machine, before anything else:

    python scripts/prepare_data.py

It produces exactly two artifacts that the rest of the pipeline treats as read-only
ground truth:

    data/normalization.json   per-RGB-channel mean/std from the UNLABELED split
    data/splits.npz           the 10 official folds + their 800/200 dev splits

Deriving those once and committing to the files is not bureaucracy. A split or a
normalization constant recomputed in three different scripts is a split that will
eventually disagree in one of them, and the entire two-arm comparison rests on the claim
that Tomer's VAE arm and Lior's SimCLR arm are scored on the same images with the same
preprocessing. One file, computed once, read everywhere.

THE TEST SPLIT IS NOT READ HERE
-------------------------------
This script never loads a single test pixel. It checks that test_X.bin and test_y.bin
*exist* -- a truncated download should fail in the first 30 seconds, not three hours into
a training run -- and goes no further. The test set is touched exactly once, at the very
end, by the final evaluation script, and by both arms.

IT ALSO NEVER DOWNLOADS
-----------------------
Disk on the lab server is the binding constraint (~13 GB free at last survey, one volume,
no /scratch). A script that silently pulls a 2.5 GB tarball plus its 2.5 GB extraction
could take the whole lab's box down. So if the data is missing this prints the exact
command from the gpu-run skill and exits non-zero, and a human decides.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy

# Repo root on sys.path so "python scripts/prepare_data.py" works from the repo root
# without an editable install or a PYTHONPATH incantation.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.data import (
    build_and_save_splits,
    compute_channel_statistics,
    load_class_names,
    load_split,
    resolve_binary_directory,
)
from src.seeding import enable_deterministic, reset_seed


# Every file the pipeline will eventually need. Checked for existence up front, because
# discovering a missing fold_indices.txt after a 100-epoch pretrain is a bad afternoon.
REQUIRED_FILES = (
    "train_X.bin",
    "train_y.bin",
    "test_X.bin",
    "test_y.bin",
    "unlabeled_X.bin",
    "fold_indices.txt",
    "class_names.txt",
)

# Expected on-disk sizes in bytes, derived from the split sizes rather than typed, so a
# partially-written .bin is caught here instead of producing garbage embeddings later.
# 3 channels * 96 * 96 = 27648 bytes per image, 1 byte per label.
BYTES_PER_IMAGE = 3 * config.IMAGE_SIZE * config.IMAGE_SIZE
EXPECTED_FILE_SIZES = {
    "train_X.bin": config.NUM_TRAIN_IMAGES * BYTES_PER_IMAGE,
    "train_y.bin": config.NUM_TRAIN_IMAGES,
    "test_X.bin": config.NUM_TEST_IMAGES * BYTES_PER_IMAGE,
    "test_y.bin": config.NUM_TEST_IMAGES,
    "unlabeled_X.bin": config.NUM_UNLABELED_IMAGES * BYTES_PER_IMAGE,
}

# Copied verbatim from .claude/skills/gpu-run/SKILL.md. Printed, never executed.
DOWNLOAD_COMMAND = (
    "ssh bguserver 'cd ~/lab/Tomer_Karmazin/final_project && mkdir -p data && cd data && \\\n"
    "  curl -L -O http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz && \\\n"
    "  tar xzf stl10_binary.tar.gz && rm stl10_binary.tar.gz && du -sh stl10_binary'"
)

LOCAL_DOWNLOAD_COMMAND = (
    "curl -L -o data/stl10_binary.tar.gz "
    "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz\n"
    "tar xzf data/stl10_binary.tar.gz -C data && rm data/stl10_binary.tar.gz"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Verify STL-10, compute normalization constants and build the splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=str(config.DATA_ROOT),
        help="Directory holding stl10_binary/ (the stl10_binary directory itself is "
             "also accepted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Seed for the stratified 800/200 dev splits. Fold i uses seed + i, "
             "mirroring Lior's random_state=SEED + fold_index.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=config.FOLD_DEVELOPMENT_SIZE / config.FOLD_SIZE,
        help="Held-out fraction inside each 1000-image fold (0.20 -> 800/200).",
    )
    parser.add_argument(
        "--splits-path",
        default=str(config.DATA_ROOT / "splits.npz"),
        help="Where to write the folds and dev splits.",
    )
    parser.add_argument(
        "--normalization-path",
        default=str(config.DATA_ROOT / "normalization.json"),
        help="Where to write the per-channel mean/std.",
    )
    parser.add_argument(
        "--chunk-images",
        type=int,
        default=config.NORMALIZATION_CHUNK_IMAGES,
        help="Images per float64 accumulation chunk when scanning the unlabeled split.",
    )
    parser.add_argument(
        "--run-name",
        default="prepare_data",
        help="Prefix of the runs/<name>_<timestamp>/ record directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing splits.npz / normalization.json. Without this the "
             "script refuses, because silently rewriting the splits mid-project would "
             "invalidate every embedding and accuracy number already produced.",
    )
    return parser.parse_args()


def verify_dataset_present(data_root):
    """Return the stl10_binary directory, or print the download command and exit 1.

    Deliberately does NOT download. See the module docstring: disk on the shared lab
    server is the binding constraint and a surprise 5 GB of traffic is a human decision.
    """
    binary_directory = resolve_binary_directory(data_root)

    missing = [name for name in REQUIRED_FILES
               if not os.path.isfile(os.path.join(binary_directory, name))]

    if missing:
        print("", flush=True)
        print("STL-10 is not present (or is incomplete).", flush=True)
        print("  looked in: %s" % binary_directory, flush=True)
        print("  missing:   %s" % ", ".join(missing), flush=True)
        print("", flush=True)
        print("This script will not download it -- the lab server has one volume with", flush=True)
        print("~13 GB free and the tarball plus its extraction is ~5 GB. Download it", flush=True)
        print("deliberately, check disk first, and delete the tarball immediately:", flush=True)
        print("", flush=True)
        print("  # on the lab server (the usual case):", flush=True)
        print(DOWNLOAD_COMMAND, flush=True)
        print("", flush=True)
        print("  # locally, from the repo root:", flush=True)
        print(LOCAL_DOWNLOAD_COMMAND, flush=True)
        print("", flush=True)
        sys.exit(1)

    # Size check catches the far nastier failure: a file that exists but was truncated
    # by a dropped connection or a full disk. numpy.fromfile would happily reshape the
    # remainder and every downstream number would be quietly wrong.
    truncated = []
    for name, expected_bytes in EXPECTED_FILE_SIZES.items():
        actual_bytes = os.path.getsize(os.path.join(binary_directory, name))
        if actual_bytes != expected_bytes:
            truncated.append((name, expected_bytes, actual_bytes))

    if truncated:
        print("", flush=True)
        print("STL-10 files are present but have the wrong size -- almost certainly a", flush=True)
        print("truncated download. Delete stl10_binary/ and fetch it again.", flush=True)
        for name, expected_bytes, actual_bytes in truncated:
            print("  %-16s expected %12d bytes, found %12d"
                  % (name, expected_bytes, actual_bytes), flush=True)
        print("", flush=True)
        sys.exit(1)

    return binary_directory


def refuse_to_overwrite(paths, force):
    """Exit non-zero if any output already exists and --force was not passed."""
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not force:
        print("", flush=True)
        print("Refusing to overwrite existing artifacts:", flush=True)
        for path in existing:
            print("  %s" % path, flush=True)
        print("", flush=True)
        print("These files define the splits and the normalization constants that every", flush=True)
        print("embedding, accuracy number and figure produced so far was computed with.", flush=True)
        print("Rewriting them invalidates all of it. Pass --force if that is genuinely", flush=True)
        print("what you want.", flush=True)
        print("", flush=True)
        sys.exit(1)


def write_normalization_file(path, mean, standard_deviation, image_count, chunk_images):
    """Persist the channel statistics as JSON, with enough provenance to audit them."""
    record = {
        "mean": [float(value) for value in mean],
        "standard_deviation": [float(value) for value in standard_deviation],
        "source_split": config.NORMALIZATION_SOURCE_SPLIT,
        "source_image_count": int(image_count),
        "chunk_images": int(chunk_images),
        "pixel_scale": config.PIXEL_SCALE,
        "channel_order": "RGB",
        "units": "[0, 1] -- computed after dividing by 255",
        # Recorded so a future reader can tell at a glance that the labeled train and
        # test splits played no part in preprocessing. Using them would leak evaluation
        # data into the normalization constants.
        "excluded_splits": ["train", "test"],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    return record


def summarize_development_splits(splits):
    """Per-fold row counts and per-class dev counts, for the record CSV.

    Worth writing out rather than trusting: `stratified_split` is hand-rolled (sklearn is
    banned), so the cheapest proof that "stratified" actually happened is a table showing
    20 dev images per class in every fold.
    """
    official_folds = splits["official_folds"]
    train_labels = splits["train_labels"]

    rows = []
    for fold_index in range(len(official_folds)):
        train_positions, development_positions = splits["development_splits"][fold_index]
        fold_labels = train_labels[official_folds[fold_index]]

        development_counts = numpy.bincount(
            fold_labels[development_positions], minlength=config.NUM_CLASSES)
        train_counts = numpy.bincount(
            fold_labels[train_positions], minlength=config.NUM_CLASSES)

        rows.append({
            "fold_index": fold_index,
            "train_size": int(len(train_positions)),
            "development_size": int(len(development_positions)),
            "train_class_counts": [int(value) for value in train_counts],
            "development_class_counts": [int(value) for value in development_counts],
        })

    return rows


def write_fold_summary_csv(path, rows):
    """Hand-written CSV -- one more dependency avoided, and the format is three lines."""
    header = ["fold_index", "train_size", "development_size"]
    header += ["train_class_%d" % class_id for class_id in range(config.NUM_CLASSES)]
    header += ["development_class_%d" % class_id for class_id in range(config.NUM_CLASSES)]

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            values = [row["fold_index"], row["train_size"], row["development_size"]]
            values += row["train_class_counts"]
            values += row["development_class_counts"]
            handle.write(",".join(str(value) for value in values) + "\n")


def main():
    arguments = parse_arguments()

    # Seeding before anything else, per the project's run protocol. Nothing here uses a
    # GPU, but enable_deterministic also sets CUBLAS_WORKSPACE_CONFIG, which must happen
    # before any CUDA context exists -- so it is always called first, unconditionally.
    reset_seed(arguments.seed)
    enable_deterministic()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("prepare_data", flush=True)
    print("  data root:   %s" % arguments.data_root, flush=True)
    print("  seed:        %d" % arguments.seed, flush=True)
    print("", flush=True)

    # Both checks run before anything is created on disk, so a failed invocation leaves
    # no empty run directory behind for someone to wonder about later.
    binary_directory = verify_dataset_present(arguments.data_root)
    refuse_to_overwrite(
        [arguments.splits_path, arguments.normalization_path], arguments.force)

    record_directory = config.ensure_directory(
        config.RUNS_ROOT / ("%s_%s" % (arguments.run_name, timestamp)))

    print("STL-10 verified at %s" % binary_directory, flush=True)
    print("  record dir: %s" % record_directory, flush=True)

    class_names = load_class_names(arguments.data_root)
    print("  classes: %s" % ", ".join(class_names), flush=True)
    print("", flush=True)

    # ------------------------------------------------------------------
    # Channel statistics from the unlabeled split only
    # ------------------------------------------------------------------
    # Memory-mapped: the unlabeled split is 2.76 GB as uint8 and
    # compute_channel_statistics reads it chunk by chunk, so there is no reason to
    # materialise the whole thing. On a 16 GB laptop this is the difference between
    # working and swapping.
    print("Computing channel statistics from the UNLABELED split "
          "(%d images, %d per chunk, float64 accumulation)..."
          % (config.NUM_UNLABELED_IMAGES, arguments.chunk_images), flush=True)
    started = time.time()

    unlabeled_images, _ = load_split(
        arguments.data_root, "unlabeled", memory_map_unlabeled=True)
    mean, standard_deviation = compute_channel_statistics(
        unlabeled_images, chunk_size=arguments.chunk_images)

    print("  done in %.1f s" % (time.time() - started), flush=True)
    print("", flush=True)

    # Printed at full precision on purpose: the check that matters is a human eyeballing
    # these against the numbers Lior's notebook prints in cell 6. If they differ, the two
    # arms are not looking at the same pixels and nothing downstream is comparable.
    print("  ==> COMPARE THESE AGAINST LIOR'S NOTEBOOK (cell 6) BY EYE <==", flush=True)
    print("  mean = [%.6f, %.6f, %.6f]" % tuple(mean), flush=True)
    print("  std  = [%.6f, %.6f, %.6f]" % tuple(standard_deviation), flush=True)
    print("", flush=True)

    normalization_record = write_normalization_file(
        arguments.normalization_path,
        mean,
        standard_deviation,
        image_count=len(unlabeled_images),
        chunk_images=arguments.chunk_images,
    )
    print("wrote %s" % arguments.normalization_path, flush=True)

    # ------------------------------------------------------------------
    # Official folds and their stratified 800/200 development splits
    # ------------------------------------------------------------------
    print("", flush=True)
    print("Building the 10 official folds and their %d/%d stratified dev splits..."
          % (config.FOLD_TRAIN_SIZE, config.FOLD_DEVELOPMENT_SIZE), flush=True)

    splits = build_and_save_splits(
        arguments.data_root,
        arguments.splits_path,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
    )
    print("wrote %s" % arguments.splits_path, flush=True)
    print("", flush=True)

    fold_rows = summarize_development_splits(splits)
    for row in fold_rows:
        print("  fold %d: train %d, dev %d, dev per class %s"
              % (row["fold_index"], row["train_size"], row["development_size"],
                 row["development_class_counts"]), flush=True)
    print("", flush=True)

    # ------------------------------------------------------------------
    # Run record -- a dated copy of everything, so a stale splits.npz can always be
    # traced back to the run that produced it.
    # ------------------------------------------------------------------
    write_fold_summary_csv(str(record_directory / "fold_summary.csv"), fold_rows)

    summary = {
        "script": "prepare_data.py",
        "timestamp": timestamp,
        "seed": arguments.seed,
        "validation_fraction": arguments.validation_fraction,
        "data_root": str(arguments.data_root),
        "binary_directory": str(binary_directory),
        "class_names": class_names,
        "normalization": normalization_record,
        "splits_path": str(arguments.splits_path),
        "normalization_path": str(arguments.normalization_path),
        "num_folds": int(len(splits["official_folds"])),
        "fold_size": int(splits["official_folds"].shape[1]),
        "folds": fold_rows,
        "test_split_read": False,
    }
    with open(record_directory / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("record written to %s" % record_directory, flush=True)
    print("", flush=True)
    print("Next: scripts/pretrain_vae.py --config configs/vae_base.json", flush=True)


if __name__ == "__main__":
    main()
