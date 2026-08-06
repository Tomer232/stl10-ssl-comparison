"""Run Lior's SimCLR notebook headless, without modifying it.

`simclr_stl10_abilation_f.ipynb` is a Colab notebook: cell 4 hardcodes
`PROJECT_ROOT = Path("/content/SimCLR_STL10_Label_Efficiency")` and then mkdirs
under it, which raises PermissionError on the lab server. CLAUDE.md says his
notebook is not to be modified unless strictly necessary, so this wrapper never
writes to the .ipynb. It loads the JSON, rewrites the single PROJECT_ROOT line
IN MEMORY, and executes the result. The file on disk stays byte-identical --
`git status` after a run proves it.

Three other things the wrapper arranges:

  * TEST SPLIT. Cells 26-28 are his final benchmark on the 8000 test images and
    the results archive. `--through-cell` defaults to 25, so they are simply not
    part of the notebook that gets executed. Cells 0-25 are pretraining plus the
    development probes, and the only sklearn `train_test_split` in them acts on
    the labeled TRAIN split. The test set stays untouched until both arms lock
    their selections and `scripts/final_benchmark.py` runs once, by hand.

  * NO RE-DOWNLOAD. Cell 6 calls `datasets.STL10(..., download=True)`. torchvision
    skips the download when the five .bin files are present under
    `<root>/stl10_binary` with the expected md5s, so the wrapper symlinks the
    repo's existing `data/stl10_binary` into the run directory. Verified: all
    five files match. Disk on this box is ~95% full; a second 2.5 GB copy of
    STL-10 is not something to spend it on.

  * THE ABLATION IS FOUR PRETRAINS. Cell 20 loops over POLICY_TRANSFORMS, which
    cell 10 defines with four augmentation policies (P1_crop_color,
    P2_crop_color_blur, P3_crop_flip_color, P4_full_policy). Each one is a full
    100-epoch SimCLR run, so this is a ~20-30 hour job, not a ~5 hour one.

Usage (from the repo root, on the server):

    nohup .venv/bin/python -u scripts/run_simclr_arm.py \
        > logs/simclr.log 2>&1 &
"""

import argparse
import json
import pathlib
import re
import shutil
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "simclr_stl10_abilation_f.ipynb"

# Cells 26+ are the test-set benchmark and the results archive. Keeping 0-25
# stops at the development probes -- see the module docstring.
DEFAULT_THROUGH_CELL = 25

# The assignment to rewrite. Anchored to the start of the line so it cannot
# match a mention inside a comment or a longer identifier.
PROJECT_ROOT_PATTERN = re.compile(
    r'^PROJECT_ROOT\s*=\s*Path\((["\']).*?\1\)\s*$', re.MULTILINE)

# Cell 4 hardcodes SEED = 42. Repeating the arm at other seeds is the only way to
# put an honest +/- on his headline number, so this is rewritten the same way.
SEED_PATTERN = re.compile(r'^SEED\s*=\s*\d+\s*$', re.MULTILINE)

# Restricting the four-policy loop to a subset. Injected as its own cell rather
# than rewritten in place, because POLICY_PRIORITY is derived from
# POLICY_TRANSFORMS at the end of cell 10 and both have to stay consistent.
POLICY_FILTER_TEMPLATE = (
    "# injected by scripts/run_simclr_arm.py --policies\n"
    "POLICY_TRANSFORMS = {name: transform for name, transform in POLICY_TRANSFORMS.items()\n"
    "                     if name in %r}\n"
    "POLICY_PRIORITY = {name: priority for priority, name in enumerate(POLICY_TRANSFORMS)}\n"
    "assert POLICY_TRANSFORMS, 'policy filter matched nothing'\n"
    "print('policies restricted to:', list(POLICY_TRANSFORMS))\n"
)

# The cell that defines POLICY_TRANSFORMS / POLICY_PRIORITY; the filter goes after it.
POLICY_CELL_INDEX = 10

# --- test-benchmark mode -------------------------------------------------
#
# Cells needed to run HIS cell 26 without retraining anything:
#   2  imports, DEVICE            10  transforms, labeled_train_dataset
#   4  constants, paths, save_csv 12  SEResNetEncoder / SimCLRModel
#   6  raw splits, SSL_MEAN/STD   16  extract_features, make_linear_classifier
#   8  official folds, dev splits 24  load_frozen_encoder, full_train_features
# Cell 20 (the 100-epoch training loop) and cell 22 (which derives the SELECTED_*
# names from that loop's results) are the two that must NOT run again; the bridge
# below restores what 22 would have set, from the artifacts the first run wrote.
TEST_SETUP_CELLS = (2, 4, 6, 8, 10, 12, 16)
TEST_ENCODER_CELL = 24
TEST_BENCHMARK_CELL = 26

# Repoints the run at an existing results directory. Cell 4 mints a fresh
# timestamped RUN_ID, which would put the outputs somewhere the trained encoder
# is not.
RESULTS_ROOT_BRIDGE = """\
# injected by scripts/run_simclr_arm.py --test-benchmark
import json as _json
from pathlib import Path as _Path
RESULTS_ROOT = _Path(%r)
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
HISTORIES_ROOT = RESULTS_ROOT / "histories"
PROBES_ROOT = RESULTS_ROOT / "development_probes"
FINAL_ROOT = RESULTS_ROOT / "final_benchmark"
CLASSIFIERS_ROOT = FINAL_ROOT / "linear_classifiers"
for _folder in (CHECKPOINTS_ROOT, HISTORIES_ROOT, PROBES_ROOT, FINAL_ROOT, CLASSIFIERS_ROOT):
    _folder.mkdir(parents=True, exist_ok=True)
print("RESULTS_ROOT repointed to", RESULTS_ROOT)
"""

# Restores exactly what cell 22 wrote to final_benchmark/selection.json, so the
# test benchmark evaluates the policy/epoch/C that the DEVELOPMENT numbers chose
# -- read from disk rather than re-derived, so nothing can drift.
SELECTION_BRIDGE = """\
# injected by scripts/run_simclr_arm.py --test-benchmark
with (FINAL_ROOT / "selection.json").open(encoding="utf-8") as _f:
    _selection = _json.load(_f)
SELECTED_POLICY = _selection["policy_name"]
SELECTED_EPOCH = int(_selection["epoch"])
SELECTED_C = float(_selection["c"])
FINAL_ENCODER_PATH = _Path(_selection["encoder_path"])
assert FINAL_ENCODER_PATH.exists(), FINAL_ENCODER_PATH
print("restored selection:", SELECTED_POLICY, "epoch", SELECTED_EPOCH, "C", SELECTED_C)
"""

# The five files torchvision's STL10._check_integrity looks for. If these are in
# place the download=True in cells 6 and 26 is a no-op.
STL10_FILES = ("train_X.bin", "train_y.bin", "unlabeled_X.bin",
               "test_X.bin", "test_y.bin")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=pathlib.Path, default=NOTEBOOK,
                        help="Lior's notebook. Read only, never written.")
    parser.add_argument("--run-directory", type=pathlib.Path, default=None,
                        help="Where the notebook's PROJECT_ROOT is pointed. "
                             "Default: runs/simclr_seed42.")
    parser.add_argument("--through-cell", type=int, default=DEFAULT_THROUGH_CELL,
                        help="Execute cells 0..N inclusive. Default %d, which stops "
                             "before the test-set cell. Raising it past 25 spends the "
                             "one test evaluation." % DEFAULT_THROUGH_CELL)
    parser.add_argument("--kernel", default="python3",
                        help="Jupyter kernel name. Must be the venv's kernel.")
    parser.add_argument("--timeout", type=int, default=60 * 60 * 12,
                        help="Per-cell timeout in seconds. The policy loop is one "
                             "cell covering all four pretrains, so this is generous.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Rewrite cell 4's SEED. Default: leave his 42 alone. "
                             "Use for the extra seeds the +/- column needs.")
    parser.add_argument("--policies", nargs="+", default=None,
                        help="Restrict the cell-20 policy loop to these names, e.g. "
                             "P4_full_policy. Default: all four, which is the ablation. "
                             "A single policy is the right choice for seed repeats -- the "
                             "policy was already selected at seed 42.")
    parser.add_argument("--test-benchmark", type=pathlib.Path, default=None,
                        help="THE TEST SPLIT. Path to an existing results/<RUN_ID> "
                             "directory from a completed run. Skips all training and "
                             "runs his cell 26 against the encoder that run selected. "
                             "Requires --confirm-test-evaluation.")
    parser.add_argument("--confirm-test-evaluation", action="store_true",
                        help="Required by --test-benchmark. The 8000 test images are "
                             "read once, at the end, by both arms.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do the rewrite and the data linking, print the plan, "
                             "and exit without executing anything.")
    return parser.parse_args()


def verify_stl10(data_root):
    """Confirm the five .bin files exist so download=True stays a no-op."""
    binary_directory = data_root / "stl10_binary"
    missing = [name for name in STL10_FILES
               if not (binary_directory / name).exists()]
    if missing:
        raise SystemExit(
            "STL-10 files missing from %s: %s\nRun scripts/prepare_data.py first, or "
            "the notebook will re-download 2.5 GB onto a disk that is ~95%% full."
            % (binary_directory, ", ".join(missing)))
    return binary_directory


def link_data(run_directory, source_binary_directory):
    """Symlink <run>/data/stl10_binary -> the repo's copy.

    A symlink rather than a copy: the notebook only reads these, and a second
    2.5 GB of STL-10 is not affordable here.
    """
    data_root = run_directory / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    link = data_root / "stl10_binary"
    if link.is_symlink():
        if link.resolve() == source_binary_directory.resolve():
            return data_root
        link.unlink()
    elif link.exists():
        raise SystemExit(
            "%s already exists and is not a symlink. Refusing to touch it -- move it "
            "aside by hand if it is stale." % link)

    link.symlink_to(source_binary_directory, target_is_directory=True)
    return data_root


def rewrite_project_root(notebook, run_directory):
    """Point PROJECT_ROOT at run_directory, in memory only.

    Returns (notebook_dict, before_line, after_line). Raises if the assignment is
    not found exactly once -- silently executing his notebook against /content/
    would fail confusingly, and silently matching twice would mean the config
    cell is not what this wrapper thinks it is.
    """
    replacement = 'PROJECT_ROOT = Path("%s")' % run_directory

    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        found = PROJECT_ROOT_PATTERN.findall(source)
        if not found:
            continue
        original = PROJECT_ROOT_PATTERN.search(source).group(0)
        matches.append(original)
        cell["source"] = PROJECT_ROOT_PATTERN.sub(replacement, source).splitlines(keepends=True)

    if len(matches) != 1:
        raise SystemExit(
            "Expected exactly one top-level `PROJECT_ROOT = Path(...)` assignment in the "
            "notebook, found %d. The notebook's config cell has changed -- re-read it "
            "before trusting this wrapper." % len(matches))

    return notebook, matches[0], replacement


def rewrite_seed(notebook, seed):
    """Point cell 4's SEED at `seed`, in memory only. Same contract as PROJECT_ROOT."""
    replacement = "SEED = %d" % int(seed)
    matches = []

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not SEED_PATTERN.search(source):
            continue
        matches.append(SEED_PATTERN.search(source).group(0).strip())
        cell["source"] = SEED_PATTERN.sub(replacement, source).splitlines(keepends=True)

    if len(matches) != 1:
        raise SystemExit(
            "Expected exactly one top-level `SEED = <int>` assignment, found %d. The "
            "notebook's config cell has changed -- re-read it before trusting this "
            "wrapper." % len(matches))

    return notebook, matches[0], replacement


def restrict_policies(notebook, policies):
    """Insert a cell after the policy definitions that filters them down."""
    source = POLICY_FILTER_TEMPLATE % (list(policies),)
    cell = {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
            "outputs": [], "execution_count": None}
    notebook["cells"].insert(POLICY_CELL_INDEX + 1, cell)
    return notebook


def build_test_benchmark_notebook(notebook, results_directory):
    """Assemble a notebook that runs HIS cell 26 against an already-trained encoder.

    THIS OPENS THE TEST SPLIT. It is the counterpart to
    scripts/final_benchmark.py on our side: one evaluation, at the end, on the
    configuration the development numbers already locked.

    Cell 26 is taken verbatim -- the point of executing his notebook rather than
    reimplementing his protocol is that the test number is produced by his code,
    so a transcription slip cannot flatter or penalise either arm.
    """
    def code_cell(source):
        return {"cell_type": "code", "metadata": {},
                "source": source.splitlines(keepends=True),
                "outputs": [], "execution_count": None}

    cells = []
    for index in TEST_SETUP_CELLS:
        cells.append(notebook["cells"][index])
        if index == 4:
            cells.append(code_cell(RESULTS_ROOT_BRIDGE % (str(results_directory),)))

    cells.append(code_cell(SELECTION_BRIDGE))
    cells.append(notebook["cells"][TEST_ENCODER_CELL])
    cells.append(notebook["cells"][TEST_BENCHMARK_CELL])

    notebook["cells"] = cells
    return notebook


def truncate(notebook, through_cell):
    """Keep cells 0..through_cell inclusive."""
    total = len(notebook["cells"])
    if through_cell >= total - 1:
        print("  *** --through-cell %d keeps every cell, INCLUDING the test-set "
              "benchmark. ***" % through_cell, flush=True)
    notebook["cells"] = notebook["cells"][:through_cell + 1]
    return notebook, total


def collapse_progress_bars(text):
    """Keep only the final state of each tqdm bar.

    tqdm redraws in place with a carriage return, which a notebook captures as
    one enormous single-line string -- the channel-statistics bar alone is 196
    redraws. Over four 100-epoch pretrains that is a log nobody can read, so each
    line is reduced to whatever followed its last '\\r'.
    """
    return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


def report_cell(cell, cell_index, execute_reply):
    """Print a cell's stdout as it finishes, so the log is followable live.

    nbclient calls this hook with keyword arguments, so the parameter names are
    part of the contract: it passes `cell_index`, not `index`.
    """
    del execute_reply
    if cell.get("cell_type") != "code":
        return
    print("\n===== cell %d done (%s) =====" % (cell_index, time.strftime("%H:%M:%S")), flush=True)
    for output in cell.get("outputs", []):
        text = output.get("text")
        if isinstance(text, list):
            text = "".join(text)
        if text:
            print(collapse_progress_bars(text), end="", flush=True)
        if output.get("output_type") == "error":
            print("\n".join(output.get("traceback", [])), flush=True)


def main():
    arguments = parse_arguments()

    # Absolute: the notebook writes every artifact under PROJECT_ROOT, and a
    # relative path would resolve against the kernel's cwd instead of here.
    run_directory = (arguments.run_directory or (REPO_ROOT / "runs" / "simclr_seed42")).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)

    source_binary_directory = verify_stl10(REPO_ROOT / "data")
    data_root = link_data(run_directory, source_binary_directory)

    with arguments.notebook.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    if arguments.test_benchmark is not None and not arguments.confirm_test_evaluation:
        raise SystemExit(
            "--test-benchmark reads the 8000 STL-10 test images. That happens once, at "
            "the very end, for both arms. Pass --confirm-test-evaluation if that is "
            "genuinely where the project is.")

    notebook, before, after = rewrite_project_root(notebook, run_directory)

    seed_before = seed_after = None
    if arguments.seed is not None:
        notebook, seed_before, seed_after = rewrite_seed(notebook, arguments.seed)

    if arguments.test_benchmark is not None:
        total_cells = len(notebook["cells"])
        notebook = build_test_benchmark_notebook(
            notebook, arguments.test_benchmark.resolve())
        through_cell = len(notebook["cells"]) - 1
    else:
        # Insert BEFORE truncating so the inserted cell shifts --through-cell along
        # with everything after it, rather than silently pushing the test cell into
        # range.
        through_cell = arguments.through_cell
        if arguments.policies:
            notebook = restrict_policies(notebook, arguments.policies)
            if through_cell > POLICY_CELL_INDEX:
                through_cell += 1

        notebook, total_cells = truncate(notebook, through_cell)

    print("SimCLR arm -- Lior's notebook, run headless")
    print("  notebook       %s (READ ONLY -- not modified)" % arguments.notebook)
    if arguments.test_benchmark is not None:
        print("  MODE           *** TEST BENCHMARK -- READS THE 8000 TEST IMAGES ***")
        print("  cells          setup %s + selection bridge + %d + %d (no training)"
              % (list(TEST_SETUP_CELLS), TEST_ENCODER_CELL, TEST_BENCHMARK_CELL))
        print("  results        %s" % arguments.test_benchmark.resolve())
    else:
        # `through_cell` is the index AFTER any injected cells shifted things along,
        # so the test-split check has to use it and not the raw argument.
        test_cell_index = 26 + (1 if arguments.policies else 0)
        print("  cells          0..%d of %d  (%s)"
              % (through_cell, total_cells,
                 "test split NOT touched" if through_cell < test_cell_index
                 else "*** INCLUDES THE TEST SPLIT ***"))
    print("  rewrote        %s" % before)
    print("       ->        %s" % after)
    if seed_before is not None:
        print("  rewrote        %s" % seed_before)
        print("       ->        %s" % seed_after)
    if arguments.policies:
        print("  policies       restricted to %s (of four)" % (list(arguments.policies),))
    print("  data           %s -> %s" % (data_root / "stl10_binary", source_binary_directory))
    if arguments.test_benchmark is None:
        policy_count = len(arguments.policies) if arguments.policies else 4
        print("  policies       %d x 100 epochs" % policy_count)
        # Measured on the 5090 over the seed-42 run: 7.0 h wall for four policies,
        # i.e. ~1.05 min/epoch plus ~0.5 min per development probe.
        print("  expect         ~%.1f hours (measured: ~1.75 h per policy)"
              % (1.75 * policy_count), flush=True)
    else:
        print("  expect         ~5 minutes", flush=True)

    if arguments.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    import nbformat
    from nbclient import NotebookClient

    executable = nbformat.reads(json.dumps(notebook), as_version=4)

    client = NotebookClient(
        executable,
        timeout=arguments.timeout,
        kernel_name=arguments.kernel,
        allow_errors=False,
        # The notebook resolves its own absolute paths, but relative reads should
        # still land in the repo rather than wherever this was launched from.
        resources={"metadata": {"path": str(REPO_ROOT)}},
        on_cell_executed=report_cell,
    )

    started_at = time.time()
    status = 0
    try:
        client.execute()
    except Exception as error:                      # noqa: BLE001 -- log and re-raise as exit code
        print("\n*** EXECUTION FAILED: %s: %s" % (type(error).__name__, error), flush=True)
        status = 1
    finally:
        executed_path = run_directory / "executed_notebook.ipynb"
        with executed_path.open("w", encoding="utf-8") as handle:
            nbformat.write(executable, handle)
        elapsed = time.time() - started_at
        print("\nelapsed  %.1f h" % (elapsed / 3600.0))
        print("executed notebook written to %s" % executed_path)
        print("results under %s" % (run_directory / "results"), flush=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
