#!/usr/bin/env bash
# Environment and data setup. Idempotent, and trains nothing.
#
#   bash scripts/setup.sh            create the venv, install torch, fetch STL-10
#   bash scripts/setup.sh --check    verify an existing setup without changing anything
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV=".venv"
DATA_ROOT="data"
STL10_URL="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
MINIMUM_FREE_GB=12

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

say() { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- disk
say "disk"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free space: ${FREE_GB} GB"
if (( FREE_GB < MINIMUM_FREE_GB )); then
    fail "need at least ${MINIMUM_FREE_GB} GB free, found ${FREE_GB} GB.
A run that fills the root filesystem takes the whole machine down. Free space first."
fi

# ---------------------------------------------------------------- gpu
say "gpu"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo
    echo "processes currently holding VRAM:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    echo "(the desktop compositor holding ~130 MiB is normal; a python process is not)"
else
    echo "nvidia-smi not found. The notebook will run on CPU, which is not practical for a"
    echo "full pretrain -- expect weeks rather than days."
fi

# ---------------------------------------------------------------- python
say "python environment"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" > /dev/null 2>&1 || fail "no $PYTHON_BIN on PATH"
"$PYTHON_BIN" --version

if (( CHECK_ONLY == 0 )); then
    [[ -d "$VENV" ]] || "$PYTHON_BIN" -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    # cu128 is the minimum for Blackwell (sm_120). On an older card the default index is
    # fine, but this build works on both.
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
    pip install --quiet numpy matplotlib jupyter nbconvert ipykernel
else
    [[ -d "$VENV" ]] || fail "$VENV does not exist; run without --check first"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

python - <<'PY'
import torch
print("torch     :", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    print("capability: sm_%d%d" % capability)
    if capability >= (12, 0) and int(torch.version.cuda.split(".")[0]) < 12:
        raise SystemExit("this GPU needs a CUDA 12.8+ build of torch")
PY

# ---------------------------------------------------------------- data
say "stl-10"
mkdir -p "$DATA_ROOT"
if [[ -d "$DATA_ROOT/stl10_binary" ]]; then
    echo "already extracted"
elif (( CHECK_ONLY == 1 )); then
    fail "$DATA_ROOT/stl10_binary is missing; run without --check first"
else
    echo "downloading (~2.5 GB)"
    curl -fL --progress-bar "$STL10_URL" -o "$DATA_ROOT/stl10_binary.tar.gz"
    tar -xzf "$DATA_ROOT/stl10_binary.tar.gz" -C "$DATA_ROOT"
    # Deleted immediately: the tarball is as large as the extracted data and disk is the
    # binding constraint on most machines this runs on.
    rm -f "$DATA_ROOT/stl10_binary.tar.gz"
fi

for f in train_X.bin train_y.bin test_X.bin test_y.bin unlabeled_X.bin \
         fold_indices.txt class_names.txt; do
    [[ -f "$DATA_ROOT/stl10_binary/$f" ]] || fail "missing $DATA_ROOT/stl10_binary/$f"
done
echo "all seven files present"
du -sh "$DATA_ROOT/stl10_binary"

# ---------------------------------------------------------------- rule check
say "build-from-zero check"
python tests/check_imports.py

say "ready"
echo "Open the notebook, or run it headless -- see RUNBOOK.md for the detach-safe pattern."
