#!/usr/bin/env bash
#
# One-time setup of this repo on the lab RTX 5090 server (Ubuntu 24.04).
#
#   bash scripts/bootstrap_server.sh            # full setup
#   bash scripts/bootstrap_server.sh --check    # verify only, change nothing
#
# Idempotent: safe to re-run. Every step checks whether it is already done.
#
# It does NOT train anything. When it finishes, the next command is
# `bash scripts/run_all.sh plan`.
#
# WHAT IT ASSUMES ABOUT THE BOX (surveyed 2026-08-04)
#   * Ubuntu 24.04, python3 = 3.12.3, and NO `python` binary at all
#   * one RTX 5090 (Blackwell, sm_120) -> cu128 is the MINIMUM torch build
#   * account `labadmin` is SHARED by the whole lab, and its global git identity
#     belongs to someone else -- hence the per-repo identity below
#   * no tmux, no screen, no Slurm: long runs detach with nohup
#   * one volume, no /scratch, no quotas -- a runaway job fills / for everyone

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
DATA_ROOT="$REPO_ROOT/data"

# Enough for the venv (~4 GB), STL-10 (~2.5 GB extracted plus the tarball while
# it is being unpacked), and room for checkpoints and embedding caches.
REQUIRED_GB=20

GIT_NAME="${GIT_NAME:-Tomer Karmazin}"
GIT_EMAIL="${GIT_EMAIL:-tomer@jeepsea.co.il}"

TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

STL10_URL="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

say()  { printf '\n=== %s\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
die()  { printf '\n  [FAIL] %s\n\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Refuse to run anywhere but Linux
# ---------------------------------------------------------------------------
say "environment"
if [ "$(uname -s)" != "Linux" ]; then
    die "this script is for the Ubuntu server. On the laptop there is no CUDA GPU
         and nothing here should be trained. Use the /gpu-run skill instead."
fi
ok "Linux: $(uname -sr)"
command -v python3 >/dev/null || die "python3 not found"
ok "python3: $(python3 --version 2>&1)"

# ---------------------------------------------------------------------------
# 1. Disk. The most likely way a run dies on this box.
# ---------------------------------------------------------------------------
say "disk"
AVAIL_GB="$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')"
printf '  free on this volume: %s GB (need >= %s GB)\n' "$AVAIL_GB" "$REQUIRED_GB"
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
    die "not enough free disk. There are no quotas on this box, so a job that
         fills / takes the whole lab's machine down. Free space before continuing.
         Never delete other people's files to make room -- ask first."
fi
ok "enough headroom"

# ---------------------------------------------------------------------------
# 2. GPU
# ---------------------------------------------------------------------------
say "gpu"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
               --format=csv,noheader | sed 's/^/  /'
    BUSY="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader || true)"
    if [ -n "$BUSY" ]; then
        warn "another process is holding the GPU:"
        printf '%s\n' "$BUSY" | sed 's/^/         /'
        warn "nothing arbitrates GPU access here. Coordinate before training."
    else
        ok "no compute processes on the GPU"
    fi
else
    warn "nvidia-smi not found -- cannot verify the GPU"
fi

# ---------------------------------------------------------------------------
# 3. Idle-suspend. A 100-epoch pretrain left overnight will otherwise suspend.
# ---------------------------------------------------------------------------
say "idle-suspend"
SLEEP_MASKED=1
for unit in sleep.target suspend.target; do
    state="$(systemctl is-enabled "$unit" 2>/dev/null || true)"
    [ "$state" = "masked" ] || SLEEP_MASKED=0
done
if [ "$SLEEP_MASKED" -eq 1 ]; then
    ok "sleep.target and suspend.target are masked"
else
    warn "the machine can still suspend itself mid-training. Whoever owns the box"
    warn "should run (needs sudo, changes machine-wide behaviour):"
    warn "  sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"
    warn "  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'"
fi

# ---------------------------------------------------------------------------
# 4. Per-repo git identity
# ---------------------------------------------------------------------------
say "git identity"
if [ "$CHECK_ONLY" -eq 0 ]; then
    git config user.name  "$GIT_NAME"
    git config user.email "$GIT_EMAIL"
fi
printf '  %s <%s>  (repo-local; the global identity on this shared account is someone else)\n' \
       "$(git config user.name || echo unset)" "$(git config user.email || echo unset)"

# ---------------------------------------------------------------------------
# 5. The venv. Ours alone -- never touch hri_env or rrnlp_env.
# ---------------------------------------------------------------------------
say "virtualenv"
if [ -x "$PYTHON" ]; then
    ok "exists: $VENV"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "missing: $VENV"
else
    python3 -m venv "$VENV"
    "$PYTHON" -m pip install --quiet --upgrade pip
    ok "created: $VENV"
fi

say "torch"
if [ -x "$PYTHON" ] && "$PYTHON" -c "import torch" 2>/dev/null; then
    ok "already installed"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "torch not installed"
else
    printf '  installing %s from %s (~4 GB, several minutes)\n' "$TORCH_SPEC" "$TORCH_INDEX"
    "$PYTHON" -m pip install --index-url "$TORCH_INDEX" "$TORCH_SPEC"
    "$PYTHON" -m pip install numpy matplotlib
    ok "installed"
fi

if [ -x "$PYTHON" ] && "$PYTHON" -c "import torch" 2>/dev/null; then
    "$PYTHON" - <<'PYEOF' || die "torch cannot see the GPU"
import torch
print("  torch %s  cuda %s  available=%s"
      % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("  device: %s" % torch.cuda.get_device_name(0))
capability = torch.cuda.get_device_capability(0)
print("  compute capability: sm_%d%d" % capability)
if capability[0] < 12:
    print("  note: expected sm_120 (Blackwell) for an RTX 5090")
PYEOF
    ok "torch sees the GPU"
fi

# Deliberately absent from requirements.txt: sklearn, scipy, faiss, networkx.
# The lecturer's build-from-0 constraint forbids them. If one of them is present
# in this venv, something has gone wrong.
if [ -x "$PYTHON" ]; then
    LEAKED="$("$PYTHON" -m pip list --format=freeze 2>/dev/null \
              | grep -iE '^(scikit-learn|scipy|faiss|networkx)' || true)"
    if [ -n "$LEAKED" ]; then
        warn "banned libraries present in the venv (build-from-0 violation):"
        printf '%s\n' "$LEAKED" | sed 's/^/         /'
    else
        ok "no banned libraries in the venv"
    fi
fi

# ---------------------------------------------------------------------------
# 6. STL-10
# ---------------------------------------------------------------------------
say "stl-10 data"
if [ -d "$DATA_ROOT/stl10_binary" ]; then
    ok "present: $DATA_ROOT/stl10_binary ($(du -sh "$DATA_ROOT/stl10_binary" | cut -f1))"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "missing: $DATA_ROOT/stl10_binary"
else
    printf '  downloading ~2.5 GB from %s\n' "$STL10_URL"
    mkdir -p "$DATA_ROOT"
    (
        cd "$DATA_ROOT"
        curl -L --fail -O "$STL10_URL"
        tar xzf stl10_binary.tar.gz
        # Deleted immediately: keeping both the tarball and the extracted copy
        # doubles the footprint for no benefit.
        rm -f stl10_binary.tar.gz
    )
    ok "extracted, tarball removed"
fi

# ---------------------------------------------------------------------------
# 7. Splits and normalization constants, built once
# ---------------------------------------------------------------------------
say "splits + normalization"
if [ -f "$DATA_ROOT/splits.npz" ] && [ -f "$DATA_ROOT/normalization.json" ]; then
    ok "already built"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "not built yet (run without --check, or: bash scripts/run_all.sh prepare)"
elif [ -x "$PYTHON" ] && [ -d "$DATA_ROOT/stl10_binary" ]; then
    "$PYTHON" -u scripts/prepare_data.py --data-root "$DATA_ROOT"
    ok "built"
fi

# ---------------------------------------------------------------------------
# 8. Prove the code works on this machine before any GPU time is spent
# ---------------------------------------------------------------------------
say "smoke test"
if [ -x "$PYTHON" ]; then
    if "$PYTHON" -u tests/smoke_test.py; then
        ok "all asserts passed"
    else
        die "the smoke test failed. Do NOT start training -- fix this first."
    fi
else
    warn "skipped (no venv)"
fi

say "build-from-0 hook"
if bash .claude/hooks/check-imports.sh; then
    ok "no banned imports in src/ scripts/ notebooks/ tests/"
else
    die "banned imports found -- see above"
fi

# ---------------------------------------------------------------------------
say "ready"
cat <<'EOF'
  Next:
    bash scripts/run_all.sh plan        # stage table + rough wall-clock, runs nothing

  Then, one stage at a time, detached (there is no tmux on this box):
    mkdir -p logs
    nohup bash scripts/run_all.sh pretrain > logs/pretrain.log 2>&1 &
    tail -f logs/pretrain.log

  The test split stays untouched until the very end: stage `final` is not part
  of `all` and refuses to run without results/selection.json.
EOF
