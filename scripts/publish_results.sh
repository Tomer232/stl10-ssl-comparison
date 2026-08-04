#!/usr/bin/env bash
#
# Collect the small, reportable artifacts out of runs/ into results/ and commit
# them, so Lior and the report can read the numbers without needing the server.
#
#   bash scripts/publish_results.sh                 # collect + show what changed
#   bash scripts/publish_results.sh --commit        # collect + git commit
#   bash scripts/publish_results.sh --commit --push # ... and push (needs auth)
#
# WHAT GOES IN AND WHAT STAYS OUT
# --------------------------------
# In:  *.csv, *.json, *.md, *.png  -- metrics, summaries, selections, figures.
#      Kilobytes. These are the report.
# Out: checkpoints (*.pt), embedding caches (*.npy/*.npz), logs.
#      The embedding cache alone is ~107 MB per beta and GitHub would reject
#      the checkpoints. They stay on the server; they are reproducible from a
#      checkpoint and a seed, and nothing in the report cites them directly.
#
# runs/ is gitignored and results/ is tracked -- that is the whole distinction
# this script exists to enforce.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNS="${RUNS:-$REPO_ROOT/runs}"
RESULTS="${RESULTS:-$REPO_ROOT/results}"

DO_COMMIT=0
DO_PUSH=0
for argument in "$@"; do
    case "$argument" in
        --commit) DO_COMMIT=1 ;;
        --push)   DO_PUSH=1 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

[ -d "$RUNS" ] || { echo "no runs/ directory yet -- nothing to publish"; exit 0; }

mkdir -p "$RESULTS"

echo "collecting reportable artifacts from $RUNS"
COPIED=0
while IFS= read -r -d '' source_file; do
    relative="${source_file#"$RUNS"/}"
    destination="$RESULTS/$relative"
    mkdir -p "$(dirname "$destination")"
    cp -p "$source_file" "$destination"
    COPIED=$((COPIED + 1))
done < <(find "$RUNS" -type f \
              \( -name '*.csv' -o -name '*.json' -o -name '*.md' -o -name '*.png' \) \
              -print0)

echo "  copied $COPIED file(s) into $RESULTS"

# A guard, not a formality: a stray checkpoint in results/ would be committed,
# and a 100 MB blob in git history cannot be removed without a force-push.
OVERSIZED="$(find "$RESULTS" -type f -size +5M 2>/dev/null || true)"
if [ -n "$OVERSIZED" ]; then
    echo ""
    echo "  REFUSING TO COMMIT: files over 5 MB found in results/:"
    printf '%s\n' "$OVERSIZED" | sed 's/^/    /'
    echo "  results/ is for kilobyte-scale summaries. Move these out first."
    exit 1
fi

echo ""
git status --short -- "$RESULTS" || true

if [ "$DO_COMMIT" -eq 0 ]; then
    echo ""
    echo "Nothing committed (dry run). Re-run with --commit when the numbers are final."
    exit 0
fi

if [ -z "$(git status --porcelain -- "$RESULTS")" ]; then
    echo ""
    echo "results/ is unchanged -- nothing to commit."
    exit 0
fi

git add -- "$RESULTS"
git commit -q -m "Publish results

Collected from runs/ by scripts/publish_results.sh. Summaries, metrics and
selection files only -- checkpoints and embedding caches stay on the server.
"
echo "committed."

if [ "$DO_PUSH" -eq 1 ]; then
    # This shared account has no git credential helper and its gh login belongs
    # to someone else, so a push from here may well fail. That is fine: the
    # commit is made either way and can be pushed from the laptop.
    if git push; then
        echo "pushed."
    else
        echo ""
        echo "push failed -- the commit is safe locally. Either configure a"
        echo "deploy key for this repo on the server, or pull it from the laptop:"
        echo "  git fetch <this-server> && git merge"
    fi
fi
