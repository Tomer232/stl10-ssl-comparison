#!/usr/bin/env bash
# Enforces the lecturer's "build from 0" constraint on Tomer's arm of the project.
#
# Runs as a PostToolUse hook on Write|Edit. Rather than parsing the tool payload
# (Windows paths inside JSON are a pain to unescape), it just rescans the source
# tree — it is small and the grep is free.
#
# Exit 2 feeds stderr back to Claude as a correction.
#
# Scope: src/ scripts/ notebooks/ tests/ only.
#   - reference/ holds Lior's notebook, which legitimately uses sklearn.
#   - sandbox/ is the deliberate escape hatch for throwaway validation checks.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 0
cd "$ROOT" || exit 0

SCAN_DIRS=()
for d in src scripts notebooks tests; do
  [ -d "$d" ] && SCAN_DIRS+=("$d")
done
[ ${#SCAN_DIRS[@]} -eq 0 ] && exit 0

# Matches both plain source lines and the JSON-string form used inside .ipynb cells.
BANNED='(^|")[[:space:]]*(import|from)[[:space:]]+(sklearn|scipy|faiss|networkx|annoy|pynndescent|umap|cuml|torch_geometric|torchvision\.models)\b'

HITS="$(grep -rnE --include='*.py' --include='*.ipynb' "$BANNED" "${SCAN_DIRS[@]}" 2>/dev/null)"

if [ -n "$HITS" ]; then
  {
    echo "BUILD-FROM-0 VIOLATION — the lecturer forbids these libraries in Tomer's arm."
    echo ""
    echo "$HITS"
    echo ""
    echo "Hand-write it instead: KNN graph, label propagation, metrics (from a bincount"
    echo "confusion matrix), union-find for connected components, the ResNet trunk."
    echo "PyTorch and NumPy are allowed."
    echo ""
    echo "If this is a one-off correctness check against a reference implementation, it"
    echo "belongs in sandbox/ — gitignored, exempt from this hook, and it does not ship."
  } >&2
  exit 2
fi

exit 0
