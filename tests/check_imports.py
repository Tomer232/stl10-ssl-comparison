"""Enforce the build-from-zero rule.

The project's central constraint is that the pipeline uses PyTorch and NumPy only. The
notebook's final section deliberately imports scikit-learn and torchvision to check the
hand-written components against the implementations they replace; those cells are the only
permitted exception, and they are identified by the marker below rather than by cell number,
so inserting cells cannot silently widen the exemption.

Run from the repository root:

    python tests/check_imports.py

Exits non-zero and lists every violation if the rule is broken.
"""

import ast
import json
import pathlib
import sys

BANNED_TOP_LEVEL_MODULES = {
    "sklearn",
    "scipy",
    "faiss",
    "networkx",
    "torchvision",
    "annoy",
    "nmslib",
    "pynndescent",
}

# Everything from this marker onward is the verification section.
VERIFICATION_MARKER = "VERIFICATION = []"

NOTEBOOK_GLOB = "*.ipynb"
PYTHON_GLOBS = ("scripts/**/*.py", "tests/**/*.py")

# Lior's original implementation is preserved as provenance for the contrastive arm and is
# not part of the pipeline. The rule was never meant to bind it.
EXEMPT_PATHS = {"reference"}


def imported_modules(source):
    """Yield (line_number, top_level_module) for every import in `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SyntaxError("could not parse: %s" % error)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.lineno, node.module.split(".")[0]


def check_notebook(path):
    """Return a list of violation strings for one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    verification_starts_at = len(cells)
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and VERIFICATION_MARKER in "".join(cell["source"]):
            verification_starts_at = index
            break

    violations = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code" or index >= verification_starts_at:
            continue
        source = "".join(cell["source"])
        for line_number, module in imported_modules(source):
            if module in BANNED_TOP_LEVEL_MODULES:
                violations.append(
                    "%s: cell %d line %d imports %r, which the build-from-zero rule forbids "
                    "outside the verification section (cell %d onward)"
                    % (path, index, line_number, module, verification_starts_at)
                )
    return violations


def check_python_file(path):
    violations = []
    for line_number, module in imported_modules(path.read_text(encoding="utf-8")):
        if module in BANNED_TOP_LEVEL_MODULES:
            violations.append("%s:%d imports %r, which the build-from-zero rule forbids"
                              % (path, line_number, module))
    return violations


def main():
    repository_root = pathlib.Path(__file__).resolve().parent.parent
    violations = []
    checked = 0

    for path in sorted(repository_root.glob(NOTEBOOK_GLOB)):
        if EXEMPT_PATHS & set(path.relative_to(repository_root).parts):
            continue
        violations.extend(check_notebook(path))
        checked += 1

    for pattern in PYTHON_GLOBS:
        for path in sorted(repository_root.glob(pattern)):
            if EXEMPT_PATHS & set(path.relative_to(repository_root).parts):
                continue
            if path.name == pathlib.Path(__file__).name:
                continue
            violations.extend(check_python_file(path))
            checked += 1

    if violations:
        print("build-from-zero check FAILED (%d violation(s) across %d file(s)):\n"
              % (len(violations), checked))
        for violation in violations:
            print("  -", violation)
        return 1

    print("build-from-zero check passed: %d file(s), no banned imports outside the "
          "verification section." % checked)
    print("banned:", ", ".join(sorted(BANNED_TOP_LEVEL_MODULES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
