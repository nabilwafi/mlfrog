"""Materialize sourceless .pyc next to package dirs so a source-stripped tree runs.

The codebase was stripped to Python 3.14 bytecode (only __pycache__/*.pyc remain,
no .py). Python can still import a *sourceless* module if the .pyc sits next to
where the .py would be (e.g. pkg/mod.pyc instead of pkg/__pycache__/mod.cpython-314.pyc).

This copies each __pycache__/<name>.cpython-<ver>.pyc -> <parent>/<name>.pyc, but
ONLY when <parent>/<name>.py is absent (never shadow real source).

Usage:
  python tools/materialize_pyc.py             # report
  python tools/materialize_pyc.py --write      # copy sourceless pyc
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"venv", ".git", "artifacts", "__pycache__node"}


def iter_targets():
    for pyc in ROOT.rglob("__pycache__/*.pyc"):
        if any(part in SKIP_DIRS for part in pyc.parts):
            continue
        # feature.cpython-314.pyc -> feature
        stem = pyc.name.split(".")[0]
        parent = pyc.parent.parent  # drop __pycache__
        src = parent / f"{stem}.py"
        dest = parent / f"{stem}.pyc"
        if src.exists():
            continue  # real source present; never shadow
        yield pyc, dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    n = skipped = 0
    for pyc, dest in iter_targets():
        if dest.exists():
            skipped += 1
            continue
        if args.write:
            shutil.copy2(pyc, dest)
        n += 1
    verb = "copied" if args.write else "would copy"
    print(f"{verb} {n} sourceless .pyc (already present: {skipped})")
    if not args.write:
        print("re-run with --write to apply")


if __name__ == "__main__":
    main()
