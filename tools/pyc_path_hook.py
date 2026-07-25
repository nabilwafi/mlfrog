"""Path hook: import modules from __pycache__/*.pyc when the .py source is missing.

Needed because this tree was source-stripped to Python 3.14 bytecode.
Install once at process start (apps/*.py call install()).

ponytail: covers cpython-314 sourceless packages only; rebuild .py when you can.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_INSTALLED = False


class _PycFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if path is None:
            search = [Path(p) for p in sys.path if p]
        else:
            search = [Path(p) for p in path]
        parts = fullname.split(".")
        name = parts[-1]
        for root in search:
            # package: root/pkg/__pycache__/__init__.cpython-*.pyc
            # module:  root/pkg/__pycache__/mod.cpython-*.pyc
            if len(parts) == 1:
                base = root / name
            else:
                # parent package already on path as its __path__ entries
                base = root / name if (root / name).is_dir() or list((root / "__pycache__").glob(f"{name}.cpython-*.pyc")) else root / name

            # Try as package
            pkg_dir = root / name if len(parts) >= 1 else None
            if pkg_dir and pkg_dir.is_dir():
                init_hits = list((pkg_dir / "__pycache__").glob("__init__.cpython-*.pyc"))
                if init_hits and not (pkg_dir / "__init__.py").exists():
                    return importlib.util.spec_from_file_location(
                        fullname,
                        init_hits[0],
                        loader=importlib.machinery.SourcelessFileLoader(fullname, str(init_hits[0])),
                        submodule_search_locations=[str(pkg_dir)],
                    )
            # Try as module inside a path entry
            cache = root / "__pycache__"
            hits = list(cache.glob(f"{name}.cpython-*.pyc")) if cache.is_dir() else []
            py = root / f"{name}.py"
            if hits and not py.exists():
                return importlib.util.spec_from_file_location(
                    fullname,
                    hits[0],
                    loader=importlib.machinery.SourcelessFileLoader(fullname, str(hits[0])),
                )
        return None


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    # Insert before PathFinder so we catch missing sources first.
    sys.meta_path.insert(0, _PycFinder())
    _INSTALLED = True


# Auto-install when this module is imported.
install()
