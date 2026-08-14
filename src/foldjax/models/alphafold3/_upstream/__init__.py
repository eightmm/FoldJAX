"""Import the vendored AlphaFold 3 package under its own name.

Upstream's modules import each other absolutely (``from alphafold3.model import
...``), and its C++ extensions are compiled with ``alphafold3.*`` module names,
so rewriting the tree to a FoldJAX-prefixed spelling -- the OpenFold3 vendoring
approach -- would strand every compiled module. Registering the vendored root
as ``alphafold3`` keeps both the Python and the pybind halves verbatim.

FoldJAX's managed-runtime registration gives this versioned package explicit
precedence. An external checkout is accepted only through the backend's
``source`` option; a package already imported by another caller is never
replaced in place.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_MANAGED_PACKAGE_ATTR = "__foldjax_managed_package__"


def _managed_package(module: object) -> Path | None:
    value = getattr(module, _MANAGED_PACKAGE_ATTR, None)
    if value is None:
        return None
    try:
        return Path(value).resolve()
    except (OSError, TypeError, ValueError):
        return None


def _configure_libcifpp(package: Path, *, replace_managed: bool = False) -> None:
    """Point the compiled extension at this package's chemistry dictionaries."""
    import os

    from foldjax.models.alphafold3 import build

    data = package.parent / "share" / "libcifpp"
    if not data.is_dir():
        return
    current = os.environ.get("LIBCIFPP_DATA_DIR")
    current_path = Path(current) if current else None
    current_valid = current_path is not None and current_path.is_dir()
    looks_managed = (
        current_path is not None
        and "runtime" in current_path.parts
        and "alphafold3" in current_path.parts
    )
    if (
        current is None
        or not current_valid
        or build.is_managed_origin(current_path / "placeholder")
        or (replace_managed and looks_managed)
    ):
        os.environ["LIBCIFPP_DATA_DIR"] = str(data)


def _same_package(module: object, package: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is None:
        return False
    try:
        return Path(origin).resolve().parent == package.resolve()
    except OSError:
        return False


def ensure_registered(*, package: Path | None = None) -> None:
    """Make ``import alphafold3`` resolve, preferring an installed copy.

    ``package`` is used only while FoldJAX is generating its runtime overlay,
    before the full-runtime completion marker can legitimately expose it via
    :func:`build.active_package`.
    """
    from foldjax.models.alphafold3 import build

    requested = Path(package) if package is not None else None
    existing = sys.modules.get("alphafold3")
    if existing is not None:
        managed_package = _managed_package(existing)
        existing_managed = managed_package is not None or build.is_managed_origin(
            getattr(existing, "__file__", None)
        )
        if requested is None and not existing_managed:
            return
        desired = requested or build.active_package()
        if _same_package(existing, desired):
            _configure_libcifpp(desired, replace_managed=managed_package is not None)
            return
        if not existing_managed:
            raise RuntimeError(
                "an independently installed AlphaFold 3 package is already "
                "imported; FoldJAX will not replace it with a generated runtime"
            )
        raise RuntimeError(
            "a different FoldJAX-managed AlphaFold 3 runtime is already imported; "
            "restart this Python process after upgrading FoldJAX"
        )
    found = importlib.util.find_spec("alphafold3")
    if found is not None and not build.is_managed_origin(found.origin):
        if requested is None:
            return
        raise RuntimeError(
            "an independently installed AlphaFold 3 package is available; "
            "FoldJAX will not replace it with a generated runtime"
        )

    package = requested or build.active_package()
    _configure_libcifpp(package, replace_managed=True)
    if package == build.source_package():
        module = importlib.import_module(f"{__name__}.alphafold3")
    else:
        sys.path.insert(0, str(package.parent))
        try:
            module = importlib.import_module("alphafold3")
        finally:
            try:
                sys.path.remove(str(package.parent))
            except ValueError:  # pragma: no cover - defensive against import hooks
                pass
    setattr(module, _MANAGED_PACKAGE_ATTR, str(package.resolve()))
    sys.modules["alphafold3"] = module
