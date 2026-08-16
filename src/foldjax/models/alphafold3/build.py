"""Build AlphaFold 3's generated half in FoldJAX's writable runtime store.

The vendored tree at ``._upstream`` contains the complete Python/C++ source,
but not the ABI-specific ``alphafold3.cpp`` extension or the two generated CCD
pickles. A wheel's site-packages may be immutable, so generated files live in a
source-and-ABI-keyed directory below ``$FOLDJAX_HOME/runtime/alphafold3``.

First use is guarded by both a thread lock and an inter-process file lock.
Extension files are built in a sibling staging tree and atomically published;
the large CCD pickles are each generated into a temporary file and renamed
only after validation. Completion markers ensure an interrupted first use is
repaired rather than mistaken for a usable installation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_UPSTREAM = Path(__file__).parent / "_upstream"
_PACKAGE = _UPSTREAM / "alphafold3"
_PICKLES = (
    "constants/converters/ccd.pickle",
    "constants/converters/chemical_component_sets.pickle",
)
_LIBCIFPP_FILES = (
    "components.cif",
    "mmcif_ddl.dic",
    "mmcif_ma.dic",
    "mmcif_pdbx.dic",
)
_EXTENSION_MARKER = ".foldjax-extension-complete"
_READY_MARKER = ".foldjax-complete"
_RUNTIME_SCHEMA = "4"
_THREAD_LOCK = RLock()


def runtime_base() -> Path:
    """Root shared by every source/ABI generation of AlphaFold 3."""
    from foldjax.paths import runtime_dir

    return runtime_dir("alphafold3")


def _fingerprint_files() -> list[Path]:
    paths = [_UPSTREAM / "CMakeLists.txt", _UPSTREAM / "pyproject.toml"]
    paths.extend(
        path
        for path in _PACKAGE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pickle", ".so"))
    )
    return sorted(paths, key=lambda path: path.relative_to(_UPSTREAM).as_posix())


def runtime_key() -> str:
    """Stable identity of the vendored source and this interpreter's ABI."""
    import numpy as np

    digest = hashlib.sha256()
    identity = (
        _RUNTIME_SCHEMA,
        sys.implementation.cache_tag or "unknown-python",
        str(sysconfig.get_config_var("SOABI") or "unknown-soabi"),
        sysconfig.get_platform(),
        platform.machine(),
        # The extension is compiled against NumPy's C headers. Pinning the
        # isolated build to this version below and including it here prevents a
        # runtime built against one NumPy ABI from being reused with another.
        np.__version__,
    )
    digest.update("\0".join(identity).encode())
    for path in _fingerprint_files():
        relative = path.relative_to(_UPSTREAM).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:24]


def runtime_root() -> Path:
    """Runtime directory for exactly this source and Python ABI."""
    return runtime_base() / runtime_key()


def runtime_package() -> Path:
    return runtime_root() / "alphafold3"


def source_package() -> Path:
    return _PACKAGE


def source_version() -> str:
    """Version of the vendored package and runner snapshot."""
    version_source = (_PACKAGE / "version.py").read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_source)
    if match is None:
        raise RuntimeError("vendored AlphaFold 3 version.py has no __version__")
    return match.group(1)


def _extension_suffix() -> str:
    return str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _extension(package: Path) -> Path | None:
    candidate = package / f"cpp{_extension_suffix()}"
    return candidate if _nonempty(candidate) else None


def _libcifpp_complete(root: Path) -> bool:
    data = root / "share" / "libcifpp"
    return all(_nonempty(data / name) for name in _LIBCIFPP_FILES)


def _marker_matches(path: Path) -> bool:
    try:
        return path.read_text(encoding="utf-8").strip() == runtime_key()
    except OSError:
        return False


def _runtime_extensions_complete() -> bool:
    root = runtime_root()
    return (
        _marker_matches(root / _EXTENSION_MARKER)
        and _extension(root / "alphafold3") is not None
        and _libcifpp_complete(root)
    )


def _runtime_is_complete() -> bool:
    root = runtime_root()
    return (
        _runtime_extensions_complete()
        and _marker_matches(root / _READY_MARKER)
        and all(_nonempty(root / "alphafold3" / name) for name in _PICKLES)
    )


def runtime_blocker() -> str | None:
    """Why the managed runtime cannot be selected in this Python process.

    An independently *installed* package is deliberately ignored: FoldJAX's
    source/ABI-keyed runtime owns the default execution path. Only a package
    already imported into this process is unsafe to replace because its native
    extension may already have initialized process-global state.
    """
    loaded = sys.modules.get("alphafold3")
    if loaded is None:
        return None
    recorded = getattr(loaded, "__foldjax_managed_package__", None)
    if recorded is not None:
        try:
            if Path(recorded).resolve() == runtime_package().resolve():
                return None
        except (OSError, TypeError, ValueError):
            pass
        return (
            "A different FoldJAX-managed AlphaFold 3 runtime is already "
            "imported. Restart Python after changing FOLDJAX_HOME or upgrading "
            "FoldJAX."
        )
    origin = getattr(loaded, "__file__", None)
    if origin is not None and is_managed_origin(origin):
        return None
    location = "unknown location" if origin is None else str(Path(origin).parent)
    return (
        "An external AlphaFold 3 package is already imported from "
        f"{location}. Restart Python before using FoldJAX's managed runtime; "
        "FoldJAX will not replace a loaded native package. External checkouts "
        "are used only through the explicit `source` backend option."
    )


def is_ready() -> bool:
    """Whether FoldJAX's managed AlphaFold 3 runtime is selectable now."""
    return runtime_blocker() is None and _runtime_is_complete()


def active_package() -> Path:
    """Complete package tree that the import shim should register."""
    if _runtime_is_complete():
        return runtime_package()
    raise RuntimeError(
        "the AlphaFold 3 runtime has not been prepared. Run "
        "`foldjax runtime prepare --model alphafold3` once; it builds the "
        "ABI-specific extension and CCD tables under the FoldJAX runtime store. "
        "(Library callers: foldjax.models.alphafold3.build.ensure_ready().)"
    )


def is_managed_origin(origin: str | Path | None) -> bool:
    """Whether an import spec points at FoldJAX's source or runtime overlay."""
    if origin is None:
        return False
    try:
        resolved = Path(origin).resolve()
        return resolved.is_relative_to(_UPSTREAM.resolve()) or resolved.is_relative_to(
            runtime_base().resolve()
        )
    except (OSError, RuntimeError, ValueError):
        return False


@contextmanager
def _managed_package_on_path(package: Path) -> Iterator[None]:
    """Give one managed package precedence during its controlled import.

    An unrelated installed ``alphafold3`` must not win ``find_spec`` merely
    because it appears earlier on the ordinary environment path. The temporary
    entry exists only while the vendored import shim validates and registers
    the selected source/ABI-keyed package.
    """
    parent = str(package.parent)
    sys.path.insert(0, parent)
    try:
        yield
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:  # pragma: no cover - defensive against import hooks
            pass


@contextmanager
def _runtime_lock() -> Iterator[None]:
    """Serialize first use across FoldJAX threads and Linux GPU processes."""
    with _THREAD_LOCK:
        base = runtime_base()
        base.mkdir(parents=True, exist_ok=True)
        with (base / ".build.lock").open("a+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - FoldJAX targets Linux/CUDA
                yield
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _copy_source_tree(destination: Path) -> Path:
    """Copy source contents without inheriting immutable installation modes."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(_PACKAGE.rglob("*")):
        relative = source.relative_to(_PACKAGE)
        if "__pycache__" in relative.parts or source.name.endswith(
            (".pyc", ".pickle", ".so")
        ):
            continue
        target = destination / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(target.stat().st_mode | 0o700)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(target.stat().st_mode | 0o600)
    return destination


def _write_marker(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(runtime_key() + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_command(stage: Path, wheels: Path) -> list[str]:
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "build", "--wheel", "--out-dir", str(wheels), str(stage)]
    if importlib.util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheels),
            str(stage),
        ]
    raise RuntimeError(
        "AlphaFold 3 first use must compile its CPU extension, but neither "
        "`uv` nor this interpreter's `pip` module is available"
    )


def _stage_build_project(stage: Path, package: Path) -> None:
    import numpy as np

    (stage / "src").mkdir(parents=True)
    pyproject = (_UPSTREAM / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = pyproject.replace(
        'dynamic = ["version"]', f'version = "{source_version()}"'
    ).replace(
        'metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"',
        "",
    ).replace('sdist.include = ["src/alphafold3/_version.py"]', "")
    pyproject = "\n".join(
        line
        for line in pyproject.splitlines()
        if "setuptools_scm" not in line
        and "setuptools-scm" not in line
        and "fallback_version" not in line
    )
    pyproject = pyproject.replace(
        '    "numpy",', f'    "numpy=={np.__version__}",', 1
    )
    if f'    "numpy=={np.__version__}",' not in pyproject:
        raise RuntimeError(
            "vendored AlphaFold 3 build metadata no longer exposes its NumPy "
            "build requirement; refusing an ABI-unpinned extension build"
        )
    (stage / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (stage / "README.md").write_text(
        "Vendored AlphaFold 3 build stage.\n", encoding="utf-8"
    )
    shutil.copyfile(_UPSTREAM / "LICENSE", stage / "LICENSE")
    for name in (
        "OUTPUT_TERMS_OF_USE.md",
        "WEIGHTS_PROHIBITED_USE_POLICY.md",
        "WEIGHTS_TERMS_OF_USE.md",
    ):
        shutil.copyfile(_UPSTREAM / name, stage / name)
    shutil.copyfile(_UPSTREAM / "CMakeLists.txt", stage / "CMakeLists.txt")
    (stage / "src" / "alphafold3").symlink_to(package)


def _extract_build_wheel(wheel: Path, package: Path, root: Path) -> None:
    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if name.startswith("alphafold3/cpp") and name.endswith(
                _extension_suffix()
            ):
                target = package / f"cpp{_extension_suffix()}"
            elif name.startswith("share/libcifpp/"):
                relative = Path(name).relative_to("share/libcifpp")
                target = root / "share" / "libcifpp" / relative
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _build_extension(package: Path, root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="foldjax-af3-build-") as scratch:
        stage = Path(scratch) / "project"
        _stage_build_project(stage, package)
        wheels = Path(scratch) / "wheels"
        subprocess.run(_build_command(stage, wheels), check=True)
        built = sorted(wheels.glob("alphafold3-*.whl"))
        if len(built) != 1:
            raise RuntimeError(
                "AlphaFold 3 extension build did not produce exactly one wheel"
            )
        _extract_build_wheel(built[0], package, root)


def _ensure_extensions_locked() -> Path:
    if _runtime_extensions_complete():
        extension = _extension(runtime_package())
        assert extension is not None
        return extension

    root = runtime_root()
    if root.exists():
        shutil.rmtree(root)
    for abandoned in runtime_base().glob(".*-*"):
        if abandoned.is_dir() and abandoned.name != ".build.lock":
            shutil.rmtree(abandoned)
    staged = Path(tempfile.mkdtemp(prefix=f".{runtime_key()}-", dir=runtime_base()))
    try:
        package = _copy_source_tree(staged / "alphafold3")
        # Ignored ``cpp*.so`` files in a source checkout carry no provenance
        # marker. Reusing one can silently pair changed Python/C++ source with
        # a stale binary, so every source/ABI key builds its own extension.
        _build_extension(package, staged)
        if _extension(package) is None or not _libcifpp_complete(staged):
            raise RuntimeError(
                "AlphaFold 3 extension build left native or libcifpp files missing"
            )
        _write_marker(staged / _EXTENSION_MARKER)
        staged.replace(root)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    extension = _extension(runtime_package())
    if extension is None:
        raise RuntimeError("AlphaFold 3 extension publication failed")
    return extension


def compiled_module() -> Path | None:
    """Current-ABI ``alphafold3.cpp`` extension, if safely published."""
    if _runtime_extensions_complete():
        return _extension(runtime_package())
    return None


def ensure_extensions() -> Path:
    """Build and atomically publish ``alphafold3.cpp`` and libcifpp data."""
    with _runtime_lock():
        return _ensure_extensions_locked()


def _generate_atomic(target: Path, generate: Callable[[Path], None]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        generate(temporary)
        if not _nonempty(temporary):
            raise RuntimeError(f"AlphaFold 3 generator produced an empty {target.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_ccd_locked() -> None:
    if _runtime_is_complete():
        return
    _ensure_extensions_locked()
    package = runtime_package()
    converters = package / "constants" / "converters"
    for abandoned in converters.glob(".*.tmp"):
        abandoned.unlink(missing_ok=True)

    from foldjax.models.alphafold3 import _upstream as shim

    with _managed_package_on_path(package):
        shim.ensure_registered(package=package)
    ccd = converters / "ccd.pickle"
    if not _nonempty(ccd):
        from alphafold3.constants.converters import ccd_pickle_gen

        components = runtime_root() / "share" / "libcifpp" / "components.cif"
        _generate_atomic(
            ccd,
            lambda target: ccd_pickle_gen.main(["", str(components), str(target)]),
        )
    chemical_sets = converters / "chemical_component_sets.pickle"
    if not _nonempty(chemical_sets):
        from alphafold3.constants.converters import chemical_component_sets_gen

        _generate_atomic(
            chemical_sets,
            lambda target: chemical_component_sets_gen.main(["", str(target)]),
        )
    if not all(_nonempty(package / name) for name in _PICKLES):
        raise RuntimeError("AlphaFold 3 CCD generation left artifacts missing")
    _write_marker(runtime_root() / _READY_MARKER)


def ensure_ccd() -> None:
    """Atomically materialize AlphaFold 3's generated chemistry tables."""
    with _runtime_lock():
        _ensure_ccd_locked()


def ensure_ready() -> None:
    """Prepare a complete vendored runtime once, repairing interrupted work."""
    blocker = runtime_blocker()
    if blocker is not None:
        raise RuntimeError(blocker)
    if is_ready():
        return
    with _runtime_lock():
        blocker = runtime_blocker()
        if blocker is not None:
            raise RuntimeError(blocker)
        if is_ready():
            return
        _ensure_extensions_locked()
        _ensure_ccd_locked()


def register_runtime() -> Path:
    """Register FoldJAX's prepared package under the upstream import name.

    This is the only default registration path. It intentionally ignores an
    unimported third-party installation and refuses to replace an external
    package that has already been imported into the process.
    """
    ensure_ready()
    package = active_package()
    from foldjax.models.alphafold3 import _upstream as shim

    with _managed_package_on_path(package):
        shim.ensure_registered(package=package)
    return package
