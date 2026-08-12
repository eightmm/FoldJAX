"""Build the vendored AlphaFold 3's compiled half, once, in place.

The vendored tree at ``._upstream`` is complete Python but AlphaFold 3 keeps
its structure/mmCIF machinery in one pybind module, ``alphafold3.cpp``, and
its chemical-component tables in two generated pickles. Neither belongs in
git -- one is an ABI-specific binary, the other is 513 MB of generated data --
so both are produced on the machine that will use them, the same class of
one-time step as fetching weights.

``ensure_extensions`` stages upstream's own build (its pyproject and
CMakeLists are vendored beside the source) in a scratch directory, wheels it
with the current interpreter, and copies ``cpp*.so`` into the vendored
package. ``ensure_ccd`` copies the pickles from an installed ``alphafold3``
distribution when one exists, and otherwise runs upstream's ``build_data``
generator against libcifpp's component dictionary. Both are no-ops when the
artifact is already in place.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_UPSTREAM = Path(__file__).parent / "_upstream"
_PACKAGE = _UPSTREAM / "alphafold3"
_PICKLES = (
    "constants/converters/ccd.pickle",
    "constants/converters/chemical_component_sets.pickle",
)


def compiled_module() -> Path | None:
    """The vendored ``alphafold3.cpp`` extension, if it has been built."""
    matches = sorted(_PACKAGE.glob("cpp*.so"))
    return matches[0] if matches else None


def ensure_extensions() -> Path:
    """Build ``alphafold3.cpp`` into the vendored tree if it is missing."""
    existing = compiled_module()
    if existing is not None:
        return existing

    with tempfile.TemporaryDirectory(prefix="foldjax-af3-build-") as scratch:
        stage = Path(scratch) / "project"
        (stage / "src").mkdir(parents=True)
        pyproject = (_UPSTREAM / "pyproject.toml").read_text()
        # The staged copy is not a git checkout, so setuptools_scm has no
        # version to derive; pin the vendored revision's own. LICENSE rides
        # along for the same reason: metadata validation wants the file.
        pyproject = pyproject.replace(
            'dynamic = ["version"]', 'version = "3.0.4"'
        ).replace(
            'metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"',
            "",
        ).replace(
            'sdist.include = ["src/alphafold3/_version.py"]', ""
        )
        pyproject = "\n".join(
            line
            for line in pyproject.splitlines()
            if "setuptools_scm" not in line
            and "setuptools-scm" not in line
            and "fallback_version" not in line
        )
        (stage / "pyproject.toml").write_text(pyproject)
        # Metadata validation wants the files pyproject names, not their prose.
        (stage / "README.md").write_text("Vendored AlphaFold 3 build stage.\n")
        shutil.copy2(_UPSTREAM / "LICENSE", stage / "LICENSE")
        for name in (
            "OUTPUT_TERMS_OF_USE.md",
            "WEIGHTS_PROHIBITED_USE_POLICY.md",
            "WEIGHTS_TERMS_OF_USE.md",
        ):
            shutil.copy2(_UPSTREAM / name, stage / name)
        shutil.copy2(_UPSTREAM / "CMakeLists.txt", stage / "CMakeLists.txt")
        (stage / "src" / "alphafold3").symlink_to(_PACKAGE)
        wheels = Path(scratch) / "wheels"
        # `uv build` runs the PEP 517 backend (scikit-build-core -> CMake) in
        # an isolated environment; uv venvs carry no pip of their own.
        uv = shutil.which("uv")
        if uv is not None:
            command = [uv, "build", "--wheel", "--out-dir", str(wheels), str(stage)]
        else:
            command = [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(stage),
            ]
        subprocess.run(command, check=True)
        wheel = next(wheels.glob("alphafold3-*.whl"))
        import zipfile

        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.startswith("alphafold3/cpp") and name.endswith(".so"):
                    target = _PACKAGE / Path(name).name
                    with archive.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    built = compiled_module()
    if built is None:
        raise RuntimeError("the AlphaFold 3 extension build produced no cpp*.so")
    return built


def ensure_ccd() -> None:
    """Put the chemical-component pickles in place if they are missing."""
    missing = [name for name in _PICKLES if not (_PACKAGE / name).exists()]
    if not missing:
        return

    spec = importlib.util.find_spec("alphafold3")
    if spec is not None and spec.origin is not None:
        installed = Path(spec.origin).parent
        if installed != _PACKAGE:
            copied = []
            for name in missing:
                source = installed / name
                if source.exists():
                    shutil.copy2(source, _PACKAGE / name)
                    copied.append(name)
            missing = [name for name in missing if name not in copied]
    if not missing:
        return

    # The cold path: upstream's own generator, against libcifpp's component
    # dictionary (the CMake build installs it into the environment).
    from foldjax.models.alphafold3 import _upstream as shim

    shim.ensure_registered()
    from alphafold3 import build_data

    build_data.build_data()
    still = [name for name in _PICKLES if not (_PACKAGE / name).exists()]
    if still:
        raise RuntimeError(f"CCD generation left artifacts missing: {still}")


def ensure_ready() -> None:
    """Everything the vendored predict path needs, built or copied once."""
    ensure_extensions()
    ensure_ccd()
