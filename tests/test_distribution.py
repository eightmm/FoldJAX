"""The built wheel carries files that editable installs get from the checkout."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
AF3_UPSTREAM = Path("src/foldjax/models/alphafold3/_upstream")
OF3_RESOURCES = Path(
    "src/foldjax/models/openfold3/_upstream/openfold3/core/data/resources"
)
OF3_CONFIG = Path(
    "src/foldjax/models/openfold3/_upstream/openfold3/projects/of3_all_atom/config"
)
FORBIDDEN_RUNTIME_IMPORTS = {
    "esm",
    "lightning",
    "pytorch_lightning",
    "torch",
    "torchmetrics",
}


def test_supported_python_and_accelerator_runtime_are_consistent() -> None:
    """Metadata, lock, lint, and CI must name one supported runtime family."""
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = document["project"]
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert sys.version_info[:2] == (3, 13)
    assert project["requires-python"] == ">=3.13,<3.14"
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
    assert document["tool"]["ruff"]["target-version"] == "py313"
    assert (ROOT / ".python-version").read_text().strip() == "3.13"
    assert lock["requires-python"] == "==3.13.*"
    assert 'UV_PYTHON: "3.13"' in workflow
    assert workflow.count('python-version: "3.13"') == 2

    assert "jax==0.11.1" in dependencies
    assert "cuequivariance==0.11.1" in dependencies
    assert "cuequivariance-jax==0.11.1" in dependencies
    assert "flax==0.12.9" in dependencies
    assert "qwix==0.1.8" in dependencies
    assert "tokamax==0.0.13" in dependencies
    assert "jax[cuda12]==0.11.1" in extras["cuda12"]
    assert "jax[cuda13]==0.11.1" in extras["cuda13"]
    assert "cuequivariance-ops-jax-cu12==0.11.1" in extras["cuda12"]
    assert "cuequivariance-ops-jax-cu13==0.11.1" in extras["cuda13"]
    assert "triton==3.7.1" in extras["cuda12"]
    assert "triton==3.7.1" in extras["cuda13"]
    assert "dm-haiku==0.0.17" in extras["alphafold3"]


def test_alphafold3_uv_build_uses_the_active_python(
    tmp_path: Path, monkeypatch
) -> None:
    """The first-use extension wheel must match this process's Python ABI."""
    from foldjax.models.alphafold3 import build

    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/uv")

    command = build._build_command(tmp_path / "stage", tmp_path / "wheels")

    assert command[:4] == ["/usr/bin/uv", "build", "--python", sys.executable]


def _forbidden_python_imports(root: Path) -> list[str]:
    """Return direct forbidden imports below one installable package tree."""

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            for module in modules:
                if module.partition(".")[0] in FORBIDDEN_RUNTIME_IMPORTS:
                    relative = path.relative_to(root)
                    offenders.append(f"{relative}:{node.lineno}: {module}")
    return offenders


def test_installable_source_contains_no_external_tensor_runtime_imports() -> None:
    """Parity adapters belong in tests, never in the shipped source tree."""

    assert _forbidden_python_imports(ROOT / "src" / "foldjax") == []


def test_published_dependency_sets_do_not_install_torch() -> None:
    """Every advertised FoldJAX install remains a JAX-only environment.

    Reference/parity suites may compare against publisher implementations when
    a developer supplies those environments separately.  They are not product
    features and must never make ``uv sync --all-extras`` install PyTorch.
    """
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = document["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    for group in document.get("dependency-groups", {}).values():
        requirements.extend(group)

    forbidden = ("torch", "pytorch", "lightning", "torchmetrics", "fair-esm")
    offenders = [
        requirement
        for requirement in requirements
        if any(name in requirement.lower() for name in forbidden)
    ]
    assert offenders == []

    locked = tomllib.loads((ROOT / "uv.lock").read_text())
    locked_names = {package["name"].lower() for package in locked["package"]}
    locked_offenders = sorted(
        name
        for name in locked_names
        if any(token in name for token in forbidden)
    )
    assert locked_offenders == []


def test_public_runtime_import_graph_blocks_external_tensor_runtimes() -> None:
    """Import production surfaces with a hard Torch/fair-esm tripwire."""
    probe = r'''
import importlib
import importlib.abc
import sys
import tomllib

blocked_roots = {"esm", "lightning", "pytorch_lightning", "torch", "torchmetrics"}

class ExternalRuntimeBlocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked_roots:
            raise AssertionError(f"production imported forbidden module {fullname}")
        return None

sys.meta_path.insert(0, ExternalRuntimeBlocked())
with open("pyproject.toml", "rb") as handle:
    document = tomllib.load(handle)
script_modules = tuple(
    sorted(
        {
            target.partition(":")[0]
            for target in document["project"]["scripts"].values()
        }
    )
)
modules = (
    "foldjax",
    "foldjax.api",
    "foldjax.assets",
    "foldjax.backends.alphafold3",
    "foldjax.backends.boltz2",
    "foldjax.backends.esmfold2",
    "foldjax.backends.opendde",
    "foldjax.backends.openfold3",
    "foldjax.backends.protenix",
    "foldjax.models.boltz2.api",
    "foldjax.models.esmfold2.inference",
    "foldjax.models.opendde.cli.predict",
    "foldjax.models.openfold3.data",
    "foldjax.models.openfold3.inference",
    "foldjax.models.protenix.cli.predict",
    *script_modules,
)
for name in modules:
    importlib.import_module(name)
assert not any(name.partition(".")[0] in blocked_roots for name in sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_real_weight_import_blocker_cannot_be_caught_as_optional_dependency() -> None:
    """An ``except ImportError`` probe must not hide a forbidden import attempt."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "tests" / "no_torch")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "try:\n import torch\nexcept ImportError:\n pass\n",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode != 0
    assert "JAX-only verification blocked an import of 'torch'" in completed.stdout


def _clean_build_tree(destination: Path) -> list[Path]:
    """Copy distribution inputs without checkout-only generated artifacts.

    This deliberately does not depend on ``.git``: release tests must also run
    from an unpacked sdist. New source files in the working tree are included,
    while ignored AlphaFold 3 binaries/pickles and Python caches are excluded.
    """
    relative_paths = [
        Path(name)
        for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE")
        if (ROOT / name).is_file()
    ]
    relative_paths.extend(
        path.relative_to(ROOT)
        for path in (ROOT / "src").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
        and path.suffix not in {".pyc", ".pickle", ".so"}
    )
    for relative in relative_paths:
        source = ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return relative_paths


def _wheel_path(source_path: Path) -> str:
    return source_path.relative_to("src").as_posix()


def _extract_wheel_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract wheel members only after proving each stays in destination."""
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        assert target.is_relative_to(root), member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def test_wheel_carries_alphafold3_first_use_build_inputs(tmp_path: Path) -> None:
    """A clean wheel must be able to run ``build.ensure_extensions``.

    The checkout may already contain an ignored ``cpp*.so`` and generated CCD
    pickles, so building it directly can hide a broken source distribution. A
    clean tree reproduces what a fresh clone and a release build see.
    """
    source_tree = tmp_path / "source"
    inputs = _clean_build_tree(source_tree)
    output = tmp_path / "dist"
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("distribution smoke requires the uv build frontend")
    completed = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(output), str(source_tree)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout

    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())

    required = {
        _wheel_path(AF3_UPSTREAM / name)
        for name in (
            "CMakeLists.txt",
            "pyproject.toml",
            "LICENSE",
            "NOTICE",
            "OUTPUT_TERMS_OF_USE.md",
            "WEIGHTS_PROHIBITED_USE_POLICY.md",
            "WEIGHTS_TERMS_OF_USE.md",
        )
    }
    required.update(
        _wheel_path(path)
        for path in inputs
        if path.is_relative_to(AF3_UPSTREAM)
        and path.suffix in {".cc", ".h", ".py", ".pyi"}
    )
    required.add("foldjax/redaction.py")
    required.update(
        {
            "foldjax/models/openfold3/LICENSE",
            "foldjax/models/openfold3/NOTICE",
        }
    )
    required.add(
        "foldjax/backends/_alphafold3_upstream/run_alphafold.py"
    )
    openfold_resources = {
        _wheel_path(path)
        for path in inputs
        if (
            path.is_relative_to(OF3_RESOURCES)
            and path.suffix in {".json", ".sdf"}
        )
        or (path.is_relative_to(OF3_CONFIG) and path.suffix == ".yml")
    }
    assert len(openfold_resources) == 24
    required.update(openfold_resources)

    missing = sorted(required - members)
    message = "required runtime/package files missing from wheel:\n"
    assert not missing, message + "\n".join(missing)

    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        _extract_wheel_safely(archive, installed)
    assert _forbidden_python_imports(installed / "foldjax") == []
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(installed)!r});"
                "import foldjax,foldjax.redaction;"
                "from foldjax.models.alphafold3 import build;"
                "assert build.source_package().is_dir();"
                "assert (build.source_package().parent/'CMakeLists.txt').is_file();"
                "from importlib.resources import files;"
                "resources=files("
                "'foldjax.models.openfold3._upstream.openfold3.core.data.resources'"
                ");"
                "assert resources.joinpath("
                "'canonical_protein_data/reference_mols/ALA.sdf'"
                ").is_file();"
                "config=files("
                "'foldjax.models.openfold3._upstream.openfold3.projects."
                "of3_all_atom.config'"
                ");"
                "assert config.joinpath('model_setting_presets.yml').is_file()"
            ),
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert probe.returncode == 0, probe.stdout


def test_alphafold3_first_use_outputs_go_to_the_writable_store(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    upstream = tmp_path / "installed" / "_upstream"
    package = upstream / "alphafold3"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "version.py").write_text("__version__ = '3.0.2'\n")
    for name, contents in {
        "pyproject.toml": (
            '[build-system]\nrequires = [\n    "numpy",\n]\n'
            '[project]\ndynamic = ["version"]\n'
        ),
        "CMakeLists.txt": "# staged build\n",
        "LICENSE": "Apache-2.0\n",
        "OUTPUT_TERMS_OF_USE.md": "terms\n",
        "WEIGHTS_PROHIBITED_USE_POLICY.md": "policy\n",
        "WEIGHTS_TERMS_OF_USE.md": "terms\n",
    }.items():
        (upstream / name).write_text(contents)

    home = tmp_path / "writable-home"
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setattr(build, "_UPSTREAM", upstream)
    monkeypatch.setattr(build, "_PACKAGE", package)
    extension_name = f"cpp{build._extension_suffix()}"

    def fake_build(command, *, check):
        assert check is True
        project = Path(command[-1])
        staged = (project / "pyproject.toml").read_text()
        assert f'version = "{build.source_version()}"' in staged
        import numpy as np

        assert f'"numpy=={np.__version__}"' in staged
        output = Path(command[command.index("--out-dir") + 1])
        output.mkdir(parents=True)
        with zipfile.ZipFile(output / "alphafold3-3.0.4.whl", "w") as archive:
            archive.writestr(f"alphafold3/{extension_name}", b"extension")
            for name in build._LIBCIFPP_FILES:
                archive.writestr(f"share/libcifpp/{name}", b"runtime data\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build.subprocess, "run", fake_build)
    # Reproduce an immutable wheel/Nix-style installation. Runtime copies must
    # not inherit modes that make their second write or generated files fail.
    (package / "__init__.py").chmod(0o444)
    package.chmod(0o555)
    try:
        built = build.ensure_extensions()
    finally:
        package.chmod(0o755)
        (package / "__init__.py").chmod(0o644)

    assert built == build.runtime_package() / extension_name
    assert built.read_bytes() == b"extension"
    assert build.runtime_root().is_relative_to(home / "runtime/alphafold3")
    assert (build.runtime_root() / "share/libcifpp/components.cif").is_file()
    assert (build.runtime_root() / build._EXTENSION_MARKER).is_file()
    assert (build.runtime_package() / "__init__.py").stat().st_mode & 0o200
    assert not list(package.glob("cpp*.so")), "site-packages/source stays untouched"


def test_alphafold3_runtime_key_changes_with_source(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    upstream = tmp_path / "_upstream"
    package = upstream / "alphafold3"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("REVISION = 1\n")
    (upstream / "CMakeLists.txt").write_text("# build\n")
    (upstream / "pyproject.toml").write_text("# package\n")
    monkeypatch.setattr(build, "_UPSTREAM", upstream)
    monkeypatch.setattr(build, "_PACKAGE", package)

    first = build.runtime_key()
    module.write_text("REVISION = 2\n")

    assert build.runtime_key() != first


def test_alphafold3_runtime_key_changes_with_numpy_build_abi(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np

    from foldjax.models.alphafold3 import build

    upstream = tmp_path / "_upstream"
    package = upstream / "alphafold3"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (upstream / "CMakeLists.txt").write_text("# build\n")
    (upstream / "pyproject.toml").write_text("# package\n")
    monkeypatch.setattr(build, "_UPSTREAM", upstream)
    monkeypatch.setattr(build, "_PACKAGE", package)

    first = build.runtime_key()
    monkeypatch.setattr(np, "__version__", "999.0-test")

    assert build.runtime_key() != first


def test_alphafold3_readiness_ignores_an_unimported_distribution(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    external = tmp_path / "site-packages/alphafold3/__init__.py"
    external.parent.mkdir(parents=True)
    external.touch()
    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    monkeypatch.setattr(
        build.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(external)),
    )
    monkeypatch.setattr(build, "_runtime_is_complete", lambda: True)

    assert build.is_ready() is True
    assert build.runtime_blocker() is None


def test_alphafold3_readiness_requires_the_managed_runtime(
    monkeypatch,
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    monkeypatch.setattr(build, "_runtime_is_complete", lambda: False)

    assert build.is_ready() is False
    assert build.runtime_blocker() is None


def test_alphafold3_loaded_external_package_blocks_managed_selection(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    origin = tmp_path / "site-packages/alphafold3/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    package = ModuleType("alphafold3")
    package.__file__ = str(origin)
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setattr(build, "_runtime_is_complete", lambda: True)
    monkeypatch.setattr(
        build,
        "_ensure_extensions_locked",
        lambda: pytest.fail("a loaded native package must not be replaced"),
    )

    assert build.is_ready() is False
    blocker = build.runtime_blocker()
    assert blocker is not None
    assert "already imported" in blocker
    assert str(origin.parent) in blocker
    with pytest.raises(RuntimeError, match="already imported"):
        build.ensure_ready()


def test_alphafold3_readiness_uses_the_managed_runtime_without_an_install(
    monkeypatch,
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    monkeypatch.setattr(build, "_runtime_is_complete", lambda: True)

    assert build.is_ready() is True


def test_alphafold3_managed_path_has_temporary_import_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    package = tmp_path / "runtime/key/alphafold3"
    package.mkdir(parents=True)
    before = tuple(sys.path)

    with build._managed_package_on_path(package):
        assert sys.path[0] == str(package.parent)

    assert tuple(sys.path) == before


def test_alphafold3_never_reuses_unmarked_checkout_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import build

    upstream = tmp_path / "checkout" / "_upstream"
    package = upstream / "alphafold3"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "version.py").write_text("__version__ = '3.0.2'\n")
    (upstream / "CMakeLists.txt").write_text("# build\n")
    (upstream / "pyproject.toml").write_text("# package\n")
    source_extension = package / f"cpp{build._extension_suffix()}"
    source_extension.write_bytes(b"stale extension")
    for relative in build._PICKLES:
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stale generated data")
    for name in build._LIBCIFPP_FILES:
        target = upstream / "share/libcifpp" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stale dictionary")

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(build, "_UPSTREAM", upstream)
    monkeypatch.setattr(build, "_PACKAGE", package)
    calls = []

    def fresh_build(runtime_package: Path, runtime_root: Path) -> None:
        calls.append(runtime_package)
        (runtime_package / f"cpp{build._extension_suffix()}").write_bytes(b"fresh")
        for name in build._LIBCIFPP_FILES:
            target = runtime_root / "share/libcifpp" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fresh dictionary")

    monkeypatch.setattr(build, "_build_extension", fresh_build)

    compiled = build.ensure_extensions()

    assert calls
    assert compiled.read_bytes() == b"fresh"
    assert source_extension.read_bytes() == b"stale extension"


def test_alphafold3_interrupted_ccd_generation_repairs_on_retry(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import _upstream as shim
    from foldjax.models.alphafold3 import build

    upstream = tmp_path / "installed" / "_upstream"
    package = upstream / "alphafold3"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (upstream / "CMakeLists.txt").write_text("# build\n")
    (upstream / "pyproject.toml").write_text("# package\n")
    home = tmp_path / "home"
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setattr(build, "_UPSTREAM", upstream)
    monkeypatch.setattr(build, "_PACKAGE", package)

    runtime_package = build.runtime_package()
    runtime_package.mkdir(parents=True)
    (runtime_package / f"cpp{build._extension_suffix()}").write_bytes(b"extension")
    for name in build._LIBCIFPP_FILES:
        target = build.runtime_root() / "share/libcifpp" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"runtime data")
    build._write_marker(build.runtime_root() / build._EXTENSION_MARKER)

    monkeypatch.setattr(shim, "ensure_registered", lambda **kwargs: None)
    alphafold3 = ModuleType("alphafold3")
    alphafold3.__file__ = str(runtime_package / "__init__.py")
    alphafold3.__foldjax_managed_package__ = str(runtime_package)
    constants = ModuleType("alphafold3.constants")
    converters = ModuleType("alphafold3.constants.converters")
    ccd_generator = SimpleNamespace(
        main=lambda arguments: Path(arguments[-1]).write_bytes(b"complete ccd")
    )
    attempts = 0

    def generate_sets(arguments) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("interrupted")
        Path(arguments[-1]).write_bytes(b"complete sets")

    sets_generator = SimpleNamespace(main=generate_sets)
    converters.ccd_pickle_gen = ccd_generator
    converters.chemical_component_sets_gen = sets_generator
    for name, module in {
        "alphafold3": alphafold3,
        "alphafold3.constants": constants,
        "alphafold3.constants.converters": converters,
        "alphafold3.constants.converters.ccd_pickle_gen": ccd_generator,
        "alphafold3.constants.converters.chemical_component_sets_gen": sets_generator,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(RuntimeError, match="interrupted"):
        build.ensure_ready()
    assert not (build.runtime_root() / build._READY_MARKER).exists()
    assert not list(runtime_package.rglob("*.tmp"))

    build.ensure_ready()

    assert build.active_package() == runtime_package
    assert (build.runtime_root() / build._READY_MARKER).is_file()
