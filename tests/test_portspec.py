"""The per-port table, and the two properties that make it safe to have one.

The table replaced forty `request.model == "..."` branches in `manifest.py`,
`assets.py` and `input.py`. Two things about it are load-bearing rather than
stylistic, and both are asserted here:

* **The tracked-file set per port is persisted.** Every run manifest records the
  stat identity of each implementation file the model's prediction depends on,
  and resume compares them (`manifest.py`'s `_input_dependencies`). Adding or
  dropping one changes which finished runs can be reused, so the declarations
  are frozen in `tests/data/port_manifest_sources.json` and a change there has
  to be a deliberate one; what they resolve to is held to a property instead --
  a port's featurizer, its data assets, the common-job writer and every
  `foldjax.models` module its code imports.
* **Reading the table imports nothing.** `manifest.py` keeps the AlphaFold 3
  runtime import local so ordinary manifest use does not pay for it; a table
  that imported the ports to describe them would undo that for all six at once.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from foldjax import assets, manifest, portspec, registry
from foldjax import input as input_module

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "port_manifest_sources.json").read_text()
)
CANONICAL = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")


def test_the_table_covers_every_canonical_port_and_nothing_else() -> None:
    """One entry per model the registry can build, keyed by canonical id."""
    assert tuple(sorted(portspec.PORTS)) == CANONICAL
    assert registry.available_models() == CANONICAL
    assert set(input_module._TARGETS) == set(portspec.PORTS)
    for model, spec in portspec.PORTS.items():
        assert spec.model == model
        assert spec.input_dialect
        assert spec.asset_profiles[0] == portspec.RELEASED_PROFILE
        # Every non-default profile must name its builder, or `assets_for`
        # would raise a KeyError where it used to fall through to ESMFold2's
        # download filter.
        assert set(spec.asset_profiles[1:]) == set(spec.asset_profile_providers)


def test_the_table_never_normalises_an_alias() -> None:
    """Aliases resolve in exactly one place: `registry.normalize_model_name`."""
    assert portspec.PORTS.get("af3") is None
    assert portspec.PORTS.get("boltz") is None
    assert registry.normalize_model_name("af3") == "alphafold3"
    assert registry.normalize_model_name("  BOLTZ-JAX ") == "boltz2"
    for model, spec in portspec.PORTS.items():
        assert portspec.ALIASES[model] == model
        for alias in spec.aliases:
            assert registry.normalize_model_name(alias) == model


def test_an_unknown_model_is_refused_exactly_as_before() -> None:
    """Three layers reject a name the table has no entry for, each its own way."""
    with pytest.raises(ValueError) as registry_error:
        registry.normalize_model_name("esmfold3")
    assert str(registry_error.value) == (
        "unknown model 'esmfold3'; choose one of alphafold3, boltz2, esmfold2, "
        "opendde, openfold3, protenix"
    )

    with pytest.raises(ValueError, match="unknown model 'esmfold3'"):
        assets.available_profiles("esmfold3")

    assert input_module.common_schema_features("esmfold3") == ()

    # The neutral manifest layer does not reject: a backend override may run a
    # model the table does not describe, and such a run tracks no source files
    # rather than becoming unverifiable.
    assert manifest.implementation_dependency_paths("esmfold3") == ()


def test_only_declared_providers_resolve() -> None:
    """The indirection is a lookup, not a way to import an arbitrary name."""
    for reference in portspec.PROVIDERS:
        assert portspec.provider(reference) is not None
    with pytest.raises(ValueError, match="undeclared port table provider"):
        portspec.provider("os:system")


@pytest.mark.parametrize("model", CANONICAL)
def test_manifest_source_specs_match_the_frozen_fixture(model: str) -> None:
    """The declarations, in order. A diff here is a resume-compatibility change."""
    tokens = [source.token for source in portspec.PORTS[model].manifest_sources]
    assert tokens == FIXTURE["spec_tokens"][model]


PACKAGE = Path(manifest.__file__).parent

#: Files in a port's tree that are not data a prediction reads.
_NOT_DATA = frozenset({"LICENSE", "NOTICE"})


def _tracked(model: str) -> set[Path]:
    return set(manifest.implementation_dependency_paths(model))


def _foldjax_imports(path: Path) -> set[Path]:
    """Every `foldjax` module file ``path`` imports, at any scope."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "foldjax":
                found.add(node.module)
                found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] == "foldjax"
            )
    files: set[Path] = set()
    for name in found:
        stem = PACKAGE.parent / name.replace(".", "/")
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if candidate.is_file():
                files.add(candidate)
                break
    return files


def _model_closure(model: str) -> set[Path]:
    """`foldjax.models` files reachable by import from a port's own code."""
    pending = [
        *(PACKAGE / "models" / model).rglob("*.py"),
        PACKAGE / "backends" / f"{model}.py",
    ]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pending.extend(
            imported
            for imported in _foldjax_imports(path)
            if imported.is_relative_to(PACKAGE / "models")
        )
    return {path for path in seen if path.is_relative_to(PACKAGE / "models")}


@pytest.mark.parametrize("model", CANONICAL)
def test_tracked_files_exist_once_each(model: str) -> None:
    tracked = manifest.implementation_dependency_paths(model)
    assert len(set(tracked)) == len(tracked)
    for path in tracked:
        assert path.is_file(), path


@pytest.mark.parametrize("model", CANONICAL)
def test_each_port_binds_its_featurizer_data_and_input_writer(model: str) -> None:
    """The featurizer, the assets it loads, and the common-job writer.

    Re-derived from the tree rather than listed, so a featurizer module or
    table added later is held to the same contract. Before 2026-10-06 none of
    these were bound: an edit to Protenix's `featurize_json.py` or its CCD
    tables let a stale prediction satisfy `--resume`.
    """
    tracked = _tracked(model)
    assert PACKAGE / "input.py" in tracked
    assert PACKAGE / "backends" / f"{model}.py" in tracked
    featurizer = PACKAGE / "models" / model / "data"
    if model == "alphafold3":
        # Featurized by the vendored upstream tree, which `manifest.py` binds
        # on the managed route only; FoldJAX owns no featurizer for it.
        assert not featurizer.exists()
        return
    data = [
        path
        for path in featurizer.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name not in _NOT_DATA
    ]
    assert any(path.suffix == ".py" for path in data)
    missing = sorted(
        str(path.relative_to(PACKAGE)) for path in data if path not in tracked
    )
    assert missing == []


@pytest.mark.parametrize("model", CANONICAL)
def test_each_port_binds_every_model_module_its_code_imports(model: str) -> None:
    """A module of another port, or a shared helper, changes predictions too.

    OpenDDE featurizes through Protenix's `featurize_protein_json`; ESMFold2
    and OpenFold3 run Boltz-2's native norm. An import that reaches a module
    the table does not bind has to be a deliberate decision, made here.
    """
    tracked = _tracked(model)
    missing = sorted(
        str(path.relative_to(PACKAGE))
        for path in _model_closure(model)
        if path not in tracked
        and not path.is_relative_to(PACKAGE / "models" / "alphafold3")
    )
    assert missing == []


def test_protenix_binds_its_featurizer_and_ccd_tables() -> None:
    """The three files the 2026-10-06 regression check found unbound."""
    tracked = _tracked("protenix")
    for relative in (
        "models/protenix/data/featurize_json.py",
        "models/protenix/data/ccd_nucleotides.npz",
        "input.py",
    ):
        assert PACKAGE / relative in tracked
        assert PACKAGE / relative in _tracked("opendde")


@pytest.mark.parametrize(
    "model, source",
    [
        ("protenix", "models/protenix/data/featurize_json.py"),
        ("protenix", "models/protenix/data/ccd_nucleotides.npz"),
        ("protenix", "input.py"),
        ("opendde", "models/protenix/data/featurize_json.py"),
        ("opendde", "models/opendde/data/opendde_std_reference.npz"),
        ("boltz2", "models/boltz2/data/featurize.py"),
        ("boltz2", "template_search.py"),
        # ESMFold2 is absent: the stand-in checkpoint here has no
        # `config.json` beside it, so that run is never resumable at all.
        ("openfold3", "models/openfold3/data/featurize.py"),
        (
            "openfold3",
            "models/openfold3/_upstream/openfold3/core/data/io/structure/cif.py",
        ),
    ],
)
def test_a_changed_featurizer_file_invalidates_resume(
    tmp_path: Path, monkeypatch, model: str, source: str
) -> None:
    """End to end through `predict_batch`: the bound file is what resume checks."""
    import dataclasses

    import foldjax
    from tests.test_resume_manifest import _backends, _request

    target = (PACKAGE / source).resolve()
    calls: list[tuple[str, str, int]] = []
    request = _request(
        tmp_path, model=model, options={}, padding=None, representations=None
    )
    with _backends(calls):
        foldjax.predict(request)
        unchanged = foldjax.predict_batch(dataclasses.replace(request, resume=True))
        assert unchanged.skipped == (request.output_dir,)
        original = manifest.path_stat_identity

        def edited(path):
            identity = original(path)
            if Path(path) == target:
                identity = {**identity, "stat_signature": "edited"}
            return identity

        monkeypatch.setattr(manifest, "path_stat_identity", edited)
        resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))
    assert len(calls) == 2
    assert resumed.skipped == ()


def test_asset_profiles_come_from_the_table() -> None:
    """`available_profiles` reports the table, default first."""
    for model, spec in portspec.PORTS.items():
        assert assets.available_profiles(model) == spec.asset_profiles
    assert assets.available_profiles("esmfold2") == (
        portspec.RELEASED_PROFILE,
        portspec.STRUCTURE_ONLY_PROFILE,
    )
    assert assets.available_profiles("of3") == (portspec.RELEASED_PROFILE,)


def test_staging_specs_name_a_directory_rather_than_a_relative_path() -> None:
    """Boltz-2 stages beside its weight directory; these paths get unlinked."""
    root = Path("/weights/boltz2")
    (native, mols) = portspec.PORTS["boltz2"].asset_staging
    assert native.directory(root) == Path("/weights")
    assert mols.directory(root) == root
    esmc = portspec.PORTS["esmfold2"].asset_staging[1]
    assert esmc.directory(root) == root / "esmc"


#: Storage model -> the FoldJAX-owned temporary trees its lock may remove.
#: Every other decoy planted below has to survive, which is the part the old
#: elif-chain encoded implicitly: one model's staging name is another's data.
_STAGING_REMOVED = {
    "boltz2": ("../.foldjax-boltz-native-1", ".foldjax-mols-1"),
    "opendde": (".foldjax-opendde-native-1",),
    "protenix": (".foldjax-protenix-native-1",),
    "esmfold2": (".foldjax-stage-1", "esmc/.foldjax-stage-1"),
    "openfold3": (".foldjax-stage-1",),
    "alphafold3": (),
    "protenix-v2": (".foldjax-protenix-variant-1",),
    # An internal OpenDDE root has never had a cleanup rule, and inventing one
    # here would delete a tree no conversion in this store wrote.
    "opendde-abag": (),
}
_STAGING_DECOYS = (
    "../.foldjax-boltz-native-1",
    "../.foldjax-other-native-1",
    ".foldjax-boltz-native-1",
    ".foldjax-mols-1",
    ".foldjax-opendde-native-1",
    ".foldjax-protenix-native-1",
    ".foldjax-protenix-variant-1",
    ".foldjax-stage-1",
    "esmc/.foldjax-stage-1",
    "model.safetensors",
)


@pytest.mark.parametrize("model", sorted(_STAGING_REMOVED))
def test_staging_cleanup_removes_only_this_model_s_own_trees(
    model: str, tmp_path: Path, monkeypatch
) -> None:
    """It unlinks and rmtree's, so what it matches is a safety boundary."""
    from foldjax import paths

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_repository_store", lambda: None)
    root = paths.weights_dir(model)
    for relative in _STAGING_DECOYS:
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "partial.bin").write_bytes(b"x")

    assets._cleanup_abandoned_staging(model)

    removed = {
        relative
        for relative in _STAGING_DECOYS
        if not (root / relative).exists()
    }
    assert removed == set(_STAGING_REMOVED[model])


def test_importing_the_table_imports_no_runtime() -> None:
    """A cold process: reading a port's facts must cost no JAX, torch or port.

    The `foldjax` package is registered as a bare stub whose `__path__` points
    at the source tree, so `foldjax/__init__.py` never runs. Without that the
    test would be vacuous: the package root imports `api`, which imports
    `manifest`, `input` and `registry` already, and an allow-list for what the
    root drags in would hide a leak these modules introduced themselves.

    `msa_search` and `result_validation` are named separately because they
    were lifted out of `input.py` and `api.py`. `input` imports `msa_search`
    at module scope today, so naming it keeps the guard if that import ever
    goes lazy; `result_validation` is reached only through `api`, which this
    test cannot import at all -- `api` imports a backend base class -- so a
    runtime import it acquired would otherwise go unnoticed.

    The subprocess is the point as well -- an in-process check cannot see a
    top-level import in a module some earlier test already loaded.
    """
    script = r"""
import sys
from pathlib import Path
from types import ModuleType

stub = ModuleType("foldjax")
stub.__path__ = [str(Path("src/foldjax").resolve())]
sys.modules["foldjax"] = stub

import foldjax.portspec
import foldjax.manifest
import foldjax.input
import foldjax.msa_search
import foldjax.registry
import foldjax.result_validation

leaked = sorted(
    name
    for name in sys.modules
    if name.partition(".")[0] in {"jax", "jaxlib", "torch"}
    or name.startswith("foldjax.backends")
    or name.startswith("foldjax.models.")
)
assert not leaked, leaked
assert "foldjax.portspec" in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_the_table_alone_imports_only_the_standard_library() -> None:
    """Loaded outside the package, `portspec` pulls in no `foldjax` at all."""
    script = r"""
import importlib.util
import sys
from pathlib import Path

path = Path("src/foldjax/portspec.py").resolve()
spec = importlib.util.spec_from_file_location("_portspec_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules["_portspec_probe"] = module
spec.loader.exec_module(module)

assert len(module.PORTS) == 6, sorted(module.PORTS)
leaked = sorted(name for name in sys.modules if name.partition(".")[0] == "foldjax")
assert not leaked, leaked
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
