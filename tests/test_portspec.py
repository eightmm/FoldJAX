"""The per-port table, and the two properties that make it safe to have one.

The table replaced forty `request.model == "..."` branches in `manifest.py`,
`assets.py` and `input.py`. Two things about it are load-bearing rather than
stylistic, and both are asserted here:

* **The tracked-file set per port is persisted.** Every run manifest records the
  stat identity of each implementation file the model's prediction depends on,
  and resume compares them (`manifest.py`'s `_input_dependencies`). Adding or
  dropping one changes which finished runs can be reused, so the lists are
  frozen in `tests/data/port_manifest_sources.json` and a change there has to
  be a deliberate one.
* **Reading the table imports nothing.** `manifest.py` keeps the AlphaFold 3
  runtime import local so ordinary manifest use does not pay for it; a table
  that imported the ports to describe them would undo that for all six at once.
"""

from __future__ import annotations

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


@pytest.mark.parametrize(
    "model", [name for name in CANONICAL if name not in FIXTURE["globbed_ports"]]
)
def test_tracked_files_match_the_frozen_fixture(model: str) -> None:
    """The resolved per-port list, as it was before the table existed."""
    package = Path(manifest.__file__).parent
    tracked = manifest.implementation_dependency_paths(model)
    relative = sorted(str(path.relative_to(package)) for path in tracked)
    assert relative == FIXTURE["tracked_files"][model]
    assert len(set(tracked)) == len(tracked)
    for path in tracked:
        assert path.is_file(), path


def test_boltz2_tracks_its_whole_model_tree_by_glob() -> None:
    """Boltz-2's numerics live across forty files, so the policy is the tree.

    Freezing the listing would make adding a module to the port a test failure
    while leaving the actual contract -- every ``.py`` under `models/`, sorted
    -- unasserted. This re-derives it instead.
    """
    package = Path(manifest.__file__).parent
    tracked = manifest.implementation_dependency_paths("boltz2")
    expected = [
        package / "models/boltz2/api.py",
        package / "models/boltz2/compile_policy.py",
        *sorted((package / "models/boltz2/models").rglob("*.py")),
        package / "models/_cueq.py",
    ]
    assert list(tracked) == expected
    assert len(expected) > 40


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

    `msa_search` is named separately because it was lifted out of `input.py`,
    and a module reached only through a search that most jobs never run is
    exactly where such an import goes unnoticed.

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
