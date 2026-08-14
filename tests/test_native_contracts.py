"""Checks against the real backend packages rather than test doubles.

The unit tests use fakes so they stay fast and weight-free. These tests import
the actual vendored ports and assert the contracts FoldJAX depends on, so a
native rename or signature change fails here instead of at inference time.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

from foldjax.input import materialize_native_input
from foldjax.registry import capabilities
from foldjax.schema import PredictionRequest


def _materialize(job: dict, model: str, tmp_path: Path) -> Path:
    source = tmp_path / "common.json"
    source.write_text(json.dumps(job))
    return materialize_native_input(
        source, capabilities(model), tmp_path / model, seed=1
    )


def test_materialized_boltz_job_has_a_suffix_boltz_accepts(tmp_path: Path) -> None:
    """Boltz-2 rejects any other suffix with RuntimeError before parsing."""
    # Read the parser's dispatch table as source: importing it needs Boltz's
    # torch/lightning featurization extra, which the base environment omits.
    source = (Path(_boltz_jax_root()) / "data" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    assert 'path.suffix.lower() in (".yml", ".yaml")' in source

    path = _materialize(
        {"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]},
        "boltz2",
        tmp_path,
    )
    assert path.suffix.lower() in (".yml", ".yaml")
    assert yaml.safe_load(path.read_text())["version"] == 1


def _boltz_jax_root() -> Path:
    import foldjax.models.boltz2

    return Path(foldjax.models.boltz2.__file__).parent


def test_protenix_cli_accepts_the_flags_the_adapter_emits() -> None:
    """Every Protenix option the adapter forwards must exist on its CLI."""
    from foldjax.backends.protenix import _CLI_OPTIONS
    from foldjax.models.protenix.cli import predict as native

    source = inspect.getsource(native)
    for option in sorted(_CLI_OPTIONS | {"compile_cache", "output_format", "seed"}):
        assert f'"--{option.replace("_", "-")}"' in source, option


def test_opendde_cli_accepts_the_flags_the_adapter_emits() -> None:
    from foldjax.backends.opendde import _CLI_OPTIONS
    from foldjax.models.opendde.cli import predict as native

    source = inspect.getsource(native)
    for option in sorted(_CLI_OPTIONS | {"compile_cache", "include_raw", "seed"}):
        assert f'"--{option.replace("_", "-")}"' in source, option


def test_materialized_opendde_job_parses_with_native_loader(tmp_path: Path) -> None:
    from foldjax.models.opendde.data.featurize_json import load_jobs

    path = _materialize(
        {"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]},
        "opendde",
        tmp_path,
    )
    jobs = load_jobs(path)
    assert len(jobs) == 1
    assert jobs[0]["modelSeeds"] == [1]
    assert jobs[0]["sequences"][0]["proteinChain"]["sequence"] == "ACD"


def test_boltz_predict_accepts_the_keywords_the_adapter_passes() -> None:
    # Reading the signature from source keeps this contract check off JAX's import
    # cost; prediction itself and its NumPy featurizer are part of the base install.
    source = (_boltz_jax_root() / "api.py").read_text(encoding="utf-8")
    signature = source.split("def predict(", 1)[1].split(") -> dict", 1)[0]
    for keyword in ("input", "weights", "mols", "out_dir", "seed", "compile_cache"):
        assert f"{keyword}:" in signature, keyword


def test_boltz_backend_reports_a_missing_base_dependency(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.backends import boltz2

    class Lazy:
        @property
        def predict(self):
            raise ModuleNotFoundError(name="required_package")

    monkeypatch.setattr(boltz2, "import_module", lambda name: Lazy())
    request = PredictionRequest(
        model="boltz2",
        input=_write_yaml(tmp_path),
        weights=tmp_path,
        output_dir=tmp_path / "out",
        options={"mols": tmp_path},
    )
    with pytest.raises(ModuleNotFoundError, match="base installation") as error:
        boltz2.Boltz2Backend().predict(request)
    assert "boltz-preprocess" not in str(error.value)


def _write_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "job.yaml"
    path.write_text("version: 1\n")
    return path


def test_native_weight_files_from_before_vendoring_still_load() -> None:
    """Native weight files pickle their parameter NamedTuples by module name.

    Anything exported while Protenix/OpenDDE were standalone packages records
    ``protenix_jax.*`` / ``opendde_jax.*``, so those names must keep resolving
    to the vendored classes without widening what the unpickler accepts.
    """
    import io
    import pickle

    from foldjax.models.opendde.models.model import OpenDDEInferenceParams
    from foldjax.models.protenix.bridge.weights_io import _NativeWeightsUnpickler
    from foldjax.models.protenix.models.diffusion.atom import (
        AtomAttentionEncoderParams,
    )

    unpickler = _NativeWeightsUnpickler(io.BytesIO(b""))
    resolved = {
        # pre-vendoring module name -> class it must still resolve to
        ("protenix_jax.models.diffusion.atom", "AtomAttentionEncoderParams"): (
            AtomAttentionEncoderParams
        ),
        # pre-vendoring name that also needs the legacy in-repo reshuffle
        ("protenix_jax.models.atom", "AtomAttentionEncoderParams"): (
            AtomAttentionEncoderParams
        ),
        ("opendde_jax.models.model", "OpenDDEInferenceParams"): OpenDDEInferenceParams,
        ("foldjax.models.opendde.models.model", "OpenDDEInferenceParams"): (
            OpenDDEInferenceParams
        ),
    }
    for (module, name), expected in resolved.items():
        assert unpickler.find_class(module, name) is expected, module

    for module, name in (
        ("os", "system"),
        ("builtins", "eval"),
        ("protenix_jax.cli.predict", "main"),
        ("foldjax.models.protenix.cli.predict", "main"),
    ):
        with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
            unpickler.find_class(module, name)


def test_native_weights_round_trip_through_the_vendored_writer(tmp_path: Path) -> None:
    import numpy as np

    from foldjax.models.protenix.bridge.weights_io import (
        load_native_weights,
        save_native_weights,
    )
    from foldjax.models.protenix.models.diffusion.atom import (
        AtomAttentionEncoderParams,
    )

    leaves = [np.zeros((2, 2), "float32")] * len(AtomAttentionEncoderParams._fields)
    path = tmp_path / "weights.npz"
    save_native_weights(path, {"encoder": AtomAttentionEncoderParams(*leaves)})

    loaded = load_native_weights(path)["encoder"]
    assert isinstance(loaded, AtomAttentionEncoderParams)
    assert loaded[0].shape == (2, 2)


def test_opendde_adapter_restores_every_environment_variable_the_cli_exports(
    tmp_path: Path, monkeypatch
) -> None:
    """FoldJAX runs OpenDDE's CLI in-process, so its exports must not persist."""
    import os
    import re

    from foldjax.backends import opendde
    from foldjax.models.opendde.cli import predict as native

    source = inspect.getsource(native)
    pattern = r'os\.environ(?:\.setdefault)?[\[(]\s*"([A-Z_]+)"'
    exported = set(re.findall(pattern, source))
    assert exported, "no environment exports found in the native CLI"
    assert exported <= set(opendde._EXPORTED_ENVIRONMENT), sorted(
        exported - set(opendde._EXPORTED_ENVIRONMENT)
    )

    monkeypatch.delenv("PROTENIX_CCD_COMPONENTS_FILE", raising=False)
    monkeypatch.setenv("PROTENIX_KALIGN_BINARY", "/pre-existing/kalign")

    def fake_main(argv):
        os.environ["PROTENIX_CCD_COMPONENTS_FILE"] = "/leaked/components.cif"
        os.environ["PROTENIX_KALIGN_BINARY"] = "/leaked/kalign"
        # The real CLI returns the paths it wrote; this run writes none.
        return []

    monkeypatch.setattr(
        opendde, "import_module", lambda name: type("M", (), {"main": fake_main})
    )
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "w.npz"
    weights.write_bytes(b"")
    request = PredictionRequest(
        model="opendde",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=1,
    )
    opendde.OpenDDEBackend().predict(request)

    assert "PROTENIX_CCD_COMPONENTS_FILE" not in os.environ
    assert os.environ["PROTENIX_KALIGN_BINARY"] == "/pre-existing/kalign"
