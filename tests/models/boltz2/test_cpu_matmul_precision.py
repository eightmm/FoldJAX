"""Boltz-2's released `high` matmul policy is TF32, which only a GPU has.

Off a GPU the cuEquivariance reference path hands `TF32_TF32_F32` to
`dot_general` and the CPU refuses it, so an omitted policy failed every CPU
affinity stage. Omitted, it now resolves to `highest` off a GPU; spelled
`high`, it is refused while planning.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models import _pallas_pair
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, **options) -> PredictionRequest:
    job = tmp_path / "job.yaml"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=job,
        input_format="boltz",
        weights=weights,
        output_dir=tmp_path / "out",
        seed=1,
        cache_dir=tmp_path / "cache",
        options={"mols": mols, **options},
    )


def _scoped_precision(tmp_path: Path, monkeypatch, **options) -> str | None:
    """The policy the native call ran under."""
    from foldjax.execution import resolved_matmul_precision

    seen: dict[str, object] = {}

    def native_predict(**kwargs):
        seen["precision"] = resolved_matmul_precision("unset")
        return {
            "coords": np.zeros((1, 2, 3)),
            "plddt": np.ones((1, 2)),
            "out_path": None,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    Boltz2Backend().predict(_request(tmp_path, **options))
    return seen["precision"]


@pytest.mark.parametrize("gpu", [True, False], ids=["gpu", "cpu"])
def test_an_omitted_policy_is_the_one_the_platform_has(
    tmp_path, monkeypatch, gpu
) -> None:
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: gpu)
    backend = Boltz2Backend()
    omitted = backend.cache_profile(_request(tmp_path))
    assert omitted["matmul_precision"] == ("high" if gpu else "highest")
    # The realised value spelled out names the omitted run's namespace.
    spelled = backend.cache_profile(
        _request(tmp_path, matmul_precision="high" if gpu else "highest")
    )
    assert spelled == omitted
    # On a GPU the scope is left to the port's own pin, as it always was; off
    # one the run is scoped `highest`.
    assert _scoped_precision(tmp_path, monkeypatch) == ("unset" if gpu else "highest")


def test_a_spelled_high_is_refused_off_a_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: False)
    request = _request(tmp_path, matmul_precision="high")
    with pytest.raises(ValueError, match="TF32.*matmul_precision=highest"):
        Boltz2Backend().validate_request(request)
    with pytest.raises(ValueError, match="TF32"):
        Boltz2Backend().cache_profile(request)
    with pytest.raises(ValueError, match="TF32"):
        _scoped_precision(tmp_path, monkeypatch, matmul_precision="high")
    # `highest` is accepted everywhere.
    Boltz2Backend().validate_request(_request(tmp_path, matmul_precision="highest"))


def test_a_spelled_high_runs_on_a_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_pallas_pair, "gpu_process", lambda: True)
    request = _request(tmp_path, matmul_precision="high")
    Boltz2Backend().validate_request(request)
    assert _scoped_precision(tmp_path, monkeypatch, matmul_precision="high") == "high"
