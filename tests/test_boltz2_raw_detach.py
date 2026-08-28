from __future__ import annotations

import dataclasses
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import jax
import ml_dtypes
import numpy as np

from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models import _representations
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, *, seed: int = 0) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("{}\n")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / f"out-{seed}",
        seed=seed,
        options={"mols": mols, "write_fmt": None},
    )


def _jax_array_leaves(tree: object) -> list[jax.Array]:
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    ]


def test_backend_detaches_full_raw_tree_without_changing_array_bits(
    tmp_path: Path, monkeypatch
) -> None:
    coords = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    plddt = np.asarray([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32)
    float_probe = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32
    )
    bf16_probe = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan], dtype=ml_dtypes.bfloat16
    )
    device_refs: list[weakref.ReferenceType[jax.Array]] = []

    def native_predict(**kwargs):
        del kwargs
        device_float = jax.device_put(float_probe.copy())
        device_bf16 = jax.device_put(bf16_probe.copy())
        affinity = jax.device_put(np.asarray([0.125, -0.0], dtype=np.float32))
        device_refs.extend(
            (weakref.ref(device_float), weakref.ref(device_bf16), weakref.ref(affinity))
        )
        nested = {
            "float_probe": device_float,
            "bf16_probe": device_bf16,
            "affinity_pred_value": affinity,
        }
        return {
            "coords": coords,
            "plddt": plddt,
            "raw": nested,
            "affinity_pred_value": affinity,
            "record_id": "job",
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )

    request = dataclasses.replace(_request(tmp_path), num_samples=2)
    result = Boltz2Backend().predict(request)

    assert result.raw["coords"] is coords
    assert result.raw["plddt"] is plddt
    assert result.raw["affinity_pred_value"] is result.raw["raw"][
        "affinity_pred_value"
    ]
    assert not _jax_array_leaves(result.raw)
    assert isinstance(result.raw["raw"]["float_probe"], np.ndarray)
    assert isinstance(result.raw["raw"]["bf16_probe"], np.ndarray)
    assert result.raw["raw"]["float_probe"].shape == float_probe.shape
    assert result.raw["raw"]["bf16_probe"].dtype == bf16_probe.dtype
    np.testing.assert_array_equal(
        result.raw["raw"]["float_probe"].view(np.uint32),
        float_probe.view(np.uint32),
    )
    np.testing.assert_array_equal(
        result.raw["raw"]["bf16_probe"].view(np.uint16),
        bf16_probe.view(np.uint16),
    )

    gc.collect()
    assert all(reference() is None for reference in device_refs)


def test_session_results_do_not_retain_per_seed_device_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    device_refs: list[weakref.ReferenceType[jax.Array]] = []

    def native_predict(**kwargs):
        seed = kwargs["seed"]
        device_value = jax.device_put(np.asarray([seed, -0.0], dtype=np.float32))
        device_refs.append(weakref.ref(device_value))
        return {
            "coords": np.full((2, 3), seed, dtype=np.float32),
            "plddt": np.full((2,), seed / 10, dtype=np.float32),
            "raw": {"ptm": device_value},
            "record_id": "job",
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    requests = (_request(tmp_path, seed=0), _request(tmp_path, seed=1))
    backend = Boltz2Backend()

    with backend.session(requests):
        results = tuple(backend.predict(request) for request in requests)

    assert all(not _jax_array_leaves(result.raw) for result in results)
    np.testing.assert_array_equal(results[0].raw["raw"]["ptm"], [0.0, -0.0])
    np.testing.assert_array_equal(results[1].raw["raw"]["ptm"], [1.0, -0.0])
    gc.collect()
    assert all(reference() is None for reference in device_refs)


def test_trunk_result_keeps_archive_lazy_and_drops_only_archived_representation(
    tmp_path: Path, monkeypatch
) -> None:
    request = dataclasses.replace(
        _request(tmp_path),
        representations=("single",),
        stop_after="trunk",
    )
    single = np.arange(8, dtype=np.float32).reshape(2, 4)
    pair = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    device_refs: list[weakref.ReferenceType[jax.Array]] = []
    padding = {
        "primary": {
            "actual": {"tokens": 2, "atoms": 3, "msa": 1},
            "target": {"tokens": 8, "atoms": 32, "msa": 1},
        }
    }

    def native_predict(**kwargs):
        assert kwargs["representations"] == ("single",)
        device_single = jax.device_put(single[None])
        device_pair = jax.device_put(pair[None])
        diagnostic = jax.device_put(np.asarray([3, 5], dtype=np.int16))
        device_refs.extend(
            (
                weakref.ref(device_single),
                weakref.ref(device_pair),
                weakref.ref(diagnostic),
            )
        )
        archive = _representations.save(
            kwargs["representations_dir"],
            {"single": device_single},
            _representations.specs_for("boltz2"),
            model="boltz2",
        )
        return {
            "record_id": "job",
            "raw": {
                "single": device_single,
                # Not requested or archived: retain it as ordinary raw data.
                "pair": device_pair,
                "diagnostic": diagnostic,
            },
            "representations": archive,
            "padding": padding,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )

    result = Boltz2Backend().predict(request)

    assert result.samples == ()
    assert result.raw["record_id"] == "job"
    assert result.raw["padding"] == padding
    assert result.raw["representations"] == request.output_dir / "representations.npz"
    assert "single" not in result.raw["raw"]
    np.testing.assert_array_equal(result.raw["raw"]["pair"], pair[None])
    np.testing.assert_array_equal(result.raw["raw"]["diagnostic"], [3, 5])
    assert not _jax_array_leaves(result.raw)

    assert result.representations is not None
    assert result.representations.names() == ("single",)
    np.testing.assert_array_equal(result.representations["single"], single)

    gc.collect()
    assert all(reference() is None for reference in device_refs)
