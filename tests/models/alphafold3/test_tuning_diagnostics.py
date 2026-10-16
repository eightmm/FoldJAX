"""Bounded, value-free AF3 call records for comparing warm and prediction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends import _tokamax_autotune as persistent


@pytest.fixture(autouse=True)
def _record():
    persistent.start_record(strategy="autotune", persistent_installed=True)
    yield
    persistent.clear_record()


def _note(batch, *, identity=None, namespace=Path("/private/cache"), key=None):
    if key is None:
        key = np.zeros(2, np.uint32)
    if identity is None:
        identity = {
            "config": [("max_msa_depth", 1024), ("attention_backend", "triton")],
            "buckets": [256],
            "padding": True,
            "source_sha256": "a" * 64,
            "versions": {"jax": "test"},
            "device": {"device_kind": "test-gpu"},
        }
    descriptor = persistent._call_descriptor(key, batch)
    call_sha256 = persistent._sha256(persistent._canonical_bytes(descriptor))
    signature = persistent._sha256(
        persistent._canonical_bytes({"identity": identity, "call": descriptor})
    )
    persistent._note_call(
        batch=batch,
        descriptor=descriptor,
        identity=identity,
        cache_dir=namespace,
        call_sha256=call_sha256,
        signature=signature,
    )
    persistent._note_source("store")


def test_values_and_rng_contents_do_not_split_the_call_but_shapes_and_dtypes_do():
    batch = {"msa": np.zeros((4, 8), np.int8)}
    _note(batch)
    _note({"msa": np.ones((4, 8), np.int8)}, key=np.ones(2, np.uint32))
    _note({"msa": np.ones((5, 8), np.int8)})
    _note({"msa": np.ones((4, 8), np.int32)})

    record = persistent.recorded()
    first, shape, dtype = record["calls"]
    assert first["count"] == 2
    assert first["sources"] == {"store": 2}
    assert record["sources"] == {"store": 4}
    assert len({call["model_input_abi_sha256"] for call in record["calls"]}) == 3
    assert len({call["call_sha256"] for call in record["calls"]}) == 3
    assert first["store_identity_sha256"] == shape["store_identity_sha256"]
    assert first["store_identity_sha256"] == dtype["store_identity_sha256"]
    assert first["compile_config"] == {
        "max_msa_depth": 1024,
        "attention_backend": "triton",
    }
    assert first["padding"] is True
    assert first["input_abi"]["leaves"] == [
        {"path": "[0]", "shape": [2], "dtype": "uint32", "weak_type": False},
        {"path": "[1]['msa']", "shape": [4, 8], "dtype": "int8", "weak_type": False},
    ]
    assert "_active_call" not in record
    record["calls"][0]["count"] = 900
    assert persistent.recorded()["calls"][0]["count"] == 2


def test_namespace_and_runtime_axes_are_separately_identifiable():
    batch = {"msa": np.zeros((4, 8), np.int8)}
    _note(batch, namespace=Path("/private/first"))
    _note(batch, namespace=Path("/private/second"))
    _note(
        batch,
        namespace=Path("/private/second"),
        identity={"config": [("max_msa_depth", 512)], "versions": {"jax": "changed"}},
    )
    first, moved, changed = persistent.recorded()["calls"]
    assert first["call_sha256"] == moved["call_sha256"] == changed["call_sha256"]
    assert first["signature_sha256"] == moved["signature_sha256"]
    assert first["namespace_sha256"] != moved["namespace_sha256"]
    assert moved["store_identity_sha256"] != changed["store_identity_sha256"]
    assert moved["runtime_sha256"] != changed["runtime_sha256"]
    assert moved["compile_config"] != changed["compile_config"]


def test_record_never_emits_array_values_rng_values_or_nonstandard_paths():
    private_path = "/private/sequence-file.a3m"
    sentinel = 918723641
    _note(
        {private_path: np.full((4, 8), sentinel, np.int32)},
        key=np.full(2, sentinel, np.uint32),
        identity={
            "config": [
                ("attention_backend", private_path),
                ("source", private_path),
            ],
            "source_sha256": private_path,
        },
        namespace=Path(private_path),
    )
    record = persistent.recorded()
    text = json.dumps(record)
    assert private_path not in text
    assert str(sentinel) not in text
    call, = record["calls"]
    assert call["compile_config"] == {"attention_backend": "<redacted>"}
    assert call["source_sha256"] == "<redacted>"
    assert call["input_abi"]["tree"] == "<nonstandard model input tree>"
    assert [leaf["path"] for leaf in call["input_abi"]["leaves"]] == [
        "leaf[0]", "leaf[1]"
    ]


def test_call_and_leaf_records_are_bounded_and_repeats_keep_their_count():
    for length in range(10):
        _note({"msa": np.zeros((length, 8), np.int8)})
    _note({"msa": np.zeros((0, 8), np.int8)})
    record = persistent.recorded()
    assert len(record["calls"]) == 8
    assert record["calls_truncated"] == 2
    assert record["calls"][0]["count"] == 2

    persistent.start_record(strategy="autotune", persistent_installed=True)
    _note({f"feature_{index}": np.zeros((), np.int32) for index in range(140)})
    call, = persistent.recorded()["calls"]
    assert len(call["input_abi"]["leaves"]) == 128
    assert call["input_abi"]["leaf_count"] == 141
    assert len(call["input_abi"]["tree"]) <= 8192


def test_host_bucket_selection_does_not_split_a_concrete_model_call(monkeypatch):
    monkeypatch.setattr(
        persistent, "_tokamax_api", lambda: SimpleNamespace(__version__="test")
    )
    monkeypatch.setattr(persistent, "_platform_version", lambda _platform: "test")
    device = SimpleNamespace(
        platform="gpu", device_kind="test-gpu", compute_capability="9.0", core_count=80
    )

    def identity(*, buckets, padding=True, depth=1024):
        return persistent._runtime_identity(
            device=device,
            source_snapshot=(("source", "test", "test"),),
            config_identity=(("max_msa_depth", depth),),
            buckets=buckets,
            padding=padding,
        )

    warm = identity(buckets=(256,))
    predict = identity(buckets=(256, 512, 768))
    assert warm == predict
    assert warm != identity(buckets=(256,), padding=False)
    assert warm != identity(buckets=(256,), depth=512)
    device.device_kind = "other-gpu"
    assert warm != identity(buckets=(256,))
    device.device_kind = "test-gpu"
    descriptor = persistent._call_descriptor(
        np.zeros(2, np.uint32), {"msa": np.zeros((16384, 256), np.int8)}
    )
    assert persistent._canonical_bytes({"identity": warm, "call": descriptor}) == (
        persistent._canonical_bytes({"identity": predict, "call": descriptor})
    )
