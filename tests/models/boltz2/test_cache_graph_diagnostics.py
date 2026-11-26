"""Production Boltz graph identity and realized chunks without model weights."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import execution, memory_policy
from foldjax.api import resolve_cache_dir
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.executable_cache_verify import (
    UnsupportedCacheVerificationError,
    lowered_cache_key,
)
from foldjax.models.boltz2 import api as native_api
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    resolve_long_sequence_chunks,
)
from tests.test_boltz2_session import _features, _request


@pytest.fixture(autouse=True)
def isolated_record():
    execution.clear_chunking_record()
    memory_policy.clear_record()
    yield
    execution.clear_chunking_record()
    memory_policy.clear_record()


def _install_runtime(monkeypatch, tmp_path, *, n_token=512, affinity=False):
    features = _features(affinity=affinity)
    features["token_pad_mask"] = np.ones((1, n_token), np.float32)
    features["mol_type"] = np.zeros((1, n_token), np.int32)
    features["affinity_token_mask"] = np.zeros((1, n_token), np.float32)
    if affinity:
        features["affinity_token_mask"][0, -1] = 1
    features["token_to_rep_atom"] = np.zeros((1, n_token, 3), np.int64)
    features["token_to_rep_atom"][0, :, 0] = 1
    features["atom_to_token"] = np.zeros((1, 3, n_token), np.int64)
    features["atom_to_token"][0, :, 0] = 1
    monkeypatch.setattr(
        native_api, "featurize", lambda **kwargs: (features, "job", tmp_path)
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda _: {"trunk": {"weight": np.asarray([1.0], np.float32)}},
    )
    counts = {"traces": 0, "calls": 0}
    calls = []
    original_jit = native_api._boltz_jit

    def tiny_predict(params, feats, key, **kwargs):
        # The learned blocks are stubbed. The real API still chooses feature
        # storage, static kwargs, compiler policy, graph closure and owner.
        counts["traces"] += 1
        chunks = resolve_long_sequence_chunks(
            feats["token_pad_mask"].shape[-1],
            chunk_size=128,
            triangle_attention_chunk=None,
            triangle_attention_q_chunk=kwargs["triangle_attention_q_chunk"],
            token_attention_chunk=kwargs["token_attention_chunk"],
        )
        value = (
            params["trunk"]["weight"][0]
            + jax.random.uniform(key, ())
            + chunks["triangle_attention_chunk"] * 1e-6
        )
        samples = kwargs["multiplicity"]
        atoms = feats["atom_pad_mask"].shape[-1]
        tokens = feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.full((samples, atoms, 3), value),
            "plddt": jnp.ones((samples, tokens)),
            "iptm": jnp.full((samples,), value),
        }

    class ObservedRunner:
        def __init__(self, function, **kwargs):
            self.runner = original_jit(function, **kwargs)

        def lower(self, *args):
            lowered = self.runner.lower(*args)
            calls.append(lowered)
            return lowered

        def __call__(self, *args):
            counts["calls"] += 1
            self.lower(*args)
            return self.runner(*args)

        def _cache_size(self):
            return self.runner._cache_size()

        def clear_cache(self):
            self.runner.clear_cache()

    monkeypatch.setattr(native_api, "_boltz_jit", ObservedRunner)
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", tiny_predict
    )
    return counts, calls


@pytest.mark.parametrize(
    ("n_token", "expected"),
    [
        (512, (128, 128, None, None)),
        (1536, (128, 128, None, 128)),
        (2048, (128, 32, None, 128)),
        (3072, (64, 16, 256, 64)),
    ],
)
def test_record_uses_native_token_rungs(n_token, expected):
    native_api._record_prediction_chunking(
        {"token_pad_mask": np.zeros((1, n_token), np.float32)},
        {
            "multiplicity": 5,
            "diffusion_chunk_size": None,
            "run_confidence": True,
            "confidence_sequentially": True,
        },
    )
    record = execution.recorded_chunking()["profiles"][0]
    assert record["model"] == "boltz2"
    assert record["n_token"] == n_token
    assert record["policy"] == "token_rungs"
    chunks = record["chunks"]
    assert tuple(
        chunks[key]
        for key in (
            "chunk_size",
            "triangle_attention_chunk",
            "triangle_attention_q_chunk",
            "token_attention_chunk",
        )
    ) == expected
    assert chunks["diffusion_chunk_size"] == 5
    assert chunks["confidence_sample_chunk_size"] == 1
    if n_token > 2048:
        assert chunks["confidence_chunk_size"] == 128
        assert chunks["confidence_triangle_attention_chunk"] == 32
        assert chunks["confidence_triangle_attention_q_chunk"] == 512


def test_record_preserves_explicit_dense_widths_and_diffusion_chunk():
    native_api._record_prediction_chunking(
        {"token_pad_mask": np.zeros((1, 3072), np.float32)},
        {
            "multiplicity": 8,
            "diffusion_chunk_size": 3,
            "token_attention_chunk": 0,
            "triangle_attention_q_chunk": 0,
        },
    )
    chunks = execution.recorded_chunking()["profiles"][0]["chunks"]
    assert chunks["diffusion_chunk_size"] == 3
    assert chunks["token_attention_chunk"] == 0
    assert chunks["triangle_attention_q_chunk"] == 0


def test_same_gpu_kind_ignores_ecc_budget_in_namespace(tmp_path, monkeypatch):
    from foldjax import _gpu
    from foldjax.cache import runtime_profile

    request = _request(tmp_path)
    backend = Boltz2Backend()
    profiles, namespaces, runtimes = [], [], []
    for gib in (22, 24):
        device = SimpleNamespace(
            platform="gpu",
            device_kind="NVIDIA GeForce RTX 4090",
            id=0,
            process_index=0,
            local_hardware_id=0,
            memory_stats=lambda: {"bytes_limit": gib * 2**30},
        )
        with monkeypatch.context() as patch:
            patch.setattr(_gpu, "default_device", lambda *args: device)
            patch.setattr(jax, "default_device", lambda _: nullcontext())
            patch.setattr(jax, "devices", lambda **kwargs: [device])
            patch.setattr(jax, "local_device_count", lambda **kwargs: 1)
            patch.setattr(jax, "process_count", lambda: 1)
            current = dataclasses.replace(
                request,
                options={**request.options, "memory_budget_gib": float(gib)},
            )
            profiles.append(backend.cache_profile(current))
            namespaces.append(resolve_cache_dir(current, backend))
            runtimes.append(runtime_profile())
    assert profiles[0] == profiles[1]
    assert namespaces[0] == namespaces[1]
    assert runtimes[0] == runtimes[1]


def test_actual_predict_owner_records_each_call_and_lowers_once_across_ecc_budgets(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    counts, lowered = _install_runtime(monkeypatch, tmp_path)
    backend = Boltz2Backend()
    records = []
    with backend.session((request,)):
        for gib in (22, 24):
            execution.clear_chunking_record()
            native_api.predict(
                input=request.input,
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=0,
                write_fmt=None,
                memory_budget_gib=float(gib),
                _runtime=backend,
            )
            records.append(execution.recorded_chunking())
    assert counts == {"traces": 1, "calls": 2}
    assert records[0] == records[1]
    assert records[0]["profiles"][0]["n_token"] == 512
    assert lowered[0].as_text() == lowered[1].as_text()
    assert lowered_cache_key(lowered[0]) == lowered_cache_key(lowered[1])


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("prefix_supported", [False, True])
def test_lower_only_matches_the_production_graph_without_execution_or_cache_writes(
    tmp_path, monkeypatch, padded, prefix_supported
):
    from jax._src import compiler

    request = dataclasses.replace(
        _request(tmp_path),
        num_seeds=1,
        padding={"tokens": 512, "atoms": 32} if padded else None,
    )
    counts, lowered = _install_runtime(monkeypatch, tmp_path, n_token=3)
    monkeypatch.setattr(
        native_api, "_prefix_rng_is_supported", lambda: prefix_supported
    )
    backend = Boltz2Backend()
    backend.predict(request)
    production = lowered[-1]
    previous_calls = counts["calls"]
    # A verifier must not enter the persistent cache scope even when callers
    # supplied the same compile_cache path as predict.
    monkeypatch.setattr(
        native_api,
        "compilation_cache_scope",
        lambda *a, **k: pytest.fail("lower-only initialized the compilation cache"),
    )
    monkeypatch.setattr(
        compiler,
        "backend_compile_and_load",
        lambda *a, **k: pytest.fail("lower-only compiled a program"),
    )
    verified = backend.lower_cache_graphs(request)
    assert counts["calls"] == previous_calls
    assert len(verified) == 1
    name, actual = verified[0]
    assert name == "primary"
    assert actual.as_text() == production.as_text()
    assert actual.in_avals == production.in_avals
    assert actual.in_tree == production.in_tree
    assert lowered_cache_key(actual) == lowered_cache_key(production)


def test_lower_only_refuses_cp_before_feature_preparation(tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: pytest.fail("unsupported CP started feature preparation"),
    )
    with pytest.raises(UnsupportedCacheVerificationError, match="cp_devices=1"):
        native_api.predict(
            input=request.input,
            weights=request.weights,
            mols=request.options["mols"],
            cp_devices=4,
            _lower_only=True,
        )


def test_lower_only_refuses_affinity_without_executing_primary(tmp_path, monkeypatch):
    request = _request(tmp_path)
    counts, _ = _install_runtime(monkeypatch, tmp_path, affinity=True)
    with pytest.raises(UnsupportedCacheVerificationError, match="affinity stage"):
        Boltz2Backend().lower_cache_graphs(request)
    assert counts == {"traces": 0, "calls": 0}


def test_lower_only_refuses_eager_steering(tmp_path, monkeypatch):
    request = _request(tmp_path)
    _install_runtime(monkeypatch, tmp_path)
    with pytest.raises(UnsupportedCacheVerificationError, match="eager steering"):
        native_api.predict(
            input=request.input,
            weights=request.weights,
            mols=request.options["mols"],
            use_potentials=True,
            _lower_only=True,
        )


_GIB = 2**30


@pytest.mark.parametrize(
    ("n_token", "budget_gib", "expected_lazy"),
    [
        # Below BOLTZ2_PEAK.domain_tokens (1003..4888) the gate is judged at the
        # lower edge, which bounds both the law and the cache from above.
        (200, 24, False),
        (1000, 24, False),
        (1000, 48, False),
        (1000, 96, False),
        # Lower covered edges (1,003 exact domain start, 1,024 standard bucket):
        # upper (~9.0-9.1 GiB) + bf16 cache (~0.72-0.75 GiB) <= 21.6 GiB.
        (1003, 24, False),
        (1024, 24, False),
        # 3,000 tokens: upper (~31.17 GiB) + bf16 cache (6.44 GiB) = 37.61 GiB.
        # Thresholds: 21.6 GiB (24 GiB) -> lazy; 43.2 GiB (48 GiB) and
        # 86.4 GiB (96 GiB) -> eager.
        (3000, 24, True),
        (3000, 48, False),
        (3000, 96, False),
        # Upper covered edge (4,888): upper (76.65 GiB) + cache (17.09 GiB) > 86.4 GiB.
        (4888, 96, True),
        # 5,000 tokens is above BOLTZ2_PEAK.domain_tokens (1003..4888): stays lazy.
        (5000, 24, True),
        (5000, 48, True),
        (5000, 96, True),
    ],
)
def test_token_trans_bias_gate_decisions_and_arithmetic(
    n_token: int, budget_gib: int, expected_lazy: bool
) -> None:
    from foldjax.models.boltz2.bridge.export_weights import (
        TOKEN_LAYERS,
        TOKEN_TRANSFORMER_HEADS,
    )

    law = memory_policy.BOLTZ2_PEAK
    budget = memory_policy.resolve_budget(
        pool_bytes=budget_gib * _GIB,
        card_bytes=budget_gib * _GIB,
        override_gib=None,
    )
    gate_tokens = max(n_token, law.domain_tokens[0])
    cache_bytes = native_api._token_trans_bias_cache_bytes(
        gate_tokens, compute_dtype="bfloat16", diffusion_compute_dtype="float32"
    )
    assert cache_bytes == (
        TOKEN_LAYERS * TOKEN_TRANSFORMER_HEADS * gate_tokens * gate_tokens * 2
    )
    threshold = memory_policy._threshold(
        budget.budget_bytes, model="boltz2", card_bytes=budget.card_bytes
    )
    assert threshold == int(budget_gib * _GIB * memory_policy.ADMISSION_FRACTION)
    expected_from_law = (not law.covers(gate_tokens)) or (
        law.upper(gate_tokens) + cache_bytes > threshold
    )
    assert expected_from_law is expected_lazy
    assert (
        native_api._resolve_lazy_token_trans_bias(
            n_token,
            budget=budget,
            cp_devices=1,
            compute_dtype="bfloat16",
            diffusion_compute_dtype="float32",
        )
        is expected_lazy
    )


def test_token_trans_bias_gate_edges_dtypes_and_failures() -> None:
    import math

    law = memory_policy.BOLTZ2_PEAK
    budget_96 = memory_policy.resolve_budget(
        pool_bytes=96 * _GIB, card_bytes=96 * _GIB, override_gib=None
    )
    no_budget = memory_policy.resolve_budget(
        pool_bytes=None, card_bytes=None, override_gib=None
    )

    # No budget (CPU / unknown device) stays lazy.
    assert native_api._resolve_lazy_token_trans_bias(3000, budget=None) is True
    assert native_api._resolve_lazy_token_trans_bias(3000, budget=no_budget) is True
    # A planning override on a host with no device stays lazy.
    host_override = memory_policy.resolve_budget(
        pool_bytes=None, card_bytes=None, override_gib=96.0
    )
    assert host_override.budget_bytes is not None
    assert native_api._resolve_lazy_token_trans_bias(3000, budget=host_override) is True

    # Context parallelism stays lazy even when single-device budget fits.
    for cp_devices in (2, 4):
        assert (
            native_api._resolve_lazy_token_trans_bias(
                3000, budget=budget_96, cp_devices=cp_devices
            )
            is True
        )

    # Exact threshold equality: equal to threshold is eager, 1 byte below is lazy.
    cache_3k = native_api._token_trans_bias_cache_bytes(
        3000, compute_dtype="bfloat16", diffusion_compute_dtype="float32"
    )
    required = law.upper(3000) + cache_3k
    exact_budget = math.ceil(required / memory_policy.ADMISSION_FRACTION)
    while (
        memory_policy._threshold(exact_budget, model="boltz2") > required
    ):
        exact_budget -= 1
    while (
        memory_policy._threshold(exact_budget, model="boltz2") < required
    ):
        exact_budget += 1
    assert memory_policy._threshold(exact_budget, model="boltz2") == required
    assert (
        native_api._resolve_lazy_token_trans_bias(
            3000,
            budget=memory_policy.MemoryBudget(
                pool_bytes=exact_budget,
                card_bytes=exact_budget,
                override_bytes=None,
                budget_bytes=exact_budget,
                source="pool",
            ),
        )
        is False
    )
    assert (
        native_api._resolve_lazy_token_trans_bias(
            3000,
            budget=memory_policy.MemoryBudget(
                pool_bytes=exact_budget - 1,
                card_bytes=exact_budget - 1,
                override_bytes=None,
                budget_bytes=exact_budget - 1,
                source="pool",
            ),
        )
        is True
    )

    # FP32 trunk + FP32 diffusion stores float32 bias (4 bytes): doubles cache
    # and flips 3,000 tokens on 48 GiB from eager (37.6 GiB) to lazy (44.0 GiB).
    cache_3k_fp32 = native_api._token_trans_bias_cache_bytes(
        3000, compute_dtype="float32", diffusion_compute_dtype="float32"
    )
    assert cache_3k_fp32 == 2 * cache_3k
    budget_48 = memory_policy.resolve_budget(
        pool_bytes=48 * _GIB, card_bytes=48 * _GIB, override_gib=None
    )
    assert (
        native_api._resolve_lazy_token_trans_bias(
            3000,
            budget=budget_48,
            compute_dtype="float32",
            diffusion_compute_dtype="float32",
        )
        is True
    )
    # Memory a caller already holds beyond the weights comes out of the headroom.
    assert native_api._resolve_lazy_token_trans_bias(3000, budget=budget_48) is False
    assert (
        native_api._resolve_lazy_token_trans_bias(
            3000, budget=budget_48, held_bytes=17 * _GIB
        )
        is True
    )
    # CPU reports no allocator statistics, so nothing is held against the gate.
    assert native_api._allocator_bytes_beyond_params(None) is None
    # FP32 trunk + BF16 diffusion narrows the stored bias to bfloat16 (2 bytes).
    assert (
        native_api._token_trans_bias_cache_bytes(
            3000, compute_dtype="float32", diffusion_compute_dtype="bfloat16"
        )
        == cache_3k
    )

    # Loaded params supply depth and heads (float32 compute bounds the width
    # at float32); an empty list falls back to the released constants.
    custom_params = {
        "conditioned_diffusion": {
            "diffusion_conditioning": {
                "token_trans_proj_z": [
                    {"linear": {"kernel": jnp.zeros((16, 8), dtype=jnp.bfloat16)}}
                    for _ in range(6)
                ]
            }
        }
    }
    assert (
        native_api._token_trans_bias_cache_bytes(
            100,
            compute_dtype="float32",
            diffusion_compute_dtype="float32",
            params=custom_params,
        )
        == 6 * 8 * 100 * 100 * 4
    )
    empty_proj_params = {
        "conditioned_diffusion": {
            "diffusion_conditioning": {"token_trans_proj_z": []}
        }
    }
    assert (
        native_api._token_trans_bias_cache_bytes(3000, params=empty_proj_params)
        == cache_3k
    )
    # The constants the gate holds a run to are the featurizer's and predict's.
    import inspect

    from foldjax.models.boltz2.data.featurize import UPSTREAM_PARSE_CAP

    assert native_api._RELEASED_MSA_PARSE_CAP == UPSTREAM_PARSE_CAP
    assert (
        inspect.signature(native_api.predict).parameters["num_steps"].default
        == native_api._RELEASED_SAMPLING_STEPS
    )

    # The native loader's stacked layout sizes the cache from the stack, so a
    # deeper or wider checkpoint is not budgeted at the released constants.
    from foldjax.models._stacking import StackedLayers

    stacked_params = {
        "conditioned_diffusion": {
            "diffusion_conditioning": {
                "token_trans_proj_z": StackedLayers.from_layers(
                    {"linear": {"kernel": jnp.zeros((16, 20), dtype=jnp.bfloat16)}}
                    for _ in range(30)
                )
            }
        }
    }
    assert native_api._token_trans_bias_cache_bytes(
        100, params=stacked_params
    ) == (30 * 20 * 100 * 100 * 2)
    # A projection that is present but cannot be sized keeps the bias lazy.
    unsized_params = {
        "conditioned_diffusion": {
            "diffusion_conditioning": {"token_trans_proj_z": [{"linear": {}}]}
        }
    }
    assert native_api._token_trans_bias_cache_bytes(100, params=unsized_params) is None
    assert (
        native_api._resolve_lazy_token_trans_bias(
            1003, budget=budget_48, params=unsized_params
        )
        is True
    )

    # Malformed, zero, negative, and boolean inputs fail loudly.
    for bad_tokens in (0, -1, True):
        with pytest.raises(ValueError, match="n_token"):
            native_api._token_trans_bias_cache_bytes(bad_tokens)
        with pytest.raises(ValueError, match="n_token"):
            native_api._resolve_lazy_token_trans_bias(bad_tokens, budget=budget_96)
    for bad_cp in (0, -2, True):
        with pytest.raises(ValueError, match="cp_devices"):
            native_api._resolve_lazy_token_trans_bias(
                3000, budget=budget_96, cp_devices=bad_cp
            )
    with pytest.raises(ValueError, match="compute_dtype"):
        native_api._resolve_lazy_token_trans_bias(
            3000, budget=budget_96, compute_dtype="float16"
        )
    with pytest.raises(ValueError, match="diffusion_compute_dtype"):
        native_api._resolve_lazy_token_trans_bias(
            3000, budget=budget_96, diffusion_compute_dtype="float16"
        )
    with pytest.raises(ValueError, match="budget_bytes"):
        native_api._resolve_lazy_token_trans_bias(
            3000,
            budget=memory_policy.MemoryBudget(
                pool_bytes=0,
                card_bytes=0,
                override_bytes=None,
                budget_bytes=0,
                source="pool",
            ),
        )


def test_token_trans_bias_choice_splits_runner_identity_in_session(
    tmp_path, monkeypatch
) -> None:
    """Lazy vs eager resolves per budget and never shares a retained JIT runner."""
    runtime = (("runtime", "cpu"),)
    lazy_id = native_api._runner_identity(
        predict_function=test_token_trans_bias_choice_splits_runner_identity_in_session,
        predict_kwargs={"lazy_token_trans_bias": True},
        noise_mode="none",
        runtime=runtime,
    )
    eager_id = native_api._runner_identity(
        predict_function=test_token_trans_bias_choice_splits_runner_identity_in_session,
        predict_kwargs={"lazy_token_trans_bias": False},
        noise_mode="none",
        runtime=runtime,
    )
    assert lazy_id != eager_id

    request = _request(tmp_path)
    counts, _ = _install_runtime(monkeypatch, tmp_path, n_token=3000)
    monkeypatch.setattr(
        memory_policy,
        "device_memory_budget",
        # No override stands for a host with no detected device, whatever
        # card the test machine has.
        lambda *, override_gib=None: memory_policy.resolve_budget(
            pool_bytes=None,
            card_bytes=None if override_gib is None else 96 * _GIB,
            override_gib=override_gib,
        ),
    )
    seen_lazy: list[bool] = []
    original_predict = native_api.predict

    def observe_predict(params, feats, key, **kwargs):
        seen_lazy.append(kwargs["lazy_token_trans_bias"])
        samples = kwargs["multiplicity"]
        atoms = feats["atom_pad_mask"].shape[-1]
        tokens = feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.zeros((samples, atoms, 3)),
            "plddt": jnp.ones((samples, tokens)),
            "iptm": jnp.zeros((samples,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", observe_predict
    )
    backend = Boltz2Backend()
    runner_entries = []
    with backend.session((request,)):
        for gib in (None, 24.0, 48.0, 96.0):
            original_predict(
                input=request.input,
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=0,
                write_fmt=None,
                memory_check="warn",
                memory_budget_gib=gib,
                _runtime=backend,
            )
            runner_entries.append(backend._runners["primary"])

    # CPU/no-budget and 24 GiB both resolve lazy and share one runner; 48 GiB
    # and 96 GiB both resolve eager and share a distinct second runner.
    assert seen_lazy == [True, False]
    assert runner_entries[0] is runner_entries[1]
    assert runner_entries[2] is runner_entries[3]
    assert runner_entries[1][0] != runner_entries[2][0]
    assert runner_entries[1][1] is not runner_entries[2][1]
    assert counts["calls"] == 4


def test_token_trans_bias_stays_lazy_outside_the_fitted_profile(
    tmp_path, monkeypatch
) -> None:
    request = _request(tmp_path)
    _install_runtime(monkeypatch, tmp_path, n_token=3000)
    monkeypatch.setattr(
        memory_policy,
        "device_memory_budget",
        # No override stands for a host with no detected device, whatever
        # card the test machine has.
        lambda *, override_gib=None: memory_policy.resolve_budget(
            pool_bytes=None,
            card_bytes=None if override_gib is None else 96 * _GIB,
            override_gib=override_gib,
        ),
    )
    seen_lazy: list[bool] = []

    def observe_predict(params, feats, key, **kwargs):
        seen_lazy.append(kwargs["lazy_token_trans_bias"])
        samples = kwargs["multiplicity"]
        atoms = feats["atom_pad_mask"].shape[-1]
        tokens = feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.zeros((samples, atoms, 3)),
            "plddt": jnp.ones((samples, tokens)),
            "iptm": jnp.zeros((samples,)),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", observe_predict
    )

    def run(**overrides):
        native_api.predict(
            input=request.input,
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            seed=0,
            write_fmt=None,
            memory_check="warn",
            memory_budget_gib=96.0,
            **overrides,
        )

    from foldjax.schema import PaddingConfig

    run()
    run(padding=PaddingConfig())
    run(num_samples=memory_policy.CALIBRATED_NUM_SAMPLES + 1)
    run(return_confidence_logits=True)
    run(padding=PaddingConfig(tokens=3072))
    # The law was fitted on the fused denoiser attention; the blocked path and
    # a pinned chunk hold score tensors it never saw.
    run(diffusion_attention_backend="xla")
    run(token_attention_chunk=0)
    run(triangle_backend="xla")
    run(triangle_attention_q_chunk=0)
    run(glu_backend="xla")
    run(attention_backend="tokamax")
    run(max_msa_depth=32768)
    run(num_steps=5000)
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    run()
    monkeypatch.delenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND")
    monkeypatch.setattr(native_api, "MATMUL_PRECISION", "highest")
    run()
    assert seen_lazy == [False, False, True, True, True] + [True] * 10


def test_token_trans_bias_cache_bound_covers_mixed_projection_layers() -> None:
    params = {
        "conditioned_diffusion": {
            "diffusion_conditioning": {
                "token_trans_proj_z": [
                    {"linear": {"kernel": jnp.zeros((8, 2), jnp.bfloat16)}},
                    {"linear": {"kernel": jnp.zeros((8, 3), jnp.float32)}},
                ]
            }
        }
    }
    cache = native_api._token_trans_bias_cache_bytes(
        10, compute_dtype="float32", diffusion_compute_dtype="float32", params=params
    )
    # Concatenated in float32, two layers at most three channels wide.
    assert cache == 2 * 3 * 10 * 10 * 4
