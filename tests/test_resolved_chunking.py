"""Admission budgets cannot silently change native compilation settings."""

from __future__ import annotations

import dataclasses
import json
from contextlib import nullcontext
from contextvars import copy_context
from types import SimpleNamespace

import jax
import pytest

from foldjax import execution, memory_policy
from foldjax.api import resolve_cache_dir
from foldjax.manifest import describe_run
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest, PredictionResult


@pytest.fixture(autouse=True)
def isolated_chunking_record():
    execution.clear_chunking_record()
    memory_policy.clear_record()
    yield
    execution.clear_chunking_record()
    memory_policy.clear_record()


def _record(n_token=128):
    execution.record_resolved_chunking(
        model="protenix",
        n_token=n_token,
        chunks={"diffusion_chunk_size": 5, "triangle_mul_chunk_size": None},
        policy="auto",
    )


def test_record_is_bounded_deduplicated_and_context_local():
    assert execution.recorded_chunking() is None
    _record()
    _record()
    first = execution.recorded_chunking()
    assert len(first["profiles"]) == 1
    first["profiles"][0]["chunks"]["diffusion_chunk_size"] = 99
    assert (
        execution.recorded_chunking()["profiles"][0]["chunks"]["diffusion_chunk_size"]
        == 5
    )

    context = copy_context()
    context.run(execution.clear_chunking_record)
    assert context.run(execution.recorded_chunking) is None
    assert execution.recorded_chunking() is not None
    for n_token in range(129, 140):
        _record(n_token)
    record = execution.recorded_chunking()
    assert len(record["profiles"]) == 8
    assert record["profiles_truncated"] == 4
    json.dumps(record)


def _request(tmp_path, model):
    job = tmp_path / "job.json"
    job.write_text('{"entities": [{"type": "protein", "id": "A", "sequence": "AAA"}]}')
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"fixture")
    return PredictionRequest(
        model=model,
        input=job,
        weights=weights,
        cache_dir=tmp_path / "compile",
        output_dir=tmp_path / "out",
        msa="single",
        seed=0,
    )


def test_manifest_adds_only_a_resolved_record(tmp_path):
    request = _request(tmp_path, "protenix")
    result = PredictionResult(
        model="protenix", samples=(), output_dir=request.output_dir
    )
    assert "resolved_chunking" not in describe_run(request, result)
    _record()
    document = describe_run(request, result)
    assert document["resolved_chunking"] == execution.recorded_chunking()
    assert "memory_budget_gib" not in json.dumps(document["resolved_chunking"])
    from foldjax.summary import load_schema
    from tests._schema_lite import errors, undeclared

    assert errors(document, load_schema("run")) == []
    assert undeclared(document, load_schema("run")) == []


def test_prediction_clears_chunking_between_seeds_and_on_all_resumed_runs(tmp_path):
    import foldjax
    from foldjax.backends.opendde import OpenDDEBackend
    from foldjax.registry import backend_override
    from foldjax.schema import PredictionSample

    calls = []

    class Recorder(OpenDDEBackend):
        def predict(self, request):
            calls.append(request.seed)
            if request.seed == 0:
                _record()
            structure = request.output_dir / "sample.cif"
            structure.write_text("data_mock\n#\n")
            return PredictionResult(
                model="opendde",
                samples=(
                    PredictionSample(
                        seed=request.seed, structure_path=structure, scores={"ptm": 0.5}
                    ),
                ),
                output_dir=request.output_dir,
            )

    request = dataclasses.replace(
        _request(tmp_path, "opendde"),
        seed=None,
        seeds=(0, 1),
        cache_dir=None,
        use_compile_cache=False,
        resume=True,
    )
    with backend_override("opendde", Recorder):
        foldjax.predict(request)
        first = json.loads((request.output_dir / "seed_0/foldjax_run.json").read_text())
        second = json.loads(
            (request.output_dir / "seed_1/foldjax_run.json").read_text()
        )
        assert "resolved_chunking" in first
        assert "resolved_chunking" not in second
        _record(512)
        foldjax.predict(request)
    assert calls == [0, 1]
    assert execution.recorded_chunking() is None
    aggregate = json.loads((request.output_dir / "foldjax_run.json").read_text())
    assert "resolved_chunking" not in aggregate


@pytest.mark.parametrize("model", ["protenix", "openfold3", "opendde"])
def test_ecc_budget_does_not_split_namespace_for_the_same_gpu_kind(
    tmp_path, monkeypatch, model
):
    from foldjax import _gpu

    request = _request(tmp_path, model)
    backend = get_backend(model)
    namespaces, profiles, runtime_profiles = [], [], []
    # Only the public device memory statistics differ. Concrete ordinals and
    # topology are the same, so this isolates ECC capacity from placement.
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
                request, options={"memory_budget_gib": float(gib)}
            )
            profiles.append(backend.cache_profile(current))
            namespaces.append(resolve_cache_dir(current, backend))
            from foldjax.cache import runtime_profile

            runtime_profiles.append(runtime_profile())
    assert profiles[0] == profiles[1]
    assert runtime_profiles[0] == runtime_profiles[1]
    assert namespaces[0] == namespaces[1]
    assert "memory_budget_gib" not in profiles[0]


class _RecordingPool(BoundedJitPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return super().__call__(*args, **kwargs)


def _native_pool(monkeypatch, model):
    if model == "protenix":
        from foldjax.models.protenix.models import model as implementation
    else:
        from foldjax.models.opendde.models import model as implementation

    def trunk_only(features, params, schedule, **kwargs):
        # Heavy learned blocks are stubbed; the production compiled entry,
        # static-option identity and feature normalization remain exercised.
        return {"coordinate": features["restype"][..., :3]}

    monkeypatch.setattr(implementation, f"{model}_infer_static", trunk_only)
    pool = _RecordingPool(
        getattr(implementation, f"_{model}_infer_graph"),
        static_argnames=(
            *implementation.GRAPH_STATIC_ARGNAMES,
            "params_treedef",
            "params_flags",
        ),
    )
    monkeypatch.setattr(implementation, f"_compiled_{model}_infer", pool)
    return pool


def test_protenix_real_runner_keeps_one_program_across_ecc_budgets(
    tmp_path, monkeypatch
):
    from foldjax.models.protenix import runner
    from foldjax.models.protenix.cli import predict
    from foldjax.models.protenix.data import featurize_json
    from tests.models.protenix.test_model import _toy_features, _toy_params

    request = _request(tmp_path, "protenix")
    request.input.write_text('[{"name": "tiny", "modelSeeds": [0], "sequences": []}]')
    monkeypatch.setattr(
        featurize_json, "featurize_protein_json", lambda *a, **k: dict(_toy_features())
    )
    monkeypatch.setattr(runner, "_load_prepared_params", lambda *a, **k: _toy_params())
    pool = _native_pool(monkeypatch, "protenix")
    records = []
    try:
        for gib in (22, 24):
            execution.clear_chunking_record()
            predict.main(
                [
                    "--input-json",
                    str(request.input),
                    "--weights",
                    str(request.weights),
                    "--out",
                    str(request.output_dir),
                    "--model-name",
                    "protenix_mini_default_v0.5.0",
                    "--no-compile-cache",
                    "--prewarm-only",
                    "--memory-budget-gib",
                    str(gib),
                    "--num-steps",
                    "1",
                    "--num-recycles",
                    "1",
                    "--diffusion-attention-backend",
                    "xla_jit",
                    "--trunk-single-attention-backend",
                    "xla_jit",
                    "--trunk-triangle-attention-backend",
                    "xla_jit",
                    "--confidence-triangle-attention-backend",
                    "xla_jit",
                ]
            )
            records.append(execution.recorded_chunking())
        assert records[0] == records[1]
        assert records[0]["profiles"][0]["chunks"]["triangle_mul_chunk_size"] is None
        assert pool._entry_count() == 1
        assert (
            len({pool.lower(*args, **kwargs).as_text() for args, kwargs in pool.calls})
            == 1
        )
        assert memory_policy.recorded()["budget_bytes"] == 24 * 2**30
    finally:
        pool.clear_cache()


def test_opendde_real_runner_keeps_one_program_across_ecc_budgets(monkeypatch):
    from foldjax.models.opendde import runner
    from tests.models.opendde.test_padding import _opendde_features

    pool = _native_pool(monkeypatch, "opendde")
    records = []
    try:
        for gib in (22, 24):
            execution.clear_chunking_record()
            features = _opendde_features(msa_depth=3)
            memory_policy.admit(
                model="opendde",
                n_token=128,
                msa_rows=None,
                candidates=(("bf16 trunk", memory_policy.OPENDDE_BF16_PEAK),),
                budget=memory_policy.resolve_budget(
                    pool_bytes=gib * 2**30, card_bytes=gib * 2**30, override_gib=None
                ),
            )
            runner._predict(
                features,
                (),
                seed=0,
                num_samples=1,
                num_steps=1,
                num_recycles=1,
                n_queries=2,
                n_keys=4,
                diffusion_attention_backend="xla_jit",
                trunk_single_attention_backend="xla_jit",
                structural_single_attention_backend="xla_jit",
                run_confidence_scores=False,
            )
            records.append(execution.recorded_chunking())
        assert records[0] == records[1]
        assert pool._entry_count() == 1
        assert (
            len({pool.lower(*args, **kwargs).as_text() for args, kwargs in pool.calls})
            == 1
        )
        assert memory_policy.recorded()["budget_bytes"] == 24 * 2**30
    finally:
        pool.clear_cache()


def test_openfold3_native_config_and_scores_lower_once_across_ecc_budgets(monkeypatch):
    from foldjax.models.openfold3 import inference
    from tests.models.openfold3.test_pae_metric_sinking import _install_tiny_predict
    from tests.models.openfold3.test_stable_compile import _table

    features, _ = _install_tiny_predict(monkeypatch)
    params = inference.InferenceParams(*([None] * 8))
    args = (jax.random.key(0), features, params)
    lowered, configs = [], []
    inference._compiled_predict.clear_cache()
    try:
        for gib in (22, 24):
            config = inference.released_config(
                n_token=4,
                n_atom=4,
                num_samples=3,
                num_steps=1,
                num_recycles=1,
                memory_budget=memory_policy.resolve_budget(
                    pool_bytes=gib * 2**30, card_bytes=gib * 2**30, override_gib=None
                ),
                has_atomized_tokens=False,
                max_array_bytes=0,
            )
            configs.append(config)
            run = inference.compile_predict(
                config, _table(), n_chain=3, triangle_kernel="xla"
            )
            output = run(*args)
            jax.block_until_ready(output)
            lowered.append(run.lower(*args).as_text())
        assert configs[0] == configs[1]
        assert lowered[0] == lowered[1]
        assert inference._compiled_predict._cache_size() == 1
        assert memory_policy.recorded()["budget_bytes"] == 24 * 2**30
    finally:
        inference._compiled_predict.clear_cache()


def test_openfold3_adapter_records_only_the_final_padded_widths(tmp_path, monkeypatch):
    import numpy as np

    from foldjax.backends import openfold3 as adapter

    features = {
        "token_mask": np.ones((1, 8), dtype=np.float32),
        "atom_mask": np.ones((1, 16), dtype=np.float32),
    }

    class StopBeforeWeightsError(Exception):
        pass

    def stop(batch):
        raise StopBeforeWeightsError

    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            pocket_sampling_config=lambda batch: None,
            featurize_query_with_metadata=lambda *a, **k: (features, None),
            prepare_msa_cycle_features=lambda batch, depth, **kwargs: batch,
            collapse_identical_templates=lambda batch: batch,
            has_atomized_tokens=lambda batch: False,
            pad_features=lambda batch, **kwargs: batch,
            normalize_asym_ids=stop,
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            released_config=lambda **kwargs: SimpleNamespace(
                msa_depth=1024,
                num_recycles=4,
                pair_chunk_size=64 if kwargs.get("padded") else None,
                diffusion_chunk_size=5,
                per_sample_token_cutoff=0,
            )
        ),
        **{
            name: SimpleNamespace()
            for name in (
                "foldjax.models.openfold3.output",
                "foldjax.models.openfold3.bridge.chemistry",
                "foldjax.models.openfold3.bridge.checkpoint",
                "foldjax.models.openfold3.bridge.torch_mapping",
                "jax",
            )
        },
    }
    monkeypatch.setattr(adapter, "import_module", lambda name: modules[name])
    monkeypatch.setattr(
        adapter,
        "_padding_plan",
        lambda *args, **kwargs: SimpleNamespace(
            target={"tokens": 16, "atoms": 32, "templates": 4, "msa": 1024}
        ),
    )
    with pytest.raises(StopBeforeWeightsError):
        adapter.OpenFold3Backend().predict(
            dataclasses.replace(_request(tmp_path, "openfold3"), padding=True)
        )
    assert execution.recorded_chunking()["profiles"] == [
        {
            "model": "openfold3",
            "n_token": 16,
            "policy": "auto",
            "chunks": {
                "pair_chunk_size": 64,
                "diffusion_chunk_size": 5,
                "per_sample_token_cutoff": 0,
            },
        }
    ]
