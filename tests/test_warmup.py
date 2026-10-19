from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from foldjax.cache import CacheSnapshot
from foldjax.schema import (
    PredictionOutputError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)
from foldjax.warmup import CacheWarmResult, warm_cache


def _request(tmp_path: Path, **updates) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n")
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"weights")
    values = {
        "model": "boltz2",
        "input": input_path,
        "weights": weights,
        "cache_dir": tmp_path / "cache",
        "input_format": "native",
        "seeds": (7, 11),
    }
    values.update(updates)
    return PredictionRequest(**values)


@pytest.mark.parametrize("publication", ["written", "loaded", "discarded", "threshold"])
def test_af3_compile_checkpoint_requires_persistence_and_never_executes_model(
    tmp_path, monkeypatch, trust_ancestors_above_tmp_path, publication
):
    import jax
    import numpy as np
    from jax._src import compiler

    from foldjax import cache, warmup
    from foldjax._jax_cache_tiers import LayeredCache
    from foldjax.backends import _tokamax_autotune as tuning

    request = _request(tmp_path, model="alphafold3")
    namespace = tmp_path / "namespace"
    monkeypatch.setattr(warmup, "resolve_requests", lambda _r: (request,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _m: SimpleNamespace(name="alphafold3")
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_a: namespace)
    monkeypatch.setattr(warmup, "runtime_dir", lambda *_a: tmp_path / "runtime")
    monkeypatch.setattr(warmup, "device_peak_bytes", lambda: None)
    executions = []

    @jax.jit
    def apply_fn(value):
        return value @ value + 3

    def predict(run, backend, *, output_root):
        tuning.start_record(strategy="autotune", persistent_installed=True)
        tuning._note_source("store" if publication == "loaded" else "measured")
        with cache.compilation_cache_scope(
            namespace, private_directory=tmp_path / "private"
        ):
            if publication == "threshold":
                jax.config.update("jax_persistent_cache_min_compile_time_secs", 1e9)
            apply_fn(np.arange(16, dtype=np.float32).reshape(4, 4)).block_until_ready()
            executions.append("model executed")
        raise AssertionError("preparation reached model execution")

    monkeypatch.setattr(warmup, "_warm_prediction", predict)
    if publication == "loaded":
        with cache.compilation_cache_scope(
            namespace, private_directory=tmp_path / "private"
        ):
            apply_fn(np.arange(16, dtype=np.float32).reshape(4, 4)).block_until_ready()
        jax.clear_caches()
    if publication == "discarded":
        monkeypatch.setattr(LayeredCache, "put", lambda *_a: None)
    owners = [
        (compiler, "compile_or_get_cached"),
        (compiler, "_cache_read"),
        (LayeredCache, "put"),
    ]
    originals = [getattr(owner, name) for owner, name in owners]
    if publication in {"discarded", "threshold"}:
        with pytest.raises(
            warmup.PredictionError, match="did not confirm a persistent"
        ):
            with warmup._prepare_af3_model():
                warm_cache(request)
    else:
        with warmup._prepare_af3_model():
            result = warm_cache(request)
        assert result.status == "prepared"
        assert result.strategy == "compile_only" and result.samples == 0
        assert result.preparation["source"] == publication
        assert result.preparation["module"] == "jit_apply_fn"
        assert list(namespace.glob("jit_apply_fn-*-cache"))
    assert not executions
    assert [getattr(owner, name) for owner, name in owners] == originals
    assert not warmup._AF3_PREPARATION.get()
    assert not list((tmp_path / "runtime").iterdir())
    apply_fn.clear_cache()


def test_af3_checkpoint_does_not_interrupt_sequential_candidate_compilation(
    tmp_path, trust_ancestors_above_tmp_path
):
    import jax
    import numpy as np

    from foldjax import cache, warmup
    from foldjax.backends import _tokamax_autotune as tuning

    @jax.jit
    def apply_fn(value):
        return value + 7

    tuning.start_record(strategy="autotune", persistent_installed=True)
    with (
        warmup._prepare_af3_model(),
        cache.compilation_cache_scope(
            tmp_path / "cache", private_directory=tmp_path / "private"
        ),
    ):
        # A serial tuner runs on the model thread too; the benchmark's own
        # candidate guard, rather than thread identity/name, excludes it.
        token = tuning._CANDIDATE_COMPILATION.set(True)
        try:
            actual = apply_fn(np.arange(3, dtype=np.float32))
            np.testing.assert_array_equal(actual, np.arange(3) + 7)
        finally:
            tuning._CANDIDATE_COMPILATION.reset(token)
    apply_fn.clear_cache()


@pytest.mark.parametrize("tier", ["shared", "private"])
@pytest.mark.parametrize(
    "outcome", ["hit", "missing", "different_key", "disabled", "deserialize_oom"]
)
def test_af3_fresh_verification_only_loads_the_exact_prepared_model(
    tmp_path, monkeypatch, trust_ancestors_above_tmp_path, tier, outcome
):
    import jax
    import numpy as np
    from jax._src import compiler

    from foldjax import cache, warmup
    from foldjax._jax_cache_tiers import LayeredCache

    namespace, private = tmp_path / "shared", tmp_path / "private"
    if tier == "private":
        original_put = LayeredCache._put

        def put(adapter, selected, *args):
            if selected == "shared":
                raise PermissionError("read-only shared store")
            return original_put(adapter, selected, *args)

        monkeypatch.setattr(LayeredCache, "_put", put)

    @jax.jit
    def apply_fn(value):
        return value @ value + 5

    value = np.arange(16, dtype=np.float32).reshape(4, 4)
    with cache.compilation_cache_scope(namespace, private_directory=private):
        apply_fn(value).block_until_ready()
    (artifact,) = (namespace if tier == "shared" else private).glob(
        "jit_apply_fn-*-cache"
    )
    key = artifact.name.removesuffix("-cache")
    if outcome == "missing":
        artifact.unlink()
    elif outcome == "different_key":
        key = "unprepared-key"
    elif outcome == "disabled":
        monkeypatch.setattr(compiler, "_get_cache_key", lambda *_a, **_k: None)
    elif outcome == "deserialize_oom":
        from jax._src import compilation_cache

        def fail_load(*args, **kwargs):
            raise MemoryError("mock CUDA module allocation failed")

        monkeypatch.setattr(compilation_cache, "get_executable_and_time", fail_load)
    jax.clear_caches()
    compile_calls = []
    original_compile = compiler.backend_compile_and_load

    def compile_model(backend, computation, *args, **kwargs):
        compile_calls.append(str(computation.operation.attributes["sym_name"]))
        return original_compile(backend, computation, *args, **kwargs)

    monkeypatch.setattr(compiler, "backend_compile_and_load", compile_model)
    owners = (
        "_cache_read",
        "backend_compile_and_load",
    )
    originals = [getattr(compiler, name) for name in owners]
    from jax._src import compilation_cache

    original_load = compilation_cache.get_executable_and_time
    with (
        warmup._verify_af3_artifacts((key,)),
        cache.compilation_cache_scope(namespace, private_directory=private),
    ):
        if outcome == "hit":
            actual = apply_fn(value)
            np.testing.assert_array_equal(actual, value @ value + 5)
            assert warmup._AF3_VERIFICATION.get() == [
                {"cache_key": key, "source": "loaded"}
            ]
        elif outcome == "deserialize_oom":
            with pytest.raises(MemoryError, match="mock CUDA module"):
                apply_fn(value)
        else:
            with pytest.raises(warmup.PredictionError, match="prepared|compilation"):
                apply_fn(value)
    assert not compile_calls
    assert [getattr(compiler, name) for name in owners] == originals
    assert compilation_cache.get_executable_and_time is original_load
    assert warmup._AF3_VERIFICATION.get() is None
    apply_fn.clear_cache()


@pytest.mark.parametrize("private_hit", [False, True])
def test_af3_verification_keeps_shared_deserialization_oom_private_fallback(
    tmp_path, monkeypatch, trust_ancestors_above_tmp_path, private_hit
):
    import jax
    import numpy as np
    from jax._src import compilation_cache

    from foldjax import cache, warmup

    namespace, private = tmp_path / "shared", tmp_path / "private"

    @jax.jit
    def apply_fn(value):
        return value + 19

    value = np.arange(3, dtype=np.float32)
    with cache.compilation_cache_scope(namespace, private_directory=private):
        apply_fn(value).block_until_ready()
    (artifact,) = namespace.glob("jit_apply_fn-*-cache")
    if private_hit:
        copy = private / artifact.name
        copy.write_bytes(artifact.read_bytes())
        copy.chmod(0o600)
    original = compilation_cache.decompress_executable
    attempts = []

    def first_fails(payload):
        attempts.append(1)
        if len(attempts) == 1:
            raise MemoryError("shared cache module allocation failed")
        return original(payload)

    monkeypatch.setattr(compilation_cache, "decompress_executable", first_fails)
    jax.clear_caches()
    cache.clear_record()
    with (
        warmup._verify_af3_artifacts((artifact.name.removesuffix("-cache"),)),
        cache.compilation_cache_scope(namespace, private_directory=private),
    ):
        if private_hit:
            np.testing.assert_array_equal(apply_fn(value), value + 19)
            assert len(attempts) == 2
            assert cache.recorded()["loads"]["private"]["success"] == 1
        else:
            with pytest.raises(MemoryError, match="shared cache module"):
                apply_fn(value)
            assert len(attempts) == 1
    apply_fn.clear_cache()


@pytest.mark.parametrize("model", ["boltz2", "alphafold3"])
def test_warm_cache_executes_first_seed_and_removes_scratch_outputs(
    tmp_path: Path, monkeypatch, model
) -> None:
    from foldjax import warmup
    from foldjax.backends import _tokamax_autotune as tuning

    request = _request(tmp_path, model=model)
    resolved = request
    namespace = tmp_path / "cache" / "boltz2" / "weights" / "digest"
    seen: dict[str, object] = {}
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (resolved,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _model: SimpleNamespace(name=model)
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: namespace)
    monkeypatch.setattr(
        warmup,
        "runtime_dir",
        lambda model=None: tmp_path / "runtime" / (model or ""),
    )

    def fake_predict(run, *, output_root=None):
        if model == "alphafold3":
            tuning.start_record(strategy="autotune", persistent_installed=True)
        seen["seed"] = run.seed
        seen["seeds"] = run.seeds
        seen["output"] = run.output_dir
        seen["root"] = output_root
        run.output_dir.mkdir(parents=True, exist_ok=True)
        structure = run.output_dir / "sample.cif"
        structure.write_text("data_test\n")
        namespace.mkdir(parents=True, exist_ok=True)
        (namespace / "xla-entry").write_bytes(b"compiled")
        return PredictionResult(
            model=model,
            samples=(PredictionSample(seed=run.seed, structure_path=structure),),
            output_dir=run.output_dir,
        )

    monkeypatch.setattr(warmup, "_predict_resolved", fake_predict)
    monkeypatch.setattr(warmup, "device_peak_bytes", lambda: 123456)

    result = warm_cache(request)

    assert isinstance(result, CacheWarmResult)
    assert seen["seed"] == 7
    assert seen["seeds"] is None
    assert result.strategy == "execute_once"
    assert result.status == "populated"
    assert result.before == CacheSnapshot()
    assert result.after == CacheSnapshot(files=1, bytes=8)
    assert result.output_dir is None
    assert result.peak_device_bytes == 123456
    assert not Path(seen["output"]).exists()
    assert result.summary()["samples_executed"] == 1
    if model == "alphafold3":
        assert result.summary()["kernel_tuning"] == tuning.recorded()
    else:
        assert "kernel_tuning" not in result.summary()
    tuning.clear_record()


def test_warm_cache_keeps_an_explicit_output_directory(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax import warmup

    output = tmp_path / "kept"
    request = _request(tmp_path, output_dir=output, seeds=None, seed=0)
    namespace = tmp_path / "namespace"
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (request,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _model: SimpleNamespace(name="boltz2")
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: namespace)

    def fake_predict(run, *, output_root=None):
        assert output_root is None
        run.output_dir.mkdir(parents=True)
        structure = run.output_dir / "sample.cif"
        structure.write_text("data_test\n")
        return PredictionResult(
            model="boltz2",
            samples=(PredictionSample(seed=run.seed, structure_path=structure),),
            output_dir=run.output_dir,
        )

    monkeypatch.setattr(warmup, "_predict_resolved", fake_predict)

    result = warm_cache(request)

    assert isinstance(result, CacheWarmResult)
    assert result.output_dir == output
    assert (output / "sample.cif").is_file()


@pytest.mark.parametrize("tier", ["shared", "private"])
@pytest.mark.parametrize("private_path", ["directory", "entry", "neither"])
@pytest.mark.parametrize("trusted_group", [False, True])
def test_warm_warns_about_unreadable_shared_publication_only(
    tmp_path, monkeypatch, tier, private_path, trusted_group
):
    from foldjax import cache, warmup

    request = _request(tmp_path, seeds=None, seed=0)
    namespace = tmp_path / "cache" / "namespace"
    private = tmp_path / "private"
    monkeypatch.setenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, str(int(trusted_group)))
    if tier == "private":
        namespace.mkdir(parents=True, mode=0o700)
        namespace.chmod(0o700)
        old_entry = namespace / "old-cache"
        old_entry.write_bytes(b"existing shared entry")
        old_entry.chmod(0o600)
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (request,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _model: SimpleNamespace(name="boltz2")
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: namespace)
    monkeypatch.setattr(warmup, "runtime_dir", lambda *_args: tmp_path / "runtime")
    monkeypatch.setattr(
        cache,
        "recorded",
        lambda: {
            "locations": {
                "shared": str(namespace) if tier == "shared" else None,
                "private": str(private),
            },
            "writes": {tier: {"success": 1}},
        },
    )

    def predict(run, *, output_root=None):
        destination = namespace if tier == "shared" else private
        destination.mkdir(parents=True, mode=0o755)
        directory_mode = 0o2770 if trusted_group else 0o755
        entry_mode = 0o660 if trusted_group else 0o644
        if tier == "private" or private_path == "directory":
            directory_mode = 0o700
        if tier == "private" or private_path == "entry":
            entry_mode = 0o600
        destination.chmod(directory_mode)
        entry = destination / "key-cache"
        entry.write_bytes(b"compiled")
        entry.chmod(entry_mode)
        return PredictionResult(
            model="boltz2",
            samples=(PredictionSample(seed=run.seed, coordinates=[[0, 0, 0]]),),
            output_dir=run.output_dir,
        )

    monkeypatch.setattr(warmup, "_predict_resolved", predict)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = warm_cache(request)
    notices = [
        item for item in caught if "warmed shared compile cache" in str(item.message)
    ]
    assert len(notices) == (1 if tier == "shared" and private_path != "neither" else 0)
    if notices:
        message = str(notices[0].message)
        if trusted_group:
            assert "g+rx" in message and "g+r" in message
            assert "0755" not in message and "0644" not in message
        else:
            assert "0755" in message and "0644" in message
    assert result.compilation_cache["writes"][tier]["success"] == 1


def test_warm_cache_rejects_disabled_persistent_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use_compile_cache=False"):
        warm_cache(_request(tmp_path, use_compile_cache=False))


@pytest.mark.parametrize("keep_output", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_warm_uses_the_prediction_session_and_always_closes_it(
    tmp_path, monkeypatch, keep_output, fail
):
    from foldjax import warmup

    request = _request(
        tmp_path,
        model="esmfold2",
        output_dir=tmp_path / "kept" if keep_output else None,
    )
    events, outputs = [], []
    active = object()

    @contextmanager
    def session(requests):
        assert len(requests) == 1
        assert requests[0].seed == 7
        assert requests[0].seeds is None
        assert requests[0].resume is False
        events.append("enter")
        try:
            yield active
        finally:
            events.append("close")

    backend = SimpleNamespace(name="esmfold2", session_reuse=True, session=session)
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (request,))
    monkeypatch.setattr(warmup, "get_backend", lambda _model: backend)
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: tmp_path / "cache")
    monkeypatch.setattr(warmup, "runtime_dir", lambda *_args: tmp_path / "runtime")

    def predict(run, *, backend, output_root=None):
        assert backend is active
        assert events == ["enter"]
        events.append("predict")
        outputs.append(run.output_dir)
        if fail:
            raise RuntimeError("fixture failure")
        run.output_dir.mkdir(parents=True, exist_ok=True)
        structure = run.output_dir / "sample.cif"
        structure.write_text("data_fixture\n")
        return PredictionResult(
            model="esmfold2",
            samples=(PredictionSample(seed=run.seed, structure_path=structure),),
            output_dir=run.output_dir,
        )

    monkeypatch.setattr(warmup, "_predict_resolved", predict)
    if fail:
        with pytest.raises(RuntimeError, match="fixture failure"):
            warm_cache(request)
    else:
        result = warm_cache(request)
        assert result.samples == 1
    assert events == ["enter", "predict", "close"]
    if not keep_output:
        assert not outputs[0].exists()


def test_zero_delta_does_not_claim_an_unobservable_cache_hit(tmp_path: Path) -> None:
    snapshot = CacheSnapshot(files=3, bytes=400)
    result = CacheWarmResult(
        model="boltz2",
        input=tmp_path / "job.yaml",
        weights=tmp_path / "weights.jax",
        cache_dir=tmp_path / "cache",
        seed=0,
        seconds=1.0,
        before=snapshot,
        after=snapshot,
        samples=1,
    )

    assert result.status == "no_new_entry_observed"


@pytest.mark.parametrize("tier", ["shared", "private"])
@pytest.mark.parametrize("kind", ["compilation_cache", "kernel_tuning"])
@pytest.mark.parametrize("activity", ["write", "read"])
def test_warm_status_includes_observed_private_activity(
    tmp_path: Path, tier: str, kind: str, activity: str
) -> None:
    telemetry = (
        {"writes" if activity == "write" else "loads": {tier: {"success": 1}}}
        if kind == "compilation_cache"
        else {"calls": [{f"cache_{activity}_tiers": {tier: 1}}]}
    )
    result = CacheWarmResult(
        model="alphafold3",
        input=tmp_path / "job.json",
        weights=tmp_path / "weights.jax",
        cache_dir=tmp_path / "shared",
        seed=0,
        seconds=1.0,
        before=CacheSnapshot(),
        after=CacheSnapshot(),
        samples=1,
        **{kind: telemetry},
    )

    assert result.status == (
        "populated" if activity == "write" else "no_new_entry_observed"
    )
    assert result.summary()[kind] == telemetry


def test_warm_recognizes_in_memory_tokamax_configs_with_a_known_store_origin(
    tmp_path: Path,
) -> None:
    result = CacheWarmResult(
        model="alphafold3",
        input=tmp_path / "job.json",
        weights=tmp_path / "weights.jax",
        cache_dir=tmp_path / "shared",
        seed=0,
        seconds=1.0,
        before=CacheSnapshot(),
        after=CacheSnapshot(),
        samples=1,
        kernel_tuning={"calls": [{"cache_reuse_tiers": {"private": 1}}]},
    )
    assert result.status == "no_new_entry_observed"


def test_byte_lookup_alone_does_not_claim_a_loaded_executable(tmp_path: Path) -> None:
    result = CacheWarmResult(
        model="boltz2",
        input=tmp_path / "job.json",
        weights=tmp_path / "weights.jax",
        cache_dir=tmp_path / "shared",
        seed=0,
        seconds=1.0,
        before=CacheSnapshot(),
        after=CacheSnapshot(),
        samples=1,
        compilation_cache={"lookups": {"private": {"hit": 1}}},
    )
    assert result.status == "executed_no_persistent_entry_observed"


def test_plural_warm_preserves_the_generated_output_symlink_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax import warmup

    root = tmp_path / "out"
    generated_parent = root / "boltz2"
    generated_parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    generated = generated_parent / "job"
    generated.symlink_to(outside, target_is_directory=True)

    scalar = _request(tmp_path, output_dir=generated, seeds=None, seed=0)
    plural = PredictionRequest(
        models=("boltz2",),
        input=scalar.input,
        weights=scalar.weights,
        output_dir=root,
        cache_dir=scalar.cache_dir,
        input_format="native",
    )
    namespace = tmp_path / "namespace"
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (scalar,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _model: SimpleNamespace(name="boltz2")
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: namespace)

    with pytest.raises(PredictionOutputError, match="symlink"):
        warm_cache(plural)

    assert not list(outside.iterdir())
