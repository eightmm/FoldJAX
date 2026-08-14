from __future__ import annotations

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


def test_warm_cache_executes_first_seed_and_removes_scratch_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax import warmup

    request = _request(tmp_path)
    resolved = request
    namespace = tmp_path / "cache" / "boltz2" / "weights" / "digest"
    seen: dict[str, object] = {}
    monkeypatch.setattr(warmup, "resolve_requests", lambda _request: (resolved,))
    monkeypatch.setattr(
        warmup, "get_backend", lambda _model: SimpleNamespace(name="boltz2")
    )
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_args: namespace)
    monkeypatch.setattr(
        warmup,
        "runtime_dir",
        lambda model=None: tmp_path / "runtime" / (model or ""),
    )

    def fake_predict(run, *, output_root=None):
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
            model="boltz2",
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


def test_warm_cache_keeps_an_explicit_output_directory(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax import warmup

    output = tmp_path / "kept"
    request = _request(tmp_path, output_dir=output, seeds=None)
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


def test_warm_cache_rejects_disabled_persistent_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="use_compile_cache=False"):
        warm_cache(_request(tmp_path, use_compile_cache=False))


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

    scalar = _request(tmp_path, output_dir=generated, seeds=None)
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
