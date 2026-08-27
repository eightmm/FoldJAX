"""Request-scoped AlphaFold 3 runner reuse without native dependencies."""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
import weakref
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import foldjax
from foldjax.backends import alphafold3 as af3_backend
from foldjax.backends.alphafold3 import AlphaFold3Backend
from foldjax.models.alphafold3 import build as af3_build
from foldjax.registry import backend_override
from foldjax.schema import PredictionError, PredictionRequest


@dataclasses.dataclass(frozen=True)
class _FoldInput:
    name: str
    rng_seeds: tuple[int, ...] = (0,)
    shape: int = 8
    user_ccd: str | None = None

    def sanitised_name(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class _Device:
    id: int
    platform: str = "cpu"
    process_index: int = 0
    local_hardware_id: int = 0
    device_kind: str = "mock-cpu"


class _MockRuntime:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts: Counter[str] = Counter()
        self.fail_seeds: set[int] = set()
        self.mutate_on_seed: dict[int, Any] = {}
        self.ranking_score = 0.75
        self.runner_refs: list[weakref.ReferenceType[Any]] = []
        self.runner_file = tmp_path / "managed-runner.py"
        self.runner_file.write_text("# immutable managed runner\n")
        self.source_dir = tmp_path / "managed-source"
        self.source_dir.mkdir()
        (self.source_dir / "model.py").write_text("# immutable managed source\n")
        monkeypatch.setattr(af3_backend, "VENDORED_RUNNER", self.runner_file)
        monkeypatch.setattr(af3_build, "source_package", lambda: self.source_dir)

        runtime = self

        class ModelRunner:
            def __init__(self, *, config: Any, device: Any, model_dir: Path) -> None:
                runtime.counts["model_runners"] += 1
                runtime.runner_refs.append(weakref.ref(self))
                self.config = config
                self.device = device
                self.model_dir = model_dir
                self._loaded = False

                def forward(value):
                    runtime.counts["traces"] += 1
                    return value + 1

                self._compiled = jax.jit(forward)

            def run_inference(self, batch: dict[str, int], _key: Any) -> dict:
                runtime.counts["inferences"] += 1
                if not self._loaded:
                    runtime.counts["parameter_loads"] += 1
                    self._loaded = True
                shape = int(batch["shape"])
                compiled = self._compiled(jnp.zeros((shape,), dtype=jnp.float32))
                compiled.block_until_ready()
                return {"shape": shape}

        def make_model_config(**kwargs: Any) -> Any:
            runtime.counts["configs"] += 1
            return SimpleNamespace(
                supplied=kwargs,
                heads=SimpleNamespace(
                    diffusion=SimpleNamespace(eval=SimpleNamespace(steps=200))
                ),
                evoformer=SimpleNamespace(num_msa=1024),
            )

        def predict_structure(
            fold_input: _FoldInput,
            model_runner: ModelRunner,
            *,
            buckets: tuple[int, ...] | None,
        ) -> tuple[Any, ...]:
            del buckets
            seed = fold_input.rng_seeds[0]
            model_runner.run_inference({"shape": fold_input.shape}, None)
            callback = runtime.mutate_on_seed.get(seed)
            if callback is not None:
                callback()
            if seed in runtime.fail_seeds:
                raise ValueError(f"seed {seed} failed")
            return (
                SimpleNamespace(
                    seed=seed,
                    inference_results=(
                        SimpleNamespace(
                            metadata={"ranking_score": runtime.ranking_score}
                        ),
                    ),
                ),
            )

        def write_outputs(results: Any, output_dir: Path, job_name: str) -> None:
            for result in results:
                sample_dir = Path(output_dir) / f"seed-{result.seed}_sample-0"
                sample_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{job_name}_seed-{result.seed}_sample-0"
                (sample_dir / f"{prefix}_model.cif").write_text("data_prediction\n#\n")
                (sample_dir / f"{prefix}_summary_confidences.json").write_text(
                    json.dumps({"ptm": 0.5})
                )

        self.runner = ModuleType("_foldjax_test_alphafold3_runner")
        self.runner.__file__ = str(self.runner_file)
        self.runner.ModelRunner = ModelRunner
        self.runner.make_model_config = make_model_config
        self.runner.predict_structure = predict_structure
        self.runner.write_outputs = write_outputs

        def runner_path(options: dict[str, Any]) -> Path:
            runtime.counts["runner_paths"] += 1
            options.pop("source", None)
            return runtime.runner_file

        def load_runner(_path: Path) -> ModuleType:
            runtime.counts["runner_module_lookups"] += 1
            return runtime.runner

        monkeypatch.setattr(af3_backend, "_runner_path", runner_path)
        monkeypatch.setattr(af3_backend, "_load_runner", load_runner)
        monkeypatch.setattr(
            af3_backend, "_tokamax_kernel_fallback", lambda _strategy: nullcontext()
        )
        monkeypatch.setattr(af3_backend, "_model_device", lambda _device: nullcontext())
        devices = (
            _Device(id=0, local_hardware_id=0),
            _Device(id=1, local_hardware_id=1),
        )
        monkeypatch.setattr(jax, "local_devices", lambda backend=None: devices)

        folding = ModuleType("alphafold3.common.folding_input")

        def load_fold_inputs(path: Path):
            document = json.loads(Path(path).read_text())
            return iter(
                (
                    _FoldInput(
                        name=str(document.get("name", Path(path).stem)),
                        shape=int(document.get("shape", 8)),
                    ),
                )
            )

        folding.load_fold_inputs_from_path = load_fold_inputs
        common = ModuleType("alphafold3.common")
        common.folding_input = folding
        package = ModuleType("alphafold3")
        package.common = common
        monkeypatch.setitem(sys.modules, "alphafold3", package)
        monkeypatch.setitem(sys.modules, "alphafold3.common", common)
        monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)

    def reset_counts(self) -> None:
        self.counts.clear()
        self.runner_refs.clear()


@pytest.fixture
def mock_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _MockRuntime:
    return _MockRuntime(tmp_path, monkeypatch)


def _weights(root: Path, name: str = "weights") -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "af3.bin").write_bytes(b"mock parameters")
    return directory


def _request(
    root: Path,
    *,
    weights: Path,
    output: str,
    seed: int = 0,
    num_seeds: int = 1,
    shape: int = 8,
    cache_dir: Path | None = None,
    options: dict[str, Any] | None = None,
    resume: bool = False,
    on_error: str = "stop",
) -> PredictionRequest:
    input_path = root / f"job-{shape}.json"
    input_path.write_text(json.dumps({"name": "job", "shape": shape}))
    return PredictionRequest(
        model="alphafold3",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=root / output,
        seed=seed,
        num_seeds=num_seeds,
        cache_dir=cache_dir,
        use_compile_cache=cache_dir is not None,
        options={} if options is None else options,
        resume=resume,
        on_error=on_error,
    )


@pytest.mark.parametrize(
    "names",
    (
        ("af3.bin",),
        ("af3.bin.zst",),
        ("af3.0.bin", "af3.1.bin"),
        ("af3.bin].0", "af3.bin].1"),
    ),
)
def test_session_identity_selects_the_same_parameter_layout_as_upstream(
    tmp_path: Path,
    mock_runtime: _MockRuntime,
    names: tuple[str, ...],
) -> None:
    del mock_runtime
    weights = tmp_path / "parameter-layout"
    weights.mkdir()
    for name in names:
        (weights / name).write_bytes(b"parameters")

    selected = af3_backend._selected_parameter_files(weights)

    assert selected is not None
    assert tuple(path.name for path in selected) == names
    assert af3_backend._managed_asset_snapshot(weights) is not None


def test_managed_session_reuses_runner_params_and_shape_trace(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    backend = AlphaFold3Backend()
    request = _request(
        tmp_path,
        weights=_weights(tmp_path),
        output="out",
        num_seeds=3,
        cache_dir=tmp_path / "cache",
    )

    with backend_override("alphafold3", lambda: backend):
        result = foldjax.predict(request)

    assert [sample.seed for sample in result.samples] == [0, 1, 2]
    assert mock_runtime.counts["model_runners"] == 1
    assert mock_runtime.counts["parameter_loads"] == 1
    assert mock_runtime.counts["traces"] == 1
    assert mock_runtime.counts["inferences"] == 3
    assert backend._model_runner is None
    gc.collect()
    assert all(reference() is None for reference in mock_runtime.runner_refs)


def test_one_runner_compiles_once_per_distinct_shape(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    weights = _weights(tmp_path)
    requests = tuple(
        _request(
            tmp_path,
            weights=weights,
            output=f"out-{index}",
            shape=shape,
        )
        for index, shape in enumerate((8, 16, 8))
    )
    backend = AlphaFold3Backend()

    with backend.session(requests):
        for request in requests:
            backend.predict(request)

    assert mock_runtime.counts["model_runners"] == 1
    assert mock_runtime.counts["parameter_loads"] == 1
    assert mock_runtime.counts["traces"] == 2


def test_explicit_source_keeps_fresh_model_runners(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    backend = AlphaFold3Backend()
    request = _request(
        tmp_path,
        weights=_weights(tmp_path),
        output="external",
        num_seeds=2,
        options={"source": tmp_path / "external-checkout"},
    )

    with backend_override("alphafold3", lambda: backend):
        foldjax.predict(request)

    assert mock_runtime.counts["model_runners"] == 2
    assert mock_runtime.counts["parameter_loads"] == 2
    assert mock_runtime.counts["traces"] == 2


def test_unverifiable_managed_assets_are_run_fresh_and_never_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_runtime: _MockRuntime,
) -> None:
    monkeypatch.setattr(af3_backend, "_managed_asset_snapshot", lambda _path: None)
    backend = AlphaFold3Backend()
    request = _request(
        tmp_path,
        weights=_weights(tmp_path),
        output="unverifiable",
        num_seeds=2,
    )

    with backend_override("alphafold3", lambda: backend):
        foldjax.predict(request)

    assert mock_runtime.counts["model_runners"] == 2
    assert mock_runtime.counts["parameter_loads"] == 2
    assert backend._model_runner is None


def test_unverifiable_resumed_asset_poisons_before_any_runner_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_runtime: _MockRuntime,
) -> None:
    monkeypatch.setattr(af3_backend, "_managed_asset_snapshot", lambda _path: None)
    request = _request(
        tmp_path,
        weights=_weights(tmp_path),
        output="unverifiable-resume",
        num_seeds=2,
        resume=True,
    )
    backend = AlphaFold3Backend()

    with backend.session((request,)):
        backend.validate_session(request)
        with pytest.raises(PredictionError, match="cannot verify weights"):
            backend.observe_resumed(request)

    assert mock_runtime.counts == Counter()


def test_all_resumed_seeds_do_not_load_the_runner_or_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_runtime: _MockRuntime,
) -> None:
    weights = _weights(tmp_path)
    request = _request(
        tmp_path,
        weights=weights,
        output="resumable",
        num_seeds=2,
    )
    with backend_override("alphafold3", AlphaFold3Backend):
        foldjax.predict(request)

    mock_runtime.reset_counts()
    monkeypatch.setattr(
        af3_backend,
        "_runner_path",
        lambda _options: pytest.fail("resume must not resolve or build the runtime"),
    )
    monkeypatch.setattr(
        af3_backend,
        "_load_runner",
        lambda _path: pytest.fail("resume must not load the runner module"),
    )
    backend = AlphaFold3Backend()
    with backend_override("alphafold3", lambda: backend):
        report = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert len(report.skipped) == 2
    assert not report.failures
    assert mock_runtime.counts == Counter()
    assert backend._model_runner is None


@pytest.mark.parametrize("asset", ("runner", "source"))
def test_partial_resume_rejects_a_different_managed_generation(
    tmp_path: Path, mock_runtime: _MockRuntime, asset: str
) -> None:
    weights = _weights(tmp_path)
    request = _request(
        tmp_path,
        weights=weights,
        output="runner-generation",
        num_seeds=2,
    )
    with backend_override("alphafold3", AlphaFold3Backend):
        first = foldjax.predict(request)
    assert [sample.scores["ranking_score"] for sample in first.samples] == [
        0.75,
        0.75,
    ]

    target = (
        mock_runtime.runner_file
        if asset == "runner"
        else mock_runtime.source_dir / "model.py"
    )
    target.write_text(f"# replacement managed {asset}\n")
    mock_runtime.ranking_score = 0.25
    (request.output_dir / "seed_1" / "foldjax_run.json").unlink()
    mock_runtime.reset_counts()

    with backend_override("alphafold3", AlphaFold3Backend):
        report = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert report.skipped == ()
    assert not report.failures
    assert mock_runtime.counts["inferences"] == 2
    assert [sample.scores["ranking_score"] for sample in report.results[0].samples] == [
        0.25,
        0.25,
    ]


def test_continued_failure_invalidates_before_the_next_seed(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    mock_runtime.fail_seeds.add(1)
    backend = AlphaFold3Backend()
    request = _request(
        tmp_path,
        weights=_weights(tmp_path),
        output="continued",
        num_seeds=3,
        on_error="continue",
    )

    with backend_override("alphafold3", lambda: backend):
        report = foldjax.predict_batch(request)

    assert [failure.seed for failure in report.failures] == [1]
    assert len(report.results[0].samples) == 2
    assert mock_runtime.counts["model_runners"] == 2
    assert mock_runtime.counts["parameter_loads"] == 2
    assert mock_runtime.counts["traces"] == 2
    assert mock_runtime.counts["inferences"] == 3
    gc.collect()
    assert all(reference() is None for reference in mock_runtime.runner_refs)


@pytest.mark.parametrize(
    "changed_identity",
    ("weights", "device", "config", "cache", "autotune"),
)
def test_runner_identity_change_drops_the_old_model_before_rebuilding(
    tmp_path: Path,
    mock_runtime: _MockRuntime,
    changed_identity: str,
) -> None:
    first_weights = _weights(tmp_path, "weights-a")
    second_weights = (
        _weights(tmp_path, "weights-b")
        if changed_identity == "weights"
        else first_weights
    )
    first = _request(
        tmp_path,
        weights=first_weights,
        output="identity-a",
        cache_dir=tmp_path / "cache-a",
    )
    options: dict[str, Any] = {}
    second_cache = first.cache_dir
    if changed_identity == "device":
        options["device"] = 1
    elif changed_identity == "config":
        options["num_samples"] = 6
    elif changed_identity == "cache":
        second_cache = tmp_path / "cache-b"
    elif changed_identity == "autotune":
        options["kernel_autotuning"] = "heuristics"
    second = _request(
        tmp_path,
        weights=second_weights,
        output="identity-b",
        cache_dir=second_cache,
        options=options,
    )
    backend = AlphaFold3Backend()

    with backend.session((first, second)):
        backend.predict(first)
        first_reference = mock_runtime.runner_refs[0]
        backend.predict(second)
        gc.collect()
        assert first_reference() is None

    assert mock_runtime.counts["model_runners"] == 2
    assert mock_runtime.counts["parameter_loads"] == 2
    assert mock_runtime.counts["traces"] == 2
    gc.collect()
    assert all(reference() is None for reference in mock_runtime.runner_refs)


@pytest.mark.parametrize("asset", ("weights", "runner", "source"))
def test_asset_mutation_poisons_the_whole_session(
    tmp_path: Path,
    mock_runtime: _MockRuntime,
    asset: str,
) -> None:
    weights = _weights(tmp_path)
    requests = tuple(
        _request(
            tmp_path,
            weights=weights,
            output=f"mutated-{index}",
        )
        for index in range(3)
    )
    backend = AlphaFold3Backend()

    with backend.session(requests):
        backend.predict(requests[0])
        target = {
            "weights": weights / "af3.bin",
            "runner": mock_runtime.runner_file,
            "source": mock_runtime.source_dir / "model.py",
        }[asset]
        target.write_bytes(target.read_bytes() + b"changed")
        with pytest.raises(PredictionError, match="changed while"):
            backend.predict(requests[1])
        with pytest.raises(PredictionError, match="changed while"):
            backend.predict(requests[2])
        assert backend._model_runner is None

    assert mock_runtime.counts["model_runners"] == 1
    assert mock_runtime.counts["inferences"] == 1


def test_mutation_during_lazy_inference_is_not_retained(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    weights = _weights(tmp_path)
    requests = tuple(
        _request(
            tmp_path,
            weights=weights,
            output=f"during-{index}",
            seed=index,
        )
        for index in range(2)
    )
    mock_runtime.mutate_on_seed[0] = lambda: (weights / "af3.bin").write_bytes(
        b"replacement parameters"
    )
    backend = AlphaFold3Backend()

    with backend.session(requests):
        with pytest.raises(PredictionError, match="changed while"):
            backend.predict(requests[0])
        assert backend._model_runner is None
        with pytest.raises(PredictionError, match="changed while"):
            backend.predict(requests[1])

    assert mock_runtime.counts["model_runners"] == 1
    assert mock_runtime.counts["parameter_loads"] == 1
    assert mock_runtime.counts["traces"] == 1


def test_asset_recheck_does_not_mask_keyboard_interrupt(
    tmp_path: Path, mock_runtime: _MockRuntime
) -> None:
    weights = _weights(tmp_path)
    requests = tuple(
        _request(
            tmp_path,
            weights=weights,
            output=f"cancelled-{index}",
            seed=index,
        )
        for index in range(2)
    )

    def cancel_after_mutation() -> None:
        (weights / "af3.bin").write_bytes(b"replacement parameters")
        raise KeyboardInterrupt

    mock_runtime.mutate_on_seed[0] = cancel_after_mutation
    backend = AlphaFold3Backend()

    with backend.session(requests):
        with pytest.raises(KeyboardInterrupt):
            backend.predict(requests[0])
        assert backend._model_runner is None
