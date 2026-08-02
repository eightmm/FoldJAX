"""AlphaFold 3 in-process adapter for the cloned upstream runner."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)
from foldjax.scores import scalar_scores


def _runner_path(options: dict) -> Path:
    explicit = options.pop("source", None) or os.environ.get("ALPHAFOLD3_SOURCE")
    if explicit:
        path = Path(explicit) / "run_alphafold.py"
    else:
        spec = importlib.util.find_spec("alphafold3")
        if spec is None or spec.origin is None:
            raise ImportError("alphafold3 is not installed")
        path = Path(spec.origin).resolve().parents[2] / "run_alphafold.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"AlphaFold 3 runner not found: {path}; set ALPHAFOLD3_SOURCE"
        )
    return path


def _load_runner(path: Path):
    name = "_foldjax_alphafold3_runner"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AlphaFold 3 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _settle_absl_flags()
    return module


def _tokamax_kernel_fallback(strategy: str):
    """Let Tokamax measure its kernel configuration instead of guessing it.

    AlphaFold 3's transition blocks call ``tokamax.gated_linear_unit``, which
    prefers a Triton kernel and, on a cache miss, sizes it from heuristics.
    Those heuristics are not right for every device -- on an RTX PRO 6000
    Blackwell they ask for 108 KiB of shared memory against the 99 KiB it has,
    and the launch fails with ``RESOURCE_EXHAUSTED`` from inside the diffusion
    transformer. It fails *after* the kernel has compiled, so Tokamax's own
    per-implementation fallback, which only catches ``NotImplementedError``,
    never sees it.

    ``autotune`` is the mechanism Tokamax provides for exactly this: it
    benchmarks the candidate configurations on whatever device is in front of it
    and keeps one that fits, so the same command works across a fleet of
    different cards. It costs time on the first compile of each shape and is
    cached afterwards. Pass ``kernel_autotuning=heuristics`` for upstream's
    default, or ``error`` to make a cache miss loud.
    """
    from tokamax import config as tokamax_config

    return tokamax_config.autotuning_cache_miss_fallback(strategy)


def _settle_absl_flags() -> None:
    """Give absl its defaults instead of letting it parse FoldJAX's argv.

    ``run_alphafold.py`` is an absl program and defines its flags at import, and
    Tokamax reads its own flags lazily -- the first access does
    ``flags.FLAGS(sys.argv)`` if nothing has parsed yet. Reached from the
    FoldJAX CLI that argv is ``foldjax predict --model ...``, which absl rejects
    with "Unknown command line flag 'model'" from somewhere deep inside the
    diffusion transformer. FoldJAX's own CLI is the interface here, so the flags
    are marked parsed and keep their defaults; everything this backend cares
    about is passed to ``ModelRunner`` explicitly.
    """
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


#: Where the two knobs `make_model_config` does not take live in the config it
#: returns. Named so a version that moves them fails with the path it looked
#: for rather than silently leaving the released default in place.
_DIFFUSION_STEPS = ("heads", "diffusion", "eval", "steps")
_MSA_DEPTH = ("evoformer", "num_msa")


def _set_nested(config: Any, path: tuple[str, ...], value: Any) -> None:
    """Assign `value` at `path`, or explain which attribute is missing."""
    if value is None:
        return
    target = config
    for attribute in path[:-1]:
        target = getattr(target, attribute, None)
        if target is None:
            raise ValueError(
                f"this AlphaFold 3 build has no {'.'.join(path)} to set; "
                "the option cannot be honoured and is not being ignored"
            )
    if not hasattr(target, path[-1]):
        raise ValueError(
            f"this AlphaFold 3 build has no {'.'.join(path)} to set; "
            "the option cannot be honoured and is not being ignored"
        )
    setattr(target, path[-1], int(value))


class AlphaFold3Backend(Backend):
    name = "alphafold3"
    # `make_model_config` takes only four of these as parameters, but it returns
    # a plain mutable config and already sets `num_samples` by assignment. The
    # step count and the MSA depth live one level deeper -- at
    # `heads.diffusion.eval.steps` (200) and `evoformer.num_msa` (1024) -- and
    # are set the same way. They were previously reported as unsupported, which
    # made AlphaFold 3 the one backend that could not be held to the same
    # schedule as the others, and so could not be benchmarked against them.
    sampling_options: dict[str, str] = {
        "num_samples": "diffusion_samples",
        "num_steps": "diffusion_steps",
        "num_recycles": "recycles",
        "max_msa_depth": "max_msa_depth",
    }
    compile_options = (
        "diffusion_samples",
        "diffusion_steps",
        "recycles",
        "max_msa_depth",
        "buckets",
        "attention_backend",
        "return_embeddings",
        "return_distogram",
        "kernel_autotuning",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "alphafold3", "foldjax"),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        runner = _load_runner(_runner_path(options))
        import jax
        from alphafold3.common import folding_input

        if request.cache_dir is not None:
            request.cache_dir.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(request.cache_dir))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        # Selecting from the default backend keeps a CPU-only host usable for
        # smoke runs while still resolving the GPU on an accelerator host.
        platform = options.pop("platform", None)
        devices = (
            jax.local_devices(backend=platform) if platform else jax.local_devices()
        )
        device = devices[int(options.pop("device", 0))]
        config = runner.make_model_config(
            flash_attention_implementation=options.pop("attention_backend", "triton"),
            num_diffusion_samples=int(options.pop("diffusion_samples", 5)),
            num_recycles=int(options.pop("recycles", 10)),
            return_embeddings=bool(options.pop("return_embeddings", False)),
            return_distogram=bool(options.pop("return_distogram", False)),
        )
        _set_nested(config, _DIFFUSION_STEPS, options.pop("diffusion_steps", None))
        _set_nested(config, _MSA_DEPTH, options.pop("max_msa_depth", None))
        model_runner = runner.ModelRunner(
            config=config,
            device=device,
            model_dir=request.weights,
        )
        buckets = tuple(options.pop("buckets", ())) or None
        kernel_fallback = str(options.pop("kernel_autotuning", "autotune"))
        if options:
            raise ValueError(f"unsupported AlphaFold 3 options: {', '.join(options)}")
        all_results = []
        samples: list[PredictionSample] = []
        with _tokamax_kernel_fallback(kernel_fallback):
            for fold_input in folding_input.load_fold_inputs_from_path(request.input):
                fold_input = dataclasses.replace(fold_input, rng_seeds=(request.seed,))
                results = runner.predict_structure(
                    fold_input, model_runner, buckets=buckets
                )
                job_name = fold_input.sanitised_name()
                job_dir = request.output_dir / job_name
                runner.write_outputs(results, job_dir, job_name)
                all_results.extend(results)
                samples.extend(_samples(results, job_dir, job_name))
        return PredictionResult(
            model=self.name,
            samples=tuple(samples),
            output_dir=request.output_dir,
            raw=tuple(all_results),
        )


def _samples(results: Any, job_dir: Path, job_name: str) -> list[PredictionSample]:
    """Normalize AlphaFold 3 ``ResultsForSeed`` values into common samples.

    The sample layout is reconstructed from the runner's own naming rather than
    globbed, so every structure keeps the seed and ranking score that produced
    it instead of an arbitrary directory order.
    """
    samples = []
    for results_for_seed in results:
        seed = int(getattr(results_for_seed, "seed", 0))
        for index, result in enumerate(
            getattr(results_for_seed, "inference_results", ())
        ):
            prefix = f"{job_name}_seed-{seed}_sample-{index}"
            sample_dir = job_dir / f"seed-{seed}_sample-{index}"
            structure_path = sample_dir / f"{prefix}_model.cif"
            scores = scalar_scores(sample_dir / f"{prefix}_summary_confidences.json")
            metadata = getattr(result, "metadata", None)
            if isinstance(metadata, dict) and "ranking_score" in metadata:
                scores["ranking_score"] = float(metadata["ranking_score"])
            samples.append(
                PredictionSample(
                    seed=seed,
                    structure_path=(
                        structure_path if structure_path.is_file() else None
                    ),
                    scores=scores,
                    metadata={"sample": index},
                )
            )
    return samples
