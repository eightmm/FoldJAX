"""Paired, real-weight ESMFold2 performance evidence; never automatic admission.

Run this through the site's GPU queue, with exactly one visible GPU. Two common
jobs are featurized once without network searches. Each experiment/case/arm has
its own process, so allocator peaks and executable owners cannot leak between
arms. The native compact structure core receives five samples and a shared
explicit diffusion tape; every other released sampling setting remains intact.

Examples::

    python -m bench.esmfold2_performance_probe --input small.json large.yaml \
        --experiment alignment --output /scratch/esm-alignment
    python -m bench.esmfold2_performance_probe --case L250_3dha L500_4rek \
        --experiment esmc --output /scratch/esm-esmc

``--smoke`` explicitly reduces recycles/steps and is excluded from admission.
This measures the common feature/compact-LM/model-core boundary, not writers or
independent publisher preprocessing. GPU memory is the actual selected device's
process-lifetime allocator peak, including both LM staging and structure work;
this is not a measurement of a 24-GB card unless that card actually ran it.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.esmfold2_tape import _model_features, _npz, _save_npz, _save_outputs, _sha256

SAMPLES = 5
ARMS = ("baseline", "candidate", "baseline_repeat")
EXPERIMENTS = ("alignment", "esmc")


def arm_options(experiment: str, arm: str) -> dict[str, object]:
    if experiment not in EXPERIMENTS or arm not in ARMS:
        raise ValueError("unknown experiment or arm")
    return {
        "rigid_align_backend": "jacobi"
        if experiment == "alignment" and arm == "candidate"
        else "svd",
        "compile_esmc_blocks": experiment == "esmc" and arm == "candidate",
    }


def diffusion_tape(key, atom_mask, settings):
    """Replay the native sampler's actual key splits and normal draw shapes."""
    import jax
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models import diffusion

    if settings.num_samples != SAMPLES:
        raise ValueError("the paired probe requires exactly five samples")
    sample_key = jax.random.split(key, 5)[-1]
    key, initial_key = jax.random.split(sample_key)
    mask = jnp.repeat(jnp.asarray(atom_mask), SAMPLES, axis=0)
    batch, atoms = mask.shape
    rotations, translations, churn = [], [], []
    steps = len(diffusion.noise_schedule(settings.diffusion)) - 1
    for _ in range(steps):
        key, augmentation_key, churn_key = jax.random.split(key, 3)
        rotation_key, translation_key = jax.random.split(augmentation_key)
        rotations.append(jax.random.normal(rotation_key, (batch, 4), jnp.float32))
        translations.append(
            jax.random.normal(translation_key, (batch, 1, 3), jnp.float32)
        )
        churn.append(
            diffusion._atom_normal(
                churn_key, mask, dtype=jnp.float32, preserve_prefix_rng=False
            )
        )
    tape = {
        "diffusion_initial_normal": diffusion._atom_normal(
            initial_key, mask, dtype=jnp.float32, preserve_prefix_rng=False
        ),
        "diffusion_rotation_quaternions": jnp.stack(rotations),
        "diffusion_translations": jnp.stack(translations),
        "diffusion_churn_normals": jnp.stack(churn),
    }
    diffusion.validate_diffusion_tape(
        *tape.values(), steps=steps, batch=batch, atoms=atoms
    )
    return tape


def _settings(native, *, smoke: bool, alignment: str):
    from foldjax.models.esmfold2.models import model

    effective = model.with_overrides(native, num_samples=SAMPLES)
    if smoke:
        effective = model.with_overrides(effective, num_recycles=0, num_steps=2)
    return replace(
        effective,
        diffusion=replace(effective.diffusion, rigid_align_backend=alignment),
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _local_anchor(path: Path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _verify_binding(binding, weights: Path, source_root: Path):
    for filename, identity in binding["weights"].items():
        if _local_anchor(weights / filename) != identity["local_change_anchor"]:
            raise ValueError(f"checkpoint changed during the paired probe: {filename}")
    for filename, digest in binding["source"].items():
        if _sha256(source_root / filename) != digest:
            raise ValueError(f"source changed during the paired probe: {filename}")
    if _sha256(Path(__file__)) != binding["probe_sha256"]:
        raise ValueError("probe source changed during the paired experiment")


def _memory(device) -> dict[str, object]:
    stats = device.memory_stats() or {}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "device": {str(key): int(value) for key, value in stats.items()},
        "host_process_peak_rss_bytes": int(
            rss * (1 if sys.platform == "darwin" else 1024)
        ),
        "scope": (
            "fresh worker process lifetime; includes LM, compile, warm and structure"
        ),
    }


def _timed(function):
    import jax

    started = time.perf_counter()
    result = function()
    if hasattr(result, "parameters"):
        jax.block_until_ready((result.parameters, result.esmc_parameters))
    else:
        jax.block_until_ready(result)
    return result, time.perf_counter() - started


def _host_arrays(values):
    return {
        name: np.asarray(value, dtype=np.float32)
        if str(value.dtype) == "bfloat16"
        else np.asarray(value)
        for name, value in values.items()
    }


def _benchmark_runtime():
    from foldjax import oom

    oom.set_mem_fraction(oom.PREDICT_MEM_FRACTION)
    import jax

    gpus = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(gpus) != 1:
        raise RuntimeError(
            "probe requires exactly one visible GPU; submit it through the GPU queue"
        )
    jax.config.update("jax_enable_compilation_cache", False)
    return jax, gpus[0]


def _tape_location(args):
    return args.features.parent / (
        "diffusion-smoke.npz" if args.smoke else "diffusion.npz"
    )


def _tape_identity(args, settings, binding):
    return {
        "features_sha256": binding["features_sha256"],
        "seed": args.seed,
        "samples": SAMPLES,
        "diffusion": {
            k: v
            for k, v in asdict(settings.diffusion).items()
            if k != "rigid_align_backend"
        },
    }


def _prepare_tape(args):
    jax, device = _benchmark_runtime()
    from foldjax.models.esmfold2 import inference

    features = _npz(args.features)
    binding = json.loads(args.binding.read_text())
    if _sha256(args.features) != binding["features_sha256"]:
        raise ValueError("features changed before tape preparation")
    source_root = Path(inference.__file__).parents[3]
    _verify_binding(binding, args.weights, source_root)
    settings = _settings(
        inference.structure_checkpoint.load_settings(args.weights),
        smoke=args.smoke,
        alignment="svd",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    path = _tape_location(args)
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("tape preparation requires a fresh case directory")
    with jax.default_device(device):
        tape = diffusion_tape(
            inference.seed_key(args.seed), features["atom_attention_mask"], settings
        )
        _save_npz(path, _host_arrays(tape))
    _verify_binding(binding, args.weights, source_root)
    metadata = {
        "identity": _tape_identity(args, settings, binding),
        "sha256": _sha256(path),
        "device_kind": device.device_kind,
        "scope": "unmeasured fresh preparation process; all measured arms load only",
    }
    _write_json(path.with_suffix(".json"), metadata)
    _write_json(args.output / "report.json", metadata)


def _worker(args) -> None:
    jax, device = _benchmark_runtime()

    from foldjax.models._chain_profile import chain_capacity
    from foldjax.models.esmfold2 import inference
    from foldjax.models.esmfold2.models import atom, esmc, model

    options = arm_options(args.experiment, args.arm)
    # These development owners must never populate a managed serving cache.
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    features = _npz(args.features)
    binding = json.loads(args.binding.read_text())
    if _sha256(args.features) != binding["features_sha256"]:
        raise ValueError("features changed after the paired experiment was prepared")
    source_root = Path(inference.__file__).parents[3]
    _verify_binding(binding, args.weights, source_root)
    report = {
        "experiment": args.experiment,
        "arm": args.arm,
        "options": options,
        "smoke_only": args.smoke,
        "binding": binding,
        "seed": args.seed,
        "samples": SAMPLES,
        "device_kind": device.device_kind,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "allocator_environment": {
            name: os.environ.get(name)
            for name in (
                "XLA_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
            )
        },
        "persistent_compilation_cache": False,
        "rng_scope": (
            "shared explicit diffusion tape; other native consumers use "
            "unchanged key/features"
        ),
        "boundary": (
            "shared common all-atom features; staged compact LM; native "
            "structure_model.predict; managed output projection"
        ),
    }

    def phase(name):
        _write_json(destination / "phase.json", {"phase": name, "report": report})

    with jax.default_device(device):
        phase("load_language_model_stage")
        staged, report["lm_weight_load_seconds"] = _timed(
            lambda: inference.load_language_model_stage(args.weights)
        )
        native_settings = staged.settings
        original_encode = esmc.encode

        def selected_encode(*positional, **keywords):
            if keywords.get("deterministic", False):
                raise ValueError(
                    "this experiment preserves nondeterministic released policy"
                )
            keywords["compile_blocks"] = bool(options["compile_esmc_blocks"])
            return original_encode(*positional, **keywords)

        phase("language_model_cold_and_warm")
        with patch.object(esmc, "encode", selected_encode):
            first, report["lm_cold_seconds"] = _timed(
                partial(inference.language_model_embedding, features, staged)
            )
            embedding, report["lm_warm_seconds"] = _timed(
                partial(inference.language_model_embedding, features, staged)
            )
        if embedding is None:
            raise RuntimeError("a real ESM-C compact embedding is required")
        report["lm_cold_warm_max_abs_error"] = float(
            np.max(
                np.abs(
                    np.asarray(first, np.float32) - np.asarray(embedding, np.float32)
                )
            )
        )
        del first
        _save_npz(
            destination / "lm_embedding.npz",
            {"embedding": np.asarray(embedding, np.float32)},
        )
        report["memory_after_lm"] = _memory(device)
        inference.release_language_model_parameters(staged, after=embedding)
        del staged
        gc.collect()
        phase("load_structure_only")
        loaded, report["structure_weight_load_seconds"] = _timed(
            lambda: inference.load(args.weights, language_model=False)
        )
        if loaded.settings != native_settings:
            raise ValueError(
                "LM stage and structure stage loaded different configuration"
            )
        settings = _settings(
            native_settings,
            smoke=args.smoke,
            alignment=str(options["rigid_align_backend"]),
        )
        report["native_settings"] = asdict(native_settings)
        report["effective_settings"] = asdict(settings)
        report["msa_depth"] = (
            int(features["msa"].shape[1]) if "msa" in features else None
        )
        key = inference.seed_key(args.seed)
        tape_path = _tape_location(args)
        tape_metadata = tape_path.with_suffix(".json")
        metadata = json.loads(tape_metadata.read_text())
        if metadata["identity"] != _tape_identity(args, settings, binding):
            raise ValueError("the paired diffusion tape configuration differs")
        if metadata["sha256"] != _sha256(tape_path):
            raise ValueError("the paired diffusion tape contents differ")
        if metadata["device_kind"] != device.device_kind:
            raise ValueError("tape and model were prepared on different device kinds")
        tape = {name: jax.device_put(value) for name, value in _npz(tape_path).items()}
        model.diffusion.validate_diffusion_tape(
            *(
                tape[name]
                for name in (
                    "diffusion_initial_normal",
                    "diffusion_rotation_quaternions",
                    "diffusion_translations",
                    "diffusion_churn_normals",
                )
            ),
            steps=len(model.diffusion.noise_schedule(settings.diffusion)) - 1,
            batch=features["atom_attention_mask"].shape[0] * SAMPLES,
            atoms=features["atom_attention_mask"].shape[-1],
        )
        report["diffusion_tape_sha256"] = _sha256(tape_path)
        compact = inference._has_compact_token_bond_encoding(
            features,
            loaded.parameters,
            pair_width=settings.d_pair,
            compute_dtype=settings.trunk_dtype,
        )
        bound_features = inference._model_bound_features(
            features, compact_token_bond_encoding=compact
        )
        arrays = {
            name: jax.device_put(value)
            for name, value in _model_features(bound_features).items()
        }
        capacity = chain_capacity(int(np.asarray(features["asym_id"]).max()) + 1)
        grouped = inference._has_contiguous_atom_groups(features)

        def forward(prediction_key, inputs, parameters, lm_embedding, draws):
            return model.predict(
                prediction_key,
                inputs,
                parameters,
                settings=settings,
                lm_embedding=lm_embedding,
                n_chains=capacity,
                contiguous_atom_groups=grouped,
                compact_token_bond_encoding=compact,
                return_distogram_logits=False,
                return_auxiliary_outputs=False,
                **draws,
            )

        phase("lower_and_compile_structure")
        # Each worker owns a new Python function and a fresh JIT. No arm can
        # reuse a graph traced while another alignment implementation was live.
        with atom._atom_rows_per_block_scope(atom._resolve_rows_per_block(None)):
            started = time.perf_counter()
            lowered = jax.jit(forward).lower(
                key, arrays, loaded.parameters, embedding, tape
            )
            report["structure_lower_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            executable = lowered.compile()
            report["structure_compile_seconds"] = time.perf_counter() - started
        analysis = executable.memory_analysis()
        report["structure_executable_memory"] = {
            name: getattr(analysis, name)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
        phase("structure_first_and_warm")
        first, report["structure_first_seconds"] = _timed(
            lambda: executable(key, arrays, loaded.parameters, embedding, tape)
        )
        del first
        output, report["structure_warm_seconds"] = _timed(
            lambda: executable(key, arrays, loaded.parameters, embedding, tape)
        )
        host = _host_arrays(output)
        report["output_schema"] = _save_outputs(destination, "jax", host, native=False)
        report["memory"] = _memory(device)
        _verify_binding(binding, args.weights, source_root)
        report["status"] = "completed"
        _write_json(destination / "report.json", report)
        phase("completed")


def _comparison(case_root: Path, experiment: str, features):
    from bench.boltz_amp_report import compare_arrays
    from bench.esmfold2_tape_report import compare_coordinates

    roots = {arm: case_root / experiment / arm for arm in ARMS}
    metadata = {
        arm: json.loads((root / "report.json").read_text())
        for arm, root in roots.items()
    }
    tapes = {document["diffusion_tape_sha256"] for document in metadata.values()}
    if len(tapes) != 1:
        raise ValueError("arms did not consume exactly the same diffusion tape")
    coords = {
        arm: _npz(root / "jax_coords.npz")["coords"] for arm, root in roots.items()
    }
    scores = {arm: _npz(root / "jax_confidence.npz") for arm, root in roots.items()}
    lm = {
        arm: _npz(root / "lm_embedding.npz")["embedding"] for arm, root in roots.items()
    }
    between = compare_coordinates(coords["baseline"], coords["candidate"], features)
    within = compare_coordinates(
        coords["baseline"], coords["baseline_repeat"], features
    )
    return {
        "candidate_vs_baseline": between,
        "baseline_repeat_floor": within,
        "coordinate_max_abs_error": float(
            np.max(np.abs(coords["baseline"] - coords["candidate"]))
        ),
        "candidate_scores": compare_arrays(scores["baseline"], scores["candidate"]),
        "baseline_repeat_scores": compare_arrays(
            scores["baseline"], scores["baseline_repeat"]
        ),
        "lm_candidate_max_abs_error": float(
            np.max(np.abs(lm["baseline"] - lm["candidate"]))
        ),
        "lm_baseline_repeat_max_abs_error": float(
            np.max(np.abs(lm["baseline"] - lm["baseline_repeat"]))
        ),
        "measurements": metadata,
        "full_model_admission": None,
        "scope": (
            "fixed 0.05-A per-entity coordinate diagnostic plus observed rerun "
            "floor; confidence and speed remain separate evidence"
        ),
    }


def _coordinator(args) -> None:
    from foldjax.input import read_job_document
    from foldjax.models.esmfold2 import inference

    weights = args.weights.resolve()
    if weights.is_file():
        weights = weights.parent
    if args.case:
        from bench.spec import cases

        available = {entry.name: entry.job for entry in cases()}
        jobs = [available[name] for name in args.case]
    else:
        jobs = args.input
    if len(jobs) != 2:
        raise ValueError("two distinct input sizes are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    experiments = EXPERIMENTS if args.experiment == "both" else (args.experiment,)
    esmc_root = inference.esmc_directory(weights)
    checkpoint_paths = [
        weights / "config.json",
        weights / "model.safetensors",
        esmc_root / "config.json",
        weights / "ccd.pkl",
        *inference.esmc_checkpoint.shard_paths(esmc_root),
    ]
    index = esmc_root / inference.esmc_checkpoint.INDEX_NAME
    if index.exists():
        checkpoint_paths.append(index)
    weights_binding = {}
    for path in checkpoint_paths:
        anchor = _local_anchor(path)
        digest = _sha256(path)
        if anchor != _local_anchor(path):
            raise ValueError("checkpoint changed while its content was hashed")
        weights_binding[str(path.relative_to(weights))] = {
            "sha256": digest,
            "local_change_anchor": anchor,
        }
    source = {
        str(
            Path(module.__file__).relative_to(Path(inference.__file__).parents[3])
        ): _sha256(Path(module.__file__))
        for module in (
            inference,
            inference.structure_model,
            inference.structure_model.diffusion,
            inference.esmc_model,
        )
    }
    prepared = []
    for index, job in enumerate(jobs):
        job = job.resolve()
        directory = output / f"case-{index + 1}"
        directory.mkdir()
        features = inference.build_common_job_features(
            read_job_document(job),
            base_dir=job.parent,
            ccd_path=weights / "ccd.pkl",
            seed=args.seed,
        )
        path = directory / "features.npz"
        _save_npz(path, features)
        binding = {
            "job_sha256": _sha256(job),
            "features_sha256": _sha256(path),
            "weights": weights_binding,
            "source": source,
            "probe_sha256": _sha256(Path(__file__)),
            "tokens": int(np.asarray(features["token_attention_mask"]).sum()),
            "atoms": int(np.asarray(features["atom_attention_mask"]).sum()),
            "feature_boundary": (
                "same common all-atom featurizer output, no search or "
                "independent publisher preprocessing"
            ),
        }
        _write_json(directory / "binding.json", binding)
        prepared.append((directory, features, binding))
    if prepared[0][2]["tokens"] == prepared[1][2]["tokens"]:
        raise ValueError("two different token sizes are required")
    _write_json(
        output / "plan.json",
        {
            "experiments": experiments,
            "arms": ARMS,
            "samples": SAMPLES,
            "smoke_only": args.smoke,
            "coordinate_diagnostic_angstrom": 0.05,
            "full_model_admission": None,
            "cases": [item[2] for item in prepared],
        },
    )
    summaries = []
    for directory, features, _ in prepared:
        command = [
            sys.executable,
            "-m",
            "bench.esmfold2_performance_probe",
            "--prepare-tape",
            "--weights",
            str(weights),
            "--features",
            str(directory / "features.npz"),
            "--binding",
            str(directory / "binding.json"),
            "--output",
            str(directory / "tape-prepare"),
            "--seed",
            str(args.seed),
        ]
        if args.smoke:
            command.append("--smoke")
        print(f"{directory.name}: prepare shared native tape", flush=True)
        with (directory / "tape-prepare.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        for experiment in experiments:
            for arm in ARMS:
                destination = directory / experiment / arm
                command = [
                    sys.executable,
                    "-m",
                    "bench.esmfold2_performance_probe",
                    "--worker",
                    "--weights",
                    str(weights),
                    "--features",
                    str(directory / "features.npz"),
                    "--binding",
                    str(directory / "binding.json"),
                    "--output",
                    str(destination),
                    "--experiment",
                    experiment,
                    "--arm",
                    arm,
                    "--seed",
                    str(args.seed),
                ]
                if args.smoke:
                    command.append("--smoke")
                destination.parent.mkdir(parents=True, exist_ok=True)
                print(f"{directory.name}: {experiment}/{arm}", flush=True)
                with (destination.parent / f"{arm}.log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            summary = _comparison(directory, experiment, features)
            _write_json(directory / experiment / "comparison.json", summary)
            summaries.append(
                {"case": directory.name, "experiment": experiment, **summary}
            )
    _write_json(
        output / "report.json",
        {
            "smoke_only": args.smoke,
            "comparisons": summaries,
            "full_model_admission": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--input", type=Path, nargs=2)
    selection.add_argument("--case", nargs=2)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "both"), default="both")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-tape", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--features", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--binding", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.weights is None:
        from foldjax.paths import weights_dir

        args.weights = weights_dir("esmfold2")
    if args.prepare_tape:
        if not args.features or not args.binding or args.worker:
            parser.error("tape preparation requires features and binding")
        _prepare_tape(args)
    elif args.worker:
        if (
            not args.arm
            or not args.features
            or not args.binding
            or args.experiment == "both"
        ):
            parser.error("worker requires one experiment, arm, features and binding")
        _worker(args)
    else:
        if not args.input and not args.case:
            parser.error("provide two --input jobs or two benchmark --case names")
        _coordinator(args)


if __name__ == "__main__":
    main()
