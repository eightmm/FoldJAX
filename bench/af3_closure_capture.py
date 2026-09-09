"""AF3 publisher/default-dtype closure arm; actual draws observed at execution."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import resource
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

BUCKETS = (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def config_record(config):
    """Record full-run defaults absent from the publisher config schema.

    Keep explicit port values: an early stop or extra representations must
    still fail strict comparison with the native full-output run.
    Compare only after JSON serialization normalizes tuples to arrays.
    """
    record = dict(config.as_dict())
    record.setdefault("foldjax_stop_after", "full")
    record.setdefault("foldjax_return_representations", [])
    return record


def save_config(path, config):
    """Retain the raw schema beside the comparable execution record."""
    raw = config.as_dict()
    record = config_record(config)
    save(path.with_name(path.stem + "-recording.json"), {
        "raw_config": raw,
        "synthesized_fields": sorted(record.keys() - raw.keys()),
        "recording_policy": "af3-full-run-defaults-v1",
    })
    save(path, record)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, prefix + str(key) + "."))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for i, child in enumerate(value):
            result.update(flatten(child, prefix + str(i) + "."))
        return result
    array = np.asarray(value)
    if array.dtype == object and all(isinstance(x, str) for x in array.flat):
        array = array.astype(str)
    if array.dtype == object:
        raise TypeError(f"unhandled result {prefix}: {type(value)}")
    return {prefix.rstrip("."): array}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--kernel-manifest", type=Path)
    parser.add_argument("--xla-autotune-load", type=Path)
    parser.add_argument("--xla-autotune-extend", action="store_true")
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--mode", choices=("audit", "performance"), default="audit")
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--no-preprocessing-observers", action="store_true")
    args = parser.parse_args()
    if args.no_preprocessing_observers and args.mode != "performance":
        parser.error("disabling preprocessing observers requires performance mode")
    if args.warm_repeats < 1:
        parser.error("--warm-repeats must be positive")
    if args.xla_autotune_extend and (
        args.arm != "native" or args.xla_autotune_load is None
    ):
        parser.error("XLA extension requires a native producer and a parent cache")
    out = args.out.resolve()
    upstream = args.native_source.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(out / "jax-cache")
    # Share kernel decisions, never compiled executables. JAX otherwise gives
    # each arm its own per-fusion autotune cache and can override these controls.
    os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if any(term in xla_flags for term in ("autotune_results", "autotune_cache")):
        raise ValueError(
            "inherited XLA autotuning flags conflict with capture controls"
        )
    xla_flags += f" --xla_gpu_dump_autotune_results_to={out / 'xla-autotune.textproto'}"
    if args.xla_autotune_load is not None:
        args.xla_autotune_load = args.xla_autotune_load.resolve(strict=True)
        xla_flags += f" --xla_gpu_load_autotune_results_from={args.xla_autotune_load}"
        if not args.xla_autotune_extend:
            xla_flags += " --xla_gpu_require_complete_aot_autotune_results=true"
    os.environ["XLA_FLAGS"] = xla_flags.strip()
    source = args.input.resolve()
    from foldjax.models.alphafold3.build import active_package

    runtime = active_package()
    if args.arm == "native":
        sys.path.insert(0, str(upstream / "src"))
        import alphafold3

        assert Path(alphafold3.__file__).resolve().is_relative_to(upstream / "src")
        os.environ["LIBCIFPP_DATA_DIR"] = str(runtime.parent / "share/libcifpp")
        cpp = next(runtime.glob("cpp.*.so"))
        spec = importlib.util.spec_from_file_location("alphafold3.cpp", cpp)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["alphafold3.cpp"] = module
        alphafold3.cpp = module
        from alphafold3.common import resources

        resources.ROOT = runtime
        resources._DATA_ROOT = runtime
    else:
        from foldjax.models.alphafold3._upstream import ensure_registered

        ensure_registered()
    import alphafold3
    import jax
    from alphafold3.common import folding_input
    from alphafold3.model.network import featurization

    from bench.af3_feature_capture import metadata_value
    from foldjax.backends import alphafold3 as backend

    runner_path = (
        upstream / "run_alphafold.py"
        if args.arm == "native"
        else backend.VENDORED_RUNNER
    )
    runner = backend._load_runner(runner_path)
    jobs = list(folding_input.load_fold_inputs_from_path(source))
    if len(jobs) != 1 or len(jobs[0].rng_seeds) != 1:
        raise ValueError("capture requires exactly one input job and one seed")
    job = jobs[0]
    weights = args.weights.resolve()
    config = runner.make_model_config(
        num_diffusion_samples=5,
        num_recycles=10,
        flash_attention_implementation="triton",
    )
    save_config(out / "config.json", config)
    assert config.global_config.bfloat16 == "all"
    kernel_overlay = None
    kernel_provenance = {}
    if args.kernel_manifest is not None:
        import tokamax

        manifest = json.loads(args.kernel_manifest.read_text())
        encoded = json.dumps(
            manifest["result"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        if (
            manifest["format"] != "foldjax-tokamax-autotuning"
            or manifest["schema"] != 2
            or hashlib.sha256(encoded).hexdigest() != manifest["result_sha256"]
        ):
            raise ValueError("invalid fixed-kernel manifest")
        kernel_overlay = tokamax.AutotuningResult.loads(encoded.decode("ascii"))
        if kernel_overlay.device_kind != jax.devices()[0].device_kind:
            raise ValueError("fixed-kernel manifest device mismatch")
        kernel_provenance = {
            "fixed_kernel_manifest_sha256": sha(args.kernel_manifest),
            "fixed_kernel_result_sha256": manifest["result_sha256"],
            "fixed_kernel_entries": len(kernel_overlay.data),
            "fixed_kernel_control_only": True,
            "kernel_selection": "frozen_foldjax_autotune",
            "kernel_deviation": (
                "Both arms reuse the same recorded kernel configurations; "
                "uncovered configurations fail instead of retuning"
            ),
        }
    save(
        out / "provenance.json",
        {
            "arm": args.arm,
            "mode": args.mode,
            "input_sha256": sha(source),
            "weights_sha256": sha(weights / "af3.bin"),
            "wrapper_sha256": sha(Path(__file__)),
            "runner_sha256": sha(runner_path),
            "cpp_sha256": sha(next(runtime.glob("cpp.*.so"))),
            "shared_cpp_and_generated_ccd": True,
            "buckets": BUCKETS,
            "jax_version": jax.__version__,
            "runtime_versions": {
                name: importlib.metadata.version(name)
                for name in ("jaxlib", "numpy", "rdkit", "dm-haiku", "tokamax")
            },
            "matmul_precision": jax.config.jax_default_matmul_precision,
            "xla_flags": os.environ["XLA_FLAGS"],
            "xla_autotune_load_sha256": (
                sha(args.xla_autotune_load) if args.xla_autotune_load else None
            ),
            "xla_autotune_extend": args.xla_autotune_extend,
            "separate_executable_cache": True,
            "attention": "triton",
            "kernel_selection": "autotune",
            "neutral_padding": False,
            "kernel_deviation": (
                "Both arms use autotune for Blackwell shared-memory compatibility "
                "instead of native heuristic fallback"
            ),
            "source_files": {
                str(p.relative_to(Path(alphafold3.__file__).parent)): sha(p)
                for p in sorted(Path(alphafold3.__file__).parent.rglob("*.py"))
            },
            "shared_generated_assets": {
                str(p.relative_to(runtime)): sha(p)
                for p in sorted(runtime.rglob("*.pickle"))
            },
            "native_recycles": 10,
            "steps": 200,
            "samples": 5,
            **kernel_provenance,
        },
    )
    draws = Counter()
    preprocessing_draws = []
    conformers = []
    seen = []
    timings = {}

    def observe(label, key, value):
        key, value = np.asarray(key), np.asarray(value)
        signature = json.dumps(
            [
                label,
                key.tolist(),
                str(value.dtype),
                list(value.shape),
                hashlib.sha256(value.tobytes()).hexdigest(),
            ]
        )
        draws[signature] += 1

    def instrument(name, fn):
        def call(key, *pos, **kwargs):
            value = fn(key, *pos, **kwargs)
            label = name
            if name == "normal":
                if not inspect.currentframe().f_back.f_code.co_filename.endswith(
                    "network/diffusion_head.py"
                ):
                    return value
                label = {(2, 3): "rotation", (3,): "translation"}.get(value.shape)
                if label is None:
                    label = "initial" if value.ndim == 4 else "churn"
            # GPU XLA fusion rejects effect tokens in this scan. Event keys and
            # content hashes identify draws without requiring callback order.
            jax.debug.callback(
                lambda k, v: observe(label, k, v),
                jax.random.key_data(key),
                value,
                ordered=False,
            )
            return value

        return call

    original_grid = featurization._padding_consistent_rng

    def grid(fn):
        return instrument("padding_" + fn.__name__, original_grid(fn))

    original_infer = runner.ModelRunner.run_inference

    def infer(self, features, key):
        assert not seen
        seen.append(True)
        assert "__foldjax_prefix_stable_diffusion_noise" not in features
        arrays, meta = {}, {}
        for name, value in features.items():
            if isinstance(value, np.ndarray) and value.dtype != object:
                arrays[name] = value
            else:
                meta[name] = metadata_value(value)
        np.savez_compressed(out / "input.npz", **arrays)
        save(out / "input-metadata.json", meta)
        save(
            out / "preprocessing-tape.json",
            {
                "numpy_draws": preprocessing_draws,
                "rdkit_seed_and_conformer_boundary": conformers,
                "rdkit_internal_rng_observed": False,
                "observers_enabled": not args.no_preprocessing_observers,
            },
        )
        save_config(out / "effective-config.json", self._model_config)
        save(
            out / "parameters.json",
            {
                module + "/" + name: [
                    str(array.dtype),
                    list(array.shape),
                    hashlib.sha256(array.tobytes()).hexdigest(),
                ]
                for module, values in self.model_params.items()
                for name, value in values.items()
                for array in [np.asarray(value)]
            },
        )
        if args.arm == "foldjax":
            from bench.entity_parity import compare_feature_dicts

            if args.reference is None:
                raise ValueError(
                    "FoldJAX capture requires --reference native capture directory"
                )
            reference = args.reference
            with np.load(reference / "input.npz") as archive:
                audit = compare_feature_dicts(dict(archive), arrays)
            save(out / "input-audit.json", audit)
            assert not any(
                audit[k]
                for k in (
                    "missing_from_left",
                    "missing_from_right",
                    "shape_mismatches",
                    "dtype_mismatches",
                    "value_mismatches",
                )
            ), audit
            assert meta == json.loads((reference / "input-metadata.json").read_text())
        with ExitStack() as stack:
            if args.mode == "audit":
                stack.enter_context(
                    patch.object(
                        jax.random, "normal", instrument("normal", jax.random.normal)
                    )
                )
                stack.enter_context(
                    patch.object(featurization, "_padding_consistent_rng", grid)
                )
            start = time.perf_counter()
            result = original_infer(self, features, key)
            jax.effects_barrier()
            timings["first_inference_seconds"] = time.perf_counter() - start
            if args.mode == "performance":
                first = flatten(result)
                np.savez_compressed(out / "raw-first.npz", **first)
                repeats, equality = [], []
                for _ in range(args.warm_repeats):
                    start = time.perf_counter()
                    result = original_infer(self, features, key)
                    jax.effects_barrier()
                    repeats.append(time.perf_counter() - start)
                    current = flatten(result)
                    equality.append(
                        first.keys() == current.keys()
                        and all(
                            first[name].shape == value.shape
                            and first[name].dtype == value.dtype
                            and first[name].tobytes() == value.tobytes()
                            for name, value in current.items()
                        )
                    )
                timings["warm_inference_seconds"] = float(np.median(repeats))
                timings["warm_inference_repeats_seconds"] = repeats
                timings["warm_raw_bitwise_equal_to_first"] = equality
        np.savez_compressed(out / "raw.npz", **flatten(result))
        save(out / "inference-finished.json", timings)
        save(out / "tape.json", dict(sorted(draws.items())))
        if args.mode == "audit":
            counts = Counter()
            for signature, count in draws.items():
                counts[json.loads(signature)[0]] += count
            save(out / "tape-coverage.json", dict(counts))
            assert counts == {
                "initial": 1,
                "churn": 1000,
                "rotation": 1000,
                "translation": 1000,
                "padding_gumbel": 11,
            }, counts
        return result

    original_extract = runner.ModelRunner.extract_inference_results

    def extract(self, *pos, **kwargs):
        from alphafold3.model import feat_batch
        from alphafold3.model.atom_layout import atom_layout

        batch = feat_batch.Batch.from_data_dict(
            kwargs.get("batch", pos[0] if pos else None)
        )
        gather = atom_layout.compute_gather_idxs(
            source_layout=batch.convert_model_output.token_atoms_layout,
            target_layout=batch.convert_model_output.flat_output_layout,
        )
        valid = np.asarray(gather.gather_mask, dtype=bool)
        results = original_extract(self, *pos, **kwargs)
        assert len(results) == 5
        coords, confidence = [], {}
        for i, result in enumerate(results):
            st = result.predicted_structure
            coords.append(np.stack((st.atom_x, st.atom_y, st.atom_z), axis=-1))
            assert len(valid) == len(st.atom_x)
            identity = {
                name: np.asarray(
                    getattr(st, name), dtype=int if name == "res_id" else str
                )
                for name in (
                    "chain_id",
                    "chain_type",
                    "res_id",
                    "res_name",
                    "atom_name",
                    "atom_element",
                )
            }
            if i == 0:
                np.savez_compressed(out / "identity.npz", **identity)
                first_identity = identity
            else:
                assert all(
                    np.array_equal(value, first_identity[name])
                    for name, value in identity.items()
                )
            confidence.update(
                flatten(
                    {
                        "numerical": dict(result.numerical_data),
                        "metadata": dict(result.metadata),
                        "atom_plddt": st.atom_b_factor,
                    },
                    f"{i}.",
                )
            )
        np.savez_compressed(
            out / "coordinate.npz",
            coordinate=np.stack(coords),
            mask=np.broadcast_to(valid, (5, len(valid))),
        )
        np.savez_compressed(out / "confidence.npz", **confidence)
        return results

    original_random_state = np.random.RandomState

    class ObservedRandomState(original_random_state):
        def draw(self, name, *pos, **kwargs):
            state = self.get_state()
            before = [
                state[0],
                hashlib.sha256(state[1].tobytes()).hexdigest(),
                *state[2:],
            ]
            value = getattr(super(), name)(*pos, **kwargs)
            array = np.asarray(value)
            preprocessing_draws.append(
                [
                    name,
                    before,
                    str(array.dtype),
                    list(array.shape),
                    hashlib.sha256(array.tobytes()).hexdigest(),
                ]
            )
            return value

        def normal(self, *pos, **kwargs):
            return self.draw("normal", *pos, **kwargs)

        def randint(self, *pos, **kwargs):
            return self.draw("randint", *pos, **kwargs)

    from alphafold3.data.tools import rdkit_utils

    original_conformer = rdkit_utils.get_random_conformer

    def conformer(*pos, **kwargs):
        value = original_conformer(*pos, **kwargs)
        coordinates = (
            np.empty((0, 3), dtype=np.float64)
            if value is None
            else value.GetPositions()
        )
        conformers.append(
            [
                kwargs["logging_name"],
                kwargs["random_seed"],
                kwargs["max_iterations"],
                list(coordinates.shape),
                hashlib.sha256(coordinates.tobytes()).hexdigest(),
            ]
        )
        return value

    with (
        patch.object(runner.ModelRunner, "run_inference", infer),
        patch.object(runner.ModelRunner, "extract_inference_results", extract),
        ExitStack() as preprocessing_stack,
        ExitStack() as kernel_stack,
    ):
        if not args.no_preprocessing_observers:
            preprocessing_stack.enter_context(
                patch.object(np.random, "RandomState", ObservedRandomState)
            )
            preprocessing_stack.enter_context(
                patch.object(rdkit_utils, "get_random_conformer", conformer)
            )
        if kernel_overlay is not None:
            kernel_stack.enter_context(kernel_overlay)
        if args.arm == "native":
            mr = runner.ModelRunner(
                config=config, device=jax.devices()[0], model_dir=weights
            )
            with ExitStack() as stack:
                stack.enter_context(
                    backend._tokamax_kernel_fallback(
                        "error" if kernel_overlay is not None else "autotune"
                    )
                )
                result = runner.predict_structure(job, mr, buckets=BUCKETS)
            runner.write_outputs(result, out / "predictions", job.name)
        else:
            from foldjax import PredictionRequest, predict

            predict(
                PredictionRequest(
                    model="alphafold3",
                    input=source,
                    weights=weights,
                    output_dir=out / "predictions",
                    cache_dir=out / "foldjax-cache",
                    seed=job.rng_seeds[0],
                    options={
                        "buckets": list(BUCKETS),
                        "kernel_autotuning": (
                            "error" if kernel_overlay is not None else "autotune"
                        ),
                        "attention_backend": "triton",
                        "num_samples": 5,
                        "num_recycles": 10,
                    },
                )
            )
    assert len(seen) == 1
    recorded_source = json.loads((out / "provenance.json").read_text())["source_files"]
    current_source = {
        str(p.relative_to(Path(alphafold3.__file__).parent)): sha(p)
        for p in sorted(Path(alphafold3.__file__).parent.rglob("*.py"))
    }
    if current_source != recorded_source:
        raise RuntimeError("AF3 Python source changed during capture")
    provenance = json.loads((out / "provenance.json").read_text())
    provenance["xla_autotune_sha256"] = sha(
        args.xla_autotune_load
        if args.xla_autotune_load and not args.xla_autotune_extend
        else out / "xla-autotune.textproto"
    )
    save(out / "provenance.json", provenance)
    if not args.no_preprocessing_observers:
        assert preprocessing_draws and conformers
    save(
        out / "preprocessing-tape.json",
        {
            "numpy_draws": preprocessing_draws,
            "rdkit_seed_and_conformer_boundary": conformers,
            "rdkit_internal_rng_observed": False,
            "observers_enabled": not args.no_preprocessing_observers,
        },
    )
    timings["device_memory_stats"] = jax.devices()[0].memory_stats()
    timings["host_process_maxrss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    timings["instrumented"] = args.mode == "audit"
    timings["preprocessing_observers_enabled"] = not args.no_preprocessing_observers
    save(out / "finished.json", timings)
    print(source.stem, args.arm, "completed", timings, flush=True)


if __name__ == "__main__":
    main()
