"""Separate native ESMC variation, state transport, and compact projection.

Reuse a paired probe's features.npz and binding.json, without searches or
structure-model inference. Submit through the site's GPU queue. Native arms
run five fresh processes per storage mode by default, with no layer observer.
Separate opt-in capture arms add
host reads, barriers, file writes and consumers: they change the schedule and
cannot establish equivalence of the native executions.

Example::

    python -m bench.esmc_state_diagnostics --features case-1/features.npz \
        --binding case-1/binding.json --weights /weights/esmfold2 \
        --packed-length 1026 --output /scratch/esmc-diagnostic

No result from this diagnostic automatically admits a numerical default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.esmfold2_performance_probe import (
    _benchmark_runtime,
    _control_policy,
    _local_anchor,
    _memory,
    _select_encode,
    _timed,
    _verify_binding,
    _write_json,
)
from bench.esmfold2_tape import _npz, _sha256

NATIVE_ARMS = ("stack", "buffer")
CAPTURE_ARMS = ("capture-stack", "capture-buffer")
_CONTROL_TYPES = {
    "binding": (dict,),
    "packed_inputs": (dict,),
    "actual_esmc_parameters": (dict,),
    "actual_projection_parameters": (dict,),
    "esmc_settings": (dict,),
    "compute_dtype": (str,),
    "deterministic": (bool,),
    "control_policy": (dict,),
    "device_kind": (str,),
    "jax": (str,),
    "jaxlib": (str,),
    "matmul_precision": (str,),
    "xla_flags": (str,),
    "allocator_environment": (dict,),
    "execution_environment": (dict,),
    "packed_length": (int, type(None)),
    "lm_mask_pct": (int, float),
    "diagnostic_sha256": (str,),
    "compiler_observer_enabled": (bool,),
    "compiler_observer_source_sha256": (str, type(None)),
}


def _host(value):
    array = np.asarray(value)
    return array.astype(np.float32) if str(array.dtype) == "bfloat16" else array


def _array_identity(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _description(value):
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sharding": repr(getattr(value, "sharding", None)),
        "format": repr(getattr(value, "format", None)),
        "layout": repr(getattr(value, "layout", None)),
    }


def _differences(left, right):
    left, right = _host(left), _host(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "same_signature": False,
            "left": _array_identity(left),
            "right": _array_identity(right),
        }
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "same_signature": True,
        "bitwise_equal": bool(
            np.array_equal(left.view(np.uint8), right.view(np.uint8))
        ),
        "max_abs_error": float(np.max(np.abs(delta), initial=0)),
        "rmse": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        "different_values": int(np.count_nonzero(left != right)),
        "total_values": int(left.size),
        "different_fraction": float(np.mean(left != right)) if left.size else 0.0,
    }


def _first_divergence(left, right):
    if len(left) != len(right):
        return {
            "reason": "different layer counts",
            "left": len(left),
            "right": len(right),
        }
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a["identity"] != b["identity"]:
            return {"layer_index": index, "left": a, "right": b}
    return None


def _pair_controls(left, right):
    mismatched, unverified = [], []
    for name, allowed in _CONTROL_TYPES.items():
        values = [document.get(name) for document in (left, right)]
        if any(
            name not in document
            or type(value) not in allowed
            or (
                name == "compiler_observer_source_sha256"
                and document.get("compiler_observer_enabled") is True
                and value is None
            )
            or (
                name == "control_policy"
                and (
                    any(
                        type(value.get(field)) is not expected
                        for field, expected in (
                            ("deterministic_requested", bool),
                            ("deterministic_scope", str),
                            ("deterministic_compiler_options", dict),
                            ("compile_esmc_blocks", bool),
                            ("per_executable_deterministic", bool),
                        )
                    )
                    or value["deterministic_requested"] != document.get("deterministic")
                )
            )
            for document, value in zip((left, right), values, strict=True)
        ):
            unverified.append(name)
        elif json.dumps(values[0], sort_keys=True) != json.dumps(
            values[1], sort_keys=True
        ):
            mismatched.append(name)
    return {
        "controls_match": None if unverified else not mismatched,
        "mismatched_control_fields": mismatched,
        "unverified_control_fields": unverified,
    }


def _parameter_identity(parameters):
    """Actual loaded device values, separate from checkpoint-file identity."""
    records = {
        name: _array_identity(value) for name, value in sorted(parameters.items())
    }
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "leaves": records}


def _captured_layers(directory, records):
    import jax.numpy as jnp

    for record in records:
        saved = directory / record["file"]
        if _sha256(saved) != record["file_sha256"]:
            raise ValueError("captured layer file changed")
        yield jnp.asarray(
            np.load(saved, allow_pickle=False), jnp.dtype(record["dtype"])
        )


def _projection_identity(tokens, loaded, *, deterministic):
    from foldjax.executable_cache_verify import (
        UnsupportedCacheVerificationError,
        lowered_cache_key,
    )
    from foldjax.models.esmfold2 import inference

    parameters = inference._language_model_embedding_parameters(loaded.parameters)
    signature = (
        tuple(tokens.shape),
        str(tokens.dtype),
        tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in parameters.items()
        ),
    )
    owner = inference._compiled_language_model_embedding(
        str(loaded.settings.trunk_dtype),
        signature,
        *((True,) if deterministic else ()),
    )
    lowered = owner.lower(tokens, parameters)
    result = {"lowering_sha256": hashlib.sha256(lowered.as_text().encode()).hexdigest()}
    try:
        result["jax_cache_key"] = lowered_cache_key(lowered)
    except UnsupportedCacheVerificationError as error:
        result["unsupported_cache_key"] = str(error)
    return result


def _capture(
    ids,
    sequences,
    expand_map,
    parameters,
    settings,
    directory,
    *,
    storage,
    deterministic,
):
    import jax.numpy as jnp

    from foldjax.models.esmfold2.models import esmc

    dtype = esmc._hidden_state_storage_dtype(ids, sequences, parameters, "", settings)
    buffer = esmc._TokenStateStorage(expand_map, settings.n_layers, dtype)
    retained, records = [], []

    def sink(index, state):
        if storage == "stack":
            retained.append(state)
        else:
            buffer(index, state)
        record = {
            "index": index,
            **_description(state),
            "identity": _array_identity(state),
        }
        filename = f"layer-{index:03d}.npy"
        np.save(directory / filename, _host(state), allow_pickle=False)
        record.update(file=filename, file_sha256=_sha256(directory / filename))
        records.append(record)

    esmc.encode(
        ids,
        sequences,
        parameters,
        settings=settings,
        deterministic=deterministic,
        state_sink=sink,
    )
    tokens = (
        esmc.hidden_states_for_tokens(
            jnp.stack(retained), expand_map, expand_map.shape[1]
        )
        if storage == "stack"
        else buffer.destination
    )
    return tokens, records


def _replay(directory, records, expand_map, loaded, *, deterministic, barrier):
    import jax
    import jax.numpy as jnp

    from foldjax.models.esmfold2 import inference
    from foldjax.models.esmfold2.models import esmc

    result, reference, embedding_reference = {}, None, None
    # ESMC parameters are released before replay; these are transport and
    # projection observations, never a native peak-memory measurement.
    for mode in ("stack", "donated", "non-donated"):
        if mode == "stack":
            tokens = esmc.hidden_states_for_tokens(
                jnp.stack(list(_captured_layers(directory, records))),
                expand_map,
                expand_map.shape[1],
            )
        else:
            dtype = jnp.result_type(*(jnp.dtype(row["dtype"]) for row in records))
            buffer = esmc._TokenStateStorage(expand_map, len(records) - 1, dtype)
            body = esmc._store_token_state.__wrapped__
            context = (
                patch.object(esmc, "_store_token_state", jax.jit(body))
                if mode == "non-donated"
                else nullcontext()
            )
            with context:
                for index, state in enumerate(_captured_layers(directory, records)):
                    buffer(index, state)
                    if barrier:
                        buffer.destination.block_until_ready()
            tokens = buffer.destination
        jax.block_until_ready(tokens)
        host = _host(tokens)
        report = {
            "token_states": _description(tokens),
            "identity": _array_identity(tokens),
        }
        report["transport_vs_stack"] = _differences(
            reference if reference is not None else host, host
        )
        if reference is None:
            reference = host.copy()
        report["projection_program"] = _projection_identity(
            tokens,
            loaded,
            deterministic=deterministic,
        )
        embedding = inference._language_model_embedding_from_states(
            tokens,
            loaded,
            deterministic=deterministic,
        )
        jax.block_until_ready(embedding)
        compact = _host(embedding)
        report["embedding"] = _description(embedding)
        report["projection_vs_stack"] = _differences(
            embedding_reference if embedding_reference is not None else compact,
            compact,
        )
        if embedding_reference is None:
            embedding_reference = compact.copy()
        np.save(directory / f"replay-{mode}-embedding.npy", compact, allow_pickle=False)
        result[mode] = report
        del tokens, embedding, host, compact
        gc.collect()
    return result


def _worker(args):
    jax, device = _benchmark_runtime(deterministic=args.deterministic)
    from foldjax.models.esmfold2 import inference

    features, binding = _npz(args.features), json.loads(args.binding.read_text())
    if _sha256(args.features) != binding["features_sha256"]:
        raise ValueError("feature payload does not match the binding")
    source_root = Path(inference.__file__).parents[3]
    _verify_binding(binding, args.weights, source_root)
    args.output.mkdir(parents=True, exist_ok=False)
    storage = "buffer" if "buffer" in args.arm else "stack"
    report = {
        "arm": args.arm,
        "replicate": args.replicate,
        "storage": storage,
        "binding": binding,
        "diagnostic_sha256": _sha256(Path(__file__)),
        "device_kind": device.device_kind,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
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
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_LAUNCH_BLOCKING",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
        "compiler_observer_enabled": args.compiler_fingerprints,
        "compiler_observer_source_sha256": None,
        "deterministic": args.deterministic,
        "control_policy": _control_policy(args),
        "packed_length": args.packed_length,
        "scope": "LM only; no structure inference, admission or capacity claim",
        "full_model_admission": None,
        "deterministic_control_scope": (
            "process-wide deterministic XLA policy; eager/compiled LM blocks "
            "are controlled independently by --compile-esmc-blocks"
        ),
    }

    def phase(name):
        _write_json(args.output / "phase.json", {"phase": name, "report": report})

    observation = nullcontext()
    if args.compiler_fingerprints:
        from bench import compilation_profile
        from bench.compilation_profile import observe_compilation

        report["compiler_observer_source_sha256"] = _sha256(
            Path(compilation_profile.__file__)
        )
        report["compilation_observation"] = {}
        report["compiler_observer_interference"] = (
            "opt-in compiler hashes add overhead and can alter scheduling; "
            "fingerprints alone do not establish the cause of GPU variation"
        )
        observation = observe_compilation(
            report["compilation_observation"], fingerprint_hlo=True
        )
    selected_encode = _select_encode(
        inference.esmc_model.encode,
        compile_blocks=bool(getattr(args, "compile_esmc_blocks", False)),
    )
    with (
        jax.default_device(device),
        observation,
        patch.object(inference.esmc_model, "encode", selected_encode),
    ):
        phase("load")
        loaded, report["load_seconds"] = _timed(
            lambda: inference.load_language_model_stage(
                args.weights, esmc_dtype=args.esmc_dtype
            )
        )
        if args.compute_dtype is not None:
            loaded = replace(
                loaded,
                settings=replace(loaded.settings, trunk_dtype=args.compute_dtype),
            )
        report["esmc_settings"] = asdict(loaded.esmc_settings)
        report["compute_dtype"] = str(loaded.settings.trunk_dtype)
        ids, sequences, expand_map = inference.esmc_model.pack_lm_inputs(
            *(np.asarray(features[name]) for name in inference.LANGUAGE_MODEL_FEATURES),
            packed_length=args.packed_length,
        )
        report["packed_inputs"] = {
            name: _array_identity(value)
            for name, value in (
                ("ids", ids),
                ("sequence_id", sequences),
                ("expand_map", expand_map),
            )
        }
        if inference.LM_MASK_PCT != 0.0:
            raise ValueError("this diagnostic requires the unmasked native LM policy")
        report["lm_mask_pct"] = inference.LM_MASK_PCT
        report["rng_scope"] = "no LM masking; ESMC has no other random draws"
        if args.arm in NATIVE_ARMS:
            # Hashing parameters/raw layers before native evaluation would
            # synchronize transfers and alter the scheduling under test.
            phase("native_embedding")

            def native_embedding():
                return inference.language_model_embedding(
                    features,
                    loaded,
                    packed_length=args.packed_length,
                    deterministic=False,
                    state_storage=storage,
                )

            first, report["lm_cold_seconds"] = _timed(native_embedding)
            embedding, report["lm_warm_seconds"] = _timed(native_embedding)
            report["cold_warm"] = _differences(first, embedding)
            report["lm_cold_warm_max_abs_error"] = report["cold_warm"]["max_abs_error"]
            del first
            np.save(args.output / "embedding.npy", _host(embedding), allow_pickle=False)
            report["embedding"] = _description(embedding)
        else:
            phase("capture")
            report["capture_interference"] = (
                "per-layer host read/hash/file write synchronizes execution; "
                "extra consumers and stack/buffer retention alter scheduling; "
                "these are not native-arm observations"
            )
            tokens, records = _capture(
                jax.device_put(ids),
                jax.device_put(sequences),
                expand_map,
                loaded.esmc_parameters,
                loaded.esmc_settings,
                args.output,
                storage=storage,
                deterministic=False,
            )
            report["layers"] = records
            report["capture_token_states"] = _description(tokens)
            report["capture_token_identity"] = _array_identity(tokens)
            jax.block_until_ready(tokens)
            del tokens
        phase("actual_parameter_hashes")
        report["actual_esmc_parameters"] = _parameter_identity(loaded.esmc_parameters)
        report["actual_projection_parameters"] = _parameter_identity(loaded.parameters)
        inference.release_language_model_parameters(loaded, after=None)
        if args.arm in CAPTURE_ARMS:
            phase("same_captured_values_replay")
            report["replay_barrier"] = args.replay_barrier
            report["replay"] = _replay(
                args.output,
                records,
                expand_map,
                loaded,
                deterministic=False,
                barrier=args.replay_barrier,
            )
        report["memory"] = _memory(device)
        report["memory_scope"] = (
            "fresh LM-only worker lifetime; includes load, LM, parameter hashing "
            "and capture/replay when requested"
        )
        report["memory"]["scope"] = report["memory_scope"]
        _verify_binding(binding, args.weights, source_root)
        if _sha256(Path(__file__)) != report["diagnostic_sha256"]:
            raise ValueError("diagnostic source changed during execution")
        if (
            args.compiler_fingerprints
            and _sha256(Path(compilation_profile.__file__))
            != report["compiler_observer_source_sha256"]
        ):
            raise ValueError("compiler observer source changed during execution")
        report["status"] = "completed"
        _write_json(args.output / "report.json", report)
        phase("completed")


def _comparison(output):
    reports = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(output.glob("*/report.json"))
    }
    result = {
        "completed_arms": list(reports),
        "full_model_admission": None,
        "scope": (
            "observed process variation; compare only matching identities; "
            "no causal assignment"
        ),
    }
    native = {
        name: report for name, report in reports.items() if report["arm"] in NATIVE_ARMS
    }
    pairs = []
    for left, right in combinations(native, 2):
        a, b = native[left], native[right]
        group = (
            "between-storage"
            if a["storage"] != b["storage"]
            else "within-" + a["storage"]
        )
        pairs.append(
            {
                "left": left,
                "right": right,
                "group": group,
                "same_replicate": a["replicate"] == b["replicate"],
                **_pair_controls(a, b),
                **_differences(
                    np.load(output / left / "embedding.npy", allow_pickle=False),
                    np.load(output / right / "embedding.npy", allow_pickle=False),
                ),
            }
        )
    result["native_pairwise"] = pairs
    result["native_cold_warm"] = {
        name: report["cold_warm"] for name, report in native.items()
    }
    result["native_distributions"] = {}
    for group in ("within-stack", "within-buffer", "between-storage"):
        grouped = [row for row in pairs if row["group"] == group]
        observations = [
            row["max_abs_error"]
            for row in grouped
            if row["same_signature"] and row["controls_match"] is True
        ]
        result["native_distributions"][group] = {
            "pairs": len(observations),
            "excluded_pairs": len(grouped) - len(observations),
            "max_abs_errors": observations,
            "min": min(observations) if observations else None,
            "median": float(np.median(observations)) if observations else None,
            "max": max(observations) if observations else None,
        }
    if all(arm in reports for arm in CAPTURE_ARMS):
        result["instrumented_first_layer_divergence"] = _first_divergence(
            reports["capture-stack"]["layers"],
            reports["capture-buffer"]["layers"],
        )
    result["binding_matches"] = {
        field: (
            len(
                {
                    json.dumps(report[field], sort_keys=True)
                    for report in reports.values()
                }
            )
            == 1
            if len(reports) > 1 and all(field in report for report in reports.values())
            else None
        )
        for field in (
            "binding",
            "packed_inputs",
            "actual_esmc_parameters",
            "actual_projection_parameters",
            "esmc_settings",
            "compute_dtype",
        )
    }
    return result


def _reanchor_binding(
    binding_path,
    weights,
    features,
    output,
    source_root,
    *,
    reanchor_weights=True,
    rebind_probe=False,
):
    binding = json.loads(binding_path.read_text())
    if _sha256(features) != binding["features_sha256"]:
        raise ValueError("feature payload does not match the binding")
    entries = binding.get("weights")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("reanchoring requires all checkpoint content hashes")
    for filename, identity in entries.items():
        relative = Path(filename)
        expected = identity.get("sha256")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint binding paths must be relative to weights")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(
                character not in "0123456789abcdef" for character in expected.lower()
            )
        ):
            raise ValueError("reanchoring requires all checkpoint content hashes")
        path = weights / relative
        before = _local_anchor(path)
        if not reanchor_weights and before != identity["local_change_anchor"]:
            raise ValueError(f"checkpoint changed during the paired probe: {filename}")
        actual = _sha256(path)
        after = _local_anchor(path)
        if before != after:
            raise ValueError(f"checkpoint changed while hashing: {filename}")
        if actual != expected.lower():
            raise ValueError(f"checkpoint content differs: {filename}")
        if reanchor_weights:
            identity["local_change_anchor"] = after
    if rebind_probe:
        from bench import esmfold2_performance_probe

        previous = binding.get("probe_sha256")
        if (
            not isinstance(previous, str)
            or len(previous) != 64
            or any(
                character not in "0123456789abcdef" for character in previous.lower()
            )
        ):
            raise ValueError("probe rebinding requires the original creator probe hash")
        current = _sha256(Path(esmfold2_performance_probe.__file__))
        binding["probe_rebinding"] = {
            "previous_sha256": previous,
            "current_sha256": current,
        }
        binding["probe_sha256"] = current
    # All content and model-source identities remain required. Explicit
    # migrations write a new local binding; archived provenance is immutable.
    _verify_binding(binding, weights, source_root)
    destination = output / "local-binding.json"
    _write_json(destination, binding)
    return destination


def _coordinator(args):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    worker_binding = args.binding
    if getattr(args, "reanchor_weights", False) or getattr(args, "rebind_probe", False):
        import foldjax

        worker_binding = _reanchor_binding(
            args.binding,
            args.weights,
            args.features,
            args.output,
            Path(foldjax.__file__).resolve().parent.parent,
            reanchor_weights=getattr(args, "reanchor_weights", False),
            rebind_probe=getattr(args, "rebind_probe", False),
        )
    tasks = []
    if args.mode != "capture":
        tasks.extend(
            (f"{arm}-{replicate}", arm, replicate)
            for replicate in range(args.repeats)
            for arm in NATIVE_ARMS
        )
    if args.mode != "native":
        tasks.extend((arm, arm, 0) for arm in CAPTURE_ARMS)
    _write_json(
        args.output / "plan.json",
        {
            "mode": args.mode,
            "repeats_per_storage": args.repeats,
            "tasks": tasks,
            "full_model_admission": None,
            "control_policy": _control_policy(args),
            "compiler_observer_enabled": args.compiler_fingerprints,
            "input_binding_sha256": _sha256(args.binding),
            "worker_binding_sha256": _sha256(worker_binding),
            "reanchored_weights": bool(getattr(args, "reanchor_weights", False)),
            "probe_rebound": bool(getattr(args, "rebind_probe", False)),
        },
    )
    statuses = {}
    for task_name, arm, replicate in tasks:
        command = [
            sys.executable,
            "-m",
            "bench.esmc_state_diagnostics",
            "--worker",
            "--features",
            str(args.features.resolve()),
            "--binding",
            str(worker_binding.resolve()),
            "--weights",
            str(args.weights.resolve()),
            "--output",
            str(args.output / task_name),
            "--arm",
            arm,
            "--replicate",
            str(replicate),
            "--esmc-dtype",
            args.esmc_dtype,
        ]
        for name in ("packed_length", "compute_dtype"):
            if getattr(args, name) is not None:
                command.extend(
                    ("--" + name.replace("_", "-"), str(getattr(args, name)))
                )
        for name in (
            "deterministic",
            "compile_esmc_blocks",
            "replay_barrier",
            "compiler_fingerprints",
        ):
            if getattr(args, name, False):
                command.append("--" + name.replace("_", "-"))
        with (args.output / f"{task_name}.log").open("w") as log:
            statuses[task_name] = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
    _write_json(
        args.output / "comparison.json",
        {"return_codes": statuses, **_comparison(args.output)},
    )
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "binding", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("native", "capture", "both"), default="native"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="fresh native processes per storage mode; interleaved stack/buffer",
    )
    parser.add_argument("--packed-length", type=int)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--esmc-dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "process-wide deterministic XLA policy before backend initialization; "
            "block compilation is independent"
        ),
    )
    parser.add_argument(
        "--compile-esmc-blocks",
        action="store_true",
        help="compile LM blocks; independent of process-wide deterministic control",
    )
    parser.add_argument("--replay-barrier", action="store_true")
    parser.add_argument(
        "--reanchor-weights",
        action="store_true",
        help="verify weight content hashes and copy binding with current anchors",
    )
    parser.add_argument(
        "--rebind-probe",
        action="store_true",
        help=(
            "content-verify archived features, weights and model sources; "
            "copy the binding for this probe revision"
        ),
    )
    parser.add_argument(
        "--compiler-fingerprints",
        action="store_true",
        help="diagnostic compiler observer; perturbs compile-time scheduling",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--replicate", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--arm", choices=(*NATIVE_ARMS, *CAPTURE_ARMS), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.weights.is_file():
        args.weights = args.weights.parent
    if args.worker:
        if args.arm is None:
            parser.error("worker requires --arm")
        _worker(args)
        return 0
    else:
        statuses = _coordinator(args)
        return int(any(code != 0 for code in statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
