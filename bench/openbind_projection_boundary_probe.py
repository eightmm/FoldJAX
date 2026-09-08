"""Teacher-forced ordinary OpenBind projection boundary, never a model policy change.

Three real-weight Pairformer start-attention cases N16/17/437 reuse the frozen
full-module native input archives. Actual native LN/projection outputs are captured
twice. Candidate projections always consume the captured native LN output, never
the candidate LN output. Candidate LN is a separate diagnostic on original z.
The opt-in host RNE control transforms only the teacher and projection weights;
its GEMM remains the current high-precision implementation. It is not a port fix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench import openbind_triangle_ops_probe as full
from bench.boltz_historical_replay import digest, save_new
from bench.openbind_tape_adapter import UPSTREAM_COMMIT
from bench.openbind_triton_norm_probe import (
    metrics,
    save_arrays,
    write_compiler_evidence,
)

PANEL = "openbind-ordinary-projection-native-ln-teacher-v1"
PROJECTIONS = {
    "q": "mha.linear_q",
    "k": "mha.linear_k",
    "v": "mha.linear_v",
    "g": "mha.linear_g",
    "bias": "linear_z",
}
CONTROLS = ("high", "TF32_TF32_F32", "TF32_TF32_F32_X3", "F32_F32_F32")
FIELDS = ("norm", *PROJECTIONS)
CANDIDATE_OPERATOR = "src/foldjax/models/openfold3/models/primitives.py"
NATIVE_OPERATOR = "openfold3/core/model/layers/triangular_attention.py"
SHARED_NORM = "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py"


def case_specs():
    return [full.Case("pairformer", "tri_att_start", n) for n in (16, 17, 437)]


def expected_shape(case, field):
    if field == "norm":
        return case.shape
    if field == "bias":
        return (*case.shape[:-1], 4)
    if field in PROJECTIONS:
        return case.shape[1:]
    raise ValueError("unknown projection boundary")


def value_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def tf32_rne(value):
    """Host-only nearest-even TF32 grid; preserve signed zero and NaN payloads."""
    if value.dtype != np.float32:
        raise TypeError("TF32 RNE operand control requires FP32")
    bits = np.ascontiguousarray(value).view(np.uint32)
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = (
        bits + np.uint32(0xFFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    ) & np.uint32(0xFFFFE000)
    return np.where(finite, rounded, bits).view(np.float32).reshape(value.shape)


def operand_control_metadata(rounding):
    if rounding not in {"none", "rne"}:
        raise ValueError("unknown host operand rounding control")
    return {
        "mode": rounding,
        "stage": "none" if rounding == "none" else "host_before_device_transfer",
        "operands": []
        if rounding == "none"
        else ["native_norm_0", "projection_weights"],
        "rule": None
        if rounding == "none"
        else "finite: (bits + 0xfff + ((bits >> 13) & 1)) & 0xffffe000",
        "teacher_forced": True,
        "production_modified": False,
    }


def candidate_operands(norm, weights, rounding):
    """Return effective operands and hashes without mutating native evidence."""
    operand_control_metadata(rounding)
    expected = {name + ".weight" for name in PROJECTIONS.values()}
    if set(weights) != expected:
        raise ValueError("control must receive exactly the five projection weights")
    original = {"native_norm_0": norm, **weights}
    if any(
        value.dtype != np.float32 or not np.isfinite(value).all()
        for value in original.values()
    ):
        raise ValueError("teacher-forced operands must be finite FP32")
    effective = (
        original
        if rounding == "none"
        else {key: tf32_rne(value) for key, value in original.items()}
    )
    if any(not np.isfinite(value).all() for value in effective.values()):
        raise ValueError("RNE control overflowed a finite operand")
    identities = {
        "original": {key: value_identity(value) for key, value in original.items()},
        "effective": {key: value_identity(value) for key, value in effective.items()},
    }
    return (
        effective["native_norm_0"],
        {key: effective[key] for key in weights},
        identities,
    )


def selected_controls(rounding):
    operand_control_metadata(rounding)
    return CONTROLS if rounding == "none" else ("high",)


def validate_boundary(arrays, case):
    if set(arrays) != {f"{field}_{repeat}" for field in FIELDS for repeat in (0, 1)}:
        raise ValueError("boundary archive keys differ")
    repeats = {}
    for field in FIELDS:
        values = [arrays[f"{field}_{r}"] for r in (0, 1)]
        if any(
            v.shape != expected_shape(case, field)
            or v.dtype != np.float32
            or not np.isfinite(v).all()
            for v in values
        ):
            raise ValueError("boundary shape/dtype/finite contract differs")
        repeats[field] = metrics(*values)
        if not repeats[field]["bitwise_equal"]:
            raise ValueError("native boundary repeats are not bitwise")
    return repeats


def validate_inputs(observed, norm, case):
    if set(observed) != set(PROJECTIONS):
        raise ValueError("native projection input capture incomplete")
    expected = {}
    for field in PROJECTIONS:
        shape = case.shape if field == "bias" else case.shape[1:]
        expected[field] = value_identity(norm.reshape(shape))
    if observed != expected:
        raise ValueError("native projection input differs from captured LN output")


def helper_identities():
    return {**full.helper_identities(), Path(__file__).name: digest(Path(__file__))}


def source_identity(root, native, norm_implementation="generic"):
    if norm_implementation not in {"generic", "cuda-welford"}:
        raise ValueError("unknown norm implementation control")
    if native and norm_implementation != "generic":
        raise ValueError("native baseline must retain its original norm")
    identity = full.source_identity(root, native)
    relative = NATIVE_OPERATOR if native else CANDIDATE_OPERATOR
    result = {
        **identity,
        "operator_path": relative,
        "operator_sha256": digest(root / relative),
    }
    if norm_implementation == "cuda-welford":
        result["norm_control_source"] = {SHARED_NORM: digest(root / SHARED_NORM)}
    return result


def candidate_norm(module, x, params, implementation):
    if implementation == "generic":
        return module.layer_norm(x, params)
    if implementation != "cuda-welford":
        raise ValueError("unknown norm implementation control")
    from foldjax.models.boltz2.models.primitives.native_amp_norm import _cuda_layer_norm

    return _cuda_layer_norm(x, params.weight, params.bias, 1e-5)[0]


def read_reference_case(root, case, record):
    arrays = full.read_archive(root / f"{case.name}.npz", record["archive_sha256"])
    if validate_boundary(arrays, case) != record["native_repeat"]:
        raise ValueError("boundary native metrics changed")
    for r in (0, 1):
        validate_inputs(record["projection_inputs"][r], arrays[f"norm_{r}"], case)
    return arrays


def verify_reference(root, original_sha, expected_sha=None):
    payload = (root / "manifest.json").read_bytes()
    if expected_sha is not None and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("boundary reference manifest changed")
    manifest = json.loads(payload)
    if (
        manifest["panel"] != PANEL
        or manifest["mode"] != "native"
        or not manifest["completed"]
        or manifest["controls"] != list(CONTROLS)
        or manifest["upstream_reference_sha256"] != original_sha
        or manifest["source"]["commit"] != UPSTREAM_COMMIT
        or manifest["source"]["operator_path"] != NATIVE_OPERATOR
        or manifest["policy"] != full.POLICY
        or manifest.get("operand_control", {}).get("mode", "none") != "none"
    ):
        raise ValueError("boundary reference source/policy/completion differs")
    if set(manifest["cases"]) != {c.name for c in case_specs()}:
        raise ValueError("boundary case panel incomplete")
    for case in case_specs():
        record = manifest["cases"][case.name]
        if record["status"] != "ok" or record["spec"] != case.identity():
            raise ValueError("boundary case status/specification differs")
        if (
            not record["baseline_comparison"]["bitwise_equal"]
            or not record["module_repeat"]["bitwise_equal"]
        ):
            raise ValueError("capturing boundaries changed full native output")
        read_reference_case(root, case, record)
        module = full.read_archive(
            root / f"{case.name}-module.npz", record["module_archive_sha256"]
        )
        if set(module) != {"first", "second"}:
            raise ValueError("native module archive schema differs")
        if any(x.shape != case.shape or x.dtype != np.float32 for x in module.values()):
            raise ValueError("native module archive shape/dtype differs")
        if metrics(module["first"], module["second"]) != record["module_repeat"]:
            raise ValueError("native module repeat artifact differs")
    return manifest


class NativeCapture:
    """Copy before native attention's final in-place assignment aliases the LN."""

    def __init__(self, module):
        self.values, self.inputs, self.handles = {}, {}, []
        names = {
            "layer_norm": "norm",
            **{name: field for field, name in PROJECTIONS.items()},
        }
        for name, child in module.named_modules():
            if name not in names:
                continue
            field = names[name]

            def hook(mod, args, output, _field=field):
                if _field in self.values:
                    raise ValueError(
                        "unexpected repeated row chunk in frozen N<=437 panel"
                    )
                value = output.detach().cpu().numpy().copy()
                self.values[_field] = value
                if _field != "norm":
                    self.inputs[_field] = value_identity(args[0].detach().cpu().numpy())

            self.handles.append(child.register_forward_hook(hook))

    def reset(self):
        self.values, self.inputs = {}, {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def projection_group(module, norm, weights, control):
    """Project the supplied teacher with native per-projection input ranks."""
    import jax
    import jax.numpy as jnp

    if control not in CONTROLS:
        raise ValueError("unknown explicit precision control")
    output = {}
    for field, name in PROJECTIONS.items():
        x = norm if field == "bias" else norm.reshape(norm.shape[1:])
        weight = weights[name + ".weight"]
        if control == "high":
            with jax.default_matmul_precision("high"):
                output[field] = module.linear(x, module.LinearParams(weight))
        else:
            preset = getattr(jax.lax.DotAlgorithmPreset, control)
            output[field] = jnp.matmul(x, weight.T, precision=preset)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-reference", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--operand-rounding", choices=("none", "rne"), default="none")
    parser.add_argument(
        "--norm-implementation", choices=("generic", "cuda-welford"), default="generic"
    )
    args = parser.parse_args(argv)
    native = args.mode == "native"
    if native == (args.reference is not None):
        raise ValueError("candidate requires boundary reference; native must omit it")
    if native and args.operand_rounding != "none":
        raise ValueError("native baseline must not receive operand controls")
    controls = selected_controls(args.operand_rounding)
    if any(
        os.environ.get(k) == "1"
        for k in ("OF3_TRITON_EXP2", "OF3_TRITON_DYNAMIC_SHAPES")
    ):
        raise ValueError("native specialized default policy required")
    root = args.source_root.resolve(strict=True)
    identity, helpers = (
        source_identity(root, native, args.norm_implementation),
        helper_identities(),
    )
    original_sha = digest(args.upstream_reference / "manifest.json")
    original = full.verify_reference(
        args.upstream_reference, expected_manifest_sha=original_sha
    )
    reference_sha = None if native else digest(args.reference / "manifest.json")
    reference = (
        None
        if native
        else verify_reference(args.reference, original_sha, reference_sha)
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root if native else root / "src"))
    if native:
        import torch

        module = importlib.import_module(
            "openfold3.core.model.layers.triangular_attention"
        )
        if not Path(inspect.getfile(module)).resolve().is_relative_to(root):
            raise RuntimeError("wrong native import root")
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.version.hip
        ):
            raise RuntimeError("requires exactly one NVIDIA GPU")
        torch.set_float32_matmul_precision("high")
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    else:
        import jax
        import jax.numpy as jnp
        import jaxlib

        module = importlib.import_module("foldjax.models.openfold3.models.primitives")
        if Path(inspect.getfile(module)).resolve() != root / CANDIDATE_OPERATOR:
            raise RuntimeError("wrong candidate import root")
        if args.norm_implementation == "cuda-welford":
            shared = importlib.import_module(
                "foldjax.models.boltz2.models.primitives.native_amp_norm"
            )
            if Path(inspect.getfile(shared)).resolve() != root / SHARED_NORM:
                raise RuntimeError("wrong shared norm import root")
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("requires exactly one GPU")
        runtime = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": str(jax.devices()[0]),
        }
    records = {}
    for case in case_specs():
        record = {"spec": case.identity()}
        capture = None
        try:
            arrays, baseline = full.read_reference_case(
                args.upstream_reference, case, original["cases"][case.name]
            )
            weights = full.read_weights(
                args.upstream_reference, case, original["weights"][case.weight_id]
            )
            if native:
                model = module.TriangleAttention(
                    case.width, case.width // 4, 4, starting=True
                )
                model.load_state_dict(
                    {k: torch.from_numpy(v.copy()) for k, v in weights.items()},
                    strict=True,
                )
                model = model.eval().cuda()
                capture = NativeCapture(model)
                gpu = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
                boundaries, outputs, observed = {}, [], []
                with torch.no_grad():
                    for r in (0, 1):
                        capture.reset()
                        outputs.append(
                            full.call_native(model, case, gpu).cpu().numpy().copy()
                        )
                        validate_inputs(capture.inputs, capture.values["norm"], case)
                        observed.append(capture.inputs)
                        boundaries.update(
                            {
                                f"{key}_{r}": value
                                for key, value in capture.values.items()
                            }
                        )
                record["projection_inputs"] = observed
                record["archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}.npz", **boundaries
                )
                record["module_archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}-module.npz",
                    first=outputs[0],
                    second=outputs[1],
                )
                record["native_repeat"] = validate_boundary(boundaries, case)
                record["baseline_comparison"] = metrics(outputs[0], baseline)
                record["module_repeat"] = metrics(*outputs)
                if (
                    not record["baseline_comparison"]["bitwise_equal"]
                    or not record["module_repeat"]["bitwise_equal"]
                ):
                    raise ValueError(
                        "boundary capture changed native full-module output"
                    )
                del model, gpu, boundaries, outputs
            else:
                boundaries = read_reference_case(
                    args.reference, case, reference["cases"][case.name]
                )
                p = module.LayerNormParams(
                    jnp.asarray(weights["layer_norm.weight"]),
                    jnp.asarray(weights["layer_norm.bias"]),
                )
                norm_executable = (
                    jax.jit(
                        lambda x, p: candidate_norm(
                            module, x, p, args.norm_implementation
                        )
                    )
                    .lower(jnp.asarray(arrays["z"]), p)
                    .compile()
                )
                norm = np.asarray(norm_executable(jnp.asarray(arrays["z"]), p))
                norm_repeat = np.asarray(norm_executable(jnp.asarray(arrays["z"]), p))
                record["norm_repeat"] = full.compare_outputs(norm_repeat, norm, case)
                record["norm_implementation"] = args.norm_implementation
                record["norm_archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}-norm.npz", candidate=norm
                )
                record["norm_hlo"] = write_compiler_evidence(
                    args.out_dir,
                    f"{case.name}-norm",
                    {"hlo": norm_executable.as_text()},
                )
                record["normalization_only"] = full.compare_outputs(
                    norm, boundaries["norm_0"], case
                )
                effective_norm, effective_weights, operand_identities = (
                    candidate_operands(
                        boundaries["norm_0"],
                        {
                            name + ".weight": weights[name + ".weight"]
                            for name in PROJECTIONS.values()
                        },
                        args.operand_rounding,
                    )
                )
                teacher = jnp.asarray(effective_norm)
                device_weights = {
                    name: jnp.asarray(value)
                    for name, value in effective_weights.items()
                }
                record["operand_control"] = operand_control_metadata(
                    args.operand_rounding
                )
                record["operand_identities"] = operand_identities
                record["original_weight_archive_sha256"] = original["weights"][
                    case.weight_id
                ]["archive_sha256"]
                record["teacher_input_sha256"] = value_identity(boundaries["norm_0"])[
                    "sha256"
                ]
                record["effective_teacher_input_sha256"] = operand_identities[
                    "effective"
                ]["native_norm_0"]["sha256"]
                record["controls"] = {}
                for control in controls:
                    control_record = {
                        "precision": control,
                        "teacher": "native_norm_0"
                        if args.operand_rounding == "none"
                        else "RNE(native_norm_0)",
                        "candidate_norm_used": False,
                        "operand_rounding": args.operand_rounding,
                    }
                    try:
                        executable = (
                            jax.jit(
                                lambda x, w: projection_group(module, x, w, control)
                            )
                            .lower(teacher, device_weights)
                            .compile()
                        )
                    except Exception as exc:
                        control_record.update(
                            status="compile_error",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        record["controls"][control] = control_record
                        continue
                    try:
                        first = {
                            k: np.asarray(v)
                            for k, v in executable(teacher, device_weights).items()
                        }
                        second = {
                            k: np.asarray(v)
                            for k, v in executable(teacher, device_weights).items()
                        }
                        control_record["archive_sha256"] = save_arrays(
                            args.out_dir / f"{case.name}-{control}.npz",
                            **{
                                f"{k}_{r}": v
                                for r, values in enumerate((first, second))
                                for k, v in values.items()
                            },
                        )
                        control_record["compiler_evidence"] = write_compiler_evidence(
                            args.out_dir,
                            f"{case.name}-{control}",
                            {"hlo": executable.as_text()},
                        )
                        control_record["native_comparison"] = {
                            k: full.compare_outputs(
                                first[k],
                                boundaries[f"{k}_0"],
                                SimpleNamespace(
                                    shape=expected_shape(case, k), finite=True
                                ),
                            )
                            for k in PROJECTIONS
                        }
                        control_record["candidate_repeat"] = {
                            k: full.compare_outputs(
                                first[k],
                                second[k],
                                SimpleNamespace(
                                    shape=expected_shape(case, k), finite=True
                                ),
                            )
                            for k in PROJECTIONS
                        }
                        control_record["status"] = "ok"
                    except Exception as exc:
                        control_record.update(
                            status="execution_error",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    record["controls"][control] = control_record
                    del executable
                del boundaries, teacher, norm_executable
            record["status"] = "ok"
        except Exception as exc:
            record.update(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            if capture is not None:
                capture.close()
        records[case.name] = record
        save_new(args.out_dir / f"{case.name}.json", record)
        print(
            json.dumps(
                {
                    "case": case.name,
                    "status": record["status"],
                    "norm": record.get("normalization_only"),
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
    if (
        source_identity(root, native, args.norm_implementation) != identity
        or helper_identities() != helpers
    ):
        raise RuntimeError("source/helper changed during boundary probe")
    full.verify_reference(args.upstream_reference, expected_manifest_sha=original_sha)
    if not native:
        verify_reference(args.reference, original_sha, reference_sha)
    errors = sum(r["status"] != "ok" for r in records.values())
    control_errors = sum(
        c["status"] != "ok"
        for r in records.values()
        for c in r.get("controls", {}).values()
    )
    save_new(
        args.out_dir / "manifest.json",
        {
            "panel": PANEL,
            "mode": args.mode,
            "source": identity,
            "helper_source_sha256": helpers,
            "runtime": runtime,
            "policy": full.POLICY,
            "controls": list(controls),
            "operand_control": operand_control_metadata(args.operand_rounding),
            "norm_implementation": args.norm_implementation,
            "cases": records,
            "completed": not errors and not control_errors,
            "all_cases_attempted": True,
            "execution_errors": errors,
            "control_errors": control_errors,
            "upstream_reference_sha256": original_sha,
            "boundary_reference_sha256": reference_sha,
            "scope": __doc__,
            "strict_leaf_tolerance": {"atol": 1e-4, "rtol": 1e-4},
            "full_model_admission": None,
        },
    )
    return int(bool(errors or control_errors))


if __name__ == "__main__":
    raise SystemExit(main())
