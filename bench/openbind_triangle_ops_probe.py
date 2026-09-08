"""Private full triangle operators, synthetic activations and real OpenBind weights.

The fixed 52 finite cases cover four checkpoint module families, both multiplication
directions (fused residual), both PairBlock attention directions (update only),
N=16/17/65, plus all four Pairformer operators at N=437. This is operator parity,
not real trunk activations, a whole-model admission, or a performance benchmark.
Run native and candidate in separate, serialized GPU jobs; this script never queues.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.openbind_tape_adapter import UPSTREAM_COMMIT
from bench.openbind_triton_attention_probe import (
    AttentionKernelCapture,
    compare_outputs,
    summarize,
)
from bench.openbind_triton_attention_probe import (
    helper_identities as attention_helper_identities,
)
from bench.openbind_triton_attention_probe import (
    source_identity as attention_source_identity,
)
from bench.openbind_triton_attention_probe import (
    validate_launch as validate_attention_launch,
)
from bench.openbind_triton_norm_probe import save_arrays, write_compiler_evidence

PANEL = "openbind-real-weight-triangle-ops-v1"
NATIVE_OPERATOR = "openfold3/core/model/layers/triangular_multiplicative_update.py"
CANDIDATE_OPERATOR = "src/foldjax/models/openfold3/models/native_triangle_ops.py"
CHECKPOINT = "openfold3_weights/checkpoints/of3-ob-2025-06-30-174k.pt"
FAMILIES = {
    "template": ("template_embedder.template_pair_stack.blocks.0", 64),
    "msa": ("msa_module.blocks.0.pair_stack", 128),
    "pairformer": ("pairformer_stack.blocks.0.pair_stack", 128),
    "confidence": (
        "aux_heads.pairformer_embedding.pairformer_stack.blocks.0.pair_stack",
        128,
    ),
}
OPERATORS = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end")
POLICY = {
    "dtype": "float32",
    "float32_matmul_precision": "high",
    "use_triton_triangle_kernels": True,
    "use_cueq_triangle_kernels": False,
    "inplace_safe": True,
    "multiplication_add_with_inplace": True,
    "multiplication_chunk_size": 256,
    "attention_chunk_size": 1024,
    "attention_starting": True,
    "attention_end_transpose_bias": True,
    "attention_use_high_precision": False,
    "use_exp2": False,
    "dynamic": False,
    "batch": 1,
}


@dataclass(frozen=True)
class Case:
    family: str
    operator: str
    length: int

    @property
    def name(self):
        return f"{self.family}-{self.operator}-n{self.length}"

    @property
    def width(self):
        return FAMILIES[self.family][1]

    @property
    def shape(self):
        return (1, self.length, self.length, self.width)

    @property
    def finite(self):
        return True

    @property
    def weight_id(self):
        return f"{self.family}-{self.operator}"

    @property
    def prefix(self):
        return FAMILIES[self.family][0] + "." + self.operator

    @property
    def multiplication(self):
        return self.operator.startswith("tri_mul")

    def identity(self):
        return {
            **asdict(self),
            "shape": list(self.shape),
            "finite_gate": True,
            "parameter_prefix": self.prefix,
            "output_contract": "fused_residual"
            if self.multiplication
            else "update_only",
        }


def case_specs():
    return [
        Case(f, op, n) for f in FAMILIES for op in OPERATORS for n in (16, 17, 65)
    ] + [Case("pairformer", op, 437) for op in OPERATORS]


def operands(case):
    rng = np.random.default_rng(8107 + case.length + case.width)
    z = (rng.normal(size=case.shape) * 0.7).astype(np.float32)
    i, j = np.indices((case.length, case.length))
    mask = (((i + 2 * j) % 7) != 3)[None].astype(np.float32)
    return {"z": z, "mask": mask}


def validate_operands(arrays, case):
    if set(arrays) != {"z", "mask"}:
        raise ValueError("operand keys differ")
    for name, shape in (("z", case.shape), ("mask", case.shape[:-1])):
        x = arrays[name]
        if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"invalid shape/dtype/finite operand: {name}")
    i, j = np.indices((case.length, case.length))
    if not np.array_equal(arrays["mask"], (((i + 2 * j) % 7) != 3)[None]):
        raise ValueError("reference asymmetric mask changed")


def parameter_shapes(case):
    c = case.width
    if case.multiplication:
        shapes = {
            f"{norm}.{field}": (c,)
            for norm in ("layer_norm_in", "layer_norm_out")
            for field in ("weight", "bias")
        }
        shapes.update(
            {
                f"{name}.weight": (c, c)
                for name in (
                    "linear_a_g",
                    "linear_a_p",
                    "linear_b_g",
                    "linear_b_p",
                    "linear_g",
                    "linear_z",
                )
            }
        )
    else:
        shapes = {f"layer_norm.{field}": (c,) for field in ("weight", "bias")}
        shapes["linear_z.weight"] = (4, c)
        shapes.update(
            {f"mha.linear_{name}.weight": (c, c) for name in ("q", "k", "v", "g", "o")}
        )
    return shapes


def parameter_metadata(arrays, case):
    shapes = parameter_shapes(case)
    if set(arrays) != set(shapes):
        raise ValueError("checkpoint module parameter keys differ")
    result = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid checkpoint parameter: {name}")
        result[name] = {
            "checkpoint_key": case.prefix + "." + name,
            "shape": list(shape),
            "dtype": "float32",
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    return result


def helper_identities():
    return {
        **attention_helper_identities(),
        Path(__file__).name: digest(Path(__file__)),
    }


def source_identity(root, native):
    identity = attention_source_identity(root, native)
    relative = NATIVE_OPERATOR if native else CANDIDATE_OPERATOR
    paths = (
        (
            "openfold3/core/model/layers/triangular_attention.py",
            "openfold3/core/model/primitives/attention.py",
            "openfold3/core/model/primitives/normalization.py",
            "openfold3/core/utils/chunk_utils.py",
            "openfold3/core/kernels/triton/evoformer.py",
        )
        if native
        else tuple(
            "src/foldjax/models/openfold3/" + name
            for name in (
                "models/native_triton_norm.py",
                "models/native_triton_linear.py",
                "models/native_triton_attention.py",
                "models/primitives.py",
                "models/attention.py",
                "models/triangle.py",
                "bridge/torch_mapping.py",
            )
        )
    )
    if not native:
        paths += (
            "src/foldjax/models/boltz2/models/primitives/native_amp_norm.py",
            "src/foldjax/models/_cp.py",
        )
    return {
        **identity,
        "operator_path": relative,
        "operator_sha256": digest(root / relative),
        "dependency_sha256": {path: digest(root / path) for path in paths},
    }


def read_archive(path, expected_sha):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("reference archive changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_weights(root, case, record):
    arrays = read_archive(
        root / f"weights-{case.weight_id}.npz", record["archive_sha256"]
    )
    if (
        record["parameters"] != parameter_metadata(arrays, case)
        or record["prefix"] != case.prefix
    ):
        raise ValueError("reference parameter identity differs")
    return arrays


def read_reference_case(root, case, record):
    arrays = read_archive(root / f"{case.name}.npz", record["archive_sha256"])
    if set(arrays) != {"z", "mask", "first", "second"}:
        raise ValueError("reference case archive schema differs")
    first, second = arrays.pop("first"), arrays.pop("second")
    validate_operands(arrays, case)
    repeat = compare_outputs(first, second, case)
    if (
        not repeat["bitwise_equal"]
        or not repeat["both_finite"]
        or repeat != record["native_repeat"]
    ):
        raise ValueError("native repeat changed, nonfinite, or not bitwise")
    return arrays, first


def validate_calls(record, case):
    calls = record["calls"]
    if len(calls) != 2 or calls[0] != calls[1]:
        raise ValueError("native per-call metadata missing or repeats differ")
    for call in calls:
        if call["module_kwargs"] != native_kwargs(case):
            raise ValueError("native module policy differs")
        if case.multiplication:
            events = call["events"]
            a_rows = [min(256, case.length - i) for i in range(0, case.length, 256)]
            half = (case.length + 1) // 2
            b_rows = [
                min(256, end - i)
                for begin, end in ((0, half), (half, case.length))
                for i in range(begin, end, 256)
            ]
            a_calls = [e for e in events if e.get("weight") == "linear_a_g.weight"]
            b_calls = [e for e in events if e.get("weight") == "linear_b_g.weight"]
            if [e["input_shape"][1] for e in a_calls] != a_rows:
                raise ValueError("native a projection chunk schedule differs")
            axis = 1 if case.operator == "tri_mul_out" else 2
            if [e["input_shape"][axis] for e in b_calls] != b_rows:
                raise ValueError("native b projection chunk schedule differs")
            if len(call["kernels"]) != 3 * len(a_rows) + 7 * len(b_rows):
                raise ValueError("native multiplication kernel count differs")
        else:
            rows = [
                e["input_shape"][0]
                for e in call["events"]
                if e["name"] == "mha.linear_q"
            ]
            if rows != [
                min(1024, case.length - i) for i in range(0, case.length, 1024)
            ]:
                raise ValueError("native attention actual row chunks differ")
            expected = 0 if case.length <= 16 else len(rows)
            if len(call["kernels"]) != expected:
                raise ValueError("native attention stock/Triton dispatch differs")
        for kernel in call["kernels"]:
            if "ptx" not in kernel["compiler_evidence"]:
                raise ValueError("native kernel missing actual PTX")
            metadata = kernel["metadata"]
            if case.multiplication:
                if (
                    metadata["name"]
                    not in {"layernorm_kernel", "linear_kernel", "linear_fused_kernel"}
                    or metadata["num_warps"] != 4
                    or metadata["num_stages"] != 3
                    or metadata["default_dot_input_precision"] != "tf32"
                    or metadata["enable_fp_fusion"] is not True
                ):
                    raise ValueError("native multiplication compiler policy differs")
            else:
                validate_attention_launch(
                    kernel,
                    SimpleNamespace(
                        length=case.length, rows=rows[0], dim=case.width // 4
                    ),
                )


def verify_reference(root, *, expected_manifest_sha=None):
    payload = (root / "manifest.json").read_bytes()
    if (
        expected_manifest_sha is not None
        and hashlib.sha256(payload).hexdigest() != expected_manifest_sha
    ):
        raise ValueError("reference manifest changed")
    manifest = json.loads(payload)
    if (
        manifest["panel"] != PANEL
        or manifest["mode"] != "native"
        or manifest["completed"] is not True
        or manifest["policy"] != POLICY
        or manifest["source"]["commit"] != UPSTREAM_COMMIT
        or manifest["source"]["operator_path"] != NATIVE_OPERATOR
    ):
        raise ValueError("reference native source/policy/completion differs")
    if set(manifest["cases"]) != {c.name for c in case_specs()}:
        raise ValueError("reference panel incomplete")
    if set(manifest["weights"]) != {c.weight_id for c in case_specs()}:
        raise ValueError("reference weight panel incomplete")
    checkpoint = manifest["checkpoint"]
    if (
        checkpoint["relative_path"] != CHECKPOINT
        or checkpoint["size"] <= 0
        or len(checkpoint["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint["sha256"])
    ):
        raise ValueError("reference checkpoint identity invalid")
    for case in case_specs():
        record = manifest["cases"][case.name]
        if record["status"] != "ok" or record["spec"] != case.identity():
            raise ValueError("reference case specification/status differs")
        read_weights(root, case, manifest["weights"][case.weight_id])
        read_reference_case(root, case, record)
        validate_calls(record, case)
        for kernel in record["calls"][0]["kernels"]:
            for kind, sha in kernel["compiler_evidence"].items():
                if (
                    kind not in ("ptx", "ttir", "ttgir")
                    or len(sha) != 64
                    or any(c not in "0123456789abcdef" for c in sha)
                ):
                    raise ValueError("invalid compiler artifact identity")
                if digest(root / f"kernel-{sha}.{kind}.txt") != sha:
                    raise ValueError("reference compiler artifact changed")
    return manifest


def native_kwargs(case):
    base = {
        "inplace_safe": True,
        "use_cueq_triangle_kernels": False,
        "use_triton_triangle_kernels": True,
    }
    return {
        **base,
        **(
            {"_inplace_chunk_size": 256, "_add_with_inplace": True}
            if case.multiplication
            else {"chunk_size": 1024, "transpose_bias": case.operator == "tri_att_end"}
        ),
    }


def call_native(module, case, arrays):
    # Native multiplication overwrites its input. Each repeat starts from the
    # archived immutable source, including the ending-node orientation.
    z, mask = arrays["z"].clone(), arrays["mask"].clone()
    end = case.operator == "tri_att_end"
    if end:
        z, mask = z.transpose(1, 2), mask.transpose(1, 2)
    out = module(z, mask=mask, **native_kwargs(case))
    return out.transpose(1, 2) if end else out


def call_candidate(module, case, arrays, params):
    import jax.numpy as jnp

    z, mask = arrays["z"], arrays["mask"]
    if case.multiplication:
        return module.native_triangle_multiplication_residual(
            z, params, outgoing=case.operator == "tri_mul_out", mask=mask
        )
    end = case.operator == "tri_att_end"
    if end:
        z, mask = jnp.swapaxes(z, 1, 2), jnp.swapaxes(mask, 1, 2)
    out = module.native_triangle_attention_update(
        z, params, mask=mask, transpose_bias=end
    )
    return jnp.swapaxes(out, 1, 2) if end else out


class NativeRecorder:
    """Observe the real module, wrappers, ordinary submodules, and every launch."""

    def __init__(self, multiplication, attention_kernel, out_dir):
        self.events, self.kernels = [], []
        self.out_dir = out_dir
        self.weight_names = {}
        for name in ("triton_layernorm", "triton_linear", "triton_linear_fused"):
            original = getattr(multiplication, name)

            def wrapper(*args, _name=name, _fn=original, **kwargs):
                self.events.append(
                    {
                        "name": _name,
                        "input_shape": list(args[0].shape),
                        "weight": self.weight_names.get(args[1].data_ptr()),
                        "other_shape": list(kwargs["other"].shape)
                        if kwargs.get("other") is not None
                        else None,
                        "mask_shape": list(kwargs["mask"].shape)
                        if kwargs.get("mask") is not None
                        else None,
                        "add_shape": list(kwargs["add_tensor"].shape)
                        if kwargs.get("add_tensor") is not None
                        else None,
                        "apply_sigmoid": kwargs.get("apply_sigmoid", False),
                    }
                )
                return _fn(*args, **kwargs)

            setattr(multiplication, name, wrapper)
        for owner, name in (
            (multiplication, "layernorm_kernel"),
            (multiplication, "linear_kernel"),
            (multiplication, "linear_fused_kernel"),
            (attention_kernel, "_attn_fwd"),
        ):
            setattr(owner, name, self._capture(getattr(owner, name)))

    def _capture(self, kernel):
        recorder = self

        class Capture(AttentionKernelCapture):
            def __getitem__(self, grid):
                launch = super().__getitem__(grid)

                def run(*args, **kwargs):
                    result = launch(*args, **kwargs)
                    evidence = {
                        key: value
                        for key, value in self.evidence.items()
                        if key != "assembly"
                    }
                    files = {}
                    for kind in ("ptx", "ttir", "ttgir"):
                        value = self.evidence["assembly"].get(kind)
                        if not isinstance(value, str):
                            continue
                        sha = hashlib.sha256(value.encode()).hexdigest()
                        path = recorder.out_dir / f"kernel-{sha}.{kind}.txt"
                        if not path.exists():
                            write_compiler_evidence(
                                recorder.out_dir, f"kernel-{sha}", {kind: value}
                            )
                        elif digest(path) != sha:
                            raise ValueError("compiler artifact collision")
                        files[kind] = sha
                    evidence["compiler_evidence"] = files
                    recorder.kernels.append(evidence)
                    return result

                return run

        return Capture(kernel)

    def bind(self, module):
        self.weight_names = {
            p.data_ptr(): name for name, p in module.named_parameters()
        }
        handles = []
        for name, child in module.named_modules():
            if name and not list(child.children()):

                def hook(mod, args, output, _name=name):
                    self.events.append(
                        {
                            "name": _name,
                            "input_shape": list(args[0].shape),
                            "output_shape": list(output.shape),
                        }
                    )

                handles.append(child.register_forward_hook(hook))
        return handles

    def reset(self):
        self.events, self.kernels = [], []

    def record(self, case):
        return {
            "module_kwargs": native_kwargs(case),
            "events": self.events,
            "kernels": self.kernels,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    native = args.mode == "native"
    if native == (args.reference is not None):
        raise ValueError("candidate requires reference; native must omit it")
    if any(
        os.environ.get(key) == "1"
        for key in ("OF3_TRITON_EXP2", "OF3_TRITON_DYNAMIC_SHAPES")
    ):
        raise ValueError("native default exp2/dynamic policy required")
    root = args.source_root.resolve(strict=True)
    identity, helpers = source_identity(root, native), helper_identities()
    reference_sha = None if native else digest(args.reference / "manifest.json")
    reference = (
        None
        if native
        else verify_reference(args.reference, expected_manifest_sha=reference_sha)
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root if native else root / "src"))
    weights, parameters = {}, {}
    if native:
        import torch
        import triton

        multiplication = importlib.import_module(
            "openfold3.core.model.layers.triangular_multiplicative_update"
        )
        attention = importlib.import_module(
            "openfold3.core.model.layers.triangular_attention"
        )
        attention_kernel = importlib.import_module(
            "openfold3.core.kernels.triton.evoformer"
        )
        for mod in (multiplication, attention, attention_kernel):
            if not Path(inspect.getfile(mod)).resolve().is_relative_to(root):
                raise RuntimeError("native module imported outside selected root")
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.version.hip
        ):
            raise RuntimeError("requires exactly one NVIDIA CUDA GPU")
        torch.set_float32_matmul_precision("high")
        checkpoint_path = root / CHECKPOINT
        checkpoint = {
            "relative_path": CHECKPOINT,
            "sha256": digest(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
        }
        state = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=True
        )
        for case in case_specs():
            if case.weight_id in weights:
                continue
            subset = {
                k[len(case.prefix) + 1 :]: value.detach().cpu().numpy().copy()
                for k, value in state.items()
                if k.startswith(case.prefix + ".")
            }
            metadata = parameter_metadata(subset, case)
            weights[case.weight_id] = {
                "prefix": case.prefix,
                "parameters": metadata,
                "archive_sha256": save_arrays(
                    args.out_dir / f"weights-{case.weight_id}.npz", **subset
                ),
            }
            parameters[case.weight_id] = subset
        del state
        recorder = NativeRecorder(multiplication, attention_kernel, args.out_dir)
        runtime = {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        }
    else:
        import jax
        import jax.numpy as jnp
        import jaxlib

        module = importlib.import_module(
            "foldjax.models.openfold3.models.native_triangle_ops"
        )
        mapping = importlib.import_module(
            "foldjax.models.openfold3.bridge.torch_mapping"
        )
        if Path(inspect.getfile(module)).resolve() != root / CANDIDATE_OPERATOR:
            raise RuntimeError("candidate operator imported from wrong source")
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("requires exactly one GPU")
        jax.config.update("jax_default_matmul_precision", "high")
        checkpoint = reference["checkpoint"]
        weights = reference["weights"]
        runtime = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": str(jax.devices()[0]),
        }
    records = {}
    for case in case_specs():
        record = {"spec": case.identity()}
        handles = []
        arrays = gpu = params = first = second = expected = executable = (
            native_module
        ) = None
        try:
            if native:
                cls = (
                    (
                        multiplication.TriangleMultiplicationOutgoing
                        if case.operator == "tri_mul_out"
                        else multiplication.TriangleMultiplicationIncoming
                    )
                    if case.multiplication
                    else attention.TriangleAttention
                )
                native_module = (
                    cls(case.width, case.width)
                    if case.multiplication
                    else cls(case.width, case.width // 4, 4, starting=True)
                )
                native_module.load_state_dict(
                    {
                        k: torch.from_numpy(v.copy())
                        for k, v in parameters[case.weight_id].items()
                    },
                    strict=True,
                )
                native_module = native_module.eval().cuda()
                handles = recorder.bind(native_module)
                arrays = operands(case)
                validate_operands(arrays, case)
                gpu = {k: torch.from_numpy(v).cuda() for k, v in arrays.items()}
                record["calls"] = []
                outputs = []
                with torch.no_grad():
                    for _ in range(2):
                        recorder.reset()
                        outputs.append(
                            call_native(native_module, case, gpu).cpu().numpy().copy()
                        )
                        record["calls"].append(recorder.record(case))
                first, second = outputs
                record["archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}.npz",
                    **arrays,
                    first=first,
                    second=second,
                )
                record["native_repeat"] = compare_outputs(first, second, case)
                validate_calls(record, case)
            else:
                arrays, expected = read_reference_case(
                    args.reference, case, reference["cases"][case.name]
                )
                raw = read_weights(args.reference, case, weights[case.weight_id])
                mapper = (
                    mapping.map_triangle_multiplication
                    if case.multiplication
                    else mapping.map_triangle_attention
                )
                params = mapper(raw)
                gpu = {key: jnp.asarray(value) for key, value in arrays.items()}
                executable = (
                    jax.jit(lambda data, p: call_candidate(module, case, data, p))
                    .lower(gpu, params)
                    .compile()
                )
                first, second = (
                    np.asarray(executable(gpu, params)),
                    np.asarray(executable(gpu, params)),
                )
                record["archive_sha256"] = save_arrays(
                    args.out_dir / f"{case.name}.npz", first=first, second=second
                )
                record["candidate_repeat"] = compare_outputs(first, second, case)
                record["native_comparison"] = compare_outputs(first, expected, case)
                hlo = executable.as_text()
                record["compiler_evidence"] = write_compiler_evidence(
                    args.out_dir, case.name, {"hlo": hlo}
                )
                markers = (
                    (
                        "openbind_native_triangle_layer_norm",
                        "openbind_native_triangle_linear_fused",
                    )
                    if case.multiplication
                    else (
                        ()
                        if case.length <= 16
                        else ("openbind_native_triangle_attention",)
                    )
                )
                record["backend_markers"] = {
                    marker: marker in hlo for marker in markers
                }
                if not all(record["backend_markers"].values()):
                    raise RuntimeError(
                        "candidate HLO lacks required private native kernel markers"
                    )
                record["native_calls"] = reference["cases"][case.name]["calls"]
            record["status"] = "ok"
        except Exception as exc:
            record.update(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            for handle in handles:
                handle.remove()
        records[case.name] = record
        save_new(args.out_dir / f"{case.name}.json", record)
        print(
            json.dumps(
                {
                    "case": case.name,
                    "status": record["status"],
                    "comparison": record.get(
                        "native_comparison", record.get("native_repeat")
                    ),
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
        del arrays, gpu, params, first, second, expected, executable, native_module
    if source_identity(root, native) != identity or helper_identities() != helpers:
        raise RuntimeError("source/helper identity changed during probe")
    if native:
        if digest(checkpoint_path) != checkpoint["sha256"]:
            raise RuntimeError("checkpoint changed during probe")
    else:
        verify_reference(args.reference, expected_manifest_sha=reference_sha)
    summary = summarize(records, native)
    save_new(
        args.out_dir / "manifest.json",
        {
            "mode": args.mode,
            "panel": PANEL,
            "completed": summary["execution_errors"] == 0,
            "all_cases_attempted": True,
            "source": identity,
            "helper_source_sha256": helpers,
            "harness_sha256": helpers[Path(__file__).name],
            "policy": POLICY,
            "runtime": runtime,
            "checkpoint": checkpoint,
            "weights": weights,
            "cases": records,
            "summary": summary,
            "reference_manifest_sha256": reference_sha,
            "scope": __doc__,
            "strict_leaf_tolerance": {"atol": 1e-4, "rtol": 1e-4},
            "full_model_admission": None,
        },
    )
    print(json.dumps({"summary": summary}), flush=True)
    return 1 if summary["execution_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
