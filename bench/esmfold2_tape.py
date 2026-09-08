"""Capture and replay ESMFold2's native random tape for shared-core diagnostics.

This is development evidence, not a normal benchmark.  Capture requires the
publisher's torch/transformers checkout; replay deliberately imports neither.
Replay fails before weight loading unless the core supports every tape input.
Both commands use ESMFold2's released settings, five samples and seed 101.
The input NPZ is the already-built native feature dictionary, so this compares
the shared core only, not two independent preprocessing implementations.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

SAMPLES = 5
SEED = 101


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _schedule_steps(model: Any) -> int:
    """Read the native schedule with sample()'s released 256-sigma cap."""
    schedule = model.structure_head.inference_noise_schedule()
    return int((schedule <= 256.0).sum().item())


def _model_features(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    from foldjax.models.esmfold2.data.all_atom import OUTPUT_METADATA_FEATURES

    return {
        key: value
        for key, value in features.items()
        if key not in OUTPUT_METADATA_FEATURES
    }


@dataclass(frozen=True)
class TapeShapes:
    batch: int
    tokens: int
    atoms: int
    pair_width: int
    samples: int
    loops: int
    diffusion_steps: int
    msa_depth: int | None
    max_msa_depth: int | None
    msa_column_mask_rate: float


def classify_random_events(
    shapes: TapeShapes,
    *,
    initial_pair: list[np.ndarray],
    dropout: list[np.ndarray],
    rand: list[np.ndarray],
    randperm: list[np.ndarray],
    normal: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and name forward-only native draws.

    This intentionally accepts no "close enough" shape.  A changed upstream
    random consumer must be inspected before its tape can be replayed.
    """

    pair_shape = (shapes.batch, shapes.tokens, shapes.tokens, shapes.pair_width)
    atom_shape = (shapes.batch * shapes.samples, shapes.atoms, 3)
    rotation_shape = (shapes.batch * shapes.samples, 4)
    translation_shape = (shapes.batch * shapes.samples, 1, 3)
    if len(initial_pair) != 1 or tuple(initial_pair[0].shape) != pair_shape:
        raise ValueError(
            f"expected one initial pair draw {pair_shape}, got {[x.shape for x in initial_pair]}"
        )
    if len(dropout) != shapes.loops or any(
        tuple(x.shape) != pair_shape for x in dropout
    ):
        raise ValueError(
            f"expected {shapes.loops} LM dropout masks {pair_shape}, got {[x.shape for x in dropout]}"
        )
    expected_rand = (
        1 if (shapes.msa_depth or 0) > 1 and shapes.msa_column_mask_rate > 0 else 0
    )
    token_shape = (shapes.batch, shapes.tokens)
    if len(rand) != expected_rand or any(tuple(x.shape) != token_shape for x in rand):
        raise ValueError(
            f"expected {expected_rand} MSA column draws {token_shape}, got {[x.shape for x in rand]}"
        )
    needs_rows = bool(
        shapes.msa_depth
        and shapes.max_msa_depth
        and shapes.msa_depth > shapes.max_msa_depth
    )
    expected_rows = shapes.loops if needs_rows else 0
    row_shape = ((shapes.msa_depth or 0) - 1,)
    if len(randperm) != expected_rows or any(
        tuple(x.shape) != row_shape for x in randperm
    ):
        raise ValueError(
            f"expected {expected_rows} MSA row permutations {row_shape}, got {[x.shape for x in randperm]}"
        )
    expected_normals = 1 + 3 * shapes.diffusion_steps
    if len(normal) != expected_normals:
        raise ValueError(
            f"expected {expected_normals} diffusion normal draws, got {[x.shape for x in normal]}"
        )
    if tuple(normal[0].shape) != atom_shape:
        raise ValueError(
            f"initial diffusion draw must be {atom_shape}, got {normal[0].shape}"
        )
    rotations, translations, churn = normal[1::3], normal[2::3], normal[3::3]
    if any(tuple(x.shape) != rotation_shape for x in rotations):
        raise ValueError("unexpected diffusion rotation draw shape")
    if any(tuple(x.shape) != translation_shape for x in translations):
        raise ValueError("unexpected diffusion translation draw shape")
    if any(tuple(x.shape) != atom_shape for x in churn):
        raise ValueError("unexpected diffusion churn draw shape")
    result = {
        "initial_pair_state": initial_pair[0],
        "lm_dropout_masks": np.stack(dropout),
        "msa_column_keep": np.asarray(rand[0] >= shapes.msa_column_mask_rate)
        if rand
        else np.empty((0,), bool),
        "msa_row_choices": np.stack(
            [
                np.sort(
                    np.concatenate(([0], x[: (shapes.max_msa_depth or 1) - 1] + 1))
                ).astype(np.int64)
                for x in randperm
            ]
        )
        if randperm
        else np.empty((0,), np.int64),
        "diffusion_initial_normal": normal[0],
        "diffusion_rotation_quaternions": np.stack(rotations),
        "diffusion_translations": np.stack(translations),
        "diffusion_churn_normals": np.stack(churn),
    }
    return result


def _tree_identity(root: Path, suffixes: tuple[str, ...]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    }


def _run_identity(args) -> dict[str, Any]:
    import foldjax

    root = Path(foldjax.__file__).resolve().parent
    source = {"foldjax_root": str(root), "foldjax": _tree_identity(root, (".py",))}
    if args.command == "capture":
        native = args.upstream_source_root.resolve() / "src"
        source.update(native_root=str(native), native=_tree_identity(native, (".py",)))
    return {
        "runner": _sha256(Path(__file__)),
        "source": source,
        "checkpoint": _tree_identity(args.weights.resolve(), (".json", ".safetensors")),
        "input": _sha256(args.input_features),
    }


def _save_outputs(directory, prefix, output, *, native):
    names = (
        "plddt",
        "plddt_per_atom",
        "plddt_ca",
        "complex_plddt",
        "complex_iplddt",
        "ptm",
        "iptm",
        "pair_chains_iptm",
        "pae",
        "pde",
        "plddt_logits",
        "pae_logits",
        "pde_logits",
        "resolved_logits",
    )
    convert = TorchRecorder._array if native else np.asarray
    confidence = {name: convert(output[name]) for name in names if name in output}
    if not {"plddt", "complex_plddt", "ptm", "iptm"} <= confidence.keys():
        raise ValueError("native-compatible confidence outputs are missing")
    coords = {"coords": convert(output["sample_atom_coords"])}
    schema = {
        "schema_version": 1,
        "missing_optional": sorted(set(names) - confidence.keys()),
        "arrays": {},
    }
    for kind, values in (("coords", coords), ("confidence", confidence)):
        path = directory / f"{prefix}_{kind}.npz"
        _save_npz(path, values)
        schema["arrays"][path.name] = {
            "sha256": _sha256(path),
            "fields": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in values.items()
            },
        }
    return schema


def _verify_reference_outputs(directory, capture):
    schema = capture.get("output_schema")
    if not schema or schema.get("schema_version") != 1:
        raise ValueError("capture lacks the raw output schema")
    for filename, artifact in schema["arrays"].items():
        if (
            Path(filename).name != filename
            or _sha256(directory / filename) != artifact["sha256"]
        ):
            raise ValueError("native output artifact differs from capture")
        arrays = _npz(directory / filename)
        fields = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        if fields != artifact["fields"]:
            raise ValueError("native output schema differs from capture")


def compare_saved_output_bytes(first, second):
    result = {}
    for filename in ("jax_coords.npz", "jax_confidence.npz"):
        left, right = _npz(first / filename), _npz(second / filename)
        leaves = {}
        for name in sorted(left.keys() | right.keys()):
            if name not in left or name not in right:
                leaves[name] = {
                    "storage_bytes_equal": False,
                    "missing": True,
                    "both_finite": False,
                }
                continue
            a, b = left[name], right[name]
            leaves[name] = {
                "shape_equal": a.shape == b.shape,
                "dtype_equal": a.dtype == b.dtype,
                "both_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
                "storage_bytes_equal": (
                    a.shape == b.shape
                    and a.dtype == b.dtype
                    and a.tobytes() == b.tobytes()
                ),
            }
        result[filename] = {
            "leaves": leaves,
            "all_storage_bytes_equal": bool(leaves)
            and all(value["storage_bytes_equal"] for value in leaves.values()),
            "all_finite": bool(leaves)
            and all(value["both_finite"] for value in leaves.values()),
        }
    return result


def compile_replay(predict, positional, static, dynamic):
    """Compile once; subsequent calls omit the JIT's static keyword arguments."""
    return predict.lower(*positional, **static, **dynamic).compile()


def _save_lm(directory, prefix, value, *, native):
    original_dtype = str(value.dtype).removeprefix("torch.")
    array = TorchRecorder._array(value) if native else np.asarray(value)
    if original_dtype == "bfloat16":
        array = array.astype(np.float32)
    if (
        original_dtype not in ("bfloat16", "float32", "float16")
        or not np.isfinite(array).all()
    ):
        raise ValueError("LM output must be finite FP32/BF16/FP16")
    path = directory / f"{prefix}_lm.npz"
    _save_npz(path, {"lm_hidden_states": array})
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "storage_dtype": str(array.dtype),
        "original_dtype": original_dtype,
    }


def _read_lm(directory, schema, expected_shape):
    path = directory / schema["filename"]
    if (
        Path(schema["filename"]).name != schema["filename"]
        or _sha256(path) != schema["sha256"]
    ):
        raise ValueError("native LM artifact identity differs")
    arrays = _npz(path)
    if set(arrays) != {"lm_hidden_states"}:
        raise ValueError("native LM archive has unexpected fields")
    value = arrays["lm_hidden_states"]
    if (
        list(value.shape) != schema["shape"]
        or tuple(value.shape) != tuple(expected_shape)
        or str(value.dtype) != schema["storage_dtype"]
        or not np.isfinite(value).all()
    ):
        raise ValueError("native LM shape/dtype/value contract differs")
    import jax.numpy as jnp

    if schema["original_dtype"] not in ("bfloat16", "float32", "float16"):
        raise ValueError("unsupported original native LM dtype")
    restored = jnp.asarray(value, dtype=schema["original_dtype"])
    if not np.array_equal(np.asarray(restored, dtype=value.dtype), value):
        raise ValueError("native LM storage is not lossless for original dtype")
    return restored


@contextmanager
def _observe_native_lm(model, callback):
    """Observe the existing return once; do not recompute or draw randomness."""
    name = "_compute_lm_hidden_states"
    owned = name in model.__dict__
    previous = model.__dict__.get(name)
    original = getattr(model, name)
    calls = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(callback(result))
        return result

    setattr(model, name, observe)
    try:
        yield calls
    finally:
        if owned:
            setattr(model, name, previous)
        else:
            delattr(model, name)


class TorchRecorder:
    """Temporary torch patch set that records draws without perturbing RNG."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.initial_pair: list[np.ndarray] = []
        self.dropout: list[np.ndarray] = []
        self.rand: list[np.ndarray] = []
        self.randperm: list[np.ndarray] = []
        self.normal: list[np.ndarray] = []
        self._undo: list[tuple[Any, str, Any]] = []

    def _patch(self, target: Any, name: str, replacement: Any) -> None:
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy().copy()

    def __enter__(self) -> TorchRecorder:
        torch = self.torch
        original_trunc = torch.nn.init.trunc_normal_
        original_dropout = torch.nn.functional.dropout
        original_rand, original_randperm = torch.rand, torch.randperm
        original_randn, original_randn_like = torch.randn, torch.randn_like

        def trunc(tensor: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_trunc(tensor, *args, **kwargs)
            self.initial_pair.append(self._array(result))
            return result

        def dropout(
            input: Any, p: float = 0.5, training: bool = True, inplace: bool = False
        ) -> Any:
            before = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else torch.get_rng_state()
            )
            result = original_dropout(input, p=p, training=training, inplace=inplace)
            after = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else torch.get_rng_state()
            )
            try:
                if input.is_cuda:
                    torch.cuda.set_rng_state(before, input.device)
                else:
                    torch.set_rng_state(before)
                replay = original_dropout(
                    torch.ones_like(input), p=p, training=training, inplace=False
                )
                if training and p > 0:
                    self.dropout.append(self._array(replay != 0))
            finally:
                if input.is_cuda:
                    torch.cuda.set_rng_state(after, input.device)
                else:
                    torch.set_rng_state(after)
            return result

        def rand(*args: Any, **kwargs: Any) -> Any:
            result = original_rand(*args, **kwargs)
            self.rand.append(self._array(result))
            return result

        def randperm(*args: Any, **kwargs: Any) -> Any:
            result = original_randperm(*args, **kwargs)
            self.randperm.append(self._array(result))
            return result

        def randn(*args: Any, **kwargs: Any) -> Any:
            result = original_randn(*args, **kwargs)
            self.normal.append(self._array(result))
            return result

        def randn_like(*args: Any, **kwargs: Any) -> Any:
            result = original_randn_like(*args, **kwargs)
            self.normal.append(self._array(result))
            return result

        self._patch(torch.nn.init, "trunc_normal_", trunc)
        self._patch(torch.nn.functional, "dropout", dropout)
        self._patch(torch, "rand", rand)
        self._patch(torch, "randperm", randperm)
        self._patch(torch, "randn", randn)
        self._patch(torch, "randn_like", randn_like)
        return self

    def __exit__(self, *unused: object) -> None:
        for target, name, original in reversed(self._undo):
            setattr(target, name, original)


@contextmanager
def observe_native_injection(
    model, *, enabled=False, capture_msa_inputs=False, capture_coda=False
):
    values, counts, handles = {}, {}, []

    def observer(name):
        def hook(module, positional, output):
            counts[name] = counts.get(name, 0) + 1
            if capture_msa_inputs and name == "trunk":
                values["trunk_output_last"] = output.clone()
            if counts[name] != 1:
                return
            if name == "coda":
                values["coda_input"] = positional[0].clone()
                values["coda_output"] = output.clone()
            elif name == "injection":
                values["injection_input"] = positional[0].clone()
                values["injection_output"] = output.clone()
            elif name == "trunk":
                values["trunk_input"] = positional[0].clone()
                if capture_msa_inputs:
                    values["trunk_output"] = output.clone()
            else:
                values[name + "_output"] = output.clone()

        return hook

    try:
        if enabled:
            if capture_coda:
                handles.append(
                    model.parcae_coda.register_forward_hook(observer("coda"))
                )
            if capture_msa_inputs:

                def msa_inputs(module, positional, keywords):
                    counts["msa_inputs"] = counts.get("msa_inputs", 0) + 1
                    if counts["msa_inputs"] == 1:
                        values["msa_input_pair"] = keywords["x_pair"].clone()
                        values["msa_input_embedding"] = keywords["x_inputs"].clone()

                handles.append(
                    model.msa_encoder.register_forward_pre_hook(
                        msa_inputs, with_kwargs=True
                    )
                )
            for name, attr in (
                ("msa", "msa_encoder"),
                ("lm", "lm_encoder"),
                ("injection", "parcae_input_norm"),
                ("trunk", "folding_trunk"),
            ):
                module = getattr(model, attr)
                if module is None:
                    raise ValueError("injection capture requires active " + attr)
                handles.append(module.register_forward_hook(observer(name)))
        yield values, counts
    finally:
        for handle in reversed(handles):
            handle.remove()


def predict_with_injection_capture(
    predict, module, *args, capture_msa_inputs=False, capture_coda=False, **kwargs
):
    """Expose actual scan-body boundaries without reimplementing loop arithmetic."""
    import jax

    original_scan = jax.lax.scan
    original_msa = module.msa_encoder
    original_trunk = module.folding_trunk
    original_norm = module.trunk_ops._autocast_norm
    inside = False
    leaves, captured, calls = {}, {}, []

    def msa(*a, **kw):
        result = original_msa(*a, **kw)
        if inside:
            leaves["msa_output"] = result
            if capture_msa_inputs:
                leaves["msa_input_pair"], leaves["msa_input_embedding"] = a[:2]
        return result

    def trunk(x, params, prefix, **kw):
        result = original_trunk(x, params, prefix, **kw)
        if capture_coda and prefix == "parcae_coda":
            if "coda_output" in captured:
                raise ValueError("duplicate coda capture")
            captured["coda_input"], captured["coda_output"] = x, result
        if inside:
            if prefix == "lm_encoder":
                leaves["lm_output"] = result
            elif prefix == "folding_trunk":
                leaves["trunk_input"] = x
                if capture_msa_inputs:
                    leaves["trunk_output"] = result
        return result

    def norm(x, params, prefix, *a, **kw):
        result = original_norm(x, params, prefix, *a, **kw)
        if inside and prefix == "parcae_input_norm":
            leaves["injection_input"], leaves["injection_output"] = x, result
        return result

    def scan(body, init, xs=None, **kw):
        if not getattr(body, "__qualname__", "").endswith("run_loops.<locals>.body"):
            return original_scan(body, init, xs, **kw)
        calls.append(True)

        def observed(carry, inputs):
            nonlocal inside
            leaves.clear()
            inside = True
            try:
                next_carry, output = body(carry, inputs)
            finally:
                inside = False
            expected = {
                "msa_output",
                "lm_output",
                "injection_input",
                "injection_output",
                "trunk_input",
            }
            if capture_msa_inputs:
                expected |= {"msa_input_pair", "msa_input_embedding", "trunk_output"}
            if set(leaves) != expected:
                raise ValueError("JAX injection boundaries missing")
            return next_carry, (output, dict(leaves))

        carry, (outputs, boundaries) = original_scan(observed, init, xs, **kw)
        captured.update({k: v[0] for k, v in boundaries.items()})
        if capture_msa_inputs:
            captured["trunk_output_last"] = boundaries["trunk_output"][-1]
        return carry, outputs

    with (
        patch.object(module, "msa_encoder", msa),
        patch.object(module, "folding_trunk", trunk),
        patch.object(module.trunk_ops, "_autocast_norm", norm),
        patch.object(jax.lax, "scan", scan),
    ):
        output = predict(*args, **kwargs)
    if len(calls) != 1:
        raise ValueError("requires exactly one observed run_loops scan")
    if capture_coda and "coda_output" not in captured:
        raise ValueError("missing coda capture")
    return {**output, "diagnostic_injection": captured}


def _capture(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binding = _run_identity(args)
    upstream = args.upstream_source_root.resolve()
    source_file = (
        upstream
        / "src"
        / "transformers"
        / "models"
        / "esmfold2"
        / "modeling_esmfold2.py"
    )
    if not source_file.is_file():
        raise SystemExit(f"not an upstream transformers source root: {source_file}")
    sys.path.insert(0, str(upstream / "src"))
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    if Path(inspect.getfile(ESMFold2Model)).resolve() != source_file.resolve():
        raise RuntimeError("native model imported from a different source root")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.native_policy = _native_policy(
        torch, deterministic=getattr(args, "deterministic", False)
    )
    features = _npz(args.input_features)
    config = json.loads((args.weights / "config.json").read_text())
    model = ESMFold2Model.from_pretrained(str(args.weights), load_esmc=False)
    model.load_esmc(str(args.weights / "esmc"), precision="bf16")
    model = model.to("cuda").eval()
    device_features = {
        name: torch.as_tensor(value, device="cuda")
        for name, value in _model_features(features).items()
    }
    shape = TapeShapes(
        batch=int(features["token_attention_mask"].shape[0]),
        tokens=int(features["token_attention_mask"].shape[1]),
        atoms=int(features["atom_attention_mask"].shape[1]),
        pair_width=int(config["d_pair"]),
        samples=SAMPLES,
        loops=int(config.get("num_loops", 3)) + 1,
        diffusion_steps=_schedule_steps(model),
        msa_depth=None if "msa" not in features else int(features["msa"].shape[1]),
        max_msa_depth=config.get("msa_max_depth", config.get("max_msa_depth", 1024)),
        msa_column_mask_rate=float(config.get("msa_column_mask_rate", 0.1)),
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with (
        torch.no_grad(),
        TorchRecorder(torch) as recorder,
        _observe_native_lm(
            model,
            lambda value: _save_lm(args.output_dir, "upstream", value, native=True),
        ) as lm_calls,
        observe_native_injection(
            model,
            enabled=getattr(args, "capture_injection", False),
            capture_msa_inputs=getattr(args, "capture_msa_inputs", False),
            capture_coda=getattr(args, "capture_coda", False),
        ) as injection_capture,
    ):
        output = model(**device_features, num_diffusion_samples=SAMPLES)
    if len(lm_calls) != 1:
        raise ValueError("native capture requires exactly one LM output")
    args.lm_schema = lm_calls[0]
    expected_lm_shape = [
        shape.batch,
        shape.tokens,
        config["lm_num_layers"] + 1,
        config["lm_d_model"],
    ]
    if args.lm_schema["shape"] != expected_lm_shape:
        raise ValueError("native LM output has wrong token/layer/width shape")
    args.lm_arm = "native_independent_lm"
    tape = classify_random_events(
        shape,
        initial_pair=recorder.initial_pair,
        dropout=recorder.dropout,
        rand=recorder.rand,
        randperm=recorder.randperm,
        normal=recorder.normal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(args.output_dir / "features.npz", features)
    _save_npz(args.output_dir / "tape.npz", tape)
    args.output_schema = _save_outputs(args.output_dir, "upstream", output, native=True)
    if getattr(args, "capture_injection", False):
        values, counts = injection_capture
        expected_counts = {"msa", "lm", "injection", "trunk"}
        if getattr(args, "capture_msa_inputs", False):
            expected_counts.add("msa_inputs")
        if getattr(args, "capture_coda", False):
            expected_counts.add("coda")
        if set(counts) != expected_counts or any(
            count != (1 if name == "coda" else shape.loops)
            for name, count in counts.items()
        ):
            raise ValueError("native injection loop coverage differs")
        path = args.output_dir / "injection.npz"
        _save_npz(
            path, {k: v.detach().float().cpu().numpy() for k, v in values.items()}
        )
        args.injection_schema = {
            "filename": path.name,
            "sha256": _sha256(path),
            "counts": counts,
            "dtypes": {k: str(v.dtype) for k, v in values.items()},
            "scope": "first-loop observed native boundaries; no admission",
        }
    if binding != _run_identity(args):
        raise RuntimeError("source/checkpoint/input changed during capture")
    args.binding = binding
    _write_metadata(
        args,
        config,
        {
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "python_executable": sys.executable,
            "transformers": importlib.metadata.version("transformers"),
        },
    )


def _write_metadata(
    args: argparse.Namespace, config: Mapping[str, Any], versions: Mapping[str, str]
) -> None:
    lm_schema = getattr(args, "lm_schema", None)
    injection = getattr(args, "injection_schema", None)
    if injection is not None and (
        Path(injection["filename"]).name != injection["filename"]
        or _sha256(args.output_dir / injection["filename"]) != injection["sha256"]
    ):
        raise ValueError("saved injection artifact changed before completion")
    if (
        lm_schema is not None
        and _sha256(args.output_dir / lm_schema["filename"]) != lm_schema["sha256"]
    ):
        raise ValueError("saved LM artifact changed before completion")
    shim = getattr(args, "native_shim_control", None)
    if shim is not None and (
        Path(shim["filename"]).name != shim["filename"]
        or _sha256(args.output_dir / shim["filename"]) != shim["sha256"]
    ):
        raise ValueError("saved native shim artifact changed before completion")
    embedding = getattr(args, "native_input_control", None)
    if embedding is not None:
        validate_saved_input_control(args.output_dir, embedding, embedding["shape"])
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "samples": SAMPLES,
                "core_only_shared_features": True,
                "full_model_admission": None,
                "lm_arm": getattr(args, "lm_arm", "legacy_unrecorded_lm"),
                "lm_schema": getattr(args, "lm_schema", None),
                "injection_schema": getattr(args, "injection_schema", None),
                "native_policy": getattr(args, "native_policy", None),
                "jax_policy": getattr(args, "jax_policy", None),
                "repeat_forward": getattr(args, "repeat_forward", None),
                "native_shim_control": getattr(args, "native_shim_control", None),
                "native_input_control": embedding,
                "binding": getattr(args, "binding", None),
                "output_schema": getattr(args, "output_schema", None),
                "input_sha256": _sha256(args.input_features),
                "precision": {
                    "trunk": (
                        "native_cuda_bfloat16_autocast"
                        if args.command == "capture"
                        else "bfloat16_trunk_native_policy_target"
                    ),
                    "checkpoint_dtype": config.get("dtype"),
                    "conditioning": (
                        "native_mixed_with_bfloat16_z_transitions"
                        if args.command == "capture"
                        else "port_policy_not_native_verified"
                    ),
                    "language_model": "bfloat16",
                    "matmul": "highest",
                },
                "tape_sha256": _sha256(
                    args.tape
                    if args.command == "replay"
                    else args.output_dir / "tape.npz"
                ),
                "config": config,
                "versions": dict(versions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _native_policy(torch, *, deterministic=False):
    """Opt-in diagnostic; never silently install an environment or kernel policy."""
    environment = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        )
    }
    if deterministic:
        if environment["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ValueError(
                "deterministic native control requires explicit CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before launch"
            )
        if any(
            environment[name] not in (None, "0")
            for name in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
        ):
            raise ValueError(
                "deterministic native control rejects TF32 environment overrides"
            )
        if torch.cuda.is_initialized():
            raise ValueError(
                "deterministic native control must be configured before CUDA initialization"
            )
        torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "deterministic_requested": deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "environment": environment,
        "scope": "native determinism diagnostic only; not a new port acceptance tolerance",
    }


def _replay_settings(settings):
    from foldjax.models.esmfold2.models import model

    # Retain native raw confidence heads; this changes output retention only,
    # not trunk/sampler precision, schedule, or confidence-head arithmetic.
    return replace(
        model.with_overrides(settings, num_samples=SAMPLES),
        return_confidence_logits=True,
    )


def _load_replay_lm(args, inference, features, capture):
    interchange = getattr(args, "native_lm", False)
    if interchange and not capture.get("lm_schema"):
        raise ValueError("native LM interchange requires this capture's LM artifact")
    loaded = inference.load(
        args.weights,
        dtype="float32",
        esmc_dtype="bfloat16",
        language_model=not interchange,
    )
    expected = (
        *features["token_attention_mask"].shape,
        capture["config"]["lm_num_layers"] + 1,
        capture["config"]["lm_d_model"],
    )
    lm = (
        _read_lm(args.tape.parent, capture["lm_schema"], expected)
        if interchange
        else inference.language_model_states(features, loaded)
    )
    if lm is None or lm.shape != expected:
        raise ValueError("candidate LM output has wrong token/layer/width shape")
    args.lm_schema = _save_lm(args.output_dir, "jax", lm, native=False)
    args.lm_arm = (
        "native_lm_interchange_downstream_core_only"
        if interchange
        else "jax_independent_lm"
    )
    return loaded, lm, expected


def configure_autotune_control(output_dir, *, enabled=False, load=None):
    """Select fresh executable caches before JAX import; never change defaults."""
    if not enabled and load is None:
        return None
    if "jax" in sys.modules:
        raise RuntimeError("autotune control must be configured before importing JAX")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("autotune control requires a fresh output directory")
    flags = os.environ.get("XLA_FLAGS", "")
    if any(term in flags for term in ("autotune_results", "autotune_cache")):
        raise ValueError("inherited autotune flags conflict with capture controls")
    source = Path(load).resolve(strict=True) if load is not None else None
    if any(any(c.isspace() for c in str(p)) for p in (output_dir, source) if p):
        raise ValueError("autotune flag paths must not contain whitespace")
    source_hash = _sha256(source) if source is not None else None
    dump = output_dir / "xla-autotune.textproto"
    flags += f" --xla_gpu_dump_autotune_results_to={dump}"
    if source is not None:
        flags += f" --xla_gpu_load_autotune_results_from={source}"
        flags += " --xla_gpu_require_complete_aot_autotune_results=true"
    os.environ["XLA_FLAGS"] = flags.strip()
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(output_dir / "jax-cache")
    os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"
    return {
        "scope": "fresh-process autotune control; separate executable caches",
        "load": str(source) if source is not None else None,
        "load_sha256": source_hash,
        "dump": str(dump),
        "strict_complete_load": source is not None,
    }


def finish_autotune_control(control):
    if control is None:
        return None
    if (
        control["load"] is not None
        and _sha256(Path(control["load"])) != control["load_sha256"]
    ):
        raise RuntimeError("loaded autotune artifact changed during execution")
    dump = Path(control["dump"])
    if not dump.is_file() or not dump.stat().st_size:
        raise RuntimeError("autotune capture is missing or empty")
    return {**control, "dump_sha256": _sha256(dump)}


def _replay(args: argparse.Namespace) -> None:
    # Keep this function torch-free: the installed FoldJAX package must remain so.
    import inspect

    if getattr(args, "native_shim", None) and not getattr(args, "native_lm", False):
        raise ValueError("native shim control requires --native-lm")
    if getattr(args, "native_input_embedding", None) and not (
        getattr(args, "native_lm", False) and getattr(args, "native_shim", None)
    ):
        raise ValueError("native input control requires --native-lm and --native-shim")
    autotune = configure_autotune_control(
        getattr(args, "output_dir", None),
        enabled=getattr(args, "capture_autotune", False),
        load=getattr(args, "xla_autotune_load", None),
    )
    import jax

    from foldjax.models.esmfold2 import inference
    from foldjax.models.esmfold2.models import model as structure_model

    repeats = getattr(args, "repeat_forwards", 1)
    if isinstance(repeats, bool) or repeats not in (1, 2, 3):
        raise ValueError("repeat_forwards must be 1, 2 or 3")
    args.jax_policy = {
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "scope": "recorded execution policy; not a new default or acceptance gate",
    }

    required = {
        "initial_pair_state",
        "lm_dropout_masks",
        "msa_column_keep",
        "msa_row_choices",
        "diffusion_initial_normal",
        "diffusion_rotation_quaternions",
        "diffusion_translations",
        "diffusion_churn_normals",
    }
    missing = required - set(inspect.signature(structure_model.predict).parameters)
    if missing:
        raise RuntimeError(
            "ESMFold2 full tape replay is not implemented by this installed core; "
            f"missing inputs: {sorted(missing)}. Capture/classification only."
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binding = _run_identity(args)
    reference_path = args.tape.parent / "metadata.json"
    reference_sha = _sha256(reference_path)
    jax.config.update("jax_default_matmul_precision", "highest")
    args.jax_policy["matmul"] = str(jax.config.jax_default_matmul_precision)
    capture = json.loads((args.tape.parent / "metadata.json").read_text())
    if (
        not capture.get("binding")
        or capture["binding"]["checkpoint"] != binding["checkpoint"]
    ):
        raise ValueError("replay checkpoint identity differs or capture lacks binding")
    _verify_reference_outputs(args.tape.parent, capture)
    if capture["input_sha256"] != _sha256(args.input_features):
        raise ValueError("replay input differs from captured input")
    if capture["tape_sha256"] != _sha256(args.tape):
        raise ValueError("replay tape differs from capture manifest")
    if capture["config"] != json.loads((args.weights / "config.json").read_text()):
        raise ValueError("replay checkpoint configuration differs from capture")
    features, tape = _npz(args.input_features), _npz(args.tape)
    if set(tape) != required:
        raise ValueError("replay requires exactly the eight classified tape arrays")
    interchange = getattr(args, "native_lm", False)
    loaded, lm, expected_lm_shape = _load_replay_lm(args, inference, features, capture)
    settings = _replay_settings(loaded.settings)
    structure_model.validate_initial_pair_state(
        tape["initial_pair_state"],
        batch=features["token_attention_mask"].shape[0],
        tokens=features["token_attention_mask"].shape[1],
        width=settings.d_pair,
    )
    structure_model.validate_msa_tape(
        tape["msa_column_keep"],
        tape["msa_row_choices"],
        batch=features["token_attention_mask"].shape[0],
        tokens=features["token_attention_mask"].shape[1],
        depth=features["msa"].shape[1] if "msa" in features else None,
        loops=max(1, settings.num_recycles + 1),
        settings=settings,
    )
    structure_model.diffusion.validate_diffusion_tape(
        tape["diffusion_initial_normal"],
        tape["diffusion_rotation_quaternions"],
        tape["diffusion_translations"],
        tape["diffusion_churn_normals"],
        steps=len(structure_model.diffusion.noise_schedule(settings.diffusion)) - 1,
        batch=features["token_attention_mask"].shape[0] * SAMPLES,
        atoms=features["atom_attention_mask"].shape[-1],
    )
    if settings.trunk_dtype != "bfloat16":
        raise ValueError("native CUDA replay requires the BF16 trunk policy")
    arrays = {
        name: jax.numpy.asarray(value)
        for name, value in _model_features(features).items()
    }
    positional = (jax.random.key(SEED), arrays, loaded.parameters)
    dynamic = {
        "lm_hidden_states": lm,
        **{name: jax.numpy.asarray(value) for name, value in tape.items()},
    }
    static = {"settings": settings, "n_chains": int(features["asym_id"].max()) + 1}
    shim_bindings = {}
    predict_function = structure_model.predict
    if getattr(args, "native_shim", None):
        batch, tokens = features["token_attention_mask"].shape
        pair, shim_bindings = load_native_shim_control(
            args.native_shim,
            capture,
            args.tape.parent,
            args.weights,
            (batch, tokens, tokens, settings.d_pair),
        )
        pair_path = args.output_dir / "native_shim_control.npz"
        _save_npz(pair_path, {"pair": pair})
        args.native_shim_control = {
            "scope": "native pre-dropout shim pair interchange; downstream diagnostic only",
            "full_model_admission": None,
            "filename": pair_path.name,
            "sha256": _sha256(pair_path),
            "shape": list(pair.shape),
            "dtype": str(pair.dtype),
            "bindings": shim_bindings,
        }
        dynamic["diagnostic_shim_pair"] = jax.numpy.asarray(pair)

        def predict_function(*a, diagnostic_shim_pair, **kw):
            return predict_with_native_shim(
                structure_model.predict, structure_model, diagnostic_shim_pair, *a, **kw
            )

    from functools import partial

    input_bindings = {}
    if getattr(args, "native_input_embedding", None):
        batch, tokens = features["token_attention_mask"].shape
        embedding, input_bindings = load_native_input_control(
            args.native_input_embedding,
            args.tape.parent,
            args.weights,
            (batch, tokens, settings.d_inputs),
        )
        path = args.output_dir / "native_input_control.npz"
        _save_npz(path, {"embedding": embedding})
        args.native_input_control = {
            "scope": "native input embedding substituted; downstream diagnostic only",
            "full_model_admission": None,
            "filename": path.name,
            "sha256": _sha256(path),
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
            "bindings": input_bindings,
        }
        dynamic["diagnostic_input_embedding"] = jax.numpy.asarray(embedding)
        inner_predict = predict_function

        def predict_function(*a, diagnostic_input_embedding, **kw):
            return predict_with_native_inputs(
                inner_predict, structure_model, diagnostic_input_embedding, *a, **kw
            )

    from bench.esmfold2_lm_encoder_candidate import compiler_control, ffi_output_linear
    from foldjax.models.esmfold2.models import trunk

    if getattr(args, "capture_injection", False):
        unobserved_predict = predict_function

        def predict_function(*a, **kw):
            return predict_with_injection_capture(
                unobserved_predict,
                structure_model,
                *a,
                capture_msa_inputs=getattr(args, "capture_msa_inputs", False),
                capture_coda=getattr(args, "capture_coda", False),
                **kw,
            )

    options = compiler_control(getattr(args, "compiler_profile", "default"))
    policy_path = Path(inspect.getfile(compiler_control))
    policy_hash = _sha256(policy_path)
    ffi_bindings = {}
    original_linear = trunk._autocast_linear
    library = getattr(args, "ffi_library", None)
    if library:
        from bench import native_cublaslt_ffi

        if getattr(args, "xla_autotune_load", None):
            raise ValueError("FFI replay must not load a separate autotune control")
        ffi_paths = (
            library,
            Path(native_cublaslt_ffi.__file__),
            Path(__file__).with_name("native_cublaslt_ffi.cc"),
            Path(inspect.getfile(ffi_output_linear)),
        )
        ffi_bindings = {str(p.resolve()): _sha256(p) for p in ffi_paths}
        target = native_cublaslt_ffi.register(library)
        trunk._autocast_linear = partial(
            dispatch_bias_free_ffi,
            ffi=partial(ffi_output_linear, target=target),
            fallback=original_linear,
        )
    args.jax_policy.update(
        compiler_options=options,
        compiler_control_sha256=policy_hash,
        ffi_bindings=ffi_bindings,
        ffi_native_dispatch=bool(library),
        ffi_bias_policy="runtime_fallback" if library else None,
    )
    try:
        predict = jax.jit(
            predict_function,
            static_argnames=("settings", "n_chains"),
            compiler_options=options,
        )
        if repeats > 1:
            compiled = compile_replay(predict, positional, static, dynamic)
            (args.output_dir / "compiled.hlo.txt").write_text(compiled.as_text())
            output = compiled(*positional, **dynamic)
        else:
            output = predict(*positional, **static, **dynamic)
    finally:
        trunk._autocast_linear = original_linear
    jax.block_until_ready(output["sample_atom_coords"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_schema = _save_outputs(args.output_dir, "jax", output, native=False)
    if getattr(args, "capture_injection", False):
        values = output["diagnostic_injection"]
        path = args.output_dir / "injection.npz"
        arrays = {k: np.asarray(v.astype(jax.numpy.float32)) for k, v in values.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("nonfinite JAX injection boundaries")
        _save_npz(path, arrays)
        args.injection_schema = {
            "filename": path.name,
            "sha256": _sha256(path),
            "dtypes": {k: str(v.dtype) for k, v in values.items()},
            "scope": "first-loop observed JAX scan boundaries; no admission",
        }
    if repeats > 1:
        comparisons = []
        for index in range(1, repeats):
            repeated = compiled(*positional, **dynamic)
            jax.block_until_ready(repeated)
            directory = args.output_dir / f"repeat-{index}"
            directory.mkdir(exist_ok=False)
            schema = _save_outputs(directory, "jax", repeated, native=False)
            comparisons.append(
                {
                    "repeat_index": index,
                    "output_schema": schema,
                    "arrays": compare_saved_output_bytes(args.output_dir, directory),
                }
            )
            del repeated
        args.repeat_forward = {
            "scope": "same compiled executable and identical device operands",
            "count": repeats,
            "compiled_hlo_sha256": _sha256(args.output_dir / "compiled.hlo.txt"),
            "comparisons": comparisons,
            "full_model_admission": None,
        }
    if (
        binding != _run_identity(args)
        or reference_sha != _sha256(reference_path)
        or capture["tape_sha256"] != _sha256(args.tape)
        or any(_sha256(Path(p)) != h for p, h in ffi_bindings.items())
        or _sha256(policy_path) != policy_hash
        or any(_sha256(Path(p)) != h for p, h in shim_bindings.items())
        or any(_sha256(Path(p)) != h for p, h in input_bindings.items())
    ):
        raise RuntimeError("bound replay inputs changed during execution")
    args.binding = {**binding, "reference_manifest_sha256": reference_sha}
    _verify_reference_outputs(args.tape.parent, capture)
    if interchange:
        _read_lm(args.tape.parent, capture["lm_schema"], expected_lm_shape)
    args.jax_policy["autotune_control"] = finish_autotune_control(autotune)
    _write_metadata(
        args,
        json.loads((args.weights / "config.json").read_text()),
        {
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "python_executable": sys.executable,
        },
    )


def load_native_shim_control(root, capture, reference, weights, expected_shape):
    """Validate an explicit diagnostic interchange, never a runtime fallback."""
    from bench.esmfold2_lm_shim_candidate import validate_native

    report = validate_native(root, capture, reference, weights)
    if (
        report.get("precision_control")
        or report.get("allow_bf16_reduced_precision_reduction") is False
    ):
        raise ValueError("shim interchange requires native default precision")
    pair = _npz(root / "native.npz")["pair"]
    if (
        pair.shape != tuple(expected_shape)
        or pair.dtype != np.float32
        or not np.isfinite(pair).all()
    ):
        raise ValueError("native shim pair shape/dtype/finite contract differs")
    paths = [
        root / "native.npz",
        root / "report.json",
        Path(inspect.getfile(validate_native)),
    ]
    return pair, {str(p.resolve()): _sha256(p) for p in paths}


def dispatch_bias_free_ffi(x, params, prefix, *, ffi, fallback):
    """Keep biased projections on the runtime path, outside the FFI control."""
    return (fallback if prefix + ".bias" in params else ffi)(x, params, prefix)


def predict_with_native_shim(predict, module, pair, *args, **kwargs):
    """Trace with one dynamic native pair; restore the real helper on all exits."""
    original = module.language_model_pair
    calls = []

    def substitute(*unused, **ignored):
        calls.append(True)
        return pair

    module.language_model_pair = substitute
    try:
        result = predict(*args, **kwargs)
        if len(calls) != 1:
            raise ValueError("native shim control requires exactly one LM pair call")
        return result
    finally:
        module.language_model_pair = original


def load_native_input_control(root, reference, weights, expected_shape):
    """Consume a bound native embedding, never a rounded reconstruction."""
    if (root / "metadata.json").exists():
        return load_observed_native_input(root, reference, weights, expected_shape)
    report_path, archive = root / "report.json", root / "native.npz"
    report = json.loads(report_path.read_text())
    if _sha256(archive) != report["archive_sha256"]:
        raise ValueError("native input archive identity differs")
    bindings = report["bindings"]
    for path in (
        reference / "metadata.json",
        weights / "model.safetensors",
        weights / "config.json",
    ):
        if bindings.get(str(path.resolve())) != _sha256(path):
            raise ValueError("native input reference/checkpoint identity differs")
    if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("native input source/boundary binding differs")
    value = _npz(archive)["baseline"]
    if (
        value.shape != tuple(expected_shape)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("native input shape/dtype/finite contract differs")
    return value, {
        **bindings,
        str(report_path.resolve()): _sha256(report_path),
        str(archive.resolve()): _sha256(archive),
    }


def load_observed_native_input(root, reference, weights, expected_shape):
    paths = (root / "metadata.json", reference / "metadata.json")
    bindings = {str(p.resolve()): _sha256(p) for p in paths}
    observed, baseline = [json.loads(p.read_text()) for p in paths]
    for key in ("input_sha256", "tape_sha256", "config"):
        if observed[key] != baseline[key]:
            raise ValueError("observed input reference bridge differs: " + key)
    for key in ("checkpoint",):
        if observed["binding"][key] != baseline["binding"][key]:
            raise ValueError("observed input checkpoint bridge differs")
    if (
        observed["binding"]["source"]["native"]
        != baseline["binding"]["source"]["native"]
    ):
        raise ValueError("observed input native source bridge differs")
    for name, digest in observed["binding"]["checkpoint"].items():
        path = weights / name
        if (
            not path.resolve().is_relative_to(weights.resolve())
            or _sha256(path) != digest
        ):
            raise ValueError("observed input checkpoint artifact differs")
        bindings[str(path.resolve())] = digest
    for name in ("features.npz", "tape.npz", "upstream_lm.npz"):
        left, right = root / name, reference / name
        digest = _sha256(left)
        if digest != _sha256(right):
            raise ValueError("observed input archive bridge differs")
        bindings.update({str(left.resolve()): digest, str(right.resolve()): digest})
    _verify_reference_outputs(root, observed)
    schema = observed["injection_schema"]
    archive = root / "injection.npz"
    if (
        schema["filename"] != archive.name
        or _sha256(archive) != schema["sha256"]
        or schema["dtypes"].get("msa_input_embedding") != "torch.float32"
    ):
        raise ValueError("observed native embedding schema differs")
    bindings[str(archive.resolve())] = schema["sha256"]
    value = _npz(archive)["msa_input_embedding"]
    if (
        value.shape != tuple(expected_shape)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            "observed native embedding shape/dtype/finite contract differs"
        )
    if any(_sha256(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("observed input binding changed during load")
    return value, bindings


def validate_saved_input_control(root, control, expected_shape):
    if (
        control.get("filename") != "native_input_control.npz"
        or control.get("dtype") != "float32"
        or control.get("shape") != list(expected_shape)
        or control.get("full_model_admission") is not None
    ):
        raise ValueError("native input control schema differs")
    archive = root / control["filename"]
    if _sha256(archive) != control["sha256"]:
        raise ValueError("saved native input archive identity differs")
    arrays = _npz(archive)
    if set(arrays) != {"embedding"}:
        raise ValueError("native input control keys differ")
    value = arrays["embedding"]
    if (
        value.shape != tuple(expected_shape)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("saved native input shape/dtype/finite contract differs")


def predict_with_native_inputs(predict, module, embedding, *args, **kwargs):
    original, calls = module.inputs_embedding, []

    def substitute(*unused, **ignored):
        calls.append(True)
        return embedding

    module.inputs_embedding = substitute
    try:
        result = predict(*args, **kwargs)
        if len(calls) != 1:
            raise ValueError("native input control requires exactly one embedding call")
        return result
    finally:
        module.inputs_embedding = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "replay"):
        item = sub.add_parser(name)
        item.add_argument("--input-features", type=Path, required=True)
        item.add_argument("--weights", type=Path, required=True)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--capture-injection", action="store_true")
        item.add_argument("--capture-msa-inputs", action="store_true")
        item.add_argument("--capture-coda", action="store_true")
        if name == "capture":
            item.add_argument("--upstream-source-root", type=Path, required=True)
            item.add_argument(
                "--deterministic",
                action="store_true",
                help="Opt-in native deterministic control; requires prelaunch CUBLAS_WORKSPACE_CONFIG",
            )
        else:
            item.add_argument("--tape", type=Path, required=True)
            item.add_argument(
                "--repeat-forwards", type=int, choices=(1, 2, 3), default=1
            )
            item.add_argument("--capture-autotune", action="store_true")
            item.add_argument("--xla-autotune-load", type=Path)
            item.add_argument("--ffi-library", type=Path)
            item.add_argument("--native-input-embedding", type=Path)
            item.add_argument(
                "--native-shim",
                type=Path,
                help="Diagnostic native shim pair interchange; requires --native-lm",
            )
            item.add_argument(
                "--compiler-profile",
                default="default",
                choices=("default", "native-chunks-strict-rounding"),
            )
            item.add_argument(
                "--native-lm",
                action="store_true",
                help="Reuse this native capture's exact LM output; downstream-core-only diagnostic",
            )
    args = parser.parse_args()
    if args.capture_msa_inputs and not args.capture_injection:
        parser.error("--capture-msa-inputs requires --capture-injection")
    if args.capture_coda and not args.capture_injection:
        parser.error("--capture-coda requires --capture-injection")
    (_capture if args.command == "capture" else _replay)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
