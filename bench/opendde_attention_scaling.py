"""Same-QKV native Torch/cuEq attention controls; no model admission.

The native arm saves its actual divided queries. Replaying them with scale=1
isolates the query-scaling boundary without reconstructing rounded operands.
"""

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import save, sha


@contextmanager
def observe_native_matmuls(torch, attention):
    """Return the real TF32 result; compare a side call with identical operands.

    Intercept only direct calls from this publisher function, in QK/PV order.
    This is operator evidence, not an end-to-end policy or performance gate.
    """
    original = torch.matmul
    records, outputs = [], {}

    def observed(left, right, *args, **kwargs):
        if sys._getframe(1).f_code is not attention.__code__:
            return original(left, right, *args, **kwargs)
        enabled = torch.backends.cuda.matmul.allow_tf32
        if not enabled or args or kwargs or len(records) >= 2:
            raise ValueError(
                "expected exactly two native TF32 matmul calls without out"
            )
        name = ("qk", "pv")[len(records)]
        actual = original(left, right)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            reference = original(left, right)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = enabled
        # QK is subsequently mutated in-place when the publisher adds bias.
        a = actual.detach().cpu().numpy().copy()
        b = reference.detach().cpu().numpy().copy()
        outputs[f"{name}_tf32"] = a
        outputs[f"{name}_fp32_same_operands"] = b
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        records.append(
            {
                "call": name,
                "allow_tf32_actual": bool(enabled),
                "allow_tf32_reference": False,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
                "left_stride": list(left.stride()),
                "right_stride": list(right.stride()),
                "left_dtype": str(left.dtype),
                "right_dtype": str(right.dtype),
                "output_shape": list(a.shape),
                "output_dtype": str(a.dtype),
                "finite": finite,
                "bitwise_equal": a.dtype == b.dtype
                and a.shape == b.shape
                and a.tobytes() == b.tobytes(),
                "max_abs": float(
                    np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                )
                if finite
                else None,
            }
        )
        return actual

    with patch.object(torch, "matmul", observed):
        yield records, outputs
    if len(records) != 2:
        raise ValueError("missing native QK/PV matmul observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    if args.arm == "native":
        import torch
        from opendde.model.triangular.layers import _attention

        implementation_sha256 = sha(Path(_attention.__code__.co_filename))
        rng = np.random.default_rng(17)
        for tokens, width in ((32, 32), (32, 64), (437, 32)):
            shape = (1, 1, 4, tokens, width)
            values = {
                key: rng.normal(size=shape).astype(np.float32)
                for key in ("q", "k", "v")
            }
            values["bias"] = rng.normal(size=(1, 1, 4, tokens, tokens)).astype(
                np.float32
            )
            q, k, v, bias = (
                torch.from_numpy(values[key]).cuda() for key in ("q", "k", "v", "bias")
            )
            scaled = q.clone()
            scaled /= math.sqrt(width)
            values["q_scaled"] = scaled.cpu().numpy()
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                for enabled in (False, True):
                    torch.backends.cuda.matmul.allow_tf32 = enabled
                    with torch.inference_mode():
                        if enabled:
                            with observe_native_matmuls(torch, _attention) as (
                                calls,
                                call_outputs,
                            ):
                                result = _attention(scaled, k, v, [bias])
                        else:
                            result = _attention(scaled, k, v, [bias])
                    values["tf32" if enabled else "fp32"] = result.cpu().numpy()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            name = f"n{tokens}-d{width}.npz"
            np.savez_compressed(args.out / name, **values)
            matmul_name = f"n{tokens}-d{width}-matmuls.npz"
            np.savez_compressed(args.out / matmul_name, **call_outputs)
            records.append(
                {
                    "file": name,
                    "shape": shape,
                    "sha256": sha(args.out / name),
                    "matmul_file": matmul_name,
                    "matmul_sha256": sha(args.out / matmul_name),
                    "matmul_comparisons": calls,
                }
            )
        version = torch.__version__
    else:
        import jax
        import jax.numpy as jnp

        from foldjax.models._cueq import cueq_attention_core
        from foldjax.models.protenix.models.triangle.triangle import (
            _triangle_attention_block,
        )

        implementation_sha256 = sha(Path(cueq_attention_core.__code__.co_filename))
        if args.reference is None:
            parser.error("FoldJAX requires a native reference")
        source = json.loads((args.reference / "report.json").read_text())
        for record in source["records"]:
            with np.load(args.reference / record["file"]) as archive:
                values = dict(archive)
            operands = {key: jnp.asarray(value) for key, value in values.items()}
            width = record["shape"][-1]
            mask_bias = jnp.zeros((1, 1, 1, 1, record["shape"][-2]))
            metrics, outputs = {}, {}
            for precision, reference in (("high", "tf32"), ("highest", "fp32")):
                with jax.default_matmul_precision(precision):
                    for scaling in ("kernel", "native-query"):
                        if width > 32:
                            metrics[f"{precision}-{scaling}"] = {
                                "unsupported": "Installed cuEq FP32/TF32 requires D<=32"
                            }
                            continue
                        query = operands["q" if scaling == "kernel" else "q_scaled"]
                        scale = 1 / math.sqrt(width) if scaling == "kernel" else 1.0
                        result = np.asarray(
                            jax.jit(cueq_attention_core, static_argnames=("scale",))(
                                query,
                                operands["k"],
                                operands["v"],
                                operands["bias"],
                                mask_bias,
                                scale=scale,
                            )
                        )
                        key = f"{precision}-{scaling}"
                        outputs[key] = result
                        delta = result.astype(np.float64) - values[reference]
                        metrics[key] = {
                            "max_abs": float(abs(delta).max()),
                            "rmse": float(np.sqrt(np.mean(delta**2))),
                        }
                    result = np.asarray(
                        jax.jit(_triangle_attention_block)(
                            operands["q_scaled"],
                            operands["k"],
                            operands["v"],
                            mask_bias,
                            operands["bias"],
                        )
                    )
                    key = f"{precision}-xla-native-query"
                    outputs[key] = result
                    delta = result.astype(np.float64) - values[reference]
                    metrics[key] = {
                        "max_abs": float(abs(delta).max()),
                        "rmse": float(np.sqrt(np.mean(delta**2))),
                    }
            np.savez_compressed(args.out / record["file"], **outputs)
            records.append({**record, "comparisons": metrics})
            print(json.dumps(records[-1]), flush=True)
        version = jax.__version__
    save(
        args.out / "report.json",
        {
            "scope": __doc__,
            "arm": args.arm,
            "version": version,
            "records": records,
            "wrapper_sha256": sha(Path(__file__)),
            "implementation_sha256": implementation_sha256,
        },
    )


if __name__ == "__main__":
    main()
