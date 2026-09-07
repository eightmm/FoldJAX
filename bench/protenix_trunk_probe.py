"""Observe native trunk boundaries, then replay individual modules on those inputs.

Diagnostic only: module-input substitution is not end-to-end parity evidence.
The native arm retains the actual publisher runner and its precision policy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import save, sha


def native():
    from bench import protenix_closure_capture as capture

    original_install = capture.install_observers

    def install(stack, recorder, runner, *rest):
        original_install(stack, recorder, runner, *rest)
        original_predict = runner.InferenceRunner.predict

        def predict(self, data):
            roots = (
                "linear_no_bias_sinit",
                "linear_no_bias_zinit1",
                "linear_no_bias_zinit2",
                "relative_position_encoding",
                "linear_no_bias_token_bond",
                "msa_module",
                "pairformer_stack",
                "msa_module.blocks.0.outer_product_mean_msa",
            )
            blocks = ("msa_module.blocks.0.pair_stack", "pairformer_stack.blocks.0")
            names = (
                set(roots)
                | set(blocks)
                | {
                    f"{block}.{child}"
                    for block in blocks
                    for child in (
                        "tri_mul_out",
                        "tri_mul_in",
                        "tri_att_start",
                        "tri_att_end",
                        "pair_transition",
                    )
                }
            )
            modules = dict(self.model.named_modules())
            if not names <= modules.keys():
                raise ValueError(f"missing native modules: {names - modules.keys()}")
            counts = {name: 0 for name in names}
            handles = []

            def pre(name, module, args, kwargs):
                counts[name] += 1
                if counts[name] != 1:
                    return
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                bound.apply_defaults()
                recorder.bundle(f"stage__{name}__input", bound.arguments)

            def post(name, module, args, kwargs, result):
                if counts[name] == 1:
                    recorder.bundle(f"stage__{name}__output", {"result": result})

            for name in sorted(names):
                handles.append(
                    modules[name].register_forward_pre_hook(
                        lambda module, args, kwargs, name=name: pre(
                            name, module, args, kwargs
                        ),
                        with_kwargs=True,
                    )
                )
                handles.append(
                    modules[name].register_forward_hook(
                        lambda module, args, kwargs, result, name=name: post(
                            name, module, args, kwargs, result
                        ),
                        with_kwargs=True,
                    )
                )
            try:
                result = original_predict(self, data)
                if any(count < 1 for count in counts.values()):
                    raise ValueError(f"unobserved native boundaries: {counts}")
                capture.save(recorder.out / "stage-counts.json", counts)
                return result
            finally:
                for handle in handles:
                    handle.remove()

        stack.enter_context(patch.object(runner.InferenceRunner, "predict", predict))

    with patch.object(capture, "install_observers", install):
        capture.main()


def foldjax():
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.bridge.weights_io import (
        _load_native_weights_with_field_dtype,
    )
    from foldjax.models.protenix.models.primitives.primitives import transition
    from foldjax.models.protenix.models.triangle.triangle import (
        triangle_attention,
        triangle_multiplication,
    )
    from foldjax.models.protenix.models.trunk_blocks.msa import outer_product_mean
    from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_block

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    params = _load_native_weights_with_field_dtype(
        args.weights,
        jnp.bfloat16,
        frozenset({"input_embedder", "pairformer_output"}),
        prestack=False,
    ).pairformer_output
    records = []

    def check(name, p, fn, expected="result"):
        prefix = args.reference / f"stage__{name}"
        ipath = prefix.with_name(prefix.name + "__input.npz")
        opath = prefix.with_name(prefix.name + "__output.npz")
        metadata = json.loads(
            prefix.with_name(prefix.name + "__input-tree.json").read_text()
        )
        with np.load(ipath, allow_pickle=False) as archive:
            x = {
                key: jnp.asarray(
                    archive[key],
                    dtype=(
                        jnp.bfloat16
                        if metadata[key]["native_dtype"] == "torch.bfloat16"
                        else None
                    ),
                )
                for key in ("s", "z", "m", "x")
                if key in archive
            }
        with np.load(opath, allow_pickle=False) as archive:
            target = archive[expected].astype(np.float32)
        actual = np.asarray(jax.jit(fn)(x, p), dtype=np.float32)
        difference = actual - target
        row = {
            "module": name,
            "native_input_sha256": sha(ipath),
            "native_output_sha256": sha(opath),
            "max_abs_error": float(np.max(np.abs(difference))),
            "rms_error": float(np.sqrt(np.mean(difference**2))),
            "native_rms": float(np.sqrt(np.mean(target**2))),
            "foldjax_rms": float(np.sqrt(np.mean(actual**2))),
            "bitwise_equal": actual.tobytes() == target.tobytes(),
        }
        np.savez(args.out / f"{name}.npz", native=target, foldjax=actual)
        records.append(row)
        print(json.dumps(row), flush=True)

    with jax.default_matmul_precision("high"):
        check(
            "msa_module.blocks.0.outer_product_mean_msa",
            params.msa.blocks[0].outer_product_mean,
            lambda x, p: outer_product_mean(x["m"], None, p),
        )
        for name, block in (
            ("msa_module.blocks.0.pair_stack", params.msa.blocks[0].pair_stack),
            ("pairformer_stack.blocks.0", params.pairformer_stack.blocks[0]),
        ):
            check(
                name,
                block,
                lambda x, p: pairformer_block(
                    x.get("s"),
                    x["z"],
                    None,
                    p,
                    triangle_attention_backend="cueq",
                    single_attention_backend="xla_jit",
                )[1],
                expected="result.1",
            )
            for child, direction in (
                ("tri_mul_out", "outgoing"),
                ("tri_mul_in", "incoming"),
            ):
                check(
                    f"{name}.{child}",
                    getattr(block, child),
                    lambda x, p, direction=direction: (
                        x["z"] + triangle_multiplication(x["z"], None, p, direction)
                    ),
                )
            for child in ("tri_att_start", "tri_att_end"):
                check(
                    f"{name}.{child}",
                    getattr(block, child),
                    lambda x, p: triangle_attention(
                        x["x"], None, p, num_heads=4, attention_backend="cueq"
                    ),
                )
            check(
                f"{name}.pair_transition",
                block.pair_transition,
                lambda x, p: transition(x["x"], p),
            )
    source = Path(__file__).resolve().parents[1]
    save(
        args.out / "summary.json",
        {
            "scope": "isolated modules on native intermediate inputs; not parity",
            "checkpoint_sha256": sha(args.weights),
            "native_provenance_sha256": sha(args.reference / "provenance.json"),
            "source_files": {
                str(path.relative_to(source)): sha(path)
                for directory in ("src", "bench")
                for path in sorted((source / directory).rglob("*.py"))
            },
            "jax_version": jax.__version__,
            "records": records,
        },
    )


def triangle_controls():
    import jax
    import jax.numpy as jnp

    from foldjax.models._cueq import load_cueq
    from foldjax.models.protenix.bridge.weights_io import load_native_weights

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    block = load_native_weights(args.weights, prestack=False).pairformer_output
    block = block.pairformer_stack.blocks[0].tri_mul_out
    name = "stage__pairformer_stack.blocks.0.tri_mul_out"
    with np.load(args.reference / f"{name}__input.npz") as data:
        x = jnp.asarray(data["z"], dtype=jnp.bfloat16)[None]
    with np.load(args.reference / f"{name}__output.npz") as data:
        expected = data["result"]
    cuex = load_cueq()
    rows = []
    for fallback in (False, True):
        for precision in ("DEFAULT", "TF32", "IEEE"):
            for wide_norm in (False, True):
                p = jax.tree.map(lambda v: v.astype(jnp.bfloat16), block)
                if wide_norm:
                    p = p._replace(
                        layer_norm_in=block.layer_norm_in,
                        layer_norm_out=block.layer_norm_out,
                    )

                def run(x, p):
                    return cuex.triangle_multiplicative_update(
                        x=x,
                        direction="outgoing",
                        mask=jnp.ones(x.shape[:-1], x.dtype),
                        norm_in_weight=p.layer_norm_in.weight,
                        norm_in_bias=p.layer_norm_in.bias,
                        p_in_weight=jnp.concatenate(
                            (p.linear_a_p.weight, p.linear_b_p.weight)
                        ),
                        g_in_weight=jnp.concatenate(
                            (p.linear_a_g.weight, p.linear_b_g.weight)
                        ),
                        norm_out_weight=p.layer_norm_out.weight,
                        norm_out_bias=p.layer_norm_out.bias,
                        p_out_weight=p.linear_z.weight,
                        g_out_weight=p.linear_g.weight,
                        eps=1e-5,
                        precision=getattr(cuex.TriMulPrecision, precision),
                        fallback=fallback,
                    )

                with jax.default_matmul_precision("high"):
                    update = jax.jit(run)(x, p)
                    output = np.asarray((x + update)[0], dtype=np.float32)
                    raw = np.asarray(update[0], dtype=np.float32)
                row = {
                    "fallback": fallback,
                    "precision": precision,
                    "fp32_norm_affine": wide_norm,
                    "update_max_abs": float(np.max(np.abs(raw))),
                    "update_nonzero": int(np.count_nonzero(raw)),
                    "residual_max_abs_error": float(np.max(np.abs(output - expected))),
                    "residual_rms_error": float(
                        np.sqrt(np.mean((output - expected) ** 2))
                    ),
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
    save(
        args.out / "summary.json",
        {
            "scope": "controlled actual-native-input triangle kernel replay only",
            "wrapper_sha256": sha(Path(__file__)),
            "rows": rows,
            "native_input_sha256": sha(args.reference / f"{name}__input.npz"),
            "native_output_sha256": sha(args.reference / f"{name}__output.npz"),
            "checkpoint_sha256": sha(args.weights),
        },
    )


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    arms = {
        "native": native,
        "foldjax": foldjax,
        "triangle-controls": triangle_controls,
    }
    arms[mode]()
