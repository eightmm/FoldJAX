"""Native-input first MSA transition: isolate BF16 partial-sum fusion.

The manual no-barrier arm must reproduce production before the barrier arm can
be interpreted. This uses the full captured MSA, not a row-sliced native run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.af3_closure_capture import sha
from bench.boltz_closure_capture import save_new
from bench.boltz_downstream_probe import source_identity
from bench.boltz_msa_probe import verify_bound_file
from bench.boltz_relpos_probe import arrays, comparison


def transition_params(weights):
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.torch_mapping import map_transition_state_dict
    from foldjax.models.boltz2.models.trunk_blocks.trunk import _cast_trunk_params

    mapped = map_transition_state_dict(
        {f"transition.{name}": value for name, value in weights.items()}, "transition"
    )
    return _cast_trunk_params(mapped, jnp.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-norm-control", action="store_true")
    args = parser.parse_args()
    source, root = args.source_root.resolve(), args.reference.resolve()
    if Path(__file__).resolve() != source / "bench/boltz_transition_probe.py":
        raise ValueError("execute from selected source snapshot")
    report = json.loads((root / "report.json").read_text())
    if report.get("arm") != "native" or report.get("passed") is not True:
        raise ValueError("native MSA reproduction must have passed")
    verify_bound_file(
        root / "native-weights.npz", report["artifacts"]["native-weights.npz"]
    )
    values = {}
    for stage in ("input_m", "pwa", "msa_transition"):
        name = f"layers/00/{stage}"
        verify_bound_file(root / f"{name}.npz", report["stages"][name]["arrays_sha256"])
        verify_bound_file(
            root / f"{name}.tree.json", report["stages"][name]["tree_sha256"]
        )
        values[stage] = arrays(root / f"{name}.npz")[""]
    prefix = "msa_module.layers.0.msa_transition."
    weights = {
        k.removeprefix(prefix): v
        for k, v in arrays(root / "native-weights.npz").items()
        if k.startswith(prefix)
    }
    # Native FP32 eval-dropout identity promotes the PWA update before addition.
    operand = values.pop("input_m") + values.pop("pwa")
    expected = values.pop("msa_transition")
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.compile_policy import compiler_options
    from foldjax.models.boltz2.models.primitives._common import layer_norm
    from foldjax.models.boltz2.models.primitives.transition import (
        _auto_row_chunk,
        transition_forward,
    )

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("probe requires one GPU")
    params = transition_params(weights)
    x = jnp.asarray(operand)
    chunk = 32 if operand.shape[2] > 384 else None
    if chunk is None:
        raise ValueError("this control requires native hidden-chunked transition")

    def manual(params, x, *, barrier, normalizer=layer_norm):
        row_chunk = _auto_row_chunk(x, params) or x.shape[1]
        blocks = []
        for start_row in range(0, x.shape[1], row_chunk):
            block = normalizer(
                x[:, start_row : start_row + row_chunk], **params["norm"], eps=1e-5
            ).astype(jnp.bfloat16)
            out = jnp.zeros(
                (*block.shape[:-1], params["fc3"]["kernel"].shape[-1]), jnp.bfloat16
            )
            for first in range(0, params["fc1"]["kernel"].shape[-1], chunk):
                gate = block @ params["fc1"]["kernel"][:, first : first + chunk]
                hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * (
                    block @ params["fc2"]["kernel"][:, first : first + chunk]
                )
                partial = hidden @ params["fc3"]["kernel"][first : first + chunk]
                if barrier:
                    partial = jax.lax.optimization_barrier(partial)
                out = out + partial
            blocks.append(out)
        return jnp.concatenate(blocks, 1)

    sources = source_identity(source)
    args.out.mkdir(parents=True, exist_ok=False)
    arms, reference_output = {}, None
    names = ["production", "manual", "barrier"]
    if args.native_norm_control:
        names.extend(("native_norm", "native_norm_production"))
    for name in names:

        def run(params, x):
            if name in ("production", "native_norm_production"):
                return transition_forward(
                    params,
                    x,
                    chunk_size=chunk,
                    compute_dtype=jnp.bfloat16,
                    native_amp_norm=name == "native_norm_production",
                )
            if name == "native_norm":
                from bench.boltz_native_layer_norm import native_layer_norm

                def norm(x, scale, bias, eps):
                    return native_layer_norm(x, scale, bias, eps)[0]

                return manual(params, x, barrier=False, normalizer=norm)
            return manual(params, x, barrier=name == "barrier")

        with jax.default_matmul_precision("highest"):
            result = jax.jit(run, compiler_options=compiler_options("bfloat16"))(
                params, x
            )
            result.block_until_ready()
        stored = np.asarray(result.astype(jnp.float32))
        if reference_output is None:
            reference_output = stored
        arms[name] = {
            "native": comparison(stored, expected),
            "production": comparison(stored, reference_output),
        }
        with (args.out / f"{name}.npz").open("xb") as stream:
            np.savez(stream, output=stored)
        arms[name]["output_sha256"] = sha(args.out / f"{name}.npz")
    if sources != source_identity(source):
        raise ValueError("source changed during probe")
    save_new(
        args.out / "report.json",
        {
            "capture_complete": True,
            "not_model_parity_admission": True,
            "manual_reproduction": arms["manual"]["production"]["values_equal"],
            "native_norm_control": args.native_norm_control,
            "arms": arms,
            "reference_sha256": sha(root / "report.json"),
            "source": sources,
            "compiler_options": compiler_options("bfloat16"),
        },
    )
    print(json.dumps(arms))
    if not arms["manual"]["production"]["values_equal"]:
        raise RuntimeError(
            "manual baseline differs from production; control inadmissible"
        )


if __name__ == "__main__":
    main()
