"""Run the full static Chai ``forward_256`` Torch/JAX parity benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _torch_inputs(torch, device):
    bf16 = torch.bfloat16
    single_mask = torch.zeros((1, 256), dtype=torch.bool, device=device)
    single_mask[:, :16] = True
    pair_mask = single_mask[:, :, None] & single_mask[:, None, :]
    msa_mask = torch.zeros((1, 16384, 256), dtype=torch.bool, device=device)
    msa_mask[:, :8, :16] = True
    template_mask = torch.zeros((1, 4, 256, 256), dtype=torch.bool, device=device)
    template_mask[:, :2] = pair_mask[:, None]

    single_initial = torch.zeros((1, 256, 384), dtype=bf16, device=device)
    single_previous = torch.zeros_like(single_initial)
    single_initial[:, :16] = 0.125
    single_previous[:, :16] = -0.0625
    pair_initial = torch.zeros((1, 256, 256, 256), dtype=bf16, device=device)
    pair_previous = torch.zeros_like(pair_initial)
    pair_initial[:, :16, :16] = 0.03125
    pair_previous[:, :16, :16] = -0.015625
    msa = torch.zeros((1, 16384, 256, 64), dtype=bf16, device=device)
    msa[:, :8, :16] = 0.0625
    template = torch.zeros((1, 4, 256, 256, 64), dtype=bf16, device=device)
    template[:, 0, :16, :16] = 0.09375
    template[:, 1, :16, :16] = -0.046875
    return (
        single_initial,
        pair_initial,
        single_previous,
        pair_previous,
        msa,
        msa_mask,
        template,
        template_mask,
        single_mask,
        pair_mask,
    )


def _jax_inputs(jnp):
    bf16 = jnp.bfloat16
    single_mask = jnp.zeros((1, 256), dtype=bool).at[:, :16].set(True)
    pair_mask = single_mask[:, :, None] & single_mask[:, None, :]
    msa_mask = jnp.zeros((1, 16384, 256), dtype=bool)
    msa_mask = msa_mask.at[:, :8, :16].set(True)
    template_mask = jnp.zeros((1, 4, 256, 256), dtype=bool)
    template_mask = template_mask.at[:, :2].set(pair_mask[:, None])
    single_initial = jnp.zeros((1, 256, 384), dtype=bf16)
    single_initial = single_initial.at[:, :16].set(0.125)
    single_previous = jnp.zeros_like(single_initial)
    single_previous = single_previous.at[:, :16].set(-0.0625)
    pair_initial = jnp.zeros((1, 256, 256, 256), dtype=bf16)
    pair_initial = pair_initial.at[:, :16, :16].set(0.03125)
    pair_previous = jnp.zeros_like(pair_initial)
    pair_previous = pair_previous.at[:, :16, :16].set(-0.015625)
    msa = jnp.zeros((1, 16384, 256, 64), dtype=bf16)
    msa = msa.at[:, :8, :16].set(0.0625)
    template = jnp.zeros((1, 4, 256, 256, 64), dtype=bf16)
    template = template.at[:, 0, :16, :16].set(0.09375)
    template = template.at[:, 1, :16, :16].set(-0.046875)
    return (
        single_initial,
        pair_initial,
        single_previous,
        pair_previous,
        msa,
        msa_mask,
        template,
        template_mask,
        single_mask,
        pair_mask,
    )


def _metrics(actual, expected):
    import numpy as np

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    delta = actual - expected
    denom = max(float(np.linalg.norm(expected)), 1e-12)
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "nrmse": float(np.linalg.norm(delta) / denom),
        "correlation": float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1]),
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
        "norm_ratio": actual_norm / max(expected_norm, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    asset_dir = os.environ.get("CHAI_JAX_OFFICIAL_ASSET_DIR")
    default_component = (
        Path(asset_dir).expanduser() / "trunk.pt" if asset_dir else None
    )
    parser.add_argument(
        "--component",
        type=Path,
        default=default_component,
    )
    parser.add_argument("--assert-nrmse", type=float, default=0.03)
    parser.add_argument("--torch-math-sdpa", action="store_true")
    parser.add_argument("--fp32-pairformer-state", action="store_true")
    parser.add_argument("--fp32-msa-pair-state", action="store_true")
    parser.add_argument("--low-memory-msa-pair", action="store_true")
    parser.add_argument("--msa-pair-subchunk-size", type=int)
    parser.add_argument("--pairformer-subchunk-size", type=int)
    parser.add_argument(
        "--zero-component",
        action="append",
        choices=(
            "pairformer_stack",
            "msa_module",
            "template_embedder",
            "msa_opm",
            "msa_weighted",
            "msa_transition",
            "msa_pair_transition",
            "msa_triangle_multiplication",
            "msa_triangle_attention",
        ),
        default=[],
    )
    args = parser.parse_args()

    if args.component is None:
        parser.error(
            "--component is required unless CHAI_JAX_OFFICIAL_ASSET_DIR is set"
        )
    if not args.component.is_file():
        parser.error(f"official trunk component is unavailable: {args.component}")
    for name in ("msa_pair_subchunk_size", "pairformer_subchunk_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.msa_pair_subchunk_size is not None:
        args.low_memory_msa_pair = True

    import numpy as np
    import torch

    component = args.component.resolve()
    if args.torch_math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    module = torch.jit.load(str(component), map_location="cpu").eval()
    if args.zero_component:
        zero_prefixes = {
            "pairformer_stack": "pairformer_stack.",
            "msa_module": "msa_module.",
            "template_embedder": "template_embedder.",
            "msa_opm": "msa_module.outer_product_mean.",
            "msa_weighted": "msa_module.msa_pair_weighted_averaging.",
            "msa_transition": "msa_module.msa_transition.",
            "msa_pair_transition": "msa_module.pair_transition.",
            "msa_triangle_multiplication": (
                "msa_module.triangular_multiplication."
            ),
            "msa_triangle_attention": "msa_module.triangular_attention.",
        }
        with torch.no_grad():
            for name, value in module.state_dict().items():
                if any(
                    name.startswith(zero_prefixes[component])
                    for component in args.zero_component
                ):
                    value.zero_()
        state_override = {
            name: value.detach().cpu().numpy().copy()
            for name, value in module.state_dict().items()
        }
    else:
        state_override = None
    module = module.cuda()
    torch_inputs = _torch_inputs(torch, "cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        torch_output = module.forward_256(*torch_inputs)
    torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - start
    torch_peak = torch.cuda.max_memory_allocated()
    expected = tuple(value.float().cpu().numpy() for value in torch_output)
    import jax
    jax_inputs = tuple(jax.dlpack.from_dlpack(value) for value in torch_inputs)
    jax.block_until_ready(jax_inputs)
    del torch_output, torch_inputs, module
    gc.collect()
    torch.cuda.empty_cache()

    from foldjax.models.chai.bridge.component_io import load_component_state_dict
    from foldjax.models.chai.inference import (
        _run_msa_pair_block_low_memory,
        _run_pairformer_block_low_memory,
    )
    from foldjax.models.chai.models.msa import (
        _msa_pair_block,
        msa_pair_weighted_averaging,
        msa_transition,
        outer_product_mean,
    )
    from foldjax.models.chai.models.primitives import linear_bf16
    from foldjax.models.chai.models.template import template_embedding
    from foldjax.models.chai.models.trunk import (
        _pairformer_stack,
        map_trunk,
        recycling_projection,
    )

    state = (
        state_override
        if state_override is not None
        else load_component_state_dict(component)
    )
    params = map_trunk(state)
    def recycle_template(p, single_initial, pair_initial, single_prev, pair_prev,
                         template, template_mask, single_mask, pair_mask):
        del single_mask
        pair = pair_initial + recycling_projection(pair_prev, p.pair_recycling)
        single = single_initial + recycling_projection(
            single_prev, p.single_recycling
        )
        pair = template_embedding(
            pair, template, template_mask, pair_mask, p.template
        )
        return single, pair

    compiled_recycle_template = jax.jit(recycle_template)
    compiled_msa_embedding = jax.jit(
        lambda features, single, weight: features
        + linear_bf16(single, weight)[:, None]
    )
    compiled_opm = jax.jit(
        lambda msa, mask, p: outer_product_mean(
            msa, mask, p, chunk_size=4096
        )
    )
    compiled_msa_transition = jax.jit(
        lambda msa, p: msa_transition(msa, p, chunk_size=1024)
    )
    compiled_weighted = jax.jit(
        lambda msa, pair, msa_mask, pair_mask, p: msa_pair_weighted_averaging(
            msa, pair, msa_mask, pair_mask, p, chunk_size=1024
        )
    )
    compiled_msa_pair = jax.jit(_msa_pair_block)
    compiled_stack = jax.jit(
        lambda blocks, single, pair, single_mask, pair_mask: _pairformer_stack(
            single, pair, single_mask, pair_mask, blocks
        )
    )

    stage_peaks: dict[str, int | None] = {}

    def record_peak(name: str, values) -> None:
        jax.block_until_ready(values)
        stats = jax.devices()[0].memory_stats() or {}
        stage_peaks[name] = stats.get("peak_bytes_in_use")

    def run_jax(*, record_peaks: bool = False):
        single, pair = compiled_recycle_template(
            params,
            jax_inputs[0],
            jax_inputs[1],
            jax_inputs[2],
            jax_inputs[3],
            jax_inputs[6],
            jax_inputs[7],
            jax_inputs[8],
            jax_inputs[9],
        )
        if record_peaks:
            record_peak("recycle_template", (single, pair))
        msa = compiled_msa_embedding(
            jax_inputs[4], single, params.msa.linear_s2m_weight
        )
        pair_before_msa = pair
        if args.fp32_msa_pair_state:
            pair = pair.astype(jax.numpy.float32)
        jax.block_until_ready(msa)
        for index, block in enumerate(params.msa.blocks):
            pair = pair + compiled_opm(msa, jax_inputs[5], block.outer_product_mean)
            if record_peaks:
                record_peak(f"msa_block_{index}_opm", pair)
            if index < 3:
                msa_input = msa
                msa = msa_input + compiled_msa_transition(
                    msa_input, block.msa_transition
                )
                if record_peaks:
                    record_peak(f"msa_block_{index}_transition", msa)
                weighted_update = compiled_weighted(
                    msa_input,
                    pair,
                    jax_inputs[5],
                    jax_inputs[9],
                    block.weighted_averaging,
                )
                if record_peaks:
                    record_peak(f"msa_block_{index}_weighted", weighted_update)
                msa += weighted_update
                if record_peaks:
                    record_peak(f"msa_block_{index}_msa_update", msa)
            pair = (
                _run_msa_pair_block_low_memory(
                    pair,
                    jax_inputs[9],
                    block.pair,
                    subchunk_size=args.msa_pair_subchunk_size,
                )
                if args.low_memory_msa_pair
                else compiled_msa_pair(pair, jax_inputs[9], block.pair)
            )
            if record_peaks:
                record_peak(f"msa_block_{index}", (msa, pair))
        pair = pair_before_msa + pair
        if args.fp32_pairformer_state:
            single = single.astype(jax.numpy.float32)
            pair = pair.astype(jax.numpy.float32)
        if args.pairformer_subchunk_size is not None:
            for block in params.pairformer_blocks:
                single, pair = _run_pairformer_block_low_memory(
                    single,
                    pair,
                    jax_inputs[8],
                    jax_inputs[9],
                    block,
                    subchunk_size=args.pairformer_subchunk_size,
                )
            output = single, pair
        else:
            output = compiled_stack(
                params.pairformer_blocks,
                single,
                pair,
                jax_inputs[8],
                jax_inputs[9],
            )
        if record_peaks:
            record_peak("pairformer_stack", output)
        return output

    start = time.perf_counter()
    jax_output = run_jax(record_peaks=True)
    jax.block_until_ready(jax_output)
    compile_and_run_seconds = time.perf_counter() - start
    first_actual = tuple(
        np.array(jax.device_get(value), dtype=np.float32, copy=True)
        for value in jax_output
    )
    start = time.perf_counter()
    jax_output = run_jax()
    jax.block_until_ready(jax_output)
    warm_seconds = time.perf_counter() - start
    memory_stats = jax.devices()[0].memory_stats() or {}
    actual = tuple(np.asarray(value, dtype=np.float32) for value in jax_output)
    results = {
        "component": str(component),
        "torch_math_sdpa": args.torch_math_sdpa,
        "fp32_pairformer_state": args.fp32_pairformer_state,
        "fp32_msa_pair_state": args.fp32_msa_pair_state,
        "msa_pair_subchunk_size": args.msa_pair_subchunk_size,
        "pairformer_subchunk_size": args.pairformer_subchunk_size,
        "zero_component": args.zero_component,
        "torch_seconds": torch_seconds,
        "torch_peak_bytes": torch_peak,
        "jax_compile_and_run_seconds": compile_and_run_seconds,
        "jax_warm_seconds": warm_seconds,
        "jax_peak_bytes": memory_stats.get("peak_bytes_in_use"),
        "jax_stage_peak_bytes": stage_peaks,
        "jax_deterministic_exact": all(
            np.array_equal(first, second)
            for first, second in zip(first_actual, actual, strict=True)
        ),
        "single": _metrics(actual[0], expected[0]),
        "pair": _metrics(actual[1], expected[1]),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    if results["single"]["nrmse"] > args.assert_nrmse:
        raise SystemExit("single trunk parity threshold exceeded")
    if results["pair"]["nrmse"] > args.assert_nrmse:
        raise SystemExit("pair trunk parity threshold exceeded")


if __name__ == "__main__":
    main()
