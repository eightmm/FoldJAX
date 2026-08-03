"""Measure JAX Protenix against a captured noise-matched torch run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def rmsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=-1))))


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rmsd(pc @ rotation.T, qc)


def array_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    x = left.astype(np.float64, copy=False).ravel()
    y = right.astype(np.float64, copy=False).ravel()
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "max_abs": float(np.max(np.abs(x - y))),
    }


def ca_indices(features: dict) -> np.ndarray:
    names = np.asarray(features["ref_atom_name_chars"])
    chars = names.argmax(axis=-1) + 32
    return np.where(
        (chars[:, 0] == ord("C"))
        & (chars[:, 1] == ord("A"))
        & (chars[:, 2] == ord(" "))
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--no-pairformer-scan", action="store_true")
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.999,
        help="fail if any trunk tensor correlates below this with upstream's",
    )
    parser.add_argument("--bf16-trunk", action="store_true")
    parser.add_argument(
        "--diffusion-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa", "cudnn"),
        default="xla",
    )
    parser.add_argument(
        "--trunk-single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa", "cudnn"),
        default="xla",
    )
    parser.add_argument(
        "--trunk-triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        default="xla",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import foldjax.models.protenix.models.diffusion.diffusion as diffusion
    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.data.static_io import load_static_feature_npz
    from foldjax.models.protenix.models.model import cast_trunk_params
    from foldjax.models.protenix.models.predict import protenix_predict_static

    features = load_static_feature_npz(args.features)
    params = load_native_weights(args.weights)
    trunk_dtype = None
    if args.bf16_trunk:
        trunk_dtype = jnp.bfloat16
        params = cast_trunk_params(params, trunk_dtype)
    with np.load(args.torch_dir / "noise.npz") as data:
        init = data["init"].astype(np.float32)
        steps = data["steps"].astype(np.float32)
    torch_coordinate = np.load(args.torch_dir / "coordinate.npy").astype(np.float32)[0]
    with np.load(args.torch_dir / "trunk.npz") as data:
        torch_trunk = {key: data[key] for key in data.files}
    with np.load(args.torch_dir / "denoise0.npz") as data:
        torch_denoise0 = {key: data[key] for key in data.files}

    captured_denoise: list[np.ndarray] = []
    original_denoise = diffusion.diffusion_module_forward

    def record_denoise(*pos, **kwargs):
        output = original_denoise(*pos, **kwargs)
        if not captured_denoise:
            captured_denoise.append(np.asarray(jax.block_until_ready(output)))
        return output

    diffusion.diffusion_module_forward = record_denoise

    def run() -> dict:
        output = protenix_predict_static(
            params,
            features,
            key=jax.random.PRNGKey(0),
            n_sample=1,
            num_sampling_steps=args.steps,
            recycling_steps=args.cycles,
            use_pairformer_scan=not args.no_pairformer_scan,
            diffusion_attention_backend=args.diffusion_attention_backend,
            trunk_single_attention_backend=args.trunk_single_attention_backend,
            trunk_triangle_attention_backend=args.trunk_triangle_attention_backend,
            init_noise=jnp.asarray(init),
            step_noises=tuple(jnp.asarray(value) for value in steps),
            run_confidence=False,
            run_confidence_scores=False,
            centre_each_step=True,
            matmul_precision="highest",
            trunk_dtype=trunk_dtype,
            use_sampler_scan=False,
        )
        jax.block_until_ready(output["coordinate"])
        return output

    started = time.perf_counter()
    cold_output = run()
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm_output = run()
    warm_seconds = time.perf_counter() - started
    diffusion.diffusion_module_forward = original_denoise

    coordinate = np.asarray(warm_output["coordinate"])[0]
    ca = ca_indices(features)
    jax_denoise0 = captured_denoise[0]
    while jax_denoise0.ndim > 3:
        jax_denoise0 = jax_denoise0[0]
    if jax_denoise0.ndim == 3:
        jax_denoise0 = jax_denoise0[0]
    torch_d0 = torch_denoise0["x_denoised"]
    while torch_d0.ndim > 3:
        torch_d0 = torch_d0[0]
    if torch_d0.ndim == 3:
        torch_d0 = torch_d0[0]

    memory = jax.devices()[0].memory_stats() or {}
    metrics = {
        "backend": "jax",
        "precision": "bf16_trunk_fp32_diffusion" if args.bf16_trunk else "fp32_highest",
        "tokens": int(features["residue_index"].shape[-1]),
        "atoms": int(features["atom_to_token_idx"].shape[-1]),
        "msa_rows": int(features["msa"].shape[-2]),
        "cycles": args.cycles,
        "diffusion_steps": args.steps,
        "pairformer_scan": not args.no_pairformer_scan,
        "diffusion_attention_backend": args.diffusion_attention_backend,
        "trunk_single_attention_backend": args.trunk_single_attention_backend,
        "trunk_triangle_attention_backend": args.trunk_triangle_attention_backend,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "peak_vram_gb": memory.get("peak_bytes_in_use", 0) / 1e9,
        "all_atom_raw_rmsd": rmsd(coordinate, torch_coordinate),
        "all_atom_kabsch_rmsd": kabsch_rmsd(coordinate, torch_coordinate),
        "ca_count": int(ca.size),
        "ca_raw_rmsd": rmsd(coordinate[ca], torch_coordinate[ca]),
        "ca_kabsch_rmsd": kabsch_rmsd(coordinate[ca], torch_coordinate[ca]),
        "determinism_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(cold_output["coordinate"])
                    - np.asarray(warm_output["coordinate"])
                )
            )
        ),
        "s_inputs": array_metrics(
            np.asarray(warm_output["s_inputs"]), torch_trunk["s_inputs"]
        ),
        "s_trunk": array_metrics(
            np.asarray(warm_output["s_trunk"]), torch_trunk["s_trunk"]
        ),
        "z_trunk": array_metrics(
            np.asarray(warm_output["z_trunk"]), torch_trunk["z_trunk"]
        ),
        "denoise0_raw_rmsd": rmsd(jax_denoise0, torch_d0),
        "denoise0_kabsch_rmsd": kabsch_rmsd(jax_denoise0, torch_d0),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.save(args.out_dir / "coordinate.npy", coordinate)
    print(json.dumps(metrics, indent=2))

    verdict = _verdict(metrics, min_correlation=args.min_correlation)
    if verdict:
        print("\n".join(["", "PARITY FAILED:", *verdict]))
        return 1
    print(f"\nPARITY OK: every trunk tensor correlates >= {args.min_correlation:g}.")
    return 0


def _verdict(metrics: dict, *, min_correlation: float) -> list[str]:
    """Which trunk tensors fall short, as lines to print.

    Printing metrics is not checking them. This harness reported a `z_trunk`
    correlation for the whole life of the port and nothing ever read it, which
    is how a transposed MSA block survived: the check existed, but nothing
    failed when it was wrong. A threshold here is what turns the output into an
    answer.
    """
    failures = []
    for name in ("s_inputs", "s_trunk", "z_trunk"):
        correlation = metrics.get(name, {}).get("correlation")
        if correlation is None:
            failures.append(f"  {name}: not reported")
        elif not correlation >= min_correlation:
            failures.append(
                f"  {name}: correlation {correlation:.6f} < {min_correlation:g}"
            )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
