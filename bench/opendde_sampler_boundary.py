"""First sampler step on real random tapes, replacing only the denoiser with zero.

The native branch executes publisher sampler/augmentation code. This finite
operator diagnostic is not a structure, confidence or performance benchmark.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.af3_closure_capture import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("native", "foldjax"))
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.tape, allow_pickle=False) as archive:
        tape = dict(archive)
    samples, atoms, _ = tape["init_noise"].shape
    if samples != 5:
        raise ValueError("expected five captured samples")
    report = {"scope": __doc__, "tape_sha256": sha(args.tape)}
    if args.arm == "native":
        import torch
        from opendde.model import generator, utils

        torch.backends.cuda.matmul.allow_tf32 = True
        values = {key: torch.from_numpy(value).cuda() for key, value in tape.items()}
        draws = iter(
            [
                values["init_noise"],
                values["translations"][0, :, None, :],
                values["step_noises"][0],
            ]
        )
        calls, boundary = [], {}

        def randn(*a, **kw):
            value = next(draws)
            shape = kw["size"]
            if tuple(shape) != tuple(value.shape):
                raise ValueError(f"unexpected random draw {shape} vs {value.shape}")
            calls.append(tuple(shape))
            return value

        def denoise(**kw):
            boundary["x_noisy"] = kw["x_noisy"].cpu().numpy()
            boundary["t_hat"] = kw["t_hat_noise_level"].cpu().numpy()
            return torch.zeros_like(kw["x_noisy"])

        with (
            torch.inference_mode(),
            patch.object(torch, "randn", randn),
            patch.object(
                utils, "uniform_random_rotation", lambda **kw: values["rotations"][0]
            ),
        ):
            dummy = torch.zeros((1, 1), device="cuda")
            result = generator.sample_diffusion(
                denoise,
                {"atom_to_token_idx": torch.zeros(atoms, device="cuda")},
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                values["noise_schedule"][:2],
                N_sample=samples,
            )
            boundary["output"] = result.cpu().numpy()
            previous, current = (
                values["noise_schedule"][:-1],
                values["noise_schedule"][1:],
            )
            gamma = torch.where(current > 1, 0.8, 0.0)
            t_hat = previous * (gamma + 1)
            boundary["churn_amplitude"] = (
                torch.sqrt(t_hat**2 - previous**2).cpu().numpy()
            )
        if next(draws, None) is not None or len(calls) != 3:
            raise ValueError("incomplete first-step tape consumption")
        np.savez_compressed(args.out / "native.npz", **boundary)
        report.update(torch=torch.__version__, calls=calls)
    else:
        import jax
        import jax.numpy as jnp

        from foldjax.models.opendde.models.sampling import sample_diffusion

        if args.reference is None:
            parser.error("FoldJAX requires native reference")
        with np.load(args.reference / "native.npz") as archive:
            native = dict(archive)
        values = {key: jnp.asarray(value) for key, value in tape.items()}

        def compute(values):
            boundary = {}

            def denoise(x, t):
                boundary.update(x_noisy=x, t_hat=t)
                return jnp.zeros_like(x)

            result = sample_diffusion(
                denoise,
                values["noise_schedule"][:2],
                num_samples=samples,
                n_atom=atoms,
                key=None,
                init_noise=values["init_noise"],
                step_noises=values["step_noises"][:1],
                rotations=values["rotations"][:1],
                translations=values["translations"][:1],
            )
            boundary["output"] = result
            previous, current = (
                values["noise_schedule"][:-1],
                values["noise_schedule"][1:],
            )
            gamma = jnp.where(current > 1, 0.8, 0.0)
            t_hat = previous * (gamma + 1)
            boundary["churn_amplitude"] = previous * jnp.sqrt(gamma * (gamma + 2))
            boundary["churn_native_expression"] = jnp.sqrt(
                jnp.maximum(t_hat**2 - previous**2, 0)
            )
            hat_square, previous_square = jax.lax.optimization_barrier(
                (t_hat**2, previous**2)
            )
            boundary["churn_barrier"] = jnp.sqrt(hat_square - previous_square)
            return boundary

        metrics = {}
        for precision in ("high", "highest"):
            with jax.default_matmul_precision(precision):
                boundary = {
                    k: np.asarray(v) for k, v in jax.jit(compute)(values).items()
                }
            np.savez_compressed(args.out / f"{precision}.npz", **boundary)
            metrics[precision] = {}
            for key, value in boundary.items():
                reference = native[
                    "churn_amplitude" if key.startswith("churn") else key
                ]
                error = value.astype(np.float64) - reference
                metrics[precision][key] = {
                    "max_abs": float(np.max(abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "unequal": int(np.count_nonzero(error)),
                }
        report.update(jax=jax.__version__, comparisons=metrics)
    report["wrapper_sha256"] = sha(Path(__file__))
    save(args.out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
