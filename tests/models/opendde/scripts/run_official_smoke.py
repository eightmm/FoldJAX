"""Run the smallest real-weight OpenDDE-JAX inference smoke gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from foldjax.models.opendde.bridge.weights_io import load_native_weights
from foldjax.models.opendde.data.featurize_json import featurize_opendde_json, load_jobs
from foldjax.models.opendde.models.model import opendde_infer_static
from foldjax.models.protenix.models.diffusion.diffusion import inference_noise_schedule


def collect_raw_arrays(output: dict[str, object]) -> dict[str, np.ndarray]:
    """Collect array-valued raw model outputs for numerical parity probes."""

    return {
        name: np.asarray(value)
        for name, value in output.items()
        if hasattr(value, "shape") and hasattr(value, "dtype")
    }


def coordinate_error_metrics(
    coordinate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Return direct, no-alignment errors for matched sampler coordinates."""

    coordinate = np.asarray(coordinate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if coordinate.ndim != 3 or coordinate.shape[0] != 1:
        raise ValueError(
            "coordinate parity currently requires shape [1, N_atom, 3], got "
            f"{coordinate.shape}"
        )
    expected_reference = tuple(coordinate.shape[1:])
    if reference.shape == coordinate.shape:
        reference = reference[0]
    if tuple(reference.shape) != expected_reference:
        raise ValueError(
            f"reference coordinates expected shape {expected_reference}, "
            f"got {reference.shape}"
        )
    difference = coordinate[0] - reference
    return {
        "raw_coordinate_rmse_angstrom": float(np.sqrt(np.mean(np.square(difference)))),
        "all_atom_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(np.square(difference), axis=-1)))
        ),
        "coordinate_max_abs_error_angstrom": float(np.max(np.abs(difference))),
        "coordinate_mean_abs_error_angstrom": float(np.mean(np.abs(difference))),
    }


def load_random_tape(
    path: Path,
    *,
    n_step: int,
    n_sample: int,
    n_atom: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load an upstream-generated sampler tape with strict shape checks."""

    expected = {
        "noise_schedule",
        "init_noise",
        "step_noises",
        "rotations",
        "translations",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = expected.difference(archive.files)
        if missing:
            raise ValueError(f"random tape is missing arrays: {sorted(missing)}")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32) for name in expected
        }

    expected_init = (n_sample, n_atom, 3)
    expected_steps = (n_step, n_sample, n_atom, 3)
    expected_rotations = (n_step, n_sample, 3, 3)
    expected_translations = (n_step, n_sample, 3)
    expected_schedule = (n_step + 1,)
    shape_contracts = {
        "noise_schedule": expected_schedule,
        "init_noise": expected_init,
        "step_noises": expected_steps,
        "rotations": expected_rotations,
        "translations": expected_translations,
    }
    for name, expected_shape in shape_contracts.items():
        actual_shape = tuple(arrays[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} expected shape {expected_shape}, got {actual_shape}"
            )
    tape: dict[str, object] = {
        "init_noise": arrays["init_noise"],
        "step_noises": tuple(arrays["step_noises"]),
        "rotations": arrays["rotations"],
        "translations": arrays["translations"],
    }
    return arrays["noise_schedule"], tape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--random-tape", type=Path)
    parser.add_argument("--reference-coordinates", type=Path)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--n-cycle", type=int, default=1)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "highest")
    job = load_jobs(args.input_json)[0]
    features = featurize_opendde_json(
        job,
        base_dir=args.input_json.parent,
        seed=args.seed,
    )
    params = load_native_weights(args.weights)
    noise_schedule = inference_noise_schedule(n_step=args.n_step)
    random_kwargs: dict[str, object] = {}
    key: jax.Array | None = jax.random.PRNGKey(args.seed)
    if args.random_tape is not None:
        noise_schedule, random_kwargs = load_random_tape(
            args.random_tape,
            n_step=args.n_step,
            n_sample=1,
            n_atom=int(features["ref_pos"].shape[0]),
        )
        key = None
    started = time.perf_counter()
    output = opendde_infer_static(
        features,
        params,
        noise_schedule,
        key=key,
        n_sample=1,
        n_cycle=args.n_cycle,
        run_confidence=False,
        diffusion_attention_backend="xla",
        trunk_single_attention_backend="xla",
        trunk_triangle_attention_backend="xla",
        structural_single_attention_backend="xla",
        structural_triangle_attention_backend="xla",
        **random_kwargs,
    )
    coordinate = np.asarray(jax.device_get(output["coordinate"]))
    elapsed = time.perf_counter() - started
    metrics = {
        "all_finite": bool(np.all(np.isfinite(coordinate))),
        "coordinate_abs_max_angstrom": float(np.max(np.abs(coordinate))),
        "coordinate_shape": list(coordinate.shape),
        "elapsed_seconds": elapsed,
        "n_atom": int(features["ref_pos"].shape[0]),
        "n_residue_units": int(features["restype"].shape[0]),
        "n_structural_units": int(features["parent_residue_idx"].shape[0]),
        "n_cycle": args.n_cycle,
        "n_step": args.n_step,
        "seed": args.seed,
        "shared_random_tape": args.random_tape is not None,
    }
    if args.reference_coordinates is not None:
        with np.load(args.reference_coordinates, allow_pickle=False) as archive:
            if "coordinate" not in archive.files:
                raise ValueError(
                    "reference coordinate archive must contain 'coordinate'"
                )
            reference_coordinate = np.asarray(archive["coordinate"])
        metrics.update(coordinate_error_metrics(coordinate, reference_coordinate))
    if coordinate.shape != (1, metrics["n_atom"], 3):
        raise RuntimeError(
            f"unexpected coordinate shape: {coordinate.shape}; metrics={metrics}"
        )
    if not metrics["all_finite"]:
        raise RuntimeError(f"non-finite coordinates: {metrics}")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.coordinates.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.coordinates, **collect_raw_arrays(output))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
