"""Replay upstream Chai's sampler tape through the JAX port and diff the atoms.

`capture_upstream_sampler_tape.py` records every random number Chai's diffusion
sampler drew on a real job, plus the structure those numbers produced and the
trunk representations that fed the diffusion module. This script feeds the same
numbers to `foldjax.models.chai` and reports the direct, no-alignment
coordinate error between the two structures.

That is the decisive test. Both existing Chai parity routes stop short of it:
the component route (`benchmark_sampler_parity.py`) matches the tape but runs
on a synthetic context, and the end-to-end route
(`export_scientific_parity.py`) runs the real job but leaves the two PRNG
streams unmatched, so its two structures are different samples and only
comparable statistically. Confidence scores agreeing across a PRNG boundary
proves very little -- an ordering defect can leave every score intact and still
move the coordinates.

Nothing under `src/` is modified. The port's three sampler draw sites are
rebound from the outside for the duration of one call:

* `jax.random.normal`, reached only through `inference`'s own module global, so
  the rebinding cannot leak into unrelated JAX code. The first call is the
  initial position draw and every later one is a churn draw -- they have the
  same shape, so order is what separates them, and a shape that does not match
  the tape is an error rather than a silent reshape.
* `inference._random_rotations` and `inference._random_translations`.

All three are called from `execute_prepared_inference` itself, outside every
`jax.jit` boundary, so the replay never sees a tracer.

The trunk representations are captured on the way past `_compiled_diffusion_step`,
which is called from Python with concrete arrays. If the coordinates disagree,
their correlation says whether the sampler or the trunk is responsible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

TRUNK_KEYS = (
    "token_single_initial_repr",
    "token_pair_initial_repr",
    "token_single_trunk_repr",
    "token_pair_trunk_repr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path)
    parser.add_argument("--tape-dir", type=Path, required=True)
    parser.add_argument(
        "--control-tape-dir",
        type=Path,
        help="a second capture of the same job and seed; reports what upstream "
        "scores against itself, which is the floor this metric can reach",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home() / ".cache/foldjax/weights/chai",
        help="native Chai-JAX asset root holding models/chai1 and conformers.npz",
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--conformers", type=Path)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--compilation-cache", type=Path)
    parser.add_argument(
        "--no-esm",
        action="store_true",
        help="disable the ESM embedding feature; must match the capture",
    )
    return parser.parse_args()


class _RandomProxy:
    """`jax.random` with `normal` rebound; every other name passes through."""

    def __init__(self, real: Any, normal: Any) -> None:
        self._real = real
        self._normal = normal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def normal(self, *args: Any, **kwargs: Any) -> Any:
        return self._normal(*args, **kwargs)


class _JaxProxy:
    """`jax` with a rebound `random`; scoped to one module's global."""

    def __init__(self, real: Any, random: Any) -> None:
        self._real = real
        self.random = random

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class SamplerReplay:
    """Hand back the recorded draws in the order the port asks for them."""

    def __init__(self, path: Path, *, steps: int, samples: int, atoms: int) -> None:
        expected = {"sigmas", "init_noise", "step_noises", "rotations", "translations"}
        with np.load(path, allow_pickle=False) as archive:
            missing = expected.difference(archive.files)
            if missing:
                raise ValueError(f"tape is missing arrays: {sorted(missing)}")
            arrays = {
                name: np.asarray(archive[name], dtype=np.float32) for name in expected
            }
        contracts = {
            "sigmas": (steps,),
            "init_noise": (samples, atoms, 3),
            "step_noises": (steps - 1, samples, atoms, 3),
            "rotations": (steps - 1, samples, 3, 3),
            "translations": (steps - 1, samples, 3),
        }
        for name, shape in contracts.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"{name} expected shape {shape}, got {arrays[name].shape}"
                )
        self.sigmas = arrays["sigmas"]
        self.init_noise = arrays["init_noise"]
        self.step_noises = arrays["step_noises"]
        self.rotations = arrays["rotations"]
        # Upstream draws `[b, 1, 3]`; the port keeps the broadcast axis too.
        self.translations = arrays["translations"][:, :, None, :]
        self.normal_index = 0
        self.rotation_index = 0
        self.translation_index = 0
        self.diffusion_inputs: dict[str, np.ndarray] = {}

    def normal(self, key: Any, shape: Any, dtype: Any = None) -> Any:
        del key, dtype
        if self.normal_index == 0:
            value = self.init_noise
        else:
            index = self.normal_index - 1
            if index >= len(self.step_noises):
                raise RuntimeError("the sampler asked for more noise than was taped")
            value = self.step_noises[index]
        if tuple(shape) != value.shape:
            raise RuntimeError(
                f"draw {self.normal_index} wanted shape {tuple(shape)}, the tape "
                f"holds {value.shape}"
            )
        self.normal_index += 1
        return _as_device_array(value)

    def rotations_for(self, key: Any, count: int) -> Any:
        del key
        value = self.rotations[self.rotation_index]
        if value.shape[0] != count:
            raise RuntimeError(
                f"rotation draw {self.rotation_index} wanted {count} samples, the "
                f"tape holds {value.shape[0]}"
            )
        self.rotation_index += 1
        return _as_device_array(value)

    def translations_for(self, key: Any, count: int) -> Any:
        del key
        value = self.translations[self.translation_index]
        if value.shape[0] != count:
            raise RuntimeError(
                f"translation draw {self.translation_index} wanted {count} samples, "
                f"the tape holds {value.shape[0]}"
            )
        self.translation_index += 1
        return _as_device_array(value)

    def assert_exhausted(self) -> None:
        used = (self.normal_index - 1, self.rotation_index, self.translation_index)
        available = (
            len(self.step_noises),
            len(self.rotations),
            len(self.translations),
        )
        if used != available:
            raise RuntimeError(f"tape consumption mismatch: used {used} of {available}")


def _as_device_array(value: np.ndarray) -> Any:
    import jax.numpy as jnp

    return jnp.asarray(value)


def _install(inference: Any, replay: SamplerReplay):
    """Rebind the port's sampler draws; return a callable that restores them."""
    import jax

    original_jax = inference.jax
    original_rotations = inference._random_rotations
    original_translations = inference._random_translations
    original_schedule = inference.chai_noise_schedule
    original_step = inference._compiled_diffusion_step

    inference.jax = _JaxProxy(jax, _RandomProxy(jax.random, replay.normal))
    inference._random_rotations = replay.rotations_for
    inference._random_translations = replay.translations_for

    def schedule(num_timesteps: int, **kwargs: Any) -> Any:
        # Use upstream's own sigmas rather than a recomputation of them, so a
        # schedule difference shows up as an explicit number below instead of
        # as unexplained coordinate error.
        del kwargs
        if num_timesteps != len(replay.sigmas):
            raise RuntimeError(
                f"the port asked for {num_timesteps} sigmas, the tape holds "
                f"{len(replay.sigmas)}"
            )
        return _as_device_array(replay.sigmas)

    def step(*args: Any, **kwargs: Any) -> Any:
        if not replay.diffusion_inputs:
            inputs = args[8]
            replay.diffusion_inputs = {
                name: np.asarray(jax.device_get(inputs[name])) for name in TRUNK_KEYS
            }
        return original_step(*args, **kwargs)

    inference.chai_noise_schedule = schedule
    inference._compiled_diffusion_step = step

    def restore() -> None:
        inference.jax = original_jax
        inference._random_rotations = original_rotations
        inference._random_translations = original_translations
        inference.chai_noise_schedule = original_schedule
        inference._compiled_diffusion_step = original_step

    return restore


def _native_esm_archive(source: Path, destination: Path) -> Path:
    """Rewrite the captured Torch embeddings as the port's native archive."""
    from foldjax.models.chai.data.esm import save_native_esm_embeddings

    with np.load(source, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        embeddings = {
            sequence: np.asarray(archive[f"embedding_{index:06d}"], dtype=np.float32)
            for index, sequence in enumerate(manifest["sequences"])
        }
    save_native_esm_embeddings(
        embeddings,
        destination,
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        source_sha256=manifest["source_sha256"],
    )
    return destination


def _real_tokens(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the token slots the job actually uses.

    Chai pads a 132-token job to 256. What the two implementations leave in the
    124 unused slots is arbitrary -- nothing reads them -- so comparing them
    reports a disagreement that does not exist.
    """
    if array.ndim == 3:
        return array[:, mask]
    if array.ndim == 4:
        return array[:, mask][:, :, mask]
    raise ValueError(f"unexpected trunk rank: {array.shape}")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, np.float64).ravel()
    b = np.asarray(right, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _coordinate_metrics(
    coordinate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    if coordinate.shape != reference.shape:
        raise ValueError(
            f"coordinate shape {coordinate.shape} does not match the reference "
            f"{reference.shape}"
        )
    difference = np.asarray(coordinate, np.float64) - np.asarray(reference, np.float64)
    real = difference[:, np.asarray(mask, bool)]
    return {
        "all_atom_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(np.square(real), axis=-1)))
        ),
        "all_atom_rmsd_with_padding_angstrom": float(
            np.sqrt(np.mean(np.sum(np.square(difference), axis=-1)))
        ),
        "coordinate_max_abs_error_angstrom": float(np.max(np.abs(real))),
        "coordinate_mean_abs_error_angstrom": float(np.mean(np.abs(real))),
        "coordinate_correlation": _correlation(
            np.asarray(coordinate)[:, np.asarray(mask, bool)],
            np.asarray(reference)[:, np.asarray(mask, bool)],
        ),
    }


def _trunk_metrics(
    ours: dict[str, np.ndarray], reference_path: Path, token_mask: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with np.load(reference_path, allow_pickle=False) as archive:
        for name in TRUNK_KEYS:
            if name not in archive.files or name not in ours:
                continue
            left = np.asarray(ours[name], np.float64)
            right = np.asarray(archive[name], np.float64)
            real_left = _real_tokens(left, token_mask)
            real_right = _real_tokens(right, token_mask)
            result[name] = {
                "correlation": _correlation(real_left, real_right),
                "max_abs_diff": float(np.max(np.abs(real_left - real_right))),
                "correlation_with_padding": _correlation(left, right),
                "max_abs_diff_with_padding": float(np.max(np.abs(left - right))),
            }
    return result


def _control_metrics(
    tape_dir: Path, control_dir: Path, mask: np.ndarray, token_mask: np.ndarray
) -> dict[str, Any]:
    """Measure upstream against a second upstream run of the same job.

    Chai's sampler is chaotically sensitive: nineteen EDM steps amplify the
    float32 reduction-order differences a GPU is free to make between two runs
    of the *same* program. Without this control the JAX number has no scale --
    it is impossible to tell a port defect from the floor every implementation
    of this model is stuck with. The two upstream tapes are checked for
    bit-identity first, so any coordinate difference reported here is
    nondeterminism and not a different draw.
    """
    with np.load(tape_dir / "tape.npz", allow_pickle=False) as first:
        with np.load(control_dir / "tape.npz", allow_pickle=False) as second:
            identical = sorted(first.files) == sorted(second.files) and all(
                np.array_equal(first[name], second[name]) for name in first.files
            )
    if not identical:
        raise RuntimeError(
            "the control run drew a different tape; it is not a determinism control"
        )
    with np.load(tape_dir / "reference.npz", allow_pickle=False) as archive:
        reference = np.asarray(archive["coordinate"], np.float32)
    with np.load(control_dir / "reference.npz", allow_pickle=False) as archive:
        control = np.asarray(archive["coordinate"], np.float32)
    result: dict[str, Any] = {
        "description": "upstream Chai against a second upstream run, same seed, "
        "bit-identical tape",
        "tape_identical": identical,
    }
    result.update(_coordinate_metrics(control, reference, mask))
    control_trunk = control_dir / "trunk.npz"
    trunk_path = tape_dir / "trunk.npz"
    if control_trunk.is_file() and trunk_path.is_file():
        with np.load(control_trunk, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        result["trunk"] = _trunk_metrics(arrays, trunk_path, token_mask)
    return result


def main() -> int:
    args = parse_args()
    import jax

    from foldjax.models.chai import inference as chai_inference

    jax.config.update("jax_default_matmul_precision", "highest")

    bundle = args.bundle or args.weights / "models" / "chai1"
    conformers = args.conformers or args.weights / "conformers.npz"
    args.out.mkdir(parents=True, exist_ok=True)

    esm_path = None
    if not args.no_esm:
        esm_path = _native_esm_archive(
            args.tape_dir / "esm.npz", args.out / "esm_native.npz"
        )

    config = chai_inference.InferenceConfig(
        num_trunk_recycles=args.recycles,
        num_diffusion_timesteps=args.steps,
        num_diffusion_samples=args.samples,
        num_trunk_samples=1,
        seed=args.seed,
        use_esm_embeddings=not args.no_esm,
        esm_embeddings_path=esm_path,
        msa_directory=args.msa_dir,
        compilation_cache_dir=args.compilation_cache,
    )
    prepared, assets = chai_inference.prepare_inference(
        args.fasta,
        bundle_path=bundle,
        conformer_path=conformers,
        config=config,
    )
    mask = np.asarray(prepared.padded_inputs["atom_exists_mask"]).reshape(-1)
    atoms = int(mask.shape[0])

    replay = SamplerReplay(
        args.tape_dir / "tape.npz",
        steps=args.steps,
        samples=args.samples,
        atoms=atoms,
    )
    native_sigmas = np.asarray(
        jax.device_get(chai_inference.chai_noise_schedule(args.steps))
    )
    components = chai_inference.map_model_components(assets.bundle)
    restore = _install(chai_inference, replay)
    started = time.perf_counter()
    try:
        prediction = chai_inference.execute_prepared_inference(
            prepared, components, config
        )
        jax.block_until_ready(prediction.atom_coords)
    finally:
        restore()
    elapsed = time.perf_counter() - started
    replay.assert_exhausted()

    coordinate = np.asarray(jax.device_get(prediction.atom_coords))
    with np.load(args.tape_dir / "reference.npz", allow_pickle=False) as archive:
        reference = np.asarray(archive["coordinate"], np.float32)

    metrics: dict[str, Any] = {
        "contract": "end-to-end/tape-matched",
        "job": str(args.fasta),
        "seed": args.seed,
        "recycles": args.recycles,
        "diffusion_timesteps": args.steps,
        "diffusion_steps": args.steps - 1,
        "samples": args.samples,
        "use_esm_embeddings": not args.no_esm,
        "n_atom_padded": atoms,
        "n_atom_real": int(mask.sum()),
        "n_token_padded": int(prepared.model_size),
        "elapsed_seconds": round(elapsed, 3),
        "all_finite": bool(np.all(np.isfinite(coordinate))),
        "noise_schedule_max_abs_diff": float(
            np.max(np.abs(native_sigmas - replay.sigmas))
        ),
        "draws_replayed": {
            "initial_noise": 1,
            "churn_noise": replay.normal_index - 1,
            "rotations": replay.rotation_index,
            "translations": replay.translation_index,
        },
    }
    metrics.update(_coordinate_metrics(coordinate, reference, mask))

    token_mask = np.asarray(prepared.padded_inputs["token_exists_mask"], bool).reshape(
        -1
    )
    trunk_path = args.tape_dir / "trunk.npz"
    if trunk_path.is_file() and replay.diffusion_inputs:
        metrics["trunk"] = _trunk_metrics(
            replay.diffusion_inputs, trunk_path, token_mask
        )
        np.savez_compressed(args.out / "trunk_jax.npz", **replay.diffusion_inputs)

    if args.control_tape_dir is not None:
        metrics["upstream_self_control"] = _control_metrics(
            args.tape_dir, args.control_tape_dir, mask, token_mask
        )

    if not metrics["all_finite"]:
        raise RuntimeError(f"non-finite coordinates: {metrics}")
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.out / "coordinates.npz",
        coordinate=coordinate,
        reference=reference,
        # Exported so the same real-atom subset can be applied to any other
        # pair of runs, including an upstream-versus-upstream control.
        atom_exists_mask=mask,
        token_exists_mask=token_mask,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
