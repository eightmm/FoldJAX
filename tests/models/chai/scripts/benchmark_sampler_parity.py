#!/usr/bin/env python3
"""Compare the complete Torch/JAX EDM sampler with an identical random tape.

This is the noise-matched component route. It isolates sampler and diffusion
module drift from the intentionally different PyTorch and JAX PRNG streams used
by the public end-to-end APIs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from benchmark_diffusion_parity import _inputs
from reference_augmentation_tape import replay_jax

from foldjax.models.chai.models.diffusion import (
    chai_diffusion_gammas,
    chai_noise_schedule,
    edm_heun_step,
)
from foldjax.models.chai.models.diffusion_denoiser import (
    full_diffusion_denoiser,
    map_full_diffusion_denoiser,
)


def _default_component() -> Path | None:
    asset_dir = os.environ.get("CHAI_JAX_OFFICIAL_ASSET_DIR")
    return Path(asset_dir).expanduser() / "diffusion_module.pt" if asset_dir else None


def _random_rotations(rng: np.random.Generator, count: int) -> np.ndarray:
    quaternion = rng.normal(size=(count, 4)).astype(np.float32)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    quaternion *= np.where(quaternion[:, :1] < 0, -1.0, 1.0)
    real, i, j, k = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (j * j + k * k),
            2 * (i * j - k * real),
            2 * (i * k + j * real),
            2 * (i * j + k * real),
            1 - 2 * (i * i + k * k),
            2 * (j * k - i * real),
            2 * (i * k - j * real),
            2 * (j * k + i * real),
            1 - 2 * (i * i + j * j),
        ),
        axis=-1,
    ).reshape(count, 3, 3)


def _tape(seed: int, steps: int, samples: int, atoms: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "initial_noise": rng.normal(size=(samples, atoms, 3)).astype(np.float32),
        "churn_noise": rng.normal(size=(steps, samples, atoms, 3)).astype(np.float32),
        "rotations": np.stack([_random_rotations(rng, samples) for _ in range(steps)]),
        "translations": rng.normal(size=(steps, samples, 3)).astype(np.float32),
    }


def _torch_center(
    coords: torch.Tensor,
    mask: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(coords.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-4)
    centroid = (coords * weights[..., None]).sum(-2, keepdim=True)
    centered = coords - centroid
    return torch.einsum("bij,baj->bai", rotations, centered) + translations[:, None]


def _torch_denoise(
    module: torch.jit.ScriptModule,
    static_inputs: list[torch.Tensor],
    atom_tokens: torch.Tensor,
    position: torch.Tensor,
    sigma: float,
    samples: int,
    model_size: int,
) -> torch.Tensor:
    model_inputs = [
        *static_inputs,
        position[None],
        torch.full((1, samples), sigma, device="cuda"),
        atom_tokens,
    ]
    return _torch_bucket_forward(module, model_size, tuple(model_inputs))[0]


def _torch_bucket_forward(
    module: object, model_size: int, model_inputs: tuple[object, ...]
) -> object:
    """Call the released TorchScript method for one public crop bucket."""
    method_name = f"forward_{model_size}"
    method = getattr(module, method_name, None)
    if method is None or not callable(method):
        raise ValueError(f"Torch diffusion component does not expose {method_name}")
    return method(*model_inputs)


def _real_protein_inputs(
    fasta: Path,
    bundle: Path,
    conformers: Path,
    *,
    recycles: int,
    esm_model: Path | None = None,
    msa_directory: Path | None = None,
    template_hits: Path | None = None,
    template_cif_directory: Path | None = None,
    restraint: Path | None = None,
    reference_augmentation_tape: Path | None = None,
    kalign_executable: str | Path = "kalign",
) -> tuple[tuple[np.ndarray, ...], dict[str, object]]:
    """Build one shared diffusion context from real prepared features/trunk state."""
    import foldjax.models.chai.inference as inference

    config = inference.InferenceConfig(
        num_trunk_recycles=recycles,
        num_diffusion_timesteps=2,
        num_diffusion_samples=1,
        seed=0,
        use_esm_embeddings=esm_model is not None,
        esm_model_path=esm_model,
        msa_directory=msa_directory,
        template_hits_path=template_hits,
        template_cif_directory=template_cif_directory,
        constraint_path=restraint,
        kalign_executable=kalign_executable,
    )
    with replay_jax(reference_augmentation_tape):
        prepared, assets = inference.prepare_inference(
            fasta,
            bundle_path=bundle,
            conformer_path=conformers,
            config=config,
        )
    components = inference.map_model_components(assets.bundle)
    values = inference._model_padded_inputs(prepared)
    (
        diffusion_inputs,
        single_initial,
        pair_initial,
        msa_features,
        template_features,
    ) = inference._embed_and_initialize(prepared, components)
    token_mask = values["token_exists_mask"]
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    template_mask = values["template_mask"]
    template_pair_mask = template_mask[..., :, None] & template_mask[..., None, :]
    skip_templates = not bool(np.any(prepared.padded_inputs["template_mask"]))
    single, pair = single_initial, pair_initial
    for _ in range(recycles):
        msa_features, msa_mask = inference._crop_trailing_masked_msa_rows(
            msa_features, prepared.padded_inputs["msa_mask"]
        )
        single, pair = inference._run_staged_trunk(
            single_initial,
            pair_initial,
            single,
            pair,
            msa_features,
            msa_mask,
            template_features,
            template_pair_mask,
            token_mask,
            pair_mask,
            components.trunk,
            skip_templates=skip_templates,
        )
    diffusion_inputs["token_single_trunk_repr"] = single.astype(jnp.float32)
    diffusion_inputs["token_pair_trunk_repr"] = pair.astype(jnp.float32)
    ordered_names = (
        "token_single_initial_repr",
        "token_pair_initial_repr",
        "token_single_trunk_repr",
        "token_pair_trunk_repr",
        "atom_single_input_feats",
        "atom_block_pair_input_feats",
        "atom_single_mask",
        "atom_block_pair_mask",
        "token_single_mask",
        "block_indices_h",
        "block_indices_w",
    )
    static = tuple(
        np.asarray(jax.device_get(diffusion_inputs[name])) for name in ordered_names
    )
    atom_tokens = np.asarray(jax.device_get(diffusion_inputs["atom_token_indices"]))
    digest = hashlib.sha256()
    for name, value in zip(
        (*ordered_names, "atom_token_indices"), (*static, atom_tokens)
    ):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    metadata: dict[str, object] = {
        "context": "real-protein-shared-jax-prepared-and-trunk",
        "fasta": str(fasta),
        "model_size": prepared.model_size,
        "valid_atom_count": int(static[6].sum()),
        "static_input_sha256": digest.hexdigest(),
        "trunk_recycles": recycles,
        "use_esm_embeddings": esm_model is not None,
        "reference_augmentation_tape_sha256": (
            hashlib.sha256(reference_augmentation_tape.read_bytes()).hexdigest()
            if reference_augmentation_tape is not None
            else None
        ),
        "branches": {
            "msa": msa_directory is not None,
            "template": template_hits is not None,
            "restraint": restraint is not None,
        },
    }
    del components, assets, prepared, values, diffusion_inputs
    gc.collect()
    return (*static, np.empty(0), np.empty(0), atom_tokens), metadata


def _load_torch_context(
    path: Path,
) -> tuple[tuple[np.ndarray, ...], dict[str, object]]:
    ordered_names = (
        "token_single_initial_repr",
        "token_pair_initial_repr",
        "token_single_trunk_repr",
        "token_pair_trunk_repr",
        "atom_single_input_feats",
        "atom_block_pair_input_feats",
        "atom_single_mask",
        "atom_block_pair_mask",
        "token_single_mask",
        "block_indices_h",
        "block_indices_w",
    )
    with np.load(path, allow_pickle=False) as archive:
        static = tuple(np.asarray(archive[name]) for name in ordered_names)
        atom_tokens = np.asarray(archive["atom_token_indices"])
        metadata = json.loads(str(archive["metadata_json"]))
    return (*static, np.empty(0), np.empty(0), atom_tokens), metadata


def _array_drift(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        return {
            "shape": list(left.shape),
            "mismatch_count": int(np.count_nonzero(left != right)),
        }
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    delta = right64 - left64
    reference_rms = np.sqrt(np.mean(left64**2))
    correlation = None
    if left64.size > 1 and np.std(left64) > 0 and np.std(right64) > 0:
        correlation = float(np.corrcoef(left64.ravel(), right64.ravel())[0, 1])
    return {
        "shape": list(left.shape),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "nrmse": float(np.sqrt(np.mean(delta**2)) / max(reference_rms, 1e-12)),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "correlation": correlation,
    }


def _coordinate_drift(
    reference: np.ndarray, actual: np.ndarray, valid_mask: np.ndarray
) -> dict[str, float]:
    reference_valid = reference[valid_mask]
    actual_valid = actual[valid_mask]
    delta = actual_valid - reference_valid
    component_rmse = float(np.sqrt(np.mean(delta**2)))
    atom_displacements = np.linalg.norm(delta, axis=-1)
    geometry = _geometry_drift(reference, actual, valid_mask)
    return {
        # Retain the historical names for machine-readable compatibility. The
        # explicit alias prevents this per-coordinate-component quantity from
        # being mistaken for conventional per-atom RMSD.
        "coordinate_rmse": component_rmse,
        "coordinate_component_rmse": component_rmse,
        "coordinate_mae": float(np.mean(np.abs(delta))),
        "coordinate_max_abs": float(np.max(np.abs(delta))),
        "all_atom_raw_rmsd": float(np.sqrt(np.mean(atom_displacements**2))),
        "max_atom_displacement": float(np.max(atom_displacements)),
        "correlation": float(
            np.corrcoef(reference_valid.ravel(), actual_valid.ravel())[0, 1]
        ),
        **geometry,
    }


def _geometry_drift(
    reference: np.ndarray,
    actual: np.ndarray,
    valid_mask: np.ndarray,
    *,
    block_size: int = 256,
) -> dict[str, float]:
    """Return rigid-invariant endpoint metrics without materializing N² arrays."""

    aligned_squared_error = 0.0
    valid_atoms = 0
    centroid_squared_error = 0.0
    local_error_sum = 0.0
    local_pair_count = 0
    atom_lddt_sum = 0.0
    atom_lddt_count = 0
    cutoffs = np.asarray((0.5, 1.0, 2.0, 4.0), dtype=np.float64)

    for sample in range(reference.shape[0]):
        keep = valid_mask[sample]
        ref = np.asarray(reference[sample, keep], dtype=np.float64)
        mob = np.asarray(actual[sample, keep], dtype=np.float64)
        if ref.shape[0] == 0:
            continue
        ref_centroid = ref.mean(axis=0)
        mob_centroid = mob.mean(axis=0)
        centroid_squared_error += float(np.sum((mob_centroid - ref_centroid) ** 2))
        covariance = (mob - mob_centroid).T @ (ref - ref_centroid)
        left, _, right_t = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
        aligned = (mob - mob_centroid) @ (left @ correction @ right_t) + ref_centroid
        aligned_squared_error += float(np.sum((aligned - ref) ** 2))
        valid_atoms += ref.shape[0]
        per_atom_score = np.zeros(ref.shape[0], dtype=np.float64)
        per_atom_pairs = np.zeros(ref.shape[0], dtype=np.int64)

        for start in range(0, ref.shape[0], block_size):
            ref_left = ref[start : start + block_size]
            mob_left = mob[start : start + block_size]
            for right_start in range(start, ref.shape[0], block_size):
                ref_right = ref[right_start : right_start + block_size]
                mob_right = mob[right_start : right_start + block_size]
                ref_distance = np.linalg.norm(
                    ref_left[:, None] - ref_right[None], axis=-1
                )
                mob_distance = np.linalg.norm(
                    mob_left[:, None] - mob_right[None], axis=-1
                )
                local = ref_distance < 15.0
                if start == right_start:
                    local &= np.triu(np.ones(local.shape, dtype=bool), k=1)
                error_matrix = np.abs(mob_distance - ref_distance)
                error = error_matrix[local]
                local_error_sum += float(error.sum())
                local_pair_count += error.size
                pair_score = np.where(
                    local,
                    (error_matrix[..., None] < cutoffs).mean(axis=-1),
                    0.0,
                )
                left_stop = start + ref_left.shape[0]
                right_stop = right_start + ref_right.shape[0]
                per_atom_score[start:left_stop] += pair_score.sum(axis=1)
                per_atom_pairs[start:left_stop] += local.sum(axis=1)
                per_atom_score[right_start:right_stop] += pair_score.sum(axis=0)
                per_atom_pairs[right_start:right_stop] += local.sum(axis=0)
        has_neighbors = per_atom_pairs > 0
        atom_lddt_sum += float(
            np.sum(per_atom_score[has_neighbors] / per_atom_pairs[has_neighbors])
        )
        atom_lddt_count += int(has_neighbors.sum())

    if valid_atoms == 0 or local_pair_count == 0 or atom_lddt_count == 0:
        raise ValueError("geometry drift requires valid atoms and local atom pairs")
    return {
        "all_atom_kabsch_rmsd": float(np.sqrt(aligned_squared_error / valid_atoms)),
        "centroid_rmsd": float(np.sqrt(centroid_squared_error / reference.shape[0])),
        "local_pair_distance_mae": local_error_sum / local_pair_count,
        "local_pair_lddt": atom_lddt_sum / atom_lddt_count,
    }


def _threshold_failures(
    report: dict[str, object],
    *,
    max_coordinate_rmse: float | None,
    max_all_atom_raw_rmsd: float | None,
) -> list[str]:
    failures = []
    if (
        max_coordinate_rmse is not None
        and float(report["coordinate_rmse"]) > max_coordinate_rmse
    ):
        failures.append("coordinate-component RMSE threshold exceeded")
    if (
        max_all_atom_raw_rmsd is not None
        and float(report["all_atom_raw_rmsd"]) > max_all_atom_raw_rmsd
    ):
        failures.append("all-atom raw RMSD threshold exceeded")
    return failures


def _torch_sampler(
    module: torch.jit.ScriptModule,
    numpy_inputs: tuple[np.ndarray, ...],
    tape: dict[str, np.ndarray],
    sigmas: np.ndarray,
    gammas: np.ndarray,
    *,
    samples: int,
    model_size: int,
) -> np.ndarray:
    torch_static = [torch.from_numpy(value).cuda() for value in numpy_inputs[:11]]
    torch_atom_tokens = torch.from_numpy(numpy_inputs[13]).cuda()
    torch_mask = torch.from_numpy(numpy_inputs[6]).cuda().repeat(samples, 1)
    tape_device = {name: torch.from_numpy(value).cuda() for name, value in tape.items()}
    atom_pos = tape_device["initial_noise"] * float(sigmas[0])
    with torch.inference_mode():
        for index, (sigma_curr, sigma_next, gamma) in enumerate(
            zip(sigmas[:-1], sigmas[1:], gammas[:-1], strict=True)
        ):
            atom_pos = _torch_center(
                atom_pos,
                torch_mask,
                tape_device["rotations"][index],
                tape_device["translations"][index],
            )
            sigma_hat = float(sigma_curr * (1.0 + gamma))
            scale = math.sqrt(max(sigma_hat**2 - float(sigma_curr) ** 2, 1e-6))
            atom_hat = atom_pos + 1.003 * tape_device["churn_noise"][index] * scale
            denoised = _torch_denoise(
                module,
                torch_static,
                torch_atom_tokens,
                atom_hat,
                sigma_hat,
                samples,
                model_size,
            )
            derivative = (atom_hat - denoised) / sigma_hat
            euler = atom_hat + (float(sigma_next) - sigma_hat) * derivative
            denoised_next = _torch_denoise(
                module,
                torch_static,
                torch_atom_tokens,
                euler,
                float(sigma_next),
                samples,
                model_size,
            )
            derivative_next = (euler - denoised_next) / float(sigma_next)
            atom_pos = (
                euler
                + (float(sigma_next) - sigma_hat) * (derivative + derivative_next) / 2
            )
        torch.cuda.synchronize()
        result = atom_pos.float().cpu().numpy()
    return result


def _make_jax_sampler(
    params: object,
    numpy_inputs: tuple[np.ndarray, ...],
    tape: dict[str, np.ndarray],
    sigmas: np.ndarray,
    gammas: np.ndarray,
    *,
    samples: int,
) -> object:
    jax_static = tuple(jnp.asarray(value) for value in numpy_inputs[:11])
    jax_atom_tokens = jnp.asarray(numpy_inputs[13])
    jax_mask = jnp.repeat(jnp.asarray(numpy_inputs[6]), samples, axis=0)

    @jax.jit
    def step(
        position: jax.Array,
        sigma_curr: jax.Array,
        sigma_next: jax.Array,
        gamma: jax.Array,
        noise: jax.Array,
        rotations: jax.Array,
        translations: jax.Array,
    ) -> jax.Array:
        def denoise(value: jax.Array, sigma: jax.Array) -> jax.Array:
            output = full_diffusion_denoiser(
                *jax_static,
                value[None],
                jnp.broadcast_to(sigma, (1, samples)),
                jax_atom_tokens,
                params,
            )
            return output[0]

        return edm_heun_step(
            position,
            jax_mask,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            gamma=gamma,
            noise=noise,
            rotations=rotations,
            translations=translations,
            denoise=denoise,
        )

    def run() -> np.ndarray:
        actual = jnp.asarray(tape["initial_noise"]) * sigmas[0]
        for index, values in enumerate(
            zip(sigmas[:-1], sigmas[1:], gammas[:-1], strict=True)
        ):
            actual = step(
                actual,
                *map(jnp.asarray, values),
                jnp.asarray(tape["churn_noise"][index]),
                jnp.asarray(tape["rotations"][index]),
                jnp.asarray(tape["translations"][index]),
            )
        return np.asarray(jax.block_until_ready(actual))

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, default=_default_component())
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--conformers", type=Path)
    parser.add_argument("--torch-context", type=Path)
    parser.add_argument("--torch-control-context", type=Path)
    parser.add_argument("--esm-model", type=Path)
    parser.add_argument("--msa-directory", type=Path)
    parser.add_argument("--template-hits", type=Path)
    parser.add_argument("--template-cif-directory", type=Path)
    parser.add_argument("--restraint", type=Path)
    parser.add_argument("--reference-augmentation-tape", type=Path)
    parser.add_argument("--kalign-executable", default="kalign")
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--skip-hybrids", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-coordinate-rmse", type=float)
    parser.add_argument(
        "--max-all-atom-raw-rmsd",
        type=float,
        help="maximum conventional raw RMSD over valid atoms, in angstrom",
    )
    args = parser.parse_args()
    if args.component is None:
        parser.error(
            "--component is required unless CHAI_JAX_OFFICIAL_ASSET_DIR is set"
        )
    if args.timesteps < 2:
        raise ValueError("timesteps must be at least two")
    if args.warm_repeats < 1:
        raise ValueError("warm-repeats must be positive")
    real_options = (args.fasta, args.bundle, args.conformers)
    if any(value is not None for value in real_options) and not all(
        value is not None for value in real_options
    ):
        raise ValueError(
            "--fasta, --bundle, and --conformers must be supplied together"
        )

    if args.fasta is None:
        numpy_inputs = _inputs(samples=args.samples)
        context_metadata: dict[str, object] = {"context": "synthetic-component"}
    else:
        numpy_inputs, context_metadata = _real_protein_inputs(
            args.fasta,
            args.bundle,
            args.conformers,
            recycles=args.recycles,
            esm_model=args.esm_model,
            msa_directory=args.msa_directory,
            template_hits=args.template_hits,
            template_cif_directory=args.template_cif_directory,
            restraint=args.restraint,
            reference_augmentation_tape=args.reference_augmentation_tape,
            kalign_executable=args.kalign_executable,
        )
    torch_inputs = numpy_inputs
    torch_control_inputs = None
    if args.torch_context is not None:
        torch_inputs, torch_context_metadata = _load_torch_context(args.torch_context)
        if torch_inputs[6].shape != numpy_inputs[6].shape:
            raise ValueError("Torch/JAX atom mask shapes differ")
        if not np.array_equal(torch_inputs[6], numpy_inputs[6]):
            raise ValueError("Torch/JAX atom masks differ")
        if not np.array_equal(torch_inputs[13], numpy_inputs[13]):
            raise ValueError("Torch/JAX atom-to-token indices differ")
        context_metadata["context"] = "framework-native-prepared-and-trunk"
        context_metadata["torch_static_input_sha256"] = torch_context_metadata[
            "static_input_sha256"
        ]
        static_names = (
            "token_single_initial_repr",
            "token_pair_initial_repr",
            "token_single_trunk_repr",
            "token_pair_trunk_repr",
            "atom_single_input_feats",
            "atom_block_pair_input_feats",
            "atom_single_mask",
            "atom_block_pair_mask",
            "token_single_mask",
            "block_indices_h",
            "block_indices_w",
            "atom_token_indices",
        )
        static_drift = {
            name: _array_drift(torch_inputs[index], numpy_inputs[index])
            for index, name in zip((*range(11), 13), static_names, strict=True)
        }
        token_valid = torch_inputs[8].astype(bool)
        atom_valid = torch_inputs[6].astype(bool)
        pair_valid = token_valid[..., :, None] & token_valid[..., None, :]
        block_valid = torch_inputs[7].astype(bool) & numpy_inputs[7].astype(bool)
        for index in (0, 2):
            name = static_names[index]
            static_drift[name]["valid_region"] = _array_drift(
                torch_inputs[index][token_valid], numpy_inputs[index][token_valid]
            )
        for index in (1, 3):
            name = static_names[index]
            static_drift[name]["valid_region"] = _array_drift(
                torch_inputs[index][pair_valid], numpy_inputs[index][pair_valid]
            )
        static_drift["atom_single_input_feats"]["valid_region"] = _array_drift(
            torch_inputs[4][atom_valid], numpy_inputs[4][atom_valid]
        )
        static_drift["atom_block_pair_input_feats"]["valid_region"] = _array_drift(
            torch_inputs[5][block_valid], numpy_inputs[5][block_valid]
        )
        query_valid = atom_valid[0][torch_inputs[9]][..., :, None]
        key_valid = atom_valid[0][torch_inputs[10]][..., None, :]
        chemically_valid_block = query_valid & key_valid
        block_mask_delta = torch_inputs[7][0] != numpy_inputs[7][0]
        static_drift["atom_block_pair_mask"]["chemically_valid_mismatch_count"] = int(
            np.count_nonzero(block_mask_delta & chemically_valid_block)
        )
        context_metadata["static_field_order"] = list(static_names)
        context_metadata["static_input_drift"] = static_drift
    if args.torch_control_context is not None:
        if args.torch_context is None:
            raise ValueError("--torch-control-context requires --torch-context")
        torch_control_inputs, control_metadata = _load_torch_context(
            args.torch_control_context
        )
        for index in (6, 7, 8, 9, 10, 13):
            if not np.array_equal(torch_control_inputs[index], torch_inputs[index]):
                raise ValueError(
                    "Torch control context identity/masks differ at static index "
                    f"{index}"
                )
        context_metadata["torch_control_static_input_sha256"] = control_metadata[
            "static_input_sha256"
        ]
    atom_mask = numpy_inputs[6]
    atom_count = atom_mask.shape[1]
    model_size = int(numpy_inputs[8].shape[-1])
    if torch_inputs[8].shape[-1] != model_size:
        raise ValueError("Torch/JAX token-mask bucket sizes differ")
    sigmas = np.asarray(chai_noise_schedule(args.timesteps))
    gammas = np.asarray(chai_diffusion_gammas(sigmas, num_timesteps=args.timesteps))
    tape = _tape(args.seed, args.timesteps - 1, args.samples, atom_count)

    module = torch.jit.load(str(args.component), map_location="cuda").eval()
    state = {
        name: value.detach().cpu().numpy()
        for name, value in module.state_dict().items()
    }
    torch_same_static_repeat = _torch_sampler(
        module,
        torch_inputs,
        tape,
        sigmas,
        gammas,
        samples=args.samples,
        model_size=model_size,
    )
    torch_times = []
    for _ in range(args.warm_repeats):
        start = time.perf_counter()
        expected = _torch_sampler(
            module,
            torch_inputs,
            tape,
            sigmas,
            gammas,
            samples=args.samples,
            model_size=model_size,
        )
        torch_times.append(time.perf_counter() - start)
    torch_with_jax_static = None
    if args.torch_context is not None:
        torch_with_jax_static = _torch_sampler(
            module,
            numpy_inputs,
            tape,
            sigmas,
            gammas,
            samples=args.samples,
            model_size=model_size,
        )
    torch_control = None
    if torch_control_inputs is not None:
        torch_control = _torch_sampler(
            module,
            torch_control_inputs,
            tape,
            sigmas,
            gammas,
            samples=args.samples,
            model_size=model_size,
        )
    del module
    gc.collect()
    torch.cuda.empty_cache()

    params = map_full_diffusion_denoiser(state)
    jax_run = _make_jax_sampler(
        params, numpy_inputs, tape, sigmas, gammas, samples=args.samples
    )
    jax_run()
    jax_times = []
    for _ in range(args.warm_repeats):
        start = time.perf_counter()
        actual = jax_run()
        jax_times.append(time.perf_counter() - start)
    valid_mask = np.broadcast_to(atom_mask, (args.samples, atom_count))
    coordinate_metrics = _coordinate_drift(expected, actual, valid_mask)
    hybrid_metrics = None
    hybrid_single_initial_metrics = None
    hybrid_single_initial_and_trunk_metrics = None
    hybrid_masks_only_metrics = None
    shared_torch_static_metrics = None
    if args.torch_context is not None:

        def run_hybrid(indices: tuple[int, ...]) -> dict[str, float]:
            hybrid_inputs = list(numpy_inputs)
            for index in indices:
                hybrid_inputs[index] = torch_inputs[index]
            hybrid_run = _make_jax_sampler(
                params,
                tuple(hybrid_inputs),
                tape,
                sigmas,
                gammas,
                samples=args.samples,
            )
            return _coordinate_drift(expected, hybrid_run(), valid_mask)

        if not args.skip_hybrids:
            hybrid_metrics = run_hybrid((2, 3))
            hybrid_single_initial_metrics = run_hybrid((0,))
            hybrid_single_initial_and_trunk_metrics = run_hybrid((0, 2))
            hybrid_masks_only_metrics = run_hybrid((6, 7, 8))
        shared_torch_run = _make_jax_sampler(
            params,
            torch_inputs,
            tape,
            sigmas,
            gammas,
            samples=args.samples,
        )
        shared_torch = shared_torch_run()
        shared_torch_static_metrics = _coordinate_drift(
            expected, shared_torch, valid_mask
        )
    report = {
        "contract": "component/noise-matched",
        "timesteps": args.timesteps,
        "samples": args.samples,
        "seed": args.seed,
        "model_size": model_size,
        **context_metadata,
        "torch_warm_seconds": float(np.median(torch_times)),
        "jax_warm_seconds": float(np.median(jax_times)),
        "warm_speedup": float(np.median(torch_times) / np.median(jax_times)),
        **coordinate_metrics,
    }
    report["torch_same_static_repeat"] = _coordinate_drift(
        expected, torch_same_static_repeat, valid_mask
    )
    if hybrid_metrics is not None:
        report["hybrid_torch_trunk_repr_only"] = hybrid_metrics
    if hybrid_single_initial_metrics is not None:
        report["hybrid_torch_single_initial_only"] = hybrid_single_initial_metrics
    if hybrid_single_initial_and_trunk_metrics is not None:
        report["hybrid_torch_single_initial_and_single_trunk"] = (
            hybrid_single_initial_and_trunk_metrics
        )
    if hybrid_masks_only_metrics is not None:
        report["hybrid_torch_masks_only"] = hybrid_masks_only_metrics
    if shared_torch_static_metrics is not None:
        report["shared_torch_static"] = shared_torch_static_metrics
    if torch_with_jax_static is not None:
        report["torch_with_jax_static"] = _coordinate_drift(
            expected, torch_with_jax_static, valid_mask
        )
        report["cross_backend_jax_static"] = _coordinate_drift(
            torch_with_jax_static, actual, valid_mask
        )
    if torch_control is not None:
        report["torch_repeat_context"] = _coordinate_drift(
            expected, torch_control, valid_mask
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    failures = _threshold_failures(
        report,
        max_coordinate_rmse=args.max_coordinate_rmse,
        max_all_atom_raw_rmsd=args.max_all_atom_raw_rmsd,
    )
    if failures:
        raise SystemExit(
            "noise-matched sampler parity threshold exceeded: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
