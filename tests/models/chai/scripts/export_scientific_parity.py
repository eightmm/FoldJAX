"""Export a provenance-rich Chai Torch or Chai-JAX inference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from reference_augmentation_tape import replay_jax, replay_torch
from scientific_parity import (
    NUMERIC_PREPARED_TOLERANCES,
    SCHEMA_VERSION,
    read_cif_coordinates,
    save_artifact,
    sha256_file,
    source_weight_manifest_from_native_bundle,
    source_weight_manifest_from_torch,
    tensor_manifest,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _prepared_manifest(
    features: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    arrays = {}
    for prefix, values in (("features", features), ("inputs", inputs)):
        for name, value in values.items():
            array = _numpy(value)
            # Restraint controls may be None or lists of dictionaries. Their
            # enabled state is part of the explicit branch contract; generated
            # numeric restraint features are hashed with the other features.
            if array.dtype.kind in "biufSU":
                arrays[f"{prefix}.{name}"] = array
    # Torch pads categorical residue IDs with zero; JAX uses the dedicated 32
    # padding category. The token mask makes those slots semantically absent,
    # so normalize only masked positions before checking prepared features.
    residue = arrays.get("features.ResidueType")
    token_mask = arrays.get("inputs.token_exists_mask")
    if residue is not None and token_mask is not None:
        normalized = residue.copy()
        mask = np.asarray(token_mask, dtype=bool)
        if normalized.ndim == mask.ndim + 1 and normalized.shape[-1] == 1:
            normalized[~mask, 0] = 0
        else:
            normalized[~mask] = 0
        arrays["features.ResidueType"] = normalized
    return tensor_manifest(arrays)


def _ranking_from_score_files(paths: list[Path]) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                rows.setdefault(name, []).append(np.asarray(archive[name]))
    return {name: np.stack(values) for name, values in rows.items()}


def _numeric_prepared_arrays(features: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for qualified_name in NUMERIC_PREPARED_TOLERANCES:
        name = qualified_name.removeprefix("features.")
        array = _numpy(features[name])
        if array.ndim and array.shape[-1] == 1:
            array = np.squeeze(array, axis=-1)
        result[qualified_name] = array
    return result


def _base_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": args.backend,
        "fixture_id": args.fixture_id,
        "fasta_sha256": sha256_file(args.fasta),
        "config": {
            "recycles": args.recycles,
            "timesteps": args.timesteps,
            "samples": args.samples,
            "seed": args.seed,
        },
        "branches": {
            "msa": args.msa_directory is not None,
            "template": args.template_hits is not None,
            "restraint": args.restraint is not None,
        },
        # Both implementations use the same seed and schedule, but Torch and
        # JAX use different PRNG algorithms. This is distribution-matched, not
        # a claim of bitwise-identical sampler noise.
        "sampler_randomness": {
            "matching": "seed-and-distribution",
            "bitwise_identical": False,
            "reason": "PyTorch Philox and JAX Threefry generate different streams",
        },
        "reference_augmentation": (
            {
                "matching": "explicit-rigid-transform-tape",
                "sha256": sha256_file(args.reference_augmentation_tape),
            }
            if args.reference_augmentation_tape is not None
            else {"matching": "framework-native-rng"}
        ),
    }


def _save_prepared_only(
    path: Path,
    metadata: dict[str, Any],
    numeric_arrays: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **{f"prepared__{name}": value for name, value in numeric_arrays.items()},
    )


def _export_jax(args: argparse.Namespace, run_dir: Path) -> None:
    import jax

    from foldjax.models.chai.inference import (
        InferenceConfig,
        execute_prepared_inference,
        map_model_components,
        prepare_inference,
    )
    from foldjax.models.chai.output import write_prediction_outputs

    if args.bundle is None or args.conformers is None:
        raise ValueError("JAX export requires --bundle and --conformers")
    config = InferenceConfig(
        num_trunk_recycles=args.recycles,
        num_diffusion_timesteps=args.timesteps,
        num_diffusion_samples=args.samples,
        seed=args.seed,
        use_esm_embeddings=False,
        msa_directory=args.msa_directory,
        template_hits_path=args.template_hits,
        template_cif_directory=args.template_cif_directory,
        constraint_path=args.restraint,
        compilation_cache_dir=args.compilation_cache,
    )
    with replay_jax(args.reference_augmentation_tape):
        prepared, assets = prepare_inference(
            args.fasta,
            bundle_path=args.bundle,
            conformer_path=args.conformers,
            config=config,
        )
    metadata = _base_metadata(args)
    metadata["source_weight_sha256"] = source_weight_manifest_from_native_bundle(
        args.bundle
    )
    metadata["prepared_tensors"] = _prepared_manifest(
        dict(prepared.features), dict(prepared.padded_inputs)
    )
    metadata["versions"] = {"jax": jax.__version__}
    numeric_arrays = _numeric_prepared_arrays(dict(prepared.features))
    if args.prepared_only:
        _save_prepared_only(args.artifact, metadata, numeric_arrays)
        return
    components = map_model_components(assets.bundle)
    prediction = execute_prepared_inference(prepared, components, config)
    jax.block_until_ready(prediction.atom_coords)
    candidates = write_prediction_outputs(run_dir, [prediction])
    coords, atom_ids = read_cif_coordinates(candidates.cif_paths)
    score_paths = [
        path.with_name(path.name.replace("pred.", "scores.")).with_suffix(".npz")
        for path in candidates.cif_paths
    ]
    save_artifact(
        args.artifact,
        coords=coords,
        atom_ids=atom_ids,
        pae=np.asarray(candidates.pae),
        pde=np.asarray(candidates.pde),
        plddt=np.asarray(candidates.plddt),
        ranking=_ranking_from_score_files(score_paths),
        metadata=metadata,
        prepared_arrays=numeric_arrays,
    )


def _export_torch(args: argparse.Namespace, run_dir: Path) -> None:
    import torch
    from chai_lab.chai1 import (
        feature_factory,
        make_all_atom_feature_context,
        run_folding_on_context,
    )
    from chai_lab.data.collate.collate import Collate

    model_directory = (
        args.torch_model_directory
        if args.torch_model_directory is not None
        else Path(__file__).resolve().parents[2] / "chai" / "downloads" / "models_v2"
    )
    with replay_torch(args.reference_augmentation_tape):
        context = make_all_atom_feature_context(
            fasta_file=args.fasta,
            output_dir=run_dir,
            use_esm_embeddings=False,
            use_msa_server=False,
            msa_directory=args.msa_directory,
            constraint_path=args.restraint,
            use_templates_server=False,
            templates_path=args.template_hits,
            esm_device=torch.device(args.device),
        )
    # This is the same Collate contract used inside run_folding_on_context.
    batch = Collate(
        feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32
    )([context])
    prepared = _prepared_manifest(dict(batch["features"]), dict(batch["inputs"]))
    metadata = _base_metadata(args)
    metadata["source_weight_sha256"] = source_weight_manifest_from_torch(
        model_directory
    )
    metadata["prepared_tensors"] = prepared
    metadata["versions"] = {
        "torch": torch.__version__,
        "chai_lab": __import__("chai_lab").__version__,
    }
    numeric_arrays = _numeric_prepared_arrays(dict(batch["features"]))
    if args.prepared_only:
        _save_prepared_only(args.artifact, metadata, numeric_arrays)
        return
    candidates = run_folding_on_context(
        context,
        output_dir=run_dir,
        num_trunk_recycles=args.recycles,
        num_diffn_timesteps=args.timesteps,
        num_diffn_samples=args.samples,
        seed=args.seed,
        device=torch.device(args.device),
        low_memory=True,
    )
    coords, atom_ids = read_cif_coordinates(candidates.cif_paths)
    score_paths = [
        path.with_name(path.name.replace("pred.", "scores.")).with_suffix(".npz")
        for path in candidates.cif_paths
    ]
    save_artifact(
        args.artifact,
        coords=coords,
        atom_ids=atom_ids,
        pae=_numpy(candidates.pae),
        pde=_numpy(candidates.pde),
        plddt=_numpy(candidates.plddt),
        ranking=_ranking_from_score_files(score_paths),
        metadata=metadata,
        prepared_arrays=numeric_arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("torch", "jax"))
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--conformers", type=Path)
    parser.add_argument("--torch-model-directory", type=Path)
    parser.add_argument("--msa-directory", type=Path)
    parser.add_argument("--template-hits", type=Path)
    parser.add_argument("--template-cif-directory", type=Path)
    parser.add_argument("--restraint", type=Path)
    parser.add_argument("--reference-augmentation-tape", type=Path)
    parser.add_argument("--compilation-cache", type=Path)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepared-only", action="store_true")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "jax":
        _export_jax(args, args.run_dir)
    else:
        _export_torch(args, args.run_dir)
    print(json.dumps({"artifact": str(args.artifact), "backend": args.backend}))


if __name__ == "__main__":
    main()
