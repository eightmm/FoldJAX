"""foldjax.models.boltz2 predict CLI: raw YAML -> structure file (PDB/mmCIF).

Featurizes the input via foldjax.models.boltz2.data, runs the JAX structure sampler with
native weights, and writes the predicted structure.

Run:
  uv run python scripts/predict.py \
      --input X.yaml \
      --weights outputs/native_weights/boltz2_conf \
      --mols .cache/boltz/mols \
      --out-dir outputs/predictions \
      [--steps 200 --recycling 3 --compute-dtype float32|bfloat16 --fmt pdb|cif]
      [--use-msa-server]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from foldjax.models.boltz2.data.featurize import featurize_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="raw input YAML")
    p.add_argument(
        "--seq", action="append", default=[],
        help="protein sequence (repeatable); builds a job YAML when --input is "
        "omitted",
    )
    p.add_argument("--dna", action="append", default=[], help="DNA sequence")
    p.add_argument("--rna", action="append", default=[], help="RNA sequence")
    p.add_argument(
        "--ligand-ccd", action="append", default=[], help="ligand CCD code"
    )
    p.add_argument(
        "--ligand-smiles", action="append", default=[], help="ligand SMILES"
    )
    p.add_argument(
        "--weights", type=Path, default=ROOT / "outputs/native_weights/boltz2_conf"
    )
    p.add_argument(
        "--affinity-weights",
        type=Path,
        default=None,
        help="native affinity bundle; affinity jobs default to boltz2_aff next "
        "to --weights",
    )
    p.add_argument(
        "--mols", type=Path, default=ROOT / ".cache/boltz/mols"
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/predictions")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--recycling", type=int, default=3)
    p.add_argument("--diffusion-samples", type=int, default=1)
    p.add_argument("--sampling-steps-affinity", type=int, default=200)
    p.add_argument("--diffusion-samples-affinity", type=int, default=5)
    p.add_argument("--affinity-mw-correction", action="store_true")
    p.add_argument(
        "--compute-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    p.add_argument(
        "--triangle-backend",
        choices=["xla", "pallas", "tokamax", "cueq"],
        default="cueq",
    )
    p.add_argument(
        "--triangle-multiplication-backend",
        choices=["xla", "cueq"],
        default="cueq",
    )
    p.add_argument("--glu-backend", choices=["xla", "tokamax"], default="xla")
    p.add_argument("--fmt", choices=["pdb", "cif"], default="pdb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--use-potentials",
        action="store_true",
        help="enable upstream Boltz-2 FK resampling and physical/contact "
        "guidance potentials",
    )
    p.add_argument(
        "--use-msa-server",
        action="store_true",
        help="generate MSAs via the colabfold mmseqs2 server for protein chains "
        "that request one (msa not 'empty' and no precomputed a3m/csv)",
    )
    p.add_argument("--msa-server-url", default="https://api.colabfold.com")
    p.add_argument(
        "--msa-pairing-strategy", choices=["greedy", "complete"], default="greedy"
    )
    p.add_argument(
        "--msa-server-username",
        default=None,
        help="basic-auth username (or BOLTZ_MSA_USERNAME)",
    )
    p.add_argument(
        "--msa-server-password",
        default=None,
        help="basic-auth password (or BOLTZ_MSA_PASSWORD)",
    )
    p.add_argument(
        "--msa-api-key-header",
        default=None,
        help="API-key header name (default X-API-Key)",
    )
    p.add_argument(
        "--msa-api-key-value",
        default=None,
        help="API-key value (or MSA_API_KEY_VALUE)",
    )
    p.add_argument(
        "--compile-cache",
        type=Path,
        default=ROOT / "outputs/compile_cache",
        help="persistent XLA compilation cache dir (reuse compiles across runs); "
        "on by default, pass a different path to relocate",
    )
    p.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "outputs/feature_cache",
        help="persistent featurization cache dir (memoize features by input "
        "digest; on by default, pass a different path to relocate)",
    )
    p.add_argument(
        "--bucket",
        action="store_true",
        help="pad token/atom dims to a shape ladder so the compile cache hits "
        "across different-length targets (serving). Pads FLOPs (single-run "
        "loss, multi-run win); shifts real coords by ~1e-4 A (fp reassociation "
        "from padded reductions), biologically negligible.",
    )
    p.add_argument(
        "--prewarm-only",
        action="store_true",
        help="compile and populate --compile-cache for this input/profile, then "
        "exit without writing prediction files",
    )
    args = p.parse_args()
    if args.use_msa_server:
        args.msa_server_username = args.msa_server_username or os.environ.get(
            "BOLTZ_MSA_USERNAME"
        )
        args.msa_server_password = args.msa_server_password or os.environ.get(
            "BOLTZ_MSA_PASSWORD"
        )
        args.msa_api_key_value = args.msa_api_key_value or os.environ.get(
            "MSA_API_KEY_VALUE"
        )
    assert args.num_samples > 0, "--diffusion-samples must be positive"
    os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = (
        args.triangle_multiplication_backend
    )

    if args.input is None:
        any_entity = (
            args.seq or args.dna or args.rna or args.ligand_ccd or args.ligand_smiles
        )
        assert any_entity, "provide --input YAML or --seq/--dna/--rna/--ligand-*"
        from foldjax.models.boltz2.data.job_yaml import build_job_yaml

        gen_dir = args.out_dir / "prep" / "job"
        gen_dir.mkdir(parents=True, exist_ok=True)
        args.input = gen_dir / "job.yaml"
        args.input.write_text(
            build_job_yaml(
                args.seq,
                args.ligand_ccd,
                dna=args.dna,
                rna=args.rna,
                ligands_smiles=args.ligand_smiles,
                use_msa_server=args.use_msa_server,
            )
        )
        print(f"built job YAML: {args.input}")
    assert args.input.exists(), f"missing input: {args.input}"

    # --- Step 1: YAML -> processed tree + features (torch side) ---
    prep_dir = args.out_dir / "prep" / args.input.stem
    prep_dir.mkdir(parents=True, exist_ok=True)
    feats_np, manifest, struct_dir = featurize_yaml(
        args.input,
        prep_dir,
        args.mols,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        msa_pairing_strategy=args.msa_pairing_strategy,
        msa_server_username=args.msa_server_username,
        msa_server_password=args.msa_server_password,
        msa_api_key_header=args.msa_api_key_header,
        msa_api_key_value=args.msa_api_key_value,
        cache_dir=args.feature_cache,
    )
    if manifest is not None:
        records = manifest.records
        assert len(records) == 1, f"expected 1 record, got {len(records)}"
        record_id = records[0].id
    else:  # cache hit: record id is the structure npz filename
        npzs = sorted(struct_dir.glob("*.npz"))
        assert len(npzs) == 1, f"expected 1 cached structure, got {len(npzs)}"
        record_id = npzs[0].stem
        print("(feature cache hit)")
    print(f"featurized record: {record_id}")

    # --- Step 2: run the JAX sampler (no torch/boltz at runtime here) ---
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.predict import boltz2_predict

    jax.config.update("jax_default_matmul_precision", "highest")
    if args.compile_cache is not None:
        cache = args.compile_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        print(f"compile cache: {cache}")
    compute_dtype = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }[args.compute_dtype]
    steering_args = None
    if args.use_potentials:
        steering_args = {
            "fk_steering": True,
            "num_particles": 3,
            "fk_lambda": 4.0,
            "fk_resampling_interval": 3,
            "physical_guidance_update": True,
            "contact_guidance_update": True,
            "num_gd_steps": 20,
        }

    params = load_params(args.weights)
    affinity_model_params = None
    if bool(np.any(feats_np["affinity_token_mask"])):
        affinity_path = (
            args.affinity_weights
            if args.affinity_weights is not None
            else args.weights.with_name("boltz2_aff")
        )
        exists = affinity_path.is_file() or any(
            affinity_path.with_suffix(suffix).is_file()
            for suffix in (".safetensors", ".npz")
        )
        if not exists:
            raise FileNotFoundError(
                "affinity weights are required for an affinity job; "
                f"not found: {affinity_path}(.safetensors|.npz)"
            )
        affinity_model_params = load_params(affinity_path)
        if not {"trunk", "affinity"} <= affinity_model_params.keys():
            raise ValueError(
                "native affinity weights use the obsolete head-only format; "
                "rerun scripts/setup.sh"
            )

    original_tokens = int(feats_np["token_pad_mask"].shape[-1])
    original_atoms = int(feats_np["atom_pad_mask"].shape[-1])
    if args.bucket:
        from foldjax.models.boltz2.data.bucket import pad_feats, resolve_bucket_shape

        tgt_tok, tgt_atom, tgt_msa = resolve_bucket_shape(feats_np)
        feats_np, _log = pad_feats(
            feats_np, tgt_tok, tgt_atom, target_msa=tgt_msa
        )
        print(
            f"bucket: tokens {original_tokens}->{tgt_tok} "
            f"atoms {original_atoms}->{tgt_atom} msa->{tgt_msa}"
        )

    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}

    predict_kwargs = {
        "recycling_steps": args.recycling,
        "num_sampling_steps": args.steps,
        "augmentation": False,
        "run_confidence": True,
        "run_distogram": True,
        "run_bfactor": True,
        "compute_dtype": compute_dtype,
        "use_scan": True,
        "multiplicity": args.num_samples,
        "triangle_backend": args.triangle_backend,
        "glu_backend": args.glu_backend,
        "confidence_sequentially": args.num_samples > 1,
        "return_pair_chains_iptm": False,
        "recompute_nonpolymer_frames": bool(np.any(feats_np["mol_type"] == 3)),
        "steering_args": steering_args,
    }
    def run_model(model_params, model_feats, model_key):
        return boltz2_predict(model_params, model_feats, model_key, **predict_kwargs)

    model_args = (params, feats, jax.random.PRNGKey(args.seed))

    runner = run_model if args.use_potentials else jax.jit(run_model)
    out = runner(*model_args)
    coords = np.asarray(jax.block_until_ready(out["sample_atom_coords"]))
    plddt = np.asarray(out["plddt"]).reshape(-1)
    assert np.all(np.isfinite(coords)), "non-finite coordinates produced"

    if affinity_model_params is not None:
        from foldjax.models.boltz2.api import _prepare_affinity_features

        coords_batched = coords.reshape(args.num_samples, -1, 3)
        best_idx = int(np.argmax(np.asarray(out["iptm"])))
        affinity_feats_np = _prepare_affinity_features(
            input_path=args.input,
            struct_dir=struct_dir,
            record_id=record_id,
            mol_dir=args.mols,
            predicted_coords=coords_batched[best_idx : best_idx + 1],
            atom_pad_mask=feats_np["atom_pad_mask"],
            out_dir=args.out_dir / "affinity_stage",
        )
        if args.bucket:
            from foldjax.models.boltz2.data.bucket import (
                pad_feats,
                resolve_bucket_shape,
            )

            aff_tok, aff_atom, aff_msa = resolve_bucket_shape(affinity_feats_np)
            affinity_feats_np, _ = pad_feats(
                affinity_feats_np, aff_tok, aff_atom, target_msa=aff_msa
            )
        affinity_feats = {
            key: jnp.asarray(value) for key, value in affinity_feats_np.items()
        }
        affinity_kwargs = {
            **predict_kwargs,
            "recycling_steps": 5,
            "num_sampling_steps": args.sampling_steps_affinity,
            "multiplicity": args.diffusion_samples_affinity,
            "steering_args": None,
            "run_bfactor": "bfactor" in affinity_model_params,
            "confidence_sequentially": args.diffusion_samples_affinity > 1,
            "recompute_nonpolymer_frames": True,
            "affinity_mw_correction": args.affinity_mw_correction,
        }

        def run_affinity(model_params, model_feats, model_key):
            return boltz2_predict(
                model_params, model_feats, model_key, **affinity_kwargs
            )

        affinity_out = jax.jit(run_affinity)(
            affinity_model_params,
            affinity_feats,
            jax.random.PRNGKey(args.seed),
        )
        out.update(
            {
                key: value
                for key, value in affinity_out.items()
                if key.startswith("affinity_")
            }
        )

    if args.prewarm_only:
        jax.block_until_ready(out)
        print(
            "prewarmed: "
            f"tokens={feats_np['token_pad_mask'].shape[-1]} "
            f"atoms={feats_np['atom_pad_mask'].shape[-1]} "
            f"samples={args.num_samples}"
        )
        return

    # --- Step 3: write structure file ---
    from foldjax.models.boltz2.data.write.structure import write_prediction

    atom_pad_mask = feats_np["atom_pad_mask"].reshape(-1)
    padded_coords = coords.reshape(args.num_samples, -1, 3)
    coords = padded_coords[:, :original_atoms]
    plddt = np.asarray(out["plddt"]).reshape(args.num_samples, -1)
    plddt = plddt[:, :original_tokens]
    written = []
    for model_idx in range(args.num_samples):
        suffix = "" if args.num_samples == 1 else f"_model_{model_idx}"
        out_path = args.out_dir / f"{record_id}{suffix}.{args.fmt}"
        written.append(
            write_prediction(
                structure_npz=struct_dir / f"{record_id}.npz",
                coords=padded_coords[model_idx],
                atom_pad_mask=atom_pad_mask,
                out_path=out_path,
                plddts=plddt[model_idx],
                fmt=args.fmt,
            )
        )

    print(f"sample_atom_coords shape: {coords.shape}")
    print(
        f"pLDDT: min={plddt.min():.4f} mean={plddt.mean():.4f} "
        f"max={plddt.max():.4f}"
    )
    for path in written:
        print(f"WROTE {path}")
    for key in ("affinity_pred_value", "affinity_probability_binary"):
        if key in out:
            print(f"{key}: {np.asarray(out[key]).reshape(-1).tolist()}")


if __name__ == "__main__":
    main()
