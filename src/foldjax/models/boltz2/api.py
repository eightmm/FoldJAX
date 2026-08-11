"""High-level Python API: sequences / YAML -> predicted structure.

Wraps featurization + the JAX sampler so other code (incl. other JAX models)
can call Boltz-2 inference without the CLI. Heavy deps (jax, torch) are imported
lazily, so ``import foldjax.models.boltz2`` stays cheap.

  import foldjax.models.boltz2
  out = foldjax.models.boltz2.predict(seq=["MKQLED..."], ligand_ccd=["ATP"],
                          weights="outputs/native_weights/boltz2_conf",
                          mols=".cache/boltz/mols")
  out["coords"]       # (n_atom, 3) np.ndarray
  out["plddt"]        # (n_atom,) np.ndarray

For composing with other JAX models, use the lower-level
``boltz2_predict`` (a pure JAX function over params + feature pytree)
and ``foldjax.models.boltz2.load_params`` directly.
"""

from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.boltz2.data.featurize import featurize_yaml
from foldjax.models.boltz2.data.job_yaml import build_job_yaml

#: Trunk dtypes `predict` accepts, as names rather than `jnp` dtypes so that
#: importing this module does not pull in JAX. fp16 is absent deliberately: the
#: sampler's `inf` constants saturate in half precision and a fully-masked
#: softmax row then gives NaN.
COMPUTE_DTYPES = ("bfloat16", "float32")

#: Matmul precision this port runs under. Boltz-2 is the one model here whose
#: upstream asks for true float32 -- `main.py:1096` is
#: `torch.set_float32_matmul_precision("highest")`, where OpenFold3, Protenix and
#: OpenDDE all select TF32 -- so "highest" is upstream parity, not caution.
MATMUL_PRECISION = "highest"


def _pinned_matmul_precision(function):
    """Run `function` with the matmul precision scoped, not latched.

    This used to be a `jax.config.update` inside `predict`, which is
    process-global and stayed set after the call returned: a notebook that ran
    Boltz-2 and then anything else left that other thing in float32. It went
    unnoticed while every port pinned the same value; they no longer do.

    JAX is imported inside the wrapper rather than at module scope, because
    importing this module has to stay cheap and torch-free -- `test_import`
    asserts it.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        import jax

        with jax.default_matmul_precision(MATMUL_PRECISION):
            return function(*args, **kwargs)

    return wrapper


def featurize(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    mols: str | Path,
    out_dir: str | Path | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    feature_cache: str | Path | None = None,
    max_msa_depth: int | None = None,
) -> tuple[dict[str, np.ndarray], str, Path]:
    """Featurize a YAML path or bare entities.

    Returns ``(feats, record_id, struct_dir)``.
    """
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if msa_api_key_value is None:
            msa_api_key_value = os.environ.get("MSA_API_KEY_VALUE")
    work = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    if input is None:
        if not (seq or dna or rna or ligand_ccd or ligand_smiles):
            raise ValueError("provide `input` YAML or sequences/ligands")
        work.mkdir(parents=True, exist_ok=True)
        input = work / "job.yaml"
        Path(input).write_text(
            build_job_yaml(
                seq, ligand_ccd, dna=dna, rna=rna,
                ligands_smiles=ligand_smiles, use_msa_server=use_msa_server,
            )
        )
    feats, manifest, struct_dir = featurize_yaml(
        Path(input), work, Path(mols),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        msa_api_key_header=msa_api_key_header,
        msa_api_key_value=msa_api_key_value,
        cache_dir=Path(feature_cache) if feature_cache is not None else None,
        max_msa_depth=max_msa_depth,
    )
    if manifest is not None:
        record_id = manifest.records[0].id
    else:
        record_id = sorted(struct_dir.glob("*.npz"))[0].stem
    return feats, record_id, struct_dir


@_pinned_matmul_precision
def predict(
    *,
    input: str | Path | None = None,
    seq: Sequence[str] = (),
    dna: Sequence[str] = (),
    rna: Sequence[str] = (),
    ligand_ccd: Sequence[str] = (),
    ligand_smiles: Sequence[str] = (),
    weights: str | Path,
    affinity_weights: str | Path | None = None,
    mols: str | Path,
    out_dir: str | Path | None = None,
    steps: int = 200,
    recycling: int = 3,
    diffusion_samples: int = 1,
    affinity_steps: int = 200,
    affinity_diffusion_samples: int = 5,
    affinity_mw_correction: bool = False,
    seed: int = 0,
    # Upstream's predict path is `Trainer(..., precision="bf16-mixed")`
    # (`main.py:1262`), so float32 was this port inventing a setting Boltz-2
    # does not ship. The trunk runs in this dtype; the diffusion and confidence
    # modules stay float32 either way, which is the same split upstream's
    # autocast draws. Measured at 1,003 tokens over an 8,192-row alignment,
    # five samples of the released schedule: 87.3 s -> 65.0 s warm, 13,035 ->
    # 10,218 MiB peak, and against the deposited structure TM 0.9932 -> 0.9931,
    # the two crossing in both directions per sample. Passing "float32"
    # restores the previous behaviour exactly.
    compute_dtype: str = "bfloat16",
    # False by default: the full-bin pae/pde/plddt logits and the distogram are
    # program outputs nothing in the cif/JSON path reads, and XLA keeps entry
    # outputs resident alongside the temp arena for the whole run -- at 3,012
    # tokens and 5 samples that is >20 GiB of peak for arrays that were thrown
    # away. The summaries (plddt, pae, ptm/iptm, complex_*) are unaffected.
    # Pass True to get the raw logits and pdistogram back in `raw`.
    return_confidence_logits: bool = False,
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    steering_args: Mapping[str, object] | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
    feature_cache: str | Path | None = None,
    compile_cache: str | Path | None = None,
    bucket: bool = False,
    write_fmt: str | None = None,
    max_msa_depth: int | None = None,
) -> dict[str, Any]:
    """Run end-to-end Boltz-2 inference.

    Pass either ``input`` (a job YAML) or bare ``seq``/``dna``/``rna``/
    ``ligand_ccd``/``ligand_smiles``. Affinity jobs automatically load a
    sibling ``boltz2_aff`` bundle unless ``affinity_weights`` is explicit.
    Returns a dict with ``coords`` (n_atom, 3), ``plddt``, ``record_id``,
    ``raw`` (full model output), and ``out_path`` (if ``write_fmt`` is
    "pdb"/"cif"). Defaults match the Boltz-2 reference.
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.predict import boltz2_predict

    if diffusion_samples <= 0:
        raise ValueError("diffusion_samples must be positive")

    feats_np, record_id, struct_dir = featurize(
        input=input, seq=seq, dna=dna, rna=rna,
        ligand_ccd=ligand_ccd, ligand_smiles=ligand_smiles,
        mols=mols, out_dir=out_dir, use_msa_server=use_msa_server,
        msa_server_url=msa_server_url, msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        msa_api_key_header=msa_api_key_header,
        msa_api_key_value=msa_api_key_value,
        feature_cache=feature_cache,
        max_msa_depth=max_msa_depth,
    )

    if compile_cache is not None:
        cache = Path(compile_cache).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(
            f"compute_dtype must be one of {COMPUTE_DTYPES}, got {compute_dtype!r}"
        )
    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[compute_dtype]
    confidence_weights = Path(weights)
    params = load_params(confidence_weights)
    affinity_model_params = None
    affinity_requested = bool(np.any(feats_np["affinity_token_mask"]))
    if affinity_requested:
        affinity_path = (
            Path(affinity_weights)
            if affinity_weights is not None
            else confidence_weights.with_name("boltz2_aff")
        )
        if not _native_weights_exist(affinity_path):
            raise FileNotFoundError(
                "affinity weights are required for an affinity job; "
                f"not found: {affinity_path}(.safetensors|.npz)"
            )
        affinity_model_params = load_params(affinity_path)
        if not {"trunk", "affinity"} <= affinity_model_params.keys():
            raise ValueError(
                "native affinity weights use the obsolete head-only format; "
                "rerun scripts/setup.sh to export the complete affinity model"
            )

    original_tokens = int(feats_np["token_pad_mask"].shape[-1])
    original_atoms = int(feats_np["atom_pad_mask"].shape[-1])
    if bucket:
        from foldjax.models.boltz2.data.bucket import pad_feats, resolve_bucket_shape

        tgt_tok, tgt_atom, tgt_msa = resolve_bucket_shape(feats_np)
        feats_np, _ = pad_feats(
            feats_np, tgt_tok, tgt_atom, target_msa=tgt_msa
        )

    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}
    predict_kwargs = {
        "recycling_steps": recycling,
        "num_sampling_steps": steps,
        "augmentation": False,
        "steering_args": steering_args,
        "run_confidence": True,
        "run_distogram": return_confidence_logits,
        "return_confidence_logits": return_confidence_logits,
        "run_bfactor": True,
        "compute_dtype": dtype,
        "use_scan": True,
        "trunk_use_scan": True,
        "score_use_scan": True,
        "multiplicity": diffusion_samples,
        "attention_backend": attention_backend,
        "triangle_backend": triangle_backend,
        "glu_backend": glu_backend,
        "confidence_sequentially": diffusion_samples > 1,
        "return_pair_chains_iptm": False,
        "recompute_nonpolymer_frames": bool(np.any(feats_np["mol_type"] == 3)),
        # The featurizer always emits template_* arrays, zero-filled when no
        # template was given. Whether they are real is a value question, so it
        # is answered here on concrete arrays rather than inside the jit.
        "use_template": bool(np.any(feats_np.get("template_mask", np.zeros(1)) != 0)),
    }
    def run_model(model_params, model_feats, model_key):
        return boltz2_predict(model_params, model_feats, model_key, **predict_kwargs)

    model_args = (params, feats, jax.random.PRNGKey(seed))

    steering_active = steering_args is not None and any(
        bool(steering_args.get(key, False))
        for key in (
            "fk_steering",
            "physical_guidance_update",
            "contact_guidance_update",
        )
    )
    runner = run_model if steering_active else jax.jit(run_model)
    out = runner(*model_args)
    coords_batched = np.asarray(
        jax.block_until_ready(out["sample_atom_coords"])
    ).reshape(diffusion_samples, -1, 3)
    plddt_batched = np.asarray(out["plddt"]).reshape(diffusion_samples, -1)
    assert np.all(np.isfinite(coords_batched)), "non-finite coordinates produced"

    if affinity_model_params is not None:
        best_idx = int(np.argmax(np.asarray(out["iptm"])))
        affinity_feats_np = _prepare_affinity_features(
            input_path=Path(input) if input is not None else None,
            struct_dir=struct_dir,
            record_id=record_id,
            mol_dir=Path(mols),
            predicted_coords=coords_batched[best_idx : best_idx + 1],
            atom_pad_mask=feats_np["atom_pad_mask"],
            out_dir=(Path(out_dir) if out_dir is not None else struct_dir.parent)
            / "affinity_stage",
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            msa_api_key_header=msa_api_key_header,
            msa_api_key_value=msa_api_key_value,
        )
        if bucket:
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
            "num_sampling_steps": affinity_steps,
            "steering_args": None,
            "multiplicity": affinity_diffusion_samples,
            "run_bfactor": "bfactor" in affinity_model_params,
            "confidence_sequentially": affinity_diffusion_samples > 1,
            "recompute_nonpolymer_frames": True,
            "affinity_mw_correction": affinity_mw_correction,
        }

        def run_affinity(model_params, model_feats, model_key):
            return boltz2_predict(
                model_params, model_feats, model_key, **affinity_kwargs
            )

        affinity_out = jax.jit(run_affinity)(
            affinity_model_params,
            affinity_feats,
            jax.random.PRNGKey(seed),
        )
        out.update(
            {
                key: value
                for key, value in affinity_out.items()
                if key.startswith("affinity_")
            }
        )

    from foldjax.models.boltz2.data.bucket import crop_prediction_outputs

    public_out = (
        crop_prediction_outputs(out, original_tokens, original_atoms)
        if bucket
        else out
    )
    public_coords_batched = (
        coords_batched[:, :original_atoms] if bucket else coords_batched
    )
    public_plddt_batched = (
        plddt_batched[:, :original_tokens] if bucket else plddt_batched
    )
    public_coords = (
        public_coords_batched[0]
        if diffusion_samples == 1
        else public_coords_batched
    )
    public_plddt = (
        public_plddt_batched[0]
        if diffusion_samples == 1
        else public_plddt_batched
    )
    result: dict[str, Any] = {
        "coords": public_coords,
        "plddt": public_plddt,
        "record_id": record_id,
        "raw": public_out,
    }
    result.update(
        {
            key: value
            for key, value in public_out.items()
            if key.startswith("affinity_")
        }
    )
    if write_fmt is not None:
        from foldjax.models.boltz2.data.write.structure import write_prediction

        dest = Path(out_dir) if out_dir is not None else struct_dir.parent
        dest.mkdir(parents=True, exist_ok=True)
        paths = []
        for model_idx in range(diffusion_samples):
            suffix = "" if diffusion_samples == 1 else f"_model_{model_idx}"
            out_path = dest / f"{record_id}{suffix}.{write_fmt}"
            paths.append(
                write_prediction(
                    structure_npz=struct_dir / f"{record_id}.npz",
                    coords=coords_batched[model_idx],
                    atom_pad_mask=feats_np["atom_pad_mask"].reshape(-1),
                    out_path=out_path,
                    plddts=public_plddt_batched[model_idx],
                    fmt=write_fmt,
                )
            )
        if diffusion_samples == 1:
            result["out_path"] = paths[0]
        else:
            result["out_paths"] = paths
    return result


def _native_weights_exist(path: Path) -> bool:
    return path.is_file() or any(
        path.with_suffix(suffix).is_file() for suffix in (".safetensors", ".npz")
    )


def _prepare_affinity_features(
    *,
    input_path: Path | None,
    struct_dir: Path,
    record_id: str,
    mol_dir: Path,
    predicted_coords: np.ndarray,
    atom_pad_mask: np.ndarray,
    out_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    msa_api_key_header: str | None = None,
    msa_api_key_value: str | None = None,
) -> dict[str, np.ndarray]:
    from foldjax.models.boltz2.data.featurize import (
        featurize_affinity_from_prediction,
        featurize_yaml,
    )
    from foldjax.models.boltz2.data.types import Manifest

    processed_dir = struct_dir.parent
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = Manifest.load(manifest_path)
    else:
        if input_path is None:
            raise ValueError(
                "affinity feature-cache recovery requires the original input YAML"
            )
        _, manifest, regenerated_struct_dir = featurize_yaml(
            input_path,
            out_dir / "source",
            mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            msa_api_key_header=msa_api_key_header,
            msa_api_key_value=msa_api_key_value,
            cache_dir=None,
        )
        processed_dir = regenerated_struct_dir.parent
        if manifest.records[0].id != record_id:
            raise ValueError("affinity re-featurization changed the record id")
    return featurize_affinity_from_prediction(
        manifest=manifest,
        processed_dir=processed_dir,
        mol_dir=mol_dir,
        predicted_coords=predicted_coords,
        atom_pad_mask=atom_pad_mask,
        out_dir=out_dir,
    )
