#!/usr/bin/env python3
"""Export the real prepared 256-crop confidence/postprocess parity fixture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_upstream_postprocess(
    upstream_root: Path, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    python = upstream_root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"upstream Chai Python is unavailable: {python}")
    script = r"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from chai_lab.ranking.rank import get_scores, rank

d = np.load(sys.argv[1], allow_pickle=False)
t = lambda name: torch.from_numpy(d[name])
pair_centers = torch.linspace(0.0, 32.0, 129)[1::2]
plddt_centers = torch.linspace(0.0, 1.0, 101)[1::2]
ranking = rank(
    atom_coords=t("coords"),
    atom_mask=t("atom_mask")[None],
    atom_token_index=t("atom_token_index")[None],
    token_exists_mask=t("token_mask")[None],
    token_asym_id=t("token_asym_id")[None],
    token_entity_type=t("token_entity_type")[None],
    token_valid_frames_mask=t("valid_frames_mask")[None],
    lddt_logits=t("plddt_logits"),
    lddt_bin_centers=plddt_centers,
    pae_logits=t("pae_logits"),
    pae_bin_centers=pair_centers,
)
pae = (torch.softmax(t("pae_logits"), -1) * pair_centers).sum(-1)
pde = (torch.softmax(t("pde_logits"), -1) * pair_centers).sum(-1)
atom_plddt = (torch.softmax(t("plddt_logits"), -1) * plddt_centers).sum(-1)
token_sum = torch.zeros((1, d["token_mask"].size), dtype=torch.float32)
token_count = torch.zeros(d["token_mask"].size, dtype=torch.float32)
token_sum.scatter_add_(1, t("atom_token_index")[None], atom_plddt)
token_count.scatter_add_(0, t("atom_token_index"), t("atom_mask").float())
plddt = token_sum / token_count.clamp(min=1)[None]
np.savez(
    sys.argv[2],
    pae=pae.numpy(),
    pde=pde.numpy(),
    plddt=plddt.numpy(),
    total_clashes=ranking.clash_scores.total_clashes.numpy(),
    total_inter_chain_clashes=(
        ranking.clash_scores.total_inter_chain_clashes.numpy()
    ),
    **get_scores(ranking),
)
"""
    with tempfile.TemporaryDirectory(prefix="chai-confidence-") as temp:
        input_path = Path(temp) / "inputs.npz"
        output_path = Path(temp) / "outputs.npz"
        np.savez(input_path, **inputs)
        subprocess.run(
            [python, "-c", script, input_path, output_path],
            cwd=upstream_root,
            check=True,
        )
        with np.load(output_path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--conformers", type=Path, required=True)
    parser.add_argument("--torch-context", type=Path, required=True)
    parser.add_argument("--torch-artifact", type=Path, required=True)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import torch

    from foldjax.models.chai.inference import InferenceConfig, prepare_inference
    from foldjax.models.chai.models.confidence import (
        confidence_head_forward,
        map_confidence_head,
    )
    from foldjax.models.chai.ranking.frames import get_frames_and_mask

    prepared, _ = prepare_inference(
        args.fasta,
        bundle_path=args.bundle,
        conformer_path=args.conformers,
        config=InferenceConfig(use_esm_embeddings=False, seed=0),
    )
    if prepared.model_size != 256:
        raise ValueError(f"fixture must use the 256 crop, got {prepared.model_size}")
    values = {name: np.asarray(value) for name, value in prepared.padded_inputs.items()}
    with np.load(args.torch_context, allow_pickle=False) as archive:
        context = {name: archive[name] for name in archive.files}
    with np.load(args.torch_artifact, allow_pickle=False) as archive:
        torch_artifact = {name: archive[name] for name in archive.files}

    atom_count = values["atom_exists_mask"].shape[-1]
    active_atoms = int(values["atom_exists_mask"].sum())
    active_tokens = int(values["token_exists_mask"].sum())
    if torch_artifact["coords"].shape != (1, active_atoms, 3):
        raise ValueError("Torch artifact coordinates do not match prepared atom order")
    coords = np.zeros((1, atom_count, 3), dtype=np.float32)
    coords[:, :active_atoms] = torch_artifact["coords"]

    component = torch.jit.load(str(args.component), map_location="cpu").eval().cuda()
    state = {
        name: value.detach().cpu().numpy()
        for name, value in component.state_dict().items()
    }
    torch_inputs = {
        "token_single_input_repr": torch.from_numpy(
            context["token_single_initial_repr"]
        ).to("cuda", dtype=torch.bfloat16),
        "token_single_trunk_repr": torch.from_numpy(
            context["token_single_trunk_repr"]
        ).to("cuda", dtype=torch.bfloat16),
        "token_pair_trunk_repr": torch.from_numpy(
            context["token_pair_trunk_repr"]
        ).to("cuda", dtype=torch.bfloat16),
        "token_single_mask": torch.from_numpy(values["token_exists_mask"]).cuda(),
        "atom_single_mask": torch.from_numpy(values["atom_exists_mask"]).cuda(),
        "atom_coords": torch.from_numpy(coords).cuda(),
        "token_reference_atom_index": torch.from_numpy(
            values["token_ref_atom_index"]
        ).cuda(),
        "atom_token_index": torch.from_numpy(values["atom_token_index"]).cuda(),
        "atom_within_token_index": torch.from_numpy(
            values["atom_within_token_index"]
        ).cuda(),
    }
    with torch.inference_mode():
        torch_logits = component.forward_256(**torch_inputs)
    torch_logits = tuple(value.float().cpu().numpy() for value in torch_logits)
    del component, torch_inputs
    gc.collect()
    torch.cuda.empty_cache()

    params = map_confidence_head(state)
    jax_inputs = {
        "token_single_input_repr": jnp.asarray(context["token_single_initial_repr"]),
        "token_single_trunk_repr": jnp.asarray(context["token_single_trunk_repr"]),
        "token_pair_trunk_repr": jnp.asarray(context["token_pair_trunk_repr"]),
        "token_single_mask": jnp.asarray(values["token_exists_mask"]),
        "atom_single_mask": jnp.asarray(values["atom_exists_mask"]),
        "atom_coords": jnp.asarray(coords),
        "token_reference_atom_index": jnp.asarray(values["token_ref_atom_index"]),
        "atom_token_index": jnp.asarray(values["atom_token_index"]),
        "atom_within_token_index": jnp.asarray(values["atom_within_token_index"]),
    }
    jax_logits = jax.jit(confidence_head_forward)(**jax_inputs, params=params)
    jax_logits = tuple(np.asarray(value, np.float32) for value in jax_logits)
    _, valid_frames = get_frames_and_mask(
        jnp.asarray(coords),
        jnp.asarray(values["token_asym_id"]),
        jnp.asarray(values["token_residue_index"]),
        jnp.asarray(values["token_backbone_frame_mask"]),
        jnp.asarray(values["token_centre_atom_index"]),
        jnp.asarray(values["token_exists_mask"]),
        jnp.asarray(values["atom_exists_mask"]),
        jnp.asarray(values["token_backbone_frame_index"]),
        jnp.asarray(values["atom_token_index"]),
    )

    token_slice = slice(0, active_tokens)
    atom_slice = slice(0, active_atoms)
    arrays: dict[str, np.ndarray] = {
        "coords": coords[:, atom_slice],
        "token_mask": values["token_exists_mask"][0, token_slice],
        "atom_mask": values["atom_exists_mask"][0, atom_slice],
        "atom_token_index": values["atom_token_index"][0, atom_slice],
        "token_asym_id": values["token_asym_id"][0, token_slice],
        "token_entity_type": values["token_entity_type"][0, token_slice],
        "valid_frames_mask": np.asarray(valid_frames)[0, token_slice],
    }
    for prefix, logits in (("torch", torch_logits), ("jax", jax_logits)):
        arrays[f"{prefix}_pae_logits"] = logits[0][:, token_slice, token_slice]
        arrays[f"{prefix}_pde_logits"] = logits[1][:, token_slice, token_slice]
        arrays[f"{prefix}_plddt_logits"] = logits[2][:, atom_slice]
    reference = _run_upstream_postprocess(
        args.component.resolve().parents[2],
        {
            "coords": arrays["coords"],
            "token_mask": arrays["token_mask"],
            "atom_mask": arrays["atom_mask"],
            "atom_token_index": arrays["atom_token_index"],
            "token_asym_id": arrays["token_asym_id"],
            "token_entity_type": arrays["token_entity_type"],
            "valid_frames_mask": arrays["valid_frames_mask"],
            "pae_logits": arrays["torch_pae_logits"],
            "pde_logits": arrays["torch_pde_logits"],
            "plddt_logits": arrays["torch_plddt_logits"],
        },
    )
    for name in ("pae", "pde", "plddt"):
        arrays[f"torch_{name}_scores"] = reference[name]
    for name in (
        "aggregate_score",
        "ptm",
        "iptm",
        "per_chain_ptm",
        "per_chain_pair_iptm",
        "has_inter_chain_clashes",
        "chain_chain_clashes",
    ):
        arrays[f"torch_score__{name}"] = reference[name]
    arrays["torch_total_clashes"] = reference["total_clashes"]
    arrays["torch_total_inter_chain_clashes"] = reference[
        "total_inter_chain_clashes"
    ]
    metadata = {
        "schema_version": 1,
        "source": "official-chai-torch",
        "fixture_id": "protein_ligand",
        "crop_size": 256,
        "active_tokens": active_tokens,
        "active_atoms": active_atoms,
        "fasta_sha256": _sha256(args.fasta),
        "confidence_component_sha256": _sha256(args.component),
        "torch_context_sha256": _sha256(args.torch_context),
        "torch_artifact_sha256": _sha256(args.torch_artifact),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({"output": str(args.output), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
