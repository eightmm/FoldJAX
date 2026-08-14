import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.confidence_mapping import (
    map_confidence_module_state_dict,
)
from foldjax.models.boltz2.models.heads.confidence import confidence_module_forward
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "confidence_module"

TOKEN_S = 384
TOKEN_Z = 128
N_TOKENS = 6
N_ATOMS = 24
NUM_PF_LAYERS = 8

# Real boltz2_conf.ckpt confidence_model_args.
CONFIDENCE_MODEL_ARGS = {
    "use_gaussian": False,
    "num_dist_bins": 64,
    "max_dist": 22,
    "use_miniformer": False,
    "no_trunk_feats": False,
    "add_s_to_z_prod": True,
    "add_s_input_to_s": True,
    "use_s_diffusion": False,
    "add_z_input_to_z": True,
    "pairformer_args": {"num_blocks": NUM_PF_LAYERS, "num_heads": 16, "dropout": 0.0},
    "confidence_args": {
        "num_plddt_bins": 50,
        "num_pde_bins": 64,
        "num_pae_bins": 64,
        "relative_confidence": "none",
        "use_separate_heads": True,
    },
}


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def _load_torch_module(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.confidencev2 import ConfidenceModule

    module = ConfidenceModule(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        token_level_confidence=True,
        bond_type_feature=True,
        fix_sym_check=True,
        cyclic_pos_enc=True,
        conditioning_cutoff_min=4.0,
        conditioning_cutoff_max=20.0,
        **CONFIDENCE_MODEL_ARGS,
    ).eval()

    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local = key.removeprefix(f"{PREFIX}.")
        if local.startswith("pairformer_stack.layers."):
            if int(local.split(".")[2]) >= NUM_PF_LAYERS:
                continue
        module_state[local] = value
    missing, unexpected = module.load_state_dict(module_state, strict=False)
    # buffers (boundaries) may differ; ignore unexpected/missing buffers only.
    assert not [m for m in missing if not m.endswith("boundaries")], missing
    return module


def _inputs() -> dict[str, object]:
    rng = np.random.default_rng(0)
    # token -> chain assignment: two chains.
    asym = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    mol_type = np.array([0, 0, 0, 0, 3, 3], dtype=np.int64)  # PROTEIN + NONPOLYMER

    atom_to_token = np.zeros((1, N_ATOMS, N_TOKENS), dtype=np.float32)
    tok_of_atom = np.repeat(np.arange(N_TOKENS), N_ATOMS // N_TOKENS)
    atom_to_token[0, np.arange(N_ATOMS), tok_of_atom] = 1.0
    # rep atom = first atom of each token.
    token_to_rep_atom = np.zeros((1, N_TOKENS, N_ATOMS), dtype=np.float32)
    for t in range(N_TOKENS):
        token_to_rep_atom[0, t, np.where(tok_of_atom == t)[0][0]] = 1.0

    contact_conditioning = np.zeros((1, N_TOKENS, N_TOKENS, 5), dtype=np.float32)
    contact_conditioning[..., 0] = 1.0  # all "unspecified"
    contact_threshold = np.full((1, N_TOKENS, N_TOKENS), 10.0, dtype=np.float32)

    feats = {
        "token_to_rep_atom": torch.tensor(token_to_rep_atom),
        "token_pad_mask": torch.ones(1, N_TOKENS, dtype=torch.float32),
        "atom_pad_mask": torch.ones(1, N_ATOMS, dtype=torch.float32),
        "atom_to_token": torch.tensor(atom_to_token),
        "mol_type": torch.tensor(mol_type)[None],
        "asym_id": torch.tensor(asym)[None],
        "entity_id": torch.tensor(asym)[None],
        "sym_id": torch.zeros(1, N_TOKENS, dtype=torch.long),
        "residue_index": torch.arange(N_TOKENS)[None],
        "token_index": torch.arange(N_TOKENS)[None],
        "cyclic_period": torch.zeros(1, N_TOKENS, dtype=torch.long),
        "token_bonds": torch.zeros(1, N_TOKENS, N_TOKENS, 1, dtype=torch.float32),
        "type_bonds": torch.zeros(1, N_TOKENS, N_TOKENS, dtype=torch.long),
        "contact_conditioning": torch.tensor(contact_conditioning),
        "contact_threshold": torch.tensor(contact_threshold),
        "frames_idx": torch.tensor(
            np.stack([tok_of_atom_frame(tok_of_atom) for _ in range(1)], axis=0)
        ),
    }

    x_pred = torch.tensor(
        rng.standard_normal((1, N_ATOMS, 3)) * 5.0, dtype=torch.float32
    )
    s_inputs = torch.tensor(
        rng.standard_normal((1, N_TOKENS, TOKEN_S)) * 0.2, dtype=torch.float32
    )
    s = torch.tensor(
        rng.standard_normal((1, N_TOKENS, TOKEN_S)) * 0.2, dtype=torch.float32
    )
    z = torch.tensor(
        rng.standard_normal((1, N_TOKENS, N_TOKENS, TOKEN_Z)) * 0.2, dtype=torch.float32
    )
    pred_distogram_logits = torch.tensor(
        rng.standard_normal((1, N_TOKENS, N_TOKENS, 64)), dtype=torch.float32
    )
    return {
        "s_inputs": s_inputs,
        "s": s,
        "z": z,
        "x_pred": x_pred,
        "feats": feats,
        "pred_distogram_logits": pred_distogram_logits,
    }


def tok_of_atom_frame(tok_of_atom):
    # frames_idx: per token, 3 atom indices. Use first three atoms (clamped).
    frames = np.zeros((N_TOKENS, 3), dtype=np.int64)
    for t in range(N_TOKENS):
        atoms = np.where(tok_of_atom == t)[0]
        for j in range(3):
            frames[t, j] = int(atoms[min(j, len(atoms) - 1)])
    return frames


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v.numpy()) for k, v in feats.items()}


def test_confidence_module_matches_boltz_torch(checkpoint_state) -> None:
    torch_module = _load_torch_module(checkpoint_state)
    params = map_confidence_module_state_dict(
        checkpoint_state, PREFIX, num_pairformer_layers=NUM_PF_LAYERS
    )
    inp = _inputs()

    with torch.no_grad():
        expected = torch_module(
            s_inputs=inp["s_inputs"],
            s=inp["s"],
            z=inp["z"],
            x_pred=inp["x_pred"],
            feats=inp["feats"],
            pred_distogram_logits=inp["pred_distogram_logits"],
            multiplicity=1,
        )

    actual = confidence_module_forward(
        params,
        s_inputs=jnp.asarray(inp["s_inputs"].numpy()),
        s=jnp.asarray(inp["s"].numpy()),
        z=jnp.asarray(inp["z"].numpy()),
        x_pred=jnp.asarray(inp["x_pred"].numpy()),
        feats=_jax_feats(inp["feats"]),
        pred_distogram_logits=jnp.asarray(inp["pred_distogram_logits"].numpy()),
        multiplicity=1,
    )

    scalar_or_tensor = [
        "pde_logits",
        "plddt_logits",
        "resolved_logits",
        "pae_logits",
        "pde",
        "plddt",
        "pae",
        "complex_plddt",
        "complex_iplddt",
        "complex_pde",
        "complex_ipde",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
    ]
    for key in scalar_or_tensor:
        np.testing.assert_allclose(
            np.asarray(actual[key]),
            expected[key].detach().numpy(),
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"mismatch in {key}",
        )

    # pair_chains_iptm: nested dict keyed by chain id.
    exp_pci = expected["pair_chains_iptm"]
    act_pci = actual["pair_chains_iptm"]
    for c1 in exp_pci:
        for c2 in exp_pci[c1]:
            np.testing.assert_allclose(
                np.asarray(act_pci[c1][c2]),
                exp_pci[c1][c2].detach().numpy(),
                rtol=2e-3,
                atol=2e-3,
                err_msg=f"mismatch in pair_chains_iptm[{c1}][{c2}]",
            )
