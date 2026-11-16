"""Padded positions cannot change ESMFold2's real outputs.

The structure model's `predict` -- inputs embedder with its atom encoder,
one-layer trunk with the MSA and language-model branches and a Parcae coda,
diffusion sampler, distogram and confidence heads -- runs twice on one
bucket-padded two-chain-plus-ligand job (the all-atom featurizer over a stub
chemistry store, so the released CCD is not needed): once as
`normalize_msa_features` and `pad_features` produced it, once with every padded
token, atom, MSA row and column, language-model embedding row and noise-tape
row overwritten with large finite garbage. The real slices of every output
must agree bit for bit.

ESMC itself is not run: its combined embedding is a random `[1, tokens, 256]`
array handed in as the managed path does, with garbage in the padded rows. The
MSA rows are padded on the host after the released per-loop row selection, so
every padded row also sits in the loop tapes the model reads. The diffusion
draws come in through the native tapes so that the padded atoms' rows can be
overwritten; the managed padded path instead draws them in-graph and zeroes
them. The second case pins the token axis to the real token count while atoms
are still bucketed. Masks and selectors -- the attention masks,
`atom_to_token`, `distogram_atom_idx`, `msa_attention_mask` and its loop tape
-- are left as padding wrote them.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._chain_profile import chain_capacity
from foldjax.models.esmfold2 import inference, output
from foldjax.models.esmfold2.data import all_atom, chemistry
from foldjax.models.esmfold2.models import atom as atom_model
from foldjax.models.esmfold2.models import diffusion
from foldjax.models.esmfold2.models import model as structure_model
from tests.models._padding_perturbation import (
    GARBAGE,
    assert_bitwise_identical,
    assert_some_difference,
    garbage_ids,
    real_slices,
)
from tests.models._tiny_weights import load_manifest, random_state_dict

STEPS, SAMPLES = 3, 2
MSA_ROWS = 4
LM_WIDTH = 256

JOB = {
    "name": "perturbation",
    "entities": [
        {"type": "protein", "id": "A", "sequence": "MKTAYIAK"},
        {"type": "protein", "id": "B", "sequence": "GSWEL"},
        {"type": "ligand", "id": "L", "smiles": "CC(=O)O"},
    ],
}

#: The released `config.json` with every repeated stack cut to one block and
#: the sampling knobs shrunk; `settings_from_config` reads the same fields.
CONFIG = {
    "d_pair": 256,
    "inputs": {
        "d_inputs": 451,
        "atom_encoder": {"n_blocks": 1, "n_heads": 4, "swa_window_size": 128},
    },
    "n_relative_residx_bins": 32,
    "n_relative_chain_bins": 2,
    "folding_trunk": {"n_layers": 1},
    "lm_encoder": {"enabled": True, "n_layers": 1, "lm_dropout": 0.25},
    "msa_encoder": {"enabled": True, "n_layers": 1},
    "parcae": {"coda_n_layers": 1},
    "confidence_head": {"folding_trunk": {"n_layers": 1}},
    "num_loops": 1,
    "num_diffusion_samples": SAMPLES,
    "structure_head": {
        "diffusion_module": {
            "sigma_data": 16.0,
            "c_atom": 128,
            "c_token": 768,
            "c_z": 256,
            "atom_num_blocks": 1,
            "atom_num_heads": 4,
            "token_num_blocks": 1,
            "token_num_heads": 16,
        },
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "step_scale": 1.5,
        "inference_s_max": 160.0,
        "inference_s_min": 0.0004,
        "inference_p": 7.0,
        "inference_num_steps": STEPS,
    },
}

REQUIRED = (
    "sample_atom_coords",
    "plddt",
    "plddt_per_atom",
    "pae",
    "pde",
    "ptm",
    "iptm",
    "complex_plddt",
    "pair_chains_iptm",
    "distogram_logits",
)


class _StubCCD:
    """Standard protein chemistry only; the job has no CCD-coded residue."""

    def __init__(self, _path: object) -> None:
        pass

    def conformer(self, component: str):
        return chemistry.PROTEIN_REF_POS.get(component)

    def atoms(self, component: str):
        return None

    def bonds(self, component: str):
        return None

    def leaving_atoms(self, component: str) -> set[str]:
        return set()

    def idealized_position(self, component: str, atom_name: str):
        return None


@pytest.fixture(scope="module")
def settings() -> structure_model.ModelSettings:
    return structure_model.settings_from_config(CONFIG)


@pytest.fixture(scope="module")
def params() -> dict[str, jnp.ndarray]:
    state = random_state_dict(load_manifest("esmfold2"), seed=0)
    return {name: jnp.asarray(value) for name, value in state.items()}


@pytest.fixture(scope="module")
def features() -> dict[str, np.ndarray]:
    original = all_atom.get_ccd_store
    all_atom.get_ccd_store = _StubCCD
    try:
        return all_atom.build_job_features(
            JOB, base_dir=".", ccd_path="unused.pkl", seed=0
        )
    finally:
        all_atom.get_ccd_store = original


def _forward(settings, padded: dict[str, Any], params, n_chains: int):
    """`inference._run` without the compile pool: the same static choices."""

    compact_bonds = inference._has_compact_token_bond_encoding(
        padded, params, pair_width=settings.d_pair, compute_dtype=settings.trunk_dtype
    )
    contiguous = inference._has_contiguous_atom_groups(padded)
    rows = atom_model._resolve_rows_per_block(None)

    @jax.jit
    def run(arrays, params, lm, tapes):
        with atom_model._atom_rows_per_block_scope(rows):
            return structure_model.predict(
                jax.random.key(7),
                arrays,
                params,
                settings=settings,
                lm_embedding=lm,
                n_chains=chain_capacity(n_chains),
                contiguous_atom_groups=contiguous,
                compact_token_bond_encoding=compact_bonds,
                **tapes,
            )

    def forward(batch, lm, tapes):
        bound = inference._model_bound_features(
            batch, compact_token_bond_encoding=compact_bonds
        )
        arrays = {
            name: jnp.asarray(value)
            for name, value in bound.items()
            if name not in all_atom.OUTPUT_METADATA_FEATURES
        }
        return run(arrays, params, lm, tapes)

    return forward


def _tapes(key, settings, n_atoms_real: int, n_atoms: int, *, padded_rows: float):
    """The four native sampler draws; rows of padded atoms are the caller's."""

    steps = len(diffusion.noise_schedule(settings.diffusion)) - 1
    keys = jax.random.split(key, 4)
    initial = np.full((SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    churn = np.full((steps, SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    initial[:, :n_atoms_real] = jax.random.normal(keys[0], (SAMPLES, n_atoms_real, 3))
    churn[:, :, :n_atoms_real] = jax.random.normal(
        keys[1], (steps, SAMPLES, n_atoms_real, 3)
    )
    return {
        "diffusion_initial_normal": jnp.asarray(initial),
        "diffusion_rotation_quaternions": jax.random.normal(
            keys[2], (steps, SAMPLES, 4)
        ),
        "diffusion_translations": jax.random.normal(keys[3], (steps, SAMPLES, 1, 3)),
        "diffusion_churn_normals": jnp.asarray(churn),
    }


def _lm_embedding(key, n_tokens_real: int, n_tokens: int, *, padded_rows: float):
    embedding = np.full((1, n_tokens, LM_WIDTH), padded_rows, dtype=np.float32)
    embedding[:, :n_tokens_real] = jax.random.normal(key, (1, n_tokens_real, LM_WIDTH))
    return jnp.asarray(embedding)


def _perturb_padding(
    padded: dict[str, Any], t: int, a: int, m: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Garbage on every padded position of every value feature."""

    out = {k: np.array(v, copy=True) for k, v in padded.items()}
    tt = out["token_attention_mask"].shape[-1]

    def fill(name, index, value):
        if name in out:
            out[name][index] = value

    # Tokens: a fourth chain of ligand-typed residues with random residue ids
    # far down the sequence, bonded to everything, with a garbage profile.
    residues = garbage_ids(rng, (tt - t,), structure_model.NUM_RES_TYPES)
    fill("res_type", np.s_[0, t:], residues)
    fill("input_ids", np.s_[0, t:], 2)
    for name in ("asym_id", "sym_id", "entity_id"):
        fill(name, np.s_[0, t:], 7)
    for name in ("residue_index", "token_index"):
        fill(name, np.s_[0, t:], 9999)
    if tt > t:
        out["mol_type"][0, t:] = int(np.max(out["mol_type"][0, :t]))
        out["token_bonds"][0, t:, :] = 1.0
        out["token_bonds"][0, :, t:] = 1.0
    for name in ("deletion_mean", "msa_profile"):
        fill(name, np.s_[0, t:], GARBAGE)
    for name in ("token_chain_id_chars", "token_residue_name_chars"):
        fill(name, np.s_[0, t:], 1)
    # Atoms: reference conformer far away, charged, hydrogen with every name
    # character set, one shared reference space.
    for name in ("ref_pos", "ref_charge"):
        fill(name, np.s_[0, a:], GARBAGE if name == "ref_pos" else 100)
    for name in ("ref_element", "ref_atom_name_chars"):
        fill(name, np.s_[0, a:], 1)
    fill("ref_space_uid", np.s_[0, a:], 9999)
    # MSA: padded rows and padded columns, in the first-loop rows and in the
    # per-loop tapes alike.
    for name, rows, cols in (
        ("msa", np.s_[0, m:, :], np.s_[0, :, t:]),
        ("msa_loop_tape", np.s_[:, 0, m:, :], np.s_[:, 0, :, t:]),
    ):
        for index in (rows, cols):
            out[name][index] = garbage_ids(
                rng, out[name][index].shape, structure_model.NUM_RES_TYPES
            )
    for name, value in (("has_deletion", True), ("deletion_value", GARBAGE)):
        out[name][0, m:, :] = value
        out[name][0, :, t:] = value
        out[f"{name}_loop_tape"][:, 0, m:, :] = value
        out[f"{name}_loop_tape"][:, 0, :, t:] = value
    return out


def _perturb_one_real_residue(padded: dict[str, Any]) -> dict[str, Any]:
    out = dict(padded)
    res_type = np.array(out["res_type"], copy=True)
    res_type[0, 1] = (res_type[0, 1] + 3) % 20
    out["res_type"] = res_type
    return out


def _crop(padded: dict[str, Any], n_chains: int):
    def crop(name: str, value: np.ndarray) -> np.ndarray | None:
        if name == "pair_chains_iptm":
            return value[:, :n_chains, :n_chains]
        return np.asarray(output.crop_prediction({name: value}, padded)[name])

    return crop


@pytest.mark.parametrize(
    "pin_tokens", [False, True], ids=["bucket", "tokens_on_bucket"]
)
def test_padded_positions_cannot_change_real_outputs(
    settings, params, features, pin_tokens
):
    n_token = int(features["token_attention_mask"].sum())
    n_atom = int(features["atom_attention_mask"].sum())
    n_msa = int(features["msa_attention_mask"].shape[1])
    assert features["token_attention_mask"].shape[-1] == n_token
    assert n_atom < features["atom_attention_mask"].shape[-1] <= 160
    assert n_msa < MSA_ROWS
    target_tokens = n_token if pin_tokens else 24
    target_atoms = 160
    loops = max(1, settings.num_recycles + 1)

    key = jax.random.key(3)
    normalized = inference.normalize_msa_features(
        key, features, n_msa=MSA_ROWS, max_msa_depth=settings.max_msa_depth,
        total_steps=loops,
    )
    padded = inference.pad_features(
        normalized, n_token=target_tokens, n_atom=target_atoms, n_msa=MSA_ROWS
    )
    assert padded["msa_loop_tape"].shape == (loops, 1, MSA_ROWS, target_tokens)
    n_chains = int(padded["asym_id"].max()) + 1
    assert n_chains == 3
    # Padded atoms are owned by real token 0, as the helper leaves them.
    np.testing.assert_array_equal(padded["atom_to_token"][0, n_atom:], 0)

    clean = _tapes(key, settings, n_atom, target_atoms, padded_rows=0.0)
    dirty = _tapes(key, settings, n_atom, target_atoms, padded_rows=GARBAGE)
    clean_lm = _lm_embedding(key, n_token, target_tokens, padded_rows=0.0)
    dirty_lm = _lm_embedding(key, n_token, target_tokens, padded_rows=GARBAGE)
    np.testing.assert_array_equal(
        clean["diffusion_churn_normals"][..., :n_atom, :],
        dirty["diffusion_churn_normals"][..., :n_atom, :],
    )

    run = _forward(settings, padded, params, n_chains)
    crop = _crop(padded, n_chains)
    reference = real_slices(run(padded, clean_lm, clean), crop)
    garbage = _perturb_padding(
        padded, n_token, n_atom, n_msa, np.random.default_rng(11)
    )
    perturbed = real_slices(run(garbage, dirty_lm, dirty), crop)
    assert_bitwise_identical(reference, perturbed, require=REQUIRED)

    control = real_slices(run(_perturb_one_real_residue(padded), clean_lm, clean), crop)
    assert_some_difference(
        reference, control, ("sample_atom_coords", "complex_plddt", "pae")
    )
