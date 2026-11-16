"""Padded positions cannot change OpenDDE's real outputs.

The default inference forward -- residue trunk with MSA cycles and template
slots, structural-token expansion and refinement, diffusion sampler,
confidence head and ranking -- runs twice on one padded two-chain-plus-ligand
input: once as `pad_opendde_features` produced it, once with every padded
residue token, structural token, atom, MSA row and column, template row and
column and noise-tape row overwritten with large finite garbage. The real
slices of every output must agree bit for bit.

The second case pins the residue and structural token axes to their real
counts while atoms are still padded: `pad_protenix_features` then hands every
padded atom to real residue 0, and OpenDDE's own padding always hands them to
structural token 0. Masks and selectors are left as padding wrote them.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.bridge import torch_mapping as tm
from foldjax.models.opendde.data.featurize_json import featurize_opendde_json
from foldjax.models.opendde.data.padding import (
    pad_opendde_features,
    select_opendde_model_features,
)
from foldjax.models.opendde.models.model import opendde_infer_static
from foldjax.models.opendde.models.msa_sampling import (
    sample_opendde_msa_cycle_features,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    inference_noise_schedule,
)
from foldjax.schema import PaddingConfig
from tests.models._padding_perturbation import (
    GARBAGE,
    assert_bitwise_identical,
    assert_some_difference,
    garbage_ids,
    real_slices,
)
from tests.models._tiny_weights import load_manifest, random_state_dict

N_QUERIES, N_KEYS = 32, 128
STEPS, SAMPLES, RECYCLES = 3, 2, 1

JOB = {
    "name": "perturbation",
    "sequences": [
        {"proteinChain": {"sequence": "MKTAYIAK", "count": 1}},
        {"proteinChain": {"sequence": "GSWEL", "count": 1}},
        {"ligand": {"ligand": "CC(=O)O", "count": 1}},
    ],
}

REQUIRED = (
    "coordinate",
    "atom_plddt",
    "plddt",
    "pae",
    "pde",
    "token_pair_pae",
    "summary_plddt",
    "summary_ptm",
    "summary_iptm",
    "summary_ranking_score",
    "ranking_score",
    "chain_pair_iptm",
    "distogram_logits",
)


@pytest.fixture(scope="module")
def params():
    # The released mapper insists on the released four-block structural
    # refiner; the one-block manifest goes through the same component mappers
    # with the depth inferred from the keys.
    state = random_state_dict(load_manifest("opendde"), seed=0)
    return tm.OpenDDEInferenceParams(
        input_embedder=tm.map_input_feature_embedder_state_dict(
            state, "input_embedder"
        ),
        pairformer_output=tm.map_pairformer_output_state_dict(state),
        structural_expander=tm.map_structural_token_expander_state_dict(state),
        structural_refiner=tm.map_structural_refiner_state_dict(state),
        diffusion=tm.map_diffusion_module_state_dict(state, "diffusion_module"),
        distogram=tm.map_distogram_state_dict(state),
        confidence=tm.map_confidence_head_state_dict(state, "confidence_head"),
    )


@pytest.fixture(scope="module")
def features() -> dict[str, Any]:
    return featurize_opendde_json(JOB, seed=0, n_queries=N_QUERIES, n_keys=N_KEYS)


def _forward(params, features, cycles, *, n_chain, tapes):
    init_noise, step_noises, rotations, translations = tapes
    return opendde_infer_static(
        features,
        params,
        inference_noise_schedule(num_steps=STEPS),
        key=jax.random.PRNGKey(7),
        num_samples=SAMPLES,
        num_recycles=RECYCLES,
        cycle_msa_features=cycles,
        init_noise=init_noise,
        step_noises=step_noises,
        rotations=rotations,
        translations=translations,
        run_confidence_scores=True,
        return_confidence_logits=True,
        return_confidence_details=True,
        n_chain=n_chain,
    )


def _tapes(key, n_atoms_real: int, n_atoms: int, *, padded_rows: float):
    """Noise drawn for the real atoms; the padded rows are the caller's."""

    init_key, step_key, rot_key, trans_key = jax.random.split(key, 4)
    init = np.full((SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    steps = np.full((STEPS, SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    init[:, :n_atoms_real] = jax.random.normal(init_key, (SAMPLES, n_atoms_real, 3))
    steps[:, :, :n_atoms_real] = jax.random.normal(
        step_key, (STEPS, SAMPLES, n_atoms_real, 3)
    )
    rotations = jnp.broadcast_to(jnp.eye(3), (STEPS, SAMPLES, 3, 3))
    rotations = rotations + 0.01 * jax.random.normal(rot_key, rotations.shape)
    translations = jax.random.normal(trans_key, (STEPS, SAMPLES, 3))
    return jnp.asarray(init), jnp.asarray(steps), rotations, translations


def _perturb_padding(padded: dict[str, Any], cycles, plan, rng: np.random.Generator):
    """Garbage on every padded position of every value feature."""

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in padded.items()}
    t, a = plan.actual["tokens"], plan.actual["atoms"]
    s = plan.actual["structural_tokens"]
    tt, ta = plan.target["tokens"], plan.target["atoms"]
    ts = plan.target["structural_tokens"]

    def fill(name, index, value):
        if name in out:
            out[name][index] = value

    # Residue tokens.
    for name in ("restype", "profile", "deletion_mean"):
        fill(name, np.s_[t:], GARBAGE)
    for name in ("asym_id", "entity_id", "sym_id"):
        fill(name, np.s_[t:], 7)
    fill("residue_index", np.s_[t:], 9999)
    fill("token_index", np.s_[t:], 9999)
    if tt > t:
        out["token_bonds"][t:, :] = 1.0
        out["token_bonds"][:, t:] = 1.0
    # Structural tokens: a real role and polymer type on the padded tokens, a
    # far-away token index. `parent_residue_idx` and the frame/rep selectors
    # stay as written: they are indices the model gathers with.
    if ts > s:
        out["subtoken_role_id"][s:] = int(np.max(out["subtoken_role_id"][:s]))
        out["structural_polymer_type"][s:] = int(
            np.max(out["structural_polymer_type"][:s])
        )
        out["structural_token_index"][s:] = 9999
    # Atoms: reference conformer far away, charged, unknown element, ligand.
    for name in ("ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars"):
        fill(name, np.s_[a:], GARBAGE)
    fill("ref_mask", np.s_[a:], 1.0)
    fill("is_ligand", np.s_[a:], 1)
    fill("is_protein", np.s_[a:], 1)
    masked = ~np.asarray(out["pad_info"]["mask_trunked"], dtype=bool)
    out["d_lm"] = np.where(masked[..., None], GARBAGE, out["d_lm"])
    out["v_lm"] = np.where(masked[..., None], GARBAGE, out["v_lm"])
    # Template rows and columns of the padded residue tokens (the four slots
    # are the native depth, not padding; see the Protenix test).
    if tt > t:
        for name in ("template_distogram", "template_unit_vector"):
            out[name][:, t:, :] = GARBAGE
            out[name][:, :, t:] = GARBAGE
        fill("template_atom_positions", np.s_[:, t:], GARBAGE)
        fill("template_aatype", np.s_[:, t:], garbage_ids(rng, (4, tt - t), 32))
    # MSA cycles: every masked row or column gets random residues, deletions.
    dirty_cycles = []
    for cycle in cycles:
        cycle = {k: v.copy() for k, v in cycle.items()}
        gap = ~np.asarray(cycle["msa_mask"], dtype=bool)
        cycle["msa"] = np.where(gap, garbage_ids(rng, gap.shape, 32), cycle["msa"])
        cycle["has_deletion"] = np.where(gap, 1.0, cycle["has_deletion"])
        cycle["deletion_value"] = np.where(gap, GARBAGE, cycle["deletion_value"])
        assert gap.any()
        dirty_cycles.append(cycle)
    assert ta > a
    return out, tuple(dirty_cycles)


def _perturb_one_real_residue(padded: dict[str, Any]) -> dict[str, Any]:
    out = dict(padded)
    restype = np.asarray(out["restype"]).copy()
    restype[1] = np.roll(restype[1], 3)
    out["restype"] = restype
    return out


def _crop(plan):
    t, a = plan.actual["tokens"], plan.actual["atoms"]

    def crop(name: str, value: np.ndarray) -> np.ndarray | None:
        if name in {"coordinate", "atom_plddt", "plddt", "resolved"}:
            return value[:, :a]
        if name in {"pae", "pde", "token_pair_pae", "token_pair_pde"}:
            return value[:, :t, :t]
        if name in {"contact_probs", "distogram_logits"}:
            return value[:t, :t]
        return value

    return crop


@pytest.mark.parametrize(
    "pin_tokens", [False, True], ids=["bucket", "tokens_on_bucket"]
)
def test_padded_positions_cannot_change_real_outputs(params, features, pin_tokens):
    n_token = int(features["restype"].shape[0])
    n_atom = int(features["ref_pos"].shape[0])
    n_structural = int(features["parent_residue_idx"].shape[0])
    config = PaddingConfig(
        tokens=n_token if pin_tokens else 24,
        atoms=160,
        msa=8,
        structural_tokens=n_structural if pin_tokens else 64,
    )
    sampled = sample_opendde_msa_cycle_features(features, num_recycles=RECYCLES, seed=0)
    padded, cycles, plan = pad_opendde_features(
        features, sampled, config, n_queries=N_QUERIES, n_keys=N_KEYS
    )
    assert plan.actual["tokens"] == n_token and plan.actual["atoms"] == n_atom
    assert plan.actual["structural_tokens"] == n_structural
    # OpenDDE always hands padded atoms to structural token 0.
    np.testing.assert_array_equal(padded["atom_to_structural_token_idx"][n_atom:], 0)
    if pin_tokens:
        assert plan.target["tokens"] == n_token
        assert plan.target["structural_tokens"] == n_structural
        np.testing.assert_array_equal(padded["atom_to_token_idx"][n_atom:], 0)
    else:
        np.testing.assert_array_equal(padded["atom_to_token_idx"][n_atom:], n_token)
    model_features = select_opendde_model_features(padded)
    n_chain = int(np.max(features["asym_id"])) + 1

    key = jax.random.PRNGKey(3)
    clean = _tapes(key, n_atom, plan.target["atoms"], padded_rows=0.0)
    dirty = _tapes(key, n_atom, plan.target["atoms"], padded_rows=GARBAGE)
    np.testing.assert_array_equal(clean[0][:, :n_atom], dirty[0][:, :n_atom])

    run = functools.partial(_forward, params, n_chain=n_chain)
    crop = _crop(plan)
    reference = real_slices(run(model_features, cycles, tapes=clean), crop)
    garbage, dirty_cycles = _perturb_padding(
        model_features, cycles, plan, np.random.default_rng(11)
    )
    perturbed = real_slices(run(garbage, dirty_cycles, tapes=dirty), crop)
    assert_bitwise_identical(reference, perturbed, require=REQUIRED)

    control = real_slices(
        run(_perturb_one_real_residue(model_features), cycles, tapes=clean), crop
    )
    assert_some_difference(reference, control, ("coordinate", "summary_plddt", "pae"))
