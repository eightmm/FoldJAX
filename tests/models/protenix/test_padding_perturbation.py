"""Padded positions cannot change Protenix's real outputs.

The whole default inference forward -- input embedder, one-block trunk with
MSA and template modules, diffusion sampler, confidence head and ranking --
runs twice on one padded two-chain-plus-ligand input: once as the padding
helper produced it, once with every padded token, atom, MSA row and column,
template slot and noise-tape row overwritten with large finite garbage. The
real slices of coordinates, pLDDT/PAE/PDE, their summaries and the ranking
scores must agree bit for bit.

The second case pins the token axis to the real token count while atoms are
still padded, which is the corner where `pad_protenix_features` hands every
padded atom to real token 0 (`dummy_token = 0`). Masks and selectors -- the
padding masks, `has_frame`, the template masks, `distogram_rep_atom_mask`,
`pad_info["mask_trunked"]` -- are left as padding wrote them: they are what
the model is allowed to rely on, and the question is whether anything else
reaches a real output.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_protenix_inference_state_dict,
)
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.padding import pad_protenix_features
from foldjax.models.protenix.models.predict import protenix_predict_static
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
STEPS, SAMPLES = 3, 2

JOB = {
    "name": "perturbation",
    "sequences": [
        {"proteinChain": {"sequence": "MKTAYIAK", "count": 1}},
        {"proteinChain": {"sequence": "GSWEL", "count": 1}},
        {"ligand": {"ligand": "CC(=O)O", "count": 1}},
    ],
}

#: Outputs whose presence the comparison insists on.
REQUIRED = (
    "coordinate",
    "atom_plddt",
    "plddt",
    "pae",
    "pde",
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
    return map_protenix_inference_state_dict(
        random_state_dict(load_manifest("protenix"), seed=0)
    )


@pytest.fixture(scope="module")
def features() -> dict[str, Any]:
    return featurize_protein_json(JOB, seed=0, n_queries=N_QUERIES, n_keys=N_KEYS)


def _forward(params, features, *, init_noise, step_noises):
    return protenix_predict_static(
        params,
        features,
        jax.random.PRNGKey(7),
        num_samples=SAMPLES,
        num_sampling_steps=STEPS,
        recycling_steps=1,
        init_noise=init_noise,
        step_noises=step_noises,
        # The released denoiser attention is tokamax's fused kernel, which has
        # no CPU path; the subject here is masking, not the kernel.
        diffusion_attention_backend="xla_jit",
        matmul_precision="highest",
        padded_generated_schema=True,
    )


def _tapes(key, n_atoms_real: int, n_atoms: int, *, padded_rows: float):
    """Noise drawn for the real atoms; the padded rows are the caller's."""

    init_key, step_key = jax.random.split(key)
    init = np.full((SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    steps = np.full((STEPS, SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    init[:, :n_atoms_real] = jax.random.normal(init_key, (SAMPLES, n_atoms_real, 3))
    steps[:, :, :n_atoms_real] = jax.random.normal(
        step_key, (STEPS, SAMPLES, n_atoms_real, 3)
    )
    return jnp.asarray(init), jnp.asarray(steps)


def _perturb_padding(padded: dict[str, Any], plan, rng: np.random.Generator):
    """Garbage on every padded position of every value feature."""

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in padded.items()}
    t, a, m = plan.actual["tokens"], plan.actual["atoms"], plan.actual["msa"]
    tt, ta = plan.target["tokens"], plan.target["atoms"]

    def fill(name, index, value):
        if name in out:
            out[name][index] = value

    # Token axis: chain ids that look like a fourth chain, a huge residue
    # index, a ligand flag, garbage one-hots and profiles.
    for name in ("restype", "profile", "deletion_mean"):
        fill(name, np.s_[t:], GARBAGE)
    for name in ("asym_id", "entity_id", "sym_id", "mol_id"):
        fill(name, np.s_[t:], 7)
    fill("residue_index", np.s_[t:], 9999)
    fill("token_index", np.s_[t:], 9999)
    fill("token_is_ligand", np.s_[t:], True)
    for name in ("token_polymer_type", "token_entity_id", "token_copy_id"):
        if name in out and tt > t:
            out[name][t:] = int(np.max(out[name][:t]))
    # Pair axis: bonds to and from every padded token. The compact
    # relative-position bins are left alone: `normalize_relative_position_storage`
    # requires the padding sentinel on every padded pair and refuses anything
    # else, so garbage there never reaches the model.
    if tt > t:
        out["token_bonds"][t:, :] = 1.0
        out["token_bonds"][:, t:] = 1.0
    # Atom axis: reference conformer far away, charged, unknown element.
    for name in ("ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars"):
        fill(name, np.s_[a:], GARBAGE)
    fill("ref_mask", np.s_[a:], 1.0)
    fill("is_ligand", np.s_[a:], 1)
    for name in ("atom_entity_id", "atom_copy_id", "atom_residue_index"):
        fill(name, np.s_[a:], 7)
    # Local atom geometry: every masked query/key pair, including real
    # queries whose window reaches into the padded atoms.
    masked = ~np.asarray(out["pad_info"]["mask_trunked"], dtype=bool)
    out["d_lm"] = np.where(masked[..., None], GARBAGE, out["d_lm"])
    out["v_lm"] = np.where(masked[..., None], GARBAGE, out["v_lm"])
    # MSA: padded rows and padded columns with random residues and deletions.
    msa = out["msa"]
    msa[m:, :] = garbage_ids(rng, msa[m:, :].shape, 32)
    msa[:, t:] = garbage_ids(rng, msa[:, t:].shape, 32)
    for name, value in (("has_deletion", 1.0), ("deletion_value", GARBAGE)):
        out[name][m:, :] = value
        out[name][:, t:] = value
    # Templates: the padded token rows and columns of every slot. The four
    # slots themselves are the native fixed depth, not padding: an empty slot
    # is a template whose features are zero, and the embedder averages it in
    # like upstream does, with no per-slot mask -- so garbage *inside* an empty
    # slot's real rows would reach z by design, on the unpadded path too.
    if tt > t:
        for name in ("template_distogram", "template_unit_vector"):
            out[name][:, t:, :] = GARBAGE
            out[name][:, :, t:] = GARBAGE
        fill("template_atom_positions", np.s_[:, t:], GARBAGE)
        fill("template_aatype", np.s_[:, t:], garbage_ids(rng, (4, tt - t), 32))
    assert ta > a
    return out


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
        if name in {"contact_probs", "distogram_logits", "z_trunk"}:
            return value[:t, :t]
        if name in {"s_inputs", "s_trunk"}:
            return value[:t]
        return value

    return crop


@pytest.mark.parametrize(
    "pin_tokens", [False, True], ids=["bucket", "tokens_on_bucket"]
)
def test_padded_positions_cannot_change_real_outputs(params, features, pin_tokens):
    n_token = int(features["restype"].shape[0])
    n_atom = int(features["ref_pos"].shape[0])
    config = PaddingConfig(
        tokens=n_token if pin_tokens else 24, atoms=160, msa=8, templates=4
    )
    padded, plan = pad_protenix_features(
        features, config, n_queries=N_QUERIES, n_keys=N_KEYS
    )
    assert plan.actual["tokens"] == n_token and plan.actual["atoms"] == n_atom
    if pin_tokens:
        # The F1 corner: padded atoms are owned by real token 0.
        assert plan.target["tokens"] == n_token
        np.testing.assert_array_equal(padded["atom_to_token_idx"][n_atom:], 0)
    else:
        np.testing.assert_array_equal(padded["atom_to_token_idx"][n_atom:], n_token)

    key = jax.random.PRNGKey(3)
    clean = _tapes(key, n_atom, plan.target["atoms"], padded_rows=0.0)
    dirty = _tapes(key, n_atom, plan.target["atoms"], padded_rows=GARBAGE)
    np.testing.assert_array_equal(clean[0][:, :n_atom], dirty[0][:, :n_atom])

    run = functools.partial(_forward, params)
    crop = _crop(plan)
    reference = real_slices(
        run(padded, init_noise=clean[0], step_noises=clean[1]), crop
    )
    garbage = _perturb_padding(padded, plan, np.random.default_rng(11))
    perturbed = real_slices(
        run(garbage, init_noise=dirty[0], step_noises=dirty[1]), crop
    )
    assert_bitwise_identical(reference, perturbed, require=REQUIRED)

    control = real_slices(
        run(
            _perturb_one_real_residue(padded),
            init_noise=clean[0],
            step_noises=clean[1],
        ),
        crop,
    )
    assert_some_difference(reference, control, ("coordinate", "summary_plddt", "pae"))
