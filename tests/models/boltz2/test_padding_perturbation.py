"""Padded positions cannot change Boltz-2's real outputs.

`boltz2_predict` -- trunk with MSA module, diffusion sampler, distogram and
confidence heads -- runs twice on one bucket-padded two-chain-plus-ligand
input (`fixtures/two_chains_ligand.npz`, featurized once from two short
protein chains and an acetate SMILES): once as `pad_feats` produced it, once
with every padded token, atom, MSA row and column, template row and noise-tape
row overwritten with large finite garbage. The real slices of every output
must agree bit for bit.

Boltz's own featurizer already pads atoms to a multiple of 32 (108 real atoms
in 128 storage slots). Those window-padding atoms are upstream's, not the
padding helper's: upstream draws their noise and centres every sample on the
unmasked mean over its storage, so their noise and reference features reach
the real coordinates by upstream's own arithmetic, and the port keeps that
(`_native_storage_mean` in `trunk_blocks/trunk.py` stops the mean at the
native storage for exactly this reason). The atom positions this test
overwrites are therefore the bucket suffix the helper appended, from the
storage size up; the noise tape draws normal noise for the storage atoms, as
the runner's prefix-stable tape does, and the suffix rows are the garbage.

The second case pins the token axis to the real token count while atoms are
still bucketed. Masks and selectors -- the pad masks, `msa_mask`,
`atom_to_token`, `token_to_rep_atom`, `frames_idx`, the template masks -- are
left as padding wrote them.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.bridge.confidence_mapping import (
    map_confidence_module_state_dict,
)
from foldjax.models.boltz2.bridge.torch_mapping import (
    map_boltz2_graph_state_dict,
    map_distogram_state_dict,
)
from foldjax.models.boltz2.data.bucket import (
    crop_prediction_outputs,
    pad_feats,
    resolve_padding_plan,
)
from foldjax.models.boltz2.models.heads.confidence import pair_chains_capacity
from foldjax.models.boltz2.models.predict import boltz2_predict
from foldjax.schema import PaddingConfig
from tests.models._padding_perturbation import (
    GARBAGE,
    assert_bitwise_identical,
    assert_some_difference,
    garbage_ids,
    real_slices,
)
from tests.models._tiny_weights import load_manifest, random_state_dict

FEATURES = Path(__file__).parent / "fixtures/two_chains_ligand.npz"
STEPS, SAMPLES = 3, 2

REQUIRED = (
    "sample_atom_coords",
    "plddt",
    "pae",
    "pde",
    "ptm",
    "iptm",
    "complex_plddt",
    "confidence_score",
    "pair_chains_iptm",
    "pdistogram",
)


@pytest.fixture(scope="module")
def params():
    state = random_state_dict(load_manifest("boltz2"), seed=0)
    # The trunk's atom encoder projects one pair bias for all of its layers at
    # once (`layers x heads` rows); the manifest keeps one layer, so keep the
    # first layer's four heads of that projection.
    name = "input_embedder.atom_enc_proj_z.1.weight"
    state[name] = state[name][:4]
    return {
        **map_boltz2_graph_state_dict(state, token_transformer_heads=16),
        "distogram": map_distogram_state_dict(state)["distogram"],
        "confidence": map_confidence_module_state_dict(state, "confidence_module"),
    }


@pytest.fixture(scope="module")
def features() -> dict[str, np.ndarray]:
    loaded = np.load(FEATURES)
    return {name: loaded[name] for name in loaded.files}


def _forward(params, feats, *, init_noise, step_noises):
    with jax.default_matmul_precision("highest"):
        return boltz2_predict(
            params,
            feats,
            jax.random.PRNGKey(7),
            recycling_steps=1,
            num_sampling_steps=STEPS,
            multiplicity=SAMPLES,
            init_noise=init_noise,
            step_noises=step_noises,
            # TF32 is a GPU precision; the subject here is masking.
            matmul_precision="highest",
            attention_backend="xla",
            confidence_sequentially=True,
            return_pair_chains_iptm=False,
            pair_chains_capacity=pair_chains_capacity(feats["asym_id"]),
            recompute_nonpolymer_frames=True,
        )


def _tapes(key, n_storage: int, n_atoms: int, *, padded_rows: float):
    """Noise drawn for the storage atoms; the bucket rows are the caller's."""

    init_key, step_key = jax.random.split(key)
    init = np.full((SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    steps = np.full((STEPS, SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    init[:, :n_storage] = jax.random.normal(init_key, (SAMPLES, n_storage, 3))
    steps[:, :, :n_storage] = jax.random.normal(
        step_key, (STEPS, SAMPLES, n_storage, 3)
    )
    return jnp.asarray(init), jnp.asarray(steps)


def _perturb_padding(padded: dict[str, np.ndarray], plan, rng: np.random.Generator):
    """Garbage on every padded position of every value feature."""

    out = {k: v.copy() for k, v in padded.items()}
    t, m = plan.actual["tokens"], plan.actual["msa"]
    # Atoms from the storage size up: the bucket suffix (see the module doc).
    a = plan.storage["atoms"]
    tt = plan.target["tokens"]

    def fill(name, index, value):
        if name in out:
            out[name][index] = value

    # Tokens: a fourth chain of modified, cyclic ligand tokens far down the
    # sequence, with garbage one-hots and profiles.
    fill("res_type", np.s_[0, t:], 1)
    for name in ("profile", "deletion_mean", "cyclic_period"):
        fill(name, np.s_[0, t:], GARBAGE)
    for name in ("asym_id", "entity_id", "sym_id"):
        fill(name, np.s_[0, t:], 7)
    for name in ("residue_index", "token_index"):
        fill(name, np.s_[0, t:], 9999)
    for name in ("mol_type", "method_feature"):
        if tt > t:
            out[name][0, t:] = int(np.max(out[name][0, :t]))
    fill("modified", np.s_[0, t:], 1)
    if tt > t:
        for name, value in (
            ("token_bonds", 1.0),
            ("type_bonds", 1),
            ("contact_conditioning", 1),
            ("contact_threshold", GARBAGE),
        ):
            out[name][0, t:, :] = value
            out[name][0, :, t:] = value
    # Atoms: reference conformer far away, charged, every element and name
    # character at once, one shared reference space.
    for name in ("ref_pos", "ref_charge"):
        fill(name, np.s_[0, a:], GARBAGE)
    for name in ("ref_element", "ref_atom_name_chars"):
        fill(name, np.s_[0, a:], 1)
    fill("ref_space_uid", np.s_[0, a:], 9999)
    # MSA: padded rows and padded columns.
    msa = out["msa"]
    msa[0, m:, :] = garbage_ids(rng, msa[0, m:, :].shape, 33)
    msa[0, :, t:] = garbage_ids(rng, msa[0, :, t:].shape, 33)
    for name, value in (
        ("msa_paired", 1.0),
        ("has_deletion", True),
        ("deletion_value", GARBAGE),
    ):
        out[name][0, m:, :] = value
        out[name][0, :, t:] = value
    # Template rows of the padded tokens (the one slot is the featurizer's
    # dummy template; its masks stay zero).
    fill("template_restype", np.s_[0, :, t:], 1)
    for name in (
        "template_frame_rot",
        "template_frame_t",
        "template_cb",
        "template_ca",
    ):
        fill(name, np.s_[0, :, t:], GARBAGE)
    return out


def _perturb_one_real_residue(padded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(padded)
    res_type = out["res_type"].copy()
    res_type[0, 1] = np.roll(res_type[0, 1], 3)
    out["res_type"] = res_type
    return out


def _crop(plan, n_chain: int):
    def crop(name: str, value: np.ndarray) -> np.ndarray | None:
        if name == "pair_chains_iptm":
            return value[:, :n_chain, :n_chain]
        return np.asarray(
            crop_prediction_outputs(
                {name: value}, plan.actual["tokens"], plan.actual["atoms"]
            )[name]
        )

    return crop


@pytest.mark.parametrize(
    "pin_tokens", [False, True], ids=["bucket", "tokens_on_bucket"]
)
def test_padded_positions_cannot_change_real_outputs(params, features, pin_tokens):
    n_token = int(features["token_pad_mask"].sum())
    n_atom = int(features["atom_pad_mask"].sum())
    n_storage = int(features["atom_pad_mask"].shape[-1])
    n_chain = int(np.max(features["asym_id"])) + 1
    assert n_atom < n_storage, "the featurizer's own window padding is expected"
    config = PaddingConfig(tokens=n_token if pin_tokens else 24, atoms=160, msa=8)
    plan = resolve_padding_plan(features, config)
    assert plan.actual == {"tokens": n_token, "atoms": n_atom, "msa": 1}
    assert plan.storage["atoms"] == n_storage < plan.target["atoms"]
    assert plan.target["tokens"] == (n_token if pin_tokens else 24)
    padded, _ = pad_feats(
        features,
        plan.target["tokens"],
        plan.target["atoms"],
        target_msa=plan.target["msa"],
    )
    padded = {name: np.asarray(value) for name, value in padded.items()}
    # Padded atoms belong to no token: their one-hot rows are empty.
    assert int(padded["atom_to_token"][0, n_atom:].sum()) == 0

    key = jax.random.PRNGKey(3)
    clean = _tapes(key, n_storage, plan.target["atoms"], padded_rows=0.0)
    dirty = _tapes(key, n_storage, plan.target["atoms"], padded_rows=GARBAGE)
    np.testing.assert_array_equal(clean[0][:, :n_storage], dirty[0][:, :n_storage])

    run = functools.partial(_forward, params)
    crop = _crop(plan, n_chain)
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
    assert_some_difference(
        reference, control, ("sample_atom_coords", "complex_plddt", "pae")
    )
