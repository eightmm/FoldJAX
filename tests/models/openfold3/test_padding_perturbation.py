"""Padded positions cannot change OpenFold3's real outputs.

`predict` -- input embedder, one-block MSA module, template embedder and
Pairformer, diffusion sampler, distogram and confidence heads -- runs twice on
one bucket-padded two-chain-plus-ligand query featurized here with the released
CLI's query-only MSA: once as `pad_features` produced it, once with every
padded token, atom, MSA row and column, template row and column and noise-tape
row overwritten with large finite garbage. The real slices of every output must
agree bit for bit.

The MSA rows are padded on the host and then handed to the released row
planner, so the filler rows it selects for a shallow MSA are the padded ones
and carry the garbage. The four template slots are the native fixed depth, not
padding, so only their padded token rows and columns are overwritten. The
second case pins the token axis to the real token count while atoms are still
bucketed. Masks and selectors -- `token_mask`, `atom_mask`, `msa_mask`, the
template masks, `atom_to_token_index`, `start_atom_index`,
`num_atoms_per_token`, `max_atom_per_token_mask`, the MSA cycle indices -- are
left as padding wrote them: padded atoms point at real token 0 exactly as the
helper leaves them.

The triangle kernels run on XLA: the fused cuEquivariance path pins TF32
matmuls, which the CPU backend refuses.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax._openfold3_compile import triangle_backend
from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.bridge import chemistry
from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
from foldjax.models.openfold3.data import featurize as data
from foldjax.models.openfold3.output import crop_prediction
from tests.models._padding_perturbation import (
    GARBAGE,
    assert_bitwise_identical,
    assert_some_difference,
    real_slices,
)
from tests.models._tiny_weights import load_manifest, random_state_dict

STEPS, SAMPLES = 3, 2
MSA_ROWS = 4

SPEC = {
    "queries": {
        "perturbation": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "MKTAYIAK",
                },
                {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "GSWEL"},
                {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "CC(=O)O"},
            ]
        }
    }
}

REQUIRED = (
    "coordinates",
    "plddt",
    "ptm",
    "iptm",
    "chain_pair_iptm",
    "bespoke_iptm",
    "pae",
    "pde",
    "gpde",
    "pae_logits",
    "distogram_logits",
)

#: Per-chain scores sized by the chain capacity rather than a token or atom axis.
CHAIN_OUTPUTS = {"chain_pair_iptm", "chain_ptm", "bespoke_iptm"}


@pytest.fixture(scope="module")
def params() -> inference.InferenceParams:
    state = random_state_dict(load_manifest("openfold3"), seed=0)
    return map_inference_params(state, "")


@pytest.fixture(scope="module")
def features() -> dict[str, np.ndarray]:
    with pytest.warns(UserWarning, match="query-only dummy MSA"):
        return data.featurize_query(SPEC, seed=0)


def _config(n_token: int, n_atom: int) -> inference.InferenceConfig:
    return inference.released_config(
        n_token=n_token,
        n_atom=n_atom,
        num_recycles=1,
        num_samples=SAMPLES,
        num_steps=STEPS,
        pair_chunk_size=None,
        diffusion_chunk_size=None,
        msa_depth=MSA_ROWS,
        dtype="float32",
        padded=True,
    )


def _forward(config, n_chain: int):
    table = chemistry.representative_atom_table()

    @jax.jit
    def run(batch, params, tape):
        return inference.predict(
            jax.random.key(7),
            batch,
            params,
            config,
            table,
            n_chain=n_chain,
            noise_tape=tape,
        )

    def forward(batch, params, tape):
        with triangle_backend("xla"):
            out = run({k: jnp.asarray(v) for k, v in batch.items()}, params, tape)
        return {k: v for k, v in out._asdict().items() if v is not None}

    return forward


def _tape(key, n_atoms_real: int, n_atoms: int, *, padded_rows: float) -> jnp.ndarray:
    """Initial plus per-step sampler draws; rows of padded atoms are the caller's."""

    tape = np.full((STEPS + 1, SAMPLES, n_atoms, 3), padded_rows, dtype=np.float32)
    tape[:, :, :n_atoms_real] = jax.random.normal(
        key, (STEPS + 1, SAMPLES, n_atoms_real, 3)
    )
    return jnp.asarray(tape)


def _perturb_padding(padded: dict[str, Any], t: int, a: int, m: int) -> dict[str, Any]:
    """Garbage on every padded position of every value feature."""

    out = {k: np.array(v, copy=True) for k, v in padded.items()}
    tt = out["token_mask"].shape[-1]

    def fill(name, index, value):
        if name in out:
            out[name][index] = value

    # Tokens: a fourth chain of cyclic, atomized tokens that are protein, DNA,
    # RNA and ligand at once, far down the sequence, with garbage one-hots.
    for name in ("restype", "is_atomized", "is_protein", "is_dna", "is_rna"):
        fill(name, np.s_[0, t:], 1)
    fill("is_ligand", np.s_[0, t:], 1)
    for name in ("profile", "deletion_mean"):
        fill(name, np.s_[0, t:], GARBAGE)
    for name in ("asym_id", "entity_id", "sym_id"):
        fill(name, np.s_[0, t:], 7)
    for name in ("residue_index", "token_index"):
        fill(name, np.s_[0, t:], 9999)
    fill("cyclic_mask", np.s_[0, t:], True)
    if tt > t:
        out["token_bonds"][0, t:, :] = 1
        out["token_bonds"][0, :, t:] = 1
    # Atoms: reference conformer far away, charged, every element and name
    # character at once, flagged as a valid reference atom in one shared space.
    for name in ("ref_pos", "ref_charge"):
        fill(name, np.s_[0, a:], GARBAGE)
    for name in ("ref_element", "ref_atom_name_chars", "ref_mask"):
        fill(name, np.s_[0, a:], 1)
    fill("ref_space_uid", np.s_[0, a:], 9999)
    # MSA: padded rows and padded columns.
    out["msa"][0, m:, :] = 1
    out["msa"][0, :, t:] = 1
    for name, value in (("has_deletion", 1.0), ("deletion_value", GARBAGE)):
        out[name][0, m:, :] = value
        out[name][0, :, t:] = value
    # Templates: padded token rows and columns of every slot.
    if tt > t:
        for name in ("template_distogram", "template_unit_vector"):
            out[name][0, :, t:, :] = GARBAGE
            out[name][0, :, :, t:] = GARBAGE
        out["template_restype"][0, :, t:] = 1
    return out


def _perturb_one_real_residue(padded: dict[str, Any]) -> dict[str, Any]:
    out = dict(padded)
    restype = np.array(out["restype"], copy=True)
    restype[0, 1] = np.roll(restype[0, 1], 3)
    out["restype"] = restype
    return out


def _crop(padded: dict[str, Any], n_chain: int):
    empty = {field: None for field in inference.Prediction._fields}

    def crop(name: str, value: np.ndarray) -> np.ndarray | None:
        if name in CHAIN_OUTPUTS:
            if value.ndim == 3:
                return value[:, :n_chain, :n_chain]
            return value[:, :n_chain]
        prediction = inference.Prediction(**{**empty, name: value})
        return np.asarray(getattr(crop_prediction(prediction, padded), name))

    return crop


@pytest.mark.parametrize(
    "pin_tokens", [False, True], ids=["bucket", "tokens_on_bucket"]
)
def test_padded_positions_cannot_change_real_outputs(params, features, pin_tokens):
    n_token = int(features["token_mask"].sum())
    n_atom = int(features["atom_mask"].sum())
    n_msa = int(features["msa_mask"].shape[1])
    assert features["token_mask"].shape[-1] == n_token
    assert features["atom_mask"].shape[-1] == n_atom
    assert n_msa < MSA_ROWS
    target_tokens = n_token if pin_tokens else 24
    target_atoms = 160

    padded = data.pad_features(
        features,
        n_token=target_tokens,
        n_atom=target_atoms,
        n_msa=MSA_ROWS,
        n_templates=4,
    )
    # The released row planner fills a shallow MSA with invalid rows: here the
    # padded ones, every one of them, in a seeded order.
    padded = data.prepare_msa_cycle_features(
        padded, MSA_ROWS, num_recycles=1, rng=np.random.default_rng(0)
    )
    selected = sorted(padded["_foldjax_msa_cycle_indices"][0].tolist())
    assert selected == list(range(MSA_ROWS))
    padded, n_chain = data.normalize_asym_ids(padded)
    assert n_chain == 3
    # Padded atoms are owned by real token 0, as the helper leaves them.
    np.testing.assert_array_equal(padded["atom_to_token_index"][0, n_atom:], 0)
    np.testing.assert_array_equal(padded["start_atom_index"][0, n_token:], 0)

    key = jax.random.key(3)
    clean = _tape(key, n_atom, target_atoms, padded_rows=0.0)
    dirty = _tape(key, n_atom, target_atoms, padded_rows=GARBAGE)
    np.testing.assert_array_equal(clean[:, :, :n_atom], dirty[:, :, :n_atom])

    run = _forward(_config(target_tokens, target_atoms), n_chain)
    crop = _crop(padded, n_chain)
    reference = real_slices(run(padded, params, clean), crop)
    garbage = _perturb_padding(padded, n_token, n_atom, n_msa)
    perturbed = real_slices(run(garbage, params, dirty), crop)
    assert_bitwise_identical(reference, perturbed, require=REQUIRED)

    control = real_slices(run(_perturb_one_real_residue(padded), params, clean), crop)
    assert_some_difference(reference, control, ("coordinates", "plddt", "pae"))
