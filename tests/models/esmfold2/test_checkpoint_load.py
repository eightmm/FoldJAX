"""The released checkpoint, loaded and run end to end.

The port names its parameters the way upstream's `state_dict` does, so there
is no rename table to test -- but that only helps if every name the model asks
for is a name the file has. This runs the real weights on a toy complex with
every optional branch on (MSA encoder, LM encoder, language-model shim), which
is the one check that reaches all 1,594 of them.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from foldjax.models.esmfold2.bridge import checkpoint
from foldjax.models.esmfold2.models import model as jax_model
from foldjax.paths import weights_dir

N_TOKENS, N_ATOMS, MSA_DEPTH = 8, 24, 3


def _weights():
    directory = weights_dir("esmfold2")
    if not (directory / checkpoint.WEIGHTS_NAME).exists():
        pytest.skip("esmfold2 weights are not in the store")
    return directory


def _features():
    rng = np.random.default_rng(0)
    token_index = np.arange(N_TOKENS, dtype=np.int64)[None]
    msa = rng.integers(0, 20, (1, MSA_DEPTH, N_TOKENS)).astype(np.int64)
    return {
        "token_index": token_index,
        "residue_index": token_index.copy(),
        # Two chains, so the inter-chain confidence terms are real.
        "asym_id": (np.arange(N_TOKENS, dtype=np.int64) // 4)[None],
        "sym_id": np.zeros((1, N_TOKENS), dtype=np.int64),
        "entity_id": (np.arange(N_TOKENS, dtype=np.int64) // 4)[None],
        "mol_type": np.zeros((1, N_TOKENS), dtype=np.int64),
        "res_type": rng.integers(0, 20, (1, N_TOKENS)).astype(np.int64),
        "token_bonds": np.zeros((1, N_TOKENS, N_TOKENS, 1), dtype=np.float32),
        "token_attention_mask": np.ones((1, N_TOKENS), dtype=np.float32),
        "ref_pos": rng.standard_normal((1, N_ATOMS, 3)).astype(np.float32),
        "ref_element": np.full((1, N_ATOMS), 6, dtype=np.int64),
        "ref_charge": np.zeros((1, N_ATOMS), dtype=np.float32),
        "ref_atom_name_chars": np.zeros((1, N_ATOMS, 4), dtype=np.int64),
        "ref_space_uid": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
        "atom_attention_mask": np.ones((1, N_ATOMS), dtype=np.float32),
        "atom_to_token": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
        "distogram_atom_idx": (np.arange(N_TOKENS, dtype=np.int64) * 3)[None],
        "msa": msa,
        "msa_attention_mask": np.ones((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
        "has_deletion": np.zeros((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
        "deletion_value": np.zeros((1, MSA_DEPTH, N_TOKENS), dtype=np.float32),
    }


def test_the_released_settings_are_read_from_the_file() -> None:
    """Not the dataclass defaults, which differ in almost every field."""
    settings = checkpoint.load_settings(_weights())
    assert settings.trunk_n_layers == 48
    assert settings.num_loops == 3
    assert settings.msa_n_layers == 4
    assert settings.diffusion.num_steps == 14
    # The churn the dataclass calls zero. A sampler that took the default
    # would run a noiseless ODE and never say so.
    assert settings.diffusion.noise_scale == pytest.approx(1.003)


@pytest.mark.slow
def test_the_whole_model_runs_on_the_released_weights() -> None:
    directory = _weights()
    parameters = checkpoint.load_parameters(directory)
    settings = jax_model.with_overrides(
        checkpoint.load_settings(directory), num_loops=1, num_samples=2, num_steps=2
    )
    features = _features()
    hidden = np.zeros((1, N_TOKENS, 81, 2560), dtype=np.float32)

    output = jax_model.predict(
        jax.random.key(0),
        features,
        parameters,
        settings=settings,
        lm_hidden_states=hidden,
    )

    assert output["sample_atom_coords"].shape == (2, N_ATOMS, 3)
    assert output["plddt"].shape == (2, N_TOKENS)
    assert output["pair_chains_iptm"].shape == (2, 2, 2)
    for name, value in output.items():
        assert np.all(np.isfinite(np.asarray(value))), name
    # pLDDT is a probability-weighted mean over bins spanning 0 to 1.
    plddt = np.asarray(output["plddt"])
    assert plddt.min() >= 0.0 and plddt.max() <= 1.0
