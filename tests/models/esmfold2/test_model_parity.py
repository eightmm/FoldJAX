"""The whole model against torch, as far as determinism reaches.

ESMFold2 is stochastic at inference on purpose: the pair state is a random
draw, the LM pair embedding is dropped out per loop, and the sampler adds
noise. What *is* deterministic once the initial state is fixed is everything
from the features through the parcae recurrence to the distogram, and that is
what this file gates -- the longest reproducible path the model has.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")
modeling = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2")
configuration = pytest.importorskip(
    "transformers.models.esmfold2.configuration_esmfold2"
)

from foldjax.models.esmfold2.models import model as jax_model  # noqa: E402
from tests.models.esmfold2.parity import torch_state_to_numpy  # noqa: E402

D_PAIR, D_ATOM, D_TOKEN, N_HEADS = 16, 64, 32, 2
D_INPUTS = D_TOKEN // 2 + 33 + 33 + 1
N_TOKENS, N_ATOMS = 6, 18
LM_LAYERS, LM_WIDTH = 3, 8


def _config():
    return configuration.ESMFold2Config(
        d_single=24,
        d_pair=D_PAIR,
        num_recycles=1,
        num_diffusion_samples=1,
        lm_d_model=LM_WIDTH,
        lm_num_layers=LM_LAYERS,
        inputs={
            "d_inputs": D_INPUTS,
            "atom_encoder": {
                "d_atom": D_ATOM,
                "d_token": D_TOKEN,
                "n_blocks": 1,
                "n_heads": N_HEADS,
                "swa_window_size": 8,
            },
        },
        folding_trunk={"n_layers": 1},
        parcae={"coda_n_layers": 1},
        lm_encoder={"enabled": True, "n_layers": 1},
        structure_head={
            "distogram_bins": 12,
            "inference_num_steps": 2,
            "diffusion_module": {
                "c_atom": D_ATOM,
                "c_token": D_TOKEN,
                "c_z": D_PAIR,
                "c_s_inputs": D_INPUTS,
                "fourier_dim": 16,
                "atom_num_blocks": 1,
                "atom_num_heads": N_HEADS,
                "token_num_blocks": 1,
                "token_num_heads": N_HEADS,
            },
        },
        confidence_head={
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
            "distogram_bins": 12,
            "folding_trunk": {"n_layers": 1},
        },
    )


def _features(seed=0):
    rng = np.random.default_rng(seed)
    token_index = np.arange(N_TOKENS, dtype=np.int64)[None]
    return {
        "token_index": token_index,
        "residue_index": token_index.copy(),
        "asym_id": (np.arange(N_TOKENS, dtype=np.int64) // 3)[None],
        "sym_id": np.zeros((1, N_TOKENS), dtype=np.int64),
        "entity_id": (np.arange(N_TOKENS, dtype=np.int64) // 3)[None],
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
    }


def test_the_inputs_embedder_matches() -> None:
    config = _config()
    module = common.InputsEmbedder(config).eval()
    data = _features(1)
    settings = jax_model.settings_from_config(config.to_dict())
    element = np.eye(128, dtype=np.float32)[data["ref_element"]]
    chars = np.eye(64, dtype=np.float32)[data["ref_atom_name_chars"]]
    res_type = np.eye(33, dtype=np.float32)[data["res_type"]]
    profile = res_type.copy()
    deletion_mean = np.zeros((1, N_TOKENS), dtype=np.float32)

    with torch.no_grad():
        expected = module(
            aatype=torch.from_numpy(res_type),
            profile=torch.from_numpy(profile),
            deletion_mean=torch.from_numpy(deletion_mean),
            ref_pos=torch.from_numpy(data["ref_pos"]),
            atom_attention_mask=torch.from_numpy(data["atom_attention_mask"]),
            ref_space_uid=torch.from_numpy(data["ref_space_uid"]),
            ref_charge=torch.from_numpy(data["ref_charge"]),
            ref_element=torch.from_numpy(element),
            ref_atom_name_chars=torch.from_numpy(chars),
            atom_to_token=torch.from_numpy(data["atom_to_token"]),
        )
    got = jax_model.inputs_embedding(
        res_type,
        profile,
        deletion_mean,
        data["ref_pos"],
        data["atom_attention_mask"],
        data["ref_space_uid"],
        data["ref_charge"],
        element,
        chars,
        data["atom_to_token"],
        torch_state_to_numpy(module),
        prefix="",
        settings=settings,
        n_tokens=N_TOKENS,
    )
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-4, rtol=2e-4)


def test_the_language_model_shim_matches() -> None:
    """The softmax over layers, and that it reads the whole stack."""
    module = common.LanguageModelShim(
        d_z=D_PAIR, d_model=LM_WIDTH, num_layers=LM_LAYERS
    ).eval()
    with torch.no_grad():
        module.base_z_combine.copy_(torch.randn(LM_LAYERS + 1))
    hidden = (
        np.random.default_rng(2)
        .standard_normal((1, N_TOKENS, LM_LAYERS + 1, LM_WIDTH))
        .astype(np.float32)
    )
    with torch.no_grad():
        expected = module(torch.from_numpy(hidden))
    got = jax_model.language_model_pair(hidden, torch_state_to_numpy(module), "")
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-5, rtol=2e-5)


def test_the_trunk_and_distogram_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Features through the parcae recurrence to the distogram logits.

    The initial pair state is the model's only nondeterminism on this path, so
    both sides are handed the same one.
    """
    config = _config()
    module = modeling.ESMFold2Model(config).eval()
    module.set_chunk_size(None)
    data = _features(3)
    state = (
        np.random.default_rng(4)
        .standard_normal((1, N_TOKENS, N_TOKENS, D_PAIR))
        .astype(np.float32)
        * 0.1
    )
    monkeypatch.setattr(
        modeling.ESMFold2Model,
        "_init_pair_state",
        lambda self, ref: torch.from_numpy(state),
    )

    with torch.no_grad():
        expected = module(
            **{k: torch.from_numpy(v) for k, v in data.items()},
            num_sampling_steps=1,
        )["distogram_logits"]

    params = torch_state_to_numpy(module)
    params["confidence_head.boundaries"] = module.confidence_head.boundaries.numpy()
    got = jax_model.predict(
        jax.random.key(0),
        data,
        params,
        # float32 on both sides: torch's autocast is CUDA-only and this runs on
        # CPU, so a bfloat16 trunk here would compare two different models.
        # `test_the_released_trunk_is_bfloat16` is what pins the real default.
        settings=dataclasses.replace(
            jax_model.settings_from_config(config.to_dict()), trunk_dtype="float32"
        ),
        initial_pair_state=state,
    )["distogram_logits"]
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("with_msa", [False, True])
def test_native_bf16_trunk_and_distogram_replay_lm_dropout_tape(with_msa):
    """Actual tiny trunk, fixed features/LM/pair; no sampler or MSA randomness."""
    import jax.numpy as jnp

    config = _config()
    if with_msa:
        config.msa_encoder.enabled = True
        config.msa_encoder.n_layers = 1
    native = modeling.ESMFold2Model(config).eval()
    params = torch_state_to_numpy(native)
    params["confidence_head.boundaries"] = native.confidence_head.boundaries.numpy()
    features = {k: jnp.asarray(v) for k, v in _features(3).items()}
    if with_msa:
        features.update(
            msa=jnp.arange(24).reshape(1, 4, 6) % 20,
            msa_attention_mask=jnp.ones((1, 4, 6)),
            has_deletion=jnp.zeros((1, 4, 6)),
            deletion_value=jnp.zeros((1, 4, 6)),
        )
    settings = dataclasses.replace(
        jax_model.settings_from_config(config.to_dict()),
        num_recycles=1,
        per_loop_lm_dropout=True,
        lm_dropout=0.25,
        max_msa_depth=2,
        msa_column_mask_rate=0.2,
    )
    assert settings.trunk_dtype == "bfloat16"
    rng = np.random.default_rng(72)
    initial = jnp.asarray(rng.normal(size=(1, N_TOKENS, N_TOKENS, D_PAIR)), jnp.float32)
    hidden = jnp.asarray(
        rng.normal(size=(1, N_TOKENS, LM_LAYERS + 1, LM_WIDTH)), jnp.float32
    )
    masks = jnp.asarray(rng.random((2, 1, N_TOKENS, N_TOKENS, D_PAIR)) > 0.25)

    def run(key, tape, column_override=None):
        column = (
            jnp.array([[True, False, True, False, True, False]])
            if column_override is None
            else column_override
        )
        output = jax_model.predict(
            key,
            features,
            params,
            settings=settings,
            lm_hidden_states=hidden,
            initial_pair_state=initial,
            lm_dropout_masks=tape,
            msa_column_keep=column if with_msa else None,
            msa_row_choices=jnp.array([[0, 1], [0, 3]], jnp.int32)
            if with_msa
            else None,
            n_chains=2,
            stop_after_trunk=True,
            return_representations=("pair",),
        )
        return output["pair"], jax_model._distogram_logits(output["pair"], params)

    # This gate is tape/key independence in each execution mode, not a
    # BF16 eager-versus-fused arithmetic parity claim.
    for fn in (run, jax.jit(run)):
        first = fn(jax.random.key(0), masks)
        other_key = fn(jax.random.key(99), masks)
        for left, right in zip(first, other_key, strict=True):
            np.testing.assert_array_equal(left, right)
        changed = fn(jax.random.key(0), ~masks)
        assert not np.array_equal(first[0], changed[0])
        assert not np.array_equal(first[1], changed[1])
        if with_msa:
            changed_columns = fn(jax.random.key(0), masks, jnp.ones((1, 6), bool))
            assert not np.array_equal(first[0], changed_columns[0])
            assert not np.array_equal(first[1], changed_columns[1])


def test_the_released_trunk_is_bfloat16() -> None:
    """Upstream wraps four regions in `torch.amp.autocast(bfloat16)` on CUDA.

    The trunk from the inputs embedder to the parcae coda, the confidence
    head's folding trunk, the diffusion conditioning's pair transitions, and
    the language model. A port that ran the trunk in float32 would not be
    safer, it would be a different model from every published number.
    """
    settings = jax_model.settings_from_config(_config().to_dict())
    assert settings.trunk_dtype == "bfloat16"
