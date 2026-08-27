"""Schema-aware serving buckets for the two ESMFold2 model stages."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.esmfold2 import _padding_plan
from foldjax.models.esmfold2 import inference, output
from foldjax.models.esmfold2.data import chemistry, features
from foldjax.models.esmfold2.models import diffusion, esmc
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.schema import PaddingConfig


def _features() -> dict[str, np.ndarray]:
    return features.build_features([("ACDEFGHIK", "A", 0, 0)])


def _alignment_rows(
    built: dict[str, np.ndarray], rows: int
) -> dict[str, np.ndarray]:
    expanded = dict(built)
    for name in ("msa", "msa_attention_mask", "has_deletion", "deletion_value"):
        expanded[name] = np.repeat(built[name], rows, axis=1)
    for row in range(rows):
        expanded["msa"][0, row] = (
            expanded["msa"][0, row] + row
        ) % structure_model.NUM_RES_TYPES
        expanded["deletion_value"][0, row] = float(row)
        expanded["has_deletion"][0, row] = row > 0
    expanded["deletion_mean"] = expanded["deletion_value"].mean(axis=1)
    return expanded


def _store_msa(
    built: dict[str, np.ndarray], rows: int
) -> dict[str, np.ndarray]:
    return features.pad_features(
        built,
        n_token=built["token_attention_mask"].shape[-1],
        n_atom=built["atom_attention_mask"].shape[-1],
        n_msa=rows,
    )


def test_every_dynamic_feature_axis_is_padded_and_masked() -> None:
    built = _features()
    tokens = built["token_attention_mask"].shape[-1]
    atoms = built["atom_attention_mask"].shape[-1]
    padded = features.pad_features(
        built, n_token=tokens + 7, n_atom=atoms + 32, n_msa=4
    )

    assert padded["token_bonds"].shape == (1, tokens + 7, tokens + 7, 1)
    assert padded["msa"].shape == (1, 4, tokens + 7)
    assert padded["ref_pos"].shape == (1, atoms + 32, 3)
    assert int(padded["token_attention_mask"].sum()) == tokens
    assert int(padded["atom_attention_mask"].sum()) == int(
        built["atom_attention_mask"].sum()
    )
    assert int(padded["msa_attention_mask"].sum()) == int(
        built["msa_attention_mask"].sum()
    )
    assert np.all(padded["msa"][0, 1:] == chemistry.MSA_GAP_TOKEN_ID)
    assert np.all(padded["input_ids"][0, tokens:] == features.ESMC_PAD_TOKEN_ID)
    assert np.all(padded["distogram_atom_idx"][0, tokens:] == 0)


def test_padding_never_shrinks_an_existing_axis() -> None:
    built = _features()
    with pytest.raises(ValueError, match="cannot pad ESMFold2 tokens down"):
        features.pad_features(
            built,
            n_token=built["token_attention_mask"].shape[-1] - 1,
            n_atom=built["atom_attention_mask"].shape[-1],
            n_msa=built["msa_attention_mask"].shape[-2],
        )


def test_padding_plan_carries_real_storage_and_target_dimensions() -> None:
    built = _features()
    plan = _padding_plan(
        built,
        PaddingConfig(),
        max_msa_depth=1024,
        language_model_tokens=11,
    )

    assert plan.actual["tokens"] == 9
    assert plan.storage["atoms"] == built["atom_attention_mask"].shape[-1]
    assert plan.target == {
        "tokens": 256,
        "atoms": 256,
        "msa": 1,
        "language_model_tokens": 128,
    }


def test_deep_msa_plan_normalizes_to_the_active_standard_cap() -> None:
    built = _alignment_rows(_features(), 1500)

    plan = _padding_plan(
        built,
        PaddingConfig(),
        max_msa_depth=1024,
        language_model_tokens=None,
    )

    assert plan.actual["msa"] == 1500
    assert plan.storage["msa"] == 1500
    assert plan.target["msa"] == 1024


def test_explicit_msa_bucket_cannot_change_the_models_scientific_cap() -> None:
    with pytest.raises(ValueError, match="only possible when padding.msa equals"):
        _padding_plan(
            _alignment_rows(_features(), 130),
            PaddingConfig(msa=64),
            max_msa_depth=1024,
            language_model_tokens=None,
        )


def test_nonstandard_active_msa_cap_is_a_stable_target() -> None:
    plan = _padding_plan(
        _alignment_rows(_features(), 130),
        PaddingConfig(),
        max_msa_depth=100,
        language_model_tokens=None,
    )

    assert plan.target["msa"] == 100


def test_padding_plan_can_drop_only_stored_msa_suffix_padding() -> None:
    compact = _alignment_rows(_features(), 5)
    stored = _store_msa(compact, 11)

    plan = _padding_plan(
        stored,
        PaddingConfig(),
        max_msa_depth=3,
        language_model_tokens=None,
    )

    assert plan.actual["msa"] == 5
    assert plan.storage["msa"] == 11
    assert plan.target["msa"] == 3


def test_msa_padding_cannot_cross_the_active_sampling_depth() -> None:
    with pytest.raises(ValueError, match="padded rows could be sampled"):
        _padding_plan(
            _features(),
            PaddingConfig(msa=64),
            max_msa_depth=1,
            language_model_tokens=None,
        )


def test_deep_msa_normalization_keeps_query_order_and_full_profile() -> None:
    built = _alignment_rows(_features(), 5)
    original_deletion_mean = built["deletion_mean"].copy()
    key = jax.random.key(73)
    row_indices = inference.msa_loop_row_indices(
        key, depth=5, max_msa_depth=3, total_steps=4
    )

    normalized = features.normalize_msa_features(
        built, n_msa=3, row_indices=row_indices
    )

    for loop, rows in enumerate(row_indices):
        np.testing.assert_array_equal(
            normalized["msa_loop_tape"][loop], built["msa"][:, rows]
        )
        np.testing.assert_array_equal(
            normalized["deletion_value_loop_tape"][loop],
            built["deletion_value"][:, rows],
        )
    np.testing.assert_array_equal(
        normalized["msa"], built["msa"][:, row_indices[0]]
    )
    np.testing.assert_array_equal(
        normalized["deletion_mean"], original_deletion_mean
    )
    expected = np.eye(structure_model.NUM_RES_TYPES, dtype=np.float32)[
        built["msa"]
    ].mean(axis=1)
    np.testing.assert_array_equal(normalized["msa_profile"], expected)


def test_msa_profile_bounds_masked_one_hot_without_changing_bits(monkeypatch) -> None:
    rng = np.random.default_rng(47)
    msa = rng.integers(0, structure_model.NUM_RES_TYPES, size=(1, 41, 17))
    mask = rng.integers(0, 2, size=msa.shape, dtype=np.int8).astype(bool)
    mask[:, 0] = True
    safe_ids = np.where(mask, msa, 0)
    expected_one_hot = np.eye(structure_model.NUM_RES_TYPES, dtype=np.float32)[
        safe_ids
    ]
    expected_one_hot *= mask[..., None].astype(np.float32)
    expected_counts = np.clip(mask.astype(np.float32).sum(axis=1), 1.0, None)
    expected = expected_one_hot.sum(axis=1) / expected_counts[..., None]

    budget = 17 * structure_model.NUM_RES_TYPES * 6
    monkeypatch.setattr(features, "_MSA_PROFILE_TEMP_BUDGET", budget)
    real_sum = np.sum
    temporary_sizes: list[int] = []

    def tracked_sum(value, *args, **kwargs):
        temporary_sizes.append(value.nbytes)
        return real_sum(value, *args, **kwargs)

    monkeypatch.setattr(features.np, "sum", tracked_sum)
    actual = features._msa_profile(msa, mask)

    assert np.array_equal(actual, expected)
    assert len(temporary_sizes) > 1
    assert max(temporary_sizes) <= budget


def test_msa_profile_still_rejects_an_active_out_of_range_id() -> None:
    msa = np.zeros((1, 2, 3), dtype=np.int64)
    mask = np.ones_like(msa, dtype=bool)
    msa[0, 1, 2] = structure_model.NUM_RES_TYPES

    with pytest.raises(ValueError, match="residue ids outside"):
        features._msa_profile(msa, mask)


def test_msa_loop_tape_mirrors_every_released_key_split() -> None:
    key = jax.random.key(91)
    actual = inference.msa_loop_row_indices(
        key, depth=9, max_msa_depth=4, total_steps=5
    )

    _, _, _, loop_key, _ = jax.random.split(key, 5)
    expected = []
    for _ in range(5):
        loop_key, _, msa_key = jax.random.split(loop_key, 3)
        expected.append(
            np.asarray(structure_model._subsample_msa(msa_key, 9, 4))
        )

    np.testing.assert_array_equal(actual, np.stack(expected))


def test_msa_wrapper_ignores_suffix_storage_width() -> None:
    compact = _alignment_rows(_features(), 5)
    key = jax.random.key(101)
    normalized = [
        inference.normalize_msa_features(
            key,
            _store_msa(compact, storage),
            n_msa=3,
            max_msa_depth=3,
            total_steps=4,
        )
        for storage in (8, 11)
    ]

    for name in (
        "msa_loop_tape",
        "msa_attention_mask_loop_tape",
        "has_deletion_loop_tape",
        "deletion_value_loop_tape",
        "msa_profile",
        "deletion_mean",
    ):
        np.testing.assert_array_equal(normalized[0][name], normalized[1][name])
    np.testing.assert_array_equal(
        normalized[0]["deletion_mean"], compact["deletion_mean"]
    )
    expected_profile = np.eye(
        structure_model.NUM_RES_TYPES, dtype=np.float32
    )[compact["msa"]].mean(axis=1)
    np.testing.assert_array_equal(
        normalized[0]["msa_profile"], expected_profile
    )


def test_deep_msa_tape_matches_default_loop_subsampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _alignment_rows(_features(), 5)
    key = jax.random.key(113)
    total_steps = 3
    normalized = inference.normalize_msa_features(
        key,
        _store_msa(built, 11),
        n_msa=3,
        max_msa_depth=3,
        total_steps=total_steps,
    )

    def fake_msa_encoder(
        _injected,
        _x_inputs,
        _one_hot,
        _has_deletion,
        deletion_value,
        _msa_mask,
        _params,
        _prefix,
        *,
        n_layers,
    ):
        del n_layers
        per_token = jnp.sum(deletion_value, axis=-1)
        return per_token[:, :, None, None] + per_token[:, None, :, None]

    monkeypatch.setattr(structure_model, "msa_encoder", fake_msa_encoder)
    monkeypatch.setattr(
        structure_model, "folding_trunk", lambda value, *args, **kwargs: value
    )
    monkeypatch.setattr(
        structure_model, "layer_norm", lambda value, *args, **kwargs: value
    )

    raw_mask = jnp.asarray(built["msa_attention_mask"], dtype=jnp.float32)
    raw_one_hot = jax.nn.one_hot(
        jnp.asarray(built["msa"]), structure_model.NUM_RES_TYPES
    )
    default_inputs = {
        "msa_one_hot": jnp.swapaxes(raw_one_hot, 1, 2),
        "msa_mask": jnp.swapaxes(raw_mask, 1, 2),
        "has_deletion": jnp.swapaxes(
            jnp.asarray(built["has_deletion"], dtype=jnp.float32), 1, 2
        ),
        "deletion_value": jnp.swapaxes(
            jnp.asarray(built["deletion_value"], dtype=jnp.float32), 1, 2
        ),
        "x_inputs": jnp.zeros((1, 9, 1), dtype=jnp.float32),
    }
    tape_inputs = {
        "loop_tape": (
            jnp.asarray(normalized["msa_loop_tape"]),
            jnp.asarray(
                normalized["msa_attention_mask_loop_tape"], dtype=jnp.float32
            ),
            jnp.asarray(
                normalized["has_deletion_loop_tape"], dtype=jnp.float32
            ),
            jnp.asarray(
                normalized["deletion_value_loop_tape"], dtype=jnp.float32
            ),
        ),
        "x_inputs": jnp.zeros((1, 9, 1), dtype=jnp.float32),
    }
    settings = structure_model.ModelSettings(
        d_pair=1,
        msa_n_layers=1,
        max_msa_depth=3,
        trunk_n_layers=1,
        num_recycles=2,
    )
    params = {
        "parcae_log_delta": jnp.zeros((1,), dtype=jnp.float32),
        "parcae_log_a": jnp.zeros((1,), dtype=jnp.float32),
        "parcae_b_cont": jnp.ones((1, 1), dtype=jnp.float32),
        "parcae_input_norm.weight": jnp.ones((1,), dtype=jnp.float32),
        "parcae_input_norm.bias": jnp.zeros((1,), dtype=jnp.float32),
    }
    pair = jnp.zeros((1, 9, 9, 1), dtype=jnp.float32)
    pair_mask = jnp.ones((1, 9, 9), dtype=jnp.float32)
    _, _, _, loop_key, _ = jax.random.split(key, 5)

    exact = structure_model.run_loops(
        loop_key,
        pair,
        pair,
        None,
        default_inputs,
        pair_mask,
        params,
        settings=settings,
        total_steps=total_steps,
    )
    taped = structure_model.run_loops(
        loop_key,
        pair,
        pair,
        None,
        tape_inputs,
        pair_mask,
        params,
        settings=settings,
        total_steps=total_steps,
        preserve_prefix_rng=True,
    )

    np.testing.assert_array_equal(np.asarray(taped), np.asarray(exact))


def test_msa_normalization_drops_stored_dummy_rows_then_repads_them_masked() -> None:
    built = _alignment_rows(_features(), 3)
    stored = _store_msa(built, 6)
    row_indices = np.tile(np.arange(3, dtype=np.int64), (2, 1))
    normalized = features.normalize_msa_features(
        stored, n_msa=4, row_indices=row_indices
    )
    padded = features.pad_features(
        normalized,
        n_token=normalized["token_attention_mask"].shape[-1],
        n_atom=normalized["atom_attention_mask"].shape[-1],
        n_msa=4,
    )

    assert padded["msa"].shape[1] == 4
    assert not np.any(padded["msa_attention_mask"][:, 3:])
    assert not np.any(padded["has_deletion"][:, 3:])
    assert not np.any(padded["deletion_value"][:, 3:])
    assert np.all(padded["msa"][:, 3:] == chemistry.MSA_GAP_TOKEN_ID)
    assert not np.any(padded["msa_attention_mask_loop_tape"][:, :, 3:])
    assert np.all(
        padded["msa_loop_tape"][:, :, 3:] == chemistry.MSA_GAP_TOKEN_ID
    )


def test_msa_normalization_fails_closed_for_an_unknown_batch_layout() -> None:
    built = _alignment_rows(_features(), 3)
    built = {name: np.repeat(value, 2, axis=0) for name, value in built.items()}

    with pytest.raises(ValueError, match="requires one batched alignment"):
        features.normalize_msa_features(
            built,
            n_msa=2,
            row_indices=np.asarray([[0, 1]], dtype=np.int64),
        )


def test_explicit_atom_bucket_must_follow_the_native_block() -> None:
    with pytest.raises(ValueError, match="multiple of its 32-atom"):
        _padding_plan(
            _features(),
            PaddingConfig(atoms=257),
            max_msa_depth=1024,
            language_model_tokens=None,
        )


def test_esmc_packed_length_is_right_padded_without_moving_real_tokens() -> None:
    ids = np.asarray([[10, 11, 12, 1]], dtype=np.int64)
    asym = np.asarray([[0, 0, 0, 0]], dtype=np.int64)
    residues = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    mol_type = np.zeros_like(ids)
    mask = np.asarray([[1, 1, 1, 0]], dtype=np.int64)

    natural, natural_chains, natural_expand = esmc.pack_lm_inputs(
        ids, asym, residues, mol_type, mask
    )
    padded, padded_chains, padded_expand = esmc.pack_lm_inputs(
        ids, asym, residues, mol_type, mask, packed_length=16
    )

    assert padded.shape == (1, 16)
    np.testing.assert_array_equal(padded[:, : natural.shape[1]], natural)
    np.testing.assert_array_equal(
        padded_chains[:, : natural_chains.shape[1]], natural_chains
    )
    assert np.all(padded[:, natural.shape[1] :] == esmc.PAD_TOKEN_ID)
    assert np.all(padded_chains[:, natural.shape[1] :] == -1)
    np.testing.assert_array_equal(padded_expand, natural_expand)


def test_pair_state_dropout_and_column_draws_preserve_compact_prefix() -> None:
    key = jax.random.key(41)
    compact_tokens = jnp.ones((1, 3), dtype=bool)
    padded_tokens = jnp.asarray([[1, 1, 1, 0, 0]], dtype=bool)
    compact_pairs = compact_tokens[:, :, None] & compact_tokens[:, None, :]
    padded_pairs = padded_tokens[:, :, None] & padded_tokens[:, None, :]

    exact_state = structure_model._initial_pair_state_draw(
        key, compact_pairs, 4, preserve_prefix_rng=False
    )
    padded_state = structure_model._initial_pair_state_draw(
        key, padded_pairs, 4, preserve_prefix_rng=True
    )
    np.testing.assert_array_equal(
        np.asarray(padded_state[:, :3, :3]), np.asarray(exact_state)
    )

    exact_dropout = structure_model._dropout(
        key, jnp.ones((1, 3, 3, 2), dtype=jnp.float32), 0.25
    )
    padded_dropout = structure_model._dropout(
        key,
        jnp.ones((1, 5, 5, 2), dtype=jnp.float32),
        0.25,
        valid_mask=padded_pairs,
        preserve_prefix_rng=True,
    )
    np.testing.assert_array_equal(
        np.asarray(padded_dropout[:, :3, :3]), np.asarray(exact_dropout)
    )

    exact_columns = structure_model._msa_column_keep(
        key, compact_tokens, 0.1, preserve_prefix_rng=False
    )
    padded_columns = structure_model._msa_column_keep(
        key, padded_tokens, 0.1, preserve_prefix_rng=True
    )
    np.testing.assert_array_equal(
        np.asarray(padded_columns[:, :3]), np.asarray(exact_columns)
    )
    assert not np.any(np.asarray(padded_state)[:, 3:])
    assert not np.any(np.asarray(padded_state)[:, :, 3:])
    assert not np.any(np.asarray(padded_dropout)[:, 3:])
    assert not np.any(np.asarray(padded_dropout)[:, :, 3:])
    assert not np.any(np.asarray(padded_columns)[:, 3:])


def test_diffusion_atom_noise_preserves_compact_prefix_for_every_sample() -> None:
    key = jax.random.key(57)
    compact_mask = jnp.zeros((2, 32), dtype=bool).at[:, :3].set(True)
    padded_mask = jnp.zeros((2, 64), dtype=bool).at[:, :3].set(True)

    exact = diffusion._atom_normal(
        key,
        compact_mask,
        dtype=jnp.float32,
        preserve_prefix_rng=False,
    )
    padded = diffusion._atom_normal(
        key,
        padded_mask,
        dtype=jnp.float32,
        preserve_prefix_rng=True,
    )

    np.testing.assert_array_equal(
        np.asarray(padded[:, :3]), np.asarray(exact[:, :3])
    )
    assert not np.any(np.asarray(padded[:, 3:]))


def test_public_crop_and_score_drop_masked_suffixes() -> None:
    built = _features()
    tokens = built["token_attention_mask"].shape[-1]
    atoms = int(built["atom_attention_mask"].sum())
    padded = features.pad_features(
        built,
        n_token=tokens + 3,
        n_atom=built["atom_attention_mask"].shape[-1] + 32,
        n_msa=1,
    )
    prediction = {
        "sample_atom_coords": np.zeros(
            (1, padded["atom_attention_mask"].shape[-1], 3), dtype=np.float32
        ),
        "plddt_per_atom": np.ones(
            (1, padded["atom_attention_mask"].shape[-1]), dtype=np.float32
        ),
        "plddt": np.concatenate(
            [np.full((1, tokens), 0.8), np.zeros((1, 3))], axis=-1
        ),
        "pae_logits": np.zeros((1, tokens + 3, tokens + 3, 8)),
        "pae": np.zeros((1, tokens + 3, tokens + 3)),
        "pde": np.zeros((1, tokens + 3, tokens + 3)),
        "complex_plddt": np.asarray([0.8]),
    }

    cropped = output.crop_prediction(prediction, padded)

    assert cropped["sample_atom_coords"].shape == (1, atoms, 3)
    assert cropped["plddt"].shape == (1, tokens)
    assert cropped["pae_logits"].shape == (1, tokens, tokens, 8)
    assert cropped["pae"].shape == (1, tokens, tokens)
    assert cropped["pde"].shape == (1, tokens, tokens)
    assert output.sample_scores(cropped)[0]["plddt"] == pytest.approx(0.8)
    assert output.sample_scores(
        prediction, token_mask=padded["token_attention_mask"]
    )[0]["plddt"] == pytest.approx(0.8)
