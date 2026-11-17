"""Schema-aware serving buckets for the two ESMFold2 model stages."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import confidence_arrays
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
        "tokens": 128,
        # The stored atoms on the atom grid, not 24 per token (3,072).
        "atoms": 256,
        "msa": 1024,
        "language_model_tokens": 384,
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
        native_opm_params=None,
        workspace=None,
    ):
        del n_layers, workspace
        assert native_opm_params is None
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
    assert output.sample_scores(prediction, token_mask=padded["token_attention_mask"])[
        0
    ]["plddt"] == pytest.approx(0.8)


def test_same_bucket_predict_and_writer_do_not_compile_for_another_real_length(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = structure_model.ModelSettings(trunk_dtype="float32")
    loaded = inference.LoadedModel(parameters={}, settings=settings)
    traced: list[tuple[int, ...]] = []

    def tiny_predict(key, built, parameters, **kwargs):
        del key, parameters, kwargs
        traced.append(built["token_attention_mask"].shape)
        token_mask = built["token_attention_mask"].astype(jnp.float32)
        return {
            "sample_atom_coords": built["ref_pos"],
            "plddt": token_mask * 0.8,
            "plddt_per_atom": built["atom_attention_mask"].astype(jnp.float32) * 0.8,
            "pae": token_mask[:, :, None] * token_mask[:, None, :],
        }

    monkeypatch.setattr(structure_model, "predict", tiny_predict)
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", raising=False)
    built_inputs = [
        features.pad_features(
            features.build_features([(sequence, "A", 0, 0)]),
            n_token=16,
            n_atom=128,
            n_msa=4,
        )
        for sequence in ("ACD", "ACDEF")
    ]
    key = jax.random.key(0)
    inference.compiled_predict.cache_clear()
    caplog.set_level(logging.WARNING)
    try:
        with jax.log_compiles(True):
            for index, built in enumerate(built_inputs):
                caplog.clear()
                prediction = inference.predict(
                    key, built, loaded, preserve_prefix_rng=True, padded=True
                )
                written = output.write_prediction_outputs(
                    prediction, built, tmp_path / str(index), name="same"
                )
                assert written["summary"][0]["plddt"] == pytest.approx(0.8)
                assert len(written["structures"]) == 1
                with np.load(
                    confidence_arrays.staged_path(written["structures"][0])
                ) as archive:
                    tokens = int(built["token_attention_mask"].sum())
                    atoms = int(built["atom_attention_mask"].sum())
                    assert archive["token_plddt"].shape == (tokens,)
                    assert archive["atom_plddt"].shape == (atoms,)
                    assert archive["pae"].shape == (tokens, tokens)
                if index:
                    compiled = [
                        record.getMessage()
                        for record in caplog.records
                        if "Compiling jit(" in record.getMessage()
                    ]
                    assert not compiled, compiled
        assert traced == [(1, 16)]
    finally:
        inference.compiled_predict.cache_clear()


def test_writer_leaves_unused_native_arrays_on_the_device(tmp_path) -> None:
    class UnusedArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("an unused native output was copied to the host")

        def __getitem__(self, key):
            raise AssertionError("an unused native output was sliced on the device")

    built = _features()
    atoms = built["atom_attention_mask"].shape[-1]
    prediction = {
        "sample_atom_coords": np.zeros((1, atoms, 3), dtype=np.float32),
        **{
            name: UnusedArray()
            for name in ("single", "pair", "pae_logits", "distogram_logits")
        },
    }

    written = output.write_prediction_outputs(prediction, built, tmp_path, name="same")

    assert len(written["structures"]) == 1


def test_token_profile_is_independent_of_msa_depth_and_lm_chain_overhead() -> None:
    plans = [
        _padding_plan(
            _alignment_rows(_features(), rows),
            PaddingConfig(),
            max_msa_depth=1024,
            language_model_tokens=lm_length,
        )
        for rows, lm_length in ((1, 11), (80, 27))
    ]
    assert plans[0].target == plans[1].target


def test_uncapped_msa_profile_rejects_overflow_without_sampling() -> None:
    with pytest.raises(ValueError):
        _padding_plan(
            _alignment_rows(_features(), 1025),
            PaddingConfig(),
            max_msa_depth=None,
            language_model_tokens=None,
        )


def test_explicit_language_model_capacity_is_preserved() -> None:
    plan = _padding_plan(
        _features(),
        PaddingConfig(language_model_tokens=32),
        max_msa_depth=1024,
        language_model_tokens=11,
    )
    assert plan.target["language_model_tokens"] == 32


def test_lm_token_profile_covers_one_residue_per_chain() -> None:
    built = features.build_features([("A", str(i), i, i) for i in range(9)])
    natural = inference.language_model_length(built)
    assert natural == 27
    plan = _padding_plan(
        built, PaddingConfig(tokens=9),
        max_msa_depth=1024, language_model_tokens=natural,
    )
    assert plan.target["language_model_tokens"] == natural
    assert plan.target["atoms"] == 224


def test_prefix_rng_guards_unsupported_prng_and_backend_falls_back_to_tapes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataclasses
    import json
    from types import SimpleNamespace

    from foldjax.backends.esmfold2 import ESMFold2Backend
    from foldjax.schema import PredictionRequest

    key = jax.random.key(19)
    atom_mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    pair_mask = jnp.ones((1, 2, 2), dtype=jnp.float32)
    tok_mask = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    tiny_settings = structure_model.ModelSettings(
        d_pair=8,
        d_inputs=203,
        inputs_atom_n_blocks=1,
        inputs_atom_n_heads=2,
        trunk_n_layers=1,
        lm_encoder_n_layers=1,
        coda_n_layers=1,
        confidence_n_layers=1,
        msa_n_layers=1,
        num_recycles=1,
        num_samples=2,
        trunk_dtype="float32",
        diffusion=diffusion.DiffusionSettings(
            c_atom=64,
            c_token=32,
            c_z=8,
            atom_n_blocks=1,
            atom_n_heads=2,
            token_n_blocks=1,
            token_n_heads=2,
            num_steps=2,
        ),
    )
    dummy_cache = diffusion.DiffusionCache(
        atom_conditioning=jnp.zeros((2, 4, 64), jnp.float32),
        cos=jnp.ones((2, 4, 16), jnp.bfloat16),
        sin=jnp.zeros((2, 4, 16), jnp.bfloat16),
        pair=jnp.zeros((1, 2, 2, 8), jnp.float32),
        atom_to_token=jnp.zeros((2, 4), jnp.int32),
        atom_mask=jnp.ones((2, 4), jnp.float32),
        n_tokens=2,
    )

    for ctx in (jax.threefry_partitionable(False), jax.default_prng_impl("rbg")):
        with ctx:
            curr_key = jax.random.key(19)
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                diffusion._atom_normal(
                    curr_key, atom_mask, dtype=jnp.float32, preserve_prefix_rng=True
                )
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                diffusion.sample(
                    curr_key,
                    jnp.zeros((1, 2, 203), jnp.float32),
                    dummy_cache,
                    {},
                    settings=tiny_settings.diffusion,
                    num_samples=2,
                    preserve_prefix_rng=True,
                )
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                structure_model._dropout(
                    curr_key,
                    jnp.ones((1, 2, 2, 8), jnp.float32),
                    0.25,
                    valid_mask=pair_mask,
                    preserve_prefix_rng=True,
                )
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                structure_model._initial_pair_state_draw(
                    curr_key, pair_mask, 8, preserve_prefix_rng=True
                )
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                structure_model._msa_column_keep(
                    curr_key, tok_mask, 0.1, preserve_prefix_rng=True
                )
            with pytest.raises(ValueError, match="jax_threefry_partitionable=True"):
                structure_model.predict(
                    curr_key,
                    _features(),
                    {},
                    settings=tiny_settings,
                    preserve_prefix_rng=True,
                )

    # Verify compact-drawn padded_random_tapes match prefix draws on real slices.
    tapes = inference.padded_random_tapes(
        key,
        settings=tiny_settings,
        actual_tokens=2,
        target_tokens=4,
        actual_atoms=32,
        target_atoms=64,
        real_atom_mask=jnp.concatenate(
            [jnp.ones((1, 14), jnp.float32), jnp.zeros((1, 18), jnp.float32)], axis=1
        ),
        msa_depth=3,
        lm_active=True,
    )
    _root, state_k, col_k, loop_k, samp_k = jax.random.split(key, 5)
    std = (2.0 / (5.0 * tiny_settings.d_pair)) ** 0.5
    expected_z = std * jax.random.truncated_normal(
        state_k, -3.0, 3.0, (1, 2, 2, tiny_settings.d_pair), dtype=jnp.float32
    )
    np.testing.assert_array_equal(tapes["initial_pair_state"][:, :2, :2], expected_z)
    np.testing.assert_array_equal(
        tapes["msa_column_keep"][:, :2],
        jax.random.uniform(col_k, (1, 2)) >= tiny_settings.msa_column_mask_rate,
    )
    assert tapes["lm_dropout_masks"].shape == (2, 1, 4, 4, 8)
    _, init_k = jax.random.split(samp_k)
    expected_init = jax.random.normal(init_k, (2, 32, 3), dtype=jnp.float32)
    np.testing.assert_array_equal(
        tapes["diffusion_initial_normal"][:, :14], expected_init[:, :14]
    )
    assert not np.any(np.asarray(tapes["diffusion_initial_normal"][:, 14:]))

    # Backend fallback under non-partitionable PRNG: preserve_prefix_rng=False + tapes.
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            {
                "name": "fallback",
                "entities": [{"type": "protein", "id": "A", "sequence": "AC"}],
            }
        ),
        encoding="utf-8",
    )
    built_ac = features.build_features([("AC", "A", 0, 0)])
    assert int(built_ac["atom_attention_mask"].sum()) == 11
    assert built_ac["atom_attention_mask"].shape == (1, 32)
    churn_settings = dataclasses.replace(
        tiny_settings,
        diffusion=dataclasses.replace(tiny_settings.diffusion, noise_scale=1.003),
    )
    captured_calls: list[dict] = []
    fake_model = SimpleNamespace(
        settings=churn_settings,
        parameters={"token_bonds.weight": jnp.ones((8, 1), jnp.float32)},
        esmc_parameters={"w": jnp.ones((1,))},
        esmc_settings=SimpleNamespace(),
        has_language_model=True,
    )

    monkeypatch.setattr(
        ESMFold2Backend, "_load_model", lambda self, *a, **k: fake_model
    )
    monkeypatch.setattr(
        ESMFold2Backend,
        "_can_stage_language_model",
        lambda self, _inf: False,
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module",
        lambda name: (
            SimpleNamespace(
                jax=jax,
                LANGUAGE_MODEL_FEATURES=inference.LANGUAGE_MODEL_FEATURES,
                COMPACT_LANGUAGE_MODEL_API=True,
                MANAGED_CACHE_LOWERING_API=True,
                PADDED_RANDOM_TAPES_API=True,
                load=lambda *a, **k: fake_model,
                abstract_language_model_embedding=lambda *a, **k: jax.ShapeDtypeStruct(
                    (1, 8, 2560), jnp.float32
                ),
                language_model_embedding=lambda *a, **k: jnp.zeros(
                    (1, 8, 2560), jnp.float32
                ),
                build_job_features=lambda *a, **k: built_ac,
                build_common_job_features=lambda *a, **k: built_ac,
                language_model_states=lambda *a, **k: None,
                language_model_length=inference.language_model_length,
                normalize_msa_features=inference.normalize_msa_features,
                pad_features=inference.pad_features,
                padded_random_tapes=inference.padded_random_tapes,
                seed_key=inference.seed_key,
                lm_mask_key=inference.lm_mask_key,
                predict=lambda _k, _f, _m, **kw: captured_calls.append(kw) or {},
            )
            if name == "foldjax.models.esmfold2.inference"
            else (
                structure_model
                if name == "foldjax.models.esmfold2.models.model"
                else SimpleNamespace(
                    write_prediction_outputs=lambda *a, **k: {
                        "structures": (),
                        "summary": (),
                    }
                )
            )
        ),
    )
    req = PredictionRequest(
        model="esmfold2",
        input=job,
        output_dir=tmp_path / "out",
        weights=tmp_path,
        seed=19,
        input_format="foldjax",
        padding=PaddingConfig(tokens=8, atoms=64, msa=4),
        options={"no_language_model": True},
    )
    backend = ESMFold2Backend()
    exact_atom_mask = jnp.repeat(
        jnp.asarray(built_ac["atom_attention_mask"], dtype=jnp.float32), 2, axis=0
    )
    padded_atom_mask = jnp.pad(exact_atom_mask, ((0, 0), (0, 32)))
    schedule_steps = len(diffusion.noise_schedule(churn_settings.diffusion)) - 1
    monkeypatch.setattr(
        diffusion,
        "diffusion_module",
        lambda x_noisy, _t_hat, s_in, *_a, **_kw: (
            0.5 * x_noisy,
            jnp.zeros((x_noisy.shape[0], s_in.shape[1], 32), jnp.float32),
        ),
    )

    for ctx in (jax.threefry_partitionable(False), jax.default_prng_impl("rbg")):
        captured_calls.clear()
        with ctx:
            backend.predict(req)
            backend.lower_cache_graphs(req)

            exact_root = inference.seed_key(19)
            _r, exact_state_k, _col_k, _loop_k, exact_samp_k = jax.random.split(
                exact_root, 5
            )
            exact_z = std * jax.random.truncated_normal(
                exact_state_k,
                -3.0,
                3.0,
                (1, 2, 2, churn_settings.d_pair),
                dtype=jnp.float32,
            )
            step_samp_k, exact_init_k = jax.random.split(exact_samp_k)
            exact_init = diffusion._atom_normal(
                exact_init_k,
                exact_atom_mask,
                dtype=jnp.float32,
                preserve_prefix_rng=False,
            )
            exact_quats = []
            exact_trans = []
            exact_churns = []
            for _ in range(schedule_steps):
                step_samp_k, aug_k, churn_k = jax.random.split(step_samp_k, 3)
                rot_k, trans_k = jax.random.split(aug_k)
                exact_quats.append(
                    jax.random.normal(rot_k, (2, 4), dtype=jnp.float32)
                )
                exact_trans.append(
                    jax.random.normal(trans_k, (2, 1, 3), dtype=jnp.float32)
                )
                exact_churns.append(
                    diffusion._atom_normal(
                        churn_k,
                        exact_atom_mask,
                        dtype=jnp.float32,
                        preserve_prefix_rng=False,
                    )
                )

            assert captured_calls[0]["preserve_prefix_rng"] is False
            backend_tapes = captured_calls[0]["_padded_random_tapes"]
            np.testing.assert_array_equal(
                backend_tapes["initial_pair_state"][:, :2, :2], exact_z
            )
            np.testing.assert_array_equal(
                backend_tapes["diffusion_initial_normal"][:, :11],
                exact_init[:, :11],
            )
            assert not np.any(
                np.asarray(backend_tapes["diffusion_initial_normal"][:, 11:])
            )
            np.testing.assert_array_equal(
                backend_tapes["diffusion_rotation_quaternions"],
                jnp.stack(exact_quats),
            )
            np.testing.assert_array_equal(
                backend_tapes["diffusion_translations"],
                jnp.stack(exact_trans),
            )
            np.testing.assert_array_equal(
                backend_tapes["diffusion_churn_normals"][:, :, :11],
                jnp.stack(exact_churns)[:, :, :11],
            )
            assert not np.any(
                np.asarray(backend_tapes["diffusion_churn_normals"][:, :, 11:])
            )

            exact_cache = dataclasses.replace(dummy_cache, atom_mask=exact_atom_mask)
            padded_cache = dataclasses.replace(
                dummy_cache, atom_mask=padded_atom_mask, n_tokens=8
            )
            exact_coords, _ = diffusion.sample(
                exact_samp_k,
                jnp.zeros((1, 2, 203), jnp.float32),
                exact_cache,
                {},
                settings=churn_settings.diffusion,
                num_samples=2,
                preserve_prefix_rng=False,
            )
            padded_coords, _ = diffusion.sample(
                exact_samp_k,
                jnp.zeros((1, 8, 203), jnp.float32),
                padded_cache,
                {},
                settings=churn_settings.diffusion,
                num_samples=2,
                preserve_prefix_rng=False,
                diffusion_initial_normal=backend_tapes["diffusion_initial_normal"],
                diffusion_rotation_quaternions=backend_tapes[
                    "diffusion_rotation_quaternions"
                ],
                diffusion_translations=backend_tapes["diffusion_translations"],
                diffusion_churn_normals=backend_tapes["diffusion_churn_normals"],
            )
            np.testing.assert_allclose(
                np.asarray(padded_coords[:, :11]),
                np.asarray(exact_coords[:, :11]),
                rtol=1e-5,
                atol=2e-5,
            )

            assert captured_calls[1]["preserve_prefix_rng"] is False
            assert isinstance(
                captured_calls[1]["_padded_random_tapes"]["initial_pair_state"],
                jax.ShapeDtypeStruct,
            )

    # Edge and failure cases for padded_random_tapes: equal-to-limit, zero,
    # target < actual, boolean counts, and mismatched real_atom_mask shape.
    exact_limit_tapes = inference.padded_random_tapes(
        key,
        settings=tiny_settings,
        actual_tokens=2,
        target_tokens=2,
        actual_atoms=32,
        target_atoms=32,
        real_atom_mask=jnp.asarray(built_ac["atom_attention_mask"], dtype=jnp.float32),
        msa_depth=1,
        lm_active=False,
    )
    assert exact_limit_tapes["initial_pair_state"].shape == (
        1,
        2,
        2,
        tiny_settings.d_pair,
    )
    assert exact_limit_tapes["diffusion_initial_normal"].shape == (2, 32, 3)
    assert "msa_column_keep" not in exact_limit_tapes
    assert "lm_dropout_masks" not in exact_limit_tapes

    import inspect

    assert (
        tuple(inspect.signature(inference._run).parameters)[4:]
        == inference._COMPILED_PREDICT_STATIC_ARGNAMES
    )
    seen_structure_kwargs: list[dict] = []
    monkeypatch.setattr(
        structure_model,
        "predict",
        lambda _k, feat, _p, **kw: (
            seen_structure_kwargs.append({"feat_keys": set(feat), "kw": kw}) or {}
        ),
    )
    inference._run(
        key,
        {"token_attention_mask": tok_mask, "_padded_random_tapes": exact_limit_tapes},
        {},
        None,
        tiny_settings,
        1,
        False,
    )
    assert "_padded_random_tapes" not in seen_structure_kwargs[0]["feat_keys"]
    assert (
        seen_structure_kwargs[0]["kw"]["initial_pair_state"]
        is exact_limit_tapes["initial_pair_state"]
    )

    for bad_tok, tgt_tok, bad_atom, tgt_atom in (
        (0, 2, 32, 32),
        (True, 2, 32, 32),
        (3, 2, 32, 32),
        (2, 2, 0, 32),
        (2, 2, 64, 32),
    ):
        with pytest.raises(ValueError):
            inference.padded_random_tapes(
                key,
                settings=tiny_settings,
                actual_tokens=bad_tok,  # type: ignore[arg-type]
                target_tokens=tgt_tok,
                actual_atoms=bad_atom,
                target_atoms=tgt_atom,
                real_atom_mask=jnp.ones((1, 32), jnp.float32),
                msa_depth=1,
                lm_active=False,
            )
    for bad_mask in (
        jnp.ones((1, 11), jnp.float32),
        jnp.ones((32,), jnp.float32),
        jnp.ones((0, 32), jnp.float32),
    ):
        with pytest.raises(ValueError, match="real_atom_mask"):
            inference.padded_random_tapes(
                key,
                settings=tiny_settings,
                actual_tokens=2,
                target_tokens=4,
                actual_atoms=32,
                target_atoms=64,
                real_atom_mask=bad_mask,
                msa_depth=1,
                lm_active=False,
            )
