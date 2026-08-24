import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.data.bucket import (
    pad_feats,
    resolve_bucket_shape,
    resolve_padding_plan,
    select_model_features,
    select_model_features_for_padding,
)
from foldjax.schema import PaddingConfig


def _features(tokens: int = 3, atoms: int = 4, msa: int = 2):
    return {
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
        "atom_pad_mask": np.ones((1, atoms), dtype=np.float32),
        "ref_pos": np.ones((1, atoms, 3), dtype=np.float32),
        "ref_atom_name_chars": np.ones((1, atoms, 4, 64), dtype=np.float32),
        "res_type": np.ones((1, tokens, 33), dtype=np.float32),
        "token_bonds": np.ones((1, tokens, tokens), dtype=np.float32),
        "atom_to_token": np.ones((1, atoms, tokens), dtype=np.float32),
        "token_to_rep_atom": np.ones((1, tokens, atoms), dtype=np.float32),
        "coords": np.ones((1, 1, atoms, 3), dtype=np.float32),
        "frames_idx": np.ones((1, 1, tokens, 3), dtype=np.int32),
        "msa": np.ones((1, msa, tokens), dtype=np.int32),
        "msa_mask": np.ones((1, msa, tokens), dtype=np.float32),
        "profile": np.ones((1, tokens, 33), dtype=np.float32),
        "ensemble_ref_idxs": np.arange(atoms, dtype=np.int32)[None],
    }


def test_pad_feats_uses_feature_schema_not_matching_dimension_sizes() -> None:
    padded, _ = pad_feats(_features(), 8, 32, target_msa=128)

    assert padded["ref_pos"].shape == (1, 32, 3)
    assert padded["ref_atom_name_chars"].shape == (1, 32, 4, 64)
    assert padded["res_type"].shape == (1, 8, 33)
    assert padded["token_bonds"].shape == (1, 8, 8)
    assert padded["atom_to_token"].shape == (1, 32, 8)
    assert padded["token_to_rep_atom"].shape == (1, 8, 32)
    assert padded["coords"].shape == (1, 1, 32, 3)
    assert padded["frames_idx"].shape == (1, 1, 8, 3)
    assert padded["msa"].shape == (1, 128, 8)
    assert padded["msa_mask"].shape == (1, 128, 8)
    assert padded["profile"].shape == (1, 8, 33)
    assert padded["ensemble_ref_idxs"].shape == (1, 4)

    np.testing.assert_array_equal(np.asarray(padded["ref_pos"][:, :4]), 1)
    np.testing.assert_array_equal(np.asarray(padded["ref_pos"][:, 4:]), 0)
    np.testing.assert_array_equal(np.asarray(padded["msa_mask"][:, 2:]), 0)


def test_pad_feats_handles_equal_token_and_atom_counts() -> None:
    padded, _ = pad_feats(_features(tokens=4, atoms=4), 8, 32, target_msa=128)

    assert padded["atom_to_token"].shape == (1, 32, 8)
    assert padded["token_to_rep_atom"].shape == (1, 8, 32)


def test_pad_feats_truncates_msa_before_padding() -> None:
    feats = _features(tokens=6, atoms=8, msa=1100)
    padded, _ = pad_feats(feats, 256, 32, target_msa=1024)

    assert padded["msa"].shape == (1, 1024, 256)
    assert padded["msa_mask"].shape == (1, 1024, 256)
    np.testing.assert_array_equal(np.asarray(padded["msa"][:, :, :6]), 1)


def test_resolve_bucket_shape_normalizes_msa_without_overpadding_shallow_inputs(
) -> None:
    assert resolve_bucket_shape(_features(msa=1)) == (256, 256, 1)
    assert resolve_bucket_shape(_features(msa=77)) == (256, 256, 128)
    assert resolve_bucket_shape(_features(msa=249)) == (256, 256, 256)
    assert resolve_bucket_shape(_features(msa=400)) == (256, 256, 512)
    assert resolve_bucket_shape(_features(msa=900)) == (256, 256, 1024)
    assert resolve_bucket_shape(_features(msa=2000)) == (256, 256, 1024)


def test_neutral_padding_resolves_all_three_compile_shape_axes() -> None:
    plan = resolve_padding_plan(_features(msa=2), PaddingConfig())

    assert plan.actual == {"tokens": 3, "atoms": 4, "msa": 2}
    assert plan.storage == {"tokens": 3, "atoms": 4, "msa": 2}
    assert plan.target == {"tokens": 256, "atoms": 256, "msa": 64}


def test_neutral_padding_honours_exact_axis_targets() -> None:
    plan = resolve_padding_plan(
        _features(msa=2),
        PaddingConfig(tokens=512, atoms=1024, msa=128),
    )

    assert plan.target == {"tokens": 512, "atoms": 1024, "msa": 128}


def test_neutral_padding_rejects_unaligned_exact_atom_target() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        resolve_padding_plan(
            _features(), PaddingConfig(tokens=8, atoms=33, msa=2)
        )


def test_neutral_padding_never_shrinks_materialized_features() -> None:
    with pytest.raises(ValueError, match="smaller than the input size"):
        resolve_padding_plan(
            _features(tokens=8, atoms=32, msa=4),
            PaddingConfig(tokens=4),
        )


def test_atom_buckets_reuse_shapes_beyond_the_featurizers_32_alignment() -> None:
    first = resolve_bucket_shape(_features(atoms=33))
    second = resolve_bucket_shape(_features(atoms=200))

    assert first[1] == second[1] == 256


def test_public_crop_requires_real_mask_entries_to_be_a_prefix() -> None:
    feats = _features(tokens=4)
    feats["token_pad_mask"] = np.asarray([[1, 0, 1, 0]], dtype=np.float32)

    with pytest.raises(ValueError, match="contiguous prefix"):
        resolve_padding_plan(feats, PaddingConfig())


def test_neutral_padding_rejects_unprofiled_template_rows() -> None:
    feats = _features()
    feats["template_mask"] = np.zeros((1, 2, 3), dtype=np.float32)
    feats["visibility_ids"] = np.zeros((1, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="exactly one template row"):
        select_model_features_for_padding(feats, steering_active=False)


def test_neutral_padding_rejects_active_variable_length_steering() -> None:
    with pytest.raises(ValueError, match="does not support active steering"):
        select_model_features_for_padding(_features(), steering_active=True)


def test_neutral_padding_drops_features_unused_by_the_jitted_graph() -> None:
    feats = _features()
    feats["host_only_variable_archive"] = np.zeros((1, 37), dtype=np.float32)
    feats["r_set_to_rep_atom"] = np.zeros((1, 11, 4), dtype=np.float32)
    feats["token_to_rep_atom"] = np.zeros((1, 3, 4), dtype=np.float32)

    selected = select_model_features_for_padding(feats, steering_active=False)

    assert "host_only_variable_archive" not in selected
    assert "r_set_to_rep_atom" not in selected
    assert "token_to_rep_atom" in selected


def test_generic_model_feature_filter_drops_training_and_writer_arrays() -> None:
    feats = _features()
    feats["disto_target"] = np.full((1, 3, 3, 1, 64), np.nan, dtype=np.float32)
    feats["host_only_variable_archive"] = np.full(
        (1, 37), np.inf, dtype=np.float32
    )

    selected = select_model_features(feats)

    assert "disto_target" not in selected
    assert "coords" not in selected
    assert "ensemble_ref_idxs" not in selected
    assert "host_only_variable_archive" not in selected
    assert selected["token_pad_mask"] is feats["token_pad_mask"]
    assert selected["atom_pad_mask"] is feats["atom_pad_mask"]


def test_filtering_dead_nonfinite_features_does_not_change_the_executable() -> None:
    feats = {
        "token_pad_mask": np.asarray(
            [[1.0, -0.0, np.nan, np.inf, -np.inf]], dtype=np.float32
        ),
        "atom_pad_mask": np.ones((1, 4), dtype=np.float32),
        "disto_target": np.asarray([[[[[np.nan, np.inf, -np.inf]]]]]),
        "writer_only": np.asarray([np.nan, np.inf], dtype=np.float32),
    }
    filtered = select_model_features(feats)

    def graph(model_feats):
        return model_feats["token_pad_mask"] + jnp.sum(
            model_feats["atom_pad_mask"]
        )

    raw_lowered = jax.jit(graph).lower(feats)
    filtered_lowered = jax.jit(graph).lower(filtered)
    assert (
        raw_lowered.compiler_ir(dialect="hlo").as_hlo_text()
        == filtered_lowered.compiler_ir(dialect="hlo").as_hlo_text()
    )

    raw_executable = raw_lowered.compile()
    filtered_executable = filtered_lowered.compile()
    memory_fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    raw_memory = raw_executable.memory_analysis()
    filtered_memory = filtered_executable.memory_analysis()
    assert tuple(getattr(raw_memory, name) for name in memory_fields) == tuple(
        getattr(filtered_memory, name) for name in memory_fields
    )

    raw = np.asarray(raw_executable(feats))
    selected = np.asarray(filtered_executable(filtered))
    np.testing.assert_array_equal(raw, selected)
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(selected))
    np.testing.assert_array_equal(np.isposinf(raw), np.isposinf(selected))
    np.testing.assert_array_equal(np.isneginf(raw), np.isneginf(selected))
    np.testing.assert_array_equal(np.signbit(raw), np.signbit(selected))


def test_pad_feats_preserves_jax_compatible_arrays() -> None:
    padded, _ = pad_feats(_features(), 8, 32, target_msa=128)
    assert isinstance(padded["msa"], jnp.ndarray)
