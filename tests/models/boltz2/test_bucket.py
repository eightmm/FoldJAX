import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.data.bucket import pad_feats, resolve_bucket_shape


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
    assert resolve_bucket_shape(_features(msa=1)) == (256, 32, 1)
    assert resolve_bucket_shape(_features(msa=77)) == (256, 32, 128)
    assert resolve_bucket_shape(_features(msa=249)) == (256, 32, 256)
    assert resolve_bucket_shape(_features(msa=400)) == (256, 32, 512)
    assert resolve_bucket_shape(_features(msa=900)) == (256, 32, 1024)
    assert resolve_bucket_shape(_features(msa=2000)) == (256, 32, 1024)


def test_pad_feats_preserves_jax_compatible_arrays() -> None:
    padded, _ = pad_feats(_features(), 8, 32, target_msa=128)
    assert isinstance(padded["msa"], jnp.ndarray)
