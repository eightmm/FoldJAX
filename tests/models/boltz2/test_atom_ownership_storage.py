from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.data.bucket import pad_feats, select_model_features
from foldjax.models.boltz2.data.ownership import (
    ATOM_TO_TOKEN_INDEX,
    COMPACT_ATOM_TO_TOKEN,
    compact_atom_to_token_storage,
)
from foldjax.models.boltz2.models.diffusion.atom import (
    atom_to_token_index_from_feats,
    gather_token_pairs_to_atom_windows,
    gather_token_pairs_to_atom_windows_indexed,
    gather_tokens_to_atoms,
    get_indexing_matrix,
    scatter_atoms_to_tokens_mean,
    single_to_keys,
)
from foldjax.models.boltz2.models.heads.confidence import (
    _compute_frame_pred_inference,
)


def _features(
    dense: np.ndarray,
    *,
    token_pad_mask: np.ndarray | None = None,
    atom_pad_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    batch, atoms, tokens = dense.shape
    if token_pad_mask is None:
        token_pad_mask = np.ones((batch, tokens), dtype=np.float32)
    if atom_pad_mask is None:
        atom_pad_mask = np.any(dense != 0, axis=-1).astype(np.float32)
    return {
        "atom_to_token": dense,
        "token_pad_mask": token_pad_mask,
        "atom_pad_mask": atom_pad_mask,
    }


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32])
def test_native_atom_ownership_compacts_without_mutating_public_features(
    dtype,
) -> None:
    dense = np.zeros((2, 6, 4), dtype=dtype)
    dense[0, 0, 0] = dense[0, 1, 2] = dense[0, 2, 2] = dense[0, 3, 1] = 1
    dense[1, 0, 1] = dense[1, 1, 1] = dense[1, 2, 0] = 1
    token_mask = np.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
    features = _features(dense, token_pad_mask=token_mask)

    compact = compact_atom_to_token_storage(features)

    assert compact is not features
    assert "atom_to_token" not in compact
    assert compact[COMPACT_ATOM_TO_TOKEN].shape == ()
    assert compact[COMPACT_ATOM_TO_TOKEN].dtype == np.uint8
    assert int(compact[COMPACT_ATOM_TO_TOKEN]) == 1
    assert compact[ATOM_TO_TOKEN_INDEX].dtype == np.int32
    np.testing.assert_array_equal(
        compact[ATOM_TO_TOKEN_INDEX],
        np.asarray(
            [[0, 2, 2, 1, -1, -1], [1, 1, 0, -1, -1, -1]],
            dtype=np.int32,
        ),
    )
    assert features["atom_to_token"] is dense
    assert COMPACT_ATOM_TO_TOKEN not in features
    assert ATOM_TO_TOKEN_INDEX not in features


def test_checked_in_production_fixture_atom_ownership_compacts_exactly() -> None:
    fixture = Path(__file__).with_name("fixtures") / "1UBQ_A.npz"
    with np.load(fixture) as archive:
        dense = archive["atom_to_token"]
        features = {
            "atom_to_token": dense,
            "token_pad_mask": archive["token_pad_mask"],
            "atom_pad_mask": archive["atom_pad_mask"],
        }

    compact = compact_atom_to_token_storage(features)
    expected = np.where(
        np.any(dense > 0, axis=-1), np.argmax(dense, axis=-1), -1
    ).astype(np.int32)

    np.testing.assert_array_equal(compact[ATOM_TO_TOKEN_INDEX], expected)
    assert int(compact[COMPACT_ATOM_TO_TOKEN]) == 1
    assert dense.nbytes > compact[ATOM_TO_TOKEN_INDEX].nbytes * 100


def test_custom_atom_ownership_layouts_fall_back_by_identity() -> None:
    cases = []
    for value in (np.nan, np.inf, -np.inf, 0.5, -1.0):
        dense = np.zeros((1, 2, 3), dtype=np.float32)
        dense[0, 0, 1] = value
        cases.append(_features(dense))
    negative_zero = np.zeros((1, 2, 3), dtype=np.float32)
    negative_zero[0, 0, 1] = -0.0
    cases.append(_features(negative_zero))
    multi_hot = np.zeros((1, 2, 3), dtype=np.int64)
    multi_hot[0, 0, :2] = 1
    cases.append(_features(multi_hot))
    ownerless_real_atom = np.zeros((1, 2, 3), dtype=np.int64)
    ownerless_real_atom[0, 0, 0] = 1
    cases.append(
        _features(
            ownerless_real_atom,
            atom_pad_mask=np.ones((1, 2), dtype=np.float32),
        )
    )
    padded_token_owner = np.zeros((1, 2, 3), dtype=np.int64)
    padded_token_owner[0, 0, 2] = 1
    cases.append(
        _features(
            padded_token_owner,
            token_pad_mask=np.asarray([[1, 1, 0]], dtype=np.float32),
        )
    )

    for features in cases:
        assert compact_atom_to_token_storage(features) is features

    empty = _features(np.empty((1, 2, 0), dtype=np.int64))
    assert compact_atom_to_token_storage(empty) is empty
    no_atoms = _features(np.empty((1, 0, 2), dtype=np.int64))
    assert compact_atom_to_token_storage(no_atoms) is no_atoms
    wrong_shape = _features(np.zeros((1, 2, 3), dtype=np.int64))
    wrong_shape["atom_pad_mask"] = np.ones((1, 3), dtype=np.float32)
    assert compact_atom_to_token_storage(wrong_shape) is wrong_shape
    device_dense = _features(np.zeros((1, 2, 3), dtype=np.int64))
    device_dense["atom_to_token"] = jnp.asarray(device_dense["atom_to_token"])
    assert compact_atom_to_token_storage(device_dense) is device_dense


def test_dense_atom_ownership_is_authoritative_over_stale_private_fields() -> None:
    dense = np.zeros((1, 3, 2), dtype=np.int64)
    dense[0, 0, 1] = dense[0, 1, 0] = 1
    stale = {
        **_features(dense),
        COMPACT_ATOM_TO_TOKEN: np.asarray(9, np.uint8),
        ATOM_TO_TOKEN_INDEX: np.asarray([[0, 0, 0]], np.int32),
        "atom_to_token_ids_global": np.asarray([[1, 1, 1]], np.int32),
        "atom_to_token_valid": np.ones((1, 3), dtype=bool),
    }

    regenerated = compact_atom_to_token_storage(stale)

    np.testing.assert_array_equal(
        regenerated[ATOM_TO_TOKEN_INDEX], np.asarray([[1, 0, -1]], np.int32)
    )
    assert int(regenerated[COMPACT_ATOM_TO_TOKEN]) == 1
    assert "atom_to_token_ids_global" not in regenerated
    assert "atom_to_token_valid" not in regenerated

    invalid_dense = dense.astype(np.float32)
    invalid_dense[0, 0, 1] = 0.5
    invalid = {**stale, "atom_to_token": invalid_dense}
    cleaned = compact_atom_to_token_storage(invalid)
    assert cleaned["atom_to_token"] is invalid_dense
    assert COMPACT_ATOM_TO_TOKEN not in cleaned
    assert ATOM_TO_TOKEN_INDEX not in cleaned
    assert "atom_to_token_ids_global" not in cleaned
    assert "atom_to_token_valid" not in cleaned


def test_private_atom_ownership_requires_complete_validated_pair() -> None:
    base = {
        "token_pad_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "atom_pad_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
    }
    marker = np.asarray(1, dtype=np.uint8)
    payload = np.asarray([[0, 1, -1]], dtype=np.int32)
    valid = {
        **base,
        COMPACT_ATOM_TO_TOKEN: marker,
        ATOM_TO_TOKEN_INDEX: payload,
    }
    assert compact_atom_to_token_storage(valid) is valid
    with_legacy = {
        **valid,
        "atom_to_token_ids_global": payload,
        "atom_to_token_valid": payload >= 0,
    }
    cleaned = compact_atom_to_token_storage(with_legacy)
    assert "atom_to_token_ids_global" not in cleaned
    assert "atom_to_token_valid" not in cleaned
    with pytest.raises(ValueError, match="legacy CP"):
        compact_atom_to_token_storage(
            {
                **base,
                "atom_to_token_ids_global": payload,
                "atom_to_token_valid": payload >= 0,
            }
        )

    for incomplete in (
        {**base, COMPACT_ATOM_TO_TOKEN: marker},
        {**base, ATOM_TO_TOKEN_INDEX: payload},
    ):
        with pytest.raises(ValueError, match="requires both"):
            compact_atom_to_token_storage(incomplete)

    for bad_marker in (
        np.asarray(1, dtype=np.int32),
        np.asarray([1], dtype=np.uint8),
        np.asarray(2, dtype=np.uint8),
    ):
        with pytest.raises((TypeError, ValueError), match="scalar uint8|version 1"):
            compact_atom_to_token_storage(
                {
                    **base,
                    COMPACT_ATOM_TO_TOKEN: bad_marker,
                    ATOM_TO_TOKEN_INDEX: payload,
                }
            )

    malformed = (
        np.asarray([[0, 1, -1]], dtype=np.int64),
        np.asarray([[0, -1]], dtype=np.int32),
        np.asarray([[-2, 1, -1]], dtype=np.int32),
        np.asarray([[0, 3, -1]], dtype=np.int32),
        np.asarray([[0, -1, -1]], dtype=np.int32),
        np.asarray([[0, 1, 0]], dtype=np.int32),
        np.asarray([[0, 2, -1]], dtype=np.int32),
    )
    for payload_value in malformed:
        with pytest.raises((TypeError, ValueError)):
            compact_atom_to_token_storage(
                {
                    **base,
                    COMPACT_ATOM_TO_TOKEN: marker,
                    ATOM_TO_TOKEN_INDEX: payload_value,
                }
            )


def test_compact_atom_ownership_padding_uses_negative_sentinel() -> None:
    dense = np.zeros((1, 3, 2), dtype=np.int64)
    dense[0, 0, 0] = dense[0, 1, 1] = dense[0, 2, 0] = 1
    compact = compact_atom_to_token_storage(select_model_features(_features(dense)))

    padded, _ = pad_feats(compact, 6, 32, target_msa=1)

    assert "atom_to_token" not in padded
    assert padded[COMPACT_ATOM_TO_TOKEN].shape == ()
    assert int(padded[COMPACT_ATOM_TO_TOKEN]) == 1
    np.testing.assert_array_equal(
        np.asarray(padded[ATOM_TO_TOKEN_INDEX]),
        np.asarray([[0, 1, 0, *([-1] * 29)]], dtype=np.int32),
    )


def _assert_array_bits_equal(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.tobytes() == expected_array.tobytes()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_compact_routing_is_jit_bit_exact_with_padding_and_nonfinite(dtype) -> None:
    tokens, atoms, channels = 4, 8, 4
    owner = np.asarray([[0, 1, 1, 2, 3, 0, -1, -1]], dtype=np.int32)
    atom_valid = owner >= 0
    dense = np.zeros((1, atoms, tokens), dtype=np.float32)
    dense[0, np.flatnonzero(atom_valid[0]), owner[0, atom_valid[0]]] = 1
    features = _features(
        dense,
        atom_pad_mask=atom_valid.astype(np.float32),
    )
    compact = compact_atom_to_token_storage(features)
    compact_jax = {name: jnp.asarray(value) for name, value in compact.items()}
    index = atom_to_token_index_from_feats(compact_jax)

    token_values = np.arange(tokens * channels, dtype=np.float32).reshape(
        1, tokens, channels
    )
    token_values[0, 0] = [np.nan, np.inf, -np.inf, -0.0]
    atom_values = np.arange(atoms * channels, dtype=np.float32).reshape(
        1, atoms, channels
    )
    atom_values[0, 6] = [np.nan, np.inf, -np.inf, -0.0]
    token_values = jnp.asarray(token_values, dtype=dtype)
    atom_values = jnp.asarray(atom_values, dtype=dtype)
    dense_jax = jnp.asarray(dense)

    dense_gather = jax.jit(gather_tokens_to_atoms)(dense_jax, token_values)
    compact_gather = jax.jit(
        lambda values, ids, valid: gather_tokens_to_atoms(
            None, values, index=(ids, valid)
        )
    )(token_values, *index)
    dense_scatter = jax.jit(scatter_atoms_to_tokens_mean)(dense_jax, atom_values)
    compact_scatter = jax.jit(
        lambda values, ids, valid: scatter_atoms_to_tokens_mean(
            None,
            values,
            index=(ids, valid),
            num_tokens=tokens,
        )
    )(atom_values, *index)

    w, h_keys = 4, 4
    indexing = get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)
    dense_queries = dense_jax.reshape(1, atoms // w, w, tokens)
    dense_keys = single_to_keys(dense_jax, indexing, w=w, h_keys=h_keys)
    query_index = (
        index[0].reshape(1, atoms // w, w),
        index[1].reshape(1, atoms // w, w),
    )
    key_index = (
        jnp.squeeze(single_to_keys(index[0][..., None], indexing, w, h_keys), -1),
        jnp.squeeze(
            single_to_keys(
                index[1][..., None].astype(jnp.float32), indexing, w, h_keys
            ),
            -1,
        ).astype(bool),
    )
    pair_values = np.arange(tokens * tokens * channels, dtype=np.float32).reshape(
        1, tokens, tokens, channels
    )
    pair_values[0, 0, 0] = [np.nan, np.inf, -np.inf, -0.0]
    pair_values = jnp.asarray(pair_values, dtype=dtype)
    dense_pair = jax.jit(gather_token_pairs_to_atom_windows)(
        pair_values, dense_queries, dense_keys
    )
    compact_pair = jax.jit(gather_token_pairs_to_atom_windows_indexed)(
        pair_values, query_index, key_index
    )

    _assert_array_bits_equal(compact_gather, dense_gather)
    _assert_array_bits_equal(compact_scatter, dense_scatter)
    _assert_array_bits_equal(compact_pair, dense_pair)


def test_dense_low_level_atom_ownership_takes_precedence() -> None:
    dense = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]])
    features = {
        "atom_to_token": dense,
        COMPACT_ATOM_TO_TOKEN: jnp.asarray(9, dtype=jnp.uint8),
        ATOM_TO_TOKEN_INDEX: jnp.asarray([[0, 1]], dtype=jnp.int32),
        "token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "atom_pad_mask": jnp.ones((1, 2), dtype=jnp.float32),
    }

    index, valid = atom_to_token_index_from_feats(features)

    np.testing.assert_array_equal(np.asarray(index), [[1, 0]])
    np.testing.assert_array_equal(np.asarray(valid), [[True, True]])


def test_indexed_pair_windows_preserve_custom_dense_routing_bits() -> None:
    atoms, tokens, channels = 8, 4, 3
    w, h_keys = 4, 4
    rng = np.random.default_rng(20260829)
    dense = rng.normal(size=(1, atoms, tokens)).astype(np.float32)
    dense[0, 0] = [np.nan, np.inf, -np.inf, -0.0]
    dense[0, 1] = [-0.0, 0.0, -0.0, 0.0]
    dense[0, 2] = [1.0, 1.0, 0.0, 0.0]
    pair = rng.normal(size=(1, tokens, tokens, channels)).astype(np.float32)
    pair[0, 0, 0] = [np.nan, np.inf, -0.0]
    indexing = get_indexing_matrix(atoms // w, w, h_keys)

    def dense_route(mapping, values):
        queries = mapping.reshape(1, atoms // w, w, tokens)
        keys = single_to_keys(mapping, indexing, w, h_keys)
        return gather_token_pairs_to_atom_windows(values, queries, keys)

    def indexed_route(mapping, values):
        index = atom_to_token_index_from_feats({"atom_to_token": mapping})
        queries = (
            index[0].reshape(1, atoms // w, w),
            index[1].reshape(1, atoms // w, w),
        )
        keys = (
            jnp.squeeze(single_to_keys(index[0][..., None], indexing, w, h_keys), -1),
            jnp.squeeze(
                single_to_keys(
                    index[1][..., None].astype(jnp.float32), indexing, w, h_keys
                ),
                -1,
            ).astype(bool),
        )
        return gather_token_pairs_to_atom_windows_indexed(values, queries, keys)

    expected = jax.jit(dense_route)(jnp.asarray(dense), jnp.asarray(pair))
    actual = jax.jit(indexed_route)(jnp.asarray(dense), jnp.asarray(pair))

    _assert_array_bits_equal(actual, expected)


def test_low_level_atom_owner_rejects_incomplete_or_invalid_private_input() -> None:
    masks = {
        "token_pad_mask": jnp.asarray([[1, 1, 0]], dtype=jnp.float32),
        "atom_pad_mask": jnp.asarray([[1, 1, 0]], dtype=jnp.float32),
    }
    marker = jnp.asarray(1, dtype=jnp.uint8)
    payload = jnp.asarray([[0, 1, -1]], dtype=jnp.int32)
    for incomplete in (
        {**masks, COMPACT_ATOM_TO_TOKEN: marker},
        {**masks, ATOM_TO_TOKEN_INDEX: payload},
    ):
        with pytest.raises(ValueError, match="requires both"):
            atom_to_token_index_from_feats(incomplete)
    for invalid in (
        {
            **masks,
            COMPACT_ATOM_TO_TOKEN: jnp.asarray(2, dtype=jnp.uint8),
            ATOM_TO_TOKEN_INDEX: payload,
        },
        {
            **masks,
            COMPACT_ATOM_TO_TOKEN: marker,
            ATOM_TO_TOKEN_INDEX: jnp.asarray([[0, 3, -1]], dtype=jnp.int32),
        },
        {
            **masks,
            COMPACT_ATOM_TO_TOKEN: marker,
            ATOM_TO_TOKEN_INDEX: jnp.asarray([[0, 1, 0]], dtype=jnp.int32),
        },
    ):
        with pytest.raises(ValueError):
            atom_to_token_index_from_feats(invalid)


def test_confidence_frames_accept_dense_and_compact_atom_ownership_exactly() -> None:
    tokens, atoms = 2, 8
    owners = np.asarray([0, 0, 0, 0, 1, 1, 1, -1], dtype=np.int32)
    dense = np.zeros((1, atoms, tokens), dtype=np.int64)
    dense[0, np.arange(atoms - 1), owners[:-1]] = 1
    atom_mask = np.asarray([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)
    token_to_rep = np.zeros((1, tokens, atoms), dtype=np.int64)
    token_to_rep[0, 0, 0] = token_to_rep[0, 1, 4] = 1
    common = {
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
        "atom_pad_mask": atom_mask,
        "asym_id": np.asarray([[0, 1]], dtype=np.int32),
        "mol_type": np.full((1, tokens), 3, dtype=np.int32),
        "token_to_rep_atom": token_to_rep,
    }
    dense_feats = {**common, "atom_to_token": dense}
    compact_feats = compact_atom_to_token_storage(dense_feats)
    dense_feats = {name: jnp.asarray(value) for name, value in dense_feats.items()}
    compact_feats = {name: jnp.asarray(value) for name, value in compact_feats.items()}
    rng = np.random.default_rng(20260828)
    coords = jnp.asarray(rng.normal(size=(1, atoms, 3)), dtype=jnp.float32)
    frames = jnp.asarray([[[0, 1, 2], [4, 5, 6]]], dtype=jnp.int32)

    run = jax.jit(
        lambda values, model_features: _compute_frame_pred_inference(
            values,
            frames,
            model_features,
            multiplicity=1,
        )
    )
    expected = run(coords, dense_feats)
    actual = run(coords, compact_feats)

    _assert_array_bits_equal(actual, expected)


def test_compact_routing_removes_dense_jit_argument_and_cpu_temporary() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("this is an isolated CPU compiler-memory gate")

    tokens, atoms, channels = 128, 1024, 8
    w, h_keys = 32, 128
    ids = (np.arange(atoms, dtype=np.int32) % tokens)[None]
    dense = np.eye(tokens, dtype=np.float32)[ids]
    token_values = np.arange(tokens * channels, dtype=np.float32).reshape(
        1, tokens, channels
    )
    atom_values = np.arange(atoms * channels, dtype=np.float32).reshape(
        1, atoms, channels
    )
    pair_values = np.arange(tokens * tokens * channels, dtype=np.float32).reshape(
        1, tokens, tokens, channels
    )
    indexing = get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)

    def dense_graph(mapping, single, atom, pair):
        queries = mapping.reshape(1, atoms // w, w, tokens)
        keys = single_to_keys(mapping, indexing, w, h_keys)
        return (
            gather_tokens_to_atoms(mapping, single),
            scatter_atoms_to_tokens_mean(mapping, atom),
            gather_token_pairs_to_atom_windows(pair, queries, keys),
        )

    def compact_graph(owner_ids, single, atom, pair):
        index = (jnp.maximum(owner_ids, 0), owner_ids >= 0)
        queries = (
            index[0].reshape(1, atoms // w, w),
            index[1].reshape(1, atoms // w, w),
        )
        keys = (
            jnp.squeeze(single_to_keys(index[0][..., None], indexing, w, h_keys), -1),
            jnp.squeeze(
                single_to_keys(
                    index[1][..., None].astype(jnp.float32), indexing, w, h_keys
                ),
                -1,
            ).astype(bool),
        )
        return (
            gather_tokens_to_atoms(None, single, index=index),
            scatter_atoms_to_tokens_mean(None, atom, index=index, num_tokens=tokens),
            gather_token_pairs_to_atom_windows_indexed(pair, queries, keys),
        )

    dense_lowered = jax.jit(dense_graph).lower(
        dense, token_values, atom_values, pair_values
    )
    compact_lowered = jax.jit(compact_graph).lower(
        ids, token_values, atom_values, pair_values
    )
    dense_hlo = str(dense_lowered.compiler_ir(dialect="stablehlo"))
    compact_hlo = str(compact_lowered.compiler_ir(dialect="stablehlo"))
    assert f"tensor<1x{atoms}x{tokens}xf32>" in dense_hlo
    assert f"tensor<1x{atoms}x{tokens}xf32>" not in compact_hlo

    dense_executable = dense_lowered.compile()
    compact_executable = compact_lowered.compile()
    dense_memory = dense_executable.memory_analysis()
    compact_memory = compact_executable.memory_analysis()
    assert compact_memory.argument_size_in_bytes < dense_memory.argument_size_in_bytes
    assert compact_memory.temp_size_in_bytes < dense_memory.temp_size_in_bytes
    compact_flops = compact_executable.cost_analysis()["flops"]
    dense_flops = dense_executable.cost_analysis()["flops"]
    assert compact_flops < dense_flops
