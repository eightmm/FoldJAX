from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.boltz2.models.heads.affinity as affinity_head
import foldjax.models.boltz2.models.heads.confidence as confidence_head
from foldjax.models.boltz2.data.bucket import pad_feats, select_model_features
from foldjax.models.boltz2.data.ownership import (
    COMPACT_TOKEN_TO_REP_ATOM,
    TOKEN_TO_REP_ATOM_INDEX,
    compact_token_to_rep_atom_storage,
    drop_token_to_rep_atom_storage,
)
from foldjax.models.boltz2.models.diffusion.atom import (
    gather_rep_atoms_to_tokens,
    token_to_rep_atom_index_from_feats,
)
from foldjax.models.boltz2.models.heads.confidence import (
    _nearest_nonpolymer_frames,
)


def _features(dense: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "token_to_rep_atom": dense,
        "token_pad_mask": np.any(dense != 0, axis=-1).astype(np.float32),
        "atom_pad_mask": np.ones((dense.shape[0], dense.shape[2]), dtype=np.float32),
    }


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32])
def test_native_dense_storage_compacts_without_mutating_public_features(dtype) -> None:
    dense = np.zeros((2, 4, 7), dtype=dtype)
    dense[0, 0, 2] = 1
    dense[0, 1, 5] = 1
    dense[1, 0, 6] = 1
    dense[1, 3, 1] = 1
    features = _features(dense)

    compact = compact_token_to_rep_atom_storage(features)

    assert compact is not features
    assert compact[COMPACT_TOKEN_TO_REP_ATOM].shape == ()
    assert compact[COMPACT_TOKEN_TO_REP_ATOM].dtype == np.uint8
    assert int(compact[COMPACT_TOKEN_TO_REP_ATOM]) == 1
    assert compact[TOKEN_TO_REP_ATOM_INDEX].dtype == np.int32
    np.testing.assert_array_equal(
        compact[TOKEN_TO_REP_ATOM_INDEX],
        np.asarray([[2, 5, -1, -1], [6, -1, -1, 1]], dtype=np.int32),
    )
    assert "token_to_rep_atom" not in compact
    assert features["token_to_rep_atom"] is dense
    assert COMPACT_TOKEN_TO_REP_ATOM not in features
    assert TOKEN_TO_REP_ATOM_INDEX not in features


def test_checked_in_production_fixture_compacts_exactly() -> None:
    fixture = Path(__file__).with_name("fixtures") / "1UBQ_A.npz"
    with np.load(fixture) as archive:
        dense = archive["token_to_rep_atom"]
        features = {
            "token_to_rep_atom": dense,
            "token_pad_mask": archive["token_pad_mask"],
            "atom_pad_mask": archive["atom_pad_mask"],
        }

    compact = compact_token_to_rep_atom_storage(features)
    expected = np.where(
        np.any(dense > 0, axis=-1), np.argmax(dense, axis=-1), -1
    ).astype(np.int32)

    np.testing.assert_array_equal(compact[TOKEN_TO_REP_ATOM_INDEX], expected)
    assert int(compact[COMPACT_TOKEN_TO_REP_ATOM]) == 1
    assert dense.nbytes > compact[TOKEN_TO_REP_ATOM_INDEX].nbytes * 1000


def test_custom_dense_layouts_fall_back_by_identity() -> None:
    cases = []
    for value in (np.nan, np.inf, -np.inf, 0.5, -1.0):
        dense = np.zeros((1, 2, 3), dtype=np.float32)
        dense[0, 0, 1] = value
        cases.append(dense)
    negative_zero = np.zeros((1, 2, 3), dtype=np.float32)
    negative_zero[0, 0, 1] = -0.0
    cases.append(negative_zero)
    multi_hot = np.zeros((1, 2, 3), dtype=np.int64)
    multi_hot[0, 0, :2] = 1
    cases.append(multi_hot)

    for dense in cases:
        features = _features(dense)
        assert compact_token_to_rep_atom_storage(features) is features

    empty = _features(np.empty((1, 2, 0), dtype=np.int64))
    assert compact_token_to_rep_atom_storage(empty) is empty
    wrong_shape = _features(np.zeros((1, 2, 3), dtype=np.int64))
    wrong_shape["token_pad_mask"] = np.ones((1, 3), dtype=np.float32)
    assert compact_token_to_rep_atom_storage(wrong_shape) is wrong_shape
    device_dense = _features(np.zeros((1, 2, 3), dtype=np.int64))
    device_dense["token_to_rep_atom"] = jnp.asarray(
        device_dense["token_to_rep_atom"]
    )
    assert compact_token_to_rep_atom_storage(device_dense) is device_dense


def test_dense_authority_strips_or_replaces_stale_private_fields() -> None:
    invalid_dense = np.zeros((1, 2, 3), dtype=np.float32)
    invalid_dense[0, 0, 1] = 0.5
    invalid = {
        **_features(invalid_dense),
        COMPACT_TOKEN_TO_REP_ATOM: np.asarray(1, np.uint8),
        TOKEN_TO_REP_ATOM_INDEX: np.asarray([[2, -1]], np.int32),
    }

    cleaned = compact_token_to_rep_atom_storage(invalid)

    assert cleaned is not invalid
    assert cleaned["token_to_rep_atom"] is invalid_dense
    assert COMPACT_TOKEN_TO_REP_ATOM not in cleaned
    assert TOKEN_TO_REP_ATOM_INDEX not in cleaned

    valid_dense = np.zeros((1, 2, 3), dtype=np.int64)
    valid_dense[0, 0, 1] = 1
    stale = {
        **_features(valid_dense),
        COMPACT_TOKEN_TO_REP_ATOM: np.asarray(9, np.uint8),
        TOKEN_TO_REP_ATOM_INDEX: np.asarray([[2, 2]], np.int32),
    }

    regenerated = compact_token_to_rep_atom_storage(stale)

    assert int(regenerated[COMPACT_TOKEN_TO_REP_ATOM]) == 1
    np.testing.assert_array_equal(
        regenerated[TOKEN_TO_REP_ATOM_INDEX], np.asarray([[1, -1]], np.int32)
    )


def test_trunk_only_drop_removes_incomplete_ownership_without_validation() -> None:
    features = {
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        COMPACT_TOKEN_TO_REP_ATOM: np.asarray(9, dtype=np.uint8),
    }

    dropped = drop_token_to_rep_atom_storage(features)

    assert set(dropped) == {"token_pad_mask"}
    assert dropped["token_pad_mask"] is features["token_pad_mask"]
    assert COMPACT_TOKEN_TO_REP_ATOM in features


def test_private_representation_requires_complete_provenance_pair() -> None:
    base = {
        "token_pad_mask": np.asarray([[1, 0]], dtype=np.float32),
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
    }
    marker = np.asarray(1, dtype=np.uint8)
    payload = np.asarray([[0, -1]], dtype=np.int32)
    valid = {
        **base,
        COMPACT_TOKEN_TO_REP_ATOM: marker,
        TOKEN_TO_REP_ATOM_INDEX: payload,
    }
    assert compact_token_to_rep_atom_storage(valid) is valid

    with pytest.raises(ValueError, match="requires both"):
        compact_token_to_rep_atom_storage(
            {**base, TOKEN_TO_REP_ATOM_INDEX: payload}
        )
    with pytest.raises(ValueError, match="requires both"):
        compact_token_to_rep_atom_storage(
            {**base, COMPACT_TOKEN_TO_REP_ATOM: marker}
        )


@pytest.mark.parametrize(
    "marker",
    [
        np.asarray(1, dtype=np.int32),
        np.asarray([1], dtype=np.uint8),
        np.asarray(2, dtype=np.uint8),
    ],
)
def test_private_representation_rejects_bad_marker(marker: np.ndarray) -> None:
    features = {
        "token_pad_mask": np.asarray([[1, 0]], dtype=np.float32),
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        COMPACT_TOKEN_TO_REP_ATOM: marker,
        TOKEN_TO_REP_ATOM_INDEX: np.asarray([[0, -1]], np.int32),
    }

    with pytest.raises((TypeError, ValueError), match="scalar uint8|version 1"):
        compact_token_to_rep_atom_storage(features)


def test_private_index_contract_rejects_bad_dtype_shape_range_and_sentinel() -> None:
    base = {
        "token_pad_mask": np.asarray([[1, 0]], dtype=np.float32),
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        COMPACT_TOKEN_TO_REP_ATOM: np.asarray(1, dtype=np.uint8),
    }
    with pytest.raises(TypeError, match="dtype int32"):
        compact_token_to_rep_atom_storage(
            {**base, TOKEN_TO_REP_ATOM_INDEX: np.asarray([[0, -1]], np.int64)}
        )
    with pytest.raises(ValueError, match="does not match"):
        compact_token_to_rep_atom_storage(
            {**base, TOKEN_TO_REP_ATOM_INDEX: np.asarray([[0]], np.int32)}
        )
    for malformed in (
        np.asarray([[-2, 0]], np.int32),
        np.asarray([[0, 3]], np.int32),
    ):
        with pytest.raises(ValueError, match="entries must be -1"):
            compact_token_to_rep_atom_storage(
                {**base, TOKEN_TO_REP_ATOM_INDEX: malformed}
            )
    with pytest.raises(ValueError, match="-1 exactly"):
        compact_token_to_rep_atom_storage(
            {**base, TOKEN_TO_REP_ATOM_INDEX: np.asarray([[0, 1]], np.int32)}
        )
    with pytest.raises(ValueError, match="-1 exactly"):
        compact_token_to_rep_atom_storage(
            {**base, TOKEN_TO_REP_ATOM_INDEX: np.asarray([[-1, -1]], np.int32)}
        )
    padded_atom = {
        **base,
        "atom_pad_mask": np.asarray([[1, 0, 0]], dtype=np.float32),
        TOKEN_TO_REP_ATOM_INDEX: np.asarray([[1, -1]], np.int32),
    }
    with pytest.raises(ValueError, match="unpadded atoms"):
        compact_token_to_rep_atom_storage(padded_atom)


def test_compact_storage_padding_uses_negative_sentinel() -> None:
    dense = np.zeros((1, 3, 4), dtype=np.int64)
    dense[0, 0, 1] = 1
    dense[0, 1, 3] = 1
    features = {
        **_features(dense),
        "msa": np.ones((1, 1, 3), dtype=np.int32),
    }
    compact = compact_token_to_rep_atom_storage(select_model_features(features))

    padded, _ = pad_feats(compact, 6, 32, target_msa=1)

    assert "token_to_rep_atom" not in padded
    assert padded[COMPACT_TOKEN_TO_REP_ATOM].shape == ()
    assert int(padded[COMPACT_TOKEN_TO_REP_ATOM]) == 1
    np.testing.assert_array_equal(
        np.asarray(padded[TOKEN_TO_REP_ATOM_INDEX]),
        np.asarray([[1, 3, -1, -1, -1, -1]], dtype=np.int32),
    )


def _assert_float_bits_equal(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(
        actual_array.view(np.uint32), expected_array.view(np.uint32)
    )


def test_compact_gather_is_jit_and_nonfinite_bit_exact() -> None:
    dense = np.zeros((1, 6, 6), dtype=np.float32)
    dense[0, np.arange(5), np.arange(5)] = 1
    atom_values = np.asarray(
        [[[np.nan], [np.inf], [-np.inf], [-0.0], [0.0], [1.25]]],
        dtype=np.float32,
    )
    compact = compact_token_to_rep_atom_storage(_features(dense))
    compact_feats = {
        name: jnp.asarray(value) for name, value in compact.items()
    }

    def dense_graph(mapping, values):
        return gather_rep_atoms_to_tokens(mapping, values)

    def compact_graph(model_features, values):
        return gather_rep_atoms_to_tokens(
            None,
            values,
            index=token_to_rep_atom_index_from_feats(model_features),
        )

    expected = jax.jit(dense_graph)(jnp.asarray(dense), jnp.asarray(atom_values))
    actual = jax.jit(compact_graph)(compact_feats, jnp.asarray(atom_values))
    _assert_float_bits_equal(actual, expected)


def test_dense_feature_takes_precedence_over_private_index() -> None:
    dense = jnp.asarray([[[0.0, 1.0, 0.0]]])
    features = {
        "token_to_rep_atom": dense,
        COMPACT_TOKEN_TO_REP_ATOM: jnp.asarray(9, dtype=jnp.uint8),
        TOKEN_TO_REP_ATOM_INDEX: jnp.asarray([[2]], dtype=jnp.int32),
        "token_pad_mask": jnp.ones((1, 1), dtype=jnp.float32),
        "atom_pad_mask": jnp.ones((1, 3), dtype=jnp.float32),
    }

    index, valid = token_to_rep_atom_index_from_feats(features)

    np.testing.assert_array_equal(np.asarray(index), [[1]])
    np.testing.assert_array_equal(np.asarray(valid), [[True]])


def test_low_level_consumer_rejects_incomplete_and_invalid_private_input() -> None:
    masks = {
        "token_pad_mask": jnp.asarray([[1, 0]], dtype=jnp.float32),
        "atom_pad_mask": jnp.asarray([[1, 1, 1]], dtype=jnp.float32),
    }
    marker = jnp.asarray(1, dtype=jnp.uint8)
    payload = jnp.asarray([[0, -1]], dtype=jnp.int32)

    for incomplete in (
        {**masks, COMPACT_TOKEN_TO_REP_ATOM: marker},
        {**masks, TOKEN_TO_REP_ATOM_INDEX: payload},
    ):
        with pytest.raises(ValueError, match="requires both"):
            token_to_rep_atom_index_from_feats(incomplete)
    with pytest.raises(ValueError, match="version 1"):
        token_to_rep_atom_index_from_feats(
            {
                **masks,
                COMPACT_TOKEN_TO_REP_ATOM: jnp.asarray(2, dtype=jnp.uint8),
                TOKEN_TO_REP_ATOM_INDEX: payload,
            }
        )
    with pytest.raises(ValueError, match="entries must be -1"):
        token_to_rep_atom_index_from_feats(
            {
                **masks,
                COMPACT_TOKEN_TO_REP_ATOM: marker,
                TOKEN_TO_REP_ATOM_INDEX: jnp.asarray([[3, -1]], dtype=jnp.int32),
            }
        )
    with pytest.raises(ValueError, match="-1 exactly"):
        token_to_rep_atom_index_from_feats(
            {
                **masks,
                COMPACT_TOKEN_TO_REP_ATOM: marker,
                TOKEN_TO_REP_ATOM_INDEX: jnp.asarray([[0, 1]], dtype=jnp.int32),
            }
        )


def test_compact_frames_match_dense_under_jit() -> None:
    coords = jnp.asarray(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
           [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]],
        dtype=jnp.float32,
    )
    asym_id_atom = jnp.asarray([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
    atom_pad_mask = jnp.ones((1, 6), dtype=jnp.float32)
    dense = jnp.asarray(
        [[[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]]],
        dtype=jnp.float32,
    )
    index = (jnp.asarray([[1, 4, 0]], jnp.int32), jnp.asarray([[1, 1, 0]], bool))

    expected = jax.jit(_nearest_nonpolymer_frames)(
        coords, asym_id_atom, atom_pad_mask, dense
    )
    actual = jax.jit(
        lambda c, a, m, i, v: _nearest_nonpolymer_frames(
            c, a, m, None, index=(i, v)
        )
    )(coords, asym_id_atom, atom_pad_mask, *index)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_confidence_consumer_accepts_dense_and_compact_trees_exactly(
    monkeypatch
) -> None:
    monkeypatch.setattr(confidence_head, "_layer_norm", lambda value, *_args: value)
    monkeypatch.setattr(
        confidence_head, "_linear", lambda value, kernel: value * kernel
    )
    monkeypatch.setattr(
        confidence_head,
        "relative_position_forward",
        lambda _params, feats, **_kwargs: jnp.zeros(
            (*feats["token_pad_mask"].shape, feats["token_pad_mask"].shape[-1], 1),
            dtype=jnp.float32,
        ),
    )
    monkeypatch.setattr(
        confidence_head,
        "contact_conditioning_forward",
        lambda _params, feats: jnp.zeros(
            (*feats["token_pad_mask"].shape, feats["token_pad_mask"].shape[-1], 1),
            dtype=jnp.float32,
        ),
    )
    monkeypatch.setattr(
        confidence_head,
        "pairformer_module_forward",
        lambda _params, single, pair, **_kwargs: (single, pair),
    )
    monkeypatch.setattr(
        confidence_head,
        "_confidence_heads_forward",
        lambda _params, **kwargs: {
            "distance": kwargs["d"],
            "representative": kwargs["rep_atom_index"][0],
        },
    )

    tokens, atoms = 3, 5
    dense = np.zeros((1, tokens, atoms), dtype=np.int64)
    dense[0, 0, 1] = dense[0, 1, 3] = dense[0, 2, 4] = 1
    common = {
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
        "atom_pad_mask": np.ones((1, atoms), dtype=np.float32),
        "token_bonds": np.zeros((1, tokens, tokens, 1), dtype=np.float32),
        "type_bonds": np.zeros((1, tokens, tokens), dtype=np.int32),
    }
    dense_feats = {**common, "token_to_rep_atom": dense}
    compact_feats = compact_token_to_rep_atom_storage(dense_feats)
    dense_feats = {key: jnp.asarray(value) for key, value in dense_feats.items()}
    compact_feats = {
        key: jnp.asarray(value) for key, value in compact_feats.items()
    }
    scalar = jnp.asarray(0.0, dtype=jnp.float32)
    params = {
        "s_inputs_norm": {"scale": scalar, "bias": scalar},
        "s_norm": {"scale": scalar, "bias": scalar},
        "s_input_to_s": {"kernel": scalar},
        "z_norm": {"scale": scalar, "bias": scalar},
        "rel_pos": {},
        "token_bonds": {"kernel": scalar},
        "token_bonds_type": jnp.zeros((1, 1), dtype=jnp.float32),
        "contact_conditioning": {},
        "s_to_z": {"kernel": scalar},
        "s_to_z_transpose": {"kernel": scalar},
        "s_to_z_prod_in1": {"kernel": scalar},
        "s_to_z_prod_in2": {"kernel": scalar},
        "s_to_z_prod_out": {"kernel": scalar},
        "boundaries": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "dist_bin_pairwise_embed": jnp.arange(3, dtype=jnp.float32)[:, None],
        "pairformer_stack": {},
        "confidence_heads": {},
    }
    single = jnp.zeros((1, tokens, 1), dtype=jnp.float32)
    pair = jnp.zeros((1, tokens, tokens, 1), dtype=jnp.float32)
    coords = jnp.arange(atoms * 3, dtype=jnp.float32).reshape(1, atoms, 3)
    logits = jnp.zeros((1, tokens, tokens, 2), dtype=jnp.float32)

    def run(model_features):
        return confidence_head.confidence_module_forward(
            params,
            single,
            single,
            pair,
            coords,
            model_features,
            logits,
            multiplicity=1,
        )

    dense_out = jax.jit(run)(dense_feats)
    compact_out = jax.jit(run)(compact_feats)
    jax.tree.map(_assert_float_bits_equal, compact_out, dense_out)


def test_affinity_consumer_accepts_dense_and_compact_trees_exactly(
    monkeypatch
) -> None:
    monkeypatch.setattr(affinity_head, "_layer_norm", lambda value, *_args: value)
    monkeypatch.setattr(
        affinity_head, "_linear", lambda value, kernel: value * kernel
    )
    monkeypatch.setattr(
        affinity_head,
        "pairwise_conditioning_forward",
        lambda _params, *, z_trunk, token_rel_pos_feats, eps: token_rel_pos_feats,
    )
    monkeypatch.setattr(
        affinity_head,
        "pairformer_no_seq_module_forward",
        lambda _params, pair, _mask, eps: pair,
    )
    monkeypatch.setattr(
        affinity_head,
        "_affinity_heads_forward",
        lambda _params, pair, _feats, _multiplicity: {"pair": pair},
    )

    tokens, atoms = 3, 5
    dense = np.zeros((1, tokens, atoms), dtype=np.int64)
    dense[0, 0, 1] = dense[0, 1, 3] = dense[0, 2, 4] = 1
    common = {
        "token_pad_mask": np.ones((1, tokens), dtype=np.float32),
        "atom_pad_mask": np.ones((1, atoms), dtype=np.float32),
        "mol_type": np.asarray([[0, 0, 3]], dtype=np.int32),
        "affinity_token_mask": np.asarray([[0, 0, 1]], dtype=np.int32),
    }
    dense_feats = {**common, "token_to_rep_atom": dense}
    compact_feats = compact_token_to_rep_atom_storage(dense_feats)
    dense_feats = {key: jnp.asarray(value) for key, value in dense_feats.items()}
    compact_feats = {
        key: jnp.asarray(value) for key, value in compact_feats.items()
    }
    scalar = jnp.asarray(0.0, dtype=jnp.float32)
    params = {
        "z_norm": {"scale": scalar, "bias": scalar},
        "z_linear": {"kernel": scalar},
        "s_to_z_prod_in1": {"kernel": scalar},
        "s_to_z_prod_in2": {"kernel": scalar},
        "boundaries": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "dist_bin_pairwise_embed": jnp.arange(3, dtype=jnp.float32)[:, None],
        "pairwise_conditioner": {},
        "pairformer_stack": {},
        "affinity_heads": {},
    }
    single = jnp.zeros((1, tokens, 1), dtype=jnp.float32)
    pair = jnp.zeros((1, tokens, tokens, 1), dtype=jnp.float32)
    coords = jnp.arange(atoms * 3, dtype=jnp.float32).reshape(1, atoms, 3)

    def run(model_features):
        return affinity_head.affinity_module_forward(
            params, single, pair, coords, model_features, multiplicity=1
        )

    dense_out = jax.jit(run)(dense_feats)
    compact_out = jax.jit(run)(compact_feats)
    jax.tree.map(_assert_float_bits_equal, compact_out, dense_out)


def test_compact_gather_removes_dense_argument_and_cpu_temporary() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("this is an isolated CPU compiler-memory gate")

    tokens, atoms = 128, 1024
    dense = np.zeros((1, tokens, atoms), dtype=np.float32)
    dense[0, np.arange(tokens), np.arange(tokens) * 7] = 1
    index = np.argmax(dense, axis=-1).astype(np.int32)
    atom_values = np.arange(atoms * 3, dtype=np.float32).reshape(1, atoms, 3)

    dense_lowered = jax.jit(gather_rep_atoms_to_tokens).lower(dense, atom_values)
    compact_lowered = jax.jit(
        lambda ids, values: gather_rep_atoms_to_tokens(
            None, values, index=(jnp.maximum(ids, 0), ids >= 0)
        )
    ).lower(index, atom_values)
    dense_hlo = str(dense_lowered.compiler_ir(dialect="stablehlo"))
    compact_hlo = str(compact_lowered.compiler_ir(dialect="stablehlo"))

    assert f"tensor<1x{tokens}x{atoms}xf32>" in dense_hlo
    assert f"tensor<1x{tokens}x{atoms}xf32>" not in compact_hlo
    dense_executable = dense_lowered.compile()
    compact_executable = compact_lowered.compile()
    dense_memory = dense_executable.memory_analysis()
    compact_memory = compact_executable.memory_analysis()
    assert (
        compact_memory.argument_size_in_bytes * 20
        < dense_memory.argument_size_in_bytes
    )
    assert compact_memory.temp_size_in_bytes < dense_memory.temp_size_in_bytes
    dense_flops = dense_executable.cost_analysis()["flops"]
    compact_flops = compact_executable.cost_analysis()["flops"]
    assert compact_flops * 100 < dense_flops
