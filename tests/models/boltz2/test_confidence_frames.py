import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.diffusion.atom import gather_tokens_to_atoms
from foldjax.models.boltz2.models.heads.confidence import (
    _compute_collinear_mask,
    _compute_frame_pred_inference,
    _nearest_nonpolymer_frames,
)


def _legacy_nearest_nonpolymer_frames(
    pred_atom_coords: jnp.ndarray,
    asym_id_atom: jnp.ndarray,
    atom_pad_mask: jnp.ndarray,
    token_to_rep_atom: jnp.ndarray,
) -> jnp.ndarray:
    """The dense atom-by-atom implementation replaced by the compact path."""

    delta = pred_atom_coords[:, :, :, None, :] - pred_atom_coords[:, :, None, :, :]
    dist_mat = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 0.0))
    same_chain = asym_id_atom[:, :, None] == asym_id_atom[:, None, :]
    valid_atom_pair = (
        same_chain
        & atom_pad_mask[:, :, None].astype(bool)
        & atom_pad_mask[:, None, :].astype(bool)
    )
    dist_mat = jnp.where(valid_atom_pair[:, None, :, :], dist_mat, jnp.inf)
    nearest = jnp.argsort(dist_mat, axis=-1)[..., :3]
    nearest = nearest[..., jnp.array([1, 0, 2])]

    rep_atom_idx = jnp.argmax(token_to_rep_atom, axis=-1)
    gather_idx = jnp.broadcast_to(
        rep_atom_idx[:, None, :, None],
        (
            pred_atom_coords.shape[0],
            pred_atom_coords.shape[1],
            rep_atom_idx.shape[1],
            3,
        ),
    )
    return jnp.take_along_axis(nearest, gather_idx, axis=2)


def _legacy_compute_frame_pred_inference(
    pred_atom_coords: jnp.ndarray,
    frames_idx_true: jnp.ndarray,
    feats: dict[str, jnp.ndarray],
    multiplicity: int,
    recompute_nonpolymer_frames: bool = True,
) -> jnp.ndarray:
    asym_id_token = feats["asym_id"]
    asym_id_atom = gather_tokens_to_atoms(
        feats["atom_to_token"].astype(jnp.float32),
        asym_id_token[..., None].astype(jnp.float32),
    )[..., 0]

    b, _, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(
        b // multiplicity, multiplicity, -1, 3
    )
    frames_idx_pred = jnp.reshape(
        jnp.repeat(frames_idx_true, multiplicity, axis=0),
        (b // multiplicity, multiplicity, -1, 3),
    )

    mol_type = feats["mol_type"]
    token_pad_mask = feats["token_pad_mask"]
    atom_pad_mask = feats["atom_pad_mask"]

    if recompute_nonpolymer_frames:
        nonpolymer_frames = _legacy_nearest_nonpolymer_frames(
            pred_atom_coords,
            asym_id_atom,
            atom_pad_mask,
            feats["token_to_rep_atom"],
        )
        chain_atom_count = jnp.sum(
            (asym_id_token[:, :, None] == asym_id_atom[:, None, :])
            * atom_pad_mask[:, None, :],
            axis=-1,
        )
        same_chain_token = (
            asym_id_token[:, :, None] == asym_id_token[:, None, :]
        ) & token_pad_mask[:, None, :].astype(bool)
        first_chain_token = jnp.argmax(same_chain_token, axis=-1)
        first_chain_mol_type = jnp.take_along_axis(
            mol_type, first_chain_token, axis=1
        )
        use_nonpolymer_frame = (
            (first_chain_mol_type == 3)
            & (chain_atom_count >= 3)
            & token_pad_mask.astype(bool)
        )
        frames_idx_pred = jnp.where(
            use_nonpolymer_frame[:, None, :, None],
            nonpolymer_frames,
            frames_idx_pred,
        )

    bm = b // multiplicity
    idx_b = jnp.arange(bm)[:, None, None, None]
    idx_m = jnp.arange(multiplicity)[None, :, None, None]
    frames_expanded = pred_atom_coords[idx_b, idx_m, frames_idx_pred].reshape(-1, 3, 3)
    mask_collinear = _compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(bm, multiplicity, -1)
    return mask_collinear.astype(jnp.float32) * token_pad_mask[:, None, :]


def _valid_case(
    *, batch: int = 2, multiplicity: int = 3, tokens: int = 6, atoms: int = 24
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    if atoms % tokens:
        raise ValueError("the test case requires an equal atom count per token")

    rng = np.random.default_rng(20260824)
    coords = rng.normal(size=(batch * multiplicity, atoms, 3)).astype(np.float32)
    atoms_per_token = atoms // tokens
    atom_owner = np.repeat(np.arange(tokens), atoms_per_token)

    atom_to_token = np.zeros((batch, atoms, tokens), dtype=np.float32)
    token_to_rep_atom = np.zeros((batch, tokens, atoms), dtype=np.float32)
    frames_idx = np.zeros((batch, tokens, 3), dtype=np.int32)
    for batch_idx in range(batch):
        atom_to_token[batch_idx, np.arange(atoms), atom_owner] = 1.0
        for token_idx in range(tokens):
            owned = np.flatnonzero(atom_owner == token_idx)
            token_to_rep_atom[batch_idx, token_idx, owned[0]] = 1.0
            frames_idx[batch_idx, token_idx] = owned[:3]

    split = tokens // 2
    asym_id = np.broadcast_to(
        np.concatenate(
            [np.zeros(split, dtype=np.int32), np.ones(tokens - split, dtype=np.int32)]
        ),
        (batch, tokens),
    ).copy()
    mol_type = np.full((batch, tokens), 3, dtype=np.int32)
    token_pad_mask = np.ones((batch, tokens), dtype=np.float32)
    atom_pad_mask = np.ones((batch, atoms), dtype=np.float32)
    if batch > 1:
        token_pad_mask[1, -1] = 0.0
        atom_pad_mask[1, -2:] = 0.0

    feats = {
        "asym_id": jnp.asarray(asym_id),
        "atom_to_token": jnp.asarray(atom_to_token),
        "mol_type": jnp.asarray(mol_type),
        "token_pad_mask": jnp.asarray(token_pad_mask),
        "atom_pad_mask": jnp.asarray(atom_pad_mask),
        "token_to_rep_atom": jnp.asarray(token_to_rep_atom),
    }
    return jnp.asarray(coords), jnp.asarray(frames_idx), feats


def _atom_chain_ids(feats: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return gather_tokens_to_atoms(
        feats["atom_to_token"].astype(jnp.float32),
        feats["asym_id"][..., None].astype(jnp.float32),
    )[..., 0]


def _assert_same_float_classification(
    actual: jnp.ndarray, expected: jnp.ndarray
) -> None:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    np.testing.assert_array_equal(np.isnan(actual_np), np.isnan(expected_np))
    np.testing.assert_array_equal(np.isposinf(actual_np), np.isposinf(expected_np))
    np.testing.assert_array_equal(np.isneginf(actual_np), np.isneginf(expected_np))
    finite = np.isfinite(expected_np)
    np.testing.assert_array_equal(actual_np[finite], expected_np[finite])


def test_representative_rows_match_dense_legacy_for_batch_and_multiplicity() -> None:
    multiplicity = 3
    coords, frames_idx, feats = _valid_case(multiplicity=multiplicity)
    reshaped = coords.reshape(2, multiplicity, 24, 3)
    asym_id_atom = _atom_chain_ids(feats)

    expected_nearest = _legacy_nearest_nonpolymer_frames(
        reshaped,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )
    actual_nearest = _nearest_nonpolymer_frames(
        reshaped,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )
    jitted_nearest = jax.jit(_nearest_nonpolymer_frames)(
        reshaped,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )
    np.testing.assert_array_equal(actual_nearest, expected_nearest)
    np.testing.assert_array_equal(jitted_nearest, expected_nearest)

    expected_mask = _legacy_compute_frame_pred_inference(
        coords, frames_idx, feats, multiplicity
    )
    actual_mask = _compute_frame_pred_inference(
        coords, frames_idx, feats, multiplicity
    )
    jitted_mask = jax.jit(
        lambda x: _compute_frame_pred_inference(x, frames_idx, feats, multiplicity)
    )(coords)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_array_equal(jitted_mask, expected_mask)


def test_representative_chain_id_is_gathered_from_the_atom_mapping() -> None:
    coords = jnp.asarray(
        [[[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
           [21.0, 0.0, 0.0], [23.0, 0.0, 0.0]]]],
        dtype=jnp.float32,
    )
    asym_id_atom = jnp.asarray([[0.0, 0.0, 7.0, 7.0, 7.0]])
    atom_pad_mask = jnp.ones((1, 5), dtype=jnp.float32)
    token_to_rep_atom = jnp.asarray([[[0.0, 0.0, 1.0, 0.0, 0.0]]])

    actual = _nearest_nonpolymer_frames(
        coords, asym_id_atom, atom_pad_mask, token_to_rep_atom
    )

    np.testing.assert_array_equal(actual, np.asarray([[[[3, 2, 4]]]]))


def test_argmax_ties_zero_rows_and_padded_representatives_match_legacy() -> None:
    coords = jnp.asarray(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype=jnp.float32,
    )
    asym_id_atom = jnp.zeros((1, 6), dtype=jnp.float32)
    atom_pad_mask = jnp.asarray([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    token_to_rep_atom = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
                [-3.0, -2.0, -1.0, -4.0, -1.0, -5.0],
            ]
        ]
    )

    expected = _legacy_nearest_nonpolymer_frames(
        coords, asym_id_atom, atom_pad_mask, token_to_rep_atom
    )
    actual = _nearest_nonpolymer_frames(
        coords, asym_id_atom, atom_pad_mask, token_to_rep_atom
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0, 0, 0], np.asarray([1, 0, 2]))
    # The tied representative row selects atom 3, which is padded. The stable
    # full-width argsort of its all-infinity row retains atom index order.
    np.testing.assert_array_equal(actual[0, 0, 1], np.asarray([1, 0, 2]))


@pytest.mark.parametrize(
    ("atom_idx", "value"),
    [
        (1, np.nan),
        (2, np.inf),
        (4, -np.inf),
    ],
)
def test_nonfinite_coordinate_classification_matches_legacy(
    atom_idx: int, value: float
) -> None:
    multiplicity = 2
    coords, frames_idx, feats = _valid_case(
        batch=2, multiplicity=multiplicity, tokens=6, atoms=24
    )
    coords = coords.at[0, atom_idx, 1].set(value)
    reshaped = coords.reshape(2, multiplicity, 24, 3)
    asym_id_atom = _atom_chain_ids(feats)

    expected_nearest = _legacy_nearest_nonpolymer_frames(
        reshaped,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )
    actual_nearest = _nearest_nonpolymer_frames(
        reshaped,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )
    np.testing.assert_array_equal(actual_nearest, expected_nearest)

    expected_mask = _legacy_compute_frame_pred_inference(
        coords, frames_idx, feats, multiplicity
    )
    actual_mask = _compute_frame_pred_inference(
        coords, frames_idx, feats, multiplicity
    )
    _assert_same_float_classification(actual_mask, expected_mask)


def test_first_chain_molecule_type_still_controls_frame_replacement() -> None:
    coords = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    frames_idx = jnp.asarray([[[0, 1, 3], [0, 1, 3]]], dtype=jnp.int32)
    atom_to_token = jnp.asarray(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ]
    )
    feats = {
        "asym_id": jnp.zeros((1, 2), dtype=jnp.int32),
        "atom_to_token": atom_to_token,
        "mol_type": jnp.asarray([[0, 3]], dtype=jnp.int32),
        "token_pad_mask": jnp.ones((1, 2), dtype=jnp.float32),
        "atom_pad_mask": jnp.ones((1, 4), dtype=jnp.float32),
        "token_to_rep_atom": jnp.asarray(
            [[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        ),
    }

    expected = _legacy_compute_frame_pred_inference(coords, frames_idx, feats, 1)
    actual = _compute_frame_pred_inference(coords, frames_idx, feats, 1)

    np.testing.assert_array_equal(expected, np.ones((1, 1, 2), dtype=np.float32))
    np.testing.assert_array_equal(actual, expected)


def test_recompute_disabled_retains_the_legacy_frame_path() -> None:
    multiplicity = 3
    coords, frames_idx, feats = _valid_case(multiplicity=multiplicity)

    expected = _legacy_compute_frame_pred_inference(
        coords,
        frames_idx,
        feats,
        multiplicity,
        recompute_nonpolymer_frames=False,
    )
    actual = _compute_frame_pred_inference(
        coords,
        frames_idx,
        feats,
        multiplicity,
        recompute_nonpolymer_frames=False,
    )

    np.testing.assert_array_equal(actual, expected)


def test_representative_rows_remove_the_dense_atom_square_from_cpu_hlo() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("this is an isolated CPU compiler-memory gate")

    batch, multiplicity, tokens, atoms = 1, 2, 8, 128
    coords, _, feats = _valid_case(
        batch=batch, multiplicity=multiplicity, tokens=tokens, atoms=atoms
    )
    coords = coords.reshape(batch, multiplicity, atoms, 3)
    asym_id_atom = _atom_chain_ids(feats)
    args = (
        coords,
        asym_id_atom,
        feats["atom_pad_mask"],
        feats["token_to_rep_atom"],
    )

    compact_lowered = jax.jit(_nearest_nonpolymer_frames).lower(*args)
    legacy_lowered = jax.jit(_legacy_nearest_nonpolymer_frames).lower(*args)
    compact_hlo = str(compact_lowered.compiler_ir(dialect="stablehlo"))

    assert f"tensor<1x2x{atoms}x{atoms}x3xf32>" not in compact_hlo
    assert f"tensor<1x2x{tokens}x{atoms}x3xf32>" in compact_hlo

    compact_temp = compact_lowered.compile().memory_analysis().temp_size_in_bytes
    legacy_temp = legacy_lowered.compile().memory_analysis().temp_size_in_bytes
    assert compact_temp * 4 < legacy_temp
