"""Predicted-geometry frame selection for OpenFold3 confidence scores."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.frames import token_frame_atoms


def _names(values: list[str]) -> jnp.ndarray:
    encoded = np.zeros((1, len(values), 4, 64), dtype=np.int32)
    for atom, name in enumerate(values):
        for position, character in enumerate(name.ljust(4)):
            encoded[0, atom, position, ord(character) - 32] = 1
    return jnp.asarray(encoded)


def test_standard_protein_and_nucleotide_frames_use_named_atoms() -> None:
    names = ["N", "CA", "C", "O", "P", "C3'", "C1'", "C4'"]
    coordinates = jnp.asarray(
        [
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [2, 0, 0],
                [9, 9, 9],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        ],
        dtype=jnp.float32,
    )
    batch = {
        "token_mask": jnp.ones((1, 2)),
        "atom_mask": jnp.ones((1, 8)),
        "atom_to_token_index": jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1]]),
        "num_atoms_per_token": jnp.asarray([[4, 4]]),
        "start_atom_index": jnp.asarray([[0, 4]]),
        "asym_id": jnp.asarray([[1, 2]]),
        "is_atomized": jnp.zeros((1, 2)),
        "is_protein": jnp.asarray([[1, 0]]),
        "is_dna": jnp.asarray([[0, 1]]),
        "is_rna": jnp.zeros((1, 2)),
        "ref_atom_name_chars": _names(names),
    }

    (a, b, c), valid = token_frame_atoms(batch, coordinates, batch["atom_mask"])

    np.testing.assert_array_equal(np.asarray(valid), [[True, True]])
    np.testing.assert_allclose(np.asarray(a), np.asarray(coordinates[:, [0, 5]]))
    np.testing.assert_allclose(np.asarray(b), np.asarray(coordinates[:, [1, 6]]))
    np.testing.assert_allclose(np.asarray(c), np.asarray(coordinates[:, [2, 7]]))


def test_atomized_frame_uses_nearest_same_chain_atoms_and_angle() -> None:
    # Token 0 starts at atom 0; atom 1 and 2 are its closest same-chain
    # neighbours and form a right angle. Atom 3 is closer than atom 2 but belongs
    # to another chain, so it must not enter the frame.
    coordinates = jnp.asarray(
        [[[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0.5, 0]]], dtype=jnp.float32
    )
    batch = {
        "token_mask": jnp.asarray([[1, 1]]),
        "atom_mask": jnp.ones((1, 4)),
        "atom_to_token_index": jnp.asarray([[0, 0, 0, 1]]),
        "num_atoms_per_token": jnp.asarray([[3, 1]]),
        "start_atom_index": jnp.asarray([[0, 3]]),
        "asym_id": jnp.asarray([[7, 9]]),
        "is_atomized": jnp.ones((1, 2)),
        "is_protein": jnp.zeros((1, 2)),
        "is_dna": jnp.zeros((1, 2)),
        "is_rna": jnp.zeros((1, 2)),
        "ref_atom_name_chars": _names(["C1", "C2", "O1", "N1"]),
    }

    (a, b, c), valid = token_frame_atoms(batch, coordinates, batch["atom_mask"])

    assert bool(valid[0, 0])
    assert not bool(valid[0, 1])  # no two same-chain neighbours
    np.testing.assert_allclose(np.asarray(b[0, 0]), np.asarray(coordinates[0, 0]))
    selected = {tuple(np.asarray(a[0, 0])), tuple(np.asarray(c[0, 0]))}
    assert selected == {(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)}


def test_missing_standard_atom_and_nearly_collinear_ligand_are_invalid() -> None:
    coordinates = jnp.asarray(
        [[[0, 0, 0], [1, 0, 0], [2, 0.01, 0], [3, 0, 0]]], dtype=jnp.float32
    )
    batch = {
        "token_mask": jnp.asarray([[1, 1]]),
        "atom_mask": jnp.ones((1, 4)),
        "atom_to_token_index": jnp.asarray([[0, 1, 1, 1]]),
        "num_atoms_per_token": jnp.asarray([[1, 3]]),
        "start_atom_index": jnp.asarray([[0, 1]]),
        "asym_id": jnp.asarray([[1, 2]]),
        "is_atomized": jnp.asarray([[0, 1]]),
        "is_protein": jnp.asarray([[1, 0]]),
        "is_dna": jnp.zeros((1, 2)),
        "is_rna": jnp.zeros((1, 2)),
        "ref_atom_name_chars": _names(["CA", "C1", "C2", "C3"]),
    }
    _, valid = token_frame_atoms(batch, coordinates, batch["atom_mask"])
    np.testing.assert_array_equal(np.asarray(valid), [[False, False]])


def test_nonpolymer_nonatomized_token_does_not_borrow_a_nucleotide_frame() -> None:
    coordinates = jnp.asarray(
        [[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], dtype=jnp.float32
    )
    batch = {
        "token_mask": jnp.ones((1, 1)),
        "atom_mask": jnp.ones((1, 3)),
        "atom_to_token_index": jnp.zeros((1, 3), dtype=jnp.int32),
        "num_atoms_per_token": jnp.asarray([[3]]),
        "start_atom_index": jnp.asarray([[0]]),
        "asym_id": jnp.asarray([[0]]),
        "is_atomized": jnp.zeros((1, 1)),
        "is_protein": jnp.zeros((1, 1)),
        "is_dna": jnp.zeros((1, 1)),
        "is_rna": jnp.zeros((1, 1)),
        "ref_atom_name_chars": _names(["C3'", "C1'", "C4'"]),
    }

    _, valid = token_frame_atoms(batch, coordinates, batch["atom_mask"])

    np.testing.assert_array_equal(np.asarray(valid), [[False]])


def test_atomized_token_with_fewer_than_three_atoms_is_invalid_not_an_error() -> None:
    coordinates = jnp.asarray([[[0, 0, 0], [1, 0, 0]]], dtype=jnp.float32)
    batch = {
        "token_mask": jnp.ones((1, 1)),
        "atom_mask": jnp.ones((1, 2)),
        "atom_to_token_index": jnp.zeros((1, 2), dtype=jnp.int32),
        "num_atoms_per_token": jnp.asarray([[2]]),
        "start_atom_index": jnp.asarray([[0]]),
        "asym_id": jnp.asarray([[0]]),
        "is_atomized": jnp.ones((1, 1)),
        "is_protein": jnp.zeros((1, 1)),
        "is_dna": jnp.zeros((1, 1)),
        "is_rna": jnp.zeros((1, 1)),
        "ref_atom_name_chars": _names(["C1", "C2"]),
    }

    _, valid = token_frame_atoms(batch, coordinates, batch["atom_mask"])

    np.testing.assert_array_equal(np.asarray(valid), [[False]])


@pytest.mark.torch_parity
def test_frame_selection_matches_upstream_for_polymer_and_atomized_tokens(
    openfold3_source,
) -> None:
    import torch
    from openfold3.core.utils.atomize_utils import (
        get_token_frame_atoms as torch_token_frame_atoms,
    )

    # One ALA token followed by three atomized ligand tokens in another chain.
    counts = np.array([5, 1, 1, 1], dtype=np.int32)
    starts = np.array([0, 5, 6, 7], dtype=np.int32)
    owners = np.repeat(np.arange(4, dtype=np.int32), counts)
    asym_id = np.array([0, 1, 1, 1], dtype=np.int32)
    coordinates = np.array(
        [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [1, -1, 0],
          [10, 0, 0], [11, 0, 0], [10, 2, 0]]],
        dtype=np.float32,
    )
    restype = np.zeros((4, 32), dtype=np.int32)
    restype[0, 0] = 1  # ALA
    restype[1:, 20] = 1  # UNK, as atomized chemistry is encoded
    base = {
        "token_mask": np.ones(4, dtype=np.float32),
        "num_atoms_per_token": counts,
        "start_atom_index": starts,
        "asym_id": asym_id,
        "is_atomized": np.array([0, 1, 1, 1], dtype=np.int32),
        "is_protein": np.array([1, 0, 0, 0], dtype=np.int32),
        "is_dna": np.zeros(4, dtype=np.int32),
        "is_rna": np.zeros(4, dtype=np.int32),
        "restype": restype,
    }
    torch_batch = {name: torch.from_numpy(value) for name, value in base.items()}
    torch_frames, torch_valid = torch_token_frame_atoms(
        torch_batch,
        torch.from_numpy(coordinates),
        torch.ones((1, 8), dtype=torch.float32),
    )

    jax_batch = {name: jnp.asarray(value[None]) for name, value in base.items()}
    jax_batch["atom_to_token_index"] = jnp.asarray(owners[None])
    jax_batch["ref_atom_name_chars"] = _names(
        ["N", "CA", "C", "O", "CB", "C1", "C2", "O1"]
    )
    jax_frames, jax_valid = token_frame_atoms(
        jax_batch,
        jnp.asarray(coordinates),
        jnp.ones((1, 8), dtype=jnp.float32),
    )

    for actual, expected in zip(jax_frames, torch_frames, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual), expected.detach().numpy(), atol=1e-6
        )
    np.testing.assert_array_equal(
        np.asarray(jax_valid), torch_valid.detach().numpy().astype(bool)
    )
