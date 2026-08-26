from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.atomize import broadcast_token_feat_to_atoms


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_validated_owner_table_matches_boundary_reconstruction(dtype) -> None:
    counts = np.asarray([3, 0, 2, 1, 0], dtype=np.int32)
    real_owners = np.repeat(np.arange(counts.size), counts)
    n_atom = real_owners.size + 3
    owners = np.pad(real_owners, (0, n_atom - real_owners.size))[None]
    token_mask = np.asarray([[1, 1, 1, 1, 0]], dtype=np.int32)
    values = np.asarray(
        [
            [0.0, -0.0, 1.0],
            [np.nan, 2.0, 3.0],
            [np.inf, -np.inf, 4.0],
            [5.0, -6.0, 7.0],
            [8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )
    token_feat = jnp.broadcast_to(jnp.asarray(values, dtype=dtype), (3, 5, 3))

    legacy = broadcast_token_feat_to_atoms(
        jnp.asarray(token_mask),
        jnp.asarray(counts[None]),
        token_feat,
        n_atom=n_atom,
    )
    indexed = broadcast_token_feat_to_atoms(
        jnp.asarray(token_mask),
        jnp.asarray(counts[None]),
        token_feat,
        n_atom=n_atom,
        atom_to_token_index=jnp.asarray(owners),
    )

    legacy_bits = np.asarray(legacy).view(
        np.uint16 if dtype == jnp.bfloat16 else np.uint32
    )
    indexed_bits = np.asarray(indexed).view(
        np.uint16 if dtype == jnp.bfloat16 else np.uint32
    )
    np.testing.assert_array_equal(indexed_bits, legacy_bits)


def test_indexed_route_avoids_the_atom_by_token_reduction() -> None:
    n_token = 64
    n_atom = 256
    token_mask = jnp.ones((1, n_token), dtype=jnp.int32)
    counts = jnp.full((1, n_token), n_atom // n_token, dtype=jnp.int32)
    token_feat = jnp.ones((1, n_token, 8), dtype=jnp.float32)
    owners = jnp.repeat(jnp.arange(n_token), n_atom // n_token)[None]

    def indexed(tm, count, feat, owner):
        return broadcast_token_feat_to_atoms(
            tm,
            count,
            feat,
            n_atom=n_atom,
            atom_to_token_index=owner,
        )

    def legacy(tm, count, feat):
        return broadcast_token_feat_to_atoms(
            tm,
            count,
            feat,
            n_atom=n_atom,
        )

    legacy_hlo = str(
        jax.jit(legacy).lower(token_mask, counts, token_feat).compiler_ir()
    )
    indexed_hlo = str(
        jax.jit(indexed)
        .lower(token_mask, counts, token_feat, owners)
        .compiler_ir()
    )

    assert "tensor<1x256x64xi1>" in legacy_hlo
    assert (
        "tensor<1x256x64xi32>, tensor<i32>) -> tensor<1x256xi32>"
        in legacy_hlo
    )
    assert "stablehlo.reduce_window" in legacy_hlo
    assert "tensor<1x256x64xi1>" not in indexed_hlo
    assert (
        "tensor<1x256x64xi32>, tensor<i32>) -> tensor<1x256xi32>"
        not in indexed_hlo
    )
    assert "stablehlo.reduce_window" not in indexed_hlo


def test_indexed_route_requires_a_complete_owner_contract() -> None:
    args = (
        jnp.ones((1, 2)),
        jnp.ones((1, 2), dtype=jnp.int32),
        jnp.ones((1, 2, 3)),
    )
    with pytest.raises(ValueError, match="length"):
        broadcast_token_feat_to_atoms(
            *args,
            n_atom=2,
            atom_to_token_index=jnp.zeros((1, 3), dtype=jnp.int32),
        )


def test_indexed_route_keeps_count_based_padding_and_signed_zero() -> None:
    token_mask = jnp.ones((1, 2), dtype=jnp.int32)
    counts = jnp.ones((1, 2), dtype=jnp.int32)
    token_feat = jnp.asarray([[[1.0], [-0.0]]], dtype=jnp.float32)
    owners = jnp.asarray([[0, 1, 1, 1]], dtype=jnp.int32)

    legacy = broadcast_token_feat_to_atoms(
        token_mask,
        counts,
        token_feat,
        n_atom=4,
    )
    indexed = broadcast_token_feat_to_atoms(
        token_mask,
        counts,
        token_feat,
        n_atom=4,
        atom_to_token_index=owners,
    )

    np.testing.assert_array_equal(
        np.asarray(indexed).view(np.uint32),
        np.asarray(legacy).view(np.uint32),
    )
