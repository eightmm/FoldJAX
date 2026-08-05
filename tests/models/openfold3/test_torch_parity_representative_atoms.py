"""Parity for representative-atom selection against upstream's own function."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.representative_atoms import (
    token_representative_atoms,
)

from .of3_features import make_batch

pytestmark = pytest.mark.torch_parity


@pytest.fixture
def table(openfold3_source: Path):
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    return representative_atom_table()


def test_matches_upstream(openfold3_source: Path, table) -> None:
    import torch
    from openfold3.core.utils.atomize_utils import get_token_representative_atoms

    batch = make_batch()
    x = torch.randn((1, batch["atom_mask"].shape[-1], 3))
    with torch.no_grad():
        rep_x_ref, rep_mask_ref = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=batch["atom_mask"]
        )

    jax_batch = {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }
    rep_x, rep_mask = token_representative_atoms(
        jax_batch,
        jnp.asarray(x.numpy()),
        jax_batch["atom_mask"],
        table,
    )
    np.testing.assert_allclose(
        np.asarray(rep_x, dtype=np.float64),
        rep_x_ref.numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
        err_msg="representative atom positions diverged",
    )
    np.testing.assert_allclose(
        np.asarray(rep_mask, dtype=np.float64),
        rep_mask_ref.numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
        err_msg="representative atom mask diverged",
    )


def test_gathers_per_sample(openfold3_source: Path, table) -> None:
    """A leading sample axis on the coordinates must be gathered independently."""
    import torch

    batch = make_batch()
    n_atom = batch["atom_mask"].shape[-1]
    x = torch.randn((1, 3, n_atom, 3))
    jax_batch = {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }
    rep_x, _mask = token_representative_atoms(
        jax_batch, jnp.asarray(x.numpy()), jax_batch["atom_mask"], table
    )
    assert rep_x.shape == (1, 3, batch["token_mask"].shape[-1], 3)
    assert not np.allclose(np.asarray(rep_x[0, 0]), np.asarray(rep_x[0, 1]))
