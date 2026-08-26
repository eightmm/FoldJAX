from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models import confidence


def test_chain_pair_iptm_computes_each_symmetric_pair_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_compute_ptm(
        logits,
        has_frame,
        mask_i,
        **kwargs,
    ):
        del has_frame, kwargs
        calls.append(np.asarray(mask_i))
        return jnp.full((logits.shape[0],), len(calls), logits.dtype)

    monkeypatch.setattr(confidence, "compute_ptm", fake_compute_ptm)
    n_chain = 4
    actual = np.asarray(
        confidence.compute_chain_pair_iptm(
            jnp.zeros((2, n_chain, n_chain, 3), jnp.float32),
            jnp.ones((2, n_chain), bool),
            jnp.ones(n_chain, bool),
            jnp.arange(n_chain),
            n_chain=n_chain,
            bin_min=0.0,
            bin_max=1.0,
            no_bins=3,
        )
    )

    assert len(calls) == n_chain * (n_chain - 1) // 2
    np.testing.assert_array_equal(actual, np.swapaxes(actual, -1, -2))
    np.testing.assert_array_equal(
        np.diagonal(actual, axis1=-2, axis2=-1),
        np.zeros((2, n_chain), dtype=np.float32),
    )
