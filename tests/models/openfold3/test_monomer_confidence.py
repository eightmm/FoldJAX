"""Static confidence specialization for known OpenFold3 monomers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.confidence import compute_ptm


def _inputs():
    logits = jnp.arange(2 * 3 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 3, 4)
    has_frame = jnp.asarray([[1, 1, 0], [1, 1, 1]], dtype=bool)
    token_mask = jnp.asarray([1, 1, 1], dtype=bool)
    asym_id = jnp.zeros(3, dtype=jnp.int32)
    kwargs = {"bin_min": 0.0, "bin_max": 32.0, "no_bins": 4}
    ptm = compute_ptm(logits, has_frame, token_mask, **kwargs)
    return ptm, logits, has_frame, token_mask, asym_id, kwargs


def test_known_monomer_iptm_does_not_run_the_interface_reduction(monkeypatch) -> None:
    ptm, logits, has_frame, token_mask, asym_id, kwargs = _inputs()
    expected = compute_ptm(
        logits,
        has_frame,
        token_mask,
        asym_id=asym_id,
        interface=True,
        **kwargs,
    )

    def unexpected_compute(*args, **kwargs):
        raise AssertionError("a known monomer has no interface TM reduction")

    monkeypatch.setattr(inference, "compute_ptm", unexpected_compute)
    actual = inference._compute_global_iptm(
        ptm,
        logits,
        has_frame,
        token_mask,
        asym_id,
        n_chain=1,
        **kwargs,
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(actual), np.zeros_like(np.asarray(ptm)))
    assert actual.shape == ptm.shape
    assert actual.dtype == ptm.dtype


def test_unknown_chain_count_keeps_the_generic_interface_score(monkeypatch) -> None:
    ptm, logits, has_frame, token_mask, asym_id, kwargs = _inputs()
    expected = compute_ptm(
        logits,
        has_frame,
        token_mask,
        asym_id=asym_id,
        interface=True,
        **kwargs,
    )
    calls = []

    def tracked_compute(*args, **call_kwargs):
        calls.append(call_kwargs)
        return compute_ptm(*args, **call_kwargs)

    monkeypatch.setattr(inference, "compute_ptm", tracked_compute)
    actual = inference._compute_global_iptm(
        ptm,
        logits,
        has_frame,
        token_mask,
        asym_id,
        n_chain=None,
        **kwargs,
    )

    assert len(calls) == 1
    assert calls[0]["interface"] is True
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_known_monomer_hlo_omits_the_interface_softmax() -> None:
    ptm, logits, has_frame, token_mask, asym_id, kwargs = _inputs()

    def lower(n_chain):
        return jax.jit(
            lambda score, pair, frames, mask, chains: inference._compute_global_iptm(
                score,
                pair,
                frames,
                mask,
                chains,
                n_chain=n_chain,
                **kwargs,
            )
        ).lower(ptm, logits, has_frame, token_mask, asym_id)

    assert "stablehlo.exponential" in lower(None).as_text()
    assert "stablehlo.exponential" not in lower(1).as_text()
