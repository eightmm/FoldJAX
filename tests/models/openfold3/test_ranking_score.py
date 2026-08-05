"""The AF3 sample ranking score.

Weight-free arithmetic, so this is checked against the documented formula and
the ordering behaviour it has to produce rather than against a torch call: the
upstream function is entangled with RASA and batch featurization.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.clash import sample_ranking_score


def test_matches_the_documented_formula() -> None:
    ptm = jnp.asarray([0.5, 0.9])
    iptm = jnp.asarray([0.4, 0.8])
    clash = jnp.asarray([0.0, 0.0])
    disorder = jnp.asarray([0.2, 0.1])
    expected = 0.8 * np.asarray(iptm) + 0.2 * np.asarray(ptm) + 0.5 * np.asarray(
        disorder
    )
    np.testing.assert_allclose(
        np.asarray(sample_ranking_score(ptm, iptm, clash, disorder)),
        expected,
        rtol=1e-6,
    )


def test_disorder_defaults_to_zero() -> None:
    ptm = jnp.asarray([0.5])
    iptm = jnp.asarray([0.4])
    clash = jnp.asarray([0.0])
    assert float(sample_ranking_score(ptm, iptm, clash)[0]) == pytest.approx(
        0.8 * 0.4 + 0.2 * 0.5
    )


def test_iptm_dominates_ptm() -> None:
    """0.8 vs 0.2 — an implementation that swaps them would rank differently."""
    high_iptm = sample_ranking_score(
        jnp.asarray([0.0]), jnp.asarray([1.0]), jnp.asarray([0.0])
    )
    high_ptm = sample_ranking_score(
        jnp.asarray([1.0]), jnp.asarray([0.0]), jnp.asarray([0.0])
    )
    assert float(high_iptm[0]) > float(high_ptm[0])


def test_a_clash_vetoes_the_sample() -> None:
    """The 100x weight must sink a clashing sample below any clean one."""
    best_clashing = sample_ranking_score(
        jnp.asarray([1.0]), jnp.asarray([1.0]), jnp.asarray([1.0]),
        jnp.asarray([1.0]),
    )
    worst_clean = sample_ranking_score(
        jnp.asarray([0.0]), jnp.asarray([0.0]), jnp.asarray([0.0])
    )
    assert float(best_clashing[0]) < float(worst_clean[0])


def test_ranking_order_is_recoverable() -> None:
    scores = sample_ranking_score(
        jnp.asarray([0.5, 0.9, 0.7]),
        jnp.asarray([0.4, 0.85, 0.6]),
        jnp.asarray([0.0, 1.0, 0.0]),
    )
    # Sample 1 has the best confidence but clashes, so sample 2 must win.
    assert int(np.argmax(np.asarray(scores))) == 2
