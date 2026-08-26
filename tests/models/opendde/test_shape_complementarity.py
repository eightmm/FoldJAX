"""The shape-complementarity port against upstream's numbers on two chains.

Every quantity here is derived from geometry alone, so there is no checkpoint
to catch a mistake and no shape that would fail. A wrong sign on a normal, the
gap Gaussian centred on the wrong distance, a softmax that lets masked partners
into its denominator -- each produces a well-formed array of plausible
magnitude. Upstream's own output on the same coordinates is the only thing that
distinguishes them, which is what `shape_complementarity.npz` holds.

The fixture is two chains on purpose: `asym_id[i] != asym_id[j]` gates every
pair, so a single-chain case scores exactly zero and would pass against an
implementation that returns zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.opendde.models.shape_complementarity import (
    SHAPE_COMP_DEFAULTS,
    compute_shape_complementarity,
)

FIXTURE = Path(__file__).parent / "data" / "shape_complementarity.npz"


@pytest.fixture(scope="module")
def recorded():
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; run export_shape_complementarity_fixture.py")
    return np.load(FIXTURE, allow_pickle=True)


@pytest.fixture(scope="module")
def computed(recorded):
    features = {
        name[len("input_") :]: recorded[name]
        for name in recorded.files
        if name.startswith("input_")
    }
    import jax.numpy as jnp

    return compute_shape_complementarity(
        jnp.asarray(recorded["coordinate"]),
        features,
        jnp.asarray(recorded["atom_mask"]),
    )


def test_the_fixture_has_an_interface_to_score(recorded) -> None:
    """Otherwise every assertion below is `0 == 0`."""
    assert bool(recorded["shape_comp_token_mask"].any())
    assert float(recorded["shape_comp_global_pred"]) > 0.0
    assert float(recorded["shape_comp_pair_topk_mean_pred"]) > 0.0


def test_this_ports_defaults_are_upstreams(recorded) -> None:
    """The config block is copied, not imported, so it can silently drift.

    Every one of these moves the score continuously -- `gap_mean` at 4.0
    instead of 6.0 still yields a plausible number on every input.
    """
    keys = [str(k) for k in recorded["settings_keys"]]
    values = recorded["settings_values"]
    for key, value in zip(keys, values):
        if key not in SHAPE_COMP_DEFAULTS:
            continue
        assert float(SHAPE_COMP_DEFAULTS[key]) == pytest.approx(float(value)), key


@pytest.mark.parametrize(
    "field",
    [
        "shape_comp_token_pred",
        "shape_comp_global_pred",
        "shape_comp_pair_mean_pred",
        "shape_comp_pair_topk_mean_pred",
        "shape_comp_valid_pair_frac_pred",
    ],
)
def test_matches_upstream(recorded, computed, field: str) -> None:
    np.testing.assert_allclose(
        np.asarray(computed[field], np.float64),
        np.asarray(recorded[field], np.float64),
        rtol=2e-5,
        atol=2e-6,
        err_msg=field,
    )


def test_the_token_mask_matches_exactly(recorded, computed) -> None:
    """A boolean, so `allclose` would hide a one-token disagreement."""
    np.testing.assert_array_equal(
        np.asarray(computed["shape_comp_token_mask"]),
        np.asarray(recorded["shape_comp_token_mask"]).astype(bool),
    )


def test_one_chain_scores_zero(recorded) -> None:
    """The property that makes the two-chain fixture necessary.

    Not a formality: it is the reason this port's absence never showed in the
    benchmark, whose four cases are all single-chain.
    """
    import jax.numpy as jnp

    features = {
        name[len("input_") :]: recorded[name]
        for name in recorded.files
        if name.startswith("input_")
    }
    features["asym_id"] = np.zeros_like(features["asym_id"])

    output = compute_shape_complementarity(
        jnp.asarray(recorded["coordinate"]),
        features,
        jnp.asarray(recorded["atom_mask"]),
    )

    assert not bool(np.asarray(output["shape_comp_token_mask"]).any())
    assert float(output["shape_comp_global_pred"]) == 0.0
    assert float(output["shape_comp_pair_mean_pred"]) == 0.0
    # -inf must not leak out of the top-k pool when nothing is valid.
    assert float(output["shape_comp_pair_topk_mean_pred"]) == 0.0


def test_a_chunked_run_equals_an_unchunked_one(recorded, computed) -> None:
    """The chunk size bounds two quadratic arrays; it must not move the answer.

    Upstream's chunk loop reorders a reduction, and the top-32 is accumulated
    across chunks rather than taken once, so this is the assertion that the
    merge is right.
    """
    import jax.numpy as jnp

    features = {
        name[len("input_") :]: recorded[name]
        for name in recorded.files
        if name.startswith("input_")
    }
    whole = compute_shape_complementarity(
        jnp.asarray(recorded["coordinate"]),
        features,
        jnp.asarray(recorded["atom_mask"]),
        pair_chunk_size=4,
    )

    for field in ("shape_comp_token_pred", "shape_comp_pair_topk_mean_pred"):
        np.testing.assert_allclose(
            np.asarray(whole[field], np.float64),
            np.asarray(computed[field], np.float64),
            rtol=2e-5,
            atol=2e-6,
            err_msg=field,
        )


def test_density_mask_is_formed_per_chunk_instead_of_embedded_in_full() -> None:
    import jax
    import jax.numpy as jnp

    from foldjax.models.opendde.models.shape_complementarity import (
        ShapeCompTokenFeatures,
        _token_centers_and_normals,
    )

    n_token, n_atom, chunk = 16, 64, 4
    owner = np.repeat(np.arange(n_token), n_atom // n_token)
    resolved = ShapeCompTokenFeatures(
        atom_to_token_idx=owner,
        rep_atom_indices=np.arange(0, n_atom, n_atom // n_token),
        rep_atom_valid=np.ones(n_token, dtype=bool),
        token_asym_id=np.arange(n_token) % 2,
        token_role_id=np.full(n_token, -1),
        is_structural=False,
        is_protein_token=np.ones(n_token, dtype=bool),
    )

    lowered = jax.jit(
        lambda coordinate, mask: _token_centers_and_normals(
            coordinate,
            mask,
            resolved,
            n_token=n_token,
            density_sigma=1.5,
            chunk_size=chunk,
            eps=1e-6,
        )
    ).lower(
        jnp.zeros((n_atom, 3), jnp.float32),
        jnp.ones(n_atom, bool),
    )
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert f"tensor<{n_token}x{n_atom}xi1>" not in hlo
    assert f"tensor<{chunk}x{n_atom}xi1>" in hlo
