"""Torch-vs-JAX parity for the confidence metrics (AF3 SI 5.7, 5.9.1)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.confidence import (
    bin_centers,
    compute_plddt,
    compute_ptm,
    probs_to_expected_error,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-5
ATOL = 1e-5

N_SAMPLE, N_TOKEN, NO_BINS = 2, 6, 8
BIN_MIN, BIN_MAX = 0.0, 32.0


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.metrics import confidence

    return confidence


def _close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


@pytest.mark.parametrize(
    ("bin_min", "bin_max", "no_bins"),
    [(0.0, 32.0, 64), (2.0, 22.0, 64), (0.0, 1.0, 50), (0.0, 10.0, 3)],
)
def test_bin_centers_match_torch(
    openfold3_source: Path, bin_min: float, bin_max: float, no_bins: int
) -> None:
    torch = _torch()
    expected = _upstream().get_bin_centers(
        bin_min, bin_max, no_bins, device=torch.device("cpu"), dtype=torch.float32
    )
    actual = bin_centers(bin_min, bin_max, no_bins)
    assert actual.shape == (no_bins,)
    _close(actual, expected, "bin_centers")


def test_bin_centers_are_left_edges_plus_half_width(openfold3_source: Path) -> None:
    """Not linspace midpoints and not the right edges — this is easy to get wrong."""
    centers = np.asarray(bin_centers(0.0, 10.0, 5))
    np.testing.assert_allclose(centers, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_expected_error_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    probs = torch.softmax(torch.randn(2, 5, NO_BINS), dim=-1)
    expected = _upstream().probs_to_expected_error(
        probs, bin_min=BIN_MIN, bin_max=BIN_MAX, no_bins=NO_BINS
    )
    actual = probs_to_expected_error(
        jnp.asarray(probs.numpy()),
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
    )
    _close(actual, expected, "probs_to_expected_error")


def test_plddt_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    logits = torch.randn(2, 7, 50)
    expected = _upstream().compute_plddt(logits)
    actual = compute_plddt(jnp.asarray(logits.numpy()))
    _close(actual, expected, "compute_plddt")
    # pLDDT lives in [0, 1] with this binning.
    array = np.asarray(actual)
    assert array.min() >= 0.0 and array.max() <= 1.0


def _ptm_inputs(torch, *, all_frames: bool = True):
    logits = torch.randn(N_SAMPLE, N_TOKEN, N_TOKEN, NO_BINS)
    has_frame = torch.ones(N_SAMPLE, N_TOKEN)
    if not all_frames:
        has_frame[:, 2] = 0.0
    mask_i = torch.ones(N_TOKEN)
    mask_i[5] = 0.0
    asym_id = torch.tensor([0, 0, 0, 1, 1, 1])
    return logits, has_frame, mask_i, asym_id


@pytest.mark.parametrize("all_frames", [True, False])
def test_ptm_matches_torch(
    openfold3_source: Path, all_frames: bool
) -> None:
    torch = _torch()
    logits, has_frame, mask_i, _asym = _ptm_inputs(torch, all_frames=all_frames)
    expected = _upstream().compute_ptm(
        logits=logits,
        has_frame=has_frame,
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
        mask_i=mask_i,
    )
    actual = compute_ptm(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(mask_i.numpy()),
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
    )
    assert actual.shape == (N_SAMPLE,)
    _close(actual, expected, f"compute_ptm(all_frames={all_frames})")


def test_iptm_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    logits, has_frame, mask_i, asym_id = _ptm_inputs(torch)
    expected = _upstream().compute_ptm(
        logits=logits,
        has_frame=has_frame,
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
        mask_i=mask_i,
        asym_id=asym_id,
        interface=True,
    )
    actual = compute_ptm(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(mask_i.numpy()),
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
        asym_id=jnp.asarray(asym_id.numpy()),
        interface=True,
    )
    _close(actual, expected, "compute_ptm(interface=True)")


def test_iptm_differs_from_ptm(openfold3_source: Path) -> None:
    """ipTM excludes same-chain pairs, so the two must not coincide."""
    torch = _torch()
    logits, has_frame, mask_i, asym_id = _ptm_inputs(torch)
    args = (
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(mask_i.numpy()),
    )
    kwargs = {"bin_min": BIN_MIN, "bin_max": BIN_MAX, "no_bins": NO_BINS}
    ptm = compute_ptm(*args, **kwargs)
    iptm = compute_ptm(
        *args, **kwargs, asym_id=jnp.asarray(asym_id.numpy()), interface=True
    )
    assert not np.allclose(np.asarray(ptm), np.asarray(iptm), rtol=1e-3)


def test_iptm_requires_asym_id(openfold3_source: Path) -> None:
    with pytest.raises(ValueError, match="asym_id is required"):
        compute_ptm(
            jnp.zeros((1, 2, 2, NO_BINS)),
            jnp.ones((1, 2)),
            jnp.ones(2),
            bin_min=BIN_MIN,
            bin_max=BIN_MAX,
            no_bins=NO_BINS,
            interface=True,
        )


def test_ptm_and_iptm_are_jit_compilable(openfold3_source: Path) -> None:
    """Boolean indexing on a traced mask is not jittable.

    The first version of compute_ptm sliced both pair axes with `mask_i`, which
    raised NonConcreteBooleanIndexError under jit and would have forced eager
    execution during inference. The masked-sum formulation avoids that; this test
    keeps it that way.
    """
    import jax

    torch = _torch()
    logits, has_frame, mask_i, asym_id = _ptm_inputs(torch)
    args = (
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(mask_i.numpy()),
    )
    kwargs = {"bin_min": BIN_MIN, "bin_max": BIN_MAX, "no_bins": NO_BINS}

    ptm = jax.jit(lambda a, b, c: compute_ptm(a, b, c, **kwargs))(*args)
    assert ptm.shape == (N_SAMPLE,)

    iptm = jax.jit(
        lambda a, b, c, d: compute_ptm(
            a, b, c, asym_id=d, interface=True, **kwargs
        )
    )(*args, jnp.asarray(asym_id.numpy()))
    assert iptm.shape == (N_SAMPLE,)

    # And the jitted values still match the eager ones.
    np.testing.assert_allclose(
        np.asarray(ptm), np.asarray(compute_ptm(*args, **kwargs)), rtol=1e-6
    )


def test_chain_ptm_matches_torch_per_chain(openfold3_source: Path) -> None:
    """Per-chain pTM is pTM restricted to each chain's tokens."""
    from foldjax.models.openfold3.models.confidence import compute_chain_ptm

    torch = _torch()
    logits, has_frame, _mask, asym_id = _ptm_inputs(torch)
    token_mask = torch.ones(N_TOKEN)
    n_chain = 2

    expected = []
    for chain in range(n_chain):
        mask_i = token_mask.bool() & (asym_id == chain)
        expected.append(
            _upstream().compute_ptm(
                logits=logits,
                has_frame=has_frame,
                bin_min=BIN_MIN,
                bin_max=BIN_MAX,
                no_bins=NO_BINS,
                mask_i=mask_i,
            )
        )
    expected_stacked = torch.stack(expected, dim=-1)

    actual = compute_chain_ptm(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(asym_id.numpy()),
        n_chain=n_chain,
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
    )
    assert actual.shape == (N_SAMPLE, n_chain)
    _close(actual, expected_stacked, "compute_chain_ptm")


def test_chain_ptm_scores_absent_chains_zero(openfold3_source: Path) -> None:
    """A chain id with no tokens must score 0, not a spurious value."""
    from foldjax.models.openfold3.models.confidence import compute_chain_ptm

    torch = _torch()
    logits, has_frame, _mask, asym_id = _ptm_inputs(torch)
    actual = compute_chain_ptm(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.ones(N_TOKEN),
        jnp.asarray(asym_id.numpy()),
        n_chain=4,  # chains 2 and 3 do not exist
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
    )
    array = np.asarray(actual)
    assert np.all(array[:, 2:] == 0.0)
    assert np.all(array[:, :2] > 0.0)


def test_chain_ptm_is_jit_compilable(openfold3_source: Path) -> None:
    import jax

    from foldjax.models.openfold3.models.confidence import compute_chain_ptm

    torch = _torch()
    logits, has_frame, _mask, asym_id = _ptm_inputs(torch)
    fn = jax.jit(
        lambda a, b, c, d: compute_chain_ptm(
            a, b, c, d, n_chain=2, bin_min=BIN_MIN, bin_max=BIN_MAX,
            no_bins=NO_BINS,
        )
    )
    out = fn(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.ones(N_TOKEN),
        jnp.asarray(asym_id.numpy()),
    )
    assert out.shape == (N_SAMPLE, 2)


def test_chain_pair_iptm_matches_torch(openfold3_source: Path) -> None:
    """Each entry is ipTM over the union of two chains' tokens."""
    from foldjax.models.openfold3.models.confidence import compute_chain_pair_iptm

    torch = _torch()
    logits, has_frame, _mask, asym_id = _ptm_inputs(torch)
    token_mask = torch.ones(N_TOKEN)
    n_chain = 2

    chain_masks = [
        token_mask.bool() & (asym_id == chain) for chain in range(n_chain)
    ]
    expected = torch.zeros(N_SAMPLE, n_chain, n_chain)
    for i in range(n_chain):
        for j in range(i + 1, n_chain):
            value = _upstream().compute_ptm(
                logits=logits,
                has_frame=has_frame,
                mask_i=chain_masks[i] | chain_masks[j],
                asym_id=asym_id,
                interface=True,
                bin_min=BIN_MIN,
                bin_max=BIN_MAX,
                no_bins=NO_BINS,
            )
            expected[:, i, j] = value
            expected[:, j, i] = value

    actual = compute_chain_pair_iptm(
        jnp.asarray(logits.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(asym_id.numpy()),
        n_chain=n_chain,
        bin_min=BIN_MIN,
        bin_max=BIN_MAX,
        no_bins=NO_BINS,
    )
    assert actual.shape == (N_SAMPLE, n_chain, n_chain)
    _close(actual, expected, "compute_chain_pair_iptm")


def test_chain_pair_iptm_is_symmetric_with_zero_diagonal(
    openfold3_source: Path,
) -> None:
    from foldjax.models.openfold3.models.confidence import compute_chain_pair_iptm

    torch = _torch()
    logits, has_frame, _mask, asym_id = _ptm_inputs(torch)
    actual = np.asarray(
        compute_chain_pair_iptm(
            jnp.asarray(logits.numpy()),
            jnp.asarray(has_frame.numpy()),
            jnp.ones(N_TOKEN),
            jnp.asarray(asym_id.numpy()),
            n_chain=3,  # chain 2 is absent
            bin_min=BIN_MIN,
            bin_max=BIN_MAX,
            no_bins=NO_BINS,
        )
    )
    np.testing.assert_allclose(actual, np.swapaxes(actual, -1, -2), rtol=1e-6)
    assert np.all(np.diagonal(actual, axis1=-2, axis2=-1) == 0.0)
    # An absent chain contributes no interface.
    assert np.all(actual[:, 2, :] == 0.0)
    assert np.all(actual[:, :, 2] == 0.0)


def _chain_mean_reference(torch, pair, framed, n_chain):
    """Upstream's asymmetric row/column averaging, transcribed."""
    out = torch.zeros(pair.shape[0], n_chain)
    for i in range(n_chain):
        values = [pair[:, i, j] for j in range(n_chain) if j != i and framed[i]]
        values += [pair[:, j, i] for j in range(n_chain) if j != i and framed[j]]
        if values:
            out[:, i] = torch.stack(values, dim=-1).mean(dim=-1)
    return out


def test_chain_mean_iptm_matches_upstream_averaging(openfold3_source: Path) -> None:
    from foldjax.models.openfold3.models.confidence import compute_chain_mean_iptm

    torch = _torch()
    n_chain = 3
    pair = torch.rand(N_SAMPLE, n_chain, n_chain)
    pair = 0.5 * (pair + pair.transpose(-1, -2))
    pair.diagonal(dim1=-2, dim2=-1).zero_()

    # Chain 2 has no framed tokens.
    asym_id = torch.tensor([0, 0, 1, 1, 2, 2])
    token_mask = torch.ones(N_TOKEN)
    has_frame = torch.ones(N_SAMPLE, N_TOKEN)
    has_frame[:, 4:] = 0.0
    framed = [True, True, False]

    expected = _chain_mean_reference(torch, pair, framed, n_chain)
    actual = compute_chain_mean_iptm(
        jnp.asarray(pair.numpy()),
        jnp.asarray(has_frame.numpy()),
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(asym_id.numpy()),
        n_chain=n_chain,
    )
    assert actual.shape == (N_SAMPLE, n_chain)
    _close(actual, expected, "compute_chain_mean_iptm")


def test_unframed_chain_still_scores_from_framed_partners(
    openfold3_source: Path,
) -> None:
    """The column half of the average is per-partner, so chain 2 is not zeroed."""
    from foldjax.models.openfold3.models.confidence import compute_chain_mean_iptm

    torch = _torch()
    n_chain = 3
    pair = torch.full((1, n_chain, n_chain), 0.5)
    pair.diagonal(dim1=-2, dim2=-1).zero_()
    asym_id = torch.tensor([0, 0, 1, 1, 2, 2])
    has_frame = torch.ones(1, N_TOKEN)
    has_frame[:, 4:] = 0.0

    actual = np.asarray(
        compute_chain_mean_iptm(
            jnp.asarray(pair.numpy()),
            jnp.asarray(has_frame.numpy()),
            jnp.ones(N_TOKEN),
            jnp.asarray(asym_id.numpy()),
            n_chain=n_chain,
        )
    )
    # Chain 2 has no frame, so its rows are excluded, but chains 0 and 1 are
    # framed and contribute through the columns.
    assert actual[0, 2] > 0.0


def _bespoke_reference(torch, chain_mean, is_ligand_chain, n_chain):
    """Upstream's ligand-aware branch, transcribed."""
    out = torch.zeros(chain_mean.shape[0], n_chain, n_chain)
    for i in range(n_chain):
        for j in range(n_chain):
            if i == j:
                continue
            if is_ligand_chain[i]:
                out[:, i, j] = chain_mean[:, i]
            elif is_ligand_chain[j]:
                out[:, i, j] = chain_mean[:, j]
            else:
                out[:, i, j] = 0.5 * (chain_mean[:, i] + chain_mean[:, j])
    return out


def test_bespoke_iptm_matches_upstream_branching(openfold3_source: Path) -> None:
    from foldjax.models.openfold3.models.confidence import compute_bespoke_iptm

    torch = _torch()
    n_chain = 3
    chain_mean = torch.rand(N_SAMPLE, n_chain)
    # Chain 1 is a ligand: both of its tokens are ligand tokens.
    asym_id = torch.tensor([0, 0, 1, 1, 2, 2])
    is_ligand = torch.zeros(N_TOKEN)
    is_ligand[2:4] = 1.0
    token_mask = torch.ones(N_TOKEN)

    expected = _bespoke_reference(
        torch, chain_mean, [False, True, False], n_chain
    )
    actual = compute_bespoke_iptm(
        jnp.asarray(chain_mean.numpy()),
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(asym_id.numpy()),
        jnp.asarray(is_ligand.numpy()),
        n_chain=n_chain,
    )
    assert actual.shape == (N_SAMPLE, n_chain, n_chain)
    _close(actual, expected, "compute_bespoke_iptm")


def test_bespoke_iptm_ligand_takes_over_the_pair_score(
    openfold3_source: Path,
) -> None:
    """A protein-ligand pair uses the ligand's mean, not the average."""
    from foldjax.models.openfold3.models.confidence import compute_bespoke_iptm

    torch = _torch()
    chain_mean = torch.tensor([[0.2, 0.9]])  # chain 0 protein, chain 1 ligand
    asym_id = torch.tensor([0, 0, 1, 1, 1, 1])
    is_ligand = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    actual = np.asarray(
        compute_bespoke_iptm(
            jnp.asarray(chain_mean.numpy()),
            jnp.ones(N_TOKEN),
            jnp.asarray(asym_id.numpy()),
            jnp.asarray(is_ligand.numpy()),
            n_chain=2,
        )
    )
    # Both off-diagonal entries take the ligand chain's mean, not 0.55.
    assert actual[0, 0, 1] == pytest.approx(0.9)
    assert actual[0, 1, 0] == pytest.approx(0.9)


def test_bespoke_iptm_protein_pair_averages(openfold3_source: Path) -> None:
    from foldjax.models.openfold3.models.confidence import compute_bespoke_iptm

    torch = _torch()
    chain_mean = torch.tensor([[0.2, 0.8]])
    asym_id = torch.tensor([0, 0, 0, 1, 1, 1])
    actual = np.asarray(
        compute_bespoke_iptm(
            jnp.asarray(chain_mean.numpy()),
            jnp.ones(N_TOKEN),
            jnp.asarray(asym_id.numpy()),
            jnp.zeros(N_TOKEN),
            n_chain=2,
        )
    )
    assert actual[0, 0, 1] == pytest.approx(0.5)
    assert np.all(np.diagonal(actual, axis1=-2, axis2=-1) == 0.0)
