"""Torch-vs-JAX parity for inter-chain clash detection."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.clash import compute_has_clash

pytestmark = pytest.mark.torch_parity

N_CHAIN = 3


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.metrics.sample_ranking import compute_has_clash

    return compute_has_clash


def _run_both(
    torch,
    asym_id,
    positions,
    atom_mask,
    is_polymer,
    *,
    threshold=1.1,
    violation_abs=100,
    violation_frac=0.5,
):
    expected = _upstream()(
        asym_id=asym_id,
        atom_positions_predicted=positions,
        atom_mask=atom_mask.bool(),
        is_polymer=is_polymer.bool(),
        threshold=threshold,
        violation_abs=violation_abs,
        violation_frac=violation_frac,
    )
    actual = compute_has_clash(
        jnp.asarray(asym_id.numpy()),
        jnp.asarray(positions.numpy()),
        jnp.asarray(atom_mask.numpy()),
        jnp.asarray(is_polymer.numpy()),
        n_chain=N_CHAIN,
        threshold=threshold,
        violation_abs=violation_abs,
        violation_frac=violation_frac,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    return np.asarray(actual)


def _scene(torch, *, overlap: bool):
    """Two 4-atom chains plus a 2-atom chain, optionally overlapping."""
    asym_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    atom_mask = torch.ones(10)
    is_polymer = torch.ones(10)
    base = torch.arange(4, dtype=torch.float32)[:, None].repeat(1, 3)
    far = base + 100.0
    near = base + (0.1 if overlap else 0.0)
    third = base[:2] + 500.0
    positions = torch.cat([base, near if overlap else far, third], dim=0)
    return asym_id, positions.unsqueeze(0), atom_mask, is_polymer


def test_clashing_and_non_clashing_scenes_match_torch(
    openfold3_source: Path,
) -> None:
    torch = _torch()
    # A low fraction threshold makes the small overlap trip the flag.
    clashing = _run_both(
        torch, *_scene(torch, overlap=True), violation_frac=0.1
    )
    separated = _run_both(
        torch, *_scene(torch, overlap=False), violation_frac=0.1
    )
    assert clashing[0] == 1.0
    assert separated[0] == 0.0


def test_masked_atoms_are_excluded(openfold3_source: Path) -> None:
    """Masking out the overlapping atoms must remove the clash."""
    torch = _torch()
    asym_id, positions, atom_mask, is_polymer = _scene(torch, overlap=True)
    atom_mask = atom_mask.clone()
    atom_mask[4:8] = 0.0
    result = _run_both(
        torch, asym_id, positions, atom_mask, is_polymer, violation_frac=0.1
    )
    assert result[0] == 0.0


def test_non_polymer_chains_are_ignored(openfold3_source: Path) -> None:
    """A ligand chain cannot contribute a clash, however close it sits."""
    torch = _torch()
    asym_id, positions, atom_mask, is_polymer = _scene(torch, overlap=True)
    is_polymer = is_polymer.clone()
    is_polymer[4:8] = 0.0  # chain 1 becomes non-polymer
    result = _run_both(
        torch, asym_id, positions, atom_mask, is_polymer, violation_frac=0.1
    )
    assert result[0] == 0.0


def test_multiple_samples_are_scored_independently(openfold3_source: Path) -> None:
    torch = _torch()
    asym_id, clash_pos, atom_mask, is_polymer = _scene(torch, overlap=True)
    _asym, clean_pos, _mask, _poly = _scene(torch, overlap=False)
    positions = torch.cat([clash_pos, clean_pos], dim=0)
    result = _run_both(
        torch, asym_id, positions, atom_mask, is_polymer, violation_frac=0.1
    )
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_absolute_and_fractional_criteria_both_apply(
    openfold3_source: Path,
) -> None:
    """Either criterion alone must be able to trip the flag."""
    torch = _torch()
    scene = _scene(torch, overlap=True)
    # Fractional only: 16 clashes over 4 atoms is 4.0 > 0.5, but < 100 absolute.
    frac_only = _run_both(torch, *scene, violation_abs=100, violation_frac=0.5)
    assert frac_only[0] == 1.0
    # Neither: raise both bars out of reach.
    neither = _run_both(torch, *scene, violation_abs=10_000, violation_frac=99.0)
    assert neither[0] == 0.0


def test_threshold_controls_sensitivity(openfold3_source: Path) -> None:
    torch = _torch()
    scene = _scene(torch, overlap=True)
    tight = _run_both(torch, *scene, threshold=0.01, violation_frac=0.1)
    loose = _run_both(torch, *scene, threshold=5.0, violation_frac=0.1)
    assert tight[0] == 0.0
    assert loose[0] == 1.0
