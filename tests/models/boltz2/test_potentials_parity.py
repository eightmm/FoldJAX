"""Numerical parity tests: JAX potentials vs PyTorch reference.

Builds small synthetic coords + feats mirroring the keys each potential's
compute_args reads, then compares JAX ``compute`` / ``compute_gradient`` against
the boltz torch reference. Tolerances rtol/atol = 1e-4 (relaxed where noted).
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
if BOLTZ_SRC.is_dir():
    sys.path.insert(0, str(BOLTZ_SRC))

TP = pytest.importorskip("boltz.model.potentials.potentials")
TS = pytest.importorskip("boltz.model.potentials.schedules")

from foldjax.models.boltz2.models.heads import potentials as P  # noqa: E402,N812

RTOL = 1e-4
ATOL = 1e-4
RNG = np.random.default_rng(7)


def _t(x):
    return torch.from_numpy(np.asarray(x))


def _feats_torch(feats):
    out = {}
    for k, v in feats.items():
        v = np.asarray(v)
        out[k] = torch.from_numpy(v)
    return out


def _compare(jax_pot, torch_pot, coords, feats, t=0.5):
    jparams = jax_pot.compute_parameters(t)
    tparams = torch_pot.compute_parameters(t)

    tcoords = _t(coords).double() if False else _t(coords).float()
    tfeats = _feats_torch(feats)

    t_energy = torch_pot.compute(tcoords, tfeats, tparams).numpy()
    j_energy = np.asarray(jax_pot.compute(jnp.asarray(coords), feats, jparams))
    np.testing.assert_allclose(
        j_energy,
        t_energy,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{type(jax_pot).__name__} energy",
    )

    t_grad = torch_pot.compute_gradient(tcoords, tfeats, tparams).numpy()
    j_grad = np.asarray(jax_pot.compute_gradient(jnp.asarray(coords), feats, jparams))
    np.testing.assert_allclose(
        j_grad,
        t_grad,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{type(jax_pot).__name__} gradient",
    )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
def test_exponential_interpolation(t):
    for alpha in (-2.0, 0.0, 2.0):
        j = P.ExponentialInterpolation(1.0, 5.0, alpha).compute(t)
        r = TS.ExponentialInterpolation(1.0, 5.0, alpha).compute(t)
        assert abs(j - r) < 1e-9


@pytest.mark.parametrize("t", [0.0, 0.25, 0.3, 0.6, 0.75, 0.8, 1.0])
def test_piecewise_step(t):
    j = P.PiecewiseStepFunction([0.25, 0.75], [0.0, 0.5, 1.0]).compute(t)
    r = TS.PiecewiseStepFunction([0.25, 0.75], [0.0, 0.5, 1.0]).compute(t)
    assert j == r


# ---------------------------------------------------------------------------
# Synthetic feature builders
# ---------------------------------------------------------------------------

B = 2  # batch (multiplicity) samples


def _coords(n):
    return RNG.standard_normal((B, n, 3)).astype(np.float32) * 3.0


def _onehot_elements(n):
    el = np.zeros((1, n, P.const.num_elements), dtype=np.float32)
    choice = RNG.integers(1, 30, size=n)
    el[0, np.arange(n), choice] = 1.0
    return el


def test_pose_busters():
    n = 8
    npair = 10
    idx = RNG.integers(0, n, size=(2, npair)).astype(np.int64)
    lb = (RNG.uniform(1.0, 2.0, npair)).astype(np.float32)
    ub = (lb + RNG.uniform(0.5, 1.5, npair)).astype(np.float32)
    bond = RNG.integers(0, 2, npair).astype(bool)
    angle = RNG.integers(0, 2, npair).astype(bool)
    feats = {
        "rdkit_bounds_index": idx[None],
        "rdkit_lower_bounds": lb[None],
        "rdkit_upper_bounds": ub[None],
        "rdkit_bounds_bond_mask": bond[None],
        "rdkit_bounds_angle_mask": angle[None],
        "ref_element": _onehot_elements(n),
    }
    params = {"bond_buffer": 0.125, "angle_buffer": 0.125, "clash_buffer": 0.10}
    _compare(
        P.PoseBustersPotential(params),
        TP.PoseBustersPotential(params),
        _coords(n),
        feats,
    )


def test_connections():
    n = 8
    idx = RNG.integers(0, n, size=(2, 6)).astype(np.int64)
    feats = {"connected_atom_index": idx[None]}
    params = {"buffer": 2.0}
    _compare(
        P.ConnectionsPotential(params),
        TP.ConnectionsPotential(params),
        _coords(n),
        feats,
    )


def _atom_token_feats(n_atoms, asym):
    n_tok = len(asym)
    a2t = np.zeros((1, n_atoms, n_tok), dtype=np.float32)
    tok = RNG.integers(0, n_tok, size=n_atoms)
    a2t[0, np.arange(n_atoms), tok] = 1.0
    return a2t, np.asarray(asym, dtype=np.float32)[None]


def test_vdw_overlap():
    n = 10
    asym = [0, 0, 0, 1, 1, 1]  # 6 tokens, chains 0 and 1, both multi-atom
    a2t, asym_arr = _atom_token_feats(n, asym)
    feats = {
        "atom_to_token": a2t,
        "asym_id": asym_arr,
        "atom_pad_mask": np.ones((1, n), dtype=np.float32),
        "ref_element": _onehot_elements(n),
        "connected_chain_index": np.zeros((1, 2, 0), dtype=np.int64),
    }
    params = {"buffer": 0.225}
    _compare(
        P.VDWOverlapPotential(params), TP.VDWOverlapPotential(params), _coords(n), feats
    )


def test_symmetric_chain_com():
    n = 12
    asym = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # 3 multi-atom chains
    a2t, asym_arr = _atom_token_feats(n, asym)
    sym = np.array([[0, 1], [1, 2]], dtype=np.int64)  # pairs of chain ids
    feats = {
        "atom_to_token": a2t,
        "asym_id": asym_arr,
        "atom_pad_mask": np.ones((1, n), dtype=np.float32),
        "symmetric_chain_index": sym[None],
    }
    params = {"buffer": 1.0}
    _compare(
        P.SymmetricChainCOMPotential(params),
        TP.SymmetricChainCOMPotential(params),
        _coords(n),
        feats,
    )


def test_stereo_bond():
    n = 10
    idx = np.stack([RNG.permutation(n)[:4] for _ in range(5)], axis=1).astype(np.int64)
    orient = RNG.integers(0, 2, 5).astype(np.int64)
    feats = {"stereo_bond_index": idx[None], "stereo_bond_orientations": orient[None]}
    params = {"buffer": 0.5236}
    _compare(
        P.StereoBondPotential(params), TP.StereoBondPotential(params), _coords(n), feats
    )


def test_chiral_atom():
    n = 10
    idx = np.stack([RNG.permutation(n)[:4] for _ in range(5)], axis=1).astype(np.int64)
    orient = RNG.integers(0, 2, 5).astype(np.int64)
    feats = {"chiral_atom_index": idx[None], "chiral_atom_orientations": orient[None]}
    params = {"buffer": 0.5236}
    _compare(
        P.ChiralAtomPotential(params), TP.ChiralAtomPotential(params), _coords(n), feats
    )


def test_planar_bond():
    n = 10
    # planar_bond_index: (1, 6, nbonds) -> .T per sample is (nbonds, 6)
    nbonds = 3
    pbi = np.stack([RNG.permutation(n)[:6] for _ in range(nbonds)], axis=1).astype(
        np.int64
    )
    feats = {"planar_bond_index": pbi[None]}
    params = {"buffer": 0.2618}
    _compare(
        P.PlanarBondPotential(params), TP.PlanarBondPotential(params), _coords(n), feats
    )


def test_contact():
    n = 10
    npair = 8
    idx = RNG.integers(0, n, size=(2, npair)).astype(np.int64)
    union = np.repeat(np.arange(4), 2).astype(np.int64)  # 4 unions of 2
    neg = np.zeros(npair, dtype=bool)
    # Large thresholds keep contact energies small so exp(-lambda*E) does not
    # underflow to 0 in either backend (the torch reference yields NaN when a
    # whole union underflows, which the JAX port guards against; we avoid that
    # degenerate regime here to test the common path).
    thr = RNG.uniform(2.0, 5.0, npair).astype(np.float32)
    feats = {
        "contact_pair_index": idx[None],
        "contact_union_index": union[None],
        "contact_negation_mask": neg[None],
        "contact_thresholds": thr[None],
    }
    params = {"union_lambda": 0.1}
    _compare(
        P.ContactPotentital(params), TP.ContactPotentital(params), _coords(n), feats
    )


def test_template_reference():
    n_tok = 6
    n_atoms = 6
    # one forced template
    t2r = np.eye(n_atoms, dtype=np.float32)[None]  # token i -> atom i
    a2t = np.eye(n_atoms, dtype=np.float32)[None]
    mask = np.ones((1, n_tok), dtype=np.float32)
    cb = RNG.standard_normal((1, n_tok, 3)).astype(np.float32)
    feats = {
        "template_mask_cb": mask,
        "template_force": np.array([True]),
        "template_cb": cb,
        "atom_pad_mask": np.ones((1, n_atoms), dtype=np.float32),
        "token_to_rep_atom": t2r,
        "atom_to_token": a2t,
        "token_index": np.arange(n_tok, dtype=np.float32)[None],
        "template_force_threshold": np.array([2.0], dtype=np.float32),
    }
    coords = _coords(n_atoms)
    jpot = P.TemplateReferencePotential({})
    tpot = TP.TemplateReferencePotential({})
    # weighted_rigid_align via SVD: relax slightly
    jparams = jpot.compute_parameters(0.5)
    tparams = tpot.compute_parameters(0.5)
    te = tpot.compute(_t(coords).float(), _feats_torch(feats), tparams).numpy()
    je = np.asarray(jpot.compute(jnp.asarray(coords), feats, jparams))
    np.testing.assert_allclose(je, te, rtol=1e-3, atol=1e-3)
    tg = tpot.compute_gradient(_t(coords).float(), _feats_torch(feats), tparams).numpy()
    jg = np.asarray(jpot.compute_gradient(jnp.asarray(coords), feats, jparams))
    np.testing.assert_allclose(jg, tg, rtol=1e-3, atol=1e-3)
