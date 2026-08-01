"""Pure JAX port of Boltz-2 steering potentials.

Mirrors ``boltz.model.potentials.potentials`` and ``...schedules``. Potentials
are pure functions of coords + feats + scalar parameters (no learned weights,
no checkpoint). Each potential exposes:

    compute(coords, feats, parameters)          -> energy per batch sample
    compute_gradient(coords, feats, parameters) -> gradient wrt coords
    compute_parameters(t)                        -> dict of resolved params

The analytic ``compute_gradient`` math from the PyTorch reference is replicated
directly to guarantee numerical parity (verified against the torch reference in
tests/test_potentials_parity.py).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models.trunk_blocks.trunk import _weighted_rigid_align

# Vendored from boltz.data.const (the reference package is not a runtime dep).
# Kept in sync with boltz/src/boltz/data/const.py.
_NUM_ELEMENTS = 128
_VDW_RADII = [
    1.2,
    1.4,
    2.2,
    1.9,
    1.8,
    1.7,
    1.6,
    1.55,
    1.5,
    1.54,
    2.4,
    2.2,
    2.1,
    2.1,
    1.95,
    1.8,
    1.8,
    1.88,
    2.8,
    2.4,
    2.3,
    2.15,
    2.05,
    2.05,
    2.05,
    2.05,
    2.0,
    2.0,
    2.0,
    2.1,
    2.1,
    2.1,
    2.05,
    1.9,
    1.9,
    2.02,
    2.9,
    2.55,
    2.4,
    2.3,
    2.15,
    2.1,
    2.05,
    2.05,
    2.0,
    2.05,
    2.1,
    2.2,
    2.2,
    2.25,
    2.2,
    2.1,
    2.1,
    2.16,
    3.0,
    2.7,
    2.5,
    2.48,
    2.47,
    2.45,
    2.43,
    2.42,
    2.4,
    2.38,
    2.37,
    2.35,
    2.33,
    2.32,
    2.3,
    2.28,
    2.27,
    2.25,
    2.2,
    2.1,
    2.05,
    2.0,
    2.0,
    2.05,
    2.1,
    2.05,
    2.2,
    2.3,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.4,
    2.0,
    2.3,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
]


class _Const:
    num_elements = _NUM_ELEMENTS
    vdw_radii = _VDW_RADII


const = _Const()

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


class ParameterSchedule(ABC):
    def compute(self, t):  # noqa: D401
        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t):
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (
                math.exp(self.alpha * t) - 1
            ) / (math.exp(self.alpha) - 1)
        return self.start + (self.end - self.start) * t


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1
        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INF = float("inf")


def _np(x):
    """Coerce a torch/jax/numpy tensor (possibly batched) to numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _scatter_mean_coords(coords, com_index, n_out):
    """Scatter-reduce mean of coords (..., n, 3) into (..., n_out, 3)."""
    com_index = np.asarray(com_index)
    out = jnp.zeros((*coords.shape[:-2], n_out, 3), dtype=coords.dtype)
    counts = jnp.zeros((n_out,), dtype=coords.dtype)
    idx = jnp.asarray(com_index)
    out = out.at[..., idx, :].add(coords)
    counts = counts.at[idx].add(1.0)
    counts = jnp.where(counts == 0, 1.0, counts)
    return out / counts[:, None]


def _scatter_sum_atoms(grad_atom_shape, flat_index, prod, dtype):
    """Scatter-sum prod into atom grad tensor along the atom axis (-2)."""
    out = jnp.zeros(grad_atom_shape, dtype=dtype)
    idx = jnp.asarray(np.asarray(flat_index))
    out = out.at[..., idx, :].add(prod)
    return out


# ---------------------------------------------------------------------------
# Base potential
# ---------------------------------------------------------------------------


class Potential(ABC):
    def __init__(self, parameters=None):
        self.parameters = parameters

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        return {
            name: (p.compute(t) if isinstance(p, ParameterSchedule) else p)
            for name, p in self.parameters.items()
        }

    # -- core forward ------------------------------------------------------
    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        index = np.asarray(index)
        if index.shape[1] == 0:
            return jnp.zeros(coords.shape[:-2], dtype=coords.dtype)

        coords, com_index, ref_token_index = self._apply_reductions(
            coords, com_args, ref_args
        )
        ref_coords, ref_mask = (None, None)
        if ref_args is not None:
            ref_coords, ref_mask, _, _ = ref_args
        negation_mask = union_index = None
        if operator_args is not None:
            negation_mask, union_index = operator_args

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            union_index = np.asarray(union_index)
            lam = parameters["union_lambda"]
            neg_exp_energy = jnp.exp(-lam * energy)
            n_u = int(union_index.max()) + 1
            z_part = jnp.zeros((*energy.shape[:-1], n_u), dtype=energy.dtype)
            z_part = z_part.at[..., jnp.asarray(union_index)].add(neg_exp_energy)
            z_sel = z_part[..., jnp.asarray(union_index)]
            softmax_energy = jnp.where(
                z_sel == 0,
                0.0,
                neg_exp_energy / jnp.where(z_sel == 0, 1.0, z_sel),
            )
            return (energy * softmax_energy).sum(axis=-1)

        return energy.sum(axis=tuple(range(1, energy.ndim)))

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        index = np.asarray(index)
        if index.shape[1] == 0:
            return jnp.zeros_like(coords)

        coords, com_index, ref_token_index = self._apply_reductions(
            coords, com_args, ref_args
        )
        ref_coords, ref_mask = (None, None)
        if ref_args is not None:
            ref_coords, ref_mask, _, _ = ref_args
        negation_mask = union_index = None
        if operator_args is not None:
            negation_mask, union_index = operator_args

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, d_energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=True
        )

        n_terms = grad_value.shape[-3]  # number of atoms per interaction
        if union_index is not None:
            union_index = np.asarray(union_index)
            lam = parameters["union_lambda"]
            neg_exp_energy = jnp.exp(-lam * energy)
            n_u = int(union_index.max()) + 1
            ui = jnp.asarray(union_index)
            z_part = jnp.zeros((*energy.shape[:-1], n_u), dtype=energy.dtype)
            z_part = z_part.at[..., ui].add(neg_exp_energy)
            z_sel = z_part[..., ui]
            softmax_energy = jnp.where(
                z_sel == 0,
                0.0,
                neg_exp_energy / jnp.where(z_sel == 0, 1.0, z_sel),
            )
            f = jnp.zeros((*energy.shape[:-1], n_u), dtype=energy.dtype)
            f = f.at[..., ui].add(energy * softmax_energy)
            d_softmax = d_energy * softmax_energy * (1 + lam * (energy - f[..., ui]))
            d = d_softmax
        else:
            d = d_energy

        # prod = d (tiled over n_terms) * flattened grad_value
        prod = _tile_last(d, n_terms)[..., None] * _flatten_terms(grad_value)
        while prod.ndim > 3:
            prod = prod.sum(axis=tuple(range(1, prod.ndim - 2)))

        flat_index = index.reshape(-1)  # (n_terms * n_interactions,)
        grad_atom = _scatter_sum_atoms(coords.shape, flat_index, prod, coords.dtype)

        if com_index is not None:
            grad_atom = grad_atom[..., jnp.asarray(com_index), :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., jnp.asarray(ref_token_index), :]
        return grad_atom

    # -- shared reduction handling ----------------------------------------
    def _apply_reductions(self, coords, com_args, ref_args):
        com_index = None
        ref_token_index = None
        if com_args is not None:
            com_index, atom_pad_mask = com_args
            com_index = np.asarray(com_index)
            atom_pad_mask = np.asarray(atom_pad_mask).astype(bool)
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            n_out = int(unpad_com_index.max()) + 1
            coords = _scatter_mean_coords(unpad_coords, unpad_com_index, n_out)
        if ref_args is not None:
            _, _, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., np.asarray(ref_atom_index), :]
        return coords, com_index, ref_token_index

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, feats, parameters):
        raise NotImplementedError


def _tile_last(x, n):
    """torch ``x.tile(n)`` over last dim: repeat (not interleave) n times."""
    return jnp.tile(x, n)


def _flatten_terms(grad_value):
    """Flatten interaction gradients into the torch reference layout.

    Mirrors torch ``grad_value.flatten(start_dim=-3, end_dim=-2)``.
    """
    return grad_value.reshape(*grad_value.shape[:-3], -1, grad_value.shape[-1])


# ---------------------------------------------------------------------------
# Function mixins
# ---------------------------------------------------------------------------


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        value = value
        if lower_bounds is None:
            lower_bounds = jnp.full(value.shape, -INF, dtype=value.dtype)
        else:
            lower_bounds = jnp.broadcast_to(
                jnp.asarray(lower_bounds, value.dtype), value.shape
            )
        if upper_bounds is None:
            upper_bounds = jnp.full(value.shape, INF, dtype=value.dtype)
        else:
            upper_bounds = jnp.broadcast_to(
                jnp.asarray(upper_bounds, value.dtype), value.shape
            )
        k = jnp.broadcast_to(jnp.asarray(k, value.dtype), value.shape)

        if negation_mask is not None:
            negation_mask = jnp.asarray(np.asarray(negation_mask)).astype(bool)
            negation_mask = jnp.broadcast_to(negation_mask, value.shape)
            unbounded_below = jnp.isneginf(lower_bounds)
            unbounded_above = jnp.isposinf(upper_bounds)
            # torch asserts every entry is unbounded on one side or negated.
            sel_a = (~unbounded_above) & (~negation_mask)
            sel_b = (~unbounded_below) & (~negation_mask)
            new_lower = jnp.where(sel_a, upper_bounds, lower_bounds)
            new_upper = jnp.where(sel_a, INF, upper_bounds)
            new_upper = jnp.where(sel_b, new_lower, new_upper)
            new_lower = jnp.where(sel_b, -INF, new_lower)
            lower_bounds, upper_bounds = new_lower, new_upper

        neg_overflow = value < lower_bounds
        pos_overflow = value > upper_bounds

        energy = jnp.where(neg_overflow, k * (lower_bounds - value), 0.0)
        energy = jnp.where(pos_overflow, k * (value - upper_bounds), energy)
        if not compute_derivative:
            return energy

        d_energy = jnp.where(neg_overflow, -k, 0.0)
        d_energy = jnp.where(pos_overflow, k, d_energy)
        return energy, d_energy


# ---------------------------------------------------------------------------
# Variable mixins
# ---------------------------------------------------------------------------


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        i = jnp.asarray(np.asarray(index[0]))
        j = jnp.asarray(np.asarray(index[1]))
        r_ij = coords[..., i, :] - coords[..., j, :]
        r_norm = jnp.linalg.norm(r_ij, axis=-1)
        r_hat = r_ij / r_norm[..., None]
        if not compute_gradient:
            return r_norm
        grad = jnp.stack((r_hat, -r_hat), axis=-3)
        return r_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        idx = [jnp.asarray(np.asarray(index[a])) for a in range(4)]
        r_ij = coords[..., idx[0], :] - coords[..., idx[1], :]
        r_kj = coords[..., idx[2], :] - coords[..., idx[1], :]
        r_kl = coords[..., idx[2], :] - coords[..., idx[3], :]

        n_ijk = jnp.cross(r_ij, r_kj, axis=-1)
        n_jkl = jnp.cross(r_kj, r_kl, axis=-1)

        r_kj_norm = jnp.linalg.norm(r_kj, axis=-1)
        n_ijk_norm = jnp.linalg.norm(n_ijk, axis=-1)
        n_jkl_norm = jnp.linalg.norm(n_jkl, axis=-1)

        sign_phi = jnp.sign((r_kj * jnp.cross(n_ijk, n_jkl, axis=-1)).sum(axis=-1))
        cos_phi = (n_ijk * n_jkl).sum(axis=-1) / (n_ijk_norm * n_jkl_norm)
        phi = sign_phi * jnp.arccos(jnp.clip(cos_phi, -1 + 1e-8, 1 - 1e-8))

        if not compute_gradient:
            return phi

        a = ((r_ij * r_kj).sum(axis=-1) / (r_kj_norm**2))[..., None]
        b = ((r_kl * r_kj).sum(axis=-1) / (r_kj_norm**2))[..., None]

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2)[..., None]
        grad_l = -n_jkl * (r_kj_norm / n_jkl_norm**2)[..., None]
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = jnp.stack((grad_i, grad_j, grad_k, grad_l), axis=-3)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(coords, index, compute_gradient=False)
            return jnp.abs(phi)
        phi, grad = super().compute_variable(coords, index, compute_gradient=True)
        # flip gradient where phi < 0 (broadcast over the term axis -3)
        flip = (phi < 0)[..., None, :, None]
        grad = jnp.where(flip, -grad, grad)
        return jnp.abs(phi), grad


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        idx = jnp.asarray(np.asarray(index))
        sel = coords[:, idx]  # (b, 1, n, 3) since index has leading dim
        sel = sel[:, 0] if idx.ndim == 2 else sel
        ref_coords = jnp.asarray(np.asarray(ref_coords)).astype(jnp.float32)
        ref_mask_j = jnp.asarray(np.asarray(ref_mask)).astype(jnp.float32)
        coords_sel = coords[:, idx[0]] if idx.ndim == 2 else coords[:, idx]
        aligned = _weighted_rigid_align(
            ref_coords, coords_sel.astype(jnp.float32), ref_mask_j, ref_mask_j
        )
        r = coords_sel - aligned
        r_norm = jnp.linalg.norm(r, axis=-1)
        if not compute_gradient:
            return r_norm
        r_hat = r / r_norm[..., None]
        grad = (r_hat * ref_mask_j[..., None])[:, None]
        return r_norm, grad


# ---------------------------------------------------------------------------
# Concrete potentials
# ---------------------------------------------------------------------------


def _atom_vdw_radii(feats):
    vdw = np.zeros(const.num_elements, dtype=np.float32)
    vdw[1:119] = np.asarray(const.vdw_radii, dtype=np.float32)
    ref_element = _np(feats["ref_element"]).astype(np.float32)[0]
    return ref_element @ vdw


def _atom_chain_id(feats):
    a2t = _np(feats["atom_to_token"]).astype(np.float32)
    asym = _np(feats["asym_id"]).astype(np.float32)
    return (a2t @ asym[..., None]).squeeze(-1).astype(np.int64)[0]


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = _np(feats["rdkit_bounds_index"])[0]
        lower = _np(feats["rdkit_lower_bounds"])[0].astype(np.float64).copy()
        upper = _np(feats["rdkit_upper_bounds"])[0].astype(np.float64).copy()
        bond_mask = _np(feats["rdkit_bounds_bond_mask"])[0].astype(bool)
        angle_mask = _np(feats["rdkit_bounds_angle_mask"])[0].astype(bool)

        bb, ab, cb = (
            parameters["bond_buffer"],
            parameters["angle_buffer"],
            parameters["clash_buffer"],
        )
        m_b = bond_mask & ~angle_mask
        m_a = ~bond_mask & angle_mask
        m_ba = bond_mask & angle_mask
        m_n = ~bond_mask & ~angle_mask
        lower[m_b] *= 1.0 - bb
        upper[m_b] *= 1.0 + bb
        lower[m_a] *= 1.0 - ab
        upper[m_a] *= 1.0 + ab
        lower[m_ba] *= 1.0 - min(bb, ab)
        upper[m_ba] *= 1.0 + min(bb, ab)
        lower[m_n] *= 1.0 - cb
        upper[m_n] = INF

        atom_vdw = _atom_vdw_radii(feats)
        bond_cutoffs = 0.35 + atom_vdw[pair_index].mean(axis=0)
        lower[~bond_mask] = np.maximum(lower[~bond_mask], bond_cutoffs[~bond_mask])
        upper[bond_mask] = np.minimum(upper[bond_mask], bond_cutoffs[bond_mask])

        k = np.ones_like(lower)
        return (
            pair_index,
            (k.astype(np.float32), lower.astype(np.float32), upper.astype(np.float32)),
            None,
            None,
            None,
        )


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = _np(feats["connected_atom_index"])[0]
        upper = np.full((pair_index.shape[1],), parameters["buffer"], dtype=np.float32)
        k = np.ones_like(upper)
        return pair_index, (k, None, upper), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = _atom_chain_id(feats)
        atom_pad_mask = _np(feats["atom_pad_mask"])[0].astype(bool)
        chain_sizes = np.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]
        atom_vdw = _atom_vdw_radii(feats)

        n = atom_chain_id.shape[0]
        pair_index = np.array(np.triu_indices(n, 1))
        pair_pad = atom_pad_mask[pair_index].all(axis=0)
        pair_ion = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = int(atom_chain_id.max()) + 1
        cc_idx = _np(feats["connected_chain_index"])[0]
        cc_mat = np.eye(num_chains, dtype=bool)
        cc_mat[cc_idx[0], cc_idx[1]] = True
        cc_mat[cc_idx[1], cc_idx[0]] = True
        cc_mask = cc_mat[atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]]

        pair_index = pair_index[:, pair_pad & pair_ion & ~cc_mask]
        lower = atom_vdw[pair_index].sum(axis=0) * (1.0 - parameters["buffer"])
        k = np.ones_like(lower).astype(np.float32)
        return pair_index, (k, lower.astype(np.float32), None), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = _atom_chain_id(feats)
        atom_pad_mask = _np(feats["atom_pad_mask"])[0].astype(bool)
        chain_sizes = np.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = _np(feats["symmetric_chain_index"])[0]
        pair_ion = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion]
        lower = np.full((pair_index.shape[1],), parameters["buffer"], dtype=np.float32)
        k = np.ones_like(lower)
        return (
            pair_index,
            (k, lower, None),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        index = _np(feats["stereo_bond_index"])[0]
        orient = _np(feats["stereo_bond_orientations"])[0].astype(bool)
        lower = np.zeros(orient.shape, dtype=np.float32)
        upper = np.zeros(orient.shape, dtype=np.float32)
        lower[orient] = np.pi - parameters["buffer"]
        upper[orient] = INF
        lower[~orient] = -INF
        upper[~orient] = parameters["buffer"]
        k = np.ones_like(lower)
        return index, (k, lower, upper), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        index = _np(feats["chiral_atom_index"])[0]
        orient = _np(feats["chiral_atom_orientations"])[0].astype(bool)
        lower = np.zeros(orient.shape, dtype=np.float32)
        upper = np.zeros(orient.shape, dtype=np.float32)
        lower[orient] = parameters["buffer"]
        upper[orient] = INF
        upper[~orient] = -parameters["buffer"]
        lower[~orient] = -INF
        k = np.ones_like(lower)
        return index, (k, lower, upper), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = _np(feats["planar_bond_index"])[0].T
        improper = np.array([[1, 2, 3, 0], [4, 5, 0, 3]]).T
        improper_index = np.swapaxes(double_bond_index[:, improper], 0, 1).reshape(
            double_bond_index[:, improper].shape[1], -1
        )
        upper = np.full(
            (improper_index.shape[1],), parameters["buffer"], dtype=np.float32
        )
        k = np.ones_like(upper)
        return improper_index, (k, None, upper), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if "template_mask_cb" not in feats or "template_force" not in feats:
            return np.empty([1, 0], dtype=np.int64), None, None, None, None
        force = _np(feats["template_force"]).astype(bool)
        template_mask = _np(feats["template_mask_cb"])[force]
        if template_mask.shape[0] == 0:
            return np.empty([1, 0], dtype=np.int64), None, None, None, None

        ref_coords = _np(feats["template_cb"])[force].astype(np.float32).copy()
        ref_mask = _np(feats["template_mask_cb"])[force].astype(np.float32).copy()
        n_atoms = _np(feats["atom_pad_mask"]).shape[1]
        t2r = _np(feats["token_to_rep_atom"]).astype(np.float32)
        ref_atom_index = (
            (t2r @ np.arange(n_atoms, dtype=np.float32)[None, :, None])
            .squeeze(-1)
            .astype(np.int64)[0]
        )
        a2t = _np(feats["atom_to_token"]).astype(np.float32)
        tok_idx = _np(feats["token_index"]).astype(np.float32)
        ref_token_index = (a2t @ tok_idx[..., None]).squeeze(-1).astype(np.int64)[0]
        index = np.arange(template_mask.shape[-1], dtype=np.int64)[None]
        upper = np.full(template_mask.shape, INF, dtype=np.float32)
        thr = _np(feats["template_force_threshold"])[force]
        ref_idxs = np.argwhere(template_mask.astype(bool)).T
        upper[tuple(ref_idxs)] = thr[ref_idxs[0]]
        k = np.ones_like(upper)
        return (
            index,
            (k, None, upper),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = _np(feats["contact_pair_index"])[0]
        union_index = _np(feats["contact_union_index"])[0]
        negation_mask = _np(feats["contact_negation_mask"])[0]
        upper = _np(feats["contact_thresholds"])[0].astype(np.float32).copy()
        k = np.ones_like(upper)
        return index, (k, None, upper), None, None, (negation_mask, union_index)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_potentials(steering_args, boltz2=True):
    potentials = []
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        phys = steering_args["physical_guidance_update"]
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5 if phys else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if phys
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15 if phys else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.01 if phys else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1 if phys else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if phys else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05 if phys else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        contact = steering_args["contact_guidance_update"]
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if contact
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1 if contact else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )
    return potentials
