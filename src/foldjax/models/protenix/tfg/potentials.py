"""JAX implementations of Protenix training-free guidance potentials."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

CLASS_REGISTRY: dict[str, type[Potential]] = {}

_VDW_RADII = jnp.asarray(
    [
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
    ],
    dtype=jnp.float32,
)


def register(cls: type[Potential]) -> type[Potential]:
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


def _zero_energy(coords: jax.Array) -> jax.Array:
    return jnp.zeros(coords.shape[:-2], dtype=coords.dtype)


def _sum_constraints(value: jax.Array) -> jax.Array:
    return value if value.ndim == 0 else value.sum(axis=-1)


def _distance(coords: jax.Array, index: jax.Array) -> jax.Array:
    displacement = coords[..., index[0], :] - coords[..., index[1], :]
    return jnp.maximum(jnp.linalg.norm(displacement, axis=-1), 1.0e-8)


def _angle(coords: jax.Array, index: jax.Array) -> jax.Array:
    ji = coords[..., index[0], :] - coords[..., index[1], :]
    jk = coords[..., index[2], :] - coords[..., index[1], :]
    cross = jnp.cross(ji, jk)
    dot = jnp.sum(ji * jk, axis=-1) + 1.0e-8
    return jnp.arctan2(jnp.linalg.norm(cross, axis=-1), dot)


def _dihedral(coords: jax.Array, index: jax.Array) -> jax.Array:
    ij = coords[..., index[1], :] - coords[..., index[0], :]
    kj = coords[..., index[1], :] - coords[..., index[2], :]
    kl = coords[..., index[3], :] - coords[..., index[2], :]
    m = jnp.cross(ij, kj)
    n = jnp.cross(kj, kl)
    w = jnp.cross(m, n)
    phi = jnp.arctan2(jnp.linalg.norm(w, axis=-1), jnp.sum(m * n, axis=-1) + 1e-8)
    return -phi * jnp.sign(jnp.sum(ij * n, axis=-1))


def _flat_linear(
    value: jax.Array, lower: jax.Array | float | None, upper: jax.Array | float | None
) -> jax.Array:
    energy = jnp.zeros_like(value)
    if lower is not None:
        energy = energy + jnp.maximum(jnp.asarray(lower, value.dtype) - value, 0.0)
    if upper is not None:
        energy = energy + jnp.maximum(value - jnp.asarray(upper, value.dtype), 0.0)
    return energy


def _flat_parabolic(value: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    below = jnp.maximum(lower - value, 0.0)
    above = jnp.maximum(value - upper, 0.0)
    return 0.5 * (below**2 + above**2)


def _constraint_projection(coords, value_fn, active_fn):
    """Minimum-norm linearized projection, including arbitrary batch axes."""
    n_atom = coords.shape[-2]
    flat_coords = coords.reshape((-1, n_atom, 3))

    def project_one(single_coords):
        values = value_fn(single_coords)
        active = active_fn(values)
        jacobian = jax.jacrev(value_fn)(single_coords)
        jacobian = jnp.where(active[:, None, None], jacobian, 0.0)
        rows = jacobian.reshape((jacobian.shape[0], -1))
        system = rows @ rows.T
        diagonal = jnp.where(active, 1.0e-8, 1.0)
        multiplier = jnp.linalg.solve(
            system + jnp.diag(diagonal), jnp.where(active, values, 0.0)
        )
        return (-(rows.T @ multiplier)).reshape(single_coords.shape)

    return jax.vmap(project_one)(flat_coords).reshape(coords.shape)


class Potential:
    """Base potential with a JAX-autodiff coordinate gradient."""

    default_params: Mapping[str, Any] = {}

    def __init__(self, default_params: Mapping[str, Any] | None = None) -> None:
        self._default_params = dict(self.default_params)
        if default_params:
            self._default_params.update(default_params)

    def _params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        result = dict(self._default_params)
        if params:
            result.update(params)
        return result

    def energy(
        self,
        coords: jax.Array,
        feats: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> jax.Array:
        return self._energy(coords, feats, self._params(params))

    def energy_and_grad(
        self,
        coords: jax.Array,
        feats: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        resolved = self._params(params)

        def scalar_energy(value: jax.Array) -> jax.Array:
            return jnp.sum(self._energy(value, feats, resolved))

        energy = self._energy(coords, feats, resolved)
        return energy, jax.grad(scalar_energy)(coords)

    def project(
        self,
        coords: jax.Array,
        feats: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> jax.Array:
        del feats, params
        return jnp.zeros_like(coords)

    def _energy(
        self, coords: jax.Array, feats: Mapping[str, Any], params: Mapping[str, Any]
    ) -> jax.Array:
        raise NotImplementedError


@register
class InterchainBondPotential(Potential):
    default_params = {"buffer": 2.0}

    def _energy(self, coords, feats, params):
        index = feats["interchain_bond_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        return _sum_constraints(
            _flat_linear(_distance(coords, index), None, params["buffer"])
        )


@register
class PairwiseDistancePotential(Potential):
    default_params = {"bond_buffer": 0.05, "angle_buffer": 0.05, "clash_buffer": 0.05}

    def _bounds(self, feats, params):
        index = feats["pairwise_distance_index"]
        is_bond = jnp.asarray(feats["pairwise_distance_is_bond"], dtype=jnp.int32)
        is_angle = jnp.asarray(feats["pairwise_distance_is_angle"], dtype=jnp.int32)
        state = is_bond + 2 * is_angle
        minimum = min(float(params["bond_buffer"]), float(params["angle_buffer"]))
        lower_scales = (
            1.0
            - jnp.asarray(
                [
                    params["clash_buffer"],
                    params["bond_buffer"],
                    params["angle_buffer"],
                    minimum,
                ]
            )[state]
        )
        upper_scales = (
            1.0
            + jnp.asarray(
                [0.0, params["bond_buffer"], params["angle_buffer"], minimum]
            )[state]
        )
        lower = feats["pairwise_distance_lower_bound"] * lower_scales
        upper = feats["pairwise_distance_upper_bound"] * upper_scales
        upper = jnp.where(state == 0, jnp.inf, upper)
        element = jnp.argmax(feats["ref_element"], axis=-1)
        radii = _VDW_RADII[element]
        vdw_limit = 0.35 + 0.5 * (radii[index[0]] + radii[index[1]])
        lower = jnp.where(is_bond == 0, jnp.maximum(lower, vdw_limit), lower)
        upper = jnp.where(is_bond == 1, jnp.minimum(upper, vdw_limit), upper)
        return lower, upper

    def _energy(self, coords, feats, params):
        index = feats["pairwise_distance_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        lower, upper = self._bounds(feats, params)
        return _sum_constraints(_flat_parabolic(_distance(coords, index), lower, upper))

    def project(self, coords, feats, params=None):
        resolved = self._params(params)
        index = feats["pairwise_distance_index"]
        if index.shape[-1] == 0:
            return jnp.zeros_like(coords)
        lower, upper = self._bounds(feats, resolved)

        def apply_projection(value, selected):
            def violations(single_coords):
                distance = _distance(single_coords, index)
                target = jnp.where(distance < lower, lower, upper)
                return distance - target

            def active(violations):
                return selected & (jnp.abs(violations) > 0.0)

            return _constraint_projection(value, violations, active)

        angle_delta = apply_projection(
            coords, jnp.asarray(feats["pairwise_distance_is_angle"], dtype=bool)
        )
        bond_delta = apply_projection(
            coords + angle_delta,
            jnp.asarray(feats["pairwise_distance_is_bond"], dtype=bool),
        )
        return angle_delta + bond_delta


@register
class StereoBondPotential(Potential):
    default_params = {"buffer": 0.52360}

    def _energy(self, coords, feats, params):
        index = feats["stereo_bond_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        value = jnp.abs(_dihedral(coords, index))
        orientation = feats["stereo_bond_orientation"] > 0.5
        lower = jnp.where(orientation, jnp.pi - params["buffer"], -jnp.inf)
        upper = jnp.where(orientation, jnp.inf, params["buffer"])
        return _sum_constraints(_flat_linear(value, lower, upper))


@register
class ChiralAtomPotential(Potential):
    default_params = {"buffer": 0.34906, "scale_x": True}

    def _energy(self, coords, feats, params):
        index = feats["chiral_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        value = _dihedral(coords, index)
        orientation = feats["chiral_orientation"] > 0
        lower = jnp.where(orientation, params["buffer"], -jnp.inf)
        upper = jnp.where(orientation, jnp.inf, -params["buffer"])
        return _sum_constraints(_flat_linear(value, lower, upper))

    def project(self, coords, feats, params=None):
        resolved = self._params(params)
        index = feats["chiral_index"]
        if index.shape[-1] == 0:
            return jnp.zeros_like(coords)
        orientation = jnp.asarray(feats["chiral_orientation"], dtype=coords.dtype)

        def violations(single_coords):
            return _dihedral(single_coords, index) * orientation - resolved["buffer"]

        delta = _constraint_projection(coords, violations, lambda value: value < 0.0)
        if not bool(resolved.get("scale_x", True)):
            return delta

        affected = np.unique(np.asarray(index).reshape(-1))
        atom_to_token = np.asarray(feats["atom_to_token_idx"])
        atom_chain = np.asarray(feats["asym_id"])[atom_to_token]
        for chain in np.unique(atom_chain[affected]):
            atom_indices = affected[atom_chain[affected] == chain]
            selected = coords[..., atom_indices, :]
            selected_delta = delta[..., atom_indices, :]
            center = selected.mean(axis=-2, keepdims=True)
            radius = jnp.mean(
                jnp.sum((selected - center) ** 2, axis=-1),
                axis=-1,
                keepdims=True,
            )[..., None]
            moved = selected + selected_delta
            moved_center = moved.mean(axis=-2, keepdims=True)
            moved_radius = jnp.mean(
                jnp.sum((moved - moved_center) ** 2, axis=-1),
                axis=-1,
                keepdims=True,
            )[..., None]
            moved = (moved - moved_center) * jnp.sqrt(
                radius / jnp.maximum(moved_radius, 1.0e-12)
            ) + center
            delta = delta.at[..., atom_indices, :].set(moved - selected)
        return delta


@register
class PlanarImproperPotential(Potential):
    default_params = {"buffer": 0.1309}

    def _energy(self, coords, feats, params):
        del params
        index = feats["planar_improper_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        p1, p3, center, p4 = (coords[..., index[i], :] for i in range(4))
        ji, jk, jl = p1 - center, p3 - center, p4 - center
        ji = ji / jnp.maximum(jnp.linalg.norm(ji, axis=-1, keepdims=True), 1e-8)
        jk = jk / jnp.maximum(jnp.linalg.norm(jk, axis=-1, keepdims=True), 1e-8)
        jl = jl / jnp.maximum(jnp.linalg.norm(jl, axis=-1, keepdims=True), 1e-8)
        normal = jnp.cross(-ji, jk)
        normal = normal / jnp.maximum(
            jnp.linalg.norm(normal, axis=-1, keepdims=True), 1e-8
        )
        cosine = jnp.clip(jnp.sum(normal * jl, axis=-1), -1.0, 1.0)
        return _sum_constraints(1.0 - jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0)))


@register
class LinearBondPotential(Potential):
    default_params = {"buffer": 0.08726646259}

    def _energy(self, coords, feats, params):
        index = feats["linear_triple_bond_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        return _sum_constraints(
            _flat_linear(_angle(coords, index), jnp.pi - params["buffer"], None)
        )


@register
class ExperimentalTorsionPotential(Potential):
    def _energy(self, coords, feats, params):
        del params
        index = feats["experimental_torsion_index"]
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        phi = _dihedral(coords, index)
        orders = jnp.arange(1, 7, dtype=coords.dtype)
        energy = feats["experimental_torsion_force_constant"] * (
            1.0
            + feats["experimental_torsion_sign"] * jnp.cos(phi[..., :, None] * orders)
        )
        return energy.sum(axis=(-1, -2))


@register
class VinaStericPotential(Potential):
    default_params = {"buffer": 0.225}

    @staticmethod
    def _candidates(feats):
        token_chain = np.asarray(feats["asym_id"])
        atom_to_token = np.asarray(feats["atom_to_token_idx"])
        atom_chain = token_chain[atom_to_token]
        counts = {
            int(chain): int(np.sum(atom_chain == chain))
            for chain in np.unique(atom_chain)
        }
        prohibited: set[tuple[int, int]] = set()
        bonds = np.asarray(feats.get("interchain_bond_index", np.empty((2, 0), int)))
        for left, right in bonds.T:
            pair = tuple(sorted((int(atom_chain[left]), int(atom_chain[right]))))
            prohibited.add(pair)
        pairs = [
            (left, right)
            for left in range(len(atom_chain))
            for right in range(left + 1, len(atom_chain))
            if atom_chain[left] != atom_chain[right]
            and counts[int(atom_chain[left])] > 1
            and counts[int(atom_chain[right])] > 1
            and tuple(sorted((int(atom_chain[left]), int(atom_chain[right]))))
            not in prohibited
        ]
        if not pairs:
            return jnp.empty((2, 0), dtype=jnp.int32), jnp.empty(
                (0,), dtype=jnp.float32
            )
        index = jnp.asarray(pairs, dtype=jnp.int32).T
        element = jnp.argmax(feats["ref_element"], axis=-1)
        radii = _VDW_RADII[element]
        return index, radii[index[0]] + radii[index[1]]

    def _energy(self, coords, feats, params):
        index, equilibrium = self._candidates(feats)
        if index.shape[-1] == 0:
            return _zero_energy(coords)
        distance = _distance(coords, index)
        difference = distance - equilibrium
        norm = difference / 0.5
        g1 = -0.0356 * jnp.exp(-(norm**2))
        g2 = -0.00516 * jnp.exp(-(((difference - 3.0) / 2.0) ** 2))
        repulsion = 0.840 * jnp.where(difference < 0, difference**2, 0.0)
        active = distance < equilibrium * (1.0 - params["buffer"])
        return _sum_constraints(jnp.where(active, g1 + g2 + repulsion, 0.0))
