from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from foldjax.models.protenix.tfg import TFGEngine, parse_tfg_config, schedule_from_cfg
from foldjax.models.protenix.tfg.potentials import (
    CLASS_REGISTRY,
    ChiralAtomPotential,
    InterchainBondPotential,
)


def _coords() -> jnp.ndarray:
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [1.8, 1.1, 0.2],
            [2.7, 1.2, 1.0],
            [0.2, 2.0, 0.5],
            [1.3, 2.2, 1.1],
        ],
        dtype=jnp.float32,
    )


def _features() -> dict[str, jnp.ndarray]:
    ref_element = jnp.zeros((6, 118), dtype=jnp.float32).at[:, 6].set(1.0)
    return {
        "interchain_bond_index": jnp.asarray([[0], [5]], dtype=jnp.int32),
        "pairwise_distance_index": jnp.asarray([[0, 1], [4, 5]], dtype=jnp.int32),
        "pairwise_distance_is_bond": jnp.asarray([0, 1], dtype=jnp.int32),
        "pairwise_distance_is_angle": jnp.asarray([0, 0], dtype=jnp.int32),
        "pairwise_distance_lower_bound": jnp.asarray([2.0, 1.0]),
        "pairwise_distance_upper_bound": jnp.asarray([4.0, 1.5]),
        "ref_element": ref_element,
        "stereo_bond_index": jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32),
        "stereo_bond_orientation": jnp.asarray([1.0]),
        "chiral_index": jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32),
        "chiral_orientation": jnp.asarray([1.0]),
        "asym_id": jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
        "atom_to_token_idx": jnp.arange(6, dtype=jnp.int32),
        "planar_improper_index": jnp.asarray([[0], [1], [2], [3]], dtype=jnp.int32),
        "planar_improper_is_carbonyl": jnp.asarray([0.0]),
        "linear_triple_bond_index": jnp.asarray([[0], [1], [2]], dtype=jnp.int32),
        "experimental_torsion_index": jnp.asarray(
            [[0], [1], [2], [3]], dtype=jnp.int32
        ),
        "experimental_torsion_force_constant": jnp.ones((1, 6), dtype=jnp.float32),
        "experimental_torsion_sign": jnp.asarray([[1, -1, 1, -1, 1, -1]]),
    }


def test_schedule_and_config_validation() -> None:
    assert schedule_from_cfg(2.0)(0.4) == 2.0
    assert (
        schedule_from_cfg(
            {"type": "exp_interpolation", "start": 0.0, "end": 2.0, "alpha": 0.0}
        )(0.25)
        == 0.5
    )
    with pytest.raises(KeyError, match="Unsupported keys"):
        parse_tfg_config({"enable": False, "typo": 1})
    with pytest.raises(ValueError, match="no terms"):
        parse_tfg_config({"enable": True})
    with pytest.raises(KeyError, match="Unknown potential"):
        parse_tfg_config(
            {"enable": True, "terms": {"MissingPotential": {"weight": 1.0}}}
        )


@pytest.mark.parametrize("name", sorted(CLASS_REGISTRY))
def test_all_registered_potentials_have_finite_energy_and_gradient(name: str) -> None:
    potential = CLASS_REGISTRY[name]()
    energy, gradient = potential.energy_and_grad(_coords(), _features())
    assert energy.shape == ()
    assert gradient.shape == _coords().shape
    assert np.isfinite(np.asarray(energy)).all()
    assert np.isfinite(np.asarray(gradient)).all()


def test_interchain_bond_matches_closed_form() -> None:
    coords = _coords()
    feats = _features()
    energy, gradient = InterchainBondPotential().energy_and_grad(
        coords, feats, {"buffer": 1.0}
    )
    distance = np.linalg.norm(np.asarray(coords[0] - coords[5]))
    np.testing.assert_allclose(energy, max(distance - 1.0, 0.0), rtol=1e-6)
    expected = np.zeros_like(coords)
    expected[0] = (coords[0] - coords[5]) / distance
    expected[5] = -expected[0]
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-6)


def test_engine_step_changes_coordinates_and_validates_features() -> None:
    config = parse_tfg_config(
        {
            "enable": True,
            "mu": 0.1,
            "steps": {"tfg_inner": 2, "projection_outer": 0},
            "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
        }
    )
    engine = TFGEngine(config)
    coords = _coords()
    guided = engine.step(
        lambda x, noise: x,
        x=coords,
        t_hat=jnp.asarray(2.0),
        c_tau=jnp.asarray(1.0),
        step_scale_eta=1.0,
        step_i=0,
        num_diffusion_steps=2,
        input_feature_dict=_features(),
    )
    assert not np.array_equal(np.asarray(guided), np.asarray(coords))
    with pytest.raises(KeyError, match="missing required"):
        engine.refine(coords, {}, t=0.5, step_i=0)


def test_engine_mc_smoothing_is_keyed_and_deterministic() -> None:
    config = parse_tfg_config(
        {
            "enable": True,
            "mu": 0.2,
            "mc": {"std": 0.5, "batch": 8},
            "steps": {"tfg_inner": 1, "projection_outer": 0},
            "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
        }
    )
    engine = TFGEngine(config)
    kwargs = dict(
        denoise_net=lambda x, _: x,
        x=_coords(),
        t_hat=jnp.asarray(2.0),
        c_tau=jnp.asarray(1.0),
        step_scale_eta=1.0,
        step_i=0,
        num_diffusion_steps=2,
        input_feature_dict=_features(),
    )
    first = engine.step(key=jr.key(7), **kwargs)
    repeated = engine.step(key=jr.key(7), **kwargs)
    other = engine.step(key=jr.key(8), **kwargs)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert np.isfinite(first).all()
    with pytest.raises(ValueError, match="PRNG key"):
        engine.step(key=None, **kwargs)


def test_chiral_projection_preserves_affected_chain_center_and_radius() -> None:
    coords = _coords()
    feats = _features()
    feats["asym_id"] = jnp.zeros((6,), dtype=jnp.int32)
    feats["chiral_orientation"] = jnp.asarray([-1.0])
    delta = ChiralAtomPotential().project(coords, feats)
    moved = coords + delta
    affected = jnp.asarray([0, 1, 2, 3])

    def center_and_rg(value):
        selected = value[affected]
        center = selected.mean(axis=0)
        rg = jnp.mean(jnp.sum((selected - center) ** 2, axis=-1))
        return center, rg

    old_center, old_rg = center_and_rg(coords)
    new_center, new_rg = center_and_rg(moved)
    assert np.linalg.norm(np.asarray(delta)) > 0
    np.testing.assert_allclose(new_center, old_center, atol=2e-6)
    np.testing.assert_allclose(new_rg, old_rg, rtol=2e-5, atol=2e-6)
