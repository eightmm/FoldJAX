"""Statically empty contact guidance must not force the Python steering loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.heads import potentials
from foldjax.models.boltz2.models.trunk_blocks import trunk


def _steering(**overrides):
    return {
        "fk_steering": False,
        "physical_guidance_update": False,
        "contact_guidance_update": True,
        "num_gd_steps": 2,
        **overrides,
    }


def _features():
    return {
        "token_pad_mask": jnp.ones((1, 2), dtype=bool),
        "atom_pad_mask": jnp.ones((1, 4), dtype=bool),
        "contact_pair_index": jnp.empty((1, 2, 0), dtype=jnp.int32),
        "contact_union_index": jnp.empty((1, 0), dtype=jnp.int32),
        "contact_negation_mask": jnp.empty((1, 0), dtype=bool),
        "contact_thresholds": jnp.empty((1, 0), dtype=jnp.float32),
    }


@pytest.fixture
def tiny_sampler(monkeypatch):
    monkeypatch.setattr(
        trunk, "diffusion_conditioning_forward", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        trunk,
        "_preconditioned_score_forward",
        lambda *_args, r_noisy, **_kwargs: r_noisy * 0.75,
    )
    # Replay an explicit schedule carried as a JIT argument, just like a
    # captured native sigma tape; no schedule value is a Python constant.
    monkeypatch.setattr(
        trunk, "_sample_schedule", lambda _steps, *, sigma_max, **_kwargs: sigma_max
    )
    params = {
        "trunk": {},
        "conditioned_diffusion": {"diffusion_conditioning": {}, "score_model": {}},
    }
    supplied_trunk = {
        "s": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        "z": jnp.zeros((1, 2, 2, 4), dtype=jnp.float32),
        "s_inputs": jnp.zeros((1, 2, 4), dtype=jnp.float32),
        "relative_position_encoding": jnp.zeros((1, 2, 2, 4), dtype=jnp.float32),
    }

    def run(sigmas, init_noise, step_noises, feats, steering, *, use_scan):
        steps, multiplicity = step_noises.shape[:2]
        rotations = jnp.broadcast_to(jnp.eye(3), (steps, multiplicity, 3, 3))
        translations = jnp.full((steps, multiplicity, 1, 3), 0.125)
        return trunk.boltz2_sample_forward(
            params,
            feats,
            jax.random.key(5),
            num_sampling_steps=steps,
            multiplicity=multiplicity,
            sigma_max=sigmas,
            augmentation=True,
            alignment_reverse_diff=False,
            steering_args=steering,
            init_noise=init_noise,
            step_noises=step_noises,
            aug_transforms=(rotations, translations),
            use_scan=use_scan,
            trunk=supplied_trunk,
        )["sample_atom_coords"]

    return run


@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("multiplicity", [1, 5])
def test_jitted_empty_guidance_matches_disabled_with_dynamic_tapes(
    tiny_sampler, use_scan, multiplicity
):
    rng = np.random.default_rng(23)
    sigmas = jnp.asarray([8.0, 3.0, 1.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.asarray(rng.normal(size=(multiplicity, 4, 3)), jnp.float32)
    step_noises = jnp.asarray(rng.normal(size=(3, multiplicity, 4, 3)), jnp.float32)
    feats = _features()
    steering = _steering()
    original_flags = dict(steering)

    def evaluate(sigmas, initial, steps, features):
        guided = tiny_sampler(
            sigmas, initial, steps, features, steering, use_scan=use_scan
        )
        disabled = tiny_sampler(
            sigmas, initial, steps, features, None, use_scan=use_scan
        )
        return guided, disabled

    guided, disabled = jax.jit(evaluate)(sigmas, init_noise, step_noises, feats)
    np.testing.assert_array_equal(guided, disabled)
    assert np.isfinite(np.asarray(guided)).all()
    assert steering == original_flags


@pytest.mark.parametrize("multiplicity", [1, 5])
def test_empty_guidance_matches_the_previous_zero_gradient_loop(
    tiny_sampler, monkeypatch, multiplicity
):
    rng = np.random.default_rng(42)
    args = (
        jnp.asarray([8.0, 3.0, 1.0, 0.0], jnp.float32),
        jnp.asarray(rng.normal(size=(multiplicity, 4, 3)), jnp.float32),
        jnp.asarray(rng.normal(size=(3, multiplicity, 4, 3)), jnp.float32),
        _features(),
        _steering(),
    )
    fast = tiny_sampler(*args, use_scan=False)
    monkeypatch.setattr(
        trunk, "_contact_guidance_is_statically_empty", lambda *_args: False
    )
    previous = tiny_sampler(*args, use_scan=False)
    np.testing.assert_array_equal(fast, previous)


@pytest.mark.parametrize("template_field", [None, "template_mask_cb", "template_force"])
def test_empty_contact_and_absent_template_input_are_proven_empty(template_field):
    feats = _features()
    if template_field is not None:
        feats[template_field] = jnp.ones((1, 2), dtype=bool)
    assert trunk._contact_guidance_is_statically_empty(feats, _steering())


@pytest.mark.parametrize(
    "index",
    [
        None,
        [],
        jnp.empty((2, 0), dtype=jnp.int32),
        jnp.empty((1, 3, 0), dtype=jnp.int32),
        jnp.empty((0, 2, 0), dtype=jnp.int32),
        jnp.empty((2, 2, 0), dtype=jnp.int32),
        jnp.empty((1, 2, 0, 1), dtype=jnp.int32),
        jnp.empty((1, 2, 0), dtype=jnp.float32),
        jnp.empty((1, 2, 0), dtype=bool),
        jnp.zeros((1, 2, 1), dtype=jnp.int32),
    ],
)
def test_missing_malformed_or_nonempty_contacts_are_not_elided(index):
    feats = _features()
    feats["contact_pair_index"] = index
    assert not trunk._contact_guidance_is_statically_empty(feats, _steering())


def test_missing_contact_index_is_not_elided():
    feats = _features()
    del feats["contact_pair_index"]
    assert not trunk._contact_guidance_is_statically_empty(feats, _steering())


@pytest.mark.parametrize("force", [False, True])
def test_present_template_inputs_are_not_inspected_or_elided(force):
    feats = _features()
    feats["template_force"] = jnp.full((1, 1), force)
    feats["template_mask_cb"] = jnp.zeros((1, 1, 2), dtype=bool)
    # Even all-zero masks/force values are not a static absence proof.
    actual = jax.jit(
        lambda features: trunk._contact_guidance_is_statically_empty(
            features, _steering()
        )
    )(feats)
    assert not bool(actual)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fk_steering": True},
        {"physical_guidance_update": True},
        {"contact_guidance_update": False},
        {"fk_steering": None},
        {"physical_guidance_update": 0},
        {"contact_guidance_update": "true"},
    ],
)
def test_other_or_unknown_modes_are_not_elided(overrides):
    assert not trunk._contact_guidance_is_statically_empty(
        _features(), _steering(**overrides)
    )


@pytest.mark.parametrize("flags", [None, {}, {"contact_guidance_update": True}])
def test_missing_modes_are_not_elided(flags):
    assert not trunk._contact_guidance_is_statically_empty(_features(), flags)


def test_traced_flags_are_not_converted_to_python_bool():
    def evaluate(flag):
        return trunk._contact_guidance_is_statically_empty(
            _features(), _steering(contact_guidance_update=flag)
        )

    assert not bool(jax.jit(evaluate)(jnp.asarray(True)))


def test_both_native_contact_potential_types_have_exactly_zero_gradient():
    coords = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    selected = potentials.get_potentials(_steering(), boltz2=True)
    assert [type(p).__name__ for p in selected] == [
        "ContactPotentital", "TemplateReferencePotential"
    ]
    for potential in selected:
        parameters = potential.compute_parameters(1.0)
        np.testing.assert_array_equal(
            potential.compute_gradient(coords, _features(), parameters),
            jnp.zeros_like(coords),
        )
