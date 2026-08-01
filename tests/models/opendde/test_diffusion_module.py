from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import foldjax.models.opendde.bridge.torch_mapping as mapping_impl
import foldjax.models.opendde.models.diffusion_module as diffusion_impl


class _DummyParams(NamedTuple):
    conditioning: object
    marker: object


def test_full_diffusion_mapper_replaces_base_conditioning(monkeypatch) -> None:
    base_conditioning = object()
    opendde_conditioning = object()
    base = _DummyParams(conditioning=base_conditioning, marker=object())

    monkeypatch.setattr(
        mapping_impl,
        "_map_diffusion_module_state_dict",
        lambda state, prefix: base,
    )
    monkeypatch.setattr(
        mapping_impl,
        "map_diffusion_conditioning_state_dict",
        lambda state, prefix: opendde_conditioning,
    )

    actual = mapping_impl.map_diffusion_module_state_dict({}, "module")

    assert actual.conditioning is opendde_conditioning
    assert actual.marker is base.marker


def test_diffusion_f_delegates_preconditioned_tensors_and_bias(
    monkeypatch,
) -> None:
    single_s = jnp.ones((1, 2, 3), dtype=jnp.float32)
    pair_z = jnp.ones((2, 2, 2), dtype=jnp.float32)
    structural_bias = jnp.asarray([[0.0, 0.4], [-0.2, 0.0]])
    captured = {}

    def fake_conditioning(*args, **kwargs):
        captured["conditioning"] = (args, kwargs)
        return single_s, pair_z

    def fake_core(*args, **kwargs):
        captured["core"] = (args, kwargs)
        return jnp.full((1, 3, 3), 0.25, dtype=jnp.float32)

    monkeypatch.setattr(diffusion_impl, "diffusion_conditioning", fake_conditioning)
    monkeypatch.setattr(
        diffusion_impl,
        "_protenix_diffusion_module_f_forward",
        fake_core,
    )

    result = diffusion_impl.diffusion_module_f_forward(
        *_positional_inputs(),
        _DummyParams(conditioning=object(), marker=object()),
        extra_attn_bias=structural_bias,
        n_token=2,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
    )

    np.testing.assert_allclose(result, 0.25)
    _, core_kwargs = captured["core"]
    np.testing.assert_allclose(core_kwargs["conditioned_single_s"], single_s)
    np.testing.assert_allclose(core_kwargs["pair_z"], pair_z)
    np.testing.assert_allclose(core_kwargs["extra_attn_bias"], structural_bias)


def test_diffusion_forward_applies_opendde_edm_rescale(monkeypatch) -> None:
    r_update = jnp.full((1, 3, 3), 2.0, dtype=jnp.float32)
    captured = {}

    def fake_f(*args, **kwargs):
        captured["r_noisy"] = args[9]
        return r_update

    monkeypatch.setattr(diffusion_impl, "diffusion_module_f_forward", fake_f)
    inputs = list(_positional_inputs())
    x_noisy = jnp.full((1, 3, 3), 10.0, dtype=jnp.float32)
    noise = jnp.asarray([4.0], dtype=jnp.float32)
    inputs[9] = x_noisy
    inputs[10] = noise

    actual = diffusion_impl.diffusion_module_forward(
        *inputs,
        _DummyParams(conditioning=object(), marker=object()),
        n_token=2,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
    )

    expected_r_noisy = x_noisy / np.sqrt(4.0**2 + 4.0**2)
    expected = x_noisy / 2.0 + 4.0 / np.sqrt(2.0) * r_update
    np.testing.assert_allclose(captured["r_noisy"], expected_r_noisy, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def _positional_inputs() -> tuple[jnp.ndarray | dict[str, jnp.ndarray], ...]:
    return (
        jnp.asarray([0, 1, 1], dtype=jnp.int32),
        jnp.zeros((3, 3), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.ones((3,), dtype=jnp.float32),
        jnp.zeros((3, 4, 64), dtype=jnp.float32),
        jnp.zeros((3, 128), dtype=jnp.float32),
        jnp.zeros((2, 2, 4, 3), dtype=jnp.float32),
        jnp.ones((2, 2, 4, 1), dtype=jnp.float32),
        {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)},
        jnp.ones((1, 3, 3), dtype=jnp.float32),
        jnp.asarray([4.0], dtype=jnp.float32),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
    )
