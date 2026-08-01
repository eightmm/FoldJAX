from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.bridge.torch_mapping import (
    map_diffusion_conditioning_state_dict,
    map_released_diffusion_conditioning_state_dict,
)
from foldjax.models.opendde.models.diffusion_conditioning import (
    diffusion_conditioning,
    diffusion_conditioning_prepare_cache,
)
from foldjax.models.protenix.models.primitives.primitives import layer_norm, linear
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    fourier_embedding,
    relative_position_encoding,
)


def test_diffusion_conditioning_compresses_structural_pair_trunk() -> None:
    state = _state()
    params = map_diffusion_conditioning_state_dict(state, "dc")
    relp = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    z_trunk = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4) / 10.0
    s_inputs = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    s_trunk = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    noise = jnp.asarray([0.0, 8.0], dtype=jnp.float32)

    actual_s, actual_z = diffusion_conditioning(
        noise,
        relp,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        sigma_data=4.0,
    )

    compressed_z = linear(
        layer_norm(z_trunk, params.layernorm_z_trunk),
        params.linear_z_trunk,
    )
    expected_z = linear(
        layer_norm(
            jnp.concatenate(
                [compressed_z, relative_position_encoding(relp, params.relpe)],
                axis=-1,
            ),
            params.layernorm_z,
        ),
        params.linear_z,
    )
    base_s = linear(
        layer_norm(
            jnp.concatenate([s_trunk, s_inputs], axis=-1),
            params.layernorm_s,
        ),
        params.linear_s,
    )
    noise_ratio = jnp.maximum(noise / 4.0, 1.0e-10)
    noise_embedding = fourier_embedding(
        jnp.log(noise_ratio) / 4.0,
        params.fourier,
    )
    noise_embedding = linear(
        layer_norm(noise_embedding, params.layernorm_n),
        params.linear_n,
    )
    expected_s = base_s[None, :, :] + noise_embedding[:, None, :]

    np.testing.assert_allclose(actual_z, expected_z, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-5)
    assert np.isfinite(np.asarray(actual_s)).all()


def test_diffusion_conditioning_reuses_compressed_pair_cache() -> None:
    params = map_diffusion_conditioning_state_dict(_state(), "dc")
    relp = jnp.ones((2, 2, 2), dtype=jnp.float32)
    z_trunk = jnp.ones((2, 2, 4), dtype=jnp.float32)
    cache = diffusion_conditioning_prepare_cache(relp, z_trunk, params)

    _, actual = diffusion_conditioning(
        jnp.asarray([4.0], dtype=jnp.float32),
        relp,
        jnp.ones((2, 2), dtype=jnp.float32),
        jnp.ones((2, 3), dtype=jnp.float32),
        z_trunk,
        params,
        pair_z=cache,
        sigma_data=4.0,
    )

    np.testing.assert_allclose(actual, cache, rtol=0.0, atol=0.0)


def test_diffusion_conditioning_mapper_exposes_384_to_128_contract() -> None:
    params = map_diffusion_conditioning_state_dict(_state(), "dc")

    assert params.layernorm_z_trunk.weight.shape == (4,)
    assert params.layernorm_z_trunk.bias is None
    assert params.linear_z_trunk.weight.shape == (2, 4)
    assert params.layernorm_z.weight.shape == (4,)
    assert params.linear_z.weight.shape == (2, 4)


def test_released_diffusion_conditioning_validates_checkpoint_dimensions() -> None:
    params = map_released_diffusion_conditioning_state_dict(
        _state(
            c_z_trunk=384,
            c_z_diffusion=128,
            c_s=384,
            c_s_inputs=449,
            c_noise=256,
            relp_channels=139,
        ),
        "dc",
    )

    assert params.linear_z_trunk.weight.shape == (128, 384)
    assert params.linear_s.weight.shape == (384, 833)
    with pytest.raises(ValueError, match="released diffusion conditioning"):
        map_released_diffusion_conditioning_state_dict(_state(), "dc")


def _state(
    *,
    c_z_trunk: int = 4,
    c_z_diffusion: int = 2,
    c_s: int = 3,
    c_s_inputs: int = 2,
    c_noise: int = 4,
    relp_channels: int = 2,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(51)
    state = {
        "dc.relpe.linear_no_bias.weight": _random(
            rng,
            (c_z_diffusion, relp_channels),
        ),
        "dc.layernorm_z_trunk.weight": np.ones(
            (c_z_trunk,),
            dtype=np.float32,
        ),
        "dc.linear_no_bias_z_trunk.weight": _random(
            rng,
            (c_z_diffusion, c_z_trunk),
        ),
        "dc.layernorm_z.weight": np.ones(
            (2 * c_z_diffusion,),
            dtype=np.float32,
        ),
        "dc.linear_no_bias_z.weight": _random(
            rng,
            (c_z_diffusion, 2 * c_z_diffusion),
        ),
        "dc.layernorm_s.weight": np.ones(
            (c_s + c_s_inputs,),
            dtype=np.float32,
        ),
        "dc.linear_no_bias_s.weight": _random(
            rng,
            (c_s, c_s + c_s_inputs),
        ),
        "dc.fourier_embedding.w": np.linspace(
            0.1,
            0.4,
            c_noise,
            dtype=np.float32,
        ),
        "dc.fourier_embedding.b": np.linspace(
            0.2,
            0.5,
            c_noise,
            dtype=np.float32,
        ),
        "dc.layernorm_n.weight": np.ones((c_noise,), dtype=np.float32),
        "dc.linear_no_bias_n.weight": _random(rng, (c_s, c_noise)),
    }
    for name, channels in (
        ("transition_z1", c_z_diffusion),
        ("transition_z2", c_z_diffusion),
        ("transition_s1", c_s),
        ("transition_s2", c_s),
    ):
        _add_zero_transition(state, f"dc.{name}", channels)
    return state


def _add_zero_transition(
    state: dict[str, np.ndarray],
    prefix: str,
    channels: int,
) -> None:
    hidden = 2 * channels
    state[f"{prefix}.layernorm1.weight"] = np.ones(
        (channels,),
        dtype=np.float32,
    )
    state[f"{prefix}.layernorm1.bias"] = np.zeros(
        (channels,),
        dtype=np.float32,
    )
    state[f"{prefix}.linear_no_bias_a.weight"] = np.zeros(
        (hidden, channels),
        dtype=np.float32,
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = np.zeros(
        (hidden, channels),
        dtype=np.float32,
    )
    state[f"{prefix}.linear_no_bias.weight"] = np.zeros(
        (channels, hidden),
        dtype=np.float32,
    )


def _random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    return rng.normal(scale=0.2, size=shape).astype(np.float32)
