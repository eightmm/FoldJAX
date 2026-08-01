from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.diffusion_conditioning import (
    diffusion_conditioning,
    map_diffusion_conditioning,
    norm_linear,
)

pytestmark = pytest.mark.official_parity

@pytest.fixture(scope="module")
def diffusion_module(official_asset_path):
    torch = pytest.importorskip("torch")
    path = official_asset_path("diffusion_module.pt")
    return torch.jit.load(str(path), map_location="cpu").eval()


@pytest.fixture(scope="module")
def conditioning_state(diffusion_module):
    return {
        key: value.detach().cpu().numpy()
        for key, value in diffusion_module.state_dict().items()
        if key.startswith("diffusion_conditioning.")
    }


def test_mapping_covers_all_31_conditioning_tensors(conditioning_state) -> None:
    params = map_diffusion_conditioning(conditioning_state)
    assert len(conditioning_state) == 31
    assert len(jax.tree.leaves(params)) == 31


@pytest.mark.parametrize(
    ("child_name", "parameter_name", "shape"),
    [
        ("token_in_proj", "token_single_projection", (1, 4, 768)),
        ("token_pair_proj", "token_pair_projection", (1, 4, 4, 512)),
    ],
)
def test_input_projection_matches_official_callable(
    diffusion_module,
    conditioning_state,
    child_name: str,
    parameter_name: str,
    shape: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(31)
    value = torch.randn(shape, generator=generator)
    child = getattr(diffusion_module.diffusion_conditioning, child_name)
    with torch.no_grad():
        expected = child(value).numpy()
    params = getattr(map_diffusion_conditioning(conditioning_state), parameter_name)
    actual = norm_linear(jnp.asarray(value.numpy()), params)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def _torch_conditioning_reference(module, inputs):
    torch = pytest.importorskip("torch")
    conditioning = module.diffusion_conditioning

    pair_input = torch.cat(
        [inputs["token_pair_trunk_repr"], inputs["token_pair_initial_repr"]],
        dim=-1,
    )
    pair = conditioning.token_pair_proj(pair_input)
    pair = pair + conditioning.pair_trans1(pair)
    pair = pair + conditioning.pair_trans2(pair)
    pair = conditioning.pair_ln(pair)

    single_input = torch.cat(
        [inputs["token_single_initial_repr"], inputs["token_single_trunk_repr"]],
        dim=-1,
    )
    single = conditioning.token_in_proj(single_input)
    sigma = inputs["noise_sigma"].clamp_min(torch.finfo(torch.float32).eps)
    frequencies = (
        torch.log(sigma).mul(0.25)[..., None] * conditioning.fourier_embedding.weights
        + conditioning.fourier_embedding.bias
    ) * (2.0 * math.pi)
    noise = torch.cos(frequencies)[:, :, None, :]
    noise = getattr(conditioning.fourier_proj, "0")(noise)
    noise = getattr(conditioning.fourier_proj, "1")(noise)
    single = single[:, None] + noise
    single = single + conditioning.single_trans1(single)
    single = single + conditioning.single_trans2(single)
    single = conditioning.single_ln(single)
    return single, pair


def test_complete_conditioning_matches_official_children(
    diffusion_module, conditioning_state
) -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(37)
    torch_inputs = {
        "token_single_initial_repr": torch.randn(1, 4, 384, generator=generator),
        "token_pair_initial_repr": torch.randn(1, 4, 4, 256, generator=generator),
        "token_single_trunk_repr": torch.randn(1, 4, 384, generator=generator),
        "token_pair_trunk_repr": torch.randn(1, 4, 4, 256, generator=generator),
        "noise_sigma": torch.tensor([[0.0, 0.25, 4.0]]),
    }
    with torch.no_grad():
        expected_single, expected_pair = _torch_conditioning_reference(
            diffusion_module, torch_inputs
        )
    jax_inputs = {
        key: jnp.asarray(value.numpy()) for key, value in torch_inputs.items()
    }
    actual_single, actual_pair = diffusion_conditioning(
        **jax_inputs, params=map_diffusion_conditioning(conditioning_state)
    )
    assert actual_single.shape == (1, 3, 4, 384)
    assert actual_pair.shape == (1, 4, 4, 256)
    np.testing.assert_allclose(
        np.asarray(actual_single),
        expected_single.numpy(),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual_pair),
        expected_pair.numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_conditioning_rejects_non_matrix_sigma(conditioning_state) -> None:
    params = map_diffusion_conditioning(conditioning_state)
    with pytest.raises(ValueError, match="batch, samples"):
        diffusion_conditioning(
            jnp.zeros((1, 2, 384)),
            jnp.zeros((1, 2, 2, 256)),
            jnp.zeros((1, 2, 384)),
            jnp.zeros((1, 2, 2, 256)),
            jnp.ones((1,)),
            params,
        )
