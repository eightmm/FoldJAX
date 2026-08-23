"""OpenFold3 repeated blocks reach ``lax.scan`` already stacked."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._stacking import StackedLayers
from foldjax.models.openfold3.bridge import torch_mapping
from foldjax.models.openfold3.models.stacking import scan_stack


class _Block(NamedTuple):
    weight: Any
    bias: Any


def _blocks(*, width: int = 256) -> tuple[_Block, ...]:
    rng = np.random.default_rng(20260823)
    return tuple(
        _Block(
            rng.standard_normal((width, width), dtype=np.float32),
            rng.standard_normal((width,), dtype=np.float32),
        )
        for _ in range(4)
    )


def _run(carry: jax.Array, blocks: Any) -> jax.Array:
    return scan_stack(
        lambda state, block: state + jnp.sum(block.weight) + jnp.sum(block.bias),
        carry,
        blocks,
    )


def test_prestacked_scan_is_bitwise_equal_without_parameter_concatenates() -> None:
    plain = _blocks()
    prestacked = StackedLayers.from_layers(plain)

    plain_executable = jax.jit(_run).lower(jnp.float32(0), plain).compile()
    stacked_executable = jax.jit(_run).lower(jnp.float32(0), prestacked).compile()
    plain_result = plain_executable(jnp.float32(0), plain)
    stacked_result = stacked_executable(jnp.float32(0), prestacked)

    np.testing.assert_array_equal(np.asarray(stacked_result), np.asarray(plain_result))
    assert plain_executable.as_text().lower().count("concatenate(") > 0
    assert stacked_executable.as_text().lower().count("concatenate(") == 0

    stacked_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(prestacked.stacked))
    plain_temp = plain_executable.memory_analysis().temp_size_in_bytes
    stacked_temp = stacked_executable.memory_analysis().temp_size_in_bytes
    assert plain_temp - stacked_temp >= stacked_bytes


def test_inference_mapping_prestacks_on_host_then_transfers_once(monkeypatch) -> None:
    layers = _blocks(width=3)
    simple = {"value": np.ones((2,), dtype=np.float32)}
    mapped_linear = torch_mapping.map_linear(
        {"weight": np.ones((2, 2), dtype=np.float32)}, bias=False
    )
    assert isinstance(mapped_linear.weight, np.ndarray)

    monkeypatch.setattr(
        torch_mapping, "map_trunk", lambda *_args, **_kwargs: {"blocks": layers}
    )
    monkeypatch.setattr(
        torch_mapping,
        "map_diffusion_conditioning",
        lambda *_args, **_kwargs: {"transition_s": layers},
    )
    for name in (
        "map_denoiser",
        "map_pairformer_embedding",
        "map_atom_logit_head",
        "map_pair_head",
    ):
        monkeypatch.setattr(torch_mapping, name, lambda *_args, **_kwargs: simple)

    state = {"aux_heads.pae.stub": np.zeros((1,), dtype=np.float32)}
    params = torch_mapping.map_inference_params(state, "")
    plain_params = torch_mapping.map_inference_params(state, "", prestack=False)

    assert isinstance(params.trunk["blocks"], StackedLayers)
    assert isinstance(plain_params.trunk["blocks"], tuple)
    assert isinstance(params.diffusion_conditioning["transition_s"], tuple)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(params))
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(plain_params))
    for index, expected in enumerate(layers):
        actual = params.trunk["blocks"][index]
        np.testing.assert_array_equal(np.asarray(actual.weight), expected.weight)
        np.testing.assert_array_equal(np.asarray(actual.bias), expected.bias)
