"""Compare one JAX module against the torch module it was ported from.

The other ports in this repository each grew a `*_checkpoint_parity` suite,
and they are the reason those ports can be trusted: a module that produces
plausible numbers is not a module that produces upstream's numbers, and the
difference is invisible until something downstream is wrong for a reason
nobody can find. Written first here, before any module exists, so no module
can land without one.

The comparison is deliberately narrow: real released weights, real shapes, one
module at a time, and a tolerance tight enough that a wrong axis order or a
missing gate cannot pass. Random weights would not do -- several of these
modules are gates and normalisations whose bugs only show on the value
distribution the training produced.

    from tests.models.esmfold2.parity import assert_module_matches

    assert_module_matches(
        torch_module=upstream.Transition(384),
        jax_apply=lambda params, x: transition_forward(params, x),
        inputs={"x": (2, 37, 384)},
    )
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest

#: Tight on purpose. float32 through a handful of matmuls agrees far better
#: than this; anything looser stops distinguishing "ported" from "close".
ATOL = 2e-5
RTOL = 2e-5


def torch_state_to_numpy(module: Any) -> dict[str, np.ndarray]:
    """`state_dict` as plain arrays, which is what the mappers consume."""
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in module.state_dict().items()
    }


def random_inputs(
    shapes: Mapping[str, Sequence[int]], *, seed: int = 0
) -> dict[str, np.ndarray]:
    """Normal inputs, fixed seed, float32 -- the dtype the weights ship in."""
    generator = np.random.default_rng(seed)
    return {
        name: generator.standard_normal(tuple(shape)).astype(np.float32)
        for name, shape in shapes.items()
    }


def assert_module_matches(
    torch_module: Any,
    jax_apply: Callable[..., Any],
    inputs: Mapping[str, Sequence[int]],
    *,
    map_params: Callable[[Mapping[str, np.ndarray]], Any] | None = None,
    seed: int = 0,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    """Run both on the same arrays and require the same answer.

    `map_params` turns the torch state dict into whatever the JAX function
    takes; without it the state dict is passed through, which is what the
    simplest modules want.
    """
    torch = pytest.importorskip("torch")

    torch_module = torch_module.eval()
    arrays = random_inputs(inputs, seed=seed)

    with torch.no_grad():
        expected = torch_module(
            **{name: torch.from_numpy(value) for name, value in arrays.items()}
        )
    if isinstance(expected, tuple):
        expected = expected[0]
    expected = expected.detach().cpu().numpy()

    state = torch_state_to_numpy(torch_module)
    params = map_params(state) if map_params is not None else state
    got = np.asarray(jax_apply(params, **arrays))

    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)
