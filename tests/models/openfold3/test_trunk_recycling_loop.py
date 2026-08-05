"""Recycling as a loop must equal recycling unrolled.

``trunk`` emits its recycling body once and iterates, rather than emitting it four
times. That is only safe because every cycle is the same computation on a different
carry, and because the MSA embedding -- now built once before the loop -- does not
depend on the cycle. Both would break silently: an unrolled and a looped trunk that
disagree still return finite representations and still produce a structure.

Run against randomized parameters rather than the released checkpoint so the gate
does not need a 2 GB download, and with more than one cycle so a bug that only
shows up on the second iteration has somewhere to appear.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch_parity


#: The shared fixture runs a single cycle, which cannot show a recycling bug at
#: all. Three is the smallest count that exercises a carry read twice.
CYCLES = 3


@pytest.fixture(scope="module")
def case(openfold3_source, randomized):
    from .test_inference_end_to_end import _batch, _config, _params, _torch

    torch = _torch()
    return (
        _batch(torch),
        _params(torch, randomized),
        _config()._replace(num_cycles=CYCLES),
    )


def _run(batch, params, config, **overrides):
    import jax

    from foldjax.models.openfold3.models.trunk import trunk

    return jax.jit(
        lambda b, p: trunk(
            b,
            p,
            num_cycles=config.num_cycles,
            n_query=config.n_query,
            n_key=config.n_key,
            atom_heads=config.atom_heads,
            n_token=config.n_token,
            max_relative_idx=config.max_relative_idx,
            max_relative_chain=config.max_relative_chain,
            no_heads_msa=config.no_heads_msa,
            no_heads_pair=config.no_heads_pair,
            no_heads_pair_bias=config.no_heads_pair_bias,
            **overrides,
        )
    )(batch, params.trunk)


def test_looped_recycling_matches_unrolled(case) -> None:
    batch, params, config = case
    if config.num_cycles < 2:
        pytest.skip("one cycle cannot show a recycling difference")

    looped = _run(batch, params, config, scan_cycles=True)
    unrolled = _run(batch, params, config, scan_cycles=False)

    for name, left, right in zip(
        ("s_input", "s", "z"), unrolled, looped, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(right, dtype=np.float64),
            np.asarray(left, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{name} differs between looped and unrolled recycling",
        )


def test_recycling_actually_changes_the_representations(case) -> None:
    """Guard the test above from passing for the wrong reason.

    If recycling were a no-op -- a projection that returned its input, a carry that
    was dropped -- then looped and unrolled would agree trivially and the gate would
    prove nothing. So one cycle and several cycles have to *differ*.
    """
    batch, params, config = case
    if config.num_cycles < 2:
        pytest.skip("one cycle cannot show a recycling difference")

    one = _run(batch, params, config._replace(num_cycles=1), scan_cycles=True)
    many = _run(batch, params, config, scan_cycles=True)
    assert not np.allclose(
        np.asarray(one[2], dtype=np.float64),
        np.asarray(many[2], dtype=np.float64),
        rtol=1e-4,
        atol=1e-4,
    ), "the pair representation is unchanged by extra cycles; recycling is inert"
