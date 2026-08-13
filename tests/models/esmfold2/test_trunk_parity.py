"""The pair trunk against torch, module by module.

These are the modules whose bugs are silent: a triangular contraction with the
axes transposed, a gate applied to the wrong tensor, a mean divided in the
wrong place. Each produces the right shapes and plausible values, and each
would be found only much later, in a structure nobody can explain.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")
modeling = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2")

from foldjax.models.esmfold2.models import trunk  # noqa: E402
from tests.models.esmfold2.parity import (  # noqa: E402
    random_inputs,
    torch_state_to_numpy,
)

ATOL = 2e-5


def _compare(module, run, arrays, *, seed=0):
    module = module.eval()
    with torch.no_grad():
        expected = module(**{k: torch.from_numpy(v) for k, v in arrays.items()})
    if isinstance(expected, tuple):
        expected = expected[0]
    got = np.asarray(run(torch_state_to_numpy(module), **arrays))
    np.testing.assert_allclose(got, expected.numpy(), atol=ATOL, rtol=ATOL)


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplicative_matches(outgoing: bool) -> None:
    """Both flows: the einsum differs only in which axis is contracted."""
    module = common.TriangleMultiplicativeUpdate(dim=16, _outgoing=outgoing)
    arrays = random_inputs({"z": (1, 5, 5, 16)})
    arrays["mask"] = np.ones((1, 5, 5), dtype=np.float32)
    _compare(
        module,
        lambda p, z, mask: trunk.triangle_multiplicative(
            z, p, "", outgoing=outgoing, mask=mask
        ),
        arrays,
    )


def test_a_masked_pair_is_zeroed_before_the_contraction() -> None:
    """The mask multiplies the gated signal, so masked rows contribute nothing."""
    module = common.TriangleMultiplicativeUpdate(dim=16, _outgoing=True)
    arrays = random_inputs({"z": (1, 6, 6, 16)}, seed=3)
    mask = np.ones((1, 6, 6), dtype=np.float32)
    mask[:, 4:, :] = 0.0
    mask[:, :, 4:] = 0.0
    arrays["mask"] = mask
    _compare(
        module,
        lambda p, z, mask: trunk.triangle_multiplicative(
            z, p, "", outgoing=True, mask=mask
        ),
        arrays,
    )


def test_transition_keeps_its_residual_and_pair_transition_does_not() -> None:
    """Same parameters, same shapes; only the residual tells them apart."""
    with_residual = common.Transition(d_model=16, expansion_ratio=2)
    _compare(
        with_residual,
        lambda p, x: trunk.transition(x, p, "", residual=True),
        random_inputs({"x": (1, 4, 4, 16)}),
    )

    without = modeling.PairTransition(d_model=16, expansion_ratio=2)
    _compare(
        without,
        lambda p, x: trunk.transition(x, p, "", residual=False),
        random_inputs({"x": (1, 4, 4, 16)}, seed=1),
    )


def test_pair_update_block_matches() -> None:
    """Sequential residuals: the incoming update reads the outgoing one's write."""
    module = common.PairUpdateBlock(d_pair=16, expansion_ratio=2)
    arrays = random_inputs({"pair": (1, 5, 5, 16)}, seed=2)
    arrays["pair_attention_mask"] = np.ones((1, 5, 5), dtype=np.float32)
    _compare(
        module,
        lambda p, pair, pair_attention_mask: trunk.pair_update_block(
            pair, p, "", mask=pair_attention_mask
        ),
        arrays,
    )


def test_the_block_stack_matches() -> None:
    module = common.FoldingTrunk(n_layers=3, d_pair=16, expansion_ratio=2)
    arrays = random_inputs({"pair": (1, 5, 5, 16)}, seed=4)
    arrays["pair_attention_mask"] = np.ones((1, 5, 5), dtype=np.float32)
    _compare(
        module,
        lambda p, pair, pair_attention_mask: trunk.folding_trunk(
            pair, p, "", n_layers=3, mask=pair_attention_mask
        ),
        arrays,
    )


def test_outer_product_mean_divides_after_the_projection() -> None:
    """Dividing before would leave `Wout`'s bias unscaled -- a plausible wrong."""
    module = common.OuterProductMean(d_msa=12, d_hidden=4, d_pair=16)
    arrays = random_inputs({"m": (1, 5, 3, 12)}, seed=5)
    arrays["msa_attention_mask"] = np.ones((1, 5, 3), dtype=np.float32)
    _compare(
        module,
        lambda p, m, msa_attention_mask: trunk.outer_product_mean(
            m, p, "", msa_mask=msa_attention_mask
        ),
        arrays,
    )


def test_msa_pair_weighted_averaging_matches() -> None:
    module = common.MSAPairWeightedAveraging(
        d_msa=12, d_pair=16, n_heads=2, head_width=4
    )
    arrays = random_inputs(
        {"msa_repr": (1, 5, 3, 12), "pair_repr": (1, 5, 5, 16)}, seed=6
    )
    arrays["pair_attention_mask"] = np.ones((1, 5, 5), dtype=np.float32)
    _compare(
        module,
        lambda p, msa_repr, pair_repr, pair_attention_mask: (
            trunk.msa_pair_weighted_averaging(
                msa_repr, pair_repr, p, "", pair_mask=pair_attention_mask
            )
        ),
        arrays,
    )
