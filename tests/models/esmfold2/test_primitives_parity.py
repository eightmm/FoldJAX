"""Each primitive against the torch module it was ported from.

Real modules, real shapes, random weights here rather than released ones --
these are the pieces whose bugs are structural (a swapped split half, an
affine term on the wrong normalisation), and those show on any weights. The
blocks built from them get released-weight parity of their own.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
upstream = pytest.importorskip(
    "transformers.models.esmfold2.modeling_esmfold2_common"
)

from foldjax.models.esmfold2.models import primitives  # noqa: E402
from tests.models.esmfold2.parity import assert_module_matches  # noqa: E402


def test_swiglu_splits_the_packed_projection_the_way_upstream_does() -> None:
    """The first half is gated through silu; swapping them still runs."""
    assert_module_matches(
        upstream.SwiGLU(in_features=32, hidden_features=48, out_features=32),
        lambda params, x: primitives.swiglu(x, params),
        {"x": (2, 7, 32)},
    )


def test_swiglu_mlp_matches() -> None:
    assert_module_matches(
        upstream.SwiGLUMLP(d_model=32, expansion_ratio=2),
        lambda params, x: primitives.swiglu(x, params),
        {"x": (2, 5, 32)},
    )


def test_transition_layer_matches() -> None:
    assert_module_matches(
        upstream.TransitionLayer(d_model=32, n=2),
        lambda params, x: primitives.transition_layer(x, params),
        {"x": (2, 6, 32)},
    )


def test_adaptive_layer_norm_matches() -> None:
    """Two normalisations with different affine sets, one gate, one shift."""
    assert_module_matches(
        upstream.AdaptiveLayerNorm(d_model=32, d_cond=16),
        lambda params, a, s: primitives.adaptive_layer_norm(a, s, params),
        {"a": (2, 5, 32), "s": (2, 5, 16)},
    )


def test_fourier_embedding_reads_its_frozen_buffers() -> None:
    """w and b are drawn once and shipped; redrawing them changes the model."""
    module = upstream.FourierEmbedding(c=16).eval()
    t = np.asarray([0.1, 2.5, 7.0], dtype=np.float32)

    with torch.no_grad():
        expected = module(torch.from_numpy(t)).numpy()

    state = {name: value.numpy() for name, value in module.state_dict().items()}
    got = np.asarray(primitives.fourier_embedding(t, state))
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)


def test_layer_norm_matches_with_and_without_affine() -> None:
    x = np.random.default_rng(0).standard_normal((3, 9)).astype(np.float32)
    weight = np.random.default_rng(1).standard_normal(9).astype(np.float32)

    with torch.no_grad():
        bare = torch.nn.functional.layer_norm(torch.from_numpy(x), (9,)).numpy()
        scaled = torch.nn.functional.layer_norm(
            torch.from_numpy(x), (9,), torch.from_numpy(weight), None
        ).numpy()

    np.testing.assert_allclose(
        np.asarray(primitives.layer_norm(x)), bare, atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(primitives.layer_norm(x, weight)), scaled, atol=2e-5, rtol=2e-5
    )
