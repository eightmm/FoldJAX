"""The workspace-free Kabsch solve against the previous FP32 SVD and FP64.

Coordinates are compared when the optimum is unique. Rank-deficient and
reflection-degenerate inputs have nonunique rotations; their proper-rotation
objective, rather than an arbitrary SVD basis, is the numerical contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion as d


def _svd_align(x, target, weights, mask, *, backend="svd"):
    """The pre-change weighted_rigid_align, including FP32 products."""
    del backend
    precision = jax.lax.Precision.HIGHEST
    w = (mask * weights).astype(jnp.float32)[..., None]
    x, target = x.astype(jnp.float32), target.astype(jnp.float32)
    denominator = jnp.clip(jnp.sum(w, axis=-2, keepdims=True), min=1e-8)
    center = jnp.sum(x * w, axis=-2, keepdims=True) / denominator
    target_center = jnp.sum(target * w, axis=-2, keepdims=True) / denominator
    centered, target_centered = x - center, target - target_center
    correlation = jnp.einsum(
        "bni,bnj->bij", w * target_centered, centered, precision=precision
    )
    u, _, vh = jnp.linalg.svd(correlation)
    determinant = jnp.linalg.det(jnp.matmul(u, vh, precision=precision))
    flip = jnp.stack(
        (jnp.ones_like(determinant), jnp.ones_like(determinant), determinant),
        axis=-1,
    )
    rotation = jnp.matmul(u, flip[..., None] * vh, precision=precision)
    return (
        jnp.matmul(centered, rotation.swapaxes(-1, -2), precision=precision)
        + target_center
    )


def _svd_rotation64(correlation):
    u, _, vh = np.linalg.svd(np.asarray(correlation, np.float64))
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    return u @ vh


@pytest.mark.parametrize("backend", [None, "svd"])
def test_released_default_is_the_unchanged_svd(backend):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 32, 3)).astype("f4")
    target = rng.normal(size=x.shape).astype("f4")
    mask = np.ones(x.shape[:2], "f4")
    kwargs = {} if backend is None else {"backend": backend}
    result = jax.jit(lambda *a: d.weighted_rigid_align(*a, **kwargs))(
        x, target, mask, mask
    )
    expected = jax.jit(_svd_align)(x, target, mask, mask)
    np.testing.assert_array_equal(result, expected)
    assert d.DiffusionSettings().rigid_align_backend == "svd"


def test_unknown_alignment_backend_is_rejected():
    x = jnp.zeros((1, 4, 3), jnp.float32)
    mask = jnp.ones(x.shape[:2], jnp.float32)
    with pytest.raises(ValueError, match="alignment backend"):
        d.weighted_rigid_align(x, x, mask, mask, backend="power")


def test_random_rotation_matches_fp64_and_remains_proper():
    correlation = np.random.default_rng(17).normal(size=(4096, 3, 3)).astype("f4")
    rotation = np.asarray(jax.jit(d._rotation_from_correlation)(correlation))
    np.testing.assert_allclose(
        rotation, _svd_rotation64(correlation), atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        rotation @ rotation.swapaxes(-1, -2),
        np.broadcast_to(np.eye(3), rotation.shape),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 40.0, 1000.0])
@pytest.mark.parametrize("axes", [(1, 1, 1), (1, 1, 0), (1, 1e-3, 1e-3)])
def test_weighted_masked_coordinates_match_previous_svd(scale, axes):
    rng = np.random.default_rng(47)
    x = rng.normal(size=(128, 64, 3)).astype("f4") * np.asarray(axes, "f4")
    target = rng.normal(size=x.shape).astype("f4")
    weights = rng.uniform(0.0, 1.0, size=x.shape[:2]).astype("f4")
    mask = (rng.uniform(size=weights.shape) > 0.2).astype("f4")
    x, target = x * scale, target * scale
    actual = np.asarray(
        jax.jit(lambda *a: d.weighted_rigid_align(*a, backend="jacobi"))(
            x, target, weights, mask
        )
    )
    expected = np.asarray(jax.jit(_svd_align)(x, target, weights, mask))
    np.testing.assert_allclose(actual / scale, expected / scale, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "singular_values",
    [(3.0, 2.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1e-4, 1e-8), (1, 0, 0), (0, 0, 0)],
)
def test_reflections_and_rank_deficiency_preserve_optimal_proper_objective(
    singular_values,
):
    rng = np.random.default_rng(92)
    left = np.linalg.qr(rng.normal(size=(32, 3, 3)))[0]
    right = np.linalg.qr(rng.normal(size=(32, 3, 3)))[0]
    # Negative determinant requires the proper-rotation reflection correction.
    left[..., :, 0] *= -1
    correlation = (
        (left * np.asarray(singular_values)[None, None, :]) @ right.swapaxes(-1, -2)
    ).astype("f4")
    rotation = np.asarray(jax.jit(d._rotation_from_correlation)(correlation))
    expected = _svd_rotation64(correlation)
    actual_objective = np.sum(rotation * correlation, axis=(-2, -1))
    expected_objective = np.sum(expected * correlation, axis=(-2, -1))
    np.testing.assert_allclose(
        actual_objective, expected_objective, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


@pytest.mark.parametrize("valid_atoms", [0, 1, 2, 16])
def test_degenerate_and_empty_masks_keep_valid_coordinates(valid_atoms):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 16, 3)).astype("f4")
    target = rng.normal(size=x.shape).astype("f4")
    mask = np.broadcast_to(np.arange(16) < valid_atoms, x.shape[:2]).astype("f4")
    # Zero/one/two valid atoms are underdetermined. The positions actually used
    # by the sampler still coincide, even if padding rotates differently.
    actual = np.asarray(
        jax.jit(lambda *a: d.weighted_rigid_align(*a, backend="jacobi"))(
            x, target, mask, mask
        )
    )
    expected = np.asarray(jax.jit(_svd_align)(x, target, mask, mask))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[:, :valid_atoms], expected[:, :valid_atoms], atol=2e-5, rtol=2e-5
    )


def test_alignment_lowering_contains_no_solver_custom_call():
    x = jax.ShapeDtypeStruct((2, 64, 3), jnp.float32)
    mask = jax.ShapeDtypeStruct((2, 64), jnp.float32)
    hlo = (
        jax.jit(lambda *a: d.weighted_rigid_align(*a, backend="jacobi"))
        .lower(x, x, mask, mask)
        .as_text()
    )
    assert "custom_call" not in hlo
    assert "@svd" not in hlo.lower()
    assert "@eigh" not in hlo.lower()


@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_noise_sampler_matches_previous_alignment(monkeypatch, compiled):
    settings = d.DiffusionSettings(
        num_steps=3, c_token=2, noise_scale=1.003, rigid_align_backend="jacobi"
    )
    atom_mask = jnp.asarray([[1.0] * 15 + [0.0]], jnp.float32)
    cache = SimpleNamespace(atom_mask=atom_mask, n_tokens=2)

    def denoise(x, sigma, *args, **kwargs):
        coordinates = x / (1 + sigma[:, None, None])
        coordinates = coordinates * jnp.asarray([1.0, 0.9, 1.1])
        return coordinates, jnp.broadcast_to(x.mean(), (1, 2, 2))

    monkeypatch.setattr(d, "diffusion_module", denoise)
    rng = np.random.default_rng(42)
    names = (
        "diffusion_initial_normal",
        "diffusion_rotation_quaternions",
        "diffusion_translations",
        "diffusion_churn_normals",
    )
    shapes = ((1, 16, 3), (3, 1, 4), (3, 1, 1, 3), (3, 1, 16, 3))
    tape = {
        name: jnp.asarray(rng.normal(size=shape), jnp.float32)
        for name, shape in zip(names, shapes, strict=True)
    }

    def run():
        return d.sample(
            jax.random.key(5),
            jnp.zeros((1, 2, 2)),
            cache,
            {},
            settings=settings,
            **tape,
        )

    actual = jax.jit(run)() if compiled else run()
    monkeypatch.setattr(d, "weighted_rigid_align", _svd_align)
    expected = jax.jit(lambda: run())() if compiled else run()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, atol=2e-4, rtol=2e-5)
