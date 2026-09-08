from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion as d


def _fixture(monkeypatch):
    settings = d.DiffusionSettings(num_steps=3, c_token=2, noise_scale=1.003)
    cache = SimpleNamespace(
        atom_mask=jnp.array([[1, 1, 1, 0]], jnp.float32), n_tokens=2
    )

    def denoise(x, sigma, *args, **kwargs):
        return x / (1 + sigma[:, None, None]), jnp.broadcast_to(x.mean(), (1, 2, 2))

    monkeypatch.setattr(d, "diffusion_module", denoise)
    rng = np.random.default_rng(9)
    tape = dict(
        zip(
            (
                "diffusion_initial_normal",
                "diffusion_rotation_quaternions",
                "diffusion_translations",
                "diffusion_churn_normals",
            ),
            [
                jnp.asarray(rng.normal(size=s), jnp.float32)
                for s in ((1, 4, 3), (3, 1, 4), (3, 1, 1, 3), (3, 1, 4, 3))
            ],
            strict=True,
        )
    )

    def run(key, **draws):
        return d.sample(
            key, jnp.zeros((1, 2, 2)), cache, {}, settings=settings, **draws
        )

    return run, tape


@pytest.mark.parametrize("compiled", [False, True])
def test_sampler_tape_is_key_independent(monkeypatch, compiled):
    run, tape = _fixture(monkeypatch)
    if compiled:
        run = jax.jit(run)
    first = run(jax.random.key(0), **tape)
    second = run(jax.random.key(99), **tape)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    changed = dict(tape, diffusion_churn_normals=tape["diffusion_churn_normals"] + 1)
    assert not np.array_equal(first[0], run(jax.random.key(0), **changed)[0])


def test_native_augmentation_transforms_current_and_previous_without_mask_zero():
    x = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    q = jnp.array([[1, 2, 3, 4]], jnp.float32)
    t = jnp.array([[[4, 5, 6]]], jnp.float32)
    mask = jnp.array([[1, 1, 1, 0]], jnp.float32)
    first, second = d.center_random_augmentation(
        jax.random.key(0), x, mask, x + 2, quaternion=q, translation=t
    )
    mean = x[:, :3].mean(axis=1, keepdims=True)
    r = d.quaternion_to_rotation(q)
    np.testing.assert_allclose(first, (x - mean) @ r + t, atol=1e-6)
    np.testing.assert_allclose(second, (x + 2 - mean) @ r + t, atol=1e-6)
    assert np.any(np.asarray(first[:, 3]) != 0)


@pytest.mark.parametrize(
    "bad",
    [
        "partial",
        "shape",
        "dtype",
        "nan",
        "zero_q",
        "huge_q",
        "subnormal_q",
        "near_overflow_q",
    ],
)
def test_sampler_rejects_bad_tape(monkeypatch, bad):
    run, tape = _fixture(monkeypatch)
    if bad == "partial":
        tape.pop("diffusion_translations")
    elif bad == "shape":
        tape["diffusion_churn_normals"] = tape["diffusion_churn_normals"][:2]
    elif bad == "dtype":
        tape["diffusion_initial_normal"] = tape["diffusion_initial_normal"].astype(
            jnp.bfloat16
        )
    elif bad == "nan":
        tape["diffusion_initial_normal"] = (
            tape["diffusion_initial_normal"].at[0, 0, 0].set(jnp.nan)
        )
    else:
        tape["diffusion_rotation_quaternions"] = jnp.full(
            (3, 1, 4),
            {
                "zero_q": 0,
                "huge_q": 1e30,
                "subnormal_q": 1e-19,
                "near_overflow_q": 8e18,
            }[bad],
            jnp.float32,
        )
    with pytest.raises(ValueError, match="diffusion tape"):
        run(jax.random.key(0), **tape)


def test_explicit_value_preflight_is_required_before_dynamic_jit(monkeypatch):
    _, tape = _fixture(monkeypatch)
    values = tuple(tape.values())
    d.validate_diffusion_tape(*values, steps=3, batch=1, atoms=4)
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda initial: d.validate_diffusion_tape(
                initial, *values[1:], steps=3, batch=1, atoms=4
            )
        )(values[0])


def test_tape_matches_ordinary_rng_draw_order_and_schedule(monkeypatch):
    run, _ = _fixture(monkeypatch)
    key = jax.random.key(8)
    next_key, initial_key = jax.random.split(key)
    q, t, churn = [], [], []
    for _ in range(3):
        next_key, aug, noise = jax.random.split(next_key, 3)
        rotation, translation = jax.random.split(aug)
        q.append(jax.random.normal(rotation, (1, 4)))
        t.append(jax.random.normal(translation, (1, 1, 3)))
        churn.append(jax.random.normal(noise, (1, 4, 3)))
    taped = run(
        key,
        diffusion_initial_normal=jax.random.normal(initial_key, (1, 4, 3)),
        diffusion_rotation_quaternions=jnp.stack(q),
        diffusion_translations=jnp.stack(t),
        diffusion_churn_normals=jnp.stack(churn),
    )
    ordinary = run(key)
    for a, b in zip(taped, ordinary, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
