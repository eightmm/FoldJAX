import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.augmentation import (
    AugmentationTape,
    centre_augmentation_from_draws,
    centre_random_augmentation,
    validate_augmentation_tape,
)


@pytest.mark.parametrize("compiled", [False, True])
def test_empty_sample_mask_matches_native_zero_output(compiled):
    coordinates = jnp.arange(42, dtype=jnp.float32).reshape(2, 7, 3)
    mask = jnp.ones((2, 7)).at[0].set(0)
    fn = jax.jit(centre_random_augmentation) if compiled else centre_random_augmentation
    result = fn(jax.random.key(17), coordinates, mask)
    assert np.isfinite(np.asarray(result)).all()
    np.testing.assert_array_equal(result[0], np.zeros((7, 3), np.float32))
    # Guard the ordinary row against changing RNG or valid-atom arithmetic.
    control = fn(jax.random.key(17), coordinates, jnp.ones_like(mask))
    np.testing.assert_array_equal(result[1], control[1])


def test_native_draws_use_quaternion_normalisation_and_masked_centroid():
    coordinates = jnp.asarray(
        [[[0, 0, 0], [2, 0, 0], [1e6, 1e6, 1e6]]] * 2, dtype=jnp.float32
    )
    mask = jnp.asarray([[1, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    quaternion = jnp.asarray([[2, 0, 0, 2], [2, 0, 0, 2]], dtype=jnp.float32)
    translation = jnp.asarray([[3, 4, 5], [3, 4, 5]], dtype=jnp.float32)
    actual = jax.jit(centre_augmentation_from_draws)(
        coordinates, mask, quaternion, translation
    )
    expected = np.asarray([[[3, 3, 5], [3, 5, 5], [0, 0, 0]], [[0, 0, 0]] * 3])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_native_batch_axis_is_removed_without_changing_draw_values():
    q = np.arange(40, dtype=np.float32).reshape(2, 1, 5, 4)
    t = np.arange(30, dtype=np.float32).reshape(2, 1, 5, 3)
    tape = validate_augmentation_tape(AugmentationTape(q, t), steps=2, samples=5)
    np.testing.assert_array_equal(tape.quaternions, q[:, 0])
    np.testing.assert_array_equal(tape.translations, t[:, 0])


@pytest.mark.parametrize(
    ("q_shape", "t_shape"),
    [
        ((1, 5, 4), (2, 5, 3)),
        ((2, 4, 4), (2, 5, 3)),
        ((2, 5, 3), (2, 5, 3)),
        ((2, 2, 5, 4), (2, 1, 5, 3)),
        ((2, 5, 4), (3, 5, 3)),
        ((2, 5, 4), (2, 5, 1, 3)),
    ],
)
def test_malformed_native_tape_shape_is_rejected(q_shape, t_shape):
    tape = AugmentationTape(jnp.ones(q_shape), jnp.ones(t_shape))
    with pytest.raises(ValueError, match="expected shape"):
        validate_augmentation_tape(tape, steps=2, samples=5)


def test_tape_type_and_dtype_are_not_silently_coerced():
    q = jnp.ones((2, 5, 4))
    t = jnp.ones((2, 5, 3))
    with pytest.raises(TypeError, match="must be an AugmentationTape"):
        validate_augmentation_tape((q, t), steps=2, samples=5)
    with pytest.raises(TypeError, match="floating dtype"):
        validate_augmentation_tape(
            AugmentationTape(q.astype(jnp.int32), t), steps=2, samples=5
        )
    with pytest.raises(TypeError, match="draw dtypes must match"):
        validate_augmentation_tape(
            AugmentationTape(q.astype(jnp.bfloat16), t), steps=2, samples=5
        )
    with pytest.raises(TypeError, match="does not match coordinate dtype"):
        validate_augmentation_tape(
            AugmentationTape(q, t), steps=2, samples=5, dtype=jnp.bfloat16
        )


def test_replay_preserves_the_ordinary_random_draw_mapping():
    key = jax.random.key(82)
    q_key, t_key = jax.random.split(key)
    q = jax.random.normal(q_key, (5, 4))
    t = jax.random.normal(t_key, (5, 3))
    x = jnp.arange(105, dtype=jnp.float32).reshape(5, 7, 3) / 10
    mask = jnp.ones((5, 7))
    expected = centre_random_augmentation(key, x, mask)
    actual = centre_augmentation_from_draws(x, mask, q, t)
    np.testing.assert_array_equal(actual, expected)


def test_tape_staging_cannot_silently_narrow_numpy_float64():
    with jax.enable_x64(False):
        q = np.ones((2, 5, 4), dtype=np.float64)
        t = np.ones((2, 5, 3), dtype=np.float64)
        with pytest.raises(TypeError, match="cannot be preserved"):
            validate_augmentation_tape(AugmentationTape(q, t), steps=2, samples=5)


@pytest.mark.parametrize(
    "bad", ["zero", "nan", "inf", "tiny", "subnormal", "huge", "translation_nan"]
)
def test_concrete_value_gate_rejects_invalid_native_draws(bad):
    q = jnp.ones((2, 5, 4))
    t = jnp.ones((2, 5, 3))
    if bad == "translation_nan":
        t = t.at[0, 0, 0].set(jnp.nan)
    else:
        value = {
            "zero": 0,
            "nan": jnp.nan,
            "inf": jnp.inf,
            "tiny": 1e-35,
            "subnormal": 1e-20,
            "huge": 1e38,
        }[bad]
        q = q.at[0, 0].set(value)
    with pytest.raises(ValueError, match="finite"):
        validate_augmentation_tape(
            AugmentationTape(q, t), steps=2, samples=5, check_values=True
        )


def test_traced_value_validation_requires_explicit_host_preflight():
    def validate(q):
        return validate_augmentation_tape(
            AugmentationTape(q, jnp.ones((2, 5, 3))),
            steps=2,
            samples=5,
            check_values=True,
        ).quaternions

    with pytest.raises(ValueError, match="require concrete arrays"):
        jax.jit(validate)(jnp.ones((2, 5, 4)))


def test_direct_traced_zero_quaternion_retains_native_nonfinite_semantics():
    x = jnp.arange(21, dtype=jnp.float32).reshape(1, 7, 3)
    mask = jnp.ones((1, 7))
    translation = jnp.zeros((1, 3))
    apply = jax.jit(centre_augmentation_from_draws)
    # Runtime arithmetic must not hide an invalid tape by changing zero to an
    # identity quaternion. Public prediction entry points reject this earlier.
    invalid = apply(x, mask, jnp.zeros((1, 4)), translation)
    assert not np.isfinite(np.asarray(invalid)).all()
    normal = apply(x, mask, jnp.asarray([[2.0, 0.0, 0.0, 2.0]]), translation)
    small = apply(x, mask, jnp.asarray([[2e-10, 0.0, 0.0, 2e-10]]), translation)
    np.testing.assert_allclose(small, normal, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "dtype,value", [(jnp.float16, 1000.0), (jnp.float16, 1e-4), (jnp.bfloat16, 1e-20)]
)
def test_low_precision_tape_is_not_admitted_by_an_fp32_host_norm(dtype, value):
    q = jnp.full((2, 5, 4), value, dtype=dtype)
    t = jnp.zeros((2, 5, 3), dtype=dtype)
    # These cases have finite FP32 norms but invalid norms in their real dtype.
    assert np.isfinite(np.linalg.norm(np.asarray(q, np.float32), axis=-1)).all()
    assert np.all(np.linalg.norm(np.asarray(q, np.float32), axis=-1) > 0)
    actual_norm = np.asarray(jnp.linalg.norm(q, axis=-1), dtype=np.float32)
    assert not np.isfinite(actual_norm).all() or np.any(actual_norm == 0)
    with pytest.raises(TypeError, match="requires FP32/FP64"):
        validate_augmentation_tape(
            AugmentationTape(q, t), steps=2, samples=5, check_values=True
        )
