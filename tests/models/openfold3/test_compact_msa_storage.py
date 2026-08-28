"""OpenFold3's private categorical MSA storage preserves the released graph."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.data import _numpy_featurization as numpy_features
from foldjax.models.openfold3.data import (
    compact_msa_features,
    pad_features,
    subsample_msa_rows,
)
from foldjax.models.openfold3.data.featurize import (
    _COMPACT_MSA_INDICES,
    _COMPACT_MSA_MARKER,
)
from foldjax.models.openfold3.data.validation import validate_features
from foldjax.models.openfold3.models.input_embedders import (
    MSAEmbedderParams,
    msa_embedder,
)
from foldjax.models.openfold3.models.primitives import LinearParams
from tests.models.cp_probe_env import inherited_environment
from tests.models.openfold3.feature_fixture import minimal_features


def _params(dtype: jnp.dtype, *, nonfinite: bool = False) -> MSAEmbedderParams:
    rng = np.random.default_rng(29)
    msa_weight = rng.normal(size=(11, 34)).astype(np.float32)
    if nonfinite:
        msa_weight[0, 1] = np.nan
        msa_weight[1, 7] = np.inf
        msa_weight[2, 9] = -np.inf
        msa_weight[3, 0] = -0.0
    return MSAEmbedderParams(
        linear_m=LinearParams(
            weight=jnp.asarray(msa_weight, dtype=dtype),
            bias=jnp.asarray(rng.normal(size=(11,)), dtype=dtype),
        ),
        linear_s_input=LinearParams(
            weight=jnp.asarray(rng.normal(size=(11, 7)), dtype=dtype),
            bias=jnp.asarray(rng.normal(size=(11,)), dtype=dtype),
        ),
    )


def _varied_msa_features(*, rows: int = 3, tokens: int = 5) -> dict[str, np.ndarray]:
    features = minimal_features(
        tokens=tokens,
        atoms=max(tokens, tokens + 2),
        msa_rows=rows,
    )
    categories = np.arange(rows * tokens, dtype=np.int32).reshape(rows, tokens) % 32
    features["msa"].fill(0)
    np.put_along_axis(
        features["msa"],
        categories[None, ..., None],
        1,
        axis=-1,
    )
    values = np.arange(rows * tokens, dtype=np.float32).reshape(1, rows, tokens)
    features["has_deletion"] = (values % 2).astype(np.float32)
    features["deletion_value"] = np.arctan(values / 3.0).astype(np.float32)
    return features


def _assert_same(actual: jax.Array, expected: jax.Array) -> None:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    assert actual_np.dtype == expected_np.dtype
    assert actual_np.shape == expected_np.shape
    assert actual_np.tobytes() == expected_np.tobytes()


def test_compaction_round_trips_dense_categories_and_zero_padding() -> None:
    dense = pad_features(
        _varied_msa_features(),
        n_token=8,
        n_atom=13,
        n_msa=6,
        n_templates=1,
    )

    compact = compact_msa_features(dense)

    assert "msa" not in compact
    indices = compact[_COMPACT_MSA_INDICES]
    assert indices.shape == (1, 6, 8)
    assert indices.dtype == np.uint8
    np.testing.assert_array_equal(indices[0, :3, :5], np.arange(15).reshape(3, 5))
    assert np.all(indices[0, 3:] == 32)
    assert np.all(indices[0, :3, 5:] == 32)
    assert compact[_COMPACT_MSA_MARKER].shape == ()
    validate_features(compact, _allow_compact_msa=True)

    # A second serving bucket pads compact indices directly. Its suffix must
    # remain the zero-vector sentinel, not category zero.
    repadded = pad_features(
        compact,
        n_token=10,
        n_atom=15,
        n_msa=8,
        n_templates=1,
    )
    assert np.all(repadded[_COMPACT_MSA_INDICES][0, 6:] == 32)
    assert np.all(repadded[_COMPACT_MSA_INDICES][0, :, 8:] == 32)
    validate_features(repadded, _allow_compact_msa=True)


def test_compact_subsampling_selects_the_same_rows_as_dense() -> None:
    dense = _varied_msa_features(rows=6, tokens=5)
    dense["msa_mask"][:, [0, 4]] = 0
    compact = compact_msa_features(dense)

    dense_selected = subsample_msa_rows(dense, 3)
    compact_selected = subsample_msa_rows(compact, 3)

    np.testing.assert_array_equal(
        compact_selected[_COMPACT_MSA_INDICES],
        np.argmax(dense_selected["msa"], axis=-1).astype(np.uint8),
    )
    for name in ("msa_mask", "has_deletion", "deletion_value"):
        np.testing.assert_array_equal(compact_selected[name], dense_selected[name])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda msa: msa.__setitem__((0, 0, 0, 1), 1),
        lambda msa: msa.__setitem__((0, 0, 0, 0), -1),
        lambda msa: msa.__setitem__((0, 0, 0, 0), 2),
    ],
)
def test_invalid_dense_custom_msa_does_not_activate_private_storage(mutate) -> None:
    features = _varied_msa_features()
    mutate(features["msa"])
    features[_COMPACT_MSA_MARKER] = np.zeros((), dtype=np.float32)
    features[_COMPACT_MSA_INDICES] = np.zeros(
        features["msa"].shape[:-1], dtype=np.uint8
    )

    compact = compact_msa_features(features)

    assert compact["msa"] is features["msa"]
    assert _COMPACT_MSA_MARKER not in compact
    assert _COMPACT_MSA_INDICES not in compact


def test_wrong_dense_dtype_and_shape_keep_the_public_mapping() -> None:
    for msa in (
        np.zeros((1, 2, 3, 32), dtype=np.float32),
        np.zeros((1, 2, 3, 31), dtype=np.int32),
        np.zeros((2, 3, 32), dtype=np.int32),
        np.zeros((2, 2, 3, 32), dtype=np.int32),
    ):
        features = {"msa": msa}
        compact = compact_msa_features(features)
        assert compact["msa"] is msa
        assert _COMPACT_MSA_MARKER not in compact


def test_precursor_can_emit_compact_storage_without_building_one_hot() -> None:
    rows, tokens = 4, 6
    indices = np.arange(rows * tokens, dtype=np.int64).reshape(rows, tokens) % 32
    precursor = numpy_features._MsaPrecursor(
        msa_index=indices,
        deletion_matrix=np.arange(rows * tokens, dtype=np.int64).reshape(rows, tokens),
        n_rows_paired=3,
        msa_mask=np.ones((rows, tokens), dtype=np.float32),
        profile=np.zeros((tokens, 32), dtype=np.float32),
        deletion_mean=np.zeros(tokens, dtype=np.float32),
    )

    dense = numpy_features._tensorize_msa_precursor(precursor)
    compact = numpy_features._tensorize_msa_precursor(
        precursor, compact_msa=True
    )

    assert "msa" not in compact
    np.testing.assert_array_equal(compact[_COMPACT_MSA_INDICES], indices)
    np.testing.assert_array_equal(
        dense["msa"], np.eye(32, dtype=np.int32)[compact[_COMPACT_MSA_INDICES]]
    )
    for name in (
        "has_deletion",
        "deletion_value",
        "deletion_mean",
        "profile",
        "num_paired_seqs",
        "msa_mask",
    ):
        np.testing.assert_array_equal(compact[name], dense[name], err_msg=name)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_dense_and_compact_embedding_are_byte_identical(dtype: jnp.dtype) -> None:
    dense = pad_features(
        _varied_msa_features(),
        n_token=8,
        n_atom=13,
        n_msa=6,
        n_templates=1,
    )
    compact = compact_msa_features(dense)
    params = _params(dtype)
    s_input = jnp.asarray(
        np.random.default_rng(31).normal(size=(1, 8, 7)), dtype=dtype
    )

    run = jax.jit(lambda batch: msa_embedder(batch, s_input, params)[0])
    expected = run(dense)
    actual = run(compact)
    jax.block_until_ready((expected, actual))

    _assert_same(actual, expected)


def test_nonfinite_weights_and_signed_zero_keep_dense_dot_semantics() -> None:
    dense = pad_features(
        _varied_msa_features(),
        n_token=7,
        n_atom=12,
        n_msa=5,
        n_templates=1,
    )
    compact = compact_msa_features(dense)
    params = _params(jnp.float32, nonfinite=True)
    s_input = jnp.zeros((1, 7, 7), dtype=jnp.float32)

    expected = jax.jit(lambda batch: msa_embedder(batch, s_input, params)[0])(dense)
    actual = jax.jit(lambda batch: msa_embedder(batch, s_input, params)[0])(compact)
    expected_np = np.asarray(expected)
    actual_np = np.asarray(actual)

    np.testing.assert_array_equal(np.isnan(actual_np), np.isnan(expected_np))
    finite = np.isfinite(expected_np)
    assert actual_np[finite].tobytes() == expected_np[finite].tobytes()
    zeros = finite & (expected_np == 0)
    np.testing.assert_array_equal(
        np.signbit(actual_np[zeros]), np.signbit(expected_np[zeros])
    )


def test_dense_input_wins_over_stale_or_partial_private_keys() -> None:
    rng = np.random.default_rng(37)
    dense = {
        "msa": jnp.asarray(rng.normal(size=(1, 2, 3, 32)), dtype=jnp.float32),
        "has_deletion": jnp.asarray(rng.normal(size=(1, 2, 3)), dtype=jnp.float32),
        "deletion_value": jnp.asarray(
            rng.normal(size=(1, 2, 3)), dtype=jnp.float32
        ),
        "msa_mask": jnp.ones((1, 2, 3), dtype=jnp.float32),
    }
    dirty = {
        **dense,
        _COMPACT_MSA_MARKER: jnp.ones((2,), dtype=jnp.int32),
        _COMPACT_MSA_INDICES: jnp.full((1, 2, 3), 255, dtype=jnp.uint8),
    }
    params = _params(jnp.float32)
    s_input = jnp.asarray(rng.normal(size=(1, 3, 7)), dtype=jnp.float32)

    expected, _ = msa_embedder(dense, s_input, params)
    actual, _ = msa_embedder(dirty, s_input, params)
    _assert_same(actual, expected)

    with pytest.raises(KeyError, match="complete private"):
        msa_embedder(
            {
                name: value
                for name, value in dirty.items()
                if name not in {"msa", _COMPACT_MSA_INDICES}
            },
            s_input,
            params,
        )


def test_compact_validation_rejects_bad_marker_index_and_sentinel_precedence() -> None:
    compact = compact_msa_features(_varied_msa_features())
    validate_features(compact, _allow_compact_msa=True)

    bad_marker = dict(compact)
    bad_marker[_COMPACT_MSA_MARKER] = np.ones((), dtype=np.float32)
    with pytest.raises(ValueError, match="marker"):
        validate_features(bad_marker, _allow_compact_msa=True)

    bad_index = dict(compact)
    bad_index[_COMPACT_MSA_INDICES] = compact[_COMPACT_MSA_INDICES].copy()
    bad_index[_COMPACT_MSA_INDICES][0, 0, 0] = 33
    with pytest.raises(ValueError, match=r"\[0, 32\]"):
        validate_features(bad_index, _allow_compact_msa=True)

    bad_sentinel = dict(compact)
    bad_sentinel[_COMPACT_MSA_INDICES] = compact[_COMPACT_MSA_INDICES].copy()
    bad_sentinel[_COMPACT_MSA_INDICES][0, 0, 0] = 32
    with pytest.raises(ValueError, match="sentinel"):
        validate_features(bad_sentinel, _allow_compact_msa=True)

    partial = dict(compact)
    del partial[_COMPACT_MSA_MARKER]
    with pytest.raises(ValueError, match="missing"):
        validate_features(partial, _allow_compact_msa=True)


def test_dense_and_compact_pytrees_use_two_bounded_jit_entries() -> None:
    dense = _varied_msa_features()
    compact = compact_msa_features(dense)
    params = _params(jnp.float32)
    s_input = jnp.zeros((1, 5, 7), dtype=jnp.float32)
    run = jax.jit(lambda batch: msa_embedder(batch, s_input, params)[0])
    try:
        first = run(dense)
        second = run(compact)
        repeated = run(compact)
        jax.block_until_ready((first, second, repeated))
        assert run._cache_size() == 2
        _assert_same(second, first)
        _assert_same(repeated, first)
    finally:
        run.clear_cache()


def test_compact_storage_scales_entry_bytes_without_widening_cpu_temp() -> None:
    rows, tokens, c_s, c_m = 32, 64, 16, 16
    categories = (
        np.arange(rows * tokens, dtype=np.int32).reshape(1, rows, tokens) % 32
    )
    dense = {
        "msa": np.eye(32, dtype=np.int32)[categories],
        "has_deletion": np.zeros((1, rows, tokens), dtype=np.float32),
        "deletion_value": np.zeros((1, rows, tokens), dtype=np.float32),
        "msa_mask": np.ones((1, rows, tokens), dtype=np.float32),
    }
    compact = compact_msa_features(dense)
    params = MSAEmbedderParams(
        LinearParams(jnp.ones((c_m, 34), dtype=jnp.float32)),
        LinearParams(jnp.ones((c_m, c_s), dtype=jnp.float32)),
    )
    s_input = jnp.ones((1, tokens, c_s), dtype=jnp.float32)
    run = jax.jit(lambda batch: msa_embedder(batch, s_input, params)[0])

    dense_lowered = run.lower(dense)
    dense_compiled = dense_lowered.compile()
    compact_lowered = run.lower(compact)
    compact_compiled = compact_lowered.compile()
    dense_memory = dense_compiled.memory_analysis()
    compact_memory = compact_compiled.memory_analysis()

    assert (
        compact_memory.argument_size_in_bytes * 4
        < dense_memory.argument_size_in_bytes
    )
    assert compact_memory.output_size_in_bytes == dense_memory.output_size_in_bytes
    assert compact_memory.temp_size_in_bytes == dense_memory.temp_size_in_bytes
    # One iota/compare one-hot reconstruction adds a constant-size fragment; it
    # must not make StableHLO grow with the 32-channel host representation.
    assert len(compact_lowered.as_text()) <= len(dense_lowered.as_text()) + 5_000
    _assert_same(compact_compiled(compact), dense_compiled(dense))


_CP_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.openfold3.data import compact_msa_features
    from foldjax.models.openfold3.models.input_embedders import (
        MSAEmbedderParams,
        msa_embedder,
    )
    from foldjax.models.openfold3.models.primitives import LinearParams

    assert jax.device_count() == 4, jax.devices()
    rows, tokens = 8, 8
    categories = np.arange(rows * tokens, dtype=np.int32).reshape(1, rows, tokens) % 32
    msa = np.eye(32, dtype=np.int32)[categories]
    dense = {
        "msa": msa,
        "has_deletion": np.zeros((1, rows, tokens), dtype=np.float32),
        "deletion_value": np.zeros((1, rows, tokens), dtype=np.float32),
        "msa_mask": np.ones((1, rows, tokens), dtype=np.float32),
    }
    compact = compact_msa_features(dense)
    params = MSAEmbedderParams(
        LinearParams(jnp.arange(34 * 8, dtype=jnp.float32).reshape(8, 34) / 97),
        LinearParams(jnp.arange(6 * 8, dtype=jnp.float32).reshape(8, 6) / 89),
    )
    s_input = jnp.arange(tokens * 6, dtype=jnp.float32).reshape(1, tokens, 6) / 31
    traced = []

    def build(source):
        def run():
            traced.append(cp_layout())
            return msa_embedder(source, s_input, params)[0]
        return run

    reference = jax.device_get(jax.jit(build(dense))())
    for layout in ("1d", "2d"):
        jax.clear_caches()
        with context_parallel(4, layout=layout):
            dense_result = jax.device_get(jax.jit(build(dense))())
            compact_result = jax.device_get(jax.jit(build(compact))())
        assert dense_result.tobytes() == reference.tobytes(), layout
        assert compact_result.tobytes() == reference.tobytes(), layout

    assert traced == [None, "1d", "1d", "2d", "2d"], traced
    print("OPENFOLD3_COMPACT_MSA_CP_OK")
    """
)


def test_compact_msa_matches_dense_on_forced_four_cpu_devices() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_COMPACT_MSA_CP_OK" in completed.stdout
