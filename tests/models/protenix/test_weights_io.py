from __future__ import annotations

import gc
import io
import pickle
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models import _stacking
from foldjax.models.protenix.bridge import weights_io
from foldjax.models.protenix.bridge.weights_io import (
    _device_narrow_batch,
    _load_native_weights_with_field_dtype,
    _NativeWeightsUnpickler,
    load_native_weights,
    save_native_weights,
)
from foldjax.models.protenix.models.model import (
    ProtenixInferenceParams,
    cast_trunk_params,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams


def test_native_weights_roundtrip_namedtuple_tree(tmp_path) -> None:
    path = tmp_path / "weights.pkl.gz"
    params = {
        "linear": LinearParams(
            weight=jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
            bias=None,
        ),
        "flag": True,
    }

    save_native_weights(path, params)
    restored = load_native_weights(path)

    assert isinstance(restored["linear"], LinearParams)
    assert restored["linear"].bias is None
    assert restored["flag"] is True
    assert restored["linear"].weight.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(restored["linear"].weight),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )


def test_native_weights_roundtrip_uncompressed(tmp_path) -> None:
    path = tmp_path / "weights.pkl"
    params = LinearParams(
        weight=jnp.asarray([[3.0]], dtype=jnp.float32),
        bias=jnp.asarray([4.0], dtype=jnp.float32),
    )

    save_native_weights(path, params, compress=False)
    restored = load_native_weights(path)

    np.testing.assert_array_equal(np.asarray(restored.weight), [[3.0]])
    np.testing.assert_array_equal(np.asarray(restored.bias), [4.0])


def test_native_weights_resolve_pre_layout_parameter_modules() -> None:
    unpickler = _NativeWeightsUnpickler(io.BytesIO())

    restored_type = unpickler.find_class(
        "foldjax.models.protenix.models.primitives", "LinearParams"
    )

    assert restored_type is LinearParams


def test_native_weights_reject_arbitrary_pickle_globals(tmp_path) -> None:
    class ArbitraryReducer:
        def __reduce__(self):
            return eval, ("40 + 2",)

    path = tmp_path / "untrusted.pkl"
    with path.open("wb") as handle:
        pickle.dump(ArbitraryReducer(), handle)

    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        load_native_weights(path)


def test_failed_consuming_prestack_does_not_retain_the_private_loaded_tree(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "weights.pkl"
    params = {
        "first": [
            {
                "weight": np.full((8, 8), index, dtype=np.float32),
                "bias": np.full(8, index, dtype=np.float32),
            }
            for index in (1, 2)
        ],
        "second": [
            {
                "weight": np.full((8, 8), index, dtype=np.float32),
                "bias": np.full(8, index, dtype=np.float32),
            }
            for index in (101, 102)
        ],
    }
    save_native_weights(path, params, compress=False)
    references: list[weakref.ReferenceType[np.ndarray]] = []
    original = _stacking.stack_leaves

    def fail_on_second_stack(*leaves):
        references.extend(
            weakref.ref(leaf) for leaf in leaves if isinstance(leaf, np.ndarray)
        )
        if float(np.asarray(leaves[0]).reshape(-1)[0]) >= 100:
            raise RuntimeError("synthetic stack failure")
        return original(*leaves)

    monkeypatch.setattr(_stacking, "stack_leaves", fail_on_second_stack)

    with pytest.raises(RuntimeError, match="synthetic stack failure") as error:
        load_native_weights(path)
    del error
    gc.collect()

    assert references
    assert all(reference() is None for reference in references)


def test_prepared_trunk_load_matches_postload_device_cast_bits(tmp_path) -> None:
    """The bounded loader has the exact tree and leaf bytes of the old route."""

    path = tmp_path / "weights.pkl"
    finite_bits = np.asarray(
        [
            0x00000000,  # +0
            0x80000000,  # -0
            0x00000001,  # smallest FP32 subnormal
            0x007FFFFF,  # largest FP32 subnormal
            0x3F7F7FFF,  # immediately below one BF16 rounding boundary
            0x3F7F8000,  # tie
            0x3F7F8001,  # immediately above the tie
            0x7F7FFFFF,  # largest finite FP32
        ],
        dtype=np.uint32,
    ).view(np.float32)
    nonfinite_bits = np.asarray(
        [0x7F800000, 0xFF800000, 0x7F800001, 0xFFC12345], dtype=np.uint32
    ).view(np.float32)
    params = ProtenixInferenceParams(
        input_embedder={
            "finite": finite_bits,
            # With JAX x64 disabled, the historical path first narrows this to
            # FP32 on device and only then to BF16. The prepared loader must
            # take its device fallback rather than double-round on the host.
            "float64": np.asarray(
                [1.0000000596046448, -1.0000000596046448], dtype=np.float64
            ),
        },
        pairformer_output={"nonfinite": nonfinite_bits},
        diffusion={"float": np.asarray([1.25, -3.5], dtype=np.float32)},
        distogram={"integer": np.asarray([2, 3], dtype=np.int32)},
        confidence={"flag": True, "absent": None},
    )
    save_native_weights(path, params, compress=False)

    reference = cast_trunk_params(load_native_weights(path), jnp.bfloat16)
    actual = _load_native_weights_with_field_dtype(
        path,
        jnp.bfloat16,
        frozenset({"input_embedder", "pairformer_output"}),
        cast_batch_bytes=8,
    )

    assert jax.tree.structure(actual) == jax.tree.structure(reference)
    reference_leaves = jax.tree.leaves(reference, is_leaf=lambda leaf: leaf is None)
    actual_leaves = jax.tree.leaves(actual, is_leaf=lambda leaf: leaf is None)
    assert len(actual_leaves) == len(reference_leaves)
    for expected, observed in zip(reference_leaves, actual_leaves, strict=True):
        if hasattr(expected, "dtype"):
            assert type(observed) is type(expected)
            assert observed.dtype == expected.dtype
            assert observed.shape == expected.shape
            assert np.asarray(observed).tobytes() == np.asarray(expected).tobytes()
        elif expected is None or isinstance(expected, bool):
            assert observed is expected
        else:
            assert type(observed) is type(expected)
            assert observed == expected


def test_device_narrow_batch_releases_wide_buffers_only_after_completion(
    monkeypatch,
) -> None:
    """The NaN-safe fallback never leaves a completed FP32 wrapper resident."""

    events: list[tuple[str, int]] = []
    wide_values = []

    class Narrow:
        def __init__(self, index: int) -> None:
            self.index = index
            self.dtype = jnp.dtype(jnp.bfloat16)
            self.ready = False

        def block_until_ready(self):
            self.ready = True
            events.append(("ready", self.index))
            return self

    class Wide:
        def __init__(self, index: int) -> None:
            self.index = index
            self.dtype = jnp.dtype(jnp.float32)
            self.narrowed: Narrow | None = None
            self.deleted = False

        def astype(self, _dtype):
            self.narrowed = Narrow(self.index)
            events.append(("cast", self.index))
            return self.narrowed

        def delete(self) -> None:
            assert self.narrowed is not None and self.narrowed.ready
            self.deleted = True
            events.append(("delete", self.index))

    def fake_asarray(_value):
        wide = Wide(len(wide_values))
        wide_values.append(wide)
        return wide

    monkeypatch.setattr(weights_io.jnp, "asarray", fake_asarray)

    narrowed = _device_narrow_batch(
        [np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)],
        jnp.dtype(jnp.bfloat16),
    )

    assert [value.index for value in narrowed] == [0, 1]
    assert all(value.deleted for value in wide_values)
    assert events == [
        ("cast", 0),
        ("cast", 1),
        ("ready", 0),
        ("ready", 1),
        ("delete", 0),
        ("delete", 1),
    ]
