from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.protenix.data.static_io import (
    flatten_output_dict,
    load_static_feature_npz,
    save_output_npz,
    save_static_feature_npz,
)


def test_load_static_feature_npz_groups_pad_info(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        restype=np.zeros((2, 32), dtype=np.float32),
        **{"pad_info.mask_trunked": np.ones((2, 2, 4), dtype=bool)},
    )

    features = load_static_feature_npz(path)

    assert set(features) == {"restype", "pad_info"}
    np.testing.assert_array_equal(features["restype"], np.zeros((2, 32)))
    np.testing.assert_array_equal(
        features["pad_info"]["mask_trunked"],
        np.ones((2, 2, 4), dtype=bool),
    )


def test_save_static_feature_npz_roundtrips_pad_info(tmp_path) -> None:
    path = tmp_path / "features.npz"
    save_static_feature_npz(
        path,
        {
            "restype": np.zeros((2, 32), dtype=np.float32),
            "pad_info": {"mask_trunked": np.ones((2, 2, 4), dtype=bool)},
        },
    )

    features = load_static_feature_npz(path)

    assert set(features) == {"restype", "pad_info"}
    np.testing.assert_array_equal(features["restype"], np.zeros((2, 32)))
    np.testing.assert_array_equal(
        features["pad_info"]["mask_trunked"],
        np.ones((2, 2, 4), dtype=bool),
    )


def test_flatten_output_dict_skips_trunk_by_default() -> None:
    output = {
        "coordinate": np.ones((1, 3, 3)),
        "s_inputs": np.ones((2, 67)),
        "nested": {"value": np.asarray([1.0])},
    }

    flat, omitted = flatten_output_dict(output)

    assert set(flat) == {"coordinate", "nested.value"}
    assert omitted == ()


def test_flatten_output_dict_omits_oversized_arrays_without_copying() -> None:
    """An output larger than the budget must be reported, not transferred.

    At 2030 tokens one confidence output is 37 GiB -- bigger than the model's own
    peak -- and copying it to host killed a prediction that had already finished.
    The size comes from the array's metadata, so the decision costs no transfer.
    """
    output = {
        "coordinate": np.ones((1, 2, 3), dtype=np.float32),
        "big": np.zeros((256, 256, 64), dtype=np.float32),
    }

    flat, omitted = flatten_output_dict(output, max_bytes=1 << 20)

    assert set(flat) == {"coordinate"}
    assert [name for name, _ in omitted] == ["big"]
    assert omitted[0][1] == output["big"].nbytes


def test_flatten_output_dict_can_be_told_to_keep_everything() -> None:
    output = {"big": np.zeros((64, 64), dtype=np.float32)}
    flat, omitted = flatten_output_dict(output, max_bytes=None)
    assert set(flat) == {"big"}
    assert omitted == ()


def test_save_output_npz_reports_what_it_left_out(tmp_path) -> None:
    path = tmp_path / "out.npz"
    omitted = save_output_npz(
        path,
        {
            "coordinate": np.ones((1, 2, 3), dtype=np.float32),
            "big": np.zeros((256, 256, 64), dtype=np.float32),
        },
        max_bytes=1 << 20,
    )
    assert [name for name, _ in omitted] == ["big"]
    with np.load(path) as data:
        assert set(data.files) == {"coordinate"}


def test_save_output_npz_writes_compressed_arrays(tmp_path) -> None:
    path = tmp_path / "out.npz"
    save_output_npz(
        path,
        {
            "coordinate": np.ones((1, 2, 3)),
            "s_trunk": np.ones((2, 2)),
        },
    )

    with np.load(path) as data:
        assert set(data.files) == {"coordinate"}
        np.testing.assert_array_equal(data["coordinate"], np.ones((1, 2, 3)))


def test_static_feature_npz_roundtrips_nested_constraint_features(tmp_path) -> None:
    path = tmp_path / "features.npz"
    features = {
        "restype": np.eye(2, dtype=np.float32),
        "pad_info": {"mask_trunked": np.array([[True, False]])},
        "constraint_feature": {
            "contact": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            "contact_atom": np.zeros((2, 2, 2), dtype=np.float32),
            "pocket": np.ones((2, 2, 1), dtype=np.float32),
        },
    }

    save_static_feature_npz(path, features)
    loaded = load_static_feature_npz(path)

    assert loaded.keys() == features.keys()
    for group in ("pad_info", "constraint_feature"):
        assert loaded[group].keys() == features[group].keys()
        for key in features[group]:
            np.testing.assert_array_equal(loaded[group][key], features[group][key])


def test_a_list_of_arrays_is_sized_not_skipped() -> None:
    """A list has no ``nbytes``, and asarray on one stacks it on the device.

    This is the case that actually broke: an output held per-sample arrays in a
    list, the attribute check let it through, and ``np.asarray`` triggered a 37 GiB
    device-side stack after the model had already finished.
    """
    from foldjax.models.protenix.data.static_io import _output_nbytes

    arrays = [np.zeros((128, 128), dtype=np.float32) for _ in range(4)]
    assert _output_nbytes(arrays) == sum(a.nbytes for a in arrays)
    assert _output_nbytes(tuple(arrays)) == sum(a.nbytes for a in arrays)
    assert _output_nbytes(np.zeros(4, dtype=np.float32)) == 16
    # Unsizeable values must report None rather than a wrong number.
    assert _output_nbytes("not an array") is None
    assert _output_nbytes([np.zeros(2), "mixed"]) is None


def test_an_oversized_list_output_is_omitted(tmp_path) -> None:
    output = {
        "coordinate": np.ones((1, 2, 3), dtype=np.float32),
        "per_sample": [np.zeros((256, 256), dtype=np.float32) for _ in range(8)],
    }
    flat, omitted = flatten_output_dict(output, max_bytes=1 << 20)
    assert set(flat) == {"coordinate"}
    assert [name for name, _ in omitted] == ["per_sample"]


def test_a_transfer_that_runs_out_of_memory_is_recorded_not_raised() -> None:
    """A size check cannot predict every failed transfer, so one must be survivable.

    Moving a JAX array to host can execute a deferred computation and needs a
    device-side staging buffer, so it can exhaust memory even when the array itself
    is well under the budget -- observed at 2030 tokens on an array whose own size
    passed. Losing that array is acceptable; losing a ten-minute prediction because
    of the file it was being written to is not.
    """

    class Exploding:
        nbytes = 1024
        shape = (4, 4)

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying ...")

    flat, omitted = flatten_output_dict(
        {"good": np.ones((2, 2), dtype=np.float32), "bad": Exploding()}
    )
    assert set(flat) == {"good"}
    assert [name for name, _ in omitted] == ["bad"]


def test_an_unrelated_error_during_transfer_still_propagates() -> None:
    """Only memory failures are absorbed; a real bug must not be swallowed."""

    class Broken:
        nbytes = 8
        shape = (2,)

        def __array__(self, dtype=None, copy=None):
            raise ValueError("something is actually wrong")

    with pytest.raises(ValueError, match="actually wrong"):
        flatten_output_dict({"bad": Broken()})


def test_the_reporter_sees_every_output() -> None:
    """The report is how the offending array gets identified next time."""
    lines: list[str] = []
    flatten_output_dict(
        {
            "coordinate": np.ones((1, 2, 3), dtype=np.float32),
            "nested": {"value": np.zeros(4, dtype=np.float32)},
        },
        report=lines.append,
    )
    joined = "\n".join(lines)
    assert "coordinate" in joined
    assert "nested.value" in joined
    assert "GiB" in joined
