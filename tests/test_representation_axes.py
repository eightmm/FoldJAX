"""What the archive stores must have the axes the manifest names.

Three of the five producers handed back a leading batch axis of one, so the
manifest said `shape: [1, 254, 254, 128]` beside `axes: ["token", "token",
"channel"]` -- four numbers against three names, in the one file a consumer
reads to learn what it is holding. The other two wrote three of each, so the
same feature had two contracts depending on the model behind it.
"""

import json

import numpy as np
import pytest

from foldjax.models import _representations


def _specs():
    return _representations.specs_for("boltz2")


def test_a_batch_of_one_is_dropped_to_match_the_named_axes(tmp_path) -> None:
    _representations.save(
        tmp_path,
        {"pair": np.zeros((1, 7, 7, 3), dtype=np.float32)},
        _specs(),
        model="boltz2",
    )
    stored = _representations.load(tmp_path)
    assert stored["pair"].shape == (7, 7, 3)


def test_the_manifest_names_one_axis_per_dimension(tmp_path) -> None:
    _representations.save(
        tmp_path,
        {
            "pair": np.zeros((1, 7, 7, 3), dtype=np.float32),
            "single": np.zeros((7, 5), dtype=np.float32),
        },
        _specs(),
        model="boltz2",
    )
    manifest = json.loads((tmp_path / _representations.MANIFEST_NAME).read_text())
    for name, entry in manifest["representations"].items():
        assert len(entry["shape"]) == len(entry["axes"]), name


def test_an_array_that_is_not_its_spec_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="axes"):
        _representations.save(
            tmp_path,
            {"pair": np.zeros((7, 3), dtype=np.float32)},
            _specs(),
            model="boltz2",
        )


def test_a_leading_axis_that_is_not_one_is_refused(tmp_path) -> None:
    """A batch of one carries nothing; a batch of five is data being lost."""
    with pytest.raises(ValueError, match="axes"):
        _representations.save(
            tmp_path,
            {"pair": np.zeros((5, 7, 7, 3), dtype=np.float32)},
            _specs(),
            model="boltz2",
        )
