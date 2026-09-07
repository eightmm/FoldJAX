import pytest

from bench.precision_policy import coordinate_gate


def test_gate_includes_boundary_and_every_sample():
    assert coordinate_gate({"A": [0.05] * 5})["coordinate_gate_passed"]
    assert not coordinate_gate({"A": [0] * 4 + [0.050001]})["coordinate_gate_passed"]
    assert not coordinate_gate({"A": [0] * 5, "B": [0.06] * 5})[
        "coordinate_gate_passed"
    ]


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0] * 4,
        [0] * 6,
        [float("nan")] * 5,
        [float("inf")] * 5,
        [-1] * 5,
        [None] * 5,
        [True] * 5,
    ],
)
def test_gate_rejects_incomplete_or_invalid_measurements(values):
    with pytest.raises(ValueError):
        coordinate_gate({"A": values})


def test_gate_rejects_empty_report():
    with pytest.raises(ValueError):
        coordinate_gate({})


def test_gate_cannot_hide_an_entity_through_key_normalization():
    with pytest.raises(ValueError, match="collide"):
        coordinate_gate({1: [1] * 5, "1": [0] * 5})
