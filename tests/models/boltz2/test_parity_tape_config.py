"""Numerical controls must select multiplication separately from attention."""

import sys

import numpy as np
import pytest

from tests.models.boltz2.scripts.parity_matched_tape import (
    captured_sampler_trunk,
    parse_args,
)


def test_upstream_trunk_requires_captured_relative_position():
    with pytest.raises(ValueError, match="relative_position_encoding"):
        captured_sampler_trunk(dict.fromkeys(("s", "z", "s_inputs"), np.zeros(1)))


def test_upstream_trunk_preserves_all_captured_values():
    native = {
        name: np.full((1, 2, 3), index + 0.125, dtype=np.float32)
        for index, name in enumerate(
            ("s", "z", "s_inputs", "relative_position_encoding")
        )
    }
    actual = captured_sampler_trunk(native)
    assert actual.keys() == native.keys()
    for name in native:
        np.testing.assert_array_equal(actual[name], native[name])


@pytest.mark.parametrize("attention", ["xla", "cueq"])
def test_control_defaults_to_xla_multiplication(monkeypatch, attention):
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    monkeypatch.setattr(
        sys, "argv", ["parity", "--tape-dir", "unused", "--triangle-backend", attention]
    )
    args = parse_args()
    assert args.triangle_backend == attention
    assert args.triangle_multiplication_backend == "xla"


def test_control_accepts_explicit_fused_multiplication(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["parity", "--tape-dir", "unused", "--triangle-multiplication-backend", "cueq"],
    )
    assert parse_args().triangle_multiplication_backend == "cueq"
