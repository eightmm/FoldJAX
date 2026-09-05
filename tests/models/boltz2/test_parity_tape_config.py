"""Numerical controls must select multiplication separately from attention."""

import sys

import pytest

from tests.models.boltz2.scripts.parity_matched_tape import parse_args


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
