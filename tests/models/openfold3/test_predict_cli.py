"""The prediction CLI's decisions, without needing a GPU or the released weights.

This is the entry point users actually run, and it was the only one of the four with
no test. What is covered here is what it *decides* -- refusing incomplete features,
resolving the sampling knobs, reporting memory and warning when a target is close to
filling the device -- rather than the arithmetic, which the parity gate owns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.cli.predict import _parser, _report_peak_memory, main
from foldjax.models.openfold3.data import MODEL_FEATURES


class _Device:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


class _Jax:
    def __init__(self, device):
        self._device = device

    def devices(self):
        return [self._device]


class _Config:
    def __init__(self, chunk):
        self.pair_chunk_size = chunk


def test_defaults_match_the_released_settings() -> None:
    """The sampling knobs default to "whatever released_config says", not to numbers.

    Hardcoding 5/200/4 here would let the CLI drift from the config the checkpoint
    was produced under.
    """
    args = _parser().parse_args(["f.npz", "--checkpoint", "w.pt", "-o", "out"])
    assert args.samples is None
    assert args.steps is None
    assert args.cycles is None
    assert args.repeats == 1
    assert args.all_arrays is False


def test_incomplete_features_are_refused(tmp_path: Path, capsys) -> None:
    """A missing feature must fail before the checkpoint is loaded.

    Otherwise it surfaces as a KeyError inside a traced function, minutes into a
    compile, naming a feature rather than the file that lacks it.
    """
    path = tmp_path / "features.npz"
    np.savez(path, token_mask=np.ones((1, 4), dtype=np.float32))
    assert main([str(path), "--checkpoint", "missing.pt", "-o", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "features are incomplete" in out
    # The message has to name what is absent, and there is a lot absent here.
    assert "ref_pos" in out
    assert len(MODEL_FEATURES) > 30


def test_peak_memory_is_reported_with_the_chunk_that_bounded_it(capsys) -> None:
    jax = _Jax(_Device({"peak_bytes_in_use": 26 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(483))
    out = capsys.readouterr().out
    assert "26.00 GiB" in out
    assert "483" in out
    # Well inside the device, so no warning.
    assert "will not fit" not in out


def test_a_nearly_full_device_is_called_out(capsys) -> None:
    """Fitting at 97% is not the same as fitting, and the user should hear so."""
    jax = _Jax(_Device({"peak_bytes_in_use": 93 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(123))
    out = capsys.readouterr().out
    assert "97%" in out
    assert "will not fit" in out


def test_no_chunking_is_reported_as_such(capsys) -> None:
    jax = _Jax(_Device({"peak_bytes_in_use": 2 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(None))
    assert "no chunking" in capsys.readouterr().out


def test_a_device_without_memory_stats_is_silent(capsys) -> None:
    """CPU devices expose no allocator stats; printing zero would be a lie."""
    _report_peak_memory(_Jax(object()), _Config(None))
    assert capsys.readouterr().out == ""

    _report_peak_memory(_Jax(_Device(None)), _Config(None))
    assert capsys.readouterr().out == ""

    _report_peak_memory(_Jax(_Device({})), _Config(None))
    assert capsys.readouterr().out == ""


def test_missing_bytes_limit_still_reports_the_peak(capsys) -> None:
    """Some platforms omit the limit; the peak is the useful half anyway."""
    jax = _Jax(_Device({"peak_bytes_in_use": 5 * 2**30}))
    _report_peak_memory(jax, _Config(None))
    out = capsys.readouterr().out
    assert "5.00 GiB" in out
    assert "%" not in out


@pytest.mark.parametrize("flag", ["--samples", "--steps", "--cycles", "--repeats"])
def test_sampling_knobs_are_integers(flag: str) -> None:
    args = _parser().parse_args(
        ["f.npz", "--checkpoint", "w.pt", "-o", "out", flag, "7"]
    )
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == 7
