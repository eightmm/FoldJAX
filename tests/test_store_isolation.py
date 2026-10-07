"""The suite runs against an empty store unless a test asks for the real one."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from foldjax import ccd, paths


def test_an_ordinary_test_is_pointed_away_from_the_real_store(
    _isolated_foldjax_home: Path,
) -> None:
    assert paths.foldjax_home() == _isolated_foldjax_home
    assert ccd.components_file() is None
    for name in (
        "PROTENIX_CCD_COMPONENTS_FILE",
        "PROTENIX_CCD_RDKIT_MOL_FILE",
        "PROTENIX_TEMPLATE_MMCIF_DIR",
    ):
        assert name not in os.environ


@pytest.mark.real_store
def test_the_marker_keeps_the_store_the_session_found(
    _isolated_foldjax_home: Path,
) -> None:
    assert os.environ.get("FOLDJAX_HOME") != str(_isolated_foldjax_home)
    assert paths.foldjax_home() != _isolated_foldjax_home


def test_the_fixture_keeps_the_store_the_session_found(
    real_store: None, _isolated_foldjax_home: Path
) -> None:
    assert paths.foldjax_home() != _isolated_foldjax_home



def test_a_test_undoing_its_monkeypatch_stays_isolated(
    monkeypatch: pytest.MonkeyPatch, _isolated_foldjax_home: Path
) -> None:
    """A test calling `monkeypatch.undo()` once restored the real store."""
    monkeypatch.setenv("FOLDJAX_PROGRESS", "0")
    monkeypatch.undo()
    assert paths.foldjax_home() == _isolated_foldjax_home
