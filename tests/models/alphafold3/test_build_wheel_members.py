"""The AlphaFold 3 build wheel's data members stay inside the runtime root."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from foldjax.models.alphafold3.build import _extract_build_wheel


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def test_libcifpp_data_is_extracted_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    package = root / "alphafold3"
    wheel = _wheel(tmp_path / "w.whl", {"share/libcifpp/components.cif": b"data_"})
    _extract_build_wheel(wheel, package, root)
    assert (root / "share" / "libcifpp" / "components.cif").read_bytes() == b"data_"


def test_a_member_that_climbs_out_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    package = root / "alphafold3"
    wheel = _wheel(
        tmp_path / "w.whl", {"share/libcifpp/../../../escaped.txt": b"payload"}
    )
    with pytest.raises(RuntimeError, match="escapes"):
        _extract_build_wheel(wheel, package, root)
    assert not list(tmp_path.rglob("escaped.txt"))
