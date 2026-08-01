from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge.conformer_io import (
    ConformerData,
    load_native_conformers,
    save_native_conformers,
)


def _conformer(offset: float = 0.0) -> ConformerData:
    return ConformerData(
        position=np.asarray([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], np.float32),
        element=np.asarray([6, 8], np.int32),
        charge=np.asarray([0, -1], np.int32),
        atom_names=("C", "O"),
        bonds=((0, 1),),
        symmetries=np.asarray([[0], [1]], np.int64),
    )


def test_native_conformer_roundtrip_is_torch_free_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "conformers.npz"
    save_native_conformers(
        {"ALA": _conformer(), "LIG": _conformer(2.0)},
        path,
        asset_version="conformers_v1",
        source_sha256="a" * 64,
    )

    loaded = load_native_conformers(path, expected_asset_version="conformers_v1")

    assert list(loaded) == ["ALA", "LIG"]
    assert loaded["ALA"].atom_names == ("C", "O")
    assert loaded["ALA"].bonds == ((0, 1),)
    np.testing.assert_array_equal(loaded["LIG"].position, _conformer(2.0).position)


def test_native_conformer_rejects_invalid_shapes(tmp_path: Path) -> None:
    invalid = _conformer()._replace(element=np.asarray([6], np.int32))
    with pytest.raises(ValueError, match="atom count"):
        save_native_conformers(
            {"ALA": invalid},
            tmp_path / "bad.npz",
            asset_version="v1",
            source_sha256="b" * 64,
        )


def test_native_conformer_rejects_wrong_asset_version(tmp_path: Path) -> None:
    path = tmp_path / "conformers.npz"
    save_native_conformers(
        {"ALA": _conformer()},
        path,
        asset_version="v1",
        source_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="asset version"):
        load_native_conformers(path, expected_asset_version="v2")

def test_packed_archives_round_trip_and_keep_per_residue_dtypes(tmp_path) -> None:
    """The packed layout must not widen a residue's arrays.

    Version 1 stored four npz members per residue: 172,700 for the released
    asset, which took ~203 s to read because each member costs a seek, a header
    parse, and a zipfile handle. Version 2 concatenates them, and the trap that
    introduces is dtype promotion — 1,880 of the released residues carry int32
    symmetries and the rest int64, so concatenating silently widens them.
    """
    import numpy as np

    from foldjax.models.chai.bridge.conformer_io import (
        ConformerData,
        load_native_conformers,
        save_native_conformers,
    )

    narrow = ConformerData(
        position=np.zeros((3, 3), np.float32),
        element=np.asarray([6, 7, 8], np.int32),
        charge=np.zeros(3, np.int32),
        atom_names=("N", "CA", "C"),
        bonds=((0, 1), (1, 2)),
        # zero-width and int32, which is the case that got promoted
        symmetries=np.zeros((3, 0), np.int32),
    )
    wide = ConformerData(
        position=np.ones((2, 3), np.float32),
        element=np.asarray([1, 1], np.int32),
        charge=np.zeros(2, np.int32),
        atom_names=("H1", "H2"),
        bonds=(),
        symmetries=np.asarray([[0, 1], [1, 0]], np.int64),
    )
    path = tmp_path / "conformers.npz"
    save_native_conformers(
        {"AAA": narrow, "BBB": wide},
        path,
        asset_version="conformers_test",
        source_sha256="0" * 64,
    )
    loaded = load_native_conformers(path)

    for name, expected in (("AAA", narrow), ("BBB", wide)):
        actual = loaded[name]
        for field in ("position", "element", "charge", "symmetries"):
            want, got = getattr(expected, field), getattr(actual, field)
            assert got.dtype == want.dtype, f"{name}.{field} dtype changed"
            assert got.shape == want.shape, f"{name}.{field} shape changed"
            np.testing.assert_array_equal(got, want)
        assert actual.atom_names == expected.atom_names
        assert actual.bonds == expected.bonds


def test_ragged_symmetry_widths_survive_packing(tmp_path) -> None:
    """Symmetries are ragged in their second axis, so they pack flat."""
    import numpy as np

    from foldjax.models.chai.bridge.conformer_io import (
        ConformerData,
        load_native_conformers,
        save_native_conformers,
    )

    residues = {}
    for index, width in enumerate((0, 1, 5)):
        atoms = index + 2
        residues[f"R{index}"] = ConformerData(
            position=np.full((atoms, 3), index, np.float32),
            element=np.full(atoms, index, np.int32),
            charge=np.zeros(atoms, np.int32),
            atom_names=tuple(f"A{i}" for i in range(atoms)),
            bonds=(),
            symmetries=np.arange(atoms * width, dtype=np.int64).reshape(atoms, width),
        )
    path = tmp_path / "ragged.npz"
    save_native_conformers(
        residues, path, asset_version="v", source_sha256="a" * 64
    )
    loaded = load_native_conformers(path)
    for name, expected in residues.items():
        np.testing.assert_array_equal(loaded[name].symmetries, expected.symmetries)
        assert loaded[name].symmetries.shape == expected.symmetries.shape
