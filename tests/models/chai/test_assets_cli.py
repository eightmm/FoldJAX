"""Installed native-asset export command contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from foldjax.models.chai.bridge.bundle_io import COMPONENT_TENSOR_COUNTS
from foldjax.models.chai.bridge.conformer_io import load_native_conformers
from foldjax.models.chai.cli.main import main


def test_assets_help_discovers_all_exporters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["assets", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "bundle" in output
    assert "conformers" in output
    assert "esm2" in output


def test_assets_bundle_maps_strict_official_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = tmp_path / "components"
    components.mkdir()
    for name in COMPONENT_TENSOR_COUNTS:
        (components / f"{name}.pt").write_bytes(name.encode())
    captured: dict[str, Any] = {}

    def fake_export(sources: dict[str, Path], destination: Path, **metadata: str):
        captured.update(sources=sources, destination=destination, **metadata)
        destination.mkdir()
        (destination / "manifest.json").write_text("{}", encoding="utf-8")
        return {"components": {name: {} for name in sources}}

    monkeypatch.setattr(
        "foldjax.models.chai.cli.assets.export_native_bundle", fake_export
    )
    output = tmp_path / "native"
    assert (
        main(
            [
                "assets",
                "bundle",
                "--components-directory",
                str(components),
                "--output",
                str(output),
                "--chai-source",
                "chai-lab/chai-1",
                "--chai-release",
                "v0.6.1",
            ]
        )
        == 0
    )

    assert captured["sources"] == {
        name: components / f"{name}.pt" for name in COMPONENT_TENSOR_COUNTS
    }
    assert captured["destination"] == output
    assert captured["chai_source"] == "chai-lab/chai-1"
    assert captured["chai_release"] == "v0.6.1"


def test_assets_bundle_fails_before_export_when_component_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    components = tmp_path / "components"
    components.mkdir()

    assert (
        main(
            [
                "assets",
                "bundle",
                "--components-directory",
                str(components),
                "--output",
                str(tmp_path / "native"),
                "--chai-source",
                "chai-lab/chai-1",
                "--chai-release",
                "v0.6.1",
            ]
        )
        == 1
    )
    assert "missing official component" in capsys.readouterr().err


def test_assets_conformers_exports_from_explicit_chai_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chai_source = tmp_path / "chai-lab"
    (chai_source / "chai_lab/data/sources").mkdir(parents=True)
    source = tmp_path / "conformers.apkl"
    source.write_bytes(b"tiny-official-conformers")

    class FakeTensor:
        def __init__(self, value: np.ndarray):
            self.value = value

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    record = SimpleNamespace(
        position=FakeTensor(np.asarray([[0.0, 1.0, 2.0]], np.float32)),
        element=FakeTensor(np.asarray([6], np.int32)),
        charge=FakeTensor(np.asarray([0], np.int32)),
        atom_names=("C",),
        bonds=(),
        symmetries=FakeTensor(np.asarray([[0]], np.int64)),
    )
    antipickle = ModuleType("antipickle")
    antipickle.load = lambda path, adapters: {"LIG": record}  # type: ignore[attr-defined]
    rdkit = ModuleType("chai_lab.data.sources.rdkit")
    rdkit._get_adapters = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "antipickle", antipickle)
    monkeypatch.setitem(sys.modules, "chai_lab.data.sources.rdkit", rdkit)

    output = tmp_path / "conformers.npz"
    assert (
        main(
            [
                "assets",
                "conformers",
                "--chai-source-directory",
                str(chai_source),
                "--source",
                str(source),
                "--output",
                str(output),
                "--asset-version",
                "conformers_v1",
            ]
        )
        == 0
    )

    loaded = load_native_conformers(
        output, expected_asset_version="conformers_v1"
    )
    assert tuple(loaded) == ("LIG",)
    np.testing.assert_array_equal(loaded["LIG"].element, np.asarray([6], np.int32))


def test_assets_esm2_maps_official_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    source.write_bytes(b"mocked-in-test")
    output = tmp_path / "esm2"
    captured: dict[str, Path] = {}

    def fake_export(source_path: Path, output_path: Path) -> dict[str, Any]:
        captured.update(source=source_path, output=output_path)
        output_path.mkdir()
        return {"model": {"layers": 36}, "source_sha256": "a" * 64}

    monkeypatch.setattr(
        "foldjax.models.chai.cli.assets.export_native_esm2", fake_export
    )
    assert (
        main(
            [
                "assets",
                "esm2",
                "--source",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured == {"source": source, "output": output}
