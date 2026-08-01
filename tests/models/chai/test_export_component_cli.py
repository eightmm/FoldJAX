from __future__ import annotations

from pathlib import Path

from foldjax.models.chai.cli import export_component


def test_export_component_cli_reports_tensor_count(
    monkeypatch, capsys, tmp_path
) -> None:
    source = tmp_path / "trunk.pt"
    destination = tmp_path / "trunk.npz"
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        export_component,
        "convert_component_to_native",
        lambda src, dst: calls.append((Path(src), Path(dst))) or 1398,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "chai-jax-export-component",
            "--component",
            str(source),
            "--output",
            str(destination),
        ],
    )

    export_component.main()

    assert calls == [(source, destination)]
    assert "1398 tensors" in capsys.readouterr().out
