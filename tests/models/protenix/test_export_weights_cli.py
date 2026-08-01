from pathlib import Path

import numpy as np

from foldjax.models.protenix.bridge import export_weights


def test_export_weights_cli_maps_checkpoint_and_writes_native(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "model.pkl"
    params = {"weight": np.asarray([1.0], dtype=np.float32)}
    seen = {}

    monkeypatch.setattr(export_weights, "load_torch_checkpoint", lambda path: params)

    def fake_save(path, value, *, compress):
        seen.update(path=Path(path), value=value, compress=compress)
        Path(path).touch()

    monkeypatch.setattr(export_weights, "save_native_weights", fake_save)

    export_weights.main(
        ["--checkpoint", str(checkpoint), "--out", str(output), "--no-compress"]
    )

    assert seen == {"path": output, "value": params, "compress": False}
    assert output.is_file()


def test_export_weights_cli_rejects_missing_checkpoint(tmp_path):
    try:
        export_weights.main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--out",
                str(tmp_path / "out.pkl"),
            ]
        )
    except SystemExit as exc:
        assert "missing checkpoint" in str(exc)
    else:
        raise AssertionError("missing checkpoint must fail")
