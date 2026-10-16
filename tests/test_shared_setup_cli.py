from __future__ import annotations

import json

import pytest

from foldjax import assets
from foldjax.cli import main


def test_shared_setup_is_inspection_only_by_default(tmp_path, monkeypatch, capsys):
    home = tmp_path / "absent"
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setattr(
        assets,
        "fetch",
        lambda *_args, **_kwargs: pytest.fail("permission audit downloaded weights"),
    )
    assert main(["setup", "--shared", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["apply"] is False
    assert not home.exists()
    assert report["counts"]["issues"]


def test_shared_setup_apply_repairs_old_tuning_files(tmp_path, monkeypatch, capsys):
    home = tmp_path / "store"
    old = (
        home
        / "compile"
        / "alphafold3"
        / "weights"
        / "profile"
        / ".tokamax-autotuning-v2"
    )
    old.mkdir(parents=True, mode=0o700)
    entry = old / "saved.json"
    entry.write_text("{}")
    entry.chmod(0o600)
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    assert main(["setup", "--shared", "--apply", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    assert old.stat().st_mode & 0o777 == 0o755
    assert entry.stat().st_mode & 0o777 == 0o644
    assert (home / "msa").stat().st_mode & 0o7777 == 0o1777


def test_shared_setup_text_explains_skipped_unsafe_paths(tmp_path, monkeypatch, capsys):
    home = tmp_path / "store"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "weights").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    assert main(["setup", "--shared"]) == 1
    text = capsys.readouterr().out
    assert "weights" in text and "symlink is not followed" in text
    assert "skipped:" in text
    assert not list(outside.iterdir())


@pytest.mark.parametrize("flags", [["--apply"], ["--group", "nobody"], ["--json"]])
def test_shared_only_flags_do_not_accidentally_download(flags, monkeypatch):
    monkeypatch.setattr(
        assets,
        "fetch",
        lambda *_args, **_kwargs: pytest.fail("invalid flags downloaded weights"),
    )
    with pytest.raises(ValueError, match="require setup --shared"):
        main(["setup", *flags])
