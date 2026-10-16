"""Shared-store permissions, explicit repair, and descriptor trust boundaries."""

from __future__ import annotations

import argparse
import errno
import grp
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from foldjax import doctor, shared_install
from foldjax.shared_install import audit_shared_install


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "store"
    report = audit_shared_install(home, apply=True)
    assert report["ok"], report
    return home


def _rows(report):
    return {row["relative_path"]: row for row in report["paths"]}


def test_inspection_does_not_create_an_absent_home(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "absent"
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setattr(os, "mkdir", lambda *a, **kw: pytest.fail("inspection mkdir"))
    monkeypatch.setattr(os, "fchmod", lambda *a: pytest.fail("inspection chmod"))
    report = audit_shared_install()
    assert report["home"] == str(home)
    assert report["counts"]["changes"] == 0
    assert not report["ok"]
    assert not home.exists()
    json.dumps(report)


def test_apply_creates_policy_roots_despite_a_private_umask(tmp_path: Path) -> None:
    home = tmp_path / "store"
    previous = os.umask(0o077)
    try:
        report = audit_shared_install(home, apply=True)
    finally:
        os.umask(previous)
    assert report["ok"], report
    for name in ("weights", "runtime", "assets", "downloads", "compile", "templates"):
        assert _mode(home / name) == 0o755
    for name in ("msa", "templates/hits", "templates/mmcif"):
        assert _mode(home / name) == 0o1777
    assert audit_shared_install(home)["ok"]
    assert audit_shared_install(home, apply=True)["counts"]["changes"] == 0


def test_private_weights_unwritable_msa_and_private_tokamax_are_diagnosed(
    tmp_path: Path, monkeypatch
) -> None:
    home = _home(tmp_path)
    weight = home / "weights" / "boltz2.safetensors"
    weight.write_bytes(b"weight")
    weight.chmod(0o600)
    (home / "msa").chmod(0o755)
    tuning = home / "compile" / "af3" / ".tokamax-autotuning-v2"
    tuning.mkdir(parents=True)
    tuning.chmod(0o700)
    result = tuning / "measurement.json"
    result.write_text("{}")
    result.chmod(0o600)
    modes = {p: _mode(p) for p in (weight, home / "msa", tuning, result)}
    owner = home.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)

    def recipient_access(parent_fd, name, mode):
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return info.st_mode & mode == mode

    monkeypatch.setattr(shared_install, "_access", recipient_access)
    report = audit_shared_install(home)
    rows = _rows(report)
    assert report["shared"] is True
    assert report["owner_uid"] == owner
    assert rows["weights"]["private_files"] == 1
    assert rows["compile"]["private_files"] == 1
    assert rows["msa"]["writable"] is False
    assert {
        "model weights",
        "MSA auto search",
        "persistent compilation and Tokamax autotuning",
    } <= set(report["failing_features"])
    assert {p: _mode(p) for p in modes} == modes
    assert report["counts"]["changes"] == 0


def test_apply_repairs_old_tokamax_and_preserves_runtime_executables(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path)
    tuning = home / "compile" / "af3" / ".tokamax-autotuning-v2" / "device"
    tuning.mkdir(parents=True)
    tuning.parent.chmod(0o700)
    tuning.chmod(0o700)
    result = tuning / "result.json"
    result.write_text("{}")
    result.chmod(0o600)
    executable = home / "runtime" / "tool"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o700)
    report = audit_shared_install(home, apply=True)
    assert report["ok"], report
    assert _mode(tuning.parent) == _mode(tuning) == 0o755
    assert _mode(result) == 0o644
    assert _mode(executable) == 0o755
    assert _rows(report)["compile"]["private_files"] == 0


def test_sticky_roots_do_not_make_existing_content_world_writable(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path)
    entry = home / "msa" / "digest"
    entry.mkdir(mode=0o700)
    result = entry / "result.a3m"
    result.write_bytes(b">query\nAA\n")
    result.chmod(0o600)
    owner = result.stat().st_uid
    report = audit_shared_install(home, apply=True)
    assert report["ok"], report
    assert _mode(home / "msa") == 0o1777
    assert _mode(entry) == 0o755
    assert _mode(result) == 0o644
    assert result.stat().st_uid == owner


def test_group_policy_is_setgid_group_writable_and_advises_compile_trust(
    tmp_path: Path,
) -> None:
    home = _home(tmp_path)
    weight = home / "weights" / "checkpoint"
    weight.write_bytes(b"weights")
    group = grp.getgrgid(os.getegid()).gr_name
    report = audit_shared_install(home, apply=True, group=group)
    assert report["ok"], report
    for row in report["paths"]:
        assert _mode(Path(row["path"])) == 0o2775
        assert Path(row["path"]).stat().st_gid == os.getegid()
    assert _mode(weight) == 0o664
    assert any("FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1" in x for x in report["advice"])
    assert audit_shared_install(home, group=group)["ok"]


@pytest.mark.parametrize("apply", [False, True])
def test_invalid_group_is_rejected_before_any_filesystem_work(
    tmp_path: Path, monkeypatch, apply: bool
) -> None:
    home = tmp_path / "absent"

    def absent(name):
        raise KeyError(name)

    monkeypatch.setattr(grp, "getgrnam", absent)
    with pytest.raises(ValueError, match="unknown shared-install group"):
        audit_shared_install(home, apply=apply, group="missing-group")
    assert not home.exists()


def test_nonmember_group_is_rejected_before_creation(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "absent"
    monkeypatch.setattr(grp, "getgrnam", lambda _: SimpleNamespace(gr_gid=987654))
    monkeypatch.setattr(os, "geteuid", lambda: 12345)
    monkeypatch.setattr(os, "getegid", lambda: 12345)
    monkeypatch.setattr(os, "getgroups", lambda: [12345])
    with pytest.raises(PermissionError, match="not a member"):
        audit_shared_install(home, apply=True, group="other-group")
    assert not home.exists()


def test_symlinks_and_hardlinks_cannot_change_an_outside_file(tmp_path: Path) -> None:
    home = _home(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    secret = outside / "secret"
    secret.write_bytes(b"private")
    secret.chmod(0o600)
    (home / "weights" / "linked-directory").symlink_to(outside)
    (home / "assets" / "linked-file").symlink_to(secret)
    os.link(secret, home / "weights" / "hardlink")
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert _mode(outside) == 0o700
    assert _mode(secret) == 0o600
    assert secret.read_bytes() == b"private"
    assert any("hard links" in row["reason"] for row in report["skipped"])
    assert sum("symlink" in row["reason"] for row in report["skipped"]) == 2


@pytest.mark.parametrize("ancestor", [False, True])
def test_symlinked_home_or_ancestor_is_never_followed(
    tmp_path: Path, ancestor: bool
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(outside)
    home = link / "store" if ancestor else link
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert report["counts"]["changes"] == 0
    assert _mode(outside) == 0o700
    assert not (outside / "weights").exists()
    assert not (outside / "store").exists()


def test_readonly_filesystem_failure_is_reported_and_old_data_remains(
    tmp_path: Path, monkeypatch
) -> None:
    home = _home(tmp_path)
    result = home / "compile" / "old.json"
    result.write_bytes(b"data")
    result.chmod(0o600)

    def readonly(*_):
        raise OSError(errno.EROFS, "read-only filesystem")

    monkeypatch.setattr(os, "fchmod", readonly)
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert report["counts"]["changes"] == 0
    assert any("read-only filesystem" in row["reason"] for row in report["skipped"])
    assert _mode(result) == 0o600
    assert result.read_bytes() == b"data"


@pytest.mark.parametrize("after_open", [False, True])
def test_an_unsafe_replacement_between_stat_and_open_is_not_modified(
    tmp_path: Path, monkeypatch, after_open: bool
) -> None:
    home = _home(tmp_path)
    weight = home / "weights" / "checkpoint"
    weight.write_bytes(b"weight")
    weight.chmod(0o600)
    outside = tmp_path / "private"
    outside.write_bytes(b"private")
    outside.chmod(0o600)
    original = os.open

    def swap(name, flags, *args, **kwargs):
        fd = original(name, flags, *args, **kwargs) if after_open else None
        if name == "checkpoint":
            weight.unlink()
            weight.symlink_to(outside)
        return fd if fd is not None else original(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert _mode(outside) == 0o600
    assert outside.read_bytes() == b"private"


def test_unmanaged_home_files_are_left_alone(tmp_path: Path) -> None:
    home = _home(tmp_path)
    private = home / "private-not-managed"
    private.write_bytes(b"private")
    private.chmod(0o600)
    assert audit_shared_install(home, apply=True)["ok"]
    assert _mode(private) == 0o600


def test_missing_ancestor_is_not_created_outside_the_selected_home(
    tmp_path: Path,
) -> None:
    home = tmp_path / "missing-parent" / "store"
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert report["counts"]["changes"] == 0
    assert not home.parent.exists()


def test_filesystem_root_is_refused() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        audit_shared_install(Path("/"), apply=True)


def test_doctor_reports_shared_permissions_without_repairing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = _home(tmp_path)
    weight = home / "weights" / "checkpoint"
    weight.write_bytes(b"weight")
    weight.chmod(0o600)
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setattr(doctor, "available_models", lambda: ())
    monkeypatch.setattr(os, "geteuid", lambda: home.stat().st_uid + 1)
    assert doctor.run_doctor(argparse.Namespace(json=True)) == 0
    report = json.loads(capsys.readouterr().out)["shared_install"]
    assert report["shared"] is True
    assert _rows(report)["weights"]["private_files"] == 1
    assert "model weights" in report["failing_features"]
    assert _mode(weight) == 0o600
    assert report["counts"]["changes"] == 0


def test_doctor_infers_an_existing_group_installation(tmp_path: Path) -> None:
    home = _home(tmp_path)
    group = grp.getgrgid(os.getegid()).gr_name
    assert audit_shared_install(home, apply=True, group=group)["ok"]
    report = doctor._shared_install_report(home)
    assert report["group"] == group
    assert report["ok"], report


def test_apply_reports_foreign_owned_entries_it_cannot_repair(
    tmp_path: Path, monkeypatch
) -> None:
    home = _home(tmp_path)
    weight = home / "weights" / "private"
    weight.write_bytes(b"weight")
    weight.chmod(0o600)
    monkeypatch.setattr(os, "geteuid", lambda: home.stat().st_uid + 1)
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert _mode(weight) == 0o600
    assert any("owner or administrator" in x["reason"] for x in report["skipped"])


def test_special_files_are_reported_without_opening_them(tmp_path: Path) -> None:
    home = _home(tmp_path)
    fifo = home / "assets" / "fifo"
    os.mkfifo(fifo, 0o600)
    report = audit_shared_install(home, apply=True)
    assert not report["ok"]
    assert any(x["reason"] == "special file is not opened" for x in report["skipped"])
    assert stat.S_IMODE(fifo.lstat().st_mode) == 0o600
