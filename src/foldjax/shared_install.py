"""Inspect or explicitly repair permissions inside one managed shared store.

Traversal and changes stay relative to pinned, no-follow descriptors. Symlinks,
mount boundaries, and files with another hard link are never chmod/chgrp targets.
Only the named managed trees are visited; arbitrary files in the home are left
alone. This module imports no numerical or optional model runtime.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from foldjax.paths import foldjax_home

# Shared search roots permit publication; published content keeps its owner.
_ROOTS = (
    ("weights", "model weights", False),
    ("runtime", "generated model runtime", False),
    ("assets", "chemistry assets", False),
    ("downloads", "weight downloads and conversion", False),
    ("compile", "persistent compilation and Tokamax autotuning", False),
    ("msa", "MSA auto search", True),
    ("templates/hits", "template hit search", True),
    ("templates/mmcif", "template structure downloads", True),
)
_DETAIL_LIMIT = 500


def _group_id(group: str | None, *, apply: bool) -> int | None:
    if group is None:
        return None
    import grp

    try:
        gid = grp.getgrnam(group).gr_gid
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown shared-install group: {group!r}") from error
    if apply and os.geteuid() != 0 and gid not in {*os.getgroups(), os.getegid()}:
        raise PermissionError(f"current user is not a member of group {group!r}")
    return gid


def _directory_flags() -> int:
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise RuntimeError(
            "shared installation requires no-follow directory descriptors"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _mode(info: os.stat_result) -> str:
    return f"{stat.S_IMODE(info.st_mode):04o}"


def _access(parent_fd: int, name: str, mode: int) -> bool:
    return os.access(
        name, mode, dir_fd=parent_fd, effective_ids=True, follow_symlinks=False
    )


class _Audit:
    def __init__(self, home: Path, apply: bool, group: str | None, gid: int | None):
        self.home = home
        self.apply = apply
        self.gid = gid
        self.device: int | None = None
        self.failures: set[str] = set()
        self.identities: dict[str, tuple[int, int]] = {}
        self.result: dict[str, Any] = {
            "home": str(home),
            "apply": apply,
            "group": group,
            "shared": None,
            "owner_uid": None,
            "current_uid": os.geteuid(),
            "ok": False,
            "paths": [],
            "issues": [],
            "changes": [],
            "skipped": [],
            "counts": {"issues": 0, "changes": 0, "skipped": 0},
            "advice": [],
        }
        if group is not None:
            self.result["advice"].append(
                "Set FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1 for every account that "
                "uses this trusted group-writable compilation cache."
            )

    def detail(self, kind: str, **values: Any) -> None:
        self.result["counts"][kind] += 1
        if len(self.result[kind]) < _DETAIL_LIMIT:
            self.result[kind].append(values)

    def issue(self, path: str, code: str, feature: str, message: str) -> None:
        self.failures.add(feature)
        self.detail("issues", path=path, code=code, feature=feature, message=message)

    def skip(self, path: str, feature: str, reason: str) -> None:
        self.detail("skipped", path=path, reason=reason)
        self.issue(path, "unsafe_or_inaccessible", feature, reason)

    def directory_mode(self, *, publication_root: bool = False) -> int:
        if self.gid is not None:
            return 0o2775
        return 0o1777 if publication_root else 0o755

    def row(
        self, relative: str, feature: str, writable: bool = False
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "path": str(self.home / relative),
            "relative_path": relative,
            "feature": feature,
            "exists": False,
            "owner_uid": None,
            "mode": None,
            "expected_mode": f"{self.directory_mode(publication_root=writable):04o}",
            "readable": False,
            "writable": False,
            "searchable": False,
            "files": 0,
            "directories": 0,
            "private_files": 0,
            "temporary_directories": 0,
        }
        self.result["paths"].append(row)
        return row

    def normalize(
        self,
        fd: int,
        parent_fd: int,
        name: str,
        relative: str,
        expected: int,
        row: dict[str, Any],
        *,
        regular: bool,
    ) -> os.stat_result:
        info = os.fstat(fd)
        path = str(self.home / relative)
        needs_change = stat.S_IMODE(info.st_mode) != expected or (
            self.gid is not None and info.st_gid != self.gid
        )
        if needs_change and regular and info.st_nlink != 1:
            self.skip(path, row["feature"], "regular file has multiple hard links")
        elif needs_change and self.apply:
            before = info
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if _identity(current) != _identity(info):
                    raise OSError("entry changed while inspecting it")
                if os.geteuid() not in (0, info.st_uid):
                    raise PermissionError(
                        "only the owner or administrator can repair it"
                    )
                if regular and os.fstat(fd).st_nlink != 1:
                    raise OSError("regular file acquired another hard link")
                if self.gid is not None and info.st_gid != self.gid:
                    os.fchown(fd, -1, self.gid)
                os.fchmod(fd, expected)
            except OSError as error:
                self.skip(path, row["feature"], str(error))
            info = os.fstat(fd)
            if _mode(before) != _mode(info) or before.st_gid != info.st_gid:
                self.detail(
                    "changes",
                    path=path,
                    old_mode=_mode(before),
                    new_mode=_mode(info),
                    old_gid=before.st_gid,
                    new_gid=info.st_gid,
                )
        if stat.S_IMODE(info.st_mode) != expected:
            self.issue(
                path,
                "mode",
                row["feature"],
                f"mode {_mode(info)}; expected {expected:04o}",
            )
        if self.gid is not None and info.st_gid != self.gid:
            self.issue(
                path,
                "group",
                row["feature"],
                f"group {info.st_gid}; expected {self.gid}",
            )
        return info

    def open_root(self, parent_fd: int, name: str, relative: str) -> int:
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        if _identity(os.fstat(fd)) != self.identities.get(relative):
            os.close(fd)
            raise OSError("managed root changed while opening it")
        return fd

    def visit(
        self,
        parent_fd: int,
        name: str,
        relative: str,
        row: dict[str, Any],
        *,
        recursive: bool = True,
        publication_root: bool = False,
        depth: int = 0,
    ) -> None:
        path = str(self.home / relative)
        private = False
        try:
            try:
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not self.apply or depth != 0:
                    self.issue(
                        path, "missing", row["feature"], "managed path does not exist"
                    )
                    return
                os.mkdir(
                    name,
                    self.directory_mode(publication_root=publication_root),
                    dir_fd=parent_fd,
                )
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                self.detail("changes", path=path, created=True)
            if (
                relative.rpartition("/")[0] == "runtime/cache-warm"
                and name.startswith(".")
                and stat.S_ISDIR(info.st_mode)
                and info.st_dev == self.device
            ):
                # Active warm outputs are private and short-lived. Neither
                # inspection nor repair may traverse or publish their contents.
                row["temporary_directories"] += 1
                return
            if depth == 0:
                self.identities[relative] = _identity(info)
                row.update(
                    exists=True,
                    owner_uid=info.st_uid,
                    mode=_mode(info),
                    readable=_access(parent_fd, name, os.R_OK),
                    writable=_access(parent_fd, name, os.W_OK),
                    searchable=_access(parent_fd, name, os.X_OK),
                )
            if stat.S_ISLNK(info.st_mode):
                self.skip(path, row["feature"], "symlink is not followed")
                return
            if self.device is not None and info.st_dev != self.device:
                self.skip(path, row["feature"], "mounted filesystem is not traversed")
                return
            directory = stat.S_ISDIR(info.st_mode)
            if not directory and not stat.S_ISREG(info.st_mode):
                self.skip(path, row["feature"], "special file is not opened")
                return
            if depth == 0 and not directory:
                self.skip(path, row["feature"], "managed root is not a directory")
                return
            if directory:
                row["directories"] += 1
            else:
                row["files"] += 1
                private = info.st_mode & 0o044 == 0
                row["private_files"] += int(private)
            flags = (
                _directory_flags()
                if directory
                else (
                    os.O_RDONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
            )
            fd = os.open(name, flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(fd)
                if _identity(opened) != _identity(info):
                    self.skip(path, row["feature"], "entry changed while opening it")
                    return
                expected = (
                    self.directory_mode(publication_root=publication_root)
                    if directory
                    else (0o664 if self.gid is not None else 0o644)
                )
                if (
                    not directory
                    and relative.split("/", maxsplit=1)[0] == "runtime"
                    and info.st_mode & 0o111
                ):
                    expected |= 0o111
                info = self.normalize(
                    fd, parent_fd, name, relative, expected, row, regular=not directory
                )
                readable = _access(parent_fd, name, os.R_OK)
                searchable = not directory or _access(parent_fd, name, os.X_OK)
                if not readable or not searchable:
                    self.issue(
                        path,
                        "access",
                        row["feature"],
                        "current user cannot read/traverse this entry",
                    )
                if depth == 0:
                    row.update(
                        mode=_mode(info),
                        readable=readable,
                        writable=_access(parent_fd, name, os.W_OK),
                        searchable=searchable,
                    )
                    if publication_root and not row["writable"]:
                        self.issue(
                            path,
                            "write_access",
                            row["feature"],
                            "current user cannot publish cache entries",
                        )
                if directory and recursive:
                    if depth >= 128:
                        self.skip(
                            path, row["feature"], "directory nesting exceeds 128 levels"
                        )
                        return
                    for child in sorted(os.listdir(fd)):
                        self.visit(
                            fd, child, f"{relative}/{child}", row, depth=depth + 1
                        )
                elif not directory:
                    row["private_files"] += int(info.st_mode & 0o044 == 0) - int(
                        private
                    )
                    private = info.st_mode & 0o044 == 0
            finally:
                os.close(fd)
        except OSError as error:
            self.skip(path, row["feature"], str(error))
        finally:
            if private:
                self.issue(
                    path,
                    "private_file",
                    row["feature"],
                    "owner-only file prevents other accounts from reading it",
                )

    def finish(self) -> dict[str, Any]:
        self.result["failing_features"] = sorted(self.failures)
        self.result["ok"] = self.result["counts"]["issues"] == 0
        self.result["details_truncated"] = any(
            self.result["counts"][kind] > len(self.result[kind])
            for kind in ("issues", "changes", "skipped")
        )
        return self.result


def audit_shared_install(
    home: Path | None = None,
    *,
    apply: bool = False,
    group: str | None = None,
) -> dict[str, Any]:
    """Inspect managed permissions; mutate them only with explicit ``apply``.

    Group membership is checked before any mutation. Default search roots are
    sticky 1777; their published descendants remain 755/644 and retain their
    owners. Group mode uses 2775/664 throughout, preserving runtime executables.
    Symlinked ancestors or managed entries and multiply linked files needing a
    change are reported and left alone. Missing ancestors outside the selected
    home are never created.
    """
    gid = _group_id(group, apply=apply)
    home = Path(
        os.path.abspath((home if home is not None else foldjax_home()).expanduser())
    )
    if home == Path(home.anchor):
        raise ValueError("filesystem root cannot be used as a shared installation home")
    audit = _Audit(home, apply, group, gid)
    flags = _directory_flags()
    parent_fd = os.open(home.anchor, flags)
    try:
        for component in home.parts[1:-1]:
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        row = audit.row(".", "shared home access")
        audit.visit(parent_fd, home.name or ".", ".", row, recursive=False)
        audit.result.update(
            owner_uid=row["owner_uid"],
            shared=None
            if row["owner_uid"] is None
            else row["owner_uid"] != os.geteuid(),
        )
        if not row["exists"]:
            return audit.finish()
        home_fd = audit.open_root(parent_fd, home.name, ".")
        try:
            info = os.fstat(home_fd)
            audit.device = info.st_dev
            audit.result.update(
                owner_uid=info.st_uid, shared=info.st_uid != os.geteuid()
            )
            for relative, feature, writable in _ROOTS:
                if relative.startswith("templates/"):
                    continue
                audit.visit(
                    home_fd,
                    relative,
                    relative,
                    audit.row(relative, feature, writable),
                    publication_root=writable,
                )
            template_row = audit.row("templates", "template cache access")
            audit.visit(
                home_fd, "templates", "templates", template_row, recursive=False
            )
            if template_row["exists"]:
                template_fd = audit.open_root(home_fd, "templates", "templates")
                try:
                    for relative, feature, writable in _ROOTS:
                        if relative.startswith("templates/"):
                            audit.visit(
                                template_fd,
                                relative.split("/")[1],
                                relative,
                                audit.row(relative, feature, writable),
                                publication_root=writable,
                            )
                finally:
                    os.close(template_fd)
        finally:
            os.close(home_fd)
    except OSError as error:
        audit.skip(str(home), "shared home access", str(error))
    finally:
        os.close(parent_fd)
    return audit.finish()
