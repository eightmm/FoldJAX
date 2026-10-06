"""Filesystem helpers that several FoldJAX modules had written out identically.

Only bodies that are the same *behaviour* live here. Two near-neighbours were
compared and deliberately left where they are, because sharing them would have
changed what a caller gets:

- the two ``_write_text_atomic`` implementations (``assets.py``, ``input.py``)
  differ in the mode of the file they leave behind. `assets` stages through
  `tempfile.NamedTemporaryFile`, which `mkstemp`s at ``0600``, so the replaced
  file ends up owner-only; `input` stages inside a `TemporaryDirectory` and
  writes with `Path.write_text`, so the replaced file ends up ``0666 & ~umask``
  (``0664`` here). They also leave different debris mid-write -- a dotted
  ``.tmp`` sibling versus a temporary directory -- which the readiness scans in
  `assets` walk.
- ``cli._format_bytes`` and ``report._bytes`` render different text. The CLI
  prints whole bytes without decimals (``512 B``) and accepts only an `int`;
  the report prints one decimal at every scale (``512.0 B``) and renders
  ``"-"`` for anything non-numeric or non-positive.

Nothing here imports anything but the standard library, so the module stays
usable from every layer, including the pre-JAX part of the CLI.

The three notions of *checkpoint identity* are not here either, and must not be
merged into one: `assets` persists size + mtime_ns + published SHA, `manifest`
persists a six-field stat tuple plus a tree digest, and `cache` derives
``str(path) + ":" + st_size``. The first two are written to disk and compared
for equality, so unifying them is an on-disk schema change.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

#: Read size for streamed hashing. Large enough that hashing a multi-GB
#: checkpoint is not syscall-bound, small enough never to hold the file.
_HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, read in bounded chunks.

    Streaming rather than `Path.read_bytes` is the contract, not an
    optimisation: these are checkpoints and alignment artifacts, and one of
    them is hashed on a path that must never hold the whole file in memory.
    `Path.open` rather than the builtin is also the contract -- it is the seam
    `tests/test_resume_manifest.py` patches to observe which files a manifest
    identity actually reads.
    """
    digest = hashlib.sha256()
    update_digest_from_file(digest, path)
    return digest.hexdigest()


def update_digest_from_file(digest: Any, path: Path) -> None:
    """Feed a file's bytes into a running ``hashlib`` digest, in bounded chunks.

    For a digest that covers more than one file -- a featurization request
    folds its job text and every file it names into one key -- where hashing
    each file separately would change the key every existing cache holds.
    """
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)


def ordinary_file_mode() -> int:
    """The mode an ordinary ``open(..., "w")`` would give: ``0o666 & ~umask``.

    For files staged through `tempfile`, which creates them ``0600``: chmod
    the staged file to this before ``os.replace`` so a published result stays
    readable to whoever reads the directory around it.

    The umask is read from ``/proc/self/status`` where the kernel reports it
    (Linux 4.7+), because the only portable way to read it -- set it and put
    it back -- briefly leaves the process at ``0`` for any thread creating a
    file meanwhile. Elsewhere it falls back to that set-and-restore.
    """
    return 0o666 & ~_current_umask()


def _current_umask() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0)
    os.umask(mask)
    return mask


def nonempty_file(path: Path) -> bool:
    """Whether ``path`` is a regular, non-empty file right now."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


_UNSAFE_NAME = re.compile(r"[^\w.-]+", flags=re.UNICODE)


def safe_job_name(name: str, *, limit: int = 120) -> str:
    """Return a readable filename component that cannot escape its run root.

    The one rule every writer that turns a job name into a path follows --
    FoldJAX's common layout, Protenix/OpenDDE's original-style tree and
    OpenFold3's native files -- so one job lands under one name whichever
    model wrote it. A name of ASCII letters, digits, ``_``, ``.`` and ``-``
    that neither starts nor ends with ``.``/``_`` comes back unchanged.
    """
    original = str(name).strip()
    safe = _UNSAFE_NAME.sub("_", original.replace("/", "_").replace("\\", "_"))
    safe = safe.strip("._") or "prediction"
    if len(safe.encode("utf-8")) <= limit:
        return safe
    digest = hashlib.sha256(original.encode()).hexdigest()[:8]
    prefix_bytes = safe.encode("utf-8")[: limit - len(digest) - 1]
    prefix = prefix_bytes.decode("utf-8", errors="ignore").rstrip("._-")
    return f"{prefix or 'prediction'}-{digest}"
