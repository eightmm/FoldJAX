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
from pathlib import Path

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
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_file(path: Path) -> bool:
    """Whether ``path`` is a regular, non-empty file right now."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
