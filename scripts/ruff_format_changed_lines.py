"""Run ``ruff format`` on the lines a commit changes, not on whole files.

A pre-commit hook. Hundreds of files in this tree predate ``ruff format``, so
the stock ``ruff-format`` hook would rewrite every file the first time anyone
touched one line of it, burying the change in unrelated reformatting. This
formats each staged hunk with ``ruff format --range`` instead; a new file is one
hunk and is formatted whole. Ruff's own ``extend-exclude`` still applies.

Hunks are formatted last to first, so reformatting one cannot shift the line
numbers of a hunk still waiting. Like any formatting hook, it exits non-zero
when it changed a file, and the commit is retried after ``git add``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator, Sequence

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def staged_ranges(path: str) -> Iterator[tuple[int, int]]:
    """1-based inclusive line ranges the index adds or changes in ``path``."""
    diff = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--no-color", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in diff.splitlines():
        match = _HUNK.match(line)
        if match is None:
            continue
        start, count = int(match.group(1)), int(match.group(2) or "1")
        if count:
            yield start, start + count - 1


def main(paths: Sequence[str]) -> int:
    status = 0
    for path in paths:
        with open(path, "rb") as handle:
            before = handle.read()
        for start, end in sorted(staged_ranges(path), reverse=True):
            # A range ends *before* its end position, so the last changed line
            # is covered by ending at column 1 of the line after it.
            span = f"--range={start}:1-{end + 1}:1"
            command = [sys.executable, "-m", "ruff", "format", span, path]
            result = subprocess.run(command)
            status |= result.returncode
        with open(path, "rb") as handle:
            if handle.read() != before:
                print(f"reformatted changed lines in {path}")
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
