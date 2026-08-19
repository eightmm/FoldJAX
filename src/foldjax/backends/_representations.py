"""Turn what a model wrote into the result object the common API hands back.

Each backend drives a different native writer, and those writers put their
output in different trees. The common layer pins one destination -- the
request's output directory -- so a caller comparing two models does not have
to know either one's layout: `result.representations.path` is in the same
place whichever model ran.
"""

from __future__ import annotations

import json
from pathlib import Path

from foldjax.models._representations import ARCHIVE_NAME, MANIFEST_NAME
from foldjax.schema import Representations


def _representations_result(
    model: str,
    output_dir: Path | None,
    wanted: tuple[str, ...],
) -> Representations | None:
    """Describe the archive a run wrote, or None when none was asked for.

    A request that asked for representations and got no archive is a bug in
    the backend wiring rather than a user error, so this is quiet about a
    missing file: the caller sees `representations is None` and the run's own
    output says what happened.
    """
    if not wanted or output_dir is None:
        return None
    archive = Path(output_dir) / ARCHIVE_NAME
    if not archive.is_file():
        return None
    manifest_path = Path(output_dir) / MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text()).get("representations", {})
    return Representations(model=model, path=archive, manifest=manifest)
