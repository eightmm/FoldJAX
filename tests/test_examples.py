"""The shipped example jobs must actually parse and translate.

An example that no longer matches the schema is worse than none: it is the
first thing a new user copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.api import detect_input_format
from foldjax.input import materialize_native_input
from foldjax.registry import available_models, capabilities

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _jobs() -> list[Path]:
    return sorted(
        path for path in EXAMPLES.iterdir() if path.suffix in {".json", ".yaml"}
    )


def test_there_is_at_least_one_example() -> None:
    assert _jobs(), "the repository ships no runnable example job"


@pytest.mark.parametrize("job", _jobs(), ids=lambda path: path.name)
def test_every_example_is_common_schema(job: Path) -> None:
    assert detect_input_format(job) == "foldjax"


@pytest.mark.parametrize("job", _jobs(), ids=lambda path: path.name)
@pytest.mark.parametrize("model", available_models())
def test_every_example_translates_to_every_backend(
    job: Path, model: str, tmp_path: Path
) -> None:
    """The common schema's promise is that one file runs anywhere."""
    written = materialize_native_input(
        job, capabilities(model), tmp_path / model / job.stem, seed=101
    )
    assert written.is_file()
    assert written.read_text().strip()
