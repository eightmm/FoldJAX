"""The shipped example jobs must actually parse and translate.

An example that no longer matches the schema is worse than none: it is the
first thing a new user copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.api import detect_input_format
from foldjax.input import materialize_native_input, read_job_document
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
    """The common schema's promise: one file runs on every backend that can
    express what it names, and is refused by name on every backend that cannot.

    Only ESMFold2 currently refuses anything. Upstream ESMFold2 supports all
    biomolecules, but FoldJAX has not yet ported its ligand and nucleic-acid
    feature builder. The refusal is the point being tested: dropping the ligand
    and folding the protein would answer a different question than the file asks.
    """
    document = read_job_document(job)
    named = {entity.get("type") for entity in document.get("entities", [])}
    supported = set(capabilities(model).entity_types)
    if not named <= supported:
        with pytest.raises(ValueError):
            materialize_native_input(
                job, capabilities(model), tmp_path / model / job.stem, seed=101
            )
        return
    written = materialize_native_input(
        job, capabilities(model), tmp_path / model / job.stem, seed=101
    )
    assert written.is_file()
    assert written.read_text().strip()
