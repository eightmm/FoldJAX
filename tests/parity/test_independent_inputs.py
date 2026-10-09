"""Hand-written native inputs must featurize as the common job translates.

Every GPU check records that the upstream arm read the native input FoldJAX's
own writer produced for the same job, which proves the two runs saw one
document and nothing about whether that document says what the common job
says. ``bench/independent_inputs.py`` holds a native document per backend
written from the upstream's own format reference, featurizes it and the
writer's output with the same torch-free featurizer in separate processes
(RDKit's unseeded conformer stream is process-global) and compares the arrays
bitwise. This module is its pytest face.

Marked ``cpu_parity`` like the capture replays, since it reads released
assets (the CCD, Boltz-2's molecule directory, ESMFold2's ``ccd.pkl``); it
replays no capture and ships no manifest entry. The assets come from the
store ``FOLDJAX_INDEPENDENT_INPUTS_STORE`` names, read only: the test builds
its own scratch ``FOLDJAX_HOME`` with file symlinks to the two CCD files and
passes every other path explicitly, so a developer's store is never written.
Selected without that store it fails rather than skips, as the subset's
policy has it (``docs/parity-cpu.md``): a skipped parity test reads as a pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench import independent_inputs as harness

pytestmark = pytest.mark.cpu_parity

_STORE = "FOLDJAX_INDEPENDENT_INPUTS_STORE"


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> harness.Assets:
    store = os.environ.get(_STORE)
    if not store:
        pytest.fail(
            f"set {_STORE} to a fetched FoldJAX store (read only), e.g. the "
            "checkout's .foldjax after `foldjax weights fetch --model boltz2 "
            "esmfold2 protenix`"
        )
    try:
        resolved = harness.Assets.from_store(Path(store))
    except FileNotFoundError as error:
        pytest.fail(str(error))
    harness.prepare_home(resolved, tmp_path_factory.mktemp("independent-home"))
    return resolved


@pytest.fixture(scope="module")
def results(
    assets: harness.Assets, tmp_path_factory: pytest.TempPathFactory
) -> list[dict[str, object]]:
    return harness.run(
        harness.build_cases(),
        harness.MODELS,
        assets=assets,
        root=tmp_path_factory.mktemp("independent-cases"),
    )


def _cells(results: list[dict[str, object]]) -> dict[tuple[str, str], dict]:
    return {(str(r["case"]), str(r["model"])): r for r in results}


def test_every_cell_resolves(results: list[dict[str, object]]) -> None:
    """A cell that errors or lacks a native document certifies nothing."""
    unresolved = {
        (r["case"], r["model"]): r.get("error") or r["status"]
        for r in results
        if r["status"] in ("error", "no-native")
    }
    assert unresolved == {}


def test_the_writer_does_what_the_capability_table_promises(
    results: list[dict[str, object]],
) -> None:
    """Materialized, refused or dropped -- as ``common_schema_features`` says."""
    surprises = {
        (r["case"], r["model"]): (r["expected"], r["writer"], r.get("records"))
        for r in results
        if str(r["status"]).startswith("unexpected")
    }
    assert surprises == {}


def test_hand_written_natives_featurize_identically(
    results: list[dict[str, object]],
) -> None:
    """Bitwise, after the featurizer's own rerun floor is subtracted."""
    mismatches = {
        (r["case"], r["model"]): r["comparison"]
        for r in results
        if r["status"] == "mismatch"
    }
    assert mismatches == {}


def test_no_match_is_vacuous(results: list[dict[str, object]]) -> None:
    """Both arms must show the feature: a shared drop compares equal too."""
    vacuous = {
        (r["case"], r["model"]): r["vacuous"]
        for r in results
        if r["status"] == "vacuous"
    }
    assert vacuous == {}


def test_the_table_covers_the_chemistry_it_claims(
    results: list[dict[str, object]],
) -> None:
    """The rows the gap report asks for actually ran as matches."""
    cells = _cells(results)
    required = {
        ("monomer_msa", "boltz2"),
        ("heteromer_paired", "protenix"),
        ("homomer_ligand_ccd", "openfold3"),
        ("ligand_smiles", "boltz2"),
        ("modified_residue", "protenix"),
        ("glycan_bonds", "boltz2"),
        ("glycan_bonds", "esmfold2"),
        ("nucleic", "opendde"),
        ("template_mapped", "protenix"),
        ("template_bare", "openfold3"),
        ("pocket_explicit", "protenix"),
        ("pocket_default", "openfold3"),
        ("contact_explicit", "boltz2"),
    }
    missing = {key for key in required if cells.get(key, {}).get("status") != "match"}
    assert missing == set()
