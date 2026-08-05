"""Parsing real depositions into a neutral target description.

Marked ``network`` because these fetch from RCSB. The cache makes repeat runs
local, but a first run needs access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._foldbench import TARGETS, by_size, load_target, parse_target
from tests._foldbench.rcsb import UnsupportedEntityError
from tests._foldbench.specs import openfold3_query

pytestmark = pytest.mark.network


def test_every_target_parses() -> None:
    for spec in TARGETS:
        target = load_target(spec.pdb_id)
        assert target.pdb_id == spec.pdb_id
        assert target.polymers, f"{spec.pdb_id} has no polymers"
        # The curated token estimate is what the size ordering relies on, so it
        # must track what the file actually contains.
        assert abs(target.polymer_tokens - spec.approx_tokens) <= 2, (
            f"{spec.pdb_id}: {target.polymer_tokens} tokens, "
            f"spec says ~{spec.approx_tokens}"
        )


def test_the_set_spans_lengths_and_compositions() -> None:
    """The point of the set is coverage; a regression to protein-only would pass
    every other test in this repo."""
    targets = [load_target(spec.pdb_id) for spec in TARGETS]
    types = {kind for target in targets for kind in target.molecule_types}
    assert {"protein", "dna", "rna"} <= types
    assert any(target.ligands for target in targets), "no ligands anywhere"
    assert any(
        len(polymer.chain_ids) > 1 for target in targets for polymer in target.polymers
    ), "no entity with repeated chains"
    assert any(len(target.molecule_types) > 1 for target in targets), "no mixed target"
    sizes = sorted(target.polymer_tokens for target in targets)
    # The upper bound is the load-bearing half. Pair attention is cubic in token
    # count, so a port that is comfortable at 600 tokens can need 267 GiB at 2000;
    # a set that stopped below ~1500 could not have shown that, and this repo's
    # earlier conclusions about chunking were wrong for exactly that reason.
    assert sizes[0] < 50, f"nothing small enough for a fast check: {sizes}"
    assert sizes[-1] >= 2000, f"nothing at the scale that breaks ports: {sizes}"
    assert any(1200 <= size <= 1700 for size in sizes), (
        f"no target between the mid range and the top, so the cost curve has a "
        f"gap where it matters: {sizes}"
    )
    # Reaching 2000 tokens with one long chain would miss repeated-chain
    # tokenization, which is the harder case and the one 3U7Q covers.
    largest = max(targets, key=lambda target: target.polymer_tokens)
    assert sum(len(polymer.chain_ids) for polymer in largest.polymers) > 1, (
        f"{largest.pdb_id} is the largest target and is a single chain"
    )


def test_waters_are_not_ligands() -> None:
    """Every entry has waters and no model takes them, so counting them as ligands
    would make the ligand coverage assertion above meaningless."""
    target = load_target("1UBQ")
    assert "HOH" not in target.ligands
    assert target.ligands == ()


def test_repeated_chains_are_one_entity() -> None:
    target = load_target("4HHB")
    assert len(target.polymers) == 2
    assert all(len(polymer.chain_ids) == 2 for polymer in target.polymers)
    assert target.polymer_tokens == sum(
        len(p.sequence) * 2 for p in target.polymers
    )
    assert "HEM" in target.ligands


def test_by_size_is_ordered_and_bounded() -> None:
    ordered = by_size()
    assert [t.approx_tokens for t in ordered] == sorted(
        t.approx_tokens for t in ordered
    )
    assert all(t.approx_tokens <= 200 for t in by_size(200))


def test_an_unknown_polymer_type_is_refused(tmp_path: Path) -> None:
    """Silently dropping an entity would change the target without saying so."""
    cif = tmp_path / "fake.cif"
    cif.write_text(
        "data_FAKE\n"
        "loop_\n"
        "_entity_poly.type\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        "_entity_poly.pdbx_strand_id\n"
        "'polysaccharide(D)' AAA A\n"
    )
    with pytest.raises(UnsupportedEntityError, match="polysaccharide"):
        parse_target(cif)


def test_ligands_become_their_own_chains_with_fresh_ids() -> None:
    """Ligands have no chain id in the deposition, and a synthetic one must not
    collide with a polymer's."""
    target = load_target("4HHB")
    spec = openfold3_query(target)
    chains = spec["queries"]["4HHB"]["chains"]
    polymer_ids = {
        i
        for c in chains
        if c["molecule_type"] != "ligand"
        for i in c["chain_ids"]
    }
    ligand_ids = [
        i for c in chains if c["molecule_type"] == "ligand" for i in c["chain_ids"]
    ]
    assert len(ligand_ids) == len(set(ligand_ids))
    assert not (set(ligand_ids) & polymer_ids)
    assert [c["ccd_codes"] for c in chains if c["molecule_type"] == "ligand"] == [
        ["HEM"],
        ["PO4"],
    ]


def test_ligands_can_be_excluded() -> None:
    target = load_target("3HTB")
    spec = openfold3_query(target, include_ligands=False)
    kinds = {c["molecule_type"] for c in spec["queries"]["3HTB"]["chains"]}
    assert "ligand" not in kinds
