"""Building a mixed-entity benchmark set: tokens, schema, alignments, validation.

Offline except for one ``network`` test. The mmCIF fixture stands in for a real
deposition so the whole selection-to-validation path can run without RCSB, and
the reference job file is the real one from the 2026-09-10 set, so the schema
assertions compare against bytes that a benchmark actually ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("biotite", reason="composition parsing reads mmCIF with biotite")

from bench.mixed_cases import (  # noqa: E402
    BUCKETS,
    EXCLUDED_COMPONENTS,
    STRUCTURAL_IONS,
    Composition,
    Entity,
    bucket_band,
    bucket_of,
    build_job,
    case_msa_requests,
    case_name,
    dump_job,
    dump_sequences,
    format_reports,
    heavy_atom_counts,
    job_sequence,
    materialise,
    parse_composition,
    sanitise_a3m,
    search_query,
    select_candidates,
    sequence_key,
    token_count,
    validate_set,
)

DATA = Path(__file__).parent / "data"
MINI_CIF = DATA / "mixed_cases_mini.cif"
REFERENCE_JOB = DATA / "mixed_cases_reference_job.json"
REFERENCE_SEQUENCES = DATA / "mixed_cases_reference_sequences.json"

#: What the fixture's components cost, so no test needs the network. Passing this
#: explicitly bypasses `heavy_atom_counts`, which is where ions are known to be
#: monatomic, so the ions have to be spelled out here too.
HEAVY_ATOMS = {"ZN": 1, "MG": 1, "MN": 1, "GTP": 32, "GOL": 6}


@pytest.fixture
def mini() -> Composition:
    return parse_composition(MINI_CIF, "MINI")


# --------------------------------------------------------------------------- #
# Composition and the deny list
# --------------------------------------------------------------------------- #


def test_ligand_copies_come_from_struct_asym(mini: Composition) -> None:
    """The copy count is the whole reason this does not reuse ``parse_target``.

    ``pdbx_entity_nonpoly`` has one row per ligand *entity*, so a zinc present
    three times appears once. Reading the count from there would give every
    ligand one chain and undercount the tokens of exactly the entries this set
    exists to cover.
    """
    ligands = {entity.ccd: entity.copies for entity in mini.ligands}
    assert ligands == {"ZN": 3, "GTP": 1}
    protein = next(e for e in mini.entities if e.kind == "protein")
    assert protein.copies == 2


def test_additives_and_solvent_are_dropped(mini: Composition) -> None:
    codes = {entity.ccd for entity in mini.ligands}
    assert "GOL" not in codes, "glycerol is a cryoprotectant, not part of the assembly"
    assert "HOH" not in codes


def test_solvent_can_be_kept_when_asked() -> None:
    """The exclusion is a policy of this set, not a property of the parser."""
    everything = parse_composition(MINI_CIF, "MINI", include_solvent=True)
    codes = {entity.ccd for entity in everything.ligands}
    assert {"GOL", "HOH"} <= codes


def test_the_deny_list_covers_the_named_additives() -> None:
    named = {"HOH", "GOL", "EDO", "PEG", "SO4", "PO4", "CL", "NA", "ACT", "DMS", "MPD"}
    assert named <= EXCLUDED_COMPONENTS


def test_structural_ions_survive_the_deny_list() -> None:
    assert not (STRUCTURAL_IONS & EXCLUDED_COMPONENTS)
    assert {"ZN", "MG", "CA", "MN", "FE", "K"} == set(STRUCTURAL_IONS)


def test_potassium_diverges_from_the_curated_additive_list() -> None:
    """A documented disagreement, asserted so it cannot be undone by accident.

    ``tests._foldbench.targets`` denies K as a buffer salt, which is right for a
    cost measurement of an arbitrary entry. This set is chosen to exercise ion
    handling, and K is the coordinating ion of K+ channels and G-quadruplexes,
    so it is re-admitted here. If that list ever stops denying it, this test
    fails and the comment in ``mixed_cases`` needs rewriting.
    """
    from tests._foldbench.targets import CRYSTALLIZATION_ADDITIVES

    assert "K" in CRYSTALLIZATION_ADDITIVES
    assert "K" not in EXCLUDED_COMPONENTS
    for salt in ("NA", "CL"):
        assert salt in CRYSTALLIZATION_ADDITIVES and salt in EXCLUDED_COMPONENTS


def test_resolution_is_read_from_the_deposition(mini: Composition) -> None:
    assert mini.resolution == pytest.approx(2.10)


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


def test_tokens_count_polymer_residues_and_ligand_heavy_atoms(
    mini: Composition,
) -> None:
    # 10 residues x 2 chains + 4 nucleotides + 3 zinc + one 32-atom GTP.
    assert mini.polymer_tokens == 24
    assert token_count(mini.entities, HEAVY_ATOMS) == 24 + 3 + 32


def test_an_organic_ligand_is_not_one_token(mini: Composition) -> None:
    """The defect this rule replaces.

    Counting one token per ligand copy is right for ions and wrong for anything
    else, and it is what the hand-built set recorded. On the fixture it hides 31
    tokens; on 5NPK it hid 200.
    """
    per_copy = mini.polymer_tokens + sum(e.copies for e in mini.ligands)
    assert per_copy == 28
    assert token_count(mini.entities, HEAVY_ATOMS) == 59


def test_ions_cost_no_request() -> None:
    def refuse(code: str) -> int:
        raise AssertionError(f"asked the network for the monatomic ion {code}")

    assert heavy_atom_counts(STRUCTURAL_IONS, fetch=refuse) == dict.fromkeys(
        STRUCTURAL_IONS, 1
    )


def test_heavy_atom_counts_are_cached_between_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(code: str) -> int:
        calls.append(code)
        return 32

    cache = tmp_path / "chemcomp.json"
    assert heavy_atom_counts(["GTP"], cache_path=cache, fetch=fetch) == {"GTP": 32}
    assert heavy_atom_counts(["GTP"], cache_path=cache, fetch=fetch) == {"GTP": 32}
    assert calls == ["GTP"], "the second call should have read the cache"
    assert json.loads(cache.read_text()) == {"GTP": 32}


def test_an_unknown_ligand_is_refused_rather_than_guessed(mini: Composition) -> None:
    with pytest.raises(KeyError):
        token_count(mini.entities, {"ZN": 1})


def test_bucket_bands_and_names() -> None:
    assert bucket_band(2000, 0.25) == (1500, 2500)
    assert bucket_of("mixed_3k_5npk") == 3000
    assert bucket_of("mixed_mini") is None
    assert case_name("5NPK", 3000) == "mixed_3k_5npk"
    assert BUCKETS == (1000, 2000, 3000, 4000, 5000)


# --------------------------------------------------------------------------- #
# Job schema
# --------------------------------------------------------------------------- #


def reference_composition() -> Composition:
    """4XWW rebuilt by hand from its deposition, sequences read from the fixture.

    Hand-written rather than parsed so the test does not need the 1.8 MB mmCIF,
    and so a change to ``parse_composition`` cannot make ``build_job``'s byte
    comparison pass for the wrong reason.
    """
    job = json.loads(REFERENCE_JOB.read_text())
    by_type = {entity["type"]: entity for entity in job["entities"]}
    return Composition(
        pdb_id="4XWW",
        entities=(
            Entity(1, "protein", 2, sequence=by_type["protein"]["sequence"]),
            Entity(2, "rna", 2, sequence=by_type["rna"]["sequence"]),
            Entity(3, "ligand", 4, ccd="ZN"),
            Entity(4, "ligand", 2, ccd="MN"),
        ),
        resolution=2.05,
    )


def test_build_job_reproduces_the_reference_bytes() -> None:
    """Chain ids, entity order, grouping, MSA paths and formatting, at once.

    Byte equality is the point. The chain ids in the reference are *not* the
    deposition's -- 4XWW's RNA strands are D and E, and the job calls them C and
    D -- because excluded components leave gaps and ligands have no strand id at
    all, so every entity is relabelled in entity order. A structural comparison
    would pass while emitting the deposition's ids.
    """
    job = build_job(reference_composition(), "mixed_1k_4xww")
    assert dump_job(job) == REFERENCE_JOB.read_text()


def test_the_sequences_entry_reproduces_the_reference_bytes() -> None:
    job = build_job(reference_composition(), "mixed_1k_4xww")
    document = {
        "mixed_1k_4xww": {
            "length": token_count(reference_composition().entities, HEAVY_ATOMS),
            "sequence": job_sequence(job),
            "pdb": "4XWW",
        }
    }
    assert dump_sequences(document) == REFERENCE_SEQUENCES.read_text()


def test_the_job_carries_no_key_the_harness_refuses() -> None:
    """The harness's own answer, not a restatement of its key set.

    ``compatibility`` is what decides whether a backend can run a document, so
    asking it is the only assertion that cannot drift from the thing it checks.
    """
    from foldjax.input import compatibility

    job = build_job(reference_composition(), "mixed_1k_4xww")
    for model in ("boltz2", "protenix", "openfold3"):
        assert compatibility(job, model) is None, model


def test_the_harness_really_refuses_a_pdb_key() -> None:
    """Without this the test above would pass against a harness that checks nothing."""
    from foldjax.input import compatibility

    job = build_job(reference_composition(), "mixed_1k_4xww")
    assert compatibility({**job, "pdb": "4XWW"}, "boltz2") == (
        "unsupported top-level fields: 'pdb'"
    )


def test_identical_chains_are_one_entity(mini: Composition) -> None:
    job = build_job(mini, "mixed_mini")
    protein = next(e for e in job["entities"] if e["type"] == "protein")
    assert protein["id"] == ["A", "B"]
    zinc = next(e for e in job["entities"] if e.get("ccd") == "ZN")
    assert zinc["id"] == ["D", "E", "F"]
    assert "sequence" not in zinc


def test_only_proteins_get_an_alignment_path(mini: Composition) -> None:
    job = build_job(mini, "mixed_mini")
    with_msa = [e["type"] for e in job["entities"] if "unpaired_msa" in e]
    assert with_msa == ["protein"]
    assert case_msa_requests(job) == [
        ("mixed_mini_p1", "MKTAYIAKQR", "../msa/mixed_mini_p1_unpaired.a3m")
    ]


def test_proteins_are_numbered_in_entity_order() -> None:
    """`_p2` is the second protein, not the second entity.

    A nucleic acid between two proteins would otherwise shift every alignment
    path by one and silently pair each protein with another's MSA.
    """
    composition = Composition(
        pdb_id="X",
        entities=(
            Entity(1, "protein", 1, sequence="AAAA"),
            Entity(2, "dna", 1, sequence="GATC"),
            Entity(3, "protein", 2, sequence="CCCC"),
        ),
    )
    built = build_job(composition, "mixed_x")
    assert [label for label, _, _ in case_msa_requests(built)] == [
        "mixed_x_p1",
        "mixed_x_p2",
    ]
    assert built["entities"][2]["id"] == ["C", "D"]


# --------------------------------------------------------------------------- #
# a3m sanitising
# --------------------------------------------------------------------------- #

QUERY = "MKTAYIAKQR"


def a3m(*records: tuple[str, str]) -> str:
    return "".join(f">{name}\n{sequence}\n" for name, sequence in records)


def test_nul_bytes_are_stripped() -> None:
    """The defect that breaks four parsers.

    The ColabFold archive carries trailing NULs, and OpenFold3, Boltz-2,
    ESMFold2 and AlphaFold3 all read the file as text.
    """
    dirty = a3m(("query", QUERY), ("hit", "MKTAYIAKQR")) + "\x00\x00\x00"
    result = sanitise_a3m(dirty, QUERY)
    assert result.stripped_nuls == 3
    assert "\x00" not in result.text
    assert result.records == 2


def test_records_with_the_wrong_column_count_are_dropped() -> None:
    text = a3m(
        ("query", QUERY),
        ("good", "MKTAYIAKQR"),
        ("truncated", "MKTAY"),
        ("gapped_ok", "MKT---AKQR"),
        ("too_long", "MKTAYIAKQRRR"),
    )
    result = sanitise_a3m(text, QUERY)
    assert result.dropped_columns == 2
    assert result.records == 3
    assert "truncated" not in result.text and "too_long" not in result.text
    assert "gapped_ok" in result.text


def test_lowercase_insertions_do_not_count_as_columns() -> None:
    """An a3m insertion belongs to no column; counting it would drop valid rows."""
    text = a3m(("query", QUERY), ("insert", "MKTAYIAkkkKQR"))
    result = sanitise_a3m(text, QUERY)
    assert result.dropped_columns == 0
    assert result.records == 2


def test_duplicate_copies_of_the_query_are_kept() -> None:
    """Every file in the reference set has between one and three of them."""
    text = a3m(("query", QUERY), ("hit", "MKTAYIAKQR"), ("again", QUERY))
    assert sanitise_a3m(text, QUERY).records == 3


def test_a_first_record_that_is_not_the_query_is_refused() -> None:
    text = a3m(("wrong", "AAAAAAAAAA"), ("query", QUERY))
    with pytest.raises(ValueError, match="does not match the requested sequence"):
        sanitise_a3m(text, QUERY)


def test_an_empty_or_headerless_alignment_is_refused() -> None:
    with pytest.raises(ValueError, match="empty or has no query"):
        sanitise_a3m("", QUERY)
    with pytest.raises(ValueError, match="does not start with a FASTA header"):
        sanitise_a3m("MKTAYIAKQR\n", QUERY)


def test_the_cache_key_is_the_sequence_not_the_case() -> None:
    """A chain repeated across two cases is searched once."""
    assert sequence_key(QUERY) == sequence_key(f" {QUERY.lower()} \n")
    assert sequence_key(QUERY) != sequence_key(QUERY + "A")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def build_set(tmp_path: Path, case: str = "mixed_mini") -> Path:
    """A complete, valid one-case set built from the fixture deposition."""
    cifs = tmp_path / "cif"
    cifs.mkdir()
    (cifs / "MINI.cif").write_text(MINI_CIF.read_text())
    root = tmp_path / "set"
    materialise("MINI", case, root, cif_directory=cifs, heavy_atoms=HEAVY_ATOMS)
    job = json.loads((root / "jobs" / f"{case}.json").read_text())
    for _, sequence, relative in case_msa_requests(job):
        path = (root / "jobs" / relative).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(a3m(("query", sequence), ("hit", sequence)))
    return root


def test_a_materialised_set_validates_clean(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    reports = validate_set(root, heavy_atoms=HEAVY_ATOMS)
    assert len(reports) == 1
    assert reports[0].problems == ()
    assert reports[0].tokens == 59
    assert reports[0].recorded_length == 59
    assert reports[0].alignments == 1
    assert "1/1 cases clean" in format_reports(reports)


def test_a_missing_alignment_is_reported(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    next((root / "msa").glob("*.a3m")).unlink()
    (problem,) = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert "is missing" in problem


def test_a_nul_in_an_alignment_is_reported(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    path = next((root / "msa").glob("*.a3m"))
    path.write_bytes(path.read_bytes() + b"\x00")
    problems = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert any("NUL" in problem for problem in problems)


def test_a_ragged_alignment_is_reported(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    path = next((root / "msa").glob("*.a3m"))
    path.write_text(path.read_text() + ">short\nMKT\n")
    problems = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert any("wrong column count" in problem for problem in problems)


def test_an_alignment_whose_query_moved_is_reported(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    path = next((root / "msa").glob("*.a3m"))
    path.write_text(a3m(("other", "AAAAAAAAAA"), ("query", "MKTAYIAKQR")))
    problems = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert any("first record is not the query" in problem for problem in problems)


def test_a_token_count_outside_the_band_is_reported(tmp_path: Path) -> None:
    """The fixture is 59 tokens, so claiming the 1k rung has to fail."""
    root = build_set(tmp_path, case="mixed_1k_mini")
    report = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0]
    assert report.bucket == 1000
    assert report.in_band is False
    assert any("outside 750-1250" in problem for problem in report.problems)


def test_a_key_the_harness_refuses_is_reported(tmp_path: Path) -> None:
    root = build_set(tmp_path)
    path = root / "jobs" / "mixed_mini.json"
    job = json.loads(path.read_text())
    path.write_text(json.dumps({**job, "pdb": "MINI"}, indent=2, sort_keys=True))
    problems = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert any(
        "refuses this job" in problem and "pdb" in problem for problem in problems
    )


def test_a_sequences_entry_that_drifted_from_the_job_is_reported(
    tmp_path: Path,
) -> None:
    root = build_set(tmp_path)
    path = root / "sequences.json"
    document = json.loads(path.read_text())
    document["mixed_mini"]["sequence"] = "AAAA"
    path.write_text(json.dumps(document, indent=1))
    problems = validate_set(root, heavy_atoms=HEAVY_ATOMS)[0].problems
    assert any("does not match the job" in problem for problem in problems)


def test_the_table_shows_both_token_numbers(tmp_path: Path) -> None:
    """A set built before the heavy-atom rule must not look identical to one after."""
    root = build_set(tmp_path)
    path = root / "sequences.json"
    document = json.loads(path.read_text())
    document["mixed_mini"]["length"] = 28  # the old one-token-per-copy count
    path.write_text(json.dumps(document, indent=1))
    table = format_reports(validate_set(root, heavy_atoms=HEAVY_ATOMS))
    assert "length off by +31" in table


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_the_search_query_brackets_the_band_rather_than_matching_it() -> None:
    """RCSB counts polymer monomers, which is a lower bound on tokens.

    Asking it for the token band would drop every entry whose ligands make up
    the difference, which is the kind of entry this set wants.
    """
    query = search_query(2000, tolerance=0.25, max_resolution=3.0)
    terminals = {
        node["parameters"]["attribute"]: node["parameters"]
        for node in query["query"]["nodes"]
    }
    monomers = terminals["rcsb_entry_info.deposited_polymer_monomer_count"]
    assert monomers["value"] == {"from": 1200, "to": 2500}
    assert terminals["rcsb_entry_info.resolution_combined"]["value"] == 3.0
    assert terminals["rcsb_entry_info.nonpolymer_entity_count"]["value"] == 0
    assert (
        terminals["rcsb_entry_info.selected_polymer_entity_types"]["value"]
        == "Protein/NA"
    )
    # `sort_by`, not `attribute`. The sort clause spells the field differently
    # from a terminal's parameters and the endpoint answers the other spelling
    # with a bare HTTP 400, which is not a message anyone would decode twice.
    assert query["request_options"]["sort"] == [
        {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
    ]
    # Serialisable, because it travels as a URL query parameter.
    assert json.loads(json.dumps(query)) == query


def test_resolution_can_be_left_unconstrained() -> None:
    query = search_query(1000, max_resolution=None)
    attributes = {node["parameters"]["attribute"] for node in query["query"]["nodes"]}
    assert "rcsb_entry_info.resolution_combined" not in attributes


def test_candidates_prefer_composition_over_closeness(monkeypatch) -> None:
    """A protein-plus-zinc entry at the exact rung loses to a mixed one nearby.

    Ranking on token distance alone would rebuild the protein-only set this
    module exists to replace.
    """
    import bench.mixed_cases as module

    plain = Composition(
        "PLAIN",
        (
            Entity(1, "protein", 1, sequence="A" * 1000),
            Entity(2, "ligand", 1, ccd="ZN"),
        ),
        resolution=1.5,
    )
    mixed = Composition(
        "MIXED",
        (
            Entity(1, "protein", 1, sequence="A" * 900),
            Entity(2, "dna", 1, sequence="GATC" * 5),
            Entity(3, "ligand", 2, ccd="GTP"),
        ),
        resolution=2.8,
    )
    by_id = {"PLAIN": plain, "MIXED": mixed}
    monkeypatch.setattr(module, "search_entries", lambda query: ["PLAIN", "MIXED"])
    monkeypatch.setattr(module, "fetch_cif", lambda pdb, directory=None: pdb)
    monkeypatch.setattr(module, "parse_composition", lambda path, pdb: by_id[pdb])

    ranked = select_candidates(1000, heavy_atoms=HEAVY_ATOMS, polymer_types=("any",))
    assert [record["pdb"] for record in ranked] == ["MIXED", "PLAIN"]
    assert ranked[0]["tokens"] == 900 + 20 + 64
    assert ranked[0]["in_band"] is True
    assert ranked[0]["refused"] is None


def test_an_entry_the_harness_refuses_never_wins_its_bucket(monkeypatch) -> None:
    """7MJW is why this term outranks every other.

    It is 1029 tokens, protein with RNA and four distinct ligands at 1.4 A --
    the best 1k candidate on every axis a token count can see -- and its RNA
    carries a modified nucleotide whose canonical one-letter mapping is `X`.
    Every backend refuses it, so recommending it costs an MSA search and a
    materialise before anything says so.
    """
    import bench.mixed_cases as module

    refused = Composition(
        "REFUSD",
        (
            Entity(1, "protein", 2, sequence="M" * 457),
            # The `X` is what the deposition's canonical sequence actually says.
            Entity(2, "rna", 2, sequence="GGACUGAAXAUCC"),
            Entity(3, "ligand", 3, ccd="MG"),
        ),
        resolution=1.4,
    )
    plain = Composition(
        "PLAIN",
        (
            Entity(1, "protein", 1, sequence="A" * 900),
            Entity(2, "dna", 1, sequence="GATC" * 5),
            Entity(3, "ligand", 2, ccd="GTP"),
        ),
        resolution=2.9,
    )
    by_id = {"REFUSD": refused, "PLAIN": plain}
    monkeypatch.setattr(module, "search_entries", lambda query: ["REFUSD", "PLAIN"])
    monkeypatch.setattr(module, "fetch_cif", lambda pdb, directory=None: pdb)
    monkeypatch.setattr(module, "parse_composition", lambda path, pdb: by_id[pdb])

    ranked = select_candidates(1000, heavy_atoms=HEAVY_ATOMS, polymer_types=("any",))
    assert [record["pdb"] for record in ranked] == ["PLAIN", "REFUSD"]
    assert ranked[0]["refused"] is None
    assert "unsupported residue 'X'" in ranked[1]["refused"]
    # Still in band -- it is refused on its residues, not on its size, and the
    # report has to keep saying so or the next reader re-picks it.
    assert ranked[1]["in_band"] is True


def test_an_unparsable_entry_is_recorded_rather_than_raised(monkeypatch) -> None:
    import bench.mixed_cases as module
    from tests._foldbench.rcsb import UnsupportedEntityError

    def explode(path, pdb):
        raise UnsupportedEntityError("polysaccharide(D)")

    monkeypatch.setattr(module, "search_entries", lambda query: ["BAD"])
    monkeypatch.setattr(module, "fetch_cif", lambda pdb, directory=None: pdb)
    monkeypatch.setattr(module, "parse_composition", explode)
    (record,) = select_candidates(1000, heavy_atoms=HEAVY_ATOMS, polymer_types=("any",))
    assert record == {"pdb": "BAD", "skipped": "polysaccharide(D)"}


@pytest.mark.network
def test_a_bucket_query_returns_real_entries() -> None:
    """One small live query, so a schema change at RCSB is caught here."""
    from bench.mixed_cases import search_entries

    identifiers = search_entries(search_query(1000, rows=5))
    assert identifiers, "no entry in the 1k band; the query or the schema moved"
    assert all(len(identifier) == 4 for identifier in identifiers)
