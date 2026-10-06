"""``templates='auto'``: searched templates, per-model selection and delivery.

Nothing here reaches the network: the ColabFold server and RCSB are replaced by
fake transports, and the structures are small synthetic mmCIFs. The one live
test at the end runs only with ``FOLDJAX_LIVE_TEMPLATE_SEARCH=1``.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from foldjax import template_search
from foldjax.input import materialize_native_input
from foldjax.registry import capabilities
from foldjax.schema import PredictionRequest
from foldjax.search.msa import HttpResponse, SearchError
from foldjax.search.templates import (
    LocalTemplateHitsClient,
    RemoteTemplateHitsClient,
    StructureStore,
    TemplateHitsPipeline,
    parse_m8,
)

QUERY = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}  # fmt: skip


def _mmcif(entry: str, released: str, chains: list[tuple]) -> str:
    """A minimal mmCIF; ``chains`` is ``(label, author, sequence, unresolved)``.

    ``unresolved`` holds 1-based positions listed in the polymer sequence but
    given no coordinates, as a real deposition's disordered termini are.
    """
    lines = [
        f"data_{entry.upper()}",
        f"_entry.id {entry.upper()}",
        "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.revision_date",
        f"1 {released}",
        "2 2030-01-01",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
    ]
    lines += [f"{i} 'polypeptide(L)'" for i in range(1, len(chains) + 1)]
    lines += [
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
    ]
    for i, (_, _, sequence, _) in enumerate(chains, 1):
        lines += [f"{i} {n} {_THREE[a]}" for n, a in enumerate(sequence, 1)]
    lines += ["loop_", "_struct_asym.id", "_struct_asym.entity_id"]
    lines += [f"{chain[0]} {i}" for i, chain in enumerate(chains, 1)]
    names = sorted({_THREE[a] for chain in chains for a in chain[2]})
    lines += ["loop_", "_chem_comp.id", "_chem_comp.type"]
    lines += [f"{name} 'L-peptide linking'" for name in names]
    tags = (
        "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
        "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x "
        "Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id auth_asym_id "
        "pdbx_PDB_model_num"
    ).split()
    lines += ["loop_", *(f"_atom_site.{tag}" for tag in tags)]
    serial = 0
    for i, (label, author, sequence, unresolved) in enumerate(chains, 1):
        for n, residue in enumerate(sequence, 1):
            if n in unresolved:
                continue
            for atom, element, dx in (
                ("N", "N", 0.0),
                ("CA", "C", 1.4),
                ("C", "C", 2.8),
                ("O", "O", 3.6),
                ("CB", "C", 1.9),
            ):
                if atom == "CB" and residue == "G":
                    continue
                serial += 1
                lines.append(
                    f"ATOM {serial} {element} {atom} . {_THREE[residue]} {label} "
                    f"{i} {n} ? {3.8 * n + dx:.3f} {10.0 * i:.3f} "
                    f"{0.7 * (atom == 'CB'):.3f} 1.00 10.0 {n} {author} 1"
                )
    return "\n".join(lines) + "\n"


#: The structures the fake RCSB serves. ``2abc``'s author chain X is label A;
#: ``3abc`` is identical to the query (a self-hit AlphaFold 3 removes);
#: ``4abc`` is released after AlphaFold 3's 2021-09-30 cutoff; ``5abc`` is
#: released on it.
STRUCTURES = {
    "2abc": _mmcif(
        "2abc", "2001-02-03", [("A", "X", "GS" + QUERY[2:28] + "PP", {1, 2})]
    ),
    "3abc": _mmcif("3abc", "2005-01-01", [("A", "A", QUERY, set())]),
    "4abc": _mmcif("4abc", "2022-06-01", [("A", "B", QUERY[:30] + "GG", set())]),
    "5abc": _mmcif("5abc", "2021-09-30", [("A", "C", "W" + QUERY[1:29], set())]),
}

M8 = "\n".join(
    "\t".join(map(str, row))
    for row in (
        ("101", "4abc_B", 0.95, 30, 0, 0, 1, 30, 1, 30, 1e-30, 200),
        ("101", "3abc_A", 1.0, 33, 0, 0, 1, 33, 1, 33, 1e-40, 250),
        ("101", "2abc_X", 0.9, 26, 0, 0, 3, 28, 3, 28, 1e-20, 150),
        ("101", "5abc_C", 0.9, 28, 0, 0, 2, 29, 2, 29, 1e-20, 140),
        ("101", "9zzz_A", 0.5, 20, 0, 0, 1, 20, 1, 20, 1e-5, 50),
    )
)


def _server(m8: str = M8, calls: list | None = None):
    """A ColabFold server whose ``ticket/msa`` archive holds ``pdb70.m8``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = m8.encode()
        info = tarfile.TarInfo("pdb70.m8")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive = buffer.getvalue()

    def transport(method, url, data, headers, timeout):
        if calls is not None:
            calls.append((method, url, data))
        if url.endswith("ticket/msa"):
            return HttpResponse(200, b'{"status":"COMPLETE","id":"t-1"}')
        if url.endswith("result/download/t-1"):
            return HttpResponse(200, archive)
        raise AssertionError(url)

    return transport


def _rcsb(calls: list | None = None, structures=None, unreachable=()):
    """A fake RCSB; ``unreachable`` ids raise as a dropped connection does."""
    structures = STRUCTURES if structures is None else structures

    def transport(method, url, data, headers, timeout):
        if calls is not None:
            calls.append(url)
        name = url.rsplit("/", 1)[-1].removesuffix(".cif").lower()
        if name in unreachable or "*" in unreachable:
            raise OSError("connection refused")
        if name not in structures:
            return HttpResponse(404, b"not found")
        return HttpResponse(200, structures[name].encode())

    return transport


def _route(tmp_path, monkeypatch, *, m8=M8, structures=None, unreachable=()):
    """Point `foldjax.template_search` at the fake server and RCSB.

    A dropped connection is retried with backoff; the fake's are permanent, so
    the waits between attempts are skipped.
    """
    monkeypatch.setattr("foldjax.search.msa.time.sleep", lambda _seconds: None)
    calls: dict[str, list] = {"server": [], "rcsb": []}
    cache = tmp_path / "home" / "templates"

    def hits_pipeline():
        client = RemoteTemplateHitsClient(
            "https://colabfold.invalid",
            version="test",
            transport=_server(m8, calls=calls["server"]),
            poll_interval=0,
        )
        pipeline = TemplateHitsPipeline(
            cache / "hits", client, options={"host": "https://colabfold.invalid"}
        )
        return pipeline, {"kind": "remote", "host": "https://colabfold.invalid"}

    def structure_store():
        return StructureStore(
            cache / "mmcif",
            transport=_rcsb(calls["rcsb"], structures, unreachable),
        )

    monkeypatch.setattr(template_search, "_hits_pipeline", hits_pipeline)
    monkeypatch.setattr(template_search, "_structure_store", structure_store)
    return calls


@pytest.fixture
def searched(tmp_path, monkeypatch):
    return _route(tmp_path, monkeypatch)


def _job(tmp_path: Path, **entity) -> Path:
    body = {"type": "protein", "id": "Q", "sequence": QUERY, **entity}
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "t", "entities": [body]}))
    return path


def _materialize(source: Path, model: str, **kwargs):
    records: list = []
    options = kwargs.pop("options", None)
    path = materialize_native_input(
        source,
        capabilities(model),
        source.parent / f"out-{model}",
        seed=1,
        msa="single",
        options=options,
        templates="auto",
        template_search=records,
        **kwargs,
    )
    return path, records


# -- the hits and the structures ---------------------------------------------


def test_m8_hits_are_ordered_by_e_value_keeping_ties_stable():
    hits = parse_m8(M8)
    assert [hit.target for hit in hits] == [
        "3abc_A",
        "4abc_B",
        "2abc_X",
        "5abc_C",
        "9zzz_A",
    ]
    assert hits[0].pdb_id == "3abc" and hits[2].chain_id == "X"
    assert [hit.rank for hit in hits] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="not '<pdb id>_<chain>'"):
        parse_m8("101\tnot-a-hit\t1\t1\t0\t0\t1\t1\t1\t1\t1\t1\n")
    with pytest.raises(ValueError, match="not '<pdb id>_<chain>'"):
        parse_m8("101\t../x_A\t1\t1\t0\t0\t1\t1\t1\t1\t1\t1\n")
    with pytest.raises(ValueError, match="columns"):
        parse_m8("101\t1abc_A\t1\n")


def test_remote_hits_come_from_the_msa_ticket_and_are_cached(tmp_path):
    calls: list = []
    client = RemoteTemplateHitsClient(
        "https://colabfold.invalid",
        version="v",
        transport=_server(calls=calls),
        poll_interval=0,
    )
    pipeline = TemplateHitsPipeline(tmp_path, client, options={"host": "h"})
    first = pipeline.search(QUERY)
    assert Path(first["hitsPath"]).read_text() == M8
    method, url, data = calls[0]
    assert (method, url) == ("POST", "https://colabfold.invalid/ticket/msa")
    # OpenFold3 reads pdb70.m8 out of the ordinary env MSA job.
    assert b"mode=env" in data
    provenance = json.loads(Path(first["provenancePath"]).read_text())
    assert provenance["source"]["job_id"] == "t-1"
    count = len(calls)
    assert pipeline.search(QUERY.lower()) == first
    assert len(calls) == count


def test_malformed_hits_never_enter_the_cache(tmp_path):
    client = RemoteTemplateHitsClient(
        "https://x.invalid",
        version="v",
        transport=_server(m8="garbage\n"),
        poll_interval=0,
    )
    pipeline = TemplateHitsPipeline(tmp_path, client)
    with pytest.raises(SearchError, match="malformed"):
        pipeline.search(QUERY)
    assert not [path for path in tmp_path.iterdir() if not path.name.startswith(".")]


def test_local_template_command_writes_pdb70_m8(tmp_path):
    seen = {}

    def runner(command, **_):
        seen["command"] = command
        output = Path(command[command.index("--output") + 1])
        (output / "pdb70.m8").write_text(M8)
        import subprocess

        return subprocess.CompletedProcess(command, 0, "", "")

    client = LocalTemplateHitsClient(
        ["/opt/search.sh"], version="pdb-2026", runner=runner
    )
    payload = client.search(QUERY)
    assert payload.hits == M8
    assert seen["command"][0] == "/opt/search.sh"
    assert payload.source == {"command": ["/opt/search.sh"], "version": "pdb-2026"}


def test_structures_come_from_the_mirror_first_then_once_from_rcsb(tmp_path):
    mirror = tmp_path / "mirror" / "ab"
    mirror.mkdir(parents=True)
    (mirror / "2abc.cif").write_text(STRUCTURES["2abc"])
    calls: list = []
    store = StructureStore(
        tmp_path / "cache", local_dir=tmp_path / "mirror", transport=_rcsb(calls)
    )
    assert store.path("2ABC") == (mirror / "2abc.cif").resolve()
    assert store.path("3abc").read_text() == STRUCTURES["3abc"]
    # Downloads are kept per source, so two servers never share a file.
    downloaded = store.path("3abc").parent
    assert downloaded.parent == (tmp_path / "cache").resolve()
    assert "files-rcsb-org" in downloaded.name
    other = StructureStore(
        tmp_path / "cache",
        base_url="https://mirror.invalid/pdb",
        transport=_rcsb(calls),
    )
    assert other.path("3abc").parent != downloaded
    assert calls == [
        "https://files.rcsb.org/download/3ABC.cif",
        "https://mirror.invalid/pdb/3ABC.cif",
    ]
    with pytest.raises(SearchError, match="HTTP 404"):
        store.path("9zzz")
    offline = StructureStore(tmp_path / "other", base_url=None)
    with pytest.raises(SearchError, match="downloading is disabled"):
        offline.path("3abc")


def test_a_compressed_divided_mirror_is_read_and_unpacked_once(tmp_path):
    import gzip

    divided = tmp_path / "mirror" / "ab"
    divided.mkdir(parents=True)
    (divided / "2abc.cif.gz").write_bytes(gzip.compress(STRUCTURES["2abc"].encode()))
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "3ABC.cif.gz").write_bytes(gzip.compress(STRUCTURES["3abc"].encode()))

    def offline(*_):
        raise AssertionError("a mirrored structure was downloaded")

    store = StructureStore(
        tmp_path / "cache", local_dir=tmp_path / "mirror", transport=offline
    )
    unpacked = store.path("2abc")
    assert unpacked.suffix == ".cif" and unpacked.read_text() == STRUCTURES["2abc"]
    assert store.path("2abc") == unpacked
    other = StructureStore(tmp_path / "cache", local_dir=flat, transport=offline)
    assert other.path("3abc").read_text() == STRUCTURES["3abc"]


def test_template_structure_reads_author_label_and_observed_residues(tmp_path):
    path = tmp_path / "2abc.cif"
    path.write_text(STRUCTURES["2abc"])
    structure = template_search.read_template_structure(path)
    assert structure.release_date == date(2001, 2, 3)
    chain = structure.chain("X")
    assert chain is not None and chain.label_id == "A"
    assert chain.sequence == "GS" + QUERY[2:28] + "PP"
    assert 1 not in chain.observed and 3 in chain.observed
    # A label id is the fallback; an unknown id names nothing.
    assert structure.chain("A") is chain
    assert structure.chain("Z") is None


# -- per-model selection and delivery ----------------------------------------


@pytest.fixture
def kalign():
    return pytest.importorskip("kalign")


def test_alphafold3_keeps_its_cutoff_filters_and_zero_based_map(
    tmp_path, searched, kalign, monkeypatch
):
    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif",
        lambda source, chain_id: Path(source).read_text(),
    )
    path, records = _materialize(_job(tmp_path), "alphafold3")
    native = json.loads(path.read_text())
    templates = native["sequences"][0]["protein"]["templates"]
    (record,) = records
    kept = [item["pdb_id"] for item in record["templates"]]
    # 3abc is the query itself (max_subsequence_ratio), 4abc is after the
    # 2021-09-30 cutoff, 5abc is on it and kept (AlphaFold 3 drops only a
    # release *after* the date), 9zzz has no structure.
    assert kept == ["2abc", "5abc"]
    assert record["skipped"] == {"filter": 1, "release_date": 1, "no_structure": 1}
    assert record["cutoff"]["max_template_date"] == "2021-09-30"
    assert "run_alphafold.py" in record["cutoff"]["source"]
    first = templates[0]
    # Query residue 2 (0-based) is template residue 2 of the full chain
    # sequence, unresolved residues 0-1 included, as AlphaFold 3 counts.
    pairs = dict(zip(first["queryIndices"], first["templateIndices"], strict=True))
    assert pairs[2] == 2 and pairs[27] == 27
    assert min(first["queryIndices"]) >= 0
    assert max(first["queryIndices"]) < len(QUERY)
    written = json.loads((path.parent / "template_search.json").read_text())
    assert written == records


def test_template_max_date_overrides_the_released_cutoff(tmp_path, searched, kalign):
    _, records = _materialize(
        _job(tmp_path), "openfold3", template_max_date="2021-09-30"
    )
    (record,) = records
    # OpenFold3 drops a template released on or after its cutoff.
    assert [item["pdb_id"] for item in record["templates"]] == ["3abc", "2abc"]
    assert record["cutoff"] == {
        "max_template_date": "2021-09-30",
        "keeps_cutoff_date": False,
        "source": "template_max_date",
    }


def test_openfold3_gets_an_ordered_template_cache_its_reader_loads(
    tmp_path, searched, kalign
):
    path, records = _materialize(_job(tmp_path), "openfold3")
    (record,) = records
    # No released cutoff, no sequence filters: the first four in e-value order.
    assert record["cutoff"]["max_template_date"] is None
    assert [item["pdb_id"] for item in record["templates"]] == [
        "3abc",
        "4abc",
        "2abc",
        "5abc",
    ]
    query = json.loads(path.read_text())["queries"]["t"]
    (chain,) = query["chains"]
    cache = Path(chain["template_alignment_file_path"])
    assert chain["template_entry_chain_ids"] == ["3abc_A", "4abc_A", "2abc_A", "5abc_A"]

    from foldjax.models.openfold3.data._numpy_featurization import (
        _preprocessed_template_entries,
    )

    entries, _ = _preprocessed_template_entries(
        cache, chain["template_entry_chain_ids"]
    )
    assert list(entries) == chain["template_entry_chain_ids"]
    entry = entries["2abc_A"]
    # 1-based query position against label_seq_id, as upstream's idx_map.
    assert [3, 3] in entry.idx_map.tolist()
    assert entry.cif_path.name == "2abc.cif"


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_protenix_needs_use_template_and_gets_observed_single_chain_files(
    tmp_path, searched, kalign, model
):
    source = _job(tmp_path)
    with pytest.raises(ValueError, match="use_template=true"):
        _materialize(source, model)
    path, records = _materialize(source, model, options={"use_template": True})
    (record,) = records
    assert [item["pdb_id"] for item in record["templates"]] == ["2abc", "5abc"]
    (job,) = json.loads(path.read_text())
    sidecar = Path(job["sequences"][0]["proteinChain"]["templatesPath"])
    payload = json.loads(sidecar.read_text())
    first = payload[0]
    # 2abc's two unresolved N-terminal residues are not in the file, so its
    # observed residue 0 is chain residue 2, aligned to query residue 2.
    assert first["queryIndices"][0] == 2 and first["templateIndices"][0] == 0

    from foldjax.models.protenix.data.template_features import chain_template_dense

    aatype, positions, mask = chain_template_dense(sidecar, sequence=QUERY, skip=False)
    assert mask[0, 2].sum() > 0 and mask[0, 0].sum() == 0
    assert mask[1].sum() > 0


def test_boltz2_gets_files_and_label_chains_to_align_itself(tmp_path, searched):
    path, records = _materialize(_job(tmp_path), "boltz2")
    import yaml

    native = yaml.safe_load(path.read_text())
    templates = native["templates"]
    assert [Path(item["cif"]).stem for item in templates] == [
        "3abc",
        "4abc",
        "2abc",
        "5abc",
    ]
    # Boltz names a template chain by label_asym_id: 2abc's author X is A.
    assert templates[2]["template_id"] == ["A"]
    assert all(item["chain_id"] == ["Q"] for item in templates)
    assert "FoldJAX convenience" in records[0]["selection"]


def test_a_chain_with_its_own_templates_is_not_searched(tmp_path, searched):
    cif = tmp_path / "own.cif"
    cif.write_text(STRUCTURES["2abc"])
    source = _job(tmp_path, templates=[{"mmcif": "own.cif", "chain_id": "X"}])
    path, records = _materialize(source, "boltz2")
    assert records == [] and searched["server"] == []


def test_a_failed_search_warns_and_folds_without_templates(tmp_path, monkeypatch):
    def broken():
        raise ValueError("FOLDJAX_MSA_SERVER_URL is set to an empty value")

    monkeypatch.setattr(template_search, "_hits_pipeline", broken)
    with pytest.warns(UserWarning, match="template search for chain"):
        path, records = _materialize(_job(tmp_path), "boltz2")
    assert records[0]["error"].startswith("FOLDJAX_MSA_SERVER_URL")
    import yaml

    assert "templates" not in yaml.safe_load(path.read_text())


def _m8(*rows) -> str:
    """``.m8`` rows from ``(target, e-value)``; the other columns are inert."""
    return "\n".join(
        f"101\t{target}\t0.9\t20\t0\t0\t1\t20\t1\t20\t{e_value}\t100"
        for target, e_value in rows
    )


def _no_template_search_warning(caught) -> bool:
    return not any("template search" in str(item.message) for item in caught)


def test_missing_kalign_warns_before_anything_is_sent(tmp_path, searched, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "kalign" else real(name, *args, **kwargs)
        ),
    )
    with pytest.warns(UserWarning, match="Kalign"):
        path, records = _materialize(_job(tmp_path), "alphafold3")
    (record,) = records
    assert "kalign-python" in record["error"]
    # Checked up front: the sequence never left and nothing was downloaded.
    assert searched == {"server": [], "rcsb": []}
    native = json.loads(path.read_text())
    assert native["sequences"][0]["protein"]["templates"] == []


def test_every_download_failing_warns_and_records_why(tmp_path, monkeypatch):
    _route(tmp_path, monkeypatch, unreachable=("*",))
    with pytest.warns(UserWarning, match="none of 5 template hits was kept"):
        path, records = _materialize(_job(tmp_path), "boltz2")
    (record,) = records
    assert record["error"] == "none of 5 template hits was kept (no_structure 5)"
    assert record["templates"] == []
    import yaml

    assert "templates" not in yaml.safe_load(path.read_text())


def test_a_partial_download_failure_keeps_the_rest_quietly(
    tmp_path, monkeypatch, kalign
):
    _route(tmp_path, monkeypatch, unreachable=("2abc",))
    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif",
        lambda source, chain_id: Path(source).read_text(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, records = _materialize(_job(tmp_path), "alphafold3")
    (record,) = records
    assert [item["pdb_id"] for item in record["templates"]] == ["5abc"]
    assert record["skipped"]["no_structure"] == 2
    assert "error" not in record
    assert _no_template_search_warning(caught)


def test_a_generated_file_is_named_from_the_hit_not_the_downloaded_file(
    tmp_path, monkeypatch, kalign
):
    hostile = dict(STRUCTURES)
    hostile["2abc"] = STRUCTURES["2abc"].replace(
        "_entry.id 2ABC", "_entry.id '../../../ESCAPED'"
    )
    _route(tmp_path, monkeypatch, structures=hostile)
    path, records = _materialize(
        _job(tmp_path), "protenix", options={"use_template": True}
    )
    written = Path(records[0]["templates"][0]["observed_chain_file"])
    assert written == path.parent / "template_search" / "2abc_A.cif"
    assert written.is_file()
    # Where naming the file from `_entry.id` would have written it.
    assert not (tmp_path.parent / "escaped_A.cif").exists()
    assert not list(tmp_path.parent.glob("*scaped*"))
    assert not list(tmp_path.parent.glob("*SCAPED*"))


def test_a_planted_template_directory_symlink_is_refused(
    tmp_path, searched, kalign
):
    destination = tmp_path / "out"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "template_search").symlink_to(outside, target_is_directory=True)
    job = {"entities": [{"type": "protein", "id": "Q", "sequence": QUERY}]}
    with pytest.warns(UserWarning, match="symlink"):
        (record,) = template_search.search_templates(
            job, "protenix", max_date=None, destination=destination
        )
    assert "symlink" in record["error"]
    assert list(outside.iterdir()) == []
    assert "templates" not in job["entities"][0]


def test_alphafold3_skips_an_author_chain_spanning_two_polymers(
    tmp_path, monkeypatch, kalign
):
    structures = dict(STRUCTURES)
    structures["6abc"] = _mmcif(
        "6abc",
        "2001-01-01",
        [("A", "Z", QUERY[:30], set()), ("B", "Z", "GG" + QUERY[3:25], set())],
    )
    _route(
        tmp_path,
        monkeypatch,
        m8=_m8(("6abc_Z", 1e-30), ("2abc_X", 1e-20)),
        structures=structures,
    )
    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif",
        lambda source, chain_id: Path(source).read_text(),
    )
    _, records = _materialize(_job(tmp_path), "alphafold3")
    (record,) = records
    # AlphaFold 3 needs one polymer per template file; filtering 6abc to
    # author chain Z would leave two and fail the whole job.
    assert [item["pdb_id"] for item in record["templates"]] == ["2abc"]
    assert record["skipped"] == {"chain": 1}
    # OpenFold3 addresses label chains and keeps it.
    _, records = _materialize(_job(tmp_path), "openfold3")
    assert [item["pdb_id"] for item in records[0]["templates"]] == ["6abc", "2abc"]


def test_openfold3_features_place_searched_templates_on_their_query_residues(
    tmp_path, searched, kalign
):
    """The port's own template alignment, run on what the search wrote.

    An ``idx_map`` off by one residue would put every template residue on the
    wrong query position, which the residue types below would show.
    """
    pytest.importorskip("biotite")
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_RESIDUES_WITH_GAP_3,
    )
    from foldjax.models.openfold3.data.featurize import featurize_query

    path, records = _materialize(_job(tmp_path), "openfold3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = featurize_query(path)
    mask = np.asarray(features["template_pseudo_beta_mask"])[0]
    names = np.asarray(STANDARD_RESIDUES_WITH_GAP_3)[
        np.asarray(features["template_restype"])[0].argmax(-1)
    ]
    order = [item["pdb_id"] for item in records[0]["templates"]]
    expected = [_THREE[residue] for residue in QUERY]
    identical = order.index("3abc")
    assert mask[identical].all()
    assert names[identical].tolist() == expected
    slot = order.index("2abc")
    # 2abc's first two residues are unresolved; 2..27 copy the query.
    assert mask[slot][:2].sum() == 0
    assert mask[slot][2:28].all()
    assert names[slot][2:28].tolist() == expected[2:28]


def test_alphafold3_reads_searched_templates_where_the_map_points(
    tmp_path, searched, kalign
):
    """AlphaFold 3's own parser, on the chain filter FoldJAX runs for it."""
    from foldjax.models.alphafold3 import build

    if not build.is_ready():
        pytest.skip("the AlphaFold 3 runtime is not built for this checkout")
    path, records = _materialize(_job(tmp_path), "alphafold3")
    build.register_runtime()
    from alphafold3 import structure
    from alphafold3.common import folding_input

    fold = folding_input.Input.from_json(path.read_text(), json_path=path)
    (chain,) = fold.protein_chains
    assert len(chain.templates) == len(records[0]["templates"]) == 2
    for template in chain.templates:
        parsed = structure.from_mmcif(template.mmcif)
        (sequence,) = parsed.polymer_author_chain_single_letter_sequence().values()
        mapping = template.query_to_template_map
        same = sum(sequence[t] == QUERY[q] for q, t in mapping.items())
        # The same map read one residue off, as a 1-based slip would read it.
        shifted = sum(
            t + 1 < len(sequence) and sequence[t + 1] == QUERY[q]
            for q, t in mapping.items()
        )
        # Both templates copy at least 26 query residues; the global
        # alignment also pairs a few unequal terminal residues.
        assert same >= 26 and same > 3 * shifted


def test_esmfold2_and_unopted_protenix_refuse_at_plan_time(tmp_path):
    from foldjax.registry import get_backend

    source = _job(tmp_path)
    for model, option, message in (
        ("esmfold2", {}, "no structural-template input"),
        ("protenix", {}, "use_template=true"),
        ("opendde", {"use_template": False}, "use_template=true"),
    ):
        request = PredictionRequest(
            model=model,
            input=source,
            input_format="foldjax",
            templates="auto",
            options=option,
        )
        with pytest.raises(ValueError, match=message):
            get_backend(model).validate_request(request)
    # A native document is passed through untouched, so a search would not run.
    native = tmp_path / "native.yaml"
    native.write_text(
        "version: 1\nsequences:\n  - protein: {id: A, sequence: ACDEF, msa: empty}\n"
    )
    request = PredictionRequest(
        model="boltz2", input=native, input_format="native", templates="auto"
    )
    with pytest.raises(ValueError, match="FoldJAX-format jobs"):
        get_backend("boltz2").validate_request(request)


def test_request_validates_template_policy_and_date(tmp_path):
    job = _job(tmp_path)
    with pytest.raises(ValueError, match="templates must be one of"):
        PredictionRequest(model="boltz2", input=job, templates="yes")
    for spelling in ("30/09/2021", "2021-W39-4", "20210930"):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            PredictionRequest(
                model="boltz2",
                input=job,
                templates="auto",
                template_max_date=spelling,
            )
    with pytest.raises(ValueError, match="templates='auto'"):
        PredictionRequest(model="boltz2", input=job, template_max_date="2021-09-30")
    request = PredictionRequest(
        model="boltz2",
        input=job,
        templates="auto",
        template_max_date=" 2021-09-30 ",
    )
    assert request.template_max_date == "2021-09-30"


# -- the common schema --------------------------------------------------------


def _templated(tmp_path: Path, **template) -> Path:
    (tmp_path / "t.cif").write_text(STRUCTURES["2abc"])
    return _job(tmp_path, templates=[{"mmcif": "t.cif", **template}])


def _native(path: Path, model: str, **kwargs) -> Path:
    return materialize_native_input(
        path,
        capabilities(model),
        path.parent / f"native-{model}",
        seed=1,
        msa="single",
        **kwargs,
    )


def test_template_indices_are_zero_based_and_reach_alphafold3_verbatim(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "foldjax.input._filter_alphafold3_template_mmcif",
        lambda source, chain_id: Path(source).read_text(),
    )
    last = len(QUERY) - 1
    source = _templated(
        tmp_path, chain_id="X", query_indices=[0, 2, last], template_indices=[0, 2, 5]
    )
    native = json.loads(_native(source, "alphafold3").read_text())
    (template,) = native["sequences"][0]["protein"]["templates"]
    assert template["queryIndices"] == [0, 2, last]
    assert template["templateIndices"] == [0, 2, 5]
    # A 1-based map ends one past the sequence; it is refused, not shifted.
    one_based = _templated(
        tmp_path, query_indices=[1, len(QUERY)], template_indices=[1, 2]
    )
    with pytest.raises(ValueError, match="0-based"):
        _native(one_based, "alphafold3")
    negative = _templated(tmp_path, query_indices=[-1], template_indices=[0])
    with pytest.raises(ValueError, match="0-based"):
        _native(negative, "alphafold3")


def test_openfold3_translates_common_templates_in_both_forms(tmp_path):
    mapped = _templated(
        tmp_path, chain_id="X", query_indices=[2, 3], template_indices=[2, 3]
    )
    query = json.loads(_native(mapped, "openfold3").read_text())["queries"]["t"]
    (chain,) = query["chains"]
    assert chain["template_entry_chain_ids"] == ["2abc_A"]
    with np.load(chain["template_alignment_file_path"]) as archive:
        entries = json.loads(str(archive["entries_json"]))
    assert entries["2abc_A"]["idx_map"] == [[3, 3], [4, 4]]
    assert entries["2abc_A"]["release_date"] == "2001-02-03"

    bare = _templated(tmp_path, chain_id="X")
    query = json.loads(_native(bare, "openfold3").read_text())["queries"]["t"]
    (chain,) = query["chains"]
    assert chain["template_cif_paths"] == [str((tmp_path / "t.cif").resolve())]
    assert chain["template_cif_chain_ids"] == ["A"]
    assert "template_alignment_file_path" not in chain

    (tmp_path / "t.cif").write_text(STRUCTURES["2abc"])
    mixed = _job(
        tmp_path,
        templates=[
            {"mmcif": "t.cif", "query_indices": [2], "template_indices": [2]},
            {"mmcif": "t.cif"},
        ],
    )
    with pytest.raises(ValueError, match="mapped and unmapped"):
        _native(mixed, "openfold3")


def test_boltz2_warns_when_a_chain_id_names_two_chains(tmp_path):
    # Author B is label A, and label B is author C.
    (tmp_path / "t.cif").write_text(
        _mmcif(
            "8abc",
            "2001-01-01",
            [("A", "B", QUERY[:20], set()), ("B", "C", QUERY[5:25], set())],
        )
    )
    source = _job(tmp_path, templates=[{"mmcif": "t.cif", "chain_id": "B"}])
    with pytest.warns(UserWarning, match="also the label id of another chain"):
        path = _native(source, "boltz2")
    import yaml

    (template,) = yaml.safe_load(path.read_text())["templates"]
    assert template["template_id"] == ["A"]


def test_openfold3_no_longer_lists_templates_as_native_only():
    from foldjax.input import common_schema_features, native_only_features

    assert {"templates", "templates_unmapped"} <= set(
        common_schema_features("openfold3")
    )
    assert "templates" not in native_only_features(
        "openfold3", capabilities("openfold3")
    )


# -- the manifest --------------------------------------------------------------


def test_manifest_records_the_search_and_old_manifests_still_resume(tmp_path):
    from foldjax import manifest
    from foldjax.schema import PredictionResult, PredictionSample
    from tests.test_resume_manifest import _file, _request

    request = _request(tmp_path, templates="auto", template_max_date="2020-01-01")
    request.output_dir.mkdir()
    structure = _file(request.output_dir / "sample.cif", b"data_mock\n#\n")
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=7, structure_path=structure, scores={}),),
        output_dir=request.output_dir,
    )
    record = [{"chains": ["A"], "templates": []}]
    document = manifest.describe_run(
        request, result, directory=request.output_dir, template_search=record
    )
    assert document["templates"] == "auto"
    assert document["template_max_date"] == "2020-01-01"
    assert document["template_search"] == record
    assert manifest.matches_request(document, request, seed=7)
    plain = PredictionRequest(
        **{
            **{name: getattr(request, name) for name in request.__dataclass_fields__},
            "templates": "none",
            "template_max_date": None,
        }
    )
    assert not manifest.matches_request(document, plain, seed=7)
    # A manifest from before the field existed ran without a search.
    old = {
        key: value
        for key, value in manifest.describe_run(
            plain, result, directory=request.output_dir
        ).items()
        if key not in {"templates", "template_max_date", "template_search"}
    }
    assert manifest.matches_request(old, plain, seed=7)


# -- live ------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("FOLDJAX_LIVE_TEMPLATE_SEARCH") != "1",
    reason="sends a sequence to api.colabfold.com; set FOLDJAX_LIVE_TEMPLATE_SEARCH=1",
)
def test_live_ubiquitin_search_finds_a_released_template(tmp_path, monkeypatch):
    pytest.importorskip("kalign")
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    ubiquitin = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    source = tmp_path / "job.json"
    source.write_text(
        json.dumps(
            {
                "name": "ubq",
                "entities": [{"type": "protein", "id": "A", "sequence": ubiquitin}],
            }
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, records = _materialize(source, "openfold3")
    (record,) = records
    assert "error" not in record, record.get("error")
    assert record["hits"] > 0
    assert record["templates"], record
