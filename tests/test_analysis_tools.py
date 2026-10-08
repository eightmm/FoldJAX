"""The analysis and workflow commands: report, interfaces, compare --reference,
jobs expand/pulldown, show --screen/--interfaces, --structure-format, --shard,
plan --json, check, and results_table(as_frame=True).

Outputs come from `tests/_w5_fixtures.py`: a synthetic two-chain complex run
through the real prediction pipeline by a replay backend, so no GPU or weights
are needed. Expected interface scores were produced by the reference
``ipsae.py`` v4 (DunbrackLab/IPSAE) on the same files; expected TM-scores by
US-align ``-TMscore 1``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from foldjax import (
    accuracy,
    cli,
    confidence_arrays,
    interfaces,
    job_generators,
    slurm,
    structure_format,
)
from foldjax.registry import backend_override
from foldjax.results import _columns, load_results, results_table
from tests._w5_fixtures import (
    SEQUENCE_A,
    complex_coordinates,
    replay_backend,
    run_replay,
    token_maps,
    write_cif,
    write_job,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))


def _batch(tmp_path: Path, **kwargs) -> Path:
    root = tmp_path / "out"
    model = kwargs.pop("model", "boltz2")
    run_replay(
        root / model / "pair", model=model, scratch=tmp_path / "scratch", **kwargs
    )
    return root


# ---------------------------------------------------------------- interfaces


def test_ipsae_matches_the_reference_script(tmp_path: Path) -> None:
    root = _batch(tmp_path, ligand=True)
    result = interfaces.sample_interfaces(root / "boltz2" / "pair" / "seed-7_sample-01")
    assert result["skipped"] is None
    derived = result["derived"]
    assert derived["plddt_source"] == "token_plddt"
    # The six ligand tokens are dropped: 30 + 24 residues.
    assert derived["residues"] == 54
    rows = {(p["chain1"], p["chain2"], p["direction"]): p for p in derived["pairs"]}
    # ipsae.py v4 output on these files (pae_cutoff 10, dist_cutoff 10).
    shared = (0.013944, 0.075441, 0.053945)
    tail = (0.1485, 0.0391, 0.1581)
    expected = {
        ("A", "B", "asym"): (*shared, 0.073596, *tail),
        ("B", "A", "asym"): (*shared, 0.068562, *tail),
        ("A", "B", "max"): (*shared, 0.073596, *tail),
    }
    keys = (
        "ipsae",
        "ipsae_d0chn",
        "ipsae_d0dom",
        "iptm_d0chn",
        "pdockq",
        "pdockq2",
        "lis",
    )
    for pair, values in expected.items():
        for key, value in zip(keys, values, strict=True):
            assert rows[pair][key] == pytest.approx(value, abs=6e-5), (pair, key)
    for key, value in (("n0res", 22), ("n0chn", 54), ("n0dom", 44)):
        assert rows[("A", "B", "max")][key] == value
    assert rows[("A", "B", "asym")]["dist1"] == 3


def test_d0_floors_differ_as_in_the_reference() -> None:
    # Scalar d0 (d0chn/d0dom) is 1.0 up to L=27; the array form floors L at 26.
    assert interfaces.calc_d0(27, "protein") == 1.0
    assert interfaces.calc_d0_array([27], "protein")[0] == pytest.approx(
        1.24 * 12 ** (1 / 3) - 1.8
    )
    assert interfaces.calc_d0(10, "nucleic_acid") == 2.0


def test_pdockq_reads_plddt_at_the_cb_atom(tmp_path: Path) -> None:
    chains = complex_coordinates()
    structure = write_cif(tmp_path / "s" / "x.cif", chains)
    n_atoms = sum(len(atoms) for residues in chains.values() for _, atoms in residues)
    names = [
        name for residues in chains.values() for _, atoms in residues for name in atoms
    ]
    atom_plddt = np.where(np.asarray(names) == "CB", 50.0, 90.0)
    # Glycine has no CB: its CA stands in, at 90.
    maps = token_maps(chains)
    confidence_arrays.write(
        tmp_path / "s" / confidence_arrays.FILENAME,
        model="alphafold3",
        arrays={
            **maps,
            "pae": np.full((len(maps["token_chain_id"]),) * 2, 3.0),
            "atom_plddt": atom_plddt,
        },
        scales={"atom_plddt": "0-100"},
    )
    assert len(atom_plddt) == n_atoms
    arrays = confidence_arrays.load_confidence_arrays(tmp_path / "s")
    residues = interfaces.residues_for(structure, arrays)
    assert set(np.unique(residues.plddt)) <= {50.0, 90.0}
    assert (residues.plddt == 50.0).sum() == sum(
        name != "GLY" for c in chains.values() for name, _ in c
    )


def test_atom_token_index_places_pae_like_the_residue_maps(tmp_path: Path) -> None:
    # Protenix, OpenDDE and OpenFold3 write atom_token_index; Boltz-2 and
    # AlphaFold 3 only the token maps. Both routes must pick the same tokens.
    chains = complex_coordinates(ligand=True)
    maps = token_maps(chains)
    owners, token = [], 0
    for residues in chains.values():
        for name, atoms in residues:
            if name == "BNZ":
                owners += list(range(token, token + len(atoms)))
                token += len(atoms)
            else:
                owners += [token] * len(atoms)
                token += 1
    n = len(maps["token_chain_id"])
    rng = np.random.default_rng(0)
    pae = rng.uniform(1.0, 20.0, size=(n, n))
    results = []
    for label, extra in (("maps", {}), ("atoms", {"atom_token_index": owners})):
        directory = tmp_path / label
        structure = write_cif(directory / "x.cif", chains)
        confidence_arrays.write(
            directory / confidence_arrays.FILENAME,
            model="protenix",
            arrays={**maps, **extra, "pae": pae, "token_plddt": np.full(n, 80.0)},
            scales={"token_plddt": "0-100"},
        )
        results.append(interfaces.sample_interfaces(directory, structure))
    assert results[0]["derived"]["pairs"] == results[1]["derived"]["pairs"]


def test_a_sample_without_pae_is_skipped_with_the_reason(tmp_path: Path) -> None:
    root = _batch(tmp_path, with_pae=False)
    rows = interfaces.interface_rows(root)
    assert rows and all("no PAE" in row["skipped"] for row in rows)
    assert all("not returned by this replay" in row["skipped"] for row in rows)


def test_show_interfaces_keeps_native_and_derived_apart(tmp_path: Path, capsys) -> None:
    root = _batch(tmp_path)
    assert cli.main(["show", str(root), "--format", "csv", "--interfaces"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 2
    for column in ("derived.ipsae.A-B", "derived.pdockq.A-B", "derived.lis.A-B"):
        assert rows[0][column] != ""
    # Boltz-2's managed run returns no chain-pair ipTM: native stays empty.
    assert rows[0]["native.chain_pair_iptm.A-B"] == ""
    assert "score.ipsae" not in rows[0]


def test_interfaces_command_formats(tmp_path: Path, capsys) -> None:
    root = _batch(tmp_path)
    assert cli.main(["interfaces", str(root), "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == interfaces.INTERFACES_SCHEMA
    assert "Dunbrack" in document["method"]["citations"]["ipsae"]
    assert cli.main(["interfaces", str(root)]) == 0
    assert "within-model only" in capsys.readouterr().out


# -------------------------------------------------------------------- report


def test_report_is_self_contained_and_copes_without_pae(tmp_path: Path) -> None:
    root = tmp_path / "out"
    run_replay(root / "boltz2" / "pair", scratch=tmp_path / "s1")
    run_replay(
        root / "alphafold3" / "pair",
        model="alphafold3",
        with_pae=False,
        scratch=tmp_path / "s2",
    )
    assert cli.main(["report", str(root)]) == 0
    page = (root / "foldjax_report.html").read_text()
    assert "<script" not in page and "http://" not in page and "https://" not in page
    assert page.count('class="card"') == 2
    assert "data:image/png;base64," in page  # boltz2's PAE
    assert "not returned by this replay" in page  # alphafold3's absent PAE
    assert page.count('aria-label="per-residue pLDDT"') == 2
    assert "confidence_score" in page and "seeds" in page


# ------------------------------------------------------------------ accuracy


def _homodimer(path: Path, *, swap: bool, shift: float = 0.0) -> Path:
    chains = complex_coordinates()
    a, b = chains["A"], [(name, atoms) for name, atoms in chains["A"]]
    moved = [
        (name, {k: v + np.array([9.0, 0.0, 3.0]) for k, v in atoms.items()})
        for name, atoms in b
    ]
    first, second = (moved, a) if swap else (a, moved)

    def nudge(chain, residue, atom, xyz):
        return xyz + (np.array([shift, 0, 0]) if chain == "B" and residue > 20 else 0)

    return write_cif(path, {"A": first, "B": second}, transform=nudge)


def test_homomer_chains_are_permuted_to_the_reference(tmp_path: Path) -> None:
    predicted = _homodimer(tmp_path / "p.cif", swap=False)
    reference = _homodimer(tmp_path / "r.cif", swap=True)
    scored = accuracy.score_structure(predicted, reference)
    assert scored["chain_map"] == {"A": "B", "B": "A"}
    assert scored["rmsd_ca"] == pytest.approx(0.0, abs=1e-6)
    assert scored["lddt"] == pytest.approx(1.0)
    assert scored["lddt_ca"] == pytest.approx(1.0)
    assert scored["tm"] == pytest.approx(1.0)
    perturbed = accuracy.score_structure(
        _homodimer(tmp_path / "q.cif", swap=False, shift=3.0), reference
    )
    assert perturbed["lddt_ca"] < 0.95 and perturbed["tm"] < 1.0


@pytest.mark.parametrize(
    ("n", "bend", "usalign"), [(40, 1.5, 0.5712), (120, 2.5, 0.6799), (18, 0.8, 0.4776)]
)
def test_tm_score_matches_usalign(n: int, bend: float, usalign: float) -> None:
    i = np.arange(n)
    target = np.stack(
        [2.3 * np.cos(np.radians(100 * i)), 2.3 * np.sin(np.radians(100 * i)), 1.5 * i],
        1,
    )
    mobile = target + np.stack(
        [bend * np.sin(i / 3.0), bend * np.cos(i / 5.0), 0 * i], 1
    )
    mobile[n * 3 // 4 :] += np.array([4.0, -3.0, 2.0])
    value = accuracy.tm_score(np.round(mobile, 3), np.round(target, 3))
    assert value == pytest.approx(usalign, abs=6e-5)


def test_ligand_rmsd_resolves_ring_symmetry_and_measures_a_shift(
    tmp_path: Path,
) -> None:
    chains = complex_coordinates(ligand=True)
    reference = write_cif(tmp_path / "r.cif", chains)
    ring = chains["L"][0][1]
    names = list(ring)
    # Rename around the ring: a name-for-name RMSD would be 1.39 A.
    rotated = {names[(k + 1) % 6]: ring[names[k]] for k in range(6)}
    predicted = write_cif(tmp_path / "p.cif", {**chains, "L": [("BNZ", rotated)]})
    scored = accuracy.score_structure(predicted, reference, metrics=("lig_rmsd",))
    assert scored["lig_rmsd"] == pytest.approx(0.0, abs=1e-6)
    shifted = {k: v + np.array([0.0, 0.0, 1.5]) for k, v in ring.items()}
    moved = write_cif(tmp_path / "m.cif", {**chains, "L": [("BNZ", shifted)]})
    assert accuracy.score_structure(moved, reference, metrics=("lig_rmsd",))[
        "lig_rmsd"
    ] == pytest.approx(1.5, abs=1e-6)


def test_a_smiles_ligand_matches_a_ccd_reference_by_graph(tmp_path: Path) -> None:
    # A SMILES ligand comes back under another residue name and non-CCD atom
    # names: the composition finds the copy, the bond graph the atom map.
    chains = complex_coordinates(ligand=True)
    reference = write_cif(tmp_path / "r.cif", chains)
    ring = list(chains["L"][0][1].values())
    renamed = {f"CX{k + 1}": ring[(k + 2) % 6] for k in range(6)}
    predicted = write_cif(tmp_path / "p.cif", {**chains, "L": [("UNL", renamed)]})
    scored = accuracy.score_structure(predicted, reference, metrics=("lig_rmsd",))
    (ligand,) = scored["ligands"]
    assert ligand["atom_map"] == "graph" and ligand["reference"].endswith("BNZ1")
    assert scored["lig_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_shard_passes_structure_inputs_through(tmp_path: Path) -> None:
    deposited = Path("structure:/data/8xyz.cif")
    selected, summary = slurm.shard_inputs([deposited], 0, 1)
    assert selected == [deposited] and summary["units_total"] == 1


def test_dockq_without_the_tool_names_how_to_get_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("FOLDJAX_DOCKQ", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setitem(sys.modules, "DockQ", None)
    predicted = _homodimer(tmp_path / "p.cif", swap=False)
    scored = accuracy.score_structure(predicted, predicted, metrics=("dockq", "tm"))
    assert "uv tool install" in scored["errors"]["dockq"]
    assert scored["tm"] == pytest.approx(1.0)


def test_compare_reference_adds_accuracy_rows_and_columns(
    tmp_path: Path, capsys
) -> None:
    root = _batch(tmp_path)
    reference = write_cif(tmp_path / "ref.cif", complex_coordinates())
    out = tmp_path / "cmp"
    assert (
        cli.main(
            [
                "compare",
                str(root),
                "--reference",
                str(reference),
                "--metrics",
                "lddt,tm,rmsd_ca",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    pairs = list(csv.DictReader((out / "compare.csv").open()))
    scored = [row for row in pairs if row["reference"] == "reference:ref"]
    assert len(scored) == 2
    assert all(float(row["tm"]) == pytest.approx(1.0) for row in scored)
    structures = list(csv.DictReader((out / "compare_structures.csv").open()))
    assert {"lddt", "tm", "accuracy_chain_map"} <= set(structures[0])
    document = json.loads((out / "compare.json").read_text())
    assert document["accuracy"]["definitions"]["comparable_across_models"] is True
    with pytest.raises(ValueError, match="give --reference"):
        cli.main(["compare", str(root), "--metrics", "tm"])


# ---------------------------------------------------------------- generators


def test_jobs_expand_names_are_stable_and_alignments_reused(tmp_path: Path) -> None:
    target = write_job(tmp_path / "t", "kin")
    library = tmp_path / "lib.smi"
    library.write_text("CCO ethanol\nc1ccccc1\nCC(=O)O acetic acid\n")
    document, summary = job_generators.expand_ligands(target, library, affinity=True)
    names = [job["name"] for job in document["jobs"]]
    assert names[0] == "kin__ethanol" and names[2] == "kin__acetic_acid"
    assert names[1].startswith("kin__lig-")
    first = document["jobs"][0]
    assert Path(first["entities"][0]["unpaired_msa"]).is_absolute()
    assert first["entities"][-1] == {"type": "ligand", "id": "C", "smiles": "CCO"}
    assert first["properties"] == [{"affinity": {"binder": "C"}}]
    library.write_text("CC(=O)O acetic acid\nc1ccccc1\nCCO ethanol\nCCN\n")
    reordered, _ = job_generators.expand_ligands(target, library)
    assert {job["name"] for job in reordered["jobs"]} >= set(names)
    assert summary["jobs"] == 3


def test_jobs_expand_reads_sdf_and_refuses_bad_records(tmp_path: Path) -> None:
    from rdkit import Chem

    target = write_job(tmp_path / "t", "kin")
    writer = Chem.SDWriter(str(tmp_path / "lib.sdf"))
    for smiles, title in (("CCO", "eth"), ("c1ccccc1", "")):
        molecule = Chem.MolFromSmiles(smiles)
        molecule.SetProp("_Name", title)
        writer.write(molecule)
    writer.close()
    document, _ = job_generators.expand_ligands(target, tmp_path / "lib.sdf")
    assert document["jobs"][0]["name"] == "kin__eth"
    (tmp_path / "bad.smi").write_text("CCO a\nnot_smiles b\n")
    with pytest.raises(ValueError, match="--skip-invalid"):
        job_generators.expand_ligands(target, tmp_path / "bad.smi")
    kept, summary = job_generators.expand_ligands(
        target, tmp_path / "bad.smi", skip_invalid=True
    )
    assert len(kept["jobs"]) == 1 and len(summary["skipped"]) == 1


def test_pulldown_pairs(tmp_path: Path) -> None:
    (tmp_path / "baits.fasta").write_text(">b1 bait\nMKTAYIAK\n>b2\nGSHMAELK\n")
    (tmp_path / "cands.fasta").write_text(">c1\nMKKLLV\n")
    (tmp_path / "msa").mkdir()
    (tmp_path / "msa" / "b1.a3m").write_text(">q\nMKTAYIAK\n")
    document, summary = job_generators.pulldown(
        tmp_path / "baits.fasta", tmp_path / "cands.fasta", msa_dir=tmp_path / "msa"
    )
    assert [job["name"] for job in document["jobs"]] == ["b1__c1", "b2__c1"]
    assert document["jobs"][0]["entities"][0]["unpaired_msa"].endswith("b1.a3m")
    assert summary["missing_msas"] == ["b2", "c1"]
    every, summary = job_generators.pulldown(
        tmp_path / "baits.fasta", tmp_path / "cands.fasta", all_vs_all=True
    )
    assert summary["jobs"] == 3
    assert (
        cli.main(
            [
                "jobs",
                "pulldown",
                "--baits",
                str(tmp_path / "baits.fasta"),
                "--all-vs-all",
                "--out",
                str(tmp_path / "p.json"),
            ]
        )
        == 0
    )
    assert len(json.loads((tmp_path / "p.json").read_text())["jobs"]) == 1


def test_screen_table_ranks_within_each_model(tmp_path: Path, capsys) -> None:
    root = tmp_path / "out"
    run_replay(root / "boltz2" / "pair", scratch=tmp_path / "s1")
    run_replay(root / "protenix" / "pair", model="protenix", scratch=tmp_path / "s2")
    table = job_generators.screen_table(results_table(root))
    assert {row["model"] for row in table} == {"boltz2", "protenix"}
    assert all(row["rank_within_model"] == 1 for row in table)
    boltz = next(row for row in table if row["model"] == "boltz2")
    # The model's own best sample (sample 0) supplies the affinity.
    assert boltz["score.affinity_pred_value"] == 1.5 and boltz["sample"] == 0
    assert cli.main(["show", str(root), "--screen"]) == 0
    assert "within each model only" in capsys.readouterr().out


# ------------------------------------------------------- structure format


def test_pdb_copy_and_its_limits(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "x.cif", complex_coordinates(ligand=True))
    pdb = structure_format.convert(cif)
    assert pdb.suffix == ".pdb" and "HETATM" in pdb.read_text()
    chains = complex_coordinates()
    wide = write_cif(tmp_path / "w.cif", {"AB": chains["A"], "C": chains["B"]})
    with pytest.raises(structure_format.StructureFormatError, match="chain ids longer"):
        structure_format.convert(wide)
    problems = structure_format.preflight(
        [
            {
                "name": "j",
                "entities": [
                    {"type": "protein", "id": ["A", "BB"]},
                    {"type": "ligand", "id": "L", "ccd": "A1BIQ"},
                ],
            }
        ]
    )
    assert len(problems) == 2


def _predict(
    tmp_path: Path, extra: list[str], *, inputs: list[Path], model: str = "boltz2"
):
    weights = tmp_path / "w.jax"
    weights.write_bytes(b"replayed")
    argv = [
        "predict",
        "--model",
        model,
        "--input",
        *map(str, inputs),
        "--weights",
        str(weights),
        "--seed",
        "7",
        "--no-cache",
        "--quiet",
        "--json",
        *extra,
    ]
    with backend_override(model, replay_backend(model)):
        return cli.main(argv)


def test_structure_format_both_writes_pdb_beside_cif(tmp_path: Path, capsys) -> None:
    job = write_job(tmp_path / "jobs", "pair")
    out = tmp_path / "out"
    assert (
        _predict(
            tmp_path,
            ["--output-dir", str(out), "--structure-format", "both"],
            inputs=[job],
        )
        == 0
    )
    capsys.readouterr()
    sample = out / "seed-7_sample-00"
    assert (sample / "pair_seed-7_sample-00.pdb").is_file()
    # The mmCIF stays the record the manifest verifies.
    (row, _) = results_table(out)
    assert row["structure_path"].endswith(".cif") and row["structure_verified"]


def test_structure_format_pdb_refuses_before_running(tmp_path: Path) -> None:
    job = tmp_path / "wide.json"
    job.write_text(
        json.dumps(
            {
                "name": "wide",
                "entities": [{"type": "protein", "id": "AB", "sequence": SEQUENCE_A}],
            }
        )
    )
    fasta = tmp_path / "wide.fasta"
    fasta.write_text(f">AB\n{SEQUENCE_A}\n")
    with pytest.raises(ValueError, match="'AB' is longer"):
        _predict(
            tmp_path,
            ["--structure-format", "pdb", "--output-dir", str(tmp_path / "f")],
            inputs=[fasta],
        )
    with pytest.raises(ValueError, match="cannot hold this job"):
        _predict(
            tmp_path,
            ["--structure-format", "pdb", "--output-dir", str(tmp_path / "o")],
            inputs=[job],
        )


# --------------------------------------------------------------------- shard


def test_parse_shard() -> None:
    assert slurm.parse_shard("1/4") == (1, 4)
    env = {
        "SLURM_ARRAY_TASK_ID": "3",
        "SLURM_ARRAY_TASK_MIN": "1",
        "SLURM_ARRAY_TASK_COUNT": "5",
    }
    assert slurm.parse_shard("auto", env) == (2, 5)
    assert slurm.parse_shard("auto/8", env) == (2, 8)
    for bad in ("4/4", "x/2", "3", "auto"):
        with pytest.raises(ValueError):
            slurm.parse_shard(bad, {} if bad == "auto" else env)


def test_shards_cover_the_batch_with_identical_job_documents(tmp_path: Path) -> None:
    from foldjax.input import expand_jobs_file

    target = write_job(tmp_path / "t", "kin")
    library = tmp_path / "lib.smi"
    library.write_text("CCO a\nCCN b\nCCC c\n")
    document, _ = job_generators.expand_ligands(target, library)
    # Relative alignment paths, as a hand-written jobs file would have them.
    for job in document["jobs"]:
        job["entities"][0]["unpaired_msa"] = "t/a.a3m"
    jobs_file = job_generators.write_jobs(document, tmp_path / "jobs.json")
    plain = write_job(tmp_path / "p", "solo")
    whole = {path.name: path for path, _ in expand_jobs_file(jobs_file)}
    seen = []
    for index in range(2):
        selected, summary = slurm.shard_inputs([jobs_file, plain], index, 2)
        assert summary["units_total"] == 4
        for path in selected:
            if path == plain:
                seen.append("solo")
                continue
            for generated, _ in expand_jobs_file(path):
                assert generated == whole[generated.name]
                seen.append(generated.stem)
    assert sorted(seen) == sorted(["kin__a", "kin__b", "kin__c", "solo"])


def test_a_one_input_shard_keeps_the_batch_layout(tmp_path: Path, capsys) -> None:
    jobs = [write_job(tmp_path / "jobs", name) for name in ("one", "two")]
    out = tmp_path / "out"
    assert (
        _predict(tmp_path, ["--output-dir", str(out), "--shard", "1/2"], inputs=jobs)
        == 0
    )
    capsys.readouterr()
    assert (out / "boltz2" / "two" / "foldjax_run.json").is_file()
    assert not (out / "boltz2" / "one").exists()


# ---------------------------------------------------------------------- plan


def test_plan_resources_from_the_law() -> None:
    from foldjax import memory_policy

    document = {
        "entities": [
            {"type": "protein", "id": ["A", "B"], "sequence": "M" * 600},
            {"type": "ligand", "id": "L", "smiles": "c1ccccc1"},
        ]
    }
    record = slurm.plan_resources("boltz2", document)
    assert record["tokens_estimate"] == 1206 and record["state"] == "estimated"
    upper = memory_policy.BOLTZ2_PEAK.upper(1206)
    assert record["min_device_memory_gib"] == math.ceil(
        upper / memory_policy.ADMISSION_FRACTION / 2**30
    )
    assert (
        record["gres"] == "gpu:1"
        and record["mem"] is None
        and "host" in record["mem_reason"]
    )
    assert slurm.plan_resources("opendde", document)["state"] == "unknown"
    assert "MSA row" in slurm.plan_resources("protenix", document)["reason"]
    small = slurm.plan_resources(
        "boltz2", {"entities": [{"type": "protein", "id": "A", "sequence": "MK"}]}
    )
    assert small["state"] == "unknown" and "does not extrapolate" in small["reason"]


def test_plan_resources_does_not_size_a_card_outside_the_fitted_composition() -> None:
    """OpenFold3's law is protein-only; its admission calls a nucleic-acid or
    ligand run unknown (5NPK peaked 11 GiB over the upper estimate), and the
    plan must not suggest a card from that estimate either.
    """
    from foldjax import memory_policy

    protein = {"type": "protein", "id": ["A", "B"], "sequence": "M" * 600}
    record = slurm.plan_resources("openfold3", {"entities": [protein]})
    assert record["state"] == "estimated" and record["min_device_memory_gib"]

    for other, label in (
        ({"type": "ligand", "id": "L", "smiles": "c1ccccc1"}, "6 nucleic-acid"),
        ({"type": "dna", "id": "D", "sequence": "ACGT"}, "4 nucleic-acid"),
        ({"type": "ligand", "id": "L", "ccd": "ATP"}, "at least 1 nucleic-acid"),
    ):
        record = slurm.plan_resources("openfold3", {"entities": [protein, other]})
        assert record["state"] == "unknown"
        assert record["min_device_memory_gib"] is None
        assert record["reason"].startswith(label)
        assert "protein-only" in record["reason"]
        # The estimate itself stays, as a lower bound.
        assert record["upper_gib"] == round(
            memory_policy.OPENFOLD3_CHUNKED_PEAK.upper(record["tokens_estimate"])
            / 2**30,
            2,
        )
    # Only a law whose admission checks the composition is affected.
    mixed = {"entities": [protein, {"type": "ligand", "id": "L", "smiles": "CCO"}]}
    assert slurm.plan_resources("boltz2", mixed)["state"] == "estimated"


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({}, None),
        ({"confidence_dtype": "bfloat16"}, None),
        ({"confidence_dtype": "float32"}, "a float32 confidence head"),
        ({"confidence_dtype": "fp32"}, "a float32 confidence head"),
        ({"dtype": "bf16", "confidence_dtype": "f32"}, "a float32 confidence head"),
        ({"dtype": "float32"}, "a float32 trunk"),
        ({"compute_dtype": "fp32"}, "a float32 trunk"),
        ({"dtype": "float32", "confidence_dtype": "bfloat16"}, "a float32 trunk"),
    ],
)
def test_plan_resources_does_not_size_a_card_for_a_float32_openfold3_region(
    options, reason
) -> None:
    """OpenFold3's law was fitted at the bfloat16 trunk and head. A float32
    head behind that trunk measured +11.6 GiB at 3,012 tokens and ran out of
    memory at 4,888 where the law admitted it, so admission records either
    float32 region as unknown, and the plan must say the same.
    """
    protein = {"type": "protein", "id": ["A", "B"], "sequence": "M" * 600}
    record = slurm.plan_resources("openfold3", {"entities": [protein]}, options=options)
    if reason is None:
        assert record["state"] == "estimated" and record["min_device_memory_gib"]
        return
    assert record["state"] == "unknown"
    assert record["min_device_memory_gib"] is None
    assert record["reason"].startswith(f"{reason}: the estimate is a lower bound")
    assert record["upper_gib"]
    assert record["exceeds_profile"] == [reason]
    # Only OpenFold3 has a confidence head knob; another port's law is
    # untouched by it. (A float32 trunk is Boltz-2's float32 pair stream too.)
    if "dtype" not in options and "compute_dtype" not in options:
        assert (
            slurm.plan_resources("boltz2", {"entities": [protein]}, options=options)[
                "state"
            ]
            == "estimated"
        )


_PADDING_PROTEIN = {"type": "protein", "id": ["A", "B"], "sequence": "M" * 1050}


@pytest.mark.parametrize("model", ["boltz2", "openfold3", "esmfold2"])
def test_plan_resources_does_not_size_a_card_for_a_padded_run(model: str) -> None:
    """Every port's law was fitted unpadded, and its admission records a padded
    run's "fits" as unknown (measured, padded runs exceeded the upper estimate
    by up to 1.69x), so the plan must not size a card for one either.
    """
    document = {"entities": [_PADDING_PROTEIN]}
    unpadded = slurm.plan_resources(model, document)
    assert unpadded["state"] == "estimated" and unpadded["min_device_memory_gib"]
    assert "exceeds_profile" not in unpadded
    record = slurm.plan_resources(model, document, padding=True)
    assert record["state"] == "unknown"
    assert record["min_device_memory_gib"] is None
    assert record["exceeds_profile"] == ["serving padding"]
    assert record["reason"].startswith(
        "serving padding: the estimate is a lower bound"
    )
    # The estimate itself stays, as a lower bound.
    assert record["upper_gib"] == unpadded["upper_gib"]


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_plan_resources_lists_padding_where_the_state_is_already_unknown(
    model: str,
) -> None:
    document = {"entities": [_PADDING_PROTEIN]}
    assert "exceeds_profile" not in slurm.plan_resources(model, document)
    record = slurm.plan_resources(model, document, padding=True)
    assert record["state"] == "unknown" and record["min_device_memory_gib"] is None
    assert record["exceeds_profile"] == ["serving padding"]
    # AlphaFold 3 has no law, so its admission has no profile to exceed.
    assert "exceeds_profile" not in slurm.plan_resources(
        "alphafold3", document, padding=True
    )


@pytest.mark.parametrize(
    ("options", "reasons"),
    [
        ({}, []),
        ({"matmul_precision": "high"}, []),
        ({"compute_dtype": "bfloat16", "pair_residual_dtype": "auto"}, []),
        ({"pair_residual_dtype": "float32"}, ["a float32 pair residual stream"]),
        ({"pair_residual_dtype": "fp32"}, ["a float32 pair residual stream"]),
        ({"dtype": "float32"}, ["a float32 pair residual stream"]),
        ({"compute_dtype": "fp32"}, ["a float32 pair residual stream"]),
        ({"matmul_precision": "highest"}, ["matmul_precision=highest"]),
        (
            {"compute_dtype": "float32", "matmul_precision": "highest"},
            ["a float32 pair residual stream", "matmul_precision=highest"],
        ),
    ],
)
def test_plan_resources_reads_boltz2_precision_options(
    options: dict[str, str], reasons: list[str]
) -> None:
    """Boltz-2's law was fitted at the bfloat16 pair stream and `high`
    matmuls; its admission records a float32 stream (spelled, or "auto" under
    a float32 compute dtype) or `matmul_precision=highest` as unknown.
    """
    document = {"entities": [_PADDING_PROTEIN]}
    record = slurm.plan_resources("boltz2", document, options=options)
    if not reasons:
        assert record["state"] == "estimated" and "exceeds_profile" not in record
        return
    assert record["state"] == "unknown" and record["min_device_memory_gib"] is None
    assert record["exceeds_profile"] == reasons
    # Admission's order: serving padding first.
    padded = slurm.plan_resources("boltz2", document, options=options, padding=True)
    assert padded["exceeds_profile"] == ["serving padding", *reasons]
    assert padded["reason"].startswith("; ".join(["serving padding", *reasons]))


def test_plan_resources_calls_openfold3_pocket_guided_sampling_unknown() -> None:
    """OpenFold3 samples pocket-guided for every query with a pocket
    constraint -- a second rollout plus the proposal search -- which its
    admission records as unknown.
    """
    from foldjax import memory_policy

    ligand = {"type": "ligand", "id": "L", "smiles": "c1ccccc1"}
    document = {
        "entities": [_PADDING_PROTEIN, ligand],
        "constraints": [{"pocket": {"binder": "L", "contacts": [["A", 5]]}}],
    }
    record = slurm.plan_resources(
        "openfold3", document, options={"dtype": "float32"}, padding=True
    )
    assert record["state"] == "unknown" and record["min_device_memory_gib"] is None
    # `released_config`'s order and words.
    assert record["exceeds_profile"] == [
        "serving padding",
        "a float32 trunk",
        "pocket-guided sampling",
        memory_policy.non_protein_reason(6),
    ]
    # A pocket is OpenFold3's sampler; Boltz-2's law does not read it.
    assert "exceeds_profile" not in slurm.plan_resources("boltz2", document)


def test_plan_resources_stays_jax_free() -> None:
    """`foldjax plan` must not import JAX, including the Boltz-2 and OpenFold3
    adapter tables the profile check reads."""
    import subprocess

    script = r"""
import sys
from foldjax import slurm

protein = {"type": "protein", "id": ["A", "B"], "sequence": "M" * 1050}
pocket = {"entities": [protein, {"type": "ligand", "id": "L", "smiles": "CCO"}],
          "constraints": [{"pocket": {"binder": "L", "contacts": [["A", 5]]}}]}
for model, options in (
    ("boltz2", {"dtype": "fp32", "matmul_precision": "highest"}),
    ("openfold3", {"confidence_dtype": "fp32"}),
):
    assert slurm.plan_resources(model, pocket, options=options, padding=True)[
        "exceeds_profile"
    ]
assert "jax" not in sys.modules, "plan imported jax"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout


def test_plan_json_adds_the_slurm_block(tmp_path: Path, capsys) -> None:
    job = write_job(tmp_path / "jobs", "pair")
    weights = tmp_path / "w.jax"
    weights.write_bytes(b"x")
    assert (
        cli.main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(job),
                "--weights",
                str(weights),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["slurm"]["gres"] == "gpu:1"
    assert payload["slurm"]["tokens_estimate"] == 54


@pytest.mark.parametrize(
    ("option", "state"),
    [(None, "estimated"), ("bfloat16", "estimated"), ("fp32", "unknown")],
)
def test_plan_json_reads_openfold3_confidence_dtype(
    tmp_path: Path, capsys, option: str | None, state: str
) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    argv = [
        "plan",
        "--model",
        "openfold3",
        "--sequence",
        "M" * 1200,
        "--msa",
        "single",
        "--weights",
        str(weights),
        "--json",
    ]
    if option is not None:
        argv += ["--option", f"confidence_dtype={option}"]
    assert cli.main(argv) == 0
    block = json.loads(capsys.readouterr().out)["slurm"]
    assert block["tokens_estimate"] == 1200
    assert block["state"] == state
    assert (block["min_device_memory_gib"] is None) == (state == "unknown")


@pytest.mark.parametrize("padding", [False, True])
def test_plan_json_reads_padding(tmp_path: Path, capsys, padding: bool) -> None:
    weights = tmp_path / "w.jax"
    weights.write_bytes(b"x")
    argv = [
        "plan",
        "--model",
        "boltz2",
        "--sequence",
        "M" * 1200,
        "--msa",
        "single",
        "--weights",
        str(weights),
        "--json",
    ]
    if padding:
        argv.append("--padding")
    assert cli.main(argv) == 0
    block = json.loads(capsys.readouterr().out)["slurm"]
    assert block["tokens_estimate"] == 1200
    if not padding:
        assert block["state"] == "estimated" and block["min_device_memory_gib"]
        return
    assert block["state"] == "unknown" and block["min_device_memory_gib"] is None
    assert block["exceeds_profile"] == ["serving padding"]


@pytest.mark.parametrize(
    "inputs",
    [
        ["--sequence", "MKTAYIAKQR"],
        ["--input", "jobs.json", "--shard", "0/2"],
        ["--input", "jobs.json"],
    ],
    ids=["sequence", "shard", "jobs-file"],
)
def test_plan_json_writes_nothing_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, inputs: list[str]
) -> None:
    home = tmp_path / "home"
    workdir = tmp_path / "work"
    workdir.mkdir()
    sequences = ("MKTAYIAKQR", "MKTAYIAKQRQ", "MKTAYIAKQRQQ")
    (tmp_path / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": name,
                        "entities": [{"type": "protein", "id": "A", "sequence": s}],
                    }
                    for name, s in zip("abc", sequences, strict=True)
                ]
            }
        )
    )
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    monkeypatch.chdir(workdir)
    inputs = [str(tmp_path / v) if v == "jobs.json" else v for v in inputs]
    argv = ["plan", "--model", "esmfold2", *inputs, "--weights", str(weights)]
    assert cli.main([*argv, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    runs = payload if isinstance(payload, list) else [payload]
    assert sorted(p for p in home.rglob("*")) == []
    assert list(workdir.iterdir()) == []
    # The slurm block still read each generated document before it went away.
    assert sorted(run["slurm"]["tokens_estimate"] for run in runs) == sorted(
        len(s)
        for s in (
            sequences[:1]
            if "--sequence" in inputs
            else sequences[::2]
            if "--shard" in inputs
            else sequences
        )
    )
    assert all(run["input"].startswith(str(home / "runtime" / "jobs")) for run in runs)


# -------------------------------------------------------------- frame, check


def test_results_table_as_frame(tmp_path: Path) -> None:
    root = _batch(tmp_path)
    rows = results_table(root)
    frame = results_table(load_results(root), as_frame=True)
    assert list(frame.columns) == _columns(rows)
    assert len(frame) == len(rows) == 2


def test_check_without_posebusters_says_how_to_install(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.doctor import install_command

    root = _batch(tmp_path, ligand=True)
    monkeypatch.setitem(sys.modules, "posebusters", None)
    with pytest.raises(ModuleNotFoundError) as error:
        cli.main(["check", str(root)])
    # The one command doctor would print for this installation, not a uv line
    # for a pip install (or a bare `uv sync --extra`, which drops the others).
    assert str(error.value).endswith(f"`{install_command('posebusters')}`")
    assert "--extra posebusters" in install_command("posebusters") or (
        "foldjax[posebusters]" in install_command("posebusters")
    )


def test_check_with_posebusters(tmp_path: Path) -> None:
    pytest.importorskip("posebusters")
    from foldjax.checks import check_directory

    root = _batch(tmp_path, ligand=True)
    rows = check_directory(root)
    assert len(rows) == 2 and all(row["ligand"] == "BNZ" for row in rows)
    assert all("pb_valid" in row for row in rows)


@pytest.mark.parametrize(
    ("record", "valid", "failed", "not_computed"),
    [
        ({"bond_lengths": True, "clashes": True}, True, [], []),
        # A check PoseBusters could not compute is not a check that passed.
        ({"bond_lengths": True, "clashes": None}, False, [], ["clashes"]),
        ({"bond_lengths": True, "clashes": float("nan")}, False, [], ["clashes"]),
        ({"bond_lengths": False, "clashes": True}, False, ["bond_lengths"], []),
        # No checks at all is not "every check passed".
        ({"mol_pred_loaded_note": "text only"}, False, [], []),
    ],
    ids=["all-pass", "none-not-computed", "nan-not-computed", "one-fails", "empty"],
)
def test_pb_valid_counts_only_checks_that_passed(
    tmp_path: Path,
    monkeypatch,
    record: dict,
    valid: bool,
    failed: list[str],
    not_computed: list[str],
) -> None:
    """The verdict rule, on a stand-in PoseBusters: PB-valid is all-True."""
    from types import ModuleType, SimpleNamespace

    from foldjax.checks import check_directory

    configs: list[str] = []

    class _PoseBusters:
        def __init__(self, config: str) -> None:
            configs.append(config)

        def bust(self, mol_pred, mol_cond=None):
            assert mol_pred is not None
            return SimpleNamespace(iloc=[SimpleNamespace(to_dict=lambda: record)])

    posebusters = ModuleType("posebusters")
    posebusters.PoseBusters = _PoseBusters
    monkeypatch.setitem(sys.modules, "posebusters", posebusters)

    rows = check_directory(_batch(tmp_path, ligand=True))

    assert configs and len(rows) == len(configs)
    for row in rows:
        assert row["ligand"] == "BNZ"
        assert row["pb_valid"] is valid
        assert row["pb_failed"] == failed
        assert row["pb_not_computed"] == not_computed
