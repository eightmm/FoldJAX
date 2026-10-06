"""`plan` refuses what `predict` refuses, before anything is written or drawn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foldjax.api
from foldjax.api import detect_input_format
from foldjax.backends.base import Backend
from foldjax.cli import main
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
)

SEQUENCE = "MKTAYIAKQRQISFVK"

#: A three-component dictionary in the released file's sorted order, with a
#: one-atom component in key-value form as an ion is written.
COMPONENTS = """data_ATP
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
ATP PG P
ATP PA P
ATP "O5'" O
#
data_NA
#
_chem_comp_atom.comp_id NA
_chem_comp_atom.atom_id NA
#
data_THR
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
THR N N
THR CA C
THR OG1 O
#
"""

_PROTEIN = f"{{type: protein, id: A, sequence: {SEQUENCE}, unpaired_msa: q.a3m}}"

#: The audit's malformed jobs, and the sentence each is refused with.
MALFORMED = {
    "empty_sequence": (
        'entities:\n  - {type: protein, id: A, sequence: "", unpaired_msa: q.a3m}\n',
        "requires a non-empty string sequence",
    ),
    "bad_letter": (
        "entities:\n  - {type: protein, id: A, sequence: MKTJAYB1, "
        "unpaired_msa: q.a3m}\n",
        "unsupported residue '1' at position 8",
    ),
    "rna_t": (
        "entities:\n  - {type: rna, id: A, sequence: ACGTU}\n",
        "unsupported residue 'T'",
    ),
    "duplicate_id": (
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: A, ccd: ATP}}\n",
        "duplicate chain id",
    ),
    "blank_id": (
        f'entities:\n  - {{type: protein, id: "", sequence: {SEQUENCE}, '
        "unpaired_msa: q.a3m}\n",
        "non-empty id",
    ),
    "missing_msa": (
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}, "
        "unpaired_msa: nope.a3m}\n",
        "names unpaired_msa 'nope.a3m', and there is no such file",
    ),
    "modification_range": (
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}, "
        "unpaired_msa: q.a3m, modifications: [{ccd: SEP, position: 99}]}\n",
        "modification position 99 outside sequence length 16",
    ),
    "bond_atom": (
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, ccd: ATP}}\n"
        "bonds:\n  - [[A, 3, XX9], [L, 1, PA]]\n",
        "bond atom 'XX9' is not an atom of residue 3 (THR) of chain 'A'",
    ),
    "unknown_ccd": (
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, ccd: QQQZ}}\n",
        "CCD code 'QQQZ' is not in the wwPDB Chemical Component Dictionary",
    ),
    "bad_smiles": (
        f'entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, smiles: "C1CC(("}}\n',
        "is not a valid SMILES string",
    ),
    "ccd_and_smiles": (
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, ccd: ATP, "
        "smiles: CCO}\n",
        "either ccd or smiles, not both",
    ),
    "bad_type": (
        f"entities:\n  - {{type: peptide, id: A, sequence: {SEQUENCE}}}\n",
        "unsupported entity type: 'peptide'",
    ),
    "misspelled_entities": (
        f"entitys:\n  - {{type: protein, id: A, sequence: {SEQUENCE}}}\n",
        "did you mean 'entities'?",
    ),
    "duplicate_job_names": (
        f"jobs:\n  - name: a\n    entities: [{_PROTEIN}]\n"
        f"  - name: a\n    entities: [{_PROTEIN}]\n",
        "repeats the name of jobs[0]",
    ),
    "unnamed_job": (
        f"jobs:\n  - entities: [{_PROTEIN}]\n",
        "needs a non-empty name",
    ),
    "yaml_syntax": (
        "entities:\n  - {type: protein, id: A, sequence: MKTA\n",
        "is not readable as YAML",
    ),
    "unknown_entity_field": (
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}, "
        "unpaired_msa: q.a3m, msa: q.a3m}\n",
        "unsupported protein entity fields: 'msa'",
    ),
    "lowercase_ccd": (
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, ccd: atp}}\n",
        "uppercase",
    ),
}


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "store"
    (home / "assets").mkdir(parents=True)
    (home / "assets" / "components.cif").write_text(COMPONENTS, encoding="utf-8")
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    monkeypatch.setenv("FOLDJAX_PROGRESS", "0")
    monkeypatch.delenv("PROTENIX_CCD_COMPONENTS_FILE", raising=False)
    return home


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "q.a3m").write_text(f">q\n{SEQUENCE}\n", encoding="utf-8")
    (directory / "weights.safetensors").write_bytes(b"not a checkpoint")
    monkeypatch.chdir(directory)
    return directory


def _job(directory: Path, name: str, text: str) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(f"name: {name}\n{text}" if "jobs:" not in text else text)
    return path


def _plan(path: Path, *extra: str, model: str = "boltz2") -> int:
    return main(
        [
            "plan",
            "--model",
            model,
            "--input",
            str(path),
            "--weights",
            "weights.safetensors",
            *extra,
        ]
    )


class _RefusingBackend(Backend):
    """Stands in for Boltz-2; reaching `predict` means validation let it through."""

    name = "boltz2"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model="boltz2", input_formats=("foldjax", "native"))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        raise AssertionError("a refused job reached the model")


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_plan_refuses_every_malformed_job_with_predicts_message(
    case: str, store: Path, workdir: Path
) -> None:
    if case == "bad_smiles":
        pytest.importorskip("rdkit")
    text, expected = MALFORMED[case]
    path = _job(workdir, case, text)

    with pytest.raises((ValueError, FileNotFoundError)) as planned:
        _plan(path)
    assert expected in str(planned.value)
    assert "\n" not in str(planned.value) or case in {"bad_letter", "rna_t"}

    with backend_override("boltz2", _RefusingBackend):
        with pytest.raises((ValueError, FileNotFoundError)) as predicted:
            main(
                [
                    "predict",
                    "--model",
                    "boltz2",
                    "--input",
                    str(path),
                    "--weights",
                    "weights.safetensors",
                    "--output-dir",
                    str(workdir / "out"),
                    "--no-cache",
                ]
            )
    assert str(predicted.value) == str(planned.value)
    # Neither command wrote a generated job into the store.
    assert not (store / "runtime").exists()


def test_plan_refuses_a_bare_protein_as_predict_does(
    store: Path, workdir: Path
) -> None:
    path = _job(
        workdir,
        "bare",
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}}}\n",
    )
    with pytest.raises(ValueError, match="have no alignment"):
        _plan(path)
    assert _plan(path, "--msa", "single") == 0
    # ESMFold2's upstream folds from the single sequence, so it is exempt.
    assert _plan(path, model="esmfold2") == 0


def test_plan_refuses_options_the_job_cannot_carry(store: Path, workdir: Path) -> None:
    bonded = _job(
        workdir,
        "bonded",
        f"entities:\n  - {_PROTEIN}\n  - {{type: ligand, id: L, ccd: ATP}}\n"
        "bonds:\n  - [[A, 3, OG1], [L, 1, PA]]\n",
    )
    assert _plan(bonded) == 0
    with pytest.raises(ValueError, match="openfold3 cannot express bonds"):
        _plan(bonded, model="openfold3")

    with pytest.raises(ValueError, match="only Boltz-2 carries an affinity head"):
        main(
            [
                "plan",
                "--model",
                "protenix",
                "--sequence",
                SEQUENCE,
                "--ligand",
                "ATP",
                "--affinity-binder",
                "B",
                "--msa",
                "single",
                "--weights",
                "weights.safetensors",
            ]
        )


def test_the_ccd_checks_are_skipped_without_a_dictionary(
    store: Path, workdir: Path
) -> None:
    (store / "assets" / "components.cif").unlink()
    path = _job(workdir, "unknown_ccd", MALFORMED["unknown_ccd"][0])
    assert _plan(path) == 0


def test_plan_writes_no_generated_job_into_the_store(
    store: Path, workdir: Path, capsys
) -> None:
    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--sequence",
                SEQUENCE,
                "--msa",
                "single",
                "--weights",
                "weights.safetensors",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert not (store / "runtime").exists()
    assert plan["generated_input"]["entities"][0]["sequence"] == SEQUENCE
    # The path predict would store it at, and the directory it would fill.
    assert Path(plan["input"]).is_relative_to(store / "runtime" / "jobs")
    assert Path(plan["output_dir"]).name == Path(plan["input"]).stem

    jobs = workdir / "batch.yaml"
    jobs.write_text(f"jobs:\n  - name: a\n    entities: [{_PROTEIN}]\n")
    assert _plan(jobs) == 0
    assert not (store / "runtime").exists()


def test_the_default_sequence_stem_is_stable_and_content_keyed(
    store: Path, workdir: Path, capsys
) -> None:
    def output_dir(*sequences: str, name: str | None = None) -> str:
        argv = ["plan", "--model", "boltz2", "--sequence", *sequences]
        argv += ["--msa", "single", "--weights", "weights.safetensors"]
        if name:
            argv += ["--name", name]
        assert main(argv) == 0
        return json.loads(capsys.readouterr().out)["output_dir"]

    first = output_dir(SEQUENCE)
    assert Path(first).name.startswith("job-")
    assert output_dir("GGSGGSGGS") != first
    assert output_dir(SEQUENCE) == first
    assert Path(output_dir(SEQUENCE, name="mine")).name == "mine"


def test_a_refused_run_draws_no_seed(store: Path, workdir: Path, monkeypatch) -> None:
    def no_draw() -> int:
        raise AssertionError("a seed was drawn for a run that was refused")

    monkeypatch.setattr(foldjax.api, "_draw_seed", no_draw)
    path = _job(
        workdir,
        "bare",
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}}}\n",
    )
    with backend_override("boltz2", _RefusingBackend):
        with pytest.raises(ValueError, match="have no alignment"):
            foldjax.api.predict_batch(
                PredictionRequest(
                    model="boltz2",
                    input=path,
                    weights=workdir / "weights.safetensors",
                    output_dir=workdir / "out",
                )
            )


def test_keep_going_records_a_refused_job_and_runs_nothing(
    store: Path, workdir: Path
) -> None:
    path = _job(workdir, "bad", MALFORMED["unknown_ccd"][0])
    with backend_override("boltz2", _RefusingBackend):
        report = foldjax.api.predict_batch(
            PredictionRequest(
                model="boltz2",
                inputs=(path,),
                weights=workdir / "weights.safetensors",
                output_dir=workdir / "out",
                seed=1,
                on_error="continue",
            )
        )
    assert not report.results
    assert [failure.seed for failure in report.failures] == [None]
    assert "QQQZ" in report.failures[0].error


def test_auto_detection_reads_an_unsigned_mapping_as_a_foldjax_job(
    workdir: Path,
) -> None:
    misspelled = workdir / "typo.yaml"
    misspelled.write_text("entitys: []\n")
    assert detect_input_format(misspelled) == "foldjax"
    native = workdir / "native.yaml"
    native.write_text("version: 1\nsequences: []\n")
    assert detect_input_format(native) == "native"
    listed = workdir / "server.json"
    listed.write_text("[]")
    assert detect_input_format(listed) == "native"

    broken = workdir / "broken.yaml"
    broken.write_text("entities:\n  - {type: protein, id: A\n")
    with pytest.raises(ValueError, match="is not readable as YAML") as error:
        detect_input_format(broken)
    assert "\n" not in str(error.value)
    assert "line" in str(error.value)


def test_a_yaml_file_is_never_parsed_as_json(workdir: Path) -> None:
    from foldjax.backends.esmfold2 import _job_document

    path = workdir / "job.yaml"
    path.write_text(f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}}}\n")
    document, _base = _job_document(path)
    assert document["entities"][0]["sequence"] == SEQUENCE


def test_an_invalid_molecule_is_refused_in_the_common_layer(
    store: Path, workdir: Path
) -> None:
    pytest.importorskip("rdkit")
    path = _job(
        workdir,
        "valence",
        f"entities:\n  - {_PROTEIN}\n"
        "  - {type: ligand, id: L, smiles: C(C)(C)(C)(C)C}\n",
    )
    with pytest.raises(ValueError, match="is not a valid molecule"):
        _plan(path)


def test_an_unknown_modification_code_names_its_residue(
    store: Path, workdir: Path
) -> None:
    path = _job(
        workdir,
        "modified",
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}, "
        "unpaired_msa: q.a3m, modifications: [{ccd: ZZZ9, position: 3}]}\n",
    )
    with pytest.raises(ValueError, match="residue 3 of chain 'A'.*'ZZZ9'"):
        _plan(path)


def test_a_bond_to_a_modified_residue_uses_its_component(
    store: Path, workdir: Path
) -> None:
    """Position 3 is THR, but modified to ATP here: ATP's atoms apply."""
    path = _job(
        workdir,
        "bond_modified",
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}, "
        "unpaired_msa: q.a3m, modifications: [{ccd: ATP, position: 3}]}\n"
        "  - {type: ligand, id: L, ccd: NA}\n"
        "bonds:\n  - [[A, 3, OG1], [L, 1, NA]]\n",
    )
    with pytest.raises(ValueError, match=r"residue 3 \(ATP\)"):
        _plan(path)


def test_models_for_applies_the_alignment_policy(
    store: Path, workdir: Path, capsys
) -> None:
    path = _job(
        workdir,
        "bare",
        f"entities:\n  - {{type: protein, id: A, sequence: {SEQUENCE}}}\n",
    )

    def rows(*extra: str) -> dict[str, dict]:
        assert main(["models", "--for", str(path), "--json", *extra]) == 0
        return {row["model"]: row for row in json.loads(capsys.readouterr().out)}

    default = rows()
    assert default["boltz2"]["runs"] is False
    assert "have no alignment" in default["boltz2"]["reason"]
    assert default["esmfold2"]["runs"] is True
    assert rows("--msa", "single")["boltz2"]["runs"] is True

    missing = _job(workdir, "missing", MALFORMED["missing_msa"][0])
    assert main(["models", "--for", str(missing), "--json"]) == 0
    reason = json.loads(capsys.readouterr().out)[0]["reason"]
    assert "no such file" in reason


def test_models_for_answers_a_multi_job_file_per_job(
    store: Path, workdir: Path, capsys
) -> None:
    path = workdir / "batch.yaml"
    path.write_text(
        f"jobs:\n  - name: good\n    entities: [{_PROTEIN}]\n"
        "  - name: bare\n"
        f"    entities: [{{type: protein, id: A, sequence: {SEQUENCE}}}]\n"
    )
    assert main(["models", "--for", str(path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    by_job = {(row["job"], row["model"]): row for row in rows}
    assert by_job[("good", "boltz2")]["runs"] is True
    assert by_job[("bare", "boltz2")]["runs"] is False
    assert not any("unsupported top-level" in str(row["reason"]) for row in rows)

    assert main(["models", "--for", str(path)]) == 0
    text = capsys.readouterr().out
    assert "job good" in text and "job bare" in text
