"""`confidence.json` and `foldjax_run.json` from all six backends meet the contract.

Each fixture in ``tests/fixtures/outputs`` is one real sample copied from a
finished run (``sources.json`` names it): the structure, the native scores the
manifest recorded, and for Protenix and OpenDDE the native summary JSON whose
name carries the sample's rank. A replay backend hands those back through the
real `foldjax.predict` pipeline -- the real backend class's validation and input
translation, `foldjax.output.normalize`, the manifest writer -- so what is
validated is what a run writes, not a file edited to match.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

import foldjax
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample
from foldjax.scores import ranked_native_samples
from foldjax.summary import COMMON_FIELDS_NOTE, SCHEMA_VERSION, load_schema
from tests._schema_lite import errors, undeclared, unsupported_keywords

FIXTURES = Path(__file__).parent / "fixtures" / "outputs"
CASES = sorted(path.name for path in FIXTURES.iterdir() if path.is_dir())

#: Per model: where the common pLDDT comes from, and the ranking key.
EXPECTED = {
    "alphafold3": ("structure:_atom_site.B_iso_or_equiv", "ranking_score"),
    "boltz2": ("scores.complex_plddt", "confidence_score"),
    "esmfold2": ("scores.complex_plddt", "plddt"),
    "opendde": ("scores.plddt", "ranking_score"),
    "openfold3": ("scores.mean_plddt", "sample_ranking_score"),
    "protenix": ("scores.plddt", "ranking_score"),
}
NATIVE_SCALE = {
    "boltz2": 100.0,
    "esmfold2": 100.0,
    "opendde": 1.0,
    "openfold3": 1.0,
    "protenix": 1.0,
}


def test_every_model_has_a_fixture() -> None:
    assert {case.split("_", 1)[0] for case in CASES} == set(EXPECTED)


def test_the_printed_mapping_table_is_the_one_the_writer_uses() -> None:
    from foldjax.summary import mapping_table

    rows = {row["model"]: row for row in mapping_table()}
    assert set(rows) == set(EXPECTED)
    for model, (source, ranking_key) in EXPECTED.items():
        assert rows[model]["ranking_key"] == ranking_key
        if model != "alphafold3":
            assert f"scores.{rows[model]['plddt_source']}" == source


@pytest.mark.parametrize("name", ["confidence", "run"])
def test_schemas_use_only_keywords_the_validator_checks(name: str) -> None:
    assert unsupported_keywords(load_schema(name)) == set()


@pytest.mark.parametrize("name", ["confidence", "run"])
def test_schemas_carry_the_common_fields_wording(name: str) -> None:
    assert COMMON_FIELDS_NOTE in load_schema(name)["description"]


def _replay_backend(model: str, case: Path):
    base = type(get_backend(model))
    fixture = json.loads((case / "sample.json").read_text())

    class Replay(base):  # type: ignore[misc, valid-type]
        def predict(self, request: PredictionRequest) -> PredictionResult:
            job = fixture["job"]
            if "native_summary" in fixture:
                # The native layout: diffusion order, named by rank.
                native = fixture["native_summary"]
                rank = native.rsplit("_", 1)[1].split(".")[0]
                directory = (
                    request.output_dir / job / f"seed_{request.seed}" / "predictions"
                )
                directory.mkdir(parents=True)
                structure = directory / f"{job}_sample_{rank}.cif"
                shutil.copy2(case / "native" / native, directory / native)
            else:
                structure = request.output_dir / f"{job}_native.cif"
            with gzip.open(case / "structure.cif.gz", "rb") as source:
                structure.write_bytes(source.read())
            if "native_summary" in fixture:
                samples = tuple(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=path,
                        scores=scores,
                        metadata=metadata,
                    )
                    for path, scores, metadata in ranked_native_samples([structure])
                )
            else:
                samples = (
                    PredictionSample(
                        seed=request.seed,
                        structure_path=structure,
                        scores=dict(fixture["scores"]),
                        metadata=dict(fixture["metadata"]),
                    ),
                )
            return PredictionResult(
                model=model, samples=samples, output_dir=request.output_dir
            )

    return Replay, fixture


def write_job(directory: Path, name: str) -> Path:
    """A common-schema job the replay backends accept (its content is unused)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.a3m").write_text(">query\nACD\n")
    job = directory / f"{name}.json"
    job.write_text(
        json.dumps(
            {
                "name": name,
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACD",
                        "unpaired_msa": "job.a3m",
                    }
                ],
            }
        )
    )
    return job


def _run(
    tmp_path: Path,
    case_name: str,
    *,
    out: Path | None = None,
    job: Path | None = None,
) -> tuple[dict, dict, dict]:
    case = FIXTURES / case_name
    model = case_name.split("_", 1)[0]
    backend, fixture = _replay_backend(model, case)
    if job is None:
        job = write_job(tmp_path, fixture["job"])
    tmp_path.mkdir(parents=True, exist_ok=True)
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"replayed")
    out = tmp_path / "out" if out is None else out
    request = PredictionRequest(
        model=model,
        input=job,
        weights=weights,
        output_dir=out,
        seed=int(fixture["seed"]),
        use_compile_cache=False,
    )
    with backend_override(model, backend):
        foldjax.predict(request)
    seed = int(fixture["seed"])
    confidence = json.loads(
        (out / f"seed-{seed}_sample-00" / "confidence.json").read_text()
    )
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    return confidence, manifest, fixture


@pytest.mark.parametrize("case_name", CASES)
def test_written_files_validate_against_the_published_schemas(
    tmp_path: Path, case_name: str
) -> None:
    confidence, manifest, _fixture = _run(tmp_path, case_name)

    assert errors(confidence, load_schema("confidence")) == []
    assert errors(manifest, load_schema("run")) == []
    assert confidence["schema_version"] == manifest["schema_version"] == SCHEMA_VERSION
    # Open objects let a 1.x reader take a 1.y file; they must not let a writer
    # emit a field its own published contract never names.
    assert undeclared(confidence, load_schema("confidence")) == []
    assert undeclared(manifest, load_schema("run")) == []


def test_searched_alignment_records_are_declared(tmp_path: Path) -> None:
    _confidence, manifest, _fixture = _run(tmp_path, "boltz2_e9_8reh")
    schema = load_schema("run")
    # The two shapes `msa_search._search_alignments` appends.
    manifest["msa_search"] = [
        {"chain": "A", "unpaired_msa": "msa/A.a3m", "provenance": "msa/A.json"},
        {"chain": "B", "error": "server unreachable"},
    ]
    assert errors(manifest, schema) == []
    assert undeclared(manifest, schema) == []
    manifest["msa_search"] = [{"error": "no chain"}]
    assert errors(manifest, schema)


def test_alphafold3_kernel_sources_reach_the_manifest_and_do_not_leak(
    tmp_path: Path,
) -> None:
    """The option is a cache-miss policy, so the manifest records what each
    model call actually took; the next prediction must not inherit it."""
    from foldjax.backends import _tokamax_autotune

    case = FIXTURES / "alphafold3_e9_8reh"
    replay, fixture = _replay_backend("alphafold3", case)

    class Tuned(replay):  # type: ignore[misc, valid-type]
        def predict(self, request: PredictionRequest) -> PredictionResult:
            # What a `heuristics` run on a cache an `autotune` run filled does.
            _tokamax_autotune.start_record(
                strategy="heuristics", persistent_installed=True
            )
            _tokamax_autotune._note_source("store")
            return super().predict(request)

    out = tmp_path / "af3"
    job = write_job(tmp_path, fixture["job"])
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"replayed")
    with backend_override("alphafold3", Tuned):
        foldjax.predict(
            PredictionRequest(
                model="alphafold3",
                input=job,
                weights=weights,
                output_dir=out,
                seed=int(fixture["seed"]),
                use_compile_cache=False,
            )
        )
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["kernel_tuning"] == {
        "kernel_autotuning": "heuristics",
        "persistent_store": True,
        "sources": {"store": 1},
    }
    schema = load_schema("run")
    assert errors(manifest, schema) == []
    assert undeclared(manifest, schema) == []
    manifest["kernel_tuning"]["sources"] = {"guessed": 1}
    assert errors(manifest, schema)

    _confidence, other, _fixture = _run(tmp_path / "next", "boltz2_e9_8reh")
    assert "kernel_tuning" not in other


def test_a_run_without_the_optional_fields_still_validates_and_resumes(
    tmp_path: Path,
) -> None:
    from foldjax.api import resolve_requests
    from foldjax.manifest import request_mismatch

    _confidence, manifest, fixture = _run(tmp_path, "boltz2_e9_8reh")
    assert manifest["schema_version"] == "1.0"
    for name in ("msa_search", "foldjax_source"):
        manifest.pop(name, None)
    assert errors(manifest, load_schema("run")) == []
    weights = tmp_path / "weights.jax"
    (request,) = resolve_requests(
        PredictionRequest(
            model="boltz2",
            input=tmp_path / f"{fixture['job']}.json",
            weights=weights,
            output_dir=tmp_path / "out",
            seed=int(fixture["seed"]),
            use_compile_cache=False,
        )
    )
    assert request_mismatch(manifest, request, seed=int(fixture["seed"])) is None


@pytest.mark.parametrize("case_name", CASES)
def test_summary_keys_and_scales_per_model(tmp_path: Path, case_name: str) -> None:
    """The coordinator's contract: one name and one scale, with the source."""
    model, target = case_name.split("_", 1)
    confidence, _manifest, fixture = _run(tmp_path, case_name)
    summary = confidence["summary"]
    scores = confidence["scores"]
    plddt_source, ranking_key = EXPECTED[model]

    plddt = summary["plddt"]
    assert plddt["scale"] == "0-100"
    assert plddt["source"] == plddt_source
    assert 0.0 <= plddt["value"] <= 100.0
    # Real outputs: every model here is confident, so a 0-1 value left
    # unscaled would sit below 1 and fail this.
    assert plddt["value"] > 50.0
    if model == "alphafold3":
        assert "mean" in plddt["transform"] and "B-factor" in plddt["transform"]
    else:
        key = plddt_source.removeprefix("scores.")
        assert plddt["value"] == pytest.approx(scores[key] * NATIVE_SCALE[model])

    assert summary["ptm"]["scale"] == "0-1"
    assert summary["ptm"]["value"] == scores["ptm"]
    assert 0.0 <= summary["ptm"]["value"] <= 1.0

    if target == "e9_8reh":  # one chain: no interface, whatever the model wrote
        assert summary["iptm"]["value"] is None
        assert "single chain" in summary["iptm"]["reason"]
    else:  # protein + ligand in its own chain
        assert summary["iptm"]["value"] == scores["iptm"]
        assert 0.0 < summary["iptm"]["value"] <= 1.0

    ranking = summary["ranking"]
    if model == "openfold3":
        # Protein input: only the partial no-disorder score exists, and it is
        # a different quantity, so the common field is null, not substituted.
        assert ranking["value"] is None
        assert ranking["key"] == ranking_key
        assert "sample_ranking_score_no_disorder" in ranking["reason"]
    else:
        assert ranking["key"] == ranking_key
        assert ranking["value"] == scores[ranking_key]
        assert ranking["scope"] == "within one model run"
        assert ranking["higher_is_better"] is True
        assert ranking["defined_by"] == (
            "foldjax" if model == "esmfold2" else "upstream"
        )
    assert confidence["summary_note"] == COMMON_FIELDS_NOTE


def test_alphafold3_plddt_is_the_mean_of_its_per_atom_plddt(tmp_path: Path) -> None:
    """The summary is the plain mean of the per-atom B-factors. The fixture's
    structure is synthetic (AlphaFold 3 predictions are not redistributed):
    B = 60 + (37 i mod 40) over 1,000 atoms, mean 79.5. On the real run the same
    rule reproduced the native `atom_plddts` mean (96.1033)."""
    confidence, _manifest, _fixture = _run(tmp_path, "alphafold3_e9_8reh")

    assert confidence["summary"]["plddt"]["value"] == pytest.approx(79.5, abs=1e-6)
    assert confidence["summary"]["plddt"]["granularity"] == "atom"


@pytest.mark.parametrize(
    ("case_name", "rank"), [("protenix_e9_8reh", 3), ("opendde_e9_8reh", 1)]
)
def test_ranked_writers_keep_native_rank_flags_and_execution(
    tmp_path: Path, case_name: str, rank: int
) -> None:
    confidence, manifest, _fixture = _run(tmp_path, case_name)

    # Diffusion sample 0 was written as "_sample_<rank>".
    assert confidence["sample"] == 0
    assert confidence["native_rank"] == rank
    assert manifest["samples"][0]["metadata"]["native_rank"] == rank
    # A boolean flag survives as 0/1; a recycle count is not a score.
    assert confidence["scores"]["has_clash"] == 0.0
    assert "num_recycles" not in confidence["scores"]
    assert confidence["execution"] == {"num_recycles": 10}


def test_esmfold2_keeps_its_scale_note(tmp_path: Path) -> None:
    confidence, _manifest, _fixture = _run(tmp_path, "esmfold2_e9_8reh")

    assert confidence["score_notes"] == {
        "plddt_scale": "0-1 here; the structures' b-factor column is 0-100"
    }


def test_the_validator_rejects_what_the_contract_forbids(tmp_path: Path) -> None:
    confidence, _manifest, _fixture = _run(tmp_path, "boltz2_e9_8reh")
    schema = load_schema("confidence")

    unscaled = json.loads(json.dumps(confidence))
    unscaled["summary"]["plddt"]["value"] = 101.0
    assert errors(unscaled, schema)

    zero_without_reason = json.loads(json.dumps(confidence))
    zero_without_reason["summary"]["iptm"] = {"value": None}
    assert errors(zero_without_reason, schema)

    pooled = json.loads(json.dumps(confidence))
    pooled["summary"]["ranking"]["scope"] = "across models"
    assert errors(pooled, schema)

    newer_minor = json.loads(json.dumps(confidence))
    newer_minor["schema_version"] = "1.7"
    newer_minor["a_field_added_in_1_7"] = {"anything": True}
    assert errors(newer_minor, schema) == []

    next_major = json.loads(json.dumps(confidence))
    next_major["schema_version"] = "2.0"
    assert errors(next_major, schema)
