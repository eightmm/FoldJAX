"""A nucleic-acid alignment a backend never reads is refused, not dropped.

Boltz-2 and ESMFold2 read no RNA or DNA alignment, Protenix and OpenFold3 no
DNA one, and OpenDDE no DNA one and no RNA one without ``use_rna_msa=true``.
Each used to accept the document and fold the chain from its sequence alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import foldjax
from foldjax.input import IGNORE_NUCLEIC_MSA, materialize_native_input
from foldjax.manifest import MANIFEST_NAME
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override, capabilities, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

_SEQUENCES = {"dna": "ACGTAC", "rna": "ACGUAC"}

#: (model, nucleic type) pairs whose alignment the backend discards.
_IGNORED = [
    ("boltz2", "rna"),
    ("boltz2", "dna"),
    ("esmfold2", "rna"),
    ("esmfold2", "dna"),
    ("protenix", "dna"),
    ("openfold3", "dna"),
    ("opendde", "dna"),
    ("opendde", "rna"),
]


def _job(tmp_path: Path, kind: str | None) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEF\n>hit\nACDEY\n")
    entities: list[dict] = [
        {
            "type": "protein",
            "id": "A",
            "sequence": "ACDEF",
            "unpaired_msa": "protein.a3m",
        }
    ]
    if kind is not None:
        (tmp_path / f"{kind}.a3m").write_text(
            f">query\n{_SEQUENCES[kind]}\n>hit\n{_SEQUENCES[kind]}\n"
        )
        entities.append(
            {
                "type": kind,
                "id": "N",
                "sequence": _SEQUENCES[kind],
                "unpaired_msa": f"{kind}.a3m",
            }
        )
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "nucleic", "entities": entities}))
    return path


def _materialize(
    source: Path, model: str, options: dict | None = None, ignored: list | None = None
) -> Path:
    return materialize_native_input(
        source,
        capabilities(model),
        source.parent / f"out-{model}",
        seed=1,
        options=options,
        ignored=ignored,
    )


def _read(path: Path):
    text = path.read_text()
    return yaml.safe_load(text) if path.suffix == ".yaml" else json.loads(text)


def _chain_msas(model: str, native) -> dict[str, list[str]]:
    """Every alignment path the native document hands to each chain type."""
    found: dict[str, list[str]] = {}
    if model == "boltz2":
        for entry in native["sequences"]:
            (kind, body), = entry.items()
            if body.get("msa") not in (None, "empty"):
                found.setdefault(kind, []).append(body["msa"])
    elif model in {"protenix", "opendde"}:
        names = {"proteinChain": "protein", "dnaSequence": "dna", "rnaSequence": "rna"}
        for entry in native[0]["sequences"]:
            (key, body), = entry.items()
            if body.get("unpairedMsaPath"):
                found.setdefault(names[key], []).append(body["unpairedMsaPath"])
    elif model == "openfold3":
        (query,) = native["queries"].values()
        for chain in query["chains"]:
            for path in chain.get("main_msa_file_paths", []):
                found.setdefault(chain["molecule_type"], []).append(path)
    elif model == "esmfold2":
        for entity in native["entities"]:
            if entity.get("unpaired_msa"):
                found.setdefault(entity["type"], []).append(entity["unpaired_msa"])
    elif model == "alphafold3":
        for entry in native["sequences"]:
            (kind, body), = entry.items()
            if body.get("unpairedMsaPath"):
                found.setdefault(kind, []).append(body["unpairedMsaPath"])
    return found


@pytest.mark.parametrize(("model", "kind"), _IGNORED)
def test_an_ignored_nucleic_msa_is_refused_by_default(
    tmp_path: Path, model: str, kind: str
) -> None:
    source = _job(tmp_path, kind)
    with pytest.raises(ValueError) as error:
        _materialize(source, model)
    message = str(error.value)
    assert message.startswith(f"{model} cannot express ")
    assert f"{kind.upper()} unpaired_msa" in message
    assert "entity 'N'" in message
    assert f"{kind}.a3m" in message
    assert f"{IGNORE_NUCLEIC_MSA}=true" in message


@pytest.mark.parametrize(("model", "kind"), _IGNORED)
def test_the_opt_in_leaves_it_out_of_the_native_input_and_says_so(
    tmp_path: Path, model: str, kind: str
) -> None:
    source = _job(tmp_path, kind)
    ignored: list = []
    native = _read(_materialize(source, model, {IGNORE_NUCLEIC_MSA: True}, ignored))

    msas = _chain_msas(model, native)
    assert kind not in msas
    # The protein alignment is untouched by the option.
    assert len(msas["protein"]) == 1
    assert ignored == [
        {
            "chains": ["N"],
            "type": kind,
            "field": "unpaired_msa",
            "path": f"{kind}.a3m",
            "resolved_path": str((tmp_path / f"{kind}.a3m").resolve()),
            "reason": (
                f"{model} does not read {kind.upper()} alignments; dropped by "
                f"{IGNORE_NUCLEIC_MSA}=true"
            ),
        }
    ]


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("alphafold3", None),
        ("protenix", None),
        ("protenix", {IGNORE_NUCLEIC_MSA: True}),
        ("openfold3", None),
        ("openfold3", {IGNORE_NUCLEIC_MSA: True}),
        ("opendde", {"use_rna_msa": True}),
        ("opendde", {"use_rna_msa": True, IGNORE_NUCLEIC_MSA: True}),
    ],
)
def test_an_rna_msa_the_backend_reads_still_reaches_it(
    tmp_path: Path, model: str, options: dict | None
) -> None:
    source = _job(tmp_path, "rna")
    ignored: list = []
    native = _read(_materialize(source, model, options, ignored))
    msas = _chain_msas(model, native)
    assert len(msas["rna"]) == 1
    assert len(msas["protein"]) == 1
    assert ignored == []


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix"]
)
def test_a_protein_msa_is_unchanged(tmp_path: Path, model: str) -> None:
    source = _job(tmp_path, None)
    ignored: list = []
    native = _read(_materialize(source, model, ignored=ignored))
    msas = _chain_msas(model, native)
    assert list(msas) == ["protein"]
    assert len(msas["protein"]) == 1
    assert ignored == []


def test_alphafold3_does_not_take_the_option(tmp_path: Path) -> None:
    """Its parser already refuses a DNA alignment, so there is nothing to drop."""
    request = PredictionRequest(
        model="alphafold3",
        input=_job(tmp_path, None),
        input_format="foldjax",
        options={IGNORE_NUCLEIC_MSA: True},
    )
    with pytest.raises(ValueError, match=f"unsupported alphafold3 options: {IGNORE_NUCLEIC_MSA}"):
        get_backend("alphafold3").validate_request(request)


@pytest.mark.parametrize("model", ["boltz2", "esmfold2", "opendde", "openfold3", "protenix"])
def test_the_option_is_checked_while_planning(tmp_path: Path, model: str) -> None:
    job = _job(tmp_path, None)
    backend = get_backend(model)
    formats = backend.capabilities().input_formats
    common = PredictionRequest(
        model=model,
        input=job,
        input_format="foldjax",
        options={IGNORE_NUCLEIC_MSA: True},
    )
    backend.validate_request(common)
    with pytest.raises(ValueError, match=f"{IGNORE_NUCLEIC_MSA} must be a boolean"):
        backend.validate_request(
            PredictionRequest(
                model=model,
                input=job,
                input_format="foldjax",
                options={IGNORE_NUCLEIC_MSA: "yes"},
            )
        )
    if "native" in formats:
        with pytest.raises(ValueError, match="applies to FoldJAX common-schema input"):
            backend.validate_request(
                PredictionRequest(
                    model=model,
                    input=job,
                    input_format="native",
                    options={IGNORE_NUCLEIC_MSA: True},
                )
            )


def _recorder(model: str, seen: list):
    base = provider(PORTS[model].backend)

    class Recorder(base):
        def predict(self, request):
            seen.append(request)
            path = request.output_dir / f"s{request.seed}.cif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data_mock\n#\n", encoding="utf-8")
            return PredictionResult(
                model=model,
                samples=(
                    PredictionSample(
                        seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                    ),
                ),
                output_dir=request.output_dir,
            )

    return Recorder


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"not really weights")
    return path


def test_a_run_refuses_then_records_the_drop_in_its_manifest(tmp_path: Path) -> None:
    source = _job(tmp_path, "rna")
    seen: list = []

    def request(options: dict, out: str) -> PredictionRequest:
        return PredictionRequest(
            model="opendde",
            input=source,
            weights=_weights(tmp_path),
            output_dir=tmp_path / out,
            seed=3,
            options=options,
            use_compile_cache=False,
        )

    with backend_override("opendde", _recorder("opendde", seen)):
        with pytest.raises(ValueError, match="use_rna_msa=true"):
            foldjax.predict(request({}, "refused"))
        assert seen == []
        assert not (tmp_path / "refused" / MANIFEST_NAME).exists()

        foldjax.predict(request({IGNORE_NUCLEIC_MSA: True}, "dropped"))

    # The native runner never sees an option only the translation consumes.
    (ran,) = seen
    assert IGNORE_NUCLEIC_MSA not in ran.options
    native = _read(ran.input)
    assert _chain_msas("opendde", native) == {
        "protein": [str((tmp_path / "protein.a3m").resolve())]
    }

    manifest = json.loads((tmp_path / "dropped" / MANIFEST_NAME).read_text())
    assert manifest["options"][IGNORE_NUCLEIC_MSA] is True
    (record,) = manifest["ignored_msas"]
    assert record["chains"] == ["N"]
    assert record["type"] == "rna"
    assert record["resolved_path"] == str((tmp_path / "rna.a3m").resolve())


def test_a_run_without_a_dropped_alignment_records_none(tmp_path: Path) -> None:
    source = _job(tmp_path, None)
    seen: list = []
    with backend_override("opendde", _recorder("opendde", seen)):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=source,
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
                seed=3,
                use_compile_cache=False,
            )
        )
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["ignored_msas"] == []
    assert _chain_msas("opendde", _read(seen[0].input)) == {
        "protein": [str((tmp_path / "protein.a3m").resolve())]
    }


def test_required_search_does_not_demand_an_rna_alignment_nobody_reads(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("FOLDJAX_RNA_MSA_COMMAND", raising=False)
    path = tmp_path / "rna-only.json"
    path.write_text(
        json.dumps({"entities": [{"type": "rna", "id": "R", "sequence": "ACGU"}]})
    )
    materialize_native_input(
        path, capabilities("boltz2"), tmp_path / "boltz", seed=1, msa="required"
    )
    with pytest.raises(ValueError, match="no RNA search is configured"):
        materialize_native_input(
            path, capabilities("protenix"), tmp_path / "protenix", seed=1, msa="required"
        )


def test_esmfold2_features_refuse_a_nucleic_alignment(tmp_path: Path) -> None:
    """The guard for callers that build ESMFold2 features without the validator."""
    from foldjax.models.esmfold2.data.all_atom import Chain, _msa

    chain = Chain(
        chain_id="N",
        asym_id=0,
        entity_index=0,
        entity_id=0,
        sym_id=0,
        kind="rna",
        sequence="ACGU",
    )
    with pytest.raises(ValueError, match="only for protein chains"):
        _msa([chain], [], {0: tmp_path / "rna.a3m"}, msa_depth=None)
