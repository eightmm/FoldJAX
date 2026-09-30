"""A nucleic-acid alignment a backend never reads is refused, not dropped.

Boltz-2 and ESMFold2 read no RNA or DNA alignment, OpenFold3 no DNA one, and
Protenix and OpenDDE no DNA one and no RNA one without ``use_rna_msa=true``
(upstream's flag, released false by both). Each used to accept the document
and fold the chain from its sequence alone.
A nucleic ``paired_msa`` follows the same rule: only OpenFold3 reads one,
and only for RNA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    ("protenix", "rna"),
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
        ("protenix", {"use_rna_msa": True}),
        ("protenix", {"use_rna_msa": True, IGNORE_NUCLEIC_MSA: True}),
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


@pytest.mark.parametrize("model", ["opendde", "protenix"])
def test_a_run_refuses_then_records_the_drop_in_its_manifest(
    tmp_path: Path, model: str
) -> None:
    source = _job(tmp_path, "rna")
    seen: list = []

    def request(options: dict, out: str) -> PredictionRequest:
        return PredictionRequest(
            model=model,
            input=source,
            weights=_weights(tmp_path),
            output_dir=tmp_path / out,
            seed=3,
            options=options,
            use_compile_cache=False,
        )

    with backend_override(model, _recorder(model, seen)):
        with pytest.raises(ValueError) as refusal:
            foldjax.predict(request({}, "refused"))
        assert seen == []
        assert not (tmp_path / "refused" / MANIFEST_NAME).exists()

        foldjax.predict(request({IGNORE_NUCLEIC_MSA: True}, "dropped"))

    # The refusal names the entity and both ways out.
    message = str(refusal.value)
    assert "entity 'N'" in message
    assert "use_rna_msa=true" in message
    assert f"{IGNORE_NUCLEIC_MSA}=true" in message

    # The native runner never sees an option only the translation consumes.
    (ran,) = seen
    assert IGNORE_NUCLEIC_MSA not in ran.options
    native = _read(ran.input)
    assert _chain_msas(model, native) == {
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
    # Protenix reads none at its released use_rna_msa=false, either.
    materialize_native_input(
        path, capabilities("protenix"), tmp_path / "protenix", seed=1, msa="required"
    )
    with pytest.raises(ValueError, match="no RNA search is configured"):
        materialize_native_input(
            path,
            capabilities("protenix"),
            tmp_path / "protenix-rna",
            seed=1,
            msa="required",
            options={"use_rna_msa": True},
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


def test_protenix_use_rna_msa_reaches_the_features(tmp_path: Path) -> None:
    """The option is rendered as the native flag, and the featurizer reads it."""
    from foldjax.backends.protenix import ProtenixBackend
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    source = _job(tmp_path, "rna")
    # A hit unlike the query, so deduplication cannot hide it.
    (tmp_path / "rna.a3m").write_text(">query\nACGUAC\n>hit\nAGGUCC\n")
    seen: list = []
    with backend_override("protenix", _recorder("protenix", seen)):
        foldjax.predict(
            PredictionRequest(
                model="protenix",
                input=source,
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
                seed=3,
                options={"use_rna_msa": True},
                use_compile_cache=False,
            )
        )
    (ran,) = seen
    native = _read(ran.input)
    assert _chain_msas("protenix", native)["rna"] == [
        str((tmp_path / "rna.a3m").resolve())
    ]
    invocation = ProtenixBackend()._native_invocation(ran)
    assert "--use-rna-msa" in invocation.argv
    assert invocation.config_fields["use_rna_msa"] is True
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["ignored_msas"] == []

    job = native[0]
    read = featurize_protein_json(
        job, use_rna_msa=invocation.config_fields["use_rna_msa"]
    )
    with pytest.warns(RuntimeWarning, match="use_rna_msa"):
        unread = featurize_protein_json(job)
    rna = np.asarray(read["restype"]).argmax(-1) >= 21
    assert rna.any()
    # The hit "AGGUCC" (torch STD_RESIDUES: A=21 G=22 C=23 U=24) is a row of
    # the RNA columns only when the alignment was read.
    hit = [21, 22, 22, 24, 23, 23]
    assert hit in read["msa"][:, rna].tolist()
    assert hit not in unread["msa"][:, rna].tolist()


def test_protenix_defaults_to_upstreams_released_flag(tmp_path: Path) -> None:
    from foldjax.backends.protenix import ProtenixBackend

    request = PredictionRequest(
        seed=0,
        model="protenix",
        input=_job(tmp_path, None),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
    )
    invocation = ProtenixBackend()._native_invocation(request)
    assert "--use-rna-msa" not in invocation.argv
    assert invocation.config_fields["use_rna_msa"] is False


@pytest.mark.parametrize("value", ["true", 1])
def test_protenix_use_rna_msa_must_be_a_boolean(tmp_path: Path, value) -> None:
    request = PredictionRequest(
        model="protenix",
        input=_job(tmp_path, None),
        input_format="foldjax",
        options={"use_rna_msa": value},
    )
    with pytest.raises(ValueError, match="use_rna_msa must be a boolean"):
        get_backend("protenix").validate_request(request)


def test_protenix_use_rna_msa_follows_upstreams_model_list(tmp_path: Path) -> None:
    """Upstream asserts the flag only for its v1.0.0 base models and protenix-v2."""
    job = _job(tmp_path, None)
    backend = get_backend("protenix")
    for name in ("protenix-v2", "protenix_base_default_v1.0.0"):
        backend.validate_request(
            PredictionRequest(
                model="protenix",
                input=job,
                input_format="foldjax",
                options={"use_rna_msa": True, "model_name": name},
            )
        )
    with pytest.raises(ValueError, match="use_rna_msa is not supported by"):
        backend.validate_request(
            PredictionRequest(
                model="protenix",
                input=job,
                input_format="foldjax",
                options={
                    "use_rna_msa": True,
                    "model_name": "protenix_mini_esm_v0.5.0",
                },
            )
        )


# A paired alignment on a nucleic chain. Protenix writes `pairedMsaPath` on a
# DNA chain and its nucleic builder never opens it, OpenDDE shares that
# builder, and OpenFold3 maps paired alignments only for protein and RNA.

#: (model, nucleic type) pairs whose paired alignment the backend discards.
_PAIRED_IGNORED = [
    ("protenix", "dna"),
    ("opendde", "dna"),
    ("openfold3", "dna"),
]


def _paired_job(tmp_path: Path, kind: str, *, protein_paired: bool = True) -> Path:
    """A protein with its alignments, and a nucleic chain with a paired one."""
    (tmp_path / "protein.a3m").write_text(">query\nACDEF\n>hit\nACDEY\n")
    (tmp_path / "protein_paired.a3m").write_text(">query\nACDEF\n>101\nACDEW\n")
    sequence = _SEQUENCES[kind]
    (tmp_path / f"{kind}_paired.a3m").write_text(
        f">query\n{sequence}\n>101\n{sequence}\n"
    )
    protein = {
        "type": "protein",
        "id": "A",
        "sequence": "ACDEF",
        "unpaired_msa": "protein.a3m",
    }
    if protein_paired:
        protein["paired_msa"] = "protein_paired.a3m"
    entities = [
        protein,
        {
            "type": kind,
            "id": "N",
            "sequence": sequence,
            "paired_msa": f"{kind}_paired.a3m",
        },
    ]
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "nucleic", "entities": entities}))
    return path


def _chain_paired_msas(model: str, native) -> dict[str, list[str]]:
    """Every paired alignment path the native document hands to each chain type."""
    found: dict[str, list[str]] = {}
    if model in {"protenix", "opendde"}:
        names = {"proteinChain": "protein", "dnaSequence": "dna", "rnaSequence": "rna"}
        for entry in native[0]["sequences"]:
            ((key, body),) = entry.items()
            if body.get("pairedMsaPath"):
                found.setdefault(names[key], []).append(body["pairedMsaPath"])
    elif model == "openfold3":
        (query,) = native["queries"].values()
        for chain in query["chains"]:
            for path in chain.get("paired_msa_file_paths", []):
                found.setdefault(chain["molecule_type"], []).append(path)
    return found


@pytest.mark.parametrize(("model", "kind"), _PAIRED_IGNORED)
def test_an_ignored_nucleic_paired_msa_is_refused_by_default(
    tmp_path: Path, model: str, kind: str
) -> None:
    source = _paired_job(tmp_path, kind)
    with pytest.raises(ValueError) as error:
        _materialize(source, model)
    message = str(error.value)
    assert message.startswith(f"{model} cannot express a {kind.upper()} paired_msa")
    assert "entity 'N'" in message
    assert f"{kind}_paired.a3m" in message
    assert f"{IGNORE_NUCLEIC_MSA}=true" in message


@pytest.mark.parametrize(("model", "kind"), _PAIRED_IGNORED)
def test_the_opt_in_leaves_a_nucleic_paired_msa_out_and_says_so(
    tmp_path: Path, model: str, kind: str
) -> None:
    source = _paired_job(tmp_path, kind)
    ignored: list = []
    native = _read(_materialize(source, model, {IGNORE_NUCLEIC_MSA: True}, ignored))

    paired = _chain_paired_msas(model, native)
    assert kind not in paired
    # The protein's alignments are untouched by the option.
    assert len(paired["protein"]) == 1
    assert len(_chain_msas(model, native)["protein"]) == 1
    assert ignored == [
        {
            "chains": ["N"],
            "type": kind,
            "field": "paired_msa",
            "path": f"{kind}_paired.a3m",
            "resolved_path": str((tmp_path / f"{kind}_paired.a3m").resolve()),
            "reason": (
                f"{model} does not read {kind.upper()} paired alignments; "
                f"dropped by {IGNORE_NUCLEIC_MSA}=true"
            ),
        }
    ]


@pytest.mark.parametrize("options", [None, {IGNORE_NUCLEIC_MSA: True}])
def test_openfold3_still_receives_an_rna_paired_msa(
    tmp_path: Path, options: dict | None
) -> None:
    """OpenFold3 maps paired alignments for RNA, so nothing is dropped there."""
    source = _paired_job(tmp_path, "rna")
    ignored: list = []
    native = _read(_materialize(source, "openfold3", options, ignored))
    paired = _chain_paired_msas("openfold3", native)
    assert len(paired["rna"]) == 1
    assert (
        Path(paired["rna"][0]).read_text() == (tmp_path / "rna_paired.a3m").read_text()
    )
    assert len(paired["protein"]) == 1
    assert ignored == []


@pytest.mark.parametrize(
    ("model", "kind", "reason"),
    [
        # Neither dialect has a paired alignment of any kind.
        ("boltz2", "rna", "cannot express paired_msa: remove it from entity 'N'"),
        ("boltz2", "dna", "cannot express paired_msa: remove it from entity 'N'"),
        ("esmfold2", "rna", "cannot express paired_msa: remove it from entity 'N'"),
        ("esmfold2", "dna", "cannot express paired_msa: remove it from entity 'N'"),
        # Upstream Protenix and OpenDDE accept no RNA pairedMsaPath at all.
        ("protenix", "rna", "cannot express RNA paired_msa"),
        ("opendde", "rna", "cannot express RNA paired_msa"),
    ],
)
def test_the_opt_in_does_not_turn_an_inexpressible_paired_msa_into_a_drop(
    tmp_path: Path, model: str, kind: str, reason: str
) -> None:
    # No protein pairing, so the refusal can only come from the nucleic chain.
    source = _paired_job(tmp_path, kind, protein_paired=False)
    ignored: list = []
    with pytest.raises(ValueError, match=reason):
        _materialize(source, model, {IGNORE_NUCLEIC_MSA: True}, ignored)
    assert ignored == []


def test_a_run_refuses_then_records_a_dropped_paired_msa(tmp_path: Path) -> None:
    source = _paired_job(tmp_path, "dna")
    seen: list = []

    def request(options: dict, out: str) -> PredictionRequest:
        return PredictionRequest(
            model="protenix",
            input=source,
            weights=_weights(tmp_path),
            output_dir=tmp_path / out,
            seed=3,
            options=options,
            use_compile_cache=False,
        )

    with backend_override("protenix", _recorder("protenix", seen)):
        with pytest.raises(ValueError, match="a DNA paired_msa"):
            foldjax.predict(request({}, "refused"))
        assert seen == []
        assert not (tmp_path / "refused" / MANIFEST_NAME).exists()

        foldjax.predict(request({IGNORE_NUCLEIC_MSA: True}, "dropped"))

    (ran,) = seen
    assert IGNORE_NUCLEIC_MSA not in ran.options
    assert _chain_paired_msas("protenix", _read(ran.input)) == {
        "protein": [str((tmp_path / "protein_paired.a3m").resolve())]
    }

    manifest = json.loads((tmp_path / "dropped" / MANIFEST_NAME).read_text())
    (record,) = manifest["ignored_msas"]
    assert record["chains"] == ["N"]
    assert record["type"] == "dna"
    assert record["field"] == "paired_msa"
    assert record["resolved_path"] == str((tmp_path / "dna_paired.a3m").resolve())
