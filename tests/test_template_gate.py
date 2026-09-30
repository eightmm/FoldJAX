"""A common-schema template Protenix or OpenDDE would ignore is refused.

Both upstreams read a chain's templates only under ``use_template``, released
false (Protenix ``configs/configs_inference.py:36`` and
``protenix/data/template/template_featurizer.py:710``; OpenDDE
``config/inference_defaults.py:28``). The Protenix port used to read them
anyway. The contract follows the nucleic-MSA one (``tests/test_nucleic_msa.py``):
refused by default, read with ``use_template=true``, dropped and recorded in the
manifest's ``ignored_templates`` with ``ignore_templates=true``.

Requests below pass ``seed`` and ``msa`` explicitly and give every protein an
alignment, so they mean the same on a base where either default moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foldjax
from foldjax.input import IGNORE_TEMPLATES, materialize_native_input
from foldjax.manifest import MANIFEST_NAME
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override, capabilities, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

_MODELS = ("opendde", "protenix")


def _job(tmp_path: Path) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEF\n>hit\nACDEY\n")
    (tmp_path / "template.cif").write_text("data_template\n_entry.id template\n")
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "name": "templated",
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACDEF",
                        "unpaired_msa": "protein.a3m",
                        "templates": [
                            {
                                "mmcif": "template.cif",
                                "query_indices": [1, 2, 3],
                                "template_indices": [1, 2, 3],
                            }
                        ],
                    }
                ],
            }
        )
    )
    return path


def _materialize(source: Path, model: str, options=None, ignored_templates=None):
    return materialize_native_input(
        source,
        capabilities(model),
        source.parent / f"out-{model}",
        seed=1,
        msa="none",
        options=options,
        ignored_templates=ignored_templates,
    )


def _templates_path(path: Path) -> object:
    (job,) = json.loads(path.read_text())
    (entry,) = job["sequences"]
    return entry["proteinChain"].get("templatesPath")


@pytest.mark.parametrize("model", _MODELS)
def test_a_template_is_refused_at_the_released_use_template(
    tmp_path: Path, model: str
) -> None:
    with pytest.raises(ValueError) as refusal:
        _materialize(_job(tmp_path), model)
    message = str(refusal.value)
    assert f"{model} cannot express templates in the common schema" in message
    assert "entity 'A'" in message
    assert "use_template=true" in message
    assert f"{IGNORE_TEMPLATES}=true" in message


@pytest.mark.parametrize("model", _MODELS)
def test_use_template_reads_it(tmp_path: Path, model: str) -> None:
    written = _materialize(_job(tmp_path), model, options={"use_template": True})
    assert _templates_path(written)


@pytest.mark.parametrize("model", _MODELS)
def test_ignore_templates_drops_and_records_it(tmp_path: Path, model: str) -> None:
    ignored: list = []
    written = _materialize(
        _job(tmp_path),
        model,
        options={IGNORE_TEMPLATES: True},
        ignored_templates=ignored,
    )
    assert _templates_path(written) is None
    (record,) = ignored
    assert record["chains"] == ["A"]
    assert record["field"] == "templates"
    assert record["resolved_path"] == str((tmp_path / "template.cif").resolve())
    assert "use_template=true" in record["reason"]


@pytest.mark.parametrize("model", _MODELS)
def test_reading_and_dropping_at_once_is_refused(tmp_path: Path, model: str) -> None:
    with pytest.raises(ValueError, match="set one of the two"):
        _materialize(
            _job(tmp_path),
            model,
            options={"use_template": True, IGNORE_TEMPLATES: True},
        )


@pytest.mark.parametrize("model", _MODELS)
def test_the_option_is_checked_while_planning(tmp_path: Path, model: str) -> None:
    backend = get_backend(model)
    job = _job(tmp_path)
    backend.validate_request(
        PredictionRequest(
            model=model,
            input=job,
            input_format="foldjax",
            seed=1,
            msa="none",
            options={IGNORE_TEMPLATES: True},
        )
    )
    with pytest.raises(ValueError, match=f"{IGNORE_TEMPLATES} must be a boolean"):
        backend.validate_request(
            PredictionRequest(
                model=model,
                input=job,
                input_format="foldjax",
                seed=1,
                msa="none",
                options={IGNORE_TEMPLATES: "yes"},
            )
        )
    with pytest.raises(ValueError, match="applies to FoldJAX common-schema input"):
        backend.validate_request(
            PredictionRequest(
                model=model,
                input=job,
                input_format="native",
                seed=1,
                msa="none",
                options={IGNORE_TEMPLATES: True},
            )
        )


def test_backends_without_the_gate_do_not_take_the_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=IGNORE_TEMPLATES):
        get_backend("alphafold3").validate_request(
            PredictionRequest(
                model="alphafold3",
                input=_job(tmp_path),
                input_format="foldjax",
                seed=1,
                msa="none",
                options={IGNORE_TEMPLATES: True},
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


@pytest.mark.parametrize("model", _MODELS)
def test_a_run_records_the_dropped_template_in_its_manifest(
    tmp_path: Path, model: str
) -> None:
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    source = _job(tmp_path)
    seen: list = []

    def request(options: dict, out: str) -> PredictionRequest:
        return PredictionRequest(
            model=model,
            input=source,
            weights=weights,
            output_dir=tmp_path / out,
            seed=3,
            msa="none",
            options=options,
            use_compile_cache=False,
        )

    with backend_override(model, _recorder(model, seen)):
        with pytest.raises(ValueError, match="use_template=true"):
            foldjax.predict(request({}, "refused"))
        assert seen == []
        foldjax.predict(request({IGNORE_TEMPLATES: True}, "dropped"))
        foldjax.predict(request({"use_template": True}, "read"))

    dropped, read = seen
    assert IGNORE_TEMPLATES not in dropped.options
    assert _templates_path(dropped.input) is None
    assert _templates_path(read.input)
    manifest = json.loads((tmp_path / "dropped" / MANIFEST_NAME).read_text())
    assert manifest["options"][IGNORE_TEMPLATES] is True
    (record,) = manifest["ignored_templates"]
    assert record["chains"] == ["A"]
    read_manifest = json.loads((tmp_path / "read" / MANIFEST_NAME).read_text())
    assert read_manifest["ignored_templates"] == []


def test_protenix_use_template_is_rendered_and_refused_where_upstream_refuses(
    tmp_path: Path,
) -> None:
    from foldjax.backends.protenix import ProtenixBackend

    backend = ProtenixBackend()
    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    request = PredictionRequest(
        model="protenix",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=1,
        msa="none",
        options={"use_template": True},
    )
    invocation = backend._native_invocation(request)
    assert "--use-template" in invocation.argv
    assert invocation.config_fields["use_template"] is True
    assert "--use-template" not in backend._native_invocation(
        PredictionRequest(
            model="protenix",
            input=job,
            weights=weights,
            output_dir=tmp_path / "out",
            seed=1,
            msa="none",
        )
    ).argv
    assert backend.cache_profile(request)["use_template"] is True
    with pytest.raises(ValueError, match="use_template is not supported by"):
        backend.validate_native_options(
            {"use_template": True, "model_name": "protenix_mini_esm_v0.5.0"}
        )
    with pytest.raises(ValueError, match="use_template must be a boolean"):
        backend.validate_native_options({"use_template": "yes"})
