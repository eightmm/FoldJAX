"""A common-schema template Protenix or OpenDDE would ignore is dropped as upstream.

Both upstreams read a chain's templates only under ``use_template``, released
false (Protenix ``configs/configs_inference.py:36`` and
``protenix/data/template/template_featurizer.py:710``; OpenDDE
``config/inference_defaults.py:28``), and otherwise ignore them silently. The
Protenix port used to read them anyway. The contract follows the nucleic-MSA
one (``tests/test_nucleic_msa.py``): by default the template is dropped with a
warning and recorded in the manifest's ``ignored_templates``; it is read with
``use_template=true`` and refused with ``ignore_templates=false``.

Requests below pass ``seed`` and ``msa`` explicitly and give every protein an
alignment, so they mean the same on a base where either default moves.
"""

from __future__ import annotations

import json
import warnings
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


def _assert_warned(caught) -> None:
    """One warning names the chain, the field and the file it drops."""
    messages = [str(warning.message) for warning in caught]
    assert any(
        "chain(s) A " in message
        and "templates" in message
        and "'template.cif'" in message
        for message in messages
    ), messages


@pytest.mark.parametrize("model", _MODELS)
def test_ignore_templates_false_refuses_it(tmp_path: Path, model: str) -> None:
    ignored: list = []
    with pytest.raises(ValueError) as refusal:
        _materialize(
            _job(tmp_path),
            model,
            options={IGNORE_TEMPLATES: False},
            ignored_templates=ignored,
        )
    message = str(refusal.value)
    assert f"{model} cannot express templates in the common schema" in message
    assert "entity 'A'" in message
    assert "use_template=true" in message
    assert f"{IGNORE_TEMPLATES}=false" in message
    assert ignored == []


@pytest.mark.parametrize(
    "options", [{"use_template": True}, {"use_template": True, IGNORE_TEMPLATES: False}]
)
@pytest.mark.parametrize("model", _MODELS)
def test_use_template_reads_it(tmp_path: Path, model: str, options: dict) -> None:
    ignored: list = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        written = _materialize(
            _job(tmp_path), model, options=options, ignored_templates=ignored
        )
    assert _templates_path(written)
    assert ignored == []


@pytest.mark.parametrize("options", [None, {IGNORE_TEMPLATES: True}])
@pytest.mark.parametrize("model", _MODELS)
def test_a_template_is_dropped_with_a_warning_and_a_record(
    tmp_path: Path, model: str, options: dict | None
) -> None:
    """The default, and an explicit true, fold without it as upstream does."""
    ignored: list = []
    with pytest.warns(UserWarning) as caught:
        written = _materialize(
            _job(tmp_path), model, options=options, ignored_templates=ignored
        )
    _assert_warned(caught)
    assert _templates_path(written) is None
    (record,) = ignored
    assert record["chains"] == ["A"]
    assert record["field"] == "templates"
    assert record["path"] == "template.cif"
    assert record["resolved_path"] == str((tmp_path / "template.cif").resolve())
    assert "use_template=true" in record["reason"]
    assert "as upstream does" in record["reason"]


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
    for value in (True, False):
        backend.validate_request(
            PredictionRequest(
                model=model,
                input=job,
                input_format="foldjax",
                seed=1,
                msa="none",
                options={IGNORE_TEMPLATES: value},
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
def test_a_run_records_the_dropped_template_unless_the_option_refuses(
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
            foldjax.predict(request({IGNORE_TEMPLATES: False}, "refused"))
        assert seen == []
        assert not (tmp_path / "refused" / MANIFEST_NAME).exists()
        with pytest.warns(UserWarning) as caught:
            foldjax.predict(request({}, "dropped"))
        _assert_warned(caught)
        with pytest.warns(UserWarning):
            foldjax.predict(request({IGNORE_TEMPLATES: True}, "explicit"))
        foldjax.predict(request({"use_template": True}, "read"))

    dropped, explicit, read = seen
    for ran in (dropped, explicit):
        assert IGNORE_TEMPLATES not in ran.options
        assert _templates_path(ran.input) is None
    assert _templates_path(read.input)
    for out in ("dropped", "explicit"):
        manifest = json.loads((tmp_path / out / MANIFEST_NAME).read_text())
        (record,) = manifest["ignored_templates"]
        assert record["chains"] == ["A"]
        assert record["resolved_path"] == str((tmp_path / "template.cif").resolve())
    default = json.loads((tmp_path / "dropped" / MANIFEST_NAME).read_text())
    assert IGNORE_TEMPLATES not in default["options"]
    asked = json.loads((tmp_path / "explicit" / MANIFEST_NAME).read_text())
    assert asked["options"][IGNORE_TEMPLATES] is True
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
