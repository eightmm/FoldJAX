"""Several common-schema jobs in one ``{"jobs": [...]}`` file.

The contract is "exactly as a directory of single-job files would": each job
runs as its own input, into ``<out>/<model>/<job name>``, with the same
requests, native inputs, resume and failure handling. Only provenance is
added -- the manifest, the failure record and ``plan`` name the file and the
job's index -- and errors name the job by index and name.

Backends are the real adapters with ``predict`` replaced, so the native input
each one would read is really written; no weights are loaded.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import foldjax
from foldjax.api import detect_input_format, resolve_request, resolve_requests
from foldjax.backends.base import Backend
from foldjax.cli import main
from foldjax.input import expand_jobs_file, is_jobs_document, read_jobs_file
from foldjax.manifest import MANIFEST_NAME
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override
from foldjax.schema import (
    JobSource,
    ModelCapabilities,
    PredictionFailure,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

_MODELS = ("boltz2", "protenix")


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Generated job documents go to the store; keep it per test."""
    home = tmp_path / "home"
    monkeypatch.setenv("FOLDJAX_HOME", str(home))
    return home


def _job(name: str, sequence: str, *, msa: str = "msa.a3m") -> dict:
    return {
        "name": name,
        "entities": [
            {"type": "protein", "id": "A", "sequence": sequence, "unpaired_msa": msa}
        ],
    }


_JOBS = [_job("alpha", "ACDEFG"), _job("beta", "MKTAYI")]


def _alignment(directory: Path) -> None:
    # Written once: rewriting it would change a dependency resume checks.
    if not (directory / "msa.a3m").exists():
        (directory / "msa.a3m").write_text(">query\nACDEFG\n>hit\nACDEFA\n")


def _jobs_file(directory: Path, jobs: list[dict] = _JOBS, name="jobs.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _alignment(directory)
    path = directory / name
    if path.suffix == ".json":
        path.write_text(json.dumps({"jobs": jobs}))
    else:
        import yaml

        path.write_text(yaml.safe_dump({"jobs": jobs}))
    return path


def _job_directory(directory: Path, jobs: list[dict] = _JOBS) -> Path:
    """The same jobs as separate files, one level below the jobs file.

    Their alignment path is spelled ``../msa.a3m``, so both spellings name one
    file and the native documents can be compared byte for byte.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        entities = [
            {**entity, "unpaired_msa": "../msa.a3m"} for entity in job["entities"]
        ]
        (directory / f"{job['name']}.json").write_text(
            json.dumps({**job, "entities": entities})
        )
    return directory


def _weights(tmp_path: Path) -> Path:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"not really weights")
    return weights


# -- parsing and refusal --------------------------------------------------------


def test_a_jobs_file_lists_its_jobs_in_order(tmp_path: Path) -> None:
    path = _jobs_file(tmp_path)
    assert [name for name, _job in read_jobs_file(path)] == ["alpha", "beta"]
    assert detect_input_format(path) == "foldjax"


def test_a_yaml_jobs_file_reads_the_same(tmp_path: Path) -> None:
    path = _jobs_file(tmp_path, name="jobs.yaml")
    assert read_jobs_file(path) == read_jobs_file(_jobs_file(tmp_path / "json"))


def test_a_top_level_list_stays_native(tmp_path: Path) -> None:
    """The list shape belongs to AlphaFold Server and Protenix/OpenDDE."""
    path = tmp_path / "list.json"
    path.write_text(json.dumps(_JOBS))
    assert detect_input_format(path) == "native"
    assert not is_jobs_document(_JOBS)
    assert not is_jobs_document({"jobs": [], "entities": []}), "a job, not a file"


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"jobs": _JOBS, "seeds": [1]}, r"unsupported top-level fields.*'seeds'"),
        ({"jobz": _JOBS, "jobs": _JOBS}, r"'jobz' \(did you mean 'jobs'\?\)"),
        ({"jobs": []}, "jobs must be a non-empty list"),
        ({"jobs": _JOBS[0]}, "jobs must be a non-empty list"),
        ({"jobs": [_JOBS[0], "beta"]}, r"jobs\[1\] must be a job mapping"),
        (
            {"jobs": [_JOBS[0], {"entities": _JOBS[1]["entities"]}]},
            r"jobs\[1\] needs a non-empty name",
        ),
        (
            {"jobs": [_JOBS[0], {**_JOBS[1], "name": "  "}]},
            r"jobs\[1\] needs a non-empty name",
        ),
        (
            {"jobs": [_JOBS[0], _JOBS[1], {**_JOBS[1]}]},
            r"jobs\[2\] \('beta'\) repeats the name of jobs\[1\]",
        ),
        (
            {"jobs": [_job("a/b", "ACD"), _job("a_b", "ACD")]},
            r"jobs\[1\] \('a_b'\) and jobs\[0\] \('a/b'\) would both write to the "
            r"output directory 'a_b'",
        ),
    ],
)
def test_a_malformed_jobs_file_is_refused_naming_the_job(
    tmp_path: Path, document: dict, message: str
) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        read_jobs_file(path)
    request = PredictionRequest(model="boltz2", inputs=(path,), seed=1)
    with pytest.raises(ValueError, match=message):
        resolve_requests(request, draw_seeds=False)


def test_a_scalar_request_names_the_plural_spelling(tmp_path: Path) -> None:
    """One run cannot be several, the rule a directory follows."""
    path = _jobs_file(tmp_path)
    with pytest.raises(ValueError, match=r"use inputs=\('jobs.json',\)"):
        resolve_request(PredictionRequest(model="boltz2", input=path, seed=1))


def test_an_explicit_native_format_is_not_split(tmp_path: Path) -> None:
    path = _jobs_file(tmp_path)
    (resolved,) = resolve_requests(
        PredictionRequest(
            model="boltz2",
            inputs=(path,),
            input_format="native",
            seed=1,
            weights=_weights(tmp_path),
            output_dir=tmp_path / "out",
        )
    )
    assert resolved.input == path
    assert resolved.source is None


def test_job_source_validates_itself(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index must be non-negative"):
        JobSource(path=tmp_path, index=-1, name="a")
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        JobSource(path=tmp_path, index=0, name=" ")
    path = _jobs_file(tmp_path)
    with pytest.raises(ValueError, match="cannot describe several inputs"):
        PredictionRequest(
            model="boltz2",
            inputs=(path,),
            source=JobSource(path=path, index=0, name="alpha"),
        )


# -- expansion --------------------------------------------------------------------


def test_each_job_is_written_as_its_own_document(tmp_path: Path, _home) -> None:
    path = _jobs_file(tmp_path / "src")
    (alpha, alpha_source), (beta, beta_source) = expand_jobs_file(path)

    assert (alpha.name, beta.name) == ("alpha.json", "beta.json")
    assert alpha.is_relative_to(_home / "runtime" / "jobs")
    assert alpha_source == JobSource(path=path, index=0, name="alpha")
    assert beta_source == JobSource(path=path, index=1, name="beta")
    written = json.loads(alpha.read_text())
    (entity,) = written["entities"]
    assert entity["unpaired_msa"] == str((tmp_path / "src" / "msa.a3m").absolute())
    assert {key: value for key, value in written.items() if key != "entities"} == {
        "name": "alpha"
    }


def test_an_unchanged_job_keeps_its_path_when_another_changes(tmp_path: Path) -> None:
    """The generated path is the resume identity, so it follows the job alone."""
    path = _jobs_file(tmp_path)
    (alpha, _), (beta, _) = expand_jobs_file(path)
    assert [item for item, _ in expand_jobs_file(path)] == [alpha, beta]

    _jobs_file(tmp_path, [_JOBS[1], {**_JOBS[0]}, _job("gamma", "ACD")])
    (beta_again, beta_source), (alpha_again, _), _ = expand_jobs_file(path)
    assert (alpha_again, beta_again) == (alpha, beta)
    assert beta_source.index == 0

    _jobs_file(tmp_path, [_job("alpha", "ACDEFGH"), _JOBS[1]])
    (edited, _), (unchanged, _) = expand_jobs_file(path)
    assert edited != alpha and edited.name == "alpha.json"
    assert unchanged == beta


# -- running ----------------------------------------------------------------------


def _recorder(model: str, seen: list, fail: str | None = None):
    base = provider(PORTS[model].backend)

    class Recorder(base):
        def predict(self, request):
            seen.append(request)
            if fail is not None and fail in request.output_dir.name:
                raise ValueError(f"{model} could not fold {fail}")
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


def _run(request: PredictionRequest, seen: dict[str, list], **kwargs):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for model in _MODELS:
            stack.enter_context(
                backend_override(model, _recorder(model, seen[model], **kwargs))
            )
        return foldjax.predict_batch(request)


def _batch(inputs: tuple[Path, ...], out: Path, **fields):
    return PredictionRequest(
        models=_MODELS,
        inputs=inputs,
        output_dir=out,
        weights=None,
        seed=4,
        msa="none",
        use_compile_cache=False,
        **fields,
    )


def _layout(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def _without(request: PredictionRequest, *names: str) -> dict:
    return {
        field.name: getattr(request, field.name)
        for field in dataclasses.fields(request)
        if field.name not in names
    }


def test_a_jobs_file_runs_exactly_as_a_directory_of_its_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = _weights(tmp_path)
    monkeypatch.setattr(
        "foldjax.assets.resolve_weights", lambda model, profile=None: weights
    )
    jobs_file = _jobs_file(tmp_path / "file")
    directory = _job_directory(tmp_path / "file" / "separate")

    from_file = resolve_requests(_batch((jobs_file,), tmp_path / "a"))
    from_dir = resolve_requests(_batch((directory,), tmp_path / "b"))
    assert len(from_file) == len(from_dir) == 4
    for left, right in zip(from_file, from_dir, strict=True):
        assert left.input.name == right.input.name
        assert left.output_dir.relative_to(tmp_path / "a") == (
            right.output_dir.relative_to(tmp_path / "b")
        )
        assert _without(left, "input", "source", "output_dir") == _without(
            right, "input", "source", "output_dir"
        )
        assert right.source is None
    assert [request.source.index for request in from_file] == [0, 1, 0, 1]
    assert [
        str(request.output_dir.relative_to(tmp_path / "a")) for request in from_file
    ] == ["boltz2/alpha", "boltz2/beta", "protenix/alpha", "protenix/beta"]

    seen_file: dict[str, list] = {model: [] for model in _MODELS}
    seen_dir: dict[str, list] = {model: [] for model in _MODELS}
    report_file = _run(_batch((jobs_file,), tmp_path / "a"), seen_file)
    report_dir = _run(_batch((directory,), tmp_path / "b"), seen_dir)
    assert not report_file.failures and not report_dir.failures
    assert len(report_file.results) == len(report_dir.results) == 4

    assert _layout(tmp_path / "a") == _layout(tmp_path / "b")
    for model in _MODELS:
        for left, right in zip(seen_file[model], seen_dir[model], strict=True):
            # The native document each backend read is the same file.
            assert left.input.read_bytes() == right.input.read_bytes()
            assert left.input.relative_to(tmp_path / "a") == (
                right.input.relative_to(tmp_path / "b")
            )
            assert _without(left, "input", "output_dir", "source") == _without(
                right, "input", "output_dir", "source"
            )

    for index, name in enumerate(("alpha", "beta")):
        for model in _MODELS:
            file_manifest = json.loads(
                (tmp_path / "a" / model / name / MANIFEST_NAME).read_text()
            )
            dir_manifest = json.loads(
                (tmp_path / "b" / model / name / MANIFEST_NAME).read_text()
            )
            source = file_manifest["input"].pop("source")
            assert source["path"] == str(jobs_file)
            assert (source["index"], source["name"]) == (index, name)
            assert source["sha256"] is not None
            assert "source" not in dir_manifest["input"]
            for key in ("samples", "best", "options", "seeds", "sampling", "msa"):
                assert file_manifest[key] == dir_manifest[key], key
            for key in ("ignored_msas", "ignored_templates", "ignored_constraints"):
                assert file_manifest[key] == dir_manifest[key], key


class _PlainBackend(Backend):
    """A backend with no weight session, so a resumed run can be reused.

    The real adapters anchor their checkpoint before reuse and cannot verify a
    placeholder file, so resume is exercised the way the resume suite does it
    (``tests/test_resume_manifest.py``).
    """

    def __init__(self, name: str, calls: list[tuple[str, str]]) -> None:
        self.name = name
        self.calls = calls

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model=self.name, input_formats=("foldjax", "native"))

    def predict(self, request: PredictionRequest) -> PredictionResult:
        self.calls.append((self.name, request.output_dir.name))
        structure = Path(request.output_dir) / f"native-{request.seed}.cif"
        structure.write_text("data_mock\n#\n", encoding="utf-8")
        return PredictionResult(
            model=self.name,
            samples=(
                PredictionSample(
                    seed=request.seed, structure_path=structure, scores={"c": 0.5}
                ),
            ),
            output_dir=request.output_dir,
        )


def _resume(request: PredictionRequest, calls: list):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for model in _MODELS:
            backend = _PlainBackend(model, calls)
            stack.enter_context(backend_override(model, lambda b=backend: b))
        return foldjax.predict_batch(request)


def test_a_jobs_file_resumes_per_job(tmp_path: Path) -> None:
    weights = _weights(tmp_path)
    jobs_file = _jobs_file(tmp_path / "file")
    request = PredictionRequest(
        models=_MODELS,
        inputs=(jobs_file,),
        output_dir=tmp_path / "out",
        seed=4,
        msa="none",
        use_compile_cache=False,
        resume=True,
    )

    # One explicit checkpoint cannot be shared by two models, so both resolve
    # their managed one -- this placeholder.
    def run(calls: list):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "foldjax.assets.resolve_weights", lambda model, profile=None: weights
            )
            return _resume(request, calls)

    first: list = []
    assert not run(first).skipped
    assert sorted(first) == [
        ("boltz2", "alpha"),
        ("boltz2", "beta"),
        ("protenix", "alpha"),
        ("protenix", "beta"),
    ]
    again: list = []
    assert len(run(again).skipped) == 4
    assert again == []

    # Editing one job reruns that job alone, on every model, however the
    # others moved in the file.
    _jobs_file(tmp_path / "file", [_job("beta", "MKTAYIA"), _JOBS[0]])
    edited: list = []
    report = run(edited)
    assert sorted(path.name for path in report.skipped) == ["alpha", "alpha"]
    assert sorted(edited) == [("boltz2", "beta"), ("protenix", "beta")]


def test_a_failed_job_is_recorded_with_its_source_and_the_rest_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = _weights(tmp_path)
    monkeypatch.setattr(
        "foldjax.assets.resolve_weights", lambda model, profile=None: weights
    )
    # beta asks for a field neither backend's dialect has: refused per run, at
    # materialization, exactly as in a file of its own.
    bad = {**_JOBS[1], "entities": [{**_JOBS[1]["entities"][0], "colour": "red"}]}
    jobs_file = _jobs_file(tmp_path / "file", [_JOBS[0], bad])
    seen: dict[str, list] = {model: [] for model in _MODELS}
    report = _run(_batch((jobs_file,), tmp_path / "out", on_error="continue"), seen)

    assert [result.output_dir.name for result in report.results] == ["alpha", "alpha"]
    assert len(report.failures) == 2
    for failure in report.failures:
        assert failure.error.startswith(f"{jobs_file} jobs[1] ('beta'): ")
        assert "'colour'" in failure.error
        assert failure.source == JobSource(path=jobs_file, index=1, name="beta")
        assert failure.summary()["source"] == {
            "path": str(jobs_file),
            "index": 1,
            "name": "beta",
        }
        assert failure.output_dir.name == "beta"
    recorded = json.loads((tmp_path / "out" / "foldjax_failures.json").read_text())
    assert [entry["source"]["index"] for entry in recorded] == [1, 1]


def test_a_directory_failure_record_is_unchanged(tmp_path: Path) -> None:
    failure = PredictionFailure(
        model="boltz2", input=tmp_path / "a.json", error="x", error_type="ValueError"
    )
    assert "source" not in failure.summary()


def test_the_cli_plans_every_job_of_a_jobs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    jobs_file = _jobs_file(tmp_path / "file")
    weights = _weights(tmp_path)
    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(jobs_file),
                "--weights",
                str(weights),
                "--output-dir",
                str(tmp_path / "out"),
                "--seed",
                "1",
                "--no-cache",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [entry["output_dir"] for entry in payload] == [
        str(tmp_path / "out" / "boltz2" / "alpha"),
        str(tmp_path / "out" / "boltz2" / "beta"),
    ]
    assert [entry["source"] for entry in payload] == [
        {"path": str(jobs_file), "index": 0, "name": "alpha"},
        {"path": str(jobs_file), "index": 1, "name": "beta"},
    ]


def test_the_cli_hands_a_lone_jobs_file_over_as_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    jobs_file = _jobs_file(tmp_path / "file")
    seen = {}

    def fake_predict(request):
        from foldjax.schema import BatchReport

        seen["request"] = request
        return BatchReport()

    monkeypatch.setattr("foldjax.cli.predict_batch", fake_predict)
    assert main(["predict", "--model", "boltz2", "--input", str(jobs_file)]) == 0
    assert seen["request"].input is None
    assert seen["request"].inputs == (jobs_file,)
