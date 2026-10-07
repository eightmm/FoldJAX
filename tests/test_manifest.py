"""Every run records what produced it.

A directory of .cif files cannot say which model, checkpoint, schedule or seed
made them, and neither can FoldJAX once the request is gone. Two directories
from two different schedules look identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foldjax
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override
from foldjax.schema import (
    PaddingConfig,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


class _Recorder(OpenDDEBackend):
    def predict(self, request):
        path = request.output_dir / f"s{request.seed}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data_mock\n#\n", encoding="utf-8")
        return PredictionResult(
            model="opendde",
            samples=(
                PredictionSample(
                    seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                ),
            ),
            output_dir=request.output_dir,
        )


def _job(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    # An alignment, because a bare protein is refused under the default msa.
    (tmp_path / "job.a3m").write_text(">query\nACD\n")
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACD",
                        "unpaired_msa": "job.a3m",
                    }
                ]
            }
        )
    )
    return path


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.jax"
    path.write_bytes(b"not really weights")
    return path


def test_a_run_records_model_weights_schedule_and_seed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        profile="released",
        output_dir=out,
        seed=17,
        num_samples=3,
        num_steps=40,
        use_compile_cache=False,
    )
    with backend_override("opendde", _Recorder):
        foldjax.predict(request)

    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["model"] == "opendde"
    assert manifest["seeds"] == [17]
    assert manifest["sampling"] == {"num_samples": 3, "num_steps": 40}
    assert manifest["weights"]["label"] == "weights.jax"
    assert manifest["weights"]["profile"] == "released"
    # The input is hashed, not just named: job files are edited in place far
    # more often than they are renamed.
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["samples"][0]["scores"] == {"ptm": 0.5}
    assert manifest["foldjax"] == foldjax.__version__
    source = manifest["foldjax_source"]
    assert source["version"] == foldjax.__version__
    assert source["git_describe"] is None or isinstance(source["git_describe"], str)
    assert manifest["runtime"]["jax"]


def test_manifest_preserves_same_shape_boltz_static_executable_identity(
    tmp_path: Path,
) -> None:
    from foldjax import manifest

    target = {"tokens": 256, "atoms": 512, "msa": 64}
    shape_profile = {
        "primary": {
            "target": target,
            "static": {
                "use_template": False,
                "recompute_nonpolymer_frames": False,
            },
        },
        "affinity": {
            "target": target,
            "static": {
                "use_template": True,
                "recompute_nonpolymer_frames": True,
            },
        },
    }
    request = PredictionRequest(
        seed=0,
        model="boltz2",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
        padding=PaddingConfig(),
        use_compile_cache=False,
    )
    result = PredictionResult(
        model="boltz2",
        samples=(),
        output_dir=request.output_dir,
        shape_profile=shape_profile,
    )

    payload = manifest.describe_run(request, result)

    assert payload["shape_profile"] == shape_profile
    assert (
        payload["shape_profile"]["primary"]["target"]
        == payload["shape_profile"]["affinity"]["target"]
    )
    assert (
        payload["shape_profile"]["primary"]["static"]
        != payload["shape_profile"]["affinity"]["static"]
    )


def test_editing_the_job_changes_the_recorded_digest(tmp_path: Path) -> None:
    job = _job(tmp_path)
    out = tmp_path / "out"

    def run() -> str:
        request = PredictionRequest(
            model="opendde",
            input=job,
            weights=_weights(tmp_path),
            output_dir=out,
            seed=1,
            use_compile_cache=False,
        )
        with backend_override("opendde", _Recorder):
            foldjax.predict(request)
        return json.loads((out / MANIFEST_NAME).read_text())["input"]["sha256"]

    before = run()
    job.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACDE",
                        "unpaired_msa": "job.a3m",
                    }
                ]
            }
        )
    )
    assert run() != before


def test_every_seed_is_recorded_and_each_gets_its_own_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out"
    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=out,
        seeds=(2, 3),
        use_compile_cache=False,
    )
    with backend_override("opendde", _Recorder):
        foldjax.predict(request)

    top = json.loads((out / MANIFEST_NAME).read_text())
    assert top["seeds"] == [2, 3]
    assert len(top["samples"]) == 2
    for seed in (2, 3):
        per_seed = json.loads((out / f"seed_{seed}" / MANIFEST_NAME).read_text())
        assert per_seed["seeds"] == [seed]


def test_a_manifest_the_disk_refuses_is_an_error_that_leaks_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """A run without its completion marker has not finished (ENOSPC, EROFS).

    The error names the file and the OS's reason, never the text it carried.
    """
    import errno

    from foldjax import manifest

    job = _job(tmp_path)
    weights = _weights(tmp_path)
    request = PredictionRequest(
        model="opendde",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=1,
        use_compile_cache=False,
    )

    def refuse(*args, **kwargs):
        raise OSError("do-not-leak-this-secret")

    monkeypatch.setattr(Path, "write_text", refuse)
    with pytest.raises(OSError) as caught:
        manifest.write(request, PredictionResult(model="opendde"), tmp_path)
    assert "could not write the run manifest" in str(caught.value)
    assert "do-not-leak-this-secret" not in str(caught.value)
    assert caught.value.__cause__ is None

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full)
    with pytest.raises(OSError) as caught:
        manifest.write(request, PredictionResult(model="opendde"), tmp_path)
    # Kept, so the command line can name the store and the way to reclaim it.
    assert caught.value.errno == errno.ENOSPC
    assert not (tmp_path / manifest.MANIFEST_NAME).exists()


def test_describe_run_without_directory_uses_absolute_artifact_paths(
    tmp_path: Path,
) -> None:
    from foldjax import manifest

    output = tmp_path / "out"
    structure = output / "sample.cif"
    native = output / "opendde_input.json"
    output.mkdir()
    structure.write_text("data_mock\n#\n", encoding="utf-8")
    native.write_text("{}\n", encoding="utf-8")
    request = PredictionRequest(
        seed=0,
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=output,
        use_compile_cache=False,
    )
    result = PredictionResult(
        model="opendde",
        samples=(
            PredictionSample(
                seed=0,
                structure_path=structure,
                scores={"ranking_score": 0.5},
            ),
        ),
        output_dir=output,
    )

    payload = manifest.describe_run(request, result, native_input=native)

    assert payload["artifact_paths"] == "absolute"
    assert payload["input"]["native"] == str(native.resolve())
    assert payload["samples"][0]["structure_path"] == str(structure.resolve())
    assert payload["best"]["structure_path"] == str(structure.resolve())


def test_a_manifest_symlink_cannot_rewrite_an_external_file(tmp_path: Path) -> None:
    from foldjax import manifest

    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "user-owned.json"
    external.write_text("do not replace\n")
    destination = output / MANIFEST_NAME
    destination.symlink_to(external)
    request = PredictionRequest(
        seed=0,
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=output,
        use_compile_cache=False,
    )

    written = manifest.write(
        request,
        PredictionResult(model="opendde", output_dir=output),
        output,
    )

    assert written == destination
    assert external.read_text() == "do not replace\n"
    assert destination.is_file() and not destination.is_symlink()
    assert json.loads(destination.read_text())["model"] == "opendde"


def test_the_manifest_names_the_job_that_was_asked_for(tmp_path: Path) -> None:
    """Common-schema input is translated before the backend sees it.

    The translated file lives inside the output directory being described, so
    a manifest naming only it says nothing about which job produced the
    directory. The original is recorded, with the generated dialect alongside.
    """
    job = _job(tmp_path)
    out = tmp_path / "out"
    request = PredictionRequest(
        model="opendde",
        input=job,
        weights=_weights(tmp_path),
        output_dir=out,
        seed=1,
        use_compile_cache=False,
    )
    with backend_override("opendde", _Recorder):
        foldjax.predict(request)

    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["input"]["path"] == str(job)
    assert manifest["input"]["format"] == "foldjax"
    assert manifest["input"]["sha256"] == _sha256(job)
    # The generated dialect is recorded too, since that is what actually ran.
    assert manifest["input"]["native"].endswith("opendde_input.json")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_manifest_records_what_the_run_cost(tmp_path: Path, monkeypatch) -> None:
    """Time and peak device memory, because from outside neither is visible.

    JAX preallocates most of the card, so a peak read with `nvidia-smi` is the
    size of the reservation and reports the same number for a 250-token job and
    a 3000-token one. `peak_bytes_in_use` is the bytes actually held.
    """
    from foldjax import manifest

    monkeypatch.setattr(manifest, "device_peak_bytes", lambda: 12_884_901_888)
    request = PredictionRequest(
        seed=0,
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path,
        use_compile_cache=False,
    )
    result = PredictionResult(
        model=request.model, samples=(), output_dir=tmp_path, raw={}
    )
    payload = manifest.describe_run(
        request,
        result,
        cost={"seconds": 41.5, "peak_bytes": manifest.device_peak_bytes()},
    )
    assert payload["cost"] == {"seconds": 41.5, "peak_bytes": 12_884_901_888}


def test_a_run_without_a_device_still_records_its_time(tmp_path: Path) -> None:
    """A CPU-only runtime has no peak to report, and that is not a failure."""
    from foldjax import manifest

    request = PredictionRequest(
        seed=0,
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path,
        use_compile_cache=False,
    )
    payload = manifest.describe_run(
        request,
        PredictionResult(model="opendde", samples=(), output_dir=tmp_path, raw={}),
        cost={"seconds": 1.0, "peak_bytes": None},
    )
    assert payload["cost"]["peak_bytes"] is None
    assert payload["cost"]["seconds"] == 1.0


def test_manifest_serialization_failure_does_not_fail_a_prediction(
    tmp_path: Path,
) -> None:
    from foldjax import manifest

    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
        use_compile_cache=False,
    )
    result = PredictionResult(model="opendde", samples=(), output_dir=tmp_path)

    with pytest.warns(RuntimeWarning, match="could not record run provenance"):
        assert (
            manifest.write(
                request,
                result,
                tmp_path / "out",
                cost={"seconds": object()},
            )
            is None
        )
    assert not (tmp_path / "out" / manifest.MANIFEST_NAME).exists()


def test_manifest_runtime_introspection_failure_does_not_fail_a_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax import manifest

    request = PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
        use_compile_cache=False,
    )
    result = PredictionResult(model="opendde", samples=(), output_dir=tmp_path)
    monkeypatch.setattr(
        manifest,
        "describe_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )

    with pytest.warns(RuntimeWarning, match="could not record run provenance"):
        assert manifest.write(request, result, tmp_path / "out") is None


def test_nested_credentials_are_redacted_without_erasing_public_types(
    tmp_path: Path,
) -> None:
    from foldjax import manifest
    from foldjax.redaction import REDACTED

    secrets = ("outer-secret", "bearer-secret", "nested-secret")
    request = PredictionRequest(
        seed=0,
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
        use_compile_cache=False,
        options={
            "msa_server_password": secrets[0],
            "refreshToken": "camel-case-secret",
            "msa_api_key_value": "native-option-secret",
            "msa_server_url": (
                "https://user:url-secret@example.test/search?"
                "access_token=query-secret&mode=fast"
            ),
            "search": {
                "headers": {
                    "Authorization": secrets[1],
                    "Cookie": "session=private-cookie",
                    "Accept": "json",
                },
                "header_lines": [
                    f"Authorization: Bearer {secrets[1]}",
                    "X-API-Key: list-secret",
                    "Accept: application/json",
                ],
                "header_blob": (
                    f"Authorization: Bearer {secrets[1]}\n"
                    "Accept: application/json\n"
                ),
                "attempts": 3,
                "fallbacks": [{"api_key": secrets[2]}, True],
            },
            "signed_url": (
                "https://example.test/object?X-Amz-Signature=aws-secret&"
                "sig=azure-secret&mode=fast"
            ),
            "generic_api_url": "https://example.test/search?key=url-api-secret",
            "auth_header": "private-auth-header",
            "header_pairs": [
                ("Authorization", "Bearer tuple-secret"),
                ("Accept", "application/json"),
            ],
            "cli_args": [
                "--api-key=equals-secret",
                "--password",
                "following-secret",
                "--mode",
                "fast",
            ],
            "tokenizer": "esm",
            "enabled": False,
        },
    )
    payload = manifest.describe_run(
        request,
        PredictionResult(model="opendde", samples=(), output_dir=tmp_path),
    )
    options = payload["options"]

    assert options["msa_server_password"] == REDACTED
    assert options["refreshToken"] == REDACTED
    assert options["msa_api_key_value"] == REDACTED
    assert options["msa_server_url"] == (
        "https://example.test/search?access_token=%5BREDACTED%5D&mode=fast"
    )
    assert options["search"]["headers"] == {
        "Authorization": REDACTED,
        "Cookie": REDACTED,
        "Accept": "json",
    }
    assert options["search"]["header_lines"] == [
        "Authorization: [REDACTED]",
        "X-API-Key: [REDACTED]",
        "Accept: application/json",
    ]
    assert options["search"]["header_blob"] == (
        "Authorization: [REDACTED]\nAccept: application/json\n"
    )
    assert options["search"]["fallbacks"] == [{"api_key": REDACTED}, True]
    assert options["auth_header"] == REDACTED
    assert options["header_pairs"] == [
        ["Authorization", REDACTED],
        ["Accept", "application/json"],
    ]
    assert options["cli_args"] == [
        "--api-key=[REDACTED]",
        "--password",
        REDACTED,
        "--mode",
        "fast",
    ]
    assert options["signed_url"] == (
        "https://example.test/object?X-Amz-Signature=%5BREDACTED%5D&"
        "sig=%5BREDACTED%5D&mode=fast"
    )
    assert options["generic_api_url"] == (
        "https://example.test/search?key=%5BREDACTED%5D"
    )
    assert options["search"]["attempts"] == 3
    assert options["tokenizer"] == "esm"
    assert options["enabled"] is False
    rendered = json.dumps(payload)
    assert all(
        secret not in rendered
        for secret in (
            *secrets,
            "camel-case-secret",
            "native-option-secret",
            "url-secret",
            "query-secret",
            "url-api-secret",
            "list-secret",
            "tuple-secret",
            "equals-secret",
            "following-secret",
        )
    )


def _unseeded(tmp_path: Path, **fields) -> PredictionRequest:
    return PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "out",
        use_compile_cache=False,
        msa="single",
        **fields,
    )


def test_an_unseeded_run_draws_prints_and_records_its_seed(tmp_path: Path) -> None:
    """Upstream OpenDDE draws `random.randint` for a job without modelSeeds.

    FoldJAX draws too, rather than running 0, and the draw is what makes the
    run repeatable afterwards: it is printed and recorded beside its source.
    """
    import io

    from foldjax import progress

    stream = io.StringIO()
    progress.enable(stream)
    try:
        with backend_override("opendde", _Recorder):
            result = foldjax.predict(_unseeded(tmp_path))
    finally:
        progress.disable()

    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    (drawn,) = manifest["seeds"]
    assert manifest["seed_source"] == "random"
    assert 0 <= drawn < 2**31
    assert [sample.seed for sample in result.samples] == [drawn]
    assert f"drew {drawn} " in stream.getvalue()
    assert f"--seed {drawn} repeats it" in stream.getvalue()


def test_resume_reuses_the_drawn_seed_instead_of_drawing_again(
    tmp_path: Path,
) -> None:
    """A fresh draw would never match the recorded seed, so resume reuses it."""
    from foldjax.api import resolve_request

    with backend_override("opendde", _Recorder):
        first = foldjax.predict(_unseeded(tmp_path, resume=True))
    (drawn,) = [sample.seed for sample in first.samples]

    resumed = resolve_request(_unseeded(tmp_path, resume=True))
    assert (resumed.seed, resumed.seed_source) == (drawn, "random")
    fresh = {resolve_request(_unseeded(tmp_path)).seed for _ in range(4)}
    assert fresh != {drawn}


def test_a_resumed_random_seed_is_not_announced_as_drawn(tmp_path: Path) -> None:
    """The seed a resume takes back from the manifest was not drawn again."""
    import io

    from foldjax import progress

    with backend_override("opendde", _Recorder):
        first = foldjax.predict(_unseeded(tmp_path, resume=True))
    (drawn,) = [sample.seed for sample in first.samples]

    stream = io.StringIO()
    progress.enable(stream)
    try:
        with backend_override("opendde", _Recorder):
            foldjax.predict(_unseeded(tmp_path, resume=True))
    finally:
        progress.disable()
    assert "drew" not in stream.getvalue()

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    stream = io.StringIO()
    progress.enable(stream)
    try:
        with backend_override("opendde", _Recorder):
            foldjax.predict(_unseeded(fresh, resume=True))
    finally:
        progress.disable()
    assert "drew" in stream.getvalue()
    assert f"drew {drawn} " not in stream.getvalue()


def test_a_given_seed_is_recorded_as_the_callers(tmp_path: Path) -> None:
    with backend_override("opendde", _Recorder):
        foldjax.predict(_unseeded(tmp_path, seed=0))
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["seeds"] == [0]
    assert manifest["seed_source"] == "user"


@pytest.mark.parametrize(
    ("model", "seed"), [("protenix", 101), ("openfold3", 42)]
)
def test_an_unseeded_request_takes_the_upstream_fixed_seed(
    tmp_path: Path, model: str, seed: int
) -> None:
    from foldjax.api import resolve_request

    weights = tmp_path / "weights"
    weights.mkdir()
    resolved = resolve_request(
        PredictionRequest(
            model=model,
            input=_job(tmp_path),
            weights=weights,
            output_dir=tmp_path / "out",
            use_compile_cache=False,
            num_seeds=2,
        )
    )
    assert resolved.seed_source == "upstream"
    assert resolved.resolved_seeds == (seed, seed + 1)


def _native_opendde(tmp_path: Path, *seed_lists) -> Path:
    path = tmp_path / "native.json"
    jobs = []
    for index, seeds in enumerate(seed_lists):
        job = {
            "name": f"job{index}",
            "sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}],
        }
        if seeds is not None:
            job["modelSeeds"] = list(seeds)
        jobs.append(job)
    path.write_text(json.dumps(jobs))
    return path


def test_opendde_runs_every_model_seed_the_native_job_names(tmp_path: Path) -> None:
    """Upstream OpenDDE runs the job's modelSeeds when --seeds is unset.

    The adapter used to pass the request's seed (0 by default) as `--seed`,
    which the runner prefers over modelSeeds, so a two-seed job ran once
    under a seed it never named.
    """
    seen: list[int] = []

    class Seen(_Recorder):
        def predict(self, request):
            seen.append(request.seed)
            argv = self._native_invocation(request).argv
            assert argv[argv.index("--seed") + 1] == str(request.seed)
            return super().predict(request)

    out = tmp_path / "out"
    with backend_override("opendde", Seen):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_native_opendde(tmp_path, (5, 9)),
                weights=_weights(tmp_path),
                output_dir=out,
                use_compile_cache=False,
            )
        )

    assert seen == [5, 9]
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert (manifest["seeds"], manifest["seed_source"]) == ([5, 9], "job")


@pytest.mark.parametrize("model", ["opendde", "alphafold3"])
def test_model_seeds_resolve_like_upstream(tmp_path: Path, model: str) -> None:
    from foldjax.api import resolve_request

    weights = tmp_path / "weights"
    weights.mkdir()

    def resolve(path: Path, **fields) -> PredictionRequest:
        return resolve_request(
            PredictionRequest(
                model=model,
                input=path,
                weights=weights,
                output_dir=tmp_path / "out",
                use_compile_cache=False,
                **fields,
            )
        )

    one = resolve(_native_opendde(tmp_path, (7,)), num_seeds=2)
    assert (one.resolved_seeds, one.seed_source) == ((7, 8), "job")
    given = resolve(_native_opendde(tmp_path, (7,)), seed=3)
    assert (given.resolved_seeds, given.seed_source) == ((3,), "user")
    unnamed = resolve(_native_opendde(tmp_path, None))
    assert unnamed.seed_source == "random"
    with pytest.raises(ValueError, match="num_seeds counts up from one seed"):
        resolve(_native_opendde(tmp_path, (1, 2)), num_seeds=2)
    with pytest.raises(ValueError, match="name different modelSeeds"):
        resolve(_native_opendde(tmp_path, (1,), (2,)))
    # A common-schema job has no modelSeeds: the run draws, as upstream does
    # for a job without them.
    assert resolve(_job(tmp_path)).seed_source == "random"


@pytest.mark.parametrize("model", ["boltz2", "esmfold2"])
def test_an_unseeded_request_draws_where_upstream_seeds_nothing(
    tmp_path: Path, model: str
) -> None:
    from foldjax.api import resolve_request

    weights = tmp_path / "weights"
    weights.mkdir()
    request = PredictionRequest(
        model=model,
        input=_job(tmp_path),
        weights=weights,
        output_dir=tmp_path / "out",
        use_compile_cache=False,
    )
    draws = {resolve_request(request).seed for _ in range(8)}
    assert len(draws) > 1
    planned = resolve_request(request, draw_seeds=False)
    assert (planned.seed, planned.seed_source) == (None, "random")
    with pytest.raises(ValueError, match="not resolved"):
        planned.resolved_seeds


def _fake_git(monkeypatch, *, toplevel: Path, described: str = "v0.1.0-3-gabc1234"):
    """Answer the two git calls `source_describe` makes, recording each."""
    import subprocess

    from foldjax import manifest

    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        output = str(toplevel) if "rev-parse" in command else described
        return subprocess.CompletedProcess(command, 0, stdout=output + "\n")

    monkeypatch.setattr(manifest.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(manifest.subprocess, "run", run)
    return calls


@pytest.fixture
def _fresh_source_describe():
    from foldjax import manifest

    manifest.source_describe.cache_clear()
    yield
    manifest.source_describe.cache_clear()


def test_a_checkout_records_its_git_describe(monkeypatch, _fresh_source_describe):
    from foldjax import manifest

    package = Path(manifest.__file__).resolve().parent
    calls = _fake_git(monkeypatch, toplevel=package.parents[1])

    assert manifest.source_describe() == "v0.1.0-3-gabc1234"
    assert calls[1][0][-3:] == ["describe", "--always", "--dirty"]
    # `--dirty` refreshes the index unless optional locks are off, and the
    # checkout may be shared with other processes.
    assert all(env["GIT_OPTIONAL_LOCKS"] == "0" for _command, env in calls)


def test_an_enclosing_unrelated_repository_lends_no_revision(
    monkeypatch, tmp_path: Path, _fresh_source_describe
) -> None:
    """A venv inside some other checkout must not report that checkout."""
    from foldjax import manifest

    calls = _fake_git(monkeypatch, toplevel=tmp_path)

    assert manifest.source_describe() is None
    assert len(calls) == 1


def test_no_git_or_a_failing_git_records_none(
    monkeypatch, _fresh_source_describe
) -> None:
    import subprocess

    from foldjax import manifest

    monkeypatch.setattr(manifest.shutil, "which", lambda name: None)
    assert manifest.source_describe() is None

    manifest.source_describe.cache_clear()
    monkeypatch.setattr(manifest.shutil, "which", lambda name: "/usr/bin/git")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(manifest.subprocess, "run", fail)
    assert manifest.source_describe() is None


def test_esmfold2_binds_the_ccd_beside_its_checkpoint(tmp_path: Path) -> None:
    """An all-biomolecule job is featurized from it, so resume must see it."""
    from foldjax import manifest

    root = tmp_path / "esmfold2"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text("{}")
    request = PredictionRequest(
        model="esmfold2",
        input=_job(tmp_path),
        weights=root,
        output_dir=tmp_path / "out",
        seed=1,
        use_compile_cache=False,
        options={"no_language_model": True},
    )

    paths, missing = manifest._esmfold2_weight_assets(request)
    assert root / "ccd.pkl" in missing and root / "ccd.pkl" not in paths

    (root / "ccd.pkl").write_bytes(b"ccd")
    paths, missing = manifest._esmfold2_weight_assets(request)
    assert root / "ccd.pkl" in paths and not missing
