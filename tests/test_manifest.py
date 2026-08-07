"""Every run records what produced it.

A directory of .cif files cannot say which model, checkpoint, schedule or seed
made them, and neither can FoldJAX once the request is gone. Two directories
from two different schedules look identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import foldjax
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample


class _Recorder(OpenDDEBackend):
    def predict(self, request):
        path = request.output_dir / f"s{request.seed}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
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
    path.write_text(
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]})
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
    # The input is hashed, not just named: job files are edited in place far
    # more often than they are renamed.
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["samples"][0]["scores"] == {"ptm": 0.5}
    assert manifest["foldjax"] == foldjax.__version__
    assert manifest["runtime"]["jax"]


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
        json.dumps({"entities": [{"type": "protein", "id": "A", "sequence": "ACDE"}]})
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


def test_a_manifest_that_cannot_be_written_does_not_fail_the_run(
    tmp_path: Path, monkeypatch
) -> None:
    """The prediction succeeded; losing its provenance must not undo that."""
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
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_text", refuse)
    assert manifest.write(request, PredictionResult(model="opendde"), tmp_path) is None


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
