"""The one-file-in workflow: auto-detected input and model-neutral knobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foldjax
from foldjax.api import detect_input_format
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.input import read_job_document
from foldjax.registry import backend_override
from foldjax.schema import PredictionRequest

_JOB = {"entities": [{"type": "protein", "id": "A", "sequence": "ACD"}]}


def _weights(tmp_path: Path) -> Path:
    path = tmp_path / "weights.jax"
    path.touch()
    return path


# --------------------------------------------------------------------------
# input format
# --------------------------------------------------------------------------


def test_common_schema_is_detected_in_both_json_and_yaml(tmp_path: Path) -> None:
    as_json = tmp_path / "job.json"
    as_json.write_text(json.dumps(_JOB))
    as_yaml = tmp_path / "job.yaml"
    as_yaml.write_text("entities:\n  - type: protein\n    id: A\n    sequence: ACD\n")
    assert detect_input_format(as_json) == "foldjax"
    assert detect_input_format(as_yaml) == "foldjax"
    assert read_job_document(as_json) == read_job_document(as_yaml)


def test_native_dialects_are_not_mistaken_for_the_common_schema(
    tmp_path: Path,
) -> None:
    # Protenix-family native input is a list of jobs, and Boltz native is either
    # a mapping with `sequences` or a FASTA. None has `entities`.
    protenix = tmp_path / "native.json"
    protenix.write_text(json.dumps([{"name": "x", "sequences": []}]))
    boltz = tmp_path / "native.yaml"
    boltz.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n")
    fasta = tmp_path / "job.fasta"
    fasta.write_text(">protein|name=A\nACD\n")

    assert detect_input_format(protenix) == "native"
    assert detect_input_format(boltz) == "native"
    assert detect_input_format(fasta) == "native"


def test_unparseable_structured_file_is_treated_as_native(tmp_path: Path) -> None:
    broken = tmp_path / "job.json"
    broken.write_text("{not json at all")
    assert detect_input_format(broken) == "native"


def test_yaml_common_input_materializes_like_its_json_twin(tmp_path: Path) -> None:
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    as_json = tmp_path / "job.json"
    as_json.write_text(json.dumps(_JOB))
    as_yaml = tmp_path / "job.yaml"
    as_yaml.write_text("entities:\n  - type: protein\n    id: A\n    sequence: ACD\n")

    caps = capabilities("protenix")
    from_json = materialize_native_input(as_json, caps, tmp_path / "a", seed=1)
    from_yaml = materialize_native_input(as_yaml, caps, tmp_path / "b", seed=1)
    assert from_json.read_text() == from_yaml.read_text()


# --------------------------------------------------------------------------
# sampling knobs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            Boltz2Backend(),
            {"diffusion_samples": 3, "steps": 40, "recycling": 2},
        ),
        (OpenDDEBackend(), {"n_sample": 3, "n_step": 40, "n_cycle": 2}),
        (ProtenixBackend(), {"n_sample": 3, "n_step": 40, "n_cycle": 2}),
    ],
    ids=["boltz2", "opendde", "protenix"],
)
def test_neutral_knobs_become_each_backends_own_option_names(
    tmp_path: Path, backend, expected
) -> None:
    request = PredictionRequest(
        model=backend.name,
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_samples=3,
        num_steps=40,
        num_recycles=2,
    )
    assert backend.apply_sampling(request) == expected


def test_every_translated_option_is_one_the_backend_really_accepts() -> None:
    """A knob that maps onto a name the backend rejects would fail at runtime."""
    for backend in (OpenDDEBackend(), ProtenixBackend()):
        from foldjax.backends import opendde, protenix

        accepted = (
            opendde._CLI_OPTIONS if backend.name == "opendde" else protenix._CLI_OPTIONS
        )
        assert set(backend.sampling_options.values()) <= accepted, backend.name
    boltz = Boltz2Backend()
    assert set(boltz.sampling_options.values()) <= set(boltz.compile_options)


def test_setting_a_knob_and_its_native_name_together_is_rejected(
    tmp_path: Path,
) -> None:
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_samples=3,
        options={"n_sample": 5},
    )
    with pytest.raises(ValueError, match="both set"):
        OpenDDEBackend().apply_sampling(request)


def test_a_knob_the_backend_cannot_express_is_rejected(tmp_path: Path) -> None:
    """A knob with nowhere to go is an error, not a silent no-op.

    Every shipped backend now honours all four, so the rule is exercised against
    a backend that declares fewer. It is worth keeping: quietly ignoring a cap
    would return a prediction at the model's own depth while the caller believed
    theirs had been applied -- and MSA depth is the dominant memory knob, so they
    would also be comparing peaks that are not comparable.
    """
    from foldjax.backends.base import Backend

    class OneKnobBackend(Backend):
        name = "one-knob"
        sampling_options = {"num_samples": "n_sample"}

        def capabilities(self):  # pragma: no cover - not what this checks
            raise NotImplementedError

        def predict(self, request):  # pragma: no cover - not what this checks
            raise NotImplementedError

    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        max_msa_depth=1024,
    )
    with pytest.raises(ValueError, match="does not support max_msa_depth"):
        OneKnobBackend().apply_sampling(request)


def test_openfold3_now_honours_the_msa_cap_it_used_to_refuse(tmp_path: Path) -> None:
    """It has no native argument for it, so the cap is applied to the features.

    Refusing it left OpenFold3 as the one model that could not be held to the
    same MSA budget as the others, which is the budget that decides whether a
    job fits in memory at all.
    """
    from foldjax.backends.openfold3 import OpenFold3Backend

    request = PredictionRequest(
        model="openfold3",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        max_msa_depth=1024,
    )
    assert OpenFold3Backend().apply_sampling(request)["max_msa_depth"] == 1024


def test_knobs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_samples must be at least 1"):
        PredictionRequest(model="opendde", input=_job_file(tmp_path), num_samples=0)


def test_capabilities_report_the_knobs_each_backend_honours() -> None:
    """All four knobs reach every model that can express them.

    AlphaFold 3 is included: `make_model_config` takes only four arguments, but
    it returns a mutable config and already sets the sample count by
    assignment. The step count and the MSA depth sit one level deeper, at
    `heads.diffusion.eval.steps` and `evoformer.num_msa`. Reporting them as
    unsupported made AlphaFold 3 the one model that could not be held to the
    same schedule as the others -- which is exactly what comparing them needs.
    """
    # Including OpenFold3, which has no MSA-depth argument at all: the cap is
    # applied to its features instead, so the knob means one thing everywhere
    # rather than being refused on the one model.
    for name in ("alphafold3", "boltz2", "opendde", "openfold3", "protenix"):
        sampling = foldjax.capabilities(name).sampling
        assert {
            "num_samples",
            "num_steps",
            "num_recycles",
            "max_msa_depth",
        } == set(sampling), name


# --------------------------------------------------------------------------
# request resolution
# --------------------------------------------------------------------------


def _job_file(tmp_path: Path) -> Path:
    path = tmp_path / "job.json"
    if not path.exists():
        path.write_text(json.dumps(_JOB))
    return path


def test_a_bare_request_resolves_weights_output_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    store = tmp_path / "store"
    (store / "weights" / "opendde").mkdir(parents=True)
    (store / "weights" / "opendde" / "opendde.jax").write_bytes(b"weights")
    monkeypatch.setenv("FOLDJAX_HOME", str(store))
    monkeypatch.chdir(tmp_path)

    resolved = foldjax.resolve_request(
        PredictionRequest(model="opendde", input=_job_file(tmp_path))
    )
    assert resolved.weights == store / "weights" / "opendde" / "opendde.jax"
    assert resolved.output_dir == Path("foldjax-outputs") / "job"
    assert resolved.cache_dir == store / "compile"
    assert resolved.input_format == "foldjax"


def test_missing_weights_name_the_command_that_fetches_them(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "empty"))
    with pytest.raises(
        FileNotFoundError, match="foldjax weights fetch --model protenix"
    ):
        foldjax.resolve_request(
            PredictionRequest(model="protenix", input=_job_file(tmp_path))
        )


def test_explicit_values_are_never_overridden(tmp_path: Path) -> None:
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        output_dir=tmp_path / "mine",
        cache_dir=tmp_path / "cache",
        input_format="native",
    )
    resolved = foldjax.resolve_request(request)
    assert resolved.weights == tmp_path / "weights.jax"
    assert resolved.output_dir == tmp_path / "mine"
    assert resolved.cache_dir == tmp_path / "cache"
    assert resolved.input_format == "native"


def test_sampling_knobs_survive_the_whole_predict_path(tmp_path: Path) -> None:
    """End to end: a neutral knob reaches the backend under its native name."""
    from tests.test_api import DummyBackend

    class OpenDDEDummyBackend(DummyBackend):
        name = "opendde"
        sampling_options = OpenDDEBackend.sampling_options

    backend = OpenDDEDummyBackend()
    with backend_override("opendde", lambda: backend):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_job_file(tmp_path),
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
                input_format="native",
                num_samples=4,
            )
        )
    assert backend.seen is not None
    assert backend.seen.num_samples == 4
    assert OpenDDEBackend().apply_sampling(backend.seen)["n_sample"] == 4


# --------------------------------------------------------------------------
# multiple seeds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed_fields",
    [{"seed": 0, "seeds": (7, 11)}, {"seed": 7, "num_seeds": 2}],
)
def test_resolve_rejects_multi_seed_representation_capture(
    tmp_path: Path, seed_fields: dict[str, object]
) -> None:
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        input_format="native",
        weights=_weights(tmp_path),
        representations=("single",),
        **seed_fields,
    )

    with pytest.raises(ValueError, match="one representation archive"):
        foldjax.resolve_request(request)


def test_seeds_run_the_job_once_each_and_return_every_structure(
    tmp_path: Path, monkeypatch
) -> None:
    """Three of the five models take a seed list natively and two do not.

    A knob that works on half the models is not a neutral knob, so the loop
    lives in `predict` and every backend gets it identically. Each seed writes
    into its own directory, because the models that do not namespace by seed
    would otherwise overwrite each other's structures.
    """
    from foldjax.schema import PredictionResult, PredictionSample

    seen: list[tuple[int, Path]] = []

    class Recorder(OpenDDEBackend):
        def predict(self, request):
            seen.append((request.seed, request.output_dir))
            path = request.output_dir / f"s{request.seed}.cif"
            path.write_text("data_prediction\n#\n")
            return PredictionResult(
                model="opendde",
                samples=(PredictionSample(seed=request.seed, structure_path=path),),
                output_dir=request.output_dir,
            )

    out = tmp_path / "out"
    request = PredictionRequest(
        model="opendde",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        output_dir=out,
        seeds=(7, 11, 13),
        use_compile_cache=False,
    )
    with backend_override("opendde", Recorder):
        result = foldjax.predict(request)

    assert [seed for seed, _ in seen] == [7, 11, 13]
    assert [directory for _, directory in seen] == [
        out / "seed_7",
        out / "seed_11",
        out / "seed_13",
    ]
    assert [sample.seed for sample in result.samples] == [7, 11, 13]
    assert result.output_dir == out
    for seed in (7, 11, 13):
        manifest = json.loads((out / f"seed_{seed}" / "foldjax_run.json").read_text())
        assert manifest["seeds"] == [seed]


def test_num_seeds_child_manifests_name_only_the_seed_that_ran(
    tmp_path: Path,
) -> None:
    from foldjax.schema import PredictionResult, PredictionSample

    class Recorder(OpenDDEBackend):
        def predict(self, request):
            return PredictionResult(
                model="opendde",
                samples=(
                    PredictionSample(
                        seed=request.seed, coordinates=((0.0, 0.0, 0.0),)
                    ),
                ),
                output_dir=request.output_dir,
            )

    out = tmp_path / "out"
    with backend_override("opendde", Recorder):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_job_file(tmp_path),
                weights=_weights(tmp_path),
                output_dir=out,
                input_format="native",
                seed=4,
                num_seeds=2,
                use_compile_cache=False,
            )
        )

    assert json.loads((out / "seed_4" / "foldjax_run.json").read_text())["seeds"] == [4]
    assert json.loads((out / "seed_5" / "foldjax_run.json").read_text())["seeds"] == [5]


def test_one_seed_keeps_the_output_directory_it_was_given(
    tmp_path: Path, monkeypatch
) -> None:
    """The single-seed path must not gain a directory level."""
    from foldjax.schema import PredictionResult, PredictionSample

    seen: list[Path] = []

    class Recorder(OpenDDEBackend):
        def predict(self, request):
            seen.append(request.output_dir)
            return PredictionResult(
                model="opendde",
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        coordinates=((0.0, 0.0, 0.0),),
                    ),
                ),
                output_dir=request.output_dir,
            )

    out = tmp_path / "out"
    with backend_override("opendde", Recorder):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_job_file(tmp_path),
                weights=_weights(tmp_path),
                output_dir=out,
                seed=4,
                use_compile_cache=False,
            )
        )
    assert seen == [out]


def test_seed_and_seeds_together_are_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both set"):
        PredictionRequest(
            model="opendde", input=_job_file(tmp_path), seed=3, seeds=(1, 2)
        )


@pytest.mark.parametrize(
    ("seeds", "message"),
    [((), "must not be empty"), ((1, 1), "must be unique"), ((-1,), "non-negative")],
)
def test_a_malformed_seed_list_is_refused(tmp_path: Path, seeds, message) -> None:
    with pytest.raises(ValueError, match=message):
        PredictionRequest(model="opendde", input=_job_file(tmp_path), seeds=seeds)


def test_openfold3_knobs_use_the_names_its_config_takes(tmp_path: Path) -> None:
    """The map has to land on `released_config`'s own argument names.

    It mapped `num_steps` and `num_recycles` onto themselves, while the config
    and the adapter's own pop list call them `no_rollout_steps` and
    `num_cycles`. Both therefore stayed in the leftover options and the adapter
    raised "unsupported OpenFold3 options" -- two of the three knobs it
    advertised could only ever fail. The port is not installed here, so this
    checks the translation rather than a run.
    """
    from foldjax.backends.openfold3 import _COMPILE_OPTIONS, OpenFold3Backend

    backend = OpenFold3Backend()
    request = PredictionRequest(
        model="openfold3",
        input=_job_file(tmp_path),
        weights=_weights(tmp_path),
        num_samples=3,
        num_steps=40,
        num_recycles=2,
    )
    assert backend.apply_sampling(request) == {
        "num_samples": 3,
        "no_rollout_steps": 40,
        # OpenFold3's native config is the number of executed cycles, while the
        # neutral knob follows upstream's num_recycles + 1 convention.
        "num_cycles": 3,
    }
    # Every translated name must also be one the cache namespace knows about,
    # or two different schedules would share a compiled program.
    assert set(backend.sampling_options.values()) <= set(_COMPILE_OPTIONS)
