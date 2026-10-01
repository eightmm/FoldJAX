"""A native OpenDDE ``constraint`` is dropped as upstream drops it, never silently.

Only native input can carry one: the common schema has no constraint field.
The Protenix featurizer OpenDDE shares would build a ``constraint_feature``
from it, and no OpenDDE module reads that feature; upstream's inference build
warns and ignores the field (OpenDDE 1.1.1
``opendde/data/inference/json_to_feature.py:28-32``). The contract follows the
template and nucleic-MSA ones (``tests/test_template_gate.py``): dropped with a
warning and a manifest record by default, refused with
``ignore_constraints=false``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax.input import IGNORE_CONSTRAINTS, native_only_features
from foldjax.manifest import MANIFEST_NAME
from foldjax.models.opendde.data import featurize_json as fj
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override, capabilities, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

_CONSTRAINT = {
    "contact": [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 1,
            "atom1": "CA",
            "entity2": 1,
            "copy2": 1,
            "position2": 3,
            "atom2": "CA",
            "max_distance": 8.0,
        }
    ]
}


def _native(tmp_path: Path, constraint: object = _CONSTRAINT) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEF\n>hit\nACDEY\n")
    job: dict = {
        "name": "constrained",
        "modelSeeds": [3],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACDEF",
                    "count": 1,
                    "unpairedMsaPath": "protein.a3m",
                }
            }
        ],
    }
    if constraint is not None:
        job["constraint"] = constraint
    path = tmp_path / "native.json"
    path.write_text(json.dumps([job]))
    return path


def _common(tmp_path: Path) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEF\n>hit\nACDEY\n")
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "name": "plain",
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACDEF",
                        "unpaired_msa": "protein.a3m",
                    }
                ],
            }
        )
    )
    return path


def _request(path: Path, *, input_format: str, **options) -> PredictionRequest:
    return PredictionRequest(
        model="opendde",
        input=path,
        input_format=input_format,
        seed=1,
        msa="none",
        options=options,
    )


def test_the_featurizer_never_receives_the_constraint(tmp_path, monkeypatch) -> None:
    """The property, not the warning: no ``constraint_feature`` can be built.

    The shared featurizer emits ``constraint_feature`` exactly when the job it
    receives has a ``constraint`` key (protenix/data/featurize_json.py), so a
    job without the key cannot produce one.
    """
    received: list[dict] = []

    def capture(job, **_kwargs):
        received.append(job)
        raise _StopError

    monkeypatch.setattr(fj, "featurize_protein_json", capture)
    (job,) = json.loads(_native(tmp_path).read_text())
    with pytest.warns(RuntimeWarning, match="constraint"), pytest.raises(_StopError):
        fj.featurize_opendde_json(job, base_dir=tmp_path)

    (prepared,) = received
    assert "constraint" not in prepared
    assert job["constraint"] == _CONSTRAINT, "the caller's job is not mutated"


@pytest.mark.parametrize("empty", [{}, None])
def test_an_empty_constraint_is_dropped_without_a_word(tmp_path, empty) -> None:
    (job,) = json.loads(_native(tmp_path, constraint=empty).read_text())
    job["constraint"] = empty
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prepared = fj._prepare_job(job, base_dir=tmp_path)
    assert "constraint" not in prepared


def test_ignore_constraints_false_refuses_while_planning(tmp_path) -> None:
    backend = get_backend("opendde")
    native = _native(tmp_path)
    with pytest.raises(ValueError, match="'constrained'.*ignored_constraints"):
        backend.validate_request(
            _request(native, input_format="native", **{IGNORE_CONSTRAINTS: False})
        )
    # The default and an explicit true both run it, as upstream does.
    backend.validate_request(_request(native, input_format="native"))
    backend.validate_request(
        _request(native, input_format="native", **{IGNORE_CONSTRAINTS: True})
    )
    # Nothing to refuse in a job without one.
    (tmp_path / "plain").mkdir()
    plain = _native(tmp_path / "plain", constraint=None)
    backend.validate_request(
        _request(plain, input_format="native", **{IGNORE_CONSTRAINTS: False})
    )
    with pytest.raises(ValueError, match=f"{IGNORE_CONSTRAINTS} must be a boolean"):
        backend.validate_request(
            _request(native, input_format="native", **{IGNORE_CONSTRAINTS: "yes"})
        )


def test_the_option_does_not_apply_to_common_schema_input(tmp_path) -> None:
    backend = get_backend("opendde")
    common = _common(tmp_path)
    with pytest.raises(ValueError, match="common schema has no constraint field"):
        backend.validate_request(
            _request(common, input_format="foldjax", **{IGNORE_CONSTRAINTS: True})
        )
    backend.validate_request(
        _request(common, input_format="foldjax", **{IGNORE_CONSTRAINTS: False})
    )


def test_backends_that_read_constraints_do_not_take_the_option(tmp_path) -> None:
    with pytest.raises(ValueError, match=IGNORE_CONSTRAINTS):
        get_backend("protenix").validate_request(
            PredictionRequest(
                model="protenix",
                input=_native(tmp_path),
                input_format="native",
                seed=1,
                msa="none",
                options={IGNORE_CONSTRAINTS: True},
            )
        )


def test_native_only_features_say_where_a_constraint_is_read() -> None:
    opendde = native_only_features("opendde", capabilities("opendde"))
    protenix = native_only_features("protenix", capabilities("protenix"))
    assert "contact_constraints" not in opendde
    assert "pocket_constraints" not in opendde
    assert {"contact_constraints", "pocket_constraints"} <= set(protenix)
    assert {"multi_residue_ligand", "ligand_file"} <= set(opendde)


class _StopError(Exception):
    pass


def _recorder(seen: list):
    base = provider(PORTS["opendde"].backend)

    class Recorder(base):
        def predict(self, request):
            seen.append(request)
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

    return Recorder


def test_a_run_records_the_dropped_constraint_unless_the_option_refuses(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    native = _native(tmp_path)
    common = _common(tmp_path)
    seen: list = []

    def request(path: Path, options: dict, out: str) -> PredictionRequest:
        return PredictionRequest(
            model="opendde",
            input=path,
            weights=weights,
            output_dir=tmp_path / out,
            seed=3,
            msa="none",
            options=options,
            use_compile_cache=False,
        )

    with backend_override("opendde", _recorder(seen)):
        with pytest.raises(ValueError, match="ignore_constraints=false"):
            foldjax.predict(request(native, {IGNORE_CONSTRAINTS: False}, "refused"))
        assert seen == []
        assert not (tmp_path / "refused" / MANIFEST_NAME).exists()
        foldjax.predict(request(native, {}, "dropped"))
        foldjax.predict(request(native, {IGNORE_CONSTRAINTS: True}, "explicit"))
        foldjax.predict(request(common, {IGNORE_CONSTRAINTS: False}, "common"))

    dropped, explicit, ran_common = seen
    for ran in seen:
        assert IGNORE_CONSTRAINTS not in ran.options
    for out in ("dropped", "explicit"):
        manifest = json.loads((tmp_path / out / MANIFEST_NAME).read_text())
        (record,) = manifest["ignored_constraints"]
        assert record["job"] == "constrained"
        assert record["field"] == "constraint"
        assert record["keys"] == ["contact"]
        assert "ignored, as upstream does" in record["reason"]
    explicit_manifest = json.loads((tmp_path / "explicit" / MANIFEST_NAME).read_text())
    assert explicit_manifest["options"][IGNORE_CONSTRAINTS] is True
    common_manifest = json.loads((tmp_path / "common" / MANIFEST_NAME).read_text())
    assert common_manifest["ignored_constraints"] is None
    assert dropped.input == native and explicit.input == native
    assert ran_common.input_format != "foldjax"


def test_a_native_job_without_a_constraint_records_an_empty_list(tmp_path) -> None:
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    seen: list = []
    with backend_override("opendde", _recorder(seen)):
        foldjax.predict(
            PredictionRequest(
                model="opendde",
                input=_native(tmp_path, constraint=None),
                weights=weights,
                output_dir=tmp_path / "out",
                seed=3,
                msa="none",
                use_compile_cache=False,
            )
        )
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["ignored_constraints"] == []


def test_other_backends_record_no_constraint_gate(tmp_path) -> None:
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"not really weights")
    base = provider(PORTS["protenix"].backend)

    class Recorder(base):
        def predict(self, request):
            path = request.output_dir / "s.cif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data_mock\n#\n", encoding="utf-8")
            return PredictionResult(
                model="protenix",
                samples=(
                    PredictionSample(
                        seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                    ),
                ),
                output_dir=request.output_dir,
            )

    with backend_override("protenix", Recorder):
        foldjax.predict(
            PredictionRequest(
                model="protenix",
                input=_native(tmp_path),
                weights=weights,
                output_dir=tmp_path / "out",
                seed=3,
                msa="none",
                use_compile_cache=False,
            )
        )
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    assert manifest["ignored_constraints"] is None
