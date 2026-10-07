"""`run_alphafold.py`'s data-pipeline flags, as AlphaFold 3 options.

`--resolve_msa_overlaps`, `--fix_standalone_glycans` and
`--conformer_max_iterations` (`run_alphafold.py:286-319`) reach
`featurise_input` through `predict_structure`; `--max_template_date` is also
the CCD fallback cutoff (`ref_max_modified_date`, `:1065`).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends import alphafold3 as af3_backend
from foldjax.backends.alphafold3 import (
    AlphaFold3Backend,
    _predict_structure_kwargs,
    _ref_max_modified_date,
)
from foldjax.schema import PredictionRequest
from tests.test_backends import FakeFoldInput


def _request(tmp_path: Path, **options) -> PredictionRequest:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="alphafold3",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options=options,
    )


def _run(tmp_path: Path, monkeypatch, request: PredictionRequest) -> dict:
    """Run the adapter against an external runner that records its call."""

    runner_file = tmp_path / "af3" / "run_alphafold.py"
    runner_file.parent.mkdir()
    runner_file.touch()
    package_init = runner_file.parent / "src/alphafold3/__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.touch()
    request.options["source"] = runner_file.parent
    seen: dict = {}

    def predict_structure(fold_input, model_runner, *, buckets, **upstream):
        seen["upstream"] = upstream
        return (
            SimpleNamespace(
                seed=5,
                inference_results=(SimpleNamespace(metadata={"ranking_score": 1}),),
            ),
        )

    def write_outputs(written, output_dir, job_name):
        sample_dir = Path(output_dir) / "seed-5_sample-0"
        sample_dir.mkdir(parents=True)
        prefix = f"{job_name}_seed-5_sample-0"
        (sample_dir / f"{prefix}_model.cif").touch()
        (sample_dir / f"{prefix}_summary_confidences.json").write_text(
            json.dumps({"ptm": 0.7, "iptm": 0.5})
        )

    runner = SimpleNamespace(
        ModelRunner=lambda **kwargs: None,
        make_model_config=lambda **kwargs: SimpleNamespace(**kwargs),
        predict_structure=predict_structure,
        write_outputs=write_outputs,
    )
    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda path: iter((FakeFoldInput(),))
    common = ModuleType("alphafold3.common")
    common.folding_input = folding
    package = ModuleType("alphafold3")
    package.__file__ = str(package_init)
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    monkeypatch.setattr(af3_backend, "_load_runner", lambda path: runner)
    monkeypatch.setattr(af3_backend, "_set_nested", lambda *args: None)
    monkeypatch.setattr(af3_backend, "_model_device", lambda device: nullcontext())
    device = jax.devices()[0]
    monkeypatch.setattr("jax.local_devices", lambda backend=None: (device,))
    AlphaFold3Backend().predict(request)
    return seen


def test_the_options_reach_predict_structure(tmp_path: Path, monkeypatch) -> None:
    seen = _run(
        tmp_path,
        monkeypatch,
        _request(
            tmp_path,
            resolve_msa_overlaps=False,
            fix_standalone_glycans=True,
            conformer_max_iterations=7,
        ),
    )

    assert seen["upstream"] == {
        "resolve_msa_overlaps": False,
        "fix_standalone_glycans": True,
        "conformer_max_iterations": 7,
        "ref_max_modified_date": datetime.date(2021, 9, 30),
    }


def test_omitted_options_leave_the_featuriser_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    seen = _run(tmp_path, monkeypatch, _request(tmp_path))

    assert seen["upstream"] == {"ref_max_modified_date": datetime.date(2021, 9, 30)}


def test_the_ccd_cutoff_follows_the_template_cutoff(
    tmp_path: Path, monkeypatch
) -> None:
    """`run_alphafold.py` has one flag for both cutoffs (`:989`, `:1065`)."""

    assert _ref_max_modified_date(None) == datetime.date(2021, 9, 30)
    assert _ref_max_modified_date("2023-01-15") == datetime.date(2023, 1, 15)
    request = dataclasses.replace(
        _request(tmp_path), templates="auto", template_max_date="2023-01-15"
    )
    seen = _run(tmp_path, monkeypatch, request)
    assert seen["upstream"]["ref_max_modified_date"] == datetime.date(2023, 1, 15)


def test_an_external_runner_without_the_parameter_refuses_a_spelled_option() -> None:
    def older(fold_input, model_runner, *, buckets):
        pass

    assert _predict_structure_kwargs(older) == {}
    with pytest.raises(ValueError, match="does not take resolve_msa_overlaps"):
        _predict_structure_kwargs(
            older, featurisation_options={"resolve_msa_overlaps": False}
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"resolve_msa_overlaps": "no"}, "resolve_msa_overlaps"),
        ({"fix_standalone_glycans": 1}, "fix_standalone_glycans"),
        ({"conformer_max_iterations": -1}, "conformer_max_iterations"),
        ({"conformer_max_iterations": 1.5}, "conformer_max_iterations"),
    ],
)
def test_plan_refuses_malformed_values(tmp_path: Path, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        AlphaFold3Backend().validate_request(_request(tmp_path, **options))


def test_released_spellings_share_a_namespace_and_departures_fork(
    tmp_path: Path,
) -> None:
    backend = AlphaFold3Backend()
    omitted = resolve_cache_dir(_request(tmp_path), backend)

    assert (
        resolve_cache_dir(
            _request(tmp_path, resolve_msa_overlaps=True, fix_standalone_glycans=False),
            backend,
        )
        == omitted
    )
    for departure in (
        {"resolve_msa_overlaps": False},
        {"fix_standalone_glycans": True},
        {"conformer_max_iterations": 100},
    ):
        assert resolve_cache_dir(_request(tmp_path, **departure), backend) != omitted


def test_padded_and_representation_featurisation_take_them_too(
    tmp_path: Path, monkeypatch
) -> None:
    """The two routes that call `featurise_input` themselves, not the runner."""

    import numpy as np

    calls: list[dict] = []

    def featurise_input(**kwargs):
        calls.append(kwargs)
        return [{"seq_length": np.asarray(3), "seq_mask": np.ones(8)}]

    data = ModuleType("alphafold3.data")
    data.featurisation = SimpleNamespace(featurise_input=featurise_input)
    constants = ModuleType("alphafold3.constants")
    constants.chemical_components = SimpleNamespace(Ccd=lambda **kwargs: {})
    monkeypatch.setitem(sys.modules, "alphafold3", ModuleType("alphafold3"))
    monkeypatch.setitem(sys.modules, "alphafold3.data", data)
    monkeypatch.setitem(sys.modules, "alphafold3.constants", constants)
    options = {"resolve_msa_overlaps": False, "conformer_max_iterations": 3}
    fold_input = SimpleNamespace(rng_seeds=(0,), name="job", user_ccd=None)

    af3_backend._featurize_padded_structure(
        fold_input,
        buckets=(8,),
        overflow="error",
        ref_max_modified_date=datetime.date(2022, 1, 1),
        featurisation_options=options,
    )
    request = dataclasses.replace(
        _request(tmp_path),
        stop_after="inputs",
        representations=("single_inputs",),
        templates="auto",
        template_max_date="2022-02-02",
    )
    af3_backend._predict_common_representations(
        fold_input,
        None,
        SimpleNamespace(
            run_inference=lambda batch, key: {
                "representations": {"single_inputs": np.ones((3, 4))}
            }
        ),
        SimpleNamespace(),
        request=request,
        wanted=("single_inputs",),
        buckets=None,
        featurisation_options=options,
    )

    padded, unpadded = calls
    for call in calls:
        assert call["resolve_msa_overlaps"] is False
        assert call["conformer_max_iterations"] == 3
    assert padded["ref_max_modified_date"] == datetime.date(2022, 1, 1)
    assert unpadded["ref_max_modified_date"] == datetime.date(2022, 2, 2)
