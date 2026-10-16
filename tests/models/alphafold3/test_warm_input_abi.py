"""Real native CPU preprocessing behind AF3 synthetic bucket warming."""

from __future__ import annotations

import dataclasses
import datetime
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends import _tokamax_autotune as persistent
from foldjax.backends import alphafold3 as backend
from foldjax.schema import PaddingConfig, PredictionRequest

_SEQUENCE = "ACDEFGHIKLMNPQRS"
_REAL_MSA = (
    f">query\n{_SEQUENCE}\n"
    ">substitution\nACDEFGHIKLMNPQRT\n"
    ">gaps\nACDEFGHIKL--PQRS\n"
    ">insertion\nACDEaFGHIKLMNPQRS\n"
)


def _protein(sequence, *, identifier="A", msa="", templates=()):
    return {
        "protein": {
            "id": identifier,
            "sequence": sequence,
            "unpairedMsa": msa,
            "pairedMsa": "",
            "templates": list(templates),
        }
    }


@pytest.fixture
def native(alphafold3_runtime):
    from foldjax.models.alphafold3 import build

    build.register_runtime()
    from alphafold3.common import folding_input
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from alphafold3.model import features
    from alphafold3.model.network import featurization

    return SimpleNamespace(
        folding_input=folding_input,
        ccd=chemical_components.Ccd(),
        featurisation=featurisation,
        features=features,
        featurization=featurization,
        runner=backend._load_runner(backend.VENDORED_RUNNER),
    )


def _input(native, proteins):
    return native.folding_input.Input.from_json(
        json.dumps(
            {
                "name": "abi-probe",
                "modelSeeds": [0],
                "dialect": "alphafold3",
                "version": 1,
                "sequences": proteins,
            }
        )
    )


def _example(native, proteins):
    fold_input = _input(native, proteins)
    examples, plan = backend._featurize_padded_structure(
        fold_input,
        buckets=(256,),
        overflow="error",
        fixed_target=True,
    )
    assert plan.target["tokens"] == 256
    return fold_input, examples[0]


def _model_batch(native, fold_input, example):
    """Use the actual runner's filtering, dtype conversion, and device transfer."""
    captured = []
    runner = object.__new__(native.runner.ModelRunner)
    runner._device = jax.devices("cpu")[0]
    runner.model_params = {"__meta__": {"__identifier__": np.zeros(1, np.uint8)}}

    def capture(key, batch):
        captured.append((key, batch))
        return {}

    def extract(*, batch, **_kwargs):
        assert batch is example
        assert backend._PREFIX_STABLE_NOISE_FEATURE not in batch
        return []

    runner._model = capture
    runner.extract_inference_results = extract
    runner.extract_embeddings = lambda **_kwargs: None
    runner.extract_distogram = lambda **_kwargs: None
    with jax.default_device(runner._device):
        backend._predict_featurized_structure(
            fold_input, [example], runner, native.runner
        )
    (key, batch), = captured
    assert bool(batch[backend._PREFIX_STABLE_NOISE_FEATURE])
    assert all(value.dtype != object for value in batch.values())
    return key, batch


def test_single_and_real_msa_keep_the_full_native_pool_and_model_abi(native):
    single_input, single = _example(native, [_protein(_SEQUENCE)])
    real_input, real = _example(native, [_protein(_SEQUENCE, msa=_REAL_MSA)])
    single_key, single_batch = _model_batch(native, single_input, single)
    real_key, real_batch = _model_batch(native, real_input, real)

    assert single["msa"].shape == real["msa"].shape == (16384, 256)
    assert real["num_alignments"] > single["num_alignments"]
    assert not np.array_equal(real["profile"], single["profile"])
    assert np.any(real["deletion_matrix"] > 0)
    assert persistent._call_descriptor(single_key, single_batch) == (
        persistent._call_descriptor(real_key, real_batch)
    )
    assert persistent._array_tree_descriptor(single_batch) == (
        persistent._array_tree_descriptor(real_batch)
    )
    for name in ("msa", "msa_mask", "deletion_matrix", "profile", "deletion_mean"):
        np.testing.assert_array_equal(real_batch[name], real[name])
    rows = int(real["num_alignments"])
    assert np.count_nonzero(real["msa_mask"][rows:]) == 0
    assert np.count_nonzero(real["msa_mask"][:, len(_SEQUENCE):]) == 0

    native_reference = native.featurisation.featurise_input(
        fold_input=real_input,
        ccd=native.ccd,
        buckets=(256,),
        ref_max_modified_date=datetime.date(2021, 9, 30),
    )[0]
    for name in ("msa", "msa_mask", "deletion_matrix", "profile", "deletion_mean"):
        np.testing.assert_array_equal(real[name], native_reference[name])

    traces = []

    def select_msa(key, batch):
        traces.append(batch["msa"].shape)
        msa = native.features.MSA.from_data_dict(batch)
        msa, _ = native.featurization.shuffle_msa(key, msa)
        return native.featurization.truncate_msa_batch(msa, 1024)

    compiled = jax.jit(select_msa)
    first = compiled.lower(single_key, single_batch)
    second = compiled.lower(real_key, real_batch)
    assert traces == [(16384, 256)]
    assert first.as_text() == second.as_text()


@pytest.mark.parametrize("case", ("length", "composition", "multimer", "template"))
def test_other_native_shape_axes_match_the_synthetic_bucket(native, case):
    warm_input, warm = _example(native, [_protein("A")])
    if case == "length":
        proteins = [_protein("A" * 256)]
    elif case == "composition":
        proteins = [_protein(_SEQUENCE, msa=_REAL_MSA)]
    elif case == "multimer":
        proteins = [
            _protein(_SEQUENCE, msa=_REAL_MSA),
            _protein(_SEQUENCE[::-1], identifier="B"),
        ]
    else:
        template_source = _input(native, [_protein(_SEQUENCE)])
        mmcif = template_source.to_structure(native.ccd).to_mmcif()
        mmcif += "\n_pdbx_audit_revision_history.revision_date 2020-01-01\n"
        template = {
            "mmcif": mmcif,
            "queryIndices": list(range(len(_SEQUENCE))),
            "templateIndices": list(range(len(_SEQUENCE))),
        }
        proteins = [_protein(_SEQUENCE, msa=_REAL_MSA, templates=[template])]
    real_input, real = _example(native, proteins)
    warm_key, warm_batch = _model_batch(native, warm_input, warm)
    real_key, real_batch = _model_batch(native, real_input, real)

    assert persistent._call_descriptor(warm_key, warm_batch) == (
        persistent._call_descriptor(real_key, real_batch)
    )
    assert real["template_atom_positions"].shape == (4, 256, 24, 3)
    if case == "template":
        assert np.count_nonzero(real["template_atom_mask"])
    for name, value in real_batch.items():
        if ":gather_" in name or name.endswith(":input_shape"):
            assert value.shape == warm_batch[name].shape


def test_synthetic_fixed_bucket_and_auto_prediction_share_compile_namespace(tmp_path):
    source = tmp_path / "job.json"
    source.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    warm = PredictionRequest(
        model="alphafold3",
        input=source,
        input_format="foldjax",
        weights=weights,
        cache_dir=tmp_path / "cache",
        msa="single",
        padding=PaddingConfig(tokens=256),
        options={"platform": "cpu"},
    )
    predict = dataclasses.replace(warm, msa="auto", padding=PaddingConfig())
    adapter = backend.AlphaFold3Backend()

    assert adapter.cache_profile(warm) == adapter.cache_profile(predict)
    assert resolve_cache_dir(warm, adapter) == resolve_cache_dir(predict, adapter)


def test_automatic_prediction_uses_the_same_first_af3_bucket(native, tmp_path):
    source = tmp_path / "job.json"
    source.write_text("{}")
    request = PredictionRequest(
        model="alphafold3", input=source, padding=PaddingConfig()
    )
    buckets = backend._prediction_buckets(request, {})
    assert buckets[0] == 256
    sequence = (_SEQUENCE * 5)[:76]
    fold_input, fixed = _example(native, [_protein(sequence)])
    automatic, plan = backend._featurize_padded_structure(
        fold_input, buckets=buckets, overflow="error"
    )
    assert plan.target["tokens"] == 256
    fixed_key, fixed_batch = _model_batch(native, fold_input, fixed)
    auto_key, auto_batch = _model_batch(native, fold_input, automatic[0])
    assert persistent._call_descriptor(fixed_key, fixed_batch) == (
        persistent._call_descriptor(auto_key, auto_batch)
    )
