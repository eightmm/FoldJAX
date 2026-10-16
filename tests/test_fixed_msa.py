"""An explicit row cap plus an exact masked capacity fixes the MSA JIT ABI."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import jax
import numpy as np
import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.openfold3 import OpenFold3Backend, _padding_plan
from foldjax.registry import get_backend
from foldjax.schema import PaddingConfig, PredictionRequest

_MODELS = ("boltz2", "protenix", "openfold3", "opendde")
_DEPTH = 7


def _alignment(rows: int, sequence: str) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [sequence]
    for index in range(rows - 1):
        sequences.append(
            alphabet[index % 20] + alphabet[(index // 20) % 20] + sequence[2:]
        )
    assert len(set(sequences)) == rows
    return "".join(f">row{index}\n{row}\n" for index, row in enumerate(sequences))


def _boltz_boundary(rows: int):
    from foldjax.models.boltz2.data.bucket import (
        pad_feats,
        resolve_padding_plan,
        select_model_features,
    )
    from foldjax.models.boltz2.data.feature.featurizerv2 import process_msa_features
    from foldjax.models.boltz2.data.parse.a3m import _parse_a3m
    from tests.models.boltz2.test_msa_deletions import build_tokenized

    data = build_tokenized()
    data.msa[0] = _parse_a3m(io.StringIO(_alignment(rows, "ACDEFGHIKL")), taxonomy=None)
    features = process_msa_features(
        data,
        random=np.random.default_rng(42),
        max_seqs_batch=_DEPTH,
        max_seqs=_DEPTH,
    )
    features = {name: np.asarray(value)[None] for name, value in features.items()}
    real = min(rows, _DEPTH)
    assert features["msa"].shape == (1, real, 10)
    original = features["msa"].copy()
    features["token_pad_mask"] = np.ones((1, 10), np.float32)
    features["atom_pad_mask"] = np.ones((1, 10), np.float32)
    plan = resolve_padding_plan(
        features, PaddingConfig(tokens=16, atoms=32, msa=_DEPTH), max_msa_depth=_DEPTH
    )
    padded, _ = pad_feats(features, 16, 32, target_msa=plan.target["msa"])
    np.testing.assert_array_equal(np.asarray(padded["msa"])[:, :real, :10], original)
    assert not np.any(np.asarray(padded["msa_mask"])[:, real:])
    return select_model_features(padded), plan, real


def _family_boundary(model: str, rows: int):
    from foldjax.models.protenix.data.featurize_json import featurize_protein_json
    from foldjax.models.protenix.data.padding import pad_protenix_features
    from foldjax.models.protenix.models.model import _PADDED_MODEL_FEATURES
    from foldjax.models.protenix.models.trunk_blocks.msa import (
        sample_msa_cycle_index_tape,
    )

    sequence = "ACDEFG"
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sequence,
                    "unpairedMsa": _alignment(rows, sequence),
                }
            }
        ]
    }
    if model == "opendde":
        from foldjax.models.opendde.data.featurize_json import featurize_opendde_json

        featurize = featurize_opendde_json
    else:
        featurize = featurize_protein_json
    features = featurize(job, n_queries=2, n_keys=4, max_msa_depth=_DEPTH, seed=42)
    real = min(rows + 1, _DEPTH)
    assert features["msa"].shape == (real, len(sequence))
    if model == "opendde":
        from foldjax.models.opendde.data.padding import (
            pad_opendde_features,
            select_opendde_model_features,
        )
        from foldjax.models.opendde.models.msa_sampling import (
            sample_opendde_msa_cycle_features,
        )

        cycles = sample_opendde_msa_cycle_features(features, num_recycles=2, seed=42)
        padded, padded_cycles, plan = pad_opendde_features(
            features,
            cycles,
            PaddingConfig(tokens=16, atoms=64, structural_tokens=32, msa=_DEPTH),
            n_queries=2,
            n_keys=4,
        )
        for original, cycle in zip(cycles, padded_cycles, strict=True):
            assert cycle["msa"].shape == (_DEPTH, 16)
            np.testing.assert_array_equal(cycle["msa"][:real, :6], original["msa"])
            assert not np.any(cycle["msa_mask"][real:])
        return (select_opendde_model_features(padded), padded_cycles), plan, real
    padded, plan = pad_protenix_features(
        features,
        PaddingConfig(tokens=16, atoms=64, msa=_DEPTH),
        n_queries=2,
        n_keys=4,
        max_msa_depth=_DEPTH,
    )
    np.testing.assert_array_equal(padded["msa"][:real, :6], features["msa"])
    assert not np.any(padded["msa_mask"][real:])
    tape = sample_msa_cycle_index_tape(
        padded, num_recycles=2, seed=42, real_rows=real, width=_DEPTH
    )
    assert tape.row_indices.shape == (2, _DEPTH)
    assert np.all(tape.row_indices[tape.row_mask] < real)
    model_features = {k: v for k, v in padded.items() if k in _PADDED_MODEL_FEATURES}
    return (model_features, tape), plan, real


def _openfold_boundary(rows: int, depth: int | None = None):
    from foldjax.models.openfold3.data import pad_features, prepare_msa_cycle_features
    from foldjax.models.openfold3.streaming import HostMSACycles
    from tests.models.openfold3.feature_fixture import minimal_features

    depth = _DEPTH if depth is None else depth
    native_depth = min(depth, 1024)
    features = prepare_msa_cycle_features(
        minimal_features(tokens=4, atoms=7, msa_rows=rows),
        native_depth,
        num_recycles=2,
        rng=np.random.default_rng(42),
    )
    plan = _padding_plan(
        features,
        PaddingConfig(tokens=16, atoms=32, msa=depth),
        max_msa_depth=native_depth,
    )
    features = pad_features(features, n_token=16, n_atom=32, n_msa=None, n_templates=4)
    provider = HostMSACycles(features, depth=plan.target["msa"], cycles=2)
    selected = tuple(provider.select(cycle) for cycle in range(2))
    real = min(rows, native_depth)
    for cycle in selected:
        assert cycle["msa_mask"].shape == (1, depth, 16)
        assert np.count_nonzero(np.any(cycle["msa_mask"], axis=-1)) == real
        assert not np.any(cycle["msa_mask"][:, real:])
    return (provider.common, selected), plan, real


def _boundary(model: str, rows: int):
    if model == "boltz2":
        return _boltz_boundary(rows)
    if model == "openfold3":
        return _openfold_boundary(rows)
    return _family_boundary(model, rows)


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("depth", [1, 7])
def test_fixed_msa_caps_before_padding_and_keeps_jit_abi_across_real_depths(
    model, depth, monkeypatch
):
    monkeypatch.setitem(globals(), "_DEPTH", depth)
    # One query row represents --msa single; the deeper alignments represent
    # searched inputs below and above the explicit cap. Probe every numeric
    # model-bound leaf, including per-cycle selections, without loading weights.
    probe = jax.jit(lambda features: features)
    signatures = []
    lowered_programs = []
    for rows in (1, 3, 17):
        features, plan, real = _boundary(model, rows)
        expected_rows = rows + 1 if model in {"protenix", "opendde"} else rows
        assert real == min(expected_rows, _DEPTH)
        assert plan.target["msa"] == _DEPTH
        abstract = jax.tree.map(
            lambda value: jax.ShapeDtypeStruct(
                np.shape(value), jax.dtypes.canonicalize_dtype(np.asarray(value).dtype)
            ),
            features,
        )
        lowered = probe.lower(abstract)
        signatures.append(lowered.in_avals)
        lowered_programs.append(str(lowered.compiler_ir("stablehlo")))
    assert signatures[0] == signatures[1] == signatures[2]
    assert lowered_programs[0] == lowered_programs[1] == lowered_programs[2]


@pytest.mark.parametrize("depth", [128, 2048, 4096])
def test_openfold3_fixed_msa_keeps_native_cap_and_exact_compiled_capacity(
    tmp_path: Path, depth: int
):
    job = tmp_path / "job.json"
    job.write_text('{"entities": []}')
    request = PredictionRequest(
        model="openfold3",
        input=job,
        max_msa_depth=depth,
        padding=PaddingConfig(msa=depth),
    )
    native = OpenFold3Backend().apply_sampling(request)["max_msa_depth"]
    assert native == min(depth, 1024)
    features, plan, real = _openfold_boundary(1100, depth)
    assert real == native
    assert plan.target["msa"] == depth
    assert features[1][0]["msa_mask"].shape[1] == depth


@pytest.mark.parametrize("model", _MODELS)
def test_fixed_msa_single_warm_and_auto_predict_use_the_same_namespace(
    tmp_path: Path, model: str
):
    weights = tmp_path / "weights"
    weights.write_bytes(b"weight identity fixture")
    backend = get_backend(model)
    for name in ("warm", "predict"):
        (tmp_path / f"{name}.json").write_text('{"entities": []}')
    warm = PredictionRequest(
        model=model,
        input=tmp_path / "warm.json",
        weights=weights,
        cache_dir=tmp_path / "compile",
        msa="single",
        max_msa_depth=2048,
        padding=PaddingConfig(msa=2048),
    )
    predict = dataclasses.replace(warm, input=tmp_path / "predict.json", msa="auto")
    assert backend.cache_profile(warm) == backend.cache_profile(predict)
    assert resolve_cache_dir(warm, backend) == resolve_cache_dir(predict, backend)
