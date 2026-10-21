"""A bounded serving profile fixes OpenDDE's independent padded graph axes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from foldjax import _warm_process, warm_buckets
from foldjax._opendde_padding import COMPACT_PROFILE, compact_targets
from foldjax.api import resolve_cache_dir
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.cli import _parser, _request
from foldjax.models.opendde.data.padding import _resolve_auxiliary_targets
from foldjax.padding import cp_aligned_padding, cp_rows
from foldjax.registry import get_backend
from foldjax.schema import PaddingConfig, PredictionRequest


@pytest.mark.parametrize(
    "tokens,targets",
    [
        (128, (1024, 256)),
        (256, (2048, 512)),
        (384, (3072, 768)),
        (512, (4096, 1024)),
        (640, (6144, 1280)),
        (768, (6144, 1536)),
        (896, (8192, 1792)),
        (1024, (8192, 1856)),
        (1280, (12288, 2304)),
        (1536, (12288, 2752)),
    ],
)
def test_compact_v1_capacities_are_a_fixed_bounded_contract(tokens, targets):
    assert compact_targets(tokens) == targets


@pytest.mark.parametrize(
    "tokens,stored_tokens,stored_atoms,stored_structural,targets",
    [
        (1024, 952, 7584, 1816, (8192, 1856)),
        (1024, 1000, 7900, 1840, (8192, 1856)),
        (1536, 1428, 11376, 2724, (12288, 2752)),
        (1536, 1500, 12000, 2740, (12288, 2752)),
    ],
)
def test_compact_capacities_depend_on_bucket_instead_of_real_axes(
    tokens, stored_tokens, stored_atoms, stored_structural, targets
):
    assert (
        _resolve_auxiliary_targets(
            PaddingConfig(tokens=tokens, auxiliary_profile=COMPACT_PROFILE),
            token_storage=stored_tokens,
            atom_storage=stored_atoms,
            structural_storage=stored_structural,
        )
        == targets
    )


@pytest.mark.parametrize("axis", ["atoms", "structural_tokens"])
def test_compact_overflow_refuses_instead_of_creating_an_unwarmed_shape(axis):
    sizes = {
        "token_storage": 1008,
        "atom_storage": 5041,
        "structural_storage": 2016,
    }
    if axis == "atoms":
        sizes.update(atom_storage=8193, structural_storage=1816)
    with pytest.raises(ValueError, match=f"stored {axis}.*exceeds capacity"):
        _resolve_auxiliary_targets(
            PaddingConfig(tokens=1024, auxiliary_profile=COMPACT_PROFILE), **sizes
        )


def test_explicit_auxiliary_pins_override_compact_but_cannot_truncate_storage():
    config = PaddingConfig(
        tokens=1024,
        atoms=8257,
        structural_tokens=2049,
        auxiliary_profile=COMPACT_PROFILE,
    )
    sizes = {
        "token_storage": 1008,
        "atom_storage": 8193,
        "structural_storage": 2016,
    }
    assert _resolve_auxiliary_targets(config, **sizes) == (8257, 2049)
    for axis, undersized in (("atoms", 8192), ("structural_tokens", 2015)):
        with pytest.raises(ValueError, match="smaller than the input size"):
            _resolve_auxiliary_targets(
                dataclasses.replace(config, **{axis: undersized}),
                **sizes,
            )


def test_compact_targets_keep_cp_alignment_and_profile_metadata():
    config = cp_aligned_padding(
        PaddingConfig(tokens=1024, auxiliary_profile=COMPACT_PROFILE),
        cp_devices=3,
        cp_layout="1d",
    )
    assert _resolve_auxiliary_targets(
        config,
        token_storage=952,
        atom_storage=7584,
        structural_storage=1816,
    ) == (8256, 1920)
    assert cp_rows(config) == 3
    assert config.auxiliary_profile == COMPACT_PROFILE
    pinned = dataclasses.replace(config, atoms=7585, structural_tokens=1817)
    assert _resolve_auxiliary_targets(
        pinned,
        token_storage=952,
        atom_storage=7584,
        structural_storage=1816,
    ) == (7585, 1817)
    assert cp_rows(pinned) == 3


@pytest.mark.parametrize("overflow", ["error", "exact"])
def test_compact_does_not_silently_extend_its_frozen_grid(overflow):
    with pytest.raises(ValueError, match="no 2048-token bucket"):
        _resolve_auxiliary_targets(
            PaddingConfig(
                tokens=2048, auxiliary_profile=COMPACT_PROFILE, overflow=overflow
            ),
            token_storage=1700,
            atom_storage=12000,
            structural_storage=3000,
        )


def test_profile_summary_and_worker_transport_preserve_old_and_new_policies(tmp_path):
    legacy = {
        "tokens": 1024,
        "atoms": None,
        "msa": 2048,
        "templates": None,
        "structural_tokens": None,
        "language_model_tokens": None,
        "overflow": "error",
    }
    ordinary = PaddingConfig(tokens=1024, msa=2048)
    compact = dataclasses.replace(ordinary, auxiliary_profile=COMPACT_PROFILE)
    assert ordinary.summary() == legacy
    assert compact.summary() == {**legacy, "auxiliary_profile": COMPACT_PROFILE}

    source = tmp_path / "job.json"
    source.write_text('{"sequences": []}')
    for padding in (ordinary, compact):
        request = PredictionRequest(
            model="opendde", input=source, padding=padding, input_format="opendde"
        )
        packet = json.loads(json.dumps(_warm_process.request_packet(request)))
        assert _warm_process._request_from_packet(packet) == request


def test_host_profile_keeps_outer_namespace_and_actual_shapes_select_executables(
    tmp_path,
):
    source = tmp_path / "job.json"
    source.write_text('{"sequences": []}')
    weights = tmp_path / "weights.jax"
    weights.touch()
    request = PredictionRequest(
        model="opendde",
        input=source,
        input_format="opendde",
        weights=weights,
        cache_dir=tmp_path / "cache",
        padding=PaddingConfig(tokens=1024),
    )
    backend = OpenDDEBackend()
    compact = dataclasses.replace(
        request,
        padding=dataclasses.replace(request.padding, auxiliary_profile=COMPACT_PROFILE),
    )
    assert resolve_cache_dir(request, backend) == resolve_cache_dir(compact, backend)
    sizes = dict(token_storage=1008, atom_storage=5041, structural_storage=1512)
    assert _resolve_auxiliary_targets(request.padding, **sizes) != (
        _resolve_auxiliary_targets(compact.padding, **sizes)
    )


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "esmfold2", "openfold3", "protenix"]
)
def test_other_backends_reject_the_opendde_only_profile(model, tmp_path):
    source = tmp_path / "job.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="requires opendde"):
        get_backend(model).validate_request(
            PredictionRequest(
                model=model,
                input=source,
                padding=PaddingConfig(auxiliary_profile=COMPACT_PROFILE),
            )
        )


def test_cli_profile_implies_padding_and_composes_with_fixed_msa(tmp_path):
    source = tmp_path / "job.json"
    source.write_text('{"sequences": []}')
    arguments = [
        "predict",
        "--model",
        "opendde",
        "--input",
        str(source),
        "--input-format",
        "opendde",
        "--opendde-padding-profile",
        "compact",
        "--fixed-msa",
        "2048",
    ]
    request = _request(_parser().parse_args(arguments))
    assert request.padding.auxiliary_profile == COMPACT_PROFILE
    assert request.padding.msa == request.max_msa_depth == 2048
    arguments[2] = "protenix"
    with pytest.raises(ValueError, match="requires only model opendde"):
        _request(_parser().parse_args(arguments))


def test_compact_warm_auto_uses_only_supported_buckets():
    config = PaddingConfig(auxiliary_profile=COMPACT_PROFILE)
    assert warm_buckets.parse_token_buckets(
        "auto", model="opendde", padding=config
    ) == (128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536)
    with pytest.raises(ValueError, match="belong to the OpenDDE compact profile"):
        warm_buckets.parse_token_buckets("1792", model="opendde", padding=config)
    assert max(warm_buckets.parse_token_buckets("auto", model="opendde")) > 1536


def _native_graph_inputs(source: Path, padding: PaddingConfig, tmp_path: Path):
    from foldjax.input import materialize_native_input
    from foldjax.models._feature_storage import compact_msa_storage
    from foldjax.models.opendde.data.compact_categories import (
        compact_ref_atom_category_storage,
    )
    from foldjax.models.opendde.data.featurize_json import (
        featurize_opendde_json,
        load_jobs,
    )
    from foldjax.models.opendde.data.padding import (
        pad_opendde_features,
        select_opendde_model_features,
    )
    from foldjax.models.opendde.models.msa_sampling import (
        sample_opendde_msa_cycle_features,
    )
    from foldjax.models.protenix.data.template_features import (
        compact_zero_template_geometry,
        dedup_templates,
    )

    native = materialize_native_input(
        source, OpenDDEBackend().capabilities(), tmp_path, seed=101, msa="single"
    )
    features = compact_msa_storage(
        featurize_opendde_json(load_jobs(native)[0], n_queries=2, n_keys=4, seed=101)
    )
    cycles = sample_opendde_msa_cycle_features(features, num_recycles=2, seed=101)
    padded, padded_cycles, plan = pad_opendde_features(
        features, cycles, padding, n_queries=2, n_keys=4
    )
    model_features = compact_ref_atom_category_storage(
        compact_zero_template_geometry(
            dedup_templates(select_opendde_model_features(padded))
        )
    )
    for axis, mask in (
        ("tokens", "token_padding_mask"),
        ("atoms", "atom_padding_mask"),
        ("structural_tokens", "structural_token_padding_mask"),
    ):
        assert not np.any(model_features[mask][plan.actual[axis] :])
    signature = jax.tree_util.tree_map(
        lambda value: (np.shape(value), np.asarray(value).dtype.str),
        (model_features, padded_cycles),
    )
    return signature, plan


def test_native_warm_and_prediction_inputs_share_complete_graph_shapes(tmp_path):
    from foldjax.job import Job

    real = Job.from_sequences((("ACDEFGHIKLMNPQRSTVWY" * 6)[:110],)).write(
        tmp_path / "real.json"
    )
    request = PredictionRequest(
        model="opendde",
        input=real,
        padding=PaddingConfig(msa=2048, auxiliary_profile=COMPACT_PROFILE),
        msa="single",
    )
    signatures = []
    for index, length in enumerate(warm_buckets.representative_lengths(128)):
        warm = warm_buckets.bucket_request(
            request,
            model="opendde",
            bucket=128,
            length=length,
            index=index,
            jobs=tmp_path / "jobs",
        )
        sequence = json.loads(warm.input.read_text())["entities"][0]["sequence"]
        assert sequence == ("GA" * ((length + 1) // 2))[:length]
        signature, plan = _native_graph_inputs(
            warm.input, warm.padding, tmp_path / f"native-warm-{index}"
        )
        assert plan.target == {
            "tokens": 128,
            "atoms": 1024,
            "msa": 2048,
            "structural_tokens": 256,
        }
        if index == 0:
            ordinary_signature, ordinary_plan = _native_graph_inputs(
                warm.input,
                dataclasses.replace(warm.padding, auxiliary_profile=None),
                tmp_path / "native-ordinary",
            )
            assert ordinary_plan.target["atoms"] == 512
            assert ordinary_signature != signature
        signatures.append(signature)
    signature, plan = _native_graph_inputs(
        real, dataclasses.replace(request.padding, tokens=128), tmp_path / "native-real"
    )
    assert plan.actual["tokens"] == 110
    assert signature == signatures[0] == signatures[1]
