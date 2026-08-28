"""Managed Protenix atom categories stay private, compact, and lossless."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.data.compact_categories import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
    compact_ref_atom_category_storage,
    drop_dense_categories_from_writer_snapshot,
    validate_compact_ref_atom_categories,
)
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.output import write_protenix_outputs
from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models.model import (
    _restore_ref_atom_category_one_hot,
)
from foldjax.models.protenix.models.predict import protenix_predict_static
from tests.models.cp_probe_env import inherited_environment

from .test_model import _toy_features, _toy_params


def _dense_categories(n_atom: int = 4) -> dict[str, np.ndarray]:
    element = np.zeros((n_atom, 128), dtype=np.float32)
    chars = np.zeros((n_atom, 4, 64), dtype=np.float32)
    if n_atom > 0:
        element[0, 6] = 1.0
        chars[0, np.arange(4), [32, 33, 34, 35]] = 1.0
    if n_atom > 2:
        element[2, 127] = 1.0
        chars[2, :, 63] = 1.0
    if n_atom > 3:
        element[3, 8] = 1.0
        chars[3, np.arange(4), [1, 2, 3, 4]] = 1.0
    return {"ref_element": element, "ref_atom_name_chars": chars}


def _assert_same_bits(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.shape == expected_array.shape
    assert actual_array.tobytes() == expected_array.tobytes()


def test_exact_float32_categories_compact_and_restore_inside_jit() -> None:
    dense = _dense_categories()
    compact = compact_ref_atom_category_storage(dense)

    assert "ref_element" not in compact
    assert "ref_atom_name_chars" not in compact
    assert compact[COMPACT_REF_ATOM_CATEGORIES_MARKER].dtype == np.uint8
    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    np.testing.assert_array_equal(
        compact[COMPACT_REF_ELEMENT_IDS], np.asarray([6, 128, 127, 8], np.uint8)
    )
    np.testing.assert_array_equal(
        compact[COMPACT_REF_ATOM_NAME_CHAR_IDS][1], np.full(4, 64, np.uint8)
    )

    restored = jax.jit(_restore_ref_atom_category_one_hot)(compact)
    _assert_same_bits(restored["ref_element"], dense["ref_element"])
    _assert_same_bits(restored["ref_atom_name_chars"], dense["ref_atom_name_chars"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda array: array.astype(np.float64),
        lambda array: array.astype(np.float16),
        lambda array: jnp.asarray(array, dtype=jnp.float32),
        lambda array: np.full_like(array, np.nan),
        lambda array: np.full_like(array, np.inf),
        lambda array: np.full_like(array, -0.0),
        lambda array: np.full_like(array, -1.0),
        lambda array: np.full_like(array, 2.0),
    ],
)
def test_nonexact_or_non_numpy_float32_categories_fall_back(mutate) -> None:
    dense = _dense_categories()
    dense["ref_element"] = mutate(dense["ref_element"])
    compact = compact_ref_atom_category_storage(dense)

    assert compact["ref_element"] is dense["ref_element"]
    assert compact["ref_atom_name_chars"] is dense["ref_atom_name_chars"]
    assert not (set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & compact.keys())


def test_multihot_and_negative_zero_atom_names_fall_back() -> None:
    for mutate in ("multihot", "negative-zero"):
        dense = _dense_categories()
        if mutate == "multihot":
            dense["ref_atom_name_chars"][0, 0, 7] = 1.0
        else:
            dense["ref_atom_name_chars"][1, 0, 0] = -0.0
        compact = compact_ref_atom_category_storage(dense)
        assert compact["ref_element"] is dense["ref_element"]
        assert compact["ref_atom_name_chars"] is dense["ref_atom_name_chars"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ref_element", np.zeros((4, 127), dtype=np.float32)),
        ("ref_element", np.zeros((1, 4, 128), dtype=np.float32)),
        ("ref_element", np.zeros((0, 128), dtype=np.float32)),
        ("ref_atom_name_chars", np.zeros((4, 3, 64), dtype=np.float32)),
        ("ref_atom_name_chars", np.zeros((4, 4, 63), dtype=np.float32)),
    ],
)
def test_wrong_dense_category_shape_falls_back(field: str, value: np.ndarray) -> None:
    dense = _dense_categories()
    dense[field] = value
    compact = compact_ref_atom_category_storage(dense)
    assert "ref_element" in compact
    assert "ref_atom_name_chars" in compact
    assert compact[field] is value


def test_dense_categories_replace_or_strip_stale_private_provenance() -> None:
    dense = _dense_categories()
    dense.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(99, np.int16),
            COMPACT_REF_ELEMENT_IDS: np.asarray([255], np.uint8),
        }
    )
    compact = compact_ref_atom_category_storage(dense)
    assert set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) <= compact.keys()
    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    assert compact[COMPACT_REF_ELEMENT_IDS].shape == (4,)

    custom = _dense_categories()
    custom["ref_element"][0, 0] = -0.0
    custom.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
            COMPACT_REF_ELEMENT_IDS: np.zeros(4, np.uint8),
            COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((4, 4), np.uint8),
        }
    )
    fallback = compact_ref_atom_category_storage(custom)
    assert "ref_element" in fallback and "ref_atom_name_chars" in fallback
    assert not (set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & fallback.keys())

    restored = _restore_ref_atom_category_one_hot(dense)
    assert "ref_element" in restored and "ref_atom_name_chars" in restored
    assert not (set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & restored.keys())


@pytest.mark.parametrize(
    "private,match",
    [
        ({COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8)}, "incomplete"),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(2, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros(3, np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((3, 4), np.uint8),
            },
            "v1",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.int16),
                COMPACT_REF_ELEMENT_IDS: np.zeros(3, np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((3, 4), np.uint8),
            },
            "scalar uint8",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros((1, 3), np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 3, 4), np.uint8),
            },
            r"\[A\]",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([129], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 4), np.uint8),
            },
            "sentinel 128",
        ),
    ],
)
def test_marker_bearing_compact_provenance_is_strict(private, match: str) -> None:
    with pytest.raises((KeyError, ValueError), match=match):
        validate_compact_ref_atom_categories(private)
    with pytest.raises((KeyError, ValueError), match=match):
        compact_ref_atom_category_storage(private)
    with pytest.raises((KeyError, ValueError), match=match):
        _restore_ref_atom_category_one_hot(private)


def test_compact_projection_is_bit_exact_and_has_one_stable_pytree_cache_key() -> None:
    dense = _dense_categories()
    compact = compact_ref_atom_category_storage(dense)
    changed_dense = _dense_categories()
    changed_dense["ref_element"][:] = 0.0
    changed_dense["ref_element"][:, 7] = 1.0
    changed = compact_ref_atom_category_storage(changed_dense)
    element_weight = jnp.arange(128 * 8, dtype=jnp.float32).reshape(128, 8)
    char_weight = jnp.arange(256 * 8, dtype=jnp.float32).reshape(256, 8)
    traces: list[None] = []

    @jax.jit
    def project(features):
        traces.append(None)
        restored = _restore_ref_atom_category_one_hot(features)
        return (
            restored["ref_element"] @ element_weight
            + restored["ref_atom_name_chars"].reshape((-1, 256)) @ char_weight
        )

    expected = (
        jnp.asarray(dense["ref_element"]) @ element_weight
        + jnp.asarray(dense["ref_atom_name_chars"]).reshape((-1, 256)) @ char_weight
    )
    _assert_same_bits(project(compact), expected)
    project(changed).block_until_ready()
    assert len(traces) == 1

    hlo = str(project.lower(compact).compiler_ir(dialect="stablehlo"))
    assert "tensor<4xui8>" in hlo
    assert "tensor<4x4xui8>" in hlo
    main_signature = next(line for line in hlo.splitlines() if "@main(" in line)
    assert "tensor<4x128xf32>" not in main_signature
    assert "tensor<4x4x64xf32>" not in main_signature


def test_compact_projection_reduces_compiled_argument_storage() -> None:
    n_atom = 2048
    element_ids = np.arange(n_atom) % 128
    char_ids = np.arange(n_atom * 4).reshape(n_atom, 4) % 64
    dense = {
        "ref_element": np.eye(128, dtype=np.float32)[element_ids],
        "ref_atom_name_chars": np.eye(64, dtype=np.float32)[char_ids],
    }
    compact = compact_ref_atom_category_storage(dense)
    element_weight = jnp.arange(128 * 32, dtype=jnp.float32).reshape(128, 32)
    char_weight = jnp.arange(256 * 32, dtype=jnp.float32).reshape(256, 32)

    def dense_project(features):
        return (
            features["ref_element"] @ element_weight
            + features["ref_atom_name_chars"].reshape((-1, 256)) @ char_weight
        )

    def compact_project(features):
        restored = _restore_ref_atom_category_one_hot(features)
        return dense_project(restored)

    dense_lowered = jax.jit(dense_project).lower(dense)
    compact_lowered = jax.jit(compact_project).lower(compact)
    dense_output = dense_lowered.compile()(dense)
    compact_output = compact_lowered.compile()(compact)
    _assert_same_bits(compact_output, dense_output)

    dense_memory = dense_lowered.compile().memory_analysis()
    compact_memory = compact_lowered.compile().memory_analysis()
    assert compact_memory.argument_size_in_bytes * 100 < (
        dense_memory.argument_size_in_bytes
    )


def test_full_compiled_toy_prediction_is_byte_identical() -> None:
    dense = dict(_toy_features())
    dense.update(_dense_categories(n_atom=3))
    compact = compact_ref_atom_category_storage(dense)
    kwargs = {
        "key": None,
        "num_samples": 1,
        "num_sampling_steps": 1,
        "recycling_steps": 1,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "n_queries": 2,
        "n_keys": 4,
        "sigma_data": 4.0,
        "centre_each_step": False,
        "init_noise": jnp.ones((1, 3, 3), dtype=jnp.float32),
        "step_noises": (jnp.zeros((1, 3, 3), dtype=jnp.float32),),
        "graph_jit": True,
        "trunk_dtype": jnp.bfloat16,
    }

    expected = protenix_predict_static(_toy_params(), dense, **kwargs)
    actual = protenix_predict_static(_toy_params(), compact, **kwargs)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        _assert_same_bits(actual[name], value)


def test_writer_snapshot_without_dense_categories_is_byte_identical(tmp_path) -> None:
    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]},
        n_queries=2,
        n_keys=4,
    )
    pruned = drop_dense_categories_from_writer_snapshot(features)
    assert set(features) - set(pruned) == {
        "ref_element",
        "ref_atom_name_chars",
    }

    n_atom = len(features["atom_to_token_idx"])
    output = {
        "coordinate": np.arange(n_atom * 3, dtype=np.float32).reshape(1, n_atom, 3),
        "atom_plddt": np.full((1, n_atom), 0.75, dtype=np.float32),
        "summary_plddt": np.asarray([75.0], dtype=np.float32),
        "summary_ranking_score": np.asarray([0.75], dtype=np.float32),
    }
    dense_root = tmp_path / "dense"
    pruned_root = tmp_path / "pruned"
    dense_paths = write_protenix_outputs(
        dense_root,
        job_name="same-job",
        seed=7,
        output=output,
        features=features,
    )
    pruned_paths = write_protenix_outputs(
        pruned_root,
        job_name="same-job",
        seed=7,
        output=output,
        features=pruned,
    )
    assert [path.relative_to(dense_root) for path in dense_paths] == [
        path.relative_to(pruned_root) for path in pruned_paths
    ]
    for dense_path, pruned_path in zip(dense_paths, pruned_paths, strict=True):
        assert dense_path.read_bytes() == pruned_path.read_bytes()


def test_writer_snapshot_keeps_dense_fallback_when_metadata_is_incomplete() -> None:
    dense = _dense_categories()
    assert drop_dense_categories_from_writer_snapshot(dense) is dense

    partial = {**dense, "output_atom_name": np.asarray(["A"] * 4)}
    assert drop_dense_categories_from_writer_snapshot(partial) is partial

    complete_but_short = {
        **dense,
        "output_atom_name": np.asarray(["A"] * 4),
        "output_atom_element": np.asarray(["C"] * 4),
        "output_atom_res_name": np.asarray(["ALA"] * 4),
        "output_atom_chain_id": np.asarray(["A"] * 4),
        "output_atom_res_id": np.arange(3),
    }
    assert (
        drop_dense_categories_from_writer_snapshot(complete_but_short)
        is complete_but_short
    )


@pytest.mark.parametrize(
    ("mode", "expect_compact"),
    [
        ("generated-compiled", True),
        ("generated-eager", False),
        ("generated-tfg", False),
        ("static-compiled", False),
    ],
)
def test_cli_compacts_only_generated_consolidated_graph_inputs(
    tmp_path, monkeypatch, mode: str, expect_compact: bool
) -> None:
    from foldjax.models.protenix.cli import predict as predict_impl

    captured: dict[str, object] = {}

    def fake_predict(_params, features, **kwargs):
        captured["features"] = features
        captured["kwargs"] = kwargs
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    common = [
        "--model-name",
        "unknown",
        "--weights",
        str(tmp_path / "unused-weights.jax"),
        "--out",
        str(tmp_path / "unused-output.npz"),
        "--n-sample",
        "1",
        "--n-step",
        "1",
        "--n-cycle",
        "1",
        "--n-queries",
        "2",
        "--n-keys",
        "4",
        "--trunk-dtype",
        "fp32",
        "--prewarm-only",
        "--cpu-only",
        "--no-compile-cache",
    ]
    extra: list[str] = []
    if mode == "static-compiled":
        feature_path = tmp_path / "features.npz"
        save_static_feature_npz(feature_path, _toy_features())
        input_args = ["--features", str(feature_path)]
    else:
        input_path = tmp_path / "input.json"
        input_path.write_text(
            '[{"sequences": [{"proteinChain": {"sequence": "A"}}]}]',
            encoding="utf-8",
        )
        input_args = ["--input-json", str(input_path)]
        if mode == "generated-eager":
            extra.append("--no-graph-jit")
        elif mode == "generated-tfg":
            guidance_path = tmp_path / "guidance.json"
            guidance_path.write_text(
                '{"enable": true, "terms": {'
                '"InterchainBondPotential": {"weight": 0.0}}}',
                encoding="utf-8",
            )
            extra.extend(["--guidance-config", str(guidance_path)])

    predict_impl.main(
        [*common, *input_args, *extra],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )
    model_features = captured["features"]
    assert isinstance(model_features, dict)
    if expect_compact:
        assert (
            set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) <= model_features.keys()
        )
        assert "ref_element" not in model_features
        assert "ref_atom_name_chars" not in model_features
    else:
        assert "ref_element" in model_features
        assert "ref_atom_name_chars" in model_features
        assert not (
            set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & model_features.keys()
        )


def test_cli_compacts_after_padding_and_uses_zero_row_sentinels(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.protenix.cli import predict as predict_impl

    input_path = tmp_path / "input.json"
    input_path.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A"}}]}]',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_predict(_params, features, **_kwargs):
        captured.update(features)
        return {"coordinate": np.zeros((1, 8, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    predict_impl.main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(tmp_path / "unused-weights.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "unused-output.npz"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--trunk-dtype",
            "fp32",
            "--padding",
            "--pad-tokens",
            "8",
            "--pad-atoms",
            "8",
            "--pad-msa",
            "4",
            "--pad-templates",
            "4",
            "--prewarm-only",
            "--cpu-only",
            "--no-compile-cache",
        ],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: (),
    )
    element_ids = np.asarray(captured[COMPACT_REF_ELEMENT_IDS])
    char_ids = np.asarray(captured[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    atom_mask = np.asarray(captured["atom_padding_mask"]).astype(bool)
    assert element_ids.shape == (8,)
    assert char_ids.shape == (8, 4)
    np.testing.assert_array_equal(element_ids[~atom_mask], 128)
    np.testing.assert_array_equal(char_ids[~atom_mask], 64)


@pytest.mark.parametrize("mode", ["plain", "padded", "esm"])
def test_cli_releases_all_dense_category_arrays_before_loading_params(
    tmp_path, monkeypatch, mode: str
) -> None:
    from foldjax.models.protenix.cli import predict as predict_impl

    input_path = tmp_path / "input.json"
    input_path.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A"}}]}]',
        encoding="utf-8",
    )
    dense_refs: list[weakref.ReferenceType[np.ndarray]] = []
    real_compact = predict_impl.compact_ref_atom_category_storage
    real_drop = predict_impl.drop_dense_categories_from_writer_snapshot

    def remember_dense(features) -> None:
        for name in ("ref_element", "ref_atom_name_chars"):
            value = features.get(name)
            if isinstance(value, np.ndarray):
                dense_refs.append(weakref.ref(value))

    def recording_compact(features):
        remember_dense(features)
        return real_compact(features)

    def recording_drop(features):
        remember_dense(features)
        return real_drop(features)

    monkeypatch.setattr(
        predict_impl, "compact_ref_atom_category_storage", recording_compact
    )
    monkeypatch.setattr(
        predict_impl, "drop_dense_categories_from_writer_snapshot", recording_drop
    )
    if mode == "esm":

        class FakeProvider:
            def __init__(self, _model_name, *, checkpoint_dir):
                del checkpoint_dir

            def __call__(self, sequence):
                return np.ones((len(sequence), 2560), dtype=np.float32)

        monkeypatch.setattr(
            "foldjax.models.protenix.data.esm.JaxEsmProvider", FakeProvider
        )

    def params_loader(_path, _dtype, _cacheable):
        gc.collect()
        assert dense_refs
        assert all(reference() is None for reference in dense_refs)
        return ()

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static",
        lambda _params, features, **_kwargs: {
            "coordinate": np.zeros(
                (1, len(features["atom_to_token_idx"]), 3), dtype=np.float32
            )
        },
    )
    extra: list[str] = []
    if mode == "padded":
        extra = [
            "--padding",
            "--pad-tokens",
            "8",
            "--pad-atoms",
            "8",
            "--pad-msa",
            "4",
            "--pad-templates",
            "4",
        ]
    model_name = "protenix_mini_esm_v0.5.0" if mode == "esm" else "unknown"
    predict_impl.main(
        [
            "--model-name",
            model_name,
            "--weights",
            str(tmp_path / "unused-weights.jax"),
            "--input-json",
            str(input_path),
            "--out",
            str(tmp_path / "unused-output.npz"),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--trunk-dtype",
            "fp32",
            "--prewarm-only",
            "--cpu-only",
            "--no-compile-cache",
            *extra,
        ],
        _prepared_params_loader=params_loader,
    )


def test_compiled_host_gate_rejects_private_provenance_before_tracing(
    monkeypatch,
) -> None:
    called = False

    def unexpected_compiled(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", unexpected_compiled)
    features = {
        "asym_id": np.asarray([0], dtype=np.int32),
        COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, dtype=np.uint8),
    }
    with pytest.raises(KeyError, match="incomplete"):
        model_impl.protenix_infer_compiled(
            features,
            (),
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        )
    assert called is False


def test_padded_graph_allowlist_retains_all_private_category_fields(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compiled(features, *_args, **_kwargs):
        captured.update(features)
        return {}

    monkeypatch.setattr(model_impl, "_compiled_protenix_infer", fake_compiled)
    compact = compact_ref_atom_category_storage(_dense_categories(n_atom=3))
    features = {
        **compact,
        "asym_id": np.asarray([0], dtype=np.int32),
        "token_padding_mask": np.asarray([1], dtype=np.float32),
        "writer_only": np.zeros((9,), dtype=np.float32),
    }
    model_impl.protenix_infer_compiled(
        features,
        (),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        padded_generated_schema=True,
    )
    assert set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) <= captured.keys()
    assert "writer_only" not in captured


_CP_PROBE = textwrap.dedent(
    """
    import os

    import jax
    import numpy as np

    from foldjax.models._cp import context_parallel, replicate_tree
    from foldjax.models.protenix.data.compact_categories import (
        compact_ref_atom_category_storage,
    )
    from foldjax.models.protenix.models.model import (
        _restore_ref_atom_category_one_hot,
    )

    assert jax.device_count() == 4, jax.devices()
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    dense = {
        "ref_element": np.eye(128, dtype=np.float32)[[6, 7, 8]],
        "ref_atom_name_chars": np.eye(64, dtype=np.float32)[
            [[32, 32, 32, 32], [33, 33, 33, 33], [34, 34, 34, 34]]
        ],
    }
    compact = compact_ref_atom_category_storage(dense)

    def run(features):
        restored = _restore_ref_atom_category_one_hot(features)
        return restored["ref_element"], restored["ref_atom_name_chars"]

    with context_parallel(4, layout=layout):
        placed = replicate_tree(compact)
        executable = jax.jit(run).lower(placed).compile()
        element, chars = map(np.asarray, executable(placed))
        hlo = executable.as_text().lower()
    assert element.tobytes() == dense["ref_element"].tobytes()
    assert chars.tobytes() == dense["ref_atom_name_chars"].tobytes()
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    print("PROTENIX_COMPACT_CATEGORY_CP_OK")
    """
)


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_compact_categories_remain_exact_on_four_cpu_devices(layout: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_CP_PROBE_LAYOUT": layout,
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PROTENIX_COMPACT_CATEGORY_CP_OK" in completed.stdout
