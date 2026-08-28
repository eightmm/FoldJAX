"""OpenFold3's managed atom-category storage is private and lossless."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models.openfold3.data import (
    compact_ref_atom_category_storage,
    load_feature_archive,
    pad_features,
    save_features,
    validate_output_metadata,
)
from foldjax.models.openfold3.data.compact_categories import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
    validate_compact_ref_atom_categories,
)
from foldjax.models.openfold3.inference import (
    _restore_ref_atom_category_one_hot,
)
from foldjax.models.openfold3.models.frames import _named_atom_indices
from foldjax.models.openfold3.models.primitives import LinearParams, linear
from foldjax.schema import PredictionRequest
from tests.models.cp_probe_env import inherited_environment

from .feature_fixture import minimal_features
from .test_output_metadata import (
    _features as output_features,
)
from .test_output_metadata import (
    _metadata as output_metadata,
)
from .test_output_metadata import (
    _prediction as output_prediction,
)
from .test_output_metadata import (
    _table as output_table,
)


def _dense_features(*, atoms: int = 7, padded_atoms: int = 2) -> dict[str, np.ndarray]:
    dense = minimal_features(tokens=4, atoms=atoms)
    element_ids = np.arange(atoms, dtype=np.int32) % 119
    dense["ref_element"].fill(0)
    np.put_along_axis(
        dense["ref_element"], element_ids[None, :, None], 1, axis=-1
    )
    names = ("N", "CA", "C", "O", "C3'", "C1'", "C4'")
    dense["ref_atom_name_chars"].fill(0)
    for atom, name in enumerate(names[index % len(names)] for index in range(atoms)):
        for position, character in enumerate(name.ljust(4)):
            dense["ref_atom_name_chars"][
                0, atom, position, ord(character) - 32
            ] = 1
    if padded_atoms:
        dense = pad_features(
            dense,
            n_token=4,
            n_atom=atoms + padded_atoms,
            n_msa=2,
            n_templates=1,
        )
    return dense


def _category_mapping(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: features[name]
        for name in ("atom_mask", "ref_element", "ref_atom_name_chars")
    }


def _assert_same(actual: jax.Array, expected: jax.Array) -> None:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    assert actual_np.dtype == expected_np.dtype
    assert actual_np.shape == expected_np.shape
    assert actual_np.tobytes() == expected_np.tobytes()


def test_exact_categories_compact_after_padding_and_restore_inside_jit() -> None:
    dense = _dense_features(atoms=7, padded_atoms=3)
    compact = compact_ref_atom_category_storage(dense)

    assert "ref_element" not in compact
    assert "ref_atom_name_chars" not in compact
    assert compact[COMPACT_REF_ATOM_CATEGORIES_MARKER].dtype == np.uint8
    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    np.testing.assert_array_equal(
        compact[COMPACT_REF_ELEMENT_IDS][0, :7], np.arange(7, dtype=np.uint8)
    )
    assert np.all(compact[COMPACT_REF_ELEMENT_IDS][0, 7:] == 119)
    assert np.all(compact[COMPACT_REF_ATOM_NAME_CHAR_IDS][0, 7:] == 64)
    validate_compact_ref_atom_categories(compact)

    restored = jax.jit(_restore_ref_atom_category_one_hot)(compact)
    _assert_same(restored["ref_element"], dense["ref_element"])
    _assert_same(restored["ref_atom_name_chars"], dense["ref_atom_name_chars"])


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("element-float32", id="fp32-custom"),
        pytest.param("chars-bfloat16", id="bf16-custom"),
        pytest.param("element-nan", id="nonfinite"),
        pytest.param("element-negative-zero", id="signed-zero"),
        pytest.param("element-multi-hot", id="multi-hot"),
        pytest.param("live-element-zero", id="zero-live-category"),
        pytest.param("padded-char-nonzero", id="nonzero-padding"),
        pytest.param("wrong-element-shape", id="shape"),
        pytest.param("wrong-mask-dtype", id="mask-dtype"),
        pytest.param("all-padding-mask", id="empty-mask-prefix"),
        pytest.param("interleaved-mask", id="interleaved-mask"),
    ],
)
def test_custom_or_nonexact_dense_categories_fall_back(mutation: str) -> None:
    dense = _dense_features(atoms=7, padded_atoms=2)
    if mutation == "element-float32":
        dense["ref_element"] = dense["ref_element"].astype(np.float32)
    elif mutation == "chars-bfloat16":
        dense["ref_atom_name_chars"] = jnp.asarray(
            dense["ref_atom_name_chars"], dtype=jnp.bfloat16
        )
    elif mutation == "element-nan":
        dense["ref_element"] = dense["ref_element"].astype(np.float32)
        dense["ref_element"][0, 0, 0] = np.nan
    elif mutation == "element-negative-zero":
        dense["ref_element"] = dense["ref_element"].astype(np.float32)
        dense["ref_element"][0, 0, 1] = -0.0
    elif mutation == "element-multi-hot":
        dense["ref_element"][0, 0, 1] = 1
    elif mutation == "live-element-zero":
        dense["ref_element"][0, 0] = 0
    elif mutation == "padded-char-nonzero":
        dense["ref_atom_name_chars"][0, -1, 0, 0] = 1
    elif mutation == "wrong-element-shape":
        dense["ref_element"] = dense["ref_element"][..., :-1]
    elif mutation == "wrong-mask-dtype":
        dense["atom_mask"] = dense["atom_mask"].astype(np.int32)
    elif mutation == "all-padding-mask":
        dense["atom_mask"].fill(0)
        dense["ref_element"].fill(0)
        dense["ref_atom_name_chars"].fill(0)
    elif mutation == "interleaved-mask":
        dense["atom_mask"].fill(0)
        dense["atom_mask"][0, [0, 2]] = 1
        padded = ~dense["atom_mask"].astype(bool)
        dense["ref_element"][padded] = 0
        dense["ref_atom_name_chars"][padded] = 0
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(mutation)

    compact = compact_ref_atom_category_storage(dense)

    assert "ref_element" in compact
    assert "ref_atom_name_chars" in compact
    assert not (set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & compact.keys())


def test_dense_categories_win_over_stale_private_provenance() -> None:
    dense = _dense_features(padded_atoms=0)
    dense.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(99, dtype=np.int16),
            COMPACT_REF_ELEMENT_IDS: np.asarray([255], dtype=np.uint8),
        }
    )

    compact = compact_ref_atom_category_storage(dense)
    restored = _restore_ref_atom_category_one_hot(compact)

    assert int(compact[COMPACT_REF_ATOM_CATEGORIES_MARKER]) == 1
    _assert_same(restored["ref_element"], dense["ref_element"])
    _assert_same(restored["ref_atom_name_chars"], dense["ref_atom_name_chars"])


@pytest.mark.parametrize(
    ("private", "match"),
    [
        (
            {COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8)},
            "incomplete",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(2, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros((1, 2), np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 2, 4), np.uint8),
                "atom_mask": np.ones((1, 2), np.float32),
            },
            "v1",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([[120]], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 1, 4), np.uint8),
                "atom_mask": np.ones((1, 1), np.float32),
            },
            "sentinel 119",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros((1, 2), np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((2, 4), np.uint8),
                "atom_mask": np.ones((1, 2), np.float32),
            },
            "shapes",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.zeros((1, 2), np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.zeros((1, 2, 4), np.uint8),
            },
            "need atom_mask",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([[5, 119]], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.asarray(
                    [[[1, 2, 3, 4], [64, 64, 64, 64]]], np.uint8
                ),
                "atom_mask": np.ones((1, 2), np.float32),
            },
            "sentinel 119 must identify exactly",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([[5, 6]], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.asarray(
                    [[[1, 2, 3, 4], [64, 64, 64, 64]]], np.uint8
                ),
                "atom_mask": np.asarray([[1, 0]], np.float32),
            },
            "sentinel 119 must identify exactly",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([[5, 119]], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.asarray(
                    [[[1, 2, 3, 4], [5, 6, 7, 8]]], np.uint8
                ),
                "atom_mask": np.asarray([[1, 0]], np.float32),
            },
            "sentinel 64 must identify exactly",
        ),
        (
            {
                COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
                COMPACT_REF_ELEMENT_IDS: np.asarray([[5, 6, 7]], np.uint8),
                COMPACT_REF_ATOM_NAME_CHAR_IDS: np.asarray(
                    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], np.uint8
                ),
                "atom_mask": np.asarray([[1, 0, 1]], np.float32),
            },
            "real prefix",
        ),
    ],
)
def test_marker_bearing_private_provenance_is_strict(private, match: str) -> None:
    with pytest.raises((KeyError, ValueError), match=match):
        validate_compact_ref_atom_categories(private)
    with pytest.raises((KeyError, ValueError), match=match):
        compact_ref_atom_category_storage(private)
    with pytest.raises((KeyError, ValueError), match=match):
        _restore_ref_atom_category_one_hot(private)


@pytest.mark.parametrize("operation", ("call", "lower"))
def test_compiled_boundary_rejects_bad_sentinels_before_tracing(
    monkeypatch, operation: str
) -> None:
    from foldjax.models.openfold3 import inference

    class MustNotTrace:
        def __call__(self, *args, **kwargs):
            raise AssertionError("malformed compact features reached tracing")

        def lower(self, *args, **kwargs):
            raise AssertionError("malformed compact features reached lowering")

    monkeypatch.setattr(inference, "_compiled_predict", MustNotTrace())
    config = inference.released_config(
        n_token=1,
        n_atom=2,
        num_recycles=1,
        num_samples=1,
        num_steps=1,
        pair_chunk_size=None,
        msa_depth=1,
    )
    run = inference.compile_predict(config, output_table())
    malformed = {
        COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(1, np.uint8),
        COMPACT_REF_ELEMENT_IDS: np.asarray([[5, 6]], np.uint8),
        COMPACT_REF_ATOM_NAME_CHAR_IDS: np.asarray(
            [[[1, 2, 3, 4], [64, 64, 64, 64]]], np.uint8
        ),
        "atom_mask": np.asarray([[1, 0]], np.float32),
    }

    invoke = run if operation == "call" else run.lower
    with pytest.raises(ValueError, match="sentinel 119 must identify exactly"):
        invoke(jax.random.key(0), malformed, np.asarray(0.0, np.float32))


def test_compiled_boundary_strips_stale_private_keys_before_jit(monkeypatch) -> None:
    from foldjax.models.openfold3 import inference

    seen: dict[str, object] = {}

    class CaptureGraph:
        def __call__(self, key, batch, params, table, tape, mask, *, identity):
            seen["features"] = batch
            return np.asarray(7, np.int32)

    monkeypatch.setattr(inference, "_compiled_predict", CaptureGraph())
    dense = _category_mapping(_dense_features(atoms=7, padded_atoms=0))
    dense.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(99, np.int16),
            COMPACT_REF_ELEMENT_IDS: np.asarray([[255]], np.uint8),
        }
    )
    config = inference.released_config(
        n_token=4,
        n_atom=7,
        num_recycles=1,
        num_samples=1,
        num_steps=1,
        pair_chunk_size=None,
        msa_depth=1,
    )

    result = inference.compile_predict(config, output_table())(
        jax.random.key(0), dense, np.asarray(0.0, np.float32)
    )

    assert int(result) == 7
    traced = seen["features"]
    assert "ref_element" in traced
    assert "ref_atom_name_chars" in traced
    assert not (
        set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & traced.keys()
    )


def _tiny_category_predict(
    key,
    batch,
    params,
    config,
    representative_atoms,
    **kwargs,
):
    del key, config, representative_atoms, kwargs
    restored = _restore_ref_atom_category_one_hot(batch)
    return jnp.sum(restored["ref_element"], dtype=jnp.float32) + params


def _compiled_category_probe(inference, monkeypatch, features):
    monkeypatch.setattr(inference, "predict", _tiny_category_predict)
    inference._compiled_predict.clear_cache()
    config = inference.released_config(
        n_token=4,
        n_atom=features["atom_mask"].shape[1],
        num_recycles=1,
        num_samples=1,
        num_steps=1,
        pair_chunk_size=None,
        msa_depth=1,
    )
    args = (
        jax.random.key(0),
        features,
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    compiled = (
        inference.compile_predict(config, output_table()).lower(*args).compile()
    )
    return compiled, args


def test_bound_compiled_call_rejects_same_shape_malformed_private_ids(
    monkeypatch,
) -> None:
    from foldjax.models.openfold3 import inference

    dense = _category_mapping(_dense_features(atoms=7, padded_atoms=2))
    compact = compact_ref_atom_category_storage(dense)
    compiled, args = _compiled_category_probe(inference, monkeypatch, compact)
    malformed = dict(compact)
    malformed[COMPACT_REF_ELEMENT_IDS] = np.array(
        compact[COMPACT_REF_ELEMENT_IDS], copy=True
    )
    malformed[COMPACT_REF_ELEMENT_IDS][0, -1] = 5
    try:
        with pytest.raises(
            ValueError, match="sentinel 119 must identify exactly"
        ):
            compiled(args[0], malformed, args[2])
    finally:
        inference._compiled_predict.clear_cache()


def test_bound_dense_compiled_call_strips_stale_private_leaves(
    monkeypatch,
) -> None:
    from foldjax.models.openfold3 import inference

    dense = _category_mapping(_dense_features(atoms=7, padded_atoms=0))
    compiled, args = _compiled_category_probe(inference, monkeypatch, dense)
    stale = dict(dense)
    stale.update(
        {
            COMPACT_REF_ATOM_CATEGORIES_MARKER: np.asarray(99, np.int16),
            COMPACT_REF_ELEMENT_IDS: np.asarray([[255]], np.uint8),
        }
    )
    try:
        expected = compiled(*args)
        actual = compiled(args[0], stale, args[2])
        jax.block_until_ready((expected, actual))
        _assert_same(actual, expected)
    finally:
        inference._compiled_predict.clear_cache()


def _projection(
    features,
    element_weight: jax.Array,
    char_weight: jax.Array,
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    restored = _restore_ref_atom_category_one_hot(features)
    element = linear(
        restored["ref_element"].astype(dtype), LinearParams(element_weight)
    )
    chars = restored["ref_atom_name_chars"]
    chars = chars.reshape((*chars.shape[:-2], 4 * 64)).astype(dtype)
    return element + linear(chars, LinearParams(char_weight))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_dense_and_compact_existing_projections_are_byte_identical(dtype) -> None:
    dense = _category_mapping(_dense_features(atoms=11, padded_atoms=3))
    compact = compact_ref_atom_category_storage(dense)
    rng = np.random.default_rng(41)
    element_weight = jnp.asarray(rng.normal(size=(8, 119)), dtype=dtype)
    char_weight = jnp.asarray(rng.normal(size=(8, 4 * 64)), dtype=dtype)
    run = jax.jit(
        lambda source, ew, cw: _projection(source, ew, cw, dtype=dtype)
    )

    expected = run(dense, element_weight, char_weight)
    actual = run(compact, element_weight, char_weight)
    jax.block_until_ready((expected, actual))

    _assert_same(actual, expected)


def test_nonfinite_weights_and_signed_zero_keep_dense_projection_semantics() -> None:
    dense = _category_mapping(_dense_features(atoms=9, padded_atoms=3))
    compact = compact_ref_atom_category_storage(dense)
    rng = np.random.default_rng(43)
    element_weight = rng.normal(size=(8, 119)).astype(np.float32)
    char_weight = rng.normal(size=(8, 4 * 64)).astype(np.float32)
    for weight in (element_weight, char_weight):
        weight[0, 0] = np.nan
        weight[1, 1] = np.inf
        weight[2, 2] = -np.inf
        weight[3, 3] = -0.0
    run = jax.jit(
        lambda source, ew, cw: _projection(
            source, ew, cw, dtype=jnp.float32
        )
    )

    expected = np.asarray(run(dense, element_weight, char_weight))
    actual = np.asarray(run(compact, element_weight, char_weight))

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert actual[finite].tobytes() == expected[finite].tobytes()
    zeros = finite & (expected == 0)
    np.testing.assert_array_equal(
        np.signbit(actual[zeros]), np.signbit(expected[zeros])
    )


def test_frame_name_selection_reads_the_same_historical_one_hot() -> None:
    dense = _dense_features(atoms=7, padded_atoms=2)
    compact = compact_ref_atom_category_storage(dense)

    @jax.jit
    def frames(source):
        restored = _restore_ref_atom_category_one_hot(source)
        values = [
            _named_atom_indices(restored, name, n_atom=9)
            for name in ("N", "CA", "C", "C3'", "C1'", "C4'")
        ]
        return tuple(leaf for pair in values for leaf in pair)

    expected = frames(dense)
    actual = frames(compact)
    for dense_leaf, compact_leaf in zip(expected, actual, strict=True):
        _assert_same(compact_leaf, dense_leaf)


def test_compact_storage_reduces_jit_argument_bytes_and_dense_hlo_inputs() -> None:
    dense_features = _dense_features(atoms=64, padded_atoms=0)
    dense = _category_mapping(dense_features)
    compact = compact_ref_atom_category_storage(dense)
    rng = np.random.default_rng(47)
    element_weight = rng.normal(size=(8, 119)).astype(np.float32)
    char_weight = rng.normal(size=(8, 4 * 64)).astype(np.float32)
    run = jax.jit(
        lambda source, ew, cw: _projection(
            source, ew, cw, dtype=jnp.float32
        )
    )

    dense_lowered = run.lower(dense, element_weight, char_weight)
    compact_lowered = run.lower(compact, element_weight, char_weight)
    dense_compiled = dense_lowered.compile()
    compact_compiled = compact_lowered.compile()
    dense_memory = dense_compiled.memory_analysis()
    compact_memory = compact_compiled.memory_analysis()
    compact_hlo = compact_lowered.as_text()

    assert (
        compact_memory.argument_size_in_bytes * 4
        < dense_memory.argument_size_in_bytes
    )
    assert compact_memory.output_size_in_bytes == dense_memory.output_size_in_bytes
    entry = next(line for line in compact_hlo.splitlines() if "public @main" in line)
    assert "tensor<1x64xui8>" in entry
    assert "tensor<1x64x4xui8>" in entry
    assert "tensor<1x64x119xi32>" not in entry
    assert "tensor<1x64x4x64xi32>" not in entry
    _assert_same(
        compact_compiled(compact, element_weight, char_weight),
        dense_compiled(dense, element_weight, char_weight),
    )


def test_portable_archive_and_writer_sidecar_keep_dense_categories(
    tmp_path: Path,
) -> None:
    dense = output_features()
    metadata = output_metadata()
    archive = save_features(
        dense,
        tmp_path / "sidecar.npz",
        representative_atoms=output_table(),
        output_metadata=metadata,
    )

    with np.load(archive, allow_pickle=False) as payload:
        assert "ref_element" in payload.files
        assert "ref_atom_name_chars" in payload.files
        assert not (
            set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & set(payload.files)
        )
    loaded, _table, restored_metadata = load_feature_archive(archive)
    assert restored_metadata is not None
    model_features = compact_ref_atom_category_storage(loaded)

    assert "ref_element" not in model_features
    assert "ref_atom_name_chars" not in model_features
    assert "ref_element" in loaded
    assert "ref_atom_name_chars" in loaded
    exact = validate_output_metadata(restored_metadata, loaded)
    np.testing.assert_array_equal(exact.atom_name, metadata.atom_name)

    from foldjax.models.openfold3.output import write_prediction_outputs

    written = write_prediction_outputs(
        output_prediction(),
        loaded,
        tmp_path / "written",
        output_metadata=restored_metadata,
    )
    assert written["structures"][0].is_file()


@pytest.mark.parametrize(
    ("suffix", "input_format", "expected_route"),
    [
        (".json", "openfold3", "raw"),
        (".npz", "openfold3-features", "archive"),
    ],
)
def test_managed_raw_and_archive_paths_compact_only_the_model_copy(
    tmp_path: Path,
    monkeypatch,
    suffix: str,
    input_format: str,
    expected_route: str,
) -> None:
    from foldjax.models.openfold3.data import (
        compact_ref_atom_category_storage as compact_categories,
    )
    from foldjax.models.openfold3.data import normalize_asym_ids

    dense = minimal_features(tokens=4, atoms=7)
    metadata = object()
    table = object()
    prediction = object()
    seen: dict[str, object] = {}
    scores = tmp_path / "scores.json"
    scores.write_text('{"samples": []}')

    def load_feature_archive(path):
        seen["route"] = "archive"
        return dict(dense), table, metadata

    def featurize_query_with_metadata(*args, **kwargs):
        seen["route"] = "raw"
        return dict(dense), metadata

    def fake_compile(config, representative_atoms, **kwargs):
        def compiled(key, batch, params):
            seen["model_features"] = batch
            return prediction

        return compiled

    def fake_write(prediction_value, writer_features, *args, **kwargs):
        seen["writer_features"] = writer_features
        seen["writer_metadata"] = kwargs.get("output_metadata")
        return {"structures": (), "scores": scores}

    data = SimpleNamespace(
        MODEL_FEATURES=tuple(dense),
        OPTIONAL_MODEL_FEATURES=(),
        PRIVATE_MODEL_FEATURES=(),
        COMPACT_MSA_PRIVATE_FEATURES=(),
        load_feature_archive=load_feature_archive,
        featurize_query_with_metadata=featurize_query_with_metadata,
        has_compact_zero_template_pair_features=lambda batch: False,
        has_compact_msa_features=lambda batch: False,
        subsample_msa_rows=lambda batch, depth: batch,
        collapse_identical_templates=lambda batch: batch,
        compact_zero_template_pair_features=lambda batch: batch,
        compact_msa_features=lambda batch: batch,
        compact_ref_atom_category_storage=compact_categories,
        has_atomized_tokens=lambda batch: False,
        normalize_asym_ids=normalize_asym_ids,
    )
    modules = {
        "foldjax.models.openfold3.data": data,
        "foldjax.models.openfold3.inference": SimpleNamespace(
            RELEASED_MSA_DEPTH=1024,
            released_config=lambda **kwargs: SimpleNamespace(msa_depth=1024),
            compile_predict=fake_compile,
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(
            write_prediction_outputs=fake_write
        ),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: table
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda path: {}
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda state, prefix=None: "",
            prune_sample_diffusion_aliases=lambda state, *, prefix: 0,
            map_inference_params=lambda state, prefix: object(),
        ),
        "foldjax.models.openfold3.compilation": SimpleNamespace(
            enable_compilation_cache=lambda path: path
        ),
        "jax": SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
    }
    monkeypatch.setattr(
        "foldjax.backends.openfold3.import_module", lambda name: modules[name]
    )
    input_path = tmp_path / f"input{suffix}"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    request = PredictionRequest(
        model="openfold3",
        input=input_path,
        input_format=input_format,
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=None,
    )

    OpenFold3Backend().predict(request)

    assert seen["route"] == expected_route
    model_features = seen["model_features"]
    writer_features = seen["writer_features"]
    assert "ref_element" not in model_features
    assert "ref_atom_name_chars" not in model_features
    assert COMPACT_REF_ELEMENT_IDS in model_features
    assert COMPACT_REF_ATOM_NAME_CHAR_IDS in model_features
    assert "ref_element" in writer_features
    assert "ref_atom_name_chars" in writer_features
    assert not (
        set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & writer_features.keys()
    )
    assert seen["writer_metadata"] is metadata


_CP_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout, shard_pair_rows
    from foldjax.models.openfold3.data import compact_ref_atom_category_storage
    from foldjax.models.openfold3.inference import _restore_ref_atom_category_one_hot
    from foldjax.models.openfold3.models.primitives import LinearParams, linear
    from tests.models.openfold3.feature_fixture import minimal_features

    assert jax.device_count() == 4, jax.devices()
    features = minimal_features(tokens=4, atoms=12)
    dense = {
        name: features[name]
        for name in ("atom_mask", "ref_element", "ref_atom_name_chars")
    }
    compact = compact_ref_atom_category_storage(dense)
    ew = jnp.arange(8 * 119, dtype=jnp.float32).reshape(8, 119) / 97
    cw = jnp.arange(8 * 256, dtype=jnp.float32).reshape(8, 256) / 89
    traced = []

    def build(source):
        def run():
            traced.append(cp_layout())
            restored = _restore_ref_atom_category_one_hot(source)
            element = linear(
                restored["ref_element"].astype(jnp.float32), LinearParams(ew)
            )
            chars = restored["ref_atom_name_chars"].reshape(1, 12, 256)
            single = element + linear(chars.astype(jnp.float32), LinearParams(cw))
            return shard_pair_rows(single[..., :, None, :] + single[..., None, :, :])
        return run

    reference = jax.device_get(jax.jit(build(dense))())
    assert np.isfinite(reference).all()
    for layout in ("1d", "2d"):
        jax.clear_caches()
        with context_parallel(4, layout=layout):
            dense_result = jax.device_get(jax.jit(build(dense))())
            compact_result = jax.device_get(jax.jit(build(compact))())
        # Context parallelism may choose a different floating-point reduction
        # schedule from serial. The optimization contract is dense-vs-compact
        # identity under the *same* topology, which remains byte-for-byte.
        assert compact_result.tobytes() == dense_result.tobytes(), layout

    assert traced == [None, "1d", "1d", "2d", "2d"], traced
    print("OPENFOLD3_COMPACT_ATOM_CATEGORIES_CP_OK")
    """
)


def test_compact_categories_match_dense_on_forced_four_cpu_devices() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_COMPACT_ATOM_CATEGORIES_CP_OK" in completed.stdout
