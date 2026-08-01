from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.data.templates import (
    TemplateContext,
    load_native_template_context,
    save_native_template_context,
)


def _context(templates: int, tokens: int, offset: int = 0) -> TemplateContext:
    restype = np.arange(templates * tokens, dtype=np.int32).reshape(templates, tokens)
    restype = (restype + offset) % 31
    pseudo = restype % 2 == 0
    backbone = restype % 3 == 0
    distances = np.arange(templates * tokens * tokens, dtype=np.float32).reshape(
        templates, tokens, tokens
    )
    vectors = np.repeat(distances[..., None], 3, axis=-1)
    return TemplateContext(restype, pseudo, backbone, distances, vectors)


def test_template_empty_matches_official_padding_class() -> None:
    context = TemplateContext.empty(n_templates=4, n_tokens=3)
    assert context.template_restype.shape == (4, 3)
    assert np.all(context.template_restype == 31)
    assert context.num_nonnull_templates == 0
    assert not context.template_pseudo_beta_mask.any()


def test_template_pad_and_index_select_are_exact() -> None:
    source = _context(2, 3)
    selected = source.index_select(np.asarray([2, 0], np.int64))
    np.testing.assert_array_equal(
        selected.template_restype, source.template_restype[:, [2, 0]]
    )
    np.testing.assert_array_equal(
        selected.template_distances, source.template_distances[:, [2, 0]][:, :, [2, 0]]
    )

    padded = source.pad(max_templates=4, max_tokens=5)
    assert padded.template_restype.shape == (4, 5)
    np.testing.assert_array_equal(
        padded.template_restype[:2, :3], source.template_restype
    )
    assert np.all(padded.template_restype[2:] == 31)
    assert np.all(padded.template_restype[:, 3:] == 31)


def test_template_merge_matches_official_chain_block_layout() -> None:
    left = _context(1, 2)
    right = _context(2, 1, offset=7)
    merged = TemplateContext.merge([left, right])
    assert merged.template_restype.shape == (2, 3)
    np.testing.assert_array_equal(
        merged.template_restype[:, :2], left.pad(2).template_restype
    )
    np.testing.assert_array_equal(
        merged.template_restype[:, 2:], right.template_restype
    )
    np.testing.assert_array_equal(
        merged.template_distances[:, :2, :2], left.pad(2).template_distances
    )
    np.testing.assert_array_equal(
        merged.template_distances[:, 2:, 2:], right.template_distances
    )
    assert not merged.template_distances[:, :2, 2:].any()
    assert not merged.template_unit_vector[:, 2:, :2].any()


def test_native_template_roundtrip_to_batched_inputs(tmp_path: Path) -> None:
    source = _context(2, 3)
    path = tmp_path / "templates.npz"
    save_native_template_context(
        source,
        path,
        source_id="local.m8+cif",
        source_sha256="e" * 64,
        query_identity={"entity_names": ["A"], "sequence_sha256": ["f" * 64]},
    )
    loaded = load_native_template_context(path)
    np.testing.assert_array_equal(
        loaded.context.template_restype, source.template_restype
    )
    inputs = loaded.context.pad(4, 5).to_model_inputs(batched=True)
    assert inputs["template_restype"].shape == (1, 4, 5)
    assert inputs["template_unit_vector"].shape == (1, 4, 5, 5, 3)
    with pytest.raises(ValueError, match="query identity"):
        load_native_template_context(
            path, expected_query_identity={"entity_names": ["B"]}
        )


def test_native_template_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "templates.npz"
    save_native_template_context(
        _context(1, 2),
        path,
        source_id="fixture",
        source_sha256="a" * 64,
        query_identity={"entity_names": ["A"]},
    )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["template_distances"][0, 0, 0] += 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_native_template_context(path)


def test_native_template_archive_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    kwargs = {
        "source_id": "fixture",
        "source_sha256": "a" * 64,
        "query_identity": {"entity_names": ["A"]},
    }
    save_native_template_context(_context(1, 2), first, **kwargs)
    save_native_template_context(_context(1, 2), second, **kwargs)
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
