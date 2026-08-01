from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.data.esm import (
    ESM_DIMENSION,
    EsmEmbeddingContext,
    assemble_esm_context,
    generate_native_esm_embeddings,
    load_native_esm_embeddings,
    save_native_esm_embeddings,
)
from foldjax.models.chai.data.input import EntityType, Input


def _embedding(length: int, offset: float = 0.0) -> np.ndarray:
    values = np.arange(length * ESM_DIMENSION, dtype=np.float32)
    return values.reshape(length, ESM_DIMENSION) + offset


def test_native_esm_archive_roundtrip_and_padding(tmp_path: Path) -> None:
    path = tmp_path / "esm.npz"
    embeddings = {"AC": _embedding(2), "G": _embedding(1, 10.0)}
    save_native_esm_embeddings(
        embeddings,
        path,
        model_id="esm2_t36_3B_UR50D",
        model_revision="chai-traced-sdpa-fp16",
        source_sha256="a" * 64,
    )

    loaded = load_native_esm_embeddings(path)
    assert loaded.model_id == "esm2_t36_3B_UR50D"
    assert loaded.model_revision == "chai-traced-sdpa-fp16"
    np.testing.assert_array_equal(loaded.embeddings["AC"], embeddings["AC"])

    context = EsmEmbeddingContext(loaded.embeddings["AC"])
    padded = context.pad(4)
    assert padded.esm_embeddings.shape == (4, ESM_DIMENSION)
    np.testing.assert_array_equal(padded.esm_embeddings[:2], embeddings["AC"])
    assert not padded.esm_embeddings[2:].any()
    with pytest.raises(ValueError, match="model revision"):
        load_native_esm_embeddings(path, expected_model_revision="wrong")


def test_native_esm_archive_rejects_wrong_sequence_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sequence length"):
        save_native_esm_embeddings(
            {"AC": _embedding(1)},
            tmp_path / "bad.npz",
            model_id="model",
            model_revision="revision",
            source_sha256="b" * 64,
        )


def test_native_esm_archive_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "esm.npz"
    save_native_esm_embeddings(
        {"AC": _embedding(2)},
        path,
        model_id="model",
        model_revision="revision",
        source_sha256="c" * 64,
    )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["sequence_000000"][0, 0] += 1
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_native_esm_embeddings(path)


def test_assemble_esm_context_matches_upstream_blowout() -> None:
    protein = _embedding(3)
    inputs = (
        Input("ACD", EntityType.PROTEIN.value, "p"),
        Input("AC", EntityType.RNA.value, "r"),
        Input("ACD", EntityType.PROTEIN.value, "p-copy"),
    )
    token_asym_id = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3], np.int32)
    token_residue_index = np.asarray([0, 1, 2, 0, 1, 0, 1, 1, 2], np.int32)

    actual = assemble_esm_context(
        inputs,
        token_asym_id=token_asym_id,
        token_residue_index=token_residue_index,
        embeddings={"ACD": protein},
    ).esm_embeddings

    expected = np.concatenate(
        (protein, np.zeros((2, ESM_DIMENSION), np.float32), protein[[0, 1, 1, 2]])
    )
    np.testing.assert_array_equal(actual, expected)


def test_assemble_esm_context_rejects_missing_or_extra_embeddings() -> None:
    inputs = (Input("AC", EntityType.PROTEIN.value, "p"),)
    asym = np.asarray([1, 1], np.int32)
    residue = np.asarray([0, 1], np.int32)
    with pytest.raises(ValueError, match="missing ESM embedding"):
        assemble_esm_context(
            inputs,
            token_asym_id=asym,
            token_residue_index=residue,
            embeddings={},
        )
    with pytest.raises(ValueError, match="unused ESM embedding"):
        assemble_esm_context(
            inputs,
            token_asym_id=asym,
            token_residue_index=residue,
            embeddings={"AC": _embedding(2), "GG": _embedding(2)},
        )


def test_archive_digest_is_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    kwargs = {
        "model_id": "model",
        "model_revision": "revision",
        "source_sha256": "d" * 64,
    }
    save_native_esm_embeddings({"AC": _embedding(2)}, first, **kwargs)
    save_native_esm_embeddings({"AC": _embedding(2)}, second, **kwargs)
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_on_the_fly_embeddings_are_cached_by_model_and_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_sha = "a" * 64
    monkeypatch.setattr(
        "foldjax.models.chai.bridge.esm2_io.read_native_esm2_manifest",
        lambda _path: {"source_sha256": model_sha},
    )
    load_calls: list[Path] = []

    def load(path: Path) -> object:
        load_calls.append(path)
        return SimpleNamespace(
            manifest={"source_sha256": model_sha},
            global_state={},
            layers=({},),
        )

    monkeypatch.setattr("foldjax.models.chai.bridge.esm2_io.load_native_esm2", load)
    monkeypatch.setattr(
        "foldjax.models.chai.models.esm2.stack_esm2_layers", lambda _layers: {}
    )

    def forward(tokens: jnp.ndarray, _state: object, **_kwargs: object) -> jnp.ndarray:
        return jnp.broadcast_to(tokens[..., None], tokens.shape + (2560,)).astype(
            jnp.float16
        )

    monkeypatch.setattr("foldjax.models.chai.models.esm2.esm2_forward", forward)
    kwargs = {
        "model_path": tmp_path / "model",
        "cache_dir": tmp_path / "cache",
    }
    first = generate_native_esm_embeddings(["AC", "AC"], **kwargs)
    second = generate_native_esm_embeddings(["AC"], **kwargs)

    assert load_calls == [tmp_path / "model"]
    np.testing.assert_array_equal(first["AC"], second["AC"])
    assert first["AC"].shape == (2, 2560)
    assert first["AC"].dtype == np.float32
