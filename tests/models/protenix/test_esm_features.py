from __future__ import annotations

import builtins
import pickle

import numpy as np
import pytest

from foldjax import torch_archive
from foldjax.models.protenix.data.esm import (
    Esm2Config,
    EsmCheckpointError,
    JaxEsmProvider,
    add_esm_embeddings,
    convert_esm2_checkpoint,
    encode_esm2_sequence,
    esm2_forward,
    load_esm2_checkpoint,
    validate_esm_embeddings,
)

_TOY_CONFIG = Esm2Config(
    num_layers=1,
    embed_dim=8,
    attention_heads=2,
    token_dropout=True,
)


def _toy_esm2_state(*, prefixed: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260814)
    state: dict[str, np.ndarray] = {}

    def add(name: str, shape: tuple[int, ...], scale: float = 0.08) -> None:
        key = f"encoder.sentence_encoder.{name}" if prefixed else name
        state[key] = rng.normal(0.0, scale, shape).astype(np.float32)

    add("embed_tokens.weight", (33, 8), 0.2)
    prefix = "layers.0."
    for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
        add(f"{prefix}self_attn.{projection}.weight", (8, 8))
        add(f"{prefix}self_attn.{projection}.bias", (8,), 0.03)
    for layer_norm in ("self_attn_layer_norm", "final_layer_norm"):
        add(f"{prefix}{layer_norm}.weight", (8,), 0.04)
        key = (
            f"encoder.sentence_encoder.{prefix}{layer_norm}.weight"
            if prefixed
            else f"{prefix}{layer_norm}.weight"
        )
        state[key] += 1.0
        add(f"{prefix}{layer_norm}.bias", (8,), 0.02)
    add(f"{prefix}fc1.weight", (32, 8))
    add(f"{prefix}fc1.bias", (32,), 0.03)
    add(f"{prefix}fc2.weight", (8, 32))
    add(f"{prefix}fc2.bias", (8,), 0.03)
    add("emb_layer_norm_after.weight", (8,), 0.04)
    final_weight_key = (
        "encoder.sentence_encoder.emb_layer_norm_after.weight"
        if prefixed
        else "emb_layer_norm_after.weight"
    )
    state[final_weight_key] += 1.0
    add("emb_layer_norm_after.bias", (8,), 0.02)
    return state


def _save_toy_checkpoint(path, *, prefixed: bool = False) -> None:
    np.savez(path, **_toy_esm2_state(prefixed=prefixed))


def test_add_esm_embeddings_maps_entities_and_reuses_duplicate_sequences() -> None:
    job = {
        "sequences": [
            {"proteinChain": {"sequence": "AG", "count": 2}},
            {"ligand": {"ligand": "CCD_ATP", "count": 1}},
        ]
    }
    features = {
        "token_entity_id": np.array([1, 1, 1, 1, 2]),
        "residue_index": np.array([0, 1, 0, 1, 0]),
    }
    calls: list[str] = []

    def provider(sequence: str) -> np.ndarray:
        calls.append(sequence)
        return np.arange(6, dtype=np.float32).reshape(2, 3)

    result = add_esm_embeddings(features, job, provider=provider, embedding_dim=3)

    assert calls == ["AG"]
    np.testing.assert_array_equal(
        result["esm_token_embedding"],
        np.array(
            [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 0, 0]],
            dtype=np.float32,
        ),
    )


def test_add_esm_embeddings_rejects_wrong_provider_shape() -> None:
    features = {
        "token_entity_id": np.array([1, 1]),
        "residue_index": np.array([0, 1]),
    }
    job = {"sequences": [{"proteinChain": {"sequence": "AG"}}]}

    with pytest.raises(ValueError, match="embedding shape"):
        add_esm_embeddings(
            features,
            job,
            provider=lambda _: np.zeros((1, 3), dtype=np.float32),
            embedding_dim=3,
        )


def test_add_esm_embeddings_requires_entity_metadata() -> None:
    with pytest.raises(KeyError, match="token_entity_id"):
        add_esm_embeddings(
            {"residue_index": np.array([0])},
            {"sequences": []},
            provider=lambda _: np.zeros((0, 3), dtype=np.float32),
            embedding_dim=3,
        )


def test_validate_precomputed_esm_embeddings() -> None:
    features = {
        "restype": np.zeros((2, 32), dtype=np.int32),
        "esm_token_embedding": np.zeros((2, 3), dtype=np.float32),
    }
    assert validate_esm_embeddings(features, embedding_dim=3).shape == (2, 3)

    with pytest.raises(ValueError, match=r"expected \(2, 3\)"):
        validate_esm_embeddings(
            {**features, "esm_token_embedding": np.zeros((1, 3))},
            embedding_dim=3,
        )
    with pytest.raises(ValueError, match="non-finite"):
        validate_esm_embeddings(
            {**features, "esm_token_embedding": np.full((2, 3), np.nan)},
            embedding_dim=3,
        )


def test_esm2_alphabet_matches_fair_esm_1b() -> None:
    np.testing.assert_array_equal(
        encode_esm2_sequence("aGX?-"),
        np.array([0, 5, 6, 24, 3, 30, 2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="exceeds 2"):
        encode_esm2_sequence("AGX", max_sequence_length=2)


def test_esm2_toy_checkpoint_matches_fair_esm_reference(tmp_path) -> None:
    checkpoint = tmp_path / "toy.npz"
    _save_toy_checkpoint(checkpoint, prefixed=True)
    weights = load_esm2_checkpoint(checkpoint, config=_TOY_CONFIG)

    representation = np.asarray(
        esm2_forward(weights, encode_esm2_sequence("AG"), compile_layers=False)
    )

    # Generated once with fair-esm 2.0.0's ESM2 class from the exact arrays
    # returned by ``_toy_esm2_state``.  The fixture remains NumPy-only so a
    # normal FoldJAX test environment never imports PyTorch or fair-esm.
    expected = np.array(
        [
            [
                [
                    -0.283438,
                    1.1004266,
                    0.3962064,
                    0.24234603,
                    0.6044054,
                    -1.6240009,
                    -1.3788356,
                    1.03545,
                ],
                [
                    1.0587238,
                    1.0876057,
                    -0.30003408,
                    0.06972499,
                    -0.8093254,
                    -0.32689148,
                    1.002367,
                    -1.8603284,
                ],
                [
                    0.98405576,
                    1.687232,
                    -0.48819423,
                    0.1095291,
                    -0.40256554,
                    0.4201688,
                    -0.6374727,
                    -1.7318075,
                ],
                [
                    0.01817593,
                    1.9136549,
                    -1.9474292,
                    0.2587622,
                    -0.26637423,
                    0.45693967,
                    0.048149806,
                    -0.5935238,
                ],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(representation, expected, rtol=2e-5, atol=2e-5)


def test_esm2_safetensors_conversion_round_trips(tmp_path) -> None:
    source = tmp_path / "toy.npz"
    destination = tmp_path / "toy.safetensors"
    _save_toy_checkpoint(source, prefixed=True)

    assert (
        convert_esm2_checkpoint(source, destination, config=_TOY_CONFIG)
        == destination
    )
    original = load_esm2_checkpoint(source, config=_TOY_CONFIG).state_dict()
    converted = load_esm2_checkpoint(destination, config=_TOY_CONFIG).state_dict()
    assert original.keys() == converted.keys()
    for name in original:
        np.testing.assert_array_equal(converted[name], original[name])


def test_jax_esm_provider_runs_with_torch_import_blocked(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "esm2_t36_3B_UR50D.npz"
    _save_toy_checkpoint(checkpoint)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"JAX ESM provider imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provider = JaxEsmProvider(
        "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
    )

    first = provider("AG")
    second = provider("AG")

    assert first.shape == (2, 8)
    np.testing.assert_array_equal(second, first)


def test_jax_esm_provider_masks_a_reusable_language_model_width(tmp_path) -> None:
    checkpoint = tmp_path / "esm2_t36_3B_UR50D.npz"
    _save_toy_checkpoint(checkpoint)
    provider = JaxEsmProvider(
        "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
    )

    exact = provider("AG")
    padded = provider.embed("AG", target_length=8)

    assert padded.shape == exact.shape == (2, 8)
    np.testing.assert_allclose(padded, exact, rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError, match="smaller than sequence"):
        provider.embed("AG", target_length=1)


def test_jax_esm_provider_prefers_safe_checkpoints_before_official_pt(
    tmp_path,
) -> None:
    stem = tmp_path / "esm2_t36_3B_UR50D"
    official = stem.with_suffix(".pt")
    npz = stem.with_suffix(".npz")
    safetensors = stem.with_suffix(".safetensors")
    source = tmp_path / "source.npz"
    official.write_bytes(b"publisher archive placeholder")
    _save_toy_checkpoint(npz)
    _save_toy_checkpoint(source)
    convert_esm2_checkpoint(source, safetensors, config=_TOY_CONFIG)

    assert (
        JaxEsmProvider(
            "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
        ).checkpoint
        == safetensors
    )
    safetensors.unlink()
    assert (
        JaxEsmProvider(
            "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
        ).checkpoint
        == npz
    )
    npz.unlink()
    assert (
        JaxEsmProvider(
            "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
        ).checkpoint
        == official
    )


@pytest.mark.parametrize("suffix", [".pt", ".npz"])
def test_jax_esm_provider_normalises_checkpoint_failures(tmp_path, suffix) -> None:
    checkpoint = (tmp_path / "esm2_t36_3B_UR50D").with_suffix(suffix)
    if suffix == ".pt":
        checkpoint.write_bytes(b"not a torch zip archive")
    else:
        np.savez(checkpoint, unexpected=np.ones((1,), dtype=np.float32))
    provider = JaxEsmProvider(
        "esm2-3b", checkpoint_dir=tmp_path, config=_TOY_CONFIG
    )

    with pytest.raises(
        EsmCheckpointError,
        match=r"could not load ESM-2 checkpoint.*publisher-supplied.*safetensors/\.npz",
    ):
        provider("AG")


def test_load_esm2_checkpoint_normalises_unpickle_failures(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "esm2.pt"
    checkpoint.write_bytes(b"placeholder")

    def fail_unpickle(_path):
        raise pickle.UnpicklingError("unsupported persistent object")

    monkeypatch.setattr(torch_archive, "load", fail_unpickle)

    with pytest.raises(
        EsmCheckpointError,
        match=r"unsupported persistent object.*publisher-supplied",
    ):
        load_esm2_checkpoint(checkpoint, config=_TOY_CONFIG)
