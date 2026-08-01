"""Torch-free Chai ESM embedding contexts and native archives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.chai.data.asset_cache import AssetCache
from foldjax.models.chai.data.input import (
    EntityType,
    Input,
    constituents_of_modified_fasta,
)

ESM_DIMENSION = 2560
ESM_MODEL_ID = "esm2_t36_3B_UR50D"
_FORMAT = "chai-jax-esm-embeddings"
_VERSION = 1
_MANIFEST = "__chai_jax_esm_manifest__"

ESM_TOKEN_MAP = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
    ".": 29,
    "-": 30,
    "<null_1>": 31,
    "<mask>": 32,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_metadata(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": _sha256(memoryview(contiguous)),
    }


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("ESM sequence must contain non-empty ASCII letters")
    return normalized


def _input_esm_sequence(value: Input) -> str:
    constituents = constituents_of_modified_fasta(value.sequence)
    if constituents is None:
        raise ValueError(f"invalid modified protein sequence: {value.entity_name}")
    return _normalize_sequence(
        "".join(item if len(item) == 1 else "X" for item in constituents)
    )


def tokenize_esm_sequence(sequence: str) -> np.ndarray:
    """Tokenize exactly as Chai's ``DumbTokenizer(<cls>SEQ<eos>)``."""

    normalized = _normalize_sequence(sequence)
    try:
        residue_tokens = [ESM_TOKEN_MAP[residue] for residue in normalized]
    except KeyError as error:
        raise ValueError(f"unsupported ESM residue: {error.args[0]!r}") from error
    return np.asarray(
        [ESM_TOKEN_MAP["<cls>"], *residue_tokens, ESM_TOKEN_MAP["<eos>"]],
        dtype=np.int32,
    )


def _embedding_cache_path(cache_dir: Path, model_sha256: str, sequence: str) -> Path:
    sequence_sha256 = _sha256(sequence.encode())
    return cache_dir / model_sha256 / f"{sequence_sha256}.npy"


def _load_cached_embedding(
    path: Path, sequence: str, model_sha256: str
) -> np.ndarray | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"incomplete ESM embedding cache entry: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = np.load(path, allow_pickle=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid ESM embedding cache entry: {path}") from error
    expected = {
        "model_sha256": model_sha256,
        "sequence": sequence,
        "sequence_sha256": _sha256(sequence.encode()),
        "embedding_sha256": _sha256(memoryview(np.ascontiguousarray(value))),
    }
    if metadata != expected:
        raise ValueError(f"ESM embedding cache provenance mismatch: {path}")
    return _validate_embedding(sequence, value)


def _save_cached_embedding(
    path: Path, sequence: str, model_sha256: str, embedding: np.ndarray
) -> None:
    value = _validate_embedding(sequence, embedding)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    metadata = {
        "model_sha256": model_sha256,
        "sequence": sequence,
        "sequence_sha256": _sha256(sequence.encode()),
        "embedding_sha256": _sha256(memoryview(value)),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary_array = Path(output.name)
        np.save(output, value, allow_pickle=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{metadata_path.name}.",
        encoding="utf-8",
        delete=False,
    ) as output:
        temporary_metadata = Path(output.name)
        json.dump(metadata, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    try:
        os.replace(temporary_array, path)
        os.replace(temporary_metadata, metadata_path)
    except BaseException:
        temporary_array.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise


def generate_native_esm_embeddings(
    sequences: Sequence[str],
    *,
    model_path: str | Path,
    cache_dir: str | Path,
    attention_implementation: str = "xla",
) -> dict[str, np.ndarray]:
    """Generate and content-cache ESM embeddings without importing Torch."""

    from foldjax.models.chai.bridge.esm2_io import (
        load_native_esm2,
        read_native_esm2_manifest,
    )
    from foldjax.models.chai.models.esm2 import esm2_forward, stack_esm2_layers

    normalized = tuple(dict.fromkeys(_normalize_sequence(item) for item in sequences))
    if not normalized:
        return {}
    manifest = read_native_esm2_manifest(model_path)
    model_sha256 = manifest["source_sha256"]
    cache_root = Path(cache_dir)
    result: dict[str, np.ndarray] = {}
    missing = []
    for sequence in normalized:
        cached = _load_cached_embedding(
            _embedding_cache_path(cache_root, model_sha256, sequence),
            sequence,
            model_sha256,
        )
        if cached is None:
            missing.append(sequence)
        else:
            result[sequence] = cached
    if not missing:
        return result

    bundle = load_native_esm2(model_path)
    state = {
        **bundle.global_state,
        "layers": stack_esm2_layers(bundle.layers),
    }
    device_state = jax.device_put(jax.tree.map(jnp.asarray, state))

    @jax.jit
    def forward(tokens: jax.Array, params: Mapping[str, Any]) -> jax.Array:
        return esm2_forward(
            tokens,
            params,
            attention_implementation=attention_implementation,
        )

    for sequence in missing:
        tokens = jnp.asarray(tokenize_esm_sequence(sequence))[None]
        full_output = np.asarray(jax.device_get(forward(tokens, device_state)))
        output = full_output[0, 1:-1].astype(np.float32)
        output = _validate_embedding(sequence, output)
        _save_cached_embedding(
            _embedding_cache_path(cache_root, model_sha256, sequence),
            sequence,
            model_sha256,
            output,
        )
        result[sequence] = output
    return result


def generate_native_esm_embeddings_for_inputs(
    inputs: Sequence[Input],
    *,
    model_path: str | Path,
    cache_dir: str | Path,
    attention_implementation: str = "xla",
) -> dict[str, np.ndarray]:
    """Generate each unique normalized protein sequence needed by inference."""

    sequences = [
        _input_esm_sequence(value)
        for value in inputs
        if value.entity_type == EntityType.PROTEIN.value
    ]
    return generate_native_esm_embeddings(
        sequences,
        model_path=model_path,
        cache_dir=cache_dir,
        attention_implementation=attention_implementation,
    )


def _validate_embedding(sequence: str, embedding: np.ndarray) -> np.ndarray:
    value = np.asarray(embedding)
    expected = (len(sequence), ESM_DIMENSION)
    if value.shape != expected:
        raise ValueError(
            f"ESM embedding sequence length/dimension mismatch for {sequence!r}: "
            f"expected {expected}, got {value.shape}"
        )
    if value.dtype != np.float32:
        raise ValueError(f"ESM embedding must be float32 for {sequence!r}")
    if not np.isfinite(value).all():
        raise ValueError(f"ESM embedding contains non-finite values for {sequence!r}")
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class EsmEmbeddingContext:
    esm_embeddings: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.esm_embeddings)
        if value.ndim != 2 or value.shape[1] != ESM_DIMENSION:
            raise ValueError(f"ESM context must have shape (tokens, {ESM_DIMENSION})")
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("ESM context must contain finite float32 values")

    @property
    def num_tokens(self) -> int:
        return self.esm_embeddings.shape[0]

    def pad(self, max_tokens: int) -> EsmEmbeddingContext:
        if self.num_tokens > max_tokens:
            raise ValueError(f"cannot pad {self.num_tokens} ESM tokens to {max_tokens}")
        result = np.zeros((max_tokens, ESM_DIMENSION), np.float32)
        result[: self.num_tokens] = self.esm_embeddings
        return EsmEmbeddingContext(result)

    @classmethod
    def empty(cls, n_tokens: int) -> EsmEmbeddingContext:
        if n_tokens < 0:
            raise ValueError("ESM token count must be non-negative")
        return cls(np.zeros((n_tokens, ESM_DIMENSION), np.float32))


@dataclass(frozen=True)
class NativeEsmEmbeddings:
    embeddings: Mapping[str, np.ndarray]
    model_id: str
    model_revision: str
    source_sha256: str


def save_native_esm_embeddings(
    embeddings: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    source_sha256: str,
) -> None:
    """Atomically write per-sequence ESM outputs with model provenance."""
    if not embeddings:
        raise ValueError("at least one ESM embedding is required")
    if not model_id or not model_revision:
        raise ValueError("ESM model_id and model_revision are required")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

    normalized: dict[str, np.ndarray] = {}
    for raw_sequence, raw_embedding in embeddings.items():
        sequence = _normalize_sequence(raw_sequence)
        if sequence in normalized:
            raise ValueError(f"duplicate normalized ESM sequence: {sequence!r}")
        normalized[sequence] = _validate_embedding(sequence, raw_embedding)

    arrays: dict[str, np.ndarray] = {}
    sequences: list[dict[str, Any]] = []
    for index, sequence in enumerate(sorted(normalized)):
        name = f"sequence_{index:06d}"
        value = normalized[sequence]
        arrays[name] = value
        sequences.append(
            {
                "sequence": sequence,
                "sequence_sha256": _sha256(sequence.encode()),
                "array": name,
                "metadata": _array_metadata(value),
            }
        )
    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "source_sha256": source_sha256,
        "dimension": ESM_DIMENSION,
        "sequences": sequences,
    }
    manifest_array = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        np.uint8,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        np.savez(output, **{_MANIFEST: manifest_array, **arrays})
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_native_esm_embeddings(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
) -> NativeEsmEmbeddings:
    """Load a checksummed ESM archive without importing PyTorch."""
    with np.load(path, allow_pickle=False) as archive:
        if _MANIFEST not in archive.files:
            raise ValueError("native ESM manifest is missing")
        try:
            manifest = json.loads(archive[_MANIFEST].tobytes().decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native ESM manifest is invalid") from error
        if manifest.get("format") != _FORMAT or manifest.get("version") != _VERSION:
            raise ValueError("unsupported native ESM archive format")
        if manifest.get("dimension") != ESM_DIMENSION:
            raise ValueError("native ESM dimension mismatch")
        if (
            expected_model_id is not None
            and manifest.get("model_id") != expected_model_id
        ):
            raise ValueError("native ESM model ID mismatch")
        if (
            expected_model_revision is not None
            and manifest.get("model_revision") != expected_model_revision
        ):
            raise ValueError("native ESM model revision mismatch")
        records = manifest.get("sequences")
        if not isinstance(records, list) or not records:
            raise ValueError("native ESM sequence manifest is invalid")
        expected_names = {record.get("array") for record in records}
        if set(archive.files) - {_MANIFEST} != expected_names:
            raise ValueError("native ESM array names mismatch")
        embeddings: dict[str, np.ndarray] = {}
        for record in records:
            sequence = _normalize_sequence(record.get("sequence", ""))
            if record.get("sequence_sha256") != _sha256(sequence.encode()):
                raise ValueError(f"native ESM sequence hash mismatch: {sequence!r}")
            value = np.array(archive[record["array"]], copy=True)
            if _array_metadata(value) != record.get("metadata"):
                raise ValueError(f"native ESM array metadata mismatch: {sequence!r}")
            embeddings[sequence] = _validate_embedding(sequence, value)
    return NativeEsmEmbeddings(
        embeddings=embeddings,
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        source_sha256=manifest["source_sha256"],
    )


def resolve_native_esm_embeddings(
    source: str | Path,
    *,
    cache_dir: str | Path,
    expected_sha256: str | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
) -> NativeEsmEmbeddings:
    """Resolve a local/HTTPS archive through the shared integrity cache."""
    asset = AssetCache(cache_dir).resolve(source, expected_sha256=expected_sha256)
    return load_native_esm_embeddings(
        asset.path,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
    )


def assemble_esm_context(
    inputs: Sequence[Input],
    *,
    token_asym_id: np.ndarray,
    token_residue_index: np.ndarray,
    embeddings: Mapping[str, np.ndarray],
) -> EsmEmbeddingContext:
    """Reproduce Chai's per-chain residue-index ESM blow-out exactly."""
    asym_id = np.asarray(token_asym_id)
    residue_index = np.asarray(token_residue_index)
    if asym_id.ndim != 1 or residue_index.shape != asym_id.shape:
        raise ValueError("token asym and residue indices must be matching vectors")
    if len(inputs) == 0 or set(np.unique(asym_id)) != set(range(1, len(inputs) + 1)):
        raise ValueError("token asym IDs must map exactly to the ordered input chains")

    normalized_embeddings = {
        _normalize_sequence(sequence): np.asarray(value)
        for sequence, value in embeddings.items()
    }
    required: set[str] = set()
    output = np.zeros((asym_id.size, ESM_DIMENSION), np.float32)
    for chain_index, input_value in enumerate(inputs, start=1):
        chain_mask = asym_id == chain_index
        indices = residue_index[chain_mask]
        if np.any(indices < 0):
            raise ValueError(
                f"negative token residue index for {input_value.entity_name}"
            )
        if input_value.entity_type != EntityType.PROTEIN.value:
            continue
        sequence = _input_esm_sequence(input_value)
        required.add(sequence)
        if sequence not in normalized_embeddings:
            raise ValueError(f"missing ESM embedding for protein sequence {sequence!r}")
        embedding = _validate_embedding(sequence, normalized_embeddings[sequence])
        if indices.size and int(indices.max()) >= len(sequence):
            raise ValueError(
                "token residue index exceeds ESM sequence for "
                f"{input_value.entity_name}"
            )
        output[chain_mask] = embedding[indices]
    unused = set(normalized_embeddings) - required
    if unused:
        raise ValueError(f"unused ESM embedding sequences: {sorted(unused)!r}")
    return EsmEmbeddingContext(output)


__all__ = [
    "ESM_DIMENSION",
    "ESM_MODEL_ID",
    "ESM_TOKEN_MAP",
    "EsmEmbeddingContext",
    "NativeEsmEmbeddings",
    "assemble_esm_context",
    "generate_native_esm_embeddings",
    "generate_native_esm_embeddings_for_inputs",
    "load_native_esm_embeddings",
    "resolve_native_esm_embeddings",
    "save_native_esm_embeddings",
    "tokenize_esm_sequence",
]
