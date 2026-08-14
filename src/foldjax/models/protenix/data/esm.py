"""Torch-free ESM-2 preprocessing for Protenix ESM and ISM variants.

The released Protenix ``mini_esm`` and ``mini_ism`` models consume the final
representation of the fair-esm ESM-2 3B encoder.  This module implements that
encoder directly in JAX.  Official ``torch.save`` checkpoints are decoded by
FoldJAX's restricted archive reader into NumPy arrays; PyTorch and fair-esm are
neither imported nor required.

Only the representation path is materialised.  The language-model and contact
heads in the checkpoint do not contribute to ``esm_token_embedding`` and are
therefore deliberately ignored.
"""

from __future__ import annotations

import pickle
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

ESM_MODELS = {
    "esm2-3b": "esm2_t36_3B_UR50D.pt",
    "esm2-3b-ism": "esm2_t36_3B_UR50D_ism.pt",
}

# fair-esm's ``Alphabet.from_architecture("ESM-1b")``.  Keeping the exact
# published order is important: these integers index ``embed_tokens.weight``.
_STANDARD_TOKENS = (
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
)
_ALPHABET = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    *_STANDARD_TOKENS,
    "<null_1>",
    "<mask>",
)
_TOKEN_TO_INDEX = {token: index for index, token in enumerate(_ALPHABET)}
_CLS_INDEX = _TOKEN_TO_INDEX["<cls>"]
_PADDING_INDEX = _TOKEN_TO_INDEX["<pad>"]
_EOS_INDEX = _TOKEN_TO_INDEX["<eos>"]
_UNKNOWN_INDEX = _TOKEN_TO_INDEX["<unk>"]
_MASK_INDEX = _TOKEN_TO_INDEX["<mask>"]


@dataclass(frozen=True)
class Esm2Config:
    """Architecture constants for one ESM-2 encoder checkpoint."""

    num_layers: int = 36
    embed_dim: int = 2560
    attention_heads: int = 40
    token_dropout: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("ESM-2 num_layers must be positive")
        if self.embed_dim < 1:
            raise ValueError("ESM-2 embed_dim must be positive")
        if self.attention_heads < 1 or self.embed_dim % self.attention_heads:
            raise ValueError("ESM-2 embed_dim must be divisible by attention_heads")


ESM2_3B_CONFIG = Esm2Config()


class EsmCheckpointError(RuntimeError):
    """An ESM-2 checkpoint exists but cannot be decoded or validated."""


@dataclass(frozen=True)
class Esm2Weights:
    """Validated NumPy-backed representation weights for an ESM-2 encoder."""

    config: Esm2Config
    embed_tokens: np.ndarray
    layers: tuple[Mapping[str, np.ndarray], ...]
    final_layer_norm_weight: np.ndarray
    final_layer_norm_bias: np.ndarray

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return the normalised flat representation state dictionary."""

        result = {"embed_tokens.weight": self.embed_tokens}
        for layer_index, layer in enumerate(self.layers):
            prefix = f"layers.{layer_index}."
            result.update({prefix + name: value for name, value in layer.items()})
        result["emb_layer_norm_after.weight"] = self.final_layer_norm_weight
        result["emb_layer_norm_after.bias"] = self.final_layer_norm_bias
        return result


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Residue embedding provider accepted by :func:`add_esm_embeddings`."""

    def __call__(self, sequence: str) -> np.ndarray:
        """Return a ``[len(sequence), embedding_dim]`` float array."""


def encode_esm2_sequence(
    sequence: str, *, max_sequence_length: int = 4094
) -> np.ndarray:
    """Encode one protein with the exact fair-esm ESM-1b alphabet.

    The returned token vector includes the leading ``<cls>`` and trailing
    ``<eos>`` tokens used by ESM-2.  Unsupported residue symbols map to
    ``<unk>`` in the same way as fair-esm's alphabet.
    """

    if len(sequence) > max_sequence_length:
        raise ValueError(
            f"ESM sequence length {len(sequence)} exceeds {max_sequence_length}"
        )
    residue_tokens = [
        _TOKEN_TO_INDEX.get(residue, _UNKNOWN_INDEX) for residue in sequence.upper()
    ]
    return np.asarray([_CLS_INDEX, *residue_tokens, _EOS_INDEX], dtype=np.int32)


def _linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    # The published 3B archive stores reduced-precision tensors, but fair-esm
    # constructs an FP32 module before ``load_state_dict`` and PyTorch casts
    # those tensors into the module parameters.  Cast lazily per layer to match
    # that runtime without duplicating the complete checkpoint on the host.
    weight = jnp.asarray(weight, dtype=x.dtype)
    bias = jnp.asarray(bias, dtype=x.dtype)
    return (
        jnp.einsum(
            "...i,oi->...o",
            x,
            weight,
            precision=jax.lax.Precision.HIGHEST,
        )
        + bias
    )


def _layer_norm(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    eps: float,
) -> jax.Array:
    # PyTorch's LayerNorm computes these reductions in the input dtype.  ESM-2
    # checkpoints and activations are FP32, so no hidden mixed-precision policy
    # is introduced here.
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    weight = jnp.asarray(weight, dtype=x.dtype)
    bias = jnp.asarray(bias, dtype=x.dtype)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * weight + bias


def _rotate_half(x: jax.Array) -> jax.Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def _apply_rotary(x: jax.Array) -> jax.Array:
    head_dim = x.shape[-1]
    positions = jnp.arange(x.shape[-2], dtype=jnp.float32)
    inv_frequency = 1.0 / (
        10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    frequencies = jnp.einsum("i,j->ij", positions, inv_frequency)
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)
    cosine = jnp.cos(angles)[None, None, :, :].astype(x.dtype)
    sine = jnp.sin(angles)[None, None, :, :].astype(x.dtype)
    return x * cosine + _rotate_half(x) * sine


def _esm2_layer(
    x: jax.Array,
    padding_mask: jax.Array,
    params: Mapping[str, jax.Array],
    *,
    attention_heads: int,
    layer_norm_eps: float,
) -> jax.Array:
    """One pre-norm fair-esm TransformerLayer, in batch-major layout."""

    residual = x
    x = _layer_norm(
        x,
        params["self_attn_layer_norm.weight"],
        params["self_attn_layer_norm.bias"],
        layer_norm_eps,
    )
    batch_size, sequence_length, embed_dim = x.shape
    head_dim = embed_dim // attention_heads

    query = _linear(
        x, params["self_attn.q_proj.weight"], params["self_attn.q_proj.bias"]
    )
    key = _linear(
        x, params["self_attn.k_proj.weight"], params["self_attn.k_proj.bias"]
    )
    value = _linear(
        x, params["self_attn.v_proj.weight"], params["self_attn.v_proj.bias"]
    )
    query = query * (head_dim**-0.5)
    query = query.reshape(batch_size, sequence_length, attention_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, attention_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, attention_heads, head_dim)
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))

    query = _apply_rotary(query)
    key = _apply_rotary(key)
    scores = jnp.einsum(
        "bhtd,bhsd->bhts",
        query,
        key,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = jnp.where(padding_mask[:, None, None, :], -jnp.inf, scores)
    # fair-esm explicitly performs softmax in FP32.
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
        scores.dtype
    )
    attention = jnp.einsum(
        "bhts,bhsd->bhtd",
        probabilities,
        value,
        precision=jax.lax.Precision.HIGHEST,
    )
    attention = jnp.transpose(attention, (0, 2, 1, 3)).reshape(
        batch_size, sequence_length, embed_dim
    )
    attention = _linear(
        attention,
        params["self_attn.out_proj.weight"],
        params["self_attn.out_proj.bias"],
    )
    x = residual + attention

    residual = x
    x = _layer_norm(
        x,
        params["final_layer_norm.weight"],
        params["final_layer_norm.bias"],
        layer_norm_eps,
    )
    x = _linear(x, params["fc1.weight"], params["fc1.bias"])
    x = x * 0.5 * (1.0 + jax.lax.erf(x / np.sqrt(2.0)))
    x = _linear(x, params["fc2.weight"], params["fc2.bias"])
    return residual + x


_compiled_esm2_layer = jax.jit(
    _esm2_layer,
    static_argnames=("attention_heads", "layer_norm_eps"),
)


def esm2_forward(
    weights: Esm2Weights,
    tokens: np.ndarray | jax.Array,
    *,
    compile_layers: bool = True,
) -> jax.Array:
    """Run the exact ESM-2 representation path and return ``[B, T, C]``.

    Layers are compiled separately and share one executable because all 36
    have the same shapes.  Besides bounding compiler size, this lets NumPy
    checkpoint leaves stream to the accelerator a layer at a time instead of
    pinning the complete 3B model in device memory.
    """

    config = weights.config
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    if tokens.ndim != 2:
        raise ValueError(f"ESM-2 tokens must have rank 1 or 2, got {tokens.shape}")
    if tokens.shape[-1] < 2:
        raise ValueError("ESM-2 tokens must include <cls> and <eos>")

    padding_mask = tokens == _PADDING_INDEX
    x = jnp.asarray(weights.embed_tokens, dtype=jnp.float32)[tokens]
    if config.token_dropout:
        masked = tokens == _MASK_INDEX
        x = jnp.where(masked[..., None], 0.0, x)
        source_lengths = jnp.sum(~padding_mask, axis=-1)
        observed_ratio = jnp.sum(masked, axis=-1).astype(x.dtype) / source_lengths
        x = x * ((1.0 - 0.15 * 0.8) / (1.0 - observed_ratio))[:, None, None]
    x = jnp.where(padding_mask[..., None], 0.0, x)

    layer_function = _compiled_esm2_layer if compile_layers else _esm2_layer
    for layer in weights.layers:
        x = layer_function(
            x,
            padding_mask,
            layer,
            attention_heads=config.attention_heads,
            layer_norm_eps=config.layer_norm_eps,
        )
        # JAX dispatch is asynchronous.  Without a barrier the Python loop can
        # enqueue all 36 host-to-device parameter transfers before the first
        # layers finish, retaining every FP32 staging buffer and defeating the
        # layer-streaming memory contract above.  Blocking here bounds host and
        # device residency to one layer while preserving the shared executable.
        x.block_until_ready()
    return _layer_norm(
        x,
        jnp.asarray(weights.final_layer_norm_weight),
        jnp.asarray(weights.final_layer_norm_bias),
        config.layer_norm_eps,
    )


_LAYER_PARAMETER_SHAPES = {
    "self_attn.k_proj.weight": ("embed", "embed"),
    "self_attn.k_proj.bias": ("embed",),
    "self_attn.v_proj.weight": ("embed", "embed"),
    "self_attn.v_proj.bias": ("embed",),
    "self_attn.q_proj.weight": ("embed", "embed"),
    "self_attn.q_proj.bias": ("embed",),
    "self_attn.out_proj.weight": ("embed", "embed"),
    "self_attn.out_proj.bias": ("embed",),
    "self_attn_layer_norm.weight": ("embed",),
    "self_attn_layer_norm.bias": ("embed",),
    "fc1.weight": ("ffn", "embed"),
    "fc1.bias": ("ffn",),
    "fc2.weight": ("embed", "ffn"),
    "fc2.bias": ("embed",),
    "final_layer_norm.weight": ("embed",),
    "final_layer_norm.bias": ("embed",),
}


def _normalise_state_key(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "encoder.sentence_encoder.",
        "encoder.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors import SafetensorError
        from safetensors.numpy import load_file

        try:
            payload: Any = load_file(str(path))
        except SafetensorError as error:
            raise ValueError("invalid safetensors container") from error
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.array(archive[key]) for key in archive.files}
    else:
        from foldjax import torch_archive

        payload = torch_archive.load(path)

    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        payload = payload["model"]
    elif isinstance(payload, Mapping) and isinstance(
        payload.get("state_dict"), Mapping
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"ESM-2 checkpoint is not a parameter mapping: {type(payload)!r}"
        )
    return payload


def load_esm2_checkpoint(
    path: str | Path,
    *,
    config: Esm2Config = ESM2_3B_CONFIG,
) -> Esm2Weights:
    """Load and validate fair-esm ``.pt``, safetensors, or NPZ weights.

    ``.pt`` must be the zip-based ``torch.save`` format used by the official
    ESM-2 releases.  It is decoded by :mod:`foldjax.torch_archive`, which only
    reconstructs known tensor storage primitives and never imports torch.
    """

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing ESM-2 checkpoint: {checkpoint}")
    try:
        return _load_esm2_checkpoint(checkpoint, config=config)
    except (
        EOFError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        pickle.PickleError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        detail = " ".join(str(error).split()) or type(error).__name__
        if len(detail) > 240:
            detail = detail[:237] + "..."
        raise EsmCheckpointError(
            f"could not load ESM-2 checkpoint {checkpoint}: {detail}. "
            "Replace it with the publisher-supplied zip-based .pt file, or "
            "place a valid converted .safetensors/.npz file with the same stem "
            "beside it."
        ) from error


def _load_esm2_checkpoint(
    checkpoint: Path,
    *,
    config: Esm2Config,
) -> Esm2Weights:
    """Decode an existing checkpoint; the public wrapper normalises failures."""

    state: dict[str, np.ndarray] = {}
    origins: dict[str, str] = {}
    for original_key, value in _checkpoint_state(checkpoint).items():
        key = _normalise_state_key(str(original_key))
        if key in state:
            raise ValueError(
                "ESM-2 keys collide after prefix normalisation: "
                f"{origins[key]!r} and {original_key!r}"
            )
        state[key] = np.asarray(value)
        origins[key] = str(original_key)

    embed_shape = (len(_ALPHABET), config.embed_dim)
    embed_tokens = _required_array(state, "embed_tokens.weight", embed_shape)
    layers: list[Mapping[str, np.ndarray]] = []
    dimensions = {"embed": config.embed_dim, "ffn": 4 * config.embed_dim}
    for layer_index in range(config.num_layers):
        prefix = f"layers.{layer_index}."
        layer: dict[str, np.ndarray] = {}
        for name, symbolic_shape in _LAYER_PARAMETER_SHAPES.items():
            shape = tuple(dimensions[dimension] for dimension in symbolic_shape)
            layer[name] = _required_array(state, prefix + name, shape)
        layers.append(layer)

    final_weight = _required_array(
        state, "emb_layer_norm_after.weight", (config.embed_dim,)
    )
    final_bias = _required_array(
        state, "emb_layer_norm_after.bias", (config.embed_dim,)
    )
    return Esm2Weights(
        config=config,
        embed_tokens=embed_tokens,
        layers=tuple(layers),
        final_layer_norm_weight=final_weight,
        final_layer_norm_bias=final_bias,
    )


def _required_array(
    state: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        value = state[name]
    except KeyError as exc:
        raise KeyError(f"ESM-2 checkpoint is missing {name!r}") from exc
    if value.shape != shape:
        raise ValueError(
            f"ESM-2 checkpoint tensor {name!r} has shape {value.shape}, "
            f"expected {shape}"
        )
    if value.dtype.kind not in "fc":
        raise TypeError(f"ESM-2 checkpoint tensor {name!r} is not floating point")
    return value


def convert_esm2_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    config: Esm2Config = ESM2_3B_CONFIG,
) -> Path:
    """Convert an official archive to representation-only safetensors."""

    destination = Path(destination)
    if destination.suffix != ".safetensors":
        raise ValueError("ESM-2 conversion destination must end in .safetensors")
    weights = load_esm2_checkpoint(source, config=config)
    from safetensors.numpy import save_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.ascontiguousarray(value)
        for name, value in weights.state_dict().items()
    }
    save_file(
        arrays,
        str(destination),
        metadata={
            "format": "foldjax-esm2-representation-v1",
            "num_layers": str(config.num_layers),
            "embed_dim": str(config.embed_dim),
            "attention_heads": str(config.attention_heads),
            "token_dropout": str(config.token_dropout).lower(),
        },
    )
    return destination


class JaxEsmProvider:
    """Lazily produce Protenix embeddings with the native JAX ESM-2 port."""

    def __init__(
        self,
        model_name: str,
        *,
        checkpoint_dir: str | Path,
        max_sequence_length: int = 4094,
        config: Esm2Config = ESM2_3B_CONFIG,
    ) -> None:
        if model_name not in ESM_MODELS:
            raise ValueError(f"unsupported ESM model: {model_name}")
        checkpoint_dir = Path(checkpoint_dir)
        official_path = checkpoint_dir / ESM_MODELS[model_name]
        stem = official_path.with_suffix("")
        candidates = (
            stem.with_suffix(".safetensors"),
            stem.with_suffix(".npz"),
            official_path,
        )
        self.model_name = model_name
        self.checkpoint = next(
            (path for path in candidates if path.is_file()), official_path
        )
        self.max_sequence_length = max_sequence_length
        self.config = config
        self._weights: Esm2Weights | None = None

    def _load(self) -> Esm2Weights:
        if self._weights is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(
                    f"missing {self.model_name} checkpoint: {self.checkpoint}; "
                    "place the publisher-supplied .pt file (or a converted "
                    ".safetensors/.npz file with the same stem) in the checkpoint "
                    "directory"
                )
            self._weights = load_esm2_checkpoint(self.checkpoint, config=self.config)
        return self._weights

    def __call__(self, sequence: str) -> np.ndarray:
        return self.embed(sequence)

    def embed(
        self,
        sequence: str,
        *,
        target_length: int | None = None,
    ) -> np.ndarray:
        """Embed a sequence, optionally on a masked reusable residue width."""

        if target_length is not None:
            if target_length < len(sequence):
                raise ValueError(
                    f"ESM padding target {target_length} is smaller than sequence "
                    f"length {len(sequence)}"
                )
            if target_length > self.max_sequence_length:
                raise ValueError(
                    f"ESM padding target {target_length} exceeds "
                    f"{self.max_sequence_length}"
                )
        tokens = encode_esm2_sequence(
            sequence, max_sequence_length=self.max_sequence_length
        )
        if target_length is not None and target_length > len(sequence):
            tokens = np.pad(
                tokens,
                (0, target_length - len(sequence)),
                constant_values=_PADDING_INDEX,
            )
        representation = esm2_forward(self._load(), tokens)
        embedding = representation[0, 1 : 1 + len(sequence)]
        return np.asarray(embedding, dtype=np.float32)


# Compatibility for code that constructed the old provider by name.  It now
# resolves to the JAX implementation and imports neither fair-esm nor torch.
FairEsmProvider = JaxEsmProvider


def add_esm_embeddings(
    features: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    provider: EmbeddingProvider | Callable[[str], np.ndarray],
    embedding_dim: int = 2560,
) -> dict[str, Any]:
    """Attach residue-level ESM values to protein tokens in static features."""

    if "token_entity_id" not in features:
        raise KeyError("ESM features require token_entity_id")
    if "residue_index" not in features:
        raise KeyError("ESM features require residue_index")
    entity_ids = np.asarray(features["token_entity_id"])
    residue_indices = np.asarray(features["residue_index"])
    if entity_ids.shape != residue_indices.shape or entity_ids.ndim != 1:
        raise ValueError("ESM token entity and residue metadata must be rank-1")

    result = dict(features)
    token_embedding = np.zeros((len(entity_ids), embedding_dim), dtype=np.float32)
    sequence_cache: dict[str, np.ndarray] = {}
    # Keep distinct sequences as distinct calls. A token-budget batch multiplies
    # ESM-2's quadratic attention workspace by the batch size, defeating the
    # layer-streaming memory bound above. Padding also changes the compiled
    # matmul shape and can change reduction order, so it would weaken the public
    # per-sequence numerical contract. Exact duplicate sequences are still
    # computed once through this cache.
    for entity_number, wrapper in enumerate(job.get("sequences", []), start=1):
        protein = wrapper.get("proteinChain") if isinstance(wrapper, Mapping) else None
        if not isinstance(protein, Mapping):
            continue
        sequence = str(protein.get("sequence", "")).upper()
        if not sequence:
            raise ValueError(f"protein entity {entity_number} has an empty sequence")
        embedding = sequence_cache.get(sequence)
        if embedding is None:
            embedding = np.asarray(provider(sequence), dtype=np.float32)
            expected = (len(sequence), embedding_dim)
            if embedding.shape != expected:
                raise ValueError(
                    f"ESM embedding shape for entity {entity_number} is "
                    f"{embedding.shape}, expected {expected}"
                )
            sequence_cache[sequence] = embedding
        mask = entity_ids == entity_number
        positions = residue_indices[mask]
        if np.any(positions < 0) or np.any(positions >= len(sequence)):
            raise ValueError(f"invalid ESM residue index for entity {entity_number}")
        token_embedding[mask] = embedding[positions]
    result["esm_token_embedding"] = token_embedding
    return result


def validate_esm_embeddings(
    features: Mapping[str, Any], *, embedding_dim: int = 2560
) -> np.ndarray:
    """Validate a precomputed ``esm_token_embedding`` feature array.

    This is the static-NPZ entry point: callers can ship embeddings once and
    run an ESM/ISM Protenix model later without retaining the 3B checkpoint.
    """

    if "esm_token_embedding" not in features:
        raise ValueError("ESM/ISM model features require esm_token_embedding")
    embedding = np.asarray(features["esm_token_embedding"])
    if "restype" in features:
        n_token = int(np.asarray(features["restype"]).shape[-2])
    elif "token_entity_id" in features:
        n_token = int(np.asarray(features["token_entity_id"]).shape[0])
    else:
        raise ValueError(
            "cannot validate esm_token_embedding without restype or token_entity_id"
        )
    expected = (n_token, embedding_dim)
    if embedding.shape != expected:
        raise ValueError(
            f"esm_token_embedding has shape {embedding.shape}, expected {expected}"
        )
    if embedding.dtype.kind not in "fc":
        raise ValueError("esm_token_embedding must be floating point")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("esm_token_embedding contains non-finite values")
    return embedding


__all__ = [
    "ESM2_3B_CONFIG",
    "ESM_MODELS",
    "EmbeddingProvider",
    "EsmCheckpointError",
    "Esm2Config",
    "Esm2Weights",
    "FairEsmProvider",
    "JaxEsmProvider",
    "add_esm_embeddings",
    "convert_esm2_checkpoint",
    "encode_esm2_sequence",
    "esm2_forward",
    "load_esm2_checkpoint",
    "validate_esm_embeddings",
]
