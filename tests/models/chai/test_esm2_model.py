"""Isolated numerical contracts for the native ESM2 encoder."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as torch_f

from foldjax.models.chai.data.esm import tokenize_esm_sequence
from foldjax.models.chai.models.esm2 import esm2_forward, esm2_layer, stack_esm2_layers


def _layer(seed: int, hidden: int = 8, ffn: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def normal(shape: tuple[int, ...], scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float16)

    return {
        "q_weight": normal((hidden, hidden)),
        "q_bias": normal((hidden,)),
        "k_weight": normal((hidden, hidden)),
        "k_bias": normal((hidden,)),
        "v_weight": normal((hidden, hidden)),
        "v_bias": normal((hidden,)),
        "out_weight": normal((hidden, hidden)),
        "out_bias": normal((hidden,)),
        "rotary_inv_freq": np.asarray([1.0, 0.01], np.float16),
        "attention_norm_weight": np.ones(hidden, np.float16),
        "attention_norm_bias": normal((hidden,), 0.01),
        "fc1_weight": normal((ffn, hidden)),
        "fc1_bias": normal((ffn,)),
        "fc2_weight": normal((hidden, ffn)),
        "fc2_bias": normal((hidden,)),
        "final_norm_weight": np.ones(hidden, np.float16),
        "final_norm_bias": normal((hidden,), 0.01),
    }


def _torch_layer(
    x: torch.Tensor, padding_mask: torch.Tensor, layer: dict[str, np.ndarray]
) -> torch.Tensor:
    t = {name: torch.from_numpy(value.copy()) for name, value in layer.items()}
    residual = x
    normalized = torch_f.layer_norm(
        x, (x.shape[-1],), t["attention_norm_weight"], t["attention_norm_bias"], 1e-5
    )
    q = torch_f.linear(normalized, t["q_weight"], t["q_bias"])
    k = torch_f.linear(normalized, t["k_weight"], t["k_bias"])
    v = torch_f.linear(normalized, t["v_weight"], t["v_bias"])
    head_dim = 2 * t["rotary_inv_freq"].numel()
    heads = x.shape[-1] // head_dim
    q = q.reshape(*q.shape[:-1], heads, head_dim).transpose(1, 2)
    k = k.reshape(*k.shape[:-1], heads, head_dim).transpose(1, 2)
    v = v.reshape(*v.shape[:-1], heads, head_dim).transpose(1, 2)
    positions = torch.arange(q.shape[-2], dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, t["rotary_inv_freq"].float())
    embedding = torch.cat([frequencies, frequencies], dim=-1)
    cosine = embedding.cos().half()[None, None]
    sine = embedding.sin().half()[None, None]

    def rotate(value: torch.Tensor) -> torch.Tensor:
        left, right = value.chunk(2, dim=-1)
        return torch.cat([-right, left], dim=-1)

    q = q * cosine + rotate(q) * sine
    k = k * cosine + rotate(k) * sine
    attended = torch_f.scaled_dot_product_attention(
        q, k, v, attn_mask=(~padding_mask)[:, None, None, :]
    )
    attended = attended.transpose(1, 2).reshape_as(x)
    x = residual + torch_f.linear(attended, t["out_weight"], t["out_bias"])
    residual = x
    normalized = torch_f.layer_norm(
        x, (x.shape[-1],), t["final_norm_weight"], t["final_norm_bias"], 1e-5
    )
    hidden = torch_f.linear(normalized, t["fc1_weight"], t["fc1_bias"])
    hidden = hidden * 0.5 * (1.0 + torch.erf(hidden / math.sqrt(2.0)))
    return residual + torch_f.linear(hidden, t["fc2_weight"], t["fc2_bias"])


def test_tokenizer_matches_chai_ids_and_rejects_unknown_residue() -> None:
    np.testing.assert_array_equal(
        tokenize_esm_sequence("LAGXBUZO"),
        [0, 4, 5, 6, 24, 25, 26, 27, 28, 2],
    )
    with np.testing.assert_raises_regex(ValueError, "unsupported ESM residue"):
        tokenize_esm_sequence("AJ")


def test_single_layer_matches_torch_sdpa_contract() -> None:
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, 5, 8)) * 0.2).astype(np.float16)
    mask = np.asarray([[False, False, False, False, True]])
    layer = _layer(11)

    expected = _torch_layer(torch.from_numpy(x), torch.from_numpy(mask), layer)
    actual = esm2_layer(
        jnp.asarray(x),
        jnp.asarray(mask),
        layer,
        attention_implementation="xla",
    )

    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=4e-3, atol=4e-3
    )


def test_full_scan_preserves_fp16_output_and_token_dropout_scaling() -> None:
    rng = np.random.default_rng(17)
    layers = (_layer(1), _layer(2))
    embed = (rng.standard_normal((33, 8)) * 0.1).astype(np.float16)
    state = {
        "embed_tokens_weight": embed,
        "final_norm_weight": np.ones(8, np.float16),
        "final_norm_bias": np.zeros(8, np.float16),
        "layers": stack_esm2_layers(layers),
    }
    tokens = np.asarray([[0, 5, 32, 2, 1]], np.int32)
    padding = torch.from_numpy(tokens == 1)
    expected = torch.from_numpy(embed[tokens]).clone()
    expected[tokens == 32] = 0
    expected = expected * np.float16(0.88) / np.float16(1.0 - 1.0 / 4.0)
    expected[tokens == 1] = 0
    for layer in layers:
        expected = _torch_layer(expected, padding, layer)
    expected = torch_f.layer_norm(expected, (8,), eps=1e-5)

    actual = esm2_forward(jnp.asarray(tokens), state, attention_implementation="xla")

    assert actual.dtype == jnp.float16
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=5e-3, atol=5e-3
    )
