from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai import inference
from foldjax.models.chai.models import pairformer, triangle_cueq


@pytest.mark.parametrize(
    "resolver, variable",
    [
        (pairformer._triangle_attention_backend, "CHAI_JAX_TRIANGLE_ATTENTION_BACKEND"),
        (
            pairformer._triangle_multiplication_backend,
            "CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND",
        ),
    ],
)
def test_both_triangle_backends_default_to_the_fused_kernel(
    monkeypatch, resolver, variable: str
) -> None:
    """No probe, no `auto`, and no device query.

    This resolver used to read `auto` and answer from `cueq_available()` plus a
    device scan, which meant a machine missing the wheel silently ran a
    different kernel under the same configuration -- two installs, two
    numerics, one name, and the only symptom a number in a benchmark. The
    kernel now either loads or raises with the variable to set.
    """
    monkeypatch.delenv(variable, raising=False)
    assert resolver() == "cueq"


@pytest.mark.parametrize(
    "resolver, variable",
    [
        (pairformer._triangle_attention_backend, "CHAI_JAX_TRIANGLE_ATTENTION_BACKEND"),
        (
            pairformer._triangle_multiplication_backend,
            "CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND",
        ),
    ],
)
def test_the_blocked_path_is_still_reachable_by_name(
    monkeypatch, resolver, variable: str
) -> None:
    """`cueq` is a default, not a lock -- a card that cannot fit it needs a way out."""
    monkeypatch.setenv(variable, "xla")
    assert resolver() == "xla"


def test_triangle_attention_backend_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "invalid")

    with pytest.raises(ValueError, match="must be 'xla' or 'cueq'"):
        pairformer._triangle_attention_backend()


def test_no_availability_probe_survives_anywhere(monkeypatch) -> None:
    """The probe is gone from the module, not merely unused by the resolver.

    Leaving `cueq_available` importable is how the fallback comes back: the
    next site that wants to be helpful calls it.
    """
    assert not hasattr(triangle_cueq, "cueq_available")


def test_high_size_xla_host_chunk_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")

    assert inference._triangle_attention_host_chunk_size(512) == 64
    assert inference._triangle_attention_host_chunk_size(32) == 32


def test_high_size_cueq_keeps_requested_host_chunk(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "cueq")

    assert inference._triangle_attention_host_chunk_size(512) == 512


def _pair_of(tokens: int, channels: int):
    """A stand-in for a pair tensor, sized but never allocated.

    Both gates read only `.shape` and `.dtype`, and the tensors in question run
    to gigabytes, so a description of one is enough and a test can afford it.
    """
    return jax.ShapeDtypeStruct((1, tokens, tokens, channels), jnp.float32)


def test_1536_xla_selects_fully_bounded_pair_paths(monkeypatch) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")

    assert inference._msa_pair_subchunk_size(1536) == 512
    assert inference._use_low_memory_pairformer(1536)


def test_1536_cueq_still_bounds_both_paths_on_a_real_tensor(monkeypatch) -> None:
    """A `cueq` card does not exempt a pair tensor from the byte rule.

    These two assertions used to read `None` and `False` against a bare token
    count, which is not what the trunk passes: it hands both gates the tensor,
    and the byte rule decides. Only `_use_low_memory_pairformer` had that rule,
    so on a cueq card -- the only kind big enough to reach 1,531 tokens -- the
    MSA pair block ran whole, on the branch that holds eleven pair-sized
    intermediates, and raised the peak by 19,399 MiB of 44,266. Written against
    the count, this test went on passing throughout.
    """
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "cueq")

    big = _pair_of(1536, 32)  # 288 MiB, over the budget
    assert inference._msa_pair_subchunk_size(big) == 512
    assert inference._use_low_memory_pairformer(big)

    # Below the budget the count rule still governs, and on cueq it declines.
    small = _pair_of(1536, 4)  # 36 MiB
    assert inference._msa_pair_subchunk_size(small) is None
    assert not inference._use_low_memory_pairformer(small)


def test_both_pair_gates_agree_on_every_probe(monkeypatch) -> None:
    """They are one decision, and were two that disagreed.

    Nothing asserted they matched, so the byte rule could be -- and was --
    added to one of them alone.
    """
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "cueq")

    probes = [
        1024,
        1536,
        2048,
        _pair_of(1024, 32),
        _pair_of(1536, 4),
        _pair_of(2048, 64),
    ]
    for probe in probes:
        chunked = inference._msa_pair_subchunk_size(probe) is not None
        assert chunked is inference._use_low_memory_pairformer(probe), probe


@pytest.mark.parametrize("direction", [0, 1])
def test_direction_chunk_matches_forced_xla_full_rows(
    monkeypatch, direction: int
) -> None:
    monkeypatch.setenv("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "xla")
    rng = np.random.default_rng(7)
    token_count = 6
    channels = 8
    num_heads = 2
    head_dim = 4
    params = pairformer.FusedTriangleAttentionParams(
        out_scalers=jnp.asarray(rng.normal(size=channels), jnp.float32),
        pair2b_weight=jnp.asarray(
            rng.normal(size=(2 * num_heads, channels)), jnp.float32
        ),
        pair2qkvg1_weight=jnp.asarray(
            rng.normal(size=(4 * num_heads * head_dim, channels)), jnp.float32
        ),
        pair2qkvg2_weight=jnp.asarray(
            rng.normal(size=(4 * num_heads * head_dim, channels)), jnp.float32
        ),
        linear_out_weight=jnp.asarray(
            rng.normal(size=(channels, 2 * num_heads * head_dim)), jnp.float32
        ),
    )
    pair = jnp.asarray(
        rng.normal(size=(1, token_count, token_count, channels)), jnp.float32
    )
    pair_mask = jnp.asarray([[[True, True, True, True, False, False]] * token_count])
    start, size = 2, 3
    full = pairformer.fused_triangle_attention_direction(
        pair,
        pair_mask,
        params,
        direction=direction,
        attention_backend="xla",
    )
    bias = pairformer.fused_triangle_attention_bias(pair, pair_mask, params)
    value = pair if direction == 0 else jnp.swapaxes(pair, 1, 2)
    value_chunk = jnp.take(value, jnp.arange(start, start + size), axis=-3)

    chunk = pairformer.fused_triangle_attention_direction_chunk(
        value_chunk,
        bias,
        params,
        direction=direction,
    )

    np.testing.assert_array_equal(
        np.asarray(chunk), np.asarray(full[:, start : start + size])
    )
