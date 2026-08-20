from __future__ import annotations

import re
from pathlib import Path


def _replace_regex(path: str, pattern: str, replacement: str) -> None:
    content = Path(path).read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{path}: precision patch target did not match once")
    Path(path).write_text(updated, encoding="utf-8")


def _replace_once(path: str, old: str, new: str) -> None:
    content = Path(path).read_text(encoding="utf-8")
    if content.count(old) != 1:
        raise RuntimeError(f"{path}: formatting target did not match once")
    Path(path).write_text(content.replace(old, new, 1), encoding="utf-8")


def _patch_two_pass_ring() -> None:
    path = "src/foldjax/models/_cp_attention.py"
    replacement = '''    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        bias_l = permute(bias_l, bias_init0)
        bias_l = permute(bias_l, diagonal_init)
        k_l = permute(k_l, diagonal_init)
        v_l = permute(v_l, diagonal_init)
        mask_l = permute(mask_l, diagonal_init)

        # Pass 1 fixes one global row maximum before any exponentials are
        # accumulated. Rotating K/bias/mask through a complete cycle returns
        # every tile to its initial owner, so pass 2 needs no saved full-width
        # operand and the per-device memory remains O(N^2 / P).
        maximum = jnp.full(
            q_l.shape[:-1] + (1,),
            -jnp.inf,
            dtype=jnp.float32,
        )
        for _ in range(side):
            scores = jnp.matmul(
                q_l.astype(jnp.float32),
                jnp.swapaxes(k_l.astype(jnp.float32), -1, -2),
                precision=precision,
            )
            scores = (
                scores
                + bias_l.astype(jnp.float32)
                + mask_l.astype(jnp.float32)
            )
            maximum = jnp.maximum(
                maximum,
                jnp.max(scores, axis=-1, keepdims=True),
            )
            # A full cycle restores the initial tile ownership. V is not used
            # in the max pass and therefore stays at its initial owner.
            k_l = permute(k_l, kv_hop)
            mask_l = permute(mask_l, kv_hop)
            bias_l = permute(bias_l, bias_hop)

        normalizer = jnp.zeros_like(maximum)
        normalizer_correction = jnp.zeros_like(maximum)
        output = jnp.zeros(q_l.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)

        # Pass 2 evaluates every tile against the same maximum and combines
        # tile-local numerator/denominator terms with Neumaier compensation.
        # This removes the repeated online-rescaling error that becomes visible
        # after a 3x3 Pairformer stack while preserving the gather-free ring.
        for step in range(side):
            scores = jnp.matmul(
                q_l.astype(jnp.float32),
                jnp.swapaxes(k_l.astype(jnp.float32), -1, -2),
                precision=precision,
            )
            scores = (
                scores
                + bias_l.astype(jnp.float32)
                + mask_l.astype(jnp.float32)
            )
            finite_maximum = jnp.isfinite(maximum)
            positive_infinity = jnp.isposinf(maximum)
            shifted = jnp.where(
                finite_maximum,
                scores - maximum,
                -jnp.inf,
            )
            probabilities = jnp.where(
                positive_infinity,
                jnp.isposinf(scores).astype(jnp.float32),
                jnp.exp(shifted),
            )
            block_normalizer = jnp.sum(
                probabilities,
                axis=-1,
                keepdims=True,
            )
            block_output = jnp.matmul(
                probabilities,
                v_l.astype(jnp.float32),
                precision=precision,
            )
            output, output_correction = _compensated_add(
                output,
                output_correction,
                block_output,
            )
            normalizer, normalizer_correction = _compensated_add(
                normalizer,
                normalizer_correction,
                block_normalizer,
            )

            if step + 1 < side:
                k_l = permute(k_l, kv_hop)
                v_l = permute(v_l, kv_hop)
                mask_l = permute(mask_l, kv_hop)
                bias_l = permute(bias_l, bias_hop)

        output = output + output_correction
        normalizer = normalizer + normalizer_correction
        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        result = jnp.where(
            normalizer > 0,
            output / jnp.maximum(normalizer, tiny),
            jnp.zeros_like(output),
        )
        return result.astype(v_l.dtype)

    out = jax.shard_map('''
    _replace_regex(
        path,
        r"    def local_ring\(q_l, k_l, v_l, bias_l, mask_l\):.*?\n    out = jax\.shard_map\(",
        replacement,
    )


def _patch_generated_test_formatting() -> None:
    path = "tests/models/test_cp_runtime.py"
    old = '''        assert atom_placed["atom_to_token"].sharding.shard_shape(atom_map.shape) == atom_map.shape
'''
    new = '''        assert (
            atom_placed["atom_to_token"].sharding.shard_shape(atom_map.shape)
            == atom_map.shape
        )
'''
    _replace_once(path, old, new)


def main() -> None:
    _patch_two_pass_ring()
    _patch_generated_test_formatting()
    print("two-pass Fold-CP ring precision patch applied")


if __name__ == "__main__":
    main()
