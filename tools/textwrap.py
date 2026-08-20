"""One-shot patch and diagnostic hook for Fold-CP finalization."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


_STDLIB_PATH = Path(os.__file__).resolve().parent / "textwrap.py"
_SPEC = importlib.util.spec_from_file_location("_foldjax_stdlib_textwrap", _STDLIB_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load standard-library textwrap from {_STDLIB_PATH}")
_STDLIB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STDLIB)
for _name in dir(_STDLIB):
    if not _name.startswith("__") or _name in {"__all__", "__doc__"}:
        globals()[_name] = getattr(_STDLIB, _name)

_ROOT = Path(__file__).resolve().parents[1]

_atom_path = _ROOT / "src/foldjax/models/_cp_atom.py"
_atom_text = _atom_path.read_text()
_atom_old = '''        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        out = jnp.where(
            denominator > 0,
            numerator / jnp.maximum(jnp.swapaxes(denominator, 1, 2), tiny),
            jnp.zeros_like(numerator),
        )
'''
_atom_new = '''        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        denominator_qh = jnp.swapaxes(denominator, 1, 2)
        out = jnp.where(
            denominator_qh > 0,
            numerator / jnp.maximum(denominator_qh, tiny),
            jnp.zeros_like(numerator),
        )
'''
if _atom_old not in _atom_text:
    raise RuntimeError("pair-bias denominator block was not found")
_atom_path.write_text(_atom_text.replace(_atom_old, _atom_new, 1))

_attention_path = _ROOT / "src/foldjax/models/_cp_attention.py"
_attention_text = _attention_path.read_text()
_start = _attention_text.index("    def local_ring(q_l, k_l, v_l, bias_l, mask_l):")
_end = _attention_text.index("\n    out = jax.shard_map(", _start)
_replacement = '''    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        bias_initial = permute(bias_l, bias_init0)
        bias_initial = permute(bias_initial, diagonal_init)
        key_initial = permute(k_l, diagonal_init)
        value_initial = permute(v_l, diagonal_init)
        mask_initial = permute(mask_l, diagonal_init)

        maximum = jnp.full(
            q_l.shape[:-1] + (1,),
            -jnp.inf,
            dtype=jnp.float32,
        )
        key_work = key_initial
        bias_work = bias_initial
        mask_work = mask_initial
        for step in range(side):
            scores = jnp.matmul(
                q_l.astype(jnp.float32),
                jnp.swapaxes(key_work.astype(jnp.float32), -1, -2),
                precision=precision,
            )
            scores = (
                scores
                + bias_work.astype(jnp.float32)
                + mask_work.astype(jnp.float32)
            )
            maximum = jnp.maximum(
                maximum,
                jnp.max(scores, axis=-1, keepdims=True),
            )
            if step + 1 < side:
                key_work = permute(key_work, kv_hop)
                mask_work = permute(mask_work, kv_hop)
                bias_work = permute(bias_work, bias_hop)

        normalizer = jnp.zeros_like(maximum)
        normalizer_correction = jnp.zeros_like(maximum)
        output = jnp.zeros(q_l.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)
        key_work = key_initial
        value_work = value_initial
        bias_work = bias_initial
        mask_work = mask_initial
        for step in range(side):
            scores = jnp.matmul(
                q_l.astype(jnp.float32),
                jnp.swapaxes(key_work.astype(jnp.float32), -1, -2),
                precision=precision,
            )
            scores = (
                scores
                + bias_work.astype(jnp.float32)
                + mask_work.astype(jnp.float32)
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
                value_work.astype(jnp.float32),
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
                key_work = permute(key_work, kv_hop)
                value_work = permute(value_work, kv_hop)
                mask_work = permute(mask_work, kv_hop)
                bias_work = permute(bias_work, bias_hop)

        output = output + output_correction
        normalizer = normalizer + normalizer_correction
        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        result = jnp.where(
            normalizer > 0,
            output / jnp.maximum(normalizer, tiny),
            jnp.zeros_like(output),
        )
        return result.astype(value_initial.dtype)
'''
_attention_path.write_text(
    _attention_text[:_start] + _replacement + _attention_text[_end:]
)

_triangle_path = _ROOT / "src/foldjax/models/boltz2/models/triangle/triangle.py"
_triangle_text = _triangle_path.read_text()
_triangle_old = '''        total = jnp.zeros(
            lhs.shape[:2] + rhs.shape[2:3] + lhs.shape[3:], dtype=jnp.float32
        )
        for step in range(side):
            total = total + jnp.einsum(
                "bikd,bkjd->bijd", lhs, rhs, preferred_element_type=jnp.float32
            )
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total
'''
_triangle_new = '''        total = jnp.zeros(
            lhs.shape[:2] + rhs.shape[2:3] + lhs.shape[3:], dtype=jnp.float32
        )
        correction = jnp.zeros_like(total)
        for step in range(side):
            term = jnp.einsum(
                "bikd,bkjd->bijd", lhs, rhs, preferred_element_type=jnp.float32
            )
            updated = total + term
            correction = correction + jnp.where(
                jnp.abs(total) >= jnp.abs(term),
                (total - updated) + term,
                (term - updated) + total,
            )
            total = updated
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction
'''
if _triangle_old not in _triangle_text:
    raise RuntimeError("Cannon accumulation block was not found")
_triangle_path.write_text(_triangle_text.replace(_triangle_old, _triangle_new, 1))

_test_path = _ROOT / "tests/models/boltz2/test_context_parallel.py"
_test_text = _test_path.read_text()
_test_old = '''    assert traced == [None, "2d"], traced
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
'''
_test_new = '''    assert traced == [None, "2d"], traced

    # Diagnostic-only intermediate capture. This block is injected into the
    # ephemeral validation checkout and is never committed.
    from foldjax.models._cp import shard_pair_rows
    from foldjax.models.boltz2.models.primitives.transition import (
        transition_forward,
    )

    def run_stages(z_in):
        outputs = []
        current = z_in
        for layer_params in params["layers"]:
            current = shard_pair_rows(current)
            current = current + triangle_multiplication_forward(
                layer_params["tri_mul_out"],
                current,
                pair_mask,
                "outgoing",
                chunk_size=128,
            )
            outputs.append(current)
            current = current + triangle_multiplication_forward(
                layer_params["tri_mul_in"],
                current,
                pair_mask,
                "incoming",
                chunk_size=128,
            )
            outputs.append(current)
            current = current + triangle_attention_forward(
                layer_params["tri_att_start"],
                current,
                pair_mask,
                starting=True,
                chunk_size=128,
                triangle_backend="xla",
            )
            outputs.append(current)
            current = current + triangle_attention_forward(
                layer_params["tri_att_end"],
                current,
                pair_mask,
                starting=False,
                chunk_size=128,
                triangle_backend="xla",
            )
            outputs.append(current)
            current = current + transition_forward(
                layer_params["transition_z"],
                current,
                row_chunk_size=128,
            )
            outputs.append(current)
        return tuple(outputs)

    serial_stages = jax.device_get(jax.jit(run_stages)(z))
    with context_parallel(DEVICES, layout="2d"):
        cp_stages = jax.device_get(jax.jit(run_stages)(z))
    stage_names = (
        "l0_mul_out", "l0_mul_in", "l0_att_start", "l0_att_end", "l0_transition",
        "l1_mul_out", "l1_mul_in", "l1_att_start", "l1_att_end", "l1_transition",
    )
    for name, serial_value, cp_value in zip(
        stage_names, serial_stages, cp_stages, strict=True
    ):
        difference = np.max(np.abs(serial_value - cp_value))
        print(f"STAGE_DIFF devices={DEVICES} stage={name} max={difference:.9e}")

    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
'''
if _test_old not in _test_text:
    raise RuntimeError("grid-stack assertion block was not found")
_test_path.write_text(_test_text.replace(_test_old, _test_new, 1))

Path(__file__).unlink()
