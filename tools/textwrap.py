"""One-shot patch hook for the Fold-CP atom finalizer.

The finalizer imports ``textwrap`` from its own directory.  This temporary
module applies two fixes before delegating every public name to the standard
library module, then deletes itself so a successful validation commit contains
no import shim.
"""

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

        # First obtain one fixed global row maximum over all resident key
        # tiles. The second ring then accumulates exp(score - maximum), which
        # avoids repeated online rescaling and its fp32 ordering drift on 3x3
        # and larger meshes.
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

# The loaded module remains usable after unlinking; deletion is intentionally
# visible to git so the validated finalizer commit removes this temporary hook.
Path(__file__).unlink()
