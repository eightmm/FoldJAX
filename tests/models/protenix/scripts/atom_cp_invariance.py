"""What must stay true of the serial program while a mesh exists nearby.

Four properties, each with a way to get them wrong that this file exists to
catch:

1. A one-device request must be the serial program. ``context_parallel(1)``
   yields ``None`` and never sets the ContextVar, so the identity stays
   ``("serial", 1, (1, 1), ())`` and the lowering is the no-context lowering
   byte for byte -- not merely close, and with no collectives in it.
2. Serial -> context-parallel -> serial in one interpreter must return to the
   first program. A module-level cache keyed on the callable rather than on
   the topology would give the third run the sharded executable.
3. An exception raised inside the context must still restore the ContextVar,
   or every later request in the process runs under a mesh nobody asked for.
4. ``context_parallel(1)`` *inside* an active mesh raises. The nesting guard at
   ``_cp.py:280`` runs before the one-device shortcut at ``:283``, so a nominal
   one-device context does not quietly inherit the outer mesh -- it is
   refused. Pinned here because the alternative reading (inherit silently) is
   the plausible one to assume, and it is wrong.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import (
    context_parallel,
    cp_identity,
    cp_mesh,
    cp_runtime,
)
from foldjax.models.protenix.models.diffusion._cp import resolve_atom_windows
from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_decoder,
    atom_attention_encoder,
    atom_attention_encoder_prepare_diffusion_cache,
)
from tests.models.protenix.atom_cp_fixtures import (
    N_ATOM,
    N_HEADS,
    N_KEYS,
    N_QUERIES,
    N_TOKEN,
    build_case,
)

COLLECTIVE_OPS = (
    "collective-permute",
    "all-gather",
    "all-reduce",
    "reduce-scatter",
    "all-to-all",
)


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == devices, jax.devices()
    case = build_case()

    def build(cp_atom_windows: bool):
        def run(r_l, s_trunk, z_pair):
            p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_element,
                case.ref_atom_name_chars,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                z_pair,
                case.encoder,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                cp_atom_windows=cp_atom_windows,
            )
            a, q_skip, c_skip, p_skip = atom_attention_encoder(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_atom_name_chars,
                case.ref_element,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                case.encoder,
                r_l=r_l,
                s=s_trunk,
                z=z_pair,
                p_lm=p_lm,
                c_l=c_l,
                n_token=N_TOKEN,
                n_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                attention_backend="xla",
                atom_mask=case.atom_mask,
                cp_atom_windows=cp_atom_windows,
            )
            return atom_attention_decoder(
                case.atom_to_token_idx,
                a,
                q_skip,
                c_skip,
                p_skip,
                case.decoder,
                n_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                attention_backend="xla",
                atom_mask=case.atom_mask,
                cp_atom_windows=cp_atom_windows,
            )

        return run

    args = (case.r_l, case.s_trunk, case.z_pair)

    def serial_run(cp_atom_windows: bool):
        compiled = jax.jit(build(cp_atom_windows))
        value = np.asarray(jax.device_get(compiled(*args)))
        text = compiled.lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
        return value, _fingerprint(text), text.lower()

    # (1) The option's default and an explicit one-device request are the same
    # program, and the option itself changes nothing without a mesh.
    plain_value, plain_hash, plain_text = serial_run(True)
    assert cp_identity() == ("serial", 1, (1, 1), ()), cp_identity()
    jax.clear_caches()
    with context_parallel(1) as mesh:
        assert mesh is None, mesh
        assert cp_mesh() is None, cp_mesh()
        assert cp_runtime() is None, cp_runtime()
        assert cp_identity() == ("serial", 1, (1, 1), ()), cp_identity()
        assert not resolve_atom_windows(
            requested=True,
            n_atom=N_ATOM,
            n_token=N_TOKEN,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        one_device_value, one_device_hash, _ = serial_run(True)
    jax.clear_caches()
    off_value, off_hash, _ = serial_run(False)

    assert plain_hash == one_device_hash, (plain_hash, one_device_hash)
    assert plain_hash == off_hash, (plain_hash, off_hash)
    np.testing.assert_array_equal(plain_value, one_device_value)
    np.testing.assert_array_equal(plain_value, off_value)
    found = {op: plain_text.count(op) for op in COLLECTIVE_OPS}
    assert sum(found.values()) == 0, found
    assert plain_text.count('custom_call_target="sharding"') == 0
    assert plain_text.count("sharding={") == 0
    print("serial identity", cp_identity(), "hash", plain_hash[:16])
    print("serial collectives", found)

    # (2) The mesh must not leak into the serial program that follows it.
    with context_parallel(devices, layout="1d"):
        distributed = jax.jit(build(True))
        distributed(*args)
        distributed_hash = _fingerprint(
            distributed.lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
        )
    assert cp_mesh() is None, cp_mesh()
    assert cp_identity() == ("serial", 1, (1, 1), ()), cp_identity()
    after_value, after_hash, after_text = serial_run(True)
    assert distributed_hash != plain_hash
    assert after_hash == plain_hash, (after_hash, plain_hash)
    np.testing.assert_array_equal(plain_value, after_value)
    assert sum(after_text.count(op) for op in COLLECTIVE_OPS) == 0
    print("after cp hash", after_hash[:16], "cp hash", distributed_hash[:16])

    # (3) A failure inside the context still restores the ContextVar.
    class DeliberateFailureError(RuntimeError):
        pass

    try:
        with context_parallel(devices, layout="1d"):
            assert cp_mesh() is not None
            raise DeliberateFailureError
    except DeliberateFailureError:
        pass
    assert cp_runtime() is None, cp_runtime()
    assert cp_identity() == ("serial", 1, (1, 1), ()), cp_identity()
    print("contextvar restored after an exception")

    # (4) A nominal one-device context inside a mesh is refused, not inherited.
    with context_parallel(devices, layout="1d"):
        outer = cp_mesh()
        try:
            with context_parallel(1):
                raise AssertionError("nested context_parallel(1) was accepted")
        except RuntimeError as error:
            assert "does not nest" in str(error), error
            print("nested one-device context refused:", error)
        assert cp_mesh() is outer, (cp_mesh(), outer)
    assert cp_mesh() is None

    # A value nobody reads would make every assertion above vacuous.
    assert np.isfinite(plain_value).all()
    assert plain_value.shape == (jnp.asarray(case.r_l).shape[0], N_ATOM, 3)
    print("PROTENIX_ATOM_INVARIANCE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
