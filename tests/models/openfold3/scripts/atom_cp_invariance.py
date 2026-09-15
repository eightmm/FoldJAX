"""A live mesh must not reach the OpenFold3 atom graph on its own.

Three claims, all needing real devices and therefore a subprocess:

* with ``cp_atom_windows`` off the denoiser lowers to the *same program* under
  a four-device mesh as it does on one device -- the same normalised HLO hash
  the serial pin records, with no collective and no sharding annotation. That
  is what makes the option, rather than the mesh, the thing that distributes
  the atom graph, and what keeps ``--no-cp-atom-windows`` a real escape hatch;
* no plan reaches the four dispatch sites in that arm. The hash equality on its
  own does *not* say that: every guard reads ``cp_atom_windows and cp_mesh() is
  not None``, so with the option off the mesh is never consulted and the
  comparison would hold for any mesh at all. Reading the plan at a site both
  arms execute is what turns "the option was honoured" into "nothing installs a
  plan without the option";
* a shape the mesh cannot split resolves the request *down* to replicated and
  says so, with the multiples to pad to. A silent fallback here would be the
  worst outcome available: the run succeeds, the option reads as honoured, and
  the per-device memory nobody can account for is the replicated atom graph.
"""

from __future__ import annotations

import hashlib
import re
import sys
import warnings

import jax

import foldjax.models.openfold3.models.atom_blocks as atom_blocks
from foldjax.models._cp import context_parallel
from foldjax.models.openfold3.models.atom_cp import (
    atom_block_misalignment,
    atom_block_plan,
    resolve_atom_windows,
)
from foldjax.models.openfold3.models.denoiser import denoise
from tests.models.openfold3.atom_cp_fixtures import (
    ATOM_HEADS,
    N_ATOM,
    N_KEY,
    N_QUERY,
    N_TOKEN,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_case,
    build_params,
)
from tests.models.openfold3.scripts.serial_diffusion_lowering import (
    COLLECTIVE_OPS,
    _normalise,
)

DEVICES = 4


def _lower(case, params, *, distributed: bool, seen: list[bool] | None = None) -> str:
    def run(xl_noisy, t, si, si_trunk, zij):
        return denoise(
            case.batch,
            xl_noisy,
            t,
            si,
            si_trunk,
            zij,
            params,
            n_query=N_QUERY,
            n_key=N_KEY,
            atom_heads=ATOM_HEADS,
            token_heads=TOKEN_HEADS,
            n_token=N_TOKEN,
            sigma_data=SIGMA_DATA,
            cp_atom_windows=distributed,
        )

    args = (case.xl_noisy, case.t, case.si, case.si_trunk, case.zij)
    if seen is None:
        return jax.jit(run).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()

    # Patched at `atom_blocks.single_rep_to_blocks`, which
    # `cross_attention_pair_bias` imports *inside* its body and therefore looks
    # up per call, so this arm reaches the patched object whether or not a plan
    # is installed. An arm that never reached it would report "no plan" in
    # exactly the same way as one that saw none, so the caller also asserts the
    # site fired.
    original = atom_blocks.single_rep_to_blocks

    def spy(*spy_args, **spy_kwargs):
        seen.append(atom_block_plan() is not None)
        return original(*spy_args, **spy_kwargs)

    atom_blocks.single_rep_to_blocks = spy
    try:
        return jax.jit(run).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    finally:
        atom_blocks.single_rep_to_blocks = original


def main() -> int:
    assert jax.device_count() == DEVICES, jax.devices()
    case = build_case()
    params = build_params()

    seen_serial: list[bool] = []
    seen_mesh: list[bool] = []
    serial = _normalise(_lower(case, params, distributed=False, seen=seen_serial))
    jax.clear_caches()
    with context_parallel(DEVICES):
        # A fresh closure per arm: `jit` keys its cache on the callable, not on
        # the mesh ContextVar, so a reused one would never retrace and the
        # comparison would be a program against itself.
        under_mesh = _normalise(_lower(case, params, distributed=False, seen=seen_mesh))
    lowered = under_mesh.lower()
    collectives = sum(lowered.count(op) for op in COLLECTIVE_OPS)
    sharding = lowered.count('custom_call_target="sharding"') + lowered.count(
        "sharding={"
    )
    print("SERIAL", hashlib.sha256(serial.encode()).hexdigest()[:16])
    print("UNDER_MESH", hashlib.sha256(under_mesh.encode()).hexdigest()[:16])
    print("COLLECTIVES", collectives, "SHARDING", sharding)
    print("BLOCK_SITES", len(seen_serial), len(seen_mesh))
    print("PLANS_SEEN", sum(seen_serial), sum(seen_mesh))
    assert serial == under_mesh, (
        "an inactive cp_atom_windows still changed the program under a mesh"
    )
    assert collectives == 0 and sharding == 0
    # The blocking site fired on both arms -- otherwise "no plan" would be the
    # report of an arm that never ran it -- and saw no plan on either, so
    # nothing installs one without the option.
    assert len(seen_serial) == len(seen_mesh) > 0, (seen_serial, seen_mesh)
    assert not any(seen_serial) and not any(seen_mesh), (seen_serial, seen_mesh)

    with context_parallel(DEVICES):
        # `N_ATOM + N_QUERY` atoms still divides the query block but no longer
        # divides `n_query * 4`, so this is the alignment rule firing rather
        # than a shape the blocking itself rejects.
        misaligned = N_ATOM + N_QUERY
        reason = atom_block_misalignment(
            n_atom=misaligned, n_token=N_TOKEN, n_query=N_QUERY
        )
        assert reason is not None and "not a multiple" in reason, reason
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_atom_windows(
                requested=True,
                n_atom=misaligned,
                n_token=N_TOKEN,
                n_query=N_QUERY,
            )
        assert resolved is False
        assert len(caught) == 1, caught
        message = str(caught[0].message)
        print("WARNING", re.sub(r"\s+", " ", message))
        assert "stays replicated" in message
        assert f"multiple of {N_QUERY * DEVICES}" in message
        assert "PaddingConfig" in message

        # A token axis the rows cannot split is the other rule, and it names
        # the token count rather than the atom one.
        odd_tokens = atom_block_misalignment(
            n_atom=N_ATOM, n_token=N_TOKEN + 1, n_query=N_QUERY
        )
        assert odd_tokens is not None and "tokens do not divide" in odd_tokens
        print("TOKEN_REASON", odd_tokens)

        # An aligned shape resolves to True and warns about nothing.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert (
                resolve_atom_windows(
                    requested=True,
                    n_atom=N_ATOM,
                    n_token=N_TOKEN,
                    n_query=N_QUERY,
                )
                is True
            )
        assert not caught, caught

    # Outside a mesh the request is simply not a decision.
    assert (
        resolve_atom_windows(
            requested=True, n_atom=N_ATOM, n_token=N_TOKEN, n_query=N_QUERY
        )
        is False
    )
    print("OPENFOLD3_ATOM_INVARIANCE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
