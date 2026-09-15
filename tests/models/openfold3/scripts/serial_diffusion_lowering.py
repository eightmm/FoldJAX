"""Print a normalised fingerprint of the serial OpenFold3 denoiser lowering.

Run with ``foldjax`` on ``PYTHONPATH`` from two different checkouts and the two
fingerprints answer one question: did this change alter the program a serial
run executes?  The parameters and features come from
``tests.models.openfold3.atom_cp_fixtures``, which is always taken from the
tree under test's *tests* directory, so only ``foldjax`` differs between the
two runs.

The call deliberately omits ``cp_atom_windows``: the point is to compare the
call an existing caller makes, and on an older tree that keyword does not
exist.  Every context-parallel entry point is a no-op without a mesh anyway,
so the default call is the whole serial contract.

Output: ``HASH``, ``COLLECTIVES``, ``SHARDING_CALLS`` and
``SHARDING_ANNOTATIONS``, one per line.
"""

from __future__ import annotations

import hashlib
import re
import sys

import jax

from foldjax.models.openfold3.models.denoiser import denoise
from tests.models.openfold3.atom_cp_fixtures import (
    ATOM_HEADS,
    N_KEY,
    N_QUERY,
    N_TOKEN,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_case,
    build_params,
)

#: Collective and sharding operations no single-device program may contain.
COLLECTIVE_OPS = (
    "collective-permute",
    "all-gather",
    "all-reduce",
    "reduce-scatter",
    "all-to-all",
)


def _normalise(text: str) -> str:
    """Drop everything that records where the source lived, not what it does.

    ``metadata={...}`` carries ``source_file``/``source_line`` and a
    ``stack_frame_id`` index into a table of them, so two checkouts of the same
    program disagree on it as soon as any line number moves -- which a patch
    that adds a guarded branch always does.
    """

    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    case = build_case()
    params = build_params()

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
        )

    args = (case.xl_noisy, case.t, case.si, case.si_trunk, case.zij)
    text = jax.jit(run).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    normalised = _normalise(text)
    lowered = normalised.lower()
    print("HASH", hashlib.sha256(normalised.encode()).hexdigest())
    print("COLLECTIVES", sum(lowered.count(op) for op in COLLECTIVE_OPS))
    # `with_sharding_constraint` lowers to a `Sharding` custom call; an
    # explicit spec also prints as a `sharding={...}` attribute.
    print("SHARDING_CALLS", lowered.count('custom_call_target="sharding"'))
    print("SHARDING_ANNOTATIONS", lowered.count("sharding={"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
