"""Exercise OpenDDE's atom-window alignment decision under a live mesh.

``atom_window_misalignment`` is unreachable without a mesh -- its first line
returns "no context-parallel mesh is active" -- and a mesh needs a device count
fixed before JAX initialises, so this runs in a child with
``FOLDJAX_CP_PROBE_DEVICES`` / ``FOLDJAX_CP_PROBE_LAYOUT`` set.

What it asserts that Protenix' equivalent cannot: the axis OpenDDE's resolution
point names. OpenDDE diffuses over its expanded *structural* tokens, and their
automatic padding target is twice the token bucket, so a caller sent to pin
``tokens`` would pin an axis that cannot fix the shape. The function under test
is the one the graph calls -- ``models/opendde/models/model.py``'s
``_resolve_atom_windows`` -- not a re-spelling of it here.

Output: per-case lines, then ``OPENDDE_ATOM_ALIGNMENT_OK``.
"""

from __future__ import annotations

import os
import sys
import warnings

from foldjax.models._cp import context_parallel, cp_grid
from foldjax.models.opendde.models.model import _resolve_atom_windows
from foldjax.models.protenix.models.diffusion._cp import (
    atom_window_misalignment,
    require_atom_windows,
)
from tests.models.protenix.atom_cp_fixtures import (
    N_ATOM,
    N_KEYS,
    N_QUERIES,
    N_TOKEN,
)


def _resolved(**kwargs) -> tuple[bool, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _resolve_atom_windows(**kwargs)
    return value, list(caught)


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]

    aligned = dict(
        n_atom=N_ATOM,
        n_structural_token=N_TOKEN,
        n_queries=N_QUERIES,
        n_keys=N_KEYS,
    )

    # Outside a mesh the request resolves to False with nothing to say: the
    # option is not an error there, it simply has no meaning.
    value, caught = _resolved(requested=True, **aligned)
    assert value is False, value
    assert not caught, [str(w.message) for w in caught]
    print("serial ->", value)

    with context_parallel(devices, layout=layout):
        rows, cols = cp_grid()
        print(f"devices={devices} layout={layout} rows={rows} cols={cols}")

        # An aligned shape resolves to True and warns about nothing.
        value, caught = _resolved(requested=True, **aligned)
        assert value is True, value
        assert not caught, [str(w.message) for w in caught]
        print("  aligned ->", value)

        # Asking for nothing is not a misalignment report.
        value, caught = _resolved(requested=False, **aligned)
        assert value is False, value
        assert not caught, [str(w.message) for w in caught]

        # (1) An atom axis that does not divide the rows into whole query
        # windows. `+ N_QUERIES` keeps it a whole number of windows, so the
        # reason has to come from the row split rather than from the window
        # geometry.
        misaligned_atoms = N_ATOM + N_QUERIES
        assert misaligned_atoms % (N_QUERIES * rows), (misaligned_atoms, rows)
        reason = atom_window_misalignment(
            n_atom=misaligned_atoms,
            n_token=N_TOKEN,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        assert reason is not None and str(misaligned_atoms) in reason, reason
        print("  misaligned atoms ->", reason)
        try:
            require_atom_windows(
                n_atom=misaligned_atoms,
                n_token=N_TOKEN,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
            )
        except ValueError as error:
            print("  adapter refused:", error)
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("a misaligned shape was accepted")

        value, caught = _resolved(
            requested=True,
            n_atom=misaligned_atoms,
            n_structural_token=N_TOKEN,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        assert value is False, value
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
        message = str(caught[0].message)
        assert f"multiple of {N_QUERIES * rows}" in message, message
        # The OpenDDE-specific half: the axis a caller is told to pin. Spelled
        # whole rather than as a substring test, because `structural_tokens`
        # contains `tokens` and a substring test would pass on the wrong axis.
        assert "PaddingConfig(atoms=..., structural_tokens=...)" in message, message
        print("  resolver warned:", message[-120:])

        # (2) A structural token count that does not divide the mesh is the
        # other half, and on this port it is the axis nothing else guards: the
        # residue token count can divide the rows while the structural one,
        # whose automatic target is twice the token bucket, does not.
        misaligned_tokens = N_TOKEN + 1
        assert misaligned_tokens % rows, (misaligned_tokens, rows)
        token_reason = atom_window_misalignment(
            n_atom=N_ATOM,
            n_token=misaligned_tokens,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        assert token_reason is not None and "token" in token_reason, token_reason
        print("  misaligned structural tokens ->", token_reason)

        value, caught = _resolved(
            requested=True,
            n_atom=N_ATOM,
            n_structural_token=misaligned_tokens,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        assert value is False, value
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
        message = str(caught[0].message)
        assert str(misaligned_tokens) in message, message
        assert "PaddingConfig(atoms=..., structural_tokens=...)" in message, message
        print("  resolver warned:", message[-120:])

    print("OPENDDE_ATOM_ALIGNMENT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
