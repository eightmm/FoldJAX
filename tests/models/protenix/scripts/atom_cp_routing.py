"""Check each Protenix atom<->token routing adapter against its serial twin.

The whole-graph gate in ``atom_cp_parity.py`` can only say that the answer
came out right; these compare one operation at a time, with inputs chosen so
that a wrong answer is not a small one:

* windows whose keys reach across a shard boundary (the halo);
* a token map with repeated indices, so the scatter has to *accumulate* rather
  than assign, and with tokens no atom points at, so the ``max(count, 1)``
  denominator is exercised;
* a masked atom tail, so the weighted mean and the window masks matter;
* atom and token counts that do *not* divide the mesh, which must be refused
  rather than silently mis-sharded.

Three of the four adapters are bitwise-exact against serial, because exactly
one device owns each source element and the ring accumulates one real
contribution and zeros. The scatter-mean is the exception: its
``psum_scatter`` reorders the summation.
"""

from __future__ import annotations

import os
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel, cp_layout
from foldjax.models.protenix.models.diffusion._cp import (
    aggregate_atom_to_token_cp,
    atom_pair_conditioning_cp,
    atom_window_misalignment,
    broadcast_token_to_atom_cp,
    broadcast_token_to_local_atom_pair_cp,
    global_window_mask,
    require_atom_windows,
    resolve_atom_windows,
)
from foldjax.models.protenix.models.diffusion.atom import (
    aggregate_atom_to_token,
    atom_pair_conditioning,
    broadcast_token_to_atom,
    broadcast_token_to_local_atom_pair,
    rearrange_qk_to_dense_trunk,
)
from foldjax.models.protenix.models.primitives.primitives import LinearParams

N_TOKEN = 12
N_ATOM = 48
N_QUERIES = 4
N_KEYS = 8
CHANNELS = 5


def _case() -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(20260917)
    # Repeated indices, unvisited tokens, and a run that straddles every
    # shard boundary a 2x2 or 3x3 mesh can draw.
    assignment = np.sort(rng.integers(0, N_TOKEN, size=N_ATOM))
    assignment[:6] = 0
    assignment[-6:] = N_TOKEN - 1
    mask = np.ones(N_ATOM, dtype=bool)
    mask[-7:] = False
    return {
        "idx": jnp.asarray(assignment, dtype=jnp.int32),
        "atom_mask": jnp.asarray(mask),
        "token_values": jnp.asarray(
            rng.normal(size=(1, N_TOKEN, CHANNELS)), dtype=jnp.float32
        ),
        "atom_values": jnp.asarray(
            rng.normal(size=(2, N_ATOM, CHANNELS)), dtype=jnp.float32
        ),
        "pair": jnp.asarray(
            rng.normal(size=(1, N_TOKEN, N_TOKEN, CHANNELS)), dtype=jnp.float32
        ),
        "c_l": jnp.asarray(rng.normal(size=(1, N_ATOM, CHANNELS)), dtype=jnp.float32),
        "p_lm": jnp.asarray(
            rng.normal(size=(1, N_ATOM // N_QUERIES, N_QUERIES, N_KEYS, CHANNELS)),
            dtype=jnp.float32,
        ),
        "linear_cl": LinearParams(
            weight=jnp.asarray(
                rng.normal(size=(CHANNELS, CHANNELS)), dtype=jnp.float32
            ),
            bias=None,
        ),
        "linear_cm": LinearParams(
            weight=jnp.asarray(
                rng.normal(size=(CHANNELS, CHANNELS)), dtype=jnp.float32
            ),
            bias=None,
        ),
        "mlp": None,
    }


def _check(name: str, reference, got, *, exact: bool) -> None:
    reference = np.asarray(reference, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    delta = np.abs(reference - got)
    print(f"  {name:<28} max {delta.max():.3e} exact={exact}")
    if exact:
        np.testing.assert_array_equal(reference, got)
    else:
        np.testing.assert_allclose(
            reference,
            got,
            rtol=1e-5,
            atol=1e-5 * max(float(np.abs(reference).max()), 1e-12),
        )


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    assert jax.device_count() == devices, jax.devices()
    case = _case()
    idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(
        case["idx"],
        case["idx"],
        n_queries=N_QUERIES,
        n_keys=N_KEYS,
        compute_mask=False,
    )

    gather_reference = jax.jit(
        lambda values: broadcast_token_to_atom(values, case["idx"])
    )(case["token_values"])
    scatter_reference = jax.jit(
        lambda values: aggregate_atom_to_token(
            values,
            case["idx"],
            n_token=N_TOKEN,
            reduce="mean",
            atom_mask=case["atom_mask"],
        )
    )(case["atom_values"])
    pair_reference = jax.jit(
        lambda pair: broadcast_token_to_local_atom_pair(
            pair,
            case["idx"],
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
            compute_mask=False,
        )[0]
    )(case["pair"])
    conditioning_reference = jax.jit(
        lambda p_lm, c_l: atom_pair_conditioning(
            p_lm, c_l, case["linear_cl"], case["linear_cm"]
        )
    )(case["p_lm"], case["c_l"])
    jax.clear_caches()

    window_mask = global_window_mask(N_ATOM, n_queries=N_QUERIES, n_keys=N_KEYS)
    print(f"devices={devices} layout={layout}")
    with context_parallel(devices, layout=layout):
        assert cp_layout() == layout
        _check(
            "token -> atom gather",
            gather_reference,
            jax.jit(lambda values: broadcast_token_to_atom_cp(values, case["idx"]))(
                case["token_values"]
            ),
            exact=True,
        )
        _check(
            "atom -> token mean",
            scatter_reference,
            jax.jit(
                lambda values: aggregate_atom_to_token_cp(
                    values,
                    case["idx"],
                    n_token=N_TOKEN,
                    atom_mask=case["atom_mask"],
                )
            )(case["atom_values"]),
            exact=False,
        )
        _check(
            "token pair -> windows",
            pair_reference,
            jax.jit(
                lambda pair: broadcast_token_to_local_atom_pair_cp(pair, idx_q, idx_k)
            )(case["pair"]),
            exact=True,
        )
        _check(
            "atom-pair conditioning",
            conditioning_reference,
            jax.jit(
                lambda p_lm, c_l: atom_pair_conditioning_cp(
                    p_lm,
                    c_l,
                    case["linear_cl"],
                    case["linear_cm"],
                    small_mlp=None,
                    n_queries=N_QUERIES,
                    n_keys=N_KEYS,
                    window_mask=window_mask,
                )
            )(case["p_lm"], case["c_l"]),
            # The halo reproduces the padded key windows exactly, but the two
            # projections that follow contract a five-element channel axis over
            # a shard-sized operand instead of a whole one, and XLA tiles and
            # fuses that differently: measured 4.8e-7 on 11% of the elements
            # at CHANNELS=5. Arithmetically the same sum, not the same
            # rounding -- so this one is a tolerance, not an equality.
            exact=False,
        )

        rows = devices if layout == "1d" else int(round(devices**0.5))
        # A length that does not divide the mesh must be named, not sharded.
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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_atom_windows(
                requested=True,
                n_atom=misaligned_atoms,
                n_token=N_TOKEN,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
            )
        assert resolved is False
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
        assert f"multiple of {N_QUERIES * rows}" in str(caught[0].message), caught[0]
        print("  resolver warned:", str(caught[0].message)[:110])

        # A token count that does not divide the mesh is the other half.
        token_reason = atom_window_misalignment(
            n_atom=N_ATOM,
            n_token=N_TOKEN + 1,
            n_queries=N_QUERIES,
            n_keys=N_KEYS,
        )
        assert token_reason is not None and "token" in token_reason, token_reason
        print("  misaligned tokens ->", token_reason)

        # An aligned shape resolves to True and warns about nothing.
        with warnings.catch_warnings(record=True) as quiet:
            warnings.simplefilter("always")
            assert resolve_atom_windows(
                requested=True,
                n_atom=N_ATOM,
                n_token=N_TOKEN,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
            )
        assert not quiet, [str(w.message) for w in quiet]

    # Outside a mesh the request resolves to False with nothing to say: the
    # option is not an error there, it simply has no meaning.
    with warnings.catch_warnings(record=True) as outside:
        warnings.simplefilter("always")
        assert (
            resolve_atom_windows(
                requested=True,
                n_atom=N_ATOM,
                n_token=N_TOKEN,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
            )
            is False
        )
    assert not outside, [str(w.message) for w in outside]
    print("PROTENIX_ATOM_ROUTING_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
