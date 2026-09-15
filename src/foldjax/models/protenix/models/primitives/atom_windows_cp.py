"""The local atom-window trunk builder a sharded Protenix atom stream needs.

Protenix' blocked atom attention builds its key windows inside
:func:`foldjax.models.protenix.models.primitives.attention._local_qk_trunks`:
it zero-pads the key axis by ``(n_keys - n_queries) // 2`` on each side and
takes overlapping windows.  That padding reaches ``(n_keys - n_queries) // 2``
atoms beyond the block, so once the atom axis is split across devices the
operation stops being local and the array cannot simply be constrained --
it needs the neighbour's edge atoms.

Rather than thread a builder through
``diffusion_transformer_stack`` -> ``diffusion_transformer_block`` ->
``local_attention_pair_bias`` -> ``local_attention`` -- four signatures, three
of which OpenDDE also calls -- the sharded caller installs one here for the
duration of its ``shard_map`` body, the same way the active mesh itself is
task-local rather than passed down (:mod:`foldjax.models._cp`).  A plan is a
drop-in replacement for ``_local_qk_trunks``: same arguments, same four
results, so the serial site keeps exactly one code path and nothing about the
serial program depends on this module.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

import jax.numpy as jnp

from foldjax.models._cp_atom import single_to_keys_local


@dataclass(frozen=True, slots=True)
class AtomWindowPlan:
    """How one device turns its own atom shard into local query/key windows.

    ``window_mask`` is this device's slice of the global
    ``[n_windows, n_queries, n_keys]`` validity mask.  It is computed from
    global atom positions by the caller and handed in window-sharded, so the
    body needs no rank arithmetic to know which keys fall outside the
    structure.
    """

    n_queries: int
    n_keys: int
    #: Mesh axis owning the atom windows, and its size.
    axis_name: str
    axis_size: int
    window_mask: jnp.ndarray

    def to_keys(self, x: jnp.ndarray) -> jnp.ndarray:
        """``[..., A_local, C]`` -> ``[..., K_local, n_keys, C]`` with halo."""

        if x.ndim < 2:
            raise ValueError(f"atom-window keys expect [..., A, C], got {x.shape}")
        # `ppermute` moves the halo, and a boolean halo would ask the
        # collective for PRED. Both masks that reach here hold exactly 0 or 1,
        # so the round trip through float32 is value-preserving.
        boolean = x.dtype == jnp.bool_
        work = x.astype(jnp.float32) if boolean else x
        atoms, channels = work.shape[-2], work.shape[-1]
        keys = single_to_keys_local(
            work.reshape((-1, atoms, channels)),
            query_window=self.n_queries,
            key_window=self.n_keys,
            axis_name=self.axis_name,
            axis_size=self.axis_size,
        )
        keys = keys.reshape(x.shape[:-2] + keys.shape[-3:])
        return keys != 0 if boolean else keys

    def qk_trunks(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        *,
        n_queries: int,
        n_keys: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        """The sharded spelling of ``_local_qk_trunks``.

        ``q_pad`` is always zero: a sharded atom axis is required to be an
        exact multiple of ``n_queries * axis_size``, so there is no ragged
        final window to pad and crop.
        """

        if q.shape != k.shape:
            raise ValueError("local attention requires q and kv to share shape")
        if n_queries != self.n_queries or n_keys != self.n_keys:
            raise ValueError(
                "atom-window plan was built for "
                f"n_queries={self.n_queries}, n_keys={self.n_keys}; the call "
                f"asks for n_queries={n_queries}, n_keys={n_keys}"
            )
        atoms = q.shape[-2]
        if atoms % n_queries:
            raise ValueError(
                f"local atom shard {atoms} is not a multiple of the query "
                f"window {n_queries}"
            )
        local_windows = atoms // n_queries
        if local_windows != int(self.window_mask.shape[-3]):
            raise ValueError(
                "atom-window plan mask covers "
                f"{int(self.window_mask.shape[-3])} windows but the shard holds "
                f"{local_windows}"
            )
        q_trunked = q.reshape(q.shape[:-2] + (local_windows, n_queries, q.shape[-1]))
        k_trunked = self.to_keys(k)
        return q_trunked, k_trunked, self.window_mask, 0


_ACTIVE: ContextVar[AtomWindowPlan | None] = ContextVar(
    "foldjax_protenix_atom_window_plan",
    default=None,
)


def atom_window_plan() -> AtomWindowPlan | None:
    """The plan installed by the enclosing sharded body, or ``None``.

    ``None`` is the ordinary case and includes a context-parallel run whose
    atom streams are deliberately replicated, so no caller may treat a missing
    plan as an error.
    """

    return _ACTIVE.get()


@contextlib.contextmanager
def use_atom_window_plan(plan: AtomWindowPlan | None) -> Iterator[None]:
    """Install one plan for the duration of a traced sharded body."""

    token = _ACTIVE.set(plan)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


__all__ = ["AtomWindowPlan", "atom_window_plan", "use_atom_window_plan"]
