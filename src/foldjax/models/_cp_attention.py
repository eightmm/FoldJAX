"""Gather-free two-dimensional Fold-CP attention primitives.

The pair representation is tiled over a square ``cp_row x cp_col`` mesh.
Queries stay resident while key, value, mask and pair-bias tiles rotate through
a ring. An fp32 online-softmax accumulator makes the result mathematically
equivalent to dense attention without materialising a full token axis.

The ring runs one block of local pair rows at a time, in a ``lax.scan`` around
the whole rotation. Every pair row is an independent attention, so a block
needs nothing but its own rows of ``q``/``k``/``v`` -- which
:func:`ring_triangle_attention_2d_from_pair` projects inside the block -- and
its score tile and accumulators are the block's width, not the tile's. See
:func:`resolve_ring_row_block`.

:func:`gather_triangle_attention_2d_from_pair` is the ring's opt-in sibling
under the same sharding contract: per row block it gathers the block's
full-width pair rows along ``cp_col`` instead of rotating tiles, so one
normalising attention call -- cuEquivariance's on a GPU -- replaces the ring's
rotation and merge. See :data:`TRIANGLE_ATTENTION_GRIDS`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    cp_grid,
    cp_layout,
    cp_mesh,
    permute,
    transpose_perm,
)
from foldjax.models._tokamax_attention import tokamax_available

#: What a ring step may evaluate one local tile with. ``xla`` is the two-pass
#: global-maximum ring below. ``tokamax`` runs the fused Triton attention per
#: tile and merges the tiles by their softmax statistics, which is a different
#: program and different arithmetic, not a faster spelling of the same one; it
#: is GPU-only, because the entry point it needs returns softmax residuals
#: from a Pallas/Triton kernel.
#:
#: Which of the two an omitted option realises is a *backend* decision and not
#: this module's: Boltz-2 resolves it on the host against the card and the
#: layout (``backends/boltz2._realised_ring_tile_kernel``) and hands the scope
#: the answer, Protenix still resolves it to ``xla``. What this module owns is
#: the vocabulary, the refusals, and the body a bare model call runs when
#: nobody set a scope at all -- which is ``xla`` below, deliberately: a model
#: called outside an adapter must compile the portable program.
RING_TILE_KERNELS: tuple[str, ...] = ("xla", "tokamax")

#: Every tile kernel :func:`_ring_local_rows` will run, which is one more than
#: the option offers. ``xla_merge`` is the merge ring -- one rotation, tiles
#: folded by :func:`merge_softmax_statistics` -- driven by the portable tile
#: instead of the fused one. It is deliberately not a request a backend can
#: spell: it is slower than both of the others and buys nothing in a
#: prediction. What it buys is that the merge ring's rotation schedule,
#: accumulator shapes and empty-row handling are executed by the CPU gates,
#: rather than first executed on the four cards the experiment allocates.
RING_TILE_BODIES: tuple[str, ...] = ("xla", "xla_merge", "tokamax")

_TILE_KERNEL: ContextVar[str] = ContextVar(
    "foldjax_ring_tile_kernel",
    default="xla",
)


def ring_tile_kernel() -> str:
    """The tile kernel the active scope asks a 2-D ring step to use.

    A scope rather than a keyword because no model in this repository takes
    this as an argument: it would have to be threaded through every trunk,
    Pairformer, MSA, template and confidence signature between the adapter and
    the ring. ``matmul_precision`` travels the same way and for the same
    reason (``backends/base.py``).
    """

    return _TILE_KERNEL.get()


@contextmanager
def ring_tile_kernel_scope(kernel: str | None) -> Iterator[str]:
    """Run the enclosed prediction with ``kernel`` inside every 2-D ring step.

    ``None`` publishes ``xla``: the scope is still entered, so a caller need
    not branch. That is the path a caller with nothing to resolve takes --
    Protenix's adapter, and a test. Boltz-2's adapter resolves an omitted
    option on the host and passes the realised name, so what this publishes
    for that port is a word somebody decided rather than a default reached by
    omission twice.
    """

    name = "xla" if kernel is None else str(kernel)
    if name not in RING_TILE_KERNELS:
        raise ValueError(
            f"triangle_attention_ring_kernel must be one of "
            f"{RING_TILE_KERNELS}, got {name!r}"
        )
    token = _TILE_KERNEL.set(name)
    try:
        yield name
    finally:
        _TILE_KERNEL.reset(token)


#: How 2-D context-parallel triangle attention reaches the keys outside a
#: device's pair tile. ``ring`` is the released program: K/V/mask/bias tiles
#: rotate around the grid and the tiles' softmax terms are combined
#: (:func:`ring_triangle_attention_2d_from_pair`). ``gather`` streams each row
#: block's full-width pair rows to the device instead and runs one normalising
#: attention on them (:func:`gather_triangle_attention_2d_from_pair`) --
#: cuEquivariance's triangle-attention kernel on a GPU, the XLA reference
#: body :data:`GATHER_ATTENTION_BODIES` names everywhere else. Opt-in and
#: unmeasured on a card; see ``docs/context_parallel.md``.
TRIANGLE_ATTENTION_GRIDS: tuple[str, ...] = ("ring", "gather")

#: What one gathered row block is attended with. Not a request anybody spells:
#: the platform decides (:func:`resolve_gather_attention_body`), because the
#: two are the same data movement with different arithmetic in the middle, and
#: which one a host can run is not a preference.
GATHER_ATTENTION_BODIES: tuple[str, ...] = ("xla", "cueq")

_TRIANGLE_ATTENTION_GRID: ContextVar[str] = ContextVar(
    "foldjax_triangle_attention_grid",
    default="ring",
)


def triangle_attention_grid() -> str:
    """The 2-D triangle-attention algorithm the active scope selects.

    A scope for the reason :func:`ring_tile_kernel` gives: no trunk, MSA,
    template or confidence signature between an adapter and the 2-D entry
    points carries it.
    """

    return _TRIANGLE_ATTENTION_GRID.get()


@contextmanager
def triangle_attention_grid_scope(grid: str | None) -> Iterator[str]:
    """Run the enclosed model call with ``grid`` at every 2-D triangle attention.

    This is the handle a GPU harness outside the repository uses to activate
    the gather path around a bare model call::

        with context_parallel(4, layout="2d"), triangle_attention_grid_scope(
            "gather"
        ):
            out = model_forward(...)

    ``None`` publishes ``ring``, the released program, so a caller need not
    branch. What ``gather`` runs is decided when a triangle attention is
    traced (:func:`resolve_gather_attention_body`), and an explicit ``gather``
    the host cannot honour is refused there rather than downgraded.
    """

    name = "ring" if grid is None else str(grid)
    if name not in TRIANGLE_ATTENTION_GRIDS:
        raise ValueError(
            f"triangle_attention_grid must be one of "
            f"{TRIANGLE_ATTENTION_GRIDS}, got {name!r}"
        )
    token = _TRIANGLE_ATTENTION_GRID.set(name)
    try:
        yield name
    finally:
        _TRIANGLE_ATTENTION_GRID.reset(token)


def refuse_triangle_attention_grid(port: str) -> None:
    """Refuse a ``gather`` scope around a port that does not offer it.

    Boltz-2, Protenix and OpenFold3 read :func:`triangle_attention_grid` at
    every 2-D triangle attention and their backends record it. A port without
    that option must not run under the scope anyway: OpenDDE's trunk is
    Protenix's Pairformer, so it would run the gather under a cache namespace
    that never names it, and ESMFold2 has no triangle attention at all, so a
    harness would report a gather that never happened. Either way one
    namespace would hold two programs, or one program two names -- the rule
    :func:`resolve_ring_tile_kernel` states. Called on the host, before any
    work, by each such port's model entry.
    """

    grid = triangle_attention_grid()
    if grid != "ring":
        raise ValueError(
            f"{port} has no triangle_attention_grid option, so "
            f"triangle_attention_grid={grid!r} cannot have been asked for "
            "through this port; the scope is refused rather than ignored"
        )


def resolve_gather_attention_body() -> str:
    """The local attention body the gather path runs on this host.

    ``cueq`` on the GPU backend, where it is the only body: a GPU process that
    cannot import cuEquivariance is refused, never handed the XLA reference
    under the gather label -- that would be two programs under one command,
    the rule :func:`resolve_ring_tile_kernel` states. ``xla`` everywhere else,
    which is not a downgrade: off a GPU the reference body *is* what ``gather``
    means, so that its data movement is executed by the CPU gates.

    Asking initialises a JAX backend, so a backend calls this on the host
    after the backend is up, never while validating a request.
    """

    platform = jax.default_backend()
    if platform != "gpu":
        return "xla"
    from foldjax.models._cueq import load_cueq

    try:
        load_cueq()
    except RuntimeError as error:
        raise RuntimeError(
            "triangle_attention_grid='gather' runs cuEquivariance's triangle "
            "attention on a GPU, which did not import in this process"
        ) from error
    return "cueq"


#: What ``cp_fused_attention`` may name. ``off`` is the released value and the
#: only one every distributed measurement in this repository describes. The
#: other three are an opt-in experiment, meant for a card and unmeasured on
#: one, that runs a fused kernel at the two Boltz-2 diffusion attentions whose
#: operands are already entirely local inside their ``shard_map``: the
#: halo-exchanged atom windows, and the grid-transposed token tile.
#:
#: Four values rather than the three a graded ladder would need, because the
#: release policy this implements promotes sites individually -- a site's
#: accuracy and timing arm has to be measurable on its own, and ``token``
#: alone is not expressible by a ladder.
CP_FUSED_ATTENTION_REQUESTS: tuple[str, ...] = ("off", "atom", "token", "atom+token")

#: The sites a request names. The name is the mesh-local attention, not a
#: kernel: what each site runs is decided at the site, because the two do not
#: reach tokamax the same way (:func:`resolve_cp_fused_attention`).
CP_FUSED_ATTENTION_SITES: tuple[str, ...] = ("atom", "token")

_CP_FUSED_ATTENTION: ContextVar[str] = ContextVar(
    "foldjax_cp_fused_attention",
    default="off",
)


def cp_fused_attention() -> str:
    """The fused-attention sites the active scope opens under a mesh.

    A scope rather than a keyword for the reason :func:`ring_tile_kernel`
    gives: neither site is reachable from a signature a caller spells. The atom
    one sits under ``atom_transformer_forward`` inside the diffusion module's
    encoder and decoder, and the token one under every diffusion transformer
    layer.
    """

    return _CP_FUSED_ATTENTION.get()


def cp_fused_attention_sites(request: str | None = None) -> frozenset[str]:
    """The site names ``request`` opens, validating its spelling.

    ``None`` reads the active scope, so a site can ask this without knowing
    whether its caller spelled the option.
    """

    name = cp_fused_attention() if request is None else str(request)
    if name not in CP_FUSED_ATTENTION_REQUESTS:
        raise ValueError(
            f"cp_fused_attention must be one of {CP_FUSED_ATTENTION_REQUESTS}, "
            f"got {name!r}"
        )
    if name == "off":
        return frozenset()
    return frozenset(name.split("+"))


@contextmanager
def cp_fused_attention_scope(request: str | None) -> Iterator[str]:
    """Run the enclosed prediction with ``request``'s sites opened.

    ``None`` is the default: the scope is still entered, so a caller need not
    branch, and the value it publishes is the released ``off``.
    """

    name = "off" if request is None else str(request)
    # Spelling is settled here rather than at the site, so a misspelling is an
    # error before a featurizer runs.
    cp_fused_attention_sites(name)
    token = _CP_FUSED_ATTENTION.set(name)
    try:
        yield name
    finally:
        _CP_FUSED_ATTENTION.reset(token)


def resolve_cp_fused_attention(site: str) -> bool:
    """Whether ``site`` runs its fused kernel here; raise if it cannot.

    Refused rather than downgraded, for the reason
    :func:`resolve_ring_tile_kernel` states: a silent fallback would let two
    machines run two different programs under one command.

    There is deliberately no platform check, which is where this differs from
    the ring's resolver. The ring pins tokamax's Triton implementation because
    it needs the residual-returning entry point, so off a GPU it can only
    raise. These two sites are not both in that position: the atom one goes
    through ``tokamax.dot_product_attention`` with tokamax's own implementation
    order -- the call the *serial* released diffusion attention already makes
    -- and that order reaches a portable XLA implementation. So the dispatch,
    the unchecked ``shard_map`` and the sharding contract are all executable by
    a CPU gate. Which implementation a card selects is a census on the card,
    and not a property this function could assert anyway.
    """

    if site not in CP_FUSED_ATTENTION_SITES:
        raise ValueError(
            f"cp_fused_attention site must be one of "
            f"{CP_FUSED_ATTENTION_SITES}, got {site!r}"
        )
    request = cp_fused_attention()
    if site not in cp_fused_attention_sites(request):
        return False
    if cp_mesh() is None:
        raise RuntimeError(
            f"cp_fused_attention={request!r} names context-parallel attention "
            f"sites; the {site!r} site was reached with no mesh active"
        )
    if not tokamax_available():
        raise RuntimeError(
            f"cp_fused_attention={request!r} needs the tokamax package, "
            "which did not import in this process"
        )
    return True


def cp_fused_shard_map_options(fused: bool) -> dict[str, bool]:
    """``shard_map`` keywords a fused site needs and a checked one does not.

    The same bookkeeping :func:`_ring_shard_map_options` documents: a Pallas
    kernel declares its outputs as :class:`jax.ShapeDtypeStruct` with no
    ``manual_axis_type``, which ``check_vma=True`` requires of every output
    produced inside a ``shard_map``, so the kernel raises out of the
    partitioner before it computes anything.

    An unfused site passes no keyword at all, so it calls ``shard_map`` the way
    it called it before this option existed -- not ``check_vma=True``, which
    would be a second spelling of one program.
    """

    return {"check_vma": False} if fused else {}


def ring_tile_kernel_available() -> bool:
    """Whether this process could run the fused tile at all.

    Both halves of :func:`resolve_ring_tile_kernel`'s refusals as one
    question, for the one caller that has to *decide* rather than validate: a
    backend resolving an omitted option needs the answer on the host, before
    featurization, so the compilation-cache identity can record the body the
    run will realise rather than the word the caller did not type.

    Asking initialises a JAX backend, so it is for the resolution path only --
    ``resolve_cache_dir`` reads ``runtime_profile()`` first, so the backend is
    already up by the time a profile is built -- and never for request
    validation, which ``foldjax plan`` runs device-free.
    """

    return tokamax_available() and jax.default_backend() == "gpu"


def resolve_ring_tile_kernel(kernel: str | None) -> str:
    """Validate a tile-kernel request against what this process can run.

    Refused rather than downgraded. A silent fallback would make two machines
    run two different programs under one command, which is the rule
    ``foldjax.execution`` states for every kernel knob: a build that cannot
    reach a fused path says so. That refusal is what a backend's host-side
    resolution leans on: an omitted option never resolves to ``tokamax`` off a
    card, so a ``tokamax`` arriving here is an explicit request and the
    refusal lands on the caller who typed it.

    The two halves are asked separately rather than through
    :func:`ring_tile_kernel_available`, because a refusal has to name which
    half is missing and one boolean cannot.
    """

    name = "xla" if kernel is None else str(kernel)
    if name not in RING_TILE_KERNELS:
        raise ValueError(
            f"triangle_attention_ring_kernel must be one of "
            f"{RING_TILE_KERNELS}, got {name!r}"
        )
    if name == "xla":
        return name
    if not tokamax_available():
        raise RuntimeError(
            "triangle_attention_ring_kernel='tokamax' needs the tokamax "
            "package, which did not import in this process"
        )
    platform = jax.default_backend()
    if platform != "gpu":
        raise RuntimeError(
            "triangle_attention_ring_kernel='tokamax' runs a Pallas/Triton "
            f"kernel and needs the GPU backend; this process is on {platform!r}"
        )
    return name


def _flat(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    side: int,
) -> list[tuple[int, int]]:
    return [(a * side + b, c * side + d) for (a, b), (c, d) in pairs]


def triangle_bias_stage0_perm(side: int) -> list[tuple[int, int]]:
    """Flatten lower diagonals onto rows: ``(r, c) -> (r-c, c)``."""

    return _flat(
        [
            ((row, col), ((row - col) % side, col))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_stage1_perm(side: int) -> list[tuple[int, int]]:
    """Rotate flattened diagonals: ``(r, c) -> (r, c+r)``."""

    return _flat(
        [
            ((row, col), (row, (col + row) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_kv_initial_perm(side: int) -> list[tuple[int, int]]:
    """Offset K/V tiles onto the redistributed bias diagonal."""

    return triangle_bias_stage1_perm(side)


def triangle_kv_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance K/V/mask one key tile around each grid row."""

    return _flat(
        [
            ((row, col), (row, (col + 1) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance bias one matching key tile up each grid column."""

    return _flat(
        [
            ((row, col), ((row - 1) % side, col))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def _resolve_axis(axis: int, ndim: int, *, name: str) -> int:
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"{name} {axis} is out of range for rank {ndim}")
    return resolved


def _two_axis_spec(
    ndim: int,
    row_axis: int,
    col_axis: int,
) -> PartitionSpec:
    row = _resolve_axis(row_axis, ndim, name="row axis")
    col = _resolve_axis(col_axis, ndim, name="column axis")
    if row == col:
        raise ValueError("row and column axes must differ")
    entries: list[str | None] = [None] * ndim
    entries[row] = CP_ROW_AXIS
    entries[col] = CP_COL_AXIS
    return PartitionSpec(*entries)


def fold_cp_pad_width(size: int) -> int:
    """Rows to append so ``size`` divides the square mesh side."""

    side = cp_grid()[0]
    return 0 if side <= 1 else (-size) % side


def _widen(
    array: jax.Array,
    pads: Sequence[tuple[int, int]],
) -> jax.Array:
    if not any(width for _, width in pads):
        return array
    widths = [(0, 0)] * array.ndim
    for axis, width in pads:
        widths[_resolve_axis(axis, array.ndim, name="pad axis")] = (0, width)
    return jnp.pad(array, widths)


def _softmax_rescale(
    source_maximum: jax.Array,
    next_maximum: jax.Array,
) -> jax.Array:
    """Stable rescale, including empty ``-inf`` and ``+inf`` blocks."""

    finite = jnp.isfinite(source_maximum) & jnp.isfinite(next_maximum)
    positive_infinity = jnp.isposinf(source_maximum) & jnp.isposinf(next_maximum)
    difference = jnp.where(finite, source_maximum - next_maximum, 0.0)
    return jnp.where(
        positive_infinity,
        jnp.ones_like(difference),
        jnp.where(finite, jnp.exp(difference), jnp.zeros_like(difference)),
    )


def _compensated_add(
    total: jax.Array,
    correction: jax.Array,
    term: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Neumaier-add one ring tile without changing its communication schedule.

    A three-by-three mesh combines three independently reduced key tiles. Plain
    fp32 addition can lose enough low bits there to become visible after several
    Pairformer residual blocks. The correction tensor has the same local output
    shape, so the per-device asymptotic memory remains ``O(N^2/P)`` and no
    collective is introduced.
    """

    updated = total + term
    residual = jnp.where(
        jnp.abs(total) >= jnp.abs(term),
        (total - updated) + term,
        (term - updated) + total,
    )
    return updated, correction + residual


def online_softmax_update(
    output: jax.Array,
    normalizer: jax.Array,
    maximum: jax.Array,
    block_output: jax.Array,
    block_normalizer: jax.Array,
    block_maximum: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge one locally normalised key tile into an online accumulator.

    This public helper keeps its historical three-state API. The production
    ring below additionally carries compensation terms for the numerator and
    denominator so repeated residual blocks stay close to the dense fp32 path.
    """

    next_maximum = jnp.maximum(maximum, block_maximum)
    previous_scale = _softmax_rescale(maximum, next_maximum)
    block_scale = _softmax_rescale(block_maximum, next_maximum)
    return (
        previous_scale * output + block_scale * block_output,
        previous_scale * normalizer + block_scale * block_normalizer,
        next_maximum,
    )


#: Rows per local block times heads. The serial and 1-D paths pick their row
#: block from this same product (protenix ``_SCORE_ROWS_TIMES_HEADS``), because
#: one block's score tile is ``rows * heads * keys^2``: what has to stay
#: bounded is the product, and a swept optimum of 64 rows at 4 heads and ~25 at
#: 12 is the same 288 either way.
RING_SCORE_ROWS_TIMES_HEADS = 288
#: Ceiling on one block's score tile. A fixed row count still grows it as
#: ``keys^2``, so this is what narrows the block again on a very wide local
#: tile -- the case the 2-D layout exists for.
RING_SCORE_CEILING_BYTES = 8 * 1024**3
RING_MAX_ROWS_PER_BLOCK = 64
RING_MIN_ROWS_PER_BLOCK = 8


def resolve_ring_row_block(
    local_rows: int,
    *,
    heads: int,
    local_keys: int,
    requested: int | None = None,
) -> int:
    """Return the local pair rows one ring block runs at a time.

    The blocked axis is the *pair row* -- ``i`` in ``z[i, j]`` -- exactly as on
    the serial and 1-D paths. Triangle attention is a batch of independent
    attentions, one per row, and nothing in the softmax crosses a row, so one
    block of rows bounds ``q``/``k``/``v``, the score tile and the online
    accumulators together. Blocking the query axis instead, which is what this
    ring did, bounds only the score tile: ``q``, ``k``, ``v`` and both fp32
    accumulators stayed whole and the ring's live set stayed quadratic in the
    local width (1,969 MiB of loop carry at 1,024 OpenDDE structural tokens on
    a 2x2 mesh, five ``[N/2, heads, N/2, 32]`` tensors at once).

    The rule is the serial path's: ``rows * heads`` near
    :data:`RING_SCORE_ROWS_TIMES_HEADS`, capped at
    :data:`RING_MAX_ROWS_PER_BLOCK`, and narrowed further when one block's
    score tile would pass :data:`RING_SCORE_CEILING_BYTES`. The key extent
    here is already local (``N / sqrt(P)``), so the tile a given row count
    buys is ``P`` times smaller than the serial path's.

    ``requested`` is the caller's knob (``triangle_att_q_chunk_size`` /
    ``q_chunk_size`` / OpenFold3's ``chunk_size``): ``None`` takes the rule,
    a positive value is narrowed by it, and a non-positive one means one block
    -- the unblocked ring, which is the program this had before blocking
    existed and the same "dense" the serial knob spells that way.
    """

    if local_rows < 2:
        return local_rows
    if requested is not None and requested <= 0:
        return local_rows
    cap = min(RING_MAX_ROWS_PER_BLOCK, max(1, RING_SCORE_ROWS_TIMES_HEADS // heads))
    per_row = heads * local_keys * local_keys * 4
    if per_row <= 0:
        allowed = cap
    else:
        allowed = max(
            RING_MIN_ROWS_PER_BLOCK,
            min(cap, RING_SCORE_CEILING_BYTES // per_row),
        )
    block = allowed if requested is None else min(requested, allowed)
    return min(max(block, 1), local_rows)


def _rows_of(
    array: jax.Array,
    start: int | jax.Array | None,
    size: int,
    axis: int,
) -> jax.Array:
    """One block of rows, or the whole axis when ``start`` is ``None``.

    ``None`` is the single-block case and takes no slice at all, so the
    unblocked ring lowers to the program it lowered to before row blocking.
    """

    if start is None:
        return array
    return jax.lax.dynamic_slice_in_dim(array, start, size, axis=axis)


def _tile_scores(
    q_b: jax.Array,
    k_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    precision: jax.lax.Precision | None,
) -> jax.Array:
    scores = jnp.matmul(
        q_b.astype(jnp.float32),
        jnp.swapaxes(k_t.astype(jnp.float32), -1, -2),
        precision=precision,
    )
    return scores + bias_b.astype(jnp.float32) + mask_t.astype(jnp.float32)


def _tile_terms_from_scores(
    scores: jax.Array,
    v_t: jax.Array,
    maximum_b: jax.Array,
    precision: jax.lax.Precision | None,
) -> tuple[jax.Array, jax.Array]:
    finite_maximum = jnp.isfinite(maximum_b)
    positive_infinity = jnp.isposinf(maximum_b)
    shifted = jnp.where(
        finite_maximum,
        scores - maximum_b,
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
        v_t.astype(jnp.float32),
        precision=precision,
    )
    return block_output, block_normalizer


def _tile_terms(
    q_b: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    maximum_b: jax.Array,
    precision: jax.lax.Precision | None,
) -> tuple[jax.Array, jax.Array]:
    return _tile_terms_from_scores(
        _tile_scores(q_b, k_t, bias_b, mask_t, precision),
        v_t,
        maximum_b,
        precision,
    )


#: One ring step's locally normalised tile: ``(output, maximum, normalizer)``
#: in fp32, where ``output`` is ``sum_j exp(score_ij - maximum_i) v_j`` -- the
#: *unnormalised* numerator -- and ``normalizer`` is the matching denominator.
#: ``maximum`` and ``normalizer`` carry a trailing size-1 key axis so they
#: broadcast against ``output`` without a reshape.
#:
#: A tile whose every key is masked away reports ``maximum = -inf`` and zeros,
#: which is what :func:`_softmax_rescale` reads as "contributes nothing".
TileAttention = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]


def merge_softmax_statistics(
    output: jax.Array,
    output_correction: jax.Array,
    normalizer: jax.Array,
    normalizer_correction: jax.Array,
    maximum: jax.Array,
    block_output: jax.Array,
    block_maximum: jax.Array,
    block_normalizer: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fold one locally normalised tile into a compensated online accumulator.

    The standard statistics merge, ``m = max(m, m_t)`` then both sides rescaled
    onto it, with :func:`_compensated_add` carrying the numerator's and the
    denominator's low bits exactly as the two-pass ring does. The corrections
    are rescaled with their totals: a correction is a residual *of* the total
    it belongs to, so a scale the total takes and the correction does not would
    make the pair describe two different sums.

    :func:`online_softmax_update` is the same recurrence without the
    compensation, and keeps its historical three-state API for callers outside
    this module.
    """

    next_maximum = jnp.maximum(maximum, block_maximum)
    previous_scale = _softmax_rescale(maximum, next_maximum)
    block_scale = _softmax_rescale(block_maximum, next_maximum)
    output, output_correction = _compensated_add(
        output * previous_scale,
        output_correction * previous_scale,
        block_scale * block_output,
    )
    normalizer, normalizer_correction = _compensated_add(
        normalizer * previous_scale,
        normalizer_correction * previous_scale,
        block_scale * block_normalizer,
    )
    return (
        output,
        output_correction,
        normalizer,
        normalizer_correction,
        next_maximum,
    )


def tile_attention_xla(
    q_b: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The :data:`TileAttention` contract in portable XLA.

    This is the reference the fused tile is measured against, and the only one
    of the two that runs anywhere: the merge below is arithmetic, not a kernel,
    and a GPU-only tile would leave it untestable on the platform every gate in
    this repository runs on.

    It is *not* the shipped 2-D program. The default ring fixes one global row
    maximum in a first pass and never rescales (:func:`_ring_local_rows`); this
    normalises each tile against its own maximum, which is the input the
    statistics merge exists to combine and what a fused kernel can report.
    """

    scores = _tile_scores(q_b, k_t, bias_b, mask_t, precision)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    block_output, block_normalizer = _tile_terms_from_scores(
        scores,
        v_t,
        maximum,
        precision,
    )
    return block_output, maximum, block_normalizer


#: The tokamax attention implementation the option runs. One name, never a
#: sequence: `tokamax.dot_product_attention` takes a list and uses the first
#: implementation that does not raise `NotImplementedError`, which is a silent
#: fallback chain -- exactly what this option must not have. Pinning the
#: single object means an unsupported shape raises instead of quietly
#: measuring XLA under a fused label.
RING_TOKAMAX_IMPLEMENTATION = "triton"


def tile_attention_tokamax(
    q_b: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
    implementation: str = RING_TOKAMAX_IMPLEMENTATION,
    logits_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The same contract from tokamax's fused Triton attention.

    ``normalize_output=False`` leaves the numerator unnormalised and
    ``return_residuals=True`` returns ``(maximum, normalizer)``, which together
    are exactly this contract -- so the tile never materialises its
    ``[rows, heads, queries, keys]`` score tensor, which is the whole reason
    the option exists. The public ``tokamax.dot_product_attention`` returns
    only the output, so the implementation object is addressed directly and
    pinned to one name: a sequence would be a silent fallback chain.

    Two operand conversions, both of which keep the bias broadcast-free:

    * ``mask_t`` is the ring's additive mask bias, broadcast over heads and
      queries. It becomes tokamax's boolean key mask. Adding its finite part
      to ``bias_b`` instead would broadcast the two against each other and
      materialise the tile this path exists to avoid.
    * a tile with no valid key is forced back to ``(-inf, 0, 0)``. Tokamax
      masks with ``finfo.min`` rather than ``-inf`` (an all-``-inf`` row makes
      its stable softmax evaluate ``exp(nan)``), so such a row comes back as a
      finite maximum and a nonzero denominator over keys that are all absent.

    ``implementation`` exists so the operand conversions above -- the layout
    swaps, the mask, the residual reshape and the empty-row forcing, which is
    where a wiring bug would live -- can be executed by a CPU gate against
    tokamax's own XLA implementation. Both implementations go through
    `base.DotProductAttention.__call__`, so they share this contract; only the
    Triton one is a request the option can make.

    ``logits_scale`` is `1.0` for the ring, whose callers divide the query by
    ``sqrt(channels)`` in `project` before the tile ever sees it. The Boltz-2
    diffusion token site (`_cp_atom.pair_bias_attention_2d`) does not: it
    scales the logits, which is where tokamax applies this factor --
    `softmax(logits_scale * q @ k.T + bias) @ v`, so the scaled product and
    the bias meet in the same place the port's own einsum puts them.
    """

    if not tokamax_available():
        raise RuntimeError(
            "triangle_attention_ring_kernel='tokamax' needs the tokamax "
            "package, which did not import in this process"
        )
    from absl import flags
    from tokamax._src.ops.attention.api import IMPLEMENTATIONS

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["foldjax.models._cp_attention"], known_only=True)

    # f32 before the call, not after. Both tokamax implementations end their
    # forward with `out.astype(q.dtype)`, so bf16 operands -- which is what
    # the released `compute_dtype` gives this ring -- would round the
    # *unnormalised* numerator to bf16 before the merge ever sees it, and a
    # cast on the way out cannot undo that. The XLA tile already casts q, k
    # and v to f32 inside `_tile_scores`, so this is the same arithmetic and
    # the two bodies stay comparable. Running the kernel on bf16 operands for
    # the tensor-core speed the serial fused path enjoys is a separate
    # decision, and it needs the merge's error measured against this first.
    query = jnp.swapaxes(q_b.astype(jnp.float32), -3, -2)
    key = jnp.swapaxes(k_t.astype(jnp.float32), -3, -2)
    value = jnp.swapaxes(v_t.astype(jnp.float32), -3, -2)
    keys_valid = mask_t >= 0.0
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"tokamax has no {implementation!r} attention implementation in "
            f"this process; it registered {sorted(IMPLEMENTATIONS)}"
        )
    output, (maximum, normalizer) = IMPLEMENTATIONS[implementation](
        query,
        key,
        value,
        bias=bias_b,
        mask=keys_valid,
        # Never `AUTO`, which would apply `1/sqrt(channels)` on top of whatever
        # the caller already did: the ring's callers divide the query in
        # `project` before the tile sees it, and the token site passes its own
        # scale above.
        logits_scale=logits_scale,
        logits_dtype=jnp.float32,
        precision=precision,
        normalize_output=False,
        return_residuals=True,
    )
    # Residuals come back as [*B, H, T]; the ring carries [*B, H, T, 1].
    output = jnp.swapaxes(output, -3, -2).astype(jnp.float32)
    maximum = maximum[..., None].astype(jnp.float32)
    normalizer = normalizer[..., None].astype(jnp.float32)
    tile_valid = jnp.any(keys_valid, axis=-1, keepdims=True)
    empty_maximum = jnp.full_like(maximum, -jnp.inf)
    maximum = jnp.where(tile_valid, maximum, empty_maximum)
    normalizer = jnp.where(tile_valid, normalizer, jnp.zeros_like(normalizer))
    output = jnp.where(tile_valid, output, jnp.zeros_like(output))
    return output, maximum, normalizer


def _resolve_tile_attention(
    tile_kernel: str,
    precision: jax.lax.Precision | None,
) -> TileAttention:
    if tile_kernel not in RING_TILE_BODIES:
        raise ValueError(
            f"ring tile kernel must be one of {RING_TILE_BODIES}, "
            f"got {tile_kernel!r}"
        )
    if tile_kernel == "tokamax":
        return lambda *tile: tile_attention_tokamax(*tile, precision=precision)
    return lambda *tile: tile_attention_xla(*tile, precision=precision)


def _finish_block(
    output: jax.Array,
    normalizer: jax.Array,
    *,
    dtype: Any,
    gate_b: jax.Array | None,
) -> jax.Array:
    """Normalise one row block's accumulator and apply its gate.

    A query row with no valid key anywhere in the ring reaches here with a
    zero denominator, and leaves as zeros rather than as a division: the
    online recurrence is defined for an empty row and this is what it defines
    it to be.
    """

    tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
    result = jnp.where(
        normalizer > 0,
        output / jnp.maximum(normalizer, tiny),
        jnp.zeros_like(output),
    )
    result = result.astype(dtype)
    if gate_b is not None:
        # Same order the callers applied it in outside the ring: the fp32
        # accumulator is rounded to the value dtype first and only then
        # gated, and an elementwise product commutes with the caller's
        # transpose back to [..., rows, keys, heads, channels].
        result = result * gate_b
    return result


def _ring_local_rows(
    *,
    tiles: Callable[
        [int | jax.Array | None, int],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array | None],
    ],
    mask_l: jax.Array,
    bias_l: jax.Array,
    rows: int,
    block: int,
    side: int,
    heads: int,
    precision: jax.lax.Precision | None,
    tile_kernel: str = "xla",
) -> jax.Array:
    """Run the ring on one ``shard_map`` shard, a row block at a time.

    ``tiles(start, size)`` returns that block's ``q``, ``k``, ``v`` and gate --
    projected inside the block by the pair-representation entry point, sliced
    from already-projected tensors by the q/k/v one.

    ``tile_kernel`` picks the block body, and the two are different programs
    rather than two spellings of one:

    ``xla``
        the shipped two-pass ring. One global row maximum is fixed over a full
        rotation before any exponential is accumulated, and the second
        rotation combines tile terms against it. This is the default and its
        arithmetic is what every 2-D measurement in this repository describes.
    ``tokamax``
        one pass of :data:`TileAttention` per key tile, merged by
        :func:`merge_softmax_statistics`. Half the rotations and no score
        tensor, at the cost of the repeated rescaling the two-pass ring was
        written to remove. Experimental and GPU-only; see
        :func:`resolve_ring_tile_kernel`.
    ``xla_merge``
        the same merge ring with the portable tile. See
        :data:`RING_TILE_BODIES`.
    """

    if tile_kernel not in RING_TILE_BODIES:
        raise ValueError(
            f"ring tile kernel must be one of {RING_TILE_BODIES}, "
            f"got {tile_kernel!r}"
        )

    bias_init0 = triangle_bias_stage0_perm(side)
    diagonal_init = triangle_bias_stage1_perm(side)
    kv_hop = triangle_kv_ring_perm(side)
    bias_hop = triangle_bias_ring_perm(side)

    # The two initial bias skews do not depend on the row block, so they run
    # once above the loop; the ring hops inside it then start every block from
    # the same tile ownership.
    bias_l = permute(bias_l, bias_init0)
    bias_l = permute(bias_l, diagonal_init)

    def one_block(start: int | jax.Array | None, size: int) -> jax.Array:
        q_b, k_b, v_b, gate_b = tiles(start, size)
        if q_b.shape[-3] != heads or q_b.shape[-4] != size:
            raise ValueError(
                "ring row block expects [..., rows, heads, keys, channels]; "
                f"got {q_b.shape} for {size} rows and {heads} heads"
            )
        mask_b = _rows_of(mask_l, start, size, -4)
        # `diagonal_init` and `kv_hop` both keep the grid row, so a device owns
        # the same pair rows for the whole ring: slicing them before the skew
        # sends the same bytes in smaller messages and lands on the same tiles.
        k_t = permute(k_b, diagonal_init)
        v_t = permute(v_b, diagonal_init)
        mask_t = permute(mask_b, diagonal_init)
        bias_t = bias_l

        # Pass 1 fixes one global row maximum before any exponentials are
        # accumulated. Rotating K/bias/mask through a complete cycle returns
        # every tile to its initial owner, so pass 2 needs no saved full-width
        # operand and the per-device memory remains O(N^2 / P).
        maximum = jnp.full(
            q_b.shape[:-1] + (1,),
            -jnp.inf,
            dtype=jnp.float32,
        )
        for _ in range(side):
            scores = _tile_scores(q_b, k_t, bias_t, mask_t, precision)
            maximum = jnp.maximum(
                maximum,
                jnp.max(scores, axis=-1, keepdims=True),
            )
            # A full cycle restores the initial tile ownership. V is not used
            # in the max pass and therefore stays at its initial owner.
            k_t = permute(k_t, kv_hop)
            mask_t = permute(mask_t, kv_hop)
            bias_t = permute(bias_t, bias_hop)

        normalizer = jnp.zeros_like(maximum)
        normalizer_correction = jnp.zeros_like(maximum)
        output = jnp.zeros(q_b.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)

        # Pass 2 evaluates every tile against the same maximum and combines
        # tile-local numerator/denominator terms with Neumaier compensation.
        # This removes the repeated online-rescaling error that becomes visible
        # after a 3x3 Pairformer stack while preserving the gather-free ring.
        for step in range(side):
            block_output, block_normalizer = _tile_terms(
                q_b,
                k_t,
                v_t,
                bias_t,
                mask_t,
                maximum,
                precision,
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
                k_t = permute(k_t, kv_hop)
                v_t = permute(v_t, kv_hop)
                mask_t = permute(mask_t, kv_hop)
                bias_t = permute(bias_t, bias_hop)

        return _finish_block(
            output + output_correction,
            normalizer + normalizer_correction,
            dtype=v_b.dtype,
            gate_b=gate_b,
        )

    def fused_block(start: int | jax.Array | None, size: int) -> jax.Array:
        """One row block through a single rotation of fused tiles.

        The tokamax arm. A fused kernel normalises its tile against the tile's
        own maximum, so there is nothing for a first pass to fix and V has to
        travel with K from the start -- one rotation instead of two, and the
        merge carries the rescaling the two-pass body avoided.
        """

        tile_attention = _resolve_tile_attention(tile_kernel, precision)
        q_b, k_b, v_b, gate_b = tiles(start, size)
        if q_b.shape[-3] != heads or q_b.shape[-4] != size:
            raise ValueError(
                "ring row block expects [..., rows, heads, keys, channels]; "
                f"got {q_b.shape} for {size} rows and {heads} heads"
            )
        mask_b = _rows_of(mask_l, start, size, -4)
        k_t = permute(k_b, diagonal_init)
        v_t = permute(v_b, diagonal_init)
        mask_t = permute(mask_b, diagonal_init)
        bias_t = bias_l

        output = jnp.zeros(q_b.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)
        normalizer = jnp.zeros(q_b.shape[:-1] + (1,), dtype=jnp.float32)
        normalizer_correction = jnp.zeros_like(normalizer)
        # Starting at -inf rather than at the first tile's statistics so that
        # the merge is the same expression on every step: `_softmax_rescale`
        # reads a -inf source as "the accumulator holds nothing yet", which is
        # the same thing it reads an all-masked tile as.
        maximum = jnp.full_like(normalizer, -jnp.inf)

        for step in range(side):
            block_output, block_maximum, block_normalizer = tile_attention(
                q_b,
                k_t,
                v_t,
                bias_t,
                mask_t,
            )
            (
                output,
                output_correction,
                normalizer,
                normalizer_correction,
                maximum,
            ) = merge_softmax_statistics(
                output,
                output_correction,
                normalizer,
                normalizer_correction,
                maximum,
                block_output,
                block_maximum,
                block_normalizer,
            )
            if step + 1 < side:
                k_t = permute(k_t, kv_hop)
                v_t = permute(v_t, kv_hop)
                mask_t = permute(mask_t, kv_hop)
                bias_t = permute(bias_t, bias_hop)

        return _finish_block(
            output + output_correction,
            normalizer + normalizer_correction,
            dtype=v_b.dtype,
            gate_b=gate_b,
        )

    # The default keeps the two-pass body, so `xla` compiles the program the
    # ring compiled before this option existed.
    block_body = fused_block if tile_kernel != "xla" else one_block

    if block >= rows:
        return block_body(None, rows)

    full_blocks, remainder = divmod(rows, block)
    # A ragged tail is its own static block, not a padded-and-masked axis: each
    # pair row is independent, so there is nothing a mask would protect, and
    # padding the row axis would copy the pair tile to compute rows that are
    # then discarded.
    peel_start, peel_size = (
        (full_blocks * block, remainder) if remainder else (0, block)
    )
    scan_starts = jnp.arange(full_blocks, dtype=jnp.int32) * block
    if not remainder:
        scan_starts = scan_starts[1:]

    # One block is peeled off ahead of the scan -- the ragged one where there
    # is one -- for two reasons, either of which would be enough: a checked
    # `shard_map` scan refuses an initial carry that does not already carry
    # this mesh's varying-ness, which a freshly zeroed destination does not
    # (the fused body's `shard_map` is unchecked -- `_ring_shard_map_options`
    # -- so there only the second reason holds); and the peeled block makes
    # the loop's carry depend on a real tile, so XLA retires that tile before
    # the loop instead of scheduling the two side by side.
    peeled = block_body(peel_start, peel_size)
    destination = jnp.zeros(
        peeled.shape[:-4] + (rows,) + peeled.shape[-3:],
        dtype=peeled.dtype,
    )
    destination = jax.lax.dynamic_update_slice_in_dim(
        destination,
        peeled,
        peel_start,
        axis=-4,
    )

    def write_block(current: jax.Array, start: jax.Array):
        return (
            jax.lax.dynamic_update_slice_in_dim(
                current,
                block_body(start, block),
                start,
                axis=-4,
            ),
            None,
        )

    # A `lax.scan` rather than a Python loop because an unrolled loop only
    # hints: the blocks are independent, so XLA schedules them together and
    # keeps every block's tile live at once. Measured on one GPU, Boltz-2 at
    # 2,096 tokens on a 2x2 mesh then asked for 68.18 GiB against the
    # unblocked path's 56.78 GiB -- blocking made it worse. A scan holds one
    # block at a time by construction; only the output destination persists.
    destination, _ = jax.lax.scan(write_block, destination, scan_starts)
    return destination


def _check_ring_mesh() -> int:
    if cp_layout() != "2d":
        raise RuntimeError("ring_triangle_attention_2d requires an active 2-D CP mesh")
    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("context-parallel mesh is not active")
    side_row, side_col = cp_grid()
    if side_row != side_col:
        raise ValueError(f"triangle ring requires a square mesh, got {cp_grid()}")
    return side_row


def _check_ring_biases(
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    rows: int,
    tokens: int,
    ndim: int,
) -> int:
    if triangle_bias.ndim != ndim or mask_bias.ndim != ndim:
        raise ValueError(
            "triangle_bias and mask_bias must have the same rank as Q/K/V; "
            f"got {triangle_bias.ndim}, {mask_bias.ndim}, {ndim}"
        )
    if triangle_bias.shape[-2:] != (tokens, tokens):
        raise ValueError(
            "triangle-bias query/key axes do not match Q/K tokens: "
            f"{triangle_bias.shape[-2:]} vs {(tokens, tokens)}"
        )
    if mask_bias.shape[-4] != rows or mask_bias.shape[-1] != tokens:
        raise ValueError(
            "mask-bias outer/key axes do not match Q/K: "
            f"shape={mask_bias.shape}, outer={rows}, tokens={tokens}"
        )
    return triangle_bias.shape[-3]


def _pad_ring_biases(
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    pad_rows: int,
    pad_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    triangle_bias = _widen(
        triangle_bias,
        ((-2, pad_tokens), (-1, pad_tokens)),
    )
    mask_bias = _widen(mask_bias, ((-4, pad_rows),))
    if pad_tokens:
        # Padded keys are absent, not merely very unlikely. Using -inf is
        # safe because the online recurrence explicitly handles a completely
        # empty tile and a globally empty query row.
        mask_bias = jnp.concatenate(
            [
                mask_bias,
                jnp.full(
                    mask_bias.shape[:-1] + (pad_tokens,),
                    -jnp.inf,
                    dtype=mask_bias.dtype,
                ),
            ],
            axis=-1,
        )
    return triangle_bias, mask_bias


def _unpad_ring_output(
    out: jax.Array,
    *,
    mesh: Any,
    spec: PartitionSpec,
    rows: int,
    tokens: int,
    pad_rows: int,
    pad_tokens: int,
) -> jax.Array:
    if pad_rows:
        out = jax.lax.slice_in_dim(out, 0, rows, axis=-4)
    if pad_tokens:
        out = jax.lax.slice_in_dim(out, 0, tokens, axis=-2)
    if pad_rows or pad_tokens:
        out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, spec))
    return out


def _ring_shard_map_options(tile_kernel: str) -> dict[str, bool]:
    """``shard_map`` keywords the fused ring body needs and the shipped one does not.

    The fused tile is a Pallas kernel, and a Pallas kernel declares its outputs
    as :class:`jax.ShapeDtypeStruct` with no ``manual_axis_type``, which
    ``check_vma=True`` requires of every output produced inside a ``shard_map``
    (``jax/_src/pallas/core.py``, ``_convert_out_shape_to_aval``); the kernel
    raises before it computes anything. The check is bookkeeping, not
    arithmetic: what each device holds is the same either way, and the ring's
    output varies over both mesh axes, which its ``out_specs`` already says.
    Skipping it also relaxes the initial-carry rule the peeled block in
    :func:`_ring_local_rows` names, which is not that block's only reason.

    The default passes no keyword at all, so ``xla`` calls ``shard_map`` the
    way the ring called it before the option existed.
    """

    return {} if tile_kernel == "xla" else {"check_vma": False}


def ring_triangle_attention_2d(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
    q_block: int | None = None,
    tile_kernel: str = "xla",
) -> jax.Array:
    """Run exact gather-free triangle attention on a square two-dimensional mesh.

    Semantic layouts::

        query/key/value  [..., outer, heads, token, channels]
        triangle_bias   [..., 1, heads, query_token, key_token]
        mask_bias       [..., outer, 1, 1, key_token]

    ``q_block`` sets the local pair rows -- the ``outer`` axis -- one ring
    block runs at a time; see :func:`resolve_ring_row_block` for the rule and
    the knob's contract. Blocking leaves the communication schedule untouched
    for Q/K/V/mask (the same bytes in more, smaller rotations, because every
    skew and hop keeps a device's grid row) and repeats the bias rotations once
    per block. Each query row still reduces over the same whole local key axis
    in the same order, so the arithmetic is unchanged. Floating-point results
    can still move by the last bits, because XLA picks its contraction
    schedule from the operand extents.

    Q/K/V arrive projected here, so a block bounds the score tile and the
    accumulators but not the projections.
    :func:`ring_triangle_attention_2d_from_pair` projects inside the block and
    bounds those too.

    ``tile_kernel`` selects what one ring step evaluates its tile with; see
    :func:`_ring_local_rows`. The default is the shipped two-pass ring.
    """

    side = _check_ring_mesh()
    mesh = cp_mesh()

    if query.ndim < 4:
        raise ValueError(
            "triangle ring expects [..., outer, heads, token, channels], "
            f"got shape {query.shape}"
        )
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key and value must have identical global shapes; got "
            f"{query.shape}, {key.shape}, {value.shape}"
        )

    outer = query.shape[-4]
    tokens = query.shape[-2]
    heads = query.shape[-3]
    _check_ring_biases(
        triangle_bias,
        mask_bias,
        rows=outer,
        tokens=tokens,
        ndim=query.ndim,
    )

    pad_outer = fold_cp_pad_width(outer)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_outer or pad_tokens:
        query = _widen(query, ((-4, pad_outer), (-2, pad_tokens)))
        key = _widen(key, ((-4, pad_outer), (-2, pad_tokens)))
        value = _widen(value, ((-4, pad_outer), (-2, pad_tokens)))
        triangle_bias, mask_bias = _pad_ring_biases(
            triangle_bias,
            mask_bias,
            pad_rows=pad_outer,
            pad_tokens=pad_tokens,
        )

    qkv_spec = _two_axis_spec(query.ndim, -4, -2)
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)

    local_rows = (outer + pad_outer) // side
    local_keys = (tokens + pad_tokens) // side
    block = resolve_ring_row_block(
        local_rows,
        heads=heads,
        local_keys=local_keys,
        requested=q_block,
    )

    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        def tiles(start, size):
            return (
                _rows_of(q_l, start, size, -4),
                _rows_of(k_l, start, size, -4),
                _rows_of(v_l, start, size, -4),
                None,
            )

        return _ring_local_rows(
            tiles=tiles,
            mask_l=mask_l,
            bias_l=bias_l,
            rows=local_rows,
            block=block,
            side=side,
            heads=heads,
            precision=precision,
            tile_kernel=tile_kernel,
        )

    out = jax.shard_map(
        local_ring,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec),
        out_specs=qkv_spec,
        **_ring_shard_map_options(tile_kernel),
    )(query, key, value, triangle_bias, mask_bias)
    return _unpad_ring_output(
        out,
        mesh=mesh,
        spec=qkv_spec,
        rows=outer,
        tokens=tokens,
        pad_rows=pad_outer,
        pad_tokens=pad_tokens,
    )


def ring_triangle_attention_2d_from_pair(
    pair: Any,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    params: Any,
    *,
    project: Callable[[Any, Any], tuple[jax.Array, jax.Array, jax.Array, Any]],
    precision: jax.lax.Precision | None = None,
    q_block: int | None = None,
    tile_kernel: str = "xla",
) -> jax.Array:
    """Project Q/K/V one row block at a time and run the ring on each block.

    ``pair`` is the pre-projection pair representation -- one array, or a
    pytree of them sharing the ``[..., rows, keys, channels]`` geometry (the
    two operands a Q/KV-split caller holds) -- and ``project(params, rows)``
    returns that block's ``query``, ``key``, ``value`` and gate (or ``None``)
    in the ring's ``[..., rows, heads, keys, channels]`` layout, with any
    query scale already applied.

    Projecting inside the block is what keeps the ring's live set from scaling
    with the square of the local width: ``q``, ``k``, ``v`` and the gate are a
    block wide, and only the output destination spans the local rows. A linear
    contracts over the channel axis, so slicing the rows and then projecting is
    the same arithmetic as projecting and then slicing, and each row is still
    projected exactly once.

    ``tile_kernel`` selects what one ring step evaluates its tile with; see
    :func:`_ring_local_rows`. The default is the shipped two-pass ring.
    """

    side = _check_ring_mesh()
    mesh = cp_mesh()

    leaves = jax.tree.leaves(pair)
    if not leaves:
        raise ValueError("pair representation has no arrays")
    ndim = leaves[0].ndim
    if ndim < 3:
        raise ValueError(
            "pair representation expects [..., rows, keys, channels], "
            f"got shape {leaves[0].shape}"
        )
    rows = leaves[0].shape[-3]
    tokens = leaves[0].shape[-2]
    for leaf in leaves[1:]:
        if leaf.ndim != ndim or leaf.shape[-3:-1] != (rows, tokens):
            raise ValueError(
                "pair operands must share the row and key axes; got "
                f"{leaf.shape} against {leaves[0].shape}"
            )
    heads = _check_ring_biases(
        triangle_bias,
        mask_bias,
        rows=rows,
        tokens=tokens,
        ndim=ndim + 1,
    )

    pad_rows = fold_cp_pad_width(rows)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_rows or pad_tokens:
        pair = jax.tree.map(
            lambda leaf: _widen(leaf, ((-3, pad_rows), (-2, pad_tokens))),
            pair,
        )
        triangle_bias, mask_bias = _pad_ring_biases(
            triangle_bias,
            mask_bias,
            pad_rows=pad_rows,
            pad_tokens=pad_tokens,
        )

    pair_specs = jax.tree.map(
        lambda leaf: _two_axis_spec(leaf.ndim, -3, -2),
        pair,
    )
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)
    out_spec = _two_axis_spec(ndim + 1, -4, -2)

    local_rows = (rows + pad_rows) // side
    local_keys = (tokens + pad_tokens) // side
    block = resolve_ring_row_block(
        local_rows,
        heads=heads,
        local_keys=local_keys,
        requested=q_block,
    )

    def local_ring(pair_l, bias_l, mask_l, params_l):
        def tiles(start, size):
            rows_l = jax.tree.map(
                lambda leaf: _rows_of(leaf, start, size, -3),
                pair_l,
            )
            projected = project(params_l, rows_l)
            if len(projected) != 4:
                raise ValueError(
                    "project must return (query, key, value, gate); got "
                    f"{len(projected)} values"
                )
            return projected

        return _ring_local_rows(
            tiles=tiles,
            mask_l=mask_l,
            bias_l=bias_l,
            rows=local_rows,
            block=block,
            side=side,
            heads=heads,
            precision=precision,
            tile_kernel=tile_kernel,
        )

    out = jax.shard_map(
        local_ring,
        mesh=mesh,
        # Weights are replicated, like the 1-D path's `params` operand: the
        # projection contracts over the channel axis, which no shard splits.
        in_specs=(pair_specs, bias_spec, mask_spec, PartitionSpec()),
        out_specs=out_spec,
        **_ring_shard_map_options(tile_kernel),
    )(pair, triangle_bias, mask_bias, params)
    return _unpad_ring_output(
        out,
        mesh=mesh,
        spec=out_spec,
        rows=rows,
        tokens=tokens,
        pad_rows=pad_rows,
        pad_tokens=pad_tokens,
    )


# --- the gather path ---------------------------------------------------------
#
# Device (r, c) holds pair rows ``I_r`` x columns ``I_c`` and writes that tile
# of the output. Starting-node attention reads, for its pair row ``i`` and
# query column ``j``, every key ``k``:
#
#     out[i, j] = sum_k softmax_k(q[i, j] . k[i, k] + b[j, k] + m[i, k]) v[i, k]
#
# so what the device needs beyond its own tile is ``k``/``v``/``m`` over the
# FULL key axis for its own rows -- the row block's pair rows gathered along
# ``cp_col`` -- and the bias rows ``b[I_c, :]``, which no device on grid row r
# holds. The ring moved the same information a tile at a time; here each
# device receives it once and the attention is one normalising call.


def gather_triangle_bias_rows(bias_l: jax.Array, side: int) -> jax.Array:
    """Inside ``shard_map``: the local ``b[I_r, I_c]`` becomes ``b[I_c, :]``.

    A redistribution of ownership, not a transpose of the tensor's axes:
    the transpose partner ``(c, r)`` holds ``b[I_c, I_r]`` in the same axis
    order, so one ``collective_permute`` ``(c, r) -> (r, c)`` hands it over
    (:func:`~foldjax.models._cp.transpose_perm`, whose diagonal pairs
    ``(r, r) -> (r, r)`` are listed explicitly), and the ``all_gather`` along
    ``cp_row`` concatenates the key axis in mesh order ``I_0, ..., I_{s-1}``.

    Permute first and gather second, rather than the other way round, sends a
    ``1/side`` message through the transpose, which is the exchange that
    crosses the grid; it runs once per call, above the row loop.
    """

    exchanged = permute(bias_l, transpose_perm(side))
    return jax.lax.all_gather(
        exchanged,
        CP_ROW_AXIS,
        axis=exchanged.ndim - 1,
        tiled=True,
    )


def _row_block(
    array: jax.Array,
    start: int | jax.Array | None,
    size: int,
    axis: int,
    rows: int,
) -> jax.Array:
    """Local rows ``[start, start + size)``, zero-padded explicitly past ``rows``.

    A tail block is padded rather than read through ``dynamic_slice``, which
    clamps an out-of-range start and would silently hand the tail the rows
    of the block before it. Only the tail is padded, never the whole local
    tile: that copy is the size of the pair tile (2.6 GiB per device at 6,568
    tokens on 2x2). ``None`` is the single-block case and takes no slice.
    """

    if start is None:
        return array
    axis = _resolve_axis(axis, array.ndim, name="row axis")
    if isinstance(start, int) and start + size > rows:
        piece = jax.lax.slice_in_dim(array, start, rows, axis=axis)
        return _widen(piece, ((axis, start + size - rows),))
    return jax.lax.dynamic_slice_in_dim(array, start, size, axis=axis)


def _gather_keys(array: jax.Array, axis: int, tokens: int) -> jax.Array:
    """All-gather the key axis along ``cp_col`` and drop the grid padding.

    The padding sits at the global end, because the pair was widened before
    it was sharded, so ``[:tokens]`` removes exactly the keys ``k >= N``:
    they are absent from the attention rather than masked in it. That keeps a
    genuinely fully-masked row on the serial contract with either body -- the
    XLA body adds ``-1e9`` to every key and the kernel replaces every logit
    with ``-1e9``, and both then average over the ``N`` keys serial averages
    over, not over the padded extent -- and it hands the kernel the key
    extent the serial call hands it.
    """

    axis = _resolve_axis(axis, array.ndim, name="key axis")
    gathered = jax.lax.all_gather(array, CP_COL_AXIS, axis=axis, tiled=True)
    if gathered.shape[axis] == tokens:
        return gathered
    return jax.lax.slice_in_dim(gathered, 0, tokens, axis=axis)


def _projected(projected: Sequence[Any]) -> Sequence[Any]:
    if len(projected) != 4:
        raise ValueError(
            "project must return (query, key, value, gate); got "
            f"{len(projected)} values"
        )
    return projected


def gather_triangle_block_operands(
    pair_l: Any,
    mask_l: jax.Array,
    params_l: Any,
    start: int | jax.Array | None,
    *,
    size: int,
    rows: int,
    tokens: int,
    project: Callable[[Any, Any], Sequence[Any]],
) -> tuple[jax.Array, jax.Array, jax.Array, Any, jax.Array]:
    """Inside ``shard_map``: one row block's ``q, k, v, gate, mask``.

    ``q`` and the gate are projected from the local tile, ``[..., R, H, L, D]``;
    ``k`` and ``v`` from the block's pair rows gathered to full width along
    ``cp_col``, ``[..., R, H, N, D]``; the additive key mask likewise,
    ``[..., R, 1, 1, N]``. ``project`` is called twice, and each call's unused
    half -- the full-width Q/gate, the local K/V -- is dead code XLA removes,
    which the gather gate checks by counting dots rather than assuming.

    Raw pair rows travel rather than projected K/V: ``C`` channels against
    ``2 * H * D``, twice the bytes for Boltz-2's ``C = H * D = 128``.
    """

    rows_l = jax.tree.map(
        lambda leaf: _row_block(leaf, start, size, leaf.ndim - 3, rows),
        pair_l,
    )
    query, _, _, gate = _projected(project(params_l, rows_l))
    wide = jax.tree.map(
        lambda leaf: _gather_keys(leaf, leaf.ndim - 2, tokens),
        rows_l,
    )
    _, key, value, _ = _projected(project(params_l, wide))
    mask_b = _row_block(mask_l, start, size, mask_l.ndim - 4, rows)
    mask_b = _gather_keys(mask_b, mask_b.ndim - 1, tokens)
    return query, key, value, gate, mask_b


def gather_attention_xla(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array,
    mask: jax.Array,
    *,
    scale: float | None,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    """The gather path's reference body: serial XLA triangle attention's arithmetic.

    The same order as Boltz-2's serial ``_attention_block`` -- f32 scores, the
    additive mask, then the bias, one softmax, the value product rounded to
    the value dtype -- on a rectangular ``[L queries, N keys]`` problem.
    ``scale`` multiplies the f32 query when the caller's ``project`` left it
    unscaled; ``None`` means ``project`` already divided it, the ring's
    contract.
    """

    q32 = query.astype(jnp.float32)
    if scale is not None:
        q32 = q32 * jnp.asarray(scale, dtype=jnp.float32)
    scores = jnp.matmul(
        q32,
        jnp.swapaxes(key.astype(jnp.float32), -1, -2),
        precision=precision,
    )
    scores = scores + mask.astype(jnp.float32) + bias.astype(jnp.float32)
    probabilities = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(
        probabilities,
        value.astype(jnp.float32),
        precision=precision,
    ).astype(value.dtype)


def gather_attention_cueq(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array,
    mask: jax.Array,
    *,
    scale: float | None,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    """The gather path's GPU body: one cuEquivariance triangle attention.

    Rectangular: ``S_qo = L`` local query columns against ``S_kv = N`` keys,
    which the installed wrapper's ``[B, N, H, S_qo, D]`` / ``[B, N, H, S_kv,
    D]`` / bias ``[B, 1, H, S_qo, S_kv]`` contract permits. The additive mask
    is converted to the kernel's boolean (``True`` = valid) exactly once, in
    :func:`foldjax.models._cueq.cueq_attention_arguments`, the conversion the
    serial fused path already goes through. ``scale`` is applied inside the
    kernel, as the serial fused path applies it, so a caller that wants the
    released rounding passes an unscaled query and its scale here.
    """

    from foldjax.models._cueq import cueq_attention_core

    return cueq_attention_core(
        query,
        key,
        value,
        bias,
        mask,
        scale=1.0 if scale is None else float(scale),
        precision=precision,
    ).astype(value.dtype)


def _resolve_gather_attention(body: str) -> Callable[..., jax.Array]:
    if body not in GATHER_ATTENTION_BODIES:
        raise ValueError(
            f"gather attention body must be one of {GATHER_ATTENTION_BODIES}, "
            f"got {body!r}"
        )
    return gather_attention_cueq if body == "cueq" else gather_attention_xla


def _gather_local_rows(
    block_body: Callable[[int | jax.Array | None, int], jax.Array],
    *,
    rows: int,
    block: int,
) -> jax.Array:
    """Run ``block_body`` over the local pair rows, a block at a time.

    ``ceil(rows / block)`` calls of one width. The ragged tail, where there is
    one, is peeled ahead of the scan, padded explicitly inside
    :func:`_row_block` and sliced back to its real rows before it is written;
    every scanned block starts in range, so no ``dynamic_slice`` or
    ``dynamic_update_slice`` is ever clamped. The peel is also what the ring's
    loop peels for (:func:`_ring_local_rows`): a checked ``shard_map`` scan
    refuses a freshly zeroed carry, and a carry that depends on a real block
    retires that block before the loop.
    """

    if block >= rows:
        return block_body(None, rows)
    full_blocks, remainder = divmod(rows, block)
    if remainder:
        peel_start = full_blocks * block
        peeled = block_body(peel_start, block)
        peeled = jax.lax.slice_in_dim(peeled, 0, remainder, axis=peeled.ndim - 4)
        scan_starts = jnp.arange(full_blocks, dtype=jnp.int32) * block
    else:
        peel_start = 0
        peeled = block_body(0, block)
        scan_starts = jnp.arange(1, full_blocks, dtype=jnp.int32) * block
    destination = jnp.zeros(
        peeled.shape[:-4] + (rows,) + peeled.shape[-3:],
        dtype=peeled.dtype,
    )
    destination = jax.lax.dynamic_update_slice_in_dim(
        destination,
        peeled,
        peel_start,
        axis=destination.ndim - 4,
    )

    def write_block(current: jax.Array, start: jax.Array):
        return (
            jax.lax.dynamic_update_slice_in_dim(
                current,
                block_body(start, block),
                start,
                axis=current.ndim - 4,
            ),
            None,
        )

    destination, _ = jax.lax.scan(write_block, destination, scan_starts)
    return destination


def gather_triangle_attention_2d_from_pair(
    pair: Any,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    params: Any,
    *,
    project: Callable[[Any, Any], Sequence[Any]],
    scale: float | None = None,
    precision: jax.lax.Precision | None = None,
    q_block: int | None = None,
    body: str = "xla",
) -> jax.Array:
    """Triangle attention on the square grid by a streamed full-width gather.

    The external contract is :func:`ring_triangle_attention_2d_from_pair`'s:
    ``pair`` ``[..., rows, keys, C]`` sharded ``(-3 -> cp_row, -2 -> cp_col)``,
    ``triangle_bias`` ``[..., 1, H, tokens, tokens]`` ``(-2 -> cp_row, -1 ->
    cp_col)``, ``mask_bias`` ``[..., rows, 1, 1, tokens]`` additive ``(-4 ->
    cp_row, -1 -> cp_col)``, the output ``[..., rows, H, tokens, D]`` ``(-4 ->
    cp_row, -2 -> cp_col)``, and the same ``project(params, rows)``. Two
    differences: ``project`` may leave the query unscaled and pass ``scale``
    here instead, so the kernel scales it as the serial fused call does; and
    the row block bounds different buffers (below). The ending node is the
    caller's transposed problem, as for the ring.

    What moves, per call and device ``(r, c)``:

    * the bias rows ``b[I_c, :]`` once, above the row loop
      (:func:`gather_triangle_bias_rows`);
    * per row block of ``R`` local pair rows, those rows at full width and
      their mask, along ``cp_col`` (:func:`gather_triangle_block_operands`).

    Then one attention per block on ``q [.., R, H, L, D]``, ``k/v [.., R, H,
    N, D]``, ``bias [.., 1, H, L, N]``, ``mask [.., R, 1, 1, N]``, with
    ``L = P / side`` the local query columns and ``N`` the unpadded key count;
    no rotation, no statistics merge. ``body`` is ``xla``
    (:func:`gather_attention_xla`, the reference any host can run) or
    ``cueq`` (:func:`gather_attention_cueq`, GPU); only the ``cueq`` body's
    ``shard_map`` is unchecked, because the kernel's FFI outputs carry no
    varying-mesh-axis type (:func:`_ring_shard_map_options`).

    Grid padding is the ring's: both token axes are widened to ``P =
    side * ceil(N / side)`` before sharding, padded query columns and pair
    rows are computed on zeros and sliced off after, and padded keys are
    dropped from the gathered key axis (:func:`_gather_keys`).

    The row block is :func:`resolve_ring_row_block`'s with the local query
    width, so ``q_block`` means what it means to the ring. What it bounds is
    the gathered rows and K/V -- ``R x N`` per block -- and, for the XLA body
    only, an ``[R, H, L, N]`` score tile ``side`` times the ring's; the
    kernel forms none.
    """

    side = _check_ring_mesh()
    mesh = cp_mesh()
    attend = _resolve_gather_attention(body)

    leaves = jax.tree.leaves(pair)
    if not leaves:
        raise ValueError("pair representation has no arrays")
    ndim = leaves[0].ndim
    if ndim < 3:
        raise ValueError(
            "pair representation expects [..., rows, keys, channels], "
            f"got shape {leaves[0].shape}"
        )
    rows = leaves[0].shape[-3]
    tokens = leaves[0].shape[-2]
    for leaf in leaves[1:]:
        if leaf.ndim != ndim or leaf.shape[-3:-1] != (rows, tokens):
            raise ValueError(
                "pair operands must share the row and key axes; got "
                f"{leaf.shape} against {leaves[0].shape}"
            )
    heads = _check_ring_biases(
        triangle_bias,
        mask_bias,
        rows=rows,
        tokens=tokens,
        ndim=ndim + 1,
    )

    pad_rows = fold_cp_pad_width(rows)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_rows or pad_tokens:
        pair = jax.tree.map(
            lambda leaf: _widen(leaf, ((-3, pad_rows), (-2, pad_tokens))),
            pair,
        )
        triangle_bias, mask_bias = _pad_ring_biases(
            triangle_bias,
            mask_bias,
            pad_rows=pad_rows,
            pad_tokens=pad_tokens,
        )

    pair_specs = jax.tree.map(
        lambda leaf: _two_axis_spec(leaf.ndim, -3, -2),
        pair,
    )
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)
    out_spec = _two_axis_spec(ndim + 1, -4, -2)

    local_rows = (rows + pad_rows) // side
    local_cols = (tokens + pad_tokens) // side
    block = resolve_ring_row_block(
        local_rows,
        heads=heads,
        local_keys=local_cols,
        requested=q_block,
    )

    def local_gather(pair_l, bias_l, mask_l, params_l):
        bias_rows = gather_triangle_bias_rows(bias_l, side)
        if bias_rows.shape[-1] != tokens:
            bias_rows = jax.lax.slice_in_dim(
                bias_rows, 0, tokens, axis=bias_rows.ndim - 1
            )

        def block_body(start, size):
            query, key, value, gate, mask_b = gather_triangle_block_operands(
                pair_l,
                mask_l,
                params_l,
                start,
                size=size,
                rows=local_rows,
                tokens=tokens,
                project=project,
            )
            if query.shape[-3] != heads or query.shape[-4] != size:
                raise ValueError(
                    "gather row block expects [..., rows, heads, keys, "
                    f"channels]; got {query.shape} for {size} rows and "
                    f"{heads} heads"
                )
            out = attend(
                query,
                key,
                value,
                bias_rows,
                mask_b,
                scale=scale,
                precision=precision,
            )
            if gate is not None:
                # The serial order: the attention rounded to the value dtype,
                # then gated.
                out = out * gate
            return out

        return _gather_local_rows(block_body, rows=local_rows, block=block)

    out = jax.shard_map(
        local_gather,
        mesh=mesh,
        in_specs=(pair_specs, bias_spec, mask_spec, PartitionSpec()),
        out_specs=out_spec,
        **cp_fused_shard_map_options(body == "cueq"),
    )(pair, triangle_bias, mask_bias, params)
    return _unpad_ring_output(
        out,
        mesh=mesh,
        spec=out_spec,
        rows=rows,
        tokens=tokens,
        pad_rows=pad_rows,
        pad_tokens=pad_tokens,
    )
