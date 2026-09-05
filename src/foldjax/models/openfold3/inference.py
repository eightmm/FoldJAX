"""End-to-end inference: trunk -> diffusion -> confidence heads.

Joins the stages that are gated individually. Every dimension, head count and
sampler setting arrives through ``InferenceConfig`` rather than being hardcoded,
because those values come from the checkpoint. Guessing them would produce code
that runs and is not the released model.

What this does **not** do: featurization (upstream's data pipeline builds
``batch``) and output writing. Those are separate concerns from the array math.
"""

from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from foldjax._openfold3_compile import (
    canonical_cache_scope,
    inspect_cache_scope,
    observe_cache_scope,
    reset_cache_scope_tracking,
    resolve_triangle_kernel,
    triangle_backend,
)
from foldjax.execution import DIFFUSION_CHUNK_SIZE, auto_diffusion_chunk_size
from foldjax.models._cp import (
    context_parallel,
    replicate_tree,
    shard_pair_rows,
)
from foldjax.models._cp import (
    cp_shards as _active_cp_shards,
)
from foldjax.models.openfold3.data.compact_categories import (
    COMPACT_REF_ATOM_CATEGORIES_MARKER,
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
    validate_compact_ref_atom_categories,
)
from foldjax.models.openfold3.data.featurize import _MSA_CYCLE_INDICES
from foldjax.models.openfold3.models.augmentation import centre_random_augmentation
from foldjax.models.openfold3.models.confidence import (
    bin_centers,
    compute_chain_pair_iptm,
    compute_plddt,
    compute_ptm,
)
from foldjax.models.openfold3.models.denoiser import DenoiserParams, denoise
from foldjax.models.openfold3.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    pair_conditioning,
    single_conditioning,
)
from foldjax.models.openfold3.models.diffusion_schedule import noise_schedule
from foldjax.models.openfold3.models.frames import token_frame_atoms
from foldjax.models.openfold3.models.heads import (
    AtomHeadParams,
    PairformerEmbeddingParams,
    PairHeadParams,
    atom_logit_head,
    distogram_head,
    pairformer_embedding,
    predicted_aligned_error_head,
    predicted_distance_error_head,
)
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
    token_representative_atoms,
)
from foldjax.models.openfold3.models.sampler import sample_diffusion
from foldjax.models.openfold3.models.trunk import TrunkParams, trunk
from foldjax.models.openfold3.output import (
    DEFAULT_ARRAY_BUDGET_BYTES,
    plan_returned_pair_logits,
)


class InferenceConfig(NamedTuple):
    """Shapes and hyperparameters, all of which come from the checkpoint."""

    n_token: int
    n_atom: int
    n_query: int
    n_key: int
    atom_heads: int
    token_heads: int
    no_heads_msa: int
    no_heads_pair: int
    no_heads_pair_bias: int
    max_relative_idx: int
    max_relative_chain: int
    num_recycles: int
    num_samples: int
    max_atoms_per_token: int
    plddt_bins: int
    pae_bins: int
    pae_bin_max: float
    num_steps: int
    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: int = 7
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    opm_first: bool = True
    # The confidence heads' distance embedding; upstream's values are 3.25 to
    # 50.75 over 39 bins, carried forward from AF2.
    confidence_min_bin: float = 3.25
    confidence_max_bin: float = 50.75
    confidence_no_bin: int = 39
    experimentally_resolved_bins: int = 2
    # Rows of the pair representation to process at a time, upstream's
    # ``settings.memory.eval.chunk_size``. ``None`` processes them in one go, which is
    # fastest and peaks highest; it reaches both triangle attention and the pair
    # transition. Above ~800 tokens it decides whether the target runs at all, so
    # ``released_config`` picks it from the token count by default -- see
    # ``auto_pair_chunk_size`` and models/row_chunking.py.
    pair_chunk_size: int | None = None
    # Token count above which confidence re-embedding runs one diffusion sample
    # at a time. FoldJAX defaults this to zero after the released n=5 A/B showed
    # the same wall time at less than half the peak, so every multi-sample default
    # is serial. Positive values retain the numeric token threshold while the
    # released sample-width guard still bounds large direct requests; ``None``
    # explicitly keeps an always-batched graph for direct callers.
    per_sample_token_cutoff: int | None = 0
    #: Diffusion samples the rollout denoises at once. The denoiser holds its
    #: activations for every sample handed to it, so this is the model's entire
    #: growth along the sample axis: measured at 1,003 tokens, the peak went
    #: 9.0 GiB at one sample to 38.2 at thirty-two without it. It is the same
    #: name and the same default width Protenix resolves it at, because a
    #: knob that means the same thing should not have two spellings. The
    #: conditioning is widened at the point of use and the noise is narrowed
    #: rather than redrawn, so a chunk sees the numbers its samples would have
    #: seen anyway -- to float32 round-off, 9.5e-06 absolute, not to the bit.
    #: `None` denoises every sample at once.
    #:
    #: Resolved from the sample count by `released_config` rather than pinned
    #: here, so a one-sample run keeps the single unchunked rollout it had.
    diffusion_chunk_size: int | None = None
    # Alignment rows the trunk sees. Upstream subsamples inside the network, in
    # ``MSAModuleEmbedder.forward``, with ``subsample_all_msa=True`` and
    # ``min_subsampled_all_msa == max_subsampled_all_msa == 1024`` -- a draw from
    # ``randint(1024, 1025)``, so the count is fixed -- and with no ``self.training``
    # guard, so it applies at inference. Running the full alignment is therefore not
    # a more faithful choice than 1024; it is a different model. ``None`` keeps every
    # row, which is only useful for comparing against this divergence.
    msa_depth: int | None = 1024
    #: Which of the [.., N, N, bins] logits the program returns. Decided ahead
    #: of the run by `output.plan_returned_pair_logits` (released_config does
    #: this), so arrays the npz writer's budget would discard are never hauled
    #: out of the graph. The writer stays the final authority on what is
    #: written; this only controls what exists as an entry output.
    returned_pair_logits: tuple[str, ...] = (
        "pae_logits",
        "pde_logits",
        "distogram_logits",
    )
    #: Trunk representations to hand back, by the shared vocabulary in
    #: `foldjax.models._representations`. Empty means none, which is the
    #: default: these are the largest arrays the program produces.
    returned_representations: tuple[str, ...] = ()
    #: Stop once the representations exist, skipping the sampler and the
    #: confidence heads. The Prediction that comes back carries the trunk
    #: arrays and nothing else.
    stop_after_trunk: bool = False
    #: Context-parallel shard count. More than one shards the pair
    #: representations row-wise across that many devices (the JAX form of
    #: OpenDDE's Fold-CP) and requires the mesh :func:`compile_predict`
    #: activates. Part of the config so a mesh change is a new compilation,
    #: never a stale cache hit.
    cp_shards: int = 1
    #: Which context-parallel layout those shards form. ``"1d"`` splits pair
    #: rows only; ``"2d"`` is Fold-CP's square grid, which splits columns too
    #: and drops the per-device pair cost from ``O(N^2/P)`` with a full-width
    #: row of tiles to ``O(N^2/P)`` outright, at the price of a Cannon ring in
    #: the triangle multiplication. ``"auto"`` currently resolves to ``"1d"``
    #: whatever the shard count -- see :func:`resolve_cp_layout` for why the
    #: better layout is not yet the default. ``"2d"`` needs a square shard
    #: count, which is the only shape the ring schedules accept.
    cp_layout: str = "auto"
    #: Whether any active token needs the geometry-dependent atomized frame.
    #: This is derived from host features by production entry points. Keeping it
    #: static removes the sample-by-token-by-atom nearest-neighbour graph for
    #: ordinary protein, RNA and DNA inputs; True is the conservative direct-API
    #: default.
    has_atomized_tokens: bool = True


class InferenceParams(NamedTuple):
    """Every parameter group the inference path needs."""

    trunk: TrunkParams
    diffusion_conditioning: DiffusionConditioningParams
    denoiser: DenoiserParams
    pairformer_embedding: PairformerEmbeddingParams
    plddt_head: AtomHeadParams
    pae_head: PairHeadParams
    pde_head: PairHeadParams
    distogram_head: PairHeadParams
    experimentally_resolved_head: AtomHeadParams | None = None


class Prediction(NamedTuple):
    """Predicted coordinates and the confidence outputs derived from them."""

    coordinates: jnp.ndarray | None
    plddt: jnp.ndarray | None
    ptm: jnp.ndarray | None
    iptm: jnp.ndarray | None
    chain_pair_iptm: jnp.ndarray | None
    #: None when the config's ``returned_pair_logits`` excludes them: these are
    #: [num_samples, N, N, bins] entry outputs that stay resident for the whole
    #: run, and above `output.DEFAULT_ARRAY_BUDGET_BYTES` the npz writer drops
    #: them unwritten anyway. `write_arrays` already skips None fields.
    pae_logits: jnp.ndarray | None
    pde_logits: jnp.ndarray | None
    distogram_logits: jnp.ndarray | None
    experimentally_resolved_logits: jnp.ndarray | None = None
    #: The trunk's own outputs, present only when `returned_representations`
    #: asks for them. A pair representation is quadratic in token count and
    #: an entry output stays resident for the whole run, so they are off by
    #: default and written straight to disk when they are on.
    single_inputs: jnp.ndarray | None = None
    single: jnp.ndarray | None = None
    pair: jnp.ndarray | None = None


#: Bytes the triangle-attention score tensor is allowed to reach before
#: :func:`auto_pair_chunk_size` starts chunking. 8 GiB leaves room for the rest of
#: the block, the diffusion path and the weights on a 96 GiB card; it is a budget,
#: not a measured constant, and the caller can override it.
PAIR_SCORE_BUDGET_BYTES = 8 * 2**30

#: Triangle attention head count in the released architecture. Named because
#: ``auto_pair_chunk_size`` needs it before an ``InferenceConfig`` exists to read it
#: from; ``test_released_config`` checks the config's copy against upstream.
_RELEASED_PAIR_HEADS = 4
# Upstream's inference-only MSA subsampler selects this many rows. Raw FoldJAX
# preprocessing uses the same value before one-hot expansion; feature-archive
# generation remains uncapped unless its caller opts in explicitly.
RELEASED_MSA_DEPTH = 1024


def auto_pair_chunk_size(
    n_token: int,
    *,
    no_heads: int,
    budget_bytes: int = PAIR_SCORE_BUDGET_BYTES,
    dtype_bytes: int = 4,
) -> int | None:
    """Largest row chunk whose triangle-attention scores stay inside the budget.

    Triangle attention scores one pair against another for every row, so the score
    tensor is ``[rows, heads, N_token, N_token]`` -- cubic in token count. Measured
    on the released architecture at ``heads=4``, one pair block needs 27 GiB of
    temporaries at 966 tokens and **267 GiB** at 2076, essentially all of it that
    tensor. Chunking the rows caps it, exactly (see models/row_chunking.py), so the
    only question is how few rows to take, and taking fewer than necessary costs
    speed for nothing.

    This picks the largest chunk that keeps the tensor under ``budget_bytes``, which
    is what upstream's chunk-size tuner does by trial: it starts at 1024 and halves
    until the allocation succeeds. It then spreads the rows evenly over that many
    blocks, because a chunk that does not divide the token count is padded to a whole
    block and the padding is computed and thrown away: 575 rows of 966 needs two
    blocks either way, but 575 pads to 1150 and wastes a fifth of the work, while 483
    pads to nothing.

    Returns ``None`` -- meaning do not chunk, the fastest path -- when the whole
    thing already fits.

    An earlier version of this port recorded that chunking changed neither peak nor
    speed. That was a measurement error: peak was being read from the allocator's
    pool size, which had been preallocated and so was flat regardless. The knob
    works; the ruler was wrong.
    """
    if n_token <= 0:
        raise ValueError("n_token must be positive")
    per_row = no_heads * n_token * n_token * dtype_bytes
    if per_row <= 0:
        return None
    rows = budget_bytes // per_row
    if rows >= n_token:
        return None
    if rows < 1:
        # The budget cannot hold a single row. One row is the smallest chunk there
        # is, so take it: the alternative is refusing to run at all.
        return 1
    blocks = -(-n_token // int(rows))
    # Same block count, rows spread evenly, so the last block carries no padding
    # the earlier ones did not.
    return -(-n_token // blocks)


def _per_sample_confidence(config: InferenceConfig) -> bool:
    """Whether to run the confidence re-embedding one sample at a time.

    The default cutoff of zero serializes every multi-sample request: a real
    490-token, five-sample A/B measured 9,045.3 -> 4,263.5 MiB with unchanged
    wall time (33.25 -> 33.39 s), so retaining the released short-target batch
    no longer buys throughput and costs 4.7 GiB. Positive values remain token
    thresholds, with the existing safety guard still serializing requests above
    the released five-sample width. ``None`` remains the explicit always-batched
    escape hatch for direct callers.

    Measured on a real 966-token target with the released weights, the per-sample
    branch is not a trade: it is **bitwise identical** on coordinates, pTM and the PAE
    logits (pLDDT differs by 1.8e-7, one float32 bit) and **3.7x faster** -- 69.5 s
    against 255.1 s. Holding five pair representations at once costs more in memory
    traffic than it wins in parallelism at that size, so upstream's cutoff is a
    throughput setting as much as a memory one.

    Keeping the comparison here, rather than treating every integer as a boolean
    switch, preserves the public numeric cutoff contract for tuned direct calls
    without reviving the unbounded high-sample confidence graph.
    """
    cutoff = config.per_sample_token_cutoff
    return (
        cutoff is not None
        and config.num_samples > 1
        and (config.n_token > cutoff or config.num_samples > DIFFUSION_CHUNK_SIZE)
    )


def _sink_pae_metrics(config: InferenceConfig) -> bool:
    """Whether serial confidence may return PAE metrics instead of PAE logits."""
    return (
        _per_sample_confidence(config)
        and "pae_logits" not in config.returned_pair_logits
    )


def _expand_samples(value: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    """Give ``value`` a leading sample axis of ``num_samples``, without copying.

    ``broadcast_to`` rather than ``repeat``: for a leading axis of 1 the two produce
    the same value, but ``repeat`` materializes ``num_samples`` copies while a
    broadcast is a view XLA can fuse into whatever reads it. On pair-sized tensors
    the difference is gigabytes -- ``[5, 2076, 2076, 128]`` in float32 is 5.1 GiB.

    Anything without a leading axis of 1 is returned unchanged, so scalar features
    and already-expanded tensors pass through.
    """
    if jnp.ndim(value) == 0 or jnp.shape(value)[0] != 1:
        return value
    return jnp.broadcast_to(value, (num_samples, *jnp.shape(value)[1:]))


#: Matmul precision the port runs under, matching what upstream sets for itself:
#: ``torch.set_float32_matmul_precision("high")`` in
#: ``openfold3/entry_points/import_utils.py:33``. Torch's "high" is TF32 on
#: NVIDIA, and JAX spells the same thing "high".
#:
#: This was ``"highest"`` -- true float32 -- until 2026-08-10, on the reasoning
#: that TF32 "silently costs ~3 decimal digits per matmul". It does, and it costs
#: upstream the same, which is the part the reasoning missed: a port that is more
#: precise than the model it ports is not more faithful, it is running a
#: different configuration and paying for the privilege. Measured at 1,003 tokens
#: over the released schedule, TF32 against the float32 pin: **-15% warm** (96.9
#: -> 82.0 s), no change in peak memory, CA RMSD 0.011 A against a 0.005 A floor
#: from running the same program twice, confidence inside that same rerun spread
#: on both pLDDT and pTM, and TM 0.9937 against the deposited structure either
#: way.
#:
#: Deliberate ``Precision.HIGHEST`` pins elsewhere are unaffected; they exist
#: precisely because this is not HIGHEST.
_MATMUL_PRECISION = "high"


def openfold3_precision(function):
    """Run `function` with JAX's matmul precision pinned to upstream's.

    Pinning it at all is the point: the setting is process-global in JAX, so
    without a scope the port would inherit whatever another model left behind,
    and the parity gate runs on CPU where TF32 is invisible.

    This used to be a `jax.config.update` at import time, which is
    process-global: importing this port would re-specify the numerics of every
    other model sharing the process, and of every non-OpenFold3 test collected
    after it. The Chai port, since removed, hit exactly this and was fixed the
    same way. Scoping it to the port's own entry point keeps the guarantee where
    it belongs.

    Decorating :func:`predict` is sufficient for both paths. The setting has to
    be active during *tracing*, not just execution, and the port's single
    `jax.jit` -- in :func:`compile_predict` -- traces through this function, so
    the compiled executable is built under the same precision the eager path
    uses.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        from foldjax.execution import resolved_matmul_precision

        with jax.default_matmul_precision(resolved_matmul_precision(_MATMUL_PRECISION)):
            return function(*args, **kwargs)

    return wrapper


def _restore_ref_atom_category_one_hot(
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Restore private IDs to OpenFold3's historical dense ``int32`` inputs.

    This executes inside the prediction graph immediately before the existing
    atom-feature projections. Dense public features take precedence and remove
    stale private provenance. Compiled callers validate private values on the
    host before tracing; direct eager calls receive the same validation here.
    """

    has_element = "ref_element" in batch
    has_chars = "ref_atom_name_chars" in batch
    private_present = any(
        name in batch for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
    )
    if has_element or has_chars:
        if not private_present:
            return batch
        out = dict(batch)
        for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES:
            out.pop(name, None)
        return out
    if not private_present:
        return batch

    missing = [
        name
        for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
        if name not in batch
    ]
    if missing:
        raise KeyError(
            "OpenFold3 private compact ref atom categories are incomplete; "
            "missing " + ", ".join(missing)
        )
    private_values = tuple(
        batch[name] for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
    )
    if not any(isinstance(value, jax.core.Tracer) for value in private_values):
        validate_compact_ref_atom_categories(batch)

    marker = jnp.asarray(batch[COMPACT_REF_ATOM_CATEGORIES_MARKER])
    element_ids = jnp.asarray(batch[COMPACT_REF_ELEMENT_IDS])
    char_ids = jnp.asarray(batch[COMPACT_REF_ATOM_NAME_CHAR_IDS])
    if marker.shape != () or marker.dtype != jnp.dtype(jnp.uint8):
        raise ValueError(
            "OpenFold3 compact ref atom category marker must be scalar uint8"
        )
    if element_ids.dtype != jnp.dtype(jnp.uint8):
        raise ValueError("OpenFold3 compact ref element IDs must have dtype uint8")
    if char_ids.dtype != jnp.dtype(jnp.uint8):
        raise ValueError("OpenFold3 compact ref atom-name IDs must have dtype uint8")
    if (
        element_ids.ndim != 2
        or element_ids.shape[0] != 1
        or element_ids.shape[1] < 1
        or char_ids.shape != (*element_ids.shape, 4)
    ):
        raise ValueError(
            "OpenFold3 compact ref atom category IDs must have shapes (1, A) "
            "and (1, A, 4)"
        )

    out = dict(batch)
    out["ref_element"] = jax.nn.one_hot(
        element_ids.astype(jnp.int32), 119, dtype=jnp.int32
    )
    out["ref_atom_name_chars"] = jax.nn.one_hot(
        char_ids.astype(jnp.int32), 64, dtype=jnp.int32
    )
    for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES:
        out.pop(name, None)
    return out


def _compute_global_iptm(
    ptm: jnp.ndarray,
    pae_logits: jnp.ndarray,
    has_frame: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int | None,
    bin_min: float,
    bin_max: float,
    no_bins: int,
) -> jnp.ndarray:
    """Return aggregate ipTM, statically omitting its empty monomer reduction."""
    if n_chain == 1:
        return jnp.zeros_like(ptm)
    return compute_ptm(
        pae_logits,
        has_frame,
        token_mask,
        asym_id=asym_id,
        interface=True,
        bin_min=bin_min,
        bin_max=bin_max,
        no_bins=no_bins,
    )


def _expected_tm_pair_scores(
    probabilities: jnp.ndarray,
    mask_i: jnp.ndarray,
    *,
    bin_min: float,
    bin_max: float,
    no_bins: int,
) -> jnp.ndarray:
    """Collapse PAE bins to the expected TM weight for each token pair."""
    mask_i = mask_i.astype(bool)
    considered = jnp.maximum(jnp.sum(mask_i), 1).astype(probabilities.dtype)
    clipped = jnp.maximum(considered, 19.0)
    d0 = 1.24 * jnp.maximum(clipped - 15.0, 0.0) ** (1.0 / 3.0) - 1.8
    weight = 1.0 / (1.0 + (bin_centers(bin_min, bin_max, no_bins) / d0) ** 2)
    return jnp.sum(probabilities * weight, axis=-1)


def _reduce_tm_pair_scores(
    pair_scores: jnp.ndarray,
    has_frame: jnp.ndarray,
    mask_i: jnp.ndarray,
    *,
    considered_dtype: jnp.dtype,
    undefined_pairs: jnp.ndarray | None = None,
    asym_id: jnp.ndarray | None = None,
    interface: bool = False,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Apply the pTM/ipTM row mask and reduction to expected pair scores."""
    if interface and asym_id is None:
        raise ValueError("asym_id is required when interface=True")
    mask_i = mask_i.astype(bool)
    # Historical ``compute_ptm`` casts this count to the PAE-logit dtype before
    # dividing. In particular, bfloat16 rounds counts above 256; using the
    # promoted expected-score dtype here would change that established result.
    considered = jnp.maximum(jnp.sum(mask_i), 1).astype(considered_dtype)
    keep_pair = mask_i[..., :, None] & mask_i[..., None, :]
    if interface:
        keep_pair = keep_pair & (asym_id[..., :, None] != asym_id[..., None, :])
        denominator = jnp.maximum(jnp.sum(keep_pair, axis=-1), eps)
    else:
        denominator = considered
    tm_i = jnp.sum(pair_scores * keep_pair, axis=-1) / denominator
    active_row = has_frame.astype(bool) & mask_i
    tm_i = jnp.where(active_row, tm_i, 0.0)
    result = jnp.max(tm_i, axis=-1)
    if undefined_pairs is not None:
        # Historical compute_ptm multiplies every pair by its mask. IEEE
        # NaN * 0 therefore poisons an active row even when the exceptional
        # pair's column is masked, while a masked row is cleared by the final
        # row selection. Some GPU lowerings turn the compact mask multiply into
        # a select and suppress that NaN; make the scalar contract explicit.
        poison = jnp.any(undefined_pairs & active_row[..., :, None], axis=(-2, -1))
        result = jnp.where(poison, jnp.full_like(result, jnp.nan), result)
    return result


def _compact_confidence_metrics_from_pae(
    pae_logits: jnp.ndarray,
    has_frame: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int | None,
    bin_min: float,
    bin_max: float,
    no_bins: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Reduce one confidence sample without returning its quadratic PAE logits.

    This is used only across the serial-confidence map boundary. Softmax is shared
    by global and chain-pair metrics, while each considered token set retains its
    own ``d0`` and expected TM weights. Moving the reductions across that boundary
    changes only floating-point fusion order; the caller enables it under the
    confidence-only ``1e-3`` max-absolute-delta contract.
    """
    probabilities = jax.nn.softmax(pae_logits, axis=-1)
    # ``jax.nn.softmax`` defines NaN, any +Inf, and an all--Inf row as an
    # undefined distribution. Some GPU softmax lowerings do not preserve that
    # non-finite class when the sample axis is sunk inside ``lax.map``. Track it
    # independently of the pair mask and restore it at scalar reduction. A mixed
    # finite/-Inf row remains well-defined.
    # A non-NaN row has an infinite maximum exactly when it contains +Inf or
    # consists entirely of -Inf. Reuse that reduction instead of materialising
    # three bin-sized predicates; softmax already takes the same row maximum.
    undefined_softmax = jnp.any(jnp.isnan(pae_logits), axis=-1) | jnp.isinf(
        jnp.max(pae_logits, axis=-1)
    )
    kwargs = {
        "bin_min": bin_min,
        "bin_max": bin_max,
        "no_bins": no_bins,
    }
    global_scores = _expected_tm_pair_scores(probabilities, token_mask, **kwargs)
    ptm = _reduce_tm_pair_scores(
        global_scores,
        has_frame,
        token_mask,
        considered_dtype=probabilities.dtype,
        undefined_pairs=undefined_softmax,
    )
    iptm = (
        jnp.zeros_like(ptm)
        if n_chain == 1
        else _reduce_tm_pair_scores(
            global_scores,
            has_frame,
            token_mask,
            considered_dtype=probabilities.dtype,
            undefined_pairs=undefined_softmax,
            asym_id=asym_id,
            interface=True,
        )
    )

    chain_pair = None
    if n_chain is not None and n_chain > 1:
        valid = token_mask.astype(bool)
        chain_masks = [valid & (asym_id == chain) for chain in range(n_chain)]
        zero = jnp.zeros(pae_logits.shape[0], dtype=pae_logits.dtype)
        matrix = [[zero for _ in range(n_chain)] for _ in range(n_chain)]
        for i in range(n_chain):
            for j in range(i + 1, n_chain):
                pair_mask = chain_masks[i] | chain_masks[j]
                both_present = jnp.any(chain_masks[i]) & jnp.any(chain_masks[j])
                pair_scores = _expected_tm_pair_scores(
                    probabilities, pair_mask, **kwargs
                )
                value = _reduce_tm_pair_scores(
                    pair_scores,
                    has_frame,
                    pair_mask,
                    considered_dtype=probabilities.dtype,
                    undefined_pairs=undefined_softmax,
                    asym_id=asym_id,
                    interface=True,
                )
                value = jnp.where(both_present, value, 0.0)
                matrix[i][j] = value
                matrix[j][i] = value
        chain_pair = jnp.stack([jnp.stack(row, axis=-1) for row in matrix], axis=-2)
    return ptm, iptm, chain_pair


@openfold3_precision
def predict(
    key: jax.Array,
    batch: Mapping[str, jnp.ndarray],
    params: InferenceParams,
    config: InferenceConfig,
    representative_atoms: RepresentativeAtomTable,
    *,
    n_chain: int | None = None,
    noise_fn: Callable[[int, tuple[int, ...]], jnp.ndarray] | None = None,
    noise_tape: jnp.ndarray | None = None,
    noise_mask: jnp.ndarray | None = None,
    augment: bool = True,
    use_trunk_pair_embedding: bool = True,
) -> Prediction:
    """Run trunk, diffusion sampling and the confidence heads.

    Args:
        key: PRNG key for the sampler and augmentation.
        batch: featurized input.
        params: mapped parameters.
        config: shapes and hyperparameters from the checkpoint.
        n_chain: normalized active-chain count. ``1`` asserts a known monomer
            and specializes its aggregate ipTM to zero; ``None`` keeps the
            generic unknown-chain path. Counts above one also enable the
            chain-pair ipTM matrix.
        noise_fn: forwarded to the sampler to replace its random draws; supplied
            only to compare the rollout against another implementation.
        noise_tape: runtime sampler draws used by shape padding to preserve the
            real atom prefix for a fixed seed.
        noise_mask: runtime sample/atom mask used to preserve that prefix
            without materializing all rollout draws at once.
        augment: apply centred random augmentation each rollout step. Turning it
            off is for the same comparison, not for production.
        use_trunk_pair_embedding: feed the trunk pair embedding into the confidence
            re-embedding. Upstream's ``use_zij_trunk_embedding``; when false the
            pair embedding is zeroed there, and only there -- the distogram head
            still reads the unzeroed trunk output.

    Returns:
        A :class:`Prediction`.
    """
    if config.cp_shards != _active_cp_shards():
        raise RuntimeError(
            f"cp_shards={config.cp_shards} but the active context-parallel "
            f"mesh has {_active_cp_shards()} shard(s); run through "
            "compile_predict or activate context_parallel() yourself"
        )
    if (
        config.msa_depth is not None
        and batch["msa_mask"].shape[1] > config.msa_depth
        and _MSA_CYCLE_INDICES not in batch
    ):
        raise ValueError(
            "OpenFold3 direct predict with an over-depth MSA needs "
            "prepare_msa_cycle_features; fixed first-row selection is not native"
        )
    batch = _restore_ref_atom_category_one_hot(batch)
    s_input, s_trunk, z = trunk(
        batch,
        params.trunk,
        num_recycles=config.num_recycles,
        n_query=config.n_query,
        n_key=config.n_key,
        atom_heads=config.atom_heads,
        n_token=config.n_token,
        max_relative_idx=config.max_relative_idx,
        max_relative_chain=config.max_relative_chain,
        no_heads_msa=config.no_heads_msa,
        no_heads_pair=config.no_heads_pair,
        no_heads_pair_bias=config.no_heads_pair_bias,
        opm_first=config.opm_first,
        chunk_size=config.pair_chunk_size,
    )
    # This axis indexes recycling, not batch: it must not reach sample expansion.
    batch = {name: value for name, value in batch.items() if name != _MSA_CYCLE_INDICES}

    schedule = noise_schedule(
        config.num_steps,
        sigma_data=config.sigma_data,
        s_max=config.s_max,
        s_min=config.s_min,
        p=config.p,
    )

    # The sampler carries a leading sample axis, so every conditioning tensor and
    # batch feature has to line up with it. Upstream expands the batch the same
    # way; broadcasting only the coordinates leaves the atom encoder mixing a
    # 1-sized batch against an S-sized one.
    # Named ``name`` rather than ``key``: a comprehension variable is scoped to the
    # comprehension so it cannot shadow the PRNG ``key`` parameter, but a reader
    # should not have to know that to be sure.
    sampled_batch = {
        name: _expand_samples(value, config.num_samples)
        for name, value in batch.items()
    }

    # Pair conditioning takes no noise level: it is a function of the batch and the
    # trunk pair representation alone, so it is constant across every rollout step
    # and across samples. Upstream evaluates it inside each denoiser call, which at
    # the released 200 steps means 200 identical evaluations of the widest tensors
    # in the diffusion path -- the relative-position encoding is
    # ``[N_token, N_token, 139]`` and is concatenated onto the pair representation
    # before the projection. Hoisting it out is exact, not an approximation.
    # Born sharded under context parallelism; the sample axis stays a
    # broadcast view, only the row axis is split.
    # Kept at its natural leading axis of 1 and widened at the point of use, so
    # the sample width is a property of the coordinates being denoised rather
    # than of the config. That is what lets the rollout run a chunk of the
    # samples at a time: everything here is a `broadcast_to` view, so widening
    # to 5 instead of 32 costs nothing and copies nothing.
    zij_base = shard_pair_rows(
        pair_conditioning(
            batch,
            z,
            params.diffusion_conditioning,
            max_relative_idx=config.max_relative_idx,
            max_relative_chain=config.max_relative_chain,
            token_mask=batch["token_mask"],
        )
    )

    def denoise_fn(xl_noisy: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # The single path is the only one that reads the noise level, so it is the
        # only one that has to be rebuilt per step. It is still sample-independent:
        # ``t`` is one value per step, shared by every sample.
        width = xl_noisy.shape[0]
        si = _expand_samples(
            single_conditioning(
                s_input,
                s_trunk,
                t,
                params.diffusion_conditioning,
                sigma_data=config.sigma_data,
                token_mask=batch["token_mask"],
            ),
            width,
        )
        return denoise(
            {name: _expand_samples(value, width) for name, value in batch.items()},
            xl_noisy,
            t,
            si,
            _expand_samples(s_trunk, width),
            shard_pair_rows(_expand_samples(zij_base, width)),
            params.denoiser,
            n_query=config.n_query,
            n_key=config.n_key,
            atom_heads=config.atom_heads,
            token_heads=config.token_heads,
            n_token=config.n_token,
            sigma_data=config.sigma_data,
        )

    if config.stop_after_trunk:
        return Prediction(
            coordinates=None,
            plddt=None,
            ptm=None,
            iptm=None,
            chain_pair_iptm=None,
            pae_logits=None,
            pde_logits=None,
            distogram_logits=None,
            single_inputs=(
                s_input if "single_inputs" in config.returned_representations else None
            ),
            single=(s_trunk if "single" in config.returned_representations else None),
            pair=z if "pair" in config.returned_representations else None,
        )

    def sample(key, noise_tape, noise_mask):
        return sample_diffusion(
            key,
            schedule,
            (config.num_samples, config.n_atom, 3),
            denoise_fn,
            gamma_0=config.gamma_0,
            gamma_min=config.gamma_min,
            noise_scale=config.noise_scale,
            step_scale=config.step_scale,
            augment_fn=(
                (
                    # Widened to whatever `xl` carries, not to the config: the
                    # rollout may be running a chunk of the samples.
                    lambda k, xl: centre_random_augmentation(
                        k, xl, _expand_samples(batch["atom_mask"], xl.shape[0])
                    )
                )
                if augment
                else None
            ),
            noise_fn=noise_fn,
            noise_tape=noise_tape,
            noise_mask=noise_mask,
            diffusion_chunk_size=config.diffusion_chunk_size,
        )

    coordinates = sample(key, noise_tape, noise_mask)

    # The distogram head is the only one that reads the trunk pair embedding.
    disto_logits = distogram_head(z, params.distogram_head)

    # Every other head reads representations that have been re-embedded with the
    # predicted geometry, so the trunk outputs must not be passed to them.
    rep_x, rep_mask = token_representative_atoms(
        sampled_batch, coordinates, sampled_batch["atom_mask"], representative_atoms
    )
    token_mask_1 = batch["token_mask"]
    pair_mask_1 = token_mask_1[..., :, None] * token_mask_1[..., None, :]
    # Zeroing happens after the distogram head, which reads the trunk embedding
    # regardless.
    z_conf_input = shard_pair_rows(z if use_trunk_pair_embedding else jnp.zeros_like(z))
    returned = set(config.returned_pair_logits)
    return_pae = "pae_logits" in returned
    return_pde = "pde_logits" in returned

    # pTM/ipTM exclude tokens whose *predicted* coordinates cannot form a valid
    # local frame. For atomized ligands and modified residues this depends on
    # their two closest same-chain atoms and their angle. A feature archive can
    # contain a frame mask for its input/reference geometry, but reusing that
    # mask for every diffusion sample changes the confidence score, so derive it
    # unconditionally here.
    _, has_frame = token_frame_atoms(
        sampled_batch,
        coordinates,
        sampled_batch["atom_mask"],
        has_atomized_tokens=config.has_atomized_tokens,
    )
    ptm_kwargs = {
        "bin_min": 0.0,
        "bin_max": config.pae_bin_max,
        "no_bins": config.pae_bins,
    }
    token_mask = batch["token_mask"].reshape(-1)[: config.n_token]
    asym_id = batch["asym_id"].reshape(-1)[: config.n_token]

    def confidence_pair(
        si_input: jnp.ndarray,
        si: jnp.ndarray,
        zij: jnp.ndarray,
        x_pred: jnp.ndarray,
        mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ):
        """Re-embed geometry and read the two pair heads off it.

        Every argument has to arrive at the same rank. ``x_pred`` is what carries
        the sample axis, and the distance embedding built from it propagates that
        axis into the pair representation -- so passing a batch-1 single
        representation alongside a 5-sample ``x_pred`` makes the Pairformer's
        scan carry disagree with itself.
        """
        s_conf_one, z_conf_one = pairformer_embedding(
            si_input,
            si,
            zij,
            x_pred,
            params.pairformer_embedding,
            single_mask=mask,
            pair_mask=pair_mask,
            no_heads_pair=config.no_heads_pair,
            no_heads_pair_bias=config.no_heads_pair_bias,
            min_bin=config.confidence_min_bin,
            max_bin=config.confidence_max_bin,
            no_bin=config.confidence_no_bin,
            chunk_size=config.pair_chunk_size,
        )
        # The pair heads are evaluated here rather than outside so the re-embedded
        # pair representation never has to exist at sample rank. PAE is either a
        # requested output or immediately reduced by the serial caller below.
        pae_one = predicted_aligned_error_head(z_conf_one, params.pae_head)
        pde_one = (
            predicted_distance_error_head(z_conf_one, params.pde_head)
            if return_pde
            else None
        )
        return s_conf_one, pae_one, pde_one

    # Upstream's ``apply_per_sample`` maps confidence over the token cutoff.
    # FoldJAX defaults the cutoff to zero, so every multi-sample prediction takes
    # the bounded schedule. A positive threshold may batch within the released
    # width, while ``None`` is the explicit unbounded opt-out. Nothing here mixes
    # samples, so mapping and batching give the same values -- the difference is
    # whether N_sample pair representations are live at once, which can decide
    # whether prediction fits.
    per_sample_confidence = _per_sample_confidence(config)
    sink_pae_metrics = _sink_pae_metrics(config)
    if sink_pae_metrics:

        def confidence_and_metrics(one):
            s_one, pae_one, pde_one = confidence_pair(
                s_input,
                s_trunk,
                z_conf_input,
                one[0][None],
                one[1][None],
                pair_mask_1,
            )
            metrics_one = _compact_confidence_metrics_from_pae(
                pae_one,
                one[2][None],
                token_mask,
                asym_id,
                n_chain=n_chain,
                **ptm_kwargs,
            )
            return jax.tree.map(lambda leaf: leaf[0], (s_one, metrics_one, pde_one))

        s_conf, (ptm, iptm, chain_pair), pde_logits = jax.lax.map(
            confidence_and_metrics,
            (rep_x, rep_mask, has_frame),
        )
        pae_logits = None
    elif per_sample_confidence:
        s_conf, pae_logits, pde_logits = jax.lax.map(
            lambda one: jax.tree.map(
                lambda leaf: leaf[0],
                confidence_pair(
                    s_input,
                    s_trunk,
                    z_conf_input,
                    one[0][None],
                    one[1][None],
                    pair_mask_1,
                ),
            ),
            (rep_x, rep_mask),
        )
    else:
        samples = config.num_samples
        s_conf, pae_logits, pde_logits = confidence_pair(
            _expand_samples(s_input, samples),
            _expand_samples(s_trunk, samples),
            _expand_samples(z_conf_input, samples),
            rep_x,
            rep_mask,
            _expand_samples(pair_mask_1, samples),
        )

    if not sink_pae_metrics:
        pae_for_ptm = jnp.broadcast_to(
            pae_logits, (config.num_samples, *pae_logits.shape[-3:])
        )
        ptm = compute_ptm(pae_for_ptm, has_frame, token_mask, **ptm_kwargs)
        iptm = _compute_global_iptm(
            ptm,
            pae_for_ptm,
            has_frame,
            token_mask,
            asym_id,
            n_chain=n_chain,
            **ptm_kwargs,
        )
        chain_pair = None
        if n_chain is not None and n_chain > 1:
            chain_pair = compute_chain_pair_iptm(
                pae_for_ptm,
                has_frame,
                token_mask,
                asym_id,
                n_chain=n_chain,
                **ptm_kwargs,
            )

    plddt_logits = atom_logit_head(
        s_conf,
        params.plddt_head,
        batch["max_atom_per_token_mask"],
        max_atoms_per_token=config.max_atoms_per_token,
        c_out=config.plddt_bins,
        n_atom=config.n_atom,
    )
    experimentally_resolved_logits = None
    if params.experimentally_resolved_head is not None:
        experimentally_resolved_logits = atom_logit_head(
            s_conf,
            params.experimentally_resolved_head,
            batch["max_atom_per_token_mask"],
            max_atoms_per_token=config.max_atoms_per_token,
            c_out=config.experimentally_resolved_bins,
            n_atom=config.n_atom,
        )

    return Prediction(
        coordinates=coordinates,
        plddt=compute_plddt(plddt_logits),
        ptm=ptm,
        iptm=iptm,
        chain_pair_iptm=chain_pair,
        pae_logits=pae_logits if return_pae else None,
        pde_logits=pde_logits if return_pde else None,
        distogram_logits=disto_logits if "distogram_logits" in returned else None,
        experimentally_resolved_logits=experimentally_resolved_logits,
        single_inputs=(
            s_input if "single_inputs" in config.returned_representations else None
        ),
        single=s_trunk if "single" in config.returned_representations else None,
        pair=z if "pair" in config.returned_representations else None,
    )


def released_config(
    *,
    n_token: int,
    n_atom: int,
    # shared.num_recycles is 3, and upstream runs num_recycles + 1 cycles.
    num_recycles: int = 4,
    num_samples: int = 5,
    num_steps: int = 200,
    pair_chunk_size: int | None | str = "auto",  # "auto" resolves from n_token
    per_sample_token_cutoff: int | None = 0,
    diffusion_chunk_size: int | None | str = "auto",
    msa_depth: int | None = RELEASED_MSA_DEPTH,
    cp_shards: int = 1,
    cp_layout: str = "auto",
    returned_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    has_atomized_tokens: bool = True,
    max_array_bytes: int | None = DEFAULT_ARRAY_BUDGET_BYTES,
) -> InferenceConfig:
    """Return the released OpenFold3 architecture settings.

    Values are transcribed from upstream's
    ``projects/of3_all_atom/config/model_config.py``, which is the same file the
    released checkpoints were produced under. Only the sizes that depend on the
    input (token/atom counts) and the sampling knobs are arguments.

    ``max_array_bytes`` mirrors the raw-array writer's explicit budget when
    deciding which quadratic pair logits become static graph outputs. ``None``
    returns all three distributions; callers must also give the writer the same
    value if they intend to persist them.

    Verify against the checkpoint before trusting this: run
    ``openfold3-jax-inspect-checkpoint`` and check the block counts against
    ``pairformer 48``, ``msa_module 4``, ``diffusion_transformer 24`` and
    ``atom_transformer 3``. A mismatch means the weights use a different config
    than upstream's default, and these values must not be used.
    """
    resolved_chunk: int | None = (
        auto_pair_chunk_size(n_token, no_heads=_RELEASED_PAIR_HEADS)
        if isinstance(pair_chunk_size, str)
        else pair_chunk_size
    )
    return InferenceConfig(
        n_token=n_token,
        n_atom=n_atom,
        returned_pair_logits=plan_returned_pair_logits(
            n_token=n_token,
            num_samples=num_samples,
            max_bytes=max_array_bytes,
        ),
        n_query=32,
        n_key=128,
        atom_heads=4,
        token_heads=16,
        no_heads_msa=8,
        no_heads_pair=4,
        no_heads_pair_bias=16,
        max_relative_idx=32,
        max_relative_chain=2,
        num_recycles=num_recycles,
        num_samples=num_samples,
        max_atoms_per_token=23,
        plddt_bins=50,
        pae_bins=64,
        pae_bin_max=32.0,
        num_steps=num_steps,
        sigma_data=16.0,
        s_max=160.0,
        s_min=4e-4,
        p=7,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        pair_chunk_size=resolved_chunk,
        per_sample_token_cutoff=per_sample_token_cutoff,
        diffusion_chunk_size=(
            auto_diffusion_chunk_size(num_samples)
            if diffusion_chunk_size == "auto"
            else diffusion_chunk_size
        ),
        msa_depth=msa_depth,
        cp_shards=cp_shards,
        cp_layout=cp_layout,
        returned_representations=returned_representations,
        stop_after_trunk=stop_after_trunk,
        has_atomized_tokens=has_atomized_tokens,
    )


#: Block counts the released architecture is expected to have, for checking a
#: checkpoint against :func:`released_config`.
RELEASED_BLOCK_COUNTS = {
    "pairformer_stack.blocks": 48,
    "msa_module.blocks": 4,
    "diffusion_transformer.blocks": 24,
    "atom_transformer.blocks": 3,
}


def resolve_cp_layout(config: InferenceConfig) -> str:
    """Turn ``config.cp_layout`` into a layout :func:`context_parallel` accepts.

    ``"auto"`` stays on the 1-D layout. The square grid is the better design
    and is verified on CPU meshes, but every published measurement of this
    feature was taken on the 1-D layout, and a default that silently changes
    the program would make those numbers describe a configuration nobody can
    reproduce; it flips once the square grid has its own GPU evidence. An
    explicit ``"1d"``/``"2d"`` is passed through and validated by
    ``context_parallel``, so ``"2d"`` on a non-square shard count still fails
    loudly instead of quietly falling back.
    """
    if config.cp_layout != "auto":
        return config.cp_layout
    return "1d"


@dataclass(frozen=True, slots=True)
class _PredictGraphIdentity:
    """Hashable identity of every Python choice made while tracing prediction."""

    config: InferenceConfig
    n_chain: int | None
    augment: bool
    use_trunk_pair_embedding: bool
    rng_route: str
    triangle_kernel: str
    cp_topology: tuple[object, ...]
    cache_scope: str | None


def _validated_representative_atoms(
    table: RepresentativeAtomTable,
) -> RepresentativeAtomTable:
    """Validate and canonicalise the tiny chemistry table as dynamic data."""

    if not isinstance(table, RepresentativeAtomTable):
        raise TypeError("representative_atoms must be a RepresentativeAtomTable")
    arrays: list[np.ndarray] = []
    for name, value in zip(table._fields, table, strict=True):
        host = np.asarray(value)
        if host.shape != (32,):
            raise ValueError(
                f"representative atom field {name!r} must have shape (32,), "
                f"got {host.shape}"
            )
        if not np.issubdtype(host.dtype, np.number) or not np.isfinite(host).all():
            raise ValueError(
                f"representative atom field {name!r} must be finite numeric data"
            )
        # Keep these 1.6 KiB of lookups on the host. ``jax.jit`` stages the
        # complete dynamic argument tree once; converting each field with
        # ``jnp.asarray`` here would compile a separate staging program.
        arrays.append(np.asarray(host, dtype=np.float32))
    return RepresentativeAtomTable(*arrays)


def _rng_route(
    noise_tape: jnp.ndarray | None,
    noise_mask: jnp.ndarray | None,
) -> str:
    if noise_tape is not None and noise_mask is not None:
        raise ValueError("noise_tape and noise_mask are mutually exclusive")
    if noise_tape is not None:
        return "tape"
    if noise_mask is not None:
        return "mask"
    return "native"


def _cp_topology_identity(mesh, *, layout: str) -> tuple[object, ...]:
    """Return topology plus ordered physical devices captured by sharding ops."""

    if mesh is None:
        return ("serial", 1, (1, 1), (), ())
    shape = tuple(int(size) for size in mesh.devices.shape)
    grid = shape if len(shape) == 2 else (shape[0], 1)
    devices = tuple(
        (
            str(device.platform),
            int(device.process_index),
            int(device.id),
            str(device.device_kind),
        )
        for device in mesh.devices.flat
    )
    return (
        layout,
        int(mesh.devices.size),
        grid,
        tuple(str(axis) for axis in mesh.axis_names),
        devices,
    )


def _predict_for_identity(
    key: jax.Array,
    batch: Mapping[str, jnp.ndarray],
    params: InferenceParams,
    representative_atoms: RepresentativeAtomTable,
    noise_tape: jnp.ndarray | None,
    noise_mask: jnp.ndarray | None,
    *,
    identity: _PredictGraphIdentity,
) -> Prediction:
    """Prediction body for one fully resolved compile-time identity."""

    return predict(
        key,
        batch,
        params,
        identity.config,
        representative_atoms,
        n_chain=identity.n_chain,
        noise_tape=noise_tape if identity.rng_route == "tape" else None,
        noise_mask=noise_mask if identity.rng_route == "mask" else None,
        augment=identity.augment,
        use_trunk_pair_embedding=identity.use_trunk_pair_embedding,
    )


_MAX_RETAINED_EXECUTABLES = 8


class _CompiledPredictPool:
    """Bounded process-wide pool of stable OpenFold prediction JITs.

    A single module-level ``jax.jit`` deduplicates repeated factories, but its
    internal cache never evicts. OpenFold programs can be hundreds of
    megabytes, and unpadded plural requests naturally create many shapes. One
    JIT per graph identity lets us evict least-recently-used programs while
    keeping parameters and chemistry tables as ordinary dynamic arguments.
    """

    def __init__(self, limit: int = _MAX_RETAINED_EXECUTABLES) -> None:
        self._limit = int(limit)
        self._entries: OrderedDict[_PredictGraphIdentity, Any] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _new(identity: _PredictGraphIdentity):
        return jax.jit(functools.partial(_predict_for_identity, identity=identity))

    @staticmethod
    def _entry_size(compiled: Any) -> int:
        size = getattr(compiled, "_cache_size", None)
        return int(size()) if callable(size) else 1

    @staticmethod
    def _drop(compiled: Any) -> None:
        clear = getattr(compiled, "clear_cache", None)
        if callable(clear):
            clear()

    def _get(self, identity: _PredictGraphIdentity):
        compiled = self._entries.pop(identity, None)
        if compiled is None:
            while len(self._entries) >= self._limit:
                _identity, evicted = self._entries.popitem(last=False)
                self._drop(evicted)
            compiled = self._new(identity)
        self._entries[identity] = compiled
        return compiled

    def _trim(self) -> None:
        while self._entries and self._cache_size_unlocked() > self._limit:
            _identity, evicted = self._entries.popitem(last=False)
            self._drop(evicted)

    def _cache_size_unlocked(self) -> int:
        return sum(self._entry_size(entry) for entry in self._entries.values())

    def __call__(
        self,
        key: jax.Array,
        batch: Mapping[str, jnp.ndarray],
        params: InferenceParams,
        representative_atoms: RepresentativeAtomTable,
        noise_tape: jnp.ndarray | None,
        noise_mask: jnp.ndarray | None,
        *,
        identity: _PredictGraphIdentity,
    ) -> Prediction:
        with self._lock:
            compiled = self._get(identity)
            result = compiled(
                key,
                batch,
                params,
                representative_atoms,
                noise_tape,
                noise_mask,
            )
            self._trim()
            return result

    def lower(
        self,
        key: jax.Array,
        batch: Mapping[str, jnp.ndarray],
        params: InferenceParams,
        representative_atoms: RepresentativeAtomTable,
        noise_tape: jnp.ndarray | None,
        noise_mask: jnp.ndarray | None,
        *,
        identity: _PredictGraphIdentity,
    ):
        with self._lock:
            compiled = self._get(identity)
            lowered = compiled.lower(
                key,
                batch,
                params,
                representative_atoms,
                noise_tape,
                noise_mask,
            )
            self._trim()
            return lowered

    def clear_cache(self) -> None:
        with self._lock:
            for compiled in self._entries.values():
                self._drop(compiled)
            self._entries.clear()
            reset_cache_scope_tracking()

    def _cache_size(self) -> int:
        with self._lock:
            return self._cache_size_unlocked()


_compiled_predict = _CompiledPredictPool()


def _require_same_topology(identity: _PredictGraphIdentity, mesh) -> None:
    current = _cp_topology_identity(mesh, layout=identity.config.cp_layout)
    if current != identity.cp_topology:
        raise RuntimeError(
            "the OpenFold3 compiled executable was lowered for a different "
            "context-parallel device topology"
        )


def _prepare_ref_atom_category_graph_input(
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate compact host values and keep dense inputs authoritative.

    This boundary has to run for both contextual JIT calls and already-lowered
    public executables.  The latter accept new values with the lowered shapes,
    so graph-side shape checks alone cannot reject a malformed marker, sentinel,
    or padding mask.  Removing stale private leaves here also preserves the
    dense executable's PyTree identity.
    """

    validate_compact_ref_atom_categories(batch)
    if ("ref_element" in batch or "ref_atom_name_chars" in batch) and any(
        name in batch for name in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
    ):
        return {
            name: value
            for name, value in batch.items()
            if name not in COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES
        }
    return batch


def _persistent_cache_matches(scope: str | None) -> bool:
    """Whether JAX is currently configured to populate this exact scope."""

    if scope is None:
        return False
    if not getattr(jax.config, "jax_enable_compilation_cache", True):
        return False
    if getattr(jax.config, "jax_compilation_cache_max_size", -1) == 0:
        return False
    configured = getattr(jax.config, "jax_compilation_cache_dir", None)
    return configured is not None and canonical_cache_scope(str(configured)) == scope


def _persistent_cache_is_bounded(scope: str | None) -> bool:
    """Whether the active cache requires a matching access-time sidecar."""

    return _persistent_cache_matches(scope) and (
        getattr(jax.config, "jax_compilation_cache_max_size", -1) != -1
    )


class _BoundCompiledPredict:
    """Public three-argument view over a six-argument JAX executable."""

    def __init__(
        self,
        compiled: Any,
        *,
        table: RepresentativeAtomTable,
        identity: _PredictGraphIdentity,
    ) -> None:
        self._compiled = compiled
        self._table = table
        self._identity = identity

    def __call__(
        self,
        key,
        batch,
        params,
        *,
        noise_tape=None,
        noise_mask=None,
    ):
        batch = _prepare_ref_atom_category_graph_input(batch)
        route = _rng_route(noise_tape, noise_mask)
        if route != self._identity.rng_route:
            raise ValueError(
                "compiled OpenFold3 RNG route does not match the route used "
                f"during lowering: expected {self._identity.rng_route!r}, "
                f"got {route!r}"
            )
        with (
            triangle_backend(self._identity.triangle_kernel),
            context_parallel(
                self._identity.config.cp_shards,
                layout=self._identity.config.cp_layout,
            ) as mesh,
        ):
            _require_same_topology(self._identity, mesh)
            return self._compiled(
                replicate_tree(key),
                replicate_tree(batch),
                replicate_tree(params),
                replicate_tree(self._table),
                replicate_tree(noise_tape),
                replicate_tree(noise_mask),
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled, name)


class _BoundLoweredPredict:
    """Lowered OpenFold program retaining its public callable contract."""

    def __init__(
        self,
        lowered: Any,
        *,
        table: RepresentativeAtomTable,
        identity: _PredictGraphIdentity,
    ) -> None:
        self._lowered = lowered
        self._table = table
        self._identity = identity

    def compile(self, *args, **kwargs) -> _BoundCompiledPredict:
        identity = self._identity
        with (
            triangle_backend(identity.triangle_kernel),
            context_parallel(
                identity.config.cp_shards,
                layout=identity.config.cp_layout,
            ) as mesh,
        ):
            _require_same_topology(identity, mesh)
            bounded_cache = _persistent_cache_is_bounded(identity.cache_scope)
            cache_token = inspect_cache_scope(
                identity.cache_scope, repair_atime=bounded_cache
            )
            if cache_token is not None and cache_token.invalidated:
                _compiled_predict.clear_cache()
            compiled = self._lowered.compile(*args, **kwargs)
            observe_cache_scope(
                identity.cache_scope,
                token=cache_token,
                require_payload=_persistent_cache_matches(identity.cache_scope),
                require_atime=bounded_cache,
            )
        return _BoundCompiledPredict(
            compiled,
            table=self._table,
            identity=identity,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lowered, name)


def compile_predict(
    config: InferenceConfig,
    representative_atoms: RepresentativeAtomTable,
    *,
    n_chain: int | None = None,
    augment: bool = True,
    use_trunk_pair_embedding: bool = True,
    triangle_kernel: str | None = None,
    cache_scope: str | None = None,
) -> Callable[[jax.Array, Mapping[str, jnp.ndarray], InferenceParams], Prediction]:
    """Return a compiled ``predict`` bound to one configuration.

    Compilation is not optional in practice. Run eagerly, the released
    architecture executes 4 recycling cycles over 48 Pairformer blocks as
    individual dispatches; measured on an RTX PRO 6000 at 16 tokens that is ~69 s
    per call. Compiled, the same call is ~0.1 s, and the released 5-sample
    200-step rollout is ~0.9 s -- the rollout is a ``lax.scan``, so step count
    costs almost nothing at run time.

    The one-time cost is real: ~170 s to compile at 8 rollout steps and ~280 s at
    200. Repeated factory calls share a bounded module-level JIT pool, while
    :func:`predict` stays an ordinary function for parity tests. The static
    identity includes the resolved mesh devices, triangle kernel, RNG route and
    persistent-cache scope, so reuse cannot replay a graph traced under different
    ambient state. The least-recently-used programs are dropped after eight
    retained executables rather than growing process memory without a bound.

    ``params`` stays a runtime argument, so weights can be swapped -- a different
    checkpoint, or averaged weights -- without recompiling. Changing ``config``,
    the batch shapes, or any keyword here does require a new compilation.

    ``noise_fn`` is deliberately not used by production callers: baking a
    Python callback into a compiled function would defeat the scan. A concrete
    ``noise_tape`` and ``noise_mask`` remain normal runtime keyword arguments.
    Padding uses the mask so the compact random stream is preserved without
    retaining every rollout draw at once.
    """
    table = _validated_representative_atoms(representative_atoms)
    layout = "1d" if config.cp_shards <= 1 else resolve_cp_layout(config)
    compiled_config = config._replace(cp_layout=layout)
    scope = None if cache_scope is None else canonical_cache_scope(str(cache_scope))

    def invoke(
        operation,
        key,
        batch,
        params,
        *,
        noise_tape=None,
        noise_mask=None,
    ):
        # Reject malformed private provenance while values are still concrete,
        # before tracing or consulting the compiled-executable cache.
        batch = _prepare_ref_atom_category_graph_input(batch)

        # Tracing happens on the first call, so the mesh has to be active
        # here, not at factory time. A checkpoint committed to one device
        # fails the multi-device call's device-assignment check; everything
        # token-linear is replicated onto the mesh, and the graph's own
        # constraints shard the pair-shaped state from its first
        # materialization.
        route = _rng_route(noise_tape, noise_mask)
        effective_kernel = resolve_triangle_kernel(
            triangle_kernel, cp_shards=compiled_config.cp_shards
        )
        with (
            triangle_backend(effective_kernel),
            context_parallel(compiled_config.cp_shards, layout=layout) as mesh,
        ):
            identity = _PredictGraphIdentity(
                config=compiled_config,
                n_chain=n_chain,
                augment=augment,
                use_trunk_pair_embedding=use_trunk_pair_embedding,
                rng_route=route,
                triangle_kernel=effective_kernel,
                cp_topology=_cp_topology_identity(mesh, layout=layout),
                cache_scope=scope,
            )
            bounded_cache = _persistent_cache_is_bounded(scope)
            cache_token = inspect_cache_scope(scope, repair_atime=bounded_cache)
            if cache_token is not None and cache_token.invalidated:
                # An in-memory JIT hit otherwise prevents a deleted or replaced
                # persistent namespace from being populated again.
                _compiled_predict.clear_cache()
            value = operation(
                replicate_tree(key),
                replicate_tree(batch),
                replicate_tree(params),
                replicate_tree(table),
                replicate_tree(noise_tape),
                replicate_tree(noise_mask),
                identity=identity,
            )
            observe_cache_scope(
                scope,
                token=cache_token,
                require_payload=_persistent_cache_matches(scope),
                require_atime=bounded_cache,
            )
            return value, identity

    def contextual(
        key,
        batch,
        params,
        *,
        noise_tape=None,
        noise_mask=None,
    ):
        value, _identity = invoke(
            _compiled_predict,
            key,
            batch,
            params,
            noise_tape=noise_tape,
            noise_mask=noise_mask,
        )
        return value

    def lower(
        key,
        batch,
        params,
        *,
        noise_tape=None,
        noise_mask=None,
    ):
        """Lower with the same mesh and static identity as an ordinary call."""

        lowered, identity = invoke(
            _compiled_predict.lower,
            key,
            batch,
            params,
            noise_tape=noise_tape,
            noise_mask=noise_mask,
        )
        return _BoundLoweredPredict(
            lowered,
            table=table,
            identity=identity,
        )

    # ``compile_predict`` historically returned JAX's jitted callable. Keep
    # its most useful introspection surface while the actual executable cache
    # now lives on the module-level function.
    contextual.lower = lower  # type: ignore[attr-defined]
    return contextual
