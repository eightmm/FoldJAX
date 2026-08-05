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
from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.augmentation import centre_random_augmentation
from foldjax.models.openfold3.models.confidence import (
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
    num_cycles: int
    num_samples: int
    max_atoms_per_token: int
    plddt_bins: int
    pae_bins: int
    pae_bin_max: float
    no_rollout_steps: int
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
    # Token count above which the confidence re-embedding runs one diffusion sample
    # at a time. 750 is upstream's own default (``per_sample_token_cutoff`` in
    # ``model_config.py``); ``None`` always batches the samples.
    per_sample_token_cutoff: int | None = 750


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

    coordinates: jnp.ndarray
    plddt: jnp.ndarray
    ptm: jnp.ndarray
    iptm: jnp.ndarray | None
    chain_pair_iptm: jnp.ndarray | None
    pae_logits: jnp.ndarray
    pde_logits: jnp.ndarray
    distogram_logits: jnp.ndarray
    experimentally_resolved_logits: jnp.ndarray | None = None


#: Bytes the triangle-attention score tensor is allowed to reach before
#: :func:`auto_pair_chunk_size` starts chunking. 8 GiB leaves room for the rest of
#: the block, the diffusion path and the weights on a 96 GiB card; it is a budget,
#: not a measured constant, and the caller can override it.
PAIR_SCORE_BUDGET_BYTES = 8 * 2**30

#: Triangle attention head count in the released architecture. Named because
#: ``auto_pair_chunk_size`` needs it before an ``InferenceConfig`` exists to read it
#: from; ``test_released_config`` checks the config's copy against upstream.
_RELEASED_PAIR_HEADS = 4


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

    Mirrors upstream's ``apply_per_sample``, including its default cutoff of 750
    tokens.

    Measured on a real 966-token target with the released weights, the per-sample
    branch is not a trade: it is **bitwise identical** on coordinates, pTM and the PAE
    logits (pLDDT differs by 1.8e-7, one float32 bit) and **3.7x faster** -- 69.5 s
    against 255.1 s. Holding five pair representations at once costs more in memory
    traffic than it wins in parallelism at that size, so upstream's cutoff is a
    throughput setting as much as a memory one.

    Which raises a question this does not answer: 750 may be conservative, and the
    crossover may sit well below it. The default stays at upstream's value because
    matching the released configuration is worth more than an unmeasured gain; a
    caller who wants to explore it can set ``per_sample_token_cutoff`` directly.
    """
    return (
        config.per_sample_token_cutoff is not None
        and config.num_samples > 1
        and config.n_token > config.per_sample_token_cutoff
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


def openfold3_precision(function):
    """Run `function` with JAX's matmul precision pinned to "highest".

    OpenFold3's torch modules compute FP32 products unless a tensor was
    explicitly cast to BF16. JAX otherwise permits TF32 on NVIDIA GPUs, which
    silently costs ~3 decimal digits per matmul -- invisible on CPU, where the
    parity gate runs, and amplified by the diffusion rollout.

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
        with jax.default_matmul_precision("highest"):
            return function(*args, **kwargs)

    return wrapper


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
    augment: bool = True,
    use_trunk_pair_embedding: bool = True,
) -> Prediction:
    """Run trunk, diffusion sampling and the confidence heads.

    Args:
        key: PRNG key for the sampler and augmentation.
        batch: featurized input.
        params: mapped parameters.
        config: shapes and hyperparameters from the checkpoint.
        n_chain: chain-id upper bound; enables ipTM outputs when given.
        noise_fn: forwarded to the sampler to replace its random draws; supplied
            only to compare the rollout against another implementation.
        augment: apply centred random augmentation each rollout step. Turning it
            off is for the same comparison, not for production.
        use_trunk_pair_embedding: feed the trunk pair embedding into the confidence
            re-embedding. Upstream's ``use_zij_trunk_embedding``; when false the
            pair embedding is zeroed there, and only there -- the distogram head
            still reads the unzeroed trunk output.

    Returns:
        A :class:`Prediction`.
    """
    s_input, s_trunk, z = trunk(
        batch,
        params.trunk,
        num_cycles=config.num_cycles,
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

    schedule = noise_schedule(
        config.no_rollout_steps,
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
    s_trunk_s = _expand_samples(s_trunk, config.num_samples)

    # Pair conditioning takes no noise level: it is a function of the batch and the
    # trunk pair representation alone, so it is constant across every rollout step
    # and across samples. Upstream evaluates it inside each denoiser call, which at
    # the released 200 steps means 200 identical evaluations of the widest tensors
    # in the diffusion path -- the relative-position encoding is
    # ``[N_token, N_token, 139]`` and is concatenated onto the pair representation
    # before the projection. Hoisting it out is exact, not an approximation.
    zij_s = _expand_samples(
        pair_conditioning(
            batch,
            z,
            params.diffusion_conditioning,
            max_relative_idx=config.max_relative_idx,
            max_relative_chain=config.max_relative_chain,
            token_mask=batch["token_mask"],
        ),
        config.num_samples,
    )

    def denoise_fn(xl_noisy: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # The single path is the only one that reads the noise level, so it is the
        # only one that has to be rebuilt per step. It is still sample-independent:
        # ``t`` is one value per step, shared by every sample.
        si = _expand_samples(
            single_conditioning(
                s_input,
                s_trunk,
                t,
                params.diffusion_conditioning,
                sigma_data=config.sigma_data,
                token_mask=batch["token_mask"],
            ),
            config.num_samples,
        )
        return denoise(
            sampled_batch,
            xl_noisy,
            t,
            si,
            s_trunk_s,
            zij_s,
            params.denoiser,
            n_query=config.n_query,
            n_key=config.n_key,
            atom_heads=config.atom_heads,
            token_heads=config.token_heads,
            n_token=config.n_token,
            sigma_data=config.sigma_data,
        )

    coordinates = sample_diffusion(
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
                lambda k, xl: centre_random_augmentation(
                    k, xl, sampled_batch["atom_mask"]
                )
            )
            if augment
            else None
        ),
        noise_fn=noise_fn,
    )

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
    z_conf_input = z if use_trunk_pair_embedding else jnp.zeros_like(z)

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
        # pair representation never has to exist at sample rank; only its two
        # logit projections are kept.
        return (
            s_conf_one,
            predicted_aligned_error_head(z_conf_one, params.pae_head),
            predicted_distance_error_head(z_conf_one, params.pde_head),
        )

    # Upstream's ``apply_per_sample``: over ``per_sample_token_cutoff`` tokens the
    # confidence re-embedding is run one sample at a time, because it is the only
    # stage where the samples genuinely differ *and* the tensors are pair-sized.
    # Nothing here mixes samples, so mapping and batching give the same values --
    # the difference is that the batched form holds N_sample pair representations
    # at once, which at 2000 tokens is what decides whether it runs at all.
    if _per_sample_confidence(config):
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

    # pTM needs a per-sample frame mask; broadcast the token mask when the
    # featurizer did not supply one.
    has_frame = batch.get("has_frame")
    if has_frame is None:
        has_frame = jnp.broadcast_to(
            batch["token_mask"][..., :], (config.num_samples, config.n_token)
        )
    pae_for_ptm = jnp.broadcast_to(
        pae_logits, (config.num_samples, *pae_logits.shape[-3:])
    )
    ptm_kwargs = {
        "bin_min": 0.0,
        "bin_max": config.pae_bin_max,
        "no_bins": config.pae_bins,
    }
    token_mask = batch["token_mask"].reshape(-1)[: config.n_token]
    ptm = compute_ptm(pae_for_ptm, has_frame, token_mask, **ptm_kwargs)

    iptm = chain_pair = None
    if n_chain is not None:
        asym_id = batch["asym_id"].reshape(-1)[: config.n_token]
        iptm = compute_ptm(
            pae_for_ptm,
            has_frame,
            token_mask,
            asym_id=asym_id,
            interface=True,
            **ptm_kwargs,
        )
        chain_pair = compute_chain_pair_iptm(
            pae_for_ptm,
            has_frame,
            token_mask,
            asym_id,
            n_chain=n_chain,
            **ptm_kwargs,
        )

    return Prediction(
        coordinates=coordinates,
        plddt=compute_plddt(plddt_logits),
        ptm=ptm,
        iptm=iptm,
        chain_pair_iptm=chain_pair,
        pae_logits=pae_logits,
        pde_logits=pde_logits,
        distogram_logits=disto_logits,
        experimentally_resolved_logits=experimentally_resolved_logits,
    )


def released_config(
    *,
    n_token: int,
    n_atom: int,
    # shared.num_recycles is 3, and upstream runs num_recycles + 1 cycles.
    num_cycles: int = 4,
    num_samples: int = 5,
    no_rollout_steps: int = 200,
    pair_chunk_size: int | None | str = "auto",  # "auto" resolves from n_token
    per_sample_token_cutoff: int | None = 750,
) -> InferenceConfig:
    """Return the released OpenFold3 architecture settings.

    Values are transcribed from upstream's
    ``projects/of3_all_atom/config/model_config.py``, which is the same file the
    released checkpoints were produced under. Only the sizes that depend on the
    input (token/atom counts) and the sampling knobs are arguments.

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
        n_query=32,
        n_key=128,
        atom_heads=4,
        token_heads=16,
        no_heads_msa=8,
        no_heads_pair=4,
        no_heads_pair_bias=16,
        max_relative_idx=32,
        max_relative_chain=2,
        num_cycles=num_cycles,
        num_samples=num_samples,
        max_atoms_per_token=23,
        plddt_bins=50,
        pae_bins=64,
        pae_bin_max=32.0,
        no_rollout_steps=no_rollout_steps,
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
    )


#: Block counts the released architecture is expected to have, for checking a
#: checkpoint against :func:`released_config`.
RELEASED_BLOCK_COUNTS = {
    "pairformer_stack.blocks": 48,
    "msa_module.blocks": 4,
    "diffusion_transformer.blocks": 24,
    "atom_transformer.blocks": 3,
}


def compile_predict(
    config: InferenceConfig,
    representative_atoms: RepresentativeAtomTable,
    *,
    n_chain: int | None = None,
    augment: bool = True,
    use_trunk_pair_embedding: bool = True,
) -> Callable[[jax.Array, Mapping[str, jnp.ndarray], InferenceParams], Prediction]:
    """Return a compiled ``predict`` bound to one configuration.

    Compilation is not optional in practice. Run eagerly, the released
    architecture executes 4 recycling cycles over 48 Pairformer blocks as
    individual dispatches; measured on an RTX PRO 6000 at 16 tokens that is ~69 s
    per call. Compiled, the same call is ~0.1 s, and the released 5-sample
    200-step rollout is ~0.9 s -- the rollout is a ``lax.scan``, so step count
    costs almost nothing at run time.

    The one-time cost is real: ~170 s to compile at 8 rollout steps and ~280 s at
    200. This is a factory rather than a cache inside :func:`predict` so that cost
    is explicit at the call site, and so :func:`predict` stays an ordinary function
    for the parity tests.

    ``params`` stays a runtime argument, so weights can be swapped -- a different
    checkpoint, or averaged weights -- without recompiling. Changing ``config``,
    the batch shapes, or any keyword here does require a new compilation.

    ``noise_fn`` is deliberately not exposed: it exists to replay another
    implementation's random stream in tests, and baking a Python callback into a
    compiled function would defeat the scan.
    """
    return jax.jit(
        functools.partial(
            predict,
            config=config,
            representative_atoms=representative_atoms,
            n_chain=n_chain,
            augment=augment,
            use_trunk_pair_embedding=use_trunk_pair_embedding,
        )
    )
