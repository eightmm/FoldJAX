"""ESMFold2 end to end, from features to coordinates and confidences.

The trunk is the part that is unlike AlphaFold. There is no recycling: the pair
state is a **linear recurrence** -- upstream calls it parcae -- with a learned
per-channel decay,

    z <- a * z + norm(injected) B^T,   then the folding trunk,

run `num_recycles + 1` times from a *randomly initialised* state. Two consequences
follow and neither is a defect to be fixed:

* the model is stochastic at inference. The pair state is drawn from a
  truncated normal, and the LM pair embedding is dropped out at `p = 0.25`
  *per loop* with `training=True`, which upstream's own release path enables.
  Two runs of identical code give different structures.
* the loop body is identical across iterations, so it is a `lax.scan` here
  rather than twenty-one unrolled copies of a twenty-four-layer trunk.

The `msa_encoder_overwrite` flag is likewise upstream's: when set -- and it is,
by default -- the MSA encoder's output *replaces* the injected pair rather than
adding to it, so the language model's contribution reaches the recurrence only
through the encoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from foldjax.models import _capture
from foldjax.models._cp import shard_pair_rows
from foldjax.models._random import masked_prefix_draw
from foldjax.models.esmfold2.models import diffusion
from foldjax.models.esmfold2.models.atom import atom_encoder, one_hot_atom_features
from foldjax.models.esmfold2.models.embedders import (
    inputs_embedder_tail,
    msa_encoder,
    relative_position_encoding,
    single_to_pair,
)
from foldjax.models.esmfold2.models.heads import confidence_head
from foldjax.models.esmfold2.models.primitives import layer_norm, linear
from foldjax.models.esmfold2.models.trunk import folding_trunk

Params = Mapping[str, jnp.ndarray]

#: Upstream's shared constants. The residue alphabet is 33 wide, which is also
#: the width of `profile`, and the atom feature is 3 + 1 + 1 + 128 + 4*64.
NUM_RES_TYPES = 33
MAX_ATOMIC_NUMBER = 128
CHAR_VOCAB_SIZE = 64


@dataclass(frozen=True)
class ModelSettings:
    """Widths and layer counts for the whole model.

    As with `DiffusionSettings`, these are upstream's *dataclass* defaults and
    not the released checkpoint's: the release runs a 48-layer trunk against
    this 24, three loops against twenty, and has its MSA encoder enabled.
    `settings_from_config` reads the real values out of `config.json`, and the
    loader always goes through it -- a port that invents its own defaults runs
    a model nobody trained.
    """

    d_pair: int = 256
    d_inputs: int = 451
    n_residue_bins: int = 32
    n_chain_bins: int = 2
    inputs_atom_n_blocks: int = 3
    inputs_atom_n_heads: int = 4
    inputs_half_window: int = 64
    trunk_n_layers: int = 24
    #: `None` when `lm_encoder.enabled` is false, in which case the LM pair
    #: embedding is added to the injection directly instead of being refined.
    lm_encoder_n_layers: int | None = 4
    lm_dropout: float = 0.25
    per_loop_lm_dropout: bool = True
    coda_n_layers: int = 2
    confidence_n_layers: int = 4
    msa_n_layers: int | None = None
    msa_encoder_overwrite: bool = True
    max_msa_depth: int | None = 1024
    msa_column_mask_rate: float = 0.1
    num_recycles: int = 20
    num_samples: int = 8
    #: Run the confidence head one structure at a time instead of batching
    #: every sample through it. On by default: the released 32-sample
    #: configuration cannot run without it, because at `False` this head builds
    #: `bf16[32, L^2, 2048]` -- 122.8 GiB against a 95.6 GiB device. This note
    #: used to say "off by default", and was correct until that flip.
    #:
    #: Worth asking for near a memory limit: this model's peak is one temp
    #: arena sized `num_samples * L^2 * 4*c_z`, and this head is where the
    #: sample factor enters it. Measured at 1,003 tokens with five samples on
    #: a 96 GB card at the released schedule, peak falls 56.0 to 30.2 GiB and
    #: the arena 35.99 to 11.31, with warm time unchanged -- 28.3-28.6 s
    #: either way. Read that 45% as a property of that measurement rather than
    #: of the option: the arena is quadratic and this divides only its
    #: confidence-head share, so the fraction moves with length and with
    #: sample count. Coordinates are bit-identical
    #: across all five samples; the only visible difference is up to 1e-5 of
    #: pLDDT, which a float32 control collapses to 3.6e-7, so it is bfloat16
    #: reduction-order rounding from the changed matmul shapes rather than
    #: anything about the samples.
    #:
    #: It does not raise the size this model can reach. Sequencing divides
    #: only the head's share of a quadratic arena -- 3.18x here, not the 5x
    #: the sample count suggests -- and dividing a quadratic by a constant
    #: buys its square root in reachable length, about 1.8x. 2,096 tokens
    #: still does not fit on a 96 GB card, and the failure is a named tensor
    #: rather than an inference: the allocator asks for 17,994,612,736 bytes,
    #: which is `bf16[1, 2096, 2096, 2048]` byte for byte -- one sample's
    #: packed pair transition, on top of 64.6 GiB already resident. Batched,
    #: that single buffer would have been five times as large.
    #:
    #: **Everything above was measured at five samples. The release ships
    #: thirty-two, and at thirty-two the batched path does not run at all.**
    #: That inverts the decision, so the default is now on.
    #:
    #: The head runs its own `folding_trunk` over the spread pair
    #: (`heads.py`), and batching puts a sample axis on every one of that
    #: trunk's intermediates. At 1,003 residues and 32 samples, measured as
    #: the allocator's own failing requests:
    #:
    #:     bf16[32, L^2,  512]    30.701 GiB
    #:     bf16[32, L^2, 1024]    61.402 GiB
    #:     bf16[32, L^2, 2048]   122.804 GiB   <- 27.2 GiB past the card
    #:     f32 pae+pde logits     15.350 GiB
    #:
    #: The largest single intermediate needs 122.8 GiB on a 95.6 GiB device,
    #: so this is a capacity impossibility rather than a fragmentation one and
    #: no allocator setting reaches it -- confirmed by failing under both
    #: preallocation on and off, at different points, for the same reason.
    #: Sequencing divides every one of those by the sample count, giving an
    #: 11,312 MiB arena that has run to completion four times.
    #:
    #: So the justification is not that this saves memory. It is that the
    #: released configuration is arithmetically impossible without it. The
    #: saving is a consequence.
    #:
    #: The earlier note that this "does not raise the size this model can
    #: reach" stands for the length axis and is beside the point here: what
    #: was out of reach is the released *sample count* at a length the model
    #: otherwise handles.
    #:
    #: `lax.map` reorders execution and not arithmetic -- `spread` is a
    #: `jnp.repeat` and no operation in the head crosses the sample axis, so
    #: every reduction there is over bins, tokens or atoms. The 1e-5 of pLDDT
    #: is bfloat16 reduction-order rounding from the changed matmul shapes,
    #: and it was never a reason to prefer a configuration that cannot run.
    #:
    #: Protenix carries the same switch and defaults it on
    #: (`models/protenix/models/model.py`) -- corroboration, not the argument.
    #:
    #: One exposure a reader should know about. Rebuilding the batched result
    #: from the mapped one keys on a size-1 second axis: a leaf with one is
    #: squeezed onto it, a leaf without one keeps its first copy. That is
    #: correct for all seventeen leaves this head returns today, checked
    #: shape by shape. It is not future-proof -- a new sample-independent
    #: output whose second axis happened to be size one would be collapsed
    #: silently rather than loudly. Protenix's copy of the rule carries the
    #: same exposure.
    confidence_sample_sequential: bool = True
    #: Return the confidence head's full-bin logits and the PAE/PDE matrices.
    #: False by default, mirroring Boltz-2's flag of the same name and default
    #: -- that argument was had once already on the other port and this is the
    #: same problem, not a different one.
    #:
    #: Nothing reads them. `write_prediction_outputs` consumes three arrays and
    #: four scalars (`sample_atom_coords`, `plddt_per_atom`, `plddt`, and
    #: `SAMPLE_SCORES`); the backend takes its scores from that writer's
    #: summary; `representations` offers only `single` and `pair`; and `raw`
    #: carries overrides, language-model presence and padding. There is no path
    #: by which a caller receives them.
    #:
    #: They are not free to carry. Measured entry outputs at 1,003 tokens were
    #: 2,748 MiB at 5 samples and 16,264 MiB at 32 -- against 2,745 and 16,261
    #: MiB of these arrays computed from their shapes, so they are the whole
    #: term. `pae_logits` and `pde_logits` are `[S, L, L, 64]` float32 and are
    #: 15.36 GiB of the 15.88 at the released 32 samples: quadratic in tokens
    #: and linear in samples.
    #:
    #: The saving comes from their not being *entry outputs of the compiled
    #: program*, which XLA cannot eliminate however unread they are. Dropping
    #: them from the returned dict after the fact would save exactly nothing,
    #: which is why this is a static setting and not a caller-side filter.
    #:
    #: `pae` and `pde` are derived matrices rather than raw logits, and in
    #: tools of this family a PAE matrix is something users actively want. That
    #: they are unread here may describe a missing output feature rather than
    #: pure waste -- so this flag is how you turn them back on, and PAE from
    #: ESMFold2 is one argument away rather than unsupported.
    #:
    #: `distogram_logits` is deliberately NOT under this flag: it is 245 MiB
    #: and flat in sample count, and `test_model_parity` reads it. Excluding it
    #: keeps 98% of the saving without touching a test that cannot run here.
    return_confidence_logits: bool = False
    #: The width of every `TRUNK_PREFIXES` sub-tree. **This is FoldJAX's
    #: choice, not upstream's** -- the comment here used to say otherwise and
    #: was checked against the source on 2026-08-28.
    #:
    #: Upstream has exactly one autocast in the model
    #: (`transformers/models/esmfold2/modeling_esmfold2.py:2021`):
    #:
    #:     use_amp = next(self.esmc.parameters()).dtype == torch.bfloat16
    #:     with torch.autocast(..., dtype=torch.bfloat16, enabled=use_amp):
    #:         esmc_out = self.esmc(...)
    #:
    #: It wraps the ESM-C call alone, and only when ESM-C's own parameters are
    #: already bfloat16. Nothing else is autocast, so the folding trunk, the
    #: inputs embedder and the parcae coda all run at the checkpoint's
    #: parameter dtype -- and the released `config.json` says `float32`.
    #:
    #: bfloat16 is kept, and the A/B that settles it was run on 2026-08-28 at
    #: 1,003 tokens on the released schedule: float32 costs 27.38 s and
    #: 32.84 GiB against bfloat16's 16.78 s and 23.47 GiB, with pLDDT and pTM
    #: identical to four decimals. The structures differ by 0.0992 A all-atom
    #: paired by sample, against a 0.0230-0.0291 A rerun floor and a 0.9135 A
    #: spread between this model's own diffusion samples -- resolvable, and
    #: nine times smaller than the sampling it sits inside.
    #:
    #: Note that no caller can change it: `settings_from_config` does not read
    #: this field and `with_overrides` does not expose it. Flipping it would
    #: change every published ESMFold2 number, so `tests/
    #: test_esmfold2_trunk_width.py` asserts the value rather than trusting
    #: this comment to be read.
    trunk_dtype: str = "bfloat16"
    diffusion: diffusion.DiffusionSettings = field(
        default_factory=diffusion.DiffusionSettings
    )


def settings_from_config(config: Mapping[str, object]) -> ModelSettings:
    """Read a `ModelSettings` out of upstream's `config.json`.

    A mapping, not an `ESMFold2Config`, so nothing on the loading path imports
    torch or transformers; `ESMFold2Config.to_dict()` produces this shape.
    """

    def block(source: Mapping[str, object], name: str) -> Mapping[str, object]:
        value = source.get(name)
        return value if isinstance(value, Mapping) else {}

    base = ModelSettings()
    inputs = block(config, "inputs")
    atom = block(inputs, "atom_encoder")
    lm = block(config, "lm_encoder")
    msa = block(config, "msa_encoder")
    trunk = block(config, "folding_trunk")
    parcae = block(config, "parcae")
    confidence = block(config, "confidence_head")

    def number(source: Mapping[str, object], name: str, fallback: float) -> float:
        value = source.get(name, fallback)
        return fallback if value is None else float(value)  # type: ignore[arg-type]

    return ModelSettings(
        d_pair=int(number(config, "d_pair", base.d_pair)),
        d_inputs=int(number(inputs, "d_inputs", base.d_inputs)),
        n_residue_bins=int(
            number(config, "n_relative_residx_bins", base.n_residue_bins)
        ),
        n_chain_bins=int(number(config, "n_relative_chain_bins", base.n_chain_bins)),
        inputs_atom_n_blocks=int(number(atom, "n_blocks", base.inputs_atom_n_blocks)),
        inputs_atom_n_heads=int(number(atom, "n_heads", base.inputs_atom_n_heads)),
        inputs_half_window=int(
            number(atom, "swa_window_size", base.inputs_half_window * 2)
        )
        // 2,
        trunk_n_layers=int(number(trunk, "n_layers", base.trunk_n_layers)),
        lm_encoder_n_layers=(
            int(number(lm, "n_layers", 4)) if lm.get("enabled", True) else None
        ),
        lm_dropout=number(lm, "lm_dropout", base.lm_dropout),
        per_loop_lm_dropout=bool(lm.get("per_loop_lm_dropout", True)),
        coda_n_layers=int(number(parcae, "coda_n_layers", base.coda_n_layers)),
        confidence_n_layers=int(
            number(
                block(confidence, "folding_trunk"),
                "n_layers",
                base.confidence_n_layers,
            )
        ),
        msa_n_layers=(
            int(number(msa, "n_layers", 4)) if msa.get("enabled", False) else None
        ),
        msa_encoder_overwrite=bool(config.get("msa_encoder_overwrite", True)),
        # ``num_recycles`` is FoldJAX's runtime spelling.  The published
        # checkpoint schema remains upstream's ``num_loops`` and must not be
        # renamed while decoding the file.
        num_recycles=int(number(config, "num_loops", base.num_recycles)),
        num_samples=int(number(config, "num_diffusion_samples", base.num_samples)),
        diffusion=diffusion.settings_from_config(config),
    )


#: The parameters inside upstream's trunk-wide `autocast` at
#: `modeling_esmfold2.py:936`. Everything else -- the distogram head, the
#: structure head, the confidence head -- is float32 there, and the two
#: sub-regions that reopen bfloat16 for themselves are handled where they are.
#: What `return_confidence_logits=False` stops returning. Every one is derived
#: inside the confidence head from tensors the head keeps, so the scalars that
#: *are* read -- `ptm` comes from `softmax(pae_logits)` -- are unaffected: this
#: withholds the arrays, it does not skip the arithmetic.
def rebuild_batched_confidence(
    mapped: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Reshape `lax.map`'s per-sample results into what batching would return.

    Each leaf gained a leading map axis over its single-sample shape. A leaf
    whose next axis is that size-1 sample axis collapses back onto it; a
    sample-independent leaf was produced identically once per sample and only
    one copy belongs in the result.

    Module level rather than inline so the test can exercise *this* function.
    It used to be a dict comprehension inside `run`, and the test that guards
    it held a second copy of the rule -- which passes when the copy is right
    and the original is wrong, the failure mode those two can always produce
    between them.
    """
    return {
        name: (
            jnp.squeeze(value, axis=1)
            if value.ndim > 1 and value.shape[1] == 1
            else value[0]
        )
        for name, value in mapped.items()
    }


CONFIDENCE_LOGIT_OUTPUTS = (
    "pae_logits",
    "pde_logits",
    "plddt_logits",
    "resolved_logits",
    "pae",
    "pde",
)

#: Native results retained by direct callers but unused by the common backend.
#: The writer reads metadata from the original feature mapping and consumes
#: only the sample coordinates, per-atom/token pLDDT and scalar scores. Keeping
#: this projection inside the traced function also lets XLA remove the
#: per-chain matrix calculation instead of merely dropping its device result
#: after dispatch.
MANAGED_AUXILIARY_OUTPUTS = frozenset(
    {
        "atom_pad_mask",
        "residue_index",
        "entity_id",
        "plddt_ca",
        "pair_chains_iptm",
    }
)


def _project_prediction_outputs(
    output: Mapping[str, jnp.ndarray],
    *,
    return_auxiliary_outputs: bool = True,
) -> dict[str, jnp.ndarray]:
    """Select the managed writer result without changing the native default."""
    if return_auxiliary_outputs:
        return dict(output)
    return {
        name: value
        for name, value in output.items()
        if name not in MANAGED_AUXILIARY_OUTPUTS
    }


TRUNK_PREFIXES = (
    "inputs_embedder.",
    "z_init_1.",
    "z_init_2.",
    "rel_pos.",
    "token_bonds.",
    "language_model.",
    "lm_encoder.",
    "msa_encoder.",
    "folding_trunk.",
    "parcae_",
)


def _cast(params: Params, prefixes: tuple[str, ...], dtype) -> dict:
    """The named sub-trees at `dtype`, everything else untouched.

    Casting weights is how a bfloat16 region is expressed here, rather than
    wrapping ops the way torch's autocast does. For matmuls the two agree; for
    normalisations they agree because `layer_norm` takes its statistics in
    float32 whatever dtype it is handed, which is also torch's rule.
    """
    if jnp.dtype(dtype) == jnp.float32:
        return dict(params)
    return {
        name: (
            value.astype(dtype)
            if name.startswith(prefixes) and value.dtype == jnp.float32
            else value
        )
        for name, value in params.items()
    }


def _token_bonds_encoding(
    token_bonds: jnp.ndarray | None,
    params: Params,
    dtype: jnp.dtype,
    *,
    compact_token_bond_encoding: bool = False,
) -> jnp.ndarray:
    """Project token bonds, or return their exact compact signed zeros."""

    if compact_token_bond_encoding:
        weight = params["token_bonds.weight"].astype(dtype)[:, 0]
        return jnp.copysign(jnp.zeros_like(weight), weight)
    if token_bonds is None:
        raise KeyError("token_bonds")
    values = (
        token_bonds.astype(dtype)[..., None]
        if token_bonds.ndim == 3
        else token_bonds.astype(dtype)
    )
    return linear(values, params, "token_bonds")


def _distogram_logits(z: jnp.ndarray, params: Params) -> jnp.ndarray:
    """The native symmetric-pair distogram returned by the direct API."""

    return linear(z + jnp.swapaxes(z, -2, -3), params, "distogram_head")


def inputs_embedding(
    res_type_one_hot: jnp.ndarray,
    profile: jnp.ndarray,
    deletion_mean: jnp.ndarray,
    ref_pos: jnp.ndarray,
    atom_mask: jnp.ndarray,
    ref_space_uid: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_element_one_hot: jnp.ndarray,
    ref_atom_name_chars_one_hot: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    params: Params,
    prefix: str = "inputs_embedder",
    *,
    settings: ModelSettings,
    n_tokens: int,
) -> jnp.ndarray:
    """`InputsEmbedder`: the atom encoder, then three sequence features.

    This encoder has no `coords_linear` -- it is built with
    `structure_prediction=False` -- so its token output is `d_token // 2` wide
    and the concatenation reaches `d_inputs` exactly.
    """
    dot = f"{prefix}." if prefix else ""
    tokens, _, _, _ = atom_encoder(
        ref_pos,
        atom_mask,
        ref_space_uid,
        ref_charge,
        ref_element_one_hot,
        ref_atom_name_chars_one_hot,
        atom_to_token,
        params,
        f"{dot}atom_attention_encoder",
        n_blocks=settings.inputs_atom_n_blocks,
        n_heads=settings.inputs_atom_n_heads,
        half_window=settings.inputs_half_window,
        n_tokens=n_tokens,
    )
    return inputs_embedder_tail(tokens, res_type_one_hot, profile, deletion_mean)


def language_model_embedding(
    hidden_states: jnp.ndarray, params: Params, prefix: str = "language_model"
) -> jnp.ndarray:
    """Project and combine ESMC's layer stack into one token embedding.

    `base_z_combine` is a softmax over the layer axis -- all 81 of ESMC's
    hidden states including the embedding -- so the shim reads the whole stack
    rather than its last layer. Keeping this prefix separate lets a multi-seed
    session retain the small combined result instead of the complete stack.
    """
    dot = f"{prefix}." if prefix else ""
    projected = linear(
        layer_norm(
            hidden_states,
            params[f"{dot}base_z_linear.0.weight"],
            params[f"{dot}base_z_linear.0.bias"],
        ),
        params,
        f"{dot}base_z_linear.1",
    )
    weights = jax.nn.softmax(params[f"{dot}base_z_combine"], axis=0)
    return jnp.einsum("l,bnld->bnd", weights, projected)


def language_model_pair_from_embedding(
    combined: jnp.ndarray, params: Params, prefix: str = "language_model"
) -> jnp.ndarray:
    """Lift one already-combined language-model embedding to a pair."""
    dot = f"{prefix}." if prefix else ""
    pair = single_to_pair(combined, params, f"{dot}base_z_mlp.0")
    return layer_norm(
        pair,
        params[f"{dot}base_z_mlp.1.weight"],
        params[f"{dot}base_z_mlp.1.bias"],
    )


def language_model_pair(
    hidden_states: jnp.ndarray, params: Params, prefix: str = "language_model"
) -> jnp.ndarray:
    """`LanguageModelShim`: combine the PLM's layers, then lift to a pair."""
    return language_model_pair_from_embedding(
        language_model_embedding(hidden_states, params, prefix), params, prefix
    )


def _dropout(
    key: jnp.ndarray,
    x: jnp.ndarray,
    rate: float,
    *,
    valid_mask: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    """`F.dropout(..., training=True)`, which upstream leaves on at inference."""
    if rate <= 0.0:
        return x
    if preserve_prefix_rng:
        if valid_mask is None:
            raise ValueError("prefix-preserving dropout requires a validity mask")
        keep = masked_prefix_draw(
            lambda draw_key, shape: jax.random.bernoulli(draw_key, 1.0 - rate, shape),
            key,
            valid_mask,
            trailing_shape=x.shape[valid_mask.ndim :],
        )
    else:
        # Keep the exact historical call for default, unpadded inference.
        keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def _initial_pair_state_draw(
    key: jnp.ndarray,
    pair_mask: jnp.ndarray,
    width: int,
    *,
    preserve_prefix_rng: bool,
) -> jnp.ndarray:
    """Draw the trunk's initial pair state, optionally preserving real pairs."""

    shape = (*pair_mask.shape, width)
    if not preserve_prefix_rng:
        return jax.random.truncated_normal(key, -3.0, 3.0, shape, dtype=jnp.float32)
    return masked_prefix_draw(
        lambda draw_key, draw_shape: jax.random.truncated_normal(
            draw_key, -3.0, 3.0, draw_shape, dtype=jnp.float32
        ),
        key,
        pair_mask,
        trailing_shape=(width,),
    )


def _msa_column_keep(
    key: jnp.ndarray,
    token_mask: jnp.ndarray,
    rate: float,
    *,
    preserve_prefix_rng: bool,
) -> jnp.ndarray:
    """Draw one column decision per token without exposing the real length."""

    if preserve_prefix_rng:
        values = masked_prefix_draw(
            lambda draw_key, shape: jax.random.uniform(draw_key, shape),
            key,
            token_mask,
        )
    else:
        # Keep the exact historical call for default, unpadded inference.
        values = jax.random.uniform(key, token_mask.shape)
    return values >= rate


def _subsample_msa(
    key: jnp.ndarray, depth: int, max_depth: int | None
) -> jnp.ndarray | None:
    """Row indices for one loop's MSA subsample, query row kept and sorted.

    Unpadded inference retains the released per-loop draw exactly.  The opt-in
    serving path host-normalizes deep alignments to a target no larger than the
    active cap, so this branch is not entered with padded dummy rows.
    """
    if max_depth is None or depth <= 1 or depth <= max_depth:
        return None
    chosen = jax.random.permutation(key, depth - 1)[: max_depth - 1] + 1
    return jnp.sort(jnp.concatenate([jnp.zeros(1, dtype=chosen.dtype), chosen]))


def run_loops(
    key: jnp.ndarray,
    z: jnp.ndarray,
    z_init: jnp.ndarray,
    lm_pair: jnp.ndarray | None,
    msa_inputs: dict[str, jnp.ndarray] | None,
    pair_mask: jnp.ndarray,
    params: Params,
    *,
    settings: ModelSettings,
    total_steps: int,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    """The parcae recurrence, `total_steps` times.

    `a` and `b` come from the log-parameterised continuous dynamics:
    `delta = softplus(log_delta)`, `a = exp(-delta * exp(log_a))`, and
    `B = delta * B_cont`. Reading `log_a` as the decay directly -- the obvious
    misreading -- gives a stable-looking recurrence with the wrong timescale.
    """
    delta = jax.nn.softplus(params["parcae_log_delta"])
    decay = jnp.exp(-delta * jnp.exp(params["parcae_log_a"]))
    decay = decay.reshape(1, 1, 1, -1).astype(z.dtype)
    b_matrix = (delta[:, None] * params["parcae_b_cont"]).astype(z.dtype)

    dropout_on = (
        lm_pair is not None
        and settings.per_loop_lm_dropout
        and settings.lm_dropout > 0.0
    )
    loop_tape = None if msa_inputs is None else msa_inputs.get("loop_tape")
    depth = (
        0
        if msa_inputs is None or loop_tape is not None
        else int(msa_inputs["msa_one_hot"].shape[2])
    )

    def body(carry, loop_inputs):
        z, key = carry
        key, dropout_key, msa_key = jax.random.split(key, 3)

        loop_lm = lm_pair
        if loop_lm is not None and dropout_on:
            loop_lm = _dropout(
                dropout_key,
                loop_lm,
                settings.lm_dropout,
                valid_mask=pair_mask,
                preserve_prefix_rng=preserve_prefix_rng,
            )

        refined = None
        if loop_lm is not None and settings.lm_encoder_n_layers is not None:
            refined = folding_trunk(
                loop_lm,
                params,
                "lm_encoder",
                n_layers=settings.lm_encoder_n_layers,
                mask=pair_mask,
            )

        injected = z_init
        if loop_lm is not None and settings.lm_encoder_n_layers is None:
            injected = injected + loop_lm

        if msa_inputs is not None and settings.msa_n_layers is not None:
            if loop_inputs is None:
                rows = _subsample_msa(msa_key, depth, settings.max_msa_depth)
                one_hot = msa_inputs["msa_one_hot"]
                msa_mask = msa_inputs["msa_mask"]
                has_deletion = msa_inputs["has_deletion"]
                deletion_value = msa_inputs["deletion_value"]
                if rows is not None:
                    one_hot = jnp.take(one_hot, rows, axis=2)
                    msa_mask = jnp.take(msa_mask, rows, axis=2)
                    has_deletion = jnp.take(has_deletion, rows, axis=2)
                    deletion_value = jnp.take(deletion_value, rows, axis=2)
            else:
                loop_msa, loop_mask, loop_has_deletion, loop_deletion = loop_inputs
                one_hot = jax.nn.one_hot(
                    loop_msa.astype(jnp.int32),
                    NUM_RES_TYPES,
                    dtype=z_init.dtype,
                )
                one_hot = jnp.swapaxes(one_hot, 1, 2) * jnp.swapaxes(loop_mask, 1, 2)[
                    ..., None
                ].astype(z_init.dtype)
                msa_mask = jnp.swapaxes(loop_mask, 1, 2)
                # Concatenated onto `one_hot`, which is `z_init.dtype`; a
                # float32 here would widen the whole MSA encoder.
                has_deletion = jnp.swapaxes(loop_has_deletion, 1, 2).astype(
                    z_init.dtype
                )
                deletion_value = jnp.swapaxes(loop_deletion, 1, 2).astype(z_init.dtype)
            msa_pair = msa_encoder(
                injected,
                msa_inputs["x_inputs"],
                one_hot,
                has_deletion,
                deletion_value,
                msa_mask,
                params,
                "msa_encoder",
                n_layers=settings.msa_n_layers,
            )
            if settings.msa_encoder_overwrite:
                injected = msa_pair
            else:
                injected = injected + msa_pair

        if refined is not None:
            injected = injected + refined

        injected = layer_norm(
            injected,
            params["parcae_input_norm.weight"],
            params["parcae_input_norm.bias"],
        )
        z = decay * z + jnp.matmul(injected.astype(z.dtype), b_matrix.T)
        z = folding_trunk(
            z, params, "folding_trunk", n_layers=settings.trunk_n_layers, mask=pair_mask
        )
        return (z, key), None

    (z, _), _ = jax.lax.scan(body, (z, key), loop_tape, length=total_steps)
    return z


def predict(
    key: jnp.ndarray,
    features: Mapping[str, jnp.ndarray],
    params: Params,
    *,
    settings: ModelSettings,
    lm_hidden_states: jnp.ndarray | None = None,
    lm_embedding: jnp.ndarray | None = None,
    initial_pair_state: jnp.ndarray | None = None,
    n_chains: int | None = None,
    preserve_prefix_rng: bool = False,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    contiguous_atom_groups: bool = False,
    compact_token_bond_encoding: bool = False,
    return_distogram_logits: bool = True,
    return_auxiliary_outputs: bool = True,
) -> dict[str, jnp.ndarray]:
    """One full forward, returning upstream's output dictionary.

    `features` uses upstream's own key names, so a featuriser written against
    the torch model feeds this unchanged. `lm_hidden_states` is ESMC's stacked
    layer output. Internal session callers may instead supply the equivalent
    projected-and-combined `lm_embedding`; direct callers retain the historical
    raw-state API. Without either, the language-model branch is absent.

    `initial_pair_state` replaces the truncated-normal draw the trunk starts
    from. Supplying it makes the trunk -- and so the distogram -- reproducible,
    which is the only way to compare this path against torch's at all; the
    diffusion sampler stays stochastic either way.

    `n_chains` sizes the confidence head's per-chain ipTM matrix. It is read
    off `asym_id` when omitted, which is a host read of a traced value and so
    the one thing that would stop this function being jitted; pass it to jit.

    `return_distogram_logits` defaults on because this low-level function is
    the direct parity/debugging API. The common FoldJAX backend turns it off:
    its writer, scores and representation export do not consume that quadratic
    native output.

    `return_auxiliary_outputs` likewise defaults on for the native result
    contract. The managed backend turns it off because its writer reads atom
    and token metadata from `features`, does not consume per-CA pLDDT or the
    chain-pair ipTM matrix, and exports representations through their named
    capture keys.
    """
    token_mask = features["token_attention_mask"]
    atom_mask = features["atom_attention_mask"]
    batch, n_tokens = token_mask.shape
    n_samples = settings.num_samples

    res_type = features["res_type"]
    if res_type.ndim == 2:
        res_type_one_hot = jax.nn.one_hot(
            res_type.astype(jnp.int32), NUM_RES_TYPES
        ) * token_mask[..., None].astype(jnp.float32)
    else:
        res_type_one_hot = res_type.astype(jnp.float32)

    msa = features.get("msa")
    msa_mask = features.get("msa_attention_mask")
    profile = features.get("msa_profile")
    if msa is not None and profile is None:
        msa_one_hot = jax.nn.one_hot(msa.astype(jnp.int32), NUM_RES_TYPES)
        if msa_mask is not None:
            msa_one_hot = msa_one_hot * msa_mask[..., None].astype(jnp.float32)
            counts = jnp.clip(jnp.sum(msa_mask.astype(jnp.float32), axis=1), min=1.0)
            profile = jnp.sum(msa_one_hot, axis=1) / counts[..., None]
        else:
            profile = jnp.mean(msa_one_hot, axis=1)
    if profile is None:
        profile = res_type_one_hot

    deletion_mean = features.get("deletion_mean")
    if deletion_mean is None:
        deletion_mean = jnp.zeros((batch, n_tokens), dtype=jnp.float32)

    element_one_hot, chars_one_hot = one_hot_atom_features(
        features["ref_element"],
        features["ref_atom_name_chars"],
        atom_mask,
        max_atomic_number=MAX_ATOMIC_NUMBER,
        char_vocab=CHAR_VOCAB_SIZE,
    )
    atom_to_token = features["atom_to_token"] * atom_mask.astype(
        features["atom_to_token"].dtype
    )

    # Upstream opens its bfloat16 autocast here and closes it after the coda;
    # `trunk_dtype` is that region and nothing else. Weights are cast rather
    # than ops wrapped, which is the same arithmetic for matmuls and -- because
    # `layer_norm` takes its statistics in float32 whatever it is handed -- the
    # same for normalisations too.
    compute = jnp.dtype(settings.trunk_dtype)
    trunk_params = _cast(params, TRUNK_PREFIXES, compute)

    x_inputs = inputs_embedding(
        res_type_one_hot.astype(compute),
        profile.astype(compute),
        deletion_mean.astype(compute),
        features["ref_pos"].astype(compute),
        atom_mask,
        features["ref_space_uid"],
        features["ref_charge"].astype(compute),
        element_one_hot.astype(compute),
        chars_one_hot.astype(compute),
        atom_to_token,
        trunk_params,
        settings=settings,
        n_tokens=n_tokens,
    )

    # What crosses into the pair path is narrowed here, at the two linears
    # upstream's autocast region covers, because that is the tensor whose width
    # is quadratic in token count. Without this cast one float32 term reaches
    # `z_init = z_init + rel_pos + token_bonds_encoding` below, and the
    # `z.astype(z_init.dtype)` after it carries float32 through all 48 trunk
    # layers -- which is what this port did, while its settings and its
    # released config.json both said bfloat16.
    #
    # The atom encoder upstream of here is deliberately NOT narrowed, but not
    # for the reason it is tempting to give: `rms_norm`'s eps is the Python
    # default `FLOAT32_EPS` (atom.py:38) and `atom_encoder` is called without
    # one, so eps is 1.19e-7 whatever the tensors are. Narrowing it would
    # change rounding only. It is left alone because it buys almost nothing --
    # atom tensors are linear in atom count, not quadratic -- and because
    # upstream at bfloat16 derives a different eps there, so matching it is an
    # upstream-parity question that no test in this tree can currently settle
    # (the torch parity tests skip without torch, and `test_model_parity` pins
    # float32 on purpose). Separately and pre-existing: `atom_features`
    # (atom.py:265) hard-casts its inputs to float32 while its weights are
    # already bfloat16, so that module runs float32 activations against
    # bfloat16 weights. That is the same kind of leak this cast fixes, and it
    # has not been measured.
    pair_inputs = x_inputs.astype(compute)
    z_init = (
        linear(pair_inputs, trunk_params, "z_init_1")[:, :, None, :]
        + linear(pair_inputs, trunk_params, "z_init_2")[:, None, :, :]
    )
    rel_pos = relative_position_encoding(
        features["residue_index"],
        features["asym_id"],
        features["sym_id"],
        features["entity_id"],
        features["token_index"],
        trunk_params,
        "rel_pos",
        n_residue_bins=settings.n_residue_bins,
        n_chain_bins=settings.n_chain_bins,
        dtype=compute,
    )
    token_bonds_encoding = _token_bonds_encoding(
        None if compact_token_bond_encoding else features["token_bonds"],
        trunk_params,
        compute,
        compact_token_bond_encoding=compact_token_bond_encoding,
    )
    # Born sharded under context parallelism: `z_init` re-enters the
    # recurrence every loop, `rel_pos` is reused by the diffusion cache and
    # the confidence head, and the LM pair feeds the lm-encoder trunk.
    rel_pos = shard_pair_rows(rel_pos)
    z_init = shard_pair_rows(z_init + rel_pos + token_bonds_encoding)

    if lm_hidden_states is not None and lm_embedding is not None:
        raise ValueError("pass lm_hidden_states or lm_embedding, not both")
    lm_pair = None
    if lm_embedding is not None:
        lm_pair = shard_pair_rows(
            language_model_pair_from_embedding(
                lm_embedding.astype(compute), trunk_params
            )
        )
    elif lm_hidden_states is not None:
        lm_pair = shard_pair_rows(
            language_model_pair(lm_hidden_states.astype(compute), trunk_params)
        )

    pair_mask = token_mask[:, :, None].astype(jnp.float32) * token_mask[
        :, None, :
    ].astype(jnp.float32)

    key, state_key, column_key, loop_key, sample_key = jax.random.split(key, 5)
    # Upstream draws the initial pair state from a truncated normal; it is not
    # zeros, and two runs of the same code therefore differ.
    if initial_pair_state is None:
        std = (2.0 / (5.0 * z_init.shape[-1])) ** 0.5
        z = std * _initial_pair_state_draw(
            state_key,
            pair_mask,
            z_init.shape[-1],
            preserve_prefix_rng=preserve_prefix_rng,
        )
    else:
        z = initial_pair_state
    z = shard_pair_rows(z.astype(z_init.dtype))

    msa_inputs = None
    if settings.msa_n_layers is not None and msa is not None:
        loop_msa = features.get("msa_loop_tape")
        loop_mask = features.get("msa_attention_mask_loop_tape")
        loop_has_deletion = features.get("has_deletion_loop_tape")
        loop_deletion_value = features.get("deletion_value_loop_tape")
        tape_values = (
            loop_msa,
            loop_mask,
            loop_has_deletion,
            loop_deletion_value,
        )
        if any(value is not None for value in tape_values):
            if any(value is None for value in tape_values):
                raise ValueError(
                    "ESMFold2 padded MSA requires all four per-loop tape features"
                )
            assert loop_msa is not None
            assert loop_mask is not None
            assert loop_has_deletion is not None
            assert loop_deletion_value is not None
            total_steps = max(1, settings.num_recycles + 1)
            if loop_msa.shape[0] != total_steps:
                raise ValueError(
                    "ESMFold2 MSA tape loop count does not match the model: "
                    f"{loop_msa.shape[0]} versus {total_steps}"
                )
            tape_mask = loop_mask
            if settings.msa_column_mask_rate > 0 and loop_msa.shape[2] > 1:
                keep = _msa_column_keep(
                    column_key,
                    token_mask,
                    settings.msa_column_mask_rate,
                    preserve_prefix_rng=preserve_prefix_rng,
                )
                keep = jnp.broadcast_to(keep[None, :, None, :], tape_mask.shape)
                keep = keep.at[:, :, 0, :].set(True)
                tape_mask = tape_mask.astype(bool) & keep
            tape_mask = tape_mask.astype(jnp.float32)
            # Keep the compact integer tape in [loop, batch, row, token]
            # layout. One-hot expansion happens for one loop inside lax.scan,
            # avoiding a loops-times-MSA-times-token temporary.
            loop_tape = (
                loop_msa,
                tape_mask,
                loop_has_deletion,
                loop_deletion_value,
            )
            msa_inputs = {"loop_tape": loop_tape, "x_inputs": pair_inputs}
        else:
            # Keep the historical unpadded graph exactly: column masking is
            # applied to the raw alignment before each loop selects its rows.
            mask = msa_mask
            if (
                mask is not None
                and settings.msa_column_mask_rate > 0
                and msa.shape[1] > 1
            ):
                keep = _msa_column_keep(
                    column_key,
                    token_mask,
                    settings.msa_column_mask_rate,
                    preserve_prefix_rng=preserve_prefix_rng,
                )
                keep = jnp.broadcast_to(keep[:, None, :], mask.shape)
                keep = keep.at[:, 0, :].set(True)
                mask = mask.astype(bool) & keep
            mask = (
                jnp.ones_like(msa, dtype=jnp.float32) if mask is None else mask
            ).astype(jnp.float32)
            # The MSA encoder works token-major; upstream permutes on the way in
            # and zeroes the padding, because its embedding has no bias.
            one_hot = jax.nn.one_hot(
                msa.astype(jnp.int32), NUM_RES_TYPES, dtype=compute
            )
            one_hot = jnp.swapaxes(one_hot, 1, 2) * jnp.swapaxes(mask, 1, 2)[
                ..., None
            ].astype(compute)
            # These two are concatenated onto `msa_one_hot`, so their width is
            # the MSA representation's width for the whole encoder.
            zeros = jnp.zeros(one_hot.shape[:3], dtype=compute)
            msa_inputs = {
                "msa_one_hot": one_hot,
                "msa_mask": jnp.swapaxes(mask, 1, 2),
                "has_deletion": (
                    zeros
                    if features.get("has_deletion") is None
                    else jnp.swapaxes(features["has_deletion"], 1, 2).astype(compute)
                ),
                "deletion_value": (
                    zeros
                    if features.get("deletion_value") is None
                    else jnp.swapaxes(features["deletion_value"], 1, 2).astype(compute)
                ),
                "x_inputs": pair_inputs,
            }

    z = run_loops(
        loop_key,
        z,
        z_init,
        lm_pair,
        msa_inputs,
        pair_mask,
        trunk_params,
        settings=settings,
        total_steps=max(1, settings.num_recycles + 1),
        preserve_prefix_rng=preserve_prefix_rng,
    )
    z = linear(z, trunk_params, "parcae_readout")
    z = folding_trunk(
        z, trunk_params, "parcae_coda", n_layers=settings.coda_n_layers, mask=pair_mask
    )
    # Upstream's `z = z.float()`, which closes the autocast region. Everything
    # after this -- the distogram head, the sampler, the confidence head -- is
    # float32, bar the two sub-regions that open their own bfloat16 block.
    z = shard_pair_rows(z.astype(jnp.float32))
    x_inputs = x_inputs.astype(jnp.float32)
    rel_pos = rel_pos.astype(jnp.float32)
    token_bonds_encoding = token_bonds_encoding.astype(jnp.float32)

    distogram_logits = (
        _distogram_logits(z, params)
        if return_distogram_logits
        else None
    )

    cache = diffusion.build_cache(
        features["ref_pos"],
        features["ref_charge"],
        atom_mask,
        element_one_hot,
        chars_one_hot,
        features["ref_space_uid"],
        atom_to_token,
        z,
        rel_pos,
        params,
        "structure_head.diffusion_module",
        settings=settings.diffusion,
        num_samples=n_samples,
        n_tokens=n_tokens,
        trunk_dtype=compute,
    )

    def draw(sample_key):
        return diffusion.sample(
            sample_key,
            x_inputs,
            cache,
            params,
            "structure_head.diffusion_module",
            settings=settings.diffusion,
            token_mask=token_mask,
            num_samples=n_samples,
            preserve_prefix_rng=preserve_prefix_rng,
        )

    x_inputs = _capture.capture("single", x_inputs)
    z = _capture.capture("pair", z)

    if stop_after_trunk:
        # The representations exist; the sampler and the confidence head are
        # the rest of the run.
        return {
            name: value
            for name, value in (("single", x_inputs), ("pair", z))
            if name in return_representations
        }

    # Every device runs the whole sampler on its own copy, so the region
    # contains no collective. The sibling ports' context-parallel runs went
    # wrong in what their samplers communicated per diffusion step.
    coords, _ = draw(sample_key)

    if n_chains is None:
        n_chains = int(features["asym_id"].max()) + 1
    # ESMFold2 carries one single stream rather than an input embedding and
    # a separate trunk output, so it fills two of the three shared roles. Off
    # by default: the pair state is quadratic in token count.
    representations = {
        name: value
        for name, value in (("single", x_inputs), ("pair", z))
        if name in return_representations
    }
    output = dict(representations)
    if distogram_logits is not None:
        output["distogram_logits"] = distogram_logits
    output.update(
        {
            "sample_atom_coords": coords,
            "atom_pad_mask": atom_mask,
            "residue_index": features["residue_index"],
            "entity_id": features["entity_id"],
        }
    )

    def _confidence(sample_coords: jnp.ndarray, samples: int) -> dict:
        return confidence_head(
            x_inputs,
            z,
            sample_coords,
            params,
            "confidence_head",
            distogram_atom_idx=features["distogram_atom_idx"],
            token_mask=token_mask,
            atom_to_token=atom_to_token,
            atom_mask=atom_mask,
            asym_id=features["asym_id"],
            mol_type=features["mol_type"],
            n_layers=settings.confidence_n_layers,
            n_chains=n_chains,
            trunk_dtype=compute,
            num_samples=samples,
            relative_position_encoding=rel_pos,
            token_bonds_encoding=token_bonds_encoding,
            contiguous_atom_groups=contiguous_atom_groups,
        )

    if settings.confidence_sample_sequential and n_samples > 1:
        # Protenix's `confidence_sample_sequential`, same shape of answer: map
        # over the sample axis with a size-1 sample axis kept, so the head's
        # own shapes are untouched and only the batch factor disappears.
        conf = rebuild_batched_confidence(
            jax.lax.map(lambda one: _confidence(one[None], 1), coords)
        )
    else:
        conf = _confidence(coords, n_samples)
    output.update(conf)
    if not settings.return_confidence_logits:
        # Dropped here, inside the traced function, so they stop being entry
        # outputs of the compiled program. Filtering the dict in the caller
        # would leave the buffers exactly where they are.
        for name in CONFIDENCE_LOGIT_OUTPUTS:
            output.pop(name, None)
    return _project_prediction_outputs(
        output,
        return_auxiliary_outputs=return_auxiliary_outputs,
    )


def with_overrides(
    settings: ModelSettings,
    *,
    num_recycles: int | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    max_msa_depth: int | None = None,
) -> ModelSettings:
    """The four knobs a caller actually varies, applied without reconstruction."""
    updates: dict[str, object] = {}
    if num_recycles is not None:
        updates["num_recycles"] = num_recycles
    if num_samples is not None:
        updates["num_samples"] = num_samples
    if max_msa_depth is not None:
        updates["max_msa_depth"] = max_msa_depth
    if num_steps is not None:
        updates["diffusion"] = replace(settings.diffusion, num_steps=num_steps)
    return replace(settings, **updates)  # type: ignore[arg-type]


__all__ = [
    "ModelSettings",
    "inputs_embedding",
    "language_model_pair",
    "language_model_embedding",
    "language_model_pair_from_embedding",
    "predict",
    "run_loops",
    "settings_from_config",
    "with_overrides",
]
