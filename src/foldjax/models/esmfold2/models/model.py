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
import numpy as np

from foldjax.models import _capture
from foldjax.models._cp import cp_mesh, shard_pair_rows
from foldjax.models._random import masked_prefix_draw
from foldjax.models.esmfold2.models import diffusion
from foldjax.models.esmfold2.models import trunk as trunk_ops
from foldjax.models.esmfold2.models.atom import atom_encoder, one_hot_atom_features
from foldjax.models.esmfold2.models.embedders import (
    inputs_embedder_tail,
    msa_encoder,
    relative_position_encoding,
    single_to_pair,
)
from foldjax.models.esmfold2.models.heads import confidence_head
from foldjax.models.esmfold2.models.primitives import (
    _cuda_profile_divide,
    layer_norm,
    linear,
)
from foldjax.models.esmfold2.models.trunk import folding_trunk

Params = Mapping[str, jnp.ndarray]

#: Upstream's shared constants. The residue alphabet is 33 wide, which is also
#: the width of `profile`, and the atom feature is 3 + 1 + 1 + 128 + 4*64.
NUM_RES_TYPES = 33
MAX_ATOMIC_NUMBER = 128
CHAR_VOCAB_SIZE = 64

#: The values `ModelSettings.confidence_dtype` accepts.
#:
#: Two exact spellings rather than an alias table. The value joins the
#: compilation-cache identity, so `bf16` alongside `bfloat16` would open a
#: second namespace for one program and answer neither caller out of the
#: other's entry.
CONFIDENCE_DTYPES = ("float32", "bfloat16")


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
    #: Denoise the diffusion samples one at a time instead of together.
    #:
    #: The sibling of the flag above, one stage earlier, and off rather than
    #: on for a reason that is about evidence and not about taste.
    #:
    #: What it divides: the diffusion token transformer's attention logits,
    #: `float32[batch * samples, tokens, tokens, heads]`, rebuilt inside every
    #: block of every denoiser call. At the released 16 heads and 3,012
    #: tokens that is 2.7 GiB at five samples and 17.3 GiB at thirty-two.
    #: `TRUNK_PREFIXES` excludes the diffusion stack from `trunk_dtype`, so
    #: those logits are float32 and stay float32. The blocked atom attention
    #: carries the sample axis too but is linear in atoms rather than
    #: quadratic in tokens, so this one tensor is most of what sequencing the
    #: structure head can save.
    #:
    #: What it does not divide, and the reason this is off: the *trunk* has no
    #: sample axis in this port at all. `run_loops` and `folding_trunk` return
    #: `[batch, tokens, tokens, d_pair]`, and `num_samples` first appears at
    #: `diffusion.build_cache`. The measured 45 GiB peak at 2,096 residues and
    #: the 3,012-residue failure are trunk pair tensors, so a caller running
    #: the released five-sample schedule should not expect this option to move
    #: them. Turning it on for that is not a small win, it is no win.
    #:
    #: Cost: the denoiser is entered `batch * samples` times per step instead
    #: of once, so the per-call work shrinks by that factor and the launch
    #: count grows by it. The arithmetic is the same arithmetic on a narrower
    #: array, which is not bitwise -- a reduction at one row may order itself
    #: differently than at thirty-two.
    #:
    #: `confidence_sample_sequential` is independent of this and stays on:
    #: they narrow two different stages, and either may be set without the
    #: other.
    structure_sample_sequential: bool = False
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
    #: FoldJAX compute policy, not a blanket upstream parameter dtype.
    #: The pinned native source has separate CUDA BF16 autocast regions for
    #: language-model execution, confidence and diffusion pair conditioning.
    #: Their FP32 residual/norm exceptions must survive operand narrowing.
    #: Setting this field to FP32 also changes the latter two regions here;
    #: it is therefore not a pure native trunk-only precision control.
    #: Historical timing/rounded-score agreement does not establish current
    #: fixed-tape structure or raw-confidence parity; see the dated native
    #: tape and MSA-entry observer reports for the outstanding gates.
    trunk_dtype: str = "bfloat16"
    #: The confidence head's re-embedding width, opt-in and independent of
    #: `trunk_dtype`. `"float32"` is the released default and changes nothing.
    #:
    #: `trunk_dtype` already reaches the *inside* of this head: its own
    #: `folding_trunk` reopens upstream's autocast the way upstream does, so
    #: that stack runs with bfloat16 Linear operands under either value. What
    #: stays float32 is the re-embedding in front of it -- the five `s_to_z*`
    #: projections, the distance-bin gather -- and the residual stream those
    #: feed. This field narrows that, following AlphaFold 3's boundary
    #: exactly: narrow re-embedding, float32 output heads
    #: (`confidence_head.py:121-127`, `:163`, `:244`).
    #:
    #: **float32 is upstream's own width, not an accidental island.**
    #: Upstream's `ConfidenceHead.forward` runs outside every autocast region
    #: (`modeling_esmfold2.py:172-221`; the model-level bf16 context at `:936`
    #: closes at `:1030`, before the head is called at `:1061`), and the only
    #: autocast inside the head wraps its `folding_trunk` alone at `:223`,
    #: taking a float32 pair and returning `pair.add_(pair_delta.float())`.
    #: The bfloat16 Linear operands already inside the head's trunk are what
    #: `:223` gives under either value of this field.
    #:
    #: **Measured, and it buys nothing alone.** GPU rows 1111/1112 against the
    #: released default: at 1,003 tokens wall 155.16 -> 160.92 s and peak
    #: 14,733.3 -> 14,733.3 MiB; at 2,096 tokens wall 450.88 -> 450.05 s and
    #: peak 46,041.8 -> 46,042.3 MiB. The arm fires -- the 2,096-token peak
    #: moves 0.5 MiB -- and moves nothing else, because this port's peak is a
    #: folding-trunk arena the confidence head has no term in. So this is a
    #: knob for combination arms, not a recommendation on its own.
    #:
    #: It is still the safest dtype change this port offers, because the
    #: confidence head makes scores and never coordinates. Nothing inside it
    #: can move the structure. Protenix measured the equivalent narrowing at
    #: 3,012 tokens: coordinates bitwise unchanged, atom pLDDT moved at most
    #: 0.0099, chain pTM and ipTM at most 1.9e-4, PAE means at most 0.005.
    #:
    #: Independent of `confidence_sample_sequential`. That option maps the
    #: head over the sample axis; this is a trace-time constant inside the
    #: mapped body, so the two compose without interacting.
    confidence_dtype: str = "float32"
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
    "lm_encoder.",
    "msa_encoder.",
    "folding_trunk.",
    "parcae_",
)


def _cast(params: Params, prefixes: tuple[str, ...], dtype) -> dict:
    """The named sub-trees at `dtype`, everything else untouched.

    This is a storage conversion, not a general autocast implementation.
    In particular FP32 statistics cannot recover rounded affine parameters.
    The LM shim therefore receives original parameters and uses explicit
    per-operation boundaries instead of this converted subtree.
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
    native_autocast: bool = False,
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
        native_autocast=native_autocast,
    )
    return inputs_embedder_tail(tokens, res_type_one_hot, profile, deletion_mean)


def _lm_autocast_linear(x, params, prefix):
    """Native BF16 Linear boundary with its bias applied before output storage."""
    out = jnp.matmul(
        x.astype(jnp.bfloat16),
        params[f"{prefix}.weight"].astype(jnp.bfloat16).T,
        preferred_element_type=jnp.float32,
    )
    bias = params.get(f"{prefix}.bias")
    if bias is not None:
        out = out + bias.astype(jnp.bfloat16).astype(jnp.float32)
    return out.astype(jnp.bfloat16)


def language_model_embedding(
    hidden_states: jnp.ndarray,
    params: Params,
    prefix: str = "language_model",
    *,
    compute_dtype=None,
) -> jnp.ndarray:
    """Project and combine ESMC's layer stack into one token embedding.

    `base_z_combine` is a softmax over the layer axis -- all 81 of ESMC's
    hidden states including the embedding -- so the shim reads the whole stack
    rather than its last layer. Keeping this prefix separate lets a multi-seed
    session retain the small combined result instead of the complete stack.
    ``compute_dtype=bfloat16`` selects the native autocast boundaries while
    retaining original FP32 affine and softmax parameters. None is the direct
    historical helper contract, not an inferred policy from stored weights.
    """
    dot = f"{prefix}." if prefix else ""
    if compute_dtype is not None and jnp.dtype(compute_dtype) == jnp.bfloat16:
        normalized = layer_norm(
            hidden_states.astype(jnp.float32),
            params[f"{dot}base_z_linear.0.weight"],
            params[f"{dot}base_z_linear.0.bias"],
        )
        projected = _lm_autocast_linear(normalized, params, f"{dot}base_z_linear.1")
        weights = jax.nn.softmax(
            params[f"{dot}base_z_combine"].astype(jnp.float32), axis=0
        )
        return jnp.matmul(weights.astype(jnp.bfloat16), projected)
    if compute_dtype is not None:
        hidden_states = hidden_states.astype(compute_dtype)
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
    combined: jnp.ndarray,
    params: Params,
    prefix: str = "language_model",
    *,
    compute_dtype=None,
) -> jnp.ndarray:
    """Lift a combined embedding; native BF16 autocast ends in FP32 LayerNorm."""
    dot = f"{prefix}." if prefix else ""
    if compute_dtype is not None and jnp.dtype(compute_dtype) == jnp.bfloat16:
        name = f"{dot}base_z_mlp.0"
        x = _lm_autocast_linear(combined, params, f"{name}.downproject")
        pair = jnp.concatenate(
            [x[:, :, None, :] * x[:, None, :, :], x[:, :, None, :] - x[:, None, :, :]],
            axis=3,
        )
        pair = _lm_autocast_linear(pair, params, f"{name}.output_mlp.0")
        pair = jax.nn.gelu(pair.astype(jnp.float32), approximate=False).astype(
            jnp.bfloat16
        )
        pair = _lm_autocast_linear(pair, params, f"{name}.output_mlp.2")
        return layer_norm(
            pair.astype(jnp.float32),
            params[f"{dot}base_z_mlp.1.weight"],
            params[f"{dot}base_z_mlp.1.bias"],
        )
    if compute_dtype is not None:
        combined = combined.astype(compute_dtype)
    pair = single_to_pair(combined, params, f"{dot}base_z_mlp.0")
    return layer_norm(
        pair,
        params[f"{dot}base_z_mlp.1.weight"],
        params[f"{dot}base_z_mlp.1.bias"],
    )


def language_model_pair(
    hidden_states: jnp.ndarray,
    params: Params,
    prefix: str = "language_model",
    *,
    compute_dtype=None,
) -> jnp.ndarray:
    """`LanguageModelShim`: combine the PLM's layers, then lift to a pair."""
    return language_model_pair_from_embedding(
        language_model_embedding(
            hidden_states, params, prefix, compute_dtype=compute_dtype
        ),
        params,
        prefix,
        compute_dtype=compute_dtype,
    )


def _dropout(
    key: jnp.ndarray,
    x: jnp.ndarray,
    rate: float,
    *,
    valid_mask: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
    keep_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """`F.dropout(..., training=True)`, which upstream leaves on at inference."""
    if keep_mask is not None:
        if preserve_prefix_rng:
            raise ValueError("native dropout tape does not support prefix RNG padding")
        if not 0.0 < rate < 1.0:
            raise ValueError(
                "dropout keep mask requires a rate strictly between 0 and 1"
            )
        keep_mask = jnp.asarray(keep_mask)
        if keep_mask.dtype != jnp.bool_ or keep_mask.shape != x.shape:
            raise ValueError("dropout keep mask must be boolean and match input shape")
        return _dropout_scale(x, keep_mask, rate)
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
    return _dropout_scale(x, keep, rate)


def _dropout_scale(x, keep, rate):
    # PyTorch CUDA Dropout.cu (cf30153): BF16/FP16 use float acc_type,
    # float keep probability, then float scale multiplication before storage.
    # Preserve the historical FP32 path and all random draws/key splits.
    if x.dtype in (jnp.bfloat16, jnp.float16):
        probability = jnp.asarray(1.0 - rate, jnp.float32)
        scale = jnp.asarray(1.0, jnp.float32) / probability
        return (x.astype(jnp.float32) * keep.astype(jnp.float32) * scale).astype(
            x.dtype
        )
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
    keep_tape: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Draw one column decision per token without exposing the real length."""

    if keep_tape is not None:
        if preserve_prefix_rng:
            raise ValueError("native MSA tape does not support prefix RNG padding")
        return keep_tape
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


def _recurrence_update(z, decay, injected, b_matrix, *, native_rounding=False):
    carried = decay * z
    update = jnp.matmul(injected.astype(z.dtype), b_matrix.T)
    if native_rounding:
        # Torch materializes both BF16 terms before their addition. Without
        # this boundary XLA can fold the sum into GEMM beta=1 and round once.
        carried, update = jax.lax.optimization_barrier((carried, update))
    return carried + update


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
    lm_dropout_masks: jnp.ndarray | None = None,
    msa_row_choices: jnp.ndarray | None = None,
    lm_encoder_params: Params | None = None,
    pair_trunk_params: Params | None = None,
    recurrence_params: Params | None = None,
    injection_norm_params: Params | None = None,
    msa_opm_params: Params | None = None,
) -> jnp.ndarray:
    """The parcae recurrence, `total_steps` times.

    `a` and `b` come from the log-parameterised continuous dynamics:
    `delta = softplus(log_delta)`, `a = exp(-delta * exp(log_a))`, and
    `B = delta * B_cont`. Reading `log_a` as the decay directly -- the obvious
    misreading -- gives a stable-looking recurrence with the wrong timescale.
    """
    if preserve_prefix_rng and (
        lm_dropout_masks is not None or msa_row_choices is not None
    ):
        raise ValueError("native loop tapes do not support prefix RNG padding")
    if lm_dropout_masks is not None:
        if (
            lm_pair is None
            or not settings.per_loop_lm_dropout
            or not 0.0 < settings.lm_dropout < 1.0
        ):
            raise ValueError("LM dropout tape requires an active per-loop LM dropout")
        lm_dropout_masks = jnp.asarray(lm_dropout_masks)
        if lm_dropout_masks.dtype != jnp.bool_:
            raise ValueError("LM dropout tape must have boolean dtype")
        if lm_dropout_masks.shape != (total_steps, *lm_pair.shape):
            raise ValueError("LM dropout tape must have shape [loops, *lm_pair.shape]")
    # Native discretizes original FP32 parameters, then narrows the resulting
    # coefficients once. Casting the parameters first changes both dynamics.
    dynamics = params if recurrence_params is None else recurrence_params
    delta = jax.nn.softplus(dynamics["parcae_log_delta"])
    decay = jnp.exp(-delta * jnp.exp(dynamics["parcae_log_a"]))
    decay = decay.reshape(1, 1, 1, -1).astype(z.dtype)
    b_matrix = (delta[:, None] * dynamics["parcae_b_cont"]).astype(z.dtype)

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
    _, msa_row_choices = validate_msa_tape(
        None,
        msa_row_choices,
        batch=z.shape[0],
        tokens=z.shape[1],
        depth=depth,
        loops=total_steps,
        settings=settings,
        check_values=False,
    )

    def body(carry, loop_inputs):
        dropout_mask = None
        rows_tape = None
        if lm_dropout_masks is not None or msa_row_choices is not None:
            loop_inputs, dropout_mask, rows_tape = loop_inputs
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
                keep_mask=dropout_mask,
            )

        refined = None
        if loop_lm is not None and settings.lm_encoder_n_layers is not None:
            # Native operator policy applies throughout the LM encoder; a
            # teacher-forced first-block parity check is not whole-stack evidence.
            refined = folding_trunk(
                loop_lm.astype(z_init.dtype),
                params if lm_encoder_params is None else lm_encoder_params,
                "lm_encoder",
                n_layers=settings.lm_encoder_n_layers,
                mask=pair_mask,
                native_autocast=lm_encoder_params is not None,
            )

        injected = z_init
        if loop_lm is not None and settings.lm_encoder_n_layers is None:
            injected = injected + loop_lm.astype(injected.dtype)

        if msa_inputs is not None and settings.msa_n_layers is not None:
            if loop_inputs is None:
                rows = (
                    rows_tape
                    if rows_tape is not None
                    else _subsample_msa(msa_key, depth, settings.max_msa_depth)
                )
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
                native_opm_params=msa_opm_params,
            ).astype(injected.dtype)
            if settings.msa_encoder_overwrite:
                injected = msa_pair
            else:
                injected = injected + msa_pair

        if refined is not None:
            injected = injected + refined.astype(injected.dtype)

        if injection_norm_params is not None:
            injected = trunk_ops._autocast_norm(
                injected, injection_norm_params, "parcae_input_norm"
            )
        else:
            injected = layer_norm(
                injected,
                params["parcae_input_norm.weight"],
                params["parcae_input_norm.bias"],
            )
        z = _recurrence_update(
            z,
            decay,
            injected,
            b_matrix,
            native_rounding=pair_trunk_params is not None,
        )
        z = folding_trunk(
            z,
            params if pair_trunk_params is None else pair_trunk_params,
            "folding_trunk",
            n_layers=settings.trunk_n_layers,
            mask=pair_mask,
            native_autocast=pair_trunk_params is not None,
        )
        return (z, key), None

    scan_inputs = (
        (loop_tape, lm_dropout_masks, msa_row_choices)
        if lm_dropout_masks is not None or msa_row_choices is not None
        else loop_tape
    )
    (z, _), _ = jax.lax.scan(body, (z, key), scan_inputs, length=total_steps)
    return z


def validate_msa_tape(
    column, rows, *, batch, tokens, depth, loops, settings, check_values=True
):
    """Native column decisions and sorted query-preserving row indices.

    Inactive native consumers have empty recorder arrays; normalize those to
    None. Preflight concrete arrays before passing dynamic tapes through JIT.
    """
    active = settings.msa_n_layers is not None and depth is not None
    column_active = active and depth > 1 and settings.msa_column_mask_rate > 0
    rows_active = (
        active and settings.max_msa_depth is not None and depth > settings.max_msa_depth
    )
    for value, enabled, dtype, shape in (
        (column, column_active, "bool", (batch, tokens)),
        (rows, rows_active, "integer", (loops, settings.max_msa_depth)),
    ):
        if value is None:
            continue
        kind_ok = (
            value.dtype == jnp.bool_
            if dtype == "bool"
            else jnp.issubdtype(value.dtype, jnp.integer)
        )
        if not kind_ok or value.shape != (shape if enabled else (0,)):
            raise ValueError("MSA tape has wrong dtype/shape or inactive consumer")
    if rows is not None and rows_active:
        if settings.max_msa_depth < 1:
            raise ValueError("MSA tape requires a positive row cap")
        if isinstance(rows, jax.core.Tracer):
            if check_values:
                raise ValueError("MSA tape value preflight requires concrete arrays")
        else:
            array = np.asarray(rows)
            if (
                np.any(array < 0)
                or np.any(array >= depth)
                or np.any(array[:, 0] != 0)
                or np.any(array[:, 1:] <= array[:, :-1])
            ):
                raise ValueError(
                    "MSA tape rows must be sorted unique in-range indices "
                    "retaining query row"
                )
    return (column if column_active else None), (rows if rows_active else None)


def validate_initial_pair_state(value, *, batch, tokens, width, check_values=True):
    """Validate an exact floating pair-state override; preflight before JIT."""
    if value is None:
        return
    if value.shape != (batch, tokens, tokens, width) or not jnp.issubdtype(
        value.dtype, jnp.floating
    ):
        raise ValueError(
            "initial pair state requires exact [batch,tokens,tokens,width] "
            "floating array"
        )
    if isinstance(value, jax.core.Tracer):
        if check_values:
            raise ValueError("initial pair state preflight requires concrete arrays")
    elif not np.isfinite(np.asarray(value)).all():
        raise ValueError("initial pair state must be finite")


def predict(
    key: jnp.ndarray,
    features: Mapping[str, jnp.ndarray],
    params: Params,
    *,
    settings: ModelSettings,
    lm_hidden_states: jnp.ndarray | None = None,
    lm_embedding: jnp.ndarray | None = None,
    initial_pair_state: jnp.ndarray | None = None,
    lm_dropout_masks: jnp.ndarray | None = None,
    msa_column_keep: jnp.ndarray | None = None,
    msa_row_choices: jnp.ndarray | None = None,
    n_chains: int | None = None,
    diffusion_initial_normal: jnp.ndarray | None = None,
    diffusion_rotation_quaternions: jnp.ndarray | None = None,
    diffusion_translations: jnp.ndarray | None = None,
    diffusion_churn_normals: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    stop_after_inputs: bool = False,
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

    ``lm_dropout_masks`` supplies native boolean keep decisions in loop order,
    with shape ``[loops, batch, tokens, tokens, pair_width]``. It requires an
    active LM/dropout branch and does not control MSA or diffusion randomness.

    Native tape inputs cannot be combined with ``preserve_prefix_rng`` padding.
    Before dynamic JIT replay, call ``validate_initial_pair_state``,
    ``validate_msa_tape`` and ``diffusion.validate_diffusion_tape`` on concrete
    arrays: this function checks shapes while tracing, not dynamic values.
    Partial MSA/LM overrides are diagnostic controls, not full native replay;
    inactive LM dropout masks are rejected, not treated as an empty tape.

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
    if preserve_prefix_rng and any(
        value is not None
        for value in (
            initial_pair_state,
            lm_dropout_masks,
            msa_column_keep,
            msa_row_choices,
            diffusion_initial_normal,
            diffusion_rotation_quaternions,
            diffusion_translations,
            diffusion_churn_normals,
        )
    ):
        raise ValueError("native tapes do not support prefix RNG padding")
    validate_initial_pair_state(
        initial_pair_state,
        batch=batch,
        tokens=n_tokens,
        width=settings.d_pair,
        check_values=False,
    )
    msa_column_keep, msa_row_choices = validate_msa_tape(
        msa_column_keep,
        msa_row_choices,
        batch=batch,
        tokens=n_tokens,
        depth=features["msa"].shape[1] if "msa" in features else None,
        loops=max(1, settings.num_recycles + 1),
        settings=settings,
        check_values=False,
    )
    if msa_row_choices is not None and features.get("msa_loop_tape") is not None:
        raise ValueError(
            "MSA row index tape cannot be combined with preselected loop features"
        )
    if msa_column_keep is not None and features.get("msa_attention_mask") is None:
        raise ValueError("MSA column tape requires native msa_attention_mask")
    diffusion.validate_diffusion_tape(
        diffusion_initial_normal,
        diffusion_rotation_quaternions,
        diffusion_translations,
        diffusion_churn_normals,
        steps=len(diffusion.noise_schedule(settings.diffusion)) - 1,
        batch=batch * n_samples,
        atoms=atom_mask.shape[-1],
        check_values=False,
    )

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
            totals = jnp.sum(msa_one_hot, axis=1)
            if settings.trunk_dtype == "bfloat16" and cp_mesh() is None:
                profile = jax.lax.platform_dependent(
                    totals,
                    counts[..., None],
                    cuda=_cuda_profile_divide,
                    default=lambda a, b: a / b,
                )
            else:
                profile = totals / counts[..., None]
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
    # `trunk_dtype` targets that region, but blanket parameter conversion is
    # not autocast: FP32 norm statistics cannot restore rounded affine weights.
    # Pair trunks use original weights with per-operation boundaries on the
    # single-device BF16 path. Other consumers still need their own audit.
    compute = jnp.dtype(settings.trunk_dtype)
    trunk_params = _cast(params, TRUNK_PREFIXES, compute)
    native_pair_autocast = compute == jnp.bfloat16 and cp_mesh() is None
    native_pair_params = (
        {
            key: value
            for key, value in params.items()
            if key.startswith(("folding_trunk.", "parcae_coda."))
        }
        if native_pair_autocast
        else None
    )

    # These three features bypass the atom encoder and are concatenated onto
    # its result. Native retains them in FP32; autocast does not narrow cat.
    sequence_dtype = jnp.float32 if native_pair_autocast else compute
    x_inputs = inputs_embedding(
        res_type_one_hot.astype(sequence_dtype),
        profile.astype(sequence_dtype),
        deletion_mean.astype(sequence_dtype),
        features["ref_pos"].astype(sequence_dtype),
        atom_mask,
        features["ref_space_uid"],
        features["ref_charge"]
        if native_pair_autocast
        else features["ref_charge"].astype(compute),
        element_one_hot.astype(sequence_dtype),
        chars_one_hot.astype(sequence_dtype),
        atom_to_token,
        params if native_pair_autocast else trunk_params,
        settings=settings,
        n_tokens=n_tokens,
        native_autocast=native_pair_autocast,
    )

    if stop_after_inputs:
        return (
            {"single_inputs": x_inputs}
            if "single_inputs" in return_representations
            else {}
        )

    # What crosses into the pair path is narrowed here, at the two linears
    # upstream's autocast region covers, because that is the tensor whose width
    # is quadratic in token count. Without this cast one float32 term reaches
    # `z_init = z_init + rel_pos + token_bonds_encoding` below, and the
    # `z.astype(z_init.dtype)` after it carries float32 through all 48 trunk
    # layers -- which is what this port did, while its settings and its
    # released config.json both said bfloat16.
    #
    # The native atom path instead preserves FP32 reference features, norms
    # and residuals, with BF16 boundaries at its Linear and attention outputs.
    # Its FP32 sequence tail makes the concatenation FP32, so this pair-input
    # cast is still necessary even when the atom's token output is BF16.
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
                lm_embedding, params, compute_dtype=compute
            )
        )
    elif lm_hidden_states is not None:
        lm_pair = shard_pair_rows(
            language_model_pair(lm_hidden_states, params, compute_dtype=compute)
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
                    keep_tape=msa_column_keep,
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
                    keep_tape=msa_column_keep,
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
        lm_dropout_masks=lm_dropout_masks,
        msa_row_choices=msa_row_choices,
        lm_encoder_params=(
            {
                key: value
                for key, value in params.items()
                if key.startswith("lm_encoder.")
            }
            if jnp.dtype(compute) == jnp.bfloat16
            else None
        ),
        pair_trunk_params=native_pair_params,
        msa_opm_params=(
            {
                key: value
                for key, value in params.items()
                if key.startswith("msa_encoder.blocks.")
            }
            if native_pair_autocast
            else None
        ),
        recurrence_params=(
            {
                key: value
                for key, value in params.items()
                if key in {"parcae_log_delta", "parcae_log_a", "parcae_b_cont"}
            }
            if native_pair_autocast
            else None
        ),
        injection_norm_params=(
            {
                key: value
                for key, value in params.items()
                if key.startswith("parcae_input_norm.")
            }
            if native_pair_autocast
            else None
        ),
    )
    z = linear(z, trunk_params, "parcae_readout")
    z = folding_trunk(
        z,
        trunk_params if native_pair_params is None else native_pair_params,
        "parcae_coda",
        n_layers=settings.coda_n_layers,
        mask=pair_mask,
        native_autocast=native_pair_autocast,
    )
    # Upstream's `z = z.float()`, which closes the autocast region. Everything
    # after this -- the distogram head, the sampler, the confidence head -- is
    # float32, bar the two sub-regions that open their own bfloat16 block.
    z = shard_pair_rows(z.astype(jnp.float32))
    x_inputs = x_inputs.astype(jnp.float32)
    rel_pos = rel_pos.astype(jnp.float32)
    token_bonds_encoding = token_bonds_encoding.astype(jnp.float32)

    distogram_logits = _distogram_logits(z, params) if return_distogram_logits else None

    sequential_samples = settings.structure_sample_sequential and n_samples > 1
    if sequential_samples and batch != 1:
        # A rollout would have to be told which input it belongs to, and the
        # only place that lives is the pair conditioning -- the largest tensor
        # the sampler holds. Slicing it per rollout, inside the loop this
        # option exists to narrow, would cost more than the option saves.
        # Nothing in this port builds a batch above one; a caller who does is
        # told rather than handed the wrong rows.
        raise ValueError(
            "structure_sample_sequential runs one input at a time; this call "
            f"has batch {batch}"
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
        # The atom-level entries are repeated to `batch * num_samples` here.
        # Sequencing narrows them to one rollout, which is the other half of
        # the saving: a batched cache would keep a factor the mapped denoiser
        # cannot use.
        num_samples=1 if sequential_samples else n_samples,
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
            diffusion_initial_normal=diffusion_initial_normal,
            diffusion_rotation_quaternions=diffusion_rotation_quaternions,
            diffusion_translations=diffusion_translations,
            diffusion_churn_normals=diffusion_churn_normals,
            sample_sequential=sequential_samples,
        )

    x_inputs = _capture.capture("single", x_inputs)
    z = _capture.capture("pair", z)

    if stop_after_trunk:
        # The representations exist; the sampler and the confidence head are
        # the rest of the run.
        return {
            name: value
            for name, value in (
                ("single_inputs", x_inputs),
                ("single", x_inputs),
                ("pair", z),
            )
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
        for name, value in (
            ("single_inputs", x_inputs),
            ("single", x_inputs),
            ("pair", z),
        )
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
            confidence_dtype=jnp.dtype(settings.confidence_dtype),
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
    structure_sample_sequential: bool | None = None,
    glu_backend: str | None = None,
    confidence_dtype: str | None = None,
) -> ModelSettings:
    """The knobs a caller actually varies, applied without reconstruction.

    All optional, and `None` means "leave the checkpoint's value alone" --
    which for the boolean is the difference between a caller who did not ask
    and one who asked for the default.
    """
    updates: dict[str, object] = {}
    if num_recycles is not None:
        updates["num_recycles"] = num_recycles
    if structure_sample_sequential is not None:
        updates["structure_sample_sequential"] = structure_sample_sequential
    if num_samples is not None:
        updates["num_samples"] = num_samples
    if max_msa_depth is not None:
        updates["max_msa_depth"] = max_msa_depth
    if confidence_dtype is not None:
        updates["confidence_dtype"] = confidence_dtype
    # Collected, then applied once: two separate `replace` calls on
    # `settings.diffusion` would each read the *original* sub-settings, so the
    # second assignment to `updates["diffusion"]` would drop the first.
    diffusion_updates: dict[str, object] = {}
    if num_steps is not None:
        diffusion_updates["num_steps"] = num_steps
    if glu_backend is not None:
        diffusion_updates["glu_backend"] = glu_backend
    if diffusion_updates:
        updates["diffusion"] = replace(settings.diffusion, **diffusion_updates)  # type: ignore[arg-type]
    return replace(settings, **updates)  # type: ignore[arg-type]


__all__ = [
    "CONFIDENCE_DTYPES",
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
