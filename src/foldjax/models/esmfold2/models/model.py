"""ESMFold2 end to end, from features to coordinates and confidences.

The trunk is the part that is unlike AlphaFold. There is no recycling: the pair
state is a **linear recurrence** -- upstream calls it parcae -- with a learned
per-channel decay,

    z <- a * z + norm(injected) B^T,   then the folding trunk,

run `num_loops + 1` times from a *randomly initialised* state. Two consequences
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
    msa_max_depth: int | None = 1024
    msa_column_mask_rate: float = 0.1
    num_loops: int = 20
    num_samples: int = 8
    #: What upstream's `torch.amp.autocast` makes the trunk on CUDA. Four
    #: regions run in bfloat16 there -- the whole trunk from the inputs
    #: embedder to the parcae coda, the confidence head's folding trunk, the
    #: diffusion conditioning's pair transitions, and the language model --
    #: and everything else is float32. Running the trunk in float32 is not a
    #: safer choice, it is a different model from the one that was trained
    #: and the one every published number describes.
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
        inputs_atom_n_blocks=int(
            number(atom, "n_blocks", base.inputs_atom_n_blocks)
        ),
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
        num_loops=int(number(config, "num_loops", base.num_loops)),
        num_samples=int(number(config, "num_diffusion_samples", base.num_samples)),
        diffusion=diffusion.settings_from_config(config),
    )


#: The parameters inside upstream's trunk-wide `autocast` at
#: `modeling_esmfold2.py:936`. Everything else -- the distogram head, the
#: structure head, the confidence head -- is float32 there, and the two
#: sub-regions that reopen bfloat16 for themselves are handled where they are.
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


def language_model_pair(
    hidden_states: jnp.ndarray, params: Params, prefix: str = "language_model"
) -> jnp.ndarray:
    """`LanguageModelShim`: combine the PLM's layers, then lift to a pair.

    `base_z_combine` is a softmax over the layer axis -- all 81 of ESMC's
    hidden states including the embedding -- so the shim reads the whole stack
    rather than its last layer.
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
    combined = jnp.einsum("l,bnld->bnd", weights, projected)
    pair = single_to_pair(combined, params, f"{dot}base_z_mlp.0")
    return layer_norm(
        pair,
        params[f"{dot}base_z_mlp.1.weight"],
        params[f"{dot}base_z_mlp.1.bias"],
    )


def _dropout(key: jnp.ndarray, x: jnp.ndarray, rate: float) -> jnp.ndarray:
    """`F.dropout(..., training=True)`, which upstream leaves on at inference."""
    if rate <= 0.0:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def _subsample_msa(
    key: jnp.ndarray, depth: int, max_depth: int | None
) -> jnp.ndarray | None:
    """Row indices for one loop's MSA subsample, query row kept and sorted."""
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
    depth = 0 if msa_inputs is None else int(msa_inputs["msa_one_hot"].shape[2])

    def body(carry, _):
        z, key = carry
        key, dropout_key, msa_key = jax.random.split(key, 3)

        loop_lm = lm_pair
        if loop_lm is not None and dropout_on:
            loop_lm = _dropout(dropout_key, loop_lm, settings.lm_dropout)

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
            rows = _subsample_msa(msa_key, depth, settings.msa_max_depth)
            one_hot = msa_inputs["msa_one_hot"]
            msa_mask = msa_inputs["msa_mask"]
            has_deletion = msa_inputs["has_deletion"]
            deletion_value = msa_inputs["deletion_value"]
            if rows is not None:
                one_hot = jnp.take(one_hot, rows, axis=2)
                msa_mask = jnp.take(msa_mask, rows, axis=2)
                has_deletion = jnp.take(has_deletion, rows, axis=2)
                deletion_value = jnp.take(deletion_value, rows, axis=2)
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

    (z, _), _ = jax.lax.scan(body, (z, key), None, length=total_steps)
    return z


def predict(
    key: jnp.ndarray,
    features: Mapping[str, jnp.ndarray],
    params: Params,
    *,
    settings: ModelSettings,
    lm_hidden_states: jnp.ndarray | None = None,
    initial_pair_state: jnp.ndarray | None = None,
    n_chains: int | None = None,
) -> dict[str, jnp.ndarray]:
    """One full forward, returning upstream's output dictionary.

    `features` uses upstream's own key names, so a featuriser written against
    the torch model feeds this unchanged. `lm_hidden_states` is ESMC's stacked
    layer output; without it the language-model branch is simply absent, which
    is what upstream does when no PLM is loaded.

    `initial_pair_state` replaces the truncated-normal draw the trunk starts
    from. Supplying it makes the trunk -- and so the distogram -- reproducible,
    which is the only way to compare this path against torch's at all; the
    diffusion sampler stays stochastic either way.

    `n_chains` sizes the confidence head's per-chain ipTM matrix. It is read
    off `asym_id` when omitted, which is a host read of a traced value and so
    the one thing that would stop this function being jitted; pass it to jit.
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
    if msa is not None:
        msa_one_hot = jax.nn.one_hot(msa.astype(jnp.int32), NUM_RES_TYPES)
        if msa_mask is not None:
            msa_one_hot = msa_one_hot * msa_mask[..., None].astype(jnp.float32)
            counts = jnp.clip(jnp.sum(msa_mask.astype(jnp.float32), axis=1), min=1.0)
            profile = jnp.sum(msa_one_hot, axis=1) / counts[..., None]
        else:
            profile = jnp.mean(msa_one_hot, axis=1)
    else:
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
    atom_to_token = (
        features["atom_to_token"] * atom_mask.astype(features["atom_to_token"].dtype)
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

    z_init = (
        linear(x_inputs, trunk_params, "z_init_1")[:, :, None, :]
        + linear(x_inputs, trunk_params, "z_init_2")[:, None, :, :]
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
    )
    token_bonds_encoding = linear(
        features["token_bonds"].astype(jnp.float32)[..., None]
        if features["token_bonds"].ndim == 3
        else features["token_bonds"].astype(jnp.float32),
        trunk_params,
        "token_bonds",
    )
    z_init = z_init + rel_pos + token_bonds_encoding

    lm_pair = None
    if lm_hidden_states is not None:
        lm_pair = language_model_pair(lm_hidden_states.astype(compute), trunk_params)

    pair_mask = (
        token_mask[:, :, None].astype(jnp.float32)
        * token_mask[:, None, :].astype(jnp.float32)
    )

    key, state_key, column_key, loop_key, sample_key = jax.random.split(key, 5)
    # Upstream draws the initial pair state from a truncated normal; it is not
    # zeros, and two runs of the same code therefore differ.
    if initial_pair_state is None:
        std = (2.0 / (5.0 * z_init.shape[-1])) ** 0.5
        z = std * jax.random.truncated_normal(
            state_key, -3.0, 3.0, z_init.shape, dtype=jnp.float32
        )
    else:
        z = initial_pair_state
    z = z.astype(z_init.dtype)

    msa_inputs = None
    if settings.msa_n_layers is not None and msa is not None:
        mask = msa_mask
        if mask is not None and settings.msa_column_mask_rate > 0 and msa.shape[1] > 1:
            keep = (
                jax.random.uniform(column_key, (batch, n_tokens))
                >= settings.msa_column_mask_rate
            )
            keep = jnp.broadcast_to(keep[:, None, :], mask.shape)
            keep = keep.at[:, 0, :].set(True)
            mask = mask.astype(bool) & keep
        mask = (
            jnp.ones_like(msa, dtype=jnp.float32) if mask is None else mask
        ).astype(jnp.float32)
        # The MSA encoder works token-major; upstream permutes on the way in
        # and zeroes the padding, because its embedding has no bias.
        one_hot = jax.nn.one_hot(msa.astype(jnp.int32), NUM_RES_TYPES, dtype=compute)
        one_hot = jnp.swapaxes(one_hot, 1, 2) * jnp.swapaxes(mask, 1, 2)[
            ..., None
        ].astype(compute)
        zeros = jnp.zeros(one_hot.shape[:3], dtype=jnp.float32)
        msa_inputs = {
            "msa_one_hot": one_hot,
            "msa_mask": jnp.swapaxes(mask, 1, 2),
            "has_deletion": (
                zeros
                if features.get("has_deletion") is None
                else jnp.swapaxes(features["has_deletion"], 1, 2).astype(jnp.float32)
            ),
            "deletion_value": (
                zeros
                if features.get("deletion_value") is None
                else jnp.swapaxes(features["deletion_value"], 1, 2).astype(jnp.float32)
            ),
            "x_inputs": x_inputs,
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
        total_steps=max(1, settings.num_loops + 1),
    )
    z = linear(z, trunk_params, "parcae_readout")
    z = folding_trunk(
        z, trunk_params, "parcae_coda", n_layers=settings.coda_n_layers, mask=pair_mask
    )
    # Upstream's `z = z.float()`, which closes the autocast region. Everything
    # after this -- the distogram head, the sampler, the confidence head -- is
    # float32, bar the two sub-regions that open their own bfloat16 block.
    z = z.astype(jnp.float32)
    x_inputs = x_inputs.astype(jnp.float32)
    rel_pos = rel_pos.astype(jnp.float32)
    token_bonds_encoding = token_bonds_encoding.astype(jnp.float32)

    distogram_logits = linear(
        z + jnp.swapaxes(z, -2, -3), params, "distogram_head"
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
    coords, _ = diffusion.sample(
        sample_key,
        x_inputs,
        cache,
        params,
        "structure_head.diffusion_module",
        settings=settings.diffusion,
        token_mask=token_mask,
        num_samples=n_samples,
    )

    if n_chains is None:
        n_chains = int(features["asym_id"].max()) + 1
    output = {
        "distogram_logits": distogram_logits,
        "sample_atom_coords": coords,
        "atom_pad_mask": atom_mask,
        "residue_index": features["residue_index"],
        "entity_id": features["entity_id"],
    }
    output.update(
        confidence_head(
            x_inputs,
            z,
            coords,
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
            num_samples=n_samples,
            relative_position_encoding=rel_pos,
            token_bonds_encoding=token_bonds_encoding,
        )
    )
    return output


def with_overrides(
    settings: ModelSettings,
    *,
    num_loops: int | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    msa_max_depth: int | None = None,
) -> ModelSettings:
    """The four knobs a caller actually varies, applied without reconstruction."""
    updates: dict[str, object] = {}
    if num_loops is not None:
        updates["num_loops"] = num_loops
    if num_samples is not None:
        updates["num_samples"] = num_samples
    if msa_max_depth is not None:
        updates["msa_max_depth"] = msa_max_depth
    if num_steps is not None:
        updates["diffusion"] = replace(settings.diffusion, num_steps=num_steps)
    return replace(settings, **updates)  # type: ignore[arg-type]


__all__ = [
    "ModelSettings",
    "inputs_embedding",
    "language_model_pair",
    "predict",
    "run_loops",
    "settings_from_config",
    "with_overrides",
]
