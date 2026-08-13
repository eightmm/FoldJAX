"""The diffusion structure head: conditioning, the token transformer, the sampler.

Written against `docs/ports/esmfold2/diffusion-spec.md`. The denoiser is an
ordinary EDM-preconditioned network, but the sampler around it is not the ODE
it resembles, and three of its details are load-bearing:

* the Karras schedule is **clipped** at `max_inference_sigma`, so a nominal
  68-step schedule runs 48 denoiser calls -- the tail above the cap is dropped
  and the cap prepended, not rescaled;
* between the denoiser call and the Euler step, `x_noisy` is **Kabsch-aligned
  onto `x_denoised`**. Nothing downstream reveals its absence: the structures
  stay plausible and stay wrong;
* the churn term draws noise at an inflated `t_hat` and the network is queried
  there, yet with the released `noise_scale = 0` the noise itself is zero. The
  inflation is real, the noise is not, and dropping either one changes the
  answer.

Sampling is a `lax.scan` over the schedule rather than an unrolled loop --
forty-eight copies of a twelve-block transformer is a compile nobody wants --
so the per-step sigmas arrive as scanned inputs. The one arithmetic
consequence is the churn standard deviation: `sqrt(t_hat^2 - sigma^2)` loses
its cancellation as a traced expression and can differentiate to NaN, so it is
written through the identity `t_hat = sigma (1 + g)` as
`sigma * sqrt(g^2 + 2g)`, which never subtracts two near-equal numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.esmfold2.models.atom import (
    FLOAT32_EPS,
    atom_conditioning,
    atom_decoder,
    atom_encoder,
)
from foldjax.models.esmfold2.models.primitives import (
    adaptive_layer_norm,
    fourier_embedding,
    layer_norm,
    linear,
    transition_layer,
)

Params = Mapping[str, jnp.ndarray]


@dataclass(frozen=True)
class DiffusionSettings:
    """Every width and constant the structure head reads, in one place.

    The sampler alone takes fifteen of them, which is past the point where
    keyword arguments stop being readable.

    The defaults here are upstream's *dataclass* defaults, and the released
    checkpoint does not use them: its `config.json` asks for fourteen steps
    rather than sixty-eight, `p = 7` rather than 8, `gamma_0 = 0.8`,
    `step_scale = 1.5`, and a churn of `noise_scale = 1.003` where the
    dataclass says zero. Anything that means to reproduce the release must go
    through `settings_from_config`; constructing this bare gives a model that
    runs and is not the released one.
    """

    sigma_data: float = 16.0
    c_atom: int = 128
    c_token: int = 768
    c_z: int = 256
    atom_n_blocks: int = 3
    atom_n_heads: int = 4
    token_n_blocks: int = 12
    token_n_heads: int = 16
    #: `swa_window_size // 2`, the reach on each side, in packed atom rank.
    half_window: int = 64
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.0
    step_scale: float = 1.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 8.0
    num_steps: int = 68
    max_inference_sigma: float | None = 256.0


@dataclass(frozen=True)
class DiffusionCache:
    """What every denoiser call in a sampling run shares.

    Upstream rebuilds most of this per step and caches it under
    `inference_cache`; the sampler here builds it once and closes over it, which
    is the same arithmetic and keeps the scanned step function small.

    The atom-level entries are already repeated to `batch * samples`; `pair` is
    not, because it is the largest tensor in the model and the attention adds
    its bias against a folded view rather than a repeated one.
    """

    atom_conditioning: jnp.ndarray
    cos: jnp.ndarray
    sin: jnp.ndarray
    pair: jnp.ndarray
    atom_to_token: jnp.ndarray
    atom_mask: jnp.ndarray
    n_tokens: int


def attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_heads: int,
    mask: jnp.ndarray | None = None,
    num_samples: int = 1,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`AttentionPairBias`.

    Two things here are not the usual arrangement. The softmax runs over the
    *key* axis while the layout keeps heads last, so it is `axis=-2` rather
    than `-1`; and the output gate reads the **raw** conditioning `s`, not the
    normalised copy the adaptive layer norm made -- normalising it twice is
    the plausible reading and silently rescales every block's output.

    When several diffusion samples share one pair tensor, the bias is added
    against a folded view of the logits instead of a repeated `z`: the pair
    tensor is `L^2 * c_z` and repeating it is the single largest allocation
    this model can be made to do.
    """
    dot = f"{prefix}." if prefix else ""
    batch, n_queries, width = a.shape
    head_dim = width // n_heads

    if s is not None:
        x = adaptive_layer_norm(a, s, params, f"{dot}adaln", eps=eps)
    else:
        x = layer_norm(
            a,
            params[f"{dot}pre_norm.weight"],
            params[f"{dot}pre_norm.bias"],
            eps=eps,
        )

    query = linear(x, params, f"{dot}q_proj").reshape(
        batch, n_queries, n_heads, head_dim
    )
    packed = linear(x, params, f"{dot}kv_proj")
    key = packed[..., :width].reshape(batch, -1, n_heads, head_dim)
    value = packed[..., width:].reshape(batch, -1, n_heads, head_dim)
    n_keys = key.shape[1]

    logits = jnp.einsum("bihd,bjhd->bijh", query, key) * (head_dim**-0.5)

    def add_over_samples(base: jnp.ndarray, term: jnp.ndarray) -> jnp.ndarray:
        """Add a term whose batch axis may be `num_samples` times shorter."""
        if term.shape[0] == batch:
            return base + term
        folded = base.reshape(term.shape[0], -1, n_queries, n_keys, n_heads)
        return (folded + term[:, None]).reshape(batch, n_queries, n_keys, n_heads)

    if z.ndim == 4:
        pair = layer_norm(
            z, params[f"{dot}pair_norm.weight"], params[f"{dot}pair_norm.bias"], eps=eps
        )
        bias = linear(pair, params, f"{dot}pair_bias_proj")
    else:
        bias = z[..., None]
    logits = add_over_samples(logits, bias.astype(logits.dtype))

    if mask is not None:
        keep = mask.astype(bool)[:, None, :, None]
        floor = jnp.where(keep, 0.0, jnp.finfo(logits.dtype).min)
        logits = add_over_samples(logits, floor.astype(logits.dtype))

    attention = jax.nn.softmax(logits, axis=-2).astype(value.dtype)
    context = jnp.einsum("bijh,bjhd->bihd", attention, value)
    gate = jax.nn.sigmoid(linear(x, params, f"{dot}g_proj")).reshape(
        batch, n_queries, n_heads, head_dim
    )
    out = linear(
        (gate * context).reshape(batch, n_queries, width), params, f"{dot}out_proj"
    )
    if s is not None:
        out = jax.nn.sigmoid(linear(s, params, f"{dot}out_gate")) * out
    return out


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    params: Params,
    prefix: str = "",
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`ConditionedTransitionBlock`: adaLN, packed SwiGLU, gate on raw `s`."""
    dot = f"{prefix}." if prefix else ""
    if s is not None:
        x = adaptive_layer_norm(a, s, params, f"{dot}adaln", eps=eps)
    else:
        x = layer_norm(
            a,
            params[f"{dot}pre_norm.weight"],
            params[f"{dot}pre_norm.bias"],
            eps=eps,
        )
    packed = linear(x, params, f"{dot}lin_swish")
    half = packed.shape[-1] // 2
    out = linear(
        jax.nn.silu(packed[..., :half]) * packed[..., half:], params, f"{dot}lin_out"
    )
    if s is not None:
        out = jax.nn.sigmoid(linear(s, params, f"{dot}output_gate")) * out
    return out


def diffusion_transformer(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_blocks: int,
    n_heads: int,
    mask: jnp.ndarray | None = None,
    num_samples: int = 1,
) -> jnp.ndarray:
    """`DiffusionTransformer`: attention and transition, both residual."""
    dot = f"{prefix}." if prefix else ""
    for index in range(n_blocks):
        a = a + attention_pair_bias(
            a,
            s,
            z,
            params,
            f"{dot}attn_blocks.{index}",
            n_heads=n_heads,
            mask=mask,
            num_samples=num_samples,
        )
        a = a + conditioned_transition_block(
            a, s, params, f"{dot}transition_blocks.{index}"
        )
    return a


def condition_pair(
    z_trunk: jnp.ndarray,
    relative_position_encoding: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    trunk_dtype: object = jnp.float32,
) -> jnp.ndarray:
    """`DiffusionConditioning`'s pair half, which no timestep enters.

    Upstream caches it across the whole sampling run for that reason; it is
    separate here so a caller cannot accidentally recompute it per step.
    """
    dot = f"{prefix}." if prefix else ""
    z = jnp.concatenate(
        [
            z_trunk.astype(jnp.float32),
            relative_position_encoding.astype(jnp.float32),
        ],
        axis=-1,
    )
    z = linear(
        layer_norm(
            z,
            params[f"{dot}z_input_norm.weight"],
            params[f"{dot}z_input_norm.bias"],
        ),
        params,
        f"{dot}z_proj",
    )
    # Upstream reopens a bfloat16 autocast around these two transitions alone.
    # The residual is float32 either way: `z + block(z)` promotes.
    for index in range(2):
        block = {
            name: (
                value.astype(trunk_dtype)
                if name.startswith(f"{dot}z_transitions") and value.dtype == jnp.float32
                else value
            )
            for name, value in params.items()
        }
        z = z + transition_layer(
            z.astype(trunk_dtype), block, f"{dot}z_transitions.{index}"
        ).astype(jnp.float32)
    return z


def condition_single(
    t_hat: jnp.ndarray,
    s_inputs: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    sigma_data: float,
    num_samples: int = 1,
) -> jnp.ndarray:
    """`DiffusionConditioning`'s single half, which is rebuilt every step.

    The timestep enters as `0.25 * log(t / sigma_data)` -- a quarter, not a
    half, and clamped rather than shifted, so `t = 0` maps to the clamp floor
    instead of to negative infinity.
    """
    dot = f"{prefix}." if prefix else ""
    if s_inputs.shape[0] * num_samples == t_hat.shape[0]:
        s_inputs = jnp.repeat(s_inputs, num_samples, axis=0)
    s = linear(
        layer_norm(
            s_inputs.astype(jnp.float32),
            params[f"{dot}s_input_norm.weight"],
            params[f"{dot}s_input_norm.bias"],
        ),
        params,
        f"{dot}s_proj",
    )

    t_noise = 0.25 * jnp.log(
        jnp.clip(t_hat.astype(jnp.float32) / sigma_data, min=1e-20)
    )
    noise = fourier_embedding(t_noise, params, f"{dot}fourier")
    noise = linear(
        layer_norm(
            noise,
            params[f"{dot}noise_norm.weight"],
            params[f"{dot}noise_norm.bias"],
        ),
        params,
        f"{dot}noise_proj",
    )
    s = s + noise[:, None, :]
    for index in range(2):
        s = s + transition_layer(s, params, f"{dot}s_transitions.{index}")
    return s


def build_cache(
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_element_one_hot: jnp.ndarray,
    ref_atom_name_chars_one_hot: jnp.ndarray,
    ref_space_uid: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    z_trunk: jnp.ndarray,
    relative_position_encoding: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    settings: DiffusionSettings,
    num_samples: int = 1,
    n_tokens: int | None = None,
    trunk_dtype: object = jnp.float32,
) -> DiffusionCache:
    """Everything a sampling run can compute before its first denoiser call."""
    dot = f"{prefix}." if prefix else ""
    conditioning, cos, sin = atom_conditioning(
        ref_pos,
        ref_mask,
        ref_space_uid,
        ref_charge,
        ref_element_one_hot,
        ref_atom_name_chars_one_hot,
        params,
        f"{dot}atom_encoder",
        n_heads=settings.atom_n_heads,
    )
    if n_tokens is None:
        n_tokens = int(atom_to_token.max()) + 1

    def spread(x: jnp.ndarray) -> jnp.ndarray:
        return x if num_samples == 1 else jnp.repeat(x, num_samples, axis=0)

    return DiffusionCache(
        atom_conditioning=spread(conditioning),
        cos=spread(cos),
        sin=spread(sin),
        pair=condition_pair(
            z_trunk,
            relative_position_encoding,
            params,
            f"{dot}conditioning",
            trunk_dtype=trunk_dtype,
        ),
        atom_to_token=spread(atom_to_token),
        atom_mask=spread(ref_mask),
        n_tokens=n_tokens,
    )


def diffusion_module(
    x_noisy: jnp.ndarray,
    t_hat: jnp.ndarray,
    s_inputs: jnp.ndarray,
    cache: DiffusionCache,
    params: Params,
    prefix: str = "",
    *,
    settings: DiffusionSettings,
    token_mask: jnp.ndarray | None = None,
    num_samples: int = 1,
    eps: float = FLOAT32_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`DiffusionModule`: one denoiser call, returning `(x_denoised, token_repr)`.

    EDM preconditioning, written the way upstream writes it rather than the way
    the paper does: the input scale is `1 / sqrt(t^2 + sigma_d^2)` and the two
    output coefficients are `sigma_d^2 / (sigma_d^2 + t^2)` on the noisy input
    and `sigma_d t / sqrt(sigma_d^2 + t^2)` on the network's update.
    """
    dot = f"{prefix}." if prefix else ""
    sigma = settings.sigma_data
    t = jnp.reshape(t_hat.astype(jnp.float32), (-1,))
    if t.shape[0] == 1 and x_noisy.shape[0] != 1:
        t = jnp.broadcast_to(t, (x_noisy.shape[0],))

    s = condition_single(
        t,
        s_inputs,
        params,
        f"{dot}conditioning",
        sigma_data=sigma,
        num_samples=num_samples,
    )

    r_noisy = x_noisy / jnp.sqrt(t * t + sigma * sigma)[:, None, None]

    tokens, queries, conditioning, rope = atom_encoder(
        None,
        cache.atom_mask,
        None,
        None,
        None,
        None,
        cache.atom_to_token,
        params,
        f"{dot}atom_encoder",
        n_blocks=settings.atom_n_blocks,
        n_heads=settings.atom_n_heads,
        half_window=settings.half_window,
        coords=r_noisy,
        precomputed=(cache.atom_conditioning, cache.cos, cache.sin),
        n_tokens=cache.n_tokens,
        eps=eps,
    )

    tokens = tokens + linear(
        layer_norm(
            s, params[f"{dot}s_step_norm.weight"], params[f"{dot}s_step_norm.bias"]
        ),
        params,
        f"{dot}s_to_token",
    )
    tokens = diffusion_transformer(
        tokens,
        s,
        cache.pair,
        params,
        f"{dot}token_transformer",
        n_blocks=settings.token_n_blocks,
        n_heads=settings.token_n_heads,
        mask=token_mask,
        num_samples=num_samples,
    )
    tokens = layer_norm(
        tokens, params[f"{dot}token_norm.weight"], params[f"{dot}token_norm.bias"]
    )

    update = atom_decoder(
        tokens,
        queries,
        conditioning,
        cache.atom_to_token,
        cache.atom_mask,
        params,
        f"{dot}atom_decoder",
        rope=rope,
        n_blocks=settings.atom_n_blocks,
        n_heads=settings.atom_n_heads,
        half_window=settings.half_window,
        eps=eps,
    )

    sigma2, t2 = sigma * sigma, t * t
    denoised = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
    denoised = denoised + ((sigma * t) / jnp.sqrt(sigma2 + t2))[:, None, None] * update
    return denoised, tokens


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def noise_schedule(settings: DiffusionSettings) -> np.ndarray:
    """The Karras power schedule, clipped, on the host.

    Every value in it is known before the run, so it is built in NumPy and the
    sampler reads Python floats out of it -- which is also what upstream does,
    via `.item()` on each entry.

    The clip is not a rescale: entries above `max_inference_sigma` are
    *dropped* and the cap prepended in their place, so a 68-entry schedule
    becomes 49 entries and 48 denoiser calls.
    """
    steps = int(settings.num_steps)
    if steps == 1:
        schedule = np.array(
            [settings.s_max * settings.sigma_data, 0.0], dtype=np.float32
        )
    else:
        inv_p = 1.0 / float(settings.p)
        k = np.arange(steps, dtype=np.float32)
        base = settings.s_max**inv_p + (k / (steps - 1)) * (
            settings.s_min**inv_p - settings.s_max**inv_p
        )
        schedule = (settings.sigma_data * base ** float(settings.p)).astype(np.float32)
        schedule = np.concatenate([schedule, np.zeros(1, dtype=np.float32)])

    cap = settings.max_inference_sigma
    if cap is not None:
        schedule = schedule[schedule <= float(cap)]
        schedule = np.concatenate([np.full(1, float(cap), dtype=np.float32), schedule])
    return schedule


def quaternion_to_rotation(q: jnp.ndarray) -> jnp.ndarray:
    """Upstream's quaternion-to-matrix, sign convention included.

    The quaternion is normalised by a *signed* magnitude -- negative when its
    real part is -- which is what makes the map single-valued over the double
    cover. Dividing by the plain magnitude gives a valid rotation too, just not
    the same one for half the draws.
    """
    scale = jnp.sqrt(jnp.sum(q * q, axis=1))
    signs = jnp.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / jnp.sum(q * q, axis=-1)
    return jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    ).reshape(q.shape[0], 3, 3)


def center_random_augmentation(
    key: jnp.ndarray,
    x: jnp.ndarray,
    atom_mask: jnp.ndarray,
    second: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Centre on the masked centroid, then rotate and translate at random.

    The translation is one vector per structure, not per atom: upstream draws
    `randn_like(x[:, 0:1, :])` and broadcasts it.
    """
    mask = atom_mask[..., None]
    mean = jnp.sum(x * mask, axis=1, keepdims=True) / jnp.clip(
        jnp.sum(mask, axis=1, keepdims=True), min=1.0
    )
    x = x - mean
    if second is not None:
        second = second - mean

    rotation_key, translation_key = jax.random.split(key)
    rotation = quaternion_to_rotation(
        jax.random.normal(rotation_key, (x.shape[0], 4), dtype=x.dtype)
    )
    x = jnp.einsum("bmd,bds->bms", x, rotation)
    if second is not None:
        second = jnp.einsum("bmd,bds->bms", second, rotation)

    shift = jax.random.normal(
        translation_key, (x.shape[0], 1, x.shape[2]), dtype=x.dtype
    )
    x = x + shift
    if second is not None:
        second = second + shift
    return x, second


def weighted_rigid_align(
    x: jnp.ndarray, x_gt: jnp.ndarray, weights: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """Kabsch: the rigid motion taking `x` onto `x_gt`, in float32.

    Called between every denoiser call and its Euler step. The reflection
    correction is the usual one -- the third singular direction carries
    `det(U V^T)` -- and it matters here because a reflected "alignment" would
    still reduce the residual and would still look like a structure.
    """
    w = (mask * weights).astype(jnp.float32)[..., None]
    x = x.astype(jnp.float32)
    x_gt = x_gt.astype(jnp.float32)
    denominator = jnp.clip(jnp.sum(w, axis=-2, keepdims=True), min=1e-8)
    mu = jnp.sum(x * w, axis=-2, keepdims=True) / denominator
    mu_gt = jnp.sum(x_gt * w, axis=-2, keepdims=True) / denominator
    x_centred = x - mu
    gt_centred = x_gt - mu_gt

    correlation = jnp.einsum("bni,bnj->bij", w * gt_centred, x_centred)
    u, _, vh = jnp.linalg.svd(correlation)
    determinant = jnp.linalg.det(u @ vh)
    ones = jnp.ones_like(determinant)
    flip = jnp.stack([ones, ones, determinant], axis=-1)
    rotation = u @ (flip[..., None] * vh)
    return x_centred @ jnp.swapaxes(rotation, -1, -2) + mu_gt


def _step(
    carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    step: jnp.ndarray,
    *,
    denoise: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    atom_mask: jnp.ndarray,
    settings: DiffusionSettings,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
    """One denoise-align-step, shared by the scanned and the eager driver.

    `step` is `(sigma_from, sigma_to, gamma)`; under `lax.scan` all three are
    tracers, which is what makes the churn identity necessary rather than
    merely tidy. The token representation rides in the carry rather than
    coming out as a scanned output: only the last step's is wanted, and
    stacking all forty-eight would cost `steps * L * c_token` for nothing.
    """
    x, previous, key, _ = carry
    sigma_from, sigma_to, gamma = step[0], step[1], step[2]

    key, augmentation_key, churn_key = jax.random.split(key, 3)
    x, previous = center_random_augmentation(augmentation_key, x, atom_mask, previous)

    t_hat = sigma_from * (1.0 + gamma)
    # sigma * sqrt(g^2 + 2g) == sqrt(t_hat^2 - sigma^2) without the
    # cancellation; zero throughout with the released noise_scale of 0.
    churn = settings.noise_scale * sigma_from * jnp.sqrt(gamma * gamma + 2.0 * gamma)
    x_noisy = x + churn * jax.random.normal(churn_key, x.shape, dtype=x.dtype)

    x_denoised, token_repr = denoise(
        x_noisy, jnp.broadcast_to(jnp.asarray(t_hat, jnp.float32), (x.shape[0],))
    )

    x_noisy = weighted_rigid_align(x_noisy, x_denoised, atom_mask, atom_mask).astype(
        x_denoised.dtype
    )
    slope = (x_noisy - x_denoised) / t_hat
    x = x_noisy + settings.step_scale * (sigma_to - t_hat) * slope
    return (x, x_denoised, key, token_repr), None


def sample(
    key: jnp.ndarray,
    s_inputs: jnp.ndarray,
    cache: DiffusionCache,
    params: Params,
    prefix: str = "",
    *,
    settings: DiffusionSettings,
    token_mask: jnp.ndarray | None = None,
    num_samples: int = 1,
    early_exit_rmsd: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Algorithm 18, returning `(sample_atom_coords, token_repr)`.

    The default path is a `lax.scan` over the schedule. `early_exit_rmsd` --
    upstream's convergence break, which its own forward never enables --
    compares two consecutive predictions and stops, which is a host-side
    branch on a traced value; asking for it runs the same step function in a
    Python loop instead, and the result cannot be jitted.
    """
    schedule = noise_schedule(settings)
    gammas = np.where(schedule > settings.gamma_min, settings.gamma_0, 0.0)
    steps = np.stack([schedule[:-1], schedule[1:], gammas[1:]], axis=-1)

    atom_mask = cache.atom_mask.astype(jnp.float32)
    batch, n_atoms = atom_mask.shape

    key, initial_key = jax.random.split(key)
    x = float(schedule[0]) * jax.random.normal(
        initial_key, (batch, n_atoms, 3), dtype=jnp.float32
    )

    def denoise(
        x_noisy: jnp.ndarray, t_hat: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return diffusion_module(
            x_noisy,
            t_hat,
            s_inputs,
            cache,
            params,
            prefix,
            settings=settings,
            token_mask=token_mask,
            num_samples=num_samples,
        )

    def run(carry, step):
        return _step(
            carry, step, denoise=denoise, atom_mask=atom_mask, settings=settings
        )

    # `lax.scan` needs a fixed carry structure, so the previous prediction and
    # the token representation start as zeros rather than absent. Nothing reads
    # the previous prediction on the first step; the augmentation merely
    # transforms it alongside `x`, exactly as upstream does with its `None`.
    n_tokens, c_token = cache.n_tokens, settings.c_token
    carry = (
        x,
        jnp.zeros_like(x),
        key,
        jnp.zeros((batch, n_tokens, c_token), dtype=jnp.float32),
    )

    if early_exit_rmsd is None:
        carry, _ = jax.lax.scan(run, carry, jnp.asarray(steps, dtype=jnp.float32))
        return carry[0], carry[3]

    for index in range(steps.shape[0]):
        previous = carry[1]
        carry, _ = run(carry, jnp.asarray(steps[index], dtype=jnp.float32))
        x_denoised, token_repr = carry[1], carry[3]
        if index >= 1:
            aligned = weighted_rigid_align(previous, x_denoised, atom_mask, atom_mask)
            difference = (x_denoised - aligned) * atom_mask[..., None]
            rmsd = jnp.sqrt(
                jnp.sum(difference**2, axis=(-1, -2))
                / jnp.clip(jnp.sum(atom_mask, axis=-1), min=1.0)
            )
            if float(jnp.max(rmsd)) < early_exit_rmsd:
                return x_denoised, token_repr
    return carry[0], carry[3]


def settings_from_config(config: Mapping[str, object]) -> DiffusionSettings:
    """Read a `DiffusionSettings` out of upstream's `config.json`.

    A mapping rather than an `ESMFold2Config`, so the loader needs neither
    torch nor transformers; `ESMFold2Config.to_dict()` produces exactly this
    shape for anyone who has one.

    The released checkpoint departs from the dataclass defaults in most of
    these fields -- fourteen steps rather than sixty-eight, `p = 7`, and a
    churn of `1.003` where the dataclass says zero -- which is why nothing here
    is assumed and every field is read.
    """
    head: Mapping[str, object] = config.get("structure_head", {})  # type: ignore[assignment]
    module: Mapping[str, object] = head.get("diffusion_module", {})  # type: ignore[assignment]
    inputs: Mapping[str, object] = config.get("inputs", {})  # type: ignore[assignment]
    atom: Mapping[str, object] = inputs.get("atom_encoder", {})  # type: ignore[assignment]
    base = DiffusionSettings()

    def read(source: Mapping[str, object], name: str, fallback: object) -> object:
        value = source.get(name, fallback)
        return fallback if value is None else value

    return replace(
        base,
        sigma_data=float(read(module, "sigma_data", base.sigma_data)),  # type: ignore[arg-type]
        c_atom=int(read(module, "c_atom", base.c_atom)),  # type: ignore[call-overload]
        c_token=int(read(module, "c_token", base.c_token)),  # type: ignore[call-overload]
        c_z=int(read(module, "c_z", base.c_z)),  # type: ignore[call-overload]
        atom_n_blocks=int(read(module, "atom_num_blocks", base.atom_n_blocks)),  # type: ignore[call-overload]
        atom_n_heads=int(read(module, "atom_num_heads", base.atom_n_heads)),  # type: ignore[call-overload]
        token_n_blocks=int(read(module, "token_num_blocks", base.token_n_blocks)),  # type: ignore[call-overload]
        token_n_heads=int(read(module, "token_num_heads", base.token_n_heads)),  # type: ignore[call-overload]
        half_window=int(read(atom, "swa_window_size", base.half_window * 2)) // 2,  # type: ignore[call-overload]
        gamma_0=float(read(head, "gamma_0", base.gamma_0)),  # type: ignore[arg-type]
        gamma_min=float(read(head, "gamma_min", base.gamma_min)),  # type: ignore[arg-type]
        noise_scale=float(read(head, "noise_scale", base.noise_scale)),  # type: ignore[arg-type]
        step_scale=float(read(head, "step_scale", base.step_scale)),  # type: ignore[arg-type]
        s_max=float(read(head, "inference_s_max", base.s_max)),  # type: ignore[arg-type]
        s_min=float(read(head, "inference_s_min", base.s_min)),  # type: ignore[arg-type]
        p=float(read(head, "inference_p", base.p)),  # type: ignore[arg-type]
        num_steps=int(read(head, "inference_num_steps", base.num_steps)),  # type: ignore[call-overload]
    )


__all__ = [
    "DiffusionCache",
    "DiffusionSettings",
    "attention_pair_bias",
    "build_cache",
    "center_random_augmentation",
    "condition_pair",
    "condition_single",
    "conditioned_transition_block",
    "diffusion_module",
    "diffusion_transformer",
    "noise_schedule",
    "quaternion_to_rotation",
    "sample",
    "settings_from_config",
    "weighted_rigid_align",
]
