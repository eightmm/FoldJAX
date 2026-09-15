"""Compare OpenDDE's distributed diffusion atom graph against the serial one.

Run under a forced device count with ``FOLDJAX_CP_PROBE_DEVICES`` and
``FOLDJAX_CP_PROBE_LAYOUT`` set; prints per-stage residuals, the collectives
each program contains, and the lowering fingerprints, then asserts the things
that make the comparison mean something:

* the mesh variant traced a *different* program (fingerprints differ) and saw a
  live mesh and a live atom-window plan (positive tripwires);
* the sharded lowering carries the collectives the mechanism needs and no
  ``all-gather``;
* no full-width atom activation, atom-pair window cache, projected token-pair
  tensor, token pair bias or token attention logit tensor survives per device;
* every stage agrees with serial to FP32 reduction-order tolerance.

A CP parity check is vacuous if the same closure is re-jitted -- ``jit`` keys
its cache on the callable, not on the mesh ContextVar -- so each variant gets a
freshly built closure, and the fingerprint inequality is what proves the
sharded program was traced rather than a cached serial one replayed.

This is not Protenix' probe with the names changed. Three things are OpenDDE's
and are the reason it exists:

* OpenDDE's own ``diffusion_module_forward`` is the entry point, so its
  conditioning runs first and the shared network is entered through
  ``conditioned_single_s`` rather than computing its own;
* OpenDDE always supplies ``extra_attn_bias``, a replicated
  ``[N_token, N_token]`` bias added to token-attention logits whose queries the
  distributed atom graph now delivers CP-row sharded. Protenix never passes it,
  so that composition had no coverage anywhere;
* OpenDDE's sampler is its own (``models/opendde/models/sampling.py``), with a
  different key split and an ``atom_mask`` contract of its own.

Also usable on a GPU with real device counts: it takes its shapes from the
fixtures and its arms from the environment.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

import foldjax.models.protenix.models.primitives.attention as attention_module
from foldjax.models._cp import context_parallel, cp_layout
from foldjax.models.opendde.models.diffusion_module import (
    diffusion_module_f_forward,
    diffusion_module_forward,
)
from foldjax.models.opendde.models.sampling import sample_diffusion
from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_encoder_prepare_diffusion_cache,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    inference_noise_schedule,
)
from tests.models.opendde.atom_cp_fixtures import build_module_case
from tests.models.protenix.atom_cp_fixtures import (
    C_ATOM,
    C_ATOMPAIR,
    C_PAIR,
    N_ATOM,
    N_HEADS,
    N_KEYS,
    N_QUERIES,
    N_SAMPLE,
    N_TOKEN,
    N_WINDOWS,
    SIGMA_DATA,
    TOKEN_HEADS,
)

#: Reduction order changes; nothing else does. The atom<->token mean is a
#: reduce-scatter instead of one local scatter, and the sparse token-pair lookup
#: is a masked sum over rotated tiles, so agreement is to FP32 accumulation
#: noise rather than bitwise.
RELATIVE_TOLERANCE = 1e-5


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _report(name: str, reference: np.ndarray, got: np.ndarray) -> None:
    delta = np.abs(reference - got)
    scale = max(float(np.abs(reference).max()), 1e-12)
    print(
        f"  {name:<10} shape {tuple(reference.shape)} "
        f"max {delta.max():.3e} p99 {np.percentile(delta, 99):.3e} "
        f"rms {np.sqrt((delta**2).mean()):.3e} "
        f"scale {scale:.3f}"
    )
    np.testing.assert_allclose(
        reference,
        got,
        rtol=RELATIVE_TOLERANCE,
        # Scaled by the stage's own magnitude rather than an absolute floor: the
        # sampler's early steps carry coordinates hundreds of units wide and the
        # network update fractions of one, and a single constant would be
        # vacuous for the first and impossible for the second.
        atol=RELATIVE_TOLERANCE * scale,
    )


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    # Read rather than spelled: the adapter resolves an omitted
    # `diffusion_attention_backend` to `xla_jit` under a mesh, so a harness that
    # pinned `xla` here would certify the absence of a difference on the arm the
    # released CP path actually runs. Inside a sharded body `xla_jit` falls back
    # to `xla` -- the plan holds the body's tracers and cannot cross an inner
    # jit -- and that fallback is what this arm exercises.
    backend = os.environ.get("FOLDJAX_CP_PROBE_ATTENTION", "xla")
    # Also read rather than spelled: `graph_jit` -- which context parallelism
    # requires on this port too (`cli/predict.py` refuses `cp_shards > 1`
    # without it) -- sets `use_diffusion_scan` and `use_sampler_scan`, so every
    # real CP run puts both the block stack and the step loop inside `lax.scan`,
    # with the halo's `ppermute` in a scan body and the window mask as a scan
    # constant.
    scan = os.environ.get("FOLDJAX_CP_PROBE_SCAN", "0").strip().lower() in (
        "1",
        "true",
    )
    # A query chunk slices the token-attention axis the atom->token scatter now
    # delivers CP-row sharded; `chunk_policy=auto` emits one at real sizes.
    chunk = int(os.environ.get("FOLDJAX_CP_PROBE_TOKEN_CHUNK", "0")) or None
    assert jax.device_count() == devices, jax.devices()

    params, case, features = build_module_case()
    schedule = inference_noise_schedule(num_steps=3, sigma_data=SIGMA_DATA)
    token_mask = jnp.ones((N_TOKEN,), dtype=bool)
    tripwires: list[tuple[str | None, bool, int, bool]] = []

    def build(distributed: bool):
        """One fresh closure per variant; a reused one would never retrace."""

        def run(s_inputs, s_trunk, pair_z, x_noisy, t_hat, extra_attn_bias):
            z_trunk = jnp.zeros_like(pair_z)
            p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_element,
                case.ref_atom_name_chars,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                jnp.expand_dims(pair_z, axis=-4),
                params.atom_encoder,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                cp_atom_windows=distributed,
            )
            shared = dict(
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                extra_attn_bias=extra_attn_bias,
                n_token=N_TOKEN,
                atom_encoder_heads=N_HEADS,
                token_heads=TOKEN_HEADS,
                atom_decoder_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                sigma_data=SIGMA_DATA,
                attention_backend=backend,
                token_mask=token_mask,
                atom_mask=case.atom_mask,
                use_scan=scan,
                token_q_chunk_size=chunk,
                cp_atom_windows=distributed,
            )
            leading = (
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_atom_name_chars,
                case.ref_element,
                case.d_lm,
                case.v_lm,
                case.pad_info,
            )
            network = diffusion_module_f_forward(
                *leading,
                x_noisy / jnp.sqrt(SIGMA_DATA**2 + t_hat**2)[..., None, None],
                t_hat,
                None,
                s_inputs,
                s_trunk,
                z_trunk,
                params,
                **shared,
            )

            def denoise_fn(noisy, level):
                return diffusion_module_forward(
                    *leading,
                    x_noisy=noisy,
                    t_hat_noise_level=level,
                    relp_feature=None,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    params=params,
                    **shared,
                )

            step = denoise_fn(x_noisy, t_hat)
            sampled = sample_diffusion(
                denoise_fn,
                schedule,
                num_samples=N_SAMPLE,
                n_atom=N_ATOM,
                key=jax.random.key(3),
                batch_shape=(),
                use_scan=scan,
                atom_mask=case.atom_mask,
            )
            return {"network": network, "step": step, "sampler": sampled}

        return run

    def plan_probe(distributed: bool):
        """A one-stage closure whose body reports what it could see."""

        def run(s_inputs, s_trunk, pair_z, x_noisy, t_hat, extra_attn_bias):
            p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_element,
                case.ref_atom_name_chars,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                jnp.expand_dims(pair_z, axis=-4),
                params.atom_encoder,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                cp_atom_windows=distributed,
            )
            seen: list[bool] = []
            # Patched at the one site both arms reach. Patching the transformer
            # stack instead would fire on the serial arm only -- the sharded arm
            # calls it through `_cp.py`'s own namespace -- and an arm that never
            # fires reports "no plan" in exactly the same way as an arm that saw
            # none.
            original = attention_module.local_attention

            def spy(*args, **kwargs):
                seen.append(attention_module.atom_window_plan() is not None)
                return original(*args, **kwargs)

            attention_module.local_attention = spy
            try:
                out = diffusion_module_forward(
                    case.atom_to_token_idx,
                    case.ref_pos,
                    case.ref_charge,
                    case.ref_mask,
                    case.ref_atom_name_chars,
                    case.ref_element,
                    case.d_lm,
                    case.v_lm,
                    case.pad_info,
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    relp_feature=None,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=jnp.zeros_like(pair_z),
                    params=params,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    extra_attn_bias=extra_attn_bias,
                    n_token=N_TOKEN,
                    atom_encoder_heads=N_HEADS,
                    token_heads=TOKEN_HEADS,
                    atom_decoder_heads=N_HEADS,
                    n_queries=N_QUERIES,
                    n_keys=N_KEYS,
                    sigma_data=SIGMA_DATA,
                    attention_backend=backend,
                    token_mask=token_mask,
                    atom_mask=case.atom_mask,
                    use_scan=scan,
                    cp_atom_windows=distributed,
                )
            finally:
                attention_module.local_attention = original
            tripwires.append((cp_layout(), distributed, len(seen), all(seen)))
            return out

        return run

    args = (
        features["s_inputs"],
        features["s_trunk"],
        features["pair_z"],
        features["x_noisy"],
        features["t_hat"],
        features["extra_attn_bias"],
    )

    serial_compiled = jax.jit(build(False))
    reference = jax.device_get(serial_compiled(*args))
    serial_hash = _fingerprint(
        serial_compiled.lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    )
    jax.jit(plan_probe(False))(*args)
    jax.clear_caches()

    with context_parallel(devices, layout=layout):
        compiled = jax.jit(build(True))
        got = jax.device_get(compiled(*args))
        lowered = compiled.lower(*args)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        spmd = lowered.compile().as_text()
        distributed_hash = _fingerprint(hlo)
        jax.jit(plan_probe(True))(*args)

    print(
        f"devices={devices} layout={layout} attention={backend} "
        f"scan={scan} token_chunk={chunk}"
    )
    for name in ("network", "step", "sampler"):
        _report(
            name,
            np.asarray(reference[name], dtype=np.float64),
            np.asarray(got[name], dtype=np.float64),
        )

    low = hlo.lower()
    counts = {
        op: low.count(op)
        for op in (
            "collective-permute",
            "all-gather",
            "all-reduce",
            "reduce-scatter",
            "all-to-all",
        )
    }
    print("  collectives", counts)
    print("  tripwires", tripwires)
    print("  serial_hash", serial_hash[:16], "cp_hash", distributed_hash[:16])

    # Both arms must have reached the patched site the same number of times: a
    # plan reported absent by an arm that never ran the site would be a tripwire
    # that certifies nothing.
    serial_wire, distributed_wire = tripwires
    assert serial_wire[2] > 0 and serial_wire[2] == distributed_wire[2], tripwires
    assert serial_wire[:2] == (None, False) and not serial_wire[3], tripwires
    assert distributed_wire[:2] == (layout, True) and distributed_wire[3], tripwires
    assert serial_hash != distributed_hash, "the mesh variant reused a program"
    # The halo and the two ring rotations are collective permutes; the
    # atom->token mean is a reduce-scatter. A full-key gather would be an
    # `all-gather` over the atom axis, which is the thing being removed.
    assert counts["collective-permute"] > 0, counts
    assert counts["reduce-scatter"] > 0, counts
    assert counts["all-gather"] == 0, counts
    # The column `psum` of the sparse token-pair lookup exists only when there
    # is a column axis to reduce over.
    assert (counts["all-reduce"] > 0) == (layout == "2d"), counts

    full_width = {
        "atom activation": f"f32[{N_SAMPLE},{N_ATOM},{C_ATOM}]",
        "atom-pair window cache": (
            f"f32[1,{N_WINDOWS},{N_QUERIES},{N_KEYS},{C_ATOMPAIR}]"
        ),
        "projected token pair": f"f32[1,{N_TOKEN},{N_TOKEN},{C_PAIR}]",
        # Four dimensions with two full token axes: no atom-path shape can spell
        # that, which is what makes these two unambiguous at
        # N_TOKEN == N_WINDOWS. They are also the two OpenDDE's replicated
        # `extra_attn_bias` could have forced back to full width, which is the
        # composition Protenix' probe never had.
        "token pair bias": f"f32[1,{TOKEN_HEADS},{N_TOKEN},{N_TOKEN}]",
        "token attention logits": (
            f"f32[{N_SAMPLE},{TOKEN_HEADS},{N_TOKEN},{N_TOKEN}]"
        ),
    }
    # Not checked, and deliberately: the token single stream
    # `[N_SAMPLE, N_TOKEN, C_TOKEN]` is linear in the token count and the
    # global token attention still collects its K/V, so a full-width copy of it
    # is expected rather than a defect. It is also unspellable at these widths
    # -- C_TOKEN == C_ATOM and N_TOKEN == N_WINDOWS -- so an assertion on it
    # would be counting atom-path tensors under a token-path name.
    for label, shape in full_width.items():
        assert spmd.count(shape) == 0, (label, shape, spmd.count(shape))
    print("  per-device full-width intermediates", dict.fromkeys(full_width, 0))
    print("OPENDDE_ATOM_CP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
