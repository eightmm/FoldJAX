"""Compare Protenix' distributed diffusion atom graph against the serial one.

Run under a forced device count with ``FOLDJAX_CP_PROBE_DEVICES`` and
``FOLDJAX_CP_PROBE_LAYOUT`` set; prints per-stage residuals, the collectives
each program contains, and the lowering fingerprints, then asserts the things
that make the comparison mean something:

* the mesh variant traced a *different* program (fingerprints differ) and saw
  a live mesh and a live atom-window plan (positive tripwires);
* the sharded lowering carries the collectives the mechanism needs and no
  ``all-gather``;
* no full-width atom activation, atom-pair window cache, or projected
  token-pair tensor survives per device;
* every stage agrees with serial to FP32 reduction-order tolerance.

A CP parity check is vacuous if the same closure is re-jitted -- ``jit`` keys
its cache on the callable, not on the mesh ContextVar -- so each variant gets
a freshly built closure, and the fingerprint inequality is what proves the
sharded program was traced rather than a cached serial one replayed.

Also usable on a GPU with real device counts: it takes its shapes from
``tests.models.protenix.atom_cp_fixtures`` and its arms from the environment.
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
from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_decoder,
    atom_attention_encoder,
    atom_attention_encoder_prepare_diffusion_cache,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    diffusion_module_forward,
    inference_noise_schedule,
    sample_diffusion_with_module,
)
from tests.models.protenix.atom_cp_fixtures import (
    C_ATOM,
    C_ATOMPAIR,
    C_PAIR,
    N_ATOM,
    N_HEADS,
    N_KEYS,
    N_PAIR_FEATURES,
    N_QUERIES,
    N_SAMPLE,
    N_TOKEN,
    N_WINDOWS,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_module_case,
)

#: Reduction order changes; nothing else does. The atom<->token mean is a
#: reduce-scatter instead of one local scatter, and the sparse token-pair
#: lookup is a masked sum over rotated tiles, so agreement is to FP32
#: accumulation noise rather than bitwise.
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
        # Scaled by the stage's own magnitude rather than an absolute floor:
        # the sampler's early steps carry coordinates two thousand units wide
        # and the decoder update fractions of one, and a single constant would
        # be vacuous for the first and impossible for the second.
        atol=RELATIVE_TOLERANCE * scale,
    )


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    # Read rather than spelled: the adapter resolves an omitted
    # `diffusion_attention_backend` to `xla_jit` under a mesh, so a harness
    # that pinned `xla` here would certify the absence of a difference on the
    # arm the released CP path actually runs. Inside a sharded body `xla_jit`
    # falls back to `xla` -- the plan holds the body's tracers and cannot
    # cross an inner jit -- and that fallback is what this arm exercises.
    backend = os.environ.get("FOLDJAX_CP_PROBE_ATTENTION", "xla")
    # Also read rather than spelled, and for the same reason: `graph_jit` --
    # which context parallelism requires -- sets `use_diffusion_scan=True`, so
    # every real CP run puts the block stack inside `lax.scan` *inside* the
    # `shard_map` body, with the halo's `ppermute` in the scan body and the
    # plan's window mask as a scan constant. A harness pinned to the unrolled
    # stack would never trace that.
    scan = os.environ.get("FOLDJAX_CP_PROBE_SCAN", "0").strip().lower() in (
        "1",
        "true",
    )
    # A query chunk slices the token-attention axis the atom->token scatter
    # now delivers CP-row sharded; `chunk_policy=auto` emits one at real
    # sizes, so the composition is not hypothetical.
    chunk = int(os.environ.get("FOLDJAX_CP_PROBE_TOKEN_CHUNK", "0")) or None
    assert jax.device_count() == devices, jax.devices()

    params, case, features = build_module_case()
    schedule = inference_noise_schedule(num_steps=3, sigma_data=SIGMA_DATA)
    token_mask = jnp.ones((N_TOKEN,), dtype=bool)
    feats = {
        "atom_to_token_idx": case.atom_to_token_idx,
        "atom_padding_mask": case.atom_mask,
        "token_padding_mask": token_mask,
        "ref_pos": case.ref_pos,
        "ref_charge": case.ref_charge,
        "ref_mask": case.ref_mask,
        "ref_atom_name_chars": case.ref_atom_name_chars,
        "ref_element": case.ref_element,
        "d_lm": case.d_lm,
        "v_lm": case.v_lm,
        "pad_info": case.pad_info,
        "relp": jnp.zeros((N_TOKEN, N_TOKEN, N_PAIR_FEATURES), dtype=jnp.float32),
    }
    z_trunk = jnp.zeros((N_TOKEN, N_TOKEN, C_PAIR), dtype=jnp.float32)
    tripwires: list[tuple[str | None, bool, int, bool]] = []

    def build(distributed: bool):
        """One fresh closure per variant; a reused one would never retrace."""

        def run(s_inputs, s_trunk, pair_z, x_noisy, t_hat):
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
            a, q_skip, c_skip, p_skip = atom_attention_encoder(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_atom_name_chars,
                case.ref_element,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                params.atom_encoder,
                r_l=x_noisy,
                s=jnp.expand_dims(s_trunk, axis=-3),
                z=jnp.expand_dims(pair_z, axis=-4),
                p_lm=p_lm,
                c_l=c_l,
                n_token=N_TOKEN,
                n_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                attention_backend=backend,
                atom_mask=case.atom_mask,
                use_scan=scan,
                cp_atom_windows=distributed,
            )
            # Read the tripwires where the sharded body actually is: inside
            # the encoder they are set only for the duration of its
            # `shard_map`, so this records the outer mesh and an inner plan
            # observed from a stage that has one.
            update = atom_attention_decoder(
                case.atom_to_token_idx,
                a,
                q_skip,
                c_skip,
                p_skip,
                params.atom_decoder,
                n_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                attention_backend=backend,
                atom_mask=case.atom_mask,
                use_scan=scan,
                cp_atom_windows=distributed,
            )
            step = diffusion_module_forward(
                case.atom_to_token_idx,
                case.ref_pos,
                case.ref_charge,
                case.ref_mask,
                case.ref_atom_name_chars,
                case.ref_element,
                case.d_lm,
                case.v_lm,
                case.pad_info,
                x_noisy,
                t_hat,
                feats["relp"],
                s_inputs,
                s_trunk,
                z_trunk,
                params,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                n_token=N_TOKEN,
                atom_encoder_heads=N_HEADS,
                token_heads=TOKEN_HEADS,
                atom_decoder_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                sigma_data=SIGMA_DATA,
                attention_backend=backend,
                glu_backend="xla",
                token_mask=token_mask,
                atom_mask=case.atom_mask,
                use_scan=scan,
                token_q_chunk_size=chunk,
                cp_atom_windows=distributed,
            )
            sampled = sample_diffusion_with_module(
                feats,
                s_inputs,
                s_trunk,
                z_trunk,
                params,
                schedule,
                num_samples=N_SAMPLE,
                key=jax.random.key(3),
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                atom_encoder_heads=N_HEADS,
                token_heads=TOKEN_HEADS,
                atom_decoder_heads=N_HEADS,
                n_queries=N_QUERIES,
                n_keys=N_KEYS,
                sigma_data=SIGMA_DATA,
                attention_backend=backend,
                glu_backend="xla",
                use_scan=scan,
                use_sampler_scan=scan,
                token_q_chunk_size=chunk,
                cp_atom_windows=distributed,
            )
            return {
                "encoder_token": a,
                "encoder_atom": q_skip,
                "encoder_pair": p_skip,
                "decoder": update,
                "step": step,
                "sampler": sampled,
            }

        return run

    def plan_probe(distributed: bool):
        """A one-stage closure whose body reports what it could see."""

        def run(r_l, s_trunk, pair_z):
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
            # Patched at the one site both arms reach. Patching the
            # transformer stack instead would have fired on the serial arm
            # only -- the sharded arm calls it through `_cp.py`'s own
            # namespace -- and an arm that never fires reports "no plan" in
            # exactly the same way as an arm that saw none.
            original = attention_module.local_attention

            def spy(*args, **kwargs):
                seen.append(attention_module.atom_window_plan() is not None)
                return original(*args, **kwargs)

            attention_module.local_attention = spy
            try:
                out = atom_attention_encoder(
                    case.atom_to_token_idx,
                    case.ref_pos,
                    case.ref_charge,
                    case.ref_mask,
                    case.ref_atom_name_chars,
                    case.ref_element,
                    case.d_lm,
                    case.v_lm,
                    case.pad_info,
                    params.atom_encoder,
                    r_l=r_l,
                    s=jnp.expand_dims(s_trunk, axis=-3),
                    z=jnp.expand_dims(pair_z, axis=-4),
                    p_lm=p_lm,
                    c_l=c_l,
                    n_token=N_TOKEN,
                    n_heads=N_HEADS,
                    n_queries=N_QUERIES,
                    n_keys=N_KEYS,
                    attention_backend=backend,
                    atom_mask=case.atom_mask,
                    use_scan=scan,
                    cp_atom_windows=distributed,
                )[0]
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
    )
    plan_args = (features["x_noisy"], features["s_trunk"], features["pair_z"])

    serial_compiled = jax.jit(build(False))
    reference = jax.device_get(serial_compiled(*args))
    serial_hash = _fingerprint(
        serial_compiled.lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    )
    jax.jit(plan_probe(False))(*plan_args)
    jax.clear_caches()

    with context_parallel(devices, layout=layout):
        compiled = jax.jit(build(True))
        got = jax.device_get(compiled(*args))
        lowered = compiled.lower(*args)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        spmd = lowered.compile().as_text()
        distributed_hash = _fingerprint(hlo)
        jax.jit(plan_probe(True))(*plan_args)

    print(
        f"devices={devices} layout={layout} attention={backend} "
        f"scan={scan} token_chunk={chunk}"
    )
    for name in (
        "encoder_token",
        "encoder_atom",
        "encoder_pair",
        "decoder",
        "step",
        "sampler",
    ):
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
    # plan reported absent by an arm that never ran the site would be a
    # tripwire that certifies nothing.
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
    # The column `psum` of the sparse token-pair lookup exists only when
    # there is a column axis to reduce over.
    assert (counts["all-reduce"] > 0) == (layout == "2d"), counts

    full_width = {
        "atom activation": f"f32[{N_SAMPLE},{N_ATOM},{C_ATOM}]",
        "atom-pair window cache": (
            f"f32[1,{N_WINDOWS},{N_QUERIES},{N_KEYS},{C_ATOMPAIR}]"
        ),
        "projected token pair": f"f32[1,{N_TOKEN},{N_TOKEN},{C_PAIR}]",
        # Four dimensions with two full token axes: no atom-path shape can
        # spell that, which is what makes these two unambiguous at
        # N_TOKEN == N_WINDOWS. Without them the docs' claim that the token
        # bias and logits are split on both pair axes would rest on a "12"
        # that could equally be half the atom windows.
        "token pair bias": f"f32[1,{TOKEN_HEADS},{N_TOKEN},{N_TOKEN}]",
        "token attention logits": (
            f"f32[{N_SAMPLE},{TOKEN_HEADS},{N_TOKEN},{N_TOKEN}]"
        ),
    }
    for label, shape in full_width.items():
        assert spmd.count(shape) == 0, (label, shape, spmd.count(shape))
    print("  per-device full-width intermediates", {k: 0 for k in full_width})
    print("PROTENIX_ATOM_CP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
