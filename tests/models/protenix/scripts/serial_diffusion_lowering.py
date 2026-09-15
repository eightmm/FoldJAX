"""Print a normalised fingerprint of the serial Protenix denoiser lowering.

Run with ``foldjax`` on ``PYTHONPATH`` from two different checkouts and the
two fingerprints answer one question: did this change alter the program a
serial run executes?  The parameters come from
``tests.models.protenix.atom_cp_fixtures``, which is always taken from the
tree under test's *tests* directory, so only ``foldjax`` differs between the
two runs.

The call deliberately omits ``cp_atom_windows``: the point is to compare the
call an existing caller makes, and on an older tree that keyword does not
exist. Every context-parallel entry point is a no-op without a mesh anyway, so
the default call is the whole serial contract.

Output: three lines -- ``HASH <hex>``, ``COLLECTIVES <n>``, ``SHARDING <n>``.
"""

from __future__ import annotations

import hashlib
import re
import sys

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_encoder_prepare_diffusion_cache,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    diffusion_module_forward,
)
from tests.models.protenix.atom_cp_fixtures import (
    N_HEADS,
    N_KEYS,
    N_PAIR_FEATURES,
    N_QUERIES,
    N_TOKEN,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_module_case,
)

#: Collective and sharding operations no single-device program may contain.
COLLECTIVE_OPS = (
    "collective-permute",
    "all-gather",
    "all-reduce",
    "reduce-scatter",
    "all-to-all",
)


def _normalise(text: str) -> str:
    """Drop everything that records where the source lived, not what it does.

    ``metadata={...}`` carries ``source_file``/``source_line`` and a
    ``stack_frame_id`` index into a table of them, so two checkouts of the
    same program disagree on it as soon as any line number moves -- which a
    patch that adds a guarded branch always does.
    """

    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> int:
    params, case, features = build_module_case()

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
        )
        return diffusion_module_forward(
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
            jnp.zeros((N_TOKEN, N_TOKEN, N_PAIR_FEATURES), dtype=jnp.float32),
            s_inputs,
            s_trunk,
            jnp.zeros_like(pair_z),
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
            attention_backend="xla",
            glu_backend="xla",
            token_mask=jnp.ones((N_TOKEN,), dtype=bool),
            atom_mask=case.atom_mask,
        )

    args = (
        features["s_inputs"],
        features["s_trunk"],
        features["pair_z"],
        features["x_noisy"],
        features["t_hat"],
    )
    text = jax.jit(run).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    normalised = _normalise(text)
    lowered = normalised.lower()
    print("HASH", hashlib.sha256(normalised.encode()).hexdigest())
    print("COLLECTIVES", sum(lowered.count(op) for op in COLLECTIVE_OPS))
    # `with_sharding_constraint` lowers to a `Sharding` custom call; an
    # explicit spec also prints as a `sharding={...}` attribute.
    print("SHARDING_CALLS", lowered.count('custom_call_target="sharding"'))
    print("SHARDING_ANNOTATIONS", lowered.count("sharding={"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
