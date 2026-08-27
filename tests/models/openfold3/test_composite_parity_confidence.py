"""Confidence-head parity against upstream's ``AuxiliaryHeadsAllAtom``.

The gap this closes: upstream re-embeds the trunk representations with the
predicted geometry (``PairformerEmbedding``) before pLDDT, PAE, PDE and the
experimentally-resolved head. Only the distogram head reads the trunk pair
embedding. Feeding the trunk outputs to the other heads produces correctly shaped
and wrong logits, which no per-layer test can see.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_atom_logit_head,
    map_pair_head,
    map_pairformer_embedding,
)
from foldjax.models.openfold3.models.heads import (
    distogram_head,
    pairformer_embedding,
    predicted_aligned_error_head,
    predicted_distance_error_head,
)
from foldjax.models.openfold3.models.representative_atoms import (
    token_representative_atoms,
)

from .of3_features import make_batch
from .test_composite_parity_trunk import _reduced_config

pytestmark = pytest.mark.torch_parity

# A whole-model forward pass amplifies weight magnitude; see the randomized
# fixture for why the scale is lowered instead of the tolerance.
COMPOSITE_SCALE = 0.05

RTOL = 1e-4
ATOL = 1e-4

N_TOKEN, N_SAMPLES = 12, 2


@pytest.fixture(scope="module")
def reference(openfold3_source: Path, randomized):
    """Randomized for the reason given in the trunk composite test."""
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    torch.manual_seed(0)
    model = randomized(OpenFold3(_reduced_config()), scale=COMPOSITE_SCALE)
    batch = make_batch(n_token=N_TOKEN)
    n_atom = batch["atom_mask"].shape[-1]

    with torch.no_grad():
        s_input, s_trunk, z = model.run_trunk(batch=batch, num_recycles=1)
        generator = torch.Generator().manual_seed(2)
        x_pred = torch.randn((1, N_SAMPLES, n_atom, 3), generator=generator)

        # Upstream's _rollout gives every batch feature and every trunk output a
        # sample axis at dim 1 before calling the heads, and PairformerEmbedding
        # rejects inputs whose batch-dim counts disagree.
        sampled = {key: value.unsqueeze(1) for key, value in batch.items()}
        output = {
            "atom_positions_predicted": x_pred,
            "si_trunk": s_trunk.unsqueeze(1),
            "zij_trunk": z.unsqueeze(1),
        }
        aux = model.aux_heads(
            batch=sampled,
            si_input=s_input.unsqueeze(1),
            output=output,
            use_zij_trunk_embedding=True,
        )
    trunk_out = (s_input.unsqueeze(1), s_trunk.unsqueeze(1), z.unsqueeze(1))
    return model, sampled, trunk_out, x_pred, aux


def _as_jax(batch: dict) -> dict:
    return {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }


def _close(actual, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from upstream aux_heads",
    )


def test_distogram_reads_the_trunk_pair_embedding(reference) -> None:
    model, _batch, (_s_input, _s_trunk, z), _x, aux = reference
    logits = distogram_head(
        jnp.asarray(z.numpy()),
        map_pair_head(
            dict(model.aux_heads.distogram.state_dict()), layer_norm=False
        ),
    )
    _close(logits, aux["distogram_logits"], "distogram_logits")


@pytest.fixture(scope="module")
def embedded(reference):
    """Run this port's re-embedding once for the heads below."""
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    model, batch, (s_input, s_trunk, z), x_pred, _aux = reference
    config = _reduced_config().architecture
    pe_config = config.heads.pairformer_embedding

    jax_batch = _as_jax(batch)
    rep_x, rep_mask = token_representative_atoms(
        jax_batch,
        jnp.asarray(x_pred.numpy()),
        jax_batch["atom_mask"],
        representative_atom_table(),
    )
    token_mask = jax_batch["token_mask"]
    return pairformer_embedding(
        jnp.asarray(s_input.numpy()),
        jnp.asarray(s_trunk.numpy()),
        jnp.asarray(z.numpy()),
        rep_x,
        map_pairformer_embedding(
            dict(model.aux_heads.pairformer_embedding.state_dict())
        ),
        single_mask=rep_mask,
        pair_mask=token_mask[..., :, None] * token_mask[..., None, :],
        no_heads_pair=pe_config.pairformer.no_heads_pair,
        no_heads_pair_bias=pe_config.pairformer.no_heads_pair_bias,
        min_bin=pe_config.min_bin,
        max_bin=pe_config.max_bin,
        no_bin=pe_config.no_bin,
        inf=pe_config.inf,
    )


def test_re_embedding_changes_the_representations(reference, embedded) -> None:
    """Guards the gate: if re-embedding were a no-op nothing below would bite."""
    _model, _batch, (_s_input, s_trunk, z), _x, _aux = reference
    s_conf, z_conf = embedded
    assert not np.allclose(
        np.asarray(z_conf[:, 0]), z[:, 0].detach().numpy(), rtol=1e-2, atol=1e-2
    )
    assert not np.allclose(
        np.asarray(s_conf[:, 0]), s_trunk[:, 0].detach().numpy(), rtol=1e-2, atol=1e-2
    )


def test_pae_and_pde_match_upstream(reference, embedded) -> None:
    model, _batch, _trunk, _x, aux = reference
    _s_conf, z_conf = embedded
    _close(
        predicted_aligned_error_head(
            z_conf, map_pair_head(dict(model.aux_heads.pae.state_dict()))
        ),
        aux["pae_logits"],
        "pae_logits",
    )
    _close(
        predicted_distance_error_head(
            z_conf, map_pair_head(dict(model.aux_heads.pde.state_dict()))
        ),
        aux["pde_logits"],
        "pde_logits",
    )


def test_plddt_and_experimentally_resolved_match_upstream(
    reference, embedded
) -> None:
    from foldjax.models.openfold3.models.heads import atom_logit_head

    model, batch, _trunk, x_pred, aux = reference
    s_conf, _z_conf = embedded
    config = _reduced_config().architecture

    # The padded per-atom mask is built with upstream's own helper: it is an input
    # to these heads, not the thing under test, and its layout (N_token *
    # max_atoms_per_token, not N_atom) is easy to get subtly wrong.
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    n_atom = batch["atom_mask"].shape[-1]
    ref_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"],
        max_num_atoms_per_token=config.heads.max_atoms_per_token,
    )
    max_atom_mask = jnp.broadcast_to(
        jnp.asarray(ref_mask.numpy()),
        (*x_pred.shape[:-2], ref_mask.shape[-1]),
    )

    for name, head, bins in (
        ("plddt_logits", model.aux_heads.plddt, config.heads.lddt.c_out),
        (
            "experimentally_resolved_logits",
            model.aux_heads.experimentally_resolved,
            config.heads.experimentally_resolved.c_out,
        ),
    ):
        _close(
            atom_logit_head(
                s_conf,
                map_atom_logit_head(dict(head.state_dict())),
                max_atom_mask,
                max_atoms_per_token=config.heads.max_atoms_per_token,
                c_out=bins,
                n_atom=n_atom,
            ),
            aux[name],
            name,
        )


def test_zeroed_trunk_pair_embedding_matches_upstream(reference) -> None:
    """``use_zij_trunk_embedding=False`` zeroes the pair embedding for the
    re-embedding only; the distogram head still reads the trunk output.

    Upstream exposes this as a call-site flag, so the port has to reproduce both
    settings, not just the default.
    """
    import torch

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    model, batch, (s_input, s_trunk, z), x_pred, _aux = reference
    config = _reduced_config().architecture
    pe_config = config.heads.pairformer_embedding

    with torch.no_grad():
        aux_off = model.aux_heads(
            batch=batch,
            si_input=s_input,
            output={
                "atom_positions_predicted": x_pred,
                "si_trunk": s_trunk,
                "zij_trunk": z,
            },
            use_zij_trunk_embedding=False,
        )

    jax_batch = _as_jax(batch)
    rep_x, rep_mask = token_representative_atoms(
        jax_batch,
        jnp.asarray(x_pred.numpy()),
        jax_batch["atom_mask"],
        representative_atom_table(),
    )
    token_mask = jax_batch["token_mask"]
    _s_conf, z_conf = pairformer_embedding(
        jnp.asarray(s_input.numpy()),
        jnp.asarray(s_trunk.numpy()),
        jnp.zeros_like(jnp.asarray(z.numpy())),
        rep_x,
        map_pairformer_embedding(
            dict(model.aux_heads.pairformer_embedding.state_dict())
        ),
        single_mask=rep_mask,
        pair_mask=token_mask[..., :, None] * token_mask[..., None, :],
        no_heads_pair=pe_config.pairformer.no_heads_pair,
        no_heads_pair_bias=pe_config.pairformer.no_heads_pair_bias,
        min_bin=pe_config.min_bin,
        max_bin=pe_config.max_bin,
        no_bin=pe_config.no_bin,
        inf=pe_config.inf,
    )
    _close(
        predicted_aligned_error_head(
            z_conf, map_pair_head(dict(model.aux_heads.pae.state_dict()))
        ),
        aux_off["pae_logits"],
        "pae_logits (trunk pair embedding off)",
    )
    # The distogram is unaffected by the flag.
    _close(
        distogram_head(
            jnp.asarray(z.numpy()),
            map_pair_head(
                dict(model.aux_heads.distogram.state_dict()), layer_norm=False
            ),
        ),
        aux_off["distogram_logits"],
        "distogram_logits (trunk pair embedding off)",
    )
