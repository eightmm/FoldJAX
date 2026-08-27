"""End-to-end parity: ``predict`` against upstream's ``OpenFold3.forward``.

The other composite tests each gate one subsystem. This gates the whole call, so a
mistake in how the pieces are wired together -- the sample axis, which coordinates
reach the confidence heads, which representation each head reads -- fails here even
though every subsystem passes on its own.

Randomness is removed the same way as in the sampler test: the same pre-generated
noise is injected into both rollouts and augmentation is the identity on both
sides. Nothing else is stubbed; upstream runs its own trunk, rollout and heads.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
from foldjax.models.openfold3.inference import InferenceConfig, predict

from .of3_features import make_batch
from .test_composite_parity_trunk import COMPOSITE_SCALE, _reduced_config

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

# Coordinates are compared with a looser absolute tolerance than the logits.
# The rollout starts at ``noise_schedule[0]`` -- 2560 for the released schedule --
# and contracts to O(1), so the result is a difference of numbers ~2e3 larger than
# itself. One float32 ulp at that scale is 2560 * 1.2e-7 ~= 3e-4, which is what is
# observed (worst case 3.2e-4, on 2 of 978 elements, and only on CPU: XLA-CPU and
# torch-CPU reduce in different orders). It is 3e-4 Angstrom, and the confidence
# outputs -- which do not pass through that cancellation -- still agree to 1e-7.
COORD_RTOL = 1e-4
COORD_ATOL = 1e-3

N_MSA, N_TEMPL = 6, 2
STEPS, SAMPLES = 4, 2


# (blocks per stack, tokens). More than one block catches block-ordering errors;
# more tokens catches anything that only holds at a small pair matrix -- masking,
# the sequence-local atom windows, relpos clipping. The released depths at 128
# tokens are verified out of band; see PROJECT.md.
SHAPES = ((1, 12), (3, 12), (3, 64))


@pytest.fixture(scope="module", params=SHAPES, ids=lambda p: f"blocks{p[0]}x{p[1]}tok")
def reference(request, openfold3_source: Path, randomized):
    """Upstream's full forward, with its random draws replaced by fixed ones."""
    import torch
    from openfold3.core.model.structure import diffusion_module as sampler_module
    from openfold3.projects.of3_all_atom.model import OpenFold3

    torch.manual_seed(0)
    blocks, n_token = request.param
    config = _reduced_config(blocks)
    shared = config.architecture.shared
    shared.num_recycles = 0
    shared.diffusion.no_full_rollout_samples = SAMPLES
    shared.diffusion.no_full_rollout_steps = STEPS
    model = randomized(OpenFold3(config), scale=COMPOSITE_SCALE)

    batch = make_batch(n_token=n_token, n_msa=N_MSA, n_templ=N_TEMPL)
    n_atom = batch["atom_mask"].shape[-1]

    generator = torch.Generator().manual_seed(11)
    shape = (1, SAMPLES, n_atom, 3)
    draws = [torch.randn(shape, generator=generator) for _ in range(STEPS + 1)]
    remaining = list(draws)

    original_randn, original_randn_like = torch.randn, torch.randn_like
    original_augment = sampler_module.centre_random_augmentation
    sampler_module.torch.randn = lambda *a, **k: remaining.pop(0)
    sampler_module.torch.randn_like = lambda *a, **k: remaining.pop(0)
    sampler_module.centre_random_augmentation = lambda xl, atom_mask: xl
    try:
        with torch.no_grad():
            _batch_out, output = model(batch=dict(batch))
    finally:
        sampler_module.torch.randn = original_randn
        sampler_module.torch.randn_like = original_randn_like
        sampler_module.centre_random_augmentation = original_augment

    assert not remaining, "upstream drew a different number of noise tensors"
    return model, batch, draws, output, n_atom, blocks, n_token


def _config(n_atom: int, blocks: int = 1, n_token: int = 12) -> InferenceConfig:
    architecture = _reduced_config(blocks).architecture
    encoder = architecture.input_embedder.atom_attn_enc
    conditioning = architecture.diffusion_module.diffusion_conditioning
    heads = architecture.heads
    schedule = architecture.noise_schedule
    return InferenceConfig(
        n_token=n_token,
        n_atom=n_atom,
        n_query=encoder.n_query,
        n_key=encoder.n_key,
        atom_heads=encoder.no_heads,
        token_heads=architecture.diffusion_module.diffusion_transformer.no_heads,
        no_heads_msa=architecture.msa.msa_module.no_heads_msa,
        no_heads_pair=architecture.pairformer.no_heads_pair,
        no_heads_pair_bias=architecture.pairformer.no_heads_pair_bias,
        max_relative_idx=conditioning.max_relative_idx,
        max_relative_chain=conditioning.max_relative_chain,
        num_recycles=1,
        num_samples=SAMPLES,
        max_atoms_per_token=heads.max_atoms_per_token,
        plddt_bins=heads.lddt.c_out,
        pae_bins=heads.pae.c_out,
        pae_bin_max=32.0,
        num_steps=STEPS,
        sigma_data=architecture.diffusion_module.diffusion_module.sigma_data,
        s_max=schedule.s_max,
        s_min=schedule.s_min,
        p=schedule.p,
        confidence_min_bin=heads.pairformer_embedding.min_bin,
        confidence_max_bin=heads.pairformer_embedding.max_bin,
        confidence_no_bin=heads.pairformer_embedding.no_bin,
        experimentally_resolved_bins=heads.experimentally_resolved.c_out,
    )


@pytest.fixture(scope="module")
def prediction(reference):
    """This port's predict on the same weights, batch and noise."""
    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    model, batch, draws, _output, n_atom, blocks, n_token = reference
    jax_batch = {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }
    # Upstream's heads need this padded mask; it is a featurizer output, so it is
    # built with upstream's helper rather than reconstructed here.
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    jax_batch["max_atom_per_token_mask"] = jnp.asarray(
        broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["token_mask"],
            max_num_atoms_per_token=_reduced_config().architecture.heads
            .max_atoms_per_token,
        ).numpy()
    )

    params = map_inference_params(dict(model.state_dict()))
    # Guard the parametrization: if the block count did not actually change, the
    # multi-block case proves nothing about block ordering.
    assert len(params.trunk.pairformer_stack.blocks) == blocks
    assert len(params.denoiser.diffusion_transformer.blocks) == blocks
    assert len(params.pairformer_embedding.pairformer_stack.blocks) == blocks

    return predict(
        jax.random.key(0),
        jax_batch,
        params,
        _config(n_atom, blocks, n_token),
        representative_atom_table(),
        # The sample axis is folded into the leading axis here, so the injected
        # tensors are reshaped to match; they are the same numbers in the same
        # order as upstream received.
        noise_fn=lambda step, shape: jnp.asarray(
            draws[step].numpy()
        ).reshape(shape),
        augment=False,
    )


def test_coordinates_match_upstream(reference, prediction) -> None:
    _model, _batch, _draws, output, n_atom, _blocks, _n_token = reference
    expected = output["atom_positions_predicted"]
    actual = prediction.coordinates
    assert actual.shape == (SAMPLES, n_atom, 3)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.reshape(SAMPLES, n_atom, 3).detach().numpy().astype(np.float64),
        rtol=COORD_RTOL,
        atol=COORD_ATOL,
        err_msg="predicted coordinates diverged from upstream forward",
    )


@pytest.mark.parametrize(
    ("field", "key"),
    [
        ("pae_logits", "pae_logits"),
        ("pde_logits", "pde_logits"),
        ("distogram_logits", "distogram_logits"),
        ("experimentally_resolved_logits", "experimentally_resolved_logits"),
    ],
)
def test_confidence_outputs_match_upstream(
    reference, prediction, field: str, key: str
) -> None:
    _model, _batch, _draws, output, _n_atom, _blocks, _n_token = reference
    actual = getattr(prediction, field)
    expected = output[key]
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).reshape(-1),
        expected.detach().numpy().astype(np.float64).reshape(-1),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{key} diverged from upstream forward",
    )


def test_plddt_matches_upstream(reference, prediction) -> None:
    """``Prediction`` exposes pLDDT as a per-atom expectation, not logits, so the
    reference logits go through the same reduction before comparing."""
    from foldjax.models.openfold3.models.confidence import compute_plddt

    _model, _batch, _draws, output, _n_atom, _blocks, _n_token = reference
    expected = compute_plddt(jnp.asarray(output["plddt_logits"].detach().numpy()))
    np.testing.assert_allclose(
        np.asarray(prediction.plddt, dtype=np.float64).reshape(-1),
        np.asarray(expected, dtype=np.float64).reshape(-1),
        rtol=RTOL,
        atol=ATOL,
        err_msg="pLDDT diverged from upstream forward",
    )


def test_predict_honours_the_trunk_pair_embedding_flag(reference, prediction) -> None:
    """The flag has to change the confidence outputs and nothing else.

    Coordinates come from the rollout, which runs before the confidence heads, so
    they must be bit-identical; the distogram reads the trunk embedding directly, so
    it must be too.
    """
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    model, batch, draws, _output, n_atom, blocks, n_token = reference
    jax_batch = {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }
    jax_batch["max_atom_per_token_mask"] = jnp.asarray(
        broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["token_mask"],
            max_num_atoms_per_token=_reduced_config().architecture.heads
            .max_atoms_per_token,
        ).numpy()
    )
    off = predict(
        jax.random.key(0),
        jax_batch,
        map_inference_params(dict(model.state_dict())),
        _config(n_atom, blocks, n_token),
        representative_atom_table(),
        noise_fn=lambda step, shape: jnp.asarray(draws[step].numpy()).reshape(shape),
        augment=False,
        use_trunk_pair_embedding=False,
    )
    np.testing.assert_array_equal(
        np.asarray(off.coordinates), np.asarray(prediction.coordinates)
    )
    np.testing.assert_array_equal(
        np.asarray(off.distogram_logits), np.asarray(prediction.distogram_logits)
    )
    assert not np.allclose(
        np.asarray(off.pae_logits), np.asarray(prediction.pae_logits)
    )
