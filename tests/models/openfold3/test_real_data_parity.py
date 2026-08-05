"""Torch-vs-JAX parity on real depositions.

Every other parity test in this repo feeds the model a batch built here. That is
enough to check arithmetic, but not enough to check the port against the inputs it
will actually see: real entries have chain breaks, repeated chains, metal ions,
ligands with generated conformers, and mixed molecule types in one target. Those are
exactly the features that tokenization, masking and the atom/token index mapping can
get wrong while every synthetic test passes.

So these run upstream's ``OpenFold3.forward`` and this port's ``predict`` on the
same featurized deposition. Depth is reduced -- the released 48-block trunk is gated
separately -- because what varies here is the *input*, not the architecture.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
from foldjax.models.openfold3.inference import predict, released_config

from .real_targets import SUITE_LADDER, described, featurized, requires_real_targets
from .test_composite_parity_trunk import COMPOSITE_SCALE, _reduced_config

# Every test here is built on a real deposition, so the whole module needs
# foldbench as well as torch.
pytestmark = [pytest.mark.torch_parity, requires_real_targets]

RTOL = 1e-4
ATOL = 1e-4
COORD_ATOL = 1e-3

SAMPLES, STEPS = 2, 3
BLOCKS = 2


@pytest.fixture(scope="module")
def model_and_params(openfold3_source: Path, randomized):
    """One reduced-depth model reused across every target."""
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    torch.manual_seed(0)
    config = _reduced_config(BLOCKS)
    config.architecture.shared.diffusion.no_full_rollout_samples = SAMPLES
    config.architecture.shared.diffusion.no_full_rollout_steps = STEPS
    model = randomized(OpenFold3(config), scale=COMPOSITE_SCALE)
    if torch.cuda.is_available():
        model = model.cuda()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return model, map_inference_params(state), config


def _run_pair(model, params, config, features: dict, n_token: int, n_atom: int):
    """Run both implementations on identical inputs and identical noise."""
    import torch
    from openfold3.core.model.structure import diffusion_module as sampler_module

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(11)
    draws = [
        torch.randn((1, SAMPLES, n_atom, 3), generator=generator)
        for _ in range(STEPS + 1)
    ]

    batch = {
        name: torch.as_tensor(value).to(device)
        for name, value in features.items()
        if name != "max_atom_per_token_mask"
    }
    remaining = [draw.to(device) for draw in draws]
    saved = (
        sampler_module.torch.randn,
        sampler_module.torch.randn_like,
        sampler_module.centre_random_augmentation,
    )
    sampler_module.torch.randn = lambda *a, **k: remaining.pop(0)
    sampler_module.torch.randn_like = lambda *a, **k: remaining.pop(0)
    sampler_module.centre_random_augmentation = lambda xl, atom_mask: xl
    try:
        with torch.no_grad():
            _out_batch, expected = model(batch=copy.copy(batch))
    finally:
        (
            sampler_module.torch.randn,
            sampler_module.torch.randn_like,
            sampler_module.centre_random_augmentation,
        ) = saved
    assert not remaining, "upstream drew a different number of noise tensors"

    actual = predict(
        jax.random.key(0),
        {name: jnp.asarray(value) for name, value in features.items()},
        params,
        released_config(
            n_token=n_token,
            n_atom=n_atom,
            num_samples=SAMPLES,
            no_rollout_steps=STEPS,
            num_cycles=config.architecture.shared.num_recycles + 1,
        ),
        representative_atom_table(),
        noise_fn=lambda step, shape: jnp.asarray(draws[step].numpy()).reshape(shape),
        augment=False,
    )
    return actual, expected


@pytest.mark.parametrize("pdb_id", SUITE_LADDER)
def test_real_target_matches_upstream(model_and_params, pdb_id: str) -> None:
    model, params, config = model_and_params
    features = featurized(pdb_id)
    n_token = features["token_mask"].shape[-1]
    n_atom = features["atom_mask"].shape[-1]

    actual, expected = _run_pair(model, params, config, features, n_token, n_atom)

    np.testing.assert_allclose(
        np.asarray(actual.coordinates, dtype=np.float64),
        expected["atom_positions_predicted"]
        .reshape(SAMPLES, n_atom, 3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        rtol=RTOL,
        atol=COORD_ATOL,
        err_msg=f"{described(pdb_id)}: coordinates diverged",
    )
    for field, key in (
        ("pae_logits", "pae_logits"),
        ("pde_logits", "pde_logits"),
        ("distogram_logits", "distogram_logits"),
        ("experimentally_resolved_logits", "experimentally_resolved_logits"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field), dtype=np.float64).reshape(-1),
            expected[key].detach().cpu().numpy().astype(np.float64).reshape(-1),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{described(pdb_id)}: {key} diverged",
        )


def test_the_ladder_is_not_all_protein() -> None:
    """Guard on the set itself: a protein-only ladder would prove much less."""
    from tests._foldbench import load_target

    types = {
        kind for pdb_id in SUITE_LADDER for kind in load_target(pdb_id).molecule_types
    }
    assert {"protein", "dna", "rna"} <= types
    assert any(load_target(pdb_id).ligands for pdb_id in SUITE_LADDER)
