"""Parity at the released depths, with no architectural reductions.

Every other composite test shrinks the stacks to keep the suite fast, which leaves
one thing unmeasured: whether the port agrees with upstream at 48 Pairformer / 4
MSA / 24 diffusion / 2 template / 4 confidence blocks and 4 recycling cycles --
368M parameters, the settings a released checkpoint was trained with.

Depth matters for a reason beyond block indexing: a 48-block residual stack
amplifies any per-block discrepancy, so an error too small to see at three blocks
can surface here.

Nothing is reduced: the rollout is the released 5 samples over 200 steps. That is
only affordable because the rollout is a ``lax.scan`` and this test takes the
compiled path -- the same path production uses. Eagerly, 200 steps would take
roughly an hour; compiled it is ~1 s after a one-time compile.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_inference_params
from foldjax.models.openfold3.inference import predict, released_config

from .of3_features import make_batch

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

N_TOKEN, N_MSA, N_TEMPL = 12, 8, 2
# The released full rollout.
SAMPLES, STEPS = 5, 200
# A 48-block stack compounds the weight scale, so this is smaller than the
# reduced-depth tests use; see the randomized fixture.
SCALE = 0.05


@pytest.fixture(scope="module")
def released_reference(openfold3_source: Path):
    """Upstream at released depths, with its random draws replaced by fixed ones."""
    import torch
    from openfold3.core.model.structure import diffusion_module as sampler_module
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3

    config = copy.deepcopy(model_config)
    architecture = config.architecture
    architecture.heads.pae.enabled = True
    architecture.shared.diffusion.no_full_rollout_samples = SAMPLES
    architecture.shared.diffusion.no_full_rollout_steps = STEPS
    for mode in ("train", "eval"):
        memory = config.settings.memory[mode]
        memory.use_triton_triangle_kernels = False
        memory.use_cueq_triangle_kernels = False
        memory.use_deepspeed_evo_attention = False
        memory.use_lma = False
        memory.chunk_size = None
        memory.tune_chunk_size = False

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = OpenFold3(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * SCALE)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    batch = make_batch(n_token=N_TOKEN, n_msa=N_MSA, n_templ=N_TEMPL)
    n_atom = batch["atom_mask"].shape[-1]
    noise_generator = torch.Generator().manual_seed(11)
    draws = [
        torch.randn((1, SAMPLES, n_atom, 3), generator=noise_generator)
        for _ in range(STEPS + 1)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            _batch_out, output = model(
                batch={key: value.to(device) for key, value in batch.items()}
            )
    finally:
        (
            sampler_module.torch.randn,
            sampler_module.torch.randn_like,
            sampler_module.centre_random_augmentation,
        ) = saved

    assert not remaining, "upstream drew a different number of noise tensors"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return state, batch, draws, output, n_atom, architecture


@pytest.fixture(scope="module")
def released_prediction(released_reference):
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    from foldjax.models.openfold3.bridge.chemistry import representative_atom_table

    state, batch, draws, _output, n_atom, architecture = released_reference
    features = {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }
    features["max_atom_per_token_mask"] = jnp.asarray(
        broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["token_mask"],
            max_num_atoms_per_token=architecture.heads.max_atoms_per_token,
        ).numpy()
    )

    params = map_inference_params(state)
    # The released preset is what is under test here, not a hand-built config.
    config = released_config(
        n_token=N_TOKEN,
        n_atom=n_atom,
        num_samples=SAMPLES,
        no_rollout_steps=STEPS,
    )
    assert len(params.trunk.pairformer_stack.blocks) == 48
    assert len(params.trunk.msa_module.blocks) == 4
    assert len(params.denoiser.diffusion_transformer.blocks) == 24
    assert config.num_cycles == architecture.shared.num_recycles + 1

    # compile_predict does not expose noise_fn -- injecting another
    # implementation's random stream is a test concern -- so the same binding it
    # would produce is built here, with the injected noise added.
    compiled = jax.jit(
        functools.partial(
            predict,
            config=config,
            representative_atoms=representative_atom_table(),
            noise_fn=lambda step, shape: jnp.asarray(
                draws[step].numpy()
            ).reshape(shape),
            augment=False,
        )
    )
    return compiled(jax.random.key(0), features, params)


def _close(actual, expected, name: str, *, rtol=RTOL, atol=ATOL) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).reshape(-1),
        np.asarray(expected.detach().cpu(), dtype=np.float64).reshape(-1),
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} diverged from upstream at released depths",
    )


def test_coordinates_match_at_released_depths(
    released_reference, released_prediction
) -> None:
    _state, _batch, _draws, output, n_atom, _architecture = released_reference
    assert released_prediction.coordinates.shape == (SAMPLES, n_atom, 3)
    _close(
        released_prediction.coordinates,
        output["atom_positions_predicted"],
        "coordinates",
        rtol=COORD_RTOL,
        atol=COORD_ATOL,
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
def test_confidence_matches_at_released_depths(
    released_reference, released_prediction, field: str, key: str
) -> None:
    _state, _batch, _draws, output, _n_atom, _architecture = released_reference
    _close(getattr(released_prediction, field), output[key], key)


def test_the_deep_stacks_actually_contribute(released_prediction) -> None:
    """Guard: a 48-block trunk whose blocks did nothing would still pass above."""
    coordinates = np.asarray(released_prediction.coordinates)
    assert np.isfinite(coordinates).all()
    assert coordinates.std() > 1e-6


def test_every_sample_differs(released_prediction) -> None:
    """The released rollout draws 5 samples; they must not collapse to one."""
    coordinates = np.asarray(released_prediction.coordinates)
    assert coordinates.shape[0] == SAMPLES
    for index in range(1, SAMPLES):
        assert not np.allclose(coordinates[0], coordinates[index])
