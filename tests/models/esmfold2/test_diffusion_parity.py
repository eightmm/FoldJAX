"""The diffusion head against torch: attention, conditioning, denoiser, sampler.

The sampler is stochastic, so what is gated here is everything in it that is
not: the schedule it walks, the alignment it applies between the denoiser call
and the step, the rotation it builds from a quaternion, and the denoiser
itself. What is left over is three `randn` draws.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")
configuration = pytest.importorskip(
    "transformers.models.esmfold2.configuration_esmfold2"
)

from foldjax.models.esmfold2.models import diffusion  # noqa: E402
from tests.models.esmfold2.parity import torch_state_to_numpy  # noqa: E402

ATOL = 2e-5
# head_dim = c_atom // atom heads must reach 32 for the 3D rotary table, which
# needs 3*2 + 10 = 16 frequency pairs. See `test_atom_parity`.
D_TOKEN, D_PAIR, N_HEADS = 32, 16, 2
C_ATOM, C_INPUTS = 64, 24
N_TOKENS, N_ATOMS = 6, 18


def _normal(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _t(array):
    return torch.from_numpy(np.asarray(array))


def _atom_inputs(seed=0):
    rng = np.random.default_rng(seed)
    element = np.zeros((1, N_ATOMS, 128), dtype=np.float32)
    element[..., 6] = 1.0
    chars = np.zeros((1, N_ATOMS, 4, 64), dtype=np.float32)
    chars[..., 0] = 1.0
    return {
        "ref_pos": rng.standard_normal((1, N_ATOMS, 3)).astype(np.float32),
        "ref_charge": np.zeros((1, N_ATOMS), dtype=np.float32),
        "ref_mask": np.ones((1, N_ATOMS), dtype=np.float32),
        "ref_element": element,
        "ref_atom_name_chars": chars,
        "ref_space_uid": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
        "tok_idx": (np.arange(N_ATOMS, dtype=np.int64) // 3)[None],
    }


@pytest.mark.parametrize("conditioned", [True, False])
def test_attention_pair_bias_matches(conditioned: bool) -> None:
    """Softmax axis, pair bias, and which tensor the output gate reads."""
    module = common.AttentionPairBias(
        d_model=D_TOKEN,
        d_pair=D_PAIR,
        num_heads=N_HEADS,
        use_conditioning=conditioned,
    ).eval()
    a = _normal((1, N_TOKENS, D_TOKEN), 1)
    s = _normal((1, N_TOKENS, D_TOKEN), 2) if conditioned else None
    z = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 3)
    mask = np.ones((1, N_TOKENS), dtype=np.float32)
    mask[0, -2:] = 0.0

    with torch.no_grad():
        expected = module(
            a=_t(a),
            s=None if s is None else _t(s),
            z=_t(z),
            attention_mask=_t(mask).bool(),
        )
    got = diffusion.attention_pair_bias(
        a, s, z, torch_state_to_numpy(module), n_heads=N_HEADS, mask=mask
    )
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=ATOL, rtol=ATOL)


def test_the_pair_bias_is_shared_across_diffusion_samples() -> None:
    """Two samples reading one pair tensor must match two repeated ones.

    Upstream repeats `z` on the batch axis inside the block; this port adds the
    bias against a folded view instead, and the two have to agree or every
    sample past the first is reading another sample's pair.
    """
    module = common.AttentionPairBias(
        d_model=D_TOKEN, d_pair=D_PAIR, num_heads=N_HEADS
    ).eval()
    params = torch_state_to_numpy(module)
    a = _normal((2, N_TOKENS, D_TOKEN), 4)
    s = _normal((2, N_TOKENS, D_TOKEN), 5)
    z = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 6)

    folded = diffusion.attention_pair_bias(
        a, s, z, params, n_heads=N_HEADS, num_samples=2
    )
    repeated = diffusion.attention_pair_bias(
        a, s, np.repeat(z, 2, axis=0), params, n_heads=N_HEADS
    )
    np.testing.assert_allclose(np.asarray(folded), np.asarray(repeated), atol=ATOL)


def test_conditioned_transition_block_matches() -> None:
    module = common.ConditionedTransitionBlock(d_model=D_TOKEN).eval()
    a, s = _normal((1, N_TOKENS, D_TOKEN), 7), _normal((1, N_TOKENS, D_TOKEN), 8)
    with torch.no_grad():
        expected = module(_t(a), _t(s))
    got = diffusion.conditioned_transition_block(a, s, torch_state_to_numpy(module))
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=ATOL, rtol=ATOL)


def test_diffusion_transformer_matches() -> None:
    module = common.DiffusionTransformer(
        d_model=D_TOKEN, d_pair=D_PAIR, num_heads=N_HEADS, num_blocks=2
    ).eval()
    a = _normal((1, N_TOKENS, D_TOKEN), 9)
    s = _normal((1, N_TOKENS, D_TOKEN), 10)
    z = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 11)
    with torch.no_grad():
        expected, _ = module(_t(a), _t(s), _t(z))
    got = diffusion.diffusion_transformer(
        a, s, z, torch_state_to_numpy(module), n_blocks=2, n_heads=N_HEADS
    )
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=1e-4, rtol=1e-4)


def test_diffusion_conditioning_matches() -> None:
    """Both halves, including the quarter-log timestep and its clamp."""
    module = common.DiffusionConditioning(
        c_z=D_PAIR, c_s=D_TOKEN, c_s_inputs=C_INPUTS, fourier_dim=16
    ).eval()
    params = torch_state_to_numpy(module)
    s_inputs = _normal((1, N_TOKENS, C_INPUTS), 12)
    z_trunk = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 13)
    rel_pos = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 14)
    t_hat = np.array([3.5], dtype=np.float32)

    with torch.no_grad():
        expected_s, expected_z = module(
            t_hat=_t(t_hat),
            s_inputs=_t(s_inputs),
            s_trunk=None,
            z_trunk=_t(z_trunk),
            relative_position_encoding=_t(rel_pos),
        )
    got_z = diffusion.condition_pair(z_trunk, rel_pos, params)
    got_s = diffusion.condition_single(t_hat, s_inputs, params, sigma_data=16.0)
    np.testing.assert_allclose(np.asarray(got_z), expected_z.numpy(), atol=ATOL)
    np.testing.assert_allclose(np.asarray(got_s), expected_s.numpy(), atol=1e-4)


def test_the_denoiser_matches() -> None:
    """`DiffusionModule` end to end: conditioning, atom stack, EDM scaling."""
    module = common.DiffusionModule(
        c_atom=C_ATOM,
        c_token=D_TOKEN,
        c_z=D_PAIR,
        c_s_inputs=C_INPUTS,
        fourier_dim=16,
        atom_num_blocks=1,
        atom_num_heads=2,
        token_num_blocks=2,
        token_num_heads=N_HEADS,
        swa_window_size=8,
    ).eval()
    params = torch_state_to_numpy(module)
    data = _atom_inputs(15)
    s_inputs = _normal((1, N_TOKENS, C_INPUTS), 16)
    z_trunk = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 17)
    rel_pos = _normal((1, N_TOKENS, N_TOKENS, D_PAIR), 18)
    x_noisy = _normal((1, N_ATOMS, 3), 19) * 8.0
    t_hat = np.array([12.0], dtype=np.float32)
    zeros = np.zeros((1, N_TOKENS), dtype=np.int64)

    with torch.no_grad():
        expected = module(
            x_noisy=_t(x_noisy),
            t_hat=_t(t_hat),
            ref_pos=_t(data["ref_pos"]),
            ref_charge=_t(data["ref_charge"]),
            ref_mask=_t(data["ref_mask"]),
            ref_element=_t(data["ref_element"]),
            ref_atom_name_chars=_t(data["ref_atom_name_chars"]),
            ref_space_uid=_t(data["ref_space_uid"]),
            tok_idx=_t(data["tok_idx"]),
            s_inputs=_t(s_inputs),
            s_trunk=None,
            z_trunk=_t(z_trunk),
            relative_position_encoding=_t(rel_pos),
            asym_id=_t(zeros),
            residue_index=_t(zeros),
            entity_id=_t(zeros),
            token_index=_t(zeros),
            sym_id=_t(zeros),
        )["x_denoised"]

    settings = diffusion.DiffusionSettings(
        c_atom=C_ATOM,
        c_token=D_TOKEN,
        c_z=D_PAIR,
        atom_n_blocks=1,
        atom_n_heads=2,
        token_n_blocks=2,
        token_n_heads=N_HEADS,
        half_window=4,
    )
    cache = diffusion.build_cache(
        data["ref_pos"],
        data["ref_charge"],
        data["ref_mask"],
        data["ref_element"],
        data["ref_atom_name_chars"],
        data["ref_space_uid"],
        data["tok_idx"],
        z_trunk,
        rel_pos,
        params,
        settings=settings,
    )
    got, _ = diffusion.diffusion_module(
        x_noisy, t_hat, s_inputs, cache, params, settings=settings
    )
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=2e-3, rtol=2e-3)


def test_the_schedule_is_clipped_not_rescaled() -> None:
    """68 nominal steps become 48 denoiser calls, and the values still match."""
    head = common.DiffusionStructureHead(configuration.ESMFold2Config()).eval()
    with torch.no_grad():
        expected = head.inference_noise_schedule(68)
        clipped = expected[expected <= 256.0]
        clipped = torch.nn.functional.pad(clipped, (1, 0), value=256.0)

    got = diffusion.noise_schedule(diffusion.DiffusionSettings())
    np.testing.assert_allclose(got, clipped.numpy(), atol=1e-4, rtol=1e-5)
    assert got.shape[0] - 1 == 48, "the clip drops the tail above the cap"


def test_the_kabsch_alignment_matches() -> None:
    x = _normal((2, N_ATOMS, 3), 20)
    x_gt = _normal((2, N_ATOMS, 3), 21)
    mask = np.ones((2, N_ATOMS), dtype=np.float32)
    mask[1, -4:] = 0.0
    with torch.no_grad():
        expected = common.DiffusionStructureHead._weighted_rigid_align(
            _t(x), _t(x_gt), _t(mask), _t(mask)
        )
    got = diffusion.weighted_rigid_align(x, x_gt, mask, mask)
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=1e-4, rtol=1e-4)


def test_the_random_rotation_matches_for_a_given_quaternion() -> None:
    """The signed normalisation, which halves of the draws depend on."""
    torch.manual_seed(0)
    expected = common.DiffusionStructureHead._random_rotations(
        4, torch.float32, torch.device("cpu")
    )
    torch.manual_seed(0)
    q = torch.randn((4, 4), dtype=torch.float32)
    got = diffusion.quaternion_to_rotation(q.numpy())
    np.testing.assert_allclose(np.asarray(got), expected.numpy(), atol=ATOL, rtol=ATOL)
