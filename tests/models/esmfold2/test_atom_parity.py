"""The atom stack against torch: rotary tables, window, blocks, encoder, decoder."""

from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.esmfold2.models import atom  # noqa: E402


def _upstream():
    """The upstream atom stack, or a skip.

    Not a module-level `importorskip`, which skipped the window test too --
    that one is pure JAX and has no business needing torch installed.

    `transformers` moved this: it was `modeling_esmfold2_common` with free
    functions taking explicit widths, and is now `modeling_esmfold2` with
    modules built from a config. Importing the old path is how this file spent
    a long time reporting "skipped" on a machine that could have run it.
    """
    pytest.importorskip("torch")
    return pytest.importorskip(
        "transformers.models.esmfold2.modeling_esmfold2"
    )

ATOL = 2e-5
# The rotary table needs head_dim // 2 >= 3*2 + 10 = 16, so head_dim >= 32 --
# which is what the released widths give (d_atom 128 over 4 heads). A smaller
# toy would not exercise the same code path; it would fail to build one.
N_ATOMS, D_ATOM, N_HEADS = 12, 64, 2


def _atom_inputs(seed=0, n_atoms=N_ATOMS):
    rng = np.random.default_rng(seed)
    return {
        "ref_pos": rng.standard_normal((1, n_atoms, 3)).astype(np.float32),
        "ref_space_uid": np.arange(n_atoms, dtype=np.int64)[None] // 3,
        "ref_charge": np.zeros((1, n_atoms), dtype=np.float32),
        "ref_element": np.eye(128, dtype=np.float32)[np.full(n_atoms, 6)][None],
        "ref_atom_name_chars": np.eye(64, dtype=np.float32)[
            np.tile(np.array([35, 33, 0, 0]), (n_atoms, 1))
        ][None],
        "atom_to_token": (np.arange(n_atoms, dtype=np.int64)[None] // 3),
        "atom_attention_mask": np.ones((1, n_atoms), dtype=np.float32),
    }


def test_the_rotary_tables_match_upstream() -> None:
    """Same axes, same frequency layout, same half-split pairing -- exactly.

    This asserted `atol=rtol=3e-3`, without the reason `PROJECT.md` asks for
    above 1e-3, in a file that otherwise holds itself to 2e-5. Measured, the
    two sides really did differ by 1.9e-3 in float32, so the tolerance was
    sized to a real gap -- and the gap was a dtype mismatch rather than a
    disagreement. FoldJAX returns these tables in bfloat16 because upstream
    hardcodes that, and the comparison took upstream's float32 reference.
    2e-3 is what bfloat16 costs on values in [-1, 1]: the tolerance was the
    size of the cast, and would have hidden any defect smaller than it.

    Upstream takes the dtype as an argument now, where the old free function
    did not, so the two can be compared as the same thing. They are
    bit-identical, and this asserts that rather than a tolerance.
    """
    torch = pytest.importorskip("torch")
    modeling = _upstream()
    from transformers.models.esmfold2.configuration_esmfold2 import (
        EsmFold2AtomEncoderConfig,
    )

    head_dim = D_ATOM // N_HEADS
    data = _atom_inputs()

    config = EsmFold2AtomEncoderConfig()
    for name, value in (
        ("head_dim", head_dim),
        ("num_spatial_rope_pairs_per_axis", 2),
        ("num_uid_rope_pairs", 10),
        ("spatial_rope_base_frequency", 20.0),
        ("uid_rope_base_frequency", 10000.0),
    ):
        setattr(config, name, value)
    rope = modeling.EsmFold2RotaryEmbedding(config)

    with torch.no_grad():
        cos, sin = rope(
            torch.from_numpy(data["ref_pos"]),
            torch.from_numpy(data["ref_space_uid"]),
            torch.bfloat16,
        )
    got_cos, got_sin = atom.build_3d_rope(
        data["ref_pos"], data["ref_space_uid"], head_dim=head_dim
    )

    # Upstream duplicates the half-width table to the full head dim and applies
    # plain rotate-half; FoldJAX returns the half. Compare the half that exists
    # on both sides, and check upstream's two halves really are the same one.
    half = np.asarray(got_cos).shape[-1]
    assert cos.shape[-1] == 2 * half
    assert torch.equal(cos[..., :half], cos[..., half:])
    assert torch.equal(sin[..., :half], sin[..., half:])

    for name, expected, got in (("cos", cos, got_cos), ("sin", sin, got_sin)):
        reference = expected[..., :half].float().numpy()
        produced = np.asarray(got).astype(np.float32)
        assert np.array_equal(produced, reference), (
            f"{name} table is no longer bit-identical to upstream's"
        )


def test_the_window_is_measured_in_packed_rank() -> None:
    """Padding between two atoms must not push them out of each other's window."""
    valid = np.array([[1, 0, 0, 1, 1]], dtype=np.float32)
    allowed = np.asarray(atom.sliding_window_mask(valid, half_window=1))
    assert allowed[0, 0, 3], "rank 0 and rank 1 are adjacent despite two pad atoms"
    assert not allowed[0, 0, 4], "rank 0 and rank 2 are outside a window of 1"
    assert allowed[0, 1, 1], "the diagonal is always allowed, padding included"


@pytest.mark.skip(
    reason="upstream's atom encoder is built from EsmFold2Config plus an atom "
    "sub-config now, not the explicit widths this passes, and its parameter "
    "names moved with it -- so `torch_state_to_numpy` would map onto a "
    "different state dict. Rewriting it against the current API is real work, "
    "and a comparison that silently lines up the wrong tensors is worse than "
    "one that says it is not running."
)
def test_the_atom_encoder_matches() -> None:
    modeling = _upstream()
    torch = pytest.importorskip("torch")
    from tests.models.esmfold2.parity import torch_state_to_numpy

    module = modeling.ESMFold2AtomEncoder(
        d_atom=D_ATOM, d_token=D_ATOM * 2, n_blocks=2, n_heads=N_HEADS,
        swa_window_size=4, expansion_ratio=2, structure_prediction=False,
    ).eval()
    data = _atom_inputs(1)
    with torch.no_grad():
        expected = module(
            ref_pos=torch.from_numpy(data["ref_pos"]),
            atom_attention_mask=torch.from_numpy(data["atom_attention_mask"]),
            ref_space_uid=torch.from_numpy(data["ref_space_uid"]),
            ref_charge=torch.from_numpy(data["ref_charge"]),
            ref_element=torch.from_numpy(data["ref_element"]),
            ref_atom_name_chars=torch.from_numpy(data["ref_atom_name_chars"]),
            atom_to_token=torch.from_numpy(data["atom_to_token"]),
        )[0]
    got, _, _, _ = atom.atom_encoder(
        data["ref_pos"], data["atom_attention_mask"], data["ref_space_uid"],
        data["ref_charge"], data["ref_element"], data["ref_atom_name_chars"],
        data["atom_to_token"], torch_state_to_numpy(module),
        n_blocks=2, n_heads=N_HEADS, half_window=2,
    )
    np.testing.assert_allclose(
        np.asarray(got), expected.float().numpy(), atol=2e-3, rtol=2e-3
    )
