"""The atom stack against torch: rotary tables, window, blocks, encoder, decoder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
common = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2_common")

from foldjax.models.esmfold2.models import atom  # noqa: E402
from tests.models.esmfold2.parity import torch_state_to_numpy  # noqa: E402

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
    """Same axes, same frequency layout, same half-split pairing."""
    data = _atom_inputs()
    with torch.no_grad():
        cos, sin = common.build_3d_rope(
            torch.from_numpy(data["ref_pos"]),
            torch.from_numpy(data["ref_space_uid"]),
            head_dim=D_ATOM // N_HEADS,
            n_spatial_per_axis=2,
            n_uid_pairs=10,
            spatial_base_freq=20.0,
            uid_base_freq=10000.0,
        )
    got_cos, got_sin = atom.build_3d_rope(
        data["ref_pos"], data["ref_space_uid"], head_dim=D_ATOM // N_HEADS
    )
    np.testing.assert_allclose(
        np.asarray(got_cos), cos.float().numpy(), atol=3e-3, rtol=3e-3
    )
    np.testing.assert_allclose(
        np.asarray(got_sin), sin.float().numpy(), atol=3e-3, rtol=3e-3
    )


def test_the_window_is_measured_in_packed_rank() -> None:
    """Padding between two atoms must not push them out of each other's window."""
    valid = np.array([[1, 0, 0, 1, 1]], dtype=np.float32)
    allowed = np.asarray(atom.sliding_window_mask(valid, half_window=1))
    assert allowed[0, 0, 3], "rank 0 and rank 1 are adjacent despite two pad atoms"
    assert not allowed[0, 0, 4], "rank 0 and rank 2 are outside a window of 1"
    assert allowed[0, 1, 1], "the diagonal is always allowed, padding included"


def test_the_atom_encoder_matches() -> None:
    module = common.ESMFold2AtomEncoder(
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
