"""Deterministic Torch parity for Chai ranking frame construction."""

from __future__ import annotations

import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.ranking.frames import (
    abc_is_colinear,
    get_centre_positions_and_mask,
    get_frames_and_mask,
    get_single_atom_frames,
)

pytestmark = pytest.mark.official_parity


def _inputs() -> dict[str, np.ndarray]:
    coords = np.asarray(
        [
            [10.0, 0.0, 0.0],  # protein centre
            [20.0, 0.0, 0.0],  # RNA centre
            [0.0, 0.0, 0.0],  # ligand atom-token 0
            [1.0, 0.0, 0.0],  # ligand atom-token 1
            [0.0, 2.0, 0.0],  # ligand atom-token 2
            [30.0, 0.0, 0.0],  # multi-atom ligand
            [31.0, 0.0, 0.0],  # multi-atom ligand
            [40.0, 0.0, 0.0],  # DNA centre
            [0.0, 0.0, 0.0],  # atom padding
            [0.0, 0.0, 0.0],  # atom padding
        ],
        dtype=np.float32,
    )[None]
    backbone = np.asarray(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [5, 5, 5],
            [7, 7, 7],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )[None]
    return {
        "atom_coords": coords,
        "token_asym_id": np.asarray([[1, 2, 3, 3, 3, 4, 5, 0]], np.int64),
        "token_residue_index": np.zeros((1, 8), dtype=np.int64),
        "token_backbone_frame_mask": np.asarray(
            [[True, True, False, False, False, False, True, False]]
        ),
        "token_centre_atom_index": np.asarray(
            [[0, 1, 2, 3, 4, 5, 7, 0]], dtype=np.int64
        ),
        "token_exists_mask": np.asarray(
            [[True, True, True, True, True, True, True, False]]
        ),
        "atom_exists_mask": np.asarray(
            [[True, True, True, True, True, True, True, True, False, False]]
        ),
        "backbone_frame_idces": backbone,
        "atom_token_index": np.asarray(
            [[0, 1, 2, 3, 4, 5, 5, 6, 0, 0]], dtype=np.int64
        ),
    }


def _upstream_reference(
    tmp_path: Path,
    inputs: dict[str, np.ndarray],
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> dict[str, np.ndarray]:
    input_path = tmp_path / "frame_inputs.npz"
    output_path = tmp_path / "frame_reference.npz"
    np.savez(input_path, **inputs)
    script = r"""
import sys
import numpy as np
import torch
from chai_lab.data.features.token_utils import get_centre_positions_and_mask
from chai_lab.ranking.frames import get_frames_and_mask, get_single_atom_frames

d = np.load(sys.argv[1])
t = lambda name: torch.from_numpy(d[name])
args = dict(
    atom_coords=t("atom_coords"),
    token_asym_id=t("token_asym_id"),
    token_residue_index=t("token_residue_index"),
    token_backbone_frame_mask=t("token_backbone_frame_mask"),
    token_centre_atom_index=t("token_centre_atom_index"),
    token_exists_mask=t("token_exists_mask"),
    atom_exists_mask=t("atom_exists_mask"),
    atom_token_index=t("atom_token_index"),
)
centre, centre_mask = get_centre_positions_and_mask(
    args["atom_coords"],
    args["atom_exists_mask"],
    args["token_centre_atom_index"],
    args["token_exists_mask"],
)
single_indices, single_mask = get_single_atom_frames(**args)
frame_indices, frame_mask = get_frames_and_mask(
    **args,
    backbone_frame_idces=t("backbone_frame_idces"),
)
np.savez(
    sys.argv[2],
    centre=centre.numpy(),
    centre_mask=centre_mask.numpy(),
    single_indices=single_indices.numpy(),
    single_mask=single_mask.numpy(),
    frame_indices=frame_indices.numpy(),
    frame_mask=frame_mask.numpy(),
)
"""
    subprocess.run(
        [upstream_chai_python, "-c", script, input_path, output_path],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output_path) as result:
        return {key: result[key] for key in result.files}


def test_colinearity_threshold_and_degenerate_frames() -> None:
    origin = jnp.asarray([[0.0, 0.0, 0.0]])
    along_x = jnp.asarray([[1.0, 0.0, 0.0]])
    right_angle = jnp.asarray([[0.0, 1.0, 0.0]])
    assert bool(abc_is_colinear(along_x, origin, -along_x)[0])
    assert not bool(abc_is_colinear(along_x, origin, right_angle)[0])
    assert bool(abc_is_colinear(origin, origin, right_angle)[0])


def test_frame_construction_matches_upstream_modalities(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    inputs = _inputs()
    reference = _upstream_reference(
        tmp_path, inputs, upstream_chai_dir, upstream_chai_python
    )
    arrays = {key: jnp.asarray(value) for key, value in inputs.items()}
    centre, centre_mask = get_centre_positions_and_mask(
        arrays["atom_coords"],
        arrays["atom_exists_mask"],
        arrays["token_centre_atom_index"],
        arrays["token_exists_mask"],
    )
    single_indices, single_mask = get_single_atom_frames(
        atom_coords=arrays["atom_coords"],
        token_asym_id=arrays["token_asym_id"],
        token_residue_index=arrays["token_residue_index"],
        token_backbone_frame_mask=arrays["token_backbone_frame_mask"],
        token_centre_atom_index=arrays["token_centre_atom_index"],
        token_exists_mask=arrays["token_exists_mask"],
        atom_exists_mask=arrays["atom_exists_mask"],
        atom_token_index=arrays["atom_token_index"],
    )
    frame_indices, frame_mask = get_frames_and_mask(**arrays)

    np.testing.assert_allclose(np.asarray(centre), reference["centre"], atol=0, rtol=0)
    np.testing.assert_array_equal(np.asarray(centre_mask), reference["centre_mask"])
    np.testing.assert_array_equal(np.asarray(single_mask), reference["single_mask"])
    valid = np.asarray(single_mask)[..., None]
    np.testing.assert_array_equal(
        np.asarray(single_indices)[valid.repeat(3, axis=-1)],
        reference["single_indices"][valid.repeat(3, axis=-1)],
    )
    np.testing.assert_array_equal(np.asarray(frame_indices), reference["frame_indices"])
    np.testing.assert_array_equal(np.asarray(frame_mask), reference["frame_mask"])

    # Protein, RNA, and DNA retain supplied backbone frames; the three
    # atom-tokenized ligand tokens gain generated frames; multi-atom ligand and
    # padding remain invalid.
    np.testing.assert_array_equal(
        np.asarray(frame_mask),
        [[True, True, True, True, True, False, True, False]],
    )
    np.testing.assert_array_equal(
        np.asarray(frame_indices)[0, 2:5],
        [[3, 2, 4], [2, 3, 4], [2, 4, 3]],
    )
