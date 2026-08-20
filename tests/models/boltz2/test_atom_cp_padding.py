from __future__ import annotations

import numpy as np

from foldjax.models.boltz2.data.bucket import (
    align_padding_plan_for_context_parallel,
)


def test_cp_padding_aligns_pair_axes_and_supplies_a_complete_atom_halo() -> None:
    feats = {
        "token_pad_mask": np.ones((1, 7), dtype=np.float32),
        "atom_pad_mask": np.ones((1, 33), dtype=np.float32),
        "msa": np.ones((1, 1, 7), dtype=np.int32),
    }
    plan = align_padding_plan_for_context_parallel(
        feats,
        None,
        cp_rows=2,
        cp_cols=2,
        query_window=32,
        key_window=128,
    )
    assert plan.target["tokens"] == 8
    # Two query windows per CP row are needed for a 3-half-window halo.
    assert plan.target["atoms"] == 128
    assert plan.actual == {"tokens": 7, "atoms": 33, "msa": 1}
