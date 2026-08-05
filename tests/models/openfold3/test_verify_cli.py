"""The verification CLI, on synthetic checkpoints.

Synthetic rather than real weights because the decisions are what matter here --
prefix detection, the block-count gate, and refusing to proceed when a checkpoint
does not match the released config -- and a 2 GB download should not be needed to run
the suite. The released checkpoint *is* available: `hf download OpenFold/OpenFold3`
fetches it, and `test_released_checkpoint_layouts.py` checks against it when present.
An earlier version of this note said the repo was gated; that was a wrong conclusion
drawn from a raw `curl` returning 403 where the `hf` client succeeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.cli.verify_checkpoint import main
from foldjax.models.openfold3.inference import RELEASED_BLOCK_COUNTS


def _save(tmp_path: Path, state: dict) -> Path:
    from safetensors.numpy import save_file

    path = tmp_path / "weights.safetensors"
    save_file(state, path)
    return path


def test_rejects_a_checkpoint_with_wrong_block_counts(
    tmp_path: Path, capsys
) -> None:
    """One Pairformer block is not the released 48; the CLI must refuse."""
    path = _save(
        tmp_path,
        {
            "trunk.pairformer_stack.blocks.0.x.weight": np.zeros(
                (2, 2), dtype=np.float32
            )
        },
    )
    assert main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out or "MISSING" in out
    assert "do not use that preset" in out


def test_reports_every_expected_stack(tmp_path: Path, capsys) -> None:
    path = _save(
        tmp_path,
        {"trunk.pairformer_stack.blocks.0.x.weight": np.zeros(1, dtype=np.float32)},
    )
    main([str(path)])
    out = capsys.readouterr().out
    for root in RELEASED_BLOCK_COUNTS:
        assert root in out


def test_requires_sizes_alongside_a_batch(tmp_path: Path, capsys) -> None:
    """--batch without --tokens/--atoms is a usage error reported before loading.

    It must not be deferred until after the checkpoint is mapped: mapping a
    mismatched checkpoint raises KeyError, which would hide the usage mistake.
    """
    missing = tmp_path / "does-not-exist.safetensors"
    batch = tmp_path / "batch.npz"
    np.savez(batch, token_mask=np.ones((1, 4), dtype=np.float32))

    with pytest.raises(SystemExit) as excinfo:
        main([str(missing), "--batch", str(batch)])
    assert excinfo.value.code == 2
    assert "--tokens and --atoms are required" in capsys.readouterr().err


def test_layout_detection_is_reported(tmp_path: Path, capsys) -> None:
    path = _save(
        tmp_path,
        {
            "trunk.pairformer_stack.blocks.0.tri_mul_out.linear_ab_p.weight": np.zeros(
                (4, 2), dtype=np.float32
            )
        },
    )
    main([str(path)])
    assert "fused" in capsys.readouterr().out


def test_entrypoint_exits(tmp_path: Path) -> None:
    from foldjax.models.openfold3.cli.verify_checkpoint import entrypoint

    with pytest.raises(SystemExit):
        entrypoint()
