"""The checkpoint inspection CLI, exercised on a synthetic checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.openfold3.cli.inspect_checkpoint import count_blocks, main


def _write(tmp_path: Path, extra: dict | None = None) -> Path:
    from safetensors.numpy import save_file

    state = {
        "trunk.pairformer_stack.blocks.0.pair_stack.tri_mul_out.linear_a_p.weight":
            np.zeros((4, 4), dtype=np.float32),
        "trunk.pairformer_stack.blocks.1.pair_stack.tri_mul_out.linear_b_p.weight":
            np.zeros((4, 4), dtype=np.float32),
        "trunk.msa_module.blocks.0.outer_product_mean.linear_1.weight":
            np.zeros((3, 4), dtype=np.float32),
        "diffusion_module.atom_attn_enc.linear_q.0.weight":
            np.zeros((2, 4), dtype=np.float32),
    }
    state.update(extra or {})
    path = tmp_path / "weights.safetensors"
    save_file(state, path)
    return path


def test_counts_blocks_per_stack(tmp_path: Path) -> None:
    from safetensors.numpy import load_file

    state = load_file(str(_write(tmp_path)))
    assert count_blocks(state, "pairformer_stack.blocks") == 2
    assert count_blocks(state, "msa_module.blocks") == 1
    assert count_blocks(state, "template_pair_stack.blocks") is None


def test_cli_reports_structure_and_layout(tmp_path: Path, capsys) -> None:
    assert main([str(_write(tmp_path)), "--depth", "1"]) == 0
    out = capsys.readouterr().out
    assert "tensors: 4" in out
    assert "unfused" in out
    assert "trunk" in out
    assert "diffusion_module" in out
    assert "pairformer_stack.blocks" in out


def test_cli_detects_a_fused_checkpoint(tmp_path: Path, capsys) -> None:
    path = _write(
        tmp_path,
        extra={"trunk.x.linear_ab_p.weight": np.zeros((8, 4), dtype=np.float32)},
    )
    # Remove the unfused markers so only the fused ones remain.
    from safetensors.numpy import load_file, save_file

    state = {
        key: value
        for key, value in load_file(str(path)).items()
        if "linear_a_p" not in key and "linear_b_p" not in key
    }
    fused_path = tmp_path / "fused.safetensors"
    save_file(state, fused_path)

    assert main([str(fused_path)]) == 0
    assert "fused" in capsys.readouterr().out


def test_cli_grep_filters_keys(tmp_path: Path, capsys) -> None:
    assert main([str(_write(tmp_path)), "--grep", "msa_module"]) == 0
    out = capsys.readouterr().out
    assert "outer_product_mean" in out
    assert "atom_attn_enc" not in out.split("keys matching")[-1]


def test_cli_grep_respects_the_limit(tmp_path: Path, capsys) -> None:
    assert main([str(_write(tmp_path)), "--grep", "weight", "--limit", "1"]) == 0
    assert "raise --limit" in capsys.readouterr().out


def test_entrypoint_exits_with_the_status(tmp_path: Path) -> None:
    from foldjax.models.openfold3.cli.inspect_checkpoint import entrypoint

    with pytest.raises(SystemExit):
        entrypoint()
