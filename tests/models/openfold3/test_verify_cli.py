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
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.models.openfold3.cli.verify_checkpoint import _parser, main
from foldjax.models.openfold3.data import save_features
from foldjax.models.openfold3.dtype import narrow_dtype
from foldjax.models.openfold3.inference import RELEASED_BLOCK_COUNTS
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
)

from .feature_fixture import minimal_features
from .test_dtype_is_realized import _synthetic_inference_params


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


def _batch(tmp_path: Path) -> Path:
    table = RepresentativeAtomTable(
        *(np.zeros(32, dtype=np.float32) for _ in RepresentativeAtomTable._fields)
    )
    return save_features(
        minimal_features(), tmp_path / "batch.npz", representative_atoms=table
    )


def test_batch_sizes_are_derived_when_not_given() -> None:
    args = _parser().parse_args(["weights.pt", "--batch", "batch.npz"])
    assert args.tokens is None
    assert args.atoms is None


def test_optional_size_assertion_fails_before_checkpoint_loading(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "does-not-exist.safetensors"
    assert (
        main([str(missing), "--batch", str(_batch(tmp_path)), "--tokens", "5"])
        == 1
    )
    assert "batch has 4 tokens" in capsys.readouterr().out


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


def test_verifier_passes_normalized_static_chain_count(
    tmp_path: Path, monkeypatch
) -> None:
    import jax
    import jax.numpy as jnp

    from foldjax.models.openfold3 import data, inference
    from foldjax.models.openfold3.bridge import checkpoint, torch_mapping
    from foldjax.models.openfold3.cli import inspect_checkpoint

    raw = {
        "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "atom_mask": np.asarray([[1, 1]], dtype=np.float32),
        "asym_id": np.asarray([[10, 30, 999]], dtype=np.int64),
    }
    # A real `InferenceParams` tree with synthetic leaves: this command
    # narrows the loaded weights the way the predict CLI and the managed
    # backend do, and that walks named subtrees and calls `_replace` on them.
    params = _synthetic_inference_params()
    seen: dict[str, object] = {}

    monkeypatch.setattr(data, "load_features", lambda path: (raw, object()))
    monkeypatch.setattr(
        data, "prepare_msa_cycle_features", lambda features, depth, **kwargs: features
    )
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(checkpoint, "detect_fused_tri_mul", lambda state: False)
    monkeypatch.setattr(
        inspect_checkpoint,
        "count_blocks",
        lambda state, root: RELEASED_BLOCK_COUNTS[root],
    )
    monkeypatch.setattr(
        torch_mapping, "map_inference_params", lambda state, prefix: params
    )
    monkeypatch.setattr(
        inference,
        "released_config",
        # `bfloat16` rather than the shipped `float32`: this command has no
        # dtype flag, so under the default the narrowing cast is the identity
        # and the assertion at the end would hold whether or not the command
        # called it. Naming the narrow value is what makes it able to fail.
        lambda **kwargs: SimpleNamespace(
            msa_depth=1024,
            num_recycles=4,
            dtype=kwargs.get("dtype", "bfloat16"),
            confidence_dtype=kwargs.get(
                "confidence_dtype", kwargs.get("dtype", "bfloat16")
            ),
        ),
    )

    def fake_predict(key, batch, params, config, table, *, n_chain=None):
        seen.update(
            n_chain=n_chain,
            asym_id=np.asarray(batch["asym_id"]),
            trunk_dtype={
                leaf.dtype
                for leaf in jax.tree.leaves(params.trunk.pairformer_stack)
            },
        )
        values = {"coordinates": np.zeros((1, 2, 3), dtype=np.float32)}
        return SimpleNamespace(_asdict=lambda: values)

    monkeypatch.setattr(inference, "predict", fake_predict)
    monkeypatch.setattr(jnp, "asarray", np.asarray)
    monkeypatch.setattr(jax.random, "key", lambda seed: seed)
    monkeypatch.setattr(jax, "devices", lambda: ["test-device"])

    batch = tmp_path / "batch.npz"
    batch.touch()
    weights = tmp_path / "weights.pt"
    weights.touch()

    assert main([str(weights), "--batch", str(batch)]) == 0
    assert seen["n_chain"] == 2
    np.testing.assert_array_equal(seen["asym_id"], [[0, 1, 0]])
    # This command exists to report what a real run does, so it has to narrow
    # the weights the real entry points narrow. Left uncast, the config would
    # ask for bfloat16 activations against float32 parameters and every
    # matmul would promote back -- a program nothing runs.
    assert seen["trunk_dtype"] == {np.dtype(narrow_dtype("bfloat16"))}
