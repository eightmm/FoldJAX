"""Layouts the released checkpoint uses that upstream's module code does not.

A checkpoint is not just values: it is values in whatever shape the code that saved
them happened to produce. Reading the module source is not enough, and both cases
here were found only by loading the real released weights -- randomized fixtures
built from the module definitions cannot expose them, because they are built from
the definitions.
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import _squeeze_trailing


def _optional_released_checkpoint() -> Path | None:
    repository = Path(__file__).resolve().parents[3]
    explicit = os.environ.get("OPENFOLD3_CHECKPOINT")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        repository
        / ".foldjax"
        / "weights"
        / "openfold3"
        / "of3_ft3_v1.pt",
        repository.parent
        / "openfold3"
        / "openfold3_weights"
        / "checkpoints"
        / "of3_ft3_v1.pt",
    ]
    return next(
        (path for path in candidates if path is not None and path.is_file()), None
    )


def _tree_digest(tree) -> tuple[str, int, int]:
    digest = hashlib.sha256(str(jax.tree.structure(tree)).encode())
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest(), len(leaves), sum(leaf.nbytes for leaf in leaves)


def test_squeeze_trailing_accepts_the_released_column_vector() -> None:
    """``fourier_emb.w`` ships as ``(c, 1)`` though it is declared ``(c,)``.

    Upstream declares the buffer ``torch.empty(c)``, but the released
    ``of3_ft3_v1`` checkpoint stores it with a trailing axis. Left alone it turns
    ``x * w`` into a ``[..., c, 1]`` outer product, and the noise-level embedding
    then fails to add to the single representation with
    ``add got incompatible shapes: (5, 76, 384), (256, 1, 384)``.
    """
    released = np.arange(8, dtype=np.float32).reshape(8, 1)
    squeezed = _squeeze_trailing(released)
    assert squeezed.shape == (8,)
    np.testing.assert_array_equal(np.asarray(squeezed), released[:, 0])


def test_squeeze_trailing_leaves_a_declared_vector_alone() -> None:
    """A checkpoint that stores the declared shape must pass through unchanged."""
    declared = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(_squeeze_trailing(declared)), declared)


def test_squeeze_trailing_keeps_real_matrices() -> None:
    """Only trailing axes of length one go; a ``(c, k)`` weight must survive.

    Without this the same helper would quietly flatten genuine matrices if it were
    reused, which is the failure mode of a "just squeeze it" fix.
    """
    matrix = np.zeros((4, 3), dtype=np.float32)
    assert _squeeze_trailing(matrix).shape == (4, 3)


def test_chemistry_table_round_trips_through_the_feature_file(tmp_path) -> None:
    """Features must carry the chemistry table, so inference needs no upstream.

    ``predict`` used to rebuild the representative-atom table by importing
    upstream, which broke the whole point of the featurize/predict split: the
    inference half is supposed to run with no torch, no rdkit and no checkout.
    """
    from foldjax.models.openfold3.data import save_features, split_chemistry
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    fields = RepresentativeAtomTable._fields
    table = RepresentativeAtomTable(
        **{
            # The released table is indexed by the 32 model residue bins.
            # A three-entry stand-in used to exercise serialization only, but
            # it is not a table inference can safely consume and is now
            # rejected at the archive boundary.
            name: np.full(32, index, dtype=np.float32)
            for index, name in enumerate(fields)
        }
    )
    features = {"token_mask": np.ones((1, 4), dtype=np.float32)}

    path = save_features(features, tmp_path / "f.npz", representative_atoms=table)
    loaded = dict(np.load(path))
    restored_features, restored_table = split_chemistry(loaded)

    assert set(restored_features) == {"token_mask"}
    assert restored_table is not None, "the table did not survive the round trip"
    for name in fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(restored_table, name)), getattr(table, name)
        )


def test_split_chemistry_reports_a_file_written_before_the_table_travelled(
    tmp_path,
) -> None:
    """An older feature file has to be detected, not half-read.

    Returning a partially populated table would surface as a wrong prediction
    rather than an error, so the contract is all-or-``None``.
    """
    from foldjax.models.openfold3.data import split_chemistry

    features, table = split_chemistry({"token_mask": np.ones((1, 4), dtype=np.float32)})
    assert table is None
    assert set(features) == {"token_mask"}


def test_split_chemistry_rejects_a_partial_table() -> None:
    """A file with some chemistry keys missing must not produce a table."""
    from foldjax.models.openfold3.data import split_chemistry
    from foldjax.models.openfold3.data.featurize import CHEMISTRY_PREFIX
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    one = RepresentativeAtomTable._fields[0]
    _, table = split_chemistry(
        {
            "token_mask": np.ones((1, 4), dtype=np.float32),
            f"{CHEMISTRY_PREFIX}{one}": np.zeros(3, dtype=np.float32),
        }
    )
    assert table is None


@pytest.mark.torch_parity
def test_the_released_checkpoint_really_stores_a_column_vector(
    openfold3_source,
) -> None:
    """Guard the premise of the squeeze against the actual file, when present.

    Skipped rather than failed when the checkpoint is not on this machine: the
    point is to notice if a future release changes the layout, not to require
    a 2 GB download to run the suite.
    """
    from pathlib import Path

    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint

    candidates = [
        # Inside the checkout since 2026-08-06; the sibling layout came first.
        openfold3_source / "openfold3_weights" / "checkpoints" / "of3_ft3_v1.pt",
        openfold3_source.parent
        / "openfold3_weights"
        / "checkpoints"
        / "of3_ft3_v1.pt",
    ]
    path = next((c for c in candidates if Path(c).is_file()), None)
    if path is None:
        pytest.skip(f"no released checkpoint at {candidates[0]}")

    state = load_checkpoint(path)
    keys = [name for name in state if name.endswith("fourier_emb.w")]
    assert keys, "no fourier embedding buffer in the checkpoint"
    for name in keys:
        shape = np.shape(state[name])
        assert shape[1:] in ((), (1,)), (
            f"{name} has shape {shape}; the mapper only knows how to read "
            "(c,) and (c, 1)"
        )


def test_the_released_checkpoint_alias_group_is_completely_prunable() -> None:
    """Gate the memory premise against the access-gated checkpoint when present."""
    from foldjax.models.openfold3.bridge.checkpoint import load_checkpoint
    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_inference_params,
        prune_sample_diffusion_aliases,
        resolve_model_prefix,
    )

    path = _optional_released_checkpoint()
    if path is None:
        pytest.skip(
            "no released checkpoint; set OPENFOLD3_CHECKPOINT to enable this gate"
        )

    state = load_checkpoint(path)
    alias_keys = [key for key in state if ".sample_diffusion." in f".{key}"]
    alias_bytes = sum(state[key].nbytes for key in alias_keys)
    prefix = resolve_model_prefix(state)
    baseline = map_inference_params(state, prefix)
    jax.block_until_ready(baseline)
    baseline_digest = _tree_digest(baseline)
    del baseline
    gc.collect()

    removed = prune_sample_diffusion_aliases(state, prefix=prefix)
    pruned = map_inference_params(state, prefix)
    jax.block_until_ready(pruned)

    assert removed == len(alias_keys) == 767
    assert alias_bytes == 812_960_240
    assert not any(key.startswith("sample_diffusion.") for key in state)
    assert _tree_digest(pruned) == baseline_digest == (
        "9569dfaf02f752c33753d0231ee9c00e4587fc413ebe2c84c2e62070c404657b",
        593,
        1_473_188_784,
    )
