"""One job and one seed must give one input.

The reference-conformer roto-translation in `ref_pos` used to come from an
unseeded generator, so featurizing the same job twice with the same seed gave
two different model inputs. Upstream draws it from torch's global RNG, which
its `--seed` sets; here it is drawn from the job seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.boltz2 import api
from foldjax.paths import weights_dir

# A ligand and a protein, so `ref_pos` holds several independently augmented
# conformers (one per residue and one for the ligand).
_SEQUENCE = "MKTAYIAKQR"
_LIGAND_CCD = "ATP"


def _mols() -> Path:
    mols = weights_dir("boltz2") / "mols"
    if not mols.is_dir():
        pytest.skip("boltz2 molecule directory not fetched")
    return mols


def _featurize(tmp_path: Path, name: str, seed: int) -> dict[str, np.ndarray]:
    feats, _, _ = api.featurize(
        seq=[_SEQUENCE],
        ligand_ccd=[_LIGAND_CCD],
        mols=_mols(),
        out_dir=tmp_path / name,
        seed=seed,
    )
    return feats


@pytest.mark.slow
@pytest.mark.real_store
def test_same_seed_reproduces_every_feature_and_another_seed_moves_only_ref_pos(
    tmp_path: Path,
) -> None:
    first = _featurize(tmp_path, "first", seed=3)
    again = _featurize(tmp_path, "again", seed=3)
    other = _featurize(tmp_path, "other", seed=4)

    assert "ref_pos" in first
    assert first.keys() == again.keys() == other.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], again[name], err_msg=name)
        assert first[name].dtype == again[name].dtype, name

    real = first["atom_pad_mask"][0].astype(bool)
    assert not np.allclose(first["ref_pos"][0][real], other["ref_pos"][0][real]), (
        "a different seed must draw a different reference-conformer augmentation"
    )
    # The augmentation stream is separate from the featurizer's own stream
    # (upstream's constant 42, which also picks conformers), so the seed moves
    # ref_pos and nothing else.
    for name in first:
        if name != "ref_pos":
            np.testing.assert_array_equal(first[name], other[name], err_msg=name)


@pytest.mark.slow
@pytest.mark.real_store
def test_the_augmentation_is_rigid_per_conformer(tmp_path: Path) -> None:
    """Two seeds differ by a rotation and translation of each conformer only."""

    first = _featurize(tmp_path, "first", seed=0)
    other = _featurize(tmp_path, "other", seed=1)

    real = first["atom_pad_mask"][0].astype(bool)
    uid = first["ref_space_uid"][0][real]
    a = first["ref_pos"][0][real].astype(np.float64)
    b = other["ref_pos"][0][real].astype(np.float64)
    assert len(np.unique(uid)) > 2
    for group in np.unique(uid):
        members = uid == group
        da = np.linalg.norm(a[members][:, None] - a[members][None], axis=-1)
        db = np.linalg.norm(b[members][:, None] - b[members][None], axis=-1)
        np.testing.assert_allclose(da, db, atol=1e-4, err_msg=f"conformer {group}")


def test_predict_hands_its_seed_to_the_featurizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    class _StopError(Exception):
        pass

    def fake_featurize(**kwargs: object):
        seen.update(kwargs)
        raise _StopError

    monkeypatch.setattr(api, "featurize", fake_featurize)
    with pytest.raises(_StopError):
        api.predict(
            seq=["ACDE"],
            weights=tmp_path / "weights.safetensors",
            mols=tmp_path / "mols",
            out_dir=tmp_path / "out",
            seed=11,
        )

    assert seen["seed"] == 11
