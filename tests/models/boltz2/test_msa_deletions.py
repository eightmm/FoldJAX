"""The Boltz-2 `msa_deletions` option: released regression vs restored loop.

Upstream Boltz-2 v2.2.0+ zeroes every MSA deletion feature -- see
`docs/boltz2-upstream-msa-deletion-regression-2026-09-10.md`. The port keeps
that released behaviour by default and reinstates the pre-`04d27c71` loop
behind `msa_deletions=restored`.

The fixture drives `process_msa_features` on a hand-built `Tokenized` stand-in
rather than the full preprocessing pipeline: the mechanism lives entirely in
the MSA path, and the molecule database the pipeline needs is a setup asset the
rest of this directory already skips on.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.feature import featurizerv2
from foldjax.models.boltz2.data.parse.a3m import _parse_a3m

#: A four-row alignment over a ten-residue query.
#:
#: The query row carries no lowercase insertion, which is exactly what arms the
#: regression: its `del_start == del_end == 0`, so the released loop slices an
#: empty array and every later row slices that empty slice.  Each homolog keeps
#: ten non-lowercase columns so the rows stay aligned to the query.
A3M = (
    ">query\n"
    "ACDEFGHIKL\n"
    ">h1\n"
    "ACDaaEFGHIKL\n"
    ">h2\n"
    "ACDEFgGHIKL\n"
    ">h3\n"
    "AWDEFGHIKppL\n"
)

#: `(seq_idx, res_idx, deletion count)` the a3m parser records for `A3M`.
#: Row order out of `construct_paired_msa` is the sequence order here because
#: no row is annotated with a taxonomy, so no pairing reorders them.
DELETION_RECORDS = ((1, 3, 2), (2, 5, 1), (3, 9, 2))

#: Every array `process_msa_features` returns on the non-affinity path.
MSA_KEYS = (
    "msa",
    "msa_paired",
    "deletion_value",
    "has_deletion",
    "deletion_mean",
    "profile",
    "msa_mask",
)

GOLDEN = Path(__file__).parent / "fixtures/msa_deletions_released.npz"


def build_tokenized() -> SimpleNamespace:
    """A minimal `Tokenized` carrying one protein chain and `A3M`."""

    msa = _parse_a3m(io.StringIO(A3M), taxonomy=None)
    first = msa.sequences[0]
    query = msa.residues[first["res_start"] : first["res_end"]]
    num_residues = len(query)

    residues = np.zeros(num_residues, dtype=[("res_type", np.dtype("i4"))])
    residues["res_type"] = query["res_type"]
    chains = np.zeros(
        1, dtype=[("res_idx", np.dtype("i4")), ("res_num", np.dtype("i4"))]
    )
    chains["res_num"] = num_residues
    tokens = np.zeros(
        num_residues,
        dtype=[("asym_id", np.dtype("i4")), ("res_idx", np.dtype("i4"))],
    )
    tokens["res_idx"] = np.arange(num_residues)

    return SimpleNamespace(
        tokens=tokens,
        structure=SimpleNamespace(chains=chains, residues=residues),
        msa={0: msa},
        record=SimpleNamespace(id="msa-deletions-fixture"),
    )


def msa_features(**kwargs: object) -> dict[str, np.ndarray]:
    """Run `process_msa_features` over the fixture and return NumPy arrays."""

    features = featurizerv2.process_msa_features(
        data=build_tokenized(),
        random=np.random.default_rng(42),
        max_seqs_batch=const.max_msa_seqs,
        max_seqs=const.max_msa_seqs,
        **kwargs,
    )
    return {
        name: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        for name, value in features.items()
    }


def test_released_zeroes_every_deletion_feature() -> None:
    features = msa_features(msa_deletions="released")

    assert features["msa"].shape == (4, 10)
    assert not features["has_deletion"].any()
    assert not features["deletion_value"].any()
    assert not features["deletion_mean"].any()


def test_restored_recovers_one_value_per_recorded_deletion() -> None:
    features = msa_features(msa_deletions="restored")

    has_deletion = features["has_deletion"].astype(bool)
    assert int(has_deletion.sum()) == len(DELETION_RECORDS)
    assert {
        (int(row), int(token)) for row, token in zip(*np.nonzero(has_deletion))
    } == {(seq_idx, res_idx) for seq_idx, res_idx, _ in DELETION_RECORDS}

    # The released featurizer applies `pi / 2 * atan(d / 3)`, not the
    # `2 / pi` spelling the task description carries; assert the code.
    for seq_idx, res_idx, count in DELETION_RECORDS:
        assert features["deletion_value"][seq_idx, res_idx] == pytest.approx(
            math.pi / 2 * math.atan(count / 3), rel=1e-6
        )
    assert features["deletion_mean"].shape == (10,)
    assert int(np.count_nonzero(features["deletion_mean"])) == len(
        {res_idx for _, res_idx, _ in DELETION_RECORDS}
    )


def test_released_is_the_default() -> None:
    for name, value in msa_features().items():
        np.testing.assert_array_equal(
            value, msa_features(msa_deletions="released")[name], err_msg=name
        )


def test_released_matches_the_pre_change_featurizer_bitwise() -> None:
    """Pin the default against arrays captured before the option existed.

    Captured from the featurizer at `018f3f4` -- the commit that documented the
    regression and predates this option -- so the guard is against that source,
    not against another spelling of the same post-change code.
    """
    golden = np.load(GOLDEN)
    features = msa_features(msa_deletions="released")

    assert set(golden.files) == set(MSA_KEYS) == set(features)
    for name in MSA_KEYS:
        assert features[name].dtype == golden[name].dtype, name
        assert np.array_equal(features[name], golden[name]), name


def test_unknown_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="msa_deletions"):
        msa_features(msa_deletions="fixed")
