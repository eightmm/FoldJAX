"""OpenDDE resolves its chunk policy from the residue count, not the larger one.

Upstream reads the threshold table twice: `N_token` for the residue trunk and
`len(parent_residue_idx)` for the structural branch (opendde.py:1481,
1500-1502). This port fed `max(n_residue, n_structural)` to both.

That is not a rounding difference. The structural token count is a near-constant
1.95x the residue count on every bench case -- 257/132, 948/490, 1886/970,
2978/1531 -- so the trunk always resolved a band or two too low, and the bands
are far apart:

    tokens   max()   residue count
       970     256            None
      1531      32             512

48 blocks where upstream runs 3, on a contraction that re-reads the whole of
`b` once per block. Nothing failed; the only symptom was OpenDDE's time growing
as N^2.71 against upstream's N^1.66.
"""

from __future__ import annotations

import pytest

from foldjax.models.protenix.chunking import (
    protenix_dynamic_token_chunk_size,
    resolve_chunk_config,
)

#: (residue tokens, structural tokens), measured by featurizing each bench case.
#: The ratio is what makes the two counts land in different bands.
BENCH_SHAPES = ((132, 257), (490, 948), (970, 1886), (1531, 2978))


@pytest.mark.parametrize("residue, structural", BENCH_SHAPES)
def test_structural_count_is_about_twice_the_residue_count(
    residue: int, structural: int
) -> None:
    """The premise. If this stops holding, the bands below stop mattering."""
    assert 1.9 <= structural / residue <= 2.0


def test_the_two_counts_resolve_to_different_bands_where_it_matters() -> None:
    """Names the exact cases the old code got wrong, so a revert is visible."""
    assert protenix_dynamic_token_chunk_size(970) is None
    assert protenix_dynamic_token_chunk_size(1886) == 256
    assert protenix_dynamic_token_chunk_size(1531) == 512
    assert protenix_dynamic_token_chunk_size(2978) == 32


@pytest.mark.parametrize("residue, structural", BENCH_SHAPES)
def test_resolving_from_the_residue_count_never_narrows_further(
    residue: int, structural: int
) -> None:
    """The residue count is the looser of the two, at every size.

    Asserted as an inequality rather than as fixed values so it keeps meaning
    something if the threshold table is ever re-tuned: whatever the bands are,
    resolving from the larger count can only ever chunk harder.
    """
    from_residue = resolve_chunk_config(n_token=residue, n_sample=5)
    from_max = resolve_chunk_config(n_token=max(residue, structural), n_sample=5)

    def rows(value: int | None) -> float:
        return float("inf") if value is None else value

    assert rows(from_residue.triangle_mul_chunk_size) >= rows(
        from_max.triangle_mul_chunk_size
    )
    assert rows(from_residue.triangle_att_q_chunk_size) >= rows(
        from_max.triangle_att_q_chunk_size
    )


def test_the_cli_resolves_from_the_residue_count() -> None:
    """Read the call site, because that is where the wrong count was passed.

    A test of `resolve_chunk_config` alone cannot see this: the function was
    always correct, and the defect was entirely in what it was handed.
    """
    import inspect

    from foldjax.models.opendde.cli import predict

    source = inspect.getsource(predict)
    assert "n_token=n_residue" in source
    assert "n_token=max(n_residue, n_structural)" not in source
