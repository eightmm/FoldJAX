"""The feature-cache key has to move whenever the features would.

A stale hit here is not a slow run, it is a different prediction reported as
this one's.
"""

from pathlib import Path

from foldjax.models.boltz2.data.featurize import _cache_opts, _input_digest


def test_the_msa_cap_is_part_of_the_cache_key() -> None:
    """It was not, so a capped run reused an uncapped run's features.

    `max_msa_depth` selects how many alignment rows reach the model. It is the
    dominant term in peak memory and it moves accuracy, and it was absent from
    the key -- so asking for a cap after an uncapped run returned the uncapped
    features, and asking for none after a capped run returned the capped ones.
    Neither said anything.
    """
    uncapped = _cache_opts(False, "u", "greedy", None)
    capped = _cache_opts(False, "u", "greedy", 1024)
    assert uncapped != capped
    assert 1024 in capped


def test_an_alignment_named_relative_to_the_job_is_hashed(tmp_path: Path) -> None:
    """Boltz job files name alignments relative to themselves, not the CWD.

    Resolving those against the process working directory found nothing, so
    nothing was folded into the digest and the key was blind to the alignment:
    editing an a3m and re-running returned the previous features.
    """
    job = tmp_path / "job.yaml"
    a3m = tmp_path / "hits.a3m"
    job.write_text("sequences:\n  - protein:\n      msa: hits.a3m\n")
    mols = tmp_path / "mols"

    a3m.write_text(">q\nAAAA\n")
    before = _input_digest(job, mols, _cache_opts(False, "u", "greedy", None))
    a3m.write_text(">q\nAAAA\n>hit\nCCCC\n")
    after = _input_digest(job, mols, _cache_opts(False, "u", "greedy", None))

    assert before != after, "editing the alignment must invalidate the cache"


def test_an_absolute_alignment_path_still_works(tmp_path: Path) -> None:
    job = tmp_path / "job.yaml"
    a3m = tmp_path / "hits.a3m"
    a3m.write_text(">q\nAAAA\n")
    job.write_text(f"sequences:\n  - protein:\n      msa: {a3m}\n")
    mols = tmp_path / "mols"

    before = _input_digest(job, mols, _cache_opts(False, "u", "greedy", None))
    a3m.write_text(">q\nAAAA\n>hit\nCCCC\n")
    assert _input_digest(job, mols, _cache_opts(False, "u", "greedy", None)) != before
