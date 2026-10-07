"""The feature-cache key has to move whenever the features would.

A stale hit here is not a slow run, it is a different prediction reported as
this one's.
"""

from pathlib import Path

from foldjax.models.boltz2.data.featurize import _cache_opts, _input_digest


def _opts(
    max_msa_depth: int | None = None, msa_deletions: str = "released", seed: int = 0
):
    return _cache_opts(False, "u", "greedy", max_msa_depth, msa_deletions, seed)


def test_the_msa_cap_is_part_of_the_cache_key() -> None:
    """It was not, so a capped run reused an uncapped run's features.

    `max_msa_depth` selects how many alignment rows reach the model. It is the
    dominant term in peak memory and it moves accuracy, and it was absent from
    the key -- so asking for a cap after an uncapped run returned the uncapped
    features, and asking for none after a capped run returned the capped ones.
    Neither said anything.
    """
    uncapped = _opts()
    capped = _opts(1024)
    assert uncapped != capped
    assert 1024 in capped


def test_a_raised_parse_cap_retires_the_old_deep_entries() -> None:
    """A depth above 8,192 used to parse only 8,192 rows under the same key.

    The parse cap now follows such a depth, so its key has to differ from the
    one those entries were written under -- the bare tuple of the same
    options -- while every key at or below the released cap stays as it was.
    """
    deep = _opts(12000)
    assert ("parse_cap", 12000) in deep
    assert deep[:-1] == (*_opts(1024)[:4], 12000, *_opts(1024)[5:])
    assert not any(isinstance(item, tuple) for item in _opts(8192))
    assert not any(isinstance(item, tuple) for item in _opts())


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
    before = _input_digest(job, mols, _opts())
    a3m.write_text(">q\nAAAA\n>hit\nCCCC\n")
    after = _input_digest(job, mols, _opts())

    assert before != after, "editing the alignment must invalidate the cache"


def test_an_absolute_alignment_path_still_works(tmp_path: Path) -> None:
    job = tmp_path / "job.yaml"
    a3m = tmp_path / "hits.a3m"
    a3m.write_text(">q\nAAAA\n")
    job.write_text(f"sequences:\n  - protein:\n      msa: {a3m}\n")
    mols = tmp_path / "mols"

    before = _input_digest(job, mols, _opts())
    a3m.write_text(">q\nAAAA\n>hit\nCCCC\n")
    assert _input_digest(job, mols, _opts()) != before


def test_the_deletion_mode_is_part_of_the_cache_key() -> None:
    """`msa_deletions` moves three of the seven MSA arrays and nothing else.

    Without it in the key a `restored` run gets a `released` run's zeroed
    deletion features straight back, which is the whole point of the option
    silently undone -- the same failure `max_msa_depth` had above.
    """
    released = _opts()
    restored = _opts(msa_deletions="restored")
    assert released != restored
    assert "restored" in restored


def test_the_seed_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """The seed draws the reference-conformer augmentation in `ref_pos`.

    Without it in the key, a seed-1 run given a feature cache gets seed 0's
    `ref_pos` back, and the run is no longer the one its seed names.
    """
    job = tmp_path / "job.yaml"
    job.write_text("sequences:\n  - protein:\n      sequence: ACDE\n")
    mols = tmp_path / "mols"

    assert _input_digest(job, mols, _opts(seed=0)) != _input_digest(
        job, mols, _opts(seed=1)
    )
    assert _input_digest(job, mols, _opts(seed=7)) == _input_digest(
        job, mols, _opts(seed=7)
    )


def test_the_key_folds_referenced_file_bytes_into_one_digest(tmp_path: Path) -> None:
    """The key is one running SHA-256, not a digest of per-file digests.

    Spelled out here so moving the streaming loop into a shared helper cannot
    change the key every existing feature cache was written under.
    """
    import hashlib

    job = tmp_path / "job.yaml"
    a3m = tmp_path / "hits.a3m"
    a3m.write_bytes(b">q\nAAAA\n" * 300_000)
    job.write_text("sequences:\n  - protein:\n      msa: hits.a3m\n")
    mols = tmp_path / "mols"
    opts = _opts()

    expected = hashlib.sha256()
    expected.update(job.read_bytes())
    expected.update(str(mols).encode())
    expected.update(repr(opts).encode())
    expected.update(a3m.read_bytes())
    assert _input_digest(job, mols, opts) == expected.hexdigest()[:16]
