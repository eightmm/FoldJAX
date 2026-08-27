"""The arena preflight must be silent on jobs that run and loud on ones that do not.

The sizes here are the two this was bracketed against by measurement: 1,003
residues (1,902 structural tokens) runs in both dtypes, and 1,531 residues
(2,978) runs in bfloat16 and OOMs in float32 on a 95.6 GiB card.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.opendde.cli import predict as predict_cli

POOL_BYTES = int(0.9 * 95.6 * 2**30)  # what `foldjax predict` asks for on this card
N_ST_1003, N_ST_1531 = 1902, 2978


def _features(n_structural: int) -> dict[str, np.ndarray]:
    return {"structural_token_index": np.arange(n_structural, dtype=np.int64)}


@pytest.fixture
def pool(monkeypatch):
    """Pin the device budget so the test does not need a GPU."""
    from foldjax import oom

    monkeypatch.setattr(oom, "device_budget", lambda: (POOL_BYTES, int(95.6 * 2**30)))
    return POOL_BYTES


BF16 = np.dtype("bfloat16") if hasattr(np, "bfloat16") else object()


@pytest.mark.parametrize(
    "n_structural,trunk_dtype",
    [
        (N_ST_1003, None),  # 1,003 residues, float32 -- measured 39,097 MiB, runs
        (N_ST_1003, BF16),  # 1,003 residues, bfloat16 -- measured 19,797 MiB, runs
        (N_ST_1531, BF16),  # 1,531 residues, bfloat16 -- measured 48,440 MiB, runs
    ],
)
def test_silent_when_the_job_fits(pool, n_structural, trunk_dtype):
    assert predict_cli._preflight_arena(_features(n_structural), trunk_dtype) is None


def test_fires_on_the_float32_wall(pool):
    """1,531 residues in float32 is the measured OOM: ~93.6 GiB against a ~77 GiB
    budget."""
    message = predict_cli._preflight_arena(_features(N_ST_1531), None)
    assert message is not None
    assert "probably not fit" in message
    assert "--trunk-dtype bf16" in message
    # The lever must carry its own caveat, not just its name.
    assert "upstream" in message
    assert str(N_ST_1531) in message


def test_does_not_offer_a_lever_already_spent(pool):
    """Above the bfloat16 ceiling the dtype advice would be useless -- so it is not
    given."""
    message = predict_cli._preflight_arena(_features(6000), BF16)
    assert message is not None
    assert "--trunk-dtype bf16" not in message
    assert "already bfloat16" in message


def test_silent_without_a_pool(monkeypatch):
    """Preallocation off means there is no pool; inventing a ceiling is worse than
    silence."""
    from foldjax import oom

    monkeypatch.setattr(oom, "device_budget", lambda: (None, None))
    assert predict_cli._preflight_arena(_features(N_ST_1531), None) is None


def test_silent_without_a_structural_token_count(pool):
    assert predict_cli._preflight_arena({}, None) is None


def test_estimate_matches_the_measured_arenas(pool):
    """The coefficients must still reproduce the measurements they were fitted to."""
    for n_st, dtype, measured in (
        (N_ST_1003, BF16, 19797.0),
        (N_ST_1531, BF16, 48440.0),
        (N_ST_1003, None, 39097.0),
    ):
        name = "bfloat16" if dtype is not None else "float32"
        estimate = predict_cli._ARENA_MIB_PER_PAIR[name] * n_st * n_st
        assert estimate == pytest.approx(measured, rel=0.01), (n_st, name)


def test_the_warning_actually_comes_out_of_predict(pool, monkeypatch) -> None:
    """The estimator being right is not the same as the call site firing.

    Everything above tests `_preflight_arena` directly. This one goes through
    `_predict`, which is where a user meets it, and stops at the first seam
    after the call so no device work happens. It pins three things the direct
    tests cannot: that `_predict` calls the preflight at all, that it passes
    the job's own features and the dtype `cast_trunk_params` was applied with
    rather than a flag, and that it warns rather than raising.
    """

    class _StopError(Exception):
        pass

    def _stop(**_kwargs):
        raise _StopError

    # Imported inside `_predict`, so the module it comes from is the seam.
    from foldjax.models.protenix import chunking

    monkeypatch.setattr(chunking, "resolve_chunk_config", _stop)

    features = _features(N_ST_1531)
    features["restype"] = np.zeros((1, 1531), dtype=np.int64)

    with pytest.warns(RuntimeWarning, match="probably not fit") as caught:
        with pytest.raises(_StopError):
            predict_cli._predict(
                features,
                None,
                seed=0,
                num_samples=1,
                num_steps=1,
                num_recycles=1,
                n_queries=32,
                n_keys=128,
                diffusion_attention_backend="xla",
                trunk_single_attention_backend="xla",
                structural_single_attention_backend="xla",
                trunk_dtype=None,
            )

    message = str(caught[0].message)
    assert "--trunk-dtype bf16" in message
    assert "not a guarantee" in message


def test_predict_is_silent_for_a_job_that_fits(pool, monkeypatch) -> None:
    """The same path, at a size that runs, must say nothing at all."""

    class _StopError(Exception):
        pass

    def _stop(**_kwargs):
        raise _StopError

    # Imported inside `_predict`, so the module it comes from is the seam.
    from foldjax.models.protenix import chunking

    monkeypatch.setattr(chunking, "resolve_chunk_config", _stop)

    features = _features(N_ST_1003)
    features["restype"] = np.zeros((1, 1003), dtype=np.int64)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(_StopError):
            predict_cli._predict(
                features,
                None,
                seed=0,
                num_samples=1,
                num_steps=1,
                num_recycles=1,
                n_queries=32,
                n_keys=128,
                diffusion_attention_backend="xla",
                trunk_single_attention_backend="xla",
                structural_single_attention_backend="xla",
                trunk_dtype=None,
            )
