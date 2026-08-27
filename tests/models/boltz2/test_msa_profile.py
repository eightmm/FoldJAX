from __future__ import annotations

import numpy as np

from foldjax.models.boltz2.data._torch import from_numpy, torch
from foldjax.models.boltz2.data.feature import featurizerv2


def test_msa_profile_bounds_one_hot_chunks_without_changing_bits(monkeypatch) -> None:
    rng = np.random.default_rng(91)
    values = rng.integers(
        0, featurizerv2.const.num_tokens, size=(43, 19), dtype=np.int64
    )
    msa = from_numpy(values)
    expected = (
        torch.nn.functional.one_hot(
            msa, num_classes=featurizerv2.const.num_tokens
        )
        .float()
        .mean(dim=0)
        .numpy()
    )

    budget = 19 * featurizerv2.const.num_tokens * 5
    monkeypatch.setattr(featurizerv2, "_MSA_PROFILE_TEMP_BUDGET", budget)
    real_sum = np.sum
    temporary_sizes: list[int] = []

    def tracked_sum(value, *args, **kwargs):
        temporary_sizes.append(value.nbytes)
        return real_sum(value, *args, **kwargs)

    monkeypatch.setattr(featurizerv2.np, "sum", tracked_sum)
    actual = featurizerv2._msa_profile(msa).numpy()

    assert np.array_equal(actual, expected)
    assert len(temporary_sizes) > 1
    assert max(temporary_sizes) <= budget
