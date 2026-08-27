from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from foldjax.models.boltz2.data.feature import featurizerv2


def test_distogram_training_label_matches_the_previous_expression() -> None:
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0], [0.0, 8.0, 0.0]],
        ],
        dtype=np.float32,
    )
    indices = [0, 1]
    expected = featurizerv2.torch.zeros(3, 3, 2, 5)
    boundaries = featurizerv2.torch.linspace(0.0, 8.0, 4)
    for index, ensemble_index in enumerate(indices):
        centers = featurizerv2.torch.Tensor(coordinates[:, ensemble_index, :])
        distances = featurizerv2.torch.cdist(centers, centers)
        bins = (distances.unsqueeze(-1) > boundaries).sum(dim=-1).long()
        expected[:, :, index, :] = featurizerv2.one_hot(bins, num_classes=5)

    actual = featurizerv2._distogram_target(
        coordinates, indices, min_dist=0.0, max_dist=8.0, num_bins=5
    )

    assert np.array_equal(actual.numpy(), expected.numpy())


def test_inference_routes_around_the_training_distogram(monkeypatch) -> None:
    assert (
        inspect.signature(featurizerv2.process_atom_features)
        .parameters["compute_disto_target"]
        .default
        is True
    )
    seen: list[bool] = []

    monkeypatch.setattr(featurizerv2, "process_ensemble_features", lambda **kw: {})
    monkeypatch.setattr(featurizerv2, "process_token_features", lambda **kw: {})
    monkeypatch.setattr(
        featurizerv2,
        "process_atom_features",
        lambda **kwargs: seen.append(kwargs["compute_disto_target"]) or {},
    )
    monkeypatch.setattr(featurizerv2, "process_msa_features", lambda **kw: {})
    monkeypatch.setattr(
        featurizerv2, "load_dummy_templates_features", lambda **kw: {}
    )
    data = SimpleNamespace(tokens=np.zeros(1), templates=())
    featurizer = featurizerv2.Boltz2Featurizer()

    for training in (False, True):
        featurizer.process(
            data,
            random=np.random.default_rng(0),
            molecules={},
            training=training,
            max_seqs=1,
        )

    assert seen == [False, True]
