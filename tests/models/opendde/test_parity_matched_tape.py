"""Cover the MSA row injection the OpenDDE matched-tape harness depends on.

The harness's whole claim rests on one thing: that the rows it hands the JAX
trunk really are the rows upstream's MSA module read. If `build_cycle_msa`
quietly returned the wrong rows -- or the untouched alignment -- the parity run
would still print a number, and the number would be wrong in the direction that
looks like success only if you never check it. So the selection is tested
directly, including the featurizer-agreement report that decides whether the
cheaper `indices` mode is equivalent at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.models.opendde.scripts.capture_upstream_tape import _translation_samples
from tests.models.opendde.scripts.parity_matched_tape import build_cycle_msa


@pytest.mark.parametrize("precision", ["high", "highest"])
def test_explicit_matmul_precision(monkeypatch, precision):
    from tests.models.opendde.scripts.parity_matched_tape import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--tape-dir",
            "tape",
            "--input-json",
            "input.json",
            "--matmul-precision",
            precision,
        ],
    )
    assert parse_args().matmul_precision == precision


def test_default_matmul_precision_remains_highest(monkeypatch):
    from tests.models.opendde.scripts.parity_matched_tape import parse_args

    monkeypatch.setattr(
        "sys.argv", ["replay", "--tape-dir", "tape", "--input-json", "input.json"]
    )
    assert parse_args().matmul_precision == "highest"


FIELDS = ("msa", "has_deletion", "deletion_value")


@pytest.mark.parametrize("dtype", ["float32", "bf16"])
def test_replay_forwards_compute_dtype_to_real_model_call(monkeypatch, tmp_path, dtype):
    import json
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from foldjax.models.opendde.bridge import weights_io
    from foldjax.models.opendde.data import featurize_json
    from foldjax.models.opendde.models import model
    from tests.models.opendde.scripts import parity_matched_tape as runner

    (tmp_path / "tape.json").write_text(json.dumps({
        "num_recycles": 10, "num_steps": 200, "num_samples": 5, "seed": 101,
    }))
    np.savez(tmp_path / "msa.npz", rows=np.zeros((10, 1), dtype=np.int32))
    np.savez(tmp_path / "coordinate.npz", coordinate=np.zeros((5, 2, 3)))
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(
        out_dir=tmp_path, tape_dir=tmp_path, input_json=tmp_path / "input.json",
        weights=tmp_path / "weights", matmul_precision="high", trunk_dtype=dtype,
        msa_source="indices", skip_sampler=False,
    ))
    monkeypatch.setattr(featurize_json, "load_jobs", lambda _: [{}])
    monkeypatch.setattr(featurize_json, "featurize_opendde_json", lambda *a, **k: {
        "ref_pos": np.zeros((2, 3)), "restype": np.zeros((1, 32)),
    })
    monkeypatch.setattr(weights_io, "load_native_weights", lambda _: object())
    monkeypatch.setattr(model, "cast_trunk_params", lambda p, d: p)
    monkeypatch.setattr(runner, "load_tape", lambda *a, **k: (None, {}))
    monkeypatch.setattr(runner, "build_cycle_msa", lambda *a, **k: ((), {}))
    monkeypatch.setattr(runner, "trunk_stages", lambda *a, **k: {
        "after_pairformer_z": jnp.zeros((1, 1, 1)),
    })

    class ModelReachedError(Exception):
        pass

    def infer(*args, **kwargs):
        assert "trunk_dtype" in kwargs
        assert kwargs["trunk_dtype"] == (jnp.bfloat16 if dtype == "bf16" else None)
        assert kwargs["num_samples"] == 5
        assert kwargs["num_recycles"] == 10
        raise ModelReachedError

    monkeypatch.setattr(model, "opendde_infer_static", infer)
    previous_precision = jax.config.jax_default_matmul_precision
    try:
        with pytest.raises(ModelReachedError):
            runner.main()
    finally:
        jax.config.update("jax_default_matmul_precision", previous_precision)


def test_later_nonfinite_sample_cannot_hide_behind_scalar_max():
    from tests.models.opendde.scripts.parity_matched_tape import _verdict

    values = [0.0, float("nan"), 0.0, 0.0, 0.0]
    metrics = {
        "stages": {},
        "stages_deliberately_absent": True,
        "all_atom_rmsd_angstrom": max(values),
        "all_atom_rmsd_angstrom_per_sample": values,
    }
    assert _verdict(metrics, max_rmsd=0.5, min_correlation=0.999)


@pytest.mark.parametrize(
    "shape,samples",
    [((1, 1, 3), 1), ((5, 1, 3), 5), ((1, 5, 1, 3), 5)],
)
def test_capture_normalizes_translation_sample_axis(
    shape: tuple[int, ...], samples: int
) -> None:
    value = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)

    normalized = _translation_samples(value)

    assert normalized.shape == (samples, 3)
    assert normalized.dtype == np.float32


def test_capture_rejects_translation_without_singleton_atom_axis() -> None:
    with pytest.raises(RuntimeError, match="translation must have shape"):
        _translation_samples(np.zeros((5, 2, 3), dtype=np.float32))


def _features(n_row: int = 6, n_token: int = 4) -> dict[str, np.ndarray]:
    msa = np.arange(n_row * n_token, dtype=np.int32).reshape(n_row, n_token) % 31
    return {
        "msa": msa,
        "has_deletion": msa.astype(np.float32) + 100.0,
        "deletion_value": msa.astype(np.float32) + 200.0,
    }


def _archive(features: dict[str, np.ndarray], rows: np.ndarray) -> dict:
    archive = {"rows": rows}
    for name in FIELDS:
        archive[f"input_{name}"] = features[name]
        archive[f"selected_{name}"] = np.stack([features[name][row] for row in rows])
    return archive


def test_upstream_source_hands_back_the_rows_upstream_selected() -> None:
    features = _features()
    rows = np.asarray([[4, 1, 0], [2, 5, 3]], dtype=np.int64)
    archive = _archive(features, rows)

    cycles, agreement = build_cycle_msa(
        features, archive, source="upstream", num_recycles=2
    )

    assert cycles is not None and len(cycles) == 2
    for cycle_index, cycle in enumerate(cycles):
        for name in FIELDS:
            expected = features[name][rows[cycle_index]]
            np.testing.assert_array_equal(np.asarray(cycle[name]), expected)
        # Upstream uses its mask only to prioritize rows; once selected every
        # row enters the module unmasked, so a mask that is anything but ones
        # would silently reweight the alignment.
        assert np.all(np.asarray(cycle["msa_mask"]) == 1.0)
    assert agreement["upstream_sampled_rows"] == 3
    assert agreement["featurizer_msa_rows_identical"] is True


def test_indices_source_matches_upstream_source_when_featurizers_agree() -> None:
    features = _features()
    rows = np.asarray([[5, 0, 2]], dtype=np.int64)
    archive = _archive(features, rows)

    by_rows, _ = build_cycle_msa(features, archive, source="indices", num_recycles=1)
    by_arrays, _ = build_cycle_msa(features, archive, source="upstream", num_recycles=1)

    assert by_rows is not None and by_arrays is not None
    for name in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(by_rows[0][name]), np.asarray(by_arrays[0][name])
        )


def test_featurizer_disagreement_is_reported_rather_than_assumed() -> None:
    """`indices` is only equivalent if both featurizers emit the same rows.

    That equivalence is an assumption until it is measured, and a harness that
    assumed it would attribute a featurizer difference to the model.
    """
    features = _features()
    rows = np.asarray([[0, 1, 2]], dtype=np.int64)
    archive = _archive(features, rows)
    shifted = dict(features)
    shifted["msa"] = features["msa"][::-1].copy()

    _, agreement = build_cycle_msa(shifted, archive, source="upstream", num_recycles=1)

    assert agreement["featurizer_msa_rows_identical"] is False
    assert agreement["featurizer_msa_row_match_fraction"] < 1.0


def test_whole_source_leaves_the_alignment_untouched() -> None:
    features = _features()
    archive = _archive(features, np.asarray([[0, 1]], dtype=np.int64))

    cycles, agreement = build_cycle_msa(
        features, archive, source="whole", num_recycles=1
    )

    assert cycles is None
    assert agreement["jax_msa_rows"] == 6


def test_a_capture_with_the_wrong_cycle_count_is_rejected() -> None:
    features = _features()
    archive = _archive(features, np.asarray([[0, 1]], dtype=np.int64))

    with pytest.raises(ValueError, match="MSA row draws"):
        build_cycle_msa(features, archive, source="upstream", num_recycles=2)
