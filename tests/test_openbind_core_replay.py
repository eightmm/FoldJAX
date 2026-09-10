import numpy as np
import pytest

from bench.openbind_core_replay import (
    candidate_trunk_arrays,
    compiler_environment,
    model_feature_batch,
    native_trunk_arrays,
    positive_count,
    replace_trunk_components,
)
from foldjax.models.openfold3.data.featurize import MODEL_FEATURES


def test_metadata_is_not_forwarded_and_required_features_are_not_dropped():
    features = {name: np.zeros(1) for name in MODEL_FEATURES}
    features["atom_array.0.annotation.atom_name"] = np.array(["CA"])
    batch = model_feature_batch(features)
    assert set(batch) == set(MODEL_FEATURES)
    assert all(batch[name] is features[name] for name in MODEL_FEATURES)
    assert "atom_array.0.annotation.atom_name" in features
    del features["token_mask"]
    with pytest.raises(ValueError, match="token_mask"):
        model_feature_batch(features)


def test_repeat_count_must_be_positive():
    import argparse

    assert positive_count("3") == 3
    for value in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive"):
            positive_count(value)


@pytest.mark.parametrize("extra", [
    ["--backend", "cueq"],
    ["--backend", "native-private"],
    ["--backend", "xla", "--inject-native-trunk"],
    ["--backend", "xla", "--inject-candidate-trunk", "unused"],
])
def test_private_control_rejects_other_backend_or_trunk_injection(extra, capsys):
    from bench.openbind_core_replay import main

    args = ["--capture", "unused", "--checkpoint", "unused",
            "--source-root", "unused", "--out-dir", "unused",
            "--private-pair-operators", *extra]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "requires XLA and no trunk injection" in capsys.readouterr().err


def test_compiler_environment_is_allowlisted_and_preserves_absence(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    monkeypatch.setenv("PRIVATE_TEST_TOKEN", "must-not-be-recorded")
    recorded = compiler_environment()
    assert recorded["XLA_FLAGS"] == "--xla_gpu_autotune_level=0"
    assert recorded["JAX_DEFAULT_MATMUL_PRECISION"] is None
    assert "PRIVATE_TEST_TOKEN" not in recorded


def test_native_trunk_injection_requires_bound_valid_arrays(tmp_path):
    import json

    from bench.boltz_historical_replay import digest

    path = tmp_path / "trunk-00.npz"
    arrays = [np.zeros(s, np.float32) for s in
              ((1, 2, 449), (1, 2, 384), (1, 2, 2, 128))]
    np.savez(path, **{str(i): a for i, a in enumerate(arrays)})
    (tmp_path / "trace.json").write_text(json.dumps({"trunk_arrays": [{
        "file": path.name,
        "sha256": {"sha256": digest(path), "bytes": path.stat().st_size},
    }]}))
    loaded, _ = native_trunk_arrays(tmp_path, 2)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, loaded, strict=True))
    with pytest.raises(ValueError, match="shape"):
        native_trunk_arrays(tmp_path, 3)
    np.savez(path, **{"0": np.zeros(1)})
    with pytest.raises(ValueError, match="identity"):
        native_trunk_arrays(tmp_path, 2)


def test_candidate_trunk_requires_matching_noninjected_provenance(tmp_path):
    import json

    from bench.boltz_historical_replay import digest

    expected = {"input_sha256": "input", "tape_sha256": "tape"}
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(expected))
    path = tmp_path / "prediction.npz"
    np.savez(path, single_inputs=np.zeros((1, 2, 449), np.float32),
             single=np.zeros((1, 2, 384), np.float32),
             pair=np.zeros((1, 2, 2, 128), np.float32))
    (tmp_path / "finished.json").write_text(json.dumps({
        "prediction_sha256": digest(path),
    }))
    arrays, identity = candidate_trunk_arrays(tmp_path, expected, 2)
    assert len(arrays) == 3 and identity["prediction_sha256"] == digest(path)
    with pytest.raises(ValueError, match="provenance"):
        candidate_trunk_arrays(tmp_path, {"input_sha256": "wrong"}, 2)
    preflight.write_text(json.dumps({**expected, "native_trunk_injection": {"x": 1}}))
    with pytest.raises(ValueError, match="non-injected"):
        candidate_trunk_arrays(tmp_path, expected, 2)
    preflight.write_text(json.dumps(expected))
    np.savez(path, single=np.zeros(1))
    with pytest.raises(ValueError, match="identity"):
        candidate_trunk_arrays(tmp_path, expected, 2)


def test_component_replacement_preserves_each_selected_array():
    candidate = tuple(object() for _ in range(3))
    native = tuple(object() for _ in range(3))
    assert replace_trunk_components(candidate, native, "single") == (
        native[0], native[1], candidate[2]
    )
    assert replace_trunk_components(candidate, native, "pair") == (
        candidate[0], candidate[1], native[2]
    )
    with pytest.raises(ValueError, match="unknown"):
        replace_trunk_components(candidate, native, "other")


def _replay_capture(*, tokens=4, atoms=4, rows=17, templates=4):
    """A template-free query as the released featurizer stores one.

    Four copies of one empty template on a fixed-width axis, exact positive-zero
    geometry, and the dense int32 MSA one-hot the portable archive keeps.
    """
    from tests.models.openfold3.feature_fixture import minimal_features

    features = minimal_features(
        tokens=tokens, atoms=atoms, msa_rows=rows, templates=templates
    )
    for name in ("template_pseudo_beta_mask", "template_backbone_frame_mask"):
        features[name] = np.zeros_like(features[name])
    return features


def _prepared(features, *, depth, cycles, seed=5):
    from foldjax.models.openfold3.data import prepare_msa_cycle_features

    return prepare_msa_cycle_features(
        features, depth, num_recycles=cycles, rng=np.random.default_rng(seed)
    )


def test_streamed_host_features_stage_the_managed_backend_representation():
    from bench.openbind_core_replay import streamed_host_features
    from foldjax.models.openfold3.data.featurize import (
        _COMPACT_MSA_INDICES,
        _COMPACT_MSA_MARKER,
        _ZERO_TEMPLATE_PAIR_FEATURES,
        _ZERO_TEMPLATE_PAIR_MARKER,
    )
    from foldjax.models.openfold3.inference import (
        _prepare_ref_atom_category_graph_input,
        _restore_ref_atom_category_one_hot,
    )

    dense = _prepared(_replay_capture(), depth=2, cycles=2)
    staged = streamed_host_features(dense)

    # One template row, and no quadratic template array at all.
    assert staged["template_restype"].shape[1] == 1
    assert dense["template_restype"].shape[1] == 4
    assert not [name for name in _ZERO_TEMPLATE_PAIR_FEATURES if name in staged]
    assert _ZERO_TEMPLATE_PAIR_MARKER in staged
    # Categories, not one-hots, across the private model boundary.
    assert "msa" not in staged and _COMPACT_MSA_MARKER in staged
    assert staged[_COMPACT_MSA_INDICES].dtype == np.uint8
    assert "ref_element" not in staged and "ref_atom_name_chars" not in staged
    # The graph boundary accepts the composed dict and rebuilds both one-hots.
    restored = _restore_ref_atom_category_one_hot(
        _prepare_ref_atom_category_graph_input(staged)
    )
    for name in ("ref_element", "ref_atom_name_chars"):
        np.testing.assert_array_equal(np.asarray(restored[name]), dense[name])


def test_streamed_device_features_never_carry_the_source_row_count():
    from bench.openbind_core_replay import streamed_host_features
    from foldjax.models.openfold3 import streaming

    rows, depth, cycles = 17, 2, 2
    staged = streamed_host_features(
        _prepared(_replay_capture(rows=rows), depth=depth, cycles=cycles)
    )
    provider = streaming.HostMSACycles(staged, depth=depth, cycles=cycles)
    for cycle in range(cycles):
        device = {**provider.common, **provider.select(cycle)}
        assert not any(rows in v.shape for v in device.values() if hasattr(v, "shape"))
        assert all(
            device[name].shape[:2] == (1, depth)
            for name in ("msa_mask", "has_deletion", "deletion_value")
        )
    # The host union is the only place the wider storage survives.
    assert staged["msa_mask"].shape[1] <= depth * cycles < rows


def test_streamed_budget_drops_pair_logits_and_the_fused_default_keeps_them():
    from bench.openbind_core_replay import array_budget_bytes
    from foldjax.models.openfold3.inference import released_config

    def logits(**flags):
        return released_config(
            n_token=3012, n_atom=23764,
            max_array_bytes=array_budget_bytes(**flags),
        ).returned_pair_logits

    assert logits(streamed=True, all_arrays=False) == ()
    expected = ("pae_logits", "pde_logits", "distogram_logits")
    assert logits(streamed=True, all_arrays=True) == expected
    assert logits(streamed=False, all_arrays=False) == expected


def test_streamed_preparation_preserves_the_template_and_msa_embeddings(monkeypatch):
    """The compaction is a storage change; the values the model reads are equal.

    Each step has its own gate (``test_template_collapse``,
    ``test_compact_msa_storage``, ``test_compact_atom_categories``). What is new
    here is the composition, applied to the replay's own prepared features. The
    template tolerance is the one ``test_template_collapse`` established for a
    collapsed axis: absolute, one float32 epsilon, no rtol.
    """
    import jax
    import jax.numpy as jnp

    from bench.openbind_core_replay import streamed_host_features
    from foldjax.models.openfold3 import streaming
    from foldjax.models.openfold3.models.input_embedders import msa_embedder
    from foldjax.models.openfold3.models.template_module import template_embedder
    from tests.models.openfold3.test_compact_msa_storage import _params
    from tests.models.openfold3.test_template_collapse import _embedder_params

    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    tokens, channels, heads, depth, cycles = 4, 8, 2, 2, 2
    dense = _prepared(
        _replay_capture(tokens=tokens, atoms=tokens), depth=depth, cycles=cycles
    )
    staged = streamed_host_features(dense)
    params = _embedder_params(channels, heads)
    z = jax.random.normal(jax.random.key(3), (1, tokens, tokens, channels))
    pair_mask = jnp.ones((1, tokens, tokens), dtype=jnp.float32)

    def embed(batch):
        return template_embedder(
            {name: jnp.asarray(value) for name, value in batch.items()},
            z,
            params,
            pair_mask=pair_mask,
            no_heads=heads,
        )

    expected = embed(dense)
    # Zero parameters or a dead branch would satisfy any comparison below.
    assert np.abs(np.asarray(expected)).max() > 1e-3
    np.testing.assert_allclose(
        expected, embed(staged), rtol=0, atol=np.finfo(np.float32).eps
    )

    single = jnp.ones((1, tokens, 7))
    msa_params = _params(jnp.float32)
    for cycle in range(cycles):
        rows = [
            streaming.HostMSACycles(batch, depth=depth, cycles=cycles).select(cycle)
            for batch in (dense, staged)
        ]
        left, right = (msa_embedder(row, single, msa_params) for row in rows)
        assert np.abs(np.asarray(left[0])).max() > 0.0
        for one, other in zip(left, right, strict=True):
            np.testing.assert_array_equal(np.asarray(one), np.asarray(other))
