from __future__ import annotations

import json

import numpy as np
import pytest

from foldjax.models.protenix.bridge.weights_io import save_native_weights
from foldjax.models.protenix.cli.static_infer import main
from foldjax.models.protenix.data.featurize_json import featurize_protein_json
from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.embedders import RelativePositionParams

from .test_model import _toy_features, _toy_params


def test_static_infer_help_exits_without_runtime_imports(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--features" in out
    assert "--input-json" in out
    assert "--weights" in out
    assert "--checkpoint" not in out
    assert "--chunk-policy" in out
    assert "--model-name" in out
    assert "--max-msa-rows" in out
    assert "--seeds" in out
    assert "--output-format" in out
    assert "--guidance-config" in out
    assert "--rna-msa-local-command" in out
    assert "--rna-msa-search-version" in out
    assert "--template-search-command" in out
    assert "--template-search-version" in out
    assert "--template-mmcif-dir" in out
    assert "--gamma0" in out
    assert "--eta" in out
    assert "--esm-checkpoint-dir" in out
    assert "--msa-search" in out
    assert "--msa-cache-dir" in out
    assert "--prewarm-only" in out


def test_static_infer_rejects_non_object_guidance_config(tmp_path) -> None:
    config = tmp_path / "guidance.json"
    config.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="guidance config must be a JSON object"):
        main(
            [
                "--features",
                str(tmp_path / "unused.npz"),
                "--model-name",
                "unknown",
                "--weights",
                str(tmp_path / "unused.pkl"),
                "--out",
                str(tmp_path / "out.npz"),
                "--guidance-config",
                str(config),
            ]
        )


def test_the_model_name_must_be_known_or_explicitly_waived(tmp_path) -> None:
    """A checkpoint whose name says nothing must not silently pick a schedule.

    The model name selects the sampler schedule (mini runs 5 steps at
    gamma0=0, base runs 200 at 0.8), decides whether ESM/ISM conditioning is
    built at all, and carries the protenix-v2 token limit. Inferring it from
    the filename meant renaming a checkpoint quietly changed all three: an ISM
    model run from a renamed file became a different model, with a
    plausible-looking structure and no warning.
    """
    weights_path = tmp_path / "my_own_checkpoint.pkl"
    features_path = tmp_path / "toy_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    save_static_feature_npz(features_path, _toy_features())
    argv = [
        "--weights", str(weights_path),
        "--features", str(features_path),
        "--out", str(out_path),
        "--n-sample", "1", "--n-step", "1", "--n-cycle", "1",
        "--input-atom-heads", "1", "--atom-encoder-heads", "1",
        "--token-heads", "1", "--atom-decoder-heads", "1",
        "--n-queries", "2", "--n-keys", "4",
        "--sigma-data", "4.0", "--cpu-only",
    ]

    with pytest.raises(SystemExit, match="cannot tell which Protenix model"):
        main(argv)
    assert not out_path.exists()

    # Waiving it is one flag, and then the same run proceeds normally.
    main([*argv, "--model-name", "unknown"])
    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 3, 3)


def test_static_infer_runs_with_native_weights(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "toy_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    save_static_feature_npz(features_path, _toy_features())

    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 3, 3)
        assert data["distogram_logits"].shape == (2, 2, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_prewarm_populates_graph_without_writing_output(
    tmp_path, monkeypatch, capsys
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "toy_features.npz"
    out_path = tmp_path / "must_not_exist.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    save_static_feature_npz(features_path, _toy_features())
    calls = []

    def fake_predict(_params, features, **_kwargs):
        calls.append(len(features["atom_to_token_idx"]))
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--seeds",
            "7",
            "11",
            "--trunk-dtype",
            "fp32",
            "--prewarm-only",
            "--no-compile-cache",
        ]
    )

    assert calls == [3]
    assert not out_path.exists()
    assert "prewarmed:" in capsys.readouterr().out


def test_static_infer_runs_from_sequence_json_features(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "json_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]},
        n_queries=2,
        n_keys=4,
    )
    features["relp"] = np.zeros((2, 2, 2), dtype=np.float32)
    save_static_feature_npz(features_path, features)

    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 10, 3)
        assert data["distogram_logits"].shape == (2, 2, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_runs_directly_from_sequence_json(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    out_path = tmp_path / "out.npz"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)

    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 10, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_processes_every_job_and_seed(tmp_path, monkeypatch) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    out_dir = tmp_path / "outputs"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "first",
                    "sequences": [{"proteinChain": {"sequence": "A", "count": 1}}],
                },
                {
                    "name": "second",
                    "sequences": [{"proteinChain": {"sequence": "G", "count": 1}}],
                },
            ]
        )
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)

    calls = []

    def fake_predict(_params, features, **kwargs):
        calls.append(kwargs["key"])
        n_atom = len(features["atom_to_token_idx"])
        return {
            "coordinate": np.zeros((1, n_atom, 3), dtype=np.float32),
            "atom_plddt": np.full((1, n_atom), 0.5, dtype=np.float32),
            "summary_plddt": np.array([50.0], dtype=np.float32),
            "summary_ranking_score": np.array([0.5], dtype=np.float32),
        }

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(out_dir),
            "--seeds",
            "3",
            "4",
            "--output-format",
            "both",
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--cpu-only",
            "--no-compile-cache",
        ]
    )

    assert len(calls) == 4
    for name in ("first", "second"):
        for seed in (3, 4):
            prediction_dir = out_dir / name / f"seed_{seed}" / "predictions"
            assert (prediction_dir / f"{name}_sample_0.cif").is_file()
            assert (prediction_dir / "raw_output.npz").is_file()


def test_static_infer_uses_model_seeds_from_direct_json(tmp_path, monkeypatch) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    out_dir = tmp_path / "outputs"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "seeded",
                    "modelSeeds": [11, 13],
                    "sequences": [
                        {"proteinChain": {"sequence": "A", "count": 1}}
                    ],
                }
            ]
        )
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    calls = []

    def fake_predict(_params, features, **kwargs):
        calls.append(np.asarray(kwargs["key"]))
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(out_dir),
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--no-compile-cache",
        ]
    )

    assert len(calls) == 2


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], (4, 5, 0.0, 1.0)),
        (
            ["--n-cycle", "2", "--n-step", "3", "--gamma0", "0.4", "--eta", "1.2"],
            (2, 3, 0.4, 1.2),
        ),
    ],
)
def test_static_infer_uses_model_sampler_defaults_unless_overridden(
    tmp_path, monkeypatch, extra_args, expected
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    captured = {}

    def fake_predict(_params, features, **kwargs):
        captured.update(kwargs)
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    main(
        [
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(tmp_path / "out.npz"),
            "--model-name",
            "protenix_mini_default_v0.5.0",
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--no-compile-cache",
            *extra_args,
        ]
    )

    assert (
        captured["recycling_steps"],
        captured["num_sampling_steps"],
        captured["gamma0"],
        captured["step_scale_eta"],
    ) == expected


@pytest.mark.parametrize(
    ("model_name", "provider_name"),
    [
        ("protenix_mini_esm_v0.5.0", "esm2-3b"),
        ("protenix_mini_ism_v0.5.0", "esm2-3b-ism"),
    ],
)
def test_static_infer_adds_esm_for_esm_model(
    tmp_path, monkeypatch, model_name, provider_name
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    providers = []
    captured = {}

    class FakeProvider:
        def __init__(self, model_name, *, checkpoint_dir):
            providers.append((model_name, checkpoint_dir))

        def __call__(self, sequence):
            return np.ones((len(sequence), 2560), dtype=np.float32)

    def fake_predict(_params, features, **_kwargs):
        captured.update(features)
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr("foldjax.models.protenix.data.esm.JaxEsmProvider", FakeProvider)
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    checkpoint_dir = tmp_path / "esm"
    main(
        [
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(tmp_path / "out.npz"),
            "--model-name",
            model_name,
            "--esm-checkpoint-dir",
            str(checkpoint_dir),
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--no-compile-cache",
        ]
    )

    assert providers == [(provider_name, checkpoint_dir)]
    assert captured["esm_token_embedding"].shape == (1, 2560)


def test_static_infer_pads_mini_esm_language_model_and_reports_profile(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models.protenix.cli.predict import main as predict_main

    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    language_targets: list[int] = []
    captured: dict[str, object] = {}
    profiles: list[tuple[object, object]] = []

    class FakeProvider:
        max_sequence_length = 4094

        def __init__(self, _model_name, *, checkpoint_dir):
            del checkpoint_dir

        def __call__(self, sequence):
            return np.ones((len(sequence), 2560), dtype=np.float32)

        def embed(self, sequence, *, target_length):
            language_targets.append(target_length)
            return np.ones((len(sequence), 2560), dtype=np.float32)

    def fake_predict(_params, features, **kwargs):
        captured["features"] = features
        captured["kwargs"] = kwargs
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    def collect_profile(plan, static):
        profiles.append((plan, static))

    monkeypatch.setattr("foldjax.models.protenix.data.esm.JaxEsmProvider", FakeProvider)
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    predict_main(
        [
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(tmp_path / "out.npz"),
            "--model-name",
            "protenix_mini_esm_v0.5.0",
            "--trunk-dtype",
            "fp32",
            "--n-sample",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--padding",
            "--pad-tokens",
            "8",
            "--pad-atoms",
            "8",
            "--pad-msa",
            "4",
            "--pad-templates",
            "4",
            "--pad-language-model-tokens",
            "16",
            "--no-compile-cache",
        ],
        on_padding_plan=collect_profile,
    )

    features = captured["features"]
    kwargs = captured["kwargs"]
    assert language_targets == [16]
    assert features["restype"].shape[0] == 8
    assert kwargs["padded_generated_schema"] is True
    assert kwargs["step_noises"].shape == (
        kwargs["num_sampling_steps"],
        1,
        8,
        3,
    )
    plan, static = profiles[0]
    assert plan.target["language_model_tokens"] == 16
    assert static == {"chains": 1}


def test_static_esm_features_bypass_language_model_checkpoint(
    tmp_path, monkeypatch
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "features.npz"
    features = dict(_toy_features())
    n_token = int(features["restype"].shape[-2])
    expected = np.arange(n_token * 2560, dtype=np.float32).reshape(n_token, 2560)
    features["esm_token_embedding"] = expected
    save_native_weights(
        weights_path, _toy_params_with_relp_dim(139), compress=False
    )
    save_static_feature_npz(features_path, features)
    captured = {}

    class UnexpectedProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("precomputed ESM features constructed a provider")

    def fake_predict(_params, model_features, **_kwargs):
        captured.update(model_features)
        n_atom = len(model_features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.data.esm.JaxEsmProvider", UnexpectedProvider
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    main(
        [
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(tmp_path / "out.npz"),
            "--model-name",
            "protenix_mini_esm_v0.5.0",
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--no-compile-cache",
        ]
    )

    np.testing.assert_array_equal(captured["esm_token_embedding"], expected)


@pytest.mark.parametrize("mode", ["local", "remote"])
def test_static_infer_applies_enabled_msa_search_before_featurization(
    tmp_path, monkeypatch, mode
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "A", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    calls = []

    def fake_local(command, *, version):
        calls.append(("local", command, version))
        return object()

    def fake_remote(url, *, version):
        calls.append(("remote", url, version))
        return object()

    def fake_pipeline(cache_dir, backend, *, options):
        calls.append(("pipeline", cache_dir, backend, options))
        return object()

    def fake_apply(job, pipeline):
        calls.append(("apply", job, pipeline))
        return job

    def fake_predict(_params, features, **_kwargs):
        n_atom = len(features["atom_to_token_idx"])
        return {"coordinate": np.zeros((1, n_atom, 3), dtype=np.float32)}

    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.LocalMsaClient", fake_local
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.RemoteMMseqs2Client", fake_remote
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.MsaSearchPipeline", fake_pipeline
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.apply_msa_paths", fake_apply
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static", fake_predict
    )
    search_args = (
        ["--msa-local-command", "fake-search"]
        if mode == "local"
        else ["--msa-remote-url", "https://msa.invalid"]
    )
    main(
        [
            "--model-name",
            "unknown",
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(tmp_path / "out.npz"),
            "--msa-search",
            mode,
            "--msa-search-version",
            "db-v1",
            "--msa-cache-dir",
            str(tmp_path / "msa-cache"),
            *search_args,
            "--trunk-dtype",
            "fp32",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--no-compile-cache",
        ]
    )

    assert calls[0][0] == mode
    assert any(call[0] == "apply" for call in calls)


def test_static_infer_orders_msa_template_and_rna_preprocessing(
    tmp_path, monkeypatch
) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps([{"sequences": [
        {"proteinChain": {"sequence": "AAAAA"}},
        {"rnaSequence": {"sequence": "AGCUA"}},
    ]}]))
    mmcif = tmp_path / "mmcif"
    mmcif.mkdir()
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)
    calls = []

    def factory(label):
        def make(*args, **kwargs):
            calls.append((label, args, kwargs))
            return label
        return make

    def apply(label):
        def run(job, pipeline):
            calls.append((label, pipeline))
            return job
        return run

    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.LocalMsaClient", factory("msa-client")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.MsaSearchPipeline", factory("msa-pipe")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.apply_msa_paths", apply("msa-apply")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.LocalTemplateSearchClient",
        factory("template-client"),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.TemplateSearchPipeline",
        factory("template-pipe"),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.apply_template_paths",
        apply("template-apply"),
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.LocalRnaMsaClient", factory("rna-client")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.RnaMsaSearchPipeline", factory("rna-pipe")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.data.search.apply_rna_msa_paths", apply("rna-apply")
    )
    monkeypatch.setattr(
        "foldjax.models.protenix.models.predict.protenix_predict_static",
        lambda _params, features, **_kwargs: {
            "coordinate": np.zeros((1, len(features["atom_to_token_idx"]), 3))
        },
    )
    main([
        "--model-name", "unknown",
        "--weights", str(weights_path), "--input-json", str(input_json),
        "--out", str(tmp_path / "out.npz"),
        "--msa-search", "local", "--msa-local-command", "protein-search",
        "--msa-search-version", "protein-v1",
        "--template-search-command", "template-search",
        "--template-search-version", "template-v1",
        "--template-mmcif-dir", str(mmcif),
        "--rna-msa-local-command", "rna-search",
        "--rna-msa-search-version", "rna-v1",
        "--trunk-dtype", "fp32", "--n-queries", "2", "--n-keys", "4",
        "--no-compile-cache",
    ])
    applies = [call[0] for call in calls if call[0].endswith("-apply")]
    assert applies == ["msa-apply", "template-apply", "rna-apply"]


def test_static_infer_checks_v2_size_before_loading_weights(tmp_path) -> None:
    """The size check runs first, and no longer stops the run by itself.

    It used to refuse above 2,560 tokens. That limit is upstream's memory
    budget rather than an architectural bound, and this port runs past it -- so
    what has to stay true is only that the check happens before the expensive
    weight load, whichever way it goes. `--strict-token-limit` still refuses,
    and refuses there.
    """
    features_path = tmp_path / "large_features.npz"
    np.savez_compressed(
        features_path,
        restype=np.zeros((2561, 32), dtype=np.float32),
    )
    argv = [
        "--weights",
        str(tmp_path / "missing.pkl"),
        "--features",
        str(features_path),
        "--out",
        str(tmp_path / "out.npz"),
        "--model-name",
        "protenix-v2",
        "--cpu-only",
    ]

    with pytest.raises(SystemExit) as exc_info:
        main([*argv, "--strict-token-limit"])
    assert exc_info.value.code != 0
    assert "does not support n_token > 2560" in str(exc_info.value)

    # Without it the size is a warning, and what stops the run is the weights
    # that are not there -- which is how you can tell the check ran and passed
    # rather than never running.
    with pytest.warns(RuntimeWarning, match="memory budget"):
        with pytest.raises((SystemExit, FileNotFoundError, OSError)) as fell_through:
            main(argv)
    assert "does not support n_token > 2560" not in str(fell_through.value)


def _toy_params_with_relp_dim(relp_dim: int):
    params = _toy_params()
    pairformer_output = params.pairformer_output
    trunk = pairformer_output.trunk
    initial = trunk.initial
    initial = initial._replace(
        relative_position=RelativePositionParams(
            linear_no_bias=LinearParams(
                weight=np.zeros((2, relp_dim), dtype=np.float32),
                bias=None,
            )
        )
    )
    trunk = trunk._replace(initial=initial)
    pairformer_output = pairformer_output._replace(trunk=trunk)
    diffusion = params.diffusion
    conditioning = diffusion.conditioning
    conditioning = conditioning._replace(
        relpe=RelativePositionParams(
            linear_no_bias=LinearParams(
                weight=np.zeros((2, relp_dim), dtype=np.float32),
                bias=None,
            )
        )
    )
    diffusion = diffusion._replace(conditioning=conditioning)
    return params._replace(pairformer_output=pairformer_output, diffusion=diffusion)


def test_the_triangle_backend_default_lives_in_one_place() -> None:
    """The documented default has to be the effective one.

    `_triangle_attention_backend()` resolves the backend, honours
    PROTENIX_TRIANGLE_BACKEND, and defaults to the blocked XLA path -- and a
    test asserts that. But "cueq_jit" was also written as the default in the
    CLI and in two module signatures, and those win: every real call passed
    cueq_jit, which does not block rows and so does not bound the score tensor.
    The tested default was the one nothing used.
    """
    import inspect

    from foldjax.models.protenix.cli.predict import main as predict_main
    from foldjax.models.protenix.models import model as model_impl
    from foldjax.models.protenix.models import predict as predict_impl

    entry_points = (
        model_impl.protenix_infer_static,
        predict_impl.protenix_predict_static,
    )
    for function in entry_points:
        default = inspect.signature(function).parameters[
            "trunk_triangle_attention_backend"
        ].default
        assert default is None, f"{function.__name__} pins a backend of its own"

    parser_defaults = _cli_defaults(predict_main)
    assert parser_defaults["trunk_triangle_attention_backend"] is None


def _cli_defaults(entry) -> dict:
    """Parse the CLI with only its required arguments to read the defaults."""
    import argparse
    from unittest.mock import patch

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        parsed = real_parse(self, args, namespace)
        captured.update(vars(parsed))
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(SystemExit):
            entry(["--features", "x.npz", "--weights", "w.pkl", "--out", "o.npz"])
    return captured
