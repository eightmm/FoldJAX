from __future__ import annotations

import builtins

import numpy as np
import pytest

from foldjax.models.protenix.bridge.weights_io import save_native_weights
from foldjax.models.protenix.cli.predict import main
from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.embedders import RelativePositionParams

from .test_model import _toy_features, _toy_params


def test_predict_help_is_native_weight_only(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--weights" in out
    assert "--checkpoint" not in out


def test_predict_native_runtime_imports_no_torch_or_protenix(tmp_path, monkeypatch):
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "toy_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    save_static_feature_npz(features_path, _toy_features())

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"standalone runtime imported {name}")
        if name == "protenix" or name.startswith("protenix."):
            raise AssertionError(f"standalone runtime imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    main(
        [
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
        assert data["summary_plddt"].shape == (1,)


def test_predict_direct_json_to_output_imports_no_torch_or_protenix(
    tmp_path,
    monkeypatch,
):
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    out_path = tmp_path / "out.npz"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"standalone runtime imported {name}")
        if name == "protenix" or name.startswith("protenix."):
            raise AssertionError(f"standalone runtime imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    main(
        [
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
