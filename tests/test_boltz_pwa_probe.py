import json

import numpy as np
import pytest

from bench.boltz_pwa_probe import load_reference, main


@pytest.mark.parametrize("layer", [-1, 4, True])
def test_pwa_probe_rejects_invalid_layer(tmp_path, layer):
    with pytest.raises(ValueError, match="layer"):
        load_reference(tmp_path, 1, layer)


def test_later_layer_preserves_fp32_and_selects_previous_pair(tmp_path):
    from bench.af3_closure_capture import sha

    def archive(name, values):
        path = tmp_path / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        return sha(path)

    m = np.full((1, 2, 1, 64), 1.0001, np.float32)
    z = np.full((1, 1, 1, 128), 7, np.float32)
    stages = {}
    for name, values in {
        "layers/02/input_m": {"": m},
        "layers/02/pwa": {"": m + 1},
        "layers/01/layer_output": {"z": z},
    }.items():
        digest = archive(name, values)
        tree = tmp_path / f"{name}.tree.json"
        tree.write_text(json.dumps({"": {"native_dtype": "torch.float32"}}))
        stages[name] = {"arrays_sha256": digest, "tree_sha256": sha(tree)}
    artifacts = {
        "operands.npz": archive("operands", {
            "input_z": z * 0, "token_pad_mask": np.ones((1, 1), np.float32)
        }),
        "native-weights.npz": archive("native-weights", {
            f"msa_module.layers.{layer}.pair_weighted_averaging.leaf{i}":
            np.array(layer, np.float32)
            for layer in (0, 2) for i in range(8)
        }),
    }
    (tmp_path / "report.json").write_text(json.dumps({
        "arm": "native", "passed": True, "artifacts": artifacts, "stages": stages
    }))
    report, inputs, weights, expected = load_reference(tmp_path, 1, 2)
    assert report["selected_m_dtype"] == "torch.float32"
    np.testing.assert_array_equal(inputs["m"], m[:, :1])
    np.testing.assert_array_equal(inputs["z"], z)
    np.testing.assert_array_equal(expected, (m + 1)[:, :1])
    assert all(float(value) == 2 for value in weights.values())
    with (tmp_path / "layers/01/layer_output.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError):
        load_reference(tmp_path, 1, 2)


@pytest.mark.parametrize("state", [{}, {"arm": "native", "passed": False}])
def test_pwa_probe_requires_native_reproduction(tmp_path, state):
    (tmp_path / "report.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="reproduction"):
        load_reference(tmp_path, 32)


def test_native_cli_requires_explicit_upstream(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "native",
            "--reference",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
