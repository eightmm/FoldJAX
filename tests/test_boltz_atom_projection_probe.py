import numpy as np

from bench.boltz_atom_projection_probe import atom_features, compile_profiles


def test_backend_controls_preserve_precision_and_default():
    assert compile_profiles() == {"baseline": {"xla_allow_excess_precision": False}}
    profiles = compile_profiles(True)
    assert len(profiles) == 4
    assert all(p["xla_allow_excess_precision"] is False for p in profiles.values())
    assert profiles["split_k_1"]["xla_gpu_experimental_force_split_k"] == 1
    assert profiles["no_triton_no_lt"]["xla_gpu_enable_cublaslt"] is False


def test_atom_features_preserve_order_and_native_float32():
    features = {
        "ref_pos": np.ones((1, 2, 3), np.float32),
        "ref_charge": np.full((1, 2), 2, np.int64),
        "ref_element": np.full((1, 2, 128), 3, np.int64),
        "ref_atom_name_chars": np.full((1, 2, 4, 64), 4, np.int64),
    }
    result = atom_features(features)
    assert result.dtype == np.float32
    assert result.shape == (1, 2, 388)
    np.testing.assert_array_equal(result[..., :3], 1)
    np.testing.assert_array_equal(result[..., 3], 2)
    np.testing.assert_array_equal(result[..., 4:132], 3)
    np.testing.assert_array_equal(result[..., 132:], 4)
