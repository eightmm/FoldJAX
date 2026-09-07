import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_conditioning_operators import (
    assert_source_origin,
    main,
    selected_operators,
    welford4_layer_norm,
)


@pytest.mark.parametrize("width", [16, 128, 256])
def test_welford_diagnostic_has_layer_norm_semantics(width):
    rng = np.random.default_rng(10)
    x = rng.normal(size=(2, 3, width)).astype(np.float32)
    scale = rng.normal(size=width).astype(np.float32)
    bias = rng.normal(size=width).astype(np.float32)
    actual = jax.jit(welford4_layer_norm)(
        jnp.asarray(x), jnp.asarray(scale), jnp.asarray(bias), 1e-5
    )
    reference = x.astype(np.float64) - x.astype(np.float64).mean(-1, keepdims=True)
    reference /= np.sqrt(x.astype(np.float64).var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(actual, reference * scale + bias, rtol=2e-5, atol=1e-6)


def test_conditioning_operator_scope_covers_both_transitions_and_bias_families():
    names = selected_operators()
    assert len(names) == len(set(names)) == 18
    for index in (0, 1):
        for name in ("norm", "fc1", "fc2", "silu", "fc3"):
            assert f"pairwise_conditioner.transitions.{index}.{name}" in names
    for group in ("atom_enc_proj_z", "atom_dec_proj_z", "token_trans_proj_z"):
        assert f"{group}.0.0" in names
        assert f"{group}.0.1" in names


@pytest.mark.parametrize("arm", ["native", "foldjax"])
def test_existing_operator_capture_is_refused_before_framework_import(tmp_path, arm):
    with pytest.raises(FileExistsError):
        main([arm, "--reference", "missing", "--out", str(tmp_path)])


@pytest.mark.parametrize("arm", ["native", "foldjax"])
def test_operator_cli_requires_bound_arm_inputs(tmp_path, arm):
    with pytest.raises(SystemExit) as exc:
        main([arm, "--reference", "missing", "--out", str(tmp_path / "new")])
    assert exc.value.code == 2


def test_operator_imports_must_come_from_selected_source(tmp_path):
    with pytest.raises(ValueError, match="outside selected tree"):
        assert_source_origin(tmp_path, (assert_source_origin,))


def test_explicit_fma_requires_bound_native_statistics(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "foldjax",
                "--reference",
                "missing",
                "--out",
                str(tmp_path / "new"),
                "--weights",
                "missing",
                "--explicit-fma",
            ]
        )
    assert exc.value.code == 2
