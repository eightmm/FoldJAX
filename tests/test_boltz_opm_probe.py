import json

import pytest

from bench.boltz_opm_probe import execution_profiles, load_inputs, main


def test_opm_requires_completed_native_msa(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps({"arm": "native", "passed": False})
    )
    with pytest.raises(ValueError, match="reproduction"):
        load_inputs(tmp_path)


@pytest.mark.parametrize("arm", ["native", "foldjax"])
def test_opm_cli_requires_the_selected_native_reference(monkeypatch, tmp_path, arm):
    monkeypatch.setattr(
        "sys.argv",
        ["probe", arm, "--reference", str(tmp_path), "--out", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_backend_variants_are_opt_in_and_keep_highest_precision():
    assert execution_profiles() == [
        ("highest", "highest", {}),
        ("default", "default", {}),
    ]
    profiles = execution_profiles(True)
    assert len(profiles) == 8
    assert all(precision == "highest" for _, precision, _ in profiles[2:])
    assert profiles[-1][2] == {
        "xla_gpu_enable_triton_gemm": False,
        "xla_gpu_enable_cublaslt": False,
        "xla_gpu_autotune_level": 0,
    }
