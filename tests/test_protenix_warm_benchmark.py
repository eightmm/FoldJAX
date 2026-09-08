"""CPU preflight contracts for the optional warm-output capture."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parent / "models/protenix/scripts/benchmark_jax_production.py"
    spec = importlib.util.spec_from_file_location("protenix_warm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_prediction_directory_fails_before_model_loading(
    benchmark, monkeypatch, tmp_path
):
    destination = tmp_path / "prediction"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--features",
            "missing.npz",
            "--weights",
            "missing.jax",
            "--out",
            str(tmp_path / "metrics.json"),
            "--prediction-dir",
            str(destination),
        ],
    )
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("precision", [None, "default", "highest"])
def test_precision_policy_is_explicit(benchmark, monkeypatch, precision):
    argv = [
        "benchmark",
        "--features",
        "input.npz",
        "--weights",
        "weights.jax",
        "--out",
        "metrics.json",
        "--prediction-dir",
        "prediction",
    ]
    if precision is not None:
        argv += ["--matmul-precision", precision]
    monkeypatch.setattr("sys.argv", argv)
    args = benchmark.parse_args()
    assert args.matmul_precision == (precision or "high")
    assert args.prediction_dir == Path("prediction")
    assert args.msa_cycle_route == "index_tape"


def test_historical_materialized_route_remains_explicit(benchmark, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--features",
            "input.npz",
            "--weights",
            "weights.jax",
            "--out",
            "metrics.json",
            "--sample-msa-per-cycle",
            "--msa-cycle-route",
            "materialized",
        ],
    )
    args = benchmark.parse_args()
    assert args.msa_cycle_route == "materialized"
    assert args.full_depth_msa is False
