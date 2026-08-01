import subprocess
import sys
from pathlib import Path


def test_predict_cli_exposes_affinity_and_steering_controls() -> None:
    script = Path(__file__).resolve().parent / "scripts/predict.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--affinity-weights" in result.stdout
    assert "--use-potentials" in result.stdout
    assert "--diffusion-samples" in result.stdout
    assert "--sampling-steps-affinity" in result.stdout
    assert "--diffusion-samples-affinity" in result.stdout
    assert "--msa-server-username" in result.stdout
    assert "--msa-server-password" in result.stdout
    assert "--msa-api-key-header" in result.stdout
    assert "--msa-api-key-value" in result.stdout
    assert "--prewarm-only" in result.stdout
