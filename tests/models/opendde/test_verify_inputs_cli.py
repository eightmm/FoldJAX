from __future__ import annotations

import json

from foldjax.models.opendde.cli import verify_inputs


def test_verify_inputs_cli_records_all_jobs_and_feature_shapes(
    tmp_path, capsys
) -> None:
    input_root = tmp_path / "examples"
    input_root.mkdir()
    input_path = input_root / "tiny.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": "one",
                    "modelSeeds": [101],
                    "sequences": [{"proteinChain": {"sequence": "AC", "count": 1}}],
                },
                {
                    "name": "two",
                    "sequences": [{"dnaSequence": {"sequence": "AG", "count": 1}}],
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "matrix.json"

    verify_inputs.main(
        [
            "--input-root",
            str(input_root),
            "--out",
            str(output_path),
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--max-msa-rows",
            "8",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["all_finite"] is True
    assert report["n_files"] == 1
    assert report["n_jobs"] == 2
    assert [job["name"] for job in report["jobs"]] == ["one", "two"]
    assert report["jobs"][0]["n_token"] == 2
    assert report["jobs"][0]["n_atom"] == 12
    assert report["jobs"][0]["msa_shape"] == [2, 2]
    assert report["jobs"][1]["n_token"] == 2
    assert "wrote:" in capsys.readouterr().out


def test_verify_inputs_reports_a_torch_free_featurization_path(tmp_path) -> None:
    """OpenDDE preprocessing must not import torch.

    ``torch_imported`` reads ``sys.modules``, so it is only meaningful in a
    process that has not already imported torch for some other reason. FoldJAX
    runs every vendored suite in one session and the Boltz/Chai parity tests do
    import torch, so this runs the CLI in a fresh interpreter instead.
    """
    import subprocess
    import sys

    input_root = tmp_path / "examples"
    input_root.mkdir()
    (input_root / "tiny.json").write_text(
        json.dumps(
            [
                {
                    "name": "one",
                    "modelSeeds": [101],
                    "sequences": [{"proteinChain": {"sequence": "AC", "count": 1}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "matrix.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from foldjax.models.opendde.cli import verify_inputs;"
            "import sys; verify_inputs.main(sys.argv[1:])",
            "--input-root",
            str(input_root),
            "--out",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["torch_imported"] is False
    assert report["all_finite"] is True
