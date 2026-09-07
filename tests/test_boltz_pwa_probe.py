import json

import pytest

from bench.boltz_pwa_probe import load_reference, main


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
