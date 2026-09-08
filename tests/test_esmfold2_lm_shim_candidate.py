import json

import pytest

from bench.esmfold2_lm_shim_candidate import validate_native
from bench.esmfold2_tape import _sha256


@pytest.mark.parametrize(
    "changed", [None, "checkpoint_sha256", "lm_sha256", "archive_sha256", "pair_dtype"]
)
def test_native_shim_binding_fails_closed(tmp_path, changed):
    for name in ("model.safetensors", "config.json", "metadata.json", "native.npz"):
        (tmp_path / name).write_bytes(name.encode())
    metadata = {
        "lm_schema": {"sha256": "lm-content-digest"},
        "binding": {"source": {"native": {
            "transformers/models/esmfold2/modeling_esmfold2_common.py": "source-digest"
        }}},
    }
    report = {
        "source_sha256": "source-digest",
        "checkpoint_sha256": _sha256(tmp_path / "model.safetensors"),
        "config_sha256": _sha256(tmp_path / "config.json"),
        "reference_sha256": _sha256(tmp_path / "metadata.json"),
        "lm_sha256": "lm-content-digest",
        "archive_sha256": _sha256(tmp_path / "native.npz"),
        "autocast": "bfloat16", "pair_dtype": "torch.float32",
    }
    if changed:
        report[changed] = "different"
    (tmp_path / "report.json").write_text(json.dumps(report))
    if changed:
        with pytest.raises(ValueError):
            validate_native(tmp_path, metadata, tmp_path, tmp_path)
    else:
        assert validate_native(tmp_path, metadata, tmp_path, tmp_path) == report
