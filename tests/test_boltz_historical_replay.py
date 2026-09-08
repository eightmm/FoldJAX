import json
from pathlib import Path

import pytest

from bench.boltz_historical_replay import (
    CHILD,
    main,
    native_hashes,
    save_new,
    source_hashes,
    validate_api,
)


def test_child_program_compiles():
    compile(CHILD, "historical-replay-child", "exec")


def test_preflight_never_launches_and_refuses_existing_output(tmp_path, monkeypatch):
    import bench.boltz_historical_replay as replay

    monkeypatch.setattr(replay, "validate_api", lambda *_: None)
    monkeypatch.setattr(replay, "source_hashes", lambda *_: {"src/model.py": "hash"})
    monkeypatch.setattr(replay, "native_hashes", lambda *_: {"features.npz": "hash"})
    monkeypatch.setattr(
        replay.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    args = [
        "--candidate-root",
        str(tmp_path),
        "--native-capture",
        str(tmp_path),
        "--weights",
        str(checkpoint),
        "--python",
        "/unused/python",
        "--out-dir",
        str(tmp_path / "out"),
        "--preflight-only",
    ]
    assert main(args) == 0
    assert not (tmp_path / "out/completion.json").exists()
    with pytest.raises(FileExistsError):
        main(args)


def test_native_policy_and_hashes(tmp_path):
    meta = dict(
        precision="32",
        kernels=False,
        subsample_msa=False,
        num_samples=5,
        num_steps=200,
        num_recycles=3,
        seed=101,
    )
    for name in ("tape.npz", "features.npz", "trunk.npz", "coordinate.npz"):
        (tmp_path / name).write_bytes(b"fixture")
    path = tmp_path / "tape.json"
    path.write_text(json.dumps(meta))
    before = native_hashes(tmp_path)
    (tmp_path / "features.npz").write_bytes(b"changed")
    assert native_hashes(tmp_path)["features.npz"] != before["features.npz"]
    for name, value in (
        ("precision", "bf16-mixed"),
        ("kernels", True),
        ("subsample_msa", True),
        ("num_samples", 1),
        ("seed", 0),
    ):
        path.write_text(json.dumps({**meta, name: value}))
        with pytest.raises(ValueError):
            native_hashes(tmp_path)


def test_source_hashes_include_relative_names_and_reject_escape(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(ValueError):
        source_hashes(tmp_path)
    (src / "model.py").write_text("x=1")
    assert set(source_hashes(tmp_path)) == {"src/model.py"}
    (src / "escape.py").symlink_to(Path(__file__).resolve())
    with pytest.raises(ValueError, match="escapes"):
        source_hashes(tmp_path)


def test_current_api_and_explicit_option_rejection(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "tests/models/boltz2/scripts/parity_matched_tape.py"
    validate_api(root, runner)
    target = tmp_path / "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def boltz2_trunk_forward(**kwargs): pass\n"
        "def boltz2_sample_forward(**kwargs): pass\n"
        "def _sample_schedule(**kwargs): pass\n"
    )
    with pytest.raises(ValueError, match="unsupported explicit"):
        validate_api(tmp_path, runner)


def test_no_overwrite_or_invented_acceptance(tmp_path):
    path = tmp_path / "record.json"
    save_new(path, {"scientific_acceptance": None})
    with pytest.raises(FileExistsError):
        save_new(path, {"scientific_acceptance": True})
    assert json.loads(path.read_text())["scientific_acceptance"] is None


def test_fp32_runner_does_not_import_new_amp_helper_unconditionally():
    import ast

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / "tests/models/boltz2/scripts/parity_matched_tape.py").read_text()
    )
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert all(
        alias.name != "_cast_trunk_params"
        for node in main.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
