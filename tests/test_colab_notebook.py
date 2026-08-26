from __future__ import annotations

import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import pytest

import foldjax
from foldjax import (
    BatchReport,
    PredictionFailure,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)
from foldjax.registry import get_backend

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "FoldJAX_Colab.ipynb"
COLAB_COMMON_STACK = {
    "cuequivariance": "0.11.1",
    "cuequivariance-jax": "0.11.1",
    "flax": "0.12.9",
    "jaxlib": "0.11.1",
    "qwix": "0.1.8",
    "tokamax": "0.0.13",
}
COLAB_CUDA_STACK = {
    "cuequivariance-ops-cu12": "0.11.1",
    "cuequivariance-ops-jax-cu12": "0.11.1",
    "jax-cuda12-pjrt": "0.11.1",
    "jax-cuda12-plugin": "0.11.1",
    "triton": "3.7.1",
}


def _document() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[dict]:
    return [cell for cell in _document()["cells"] if cell["cell_type"] == "code"]


def _source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _cell_source(cell_id: str) -> str:
    return _source(next(cell for cell in _document()["cells"] if cell["id"] == cell_id))


def _cell_function(cell_id: str, function_name: str):
    tree = ast.parse(_cell_source(cell_id))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "os": __import__("os"),
        "Path": Path,
        "shutil": shutil,
        "subprocess": subprocess,
    }
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace[function_name]


def test_colab_notebook_is_clean_accelerator_workflow() -> None:
    document = _document()
    how_to_run = _source(document["cells"][0])

    assert document["nbformat"] == 4
    assert document["nbformat_minor"] >= 5
    assert document["metadata"]["accelerator"] == "GPU"
    assert document["metadata"]["language_info"]["version"] == "3.13"
    assert document["cells"][0]["cell_type"] == "markdown"
    assert "detects the active runtime" in how_to_run
    assert "v5e" not in how_to_run
    assert "Python 3.13" not in how_to_run
    for cell in _code_cells():
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        assert cell["metadata"]["cellView"] == "form"
    ids = [cell["id"] for cell in document["cells"]]
    assert len(ids) == len(set(ids))
    assert ids.index("configure-run") < ids.index("install-foldjax")
    assert ids.index("fetch-weights") < ids.index("build-job")
    assert ids.index("build-job") < ids.index("plan-runs")
    assert ids.index("plan-runs") < ids.index("run-predictions")
    assert ids.index("run-predictions") < ids.index("compare-results")
    assert ids.index("compare-results") < ids.index("view-structures")
    assert ids.index("view-structures") < ids.index("download-results")
    assert "PROTEIN_CHAINS" not in _cell_source("configure-run")
    assert "PROTEIN_CHAINS" in _cell_source("build-job")


def test_colab_notebook_installs_the_detected_accelerator_runtime() -> None:
    install = _cell_source("install-foldjax")
    all_source = "\n".join(_source(cell) for cell in _code_cells())

    assert 'FOLDJAX_REF = "d21757f14cb199b9ead7fa298076ad68119a0dc7"' in install
    assert 'install_extras = ["cuda12"] if ACCELERATOR_KIND == "gpu" else []' in install
    assert 'install_extras.append("alphafold3")' in install
    assert 'install_extras.append("openfold3-preprocess")' in install
    assert "','.join(install_extras)" in install
    assert "install_profile" in install
    assert '(ACCELERATOR_KIND, *install_extras)' in install
    assert "FoldJAX.git@{FOLDJAX_REF}" in install
    assert 'jax[tpu]=={JAX_VERSION}' in install
    assert '"py3Dmol>=2.0,<3"' in install
    assert all_source.index('"pip"') < all_source.index("import jax")
    assert 'sys.version_info[:2] != (3, 13)' in install
    assert 'Path("/dev/accel0").exists()' in install
    assert 'shutil.which("nvidia-smi")' in install
    assert '["nvidia-smi", "-L"]' in install
    assert '"--query-gpu=compute_cap"' in install
    assert '"Dependencies installed"' in install
    assert "The runtime will restart once" in install
    assert "signal.SIGKILL" in install
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in all_source
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND" in all_source
    assert "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND" in all_source
    assert "jax.__version__ != JAX_VERSION" in all_source
    assert "metadata.PackageNotFoundError" in all_source
    for package, version in (COLAB_COMMON_STACK | COLAB_CUDA_STACK).items():
        assert f'"{package}": "{version}"' in all_source
    assert 'device.platform == ACCELERATOR_KIND' in all_source
    assert 'required_packages.add("libtpu")' in all_source


def test_colab_install_stops_before_pip_on_wrong_python(monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "version_info", (3, 12, 12))
    monkeypatch.setattr(sys, "version", "3.12.12 (Colab runtime)")
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError) as error:
        exec(
            _cell_source("install-foldjax"),
            {"SELECTED_MODELS": ("protenix",)},
        )

    message = str(error.value)
    assert "Python 3.13" in message
    assert "3.12.12" in message
    assert "Runtime → Change runtime type → Runtime version" in message
    assert "reconnect" in message
    assert calls == []


def test_colab_tpu_install_assembles_the_libtpu_stack(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    source = _cell_source("install-foldjax").replace(
        'f"/content/.foldjax-colab-',
        f'f"{tmp_path}/.foldjax-colab-',
    )
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e-1")
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda command: calls.append(tuple(command)),
    )
    os_module = __import__("os")
    monkeypatch.setattr(os_module, "kill", lambda *_args: None)

    namespace = {
        "SELECTED_MODELS": ("openfold3",),
        "foldjax_card": lambda *_args, **_kwargs: None,
    }
    exec(source, namespace)

    assert namespace["ACCELERATOR_KIND"] == "tpu"
    assert len(calls) == 1
    assert "jax[tpu]==0.11.1" in calls[0]
    assert any("foldjax[openfold3-preprocess]" in argument for argument in calls[0])
    assert not any("cuda12" in argument for argument in calls[0])


def test_colab_rerun_reuses_the_matching_install_profile(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    cards = []
    source = _cell_source("install-foldjax").replace(
        'f"/content/.foldjax-colab-',
        f'f"{tmp_path}/.foldjax-colab-',
    )
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "test-tpu")
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda command: calls.append(tuple(command)),
    )
    os_module = __import__("os")
    monkeypatch.setattr(os_module, "kill", lambda *_args: None)
    namespace = {
        "SELECTED_MODELS": ("openfold3",),
        "foldjax_card": lambda *args, **kwargs: cards.append((args, kwargs)),
    }

    exec(source, namespace)
    exec(source, namespace)

    assert len(calls) == 1
    assert any("Dependencies ready" in str(card) for card in cards)


def test_colab_accelerator_detection_prefers_tpu_markers(monkeypatch) -> None:
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e-1")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    detect_accelerator = _cell_function("install-foldjax", "detect_accelerator")

    assert detect_accelerator() == "tpu"


def test_colab_accelerator_detection_requires_a_working_gpu_probe(
    monkeypatch,
) -> None:
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="GPU 0\n"),
    )

    detect_accelerator = _cell_function("install-foldjax", "detect_accelerator")

    assert detect_accelerator() == "gpu"


@pytest.mark.parametrize(
    ("accelerator", "probe", "expected"),
    (
        ("tpu", SimpleNamespace(returncode=0, stdout="8.0\n"), None),
        ("gpu", SimpleNamespace(returncode=0, stdout="7.5\n"), (7, 5)),
        ("gpu", SimpleNamespace(returncode=0, stdout="8.0\n"), (8, 0)),
        ("gpu", SimpleNamespace(returncode=1, stdout=""), None),
        ("gpu", SimpleNamespace(returncode=0, stdout="unknown\n"), None),
    ),
)
def test_colab_detects_gpu_compute_capability_conservatively(
    accelerator: str,
    probe: SimpleNamespace,
    expected: tuple[int, int] | None,
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or probe,
    )
    detect = _cell_function("install-foldjax", "detect_gpu_compute_capability")

    assert detect(accelerator) == expected
    assert bool(calls) is (accelerator == "gpu")
    if calls:
        assert "--query-gpu=compute_cap" in calls[0]


def test_colab_accelerator_detection_rejects_an_unusable_gpu_probe(
    monkeypatch,
) -> None:
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )

    detect_accelerator = _cell_function("install-foldjax", "detect_accelerator")

    with pytest.raises(RuntimeError, match="No supported accelerator"):
        detect_accelerator()


def test_colab_runtime_matches_the_root_project_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    assert "jax==0.11.1" in project["project"]["dependencies"]


def test_colab_gpu_check_executes_with_the_pinned_stack(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    cards = []
    device = SimpleNamespace(
        platform="gpu",
        device_kind="NVIDIA test GPU",
        memory_stats=lambda: {"bytes_limit": 16 * 2**30},
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    stack = COLAB_COMMON_STACK | COLAB_CUDA_STACK
    monkeypatch.setattr(metadata, "version", stack.__getitem__)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="NVIDIA test GPU, 16000 MiB, 999.0"
        ),
    )

    exec(
        _cell_source("check-accelerator"),
        {
            "ACCELERATOR_KIND": "gpu",
            "JAX_VERSION": "0.11.1",
            "install_marker": tmp_path / "installed",
            "subprocess": subprocess,
            "sys": sys,
            "foldjax_card": lambda *args, **kwargs: cards.append((args, kwargs)),
        },
    )

    assert any("16.0 GiB VRAM" in str(card) for card in cards)


def test_colab_tpu_check_executes_without_querying_nvidia(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    cards = []
    device = SimpleNamespace(
        platform="tpu",
        device_kind="TPU v5e",
        memory_stats=lambda: {"bytes_limit": 16 * 2**30},
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(
        metadata,
        "version",
        lambda package: (
            "0.0.46" if package == "libtpu" else COLAB_COMMON_STACK[package]
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("TPU verification queried nvidia-smi"),
    )

    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "JAX_VERSION": "0.11.1",
        "install_marker": tmp_path / "installed",
        "subprocess": subprocess,
        "sys": sys,
        "foldjax_card": lambda *args, **kwargs: cards.append((args, kwargs)),
    }
    exec(_cell_source("check-accelerator"), namespace)

    assert namespace["DEVICE_MEMORY_BYTES"] == 16 * 2**30
    assert any("16.0 GiB HBM" in str(card) for card in cards)


def test_colab_accelerator_check_reports_unavailable_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    cards = []
    device = SimpleNamespace(
        platform="tpu", device_kind="TPU", memory_stats=lambda: None
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(
        metadata,
        "version",
        lambda package: (
            "0.0.46" if package == "libtpu" else COLAB_COMMON_STACK[package]
        ),
    )

    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "JAX_VERSION": "0.11.1",
        "install_marker": tmp_path / "installed",
        "subprocess": subprocess,
        "sys": sys,
        "foldjax_card": lambda *args, **kwargs: cards.append((args, kwargs)),
    }
    exec(_cell_source("check-accelerator"), namespace)

    assert namespace["DEVICE_MEMORY_BYTES"] is None
    assert any("capacity unavailable" in str(card) for card in cards)


def test_colab_compile_cache_pack_round_trips_selected_device_and_model(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    device = SimpleNamespace(
        platform="tpu",
        device_kind="TPU v5e test",
        memory_stats=lambda: {"bytes_limit": 16 * 2**30},
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(
        metadata,
        "version",
        lambda package: (
            "0.0.46" if package == "libtpu" else COLAB_COMMON_STACK[package]
        ),
    )
    local_cache = tmp_path / "local-cache"
    archive_store = tmp_path / "drive" / "compile-cache"
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "JAX_VERSION": "0.11.1",
        "install_marker": tmp_path / "installed",
        "subprocess": subprocess,
        "sys": sys,
        "SELECTED_MODELS": ("protenix",),
        "KERNEL_PROFILE": "portable XLA",
        "PERSIST_COMPILE_CACHE_TO_DRIVE": True,
        "COMPILE_CACHE": local_cache,
        "COMPILE_CACHE_STAGING": tmp_path / "staging",
        "COMPILE_ARCHIVE_STORE": archive_store,
        "COMPILE_CACHE_LOCAL_MAX_BYTES": 2**20,
        "COMPILE_CACHE_DRIVE_MAX_BYTES": 2**20,
        "COMPILE_CACHE_MAX_AGE_DAYS": 30,
        "foldjax_card": lambda *_args, **_kwargs: None,
    }

    exec(_cell_source("check-accelerator"), namespace)

    entry = local_cache / "protenix" / "weights" / "profile" / "entry.bin"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"compiled executable")
    event = namespace["sync_compile_cache_archive"]("protenix")
    archive = namespace["model_compile_archive"]("protenix")

    assert event["status"] == "saved"
    assert archive.is_file()
    assert "jax-0.11.1-jaxlib-0.11.1" in str(archive)
    assert "TPU-v5e-test" in str(archive)
    assert "portable-XLA" in str(archive)

    shutil.rmtree(local_cache)
    restored, errors = namespace["restore_compile_cache_archives"]()

    assert errors == ()
    assert restored[0][0] == "protenix"
    assert entry.read_bytes() == b"compiled executable"


def test_colab_compile_cache_restore_rejects_unsafe_archive_members(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    device = SimpleNamespace(
        platform="tpu", device_kind="TPU", memory_stats=lambda: {}
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(
        metadata,
        "version",
        lambda package: (
            "0.0.46" if package == "libtpu" else COLAB_COMMON_STACK[package]
        ),
    )
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "JAX_VERSION": "0.11.1",
        "install_marker": tmp_path / "installed",
        "subprocess": subprocess,
        "sys": sys,
        "SELECTED_MODELS": (),
        "KERNEL_PROFILE": "portable XLA",
        "PERSIST_COMPILE_CACHE_TO_DRIVE": True,
        "COMPILE_CACHE": tmp_path / "local-cache",
        "COMPILE_CACHE_STAGING": tmp_path / "staging",
        "COMPILE_ARCHIVE_STORE": tmp_path / "drive-cache",
        "COMPILE_CACHE_LOCAL_MAX_BYTES": 2**20,
        "COMPILE_CACHE_DRIVE_MAX_BYTES": 2**20,
        "COMPILE_CACHE_MAX_AGE_DAYS": 30,
        "foldjax_card": lambda *_args, **_kwargs: None,
    }
    exec(_cell_source("check-accelerator"), namespace)
    namespace["SELECTED_MODELS"] = ("protenix",)
    archive = namespace["model_compile_archive"]("protenix")
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../escaped.bin")
        payload = b"unsafe"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    restored, errors = namespace["restore_compile_cache_archives"]()

    assert restored == ()
    assert errors and "Unsafe compilation-cache archive member" in errors[0]
    assert not archive.exists()
    assert not (tmp_path / "escaped.bin").exists()


def test_colab_compile_cache_pruning_is_age_and_size_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    device = SimpleNamespace(
        platform="tpu", device_kind="TPU", memory_stats=lambda: {}
    )
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(
        metadata,
        "version",
        lambda package: (
            "0.0.46" if package == "libtpu" else COLAB_COMMON_STACK[package]
        ),
    )
    archive_store = tmp_path / "drive-cache"
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "JAX_VERSION": "0.11.1",
        "install_marker": tmp_path / "installed",
        "subprocess": subprocess,
        "sys": sys,
        "SELECTED_MODELS": (),
        "KERNEL_PROFILE": "portable XLA",
        "PERSIST_COMPILE_CACHE_TO_DRIVE": True,
        "COMPILE_CACHE": tmp_path / "local-cache",
        "COMPILE_CACHE_STAGING": tmp_path / "staging",
        "COMPILE_ARCHIVE_STORE": archive_store,
        "COMPILE_CACHE_LOCAL_MAX_BYTES": 2**20,
        "COMPILE_CACHE_DRIVE_MAX_BYTES": 1_500,
        "COMPILE_CACHE_MAX_AGE_DAYS": 30,
        "foldjax_card": lambda *_args, **_kwargs: None,
    }
    exec(_cell_source("check-accelerator"), namespace)
    old = archive_store / "old.tar"
    recent = archive_store / "recent.tar"
    newest = archive_store / "newest.tar"
    old.write_bytes(b"o" * 600)
    recent.write_bytes(b"r" * 800)
    newest.write_bytes(b"n" * 800)
    now = time.time()
    os.utime(old, (now - 31 * 86400, now - 31 * 86400))
    os.utime(recent, (now - 10, now - 10))
    os.utime(newest, (now, now))

    namespace["prune_drive_compile_archives"]()

    assert not old.exists()
    assert not recent.exists()
    assert newest.exists()

    namespace["COMPILE_CACHE_LOCAL_MAX_BYTES"] = 1_000
    local_old = namespace["COMPILE_CACHE"] / "protenix" / "old.bin"
    local_new = namespace["COMPILE_CACHE"] / "protenix" / "new.bin"
    local_old.parent.mkdir(parents=True)
    local_old.write_bytes(b"o" * 800)
    local_new.write_bytes(b"n" * 800)
    os.utime(local_old, (now - 10, now - 10))
    os.utime(local_new, (now, now))

    namespace["prune_local_compile_cache"]()

    assert not local_old.exists()
    assert local_new.exists()


def test_colab_gpu_check_clears_marker_before_reinstall(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    marker = tmp_path / "installed"
    marker.write_text("stale", encoding="utf-8")
    device = SimpleNamespace(
        platform="gpu", device_kind="GPU", memory_stats=lambda: {}
    )
    monkeypatch.setattr(jax, "__version__", "0.11.0")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    stack = COLAB_COMMON_STACK | COLAB_CUDA_STACK
    monkeypatch.setattr(metadata, "version", stack.__getitem__)

    with pytest.raises(RuntimeError, match="reinstall the stack"):
        exec(
            _cell_source("check-accelerator"),
            {
                "ACCELERATOR_KIND": "gpu",
                "JAX_VERSION": "0.11.1",
                "install_marker": marker,
                "subprocess": subprocess,
                "sys": sys,
            },
        )

    assert not marker.exists()


def test_colab_notebook_exposes_practical_multi_model_choices() -> None:
    source = "\n".join(_source(cell) for cell in _code_cells())
    markdown = "\n".join(
        _source(cell)
        for cell in _document()["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert "MODEL_PRESET" not in source
    for field in (
        "RUN_PROTENIX",
        "RUN_OPENDDE",
        "RUN_BOLTZ2",
        "RUN_ESMFOLD2",
        "RUN_OPENFOLD3",
        "RUN_ALPHAFOLD3",
    ):
        assert f"{field} =" in source
    assert "RUN_BOLTZ2 = True" in source
    assert "PROTEIN_CHAINS" in source
    assert "DNA_CHAINS" in source
    assert "RNA_CHAINS" in source
    assert "LIGAND_CCD_CODES" in source
    assert "LIGAND_SMILES" in source
    assert 'compact.split(":")' in source
    assert 'LIGAND_CCD_CODES.split(",")' in source
    assert 'LIGAND_SMILES.split(";")' in source
    assert "PERSIST_MODEL_ASSETS_TO_DRIVE = True" in source
    assert "PERSIST_MSA_TO_DRIVE = True" in source
    assert "PERSIST_OUTPUTS_TO_DRIVE = True" in source
    assert "PERSIST_COMPILE_CACHE_TO_DRIVE = True" in source
    assert 'OUTPUT_ROOT = OUTPUT_BASE / JOB_SLUG / RUN_LABEL' in source
    assert "output_dir=OUTPUT_ROOT / model_name," in source
    assert "output_dir=OUTPUT_ROOT / model_name / JOB_SLUG" not in source
    assert "LEGACY_OUTPUT_DIRS" in source
    assert "Earlier output layout found" in source
    assert 'f"{JOB_SLUG}-{RUN_LABEL}-foldjax.zip"' in source
    assert "manually obtained weights" not in source
    assert "manual/gated parameters" not in source
    assert "user-supplied parameters downloaded directly from Google" in markdown
    assert "Keep output persistence enabled" in markdown
    assert 'RUN_MODE = "Default"' in source
    assert (
        'RUN_MODE = "Default"  # @param '
        '["Fast demo", "Default", "Custom"]'
    ) in source
    assert 'INPUT_MODE = "Form"' in source
    assert 'MAX_MSA_DEPTH = 0' in source
    assert 'PREFLIGHT_ONLY = EXECUTION == "Setup only"' in source
    assert "ESMFOLD2_PROFILE" not in source
    assert "profile_status" not in source
    assert 'REPRESENTATIONS = "None"' in source
    assert "PADDING_MODE" not in source
    assert "PADDING_REQUESTS" not in source
    assert "PAD_TOKENS" not in source
    assert 'MSA_POLICY = "auto"' in source
    assert "job_path.read_text()" not in source
    assert '{"num_samples": 1, "num_steps": 20, "num_recycles": 1}' in source
    assert "protein=protein_sequences" in source
    assert "dna=dna_sequences" in source
    assert "rna=rna_sequences" in source
    assert "ligand_ccd=ligand_ccds" in source
    assert "ligand_smiles=ligand_smiles" in source
    assert "model=model_name" in source
    assert "predict_batch(model_request)" in source
    assert "BatchReport(" in source
    assert '"dtype": "float32"' in source
    assert '"attention_kernel": "xla"' in source
    assert "CONTINUE_ON_ERROR" in source
    assert "model_info(model_name)" in source
    assert "for model_name in SELECTED_MODELS" in source
    assert "INPUT_ENTITY_TYPES" in source
    assert "INPUT_REQUIRED_FEATURES" in source
    assert "OPENFOLD3_WEIGHTS_PATH" not in source
    assert "ALPHAFOLD3_WEIGHTS_PATH" not in source
    assert 'JOB_NAME = "protein-rna-atp-demo"' in source
    assert 'RNA_CHAINS = "GGGAAACCC"' in source
    assert 'LIGAND_CCD_CODES = "ATP"' in source
    assert "RUN_OPENFOLD3 = True" in source
    assert "resume=True" in source
    assert "STRUCTURES" in source
    assert "comparison.to_csv" in source
    assert '"batch_report.json"' in source
    assert "native score" in markdown.lower()
    assert "sequential" in markdown.lower()
    assert "ColabFold MMseqs2" in markdown
    assert "ESMFold2 accepts protein, DNA, RNA" in markdown
    assert "structure+ESMC+chemistry bundle is about 26.77 GB" in markdown
    assert (
        "OpenFold3's compatible p1 checkpoint is a public managed download"
        in markdown
    )
    assert "p2/OpenBind checkpoints are not compatible" in markdown
    assert "AlphaFold3 still requires parameters" in markdown
    assert "requires a runtime reporting more than 16 GiB" in markdown
    assert "structure-only" not in markdown.lower()
    assert "same sequence and search settings are reused automatically" in markdown

    configure = _cell_source("configure-run")
    assert configure.count("# @param") == 6
    assert "PERSIST_MODEL_ASSETS_TO_DRIVE" not in configure
    assert "PERSIST_MSA_TO_DRIVE" not in configure
    assert "RUN_MODE" not in configure
    assert "NUM_SEEDS" not in configure
    assert "MSA_POLICY" not in configure
    assert "EXECUTION" not in configure
    assert "CUSTOM_NUM_SAMPLES" not in configure
    assert "DOWNLOAD_RESULTS" not in configure
    assert (
        "PERSIST_MODEL_ASSETS_TO_DRIVE = True"
        in _cell_source("configure-runtime")
    )
    assert "PERSIST_MSA_TO_DRIVE = True" in _cell_source("configure-runtime")
    assert "PERSIST_OUTPUTS_TO_DRIVE = True" in _cell_source(
        "configure-runtime"
    )
    assert "PERSIST_COMPILE_CACHE_TO_DRIVE = True" in _cell_source(
        "configure-runtime"
    )
    assert "Drive persistence is enabled by default" in markdown
    assert "JAX always compiles against fast local storage" in markdown
    assert "2.5 GiB global limit and 30-day retention" in markdown
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/FoldJAX")' in source
    assert 'DRIVE_ROOT / "model-assets"' in source
    assert 'DRIVE_ROOT / "msa"' in source
    assert 'DRIVE_ROOT / "outputs"' in source
    assert "ALPHAFOLD3_WEIGHTS_PATH" not in _cell_source("fetch-weights")
    assert "MANUAL_WEIGHT_TEXT" not in _cell_source("fetch-weights")
    assert "AlphaFold 3 private parameter location" in _cell_source("fetch-weights")
    assert (
        'MODEL_DIRECTORY_MAP["alphafold3"]["weights"]'
        in _cell_source("fetch-weights")
    )
    runtime = _cell_source("configure-runtime")
    assert "MODEL_STORE_ROOT = MODEL_ASSET_STORE" in runtime
    assert '`FoldJAX/model-assets/<model>/' in markdown
    assert 'MODEL_ASSET_STORE / "shared" / "assets"' in runtime
    assert "migrate_legacy_model_store" in runtime
    fetch_weights = _cell_source("fetch-weights")
    assert "def run_setup_command" in fetch_weights
    assert 'capture_output=True' in fetch_weights
    assert 'stage="weight download, verification, or conversion"' in fetch_weights
    assert 'stage="generated runtime preparation"' in fetch_weights
    build_job = _cell_source("build-job")
    assert 'MSA_POLICY = "auto"' in build_job
    plan_runs = _cell_source("plan-runs")
    assert "Sampling and memory" not in plan_runs
    assert "Shared custom schedule — applies to every selected model" in plan_runs
    assert "select one model for individual tuning" in plan_runs
    assert "select one model for individual tuning" in plan_runs
    assert "explicit values replace the generated `0..NUM_SEEDS-1`" in plan_runs
    assert "Optional memory override — applies to every run mode" in plan_runs
    assert 'RUN_MODE = "Default"' in plan_runs
    assert "NUM_SEEDS = 1" in plan_runs
    assert 'EXECUTION = "Predict"' in plan_runs
    assert "CONTINUE_ON_ERROR = True" in plan_runs
    assert "CUSTOM_NUM_SAMPLES = 1" in plan_runs
    assert "DOWNLOAD_RESULTS = True" in _cell_source("download-results")


def test_colab_discloses_each_checkpoint_licence_with_a_source() -> None:
    markdown = "\n".join(
        _source(cell)
        for cell in _document()["cells"]
        if cell["cell_type"] == "markdown"
    )

    expected = (
        (
            "Protenix",
            "Apache-2.0; academic and commercial use",
            "github.com/bytedance/Protenix",
        ),
        (
            "OpenDDE",
            "Apache-2.0 released checkpoints",
            "huggingface.co/aurekaresearch/OpenDDE",
        ),
        (
            "Boltz-2",
            "MIT; academic and commercial use",
            "github.com/jwohlwend/boltz",
        ),
        (
            "ESMFold2",
            "MIT for ESMFold2 and ESMC-6B; Biohub AUP applies",
            "huggingface.co/biohub/ESMFold2",
        ),
        (
            "OpenFold3",
            "Apache-2.0 model and parameters",
            "huggingface.co/OpenFold/OpenFold3",
        ),
        (
            "AlphaFold 3",
            "non-commercial use by/for non-commercial organizations",
            "github.com/google-deepmind/alphafold3",
        ),
    )
    for model, terms, source in expected:
        assert model in markdown
        assert terms in markdown
        assert source in markdown
    assert '"licence": info.weights_licence' in _cell_source("fetch-weights")
    assert '"source": info.weights_source' in _cell_source("fetch-weights")


def test_registered_checkpoint_licences_are_explicit() -> None:
    expected = {
        "alphafold3": "AlphaFold 3 Model Parameters Terms of Use",
        "boltz2": "MIT (code and weights; academic and commercial use)",
        "esmfold2": "MIT (ESMFold2 and ESMC-6B weights)",
        "opendde": "Apache-2.0 (released checkpoints and code)",
        "openfold3": "Apache-2.0 (OpenFold3 model and parameters",
        "protenix": "Apache-2.0 (model parameters; academic and commercial use)",
    }

    for model, licence in expected.items():
        assert licence in foldjax.model_info(model).weights_licence
    assert "see the" not in " ".join(
        foldjax.model_info(model).weights_licence for model in expected
    ).lower()


def test_colab_notebook_renders_a_guided_workflow_instead_of_raw_logs() -> None:
    code_by_id = {cell["id"]: _source(cell) for cell in _code_cells()}
    markdown = "\n".join(
        _source(cell)
        for cell in _document()["cells"]
        if cell["cell_type"] == "markdown"
    )

    for cell_id in (
        "configure-run",
        "install-foldjax",
        "configure-runtime",
        "check-accelerator",
        "fetch-weights",
        "build-job",
        "run-predictions",
        "compare-results",
        "view-structures",
        "download-results",
    ):
        assert "foldjax_card(" in code_by_id[cell_id]
    for cell_id in ("fetch-weights", "plan-runs", "run-predictions", "compare-results"):
        assert "foldjax_table(" in code_by_id[cell_id]
    assert "print(" not in "\n".join(code_by_id.values())
    for section in (
        "RUNTIME · STORAGE",
        "CHECKPOINTS · COMPATIBILITY",
        "ONE INPUT · MANY MODELS",
        "EXECUTION · SEQUENTIAL ACCELERATOR SESSIONS",
        "ANALYSIS · NATIVE SCORES",
        "VISUALIZATION · STRUCTURE EXPLORER",
        "EXPORT · REPRODUCIBLE BUNDLE",
    ):
        assert section in markdown


def test_colab_notebook_python_cells_are_syntactically_valid() -> None:
    for cell in _code_cells():
        ast.parse(_source(cell), filename=f"{NOTEBOOK}:{cell['id']}")


@pytest.mark.parametrize(
    (
        "accelerator",
        "gpu_capability",
        "multiplication_backend",
        "attention_backend",
        "profile",
    ),
    (
        ("gpu", (7, 5), "xla", "xla_jit", "portable XLA"),
        ("gpu", (8, 0), "cueq", "cueq_jit", "CUDA fused"),
        ("gpu", None, "xla", "xla_jit", "portable XLA"),
        ("tpu", None, "xla", "xla_jit", "portable XLA"),
    ),
)
def test_colab_configures_accelerator_specific_kernels(
    accelerator: str,
    gpu_capability: tuple[int, int] | None,
    multiplication_backend: str,
    attention_backend: str,
    profile: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _cell_source("configure-runtime")
    source = source.replace(
        'WORK_DIR = Path("/content/foldjax-colab")',
        f'WORK_DIR = Path({str(tmp_path / "work")!r})',
    )
    source = source.replace(
        'foldjax_home = Path("/content/foldjax-cache")',
        f'foldjax_home = Path({str(tmp_path / "cache")!r})',
    )
    for setting in (
        "PERSIST_MODEL_ASSETS_TO_DRIVE",
        "PERSIST_MSA_TO_DRIVE",
        "PERSIST_OUTPUTS_TO_DRIVE",
        "PERSIST_COMPILE_CACHE_TO_DRIVE",
    ):
        source = source.replace(f"{setting} = True", f"{setting} = False")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    namespace = {
        "ACCELERATOR_KIND": accelerator,
        "GPU_COMPUTE_CAPABILITY": gpu_capability,
        "SELECTED_MODELS": (),
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "PERSIST_COMPILE_CACHE_TO_DRIVE": False,
        "Path": Path,
        "os": __import__("os"),
        "foldjax_card": lambda *_args, **_kwargs: None,
    }

    exec(source, namespace)

    assert namespace["KERNEL_PROFILE"] == profile
    assert (
        namespace["os"].environ["PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND"]
        == multiplication_backend
    )
    assert (
        namespace["os"].environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"]
        == multiplication_backend
    )
    assert (
        namespace["os"].environ["PROTENIX_TRIANGLE_BACKEND"]
        == attention_backend
    )
    assert (
        ("XLA_PYTHON_CLIENT_MEM_FRACTION" in namespace["os"].environ)
        is (accelerator == "gpu")
    )


def test_colab_defaults_model_assets_msa_outputs_and_compile_cache_to_drive(
    tmp_path: Path, monkeypatch
) -> None:
    source = _cell_source("configure-runtime")
    source = source.replace(
        'WORK_DIR = Path("/content/foldjax-colab")',
        f'WORK_DIR = Path({str(tmp_path / "work")!r})',
    )
    source = source.replace(
        'foldjax_home = Path("/content/foldjax-cache")',
        f'foldjax_home = Path({str(tmp_path / "local-cache")!r})',
    )
    source = source.replace(
        'DRIVE_ROOT = Path("/content/drive/MyDrive/FoldJAX")',
        f'DRIVE_ROOT = Path({str(tmp_path / "drive-root")!r})',
    )
    cached_weight = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "models"
        / "openfold3"
        / "weights"
        / "p1.pt"
    )
    cached_download = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "downloads"
        / "openfold3"
        / "source.bin"
    )
    cached_weight.parent.mkdir(parents=True)
    cached_weight.write_bytes(b"cached")
    cached_download.parent.mkdir(parents=True)
    cached_download.write_bytes(b"download")
    cached_shared_asset = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "assets"
        / "components.v1.cif"
    )
    cached_shared_asset.parent.mkdir(parents=True)
    cached_shared_asset.write_bytes(b"shared")
    mounts = []
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mounts.append)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "GPU_COMPUTE_CAPABILITY": None,
        "SELECTED_MODELS": ("openfold3", "alphafold3"),
        "PERSIST_MODEL_ASSETS_TO_DRIVE": True,
        "PERSIST_MSA_TO_DRIVE": True,
        "PERSIST_OUTPUTS_TO_DRIVE": True,
        "Path": Path,
        "os": __import__("os"),
        "foldjax_card": lambda *_args, **_kwargs: None,
    }

    exec(source, namespace)

    assert mounts == ["/content/drive"]
    assert namespace["foldjax_home"] == tmp_path / "local-cache"
    assert namespace["DRIVE_ROOT"] == tmp_path / "drive-root"
    assert namespace["MODEL_ASSET_STORE"] == tmp_path / "drive-root" / "model-assets"
    assert namespace["MODEL_STORE_ROOT"] == namespace["MODEL_ASSET_STORE"]
    assert namespace["MSA_STORE"] == tmp_path / "drive-root" / "msa"
    assert namespace["OUTPUT_BASE"] == tmp_path / "drive-root" / "outputs"
    assert (
        namespace["COMPILE_ARCHIVE_STORE"]
        == tmp_path / "drive-root" / "compile-cache"
    )
    assert namespace["COMPILE_CACHE_DRIVE_MAX_BYTES"] == int(2.5 * 2**30)
    assert namespace["COMPILE_CACHE"] == tmp_path / "work" / "compile-cache"
    assert (namespace["foldjax_home"] / "weights").is_dir()
    assert (namespace["foldjax_home"] / "weights" / "openfold3").is_symlink()
    assert (namespace["foldjax_home"] / "assets").is_symlink()
    assert (namespace["foldjax_home"] / "msa").is_symlink()
    assert not (namespace["MODEL_ASSET_STORE"] / "weights").exists()
    assert (
        namespace["MODEL_STORE_ROOT"] / "alphafold3" / "weights"
    ).is_dir()
    assert namespace["MIGRATED_MODEL_STORES"] == (
        "openfold3/weights",
        "openfold3/downloads",
        "shared/assets",
    )
    assert not (namespace["MODEL_ASSET_STORE"] / "models").exists()
    assert (
        namespace["foldjax_home"] / "weights" / "openfold3" / "p1.pt"
    ).read_bytes() == b"cached"
    assert (
        namespace["MODEL_STORE_ROOT"] / "openfold3" / "weights" / "p1.pt"
    ).read_bytes() == b"cached"
    assert (
        namespace["MODEL_STORE_ROOT"]
        / "openfold3"
        / "downloads"
        / "source.bin"
    ).read_bytes() == b"download"
    assert (namespace["SHARED_ASSET_STORE"] / "components.v1.cif").read_bytes() == (
        b"shared"
    )

    exec(source, namespace)

    assert namespace["MIGRATED_MODEL_STORES"] == ()
    assert (
        namespace["MODEL_STORE_ROOT"] / "openfold3" / "weights" / "p1.pt"
    ).read_bytes() == b"cached"


def test_colab_can_persist_model_assets_without_persisting_msa(
    tmp_path: Path, monkeypatch
) -> None:
    source = _cell_source("configure-runtime")
    source = source.replace(
        "PERSIST_MSA_TO_DRIVE = True",
        "PERSIST_MSA_TO_DRIVE = False",
    )
    replacements = {
        'WORK_DIR = Path("/content/foldjax-colab")': (
            f'WORK_DIR = Path({str(tmp_path / "work")!r})'
        ),
        'foldjax_home = Path("/content/foldjax-cache")': (
            f'foldjax_home = Path({str(tmp_path / "local-cache")!r})'
        ),
        'DRIVE_ROOT = Path("/content/drive/MyDrive/FoldJAX")': (
            f'DRIVE_ROOT = Path({str(tmp_path / "drive-root")!r})'
        ),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    mounts = []
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mounts.append)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {
        "ACCELERATOR_KIND": "gpu",
        "GPU_COMPUTE_CAPABILITY": None,
        "SELECTED_MODELS": ("openfold3",),
        "Path": Path,
        "os": __import__("os"),
        "foldjax_card": lambda *_args, **_kwargs: None,
    }

    exec(source, namespace)

    assert mounts == ["/content/drive"]
    assert (namespace["foldjax_home"] / "weights").is_dir()
    assert (namespace["foldjax_home"] / "weights" / "openfold3").is_symlink()
    assert not (namespace["foldjax_home"] / "msa").is_symlink()
    assert namespace["MSA_STORE"] == namespace["foldjax_home"] / "msa"


def test_colab_model_store_migration_never_merges_conflicting_directories(
    tmp_path: Path, monkeypatch
) -> None:
    source = _cell_source("configure-runtime")
    replacements = {
        'WORK_DIR = Path("/content/foldjax-colab")': (
            f'WORK_DIR = Path({str(tmp_path / "work")!r})'
        ),
        'foldjax_home = Path("/content/foldjax-cache")': (
            f'foldjax_home = Path({str(tmp_path / "local-cache")!r})'
        ),
        'DRIVE_ROOT = Path("/content/drive/MyDrive/FoldJAX")': (
            f'DRIVE_ROOT = Path({str(tmp_path / "drive-root")!r})'
        ),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    old_weight = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "weights"
        / "openfold3"
        / "p1.pt"
    )
    new_weight = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "openfold3"
        / "weights"
        / "p1.pt"
    )
    old_weight.parent.mkdir(parents=True)
    new_weight.parent.mkdir(parents=True)
    old_weight.write_bytes(b"old")
    new_weight.write_bytes(b"new")
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda _path: None)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    with pytest.raises(RuntimeError, match="Cannot merge the old and new"):
        exec(
            source,
            {
                "ACCELERATOR_KIND": "gpu",
                "GPU_COMPUTE_CAPABILITY": (7, 5),
                "SELECTED_MODELS": ("openfold3",),
                "Path": Path,
                "os": __import__("os"),
                "foldjax_card": lambda *_args, **_kwargs: None,
            },
        )

    assert old_weight.read_bytes() == b"old"
    assert new_weight.read_bytes() == b"new"


def test_colab_detects_private_alphafold3_parameters_in_the_drive_store(
    tmp_path: Path, monkeypatch
) -> None:
    source = _cell_source("configure-runtime")
    replacements = {
        'WORK_DIR = Path("/content/foldjax-colab")': (
            f'WORK_DIR = Path({str(tmp_path / "work")!r})'
        ),
        'foldjax_home = Path("/content/foldjax-cache")': (
            f'foldjax_home = Path({str(tmp_path / "local-cache")!r})'
        ),
        'DRIVE_ROOT = Path("/content/drive/MyDrive/FoldJAX")': (
            f'DRIVE_ROOT = Path({str(tmp_path / "drive-root")!r})'
        ),
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    af3_parameters = (
        tmp_path
        / "drive-root"
        / "model-assets"
        / "alphafold3"
        / "weights"
        / "af3.bin"
    )
    af3_parameters.parent.mkdir(parents=True)
    af3_parameters.write_bytes(b"private parameters")
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda _path: None)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "local-cache"))
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "GPU_COMPUTE_CAPABILITY": None,
        "SELECTED_MODELS": ("alphafold3",),
        "Path": Path,
        "os": __import__("os"),
        "foldjax_card": lambda *_args, **_kwargs: None,
    }

    exec(source, namespace)

    info = foldjax.model_info("alphafold3")
    assert info.weights_ready is True
    assert info.weights_path == tmp_path / "local-cache" / "weights" / "alphafold3"


@pytest.mark.parametrize(
    (
        "accelerator",
        "gpu_capability",
        "use_fused_triangle_kernels",
        "triangle_kernel",
        "alphafold_attention",
    ),
    (
        ("gpu", (7, 5), False, "xla", "auto"),
        ("gpu", (8, 0), True, "cueq", "auto"),
        ("gpu", None, False, "xla", "auto"),
        ("tpu", None, False, "xla", "xla"),
    ),
)
def test_colab_plans_all_model_kernels_for_the_accelerator(
    accelerator: str,
    gpu_capability: tuple[int, int] | None,
    use_fused_triangle_kernels: bool,
    triangle_kernel: str,
    alphafold_attention: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    models = (
        "protenix",
        "opendde",
        "boltz2",
        "esmfold2",
        "openfold3",
        "alphafold3",
    )

    def fake_resolve(request: PredictionRequest):
        return (request,)

    monkeypatch.setattr(foldjax, "resolve_requests", fake_resolve)
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    namespace = {
        "ACCELERATOR_KIND": accelerator,
        "GPU_COMPUTE_CAPABILITY": gpu_capability,
        "USE_FUSED_TRIANGLE_KERNELS": use_fused_triangle_kernels,
        "SELECTED_MODELS": models,
        "RUN_MODE": "Default",
        "JOB_SLUG": "job",
        "job_path": job_path,
        "MODEL_WEIGHTS": dict.fromkeys(models),
        "OUTPUT_BASE": tmp_path / "outputs",
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "NUM_SEEDS": 1,
        "RESOLVED_EXPLICIT_SEEDS": None,
        "MSA_POLICY": "none",
        "MAX_MSA_DEPTH": 0,
        "CONTINUE_ON_ERROR": True,
        "MODEL_PROFILES": dict.fromkeys(models),
        "REPRESENTATIONS": "None",
        "STOP_AFTER": "Full structure",
        "Path": Path,
        "pd": __import__("pandas"),
        "foldjax_table": lambda *_args, **_kwargs: None,
    }

    exec(_cell_source("plan-runs"), namespace)

    options = namespace["MODEL_OPTIONS"]
    assert options == {
        "protenix": {
            "attention_kernel": "xla",
            "triangle_kernel": triangle_kernel,
        },
        "opendde": {"dtype": "float32", "attention_kernel": "xla"},
        "boltz2": {
            "attention_kernel": "xla",
            "triangle_kernel": triangle_kernel,
        },
        "esmfold2": {},
        "openfold3": {"triangle_kernel": triangle_kernel},
        "alphafold3": {"attention_kernel": alphafold_attention},
    }
    assert namespace["MODEL_PRECISION"]["protenix"] == "native bfloat16"
    assert namespace["MODEL_PRECISION"]["boltz2"] == "native bfloat16"
    assert namespace["MODEL_PRECISION"]["openfold3"] == "float32"


def test_colab_reports_legacy_output_layout_once_without_moving_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(foldjax, "resolve_requests", lambda request: (request,))
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    legacy = tmp_path / "outputs" / "job" / "default" / "openfold3" / "job"
    legacy.mkdir(parents=True)
    marker = legacy / "foldjax_run.json"
    marker.write_text("{}", encoding="utf-8")
    cards: list[tuple] = []
    namespace = {
        "ACCELERATOR_KIND": "tpu",
        "USE_FUSED_TRIANGLE_KERNELS": False,
        "SELECTED_MODELS": ("openfold3",),
        "JOB_SLUG": "job",
        "job_path": job_path,
        "MODEL_WEIGHTS": {"openfold3": None},
        "OUTPUT_BASE": tmp_path / "outputs",
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "MODEL_PROFILES": {"openfold3": None},
        "MSA_POLICY": "none",
        "Path": Path,
        "pd": __import__("pandas"),
        "foldjax_card": lambda *args, **kwargs: cards.append((args, kwargs)),
        "foldjax_table": lambda *_args, **_kwargs: None,
    }

    exec(_cell_source("plan-runs"), namespace)
    exec(_cell_source("plan-runs"), namespace)

    notices = [card for card in cards if card[0][0] == "Earlier output layout found"]
    assert len(notices) == 1
    assert marker.is_file()
    assert namespace["model_requests"][0].output_dir == legacy.parent


@pytest.mark.parametrize(
    ("model_name", "options"),
    (
        (
            "protenix",
            {
                "attention_kernel": "xla",
                "triangle_kernel": "xla",
            },
        ),
        ("opendde", {"dtype": "float32", "attention_kernel": "xla"}),
        (
            "boltz2",
            {
                "attention_kernel": "xla",
                "triangle_kernel": "xla",
            },
        ),
        ("esmfold2", {}),
        ("openfold3", {"triangle_kernel": "xla"}),
        ("alphafold3", {"attention_kernel": "xla"}),
        (
            "protenix",
            {"attention_kernel": "xla", "triangle_kernel": "cueq"},
        ),
        (
            "boltz2",
            {"attention_kernel": "xla", "triangle_kernel": "cueq"},
        ),
        ("openfold3", {"triangle_kernel": "cueq"}),
        ("alphafold3", {"attention_kernel": "auto"}),
    ),
)
def test_colab_model_specific_request_options_are_supported(
    model_name: str, options: dict[str, str], tmp_path: Path
) -> None:
    input_path = tmp_path / "complex.json"
    input_path.write_text(
        '{"name":"complex","entities":[{"type":"protein",'
        '"id":["A"],"sequence":"ACDEFG"}]}',
        encoding="utf-8",
    )
    weights_path = tmp_path / "weights.npz"
    weights_path.write_bytes(b"test")
    request = PredictionRequest(
        model=model_name,
        input=input_path,
        weights=weights_path,
        output_dir=tmp_path / "outputs" / model_name,
        input_format="foldjax",
        cache_dir=tmp_path / "compile-cache",
        seed=0,
        msa="none",
        options=options,
        resume=True,
        num_samples=1,
        num_steps=20,
        num_recycles=1,
    )

    backend = get_backend(model_name)
    backend.validate_request(request)
    native = backend.apply_sampling(request)
    assert {1, 20}.issubset(set(native.values()))


def test_colab_form_builds_mixed_polymer_and_ligand_input(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    build_job = _cell_source("build-job")
    build_job = build_job.replace(
        'PROTEIN_CHAINS = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP'
        'DWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE"',
        'PROTEIN_CHAINS = "ACDEFG"',
    )
    build_job = build_job.replace('DNA_CHAINS = ""', 'DNA_CHAINS = "ACGT:TGCA"')
    build_job = build_job.replace(
        'RNA_CHAINS = "GGGAAACCC"', 'RNA_CHAINS = "ACGU"'
    )
    build_job = build_job.replace(
        'LIGAND_CCD_CODES = "ATP"', 'LIGAND_CCD_CODES = "ATP, MG"'
    )
    build_job = build_job.replace(
        'LIGAND_SMILES = ""', 'LIGAND_SMILES = "CCO; C1=CC=CC=C1"'
    )
    namespace: dict = {"WORK_DIR": tmp_path, "OUTPUT_BASE": tmp_path / "outputs"}
    exec(_cell_source("configure-run"), namespace)
    exec(build_job, namespace)

    payload = json.loads(Path(namespace["job_path"]).read_text(encoding="utf-8"))
    kinds = [entity["type"] for entity in payload["entities"]]
    assert kinds.count("protein") == 1
    assert kinds.count("dna") == 2
    assert kinds.count("rna") == 1
    assert kinds.count("ligand") == 4
    assert namespace["INPUT_ENTITY_TYPES"] == {
        "protein",
        "dna",
        "rna",
        "ligand",
    }


def test_colab_default_is_a_protein_rna_ligand_openfold3_demo(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    namespace: dict = {"WORK_DIR": tmp_path, "OUTPUT_BASE": tmp_path / "outputs"}

    exec(_cell_source("configure-run"), namespace)
    exec(_cell_source("build-job"), namespace)

    payload = json.loads(Path(namespace["job_path"]).read_text(encoding="utf-8"))
    assert [entity["type"] for entity in payload["entities"]] == [
        "protein",
        "rna",
        "ligand",
    ]
    assert payload["entities"][-1]["ccd"] == "ATP"
    assert len(namespace["PROTEIN_SEQUENCES"][0]) == 78
    assert namespace["SELECTED_MODELS"] == (
        "protenix",
        "opendde",
        "boltz2",
        "openfold3",
    )
    assert 'MODEL_WEIGHTS[model_name] = None' in _cell_source("fetch-weights")


def test_colab_default_run_mode_keeps_each_models_native_sampling_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    namespace: dict = {
        "WORK_DIR": tmp_path,
        "OUTPUT_BASE": tmp_path / "outputs",
        "Path": Path,
    }
    exec(_cell_source("configure-run"), namespace)
    namespace.update(
        {
            "ACCELERATOR_KIND": "tpu",
            "USE_FUSED_TRIANGLE_KERNELS": False,
            "COMPILE_CACHE": tmp_path / "compile-cache",
            "MODEL_WEIGHTS": dict.fromkeys(namespace["SELECTED_MODELS"]),
            "MODEL_PROFILES": dict.fromkeys(namespace["SELECTED_MODELS"]),
        }
    )
    exec(_cell_source("build-job"), namespace)
    monkeypatch.setattr(foldjax, "resolve_requests", lambda request: (request,))
    plan = _cell_source("plan-runs")

    exec(plan, namespace)

    assert namespace["RUN_LABEL"] == "default"
    assert namespace["SAMPLING_OVERRIDE"] == {}
    assert namespace["EFFECTIVE_NUM_SEEDS"] == 1
    assert namespace["SEED_SOURCE"] == "generated 0..0"
    assert {row["seed count"] for row in namespace["plan_rows"]} == {1}
    assert {row["seed source"] for row in namespace["plan_rows"]} == {
        "generated 0..0"
    }
    assert {row["schedule"] for row in namespace["plan_rows"]} == {
        "native default (varies by model)"
    }
    for request in namespace["model_requests"]:
        assert request.num_samples is None
        assert request.num_steps is None
        assert request.num_recycles is None
        assert request.padding is not None
        assert request.padding.overflow == "exact"
        assert request.num_seeds == 1
        assert request.seeds is None


def test_colab_custom_schedule_seeds_msa_and_representations(
    tmp_path: Path, monkeypatch
) -> None:
    configure = _cell_source("configure-run")
    plan = _cell_source("plan-runs")
    plan = plan.replace('RUN_MODE = "Default"', 'RUN_MODE = "Custom"')
    plan = plan.replace(
        'CUSTOM_NUM_SAMPLES = 1', 'CUSTOM_NUM_SAMPLES = 2'
    )
    plan = plan.replace('CUSTOM_NUM_STEPS = 100', 'CUSTOM_NUM_STEPS = 60')
    plan = plan.replace(
        'CUSTOM_NUM_RECYCLES = 3', 'CUSTOM_NUM_RECYCLES = 2'
    )
    plan = plan.replace("NUM_SEEDS = 1", "NUM_SEEDS = 0")
    plan = plan.replace('EXPLICIT_SEEDS = ""', 'EXPLICIT_SEEDS = "7, 11"')
    plan = plan.replace('MAX_MSA_DEPTH = 0', 'MAX_MSA_DEPTH = 256')
    plan = plan.replace(
        'REPRESENTATIONS = "None"', 'REPRESENTATIONS = "Single + pair"'
    )
    plan = plan.replace(
        'STOP_AFTER = "Full structure"',
        'STOP_AFTER = "Trunk representations"',
    )
    _patch_ipython_display(monkeypatch)
    namespace = {
        "WORK_DIR": tmp_path,
        "OUTPUT_BASE": tmp_path / "outputs",
        "Path": Path,
    }
    exec(configure, namespace)
    namespace.update(
        {
            "ACCELERATOR_KIND": "tpu",
            "USE_FUSED_TRIANGLE_KERNELS": False,
            "COMPILE_CACHE": tmp_path / "compile-cache",
            "MODEL_WEIGHTS": dict.fromkeys(namespace["SELECTED_MODELS"]),
            "MODEL_PROFILES": dict.fromkeys(namespace["SELECTED_MODELS"]),
        }
    )
    exec(_cell_source("build-job"), namespace)
    monkeypatch.setattr(foldjax, "resolve_requests", lambda request: (request,))

    exec(plan, namespace)

    assert namespace["RUN_LABEL"] == "custom-s2-n60-r2"
    assert namespace["EFFECTIVE_NUM_SEEDS"] == 2
    assert namespace["SEED_SOURCE"] == "explicit values"
    for request in namespace["model_requests"]:
        assert request.resolved_seeds == (7, 11)
        assert request.seeds == (7, 11)
        assert request.num_seeds is None
        assert request.max_msa_depth == 256
        assert request.representations == ("single", "pair")
        assert request.stop_after == "trunk"
        assert request.padding is not None
        assert request.padding.overflow == "exact"
        assert (request.num_samples, request.num_steps, request.num_recycles) == (
            2,
            60,
            2,
        )


def test_colab_upload_job_uses_common_validation_and_supplied_msa(
    tmp_path: Path, monkeypatch
) -> None:
    job_path = tmp_path / "advanced.json"
    job_path.write_text(
        json.dumps(
            {
                "name": "uploaded-complex",
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACDEFG",
                        "unpaired_msa": "/content/query.a3m",
                        "modifications": [{"ccd": "SEP", "position": 2}],
                        "templates": [
                            {
                                "mmcif": "data_template\n#\n",
                                "query_indices": [1, 2],
                                "template_indices": [1, 2],
                            }
                        ],
                    },
                    {"type": "ligand", "id": "B", "ccd": "ATP"},
                ],
                "bonds": [
                    [["A", 2, "OG"], ["B", 1, "PA"]],
                ],
            }
        ),
        encoding="utf-8",
    )
    build = _cell_source("build-job")
    build = build.replace('INPUT_MODE = "Form"', 'INPUT_MODE = "Upload job"')
    build = build.replace('JOB_FILE_PATH = ""', f"JOB_FILE_PATH = {str(job_path)!r}")
    tables = []
    _patch_ipython_display(monkeypatch)
    namespace = {
        "WORK_DIR": tmp_path,
        "OUTPUT_BASE": tmp_path / "outputs",
        "Path": Path,
    }
    exec(_cell_source("configure-run"), namespace)
    namespace["SELECTED_MODELS"] = ("protenix",)
    namespace["foldjax_table"] = lambda frame, *_args, **_kwargs: tables.append(
        frame
    )

    exec(build, namespace)

    assert namespace["job_path"] == job_path
    assert namespace["JOB_SLUG"] == "uploaded-complex"
    assert namespace["INPUT_REQUIRED_FEATURES"] >= {
        "unpaired_msa",
        "modifications",
        "templates",
        "ligand_ccd",
        "bonds",
    }
    preflight = next(frame for frame in tables if "action" in frame.columns)
    protein = preflight.loc[preflight["entity"] == "protein 1"].iloc[0]
    assert protein["cache"] == "supplied"
    assert "no service contact" in protein["action"]


@pytest.mark.parametrize(
    ("cached", "expected_cache", "expected_action"),
    (
        (True, "hit", "no service contact"),
        (False, "miss", "remote search"),
    ),
)
def test_colab_msa_preflight_reports_cache_reuse_without_searching(
    cached: bool,
    expected_cache: str,
    expected_action: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    tables = []

    class StubPipeline:
        cache_dir = tmp_path / "msa"

        def _identity(self, _sequence):
            return "cache-key", {}

        def _cached(self, _directory, _sequence, _cache_key):
            return {"unpairedMsaPath": "cached.a3m"} if cached else None

    monkeypatch.setattr("foldjax.input._msa_pipeline", StubPipeline)
    monkeypatch.setattr("foldjax.input._rna_msa_pipeline", lambda: None)
    monkeypatch.setattr(
        "foldjax.input.msa_search_backend",
        lambda: {
            "protein": {"kind": "remote", "host": "https://msa.invalid"},
            "rna": {"kind": "unavailable"},
        },
    )
    namespace = {"WORK_DIR": tmp_path, "OUTPUT_BASE": tmp_path / "outputs"}
    _patch_ipython_display(monkeypatch)
    exec(_cell_source("configure-run"), namespace)
    namespace["foldjax_table"] = lambda frame, *_args, **_kwargs: tables.append(frame)

    build_job = _cell_source("build-job")
    exec(build_job, namespace)

    preflight = next(frame for frame in tables if "action" in frame.columns)
    protein = preflight.loc[preflight["entity"] == "protein 1"].iloc[0]
    rna = preflight.loc[preflight["entity"] == "RNA 1"].iloc[0]
    assert protein["cache"] == expected_cache
    assert expected_action in protein["action"]
    assert rna["cache"] == "unavailable"
    assert "no public RNA search" in rna["action"]


def test_colab_msa_preflight_rejects_required_rna_without_a_backend(
    tmp_path: Path, monkeypatch
) -> None:
    pipeline = SimpleNamespace(
        cache_dir=tmp_path / "msa",
        _identity=lambda _sequence: ("cache-key", {}),
        _cached=lambda *_args: {"unpairedMsaPath": "cached.a3m"},
    )
    monkeypatch.setattr("foldjax.input._msa_pipeline", lambda: pipeline)
    monkeypatch.setattr("foldjax.input._rna_msa_pipeline", lambda: None)
    monkeypatch.setattr(
        "foldjax.input.msa_search_backend",
        lambda: {
            "protein": {"kind": "remote", "host": "https://msa.invalid"},
            "rna": {"kind": "unavailable"},
        },
    )
    namespace = {"WORK_DIR": tmp_path, "OUTPUT_BASE": tmp_path / "outputs"}
    _patch_ipython_display(monkeypatch)
    exec(_cell_source("configure-run"), namespace)
    build_job = _cell_source("build-job").replace(
        'MSA_POLICY = "auto"', 'MSA_POLICY = "required"'
    )

    with pytest.raises(RuntimeError, match="RNA MSA is required"):
        exec(build_job, namespace)


def test_colab_form_rejects_invalid_nucleic_acid_after_setup(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    build_job = _cell_source("build-job").replace(
        'DNA_CHAINS = ""', 'DNA_CHAINS = "ACGU"'
    )
    namespace = {"WORK_DIR": tmp_path, "OUTPUT_BASE": tmp_path / "outputs"}
    exec(_cell_source("configure-run"), namespace)

    with pytest.raises(ValueError, match="DNA chain 1.*base 'U'"):
        exec(build_job, namespace)


def _fake_model_info(
    model: str,
    *,
    entity_types: tuple[str, ...],
    features: tuple[str, ...] = (),
    fetchable: bool,
    ready: bool,
    weights_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        capabilities=SimpleNamespace(
            entity_types=entity_types,
            common_schema_features=features,
        ),
        weights_fetchable=fetchable,
        weights_ready=ready,
        weights_path=weights_path,
        download_bytes=1_000_000 if fetchable else None,
        weights_licence="test licence",
        weights_source="test source",
        notes="test setup note",
        runtime=SimpleNamespace(ready=True),
    )


def _patch_ipython_display(monkeypatch) -> None:
    ipython = ModuleType("IPython")
    ipython.__path__ = []  # type: ignore[attr-defined]
    ipython_display = ModuleType("IPython.display")
    ipython_display.HTML = lambda value: value  # type: ignore[attr-defined]
    ipython_display.display = lambda *_args: None  # type: ignore[attr-defined]
    ipython.display = ipython_display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)


def _notebook_ui(*, tables: list | None = None) -> dict:
    def render_table(frame, *_args, **_kwargs) -> None:
        if tables is not None:
            tables.append(frame)

    return {
        "foldjax_card": lambda *_args, **_kwargs: None,
        "foldjax_table": render_table,
    }


def _model_directory_map(root: Path, models: tuple[str, ...]) -> dict:
    return {
        model: {
            component: root / component / model
            for component in ("downloads", "weights", "runtime")
        }
        for model in models
    }


def test_colab_storage_error_reports_target_free_space_and_requirement(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    info = _fake_model_info(
        "protenix",
        entity_types=("protein",),
        fetchable=True,
        ready=False,
        weights_path=model_store / "weights" / "protenix",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    namespace = {
        "SELECTED_MODELS": ("protenix",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "foldjax_home": tmp_path,
        "MODEL_ASSET_STORE": model_store,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("protenix",)),
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": True,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=1_000_000_000)
        ),
        "subprocess": SimpleNamespace(run=lambda *_args, **_kwargs: None),
        "sys": sys,
        **_notebook_ui(),
    }

    with pytest.raises(RuntimeError) as error:
        exec(_cell_source("fetch-weights"), namespace)

    message = str(error.value)
    assert f"Model storage {model_store}" in message
    assert "has 1.0 GB free" in message
    assert "requires about 8.0 GB" in message
    assert "select storage with more free space" in message


def test_colab_weight_fetch_reports_the_actual_failed_stage_and_message(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "boltz2",
        entity_types=("protein", "rna", "ligand"),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "weights" / "boltz2",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    cards = []

    def record_card(title, message, **kwargs):
        cards.append((title, message, kwargs))

    namespace = {
        "SELECTED_MODELS": ("boltz2",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "foldjax_home": tmp_path,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("boltz2",)),
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": SimpleNamespace(
            run=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1,
                stdout="boltz2: publisher files\n",
                stderr=(
                    "[weights] boltz2/released download error mols.tar: "
                    "download or verification failed\n"
                    "mols.tar arrived incomplete: connection dropped\n"
                ),
            )
        ),
        "sys": sys,
        "foldjax_card": record_card,
        "foldjax_table": lambda *_args, **_kwargs: None,
    }

    with pytest.raises(RuntimeError) as error:
        exec(_cell_source("fetch-weights"), namespace)

    message = str(error.value)
    assert "boltz2 weight download, verification, or conversion failed" in message
    assert "mols.tar arrived incomplete: connection dropped" in message
    assert cards[-1][0] == "boltz2 setup failed"
    metrics = dict(cards[-1][2]["metrics"])
    assert metrics["stage"] == "weight download, verification, or conversion"
    assert metrics["exit status"] == 1
    assert "mols.tar arrived incomplete" in metrics["last message"]


def test_colab_weight_fetch_updates_one_live_progress_card(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "boltz2",
        entity_types=("protein",),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "weights" / "boltz2",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(
                (
                    "[weights] boltz2/released download start: fetching\n",
                    "[foldjax-progress]\tboltz2_conf.ckpt\t20971520\t104857600\n",
                    "[foldjax-progress]\tboltz2_conf.ckpt\t104857600\t104857600\n",
                    "[weights] boltz2/released convert done: ready\n",
                )
            )

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def wait():
            return 0

    class Progress:
        instances = []

        def __init__(self, title, message):
            self.title = title
            self.updates = [(message, {})]
            self.succeeded = False
            self.__class__.instances.append(self)

        def update(self, message, **kwargs):
            self.updates.append((message, kwargs))

        def succeed(self, message, **kwargs):
            self.succeeded = True
            self.updates.append((message, kwargs))

        def fail(self, message, **kwargs):
            pytest.fail(f"unexpected progress failure: {message} {kwargs}")

    fake_subprocess = SimpleNamespace(
        PIPE=object(),
        STDOUT=object(),
        Popen=lambda *_args, **_kwargs: FakeProcess(),
        run=lambda *_args, **_kwargs: pytest.fail("used non-live subprocess path"),
    )
    namespace = {
        "SELECTED_MODELS": ("boltz2",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "foldjax_home": tmp_path,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("boltz2",)),
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": fake_subprocess,
        "threading": __import__("threading"),
        "os": __import__("os"),
        "sys": sys,
        "FoldJAXProgress": Progress,
        **_notebook_ui(),
    }

    exec(_cell_source("fetch-weights"), namespace)

    progress = Progress.instances[0]
    assert progress.title == "Preparing boltz2"
    assert progress.succeeded is True
    determinate = [
        kwargs
        for _message, kwargs in progress.updates
        if kwargs.get("total") == 104857600
    ]
    assert any(item["current"] == 20971520 for item in determinate)
    assert any(item["current"] == 104857600 for item in determinate)
    first_transfer_metrics = dict(determinate[0]["metrics"])
    assert first_transfer_metrics["downloaded"] == "20.0 MiB"


def test_colab_blocks_released_esmfold2_on_a_16_gib_device(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "esmfold2",
        entity_types=("protein", "dna", "rna", "ligand"),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "weights" / "esmfold2",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    namespace = {
        "SELECTED_MODELS": ("esmfold2",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "DEVICE_MEMORY_BYTES": 16 * 2**30,
        "foldjax_home": tmp_path,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("esmfold2",)),
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": SimpleNamespace(run=lambda *_args, **_kwargs: None),
        "sys": sys,
        **_notebook_ui(),
    }

    with pytest.raises(RuntimeError, match="ESMFold2 Released loads ESMC-6B"):
        exec(_cell_source("fetch-weights"), namespace)


def test_colab_allows_released_esmfold2_above_16_gib(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "esmfold2",
        entity_types=("protein", "dna", "rna", "ligand"),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "weights" / "esmfold2",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    namespace = {
        "SELECTED_MODELS": ("esmfold2",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "DEVICE_MEMORY_BYTES": 24 * 2**30,
        "foldjax_home": tmp_path,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("esmfold2",)),
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": SimpleNamespace(
            run=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
        "sys": sys,
        **_notebook_ui(),
    }

    exec(_cell_source("fetch-weights"), namespace)

    assert calls[0][0][0][-2:] == ["--model", "esmfold2"]


def test_colab_accepts_esmfold2_all_biomolecule_input(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "esmfold2",
        entity_types=("protein", "dna", "rna", "ligand"),
        features=(
            "unpaired_msa",
            "modifications",
            "ligand_ccd",
            "ligand_smiles",
            "bonds",
        ),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "esmfold2.safetensors",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    exec(
        _cell_source("fetch-weights"),
        {
            "SELECTED_MODELS": ("esmfold2",),
            "INPUT_ENTITY_TYPES": frozenset({"protein", "dna", "rna", "ligand"}),
            "INPUT_REQUIRED_FEATURES": frozenset(
                {"modifications", "ligand_ccd", "bonds"}
            ),
            "MODEL_PROFILES": {"esmfold2": None},
            "foldjax_home": tmp_path,
            "COMPILE_CACHE": tmp_path / "compile-cache",
            "OUTPUT_BASE": tmp_path / "outputs",
            "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
            "PERSIST_MSA_TO_DRIVE": False,
            "PERSIST_OUTPUTS_TO_DRIVE": False,
            "MODEL_ASSET_STORE": tmp_path,
            "MODEL_DIRECTORY_MAP": _model_directory_map(
                tmp_path, ("esmfold2",)
            ),
            "Path": Path,
            "display": lambda *_args: None,
            "shutil": SimpleNamespace(
                disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
            ),
            "subprocess": SimpleNamespace(
                run=lambda *args, **kwargs: calls.append((args, kwargs))
            ),
            "sys": sys,
            **_notebook_ui(),
        },
    )

    assert len(calls) == 1
    assert calls[0][0][0][-2:] == ["--model", "esmfold2"]


def test_colab_fetches_public_openfold3_p1_automatically(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    _patch_ipython_display(monkeypatch)
    info = _fake_model_info(
        "openfold3",
        entity_types=("protein", "dna", "rna", "ligand"),
        features=("ligand_ccd", "ligand_smiles"),
        fetchable=True,
        ready=False,
        weights_path=tmp_path / "managed" / "of3_ft3_v1.pt",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    namespace = {
        "SELECTED_MODELS": ("openfold3",),
        "INPUT_ENTITY_TYPES": frozenset({"protein", "ligand"}),
        "INPUT_REQUIRED_FEATURES": frozenset({"ligand_smiles"}),
        "MODEL_PROFILES": {"openfold3": None},
        "foldjax_home": tmp_path,
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("openfold3",)),
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": SimpleNamespace(
            run=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
        "sys": sys,
        **_notebook_ui(),
    }

    exec(_cell_source("fetch-weights"), namespace)

    assert namespace["MODEL_WEIGHTS"] == {"openfold3": None}
    assert len(calls) == 1
    assert calls[0][0][0][-2:] == ["--model", "openfold3"]


def test_colab_reuses_cached_openfold3_p1_without_a_fetch_process(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    tables = []
    _patch_ipython_display(monkeypatch)
    msa_entry = tmp_path / "msa" / "test-entry"
    msa_entry.mkdir(parents=True)
    (msa_entry / "non_pairing.a3m").write_text(">query\nAAAA\n", encoding="utf-8")
    info = _fake_model_info(
        "openfold3",
        entity_types=("protein", "dna", "rna", "ligand"),
        features=("ligand_ccd", "ligand_smiles"),
        fetchable=True,
        ready=True,
        weights_path=tmp_path / "managed" / "of3_ft3_v1.pt",
    )
    monkeypatch.setattr(foldjax, "model_info", lambda _model: info)
    namespace = {
        "SELECTED_MODELS": ("openfold3",),
        "INPUT_ENTITY_TYPES": frozenset({"protein"}),
        "INPUT_REQUIRED_FEATURES": frozenset(),
        "MODEL_PROFILES": {"openfold3": None},
        "foldjax_home": tmp_path,
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "OUTPUT_BASE": tmp_path / "outputs",
        "PERSIST_MODEL_ASSETS_TO_DRIVE": False,
        "PERSIST_MSA_TO_DRIVE": False,
        "PERSIST_OUTPUTS_TO_DRIVE": False,
        "MODEL_ASSET_STORE": tmp_path,
        "MODEL_DIRECTORY_MAP": _model_directory_map(tmp_path, ("openfold3",)),
        "Path": Path,
        "display": lambda *_args: None,
        "shutil": SimpleNamespace(
            disk_usage=lambda _path: SimpleNamespace(free=100_000_000_000)
        ),
        "subprocess": SimpleNamespace(
            run=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
        "sys": sys,
        **_notebook_ui(tables=tables),
    }

    exec(_cell_source("fetch-weights"), namespace)

    assert namespace["MODEL_WEIGHTS"] == {"openfold3": None}
    assert calls == []
    inventory = next(frame for frame in tables if "component" in frame.columns)
    assert set(inventory["component"]) >= {
        "openfold3 · weights",
        "MSA",
        "predictions",
    }
    msa_row = inventory.loc[inventory["component"] == "MSA"].iloc[0]
    assert msa_row["location"] == "runtime"
    assert msa_row["files"] == 1


def test_colab_prediction_and_output_cells_execute_together(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[PredictionRequest] = []
    synced: list[str] = []

    def fake_resolve(request: PredictionRequest):
        assert request.model in ("protenix", "opendde", "boltz2", "openfold3")
        return (request,)

    def fake_predict_batch(request: PredictionRequest) -> BatchReport:
        seen.append(request)
        model = request.model
        assert model is not None
        index = ("protenix", "opendde", "boltz2", "openfold3").index(model)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True)
        structure = output_dir / f"{model}.cif"
        structure.write_text(
            f"data_{model}\n_entry.id {model}\n#\n",
            encoding="utf-8",
        )
        (output_dir / "confidence.json").write_text(
            json.dumps({"ranking_score": 0.8 - index * 0.1}),
            encoding="utf-8",
        )
        (output_dir / "foldjax_run.json").write_text(
            json.dumps({"schema": 1, "model": model}),
            encoding="utf-8",
        )
        return BatchReport(
            results=(
                PredictionResult(
                    model=model,
                    samples=(
                        PredictionSample(
                            seed=0,
                            structure_path=structure,
                            scores={"ranking_score": 0.8 - index * 0.1},
                        ),
                    ),
                    output_dir=output_dir,
                ),
            )
        )

    monkeypatch.setattr(foldjax, "resolve_requests", fake_resolve)
    monkeypatch.setattr(foldjax, "predict_batch", fake_predict_batch)
    monkeypatch.setattr(foldjax.progress, "enable", lambda: None)
    _patch_ipython_display(monkeypatch)

    namespace = {
        "Path": Path,
        "display": lambda *_args: None,
        "pd": __import__("pandas"),
    }
    exec(_cell_source("configure-run"), namespace)
    namespace.update(
        {
            "WORK_DIR": tmp_path,
            "OUTPUT_BASE": tmp_path / "persistent-outputs",
            "COMPILE_CACHE": tmp_path / "compile-cache",
            "MODEL_WEIGHTS": {
                "protenix": None,
                "opendde": None,
                "boltz2": None,
                "openfold3": None,
            },
            "MODEL_PROFILES": {
                "protenix": None,
                "opendde": None,
                "boltz2": None,
                "openfold3": None,
            },
            "ACCELERATOR_KIND": "tpu",
            "USE_FUSED_TRIANGLE_KERNELS": False,
            "sync_compile_cache_archive": lambda model: (
                synced.append(model)
                or {"model": model, "status": "saved", "bytes": 1}
            ),
        }
    )
    exec(_cell_source("build-job"), namespace)
    exec(_cell_source("plan-runs"), namespace)
    exec(_cell_source("run-predictions"), namespace)
    exec(_cell_source("compare-results"), namespace)

    assert tuple(request.model for request in seen) == (
        "protenix",
        "opendde",
        "boltz2",
        "openfold3",
    )
    assert tuple(synced) == tuple(request.model for request in seen)
    assert seen[0].options == {
        "attention_kernel": "xla",
        "triangle_kernel": "xla",
    }
    assert seen[1].options == {
        "dtype": "float32",
        "attention_kernel": "xla",
    }
    assert seen[2].options == {
        "attention_kernel": "xla",
        "triangle_kernel": "xla",
    }
    assert seen[3].options == {"triangle_kernel": "xla"}
    assert all(
        (request.num_samples, request.num_steps, request.num_recycles)
        == (None, None, None)
        for request in seen
    )
    assert all(request.num_seeds == 1 for request in seen)
    assert all(request.msa == "auto" for request in seen)
    assert tuple(Path(request.output_dir) for request in seen) == tuple(
        tmp_path
        / "persistent-outputs"
        / "protein-rna-atp-demo"
        / "default"
        / model
        for model in ("protenix", "opendde", "boltz2", "openfold3")
    )
    assert tuple(namespace["comparison"]["model"]) == (
        "boltz2",
        "opendde",
        "openfold3",
        "protenix",
    )
    assert len(namespace["STRUCTURES"]) == 4

    viewer_calls: list[tuple] = []

    class FakeDropdown:
        def __init__(self, *, options, **_kwargs):
            self.options = tuple(options)
            self.value = self.options[0]
            self.observers = []

        def observe(self, callback, **_kwargs):
            self.observers.append(callback)

    class FakeOutput:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    ipywidgets = ModuleType("ipywidgets")
    ipywidgets.Dropdown = FakeDropdown  # type: ignore[attr-defined]
    ipywidgets.Layout = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    ipywidgets.Output = FakeOutput  # type: ignore[attr-defined]
    ipywidgets.VBox = lambda children: tuple(children)  # type: ignore[attr-defined]

    py3dmol = ModuleType("py3Dmol")
    viewer = SimpleNamespace(
        addModel=lambda text, structure_format: viewer_calls.append(
            ("model", text, structure_format)
        ),
        setStyle=lambda style: viewer_calls.append(("style", style)),
        setBackgroundColor=lambda color: viewer_calls.append(("background", color)),
        zoomTo=lambda: viewer_calls.append(("zoom",)),
        show=lambda: viewer_calls.append(("show",)),
    )
    py3dmol.view = lambda **_kwargs: viewer  # type: ignore[attr-defined]
    ipython = ModuleType("IPython")
    ipython.__path__ = []  # type: ignore[attr-defined]
    ipython_display = ModuleType("IPython.display")
    ipython_display.clear_output = lambda **_kwargs: None  # type: ignore[attr-defined]
    ipython.display = ipython_display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setitem(sys.modules, "ipywidgets", ipywidgets)
    monkeypatch.setitem(sys.modules, "py3Dmol", py3dmol)
    exec(_cell_source("view-structures"), namespace)

    assert namespace["prediction_selector"].options == tuple(namespace["STRUCTURES"])
    assert viewer_calls[0][0] == "model"
    assert viewer_calls[0][2] == "cif"
    assert viewer_calls[-1] == ("show",)

    downloads: list[str] = []
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=downloads.append)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    exec(_cell_source("download-results"), namespace)

    archive = Path(downloads[0])
    assert archive.is_file()
    assert archive.name == "protein-rna-atp-demo-default-foldjax.zip"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert {
        "input/job.json",
        "comparison.csv",
        "batch_report.json",
        "outputs/protenix/protenix.cif",
        "outputs/opendde/opendde.cif",
        "outputs/boltz2/boltz2.cif",
        "outputs/openfold3/openfold3.cif",
    }.issubset(names)


def test_colab_run_cell_surfaces_model_failures(
    tmp_path: Path, monkeypatch
) -> None:
    failure = PredictionFailure(
        model="opendde",
        input=tmp_path / "input.json",
        output_dir=tmp_path / "outputs" / "opendde",
        seed=0,
        error_type="MemoryError",
        error="out of memory",
    )
    displayed = []
    monkeypatch.setattr(
        foldjax,
        "predict_batch",
        lambda _request: BatchReport(failures=(failure,)),
    )
    monkeypatch.setattr(foldjax.progress, "enable", lambda: None)

    exec(
        _cell_source("run-predictions"),
        {
            "model_requests": (
                SimpleNamespace(model="opendde", resolved_seeds=(0,), msa="none"),
            ),
            "PREFLIGHT_ONLY": False,
            "display": displayed.append,
            "pd": __import__("pandas"),
            **_notebook_ui(tables=displayed),
        },
    )

    assert len(displayed) == 1
    assert displayed[0].iloc[0]["model"] == "opendde"
    assert displayed[0].iloc[0]["error_type"] == "MemoryError"


def test_colab_preflight_only_resolves_without_predicting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        foldjax,
        "predict_batch",
        lambda _request: pytest.fail("preflight must not launch inference"),
    )
    namespace = {
        "model_requests": (
            SimpleNamespace(model="protenix", resolved_seeds=(0,), msa="none"),
        ),
        "PREFLIGHT_ONLY": True,
        "plan_rows": [{"model": "protenix", "seeds": [0]}],
        "STOP_POINT": "full",
        "Path": Path,
        "pd": __import__("pandas"),
        **_notebook_ui(),
    }

    exec(_cell_source("run-predictions"), namespace)
    exec(_cell_source("compare-results"), namespace)

    assert namespace["report"] == BatchReport()
    assert namespace["STRUCTURES"] == {}
    assert tuple(namespace["comparison"]["model"]) == ("protenix",)


def test_colab_trunk_mode_reports_representations_without_structures() -> None:
    representations = SimpleNamespace(
        summary=lambda: {
            "names": ["single", "pair"],
            "path": "/content/results/representations.npz",
        }
    )
    namespace = {
        "report": BatchReport(
            results=(
                PredictionResult(
                    model="protenix",
                    representations=representations,
                ),
            )
        ),
        "PREFLIGHT_ONLY": False,
        "STOP_POINT": "trunk",
        "Path": Path,
        "pd": __import__("pandas"),
        **_notebook_ui(),
    }

    exec(_cell_source("compare-results"), namespace)

    assert namespace["STRUCTURES"] == {}
    assert namespace["comparison"].iloc[0]["representations"] == "single, pair"


def test_readme_links_to_the_colab_workflow_and_the_doc_explains_it() -> None:
    """The README was cut to a front page; the explanation moved, not away.

    Both halves are checked here because a reader reaches the notebook through
    one of two doors -- the badge at the top of the README, or its `Try it`
    section -- and what the notebook actually does now lives in `docs/colab.md`.
    A README that keeps its links while the doc loses the behaviour, or a doc
    that keeps the behaviour while the README stops pointing at it, are the two
    ways this can quietly stop being true.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    colab_doc = (ROOT / "docs" / "colab.md").read_text(encoding="utf-8")
    colab_prose = " ".join(colab_doc.split())

    assert (
        "https://colab.research.google.com/github/eightmm/FoldJAX/blob/"
        "main/notebooks/FoldJAX_Colab.ipynb"
    ) in readme
    assert "[Colab notebook](notebooks/FoldJAX_Colab.ipynb)" in readme
    assert "[docs/colab.md](docs/colab.md)" in readme
    # The one worked example the front page keeps is the multi-model one:
    # running several models over a single input is what FoldJAX does that no
    # publisher's own code does.
    assert "predict_batch" in readme
    assert 'models=("protenix", "opendde", "openfold3")' in readme

    assert "[Google Colab workflow](../notebooks/FoldJAX_Colab.ipynb)" in colab_doc
    assert "protein, DNA, RNA, CCD ligands, and SMILES ligands" in colab_prose
    assert "ESMFold2 accepts the same protein, DNA, RNA," in colab_prose
    assert "OpenFold3" in colab_doc
    # The notebook's short schedule is not any model's released default, and
    # saying so is the whole reason this document exists.
    assert "tutorial schedule" in colab_prose
