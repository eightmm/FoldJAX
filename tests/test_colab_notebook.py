from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
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

    assert 'FOLDJAX_REF = "52f8c8890956acedac5e293a6bddc2e8b5449ba4"' in install
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
    assert "PROTEIN_CHAINS" in source
    assert "DNA_CHAINS" in source
    assert "RNA_CHAINS" in source
    assert "LIGAND_CCD_CODES" in source
    assert "LIGAND_SMILES" in source
    assert 'compact.split(":")' in source
    assert 'LIGAND_CCD_CODES.split(",")' in source
    assert 'LIGAND_SMILES.split(";")' in source
    assert "PERSIST_WEIGHTS_TO_DRIVE = False" in source
    assert 'OUTPUT_ROOT = WORK_DIR / "outputs" / JOB_SLUG / RUN_LABEL' in source
    assert 'RUN_MODE = "Fast demo"' in source
    assert 'MSA_POLICY = "none"' in source
    assert "job_path.read_text()" not in source
    assert '{"num_samples": 1, "num_steps": 20, "num_recycles": 1}' in source
    assert "protein=PROTEIN_SEQUENCES" in source
    assert "dna=DNA_SEQUENCES" in source
    assert "rna=RNA_SEQUENCES" in source
    assert "ligand_ccd=LIGAND_CCDS" in source
    assert "ligand_smiles=LIGAND_SMILES_VALUES" in source
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
    assert "ALPHAFOLD3_WEIGHTS_PATH" in source
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
    ("accelerator", "multiplication_backend", "attention_backend", "profile"),
    (
        ("gpu", "cueq", "cueq_jit", "CUDA fused"),
        ("tpu", "xla", "xla_jit", "portable XLA"),
    ),
)
def test_colab_configures_accelerator_specific_kernels(
    accelerator: str,
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
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    namespace = {
        "ACCELERATOR_KIND": accelerator,
        "JOB_SLUG": "job",
        "RUN_LABEL": "fast-demo",
        "PERSIST_WEIGHTS_TO_DRIVE": False,
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


@pytest.mark.parametrize(
    ("accelerator", "triangle_kernel", "alphafold_attention"),
    (
        ("gpu", "cueq", "auto"),
        ("tpu", "xla", "xla"),
    ),
)
def test_colab_plans_all_model_kernels_for_the_accelerator(
    accelerator: str,
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
        return (
            SimpleNamespace(
                model=request.model,
                input=request.input,
                output_dir=request.output_dir,
                resolved_seeds=(0,),
                options=request.options,
                sampling=request.sampling,
            ),
    )

    monkeypatch.setattr(foldjax, "resolve_requests", fake_resolve)
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    namespace = {
        "ACCELERATOR_KIND": accelerator,
        "SELECTED_MODELS": models,
        "JOB_SLUG": "job",
        "job_path": job_path,
        "MODEL_WEIGHTS": dict.fromkeys(models),
        "OUTPUT_ROOT": tmp_path / "outputs",
        "COMPILE_CACHE": tmp_path / "compile-cache",
        "NUM_SEEDS": 1,
        "MSA_POLICY": "none",
        "CONTINUE_ON_ERROR": True,
        "FAST_SCHEDULE": {},
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
    namespace: dict = {"WORK_DIR": tmp_path}
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
    namespace: dict = {"WORK_DIR": tmp_path}

    exec(_cell_source("configure-run"), namespace)
    exec(_cell_source("build-job"), namespace)

    payload = json.loads(Path(namespace["job_path"]).read_text(encoding="utf-8"))
    assert [entity["type"] for entity in payload["entities"]] == [
        "protein",
        "rna",
        "ligand",
    ]
    assert namespace["LIGAND_CCDS"] == ("ATP",)
    assert len(namespace["PROTEIN_SEQUENCES"][0]) == 78
    assert namespace["SELECTED_MODELS"] == (
        "protenix",
        "opendde",
        "openfold3",
    )
    assert "openfold3" not in namespace["MANUAL_WEIGHT_TEXT"]


def test_colab_form_rejects_invalid_nucleic_acid_after_setup(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_ipython_display(monkeypatch)
    build_job = _cell_source("build-job").replace(
        'DNA_CHAINS = ""', 'DNA_CHAINS = "ACGU"'
    )
    namespace = {"WORK_DIR": tmp_path}
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
            "MANUAL_WEIGHT_TEXT": {},
            "foldjax_home": tmp_path,
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
        "MANUAL_WEIGHT_TEXT": {},
        "foldjax_home": tmp_path,
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
    _patch_ipython_display(monkeypatch)
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
        "MANUAL_WEIGHT_TEXT": {},
        "foldjax_home": tmp_path,
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
    assert calls == []


def test_colab_prediction_and_output_cells_execute_together(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[PredictionRequest] = []

    def fake_resolve(request: PredictionRequest):
        assert request.model in ("protenix", "opendde", "openfold3")
        return (
            SimpleNamespace(
                model=request.model,
                input=request.input,
                output_dir=request.output_dir,
                resolved_seeds=(0,),
                options=request.options,
                sampling=request.sampling,
            ),
        )

    def fake_predict_batch(request: PredictionRequest) -> BatchReport:
        seen.append(request)
        model = request.model
        assert model is not None
        index = ("protenix", "opendde", "openfold3").index(model)
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
            "OUTPUT_ROOT": tmp_path / "outputs",
            "COMPILE_CACHE": tmp_path / "compile-cache",
            "MODEL_WEIGHTS": {
                "protenix": None,
                "opendde": None,
                "openfold3": None,
            },
            "ACCELERATOR_KIND": "tpu",
        }
    )
    exec(_cell_source("build-job"), namespace)
    exec(_cell_source("plan-runs"), namespace)
    exec(_cell_source("run-predictions"), namespace)
    exec(_cell_source("compare-results"), namespace)

    assert tuple(request.model for request in seen) == (
        "protenix",
        "opendde",
        "openfold3",
    )
    assert seen[0].options == {
        "attention_kernel": "xla",
        "triangle_kernel": "xla",
    }
    assert seen[1].options == {
        "dtype": "float32",
        "attention_kernel": "xla",
    }
    assert seen[2].options == {"triangle_kernel": "xla"}
    assert all(
        (request.num_samples, request.num_steps, request.num_recycles) == (1, 20, 1)
        for request in seen
    )
    assert all(request.num_seeds == 1 for request in seen)
    assert all(request.msa == "none" for request in seen)
    assert tuple(namespace["comparison"]["model"]) == (
        "opendde",
        "openfold3",
        "protenix",
    )
    assert len(namespace["STRUCTURES"]) == 3

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
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert {
        "input/job.json",
        "comparison.csv",
        "batch_report.json",
        "outputs/protenix/protein-rna-atp-demo/protenix.cif",
        "outputs/opendde/protein-rna-atp-demo/opendde.cif",
        "outputs/openfold3/protein-rna-atp-demo/openfold3.cif",
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
                SimpleNamespace(model="opendde", num_seeds=1, msa="none"),
            ),
            "display": displayed.append,
            "pd": __import__("pandas"),
            **_notebook_ui(tables=displayed),
        },
    )

    assert len(displayed) == 1
    assert displayed[0].iloc[0]["model"] == "opendde"
    assert displayed[0].iloc[0]["error_type"] == "MemoryError"


def test_readme_links_to_and_explains_the_colab_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    prose = " ".join(readme.split())

    assert "## Examples" in readme
    assert "### Google Colab: compare models from one input" in readme
    assert (
        "https://colab.research.google.com/github/eightmm/FoldJAX/blob/"
        "main/notebooks/FoldJAX_Colab.ipynb"
    ) in readme
    assert "[Google Colab workflow](notebooks/FoldJAX_Colab.ipynb)" in readme
    assert 'models=("protenix", "opendde", "openfold3")' in readme
    assert "predict_batch" in readme
    assert "protein, DNA, RNA, CCD ligands, and SMILES ligands" in prose
    assert "is an all-biomolecule model" in readme
    assert "FoldJAX ports that input contract without Torch" in prose
    assert "OpenFold3" in readme
