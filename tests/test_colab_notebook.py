from __future__ import annotations

import ast
import json
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
COLAB_STACK = {
    "cuequivariance": "0.11.1",
    "cuequivariance-jax": "0.11.1",
    "cuequivariance-ops-cu12": "0.11.1",
    "cuequivariance-ops-jax-cu12": "0.11.1",
    "flax": "0.12.9",
    "jax-cuda12-pjrt": "0.11.1",
    "jax-cuda12-plugin": "0.11.1",
    "jaxlib": "0.11.1",
    "qwix": "0.1.8",
    "tokamax": "0.0.13",
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


def test_colab_notebook_is_clean_gpu_workflow() -> None:
    document = _document()

    assert document["nbformat"] == 4
    assert document["nbformat_minor"] >= 5
    assert document["metadata"]["accelerator"] == "GPU"
    assert document["metadata"]["language_info"]["version"] == "3.13"
    assert document["cells"][0]["cell_type"] == "markdown"
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


def test_colab_notebook_installs_supported_cuda_runtime_before_imports() -> None:
    install = _cell_source("install-foldjax")
    all_source = "\n".join(_source(cell) for cell in _code_cells())

    assert 'FOLDJAX_REF = "613c0fa99b3838db1a9ea02db35735056ce71e96"' in install
    assert "foldjax[cuda12]" in install
    assert "FoldJAX.git@{FOLDJAX_REF}" in install
    assert '"py3Dmol>=2.0,<3"' in install
    assert all_source.index('"pip"') < all_source.index("import jax")
    assert 'sys.version_info[:2] != (3, 13)' in install
    assert 'install_marker.unlink(missing_ok=True)' in install
    assert 'shutil.which("nvidia-smi")' in install
    assert "Dependencies installed; restarting the runtime once" in install
    assert "signal.SIGKILL" in install
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in all_source
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND" in all_source
    assert "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND" in all_source
    assert 'jax.__version__ != "0.11.1"' in all_source
    assert "metadata.PackageNotFoundError" in all_source
    for package, version in COLAB_STACK.items():
        assert f'"{package}": "{version}"' in all_source
    assert 'device.platform == "gpu"' in all_source


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
        exec(_cell_source("install-foldjax"), {})

    message = str(error.value)
    assert "Python 3.13" in message
    assert "3.12.12" in message
    assert "Runtime → Change runtime type → Runtime version" in message
    assert "reconnect" in message
    assert calls == []


def test_colab_runtime_matches_the_root_project_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    assert "jax==0.11.1" in project["project"]["dependencies"]


def test_colab_gpu_check_executes_with_the_pinned_stack(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    device = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(jax, "__version__", "0.11.1")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(metadata, "version", COLAB_STACK.__getitem__)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="NVIDIA test GPU, 16000 MiB, 999.0"
        ),
    )

    exec(
        _cell_source("check-gpu"),
        {
            "install_marker": tmp_path / "installed",
            "subprocess": subprocess,
            "sys": sys,
        },
    )


def test_colab_gpu_check_clears_marker_before_reinstall(
    tmp_path: Path, monkeypatch
) -> None:
    from importlib import metadata

    marker = tmp_path / "installed"
    marker.write_text("stale", encoding="utf-8")
    device = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(jax, "__version__", "0.11.0")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(metadata, "version", COLAB_STACK.__getitem__)

    with pytest.raises(RuntimeError, match="reinstall the stack"):
        exec(
            _cell_source("check-gpu"),
            {
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

    assert 'MODEL_PRESET = "Compare: Protenix + OpenDDE"' in source
    assert '"Public trio: Protenix + OpenDDE + Boltz-2"' in source
    assert '("protenix", "opendde")' in source
    assert '"boltz2"' in source
    assert "PROTEIN_CHAINS" in source
    assert 'clean_input.split(":")' in source
    assert "PERSIST_WEIGHTS_TO_DRIVE = False" in source
    assert 'OUTPUT_ROOT = WORK_DIR / "outputs" / JOB_SLUG / RUN_LABEL' in source
    assert 'RUN_MODE = "Fast demo"' in source
    assert 'MSA_POLICY = "none"' in source
    assert "job_path.read_text()" not in source
    assert '{"num_samples": 1, "num_steps": 20, "num_recycles": 1}' in source
    assert 'models=SELECTED_MODELS' in source
    assert "predict_batch(batch_request)" in source
    assert '"dtype": "float32"' in source
    assert '"attention_kernel": "xla"' in source
    assert "CONTINUE_ON_ERROR" in source
    assert "model_info(model_name)" in source
    assert "for model_name in SELECTED_MODELS" in source
    assert "resume=True" in source
    assert "STRUCTURES" in source
    assert "comparison.to_csv" in source
    assert '"batch_report.json"' in source
    assert "native score" in markdown.lower()
    assert "sequential" in markdown.lower()
    assert "ColabFold MMseqs2" in markdown


def test_colab_notebook_python_cells_are_syntactically_valid() -> None:
    for cell in _code_cells():
        ast.parse(_source(cell), filename=f"{NOTEBOOK}:{cell['id']}")


@pytest.mark.parametrize("model_name", ("protenix", "opendde", "boltz2"))
def test_colab_common_request_options_are_supported(
    model_name: str, tmp_path: Path
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
        options={
            "dtype": "float32",
            "attention_kernel": "xla",
        },
        resume=True,
        num_samples=1,
        num_steps=20,
        num_recycles=1,
    )

    backend = get_backend(model_name)
    backend.validate_request(request)
    native = backend.apply_sampling(request)
    assert {1, 20}.issubset(set(native.values()))


def test_colab_prediction_and_output_cells_execute_together(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[PredictionRequest] = []

    def fake_resolve(request: PredictionRequest):
        assert request.models == ("protenix", "opendde")
        return tuple(
            SimpleNamespace(
                model=model,
                input=request.input,
                output_dir=tmp_path / "outputs" / model / "insulin-complex",
                resolved_seeds=(0,),
                sampling=request.sampling,
            )
            for model in request.models
        )

    def fake_predict_batch(request: PredictionRequest) -> BatchReport:
        seen.append(request)
        results = []
        for index, model in enumerate(request.models or ()):
            output_dir = tmp_path / "outputs" / model / "insulin-complex"
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
            results.append(
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
                )
            )
        return BatchReport(results=tuple(results))

    monkeypatch.setattr(foldjax, "resolve_requests", fake_resolve)
    monkeypatch.setattr(foldjax, "predict_batch", fake_predict_batch)
    monkeypatch.setattr(foldjax.progress, "enable", lambda: None)

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
        }
    )
    exec(_cell_source("build-job"), namespace)
    exec(_cell_source("plan-runs"), namespace)
    exec(_cell_source("run-predictions"), namespace)
    exec(_cell_source("compare-results"), namespace)

    assert len(seen) == 1
    request = seen[0]
    assert request.models == ("protenix", "opendde")
    assert request.options == {
        "dtype": "float32",
        "attention_kernel": "xla",
    }
    assert (request.num_samples, request.num_steps, request.num_recycles) == (1, 20, 1)
    assert request.num_seeds == 1
    assert request.msa == "none"
    assert tuple(namespace["comparison"]["model"]) == ("opendde", "protenix")
    assert len(namespace["STRUCTURES"]) == 2

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
        "outputs/protenix/insulin-complex/protenix.cif",
        "outputs/opendde/insulin-complex/opendde.cif",
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
            "batch_request": object(),
            "display": displayed.append,
            "pd": __import__("pandas"),
        },
    )

    assert len(displayed) == 1
    assert displayed[0].iloc[0]["model"] == "opendde"
    assert displayed[0].iloc[0]["error_type"] == "MemoryError"


def test_readme_links_to_and_explains_the_colab_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Examples" in readme
    assert "### Google Colab: compare models from one input" in readme
    assert (
        "https://colab.research.google.com/github/eightmm/FoldJAX/blob/"
        "main/notebooks/FoldJAX_Colab.ipynb"
    ) in readme
    assert "[Google Colab workflow](notebooks/FoldJAX_Colab.ipynb)" in readme
    assert 'models=("protenix", "opendde")' in readme
    assert "predict_batch" in readme
