from __future__ import annotations

import ast
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import foldjax
from foldjax import PredictionRequest, PredictionResult, PredictionSample
from foldjax.backends.protenix import ProtenixBackend

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "FoldJAX_Colab.ipynb"


def _document() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[dict]:
    return [cell for cell in _document()["cells"] if cell["cell_type"] == "code"]


def _source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _cell_source(cell_id: str) -> str:
    return _source(next(cell for cell in _document()["cells"] if cell["id"] == cell_id))


def test_colab_notebook_is_clean_gpu_notebook() -> None:
    document = _document()

    assert document["nbformat"] == 4
    assert document["nbformat_minor"] >= 5
    assert document["metadata"]["accelerator"] == "GPU"
    assert document["metadata"]["language_info"]["version"] == "3.12"
    assert document["cells"][0]["cell_type"] == "markdown"
    for cell in _code_cells():
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    ids = [cell["id"] for cell in document["cells"]]
    assert len(ids) == len(set(ids))
    assert ids.index("build-job") < ids.index("run-prediction")
    assert ids.index("run-prediction") < ids.index("inspect-result")
    assert ids.index("inspect-result") < ids.index("view-structure")
    assert ids.index("view-structure") < ids.index("download-results")


def test_colab_notebook_installs_supported_cuda_runtime_before_imports() -> None:
    cells = _code_cells()
    install = _source(cells[0])
    all_source = "\n".join(_source(cell) for cell in cells)

    assert 'FOLDJAX_REF = "49800135a81f6d6e9cacaee547d1ea09c083f94b"' in install
    assert "foldjax[cuda12]" in install
    assert "FoldJAX.git@{FOLDJAX_REF}" in install
    assert '"py3Dmol>=2.0,<3"' in install
    assert all_source.index('"pip"') < all_source.index("import jax")
    assert 'sys.version_info[:2] != (3, 12)' in install
    assert 'shutil.which("nvidia-smi")' in install
    assert "Dependencies installed; restarting the runtime once" in install
    assert "signal.SIGKILL" in install
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in all_source
    assert "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND" in all_source
    assert 'jax.__version__ != "0.10.1"' in all_source
    assert 'device.platform == "gpu"' in all_source


def test_colab_notebook_keeps_demo_and_privacy_choices_explicit() -> None:
    source = "\n".join(_source(cell) for cell in _code_cells())

    assert 'model="protenix"' in source
    assert "PERSIST_TO_DRIVE = False" in source
    assert "FAST_DEMO = True" in source
    assert 'MSA_POLICY = "none"' in source
    assert "job_path.read_text()" not in source
    assert '{"num_samples": 1, "num_steps": 20, "num_recycles": 1}' in source
    assert '"dtype": "float32"' in source
    assert '"attention_kernel": "xla"' in source
    assert '"triangle_kernel": "xla"' in source
    assert 'cache_dir=WORK_DIR / "compile-cache"' in source
    assert "resume=True" in source
    assert '"-m"' in source
    assert '"foldjax.cli"' in source
    assert '"weights"' in source and '"fetch"' in source
    assert "math.isfinite" in source
    assert "gemmi.cif.read_file" in source
    assert "files.download" in source


def test_colab_notebook_python_cells_are_syntactically_valid() -> None:
    for cell in _code_cells():
        ast.parse(_source(cell), filename=f"{NOTEBOOK}:{cell['id']}")


def test_colab_request_uses_supported_protenix_options(tmp_path: Path) -> None:
    input_path = tmp_path / "ubiquitin.json"
    input_path.write_text(
        '{"name":"ubiquitin","entities":[{"type":"protein",'
        '"id":["A"],"sequence":"ACDEFG"}]}',
        encoding="utf-8",
    )
    request = PredictionRequest(
        model="protenix",
        input=input_path,
        output_dir=tmp_path / "outputs" / "ubiquitin-protenix-fast-demo",
        input_format="foldjax",
        cache_dir=tmp_path / "compile-cache",
        seed=0,
        msa="none",
        options={
            "dtype": "float32",
            "attention_kernel": "xla",
            "triangle_kernel": "xla",
        },
        resume=True,
        num_samples=1,
        num_steps=20,
        num_recycles=1,
    )

    backend = ProtenixBackend()
    backend.validate_request(request)
    assert backend.apply_sampling(request) == {
        "trunk_dtype": "fp32",
        "trunk_single_attention_backend": "xla_jit",
        "trunk_triangle_attention_backend": "xla_jit",
        "n_sample": 1,
        "n_step": 20,
        "n_cycle": 1,
    }


def test_colab_prediction_and_output_cells_execute_together(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[PredictionRequest] = []

    def fake_predict(request: PredictionRequest) -> PredictionResult:
        seen.append(request)
        assert request.output_dir is not None
        request.output_dir.mkdir(parents=True)
        structure = request.output_dir / "prediction.cif"
        structure.write_text("data_prediction\n_entry.id prediction\n#\n")
        (request.output_dir / "confidence.json").write_text(
            '{"ranking_score":0.8}', encoding="utf-8"
        )
        (request.output_dir / "foldjax_run.json").write_text(
            '{"schema":1}', encoding="utf-8"
        )
        return PredictionResult(
            model="protenix",
            samples=(
                PredictionSample(
                    seed=0,
                    structure_path=structure,
                    scores={"ranking_score": 0.8},
                ),
            ),
            output_dir=request.output_dir,
        )

    monkeypatch.setattr(foldjax, "predict", fake_predict)
    monkeypatch.setattr(foldjax.progress, "enable", lambda: None)
    namespace = {"Path": Path, "WORK_DIR": tmp_path}
    exec(_cell_source("build-job"), namespace)
    exec(_cell_source("run-prediction"), namespace)

    assert len(seen) == 1
    request = seen[0]
    assert request.options == {
        "dtype": "float32",
        "attention_kernel": "xla",
        "triangle_kernel": "xla",
    }
    assert (request.num_samples, request.num_steps, request.num_recycles) == (1, 20, 1)
    assert request.msa == "none"
    exec(_cell_source("inspect-result"), namespace)

    viewer_calls: list[tuple] = []

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
    monkeypatch.setitem(sys.modules, "py3Dmol", py3dmol)
    exec(_cell_source("view-structure"), namespace)
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
        assert set(bundle.namelist()) == {
            "confidence.json",
            "foldjax_run.json",
            "prediction.cif",
        }


def test_readme_links_to_the_colab_notebook() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        "https://colab.research.google.com/github/eightmm/FoldJAX/blob/"
        "main/notebooks/FoldJAX_Colab.ipynb"
    ) in readme
    assert "[Google Colab quickstart](notebooks/FoldJAX_Colab.ipynb)" in readme
