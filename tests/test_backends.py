import dataclasses
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.alphafold3 import AlphaFold3Backend, _runner_path
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.chai import ChaiBackend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, model: str, **options) -> PredictionRequest:
    input_path = tmp_path / ("job.yaml" if model == "boltz2" else "job.json")
    input_path.write_text("{}")
    if model == "opendde":
        weights = tmp_path / "opendde.jax"
        weights.touch()
    else:
        weights = tmp_path / "weights"
        weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model=model,
        input=input_path,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options=options,
    )


def test_boltz_adapter_calls_native_api_and_normalizes_samples(
    tmp_path: Path, monkeypatch
) -> None:
    mols = tmp_path / "mols"
    mols.mkdir()
    seen = {}
    output_path = tmp_path / "out" / "job.cif"

    def native_predict(**kwargs):
        seen.update(kwargs)
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.asarray([0.7, 0.8]),
            "iptm": np.asarray([0.6]),
            "out_path": output_path,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    result = Boltz2Backend().predict(_request(tmp_path, "boltz2", mols=mols))

    assert seen["input"].name == "job.yaml"
    assert seen["weights"].name == "weights"
    assert seen["seed"] == 5
    assert seen["compile_cache"] == tmp_path / "cache"
    assert result.samples[0].structure_path == output_path
    assert result.samples[0].scores == {"iptm": 0.6, "mean_plddt": 0.75}


def test_boltz_adapter_requires_molecule_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mols"):
        Boltz2Backend().predict(_request(tmp_path, "boltz2"))


def _write_protenix_outputs(out: Path) -> None:
    predictions = out / "job" / "seed_5" / "predictions"
    predictions.mkdir(parents=True)
    (predictions / "job_sample_0.cif").touch()
    (predictions / "job_summary_confidence_sample_0.json").write_text(
        json.dumps(
            {"ptm": 0.81, "iptm": 0.42, "ranking_score": 0.77, "chain_ptm": [0.8, 0.9]}
        )
    )


def test_protenix_adapter_invokes_cli_in_process(tmp_path: Path, monkeypatch) -> None:
    seen = []

    def native_main(argv):
        seen.extend(argv)
        _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    result = ProtenixBackend().predict(
        _request(tmp_path, "protenix", n_sample=2, n_step=20)
    )

    assert seen[:6] == [
        "--input-json",
        str(tmp_path / "job.json"),
        "--weights",
        str(tmp_path / "weights"),
        "--out",
        str(tmp_path / "out"),
    ]
    assert "--n-sample" in seen and "2" in seen
    assert "--n-step" in seen and "20" in seen
    assert seen[seen.index("--compile-cache") + 1] == str(tmp_path / "cache")
    assert result.samples[0].structure_path.name == "job_sample_0.cif"
    # Scalar confidence fields become common scores; arrays stay in the file.
    assert result.samples[0].scores == {
        "ptm": 0.81,
        "iptm": 0.42,
        "ranking_score": 0.77,
    }


def test_protenix_adapter_tolerates_missing_confidence_json(
    tmp_path: Path, monkeypatch
) -> None:
    def native_main(argv):
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "ranked_0.cif").touch()

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    result = ProtenixBackend().predict(_request(tmp_path, "protenix"))
    assert result.samples[0].scores == {}


def test_protenix_adapter_rejects_unknown_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported Protenix options"):
        ProtenixBackend().predict(_request(tmp_path, "protenix", nonsense=1))


def test_opendde_adapter_invokes_cli_in_process_and_normalizes_scores(
    tmp_path: Path, monkeypatch
) -> None:
    seen = []
    components_path = tmp_path / "components.cif"
    rdkit_cache_path = tmp_path / "components.cif.rdkit_mol.pkl"
    template_mmcif_dir = tmp_path / "template_mmcif"
    template_release_dates = tmp_path / "release_date_cache.json"
    template_obsolete_map = tmp_path / "obsolete_to_successor.json"
    kalign_binary = tmp_path / "kalign"

    def native_main(argv):
        seen.extend(argv)
        _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    monkeypatch.setattr(
        "foldjax.backends.opendde.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    result = OpenDDEBackend().predict(
        _request(
            tmp_path,
            "opendde",
            n_sample=2,
            n_step=20,
            n_cycle=3,
            n_queries=8,
            components_cif=components_path,
            ccd_rdkit_cache=rdkit_cache_path,
            template_mmcif_dir=template_mmcif_dir,
            template_release_dates=template_release_dates,
            template_obsolete_map=template_obsolete_map,
            kalign_binary=kalign_binary,
            include_raw=True,
        )
    )

    assert seen[:6] == [
        "--input-json",
        str(tmp_path / "job.json"),
        "--weights",
        str(tmp_path / "opendde.jax"),
        "--out",
        str(tmp_path / "out"),
    ]
    assert seen[seen.index("--seed") + 1] == "5"
    assert seen[seen.index("--n-sample") + 1] == "2"
    assert seen[seen.index("--n-step") + 1] == "20"
    assert seen[seen.index("--n-cycle") + 1] == "3"
    assert seen[seen.index("--n-queries") + 1] == "8"
    assert seen[seen.index("--components-cif") + 1] == str(components_path)
    assert seen[seen.index("--ccd-rdkit-cache") + 1] == str(rdkit_cache_path)
    assert seen[seen.index("--template-mmcif-dir") + 1] == str(template_mmcif_dir)
    assert seen[seen.index("--template-release-dates") + 1] == str(
        template_release_dates
    )
    assert seen[seen.index("--template-obsolete-map") + 1] == str(template_obsolete_map)
    assert seen[seen.index("--kalign-binary") + 1] == str(kalign_binary)
    assert seen[seen.index("--compile-cache") + 1] == str(tmp_path / "cache")
    assert "--include-raw" in seen
    assert result.model == "opendde"
    assert result.samples[0].structure_path.name == "job_sample_0.cif"
    assert result.samples[0].scores == {
        "ptm": 0.81,
        "iptm": 0.42,
        "ranking_score": 0.77,
    }


def test_opendde_adapter_rejects_unknown_and_mistyped_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported OpenDDE options"):
        OpenDDEBackend().predict(_request(tmp_path, "opendde", nonsense=1))
    with pytest.raises(ValueError, match="include_raw must be a boolean"):
        OpenDDEBackend().predict(_request(tmp_path, "opendde", include_raw="false"))


def test_opendde_adapter_requires_native_weight_file(tmp_path: Path) -> None:
    request = _request(tmp_path, "opendde")
    request.weights.unlink()
    request.weights.mkdir()
    with pytest.raises(FileNotFoundError, match="native weight file"):
        OpenDDEBackend().predict(request)

    checkpoint = tmp_path / "opendde.pt"
    checkpoint.write_bytes(b"torch checkpoint")
    request = dataclasses.replace(request, weights=checkpoint)
    with pytest.raises(ValueError, match="converted native weights"):
        OpenDDEBackend().predict(request)


def test_opendde_capabilities_match_the_current_native_runtime() -> None:
    capabilities = OpenDDEBackend().capabilities()
    assert capabilities.input_formats == ("native", "opendde", "foldjax")
    assert capabilities.supports_msa is True
    assert capabilities.supports_templates is True


@dataclasses.dataclass(frozen=True)
class FakeFoldInput:
    name: str = "job"
    rng_seeds: tuple[int, ...] = (1,)

    def sanitised_name(self) -> str:
        return self.name


def test_alphafold3_adapter_invokes_cloned_runner(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path, "alphafold3", diffusion_samples=2)
    runner_file = tmp_path / "af3" / "run_alphafold.py"
    runner_file.parent.mkdir()
    runner_file.touch()
    request.options["source"] = runner_file.parent
    seen = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            seen["runner"] = kwargs

    inference_result = SimpleNamespace(metadata={"ranking_score": 0.91})
    results = (SimpleNamespace(seed=5, inference_results=(inference_result,)),)

    def predict_structure(fold_input, model_runner, *, buckets):
        seen["fold_input"] = fold_input
        return results

    def write_outputs(written, output_dir, job_name):
        # Mirror the runner's own per-sample layout and file naming.
        sample_dir = Path(output_dir) / "seed-5_sample-0"
        sample_dir.mkdir(parents=True)
        prefix = f"{job_name}_seed-5_sample-0"
        (sample_dir / f"{prefix}_model.cif").touch()
        (sample_dir / f"{prefix}_summary_confidences.json").write_text(
            json.dumps({"ptm": 0.7, "iptm": 0.5, "chain_pair_pae_min": [[1.0]]})
        )

    runner = SimpleNamespace(
        ModelRunner=FakeRunner,
        make_model_config=lambda **kwargs: kwargs,
        predict_structure=predict_structure,
        write_outputs=write_outputs,
    )
    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda path: iter((FakeFoldInput(),))
    common = ModuleType("alphafold3.common")
    common.folding_input = folding
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    monkeypatch.setattr("foldjax.backends.alphafold3._load_runner", lambda path: runner)
    monkeypatch.setattr("jax.local_devices", lambda backend=None: ("device0",))

    result = AlphaFold3Backend().predict(request)

    assert seen["runner"]["device"] == "device0"
    assert seen["fold_input"].rng_seeds == (5,)
    assert result.raw == results
    sample = result.samples[0]
    assert sample.structure_path.name == "job_seed-5_sample-0_model.cif"
    assert sample.seed == 5
    assert sample.scores == {"ptm": 0.7, "iptm": 0.5, "ranking_score": 0.91}


def test_alphafold3_source_path_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="runner not found"):
        _runner_path({"source": tmp_path})


@dataclasses.dataclass(frozen=True)
class FakeInferenceConfig:
    seed: int | None = None
    compilation_cache_dir: Path | None = None
    num_diffusion_samples: int = 5
    use_esm_embeddings: bool = True
    msa_directory: Path | None = None
    esm_attention_implementation: str = "xla"

    def validate(self) -> None:
        return None


def _chai_assets(tmp_path: Path) -> None:
    (tmp_path / "weights" / "models" / "chai1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "weights" / "conformers.npz").touch()


def _fake_chai(tmp_path: Path, seen: dict) -> SimpleNamespace:
    cif_paths = [tmp_path / "out" / "pred.model_idx_0.cif"]

    def run_inference(fasta_file, **kwargs):
        seen.update(kwargs)
        seen["fasta_file"] = fasta_file
        for path in cif_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return SimpleNamespace(
            cif_paths=cif_paths,
            ranking_data=["ranking-0"],
            plddt=np.asarray([[0.8, 0.9]]),
        )

    return SimpleNamespace(
        InferenceConfig=FakeInferenceConfig, run_inference=run_inference
    )


def _fake_ranking(monkeypatch, scores: dict) -> None:
    rank = ModuleType("foldjax.models.chai.ranking.rank")
    rank.get_scores = lambda data: scores
    monkeypatch.setitem(sys.modules, "foldjax.models.chai.ranking.rank", rank)


def test_chai_adapter_resolves_assets_and_normalizes_samples(
    tmp_path: Path, monkeypatch
) -> None:
    _chai_assets(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(
        "foldjax.backends.chai.import_module", lambda name: _fake_chai(tmp_path, seen)
    )
    _fake_ranking(
        monkeypatch,
        {
            "aggregate_score": np.asarray([0.72]),
            "ptm": np.asarray(0.65),
            "iptm": np.asarray(0.55),
            "has_inter_chain_clashes": np.asarray([False]),
        },
    )

    result = ChaiBackend().predict(
        _request(tmp_path, "chai", num_diffusion_samples=2, msa_directory=str(tmp_path))
    )

    assert seen["bundle_path"] == tmp_path / "weights" / "models" / "chai1"
    assert seen["conformer_path"] == tmp_path / "weights" / "conformers.npz"
    assert seen["config"].seed == 5
    assert seen["config"].compilation_cache_dir == tmp_path / "cache"
    assert seen["config"].num_diffusion_samples == 2
    # Path-typed options are coerced from their JSON/CLI string form.
    assert seen["config"].msa_directory == tmp_path
    sample = result.samples[0]
    assert sample.structure_path.name == "pred.model_idx_0.cif"
    assert sample.metadata == {"rank": 0}
    assert sample.scores == {
        "aggregate_score": 0.72,
        "ptm": 0.65,
        "iptm": 0.55,
        "has_inter_chain_clashes": 0.0,
        "mean_plddt": pytest.approx(0.85),
    }


def test_chai_adapter_accepts_explicit_asset_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    conformers = tmp_path / "conf.npz"
    conformers.touch()
    seen: dict = {}
    monkeypatch.setattr(
        "foldjax.backends.chai.import_module", lambda name: _fake_chai(tmp_path, seen)
    )
    _fake_ranking(monkeypatch, {"ptm": np.asarray(0.0)})

    ChaiBackend().predict(
        _request(
            tmp_path, "chai", bundle_path=str(bundle), conformer_path=str(conformers)
        )
    )
    assert seen["bundle_path"] == bundle
    assert seen["conformer_path"] == conformers


def test_chai_adapter_fails_explicitly_on_missing_conformers(tmp_path: Path) -> None:
    (tmp_path / "weights" / "models" / "chai1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="conformer_path"):
        ChaiBackend().predict(_request(tmp_path, "chai"))


def test_chai_adapter_rejects_a_missing_asset_override(tmp_path: Path) -> None:
    _chai_assets(tmp_path)
    with pytest.raises(FileNotFoundError, match="bundle_path does not exist"):
        ChaiBackend().predict(
            _request(tmp_path, "chai", bundle_path=str(tmp_path / "nope"))
        )


def test_chai_adapter_rejects_request_owned_and_unknown_options(
    tmp_path: Path, monkeypatch
) -> None:
    _chai_assets(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(
        "foldjax.backends.chai.import_module", lambda name: _fake_chai(tmp_path, seen)
    )
    with pytest.raises(ValueError, match="set from the FoldJAX request"):
        ChaiBackend().predict(_request(tmp_path, "chai", seed=3))
    with pytest.raises(ValueError, match="unsupported chai options"):
        ChaiBackend().predict(_request(tmp_path, "chai", nonsense=1))


def test_chai_adapter_reads_option_booleans_from_their_text(tmp_path: Path) -> None:
    """``--option use_esm_embeddings=False`` arrives as the truthy string "False"."""
    from foldjax.backends.chai import _coerce

    for text in ("False", "false", "FALSE", "no", "off", "0"):
        assert _coerce("bool", text) is False, text
    for text in ("True", "true", "yes", "on", "1"):
        assert _coerce("bool", text) is True, text
    assert _coerce("bool", False) is False
    with pytest.raises(ValueError, match="cannot read"):
        _coerce("bool", "maybe")


def test_chai_adapter_keeps_its_output_directory_clear_of_materialized_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    """Chai refuses a non-empty directory, and predict() writes inputs/ first."""
    _chai_assets(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(
        "foldjax.backends.chai.import_module", lambda name: _fake_chai(tmp_path, seen)
    )
    _fake_ranking(monkeypatch, {"ptm": np.asarray(0.0)})
    request = _request(tmp_path, "chai")
    (request.output_dir / "inputs").mkdir(parents=True)
    (request.output_dir / "inputs" / "chai_input.fasta").write_text(">A\nACD\n")

    ChaiBackend().predict(request)

    assert seen["output_dir"] == request.output_dir / "predictions"
    assert seen["output_dir"] != request.output_dir
