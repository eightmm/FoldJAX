import dataclasses
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.alphafold3 import AlphaFold3Backend, _runner_path
from foldjax.backends.boltz2 import Boltz2Backend
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


def test_boltz_gives_each_sample_its_own_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    """Ranking samples is the reason to generate more than one.

    Boltz-2 reports pLDDT as [n_sample, n_token] and the pTM family as one
    value per sample. The adapter used to average pLDDT over the whole batch,
    take element 0 of iptm, and copy that one dict onto every sample -- so
    three structures came back with three identical score sets and nothing to
    rank them by, with no error to say so.
    """
    mols = tmp_path / "mols"
    mols.mkdir()
    paths = [tmp_path / "out" / f"job_model_{index}.cif" for index in range(3)]

    def native_predict(**kwargs):
        return {
            "coords": np.zeros((3, 4, 3)),
            # Per-sample means are 0.1, 0.5 and 0.9.
            "plddt": np.asarray([[0.1] * 4, [0.5] * 4, [0.9] * 4]),
            "out_paths": paths,
            "raw": {
                "iptm": np.asarray([0.11, 0.55, 0.99]),
                "ptm": np.asarray([0.12, 0.56, 0.98]),
                "ligand_iptm": np.asarray([0.13, 0.57, 0.97]),
            },
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    result = Boltz2Backend().predict(_request(tmp_path, "boltz2", mols=mols))

    assert len(result.samples) == 3
    assert [sample.scores["mean_plddt"] for sample in result.samples] == pytest.approx(
        [0.1, 0.5, 0.9]
    )
    assert [sample.scores["iptm"] for sample in result.samples] == pytest.approx(
        [0.11, 0.55, 0.99]
    )
    assert [sample.scores["ptm"] for sample in result.samples] == pytest.approx(
        [0.12, 0.56, 0.98]
    )
    assert [
        sample.scores["ligand_iptm"] for sample in result.samples
    ] == pytest.approx([0.13, 0.57, 0.97])


def test_boltz_reports_the_heads_own_plddt_aggregate(
    tmp_path: Path, monkeypatch
) -> None:
    """`complex_plddt` is the number upstream writes, so it is the comparable one.

    The confidence head computes `(plddt * token_pad_mask).sum() /
    token_pad_mask.sum()` and Boltz's own writer puts that in its confidence
    JSON. FoldJAX reported only a plain mean of the pLDDT vector, which weights
    padded and masked positions like everything else, so the two
    implementations were being compared on two different statistics.
    """
    mols = tmp_path / "mols"
    mols.mkdir()

    def native_predict(**kwargs):
        return {
            "coords": np.zeros((2, 4, 3)),
            "plddt": np.asarray([[0.2, 0.2, 0.0, 0.0], [0.8, 0.8, 0.0, 0.0]]),
            "out_paths": [tmp_path / "a.cif", tmp_path / "b.cif"],
            "raw": {"complex_plddt": np.asarray([0.40, 0.90])},
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    result = Boltz2Backend().predict(_request(tmp_path, "boltz2", mols=mols))

    # The head's masked aggregate, per sample -- not the 0.1/0.4 a plain mean
    # over the padded vector gives.
    assert [
        sample.scores["complex_plddt"] for sample in result.samples
    ] == pytest.approx([0.40, 0.90])
    assert [sample.scores["mean_plddt"] for sample in result.samples] == pytest.approx(
        [0.1, 0.4]
    )


def test_boltz_shares_a_confidence_value_that_is_not_per_sample(
    tmp_path: Path, monkeypatch
) -> None:
    """A single reported value describes the whole prediction, so it is shared.

    Indexing it by sample would read past the end or, worse, hand sample 1 a
    number belonging to something else.
    """
    mols = tmp_path / "mols"
    mols.mkdir()

    def native_predict(**kwargs):
        return {
            "coords": np.zeros((2, 4, 3)),
            "plddt": np.asarray([[0.2] * 4, [0.8] * 4]),
            "out_paths": [tmp_path / "a.cif", tmp_path / "b.cif"],
            "raw": {"iptm": np.asarray([0.42])},
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    result = Boltz2Backend().predict(_request(tmp_path, "boltz2", mols=mols))

    assert [sample.scores["iptm"] for sample in result.samples] == [0.42, 0.42]
    assert [sample.scores["mean_plddt"] for sample in result.samples] == pytest.approx(
        [0.2, 0.8]
    )


def test_boltz_adapter_requires_molecule_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mols"):
        Boltz2Backend().predict(_request(tmp_path, "boltz2"))


def _write_protenix_outputs(out: Path, samples: int = 1) -> list[Path]:
    """Write what the shared Protenix writer writes, and return what it returns.

    The real `main` hands back every path it produced, in rank order. The
    adapters read that list rather than globbing, so a double that returns
    nothing is not modelling the contract.
    """
    predictions = out / "job" / "seed_5" / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rank in range(samples):
        cif = predictions / f"job_sample_{rank}.cif"
        cif.touch()
        confidence = predictions / f"job_summary_confidence_sample_{rank}.json"
        confidence.write_text(
            json.dumps(
                {
                    "ptm": 0.81,
                    "iptm": 0.42,
                    "ranking_score": 0.77,
                    "chain_ptm": [0.8, 0.9],
                }
            )
        )
        written.extend((cif, confidence))
    return written


def test_protenix_adapter_invokes_cli_in_process(tmp_path: Path, monkeypatch) -> None:
    seen = []

    def native_main(argv):
        seen.extend(argv)
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

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
        path = out / "ranked_0.cif"
        path.touch()
        return [path]

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    result = ProtenixBackend().predict(_request(tmp_path, "protenix"))
    assert result.samples[0].scores == {}


def test_no_cache_reaches_protenix_as_an_explicit_refusal(
    tmp_path: Path, monkeypatch
) -> None:
    """Omitting the flag does not disable Protenix's cache -- it moves it.

    The native CLI defaults `--compile-cache` to `outputs/compile_cache`, a
    relative path, so a request that asked to write nothing still created a
    cache directory beside whatever the working directory happened to be.
    """
    seen = []

    def native_main(argv):
        seen.extend(argv)
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=native_main),
    )
    request = dataclasses.replace(
        _request(tmp_path, "protenix"), cache_dir=None, use_compile_cache=False
    )
    ProtenixBackend().predict(request)

    assert "--no-compile-cache" in seen
    assert "--compile-cache" not in seen


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
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

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


@pytest.mark.parametrize(
    ("model", "backend"),
    [("protenix", ProtenixBackend), ("opendde", OpenDDEBackend)],
)
def test_a_rerun_does_not_report_the_previous_runs_structures(
    tmp_path: Path, monkeypatch, model: str, backend
) -> None:
    """Only what this run wrote is this run's result.

    The output directory defaults to one derived from the input, so running the
    same job twice reuses it. These adapters used to recover their samples by
    globbing the tree for ``*.cif``, which cannot tell a structure written now
    from one left behind earlier -- so asking for one sample after a five-sample
    run returned five, four of them stale, with no error. The native CLIs now
    return the paths they wrote and the adapters read that.
    """
    out = tmp_path / "out"

    def five_samples(argv):
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]), samples=5)

    def one_sample(argv):
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]), samples=1)

    monkeypatch.setattr(
        f"foldjax.backends.{model}.import_module",
        lambda name: SimpleNamespace(main=five_samples),
    )
    first = backend().predict(_request(tmp_path, model))
    assert len(first.samples) == 5

    monkeypatch.setattr(
        f"foldjax.backends.{model}.import_module",
        lambda name: SimpleNamespace(main=one_sample),
    )
    second = backend().predict(_request(tmp_path, model))

    assert len(second.samples) == 1
    assert second.samples[0].structure_path.name == "job_sample_0.cif"
    # The four files from the first run are still on disk; they are simply not
    # claimed as results of the second.
    assert len(list(out.rglob("*.cif"))) == 5


def test_samples_come_back_in_rank_order_past_ten(tmp_path: Path, monkeypatch) -> None:
    """Rank order is what the writer emits, not what sorting the names gives.

    Protenix names its files ``job_sample_<rank>.cif``, so a lexicographic sort
    puts sample_10 second, right after sample_0. Ten or more samples is the
    common case for anything being ranked, and the top hit is the one that
    matters, so this silently mislabelled which structure was best.
    """

    def twelve(argv):
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]), samples=12)

    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda name: SimpleNamespace(main=twelve),
    )
    result = ProtenixBackend().predict(_request(tmp_path, "protenix"))

    assert [sample.structure_path.name for sample in result.samples] == [
        f"job_sample_{rank}.cif" for rank in range(12)
    ]


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


def test_openfold3_predicts_from_the_vendored_package() -> None:
    """Everything the backend imports is in-package, needing no extra.

    This test used to assert the opposite: the port was on no package index, so
    the backend raised a ModuleNotFoundError explaining that you had to install
    it yourself. Vendoring removed the condition, and the message with it.

    What is worth pinning now is the property vendoring bought -- that a plain
    `uv sync` is enough to predict with this model. Importing each module by the
    name the backend uses is what would catch the package being moved or
    renamed, which is the way this can regress.
    """
    from importlib import import_module

    for name in (
        "foldjax.models.openfold3.data",
        "foldjax.models.openfold3.inference",
        "foldjax.models.openfold3.output",
        "foldjax.models.openfold3.compilation",
        "foldjax.models.openfold3.bridge.chemistry",
        "foldjax.models.openfold3.bridge.checkpoint",
        "foldjax.models.openfold3.bridge.torch_mapping",
    ):
        assert import_module(name) is not None, name


def test_openfold3_featurization_names_the_extra_it_needs(monkeypatch) -> None:
    """Featurization runs upstream's pipeline, which needs the torch extra.

    The pipeline is vendored, so this no longer fails for want of a checkout --
    but it is still upstream's torch and lightning code, and a bare
    "No module named ..." would not say which extra supplies that. The message
    names it.

    Simulated rather than assumed: setting the module to None in `sys.modules`
    makes the import fail even in an environment that has the extra, so this
    keeps checking the message here rather than only on a machine that happens
    to lack it. The patched name is the *vendored* path, which is what caught
    this test when the pipeline moved -- the old name no longer participates, so
    the patch stopped biting and the assertion stopped meaning anything.
    """
    import sys

    from foldjax.models.openfold3.data import featurize_query

    monkeypatch.setitem(
        sys.modules,
        "foldjax.models.openfold3._upstream.openfold3.projects.of3_all_atom"
        ".config.inference_query_format",
        None,
    )
    with pytest.raises(ImportError) as error:
        featurize_query({"queries": {}})
    message = str(error.value)
    assert "openfold3-preprocess" in message
    assert "inference path itself needs none of it" in message
