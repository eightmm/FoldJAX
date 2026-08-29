import builtins
import dataclasses
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import numpy as np
import pytest

from foldjax.backends.alphafold3 import (
    _PREFIX_STABLE_NOISE_FEATURE,
    VENDORED_RUNNER,
    AlphaFold3Backend,
    _load_runner,
    _runner_path,
    _samples,
    _validated_fold_jobs,
)
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.openfold3 import (
    OpenFold3Backend,
    _compile_enabled,
    _features_and_chemistry,
)
from foldjax.backends.protenix import _RESERVED_CLI_FLAGS, ProtenixBackend
from foldjax.schema import PaddingConfig, PredictionRequest


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
    assert seen["padding"] is None
    assert result.samples[0].structure_path == output_path
    assert result.samples[0].scores == {"iptm": 0.6, "mean_plddt": 0.75}


def test_boltz_adapter_routes_padding_and_preserves_the_native_profile(
    tmp_path: Path, monkeypatch
) -> None:
    mols = tmp_path / "mols"
    mols.mkdir()
    seen = {}
    profile = {
        "actual": {"tokens": 2, "atoms": 3, "msa": 1},
        "storage": {"tokens": 2, "atoms": 5, "msa": 1},
        "target": {"tokens": 8, "atoms": 32, "msa": 1},
        "changed": True,
        "static": {
            "use_template": False,
            "recompute_nonpolymer_frames": True,
        },
    }

    def native_predict(**kwargs):
        seen.update(kwargs)
        return {
            "coords": np.zeros((2, 3, 3)),
            "plddt": np.ones((2, 2)),
            "raw": {"iptm": np.asarray([0.3, 0.7])},
            "padding": {"primary": profile},
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    padding = PaddingConfig(tokens=8, atoms=32, msa=1)
    request = dataclasses.replace(
        _request(tmp_path, "boltz2", mols=mols),
        padding=padding,
        num_samples=2,
    )

    result = Boltz2Backend().predict(request)

    assert seen["padding"] is padding
    assert result.shape_profile == profile
    assert len(result.samples) == 2
    assert result.samples[0].coordinates.shape == (3, 3)


def test_boltz_gives_each_sample_its_own_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    """Ranking samples is the reason to generate more than one.

    Boltz-2 reports pLDDT as [num_samples, n_token] and the pTM family as one
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


def test_boltz_rejects_a_single_diffusion_step_before_loading_model(
    tmp_path: Path,
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        num_steps=1,
        input_format="native",
        options={},
    )

    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        Boltz2Backend().validate_request(request)


@pytest.mark.parametrize(
    "representations",
    [("single", "pair"), ("single,pair",), ("all",)],
)
def test_backend_validates_supported_representation_spellings_before_prediction(
    tmp_path: Path, representations: tuple[str, ...]
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        input_format="native",
        representations=representations,
    )

    Boltz2Backend().validate_request(request)


def test_backend_rejects_an_unsupported_comma_separated_representation_early(
    tmp_path: Path,
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        input_format="native",
        representations=("single,structural_pair",),
    )

    with pytest.raises(ValueError, match="structural_pair.*boltz2"):
        Boltz2Backend().validate_request(request)


def test_backend_rejects_all_when_it_exposes_no_representations(
    tmp_path: Path,
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "alphafold3"),
        input_format="native",
        representations=("all",),
    )

    with pytest.raises(ValueError, match="does not expose trunk representations"):
        AlphaFold3Backend().validate_request(request)


@pytest.mark.parametrize(
    "representations",
    [("",), ("   ",), (",",), (" , ", "")],
)
def test_backend_rejects_representation_selectors_without_a_name(
    tmp_path: Path, representations: tuple[str, ...]
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        input_format="native",
        representations=representations,
        stop_after="trunk",
    )

    with pytest.raises(ValueError, match="at least one representation"):
        Boltz2Backend().validate_request(request)


@pytest.mark.parametrize(
    "seed_fields",
    [{"seed": 0, "seeds": (7, 11)}, {"num_seeds": 2}],
)
def test_backend_rejects_multi_seed_representation_capture(
    tmp_path: Path, seed_fields: dict[str, object]
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        input_format="native",
        representations=("pair",),
        **seed_fields,
    )

    with pytest.raises(ValueError, match="cannot be combined with multiple seeds"):
        Boltz2Backend().validate_request(request)


def test_backend_allows_representation_capture_with_one_resolved_seed(
    tmp_path: Path,
) -> None:
    request = dataclasses.replace(
        _request(tmp_path, "boltz2"),
        input_format="native",
        representations=("pair",),
        num_seeds=1,
    )

    Boltz2Backend().validate_request(request)


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
        _request(
            tmp_path,
            "protenix",
            num_samples=2,
            num_steps=20,
            esm_checkpoint_dir=tmp_path / "esm-checkpoint",
            cli_args=("--gamma0", "0.2"),
        )
    )

    assert seen[:6] == [
        "--input-json",
        str(tmp_path / "job.json"),
        "--weights",
        str(tmp_path / "weights"),
        "--out",
        str(tmp_path / "out"),
    ]
    assert "--num-samples" in seen and "2" in seen
    assert "--num-steps" in seen and "20" in seen
    assert seen[seen.index("--esm-checkpoint-dir") + 1] == str(
        tmp_path / "esm-checkpoint"
    )
    assert seen[seen.index("--gamma0") + 1] == "0.2"
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


def test_protenix_esm_checkpoint_dir_requires_a_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="esm_checkpoint_dir must be a path"):
        ProtenixBackend().predict(
            _request(tmp_path, "protenix", esm_checkpoint_dir=object())
        )


@pytest.mark.parametrize(
    "cli_args",
    ["--gamma0 0.2", ["--gamma0", 0.2], {"--gamma0", "0.2"}, None],
)
def test_protenix_cli_args_require_a_string_sequence(
    tmp_path: Path, cli_args: object
) -> None:
    with pytest.raises(ValueError, match="non-string sequence of strings"):
        ProtenixBackend().predict(
            _request(tmp_path, "protenix", cli_args=cli_args)
        )


@pytest.mark.parametrize("flag", sorted(_RESERVED_CLI_FLAGS))
@pytest.mark.parametrize("equals_form", [False, True])
def test_protenix_cli_args_cannot_override_adapter_owned_flags(
    tmp_path: Path, flag: str, equals_form: bool
) -> None:
    cli_args = (f"{flag}=override",) if equals_form else (flag, "override")
    with pytest.raises(ValueError, match="adapter-owned flag"):
        ProtenixBackend().predict(
            _request(tmp_path, "protenix", cli_args=cli_args)
        )


def test_protenix_cli_args_reject_argparse_abbreviations_of_owned_flags(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="--weights"):
        ProtenixBackend().predict(
            _request(tmp_path, "protenix", cli_args=("--weig", "other"))
        )


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
            num_samples=2,
            num_steps=20,
            num_recycles=3,
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
    assert seen[seen.index("--num-samples") + 1] == "2"
    assert seen[seen.index("--num-steps") + 1] == "20"
    assert seen[seen.index("--num-recycles") + 1] == "3"
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
    ("model", "backend_type", "backend_module", "default_dtype"),
    [
        ("protenix", ProtenixBackend, "foldjax.backends.protenix", "bf16"),
        ("opendde", OpenDDEBackend, "foldjax.backends.opendde", "fp32"),
    ],
)
@pytest.mark.parametrize("invalidate_between", [False, True])
def test_native_cli_backends_reuse_prepared_params_within_a_session(
    tmp_path,
    monkeypatch,
    model,
    backend_type,
    backend_module,
    default_dtype,
    invalidate_between,
) -> None:
    first = _request(tmp_path, model)
    second = dataclasses.replace(
        first,
        seed=6,
        output_dir=tmp_path / "out-second",
    )
    loads = []
    seen_params = []

    def load_prepared(path, dtype):
        value = object()
        loads.append((path, dtype, value))
        return value

    def native_main(argv, **kwargs):
        loader = kwargs.pop("_prepared_params_loader")
        seen_params.append(
            loader(Path(argv[argv.index("--weights") + 1]), default_dtype, True)
        )
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    native = SimpleNamespace(
        PREPARED_PARAMS_LOADER_API=True,
        _load_prepared_params=load_prepared,
        main=native_main,
    )
    monkeypatch.setattr(
        f"{backend_module}.import_module",
        lambda _name: native,
    )
    backend = backend_type()

    with backend.session((first, second)):
        backend.predict(first)
        if invalidate_between:
            backend.invalidate_session()
        backend.predict(second)

    assert len(loads) == (2 if invalidate_between else 1)
    assert loads[0][:2] == (first.weights, default_dtype)
    if invalidate_between:
        assert seen_params == [loads[0][2], loads[1][2]]
    else:
        assert seen_params == [loads[0][2], loads[0][2]]


def test_protenix_and_opendde_share_one_ccd_lease_until_both_sessions_exit(
    tmp_path, monkeypatch
) -> None:
    from foldjax.models import _managed_memory
    from foldjax.models.protenix.data import featurize_json

    protenix_root = tmp_path / "protenix"
    opendde_root = tmp_path / "opendde"
    protenix_root.mkdir()
    opendde_root.mkdir()
    protenix_request = _request(protenix_root, "protenix")
    opendde_request = _request(opendde_root, "opendde")
    releases = []
    cleanup = []
    original_release = featurize_json._release_external_ccd_cache

    def release() -> bool:
        releases.append("release")
        return original_release()

    def native_main(argv):
        featurize_json._EXTERNAL_CCD_MOLS = {"loaded": object()}
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    native = SimpleNamespace(main=native_main)
    monkeypatch.setattr(featurize_json, "_release_external_ccd_cache", release)
    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module", lambda _name: native
    )
    monkeypatch.setattr(
        "foldjax.backends.opendde.import_module", lambda _name: native
    )
    monkeypatch.setattr(
        _managed_memory.gc, "collect", lambda: cleanup.append("collect") or 0
    )
    monkeypatch.setattr(
        _managed_memory, "_malloc_trim", lambda: cleanup.append("trim")
    )
    protenix = ProtenixBackend()
    opendde = OpenDDEBackend()

    with protenix.session(
        (
            protenix_request,
            dataclasses.replace(protenix_request, seed=6),
        )
    ):
        protenix.predict(protenix_request)
        with opendde.session(
            (
                opendde_request,
                dataclasses.replace(opendde_request, seed=6),
            )
        ):
            opendde.predict(opendde_request)
            assert releases == []
        assert releases == []

    assert releases == ["release"]
    assert cleanup == ["collect", "trim"]
    assert featurize_json._EXTERNAL_CCD_MOLS is None


@pytest.mark.parametrize(
    ("model", "backend_type", "error"),
    [
        ("protenix", ProtenixBackend, ValueError("bad input")),
        ("opendde", OpenDDEBackend, KeyboardInterrupt()),
    ],
)
def test_scalar_native_failure_always_releases_external_ccd(
    tmp_path, monkeypatch, model, backend_type, error
) -> None:
    from foldjax.models import _managed_memory
    from foldjax.models.protenix.data import featurize_json

    request = _request(tmp_path, model)

    def native_main(_argv):
        featurize_json._EXTERNAL_CCD_MOLS = {"loaded": object()}
        raise error

    monkeypatch.setattr(
        f"foldjax.backends.{model}.import_module",
        lambda _name: SimpleNamespace(main=native_main),
    )
    monkeypatch.setattr(_managed_memory.gc, "collect", lambda: 0)
    monkeypatch.setattr(_managed_memory, "_malloc_trim", lambda: None)

    with pytest.raises(type(error)):
        backend_type().predict(request)

    assert featurize_json._EXTERNAL_CCD_MOLS is None


def test_protenix_session_defers_loading_until_native_platform_setup(
    tmp_path, monkeypatch
) -> None:
    first = _request(tmp_path, "protenix", cli_args=("--cpu-only",))
    second = dataclasses.replace(
        first,
        seed=6,
        output_dir=tmp_path / "out-second",
    )
    events = []

    def load_prepared(path, dtype):
        events.append(("load", path, dtype, os.environ.get("JAX_PLATFORMS")))
        return object()

    def native_main(argv, **kwargs):
        events.append(("main", tuple(argv)))
        assert "--cpu-only" in argv
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        loader = kwargs.pop("_prepared_params_loader")
        loader(Path(argv[argv.index("--weights") + 1]), "bf16", True)
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    native = SimpleNamespace(
        PREPARED_PARAMS_LOADER_API=True,
        _load_prepared_params=load_prepared,
        main=native_main,
    )
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda _name: native,
    )
    backend = ProtenixBackend()

    with backend.session((first, second)):
        backend.predict(first)
        backend.predict(second)

    assert [event[0] for event in events] == ["main", "load", "main"]
    assert events[1][1:] == (first.weights, "bf16", "cpu")


def test_protenix_session_does_not_retain_params_across_mini_esm_calls(
    tmp_path, monkeypatch
) -> None:
    first = _request(
        tmp_path,
        "protenix",
        model_name="protenix_mini_esm_v0.5.0",
    )
    second = dataclasses.replace(
        first,
        seed=6,
        output_dir=tmp_path / "out-second",
    )
    loads = []
    seen_params = []

    def load_prepared(path, dtype):
        value = object()
        loads.append((path, dtype, value))
        return value

    def native_main(argv, **kwargs):
        loader = kwargs.pop("_prepared_params_loader")
        seen_params.append(
            loader(Path(argv[argv.index("--weights") + 1]), "bf16", False)
        )
        return _write_protenix_outputs(Path(argv[argv.index("--out") + 1]))

    native = SimpleNamespace(
        PREPARED_PARAMS_LOADER_API=True,
        _load_prepared_params=load_prepared,
        main=native_main,
    )
    monkeypatch.setattr(
        "foldjax.backends.protenix.import_module",
        lambda _name: native,
    )
    backend = ProtenixBackend()

    with backend.session((first, second)):
        backend.predict(first)
        backend.predict(second)

    assert len(loads) == 2
    assert seen_params == [loads[0][2], loads[1][2]]


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


def test_esmfold2_rejects_text_for_boolean_native_options(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda _name: SimpleNamespace()
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2._job_chains", lambda _path: ([], {})
    )

    with pytest.raises(ValueError, match="no_language_model must be a boolean"):
        ESMFold2Backend().predict(
            _request(tmp_path, "esmfold2", no_language_model="false")
        )


def test_opendde_capabilities_match_the_current_native_runtime() -> None:
    capabilities = OpenDDEBackend().capabilities()
    assert capabilities.input_formats == ("native", "opendde", "foldjax")
    assert capabilities.supports_msa is True
    assert capabilities.supports_templates is False
    assert "warn and drop" in capabilities.input_requirements["native"].notes
    assert "reject templates and RNA MSAs" in (
        capabilities.input_requirements["foldjax"].notes
    )


@dataclasses.dataclass(frozen=True)
class FakeFoldInput:
    name: str = "job"
    rng_seeds: tuple[int, ...] = (1,)

    def sanitised_name(self) -> str:
        return self.name


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b"]
)
def test_alphafold3_rejects_unsafe_native_job_output_names(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match="one safe filename component"):
        _validated_fold_jobs(
            (FakeFoldInput(name=name),), output_dir=tmp_path / "out", seed=5
        )


def test_alphafold3_rejects_a_native_job_symlink_escape(tmp_path: Path) -> None:
    output = tmp_path / "out"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "job").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes its run directory"):
        _validated_fold_jobs(
            (FakeFoldInput(),), output_dir=output, seed=5
        )


def test_alphafold3_rejects_duplicate_job_directories_before_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda _path: iter(
        (FakeFoldInput(name="same"), FakeFoldInput(name="same"))
    )
    common = ModuleType("alphafold3.common")
    common.folding_input = folding
    package = ModuleType("alphafold3")
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    predicted = []
    runner = SimpleNamespace(
        predict_structure=lambda *_args, **_kwargs: predicted.append(True)
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._runner_path", lambda _options: Path("runner")
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._load_runner", lambda _path: runner
    )

    with pytest.raises(ValueError, match="share output directory"):
        AlphaFold3Backend().predict(_request(tmp_path, "alphafold3"))

    assert predicted == []


@pytest.mark.parametrize(
    ("option", "value"),
    [("return_embeddings", "false"), ("return_distogram", 1)],
)
def test_alphafold3_requires_real_booleans_for_output_options(
    tmp_path: Path, monkeypatch, option: str, value: object
) -> None:
    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda _path: iter((FakeFoldInput(),))
    common = ModuleType("alphafold3.common")
    common.folding_input = folding
    package = ModuleType("alphafold3")
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    configured = []
    runner = SimpleNamespace(
        make_model_config=lambda **kwargs: configured.append(kwargs)
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._runner_path", lambda _options: Path("runner")
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._load_runner", lambda _path: runner
    )
    selected_device = jax.devices()[0]
    monkeypatch.setattr(
        "jax.local_devices", lambda backend=None: (selected_device,)
    )
    request = dataclasses.replace(
        _request(tmp_path, "alphafold3", **{option: value}),
        cache_dir=None,
        use_compile_cache=False,
    )

    with pytest.raises(ValueError, match=f"{option} must be a boolean"):
        AlphaFold3Backend().predict(request)

    assert configured == []


def test_alphafold3_adapter_invokes_cloned_runner(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path, "alphafold3", num_samples=2)
    runner_file = tmp_path / "af3" / "run_alphafold.py"
    runner_file.parent.mkdir()
    runner_file.touch()
    package_init = runner_file.parent / "src/alphafold3/__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.touch()
    request.options["source"] = runner_file.parent
    seen = {}
    active_devices = []

    class FakeRunner:
        def __init__(self, **kwargs):
            seen["runner"] = kwargs

    inference_result = SimpleNamespace(metadata={"ranking_score": 0.91})
    results = (SimpleNamespace(seed=5, inference_results=(inference_result,)),)

    def predict_structure(fold_input, model_runner, *, buckets):
        seen["fold_input"] = fold_input
        seen["active_device"] = active_devices[-1] if active_devices else None
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
    package = ModuleType("alphafold3")
    package.__file__ = str(package_init)
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    monkeypatch.setattr("foldjax.backends.alphafold3._load_runner", lambda path: runner)
    selected_device = jax.devices()[0]
    monkeypatch.setattr(
        "jax.local_devices", lambda backend=None: (selected_device,)
    )

    @contextmanager
    def model_device(device):
        active_devices.append(device)
        try:
            yield
        finally:
            active_devices.pop()

    monkeypatch.setattr("foldjax.backends.alphafold3._model_device", model_device)

    result = AlphaFold3Backend().predict(request)

    assert seen["runner"]["device"] == selected_device
    assert seen["active_device"] == selected_device
    assert active_devices == []
    assert seen["fold_input"].rng_seeds == (5,)
    assert result.raw == results
    sample = result.samples[0]
    assert sample.structure_path.name == "job_seed-5_sample-0_model.cif"
    assert sample.seed == 5
    assert sample.scores == {"ptm": 0.7, "iptm": 0.5, "ranking_score": 0.91}


def test_alphafold3_adapter_routes_padding_through_native_inference(
    tmp_path: Path, monkeypatch
) -> None:
    seen = {}
    plan_summary = {
        "actual": {"tokens": 3},
        "storage": {"tokens": 3},
        "target": {"tokens": 512},
        "changed": True,
    }
    example = {
        "seq_length": np.asarray(3),
        "seq_mask": np.pad(np.ones(3), (0, 509)),
    }
    inference_result = SimpleNamespace(metadata={"ranking_score": 0.88})

    class FakeModelRunner:
        def __init__(self, **kwargs):
            seen["runner"] = kwargs

        def run_inference(self, model_batch, key):
            seen["model_batch"] = model_batch
            return {"native": np.asarray(1)}

        def extract_inference_results(self, *, batch, result, target_name):
            seen["extraction_batch"] = batch
            return (inference_result,)

        def extract_embeddings(self, *, result, num_tokens):
            return None

        def extract_distogram(self, *, result, num_tokens):
            return None

    def fake_featurize(fold_input, *, buckets, overflow, fixed_target):
        seen["preflight"] = (buckets, overflow, fixed_target)
        return (example,), SimpleNamespace(summary=lambda: plan_summary)

    def write_outputs(results, output_dir, job_name):
        sample_dir = Path(output_dir) / "seed-5_sample-0"
        sample_dir.mkdir(parents=True)
        prefix = f"{job_name}_seed-5_sample-0"
        (sample_dir / f"{prefix}_model.cif").touch()
        (sample_dir / f"{prefix}_summary_confidences.json").write_text("{}")

    runner = SimpleNamespace(
        ModelRunner=FakeModelRunner,
        ResultsForSeed=lambda **kwargs: SimpleNamespace(**kwargs),
        make_model_config=lambda **kwargs: SimpleNamespace(),
        write_outputs=write_outputs,
    )
    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda path: iter((FakeFoldInput(),))
    common = ModuleType("alphafold3.common")
    common.folding_input = folding
    package = ModuleType("alphafold3")
    monkeypatch.setitem(sys.modules, "alphafold3", package)
    monkeypatch.setitem(sys.modules, "alphafold3.common", common)
    monkeypatch.setitem(sys.modules, "alphafold3.common.folding_input", folding)
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._runner_path", lambda options: Path("runner")
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._load_runner", lambda path: runner
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._featurize_padded_structure",
        fake_featurize,
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._tokamax_kernel_fallback",
        lambda strategy: nullcontext(),
    )
    selected_device = jax.devices()[0]
    monkeypatch.setattr(
        "jax.local_devices", lambda backend=None: (selected_device,)
    )
    padding = PaddingConfig(tokens=512, overflow="exact")
    request = dataclasses.replace(
        _request(tmp_path, "alphafold3"),
        cache_dir=None,
        use_compile_cache=False,
        padding=padding,
    )

    result = AlphaFold3Backend().predict(request)

    assert seen["preflight"] == ((512,), "exact", True)
    assert bool(seen["model_batch"][_PREFIX_STABLE_NOISE_FEATURE])
    assert _PREFIX_STABLE_NOISE_FEATURE not in seen["extraction_batch"]
    assert result.shape_profile == plan_summary
    assert result.raw["padding"] == plan_summary
    assert result.samples[0].structure_path is not None


def test_alphafold3_jobs_receive_unique_common_sample_slots(tmp_path: Path) -> None:
    """Native AF3 files may contain several jobs whose local samples restart at 0."""
    inference_result = SimpleNamespace(metadata={"ranking_score": 0.91})
    results = (SimpleNamespace(seed=5, inference_results=(inference_result,)),)
    samples = []
    for job_name in ("first", "second"):
        job_dir = tmp_path / job_name
        sample_dir = job_dir / "seed-5_sample-0"
        sample_dir.mkdir(parents=True)
        prefix = f"{job_name}_seed-5_sample-0"
        (sample_dir / f"{prefix}_model.cif").write_text("data_result\n#\n")
        (sample_dir / f"{prefix}_summary_confidences.json").write_text("{}")
        samples.extend(
            _samples(
                results,
                job_dir,
                job_name,
                sample_offset=len(samples),
            )
        )

    assert [sample.metadata["sample"] for sample in samples] == [0, 1]
    assert [sample.metadata["job"] for sample in samples] == ["first", "second"]


def test_alphafold3_source_path_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="runner not found"):
        _runner_path({"source": tmp_path})


def test_a_failed_runner_import_is_removed_before_retry(
    tmp_path: Path, monkeypatch
) -> None:
    runner = tmp_path / "broken_runner.py"
    runner.write_text("PARTIAL = True\nraise RuntimeError('broken runner')\n")
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._settle_absl_flags", lambda: None
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="broken runner"):
            _load_runner(runner)


def test_runner_cache_is_scoped_to_the_resolved_source_path(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("IDENTITY = 'first'\n")
    second.write_text("IDENTITY = 'second'\n")
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._settle_absl_flags", lambda: None
    )

    assert _load_runner(first).IDENTITY == "first"
    assert _load_runner(second).IDENTITY == "second"


def test_runner_cache_changes_when_the_file_changes(
    tmp_path: Path, monkeypatch
) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("IDENTITY = 'first'\n")
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._settle_absl_flags", lambda: None
    )

    assert _load_runner(runner).IDENTITY == "first"
    runner.write_text("IDENTITY = 'second'\n")

    assert _load_runner(runner).IDENTITY == "second"


def test_an_installed_alphafold3_cannot_change_the_default_runner(
    tmp_path: Path, monkeypatch
) -> None:
    """An unimported site-package never changes FoldJAX's provenance."""
    import importlib.util

    from foldjax.models.alphafold3 import build

    installed = tmp_path / "site-packages" / "alphafold3" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.touch()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(installed)),
    )
    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    selected = []
    monkeypatch.setattr(build, "register_runtime", lambda: selected.append("managed"))

    assert _runner_path({}) == VENDORED_RUNNER
    assert selected == ["managed"]
    assert VENDORED_RUNNER.is_file()


def test_an_already_imported_external_alphafold3_is_not_replaced(
    tmp_path: Path, monkeypatch
) -> None:
    installed = tmp_path / "site-packages" / "alphafold3" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.touch()
    package = ModuleType("alphafold3")
    package.__file__ = str(installed)
    monkeypatch.setitem(sys.modules, "alphafold3", package)

    with pytest.raises(RuntimeError, match="already imported"):
        _runner_path({})


def test_a_managed_runtime_remains_identifiable_after_home_changes(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import _upstream, build

    old_package = tmp_path / "home-a/runtime/alphafold3/key/alphafold3"
    new_package = tmp_path / "home-b/runtime/alphafold3/key/alphafold3"
    old_package.mkdir(parents=True)
    new_package.mkdir(parents=True)
    module = ModuleType("alphafold3")
    module.__file__ = str(old_package / "__init__.py")
    setattr(module, _upstream._MANAGED_PACKAGE_ATTR, str(old_package))
    monkeypatch.setitem(sys.modules, "alphafold3", module)
    monkeypatch.setattr(build, "is_managed_origin", lambda origin: False)
    monkeypatch.setattr(build, "active_package", lambda: new_package)

    with pytest.raises(RuntimeError, match="different FoldJAX-managed"):
        _upstream.ensure_registered()


def test_runner_resolution_rechecks_a_loaded_managed_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import _upstream, build

    old_package = tmp_path / "old/runtime/alphafold3/key/alphafold3"
    new_package = tmp_path / "new/runtime/alphafold3/key/alphafold3"
    old_package.mkdir(parents=True)
    new_package.mkdir(parents=True)
    module = ModuleType("alphafold3")
    module.__file__ = str(old_package / "__init__.py")
    setattr(module, _upstream._MANAGED_PACKAGE_ATTR, str(old_package))
    monkeypatch.setitem(sys.modules, "alphafold3", module)
    monkeypatch.setattr(build, "active_package", lambda: new_package)
    monkeypatch.setattr(build, "runtime_package", lambda: new_package)

    with pytest.raises(RuntimeError, match="different FoldJAX-managed"):
        _runner_path({})


def test_runner_resolution_rejects_an_explicit_checkout_package_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib.util

    checkout = tmp_path / "checkout"
    (checkout / "src/alphafold3").mkdir(parents=True)
    (checkout / "src/alphafold3/__init__.py").touch()
    (checkout / "run_alphafold.py").touch()
    other = tmp_path / "other/alphafold3/__init__.py"
    other.parent.mkdir(parents=True)
    other.touch()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(other)),
    )

    with pytest.raises(RuntimeError, match="does not match the imported package"):
        _runner_path({"source": checkout})


def test_registered_runtime_restores_its_libcifpp_environment(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.alphafold3 import _upstream, build

    package = tmp_path / "runtime/alphafold3/key/alphafold3"
    data = package.parent / "share/libcifpp"
    package.mkdir(parents=True)
    data.mkdir(parents=True)
    module = ModuleType("alphafold3")
    module.__file__ = str(package / "__init__.py")
    setattr(module, _upstream._MANAGED_PACKAGE_ATTR, str(package))
    monkeypatch.setitem(sys.modules, "alphafold3", module)
    monkeypatch.setattr(build, "active_package", lambda: package)
    monkeypatch.delenv("LIBCIFPP_DATA_DIR", raising=False)

    _upstream.ensure_registered()

    assert os.environ["LIBCIFPP_DATA_DIR"] == str(data)


def test_a_registered_foldjax_alphafold3_runtime_is_rechecked(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed CCD build may leave the module registered but not complete."""
    from foldjax.models.alphafold3 import build

    managed = tmp_path / "runtime" / "alphafold3" / "__init__.py"
    managed.parent.mkdir(parents=True)
    managed.touch()
    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    monkeypatch.setattr(build, "is_managed_origin", lambda origin: True)
    calls = []
    monkeypatch.setattr(build, "register_runtime", lambda: calls.append("registered"))

    assert _runner_path({}) == VENDORED_RUNNER
    assert calls == ["registered"]


def test_an_unselected_checkout_cannot_outrank_the_managed_runner(
    tmp_path: Path, monkeypatch
) -> None:
    """Discovery alone is not consent to run arbitrary checkout code."""
    import importlib.util

    from foldjax.models.alphafold3 import build

    checkout = tmp_path / "AlphaFold3"
    origin = checkout / "src" / "alphafold3" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.touch()
    (checkout / "run_alphafold.py").touch()
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(origin))
    )
    selected = []
    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    monkeypatch.setattr(build, "register_runtime", lambda: selected.append("managed"))

    assert _runner_path({}) == VENDORED_RUNNER
    assert selected == ["managed"]


def test_alphafold3_source_environment_does_not_opt_into_external_code(
    tmp_path: Path, monkeypatch
) -> None:
    """External provenance requires the request option, not ambient state."""
    from foldjax.models.alphafold3 import build

    checkout = tmp_path / "AlphaFold3"
    checkout.mkdir()
    monkeypatch.setenv("ALPHAFOLD3_SOURCE", str(checkout))
    monkeypatch.delitem(sys.modules, "alphafold3", raising=False)
    selected = []
    monkeypatch.setattr(build, "register_runtime", lambda: selected.append("managed"))

    assert _runner_path({}) == VENDORED_RUNNER
    assert selected == ["managed"]


def test_alphafold3_import_and_default_resolution_are_torch_free() -> None:
    script = r"""
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"AlphaFold 3 runtime resolution imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from foldjax.backends import alphafold3
from foldjax.models.alphafold3 import build

assert alphafold3.VENDORED_RUNNER.is_file()
assert build.source_package().is_dir()
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_the_vendored_alphafold3_runner_matches_upstream() -> None:
    """Drift guard: an AF3 update must not leave the carried copy behind.

    Skipped without a checkout, like the other upstream comparisons -- the point
    is to notice a divergence on the machine that does the updating.
    """
    checkout = Path(__file__).resolve().parents[2] / "AlphaFold3" / "run_alphafold.py"
    if not checkout.is_file():
        pytest.skip(f"no AlphaFold 3 checkout at {checkout}")
    assert VENDORED_RUNNER.read_bytes() == checkout.read_bytes(), (
        "vendored run_alphafold.py differs from the checkout; re-copy it and "
        "update the commit in _alphafold3_upstream/NOTICE"
    )


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


def test_openfold3_backend_planning_does_not_import_the_model_runtime() -> None:
    """Cache planning stays cheap and must not initialize JAX by import."""

    script = r"""
import sys
from foldjax.backends.openfold3 import OpenFold3Backend

OpenFold3Backend().capabilities()
assert "jax" not in sys.modules
assert not any(
    name == "foldjax.models.openfold3"
    or name.startswith("foldjax.models.openfold3.")
    for name in sys.modules
), sorted(name for name in sys.modules if "openfold3" in name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_openfold3_capabilities_separate_raw_and_feature_runtime_needs() -> None:
    capabilities = OpenFold3Backend().capabilities()
    requirements = capabilities.input_requirements

    archive = requirements["openfold3-features"]
    assert archive.prediction_runtime == "jax"
    assert archive.preprocessing_runtime == "precomputed"
    assert archive.required_extras == ()
    assert archive.requires_torch is False

    for input_format in ("native", "openfold3", "foldjax"):
        raw = requirements[input_format]
        assert raw.prediction_runtime == "jax"
        assert raw.preprocessing_runtime == "jax"
        assert raw.required_extras == ("openfold3-preprocess",)
        assert raw.requires_torch is False


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_openfold3_no_compile_requires_a_real_boolean(value) -> None:
    with pytest.raises(ValueError, match="no_compile must be a boolean"):
        _compile_enabled({"no_compile": value})


def test_openfold3_no_compile_boolean_is_consumed() -> None:
    eager = {"no_compile": True}
    compiled = {"no_compile": False}
    assert _compile_enabled(eager) is False and eager == {}
    assert _compile_enabled(compiled) is True and compiled == {}


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_openfold3_all_arrays_requires_a_real_boolean(value) -> None:
    with pytest.raises(ValueError, match="all_arrays must be a boolean"):
        OpenFold3Backend().validate_native_options({"all_arrays": value})


@pytest.mark.parametrize(
    ("session_runs", "invalidate_between", "expected_loads"),
    [(1, False, 1), (2, False, 1), (2, True, 2)],
    ids=["scalar", "session", "invalidated-session"],
)
@pytest.mark.parametrize("eager", [False, True], ids=["compiled", "eager"])
@pytest.mark.parametrize("num_samples", [5, 10], ids=["released-s5", "raised-s10"])
@pytest.mark.parametrize(
    ("all_arrays", "stop_after"),
    [
        (False, "full"),
        (True, "full"),
        (False, "trunk"),
        (True, "trunk"),
    ],
    ids=["managed-default", "all-arrays", "trunk-default", "trunk-all-arrays"],
)
def test_openfold3_backend_passes_normalized_static_chain_count(
    tmp_path: Path,
    monkeypatch,
    eager: bool,
    num_samples: int,
    all_arrays: bool,
    stop_after: str,
    session_runs: int,
    invalidate_between: bool,
    expected_loads: int,
) -> None:
    from foldjax.models.openfold3.data import (
        has_atomized_tokens,
        normalize_asym_ids,
    )

    features = {
        "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "atom_mask": np.asarray([[1, 1]], dtype=np.float32),
        "asym_id": np.asarray([[5, 20, 999]], dtype=np.int64),
        "is_atomized": np.asarray([[0, 0, 1]], dtype=np.int32),
        "_foldjax_zero_template_pair_features": np.zeros((), dtype=np.float32),
        "_foldjax_compact_msa": np.zeros((), dtype=np.float32),
        "_foldjax_msa_indices": np.asarray([[[0, 1, 2]]], dtype=np.uint8),
    }
    prediction = SimpleNamespace(single=np.zeros((3, 1), dtype=np.float32))
    output_metadata = object()
    seen: dict[str, object] = {}
    checkpoint_state = {"checkpoint": object()}
    checkpoint_events: list[object] = []
    scores = tmp_path / "scores.json"
    scores.write_text('{"samples": []}')

    def fake_predict(key, batch, params, config, table, *, n_chain=None):
        seen.update(
            n_chain=n_chain,
            asym_id=np.asarray(batch["asym_id"]),
            compact_marker=batch.get("_foldjax_zero_template_pair_features"),
            compact_msa_indices=batch.get("_foldjax_msa_indices"),
        )
        return prediction

    def fake_compile(config, table, *, n_chain=None, **compile_options):
        seen["compile_options"] = compile_options

        def compiled(key, batch, params):
            seen.update(
                n_chain=n_chain,
                asym_id=np.asarray(batch["asym_id"]),
                compact_marker=batch.get("_foldjax_zero_template_pair_features"),
                compact_msa_indices=batch.get("_foldjax_msa_indices"),
            )
            return prediction

        return compiled

    def fake_write(*args, **kwargs):
        seen["output_metadata"] = kwargs.get("output_metadata")
        seen["writer_array_budget"] = kwargs.get("max_array_bytes")
        return {"structures": (), "scores": scores}

    def fake_featurize(*args, **kwargs):
        seen["preprocess_msa_depth"] = kwargs.get("msa_depth")
        seen["compact_empty_template_pairs"] = kwargs.get(
            "compact_empty_template_pairs"
        )
        seen["compact_msa"] = kwargs.get("compact_msa")
        return features, output_metadata

    def fake_released_config(**kwargs):
        seen["has_atomized_tokens"] = kwargs["has_atomized_tokens"]
        seen["config_array_budget"] = kwargs["max_array_bytes"]
        seen["config_num_samples"] = kwargs.get("num_samples", 5)
        seen["config_per_sample_token_cutoff"] = kwargs.get(
            "per_sample_token_cutoff"
        )
        seen["config_per_sample_token_cutoff_present"] = (
            "per_sample_token_cutoff" in kwargs
        )
        return SimpleNamespace(msa_depth=1024)

    def unexpected_compaction(batch):
        raise AssertionError("precompacted raw templates were compacted twice")

    def load_checkpoint(path):
        checkpoint_events.append("load")
        return checkpoint_state

    def resolve_model_prefix(state, prefix=None):
        checkpoint_events.append(("resolve", state, prefix))
        return "resolved"

    def prune_sample_diffusion_aliases(state, *, prefix):
        checkpoint_events.append(("prune", state, prefix))
        return 0

    def map_inference_params(state, prefix):
        checkpoint_events.append(("map", state, prefix))
        return object()

    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            MODEL_FEATURES=("token_mask", "atom_mask", "asym_id", "is_atomized"),
            PRIVATE_MODEL_FEATURES=(
                "_foldjax_zero_template_pair_features",
                "_foldjax_compact_msa",
                "_foldjax_msa_indices",
            ),
            COMPACT_MSA_PRIVATE_FEATURES=(
                "_foldjax_compact_msa",
                "_foldjax_msa_indices",
            ),
            featurize_query_with_metadata=fake_featurize,
            subsample_msa_rows=lambda batch, depth: batch,
            collapse_identical_templates=lambda batch: batch,
            compact_zero_template_pair_features=unexpected_compaction,
            compact_msa_features=unexpected_compaction,
            has_compact_zero_template_pair_features=lambda batch: (
                "_foldjax_zero_template_pair_features" in batch
            ),
            has_compact_msa_features=lambda batch: (
                "_foldjax_compact_msa" in batch
                and "_foldjax_msa_indices" in batch
                and "msa" not in batch
            ),
            has_atomized_tokens=has_atomized_tokens,
            normalize_asym_ids=normalize_asym_ids,
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            released_config=fake_released_config,
            compile_predict=fake_compile,
            predict=fake_predict,
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(
            write_prediction_outputs=fake_write
        ),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: object()
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=load_checkpoint
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=resolve_model_prefix,
            prune_sample_diffusion_aliases=prune_sample_diffusion_aliases,
            map_inference_params=map_inference_params,
        ),
        "foldjax.models.openfold3.compilation": SimpleNamespace(
            enable_compilation_cache=lambda path: path
        ),
        "jax": SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
    }
    monkeypatch.setattr(
        "foldjax.backends.openfold3.import_module", lambda name: modules[name]
    )
    native_options = {}
    if eager:
        native_options["no_compile"] = True
    if all_arrays:
        native_options["all_arrays"] = True
    request_changes: dict[str, object] = {
        "stop_after": stop_after,
        "num_samples": num_samples,
    }
    if stop_after == "trunk":
        request_changes["representations"] = ("single",)
    request = dataclasses.replace(
        _request(tmp_path, "openfold3", **native_options), **request_changes
    )

    backend = OpenFold3Backend()
    if session_runs == 1:
        backend.predict(request)
    else:
        second = dataclasses.replace(
            request,
            seed=6,
            output_dir=tmp_path / "out-second",
        )
        with backend.session((request, second)):
            backend.predict(request)
            if invalidate_between:
                backend.invalidate_session()
            backend.predict(second)

    assert seen["n_chain"] == 2
    assert seen["has_atomized_tokens"] is False
    expected_array_budget = None if all_arrays and stop_after == "full" else 0
    assert seen["config_array_budget"] == expected_array_budget
    assert seen["config_num_samples"] == num_samples
    assert seen["config_per_sample_token_cutoff"] is None
    assert seen["config_per_sample_token_cutoff_present"] is False
    if stop_after == "full":
        assert seen["writer_array_budget"] == expected_array_budget
        assert seen["output_metadata"] is output_metadata
    else:
        assert "writer_array_budget" not in seen
    assert seen["preprocess_msa_depth"] == 1024
    assert seen["compact_empty_template_pairs"] is True
    assert seen["compact_msa"] is True
    np.testing.assert_array_equal(seen["compact_marker"], np.zeros((), np.float32))
    np.testing.assert_array_equal(
        seen["compact_msa_indices"], np.asarray([[[0, 1, 2]]], dtype=np.uint8)
    )
    if not eager:
        assert seen["compile_options"] == {
            "triangle_kernel": None,
            "cache_scope": str(tmp_path / "cache"),
        }
    generation_events = [
        "load",
        ("resolve", checkpoint_state, None),
        ("prune", checkpoint_state, "resolved"),
        ("map", checkpoint_state, "resolved"),
    ]
    assert checkpoint_events == generation_events * expected_loads
    np.testing.assert_array_equal(seen["asym_id"], [[0, 1, 0]])


def test_openfold3_backend_executes_the_lazy_padding_noise_mask_path(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.openfold3.data import has_atomized_tokens

    features = {
        "token_mask": np.asarray([[1, 1]], dtype=np.float32),
        "atom_mask": np.asarray([[1, 1]], dtype=np.float32),
        "asym_id": np.asarray([[0, 0]], dtype=np.int64),
        "is_atomized": np.asarray([[0, 0]], dtype=np.int32),
        "msa_mask": np.ones((1, 1, 2), dtype=np.float32),
        "template_backbone_frame_mask": np.ones((1, 1, 2), dtype=np.float32),
        "template_pseudo_beta_mask": np.ones((1, 1, 2), dtype=np.float32),
    }
    prediction = object()
    output_metadata = object()
    seen: dict[str, object] = {}
    scores = tmp_path / "scores.json"
    scores.write_text('{"samples": []}')

    def pad_features(batch, *, n_token, n_atom, n_msa, n_templates):
        seen["targets"] = (n_token, n_atom, n_msa, n_templates)
        return {
            "token_mask": np.asarray([[1, 1, 0, 0]], dtype=np.float32),
            "atom_mask": np.asarray([[1, 1, 0, 0]], dtype=np.float32),
            "asym_id": np.asarray([[0, 0, 0, 0]], dtype=np.int64),
            # Dirty padding metadata must not select the atomized graph.
            "is_atomized": np.asarray([[0, 0, 1, 1]], dtype=np.int32),
            "msa_mask": np.pad(batch["msa_mask"], ((0, 0), (0, 1), (0, 2))),
            "template_backbone_frame_mask": np.pad(
                batch["template_backbone_frame_mask"],
                ((0, 0), (0, 1), (0, 2)),
            ),
            "template_pseudo_beta_mask": np.pad(
                batch["template_pseudo_beta_mask"],
                ((0, 0), (0, 1), (0, 2)),
            ),
        }

    def fake_compile(config, table, *, n_chain=None, **compile_options):
        seen["compile_options"] = compile_options

        def compiled(key, batch, params, *, noise_mask=None):
            seen["noise_mask"] = np.asarray(noise_mask)
            return prediction

        return compiled

    def compact_zero_template_pair_features(batch):
        seen["compact_shape"] = batch["token_mask"].shape
        return batch

    def fake_write(*args, **kwargs):
        return {"structures": (), "scores": scores}

    modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(
            featurize_query_with_metadata=lambda *args, **kwargs: (
                features,
                output_metadata,
            ),
            subsample_msa_rows=lambda batch, depth: batch,
            collapse_identical_templates=lambda batch: batch,
            compact_zero_template_pair_features=compact_zero_template_pair_features,
            has_atomized_tokens=has_atomized_tokens,
            pad_features=pad_features,
            normalize_asym_ids=lambda batch: (batch, 1),
        ),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            released_config=lambda **kwargs: SimpleNamespace(
                msa_depth=1024,
                num_samples=2,
            ),
            compile_predict=fake_compile,
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(
            write_prediction_outputs=fake_write
        ),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: object()
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda path: {}
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda state, prefix=None: "",
            prune_sample_diffusion_aliases=lambda state, *, prefix: 0,
            map_inference_params=lambda state, prefix: object()
        ),
        "foldjax.models.openfold3.compilation": SimpleNamespace(
            enable_compilation_cache=lambda path: path
        ),
        "jax": SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)),
        "jax.numpy": SimpleNamespace(
            asarray=np.asarray,
            broadcast_to=np.broadcast_to,
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.openfold3.import_module", lambda name: modules[name]
    )
    request = dataclasses.replace(
        _request(tmp_path, "openfold3"),
        padding=PaddingConfig(tokens=4, atoms=4, msa=2, templates=2),
    )

    result = OpenFold3Backend().predict(request)

    assert seen["targets"] == (4, 4, 2, 2)
    assert seen["compact_shape"] == (1, 4)
    assert seen["compile_options"] == {
        "triangle_kernel": None,
        "cache_scope": str(tmp_path / "cache"),
    }
    np.testing.assert_array_equal(
        seen["noise_mask"],
        [[True, True, False, False], [True, True, False, False]],
    )
    assert result.shape_profile["target"] == {
        "tokens": 4,
        "atoms": 4,
        "msa": 2,
        "templates": 2,
    }


def test_openfold3_featurization_names_the_extra_it_needs(monkeypatch) -> None:
    """Featurization names its chemistry extra without implying Torch is needed.

    The pipeline is vendored, so this no longer fails for want of a checkout --
    and its FoldJAX path is NumPy/JAX-only. A bare "No module named ..." would
    not say which chemistry extra supplies the missing package, so the message
    names both the extra and the Torch-free boundary.

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
    assert "PyTorch and Lightning are not required" in message


def test_openfold3_feature_archive_loads_without_importing_torch(
    tmp_path: Path, monkeypatch
) -> None:
    from foldjax.models.openfold3 import data
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )
    from tests.models.openfold3.feature_fixture import minimal_features

    features = minimal_features()
    table = RepresentativeAtomTable(
        *(np.zeros(32, dtype=np.float32) for _ in RepresentativeAtomTable._fields)
    )
    archive = data.save_features(
        features, tmp_path / "features.npz", representative_atoms=table
    )
    request = _request(tmp_path, "openfold3")
    request = dataclasses.replace(
        request, input=archive, input_format="openfold3-features"
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("feature inference path imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    loaded, loaded_table = _features_and_chemistry(
        request, data=data, query_id=None, ccd_file_path=None
    )

    assert set(loaded) == set(data.MODEL_FEATURES)
    assert loaded_table is not None


def test_openfold3_cold_inference_import_closure_is_torch_free() -> None:
    """Guard before the first FoldJAX import, then traverse the real closure."""
    script = r"""
import builtins
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"inference closure imported {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models.openfold3 import data, inference, output
from foldjax.models.openfold3.bridge import checkpoint, chemistry, torch_mapping

assert OpenFold3Backend().name == "openfold3"
assert data.MODEL_FEATURES
assert inference.RELEASED_BLOCK_COUNTS
assert output.RESIDUE_NAMES
assert checkpoint is not None and torch_mapping is not None
assert chemistry.representative_atom_table().cb_mask.shape
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
