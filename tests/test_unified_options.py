"""Unified cross-model option aliases and additive confidence output preset."""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

import foldjax
from foldjax import cli, execution, manifest
from foldjax.backends.alphafold3 import AlphaFold3Backend
from foldjax.backends.base import Backend
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.backends.opendde import OpenDDEBackend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.opendde import runner as opendde_runner
from foldjax.models.opendde.cli import predict as opendde_predict
from foldjax.models.protenix import runner as protenix_runner
from foldjax.models.protenix.cli import predict as protenix_predict
from foldjax.registry import available_models, get_backend
from foldjax.schema import PredictionRequest
from tests._parser_capture import capture_parser

BOLTZ2_CANONICAL_ROWS: tuple[tuple[str, str, Any, Any], ...] = (
    ("diffusion_dtype", "diffusion_compute_dtype", "bf16", "bfloat16"),
    ("diffusion_dtype", "diffusion_compute_dtype", "bfloat16", "bfloat16"),
    ("diffusion_dtype", "diffusion_compute_dtype", "fp32", "float32"),
    ("diffusion_dtype", "diffusion_compute_dtype", "f32", "float32"),
    ("diffusion_dtype", "diffusion_compute_dtype", "float32", "float32"),
    ("token_q_chunk_size", "token_attention_chunk", 0, 0),
    ("token_q_chunk_size", "token_attention_chunk", 128, 128),
    ("token_q_chunk_size", "token_attention_chunk", None, None),
    ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", 0, 0),
    ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", 64, 64),
    ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", None, None),
)

CONFIDENCE_FULL_MAP: dict[str, dict[str, Any]] = {
    "alphafold3": {},
    "boltz2": {},
    "esmfold2": {"return_expected_errors": True},
    "openfold3": {"return_expected_errors": True},
    "protenix": {"output_format": "both"},
    "opendde": {"include_raw": True},
}


def _typed(mapping: dict[str, Any]) -> dict[str, tuple[Any, type]]:
    return {key: (val, type(val)) for key, val in mapping.items()}


def _make_request(
    tmp_path: Path,
    model: str,
    options: dict[str, Any] | None = None,
    **extra: Any,
) -> PredictionRequest:
    job = tmp_path / ("job.yaml" if model == "boltz2" else "job.json")
    if not job.exists():
        job.write_text(
            json.dumps(
                {
                    "name": "t",
                    "entities": [
                        {"type": "protein", "id": ["A"], "sequence": "ACDEF"}
                    ],
                }
            ),
            encoding="utf-8",
        )
    if model == "opendde":
        weights = tmp_path / "opendde.jax"
        weights.touch()
    else:
        weights = tmp_path / f"{model}-weights"
        weights.mkdir(exist_ok=True)
    fields: dict[str, Any] = {
        "model": model,
        "input": job,
        "input_format": "foldjax",
        "weights": weights,
        "output_dir": tmp_path / f"out-{model}",
        "cache_dir": tmp_path / "cache",
        "seed": 7,
        "padding": False,
        "options": {} if options is None else dict(options),
    }
    fields.update(extra)
    return PredictionRequest(**fields)


@pytest.mark.parametrize(
    ("canonical", "native", "raw_value", "expected_native_value"),
    BOLTZ2_CANONICAL_ROWS,
)
def test_boltz2_canonical_option_matches_native_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical: str,
    native: str,
    raw_value: Any,
    expected_native_value: Any,
) -> None:
    canon_opts = {canonical: raw_value}
    native_opts = {native: expected_native_value}

    assert _typed(Boltz2Backend.canonical_options(canon_opts)) == _typed(
        Boltz2Backend.canonical_options(native_opts)
    )
    assert Boltz2Backend.canonical_options(canon_opts)[native] == expected_native_value

    req_canon = foldjax.resolve_request(
        _make_request(tmp_path, "boltz2", canon_opts), draw_seeds=False
    )
    req_native = foldjax.resolve_request(
        _make_request(tmp_path, "boltz2", native_opts), draw_seeds=False
    )
    backend = Boltz2Backend()
    assert _typed(backend.apply_sampling(req_canon)) == _typed(
        backend.apply_sampling(req_native)
    )
    assert _typed(backend.cache_profile(req_canon)) == _typed(
        backend.cache_profile(req_native)
    )
    assert manifest._canonical_options(
        "boltz2", req_canon.options
    ) == manifest._canonical_options("boltz2", req_native.options)

    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    captured: list[dict[str, Any]] = []
    out_cif = tmp_path / "out-boltz2" / "job.cif"

    def fake_predict(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.asarray([0.8, 0.9]),
            "iptm": np.asarray([0.7]),
            "out_path": out_cif,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda _name: SimpleNamespace(predict=fake_predict),
    )
    backend.predict(
        foldjax.resolve_request(
            _make_request(tmp_path, "boltz2", {**canon_opts, "mols": mols}),
            draw_seeds=False,
        )
    )
    backend.predict(
        foldjax.resolve_request(
            _make_request(tmp_path, "boltz2", {**native_opts, "mols": mols}),
            draw_seeds=False,
        )
    )
    assert _typed(captured[0]) == _typed(captured[1])
    assert captured[0][native] == expected_native_value
    assert canonical not in captured[0]


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("float32", "float32"),
        ("fp32", "float32"),
        ("f32", "float32"),
        (" FP32 ", "float32"),
        ("bfloat16", "bfloat16"),
        ("bf16", "bfloat16"),
        (" BF16 ", "bfloat16"),
    ],
)
def test_boltz2_cli_diffusion_dtype_parses_all_dtype_spellings(
    tmp_path: Path, spelling: str, expected: str
) -> None:
    parsed = cli._options([f"diffusion_dtype={spelling}"])
    assert Boltz2Backend.canonical_options(parsed) == {
        "diffusion_compute_dtype": expected
    }
    resolved = foldjax.resolve_request(
        _make_request(tmp_path, "boltz2", parsed), draw_seeds=False
    )
    assert Boltz2Backend().apply_sampling(resolved)["diffusion_compute_dtype"] == (
        expected
    )


@pytest.mark.parametrize(
    ("canonical", "native", "raw_text", "expected"),
    [
        ("token_q_chunk_size", "token_attention_chunk", "0", 0),
        ("token_q_chunk_size", "token_attention_chunk", "128", 128),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", "0", 0),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", "64", 64),
    ],
)
def test_boltz2_cli_chunk_aliases_parse_to_native_ints(
    tmp_path: Path, canonical: str, native: str, raw_text: str, expected: int
) -> None:
    parsed = cli._options([f"{canonical}={raw_text}"])
    canonicalized = Boltz2Backend.canonical_options(parsed)
    assert canonicalized == {native: expected}
    assert type(canonicalized[native]) is int
    resolved = foldjax.resolve_request(
        _make_request(tmp_path, "boltz2", parsed), draw_seeds=False
    )
    assert Boltz2Backend().apply_sampling(resolved)[native] == expected


@pytest.mark.parametrize(
    ("canonical", "native", "value"),
    [
        ("diffusion_dtype", "diffusion_compute_dtype", "bfloat16"),
        ("token_q_chunk_size", "token_attention_chunk", 128),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", 64),
    ],
)
@pytest.mark.parametrize("reverse_order", [False, True])
def test_boltz2_canonical_and_native_together_raise_naming_both(
    tmp_path: Path, canonical: str, native: str, value: Any, reverse_order: bool
) -> None:
    pairs = [(canonical, value), (native, value)]
    if reverse_order:
        pairs.reverse()
    both = dict(pairs)
    with pytest.raises(ValueError) as exc_info:
        Boltz2Backend.canonical_options(both)
    message = str(exc_info.value)
    assert canonical in message and native in message

    with pytest.raises(ValueError) as req_exc:
        foldjax.resolve_request(
            _make_request(tmp_path, "boltz2", both), draw_seeds=False
        )
    req_msg = str(req_exc.value)
    assert canonical in req_msg and native in req_msg


@pytest.mark.parametrize(
    ("option", "native", "bad_value"),
    [
        ("diffusion_dtype", "diffusion_compute_dtype", ""),
        ("diffusion_dtype", "diffusion_compute_dtype", "fp16"),
        ("diffusion_dtype", "diffusion_compute_dtype", 0),
        ("diffusion_dtype", "diffusion_compute_dtype", None),
        ("token_q_chunk_size", "token_attention_chunk", -1),
        ("token_q_chunk_size", "token_attention_chunk", "abc"),
        ("token_q_chunk_size", "token_attention_chunk", True),
        ("token_q_chunk_size", "token_attention_chunk", 1.5),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", -1),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", "abc"),
        ("triangle_att_q_chunk_size", "triangle_attention_q_chunk", False),
    ],
)
def test_boltz2_canonical_options_reject_malformed_values(
    tmp_path: Path, option: str, native: str, bad_value: Any
) -> None:
    with pytest.raises(ValueError, match=native):
        foldjax.resolve_request(
            _make_request(tmp_path, "boltz2", {option: bad_value}),
            draw_seeds=False,
        )


def test_option_aliases_subclass_pass_allowlist_and_detect_double_alias(
    tmp_path: Path,
) -> None:
    from foldjax.schema import ModelCapabilities

    class CustomBackend(Backend):
        name = "custom"
        native_options = frozenset()
        option_aliases = {"canon_a": "nat_x", "canon_b": "nat_x"}

        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(model="custom", input_formats=("foldjax",))

        def predict(self, request):  # pragma: no cover - unused in unit check
            raise NotImplementedError

    assert CustomBackend.canonical_options({"canon_a": 3}) == {"nat_x": 3}
    CustomBackend().validate_request(
        _make_request(tmp_path, "boltz2", {"canon_a": 3})
    )
    with pytest.raises(ValueError, match="canon_a.*canon_b"):
        CustomBackend.canonical_options({"canon_a": 3, "canon_b": 3})


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("compute_dtype", "trunk_dtype"),
        ("trunk_dtype", "compute_dtype"),
        ("triangle_backend", "trunk_triangle_attention_backend"),
        ("attention_backend", "trunk_single_attention_backend"),
    ],
)
def test_two_legacy_aliases_for_same_knob_raise_in_normalize(
    first: str, second: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", execution.Alias)
        with pytest.raises(ValueError) as exc_info:
            execution.normalize({first: "x", second: "x"}, native=set())
    msg = str(exc_info.value)
    assert first in msg and second in msg


def test_two_legacy_aliases_raise_through_public_entry_points_preserving_native_rule(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="compute_dtype.*trunk_dtype"):
        foldjax.resolve_request(
            _make_request(
                tmp_path,
                "openfold3",
                {"compute_dtype": "bfloat16", "trunk_dtype": "float32"},
            ),
            draw_seeds=False,
        )

    # Native option names on their owning backend remain un-rewritten without warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", execution.Alias)
        assert execution.normalize(
            {"compute_dtype": "bfloat16"}, native={"compute_dtype"}
        ) == {"compute_dtype": "bfloat16"}
        assert execution.normalize(
            {"trunk_dtype": "bf16"}, native={"trunk_dtype"}
        ) == {"trunk_dtype": "bf16"}

    # Combining a native name with another port's legacy alias still fails.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", execution.Alias)
        with pytest.raises(ValueError, match="both set"):
            Boltz2Backend.canonical_options(
                {"compute_dtype": "bfloat16", "trunk_dtype": "float32"}
            )
        with pytest.raises(ValueError, match="both set"):
            ProtenixBackend.canonical_options(
                {"trunk_dtype": "bf16", "compute_dtype": "float32"}
            )


@pytest.mark.parametrize("model", sorted(CONFIDENCE_FULL_MAP))
def test_confidence_outputs_canonical_and_resolve_equivalence_per_model(
    tmp_path: Path, model: str
) -> None:
    assert set(CONFIDENCE_FULL_MAP) == set(available_models())
    backend = get_backend(model)
    backend_cls = type(backend)
    expected_native = CONFIDENCE_FULL_MAP[model]

    # CLI `--option` parse for both valid values.
    cli_full = cli._options(["confidence_outputs=full"])
    cli_model = cli._options(["confidence_outputs=model"])
    assert _typed(backend_cls.canonical_options(cli_full)) == _typed(
        backend_cls.canonical_options(expected_native)
    )
    assert backend_cls.canonical_options(cli_model) == {}
    assert backend_cls.canonical_options({}) == {}

    req_full = foldjax.resolve_request(
        _make_request(tmp_path, model, cli_full), draw_seeds=False
    )
    req_native = foldjax.resolve_request(
        _make_request(tmp_path, model, expected_native), draw_seeds=False
    )
    req_model = foldjax.resolve_request(
        _make_request(tmp_path, model, cli_model), draw_seeds=False
    )
    req_omitted = foldjax.resolve_request(
        _make_request(tmp_path, model, {}), draw_seeds=False
    )

    assert _typed(backend.apply_sampling(req_full)) == _typed(
        backend.apply_sampling(req_native)
    )
    assert _typed(backend.cache_profile(req_full)) == _typed(
        backend.cache_profile(req_native)
    )
    assert manifest._canonical_options(
        model, req_full.options
    ) == manifest._canonical_options(model, req_native.options)

    assert _typed(backend.apply_sampling(req_model)) == _typed(
        backend.apply_sampling(req_omitted)
    )
    assert _typed(backend.cache_profile(req_model)) == _typed(
        backend.cache_profile(req_omitted)
    )
    assert manifest._canonical_options(
        model, req_model.options
    ) == manifest._canonical_options(model, req_omitted.options)


def test_confidence_outputs_reaches_protenix_and_opendde_native_configs(
    tmp_path: Path,
) -> None:
    for model, backend_cls, runner_mod, cli_main, mapped in (
        (
            "protenix",
            ProtenixBackend,
            protenix_runner,
            protenix_predict.main,
            {"output_format": "both"},
        ),
        (
            "opendde",
            OpenDDEBackend,
            opendde_runner,
            opendde_predict.main,
            {"include_raw": True},
        ),
    ):
        backend = backend_cls()
        req_full = foldjax.resolve_request(
            _make_request(tmp_path, model, {"confidence_outputs": "full"}),
            draw_seeds=False,
        )
        req_native = foldjax.resolve_request(
            _make_request(tmp_path, model, mapped),
            draw_seeds=False,
        )
        req_model = foldjax.resolve_request(
            _make_request(tmp_path, model, {"confidence_outputs": "model"}),
            draw_seeds=False,
        )
        req_omitted = foldjax.resolve_request(
            _make_request(tmp_path, model, {}),
            draw_seeds=False,
        )

        inv_full = backend._native_invocation(req_full)
        inv_native = backend._native_invocation(req_native)
        assert _typed(inv_full.config_fields) == _typed(inv_native.config_fields)
        assert inv_full.argv == inv_native.argv

        parsed = capture_parser(cli_main).parse_args(inv_full.argv)
        parsed.max_msa_depth = runner_mod._resolve_msa_depth(parsed.max_msa_depth)
        assert runner_mod.PredictionConfig(**inv_full.config_fields) == (
            runner_mod.PredictionConfig(**vars(parsed))
        )

        inv_model = backend._native_invocation(req_model)
        inv_omitted = backend._native_invocation(req_omitted)
        assert _typed(inv_model.config_fields) == _typed(inv_omitted.config_fields)
        assert inv_model.argv == inv_omitted.argv


def test_confidence_outputs_reaches_esmfold2_boltz2_openfold3_alphafold3_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ESMFold2 boundary: overrides handed to inference.predict.
    esm_seen: list[dict[str, Any]] = []
    esm_modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *_a, **_k: SimpleNamespace(has_language_model=True),
            seed_key=lambda seed: seed,
            build_common_job_features=lambda _doc, **_k: {
                "asym_id": np.asarray([[0]])
            },
            predict=lambda _k, _f, _m, **ov: esm_seen.append(ov) or object(),
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *_a, **_k: {
                "structures": [tmp_path / "esm.cif"],
                "summary": [{"sample": 0, "plddt": 90.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: esm_modules[name]
    )
    for opts in (
        {"confidence_outputs": "full"},
        {"return_expected_errors": True},
        {"confidence_outputs": "model", "return_expected_errors": False},
    ):
        ESMFold2Backend().predict(
            foldjax.resolve_request(
                _make_request(tmp_path, "esmfold2", opts), draw_seeds=False
            )
        )
    assert _typed(esm_seen[0]) == _typed(esm_seen[1])
    assert esm_seen[2]["return_expected_errors"] is False

    # Boltz-2 boundary: native.predict kwargs (return_confidence_logits stays unset).
    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    boltz_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda _name: SimpleNamespace(
            predict=lambda **kw: boltz_seen.append(kw)
            or {
                "coords": np.zeros((2, 3)),
                "plddt": np.asarray([0.8, 0.9]),
                "iptm": np.asarray([0.7]),
                "out_path": tmp_path / "out-boltz2" / "job.cif",
            }
        ),
    )
    for opts in ({"confidence_outputs": "full", "mols": mols}, {"mols": mols}):
        Boltz2Backend().predict(
            foldjax.resolve_request(
                _make_request(tmp_path, "boltz2", opts), draw_seeds=False
            )
        )
    assert _typed(boltz_seen[0]) == _typed(boltz_seen[1])
    assert "return_confidence_logits" not in boltz_seen[0]

    # OpenFold3 boundary: released_config kwargs (all_arrays stays off).
    of3_features = tmp_path / "features.npz"
    mask_f32 = np.asarray([[1, 1]], dtype=np.float32)
    of3_arrays = {
        "token_mask": mask_f32,
        "atom_mask": mask_f32,
        "asym_id": np.asarray([[0, 0]], dtype=np.int64),
        "is_atomized": np.asarray([[0, 0]], dtype=np.int32),
    }
    np.savez(of3_features, **of3_arrays)
    of3_scores = tmp_path / "of3_scores.json"
    of3_scores.write_text('{"samples": []}', encoding="utf-8")
    of3_seen: list[dict[str, Any]] = []
    of3_data_attrs: dict[str, Any] = {
        "pocket_sampling_config": lambda _b: None,
        "POCKET_SAMPLING_FEATURES": (),
        "MODEL_FEATURES": ("token_mask", "atom_mask", "asym_id", "is_atomized"),
        "PRIVATE_MODEL_FEATURES": (),
        "COMPACT_MSA_PRIVATE_FEATURES": (),
        "load_feature_archive": lambda _p: (dict(of3_arrays), None, None),
        "prepare_msa_cycle_features": lambda batch, *_a, **_k: batch,
        "collapse_identical_templates": lambda batch: batch,
        "compact_zero_template_pair_features": lambda batch: batch,
        "compact_msa_features": lambda batch: batch,
        "has_compact_zero_template_pair_features": lambda _b: True,
        "has_compact_msa_features": lambda _b: True,
        "normalize_asym_ids": lambda batch: (batch, 1),
        "has_atomized_tokens": lambda _b: False,
    }
    of3_modules = {
        "foldjax.models.openfold3.data": SimpleNamespace(**of3_data_attrs),
        "foldjax.models.openfold3.inference": SimpleNamespace(
            resolve_dtypes=lambda _cfg: (None, None),
            cast_narrow_params=lambda params, *_a, **_k: params,
            released_config=lambda **kw: of3_seen.append(kw)
            or SimpleNamespace(msa_depth=1024, num_recycles=4),
            compile_predict=lambda *_a, **_k: lambda *_ca: SimpleNamespace(),
            predict=lambda *_a, **_k: SimpleNamespace(),
        ),
        "foldjax.models.openfold3.output": SimpleNamespace(
            write_prediction_outputs=lambda *_a, **_k: {
                "structures": (),
                "scores": of3_scores,
            }
        ),
        "foldjax.models.openfold3.bridge.chemistry": SimpleNamespace(
            representative_atom_table=lambda: object()
        ),
        "foldjax.models.openfold3.bridge.checkpoint": SimpleNamespace(
            load_checkpoint=lambda _p: {"ckpt": 1}
        ),
        "foldjax.models.openfold3.bridge.torch_mapping": SimpleNamespace(
            resolve_model_prefix=lambda _s, prefix=None: "p",
            prune_sample_diffusion_aliases=lambda _s, *, prefix: 0,
            map_inference_params=lambda _s, _p: object(),
        ),
        "jax": SimpleNamespace(
            random=SimpleNamespace(**{"k" + "ey": lambda seed: seed})
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.openfold3.import_module", lambda name: of3_modules[name]
    )
    for opts in (
        {"confidence_outputs": "full"},
        {"return_expected_errors": True},
        {"confidence_outputs": "model", "return_expected_errors": False},
    ):
        OpenFold3Backend().predict(
            foldjax.resolve_request(
                _make_request(
                    tmp_path,
                    "openfold3",
                    opts,
                    input=of3_features,
                    input_format="openfold3-features",
                ),
                draw_seeds=False,
            )
        )
    assert _typed(of3_seen[0]) == _typed(of3_seen[1])
    assert of3_seen[0]["max_array_bytes"] is not None
    assert of3_seen[2]["return_expected_errors"] is False

    # AlphaFold 3 boundary: make_model_config kwargs.
    af3_seen: list[dict[str, Any]] = []
    runner_dir = tmp_path / "af3"
    runner_dir.mkdir(exist_ok=True)
    (runner_dir / "run_alphafold.py").touch()
    pkg_init = runner_dir / "src/alphafold3/__init__.py"
    pkg_init.parent.mkdir(parents=True, exist_ok=True)
    pkg_init.touch()

    def write_af3_outputs(_written: Any, output_dir: str | Path, job_name: str) -> None:
        sample_dir = Path(output_dir) / "seed-7_sample-0"
        sample_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{job_name}_seed-7_sample-0"
        (sample_dir / f"{prefix}_model.cif").touch()
        (sample_dir / f"{prefix}_summary_confidences.json").write_text(
            json.dumps({"ptm": 0.7, "iptm": 0.5, "chain_pair_pae_min": [[1.0]]}),
            encoding="utf-8",
        )

    af3_runner = SimpleNamespace(
        ModelRunner=lambda **_kw: object(),
        make_model_config=lambda **kw: af3_seen.append(dict(kw)) or SimpleNamespace(),
        predict_structure=lambda *_a, **_k: (
            SimpleNamespace(
                seed=7,
                inference_results=(SimpleNamespace(metadata={"ranking_score": 0.9}),),
            ),
        ),
        write_outputs=write_af3_outputs,
    )
    @dataclasses.dataclass(frozen=True)
    class FakeFoldInput:
        name: str = "job"
        rng_seeds: tuple[int, ...] = (1,)

        def sanitised_name(self) -> str:
            return self.name

    folding = ModuleType("alphafold3.common.folding_input")
    folding.load_fold_inputs_from_path = lambda _p: iter((FakeFoldInput(),))
    common_mod = ModuleType("alphafold3.common")
    common_mod.folding_input = folding
    pkg_mod = ModuleType("alphafold3")
    pkg_mod.__file__ = str(pkg_init)
    monkeypatch.setitem(
        __import__("sys").modules, "alphafold3", pkg_mod
    )
    monkeypatch.setitem(
        __import__("sys").modules, "alphafold3.common", common_mod
    )
    monkeypatch.setitem(
        __import__("sys").modules, "alphafold3.common.folding_input", folding
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._load_runner", lambda _p: af3_runner
    )
    monkeypatch.setattr(
        "foldjax.backends.alphafold3._set_nested", lambda *_a: None
    )
    dev = jax.devices()[0]
    monkeypatch.setattr("jax.local_devices", lambda backend=None: (dev,))

    for opts in (
        {"confidence_outputs": "full", "source": runner_dir},
        {"source": runner_dir},
        {
            "confidence_outputs": "model",
            "return_embeddings": True,
            "source": runner_dir,
        },
    ):
        AlphaFold3Backend().predict(
            foldjax.resolve_request(
                _make_request(
                    tmp_path,
                    "alphafold3",
                    opts,
                    input_format="native",
                    cache_dir=None,
                    use_compile_cache=False,
                ),
                draw_seeds=False,
            )
        )
    assert _typed(af3_seen[0]) == _typed(af3_seen[1])
    assert af3_seen[0]["return_embeddings"] is False
    assert af3_seen[0]["return_distogram"] is False
    assert af3_seen[2]["return_embeddings"] is True


@pytest.mark.parametrize(
    ("model", "conflict_key", "conflict_val"),
    [
        ("esmfold2", "return_expected_errors", True),
        ("esmfold2", "return_expected_errors", False),
        ("esmfold2", "return_expected_errors", "true"),
        ("openfold3", "return_expected_errors", True),
        ("openfold3", "return_expected_errors", False),
        ("protenix", "output_format", "both"),
        ("protenix", "output_format", "protenix"),
        ("protenix", "output_format", "npz"),
        ("opendde", "include_raw", True),
        ("opendde", "include_raw", False),
    ],
)
def test_confidence_outputs_full_conflicts_with_explicit_mapped_option(
    tmp_path: Path, model: str, conflict_key: str, conflict_val: Any
) -> None:
    backend_cls = type(get_backend(model))
    opts = {"confidence_outputs": "full", conflict_key: conflict_val}
    with pytest.raises(ValueError) as exc_info:
        backend_cls.canonical_options(opts)
    assert "confidence_outputs" in str(exc_info.value)
    assert conflict_key in str(exc_info.value)

    with pytest.raises(ValueError):
        foldjax.resolve_request(
            _make_request(tmp_path, model, opts), draw_seeds=False
        )


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--output-format", "both"],
        ["--output-format", "npz"],
        ["--output-format=both"],
        ("--output-forma", "protenix"),
    ],
)
def test_protenix_confidence_outputs_full_rejects_cli_args_output_format(
    tmp_path: Path, cli_args: Any
) -> None:
    opts = {"confidence_outputs": "full", "cli_args": cli_args}
    with pytest.raises(ValueError, match="output-format"):
        ProtenixBackend.canonical_options(opts)
    with pytest.raises(ValueError, match="output-format"):
        foldjax.resolve_request(
            _make_request(tmp_path, "protenix", opts), draw_seeds=False
        )


def test_protenix_confidence_outputs_full_allows_unrelated_or_empty_cli_args(
    tmp_path: Path,
) -> None:
    for cli_args in ((), ["--gamma0", "0.2"]):
        opts = {"confidence_outputs": "full", "cli_args": cli_args}
        canonical = ProtenixBackend.canonical_options(opts)
        assert canonical["output_format"] == "both"
        resolved = foldjax.resolve_request(
            _make_request(tmp_path, "protenix", opts), draw_seeds=False
        )
        inv = ProtenixBackend()._native_invocation(resolved)
        assert inv.config_fields["output_format"] == "both"


@pytest.mark.parametrize("model", sorted(CONFIDENCE_FULL_MAP))
@pytest.mark.parametrize(
    ("stop_after", "rep"),
    [("inputs", "single_inputs"), ("trunk", "single")],
)
def test_confidence_outputs_full_rejected_with_early_stop_after(
    tmp_path: Path, model: str, stop_after: str, rep: str
) -> None:
    req = _make_request(
        tmp_path,
        model,
        {"confidence_outputs": "full"},
        stop_after=stop_after,
        representations=(rep,),
    )
    with pytest.raises(ValueError, match="stop_after"):
        foldjax.resolve_request(req, draw_seeds=False)
    with pytest.raises(ValueError, match="stop_after"):
        get_backend(model).apply_sampling(req)

    # "model" is the omitted default and stays valid with early stop_after.
    req_model = _make_request(
        tmp_path,
        model,
        {"confidence_outputs": "model"},
        stop_after=stop_after,
        representations=(rep,),
    )
    resolved_model = foldjax.resolve_request(req_model, draw_seeds=False)
    assert resolved_model.stop_after == stop_after


@pytest.mark.parametrize("model", sorted(CONFIDENCE_FULL_MAP))
@pytest.mark.parametrize(
    "bad_value",
    ["", "all", "both", "FULL", " model ", 0, 1, True, False, None, ["full"], {}],
)
def test_confidence_outputs_rejects_bad_values_listing_both_choices(
    tmp_path: Path, model: str, bad_value: Any
) -> None:
    backend_cls = type(get_backend(model))
    with pytest.raises(ValueError) as exc_info:
        backend_cls.canonical_options({"confidence_outputs": bad_value})
    msg = str(exc_info.value)
    assert "'model'" in msg and "'full'" in msg

    with pytest.raises(ValueError) as req_exc:
        foldjax.resolve_request(
            _make_request(tmp_path, model, {"confidence_outputs": bad_value}),
            draw_seeds=False,
        )
    req_msg = str(req_exc.value)
    assert "'model'" in req_msg and "'full'" in req_msg
