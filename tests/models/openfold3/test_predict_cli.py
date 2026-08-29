"""The prediction CLI's decisions, without needing a GPU or the released weights.

This is the entry point users actually run, and it was the only one of the four with
no test. What is covered here is what it *decides* -- refusing incomplete features,
resolving the sampling knobs, reporting memory and warning when a target is close to
filling the device -- rather than the arithmetic, which the parity gate owns.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.models.openfold3.cli.predict import _parser, _report_peak_memory, main
from foldjax.models.openfold3.data import (
    MODEL_FEATURES,
    load_features,
    normalize_asym_ids,
    pad_features,
    save_features,
)
from foldjax.models.openfold3.data.compact_categories import (
    COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES,
    COMPACT_REF_ATOM_NAME_CHAR_IDS,
    COMPACT_REF_ELEMENT_IDS,
)
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
)
from foldjax.models.openfold3.output import (
    DEFAULT_ARRAY_BUDGET_BYTES,
    _output_path,
    _safe_output_name,
    atom_metadata,
    plan_returned_pair_logits,
)

from .feature_fixture import minimal_features


class _Device:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


class _Jax:
    def __init__(self, device):
        self._device = device

    def devices(self):
        return [self._device]


class _Config:
    def __init__(self, chunk):
        self.pair_chunk_size = chunk


def test_defaults_match_the_released_settings() -> None:
    """The sampling knobs default to "whatever released_config says", not to numbers.

    Hardcoding 5/200/4 here would let the CLI drift from the config the checkpoint
    was produced under.
    """
    args = _parser().parse_args(["f.npz", "--checkpoint", "w.pt", "-o", "out"])
    assert args.samples is None
    assert args.steps is None
    assert args.cycles is None
    assert args.repeats == 1
    assert args.all_arrays is False
    assert args.cache_dir is None
    assert args.no_cache is False


def test_all_arrays_help_keeps_the_quadratic_memory_warning() -> None:
    help_text = " ".join(_parser().format_help().split())
    assert "--all-arrays" in help_text
    assert "over 10 GiB at 2000 tokens" in help_text
    assert "explicit 4 GiB budget" in help_text


def _table() -> RepresentativeAtomTable:
    return RepresentativeAtomTable(
        *(np.zeros(32, dtype=np.float32) for _ in RepresentativeAtomTable._fields)
    )


def _features(*, tokens: int = 4, atoms: int = 7) -> dict[str, np.ndarray]:
    return minimal_features(tokens=tokens, atoms=atoms)


@pytest.mark.parametrize("size", [32, 39, 119])
def test_padding_uses_declared_axes_when_dimensions_collide(size: int) -> None:
    """Fixed channel widths and equal token/atom counts remain unambiguous."""
    features = {
        "token_mask": np.ones((1, size), dtype=np.float32),
        "atom_mask": np.ones((1, size), dtype=np.float32),
        "num_atoms_per_token": np.zeros((1, size), dtype=np.int32),
        "restype": np.zeros((1, size, 32), dtype=np.float32),
        "ref_pos": np.zeros((1, size, 3), dtype=np.float32),
        "ref_element": np.zeros((1, size, 119), dtype=np.float32),
        "template_distogram": np.zeros(
            (1, 1, size, size, 39), dtype=np.float32
        ),
        "token_bonds": np.zeros((1, size, size), dtype=np.float32),
        "max_atom_per_token_mask": np.zeros((1, size * 23), dtype=np.float32),
    }

    padded = pad_features(features, n_token=size + 1, n_atom=size + 2)

    assert padded["token_mask"].shape == (1, size + 1)
    assert padded["atom_mask"].shape == (1, size + 2)
    assert padded["restype"].shape == (1, size + 1, 32)
    assert padded["ref_pos"].shape == (1, size + 2, 3)
    assert padded["ref_element"].shape == (1, size + 2, 119)
    assert padded["template_distogram"].shape == (
        1,
        1,
        size + 1,
        size + 1,
        39,
    )
    assert padded["token_bonds"].shape == (1, size + 1, size + 1)


def test_chain_ids_are_normalized_from_real_tokens_only() -> None:
    features = {
        "token_mask": np.asarray([[1, 1, 1, 0, 0]], dtype=np.float32),
        "asym_id": np.asarray([[9, 3, 9, -7, 99]], dtype=np.int64),
    }

    normalized, n_chain = normalize_asym_ids(features)

    np.testing.assert_array_equal(normalized["asym_id"], [[1, 0, 1, 0, 0]])
    np.testing.assert_array_equal(features["asym_id"], [[9, 3, 9, -7, 99]])
    assert n_chain == 2

    monomer, monomer_n_chain = normalize_asym_ids(
        {
            "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
            "asym_id": np.asarray([[42, 42, 999]], dtype=np.int64),
        }
    )
    np.testing.assert_array_equal(monomer["asym_id"], [[0, 0, 0]])
    assert monomer_n_chain == 1


def test_atom_metadata_ignores_padded_chain_ids() -> None:
    features = {
        "ref_atom_name_chars": np.zeros((1, 2, 4, 64), dtype=np.float32),
        "ref_element": np.zeros((1, 2, 119), dtype=np.float32),
        "atom_mask": np.asarray([[1, 0]], dtype=np.float32),
        "atom_to_token_index": np.asarray([[0, 1]], dtype=np.int64),
        "restype": np.zeros((1, 2, 32), dtype=np.float32),
        "residue_index": np.asarray([[1, 0]], dtype=np.int64),
        # Padding's zero sorts before the real sparse ID. It used to shift the
        # only emitted chain to B.
        "asym_id": np.asarray([[8, 0]], dtype=np.int64),
    }

    assert atom_metadata(features).chain_id.tolist() == ["A"]


def test_openfold3_sampling_translates_recycles_but_keeps_native_cycles(
    tmp_path: Path,
) -> None:
    from foldjax.backends.openfold3 import OpenFold3Backend
    from foldjax.schema import PredictionRequest

    input_path = tmp_path / "input.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights.pt"
    weights.touch()
    backend = OpenFold3Backend()

    neutral = PredictionRequest(
        model="openfold3",
        input=input_path,
        weights=weights,
        num_recycles=3,
    )
    native = PredictionRequest(
        model="openfold3",
        input=input_path,
        weights=weights,
        options={"num_recycles": 3},
    )

    assert backend.apply_sampling(neutral)["num_recycles"] == 4
    assert backend.apply_sampling(native)["num_recycles"] == 3


def test_openfold3_msa_depth_is_a_cap_on_the_released_default(
    tmp_path: Path,
) -> None:
    from foldjax.backends.openfold3 import OpenFold3Backend
    from foldjax.schema import PredictionRequest

    input_path = tmp_path / "input.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights.pt"
    weights.touch()
    backend = OpenFold3Backend()

    def options(depth: int) -> dict[str, object]:
        request = PredictionRequest(
            model="openfold3",
            input=input_path,
            weights=weights,
            max_msa_depth=depth,
        )
        return backend.apply_sampling(request)

    assert options(128)["max_msa_depth"] == 128
    assert options(2048)["max_msa_depth"] == 1024


@pytest.mark.parametrize("eager", [False, True], ids=["compiled", "eager"])
@pytest.mark.parametrize("num_samples", [5, 10], ids=["released-s5", "raised-s10"])
@pytest.mark.parametrize(
    "all_arrays", [False, True], ids=["budgeted-arrays", "all-arrays"]
)
def test_prediction_cli_passes_static_chain_count_and_ignores_masked_atom_padding(
    tmp_path: Path,
    monkeypatch,
    eager: bool,
    num_samples: int,
    all_arrays: bool,
) -> None:
    import jax
    import jax.numpy as jnp

    from foldjax.models.openfold3 import data, inference, output
    from foldjax.models.openfold3.bridge import checkpoint, torch_mapping

    raw = {
        "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "atom_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "asym_id": np.asarray([[7, 42, 999]], dtype=np.int64),
        "is_atomized": np.asarray([[0, 0, 1]], dtype=np.int32),
        "ref_element": np.pad(
            np.eye(119, dtype=np.int32)[None, [5, 6]],
            ((0, 0), (0, 1), (0, 0)),
        ),
        "ref_atom_name_chars": np.pad(
            np.eye(64, dtype=np.int32)[
                np.asarray(
                    [
                        [ord(character) - 32 for character in "N".ljust(4)],
                        [ord(character) - 32 for character in "CA".ljust(4)],
                    ]
                )
            ][None],
            ((0, 0), (0, 1), (0, 0), (0, 0)),
        ),
    }
    params = SimpleNamespace(
        trunk=SimpleNamespace(pairformer_stack=SimpleNamespace(blocks=())),
        denoiser=SimpleNamespace(
            diffusion_transformer=SimpleNamespace(blocks=())
        ),
    )
    coordinates = np.zeros((1, 3, 3), dtype=np.float32)
    coordinates[0, 2, 0] = np.nan
    prediction = SimpleNamespace(coordinates=coordinates)
    seen: dict[str, object] = {}
    call_order: list[str] = []
    checkpoint_state = {"checkpoint": object()}
    checkpoint_events: list[object] = []

    monkeypatch.setattr(
        data, "load_feature_archive", lambda path: (raw, object(), None)
    )
    monkeypatch.setattr(data, "subsample_msa_rows", lambda features, depth: features)

    def compact(features):
        call_order.append("templates")
        seen["compact_asym_id"] = np.asarray(features["asym_id"])
        return features

    monkeypatch.setattr(data, "compact_zero_template_pair_features", compact)
    monkeypatch.setattr(
        data,
        "compact_msa_features",
        lambda features: (call_order.append("msa"), features)[1],
    )
    original_category_compactor = data.compact_ref_atom_category_storage

    def compact_categories(features):
        call_order.append("atom-categories")
        assert "ref_element" in features
        assert "ref_atom_name_chars" in features
        return original_category_compactor(features)

    monkeypatch.setattr(
        data, "compact_ref_atom_category_storage", compact_categories
    )

    def fake_released_config(**kwargs):
        seen["has_atomized_tokens"] = kwargs["has_atomized_tokens"]
        seen["config_array_budget"] = kwargs["max_array_bytes"]
        seen["config_num_samples"] = kwargs.get("num_samples", 5)
        seen["config_per_sample_token_cutoff_present"] = (
            "per_sample_token_cutoff" in kwargs
        )
        return SimpleNamespace(
            msa_depth=1024,
            num_samples=kwargs.get("num_samples", 5),
            num_steps=200,
            num_recycles=4,
            pair_chunk_size=None,
        )

    monkeypatch.setattr(
        inference,
        "released_config",
        fake_released_config,
    )

    def fake_predict(key, features, params, config, table, *, n_chain=None):
        seen.update(
            n_chain=n_chain,
            asym_id=np.asarray(features["asym_id"]),
            model_features=features,
        )
        return prediction

    def fake_compile(config, table, *, n_chain=None, **compile_options):
        seen["compile_options"] = compile_options

        def compiled(key, features, params):
            seen.update(
                n_chain=n_chain,
                asym_id=np.asarray(features["asym_id"]),
                model_features=features,
            )
            return prediction

        return compiled

    monkeypatch.setattr(inference, "predict", fake_predict)
    monkeypatch.setattr(inference, "compile_predict", fake_compile)
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
        return params

    monkeypatch.setattr(checkpoint, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(torch_mapping, "resolve_model_prefix", resolve_model_prefix)
    monkeypatch.setattr(
        torch_mapping,
        "prune_sample_diffusion_aliases",
        prune_sample_diffusion_aliases,
    )
    monkeypatch.setattr(torch_mapping, "map_inference_params", map_inference_params)
    monkeypatch.setattr(jnp, "asarray", np.asarray)
    monkeypatch.setattr(jax.random, "key", lambda seed: seed)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [SimpleNamespace(memory_stats=lambda: None)],
    )

    def fake_write_prediction_outputs(*args, **kwargs):
        seen["writer_array_budget"] = kwargs["max_array_bytes"]
        seen["writer_features"] = args[1]
        return {
            "structures": (),
            "scores": tmp_path / "scores.json",
            "arrays": tmp_path / "raw.npz",
            "omitted_arrays": (),
        }

    monkeypatch.setattr(
        output, "write_prediction_outputs", fake_write_prediction_outputs
    )

    features_path = tmp_path / "features.npz"
    features_path.touch()
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint_path.touch()
    argv = [
        str(features_path),
        "--checkpoint",
        str(checkpoint_path),
        "-o",
        str(tmp_path / "out"),
        "--no-cache",
    ]
    if eager:
        argv.append("--no-compile")
    if all_arrays:
        argv.append("--all-arrays")
    if num_samples != 5:
        argv.extend(("--samples", str(num_samples)))

    assert main(argv) == 0
    assert seen["n_chain"] == 2
    assert seen["has_atomized_tokens"] is False
    if not eager:
        assert seen["compile_options"] == {"cache_scope": None}
    expected_budget = None if all_arrays else DEFAULT_ARRAY_BUDGET_BYTES
    assert seen["config_array_budget"] == expected_budget
    assert seen["writer_array_budget"] == expected_budget
    assert seen["config_num_samples"] == num_samples
    assert seen["config_per_sample_token_cutoff_present"] is False
    np.testing.assert_array_equal(seen["asym_id"], [[0, 1, 0]])
    np.testing.assert_array_equal(seen["compact_asym_id"], [[0, 1, 0]])
    assert call_order == ["templates", "msa", "atom-categories"]
    model_features = seen["model_features"]
    writer_features = seen["writer_features"]
    assert "ref_element" not in model_features
    assert "ref_atom_name_chars" not in model_features
    assert COMPACT_REF_ELEMENT_IDS in model_features
    assert COMPACT_REF_ATOM_NAME_CHAR_IDS in model_features
    assert "ref_element" in writer_features
    assert "ref_atom_name_chars" in writer_features
    assert not (
        set(COMPACT_REF_ATOM_CATEGORIES_PRIVATE_FEATURES) & writer_features.keys()
    )
    assert checkpoint_events == [
        "load",
        ("resolve", checkpoint_state, None),
        ("prune", checkpoint_state, "resolved"),
        ("map", checkpoint_state, "resolved"),
    ]


def test_all_arrays_restores_every_pair_logit_above_the_default_budget() -> None:
    """The writer cannot save an array the compiled graph never returned."""
    from foldjax.models.openfold3.inference import released_config

    n_token = 2076
    budgeted = released_config(n_token=n_token, n_atom=1)
    unrestricted = released_config(
        n_token=n_token,
        n_atom=1,
        max_array_bytes=None,
    )

    assert budgeted.returned_pair_logits == plan_returned_pair_logits(
        n_token=n_token,
        num_samples=budgeted.num_samples,
        max_bytes=DEFAULT_ARRAY_BUDGET_BYTES,
    )
    assert budgeted.returned_pair_logits == ("distogram_logits",)
    assert unrestricted.returned_pair_logits == (
        "pae_logits",
        "pde_logits",
        "distogram_logits",
    )
    assert budgeted.returned_representations == ()
    assert unrestricted.returned_representations == ()


def test_pair_logit_planner_keeps_the_exact_budget_boundary() -> None:
    n_token = 3
    num_samples = 2
    bins = 1
    pair = n_token * n_token
    pae_bytes = num_samples * pair * bins * 4
    pde_bytes = pae_bytes
    distogram_bytes = pair * bins * 4
    exact_budget = 64 * 2**20 + pae_bytes + pde_bytes + distogram_bytes
    kwargs = {
        "n_token": n_token,
        "num_samples": num_samples,
        "pae_bins": bins,
        "pde_bins": bins,
        "distogram_bins": bins,
    }

    assert plan_returned_pair_logits(max_bytes=exact_budget, **kwargs) == (
        "pae_logits",
        "pde_logits",
        "distogram_logits",
    )
    assert plan_returned_pair_logits(max_bytes=exact_budget - 1, **kwargs) == (
        "pde_logits",
        "distogram_logits",
    )


def test_incomplete_features_are_refused(tmp_path: Path, capsys) -> None:
    """A missing feature must fail before the checkpoint is loaded.

    Otherwise it surfaces as a KeyError inside a traced function, minutes into a
    compile, naming a feature rather than the file that lacks it.
    """
    path = tmp_path / "features.npz"
    np.savez(path, token_mask=np.ones((1, 4), dtype=np.float32))
    assert main([str(path), "--checkpoint", "missing.pt", "-o", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "feature archive is missing" in out
    # The message has to name what is absent, and there is a lot absent here.
    assert "ref_pos" in out
    assert len(MODEL_FEATURES) > 30


def test_feature_archive_requires_embedded_chemistry(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, **_features())
    with pytest.raises(ValueError, match="no embedded chemistry table"):
        load_features(path)


def test_feature_archive_rejects_malformed_embedded_chemistry(tmp_path: Path) -> None:
    path = save_features(
        _features(), tmp_path / "bad.npz", representative_atoms=_table()
    )
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    payload["chemistry.cb_mask"] = np.zeros(31, dtype=np.float32)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="no embedded chemistry table"):
        load_features(path)


def test_save_and_load_features_use_the_path_they_report(tmp_path: Path) -> None:
    path = save_features(
        _features(), tmp_path / "FEATURES.NPZ", representative_atoms=_table()
    )
    assert path == tmp_path / "FEATURES.npz"
    assert path.is_file()
    loaded, table = load_features(path)
    assert set(loaded) == set(MODEL_FEATURES)
    assert isinstance(table, RepresentativeAtomTable)


def test_saving_features_replaces_a_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"owned by user")
    target = tmp_path / "features.npz"
    target.symlink_to(outside)

    path = save_features(_features(), target, representative_atoms=_table())

    assert path == target and path.is_file() and not path.is_symlink()
    assert outside.read_bytes() == b"owned by user"
    assert set(load_features(path)[0]) == set(MODEL_FEATURES)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../escape", "/tmp/escape", "a/b", "a\\b", "a\nb"]
)
def test_output_names_are_single_safe_filename_components(name: str) -> None:
    with pytest.raises(ValueError, match="one non-empty filename component"):
        _safe_output_name(name)


def test_existing_output_symlinks_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    outside = tmp_path / "outside.npz"
    outside.touch()
    (root / "prediction_raw.npz").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its output directory"):
        _output_path(root, "prediction_raw.npz")


def test_featurize_and_predict_default_to_the_same_seed() -> None:
    from foldjax.models.openfold3.cli.featurize import _parser as featurize_parser

    featurize = featurize_parser().parse_args(["query.json", "-o", "f.npz"])
    predict = _parser().parse_args(["f.npz", "--checkpoint", "w.pt", "-o", "out"])
    assert featurize.seed == predict.seed == 0


def test_cache_location_and_opt_out_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "f.npz",
                "--checkpoint",
                "w.pt",
                "-o",
                "out",
                "--cache-dir",
                "cache",
                "--no-cache",
            ]
        )
    assert error.value.code == 2


def test_peak_memory_is_reported_with_the_chunk_that_bounded_it(capsys) -> None:
    jax = _Jax(_Device({"peak_bytes_in_use": 26 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(483))
    out = capsys.readouterr().out
    assert "26.00 GiB" in out
    assert "483" in out
    # Well inside the device, so no warning.
    assert "will not fit" not in out


def test_a_nearly_full_device_is_called_out(capsys) -> None:
    """Fitting at 97% is not the same as fitting, and the user should hear so."""
    jax = _Jax(_Device({"peak_bytes_in_use": 93 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(123))
    out = capsys.readouterr().out
    assert "97%" in out
    assert "will not fit" in out


def test_no_chunking_is_reported_as_such(capsys) -> None:
    jax = _Jax(_Device({"peak_bytes_in_use": 2 * 2**30, "bytes_limit": 96 * 2**30}))
    _report_peak_memory(jax, _Config(None))
    assert "no chunking" in capsys.readouterr().out


def test_a_device_without_memory_stats_is_silent(capsys) -> None:
    """CPU devices expose no allocator stats; printing zero would be a lie."""
    _report_peak_memory(_Jax(object()), _Config(None))
    assert capsys.readouterr().out == ""

    _report_peak_memory(_Jax(_Device(None)), _Config(None))
    assert capsys.readouterr().out == ""

    _report_peak_memory(_Jax(_Device({})), _Config(None))
    assert capsys.readouterr().out == ""


def test_missing_bytes_limit_still_reports_the_peak(capsys) -> None:
    """Some platforms omit the limit; the peak is the useful half anyway."""
    jax = _Jax(_Device({"peak_bytes_in_use": 5 * 2**30}))
    _report_peak_memory(jax, _Config(None))
    out = capsys.readouterr().out
    assert "5.00 GiB" in out
    assert "%" not in out


@pytest.mark.parametrize("flag", ["--samples", "--steps", "--cycles", "--repeats"])
def test_sampling_knobs_are_integers(flag: str) -> None:
    args = _parser().parse_args(
        ["f.npz", "--checkpoint", "w.pt", "-o", "out", flag, "7"]
    )
    assert getattr(args, flag.lstrip("-").replace("-", "_")) == 7
