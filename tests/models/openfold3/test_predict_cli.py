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
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
)
from foldjax.models.openfold3.output import (
    _output_path,
    _safe_output_name,
    atom_metadata,
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
    assert monomer_n_chain is None


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
        options={"num_cycles": 3},
    )

    assert backend.apply_sampling(neutral)["num_cycles"] == 4
    assert backend.apply_sampling(native)["num_cycles"] == 3


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
def test_prediction_cli_passes_static_chain_count(
    tmp_path: Path, monkeypatch, eager: bool
) -> None:
    import jax
    import jax.numpy as jnp

    from foldjax.models.openfold3 import data, inference, output
    from foldjax.models.openfold3.bridge import checkpoint, torch_mapping

    raw = {
        "token_mask": np.asarray([[1, 1, 0]], dtype=np.float32),
        "atom_mask": np.asarray([[1, 1]], dtype=np.float32),
        "asym_id": np.asarray([[7, 42, 999]], dtype=np.int64),
    }
    params = SimpleNamespace(
        trunk=SimpleNamespace(pairformer_stack=SimpleNamespace(blocks=())),
        denoiser=SimpleNamespace(
            diffusion_transformer=SimpleNamespace(blocks=())
        ),
    )
    prediction = SimpleNamespace(coordinates=np.zeros((1, 2, 3), dtype=np.float32))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        data, "load_feature_archive", lambda path: (raw, object(), None)
    )
    monkeypatch.setattr(data, "subsample_msa_rows", lambda features, depth: features)
    monkeypatch.setattr(
        inference,
        "released_config",
        lambda **kwargs: SimpleNamespace(
            msa_depth=1024,
            num_samples=5,
            no_rollout_steps=200,
            num_cycles=4,
            pair_chunk_size=None,
        ),
    )

    def fake_predict(key, features, params, config, table, *, n_chain=None):
        seen.update(n_chain=n_chain, asym_id=np.asarray(features["asym_id"]))
        return prediction

    def fake_compile(config, table, *, n_chain=None):
        def compiled(key, features, params):
            seen.update(n_chain=n_chain, asym_id=np.asarray(features["asym_id"]))
            return prediction

        return compiled

    monkeypatch.setattr(inference, "predict", fake_predict)
    monkeypatch.setattr(inference, "compile_predict", fake_compile)
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(
        torch_mapping, "map_inference_params", lambda state, prefix: params
    )
    monkeypatch.setattr(jnp, "asarray", np.asarray)
    monkeypatch.setattr(jax.random, "key", lambda seed: seed)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [SimpleNamespace(memory_stats=lambda: None)],
    )
    monkeypatch.setattr(
        output,
        "write_prediction_outputs",
        lambda *args, **kwargs: {
            "structures": (),
            "scores": tmp_path / "scores.json",
            "arrays": tmp_path / "raw.npz",
            "omitted_arrays": (),
        },
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

    assert main(argv) == 0
    assert seen["n_chain"] == 2
    np.testing.assert_array_equal(seen["asym_id"], [[0, 1, 0]])


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
