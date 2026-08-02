"""Public prediction CLI contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foldjax.models.chai.cli.main import main
from foldjax.models.chai.inference import _default_cache


def test_fold_maps_official_and_native_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class Candidates:
        cif_paths = [tmp_path / "outputs/pred.model_idx_0.cif"]

    def fake_run(fasta_file: Path, **kwargs: Any) -> Candidates:
        captured.update(fasta_file=fasta_file, **kwargs)
        return Candidates()

    monkeypatch.setattr("foldjax.models.chai.cli.main.run_inference", fake_run)

    result = main(
        [
            "fold",
            str(fasta),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--bundle-path",
            str(tmp_path / "bundle"),
            "--conformer-path",
            str(tmp_path / "conformers.npz"),
            "--num-trunk-recycles",
            "4",
            "--recycle-msa-subsample",
            "1",
            "--num-trunk-samples",
            "3",
            "--num-diffn-timesteps",
            "20",
            "--num-diffn-samples",
            "2",
            "--seed",
            "7",
            "--use-msa-server",
            "--msa-cache-directory",
            str(tmp_path / "msa-cache"),
            "--compilation-cache-dir",
            str(tmp_path / "jax-cache"),
            "--fasta-names-as-cif-chains",
        ]
    )

    assert result == 0
    assert captured["fasta_file"] == fasta
    assert captured["output_dir"] == tmp_path / "outputs"
    assert captured["bundle_path"] == tmp_path / "bundle"
    assert captured["conformer_path"] == tmp_path / "conformers.npz"
    config = captured["config"]
    assert config.num_trunk_recycles == 4
    assert config.recycle_msa_subsample == 1
    assert config.num_trunk_samples == 3
    assert config.num_diffusion_timesteps == 20
    assert config.num_diffusion_samples == 2
    assert config.seed == 7
    assert config.use_msa_server is True
    assert config.msa_cache_directory == tmp_path / "msa-cache"
    assert config.compilation_cache_dir == tmp_path / "jax-cache"
    assert config.fasta_names_as_cif_chains is True
    assert "pred.model_idx_0.cif" in capsys.readouterr().out


def test_fold_defaults_match_supported_public_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class Candidates:
        cif_paths: list[Path] = []

    def fake_run(_fasta: Path, **kwargs: Any) -> Candidates:
        captured.update(kwargs)
        return Candidates()

    monkeypatch.setattr("foldjax.models.chai.cli.main.run_inference", fake_run)
    assert (
        main(
            [
                "fold",
                str(fasta),
                "--output-dir",
                str(tmp_path / "outputs"),
                "--bundle-path",
                str(tmp_path / "bundle"),
                "--conformer-path",
                str(tmp_path / "conformers.npz"),
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.num_trunk_recycles == 3
    assert config.recycle_msa_subsample == 0
    assert config.num_trunk_samples == 1
    assert config.num_diffusion_timesteps == 200
    assert config.num_diffusion_samples == 5
    assert config.use_msa_server is False
    assert config.use_esm_embeddings is True
    # The default resolves through _default_cache, which prefers the FoldJAX
    # location but keeps using a populated pre-FoldJAX ~/.cache/chai_jax. Assert
    # that contract rather than one literal path, which is machine-dependent.
    assert config.esm_model_path == _default_cache("models", "esm2_t36_3B")


def test_fold_reports_validation_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">protein|name=target\nAA\n", encoding="utf-8")

    result = main(
        [
            "fold",
            str(fasta),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--bundle-path",
            str(tmp_path / "bundle"),
            "--conformer-path",
            str(tmp_path / "conformers.npz"),
            "--num-trunk-samples",
            "0",
        ]
    )

    assert result == 1
    stderr = capsys.readouterr().err
    assert "num_trunk_samples must be positive" in stderr
    assert "Traceback" not in stderr


def test_fold_requires_existing_fasta(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "fold",
                str(tmp_path / "missing.fasta"),
                "--output-dir",
                str(tmp_path / "outputs"),
                "--bundle-path",
                str(tmp_path / "bundle"),
                "--conformer-path",
                str(tmp_path / "conformers.npz"),
            ]
        )
    assert exc_info.value.code == 2


def test_default_cache_prefers_foldjax_but_keeps_a_populated_legacy_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ESM2 bundle is multi-GB, so a pre-FoldJAX export must stay usable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # This exercises the home-directory fallback, so the two overrides that
    # come before it have to be absent -- otherwise the test passes or fails
    # depending on the environment it is run in.
    monkeypatch.delenv("FOLDJAX_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    # Neither exists: the FoldJAX location is what gets created.
    assert _default_cache("models") == tmp_path / ".cache/foldjax/chai/models"

    legacy = tmp_path / ".cache/chai_jax/models"
    legacy.mkdir(parents=True)
    assert _default_cache("models") == legacy

    current = tmp_path / ".cache/foldjax/chai/models"
    current.mkdir(parents=True)
    assert _default_cache("models") == current


def test_the_chai_cache_follows_foldjax_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOLDJAX_HOME is documented as the one root, so it must move this too.

    The ESM2 bundle is several GB and is the largest thing FoldJAX writes. It
    was resolved from `Path.home()` directly, so pointing FOLDJAX_HOME at a
    scratch filesystem still filled the user's home directory.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "scratch"))
    assert _default_cache("models") == tmp_path / "scratch/chai/models"

    monkeypatch.delenv("FOLDJAX_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _default_cache("models") == tmp_path / "xdg/foldjax/chai/models"

    # With neither set it is exactly where it always was.
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert _default_cache("models") == tmp_path / "home/.cache/foldjax/chai/models"
