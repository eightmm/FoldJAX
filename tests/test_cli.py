import dataclasses
import json
import os
from pathlib import Path

import pytest

from foldjax.cli import _options, entrypoint, main


def test_models_cli_lists_backends(capsys) -> None:
    assert main(["models"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "alphafold3",
        "boltz2",
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    ]


def test_home_can_print_one_script_friendly_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    assert main(["home", "--path", "runtime"]) == 0

    assert capsys.readouterr().out.strip() == str(tmp_path / "runtime")


def test_discovery_commands_do_not_change_the_host_memory_policy(
    monkeypatch, capsys
) -> None:
    from foldjax import oom

    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    assert main(["models"]) == 0
    capsys.readouterr()
    assert oom.FRACTION_ENV not in os.environ


def test_models_json_reports_weight_readiness(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "empty-store"))
    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)

    assert main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert [item["model"] for item in payload] == [
        "alphafold3",
        "boltz2",
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    ]
    assert all(item["weights"]["ready"] is False for item in payload)
    boltz = next(item for item in payload if item["model"] == "boltz2")
    assert boltz["weights"]["fetchable"] is True
    assert boltz["weights"]["setup"] == "foldjax weights fetch --model boltz2"
    alphafold = next(item for item in payload if item["model"] == "alphafold3")
    assert alphafold["weights"]["fetchable"] is False
    assert "Request the parameters" in alphafold["weights"]["setup"]
    assert alphafold["runtime"] == {
        "notes": alphafold["runtime"]["notes"],
        "ready": False,
        "requires_network": True,
        "setup": "foldjax runtime prepare --model alphafold3",
    }
    assert next(item for item in payload if item["model"] == "boltz2")[
        "runtime"
    ]["ready"] is True
    openfold = next(item for item in payload if item["model"] == "openfold3")
    requirements = openfold["input_requirements"]
    assert requirements["openfold3-features"] == {
        "notes": requirements["openfold3-features"]["notes"],
        "prediction_runtime": "jax",
        "preprocessing_runtime": "precomputed",
        "required_extras": [],
        "requires_torch": False,
    }
    assert requirements["foldjax"]["required_extras"] == [
        "openfold3-preprocess"
    ]
    assert requirements["foldjax"]["preprocessing_runtime"] == "jax"
    assert requirements["foldjax"]["requires_torch"] is False
    esmfold = next(item for item in payload if item["model"] == "esmfold2")
    profiles = {
        profile["profile"]: profile for profile in esmfold["weights"]["profiles"]
    }
    assert profiles["released"]["download_bytes"] == 26_765_060_727
    assert profiles["structure-only"]["download_bytes"] == 1_356_814_149
    protenix = next(item for item in payload if item["model"] == "protenix")
    protenix_profiles = {
        profile["profile"]: profile
        for profile in protenix["weights"]["profiles"]
    }
    assert protenix_profiles["mini-esm-v0.5.0"]["download_bytes"] == 6_865_874_647
    assert protenix_profiles["mini-ism-v0.5.0"]["download_bytes"] == 12_544_004_971


def test_weights_cli_passes_model_specific_profiles(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))
    calls = []

    def fake_fetch(
        model: str,
        *,
        profile: str | None = None,
        on_progress=None,
        on_event=None,
        convert: bool = True,
    ) -> Path:
        calls.append(
            (model, profile, convert, on_progress is not None, on_event is not None)
        )
        return tmp_path / "home" / "weights" / model / "model.safetensors"

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    assert (
        main(
            [
                "weights",
                "fetch",
                "--model",
                "esmfold2",
                "--profile",
                "structure-only",
            ]
        )
        == 0
    )

    assert calls == [("esmfold2", "structure-only", True, True, True)]
    assert "3 file(s)" in capsys.readouterr().out


def test_weights_cli_routes_protenix_profile_through_its_isolated_root(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    calls = []

    def fake_fetch(
        model: str,
        *,
        profile: str | None = None,
        on_progress=None,
        on_event=None,
        convert: bool = True,
    ) -> Path:
        calls.append(
            (model, profile, convert, on_progress is not None, on_event is not None)
        )
        return tmp_path / "protenix-mini-esm" / "mini.jax"

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    assert (
        main(
            [
                "weights",
                "fetch",
                "--model",
                "protenix",
                "--profile",
                "mini-esm-v0.5.0",
            ]
        )
        == 0
    )

    assert calls == [("protenix", "mini-esm-v0.5.0", True, True, True)]
    output = capsys.readouterr().out
    assert "protenix: 6 file(s)" in output
    assert "protenix-mini-esm:" not in output


def test_download_only_reports_downloaded_rather_than_ready(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    destination = tmp_path / "downloads" / "opendde"
    monkeypatch.setattr(assets, "fetch", lambda *args, **kwargs: destination)

    assert (
        main(
            [
                "weights",
                "fetch",
                "--model",
                "opendde",
                "--download-only",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert f"downloaded: {destination}" in output
    assert f"ready: {destination}" not in output


def test_weights_cli_renders_structured_conversion_events_on_stderr(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    destination = tmp_path / "weights" / "opendde.jax"

    def fake_fetch(
        model: str,
        *,
        profile: str | None = None,
        on_progress=None,
        on_event=None,
        convert: bool = True,
    ) -> Path:
        assert convert is True
        assert on_event is not None
        on_event(
            assets.AssetEvent(
                model=model,
                profile=profile or "released",
                action="convert",
                status="start",
                message="publisher checkpoint to native JAX parameters",
            )
        )
        on_event(
            assets.AssetEvent(
                model=model,
                profile=profile or "released",
                action="convert",
                status="done",
                message="publisher checkpoint to native JAX parameters",
                elapsed_seconds=1.25,
                path=destination,
            )
        )
        return destination

    monkeypatch.setattr(assets, "fetch", fake_fetch)

    assert main(["weights", "fetch", "--model", "opendde"]) == 0

    captured = capsys.readouterr()
    assert f"ready: {destination}" in captured.out
    assert "convert start" in captured.err
    assert "convert done" in captured.err
    assert "1.25s" in captured.err


def test_manual_weights_cannot_report_a_fake_download_only_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path / "home"))

    assert (
        main(
            [
                "weights",
                "fetch",
                "--model",
                "alphafold3",
                "--download-only",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "no downloadable parameter files" in captured.err
    assert "ready:" not in captured.out


def test_weights_path_passes_the_selected_profile(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return tmp_path / "model.safetensors"

    monkeypatch.setattr(assets, "resolve_weights", fake_resolve)
    assert (
        main(
            [
                "weights",
                "path",
                "--model",
                "esmfold2",
                "--profile",
                "structure-only",
            ]
        )
        == 0
    )

    assert calls == [("esmfold2", "structure-only")]
    assert capsys.readouterr().out.strip() == str(tmp_path / "model.safetensors")


def test_weights_path_accepts_a_protenix_variant_profile(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return tmp_path / "protenix-mini-ism" / "mini.jax"

    monkeypatch.setattr(assets, "resolve_weights", fake_resolve)
    assert (
        main(
            [
                "weights",
                "path",
                "--model",
                "protenix",
                "--profile",
                "mini-ism-v0.5.0",
            ]
        )
        == 0
    )

    assert calls == [("protenix", "mini-ism-v0.5.0")]
    assert capsys.readouterr().out.strip().endswith("protenix-mini-ism/mini.jax")


def test_runtime_status_is_non_mutating_and_json_friendly(
    monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)
    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("status must never prepare the runtime"),
    )

    assert main(["runtime", "status", "--model", "af3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "alphafold3"
    assert payload["ready"] is False
    assert payload["setup"] == "foldjax runtime prepare --model alphafold3"
    assert payload["requires_network"] is True


def test_shadowing_external_runtime_reports_repair_and_refuses_prepare(
    monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build
    from foldjax.schema import PredictionError

    blocker = (
        "Repair or remove the AlphaFold 3 installation; required runtime "
        "artifacts are missing and it shadows FoldJAX's managed runtime."
    )
    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: blocker)
    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("prepare must not build behind a shadowing package"),
    )

    assert main(["runtime", "status", "--model", "alphafold3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["setup"] == blocker
    assert payload["requires_network"] is False
    assert "will not replace a loaded extension" in payload["notes"]

    with pytest.raises(PredictionError, match="Repair or remove"):
        main(["runtime", "prepare", "--model", "alphafold3"])


def test_runtime_prepare_materializes_alphafold3_only_when_needed(
    monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    state = {"ready": False}
    calls = []
    monkeypatch.setattr(build, "is_ready", lambda: state["ready"])
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)

    def prepare() -> None:
        calls.append("prepare")
        state["ready"] = True

    monkeypatch.setattr(build, "ensure_ready", prepare)

    assert main(["runtime", "prepare", "--model", "alphafold-3"]) == 0
    captured = capsys.readouterr()
    assert calls == ["prepare"]
    assert "may compile and download" in captured.err
    payload = json.loads(captured.out)
    assert payload["ready"] is True
    assert payload["setup"] is None
    assert payload["requires_network"] is False


def test_runtime_prepare_is_a_clean_noop_for_other_models(
    monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("a JAX-only backend has no native runtime to prepare"),
    )

    assert main(["runtime", "prepare", "--model", "boltz"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "boltz2"
    assert payload["ready"] is True
    assert payload["setup"] is None


def test_setup_fetches_defaults_and_reports_opt_in_or_manual_models(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """One command fetches defaults and reports every deliberate opt-in."""
    from foldjax import assets
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.delenv("PROTENIX_TEMPLATE_MMCIF_DIR", raising=False)
    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)
    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("setup must report, not build, the AlphaFold runtime"),
    )
    fetched = []
    monkeypatch.setattr(
        assets, "fetch", lambda name, **kw: fetched.append(name) or tmp_path / name
    )

    assert main(["setup"]) == 0
    out = capsys.readouterr().out
    assert fetched == [
        "boltz2",
        "opendde",
        "openfold3",
        "protenix",
    ], "every model whose weights are published"
    assert "alphafold3  manual" in out and "Request the parameters" in out
    # AlphaFold 3 is the only model a person has to fetch by hand, because it
    # is the only one whose parameters are not published for redistribution.
    assert "openfold3   opt-in" not in out
    assert "esmfold2    opt-in" in out, "still opt-in, for its 26.8 GB"
    assert "foldjax setup --all" in out, "the opt-in must say how to include it"


def test_setup_all_takes_the_models_held_back_for_their_size(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`--all` is the one-command install of everything that is published.

    Size is the only reason a public model sits out of the default, so there
    has to be a way to say "take it too" without naming each one. AlphaFold 3
    is not included by any flag: its parameters are licensed rather than large,
    and FoldJAX has nothing to download.
    """
    from foldjax import assets
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    monkeypatch.delenv("PROTENIX_TEMPLATE_MMCIF_DIR", raising=False)
    monkeypatch.setattr(build, "is_ready", lambda: False)
    monkeypatch.setattr(build, "runtime_blocker", lambda: None)
    monkeypatch.setattr(
        build,
        "ensure_ready",
        lambda: pytest.fail("setup must report, not build, the AlphaFold runtime"),
    )
    fetched = []
    monkeypatch.setattr(
        assets, "fetch", lambda name, **kw: fetched.append(name) or tmp_path / name
    )

    assert main(["setup", "--all"]) == 0
    out = capsys.readouterr().out
    assert fetched == [
        "boltz2",
        "esmfold2",
        "opendde",
        "openfold3",
        "protenix",
    ], "every published model, ESMFold2 included"
    assert "alphafold3  manual" in out, "a licence is not a size"
    assert "runtime" in out
    assert "foldjax runtime prepare --model alphafold3" in out
    # The two modalities that need no weights still need an answer.
    assert "msa" in out and "ColabFold" in out
    assert "PROTENIX_TEMPLATE_MMCIF_DIR" in out


def test_setup_exit_status_follows_the_public_downloads(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))

    def explode(name, **kwargs):
        raise RuntimeError(f"{name} download failed")

    monkeypatch.setattr(assets, "fetch", explode)
    assert main(["setup"]) == 1
    assert "download failed" in capsys.readouterr().err


def test_capabilities_cli_reports_one_backend(capsys) -> None:
    """Asked by an alias, so the CLI's name resolution is covered too."""
    assert main(["capabilities", "--model", "protenix-jax"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "protenix"
    assert "foldjax" in payload["input_formats"]
    assert payload["input_requirements"]["foldjax"]["prediction_runtime"] == "jax"
    assert payload["input_requirements"]["foldjax"]["required_extras"] == []


def test_predict_cli_prints_result_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    seen = {}

    def fake_predict(request):
        from foldjax.schema import BatchReport, PredictionResult

        seen["request"] = request
        return BatchReport(
            results=(PredictionResult(model="protenix", output_dir=request.output_dir),)
        )

    monkeypatch.setattr("foldjax.cli.predict_batch", fake_predict)
    code = main(
        [
            "predict",
            "--model",
            "protenix",
            "--input",
            str(input_path),
            "--weights",
            str(weights),
            "--profile",
            "released",
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "3",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--option",
            "n_step=20",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["model"] == "protenix"
    assert seen["request"].seed == 3
    assert seen["request"].cache_dir == tmp_path / "cache"
    assert seen["request"].profile == "released"
    assert seen["request"].options == {"n_step": 20}


def test_cache_warm_cli_reports_execute_once_and_cache_delta(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import cli, warmup
    from foldjax.cache import CacheSnapshot
    from foldjax.warmup import CacheWarmResult

    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n")
    weights = tmp_path / "weights.jax"
    weights.write_bytes(b"weights")
    cache = tmp_path / "cache"
    namespace = cache / "boltz2" / "weights" / "digest"
    seen = {}

    def fake_resolve(request):
        resolved = dataclasses.replace(
            request,
            model="boltz2",
            models=None,
            input_format="native",
            weights=weights,
            cache_dir=cache,
        )
        return (resolved,)

    def fake_warm(request):
        seen["request"] = request
        print("native warm progress")
        return CacheWarmResult(
            model="boltz2",
            input=input_path,
            weights=weights,
            cache_dir=namespace,
            seed=0,
            seconds=12.5,
            before=CacheSnapshot(files=1, bytes=100),
            after=CacheSnapshot(files=3, bytes=400),
            samples=1,
            peak_device_bytes=2**30,
        )

    monkeypatch.setattr(cli, "resolve_requests", fake_resolve)
    monkeypatch.setattr(warmup, "warm_cache", fake_warm)

    assert (
        main(
            [
                "cache",
                "warm",
                "--model",
                "boltz2",
                "--input",
                str(input_path),
                "--weights",
                str(weights),
                "--cache-dir",
                str(cache),
                "--num-steps",
                "1",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["strategy"] == "execute_once"
    assert payload["status"] == "populated"
    assert payload["cache"]["new_files"] == 2
    assert payload["peak_device_bytes"] == 2**30
    assert "prediction files are discarded" in captured.err
    assert "native warm progress" in captured.err
    assert "native warm progress" not in captured.out
    assert "new_files=2" in captured.err
    assert "peak_device=1.0 GiB" in captured.err
    assert seen["request"].use_compile_cache is True


def test_plan_profile_resolves_protenix_variant_without_native_options(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax import assets

    input_path = tmp_path / "job.json"
    input_path.write_text('[{"sequences": []}]')
    weights = tmp_path / "protenix-mini-esm" / "mini.jax"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    calls = []

    def fake_resolve(model: str, *, profile: str | None = None) -> Path:
        calls.append((model, profile))
        return weights

    monkeypatch.setattr(assets, "resolve_weights", fake_resolve)
    assert (
        main(
            [
                "plan",
                "--model",
                "protenix",
                "--input",
                str(input_path),
                "--profile",
                "mini-esm-v0.5.0",
                "--no-cache",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert calls == [("protenix", "mini-esm-v0.5.0")]
    assert payload["profile"] == "mini-esm-v0.5.0"
    assert payload["weights"] == str(weights)
    assert payload["options"] == {
        "esm_checkpoint_dir": str(weights.parent),
        "model_name": "protenix_mini_esm_v0.5.0",
    }


def test_explicit_zero_seed_is_still_exclusive_with_seed_list(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    with pytest.raises(ValueError, match="--seed and --seeds are mutually exclusive"):
        main(
            [
                "predict",
                "--model",
                "protenix",
                "--input",
                str(input_path),
                "--seed",
                "0",
                "--seeds",
                "1",
                "2",
            ]
        )


def test_batch_predict_cli_prints_a_json_array(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    seen = {}

    def fake_predict(request):
        from foldjax.schema import BatchReport, PredictionResult

        seen["request"] = request
        return BatchReport(
            results=(
                PredictionResult(model="boltz2", output_dir=tmp_path / "boltz2"),
                PredictionResult(model="protenix", output_dir=tmp_path / "protenix"),
            )
        )

    monkeypatch.setattr("foldjax.cli.predict_batch", fake_predict)
    assert (
        main(
            [
                "predict",
                "--model",
                "boltz2",
                "protenix",
                "--input",
                str(input_path),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["model"] for item in payload] == ["boltz2", "protenix"]
    assert seen["request"].models == ("boltz2", "protenix")


def test_batch_plan_cli_prints_every_resolved_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "alpha.json"
    second = tmp_path / "beta.json"
    first.write_text("{}")
    second.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    output = tmp_path / "out"

    assert (
        main(
            [
                "plan",
                "--model",
                "boltz",
                "--input",
                str(first),
                str(second),
                "--weights",
                str(weights),
                "--output-dir",
                str(output),
                "--no-cache",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["model"] for item in payload] == ["boltz2", "boltz2"]
    assert [item["output_dir"] for item in payload] == [
        str(output / "boltz2" / "alpha"),
        str(output / "boltz2" / "beta"),
    ]
    assert all(item["cache_dir"] is None for item in payload)


def test_plan_cli_redacts_secret_options(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    secret = "do-not-print-this"

    assert (
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(input_path),
                "--weights",
                str(weights),
                "--no-cache",
                "--option",
                f"msa_server_password={secret}",
                "--option",
                "steering_args="
                f'{{"headers":{{"Authorization":"Bearer {secret}","safe":"ok"}}}}',
            ]
        )
        == 0
    )

    text = capsys.readouterr().out
    payload = json.loads(text)
    assert secret not in text
    assert payload["options"] == {
        "msa_server_password": "[REDACTED]",
        "steering_args": {
            "headers": {"Authorization": "[REDACTED]", "safe": "ok"}
        },
    }


@pytest.mark.parametrize(
    "extra",
    [
        ["--input-format", "native"],
        ["--option", "num_stpes=20"],
        ["--option", 'no_language_model="maybe"'],
    ],
)
def test_plan_refuses_requests_that_predict_would_reject(
    tmp_path: Path, extra: list[str]
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text('{"entities": []}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")

    with pytest.raises(ValueError):
        main(
            [
                "plan",
                "--model",
                "esmfold2",
                "--input",
                str(input_path),
                "--weights",
                str(weights),
                "--no-cache",
                *extra,
            ]
        )


def test_plan_validates_mem_fraction_without_mutating_the_allocator(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "job.json"
    input_path.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)

    with pytest.raises(ValueError, match="must be in"):
        main(
            [
                "plan",
                "--model",
                "boltz2",
                "--input",
                str(input_path),
                "--weights",
                str(weights),
                "--mem-fraction",
                "99",
            ]
        )

    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ


def test_cli_options_preserve_json_types() -> None:
    assert _options(
        ["steps=20", "enabled=true", "buckets=[256,512]", "dtype=bf16"]
    ) == {
        "steps": 20,
        "enabled": True,
        "buckets": [256, 512],
        "dtype": "bf16",
    }


def test_cli_options_require_key_value_pairs() -> None:
    with pytest.raises(ValueError, match="must be KEY=VALUE"):
        _options(["steps"])


def test_cli_options_strip_keys() -> None:
    assert _options(["  steps  =20"]) == {"steps": 20}


@pytest.mark.parametrize("item", ["=20", "   =20"])
def test_cli_options_reject_blank_keys(item: str) -> None:
    with pytest.raises(ValueError, match="key must be non-empty"):
        _options([item])


def test_cli_options_reject_duplicate_normalized_keys() -> None:
    with pytest.raises(ValueError, match="set more than once"):
        _options(["steps=20", " steps =40"])


def test_entrypoint_exits_with_the_command_status() -> None:
    with pytest.raises(SystemExit) as exit_info:
        entrypoint()
    assert exit_info.value.code == 2  # argparse: a subcommand is required


def test_user_errors_are_one_clean_line_not_a_traceback(capsys, monkeypatch) -> None:
    """A bad request is the user's problem; a traceback would imply ours."""
    from foldjax import cli

    monkeypatch.setattr(
        cli,
        "main",
        lambda argv=None: (_ for _ in ()).throw(ValueError("no such model 'x'")),
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "foldjax: no such model 'x'"
    assert "Traceback" not in captured.err


def test_memory_errors_are_one_clean_line(capsys, monkeypatch) -> None:
    from foldjax import cli

    monkeypatch.setattr(
        cli,
        "main",
        lambda argv=None: (_ for _ in ()).throw(MemoryError("GPU is full")),
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()

    assert exit_info.value.code == 2
    assert capsys.readouterr().err.strip() == "foldjax: GPU is full"


def test_unexpected_failures_keep_their_traceback(monkeypatch) -> None:
    from foldjax import cli

    def fail(argv=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "main", fail)
    with pytest.raises(RuntimeError, match="boom"):
        cli.entrypoint()


def test_a_successful_run_still_exits_zero(monkeypatch) -> None:
    from foldjax import cli

    monkeypatch.setattr(cli, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.entrypoint()
    assert exit_info.value.code == 0


def _make_generation(root: Path, name: str, *, age_days: float, size: int = 4096):
    """A runtime tree of the shape `gc` walks, aged by its mtime."""
    import os
    import time

    tree = root / name / "alphafold3" / "constants" / "converters"
    tree.mkdir(parents=True)
    (tree / "ccd.pickle").write_bytes(b"x" * size)
    when = time.time() - age_days * 86_400
    os.utime(root / name, (when, when))
    return root / name


def test_runtime_gc_removes_abandoned_generations_and_spares_the_live_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Every source edit mints a ~1 GB tree and abandons the old one in place.

    The guard that matters is age rather than reachability alone: this store is
    shared between checkouts, and a worktree at another commit has its own key
    and its own live tree. A tree prepared an hour ago is probably one of those,
    so `gc` leaves it and says it did.
    """
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    base = tmp_path / "runtime" / "alphafold3"
    base.mkdir(parents=True)
    live = _make_generation(base, "live", age_days=0.1)
    old = _make_generation(base, "old", age_days=30)
    recent = _make_generation(base, "recent", age_days=0.5)
    monkeypatch.setattr(build, "runtime_key", lambda: "live")

    assert main(["runtime", "gc", "--model", "alphafold3"]) == 0
    out = capsys.readouterr().out

    assert live.is_dir(), "the generation this source selects must survive"
    assert recent.is_dir(), "a tree another checkout may be using must survive"
    assert not old.exists(), "an abandoned tree past the age guard must go"
    assert "1 unreachable generation(s) kept as recent" in out


def test_runtime_gc_all_takes_every_generation_but_the_live_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    base = tmp_path / "runtime" / "alphafold3"
    base.mkdir(parents=True)
    live = _make_generation(base, "live", age_days=0.1)
    recent = _make_generation(base, "recent", age_days=0.5)
    monkeypatch.setattr(build, "runtime_key", lambda: "live")

    assert main(["runtime", "gc", "--model", "alphafold3", "--all"]) == 0
    capsys.readouterr()

    assert live.is_dir()
    assert not recent.exists()
    with pytest.raises(ValueError, match="refusing to remove"):
        build.remove_generation(live)


def test_runtime_gc_dry_run_reports_without_removing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from foldjax.models.alphafold3 import build

    monkeypatch.setenv("FOLDJAX_HOME", str(tmp_path))
    base = tmp_path / "runtime" / "alphafold3"
    base.mkdir(parents=True)
    _make_generation(base, "live", age_days=0.1)
    old = _make_generation(base, "old", age_days=30)
    monkeypatch.setattr(build, "runtime_key", lambda: "live")

    assert main(["runtime", "gc", "--model", "alphafold3", "--dry-run"]) == 0

    assert old.is_dir(), "a dry run must not remove anything"
    assert "would remove" in capsys.readouterr().out
