from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import jax
import pytest

from foldjax import warm_buckets
from foldjax.cache import CacheSnapshot
from foldjax.cli import _parser, _request, main
from foldjax.padding import TOKEN_BUCKETS
from foldjax.schema import PaddingConfig, PredictionRequest
from foldjax.warmup import CacheWarmResult


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("768,256,512,256", (256, 512, 768)),
        ("256..512", (256, 384, 512)),
        (" 256, 512 ", (256, 512)),
        ("auto", TOKEN_BUCKETS),
    ],
)
def test_parse_bucket_spec(spec, expected):
    assert warm_buckets.parse_token_buckets(spec, model="boltz2") == expected


@pytest.mark.parametrize(
    "spec",
    ["", "0", "1", "-2", "512..256", "0..512", "4..8", "256,,512", "256..512..768"],
)
def test_invalid_bucket_spec(spec):
    with pytest.raises(ValueError):
        warm_buckets.parse_token_buckets(spec, model="boltz2")


def test_af3_native_auto_grid_is_pinned_to_the_carried_runner():
    runner = (
        Path(warm_buckets.__file__).parent
        / "backends/_alphafold3_upstream/run_alphafold.py"
    )
    tree = ast.parse(runner.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_BUCKETS"
            for target in node.targets
        )
    )
    expected = tuple(map(int, ast.literal_eval(assignment.value.args[1])))
    assert warm_buckets.parse_token_buckets("auto", model="af3") == expected
    assert warm_buckets.parse_token_buckets("256..768", model="af3") == (256, 512, 768)
    assert warm_buckets.parse_token_buckets(
        "auto", model="af3", options={"buckets": [64, 96]}
    ) == (64, 96)


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("{}")
    weights = tmp_path / "weights.jax"
    weights.touch()
    request = PredictionRequest(
        model="boltz2",
        input=source,
        weights=weights,
        cache_dir=tmp_path / "cache",
        msa="single",
        padding=PaddingConfig(msa=7),
        max_msa_depth=7,
    )
    calls = []
    failure = {}
    monkeypatch.setattr(
        warm_buckets,
        "device_summary",
        lambda _request=None: {
            "platform": "gpu",
            "device_kind": "mock GPU",
            "card_bytes": 24 * 1024**3,
            "pool_bytes": 21 * 1024**3,
        },
    )

    def run(item):
        document = json.loads(item.input.read_text())
        assert len(document["entities"]) == 1
        length = len(document["entities"][0]["sequence"])
        calls.append((item.model, item.padding.tokens, length, item))
        assert item.padding.msa == item.max_msa_depth == 7
        assert item.msa == "single"
        if failure.get("bucket") == item.padding.tokens:
            raise failure["error"]
        logging.getLogger("jax._src.compiler").warning(
            "Compiling jit(tiny) with input shapes"
        )
        return CacheWarmResult(
            model=item.model,
            input=item.input,
            weights=weights,
            cache_dir=tmp_path / "cache" / str(item.padding.tokens),
            seed=0,
            seconds=0.1,
            before=CacheSnapshot(),
            after=CacheSnapshot(files=1, bytes=128),
            samples=1,
            shape_profile={"target": {"tokens": item.padding.tokens, "msa": 7}},
        )

    monkeypatch.setattr(warm_buckets, "warm_cache", run)
    return request, calls, failure


def test_two_real_lengths_execute_ascending_buckets_and_report_device(sweep):
    request, calls, _ = sweep
    before = jax.config.jax_log_compiles
    result = warm_buckets.warm_token_buckets(request, "512,256")
    assert [call[1] for call in calls] == [256, 256, 512, 512]
    assert calls[0][2] != calls[1][2]
    assert all(length <= bucket for _, bucket, length, _ in calls)
    assert result["device"]["card_bytes"] == 24 * 1024**3
    row = result["models"][0]
    assert row["successful_buckets"] == [256, 512]
    assert row["first_failed_bucket"] is None
    assert row["buckets"][0]["verification"]["second_length_compile_requests"] == 1
    assert jax.config.jax_log_compiles == before
    assert "mock GPU (24.0 GiB)" in warm_buckets.format_bucket_report(result)


@pytest.mark.parametrize(
    "keep_going,expected", [(False, [128, 128, 256]), (True, [128, 128, 256, 512, 512])]
)
def test_stop_on_oom_is_the_default_and_can_be_disabled(sweep, keep_going, expected):
    request, calls, failure = sweep
    failure.update(
        bucket=256,
        error=RuntimeError("cudaErrorMemoryAllocation before calling cuModuleLoad"),
    )
    result = warm_buckets.warm_token_buckets(
        request, "128,256,512", stop_on_oom=not keep_going
    )
    assert [call[1] for call in calls] == expected
    model = result["models"][0]
    assert model["first_failed_bucket"] == 256
    assert model["stopped_on_oom"] is (not keep_going)
    assert "cudaErrorMemoryAllocation" in model["failure_reason"]


def test_non_oom_failure_is_reported_without_misclassification(sweep):
    request, calls, failure = sweep
    failure.update(bucket=256, error=ValueError("bad model configuration"))
    result = warm_buckets.warm_token_buckets(request, "256,512")
    assert [call[1] for call in calls] == [256, 512, 512]
    assert not result["models"][0]["buckets"][0]["oom"]


@pytest.mark.parametrize("json_output", [False, True])
def test_bucket_cli_routes_native_progress_to_stderr_and_formats_output(
    sweep, capsys, json_output
):
    _, _, _ = sweep
    argv = [
        "cache",
        "warm",
        "--isolation",
        "in-process",
        "--model",
        "boltz2",
        "--token-buckets",
        "128",
        "--fixed-msa",
        "7",
    ]
    if json_output:
        argv.append("--json")
    assert main(argv) == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "foldjax predict --model boltz2 --input JOB.json --padding" in captured.err
    assert "--fixed-msa 7" in captured.err
    assert "Prediction command template" not in output
    if json_output:
        payload = json.loads(output)
        assert payload["strategy"] == "token_buckets"
        assert payload["models"][0]["successful_buckets"] == [128]
    else:
        assert "boltz2" in output and "successful: [128]" in output


@pytest.mark.parametrize("model", ["boltz2", "protenix", "openfold3", "opendde"])
@pytest.mark.parametrize("command", [("predict",), ("plan",), ("cache", "warm")])
def test_fixed_msa_cli_expands_exactly_to_existing_public_fields(
    tmp_path, model, command
):
    args = _parser().parse_args(
        [*command, "--model", model, "--sequence", "ACD", "--fixed-msa", "7"]
    )
    result = _request(args, jobs_root=tmp_path)
    assert result.max_msa_depth == result.padding.msa == 7
    assert result.padding.tokens is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--fixed-msa", "0"],
        ["--fixed-msa", "-1"],
        ["--fixed-msa", "7", "--max-msa-depth", "8"],
        ["--fixed-msa", "7", "--pad-msa", "8"],
    ],
)
def test_fixed_msa_rejects_invalid_or_conflicting_options(tmp_path, extra):
    args = _parser().parse_args(
        ["plan", "--model", "boltz2", "--sequence", "ACD", *extra]
    )
    with pytest.raises(ValueError, match="--fixed-msa"):
        _request(args, jobs_root=tmp_path)


@pytest.mark.parametrize("model", ["af3", "esmfold2"])
def test_fixed_msa_rejects_unsupported_models(tmp_path, model):
    args = _parser().parse_args(
        ["plan", "--model", model, "--sequence", "ACD", "--fixed-msa", "7"]
    )
    with pytest.raises(ValueError, match="unsupported"):
        _request(args, jobs_root=tmp_path)


def test_synthetic_warming_does_not_search_external_servers(sweep):
    import dataclasses

    request, calls, _ = sweep
    with pytest.raises(ValueError, match="--msa single"):
        warm_buckets.warm_token_buckets(dataclasses.replace(request, msa="auto"), "256")
    assert not calls


def test_memory_admission_refusal_stops_larger_buckets(sweep):
    request, calls, failure = sweep
    failure.update(
        bucket=256, error=MemoryError("peak estimate exceeds device memory budget")
    )
    result = warm_buckets.warm_token_buckets(request, "128,256,512")
    assert [call[1] for call in calls] == [128, 128, 256]
    assert result["models"][0]["stopped_on_oom"]


def test_generated_output_symlinks_do_not_escape_the_requested_root(sweep, tmp_path):
    import dataclasses

    request, calls, _ = sweep
    output = tmp_path / "results"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "boltz2").symlink_to(outside, target_is_directory=True)
    result = warm_buckets.warm_token_buckets(
        dataclasses.replace(request, output_dir=output), "128"
    )
    assert not calls
    assert "symlink" in result["models"][0]["failure_reason"]
    assert not list(outside.iterdir())


def test_native_af3_bucket_options_preserve_the_prediction_cache_profile(
    sweep, monkeypatch
):
    import dataclasses

    from foldjax.registry import get_backend

    request, _, _ = sweep
    request = dataclasses.replace(
        request,
        model="alphafold3",
        padding=None,
        max_msa_depth=None,
        options={"buckets": [64, 96]},
    )
    backend = get_backend("alphafold3")
    expected = backend.cache_profile(request)
    seen = []

    def run(item):
        assert item.padding is None
        assert backend.cache_profile(item) == expected
        seen.append(item)
        return CacheWarmResult(
            model="alphafold3",
            input=item.input,
            weights=request.weights,
            cache_dir=request.cache_dir,
            seed=0,
            seconds=0.01,
            before=CacheSnapshot(),
            after=CacheSnapshot(),
            samples=1,
        )

    monkeypatch.setattr(warm_buckets, "warm_cache", run)
    result = warm_buckets.warm_token_buckets(request, "auto")
    assert result["models"][0]["successful_buckets"] == [64, 96]
    assert len(seen) == 4


def test_native_af3_selection_refuses_buckets_outside_its_grid():
    with pytest.raises(ValueError, match="must belong"):
        warm_buckets.parse_token_buckets(
            "128", model="af3", options={"buckets": [64, 96]}
        )


@pytest.mark.parametrize(
    "spec,grid",
    [("auto", [1, 64]), ("1..64", [1, 64]), ("auto", [64, 65]), ("65", [64, 65])],
)
def test_native_af3_grid_needs_two_distinct_real_lengths(spec, grid):
    with pytest.raises(ValueError, match="two|at least 2"):
        warm_buckets.parse_token_buckets(spec, model="af3", options={"buckets": grid})
