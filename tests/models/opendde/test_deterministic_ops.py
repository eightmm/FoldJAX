"""Repeatable reduction orders, asked for per executable.

The setting is the shared one every port carries -- see
``foldjax.models._compile_policy`` for what it is, why it rides on the
executable instead of on the process, and
``tests/models/test_compile_policy.py`` for the proof that it reaches the
compiler at all. This file is OpenDDE's own plumbing: which owners exist,
which one a run gets, and how the flag travels from argv to both of them.

*Both* of them is what is specific here. OpenDDE builds two executables per
prediction, not one: the inference graph, and the shape-complementarity stage
that finishes the confidence output on the host after the graph has returned.
That second stage scatters atom contributions into tokens -- the exact op class
the option targets -- so wiring only the graph would report a deterministic run
whose confidence fields were not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._compile_policy import DETERMINISTIC_COMPILER_OPTIONS
from foldjax.models.opendde import postprocess as postprocess_impl
from foldjax.models.opendde.cli import predict as predict_cli
from foldjax.models.opendde.models import model as model_impl
from foldjax.models.opendde.models import shape_complementarity as shape_comp

_STATIC_ARGNAMES = (
    *model_impl.GRAPH_STATIC_ARGNAMES,
    "params_treedef",
    "params_flags",
)

_FIXTURE = Path(__file__).parent / "data" / "shape_complementarity.npz"


@pytest.fixture
def recorded_jit(monkeypatch) -> list[dict[str, Any]]:
    """Record what the pool asks ``jax.jit`` for, compiling nothing.

    Both graph owners are emptied first. They are module-level and cache by
    identity, so an earlier test in the same session that ran the same toy
    shapes would otherwise satisfy the call from its cache, record nothing,
    and let "no compiler options were passed" pass on an empty list.

    The structural-feature check runs on concrete values before the pool is
    reached and would reject the one-token stand-in below; it is not what this
    test is about, and driving it would need a full feature tree.
    """
    calls: list[dict[str, Any]] = []

    def fake_jit(function: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return lambda *args, **call_kwargs: {}

    model_impl._compiled_opendde_infer.clear_cache()
    model_impl._compiled_opendde_infer_deterministic.clear_cache()
    monkeypatch.setattr(
        model_impl, "_validate_static_structural_features", lambda *a, **k: None
    )
    monkeypatch.setattr(jax, "jit", fake_jit)
    return calls


def _run_graph(deterministic: bool) -> None:
    """Drive the compiled entry point far enough to build one executable."""
    model_impl.opendde_infer_compiled(
        {"asym_id": jnp.asarray([0], dtype=jnp.int32)},
        (),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        deterministic=deterministic,
    )


def test_off_builds_the_executable_exactly_as_it_did_before(recorded_jit) -> None:
    """The default run must be the run every recorded number describes."""
    _run_graph(False)

    assert recorded_jit == [{"static_argnames": _STATIC_ARGNAMES}]


def test_on_carries_the_options_into_the_compile(recorded_jit) -> None:
    """And the values come from the constant, not from a literal here."""
    _run_graph(True)

    assert len(recorded_jit) == 1
    assert recorded_jit[0]["static_argnames"] == _STATIC_ARGNAMES
    assert recorded_jit[0]["compiler_options"] == DETERMINISTIC_COMPILER_OPTIONS


def test_the_two_graph_policies_are_two_owners() -> None:
    """One cache entry cannot serve both: the option is part of the build."""
    default = model_impl._compiled_opendde_infer
    deterministic = model_impl._compiled_opendde_infer_deterministic

    assert default is not deterministic
    assert model_impl._infer_pool(False) is default
    assert model_impl._infer_pool(True) is deterministic
    assert default._compiler_options is None
    assert deterministic._compiler_options == DETERMINISTIC_COMPILER_OPTIONS


def test_the_shape_complementarity_stage_is_two_owners_as_well() -> None:
    """The second executable a prediction builds, and the scattering one."""
    default = shape_comp._compiled_shape_complementarity
    deterministic = shape_comp._compiled_shape_complementarity_deterministic

    assert default is not deterministic
    assert shape_comp._shape_complementarity_pool(False) is default
    assert shape_comp._shape_complementarity_pool(True) is deterministic
    assert default._compiler_options is None
    assert deterministic._compiler_options == DETERMINISTIC_COMPILER_OPTIONS


def test_the_eager_path_refuses_instead_of_running_without_the_option() -> None:
    """``--no-graph-jit`` dispatches shared primitives and owns no executable.

    Running it anyway would report a deterministic run that was not one, which
    is the failure this whole option exists to remove.
    """
    with pytest.raises(ValueError, match="deterministic reductions"):
        predict_cli._predict(
            {},
            None,
            seed=0,
            num_samples=1,
            num_steps=2,
            num_recycles=1,
            n_queries=8,
            n_keys=16,
            diffusion_attention_backend="xla",
            trunk_single_attention_backend="xla",
            structural_single_attention_backend="xla",
            graph_jit=False,
            deterministic=True,
            cycle_msa_features=(),
        )


def test_the_single_sample_stage_shape_refuses_rather_than_running_eagerly(
    tmp_path: Path,
) -> None:
    """That shape keeps the historical eager boundary and owns no executable.

    A prediction always has a sample axis, so this is unreachable from the
    CLI; a direct caller asking for both gets the refusal rather than the
    silent partial promise.
    """
    features = {
        "token_index": np.arange(2, dtype=np.int64),
        "atom_to_token_idx": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "asym_id": np.asarray([0, 1], dtype=np.int64),
        "distogram_rep_atom_mask": np.asarray([1, 0, 1, 0], dtype=bool),
    }

    with pytest.raises(ValueError, match="deterministic reductions"):
        shape_comp.compute_shape_complementarity_batched(
            jnp.zeros((4, 3), dtype=jnp.float32),
            features,
            jnp.ones((4,), dtype=bool),
            deterministic=True,
        )


def test_the_stage_runs_under_both_policies_and_agrees_with_itself() -> None:
    """Recording which owner was asked for would pass against a dead option."""
    if not _FIXTURE.exists():
        pytest.skip(f"missing {_FIXTURE}")
    recorded = np.load(_FIXTURE, allow_pickle=True)
    features = {
        name[len("input_") :]: recorded[name]
        for name in recorded.files
        if name.startswith("input_")
    }
    coordinate = np.asarray(recorded["coordinate"])
    samples = jnp.asarray(np.stack((coordinate, coordinate + 0.125)))
    atom_mask = jnp.asarray(recorded["atom_mask"])
    shape_comp._compiled_shape_complementarity.clear_cache()
    shape_comp._compiled_shape_complementarity_deterministic.clear_cache()

    plain = shape_comp.compute_shape_complementarity_batched(
        samples, features, atom_mask
    )
    assert shape_comp._compiled_shape_complementarity._entry_count() == 1
    assert shape_comp._compiled_shape_complementarity_deterministic._entry_count() == 0

    repeatable = shape_comp.compute_shape_complementarity_batched(
        samples, features, atom_mask, deterministic=True
    )
    assert shape_comp._compiled_shape_complementarity_deterministic._entry_count() == 1

    for name, value in plain.items():
        np.testing.assert_allclose(
            np.asarray(repeatable[name], dtype=np.float64),
            np.asarray(value, dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
            err_msg=name,
        )


def _stage_features() -> dict[str, np.ndarray]:
    return {
        "token_index": np.arange(2, dtype=np.int64),
        "atom_to_token_idx": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "asym_id": np.asarray([0, 1], dtype=np.int64),
        "distogram_rep_atom_mask": np.asarray([1, 0, 1, 0], dtype=bool),
    }


def test_the_host_scores_carry_the_option_to_the_stage(monkeypatch) -> None:
    """The graph returns summaries; this stage is finished afterwards.

    Both branches of ``opendde_confidence_scores`` reach it, and the one the
    CLI takes is the passthrough -- the in-graph summaries with shape
    complementarity left out because it cannot trace.
    """
    captured: list[bool] = []

    def capture(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs["deterministic"])
        return {}

    monkeypatch.setattr(shape_comp, "compute_shape_complementarity_batched", capture)
    features = _stage_features()
    output = {
        "coordinate": jnp.zeros((2, 4, 3), dtype=jnp.float32),
        "summary_ranking_score": jnp.zeros((2,), dtype=jnp.float32),
    }

    postprocess_impl.opendde_confidence_scores(
        output, features, num_recycles=1, deterministic=True
    )
    postprocess_impl.opendde_confidence_scores(output, features, num_recycles=1)

    assert captured == [True, False]


class _DefaultsCapturedError(Exception):
    pass


def test_the_cli_flag_defaults_to_off(monkeypatch) -> None:
    """Off is the shipped run, so it is the parser's answer when unasked."""
    captured: dict[str, Any] = {}

    def capture(parser: argparse.ArgumentParser, *_args: object) -> None:
        captured.update((action.dest, action) for action in parser._actions)
        raise _DefaultsCapturedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_DefaultsCapturedError):
        predict_cli.main([])

    action = captured["deterministic_ops"]
    assert action.default == "off"
    assert tuple(action.choices) == ("off", "on")


def test_the_cli_rejects_a_value_outside_the_vocabulary(tmp_path: Path) -> None:
    """Not a silent off: a misspelled value must not look like a normal run."""
    with pytest.raises(SystemExit) as failure:
        predict_cli.main(
            [
                "--input-json",
                str(tmp_path / "job.json"),
                "--weights",
                str(tmp_path / "opendde.jax"),
                "--out",
                str(tmp_path / "out"),
                "--deterministic-ops",
                "yes",
            ]
        )

    assert failure.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected"),
    (((), False), (("--deterministic-ops", "on"), True)),
    ids=("unasked", "on"),
)
def test_the_cli_threads_the_flag_to_both_stages(
    tmp_path: Path,
    monkeypatch,
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    """The flag is worth nothing if it stops at ``args``.

    Both stages, because the confidence fields are finished after the graph
    returns and a run that carried the option into only one of them is the
    partial promise this file exists to rule out.
    """
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    input_path = tmp_path / "tiny.json"
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    features = {"restype": np.zeros((2, 32), dtype=np.float32)}
    predicted: list[bool] = []
    scored: list[bool] = []

    monkeypatch.setattr(predict_cli, "_load_jobs", lambda _path: [job])
    monkeypatch.setattr(predict_cli, "_featurize", lambda *_a, **_k: features)
    monkeypatch.setattr(
        predict_cli, "_load_prepared_params", lambda _path, _dtype: object()
    )

    def capture_predict(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        predicted.append(kwargs["deterministic"])
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    def capture_score(output: Any, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        scored.append(kwargs["deterministic"])
        return dict(output)

    monkeypatch.setattr(predict_cli, "_predict", capture_predict)
    monkeypatch.setattr(predict_cli, "_score", capture_score)
    monkeypatch.setattr(
        predict_cli,
        "_write",
        lambda root, **kwargs: [root / kwargs["job_name"] / "predictions" / "tiny.cif"],
    )

    predict_cli.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            *argv,
        ]
    )

    assert predicted == [expected]
    assert scored == [expected]
