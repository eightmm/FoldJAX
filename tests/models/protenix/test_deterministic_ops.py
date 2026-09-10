"""Repeatable reduction orders, asked for per executable.

Running this port under the process-wide
``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` made it bitwise repeatable
across processes and moved one master-panel case into the pass band, at 13%
of its wall time at 1,003 tokens
(``docs/protenix-master-panel-2026-09-09.md``). An environment variable is
read once per process, so it cannot say "this prediction and not the next
one", and in a benchmark process it silently reaches every other model.

These pin the same setting carried on the executables this port builds. Two
properties matter more than the plumbing:

* with the option off, ``jax.jit`` receives exactly the arguments it received
  before this existed -- not an empty option map, which is a different
  compile -- so every measurement taken so far still describes the default
  run, and
* the flag name lives in one shared constant. The narrower
  ``--xla_gpu_exclude_nondeterministic_ops`` is still being measured at 3,012
  tokens, where the autotuner failed under this one, so switching has to be an
  edit in one place.

The first is pinned below. The second is now a property of the shared
``foldjax.models._compile_policy`` that every port reads, so
``tests/models/test_compile_policy.py`` owns it, along with the proof that
the option reaches the compiler at all rather than only reaching ``jax.jit``.
The rest of this file is this port's own plumbing: which owners exist, which
one a run gets, and how the flag travels from argv to the executable.
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
from foldjax.models.protenix.cli import predict as predict_cli
from foldjax.models.protenix.data import esm
from foldjax.models.protenix.models import model as model_impl
from foldjax.models.protenix.models import predict as predict_impl

from .test_esm_features import _TOY_CONFIG, _save_toy_checkpoint
from .test_model import _toy_features, _toy_params

_STATIC_ARGNAMES = (
    *model_impl.GRAPH_STATIC_ARGNAMES,
    "params_treedef",
    "params_flags",
)


@pytest.fixture
def recorded_jit(monkeypatch) -> list[dict[str, Any]]:
    """Record what the pool asks ``jax.jit`` for, compiling nothing.

    Both graph owners are emptied first. They are module-level and cache by
    identity, so an earlier test in the same session that ran the same toy
    shapes would otherwise satisfy the call from its cache, record nothing,
    and let "no compiler options were passed" pass on an empty list.
    """
    calls: list[dict[str, Any]] = []

    def fake_jit(function: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return lambda *args, **call_kwargs: {}

    model_impl._compiled_protenix_infer.clear_cache()
    model_impl._compiled_protenix_infer_deterministic.clear_cache()
    monkeypatch.setattr(jax, "jit", fake_jit)
    return calls


def _run_graph(deterministic: bool) -> None:
    """Drive the compiled entry point far enough to build one executable."""
    model_impl.protenix_infer_compiled(
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


def test_the_two_policies_are_two_owners() -> None:
    """One cache entry cannot serve both: the option is part of the build."""
    default = model_impl._compiled_protenix_infer
    deterministic = model_impl._compiled_protenix_infer_deterministic

    assert default is not deterministic
    assert model_impl._infer_pool(False) is default
    assert model_impl._infer_pool(True) is deterministic
    assert default._compiler_options is None
    assert deterministic._compiler_options == DETERMINISTIC_COMPILER_OPTIONS


@pytest.mark.parametrize(
    "eager",
    ({"graph_jit": False}, {"guidance_config": {"enable": True}}),
    ids=("no-graph-jit", "guidance"),
)
def test_the_eager_path_refuses_instead_of_running_without_the_option(
    eager: dict[str, Any],
) -> None:
    """It dispatches shared module-level primitives and owns no executable.

    Running it anyway would report a deterministic run that was not one, which
    is the failure this whole option exists to remove.
    """
    with pytest.raises(ValueError, match="deterministic reductions"):
        predict_impl.protenix_predict_static(
            None,
            {},
            None,
            deterministic=True,
            **eager,
        )


def test_the_language_model_encoder_takes_the_same_option(tmp_path: Path) -> None:
    """ESM/ISM profiles run it as their own executable before the graph.

    Wiring only the structure graph would make ``deterministic=on`` a partial
    promise on exactly the profiles whose embeddings feed everything else.
    """
    from foldjax.models.protenix.data.esm import (
        encode_esm2_sequence,
        esm2_forward,
        load_esm2_checkpoint,
    )

    checkpoint = tmp_path / "toy.npz"
    _save_toy_checkpoint(checkpoint)
    weights = load_esm2_checkpoint(checkpoint, config=_TOY_CONFIG)
    tokens = encode_esm2_sequence("AG")
    esm._compiled_esm2_layer.clear_cache()
    esm._compiled_esm2_layer_deterministic.clear_cache()

    plain = np.asarray(esm2_forward(weights, tokens))
    assert esm._compiled_esm2_layer._entry_count() == 1
    assert esm._compiled_esm2_layer_deterministic._entry_count() == 0

    repeatable = np.asarray(esm2_forward(weights, tokens, deterministic=True))
    assert esm._compiled_esm2_layer_deterministic._entry_count() == 1
    np.testing.assert_allclose(repeatable, plain, rtol=1e-5, atol=1e-5)

    assert esm._layer_pool(False) is esm._compiled_esm2_layer
    assert esm._layer_pool(True) is esm._compiled_esm2_layer_deterministic


def test_the_provider_carries_it_to_the_encoder(tmp_path: Path) -> None:
    """The CLI builds one provider; the option has to survive that hop."""
    from foldjax.models.protenix.data.esm import JaxEsmProvider

    provider = JaxEsmProvider(
        "esm2-3b", checkpoint_dir=tmp_path, deterministic=True
    )

    assert provider.deterministic is True
    assert JaxEsmProvider("esm2-3b", checkpoint_dir=tmp_path).deterministic is False


class _DefaultsCapturedError(Exception):
    pass


def test_the_cli_flag_defaults_to_off(monkeypatch) -> None:
    """Off is the shipped run, so it is the parser's answer when unasked."""
    captured: dict[str, Any] = {}

    def capture(parser: argparse.ArgumentParser, *_args: object) -> None:
        captured.update(
            (action.dest, action) for action in parser._actions
        )
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
                str(tmp_path / "weights.jax"),
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
def test_the_cli_threads_the_flag_to_the_prediction(
    tmp_path: Path,
    monkeypatch,
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    """The flag is worth nothing if it stops at ``args``."""
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "protenix.jax"
    weights.write_bytes(b"native fixture")

    from foldjax.models.protenix.data import featurize_json

    monkeypatch.setattr(
        featurize_json,
        "featurize_protein_json",
        lambda *_args, **_kwargs: dict(_toy_features()),
    )
    monkeypatch.setattr(
        predict_cli,
        "_load_prepared_params",
        lambda *_args, **_kwargs: _toy_params(),
    )
    captured: list[bool] = []

    def capture_predict(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs["deterministic"])
        return {}

    monkeypatch.setattr(predict_impl, "protenix_predict_static", capture_predict)

    predict_cli.main(
        [
            "--input-json",
            str(job),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            "--model-name",
            "protenix_mini_default_v0.5.0",
            "--no-compile-cache",
            "--prewarm-only",
            *argv,
        ]
    )

    assert captured == [expected]
