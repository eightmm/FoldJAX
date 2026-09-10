"""Repeatable reduction orders, asked for per OpenFold3 executable.

This port is bitwise repeatable today only because the benchmark's sbatch
environment freezes XLA's autotune results per case with a process-wide
``XLA_FLAGS``. An environment variable is read once when the process starts,
so it cannot say "this prediction and not the next one", and in a benchmark
process it reaches every other model -- including the ones whose recorded
numbers were measured without it. The setting belongs on the executable that
wants it, which is what ``foldjax.models._compile_policy`` is for.

OpenFold3 owns five of them: the fused prediction JIT, and the four stage
functions the streamed path compiles (inputs, initialize, cycle, finish).
Wiring only the fused one would make ``deterministic=on`` a partial promise on
exactly the long-sequence runs that use the streamed scheduler.

Two properties matter more than the plumbing:

* with the option off, ``jax.jit`` receives exactly the arguments it received
  before this existed -- not an empty option map, which is a different
  compile -- so every measurement taken so far still describes the default
  run, and
* the flag names live in one shared constant, so replacing them is one edit.
  ``tests/models/test_compile_policy.py`` owns that guard and the proof that
  the option reaches the compiler rather than only reaching ``jax.jit``.

The rest is this port's own plumbing: which executables carry it, which paths
refuse it because they own no executable, and how it travels from a request or
argv into the graph identity.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from foldjax.backends import openfold3 as openfold3_backend
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models._compile_policy import DETERMINISTIC_COMPILER_OPTIONS
from foldjax.models.openfold3 import inference, streaming
from foldjax.models.openfold3.cli import predict as predict_cli
from foldjax.schema import PredictionRequest
from tests.models.openfold3.test_stable_compile import _config, _table


def _identity(**changes) -> inference._PredictGraphIdentity:
    return inference._PredictGraphIdentity(
        config=_config(msa_depth=8),
        n_chain=None,
        augment=True,
        use_trunk_pair_embedding=True,
        rng_route="native",
        triangle_kernel="xla",
        cp_topology=(),
        cache_scope=None,
        **changes,
    )


@pytest.fixture
def recorded_jit(monkeypatch) -> list[dict[str, Any]]:
    """Record what each owner asks ``jax.jit`` for, compiling nothing."""

    calls: list[dict[str, Any]] = []

    def fake_jit(function: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return lambda *args, **call_kwargs: None

    monkeypatch.setattr(jax, "jit", fake_jit)
    return calls


class _RecordedPool:
    """Stand in for a compiled-executable pool and keep every identity."""

    def __init__(self) -> None:
        self.identities: list[inference._PredictGraphIdentity] = []

    def __call__(self, *args, identity):
        self.identities.append(identity)
        return "prediction"

    def clear_cache(self) -> None:
        pass


def _request(tmp_path: Path, **changes) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("version: 1\n", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    request = PredictionRequest(
        model="openfold3",
        input=input_path,
        weights=weights,
    )
    return dataclasses.replace(request, **changes) if changes else request


def test_the_fused_owner_is_built_exactly_as_it_was_before(recorded_jit) -> None:
    """The default run must be the run every recorded number describes."""
    inference._CompiledPredictPool._new(_identity())

    assert recorded_jit == [{}]


def test_the_fused_owner_carries_the_options_when_asked(recorded_jit) -> None:
    """And the values come from the shared constant, not a literal here."""
    inference._CompiledPredictPool._new(_identity(deterministic=True))

    assert recorded_jit == [{"compiler_options": DETERMINISTIC_COMPILER_OPTIONS}]


def test_the_four_streamed_stages_are_built_exactly_as_before(recorded_jit) -> None:
    """Donation on the cycle carry is the only argument that was ever there."""
    streaming._StreamedGraph(_identity())

    assert recorded_jit == [{}, {}, {"donate_argnums": (4,)}, {}]


def test_every_streamed_stage_carries_the_options_when_asked(recorded_jit) -> None:
    """All four, or the streamed path is a partial promise on long runs."""
    streaming._StreamedGraph(_identity(deterministic=True))

    assert recorded_jit == [
        {"compiler_options": DETERMINISTIC_COMPILER_OPTIONS},
        {"compiler_options": DETERMINISTIC_COMPILER_OPTIONS},
        {
            "donate_argnums": (4,),
            "compiler_options": DETERMINISTIC_COMPILER_OPTIONS,
        },
        {"compiler_options": DETERMINISTIC_COMPILER_OPTIONS},
    ]


def test_the_uncompiled_streamed_stages_stay_plain_python(recorded_jit) -> None:
    """``compiled=False`` owns no executable; nothing may be jitted for it."""
    streaming._StreamedGraph(_identity(), compiled=False)

    assert recorded_jit == []


def test_the_two_policies_are_two_graph_identities() -> None:
    """One cache entry cannot serve both: the option is part of the build."""
    assert _identity(deterministic=True) != _identity(deterministic=False)
    assert _identity() == _identity(deterministic=False)
    assert _identity().deterministic is False


def test_compile_predict_threads_the_option_into_the_identity(monkeypatch) -> None:
    """The keyword is worth nothing if it stops at the factory."""
    pool = _RecordedPool()
    monkeypatch.setattr(inference, "_compiled_predict", pool)
    args = (jax.random.key(0), {"x": jnp.ones((4,), dtype=jnp.float32)}, ())

    inference.compile_predict(_config(), _table())(*args)
    inference.compile_predict(_config(), _table(), deterministic=True)(*args)

    assert [identity.deterministic for identity in pool.identities] == [False, True]
    assert pool.identities[0] != pool.identities[1]


def test_compile_streamed_predict_threads_the_option_into_the_identity(
    monkeypatch,
) -> None:
    """Same identity, so the streamed owner partitions on it as well."""
    pool = _RecordedPool()
    monkeypatch.setattr(streaming, "_compiled_streams", pool)
    config = _config(msa_depth=8)
    args = (jax.random.key(0), {"x": jnp.ones((4,), dtype=jnp.float32)}, ())

    streaming.compile_streamed_predict(config, _table())(*args)
    streaming.compile_streamed_predict(config, _table(), deterministic=True)(*args)

    assert [identity.deterministic for identity in pool.identities] == [False, True]


def test_the_uncompiled_streamed_path_refuses_instead_of_running_without_it(
    monkeypatch,
) -> None:
    """It runs stage by stage and owns no executable to carry the option.

    Running it anyway would report a deterministic run that was not one, which
    is the failure this whole option exists to remove.
    """
    monkeypatch.setattr(streaming, "_compiled_streams", _RecordedPool())

    with pytest.raises(ValueError, match="deterministic reductions"):
        streaming.compile_streamed_predict(
            _config(msa_depth=8),
            _table(),
            deterministic=True,
            compiled=False,
        )


def test_the_backend_refuses_the_eager_path(tmp_path: Path) -> None:
    """``no_compile`` dispatches operation by operation, for the same reason.

    Refused before any preprocessing, so an hours-long featurization does not
    run to produce a result the request already disqualified.
    """
    request = _request(tmp_path, options={"deterministic": "on", "no_compile": True})

    with pytest.raises(ValueError, match="deterministic reductions"):
        OpenFold3Backend().predict(request)


def test_the_backend_takes_the_option_out_of_the_leftover_dict() -> None:
    """A value left behind would be an "unsupported OpenFold3 options" error."""
    assert "deterministic" in openfold3_backend._COMPILE_OPTIONS
    assert openfold3_backend._RELEASED_COMPILE_DEFAULTS["deterministic"] is False


def test_off_shares_the_cache_namespace_with_an_unasked_run(tmp_path: Path) -> None:
    """``off`` is the shipped run, so asking for it must not split the cache.

    That namespace is part of the graph identity too, so a second scope is a
    second in-process JIT owner as well as a second persistent directory.
    """
    backend = OpenFold3Backend()

    unset = backend.cache_profile(_request(tmp_path))
    off = backend.cache_profile(_request(tmp_path, options={"deterministic": "off"}))
    on = backend.cache_profile(_request(tmp_path, options={"deterministic": "on"}))

    assert off == unset
    assert "deterministic" not in off
    assert on["deterministic"] is True
    assert on != unset


def test_the_cli_flag_defaults_to_off() -> None:
    """Off is the shipped run, so it is the parser's answer when unasked."""
    parser = predict_cli._parser()
    action = next(
        action for action in parser._actions if action.dest == "deterministic_ops"
    )

    assert action.default == "off"
    assert tuple(action.choices) == ("off", "on")


def test_the_cli_rejects_a_value_outside_the_vocabulary(tmp_path: Path) -> None:
    """Not a silent off: a misspelled value must not look like a normal run."""
    with pytest.raises(SystemExit) as failure:
        predict_cli._parser().parse_args(
            [
                str(tmp_path / "features.npz"),
                "--checkpoint",
                str(tmp_path / "weights.pt"),
                "-o",
                str(tmp_path / "out"),
                "--deterministic-ops",
                "yes",
            ]
        )

    assert failure.value.code == 2


def test_the_cli_refuses_the_eager_path_before_reading_features(
    tmp_path: Path, capsys
) -> None:
    """``--no-compile`` is the CLI's eager path; the pair cannot be honoured."""
    exit_code = predict_cli.main(
        [
            str(tmp_path / "missing.npz"),
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "-o",
            str(tmp_path / "out"),
            "--no-compile",
            "--deterministic-ops",
            "on",
        ]
    )

    assert exit_code == 1
    assert "deterministic reductions" in capsys.readouterr().out
