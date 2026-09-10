"""AlphaFold 3 asks XLA for repeatable reductions per executable, not per process.

AF3 owns exactly one executable -- upstream's ``ModelRunner._model`` -- and it
is built inside a file FoldJAX carries rather than writes. The option therefore
travels on a subclass of that runner, which is the only place a compile option
can be added without editing an upstream-licensed source file.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import pytest

from foldjax.backends import alphafold3 as af3_backend
from foldjax.backends.alphafold3 import AlphaFold3Backend
from foldjax.execution import DETERMINISTIC_API_OPTION
from foldjax.models._compile_policy import DETERMINISTIC_COMPILER_OPTIONS
from foldjax.schema import PredictionRequest


class _Jitted:
    """Stands in for `jax.jit`'s return value, keeping its wrapping ABI.

    `_tokamax_autotune.install_store` requires `_model` to be a
    `functools.partial` whose function is lowerable, so the recorder has to
    return something `functools.partial` accepts and that carries `lower`.
    """

    def __init__(self, function: Any, **options: Any) -> None:
        self.function = function
        self.options = options

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(compile=lambda: self)


@pytest.fixture
def recorded_jit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def record(function: Any, **kwargs: Any) -> _Jitted:
        calls.append(kwargs)
        return _Jitted(function, **kwargs)

    monkeypatch.setattr(jax, "jit", record)
    return calls


def _fake_runner() -> ModuleType:
    """A module shaped like upstream's runner, with none of its dependencies.

    The vendored runner cannot be imported in a unit test -- it pulls in the
    built AlphaFold 3 native runtime -- so the symbols the subclass reads off
    the loaded module (`ModelRunner`, `hk`, `model`) are supplied here. That a
    fake supplies them at all is the property under test: the subclass must
    read them from the module FoldJAX loaded, not from a global import, or an
    external checkout's runner would be paired with the vendored network.
    """

    module = ModuleType("_foldjax_fake_alphafold3_runner")

    class ModelRunner:
        def __init__(self, *, config: Any, device: Any, model_dir: Any) -> None:
            self._model_config = config
            self._device = device
            self._model_dir = model_dir

        @functools.cached_property
        def model_params(self) -> dict[str, int]:
            return {"weight": 1}

        @functools.cached_property
        def _model(self) -> Any:
            @module.hk.transform
            def forward_fn(batch):
                return module.model.Model(self._model_config)(batch)

            return functools.partial(
                jax.jit(forward_fn.apply, device=self._device), self.model_params
            )

    module.ModelRunner = ModelRunner
    module.hk = SimpleNamespace(
        transform=lambda fn: SimpleNamespace(apply=lambda params, rng, batch: fn(batch))
    )
    module.model = SimpleNamespace(Model=lambda config: lambda batch: (config, batch))
    return module


def test_the_backend_declares_the_shared_entry_and_caches_on_it() -> None:
    """The table entry is the shared object and the option changes the program.

    Unlike `kernel_autotuning`, which selects a Tokamax kernel configuration
    without changing what is compiled, this one is a compile option: two runs
    that differ only here are two executables and must not share one
    compilation namespace.
    """
    assert (
        AlphaFold3Backend.execution_options["deterministic"]
        is DETERMINISTIC_API_OPTION["deterministic"]
    )
    assert "deterministic" in AlphaFold3Backend.compile_options
    assert af3_backend._RELEASED_COMPILE_DEFAULTS["deterministic"] is False


def _request(tmp_path: Path, **options: Any) -> PredictionRequest:
    job = tmp_path / "input.json"
    job.write_text("{}", encoding="utf-8")
    return PredictionRequest(
        model="alphafold3",
        input=job,
        output_dir=tmp_path,
        options=options,
    )


def test_the_released_default_keeps_one_namespace_and_on_selects_another(
    tmp_path: Path,
) -> None:
    """`off` is what every recorded measurement already describes.

    Spelling the released default must not move a run into a second cache
    namespace, and asking for the option must not leave it in the first.
    """
    backend = AlphaFold3Backend()
    omitted = backend.cache_profile(_request(tmp_path))
    off = backend.cache_profile(_request(tmp_path, deterministic="off"))
    on = backend.cache_profile(_request(tmp_path, deterministic="on"))

    assert "deterministic" not in omitted
    assert off == omitted
    assert on["deterministic"] is True
    assert on != off


def test_the_deterministic_runner_carries_the_option_into_the_one_executable(
    recorded_jit: list[dict[str, Any]],
) -> None:
    """The whole point: `jax.jit` is called with the policy, once.

    A subclass rather than an edit to the vendored runner, and a fresh copy of
    the constant rather than the constant itself, so nothing downstream can
    edit the policy every other port reads.
    """
    runner = _fake_runner()
    device = object()
    model_runner = af3_backend._deterministic_model_runner(
        runner, config=SimpleNamespace(), device=device, model_dir=Path("weights")
    )

    assert isinstance(model_runner, runner.ModelRunner)
    model = model_runner._model

    assert len(recorded_jit) == 1
    assert set(recorded_jit[0]) == {"device", "compiler_options"}
    assert recorded_jit[0]["device"] is device
    assert recorded_jit[0]["compiler_options"] == DETERMINISTIC_COMPILER_OPTIONS
    assert recorded_jit[0]["compiler_options"] is not DETERMINISTIC_COMPILER_OPTIONS
    # `install_store` wraps this exact shape; a plain callable would silently
    # drop the persistent Tokamax store on the deterministic route.
    assert isinstance(model, functools.partial)
    assert model.args == ({"weight": 1},)
    assert callable(model.func.lower)


def _vendored_model_jit_keywords() -> list[str]:
    tree = ast.parse(
        af3_backend.VENDORED_RUNNER.read_text(encoding="utf-8"),
        filename=str(af3_backend.VENDORED_RUNNER),
    )
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    model = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_model"
    )
    call = next(
        node
        for node in ast.walk(model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "jit"
    )
    return [keyword.arg for keyword in call.keywords if keyword.arg is not None]


def test_the_subclass_still_matches_the_runner_it_overrides() -> None:
    """The copied body is only correct while upstream's is the same shape.

    `_model` is eight lines of a file FoldJAX carries but does not write. The
    deterministic route reproduces them with one keyword added, so a version
    that builds its executable differently -- another `jax.jit` keyword, a
    different wrapper -- must fail here rather than quietly hand a repeatable
    run a program built some other way.
    """
    assert _vendored_model_jit_keywords() == ["device"]


def test_the_option_is_the_only_difference_from_the_default_route(
    recorded_jit: list[dict[str, Any]],
) -> None:
    """A run that asked for nothing must compile what it always compiled.

    Both runners are built over one module, so what separates them is the
    keyword and nothing else. `tests/test_alphafold3_session.py` covers the
    other half of that promise -- that a request which asked for nothing is
    handed upstream's own class rather than the subclass.
    """
    runner = _fake_runner()
    device = object()
    config = SimpleNamespace()

    runner.ModelRunner(config=config, device=device, model_dir=Path("w"))._model
    af3_backend._deterministic_model_runner(
        runner, config=config, device=device, model_dir=Path("w")
    )._model

    default, deterministic = recorded_jit
    assert set(deterministic) - set(default) == {"compiler_options"}
    assert default == {"device": device}
