"""Repeatable reduction orders, asked for on this port's own executables.

Boltz-2 reached repeatability so far by freezing autotuning through a
process-wide ``XLA_FLAGS`` environment: a frozen pair repeats to 1e-3 A but
not bitwise (``docs/boltz2-master-kernel-toggle-2026-09-09.md``). An
environment variable is read once when the process starts, so it cannot say
"this prediction and not the next one", and in a benchmark process it reaches
every other model whose recorded numbers were measured without it.

This port already builds one outer executable per role under a dtype policy of
its own -- bfloat16 asks XLA not to erase the publisher's rounding boundaries.
The reduction-order option has to compose with that policy rather than replace
it, which is what these pin:

* with the option off, every role's ``jax.jit`` receives exactly the arguments
  it received before this existed, in both dtypes, so every measurement taken
  so far still describes the default run;
* with it on, the shared constant is merged *onto* the dtype policy, so the
  bfloat16 run keeps its rounding boundaries and gains the reduction orders;
* the affinity executable is built under the same policy as the primary one --
  wiring only the structure graph would make ``on`` a partial promise on
  exactly the jobs that run two graphs;
* the two answers are two retained owners inside one session, because the
  option is part of how the executable is built; and
* the eager steering path refuses rather than silently running without it.

The flag names themselves are never written here: they are read out of
``foldjax.models._compile_policy``, which is the one place they are spelled.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models._compile_policy import DETERMINISTIC_COMPILER_OPTIONS
from foldjax.models.boltz2 import api, compile_policy
from tests.test_boltz2_session import _features, _request

#: What each dtype asked XLA for before a reduction-order option existed.
_DTYPE_POLICY: dict[str, dict[str, bool]] = {
    "float32": {},
    "bfloat16": {"xla_allow_excess_precision": False},
}


def _expected(dtype: str, deterministic: bool) -> dict[str, Any]:
    """The option map one executable is built under, spelled out here."""
    if not deterministic:
        return dict(_DTYPE_POLICY[dtype])
    return {**_DTYPE_POLICY[dtype], **DETERMINISTIC_COMPILER_OPTIONS}


@pytest.mark.parametrize("dtype", sorted(_DTYPE_POLICY))
def test_off_builds_the_executable_exactly_as_it_did_before(dtype, monkeypatch):
    """The default run must be the run every recorded number describes."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(jax, "jit", lambda function, **kw: calls.append(kw) or function)

    compile_policy.jit(lambda x: x, compute_dtype=dtype, deterministic=False)

    options = _expected(dtype, False)
    assert calls == [{"compiler_options": options} if options else {}]


@pytest.mark.parametrize("dtype", sorted(_DTYPE_POLICY))
def test_on_merges_the_constant_onto_the_dtype_policy(dtype, monkeypatch):
    """Both settings survive: the dtype boundary and the reduction orders."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(jax, "jit", lambda function, **kw: calls.append(kw) or function)

    compile_policy.jit(lambda x: x, compute_dtype=dtype, deterministic=True)

    assert calls == [{"compiler_options": _expected(dtype, True)}]
    assert compile_policy.compiler_options(dtype, deterministic=True) == _expected(
        dtype, True
    )


@pytest.mark.parametrize("dtype", sorted(_DTYPE_POLICY))
def test_the_answer_is_a_fresh_map_the_caller_cannot_edit(dtype):
    """A shared dict would let one run's policy reach the next one's."""
    options = compile_policy.compiler_options(dtype, deterministic=True)
    options.clear()

    assert compile_policy.compiler_options(dtype, deterministic=True) == _expected(
        dtype, True
    )
    assert DETERMINISTIC_COMPILER_OPTIONS == _expected("float32", True)


def test_an_unsupported_dtype_still_fails_before_the_option_is_read():
    """The dtype policy is resolved first, so a typo cannot reach a compile."""
    with pytest.raises(ValueError, match="unsupported Boltz compute dtype"):
        compile_policy.compiler_options("float16", deterministic=True)
    with pytest.raises(ValueError, match="unsupported Boltz compute dtype"):
        compile_policy.jit(lambda x: x, compute_dtype=None, deterministic=True)


def _patch_model(monkeypatch, tmp_path: Path, *, affinity: bool) -> Path:
    """Replace featurization, weights and the graph with cheap stand-ins."""
    affinity_path = tmp_path / "boltz2_aff.npz"
    affinity_path.write_bytes(b"test-only-affinity")
    monkeypatch.setattr(
        api, "featurize", lambda **kw: (_features(affinity=affinity), "job", tmp_path)
    )
    monkeypatch.setattr(
        api, "_prepare_affinity_features", lambda **kw: _features(affinity=affinity)
    )

    def load(path):
        params = {"trunk": {"weight": jnp.ones(1)}}
        if path == affinity_path:
            params["affinity"] = {"weight": jnp.ones(1)}
        return params

    def predict(params, feats, key, **kwargs):
        samples = kwargs["multiplicity"]
        result = {
            "sample_atom_coords": jnp.zeros((samples, 3, 3)),
            "plddt": jnp.ones((samples, 2)),
            "iptm": jnp.zeros(samples),
        }
        if "affinity" in params:
            result["affinity_pred_value"] = jnp.ones(1)
        return result

    monkeypatch.setattr("foldjax.models.boltz2.bridge.native.load_params", load)
    monkeypatch.setattr("foldjax.models.boltz2.models.predict.boltz2_predict", predict)
    return affinity_path


def _observe_factories(monkeypatch) -> list[tuple[str, str, bool]]:
    """Record which graph each outer executable owner is built for, and how."""
    built: list[tuple[str, str, bool]] = []
    actual_jit = api._boltz_jit

    def observe(function, *, compute_dtype, deterministic=False):
        built.append((function.__name__, compute_dtype, deterministic))
        return actual_jit(
            function, compute_dtype=compute_dtype, deterministic=deterministic
        )

    monkeypatch.setattr(api, "_boltz_jit", observe)
    return built


@pytest.mark.parametrize("dtype", sorted(_DTYPE_POLICY))
@pytest.mark.parametrize("deterministic", (False, True))
def test_both_roles_report_and_build_under_one_policy(
    tmp_path, monkeypatch, dtype, deterministic
):
    """An affinity job runs two graphs; a partial promise is not a promise."""
    request = _request(tmp_path, affinity=True)
    affinity_path = _patch_model(monkeypatch, tmp_path, affinity=True)
    built = _observe_factories(monkeypatch)

    output = api.predict(
        seq=["AA"],
        weights=request.weights,
        affinity_weights=affinity_path,
        mols=request.options["mols"],
        out_dir=tmp_path,
        write_fmt=None,
        compute_dtype=dtype,
        deterministic=deterministic,
    )

    assert built == [
        ("run_model", dtype, deterministic),
        ("run_affinity", dtype, deterministic),
    ]
    role = {
        "mode": "jit",
        "scope": "outer_jit",
        "compute_dtype": dtype,
        "compiler_options": _expected(dtype, deterministic),
    }
    assert output["execution_policy"] == {"primary": role, "affinity": role}


def test_the_two_policies_are_two_retained_owners(tmp_path, monkeypatch):
    """One session, one cache slot: the option has to invalidate it.

    A session retains one graph owner per role and reuses it across seeds. If
    the reduction-order policy were not part of that identity, a run that asked
    for repeatable reductions would be handed the executable compiled without
    them -- reported as deterministic, and not.
    """
    request = _request(tmp_path)
    _patch_model(monkeypatch, tmp_path, affinity=False)
    built = _observe_factories(monkeypatch)
    backend = Boltz2Backend()

    runners: list[Any] = []
    with backend.session((request,)):
        for seed, deterministic in enumerate((False, False, True)):
            api.predict(
                seq=["AA"],
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                write_fmt=None,
                seed=seed,
                deterministic=deterministic,
                _runtime=backend,
            )
            runners.append(backend._runners["primary"])

    assert [entry[2] for entry in built] == [False, True]
    assert runners[0] is runners[1]
    assert runners[1][0] != runners[2][0]
    assert runners[1][1] is not runners[2][1]


def test_eager_steering_refuses_instead_of_running_without_the_option(
    tmp_path, monkeypatch
):
    """Steering owns no outer executable, so the option would reach nothing.

    Running anyway would report a repeatable run that was not one, which is the
    failure this option exists to remove.
    """
    request = _request(tmp_path)
    _patch_model(monkeypatch, tmp_path, affinity=False)
    monkeypatch.setattr(
        api, "_boltz_jit", lambda *a, **kw: pytest.fail("unexpected jit")
    )

    def run(**overrides):
        return api.predict(
            seq=["AA"],
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            write_fmt=None,
            steering_args={"contact_guidance_update": True},
            **overrides,
        )

    with pytest.raises(ValueError, match="deterministic reductions"):
        run(deterministic=True)

    # The refusal is the option's, not steering's: off still runs eagerly.
    assert run()["execution_policy"]["primary"]["scope"] == (
        "eager_steering_not_covered"
    )


def test_asking_for_off_does_not_fragment_the_compile_cache(tmp_path):
    """Off is the released default, so naming it must reuse its namespace.

    The strip that does this matches on exact type as well as value, which is
    why the released default is a ``bool`` rather than the ``"off"`` string a
    request spells: a lookalike keeps its own persistent cache directory, and
    every warm run recorded so far would have missed it.
    """
    request = _request(tmp_path)
    backend = Boltz2Backend()

    unasked = backend.cache_profile(request)
    off = backend.cache_profile(
        dataclasses.replace(
            request, options={**request.options, "deterministic": "off"}
        )
    )
    on = backend.cache_profile(
        dataclasses.replace(request, options={**request.options, "deterministic": "on"})
    )

    assert "deterministic" not in unasked
    assert off == unasked
    assert on == {**unasked, "deterministic": True}
