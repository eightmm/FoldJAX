"""The shared reduction-order policy, and proof that it reaches XLA.

`foldjax.models._compile_policy` is the one place a deterministic-reduction
flag is spelled. Six ports read it, so the tests that matter here are the ones
a port author would otherwise have to re-derive:

* what `compiler_options` answers in each of the four `(deterministic, base)`
  corners -- in particular that "off with nothing asked for" is `None` and not
  an empty dict, because an empty option map is a different compile from no
  option map at all, and every recorded measurement describes the latter;
* that the two pools `policy_pools` builds are two owners, one of which asks
  for nothing; and
* that the option actually arrives at the compiler rather than merely being
  passed to `jax.jit`.

The last one is worth three separate mechanisms because a recording test
proves only that an argument was handed over. They are stated as what each one
proves:

(a) the values XLA is given -- read off the `CompileOptions` the backend is
    compiled with;
(b) that the CPU compiler *acts* on an entry in that map -- a public,
    observable effect (`xla_dump_to` writes files) with no private API in it,
    so at least one of the three can never skip; and
(c) that two policies can never share a persistent compilation-cache entry --
    the cache key differs.

(a) and (c) reach into `jax._src`. Both check the signature they depend on
first and skip on a mismatch, so a JAX upgrade reports "this proof no longer
applies" instead of failing as though the option had stopped working.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._compile_policy import (
    DETERMINISTIC_COMPILER_OPTIONS,
    compiler_options,
    policy_pools,
    select,
)
from foldjax.models._jit_pool import BoundedJitPool


def test_the_module_a_backend_reads_at_plan_time_does_not_import_jax() -> None:
    """A backend reads the constant while planning, before any client exists.

    Importing JAX from here would pull an accelerator client into processes
    that only wanted to know what a request means.
    """
    import foldjax

    source_root = str(Path(foldjax.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root, *filter(None, [env.get("PYTHONPATH")])]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import foldjax.models._compile_policy as m; "
            "print(m.DETERMINISTIC_COMPILER_OPTIONS and 'jax' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert completed.stdout.strip() == "False"


def test_off_with_nothing_else_asked_for_is_no_option_map_at_all() -> None:
    """`{}` would be a different compile from the one every number describes."""
    assert compiler_options(deterministic=False) is None
    assert compiler_options(deterministic=False, base={}) is None


def test_off_keeps_the_options_the_port_already_asked_for() -> None:
    """Boltz-2's bfloat16 run passes one of its own; off must not drop it."""
    base = {"xla_allow_excess_precision": False}

    result = compiler_options(deterministic=False, base=base)

    assert result == base
    assert result is not base


def test_on_merges_the_constant_onto_the_port_s_own_options() -> None:
    """Composition, not replacement: the port's key survives alongside."""
    base = {"xla_allow_excess_precision": False}

    result = compiler_options(deterministic=True, base=base)

    assert result == {**base, **DETERMINISTIC_COMPILER_OPTIONS}
    assert result is not base


def test_on_without_a_base_is_the_constant_but_not_the_constant_object() -> None:
    """A caller that edited the return value would edit every port's policy."""
    result = compiler_options(deterministic=True)

    assert result == DETERMINISTIC_COMPILER_OPTIONS
    assert result is not DETERMINISTIC_COMPILER_OPTIONS


def test_policy_pools_builds_two_owners_and_only_one_asks_for_anything() -> None:
    """The default owner has to stay the owner it was before this existed."""
    default, deterministic = policy_pools(lambda value: value, limit=4)

    assert default is not deterministic
    assert default._compiler_options is None
    assert deterministic._compiler_options == DETERMINISTIC_COMPILER_OPTIONS
    assert deterministic._compiler_options is not DETERMINISTIC_COMPILER_OPTIONS
    assert default._limit == deterministic._limit == 4


def test_policy_pools_merges_a_base_instead_of_replacing_it() -> None:
    """A port with its own compile option keeps it in both owners."""
    base = {"xla_allow_excess_precision": False}

    default, deterministic = policy_pools(lambda value: value, compiler_options=base)

    assert default._compiler_options == base
    assert deterministic._compiler_options == {
        **base,
        **DETERMINISTIC_COMPILER_OPTIONS,
    }


def test_select_hands_back_the_owner_for_the_policy() -> None:
    """One entry cannot serve both: the option is part of the build."""
    pools = policy_pools(lambda value: value)

    assert select(pools, False) is pools[0]
    assert select(pools, True) is pools[1]


def test_a_pool_that_asks_for_the_options_still_runs() -> None:
    """The recording tests would pass against an option XLA rejects."""
    pool = BoundedJitPool(
        lambda value: value + 1,
        compiler_options=DETERMINISTIC_COMPILER_OPTIONS,
    )

    assert int(pool(jnp.asarray(1, dtype=jnp.int32))) == 2


def test_the_pool_keeps_the_constant_out_of_reach() -> None:
    """A pool that stored the caller's dict could be edited through it."""
    pool = BoundedJitPool(
        lambda value: value,
        compiler_options=DETERMINISTIC_COMPILER_OPTIONS,
    )

    assert pool._compiler_options is not DETERMINISTIC_COMPILER_OPTIONS


def _compile_hook():
    """The private entry the compiled options pass through, or `None`.

    Guarded by signature rather than by version: this is what the proof reads,
    so if it no longer looks like this the proof no longer applies.
    """
    from jax._src import compiler

    entry = getattr(compiler, "backend_compile_and_load", None)
    if entry is None:
        return None
    expected = (
        "backend",
        "module",
        "executable_devices",
        "options",
        "host_callbacks",
    )
    if tuple(inspect.signature(entry).parameters) != expected:
        return None
    return compiler, entry


def test_the_options_reach_the_compiler_and_a_default_pool_carries_none(
    monkeypatch,
) -> None:
    """(a) What XLA is handed, not what `jax.jit` was handed.

    Recorded twice in one process, default first, because the interesting
    failure is a deterministic owner that quietly compiles the default
    program.
    """
    hooked = _compile_hook()
    if hooked is None:
        pytest.skip("jax._src.compiler.backend_compile_and_load has changed shape")
    compiler, real = hooked
    # Built and realized before the hook is installed. `jnp.asarray` compiles a
    # scalar convert of its own on first use in a process, which would be
    # recorded here as a third, optionless compile and make this test pass or
    # fail depending on what ran before it.
    argument = jax.block_until_ready(jnp.asarray(1, dtype=jnp.int32))
    seen: list[list[tuple[str, object]]] = []

    def record(backend, module, executable_devices, options, host_callbacks):
        seen.append(list(options.env_option_overrides))
        return real(backend, module, executable_devices, options, host_callbacks)

    monkeypatch.setattr(compiler, "backend_compile_and_load", record)
    # Distinct functions: an identical program would be answered from JAX's
    # compilation cache and record nothing at all.
    BoundedJitPool(lambda value: value + 101)(argument)
    BoundedJitPool(
        lambda value: value + 102,
        compiler_options=DETERMINISTIC_COMPILER_OPTIONS,
    )(argument)

    assert seen == [[], list(DETERMINISTIC_COMPILER_OPTIONS.items())]


def test_the_cpu_compiler_acts_on_a_pool_option_rather_than_recording_it(
    tmp_path: Path,
) -> None:
    """(b) A public, observable effect, with no private API to go stale.

    `xla_dump_to` is an ordinary entry in the same option map the
    deterministic keys travel in, and it is the only one whose effect can be
    seen from outside the compiler on CPU. Files appearing proves the map is
    honoured; it does not prove anything about reduction order, which is a GPU
    property.
    """
    destination = tmp_path / "hlo"

    pool = BoundedJitPool(
        lambda value: value * 3 + 1,
        compiler_options={"xla_dump_to": str(destination)},
    )
    assert int(pool(jnp.asarray(2, dtype=jnp.int32))) == 7

    assert sorted(path.name for path in destination.iterdir())


def test_the_two_policies_cannot_share_a_persistent_cache_entry() -> None:
    """(c) Otherwise one cache directory would hand back the wrong program.

    The compile options are part of JAX's persistent-cache key, so an
    executable built without the option can never be served to a run that
    asked for it -- including across processes sharing a cache directory.
    """
    from jax._src import compilation_cache, compiler

    if (
        "env_options_overrides"
        not in inspect.signature(compiler.get_compile_options).parameters
    ):
        pytest.skip("jax._src.compiler.get_compile_options has changed shape")
    import jax.extend as jex

    module = (
        jax.jit(lambda value: value + 5)
        .lower(jnp.asarray(1, dtype=jnp.int32))
        .compiler_ir()
    )
    backend = jex.backend.get_backend()
    devices = np.array(jax.local_devices()[:1])

    def key(options: dict[str, object] | None) -> str:
        compile_options = compiler.get_compile_options(
            num_replicas=1,
            num_partitions=1,
            device_assignment=devices.reshape((1, 1)),
            env_options_overrides=options,
            backend=backend,
        )
        return compilation_cache.get_cache_key(
            module, devices, compile_options, backend
        )

    assert key(None) != key(dict(DETERMINISTIC_COMPILER_OPTIONS))


def test_the_flag_name_is_spelled_in_exactly_one_place() -> None:
    """Because it is provisional.

    The 3,012-token measurement may replace it with the narrower
    ``xla_gpu_exclude_nondeterministic_ops``. A second copy in a CLI help
    string or a backend table would be the one that gets missed -- and with
    six ports reading this constant, a second copy is six ports' worth of
    silent divergence rather than one.

    The names searched for are read out of the constant rather than written
    here, so this survives that switch instead of failing the day it lands --
    which would be this guard reporting the opposite of what it guards.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "foldjax"
    sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    }

    assert DETERMINISTIC_COMPILER_OPTIONS
    for flag in DETERMINISTIC_COMPILER_OPTIONS:
        spelled = sorted(name for name, text in sources.items() if flag in text)
        assert spelled == ["models/_compile_policy.py"], flag
