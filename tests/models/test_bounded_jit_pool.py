from __future__ import annotations

import gc
import inspect
import threading
import weakref

import jax.numpy as jnp

from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.esmfold2 import inference as esmfold2_inference
from foldjax.models.opendde.models import model as opendde_model
from foldjax.models.protenix.data import esm as protenix_esm
from foldjax.models.protenix.models import model as protenix_model


def test_pool_evicts_the_least_recently_used_executable() -> None:
    traces: list[tuple[int, ...]] = []

    def add_offset(value, *, offset):
        traces.append(value.shape)
        return value + offset

    compiled = BoundedJitPool(
        add_offset,
        static_argnames=("offset",),
        limit=2,
    )

    assert compiled(jnp.zeros((1,)), offset=1).shape == (1,)
    first_owner = weakref.ref(next(iter(compiled._entries.values())))
    assert compiled(jnp.zeros((2,)), offset=1).shape == (2,)
    assert compiled(jnp.zeros((2,)), offset=1).shape == (2,)
    assert traces == [(1,), (2,)]

    assert compiled(jnp.zeros((3,)), offset=1).shape == (3,)
    gc.collect()
    assert first_owner() is None
    assert compiled._entry_count() == 2
    assert compiled._cache_size() == 2

    # Shape one was the least recently used entry and has to trace again.
    assert compiled(jnp.zeros((1,)), offset=1).shape == (1,)
    assert traces == [(1,), (2,), (3,), (1,)]
    assert compiled._entry_count() == 2
    assert compiled._cache_size() == 2

    compiled.clear_cache()
    assert compiled._entry_count() == 0
    assert compiled._cache_size() == 0


def test_pool_lower_keeps_the_original_public_signature() -> None:
    compiled = BoundedJitPool(
        lambda value, *, scale: value * scale,
        static_argnames=("scale",),
        limit=2,
    )
    lowered = compiled.lower(jnp.arange(3.0), scale=2)
    executable = lowered.compile()

    result = executable(jnp.arange(3.0))

    assert result.tolist() == [0.0, 2.0, 4.0]
    assert compiled._entry_count() == 0
    assert compiled._cache_size() == 0


def test_pool_accepts_static_arguments_positionally() -> None:
    traces: list[str] = []

    def add_mode(value, mode):
        traces.append(mode)
        return value + {"one": 1, "two": 2}[mode]

    compiled = BoundedJitPool(
        add_mode,
        static_argnames=("mode",),
        limit=2,
    )

    assert compiled(jnp.zeros((1,)), "one").tolist() == [1.0]
    assert compiled(jnp.zeros((1,)), "two").tolist() == [2.0]
    assert compiled(jnp.zeros((1,)), "one").tolist() == [1.0]
    assert traces == ["one", "two"]
    assert compiled._entry_count() == 2
    assert compiled._cache_size() == 2

    single = BoundedJitPool(
        add_mode,
        static_argnames=("mode",),
        limit=1,
    )
    single(jnp.zeros((1,)), "one")
    single(jnp.zeros((1,)), "two")
    assert single._entry_count() == 1
    assert single._cache_size() == 1


def test_lowering_does_not_evict_a_warm_dispatch_entry() -> None:
    traces: list[tuple[int, ...]] = []

    def identity(value):
        traces.append(value.shape)
        return value

    compiled = BoundedJitPool(identity, limit=1)
    compiled(jnp.zeros((1,)))

    lowered = compiled.lower(jnp.zeros((2,)))
    assert lowered.compile()(jnp.zeros((2,))).shape == (2,)
    assert compiled._entry_count() == 1
    assert compiled._cache_size() == 1

    compiled(jnp.zeros((1,)))
    assert traces == [(1,), (2,)]


def test_failed_first_trace_does_not_poison_the_pool() -> None:
    def reject(value, *, enabled):
        if enabled:
            raise RuntimeError("synthetic trace failure")
        return value

    compiled = BoundedJitPool(
        reject,
        static_argnames=("enabled",),
        limit=2,
    )

    try:
        compiled(jnp.zeros((1,)), enabled=True)
    except RuntimeError as exc:
        assert str(exc) == "synthetic trace failure"
    else:
        raise AssertionError("the synthetic trace must fail")

    assert compiled._entry_count() == 0
    assert compiled._cache_size() == 0


def test_cold_trace_does_not_block_a_different_warm_identity() -> None:
    cold_entered = threading.Event()
    release_cold = threading.Event()
    warm_finished = threading.Event()
    errors: list[BaseException] = []

    def wait_while_tracing(value):
        if value.shape == (2,):
            cold_entered.set()
            if not release_cold.wait(timeout=5):
                raise RuntimeError("timed out waiting to release the cold trace")
        return value + 1

    compiled = BoundedJitPool(wait_while_tracing, limit=2)
    compiled(jnp.zeros((1,)))

    def run_cold() -> None:
        try:
            compiled(jnp.zeros((2,)))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_warm() -> None:
        try:
            compiled(jnp.zeros((1,)))
            warm_finished.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    cold = threading.Thread(target=run_cold)
    warm = threading.Thread(target=run_warm)
    cold.start()
    assert cold_entered.wait(timeout=2)
    warm.start()
    try:
        assert warm_finished.wait(timeout=1)
    finally:
        release_cold.set()
        cold.join(timeout=5)
        warm.join(timeout=5)

    assert not cold.is_alive()
    assert not warm.is_alive()
    assert errors == []


def test_pool_waits_instead_of_evicting_an_active_owner() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    def controlled_trace(value):
        if value.shape == (1,):
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise RuntimeError("timed out waiting to release the first trace")
        else:
            second_entered.set()
        return value

    compiled = BoundedJitPool(controlled_trace, limit=1)

    def run(shape: tuple[int, ...]) -> None:
        try:
            compiled(jnp.zeros(shape))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=run, args=((1,),))
    second = threading.Thread(target=run, args=((2,),))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert second_entered.is_set()
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert compiled._entry_count() == 1
    assert compiled._cache_size() == 1


def test_clear_cache_waits_for_an_active_owner() -> None:
    trace_entered = threading.Event()
    release_trace = threading.Event()
    clear_started = threading.Event()
    clear_finished = threading.Event()
    errors: list[BaseException] = []

    def controlled_trace(value):
        trace_entered.set()
        if not release_trace.wait(timeout=5):
            raise RuntimeError("timed out waiting to release the trace")
        return value

    compiled = BoundedJitPool(controlled_trace, limit=1)

    def run_trace() -> None:
        try:
            compiled(jnp.zeros((1,)))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def clear() -> None:
        clear_started.set()
        compiled.clear_cache()
        clear_finished.set()

    trace_thread = threading.Thread(target=run_trace)
    clear_thread = threading.Thread(target=clear)
    trace_thread.start()
    assert trace_entered.wait(timeout=2)
    clear_thread.start()
    assert clear_started.wait(timeout=2)
    try:
        assert not clear_finished.wait(timeout=0.1)
    finally:
        release_trace.set()
        trace_thread.join(timeout=5)
        clear_thread.join(timeout=5)

    assert clear_finished.is_set()
    assert not trace_thread.is_alive()
    assert not clear_thread.is_alive()
    assert errors == []
    assert compiled._entry_count() == 0
    assert compiled._cache_size() == 0


def test_pinned_jax_exposes_owner_local_cache_hooks() -> None:
    compiled = BoundedJitPool(lambda value: value)
    owner = compiled._new()

    assert callable(getattr(owner, "_clear_cache", None))
    assert callable(getattr(owner, "_cache_size", None))


def test_discarded_lowerings_release_their_private_owners() -> None:
    compiled = BoundedJitPool(lambda value: value + 1, limit=1)
    original_new = compiled._new
    owner_refs = []

    def tracked_new():
        owner = original_new()
        owner_refs.append(weakref.ref(owner))
        return owner

    compiled._new = tracked_new
    for size in range(1, 6):
        lowered = compiled.lower(jnp.zeros((size,)))
        del lowered

    gc.collect()
    assert all(owner() is None for owner in owner_refs)
    assert compiled._entry_count() == 0
    assert compiled._cache_size() == 0


def test_whole_model_jits_use_bounded_owners() -> None:
    assert isinstance(protenix_model._compiled_protenix_infer, BoundedJitPool)
    assert isinstance(opendde_model._compiled_opendde_infer, BoundedJitPool)
    assert isinstance(esmfold2_inference._compiled_predict_pool, BoundedJitPool)
    assert isinstance(protenix_esm._compiled_esm2_layer, BoundedJitPool)
    assert protenix_model._compiled_protenix_infer._limit == 8
    assert opendde_model._compiled_opendde_infer._limit == 8
    assert esmfold2_inference._compiled_predict_pool._limit == 8
    assert protenix_esm._compiled_esm2_layer._limit == 8


def test_esmfold2_static_choices_create_distinct_bounded_owners(
    monkeypatch,
) -> None:
    parameters = tuple(inspect.signature(esmfold2_inference._run).parameters)
    assert esmfold2_inference._COMPILED_PREDICT_STATIC_ARGNAMES == parameters[4:]

    pool = esmfold2_inference._compiled_predict_pool
    created = []

    class FakeOwner:
        def __init__(self) -> None:
            self.size = 0

        def __call__(self, *_args, **_kwargs):
            self.size = 1
            return {}

        def _cache_size(self) -> int:
            return self.size

        def _clear_cache(self) -> None:
            self.size = 0

    def new_owner():
        owner = FakeOwner()
        created.append(owner)
        return owner

    settings = esmfold2_inference.structure_model.ModelSettings()
    base = (
        jnp.zeros((2,), dtype=jnp.uint32),
        {},
        {},
        None,
        settings,
        1,
        False,
        1,
        (),
        False,
        False,
        False,
        True,
        False,
        256,
    )
    compact_bonds = (*base[:11], True, *base[12:])
    no_distogram = (*base[:12], False, *base[13:])
    dense_atom_attention = (*base[:-1], 0)

    pool.clear_cache()
    monkeypatch.setattr(pool, "_new", new_owner)
    try:
        pool(*base)
        pool(*compact_bonds)
        pool(*no_distogram)
        pool(*dense_atom_attention)
        assert pool._entry_count() == 4
        assert len(created) == 4

        pool(*base)
        assert pool._entry_count() == 4
        assert len(created) == 4
    finally:
        pool.clear_cache()
