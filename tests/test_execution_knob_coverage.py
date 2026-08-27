"""Every neutral knob is answered by every backend.

`foldjax.execution.KNOBS` is a promise: a name that means the same thing on
every model. Nothing enforced it, so `matmul_precision` sat in that table
while all six backends raised "does not support matmul_precision" -- the
vocabulary advertised a knob no adapter had ever implemented.

A test that only checked the backends against each other would have been happy
with that: they agreed. This checks them against the promise.
"""

from __future__ import annotations

import importlib

import pytest

from foldjax.execution import KNOBS, translate

BACKENDS = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")


def _table(name: str) -> dict:
    module = importlib.import_module(f"foldjax.backends.{name}")
    for attribute in vars(module).values():
        options = getattr(attribute, "execution_options", None)
        if isinstance(options, dict) and options:
            return options
    raise AssertionError(f"{name} declares no execution_options")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_every_backend_answers_every_neutral_knob(backend: str, knob: str) -> None:
    """A knob in the vocabulary is a knob every adapter can be asked for.

    `dtype`, `triangle_kernel` and `attention_kernel` are per-port renames and
    a port may legitimately not have the concept -- those are declared model by
    model. `matmul_precision` is not a rename: it is the float32 matmul scope
    that JAX applies to any program, so every port can answer it, and the ones
    that pin a value of their own answer it by reading the request first.
    """
    if knob != "matmul_precision":
        pytest.skip(f"{knob} is a per-port rename, not a universal capability")
    table = _table(backend)
    assert knob in table, f"{backend} does not declare {knob}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", KNOBS["matmul_precision"])
def test_matmul_precision_translates_on_every_backend(backend: str, value: str) -> None:
    """Declared is not enough; the value has to survive translation."""
    assert translate(
        {"matmul_precision": value}, _table(backend), model=backend
    ) == {"matmul_precision": value}


def test_the_shared_entry_is_one_object_not_six_copies() -> None:
    """Six literals would be six chances to diverge; this is the same dict."""
    from foldjax.backends.base import MATMUL_PRECISION_OPTION

    for backend in BACKENDS:
        entry = _table(backend)["matmul_precision"]
        assert entry is MATMUL_PRECISION_OPTION["matmul_precision"]


def test_an_unasked_run_keeps_the_port_s_own_pin() -> None:
    """No request in flight means the recorded upstream-parity value wins.

    This is the property that makes the change safe to land: every default is
    the number that was there before, so a run nobody asked to change is the
    run it already was.
    """
    from foldjax.execution import resolved_matmul_precision

    assert resolved_matmul_precision("highest") == "highest"
    assert resolved_matmul_precision("high") == "high"


def test_a_request_overrides_a_pin_and_then_stops() -> None:
    """Inside the scope the caller wins; outside it the pin is back.

    Scoped rather than latched, for the reason both `_pinned_matmul_precision`
    and Protenix's `predict` already record: a process-global update leaves the
    next unrelated thing in the caller's precision.
    """
    from foldjax.execution import matmul_precision_scope, resolved_matmul_precision

    with matmul_precision_scope("high"):
        assert resolved_matmul_precision("highest") == "high"
    assert resolved_matmul_precision("highest") == "highest"


def test_the_scope_also_sets_jax_for_the_ports_that_pin_nothing() -> None:
    """Three ports open no scope of their own, so this one has to reach them."""
    import jax
    import jax.numpy as jnp

    from foldjax.execution import matmul_precision_scope

    def precision_of(x):
        return jax.jit(lambda a: a @ a).lower(x).as_text()

    x = jnp.ones((4, 4), dtype=jnp.float32)
    with matmul_precision_scope("highest"):
        assert "highest" in precision_of(x).lower()


def test_a_value_outside_the_vocabulary_is_refused() -> None:
    """`translate` guards the neutral path; this guards the scope itself."""
    from foldjax.execution import matmul_precision_scope

    with pytest.raises(ValueError, match="matmul_precision must be one of"):
        with matmul_precision_scope("tf32"):
            pass


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_backend_takes_the_option_out_of_the_dict(backend: str) -> None:
    """It must not survive into the model call.

    Every one of these adapters either forwards leftover options into a native
    signature or raises on them, so a value left behind is a TypeError or an
    "unsupported options" failure rather than a quiet no-op.
    """
    module = importlib.import_module(f"foldjax.backends.{backend}")
    cls = next(
        attribute
        for attribute in vars(module).values()
        if isinstance(getattr(attribute, "execution_options", None), dict)
        and getattr(attribute, "execution_options")
    )
    options = {"matmul_precision": "high", "keep": 1}
    # `self` is unused; calling through the class avoids constructing an
    # adapter, which would touch weight stores this test has no business in.
    scope = cls.matmul_precision(None, options)
    assert options == {"keep": 1}
    with scope():
        from foldjax.execution import resolved_matmul_precision

        assert resolved_matmul_precision("highest") == "high"


def _backend_class(name: str):
    module = importlib.import_module(f"foldjax.backends.{name}")
    return next(
        attribute
        for attribute in vars(module).values()
        if isinstance(getattr(attribute, "execution_options", None), dict)
        and getattr(attribute, "execution_options")
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_options_name_options_that_exist(backend: str) -> None:
    """Every cache-key name is an option some declaration actually produces.

    These names are written down three times -- in `sampling_options`, in
    `compile_options`, and again wherever the adapter reads the value -- and
    nothing makes the copies agree. Renaming the port's option and missing one
    copy does not fail loudly: an orphan in `compile_options` silently stops
    contributing to the compile-cache key, and the same slip in
    `validate_native_options` silently stops validating. That second one
    happened while these names were being unified, and only a test that
    asserted the rejection caught it.

    This checks the copy that has no behavioural test of its own.
    """
    cls = _backend_class(backend)
    declared = set(cls.sampling_options.values())
    declared |= {native for native, _values in cls.execution_options.values()}
    declared |= set(cls.native_options or ())
    orphans = sorted(set(cls.compile_options) - declared)
    assert not orphans, f"{backend} caches on options it does not declare: {orphans}"
