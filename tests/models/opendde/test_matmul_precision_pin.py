"""OpenDDE has to name the precision upstream runs, not inherit one.

`foldjax.execution.matmul_precision_scope` says in its own docstring that three
backends pin nothing and therefore take whatever JAX's process default is.
OpenDDE was one of them, and its upstream ships `"dtype": "fp32"` with
`enable_tf32: True` -- TF32 matmuls, which JAX spells `"high"`.

On the card this was measured on the default already is TF32, so the pin moves
no number today. What it removes is the dependence: a card whose default is not
TF32, or a caller who leaves a different setting in the process, silently
changed this port's arithmetic and nothing said so.
"""

from __future__ import annotations

import jax

from foldjax.execution import matmul_precision_scope
from foldjax.models.opendde.cli.predict import (
    _MATMUL_PRECISION,
    opendde_precision,
)


def _active() -> object:
    """The precision a `precision=None` dot would be traced under."""
    return jax.config.jax_default_matmul_precision


def test_the_pin_names_upstreams_tf32() -> None:
    """`high` is JAX's spelling of upstream's `enable_tf32: True`."""
    assert _MATMUL_PRECISION == "high"


def test_the_port_stops_inheriting_the_process_default(monkeypatch) -> None:
    """Whatever another model left behind must not reach this port.

    Forced onto the GPU branch, because that is the only place TF32 exists and
    the suite this runs in is a CPU one.
    """
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    observed = []

    @opendde_precision
    def entry() -> None:
        observed.append(_active())

    with jax.default_matmul_precision("highest"):
        outside = _active()
        entry()

    assert observed == ["high"], (observed, outside)
    assert outside != "high", "the ambient setting was not actually different"


def test_the_pin_is_scoped_and_restores(monkeypatch) -> None:
    """Scoped to the entry point, so importing the port re-specifies nothing."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    before = _active()

    @opendde_precision
    def entry() -> None:
        assert _active() == "high"

    entry()
    assert _active() == before


def test_the_pin_stands_aside_where_tf32_does_not_exist() -> None:
    """Asking a CPU dot for TF32 raises rather than degrading.

    `ValueError: The precision 'TF32_TF32_F32' is not supported by dot_general
    on CPU` is what a naive pin costs, and it is why this port had none. There
    is also nothing to match off the GPU: upstream's `enable_tf32` is equally
    inert there.
    """
    assert jax.default_backend() != "gpu", "this test is about the CPU branch"
    before = _active()

    @opendde_precision
    def entry() -> None:
        assert _active() == before

    entry()


def test_an_explicit_request_still_wins() -> None:
    """The pin is this port's default, not an override of the caller."""
    observed = []

    @opendde_precision
    def entry() -> None:
        observed.append(_active())

    with matmul_precision_scope("highest"):
        entry()

    assert observed == ["highest"]
