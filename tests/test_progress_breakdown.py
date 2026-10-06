from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from foldjax import progress


def _fresh_program(scale: float):
    # A new function each call, so every call is a trace and a compile.
    return jax.jit(lambda x: jnp.sin(x) * scale)


def test_breakdown_records_compiler_events_and_parts_inside_predict() -> None:
    timeline = progress.Timeline()
    with timeline.stage("predict"), timeline.recording():
        with progress.part("featurize"):
            time.sleep(0.01)
        _fresh_program(2.0)(jnp.ones((4,))).block_until_ready()
    breakdown = timeline.breakdown()
    assert breakdown is not None
    seconds, counts = breakdown["seconds"], breakdown["counts"]
    assert seconds["featurize"] >= 0.01
    for label in ("trace", "lower", "compile", "execute and host"):
        assert label in seconds
    assert counts["programs"] >= 1
    measured = sum(value for key, value in seconds.items() if key != "execute and host")
    assert measured <= timeline.summary()["predict"] + 0.02


def test_a_compile_inside_a_part_is_counted_but_not_timed_twice() -> None:
    timeline = progress.Timeline()
    with timeline.recording():
        with progress.part("weight load"):
            _fresh_program(3.0)(jnp.ones((4,))).block_until_ready()
    breakdown = timeline.breakdown()
    assert breakdown is not None
    assert set(breakdown["seconds"]) == {"weight load"}
    assert breakdown["counts"]["programs"] >= 1


def test_nothing_is_attributed_outside_recording() -> None:
    timeline = progress.Timeline()
    with progress.part("featurize"):
        _fresh_program(4.0)(jnp.ones((4,))).block_until_ready()
    assert timeline.breakdown() is None
