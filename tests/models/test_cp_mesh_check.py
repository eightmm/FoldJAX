"""A device count is not a working interconnect.

On one tested node every context-parallel run compiled, ran to completion, and
returned finite numbers in which the first shard's rows were right and every
other row was zero -- a prediction nobody would question without a
single-device run beside it. `jax.device_count()` was correct throughout. The
mesh is now asked to prove it moves data before anything is computed on it.

The half-delivery is reproduced here by handing the judgement a corrupted
gather rather than a broken machine: a test that needs the broken machine
guards nothing on any other one.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from foldjax.models import _cp


def _probe(rows: int = 8, cols: int = 4) -> np.ndarray:
    return np.arange(rows, dtype=np.float32)[:, None] * np.ones(
        (rows, cols), dtype=np.float32
    )


def test_a_complete_exchange_is_not_reported_as_a_failure() -> None:
    sent = _probe()
    assert _cp.exchange_failure(sent.copy(), sent, devices=2) is None


def test_a_half_delivered_exchange_is_named_precisely() -> None:
    sent = _probe()
    got = sent.copy()
    got[len(got) // 2 :] = 0.0
    message = _cp.exchange_failure(got, sent, devices=2)
    assert message is not None
    assert "4 of 8 rows" in message
    assert "from row 4" in message


def test_one_wrong_row_is_enough() -> None:
    """Half is what was seen; a single row is what the gate must still catch."""
    sent = _probe()
    got = sent.copy()
    got[3] += 1.0
    assert _cp.exchange_failure(got, sent, devices=4) is not None


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_a_working_mesh_passes_the_check(layout: str) -> None:
    """Forced CPU devices do exchange, so entering the context must succeed."""
    source = textwrap.dedent(
        f"""
        from foldjax.models._cp import context_parallel
        with context_parallel(4, layout="{layout}") as mesh:
            assert mesh is not None
        print("ok")
        """
    )
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", source], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
