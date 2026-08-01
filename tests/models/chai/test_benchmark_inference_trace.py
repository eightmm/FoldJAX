from __future__ import annotations

import foldjax.models.chai.inference as inference

from .scripts.benchmark_inference import (
    INSTRUMENTED_FUNCTIONS,
    _memory_snapshot,
)


def test_phase_trace_targets_current_staged_inference_boundaries() -> None:
    assert "_compiled_confidence" not in INSTRUMENTED_FUNCTIONS
    assert {
        "_embed_and_initialize",
        "_run_staged_trunk",
        "_compiled_diffusion_step",
        "_compiled_confidence_initialize",
        "_run_staged_confidence_block",
        "_compiled_confidence_project",
    } == set(INSTRUMENTED_FUNCTIONS)
    assert all(hasattr(inference, name) for name in INSTRUMENTED_FUNCTIONS)


def test_memory_snapshot_keeps_allocator_byte_and_limit_metrics() -> None:
    memory = {
        "bytes_in_use": 11,
        "peak_bytes_in_use": 23,
        "bytes_limit": 101,
        "largest_free_block_bytes": 47,
        "num_allocs": 7,
    }

    assert _memory_snapshot(memory) == {
        "bytes_in_use": 11,
        "peak_bytes_in_use": 23,
        "bytes_limit": 101,
        "largest_free_block_bytes": 47,
    }
