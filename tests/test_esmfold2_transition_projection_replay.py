from bench.esmfold2_transition_projection_replay import kernel_launches


def test_launch_extraction_excludes_host_events_and_timing():
    trace = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "linear", "args": {}},
            {
                "cat": "kernel",
                "name": "gemm",
                "dur": 12,
                "args": {"grid": [2, 3, 4], "block": [256, 1, 1], "device": 0},
            },
        ]
    }
    assert kernel_launches(trace) == [
        {
            "name": "gemm",
            "launch": {
                "grid": [2, 3, 4],
                "block": [256, 1, 1],
                "registers per thread": None,
                "shared memory": None,
            },
        }
    ]
