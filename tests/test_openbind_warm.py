import pytest

from bench.openbind_warm import measure_calls


def test_warm_timing_includes_synchronization_but_excludes_observation():
    now = [0.0]
    order = []

    def call():
        order.append("call")
        now[0] += 2
        return object()

    def sync(result):
        order.append("sync")
        now[0] += 3

    def observe(result):
        order.append("observe")
        now[0] += 100
        return {"allocator": {"peak_bytes_in_use": 123}}

    rows = measure_calls(call, sync, observe, warm_repeats=3, clock=lambda: now[0])
    assert [r["phase"] for r in rows] == ["first", "warm", "warm", "warm"]
    assert [r["seconds"] for r in rows] == [5] * 4
    assert order == ["call", "sync", "observe"] * 4


def test_prepare_is_outside_timing_and_runs_for_every_call():
    now = [0.0]
    prepared = []

    def prepare():
        now[0] += 100
        prepared.append(True)

    rows = measure_calls(
        lambda: None,
        lambda _: None,
        lambda _: {},
        warm_repeats=2,
        prepare=prepare,
        clock=lambda: now[0],
    )
    assert len(prepared) == 3
    assert [row["seconds"] for row in rows] == [0, 0, 0]


def test_native_copies_containers_without_retaining_mutations_or_cloning_storage():
    from bench.openbind_native_warm import copy_containers

    leaf = object()
    original = {"seed": [101], "nested": ({"tensor": leaf},)}
    actual = copy_containers(original)
    assert actual["nested"][0]["tensor"] is leaf
    actual["seed"].pop()
    actual["nested"][0].pop("tensor")
    assert original == {"seed": [101], "nested": ({"tensor": leaf},)}


def test_native_shared_input_guard_rejects_mutation_and_dtype_changes():
    import numpy as np

    from bench.openbind_native_warm import assert_shared_features_unchanged

    expected = {"ref_pos": np.ones((2, 3), dtype=np.float32)}
    assert_shared_features_unchanged(
        {k: v.copy() for k, v in expected.items()}, expected
    )
    with pytest.raises(AssertionError):
        assert_shared_features_unchanged({"ref_pos": expected["ref_pos"] + 1}, expected)
    with pytest.raises(ValueError, match="schema"):
        assert_shared_features_unchanged(
            {"ref_pos": expected["ref_pos"].astype(np.float64)}, expected
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_repeats_fail_before_execution(count):
    def forbidden():
        pytest.fail("must not execute")

    with pytest.raises(ValueError, match="positive integer"):
        measure_calls(forbidden, forbidden, forbidden, warm_repeats=count)


def test_warm_result_is_released_before_next_call():
    import weakref

    references = []

    class Result:
        pass

    def call():
        assert all(ref() is None for ref in references)
        value = Result()
        references.append(weakref.ref(value))
        return value

    measure_calls(call, lambda _: None, lambda _: {}, warm_repeats=2)
    assert all(ref() is None for ref in references)
