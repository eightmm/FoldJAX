from __future__ import annotations

from bench.kernel_shootout.attrib import allocation_rows, analyse_dump


def test_arena_is_separate_from_values_and_live_peak_metadata(tmp_path):
    stem = "module_0001.jit_apply_fn.gpu_after_optimizations"
    assignment = tmp_path / f"{stem}-buffer-assignment.txt"
    assignment.write_text(
        "allocation 0: size 16000000000, preallocated-temp:\n"
        " value: <8 q @0> (size=8000000000,offset=0): bf16[2000,2000,1000]{2,1,0}\n"
        " value: <9 k @0> (size=8000000000,offset=8000000000): "
        "bf16[2000,2000,1000]{2,1,0}\n"
        " value: <10 later @0> (size=8000000000,offset=0): "
        "bf16[2000,2000,1000]{2,1,0}\n"
        "allocation 1: size 4, constant:\n"
        " value: <11 small @0> (size=4,offset=0): f32[]\n"
    )
    (tmp_path / f"{stem}.txt").write_text(
        "%q = bf16[2000,2000,1000]{2,1,0} fusion(%x), kind=kLoop, "
        'metadata={op_name="triangle/query"}\n'
        "%k = bf16[2000,2000,1000]{2,1,0} custom-call(%x), "
        'custom_call_target="triangle_kernel"\n'
    )
    (tmp_path / f"{stem}-live-range.txt").write_text(
        "InstructionSequence:\n"
        "  30:q\n"
        "Live ranges at 30 (peak):\n"
        "  q{}: 8000000000 bytes\n"
        "  k{}: 8000000000 bytes\n"
    )
    assert allocation_rows(assignment, 100)[0]["value_count"] == 3
    result = analyse_dump(tmp_path, 100)
    assert result["temp"] == 16000000000
    assert result["at"] == "q"
    assert sum(row[0] for row in result["rows"]) == 16000000000
    (arena,) = result["allocations"]
    assert arena["description"] == "preallocated-temp:"
    assert len(arena["large_values"]) == 3
    assert "triangle/query" in arena["large_values"][0]["producer"]
    assert "triangle_kernel" in arena["large_values"][1]["producer"]
    assert arena["large_values"][2]["offset_bytes"] == 0


def test_cache_hit_without_dump_is_unknown_not_zero_memory(tmp_path):
    assert analyse_dump(tmp_path, 0) is None


def test_missing_live_ranges_do_not_imply_zero_peak(tmp_path):
    assignment = tmp_path / "module.gpu_after_optimizations-buffer-assignment.txt"
    assignment.write_text("allocation 0: size 1000, preallocated-temp:\n")
    result = analyse_dump(tmp_path, 0)
    assert result["temp"] == 1000
    assert result["live_peak_bytes"] is None
