"""Cache-miss explanations cannot break Pallas lowering or cache persistence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax._src import api_util, core, flattree
from jax._src.interpreters import partial_eval

from foldjax import _jax_cache_diagnostics, _pallas_gpu


@pytest.fixture
def original_diff(monkeypatch):
    original = partial_eval.diff_tracing_cache_keys
    while getattr(original, "_foldjax_keyword_diagnostics", False):
        original = original.__wrapped__
    monkeypatch.setattr(partial_eval, "diff_tracing_cache_keys", original)
    return original


def _keys():
    def identity(x):
        return x

    def values(size):
        aval = core.ShapedArray((size,), np.dtype("float32"))
        return (
            flattree.flatten_args(aval),
            api_util.debug_info("pallas triton lower_fun", identity, (aval,), {}),
        )

    new_arguments, new_debug = values(32)
    old_arguments, old_debug = values(16)
    context = ("same tracing context",)
    return (
        (context, (new_arguments, new_debug), ()),
        (
            context,
            (old_arguments,),
            (("debug_info", old_debug), ("requires_low", True)),
        ),
    )


def test_pinned_jax_keyword_explanation_failure_is_reproduced(original_diff):
    new_key, old_key = _keys()
    with pytest.raises(ValueError, match="expected at least 2, got 1"):
        original_diff(new_key, old_key)


def test_keyword_view_preserves_keys_and_explains_type_miss(original_diff):
    new_key, old_key = _keys()
    _jax_cache_diagnostics.install()
    result = partial_eval.diff_tracing_cache_keys(new_key, old_key)
    assert result[:2] == (3, 1)
    assert "now f32[32] and before f32[16]" in result[2]
    assert "normalized for diagnostics" in result[2]
    assert _jax_cache_diagnostics._diagnostic_key(new_key) is new_key
    assert len(old_key[1]) == 1
    assert dict(old_key[2])["requires_low"] is True
    assert dict(old_key[2])["debug_info"].func_name == "identity"


def test_normal_diagnostics_and_errors_are_not_suppressed(monkeypatch, original_diff):
    new_key, old_key = _keys()
    old_view = _jax_cache_diagnostics._diagnostic_key(old_key)
    expected = original_diff(new_key, old_view)
    _jax_cache_diagnostics.install()
    installed = partial_eval.diff_tracing_cache_keys
    _jax_cache_diagnostics.install()
    assert partial_eval.diff_tracing_cache_keys is installed
    assert installed(new_key, old_view) == expected
    invalid = (old_key[0], old_key[1], (("future_keyword", True),))
    assert _jax_cache_diagnostics._diagnostic_key(invalid) is invalid
    with pytest.raises(ValueError):
        installed(new_key, invalid)

    monkeypatch.setattr(jax, "__version__", "future")
    monkeypatch.setattr(partial_eval, "diff_tracing_cache_keys", original_diff)
    _jax_cache_diagnostics.install()
    assert partial_eval.diff_tracing_cache_keys is original_diff


def test_cuda_pallas_lowering_keeps_keyword_miss_logs(
    monkeypatch, caplog, original_diff
):
    from jax._src.pallas.triton import lowering
    from jax.experimental import pallas
    from jax.experimental.pallas import triton

    primitive = core.Primitive("foldjax_diagnostic_identity")
    primitive.multiple_results = True
    primitive.def_impl(lambda x: [x])
    primitive.def_abstract_eval(lambda x: [x])

    def custom_lower(x):
        return [x]

    monkeypatch.setitem(
        lowering.triton_lowering_rules,
        primitive,
        lowering.lower_fun(custom_lower, multiple_results=True),
    )

    def kernel(x, y):
        (value,) = primitive.bind(x[:])
        y[:] = value

    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("devices",),
        abstract_device=jax.sharding.AbstractDevice(
            "NVIDIA GeForce RTX 4090", 128, "gpu"
        ),
    )
    _pallas_gpu.install()
    with jax.explain_cache_misses(True), jax.sharding.use_abstract_mesh(mesh):
        for size in (16, 32):
            shape = jax.ShapeDtypeStruct((size,), np.float32)
            fn = pallas.pallas_call(
                kernel,
                out_shape=shape,
                grid=(),
                compiler_params=triton.CompilerParams(num_warps=4),
            )
            lowered = jax.jit(fn).trace(shape).lower(lowering_platforms=("cuda",))
            assert "xla.gpu.triton" in str(lowered.compiler_ir())
    assert "custom_lower" in caplog.text
    assert "now f32[32] and before f32[16]" in caplog.text
    assert "normalized for diagnostics" in caplog.text


def test_requested_debug_logs_keep_layered_miss_write_and_hit(tmp_path):
    script = textwrap.dedent(
        """
        import json,sys
        from pathlib import Path
        import jax,numpy as np
        from foldjax import _pallas_gpu
        from foldjax.cache import compilation_cache_scope,recorded
        _pallas_gpu.install()
        shared,private = map(Path,sys.argv[1:])
        with compilation_cache_scope(shared, private_directory=private):
            def model(x):
                return x + 1
            result=jax.jit(model)(np.ones((8,), dtype=np.float32))
            expected=np.full((8,),2,np.float32)
            np.testing.assert_array_equal(np.asarray(result), expected)
            print(json.dumps(recorded()))
        """
    )
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_EXPLAIN_CACHE_MISSES": "1",
        "JAX_DEBUG_LOG_MODULES": (
            "jax._src.compilation_cache,jax._src.compiler,jax._src.cache_key"
        ),
        "JAX_LOGGING_LEVEL": "WARNING",
    }
    runs = [
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path / "shared"),
                str(tmp_path / "private"),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        for _ in range(2)
    ]
    first, second = [json.loads(run.stdout) for run in runs]
    assert first["compiles"] == {"success": 1}
    assert first["writes"]["shared"]["success"] == 1
    assert second["compiles"] == {}
    assert second["loads"]["shared"]["success"] == 1
    assert "PERSISTENT COMPILATION CACHE MISS" in runs[0].stderr
    assert "Persistent compilation cache hit" in runs[1].stderr
    assert "hash of serialized computation" in runs[0].stderr
