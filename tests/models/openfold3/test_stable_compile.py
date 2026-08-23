"""The OpenFold3 factory must reuse one correctly-partitioned JIT cache."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax._openfold3_compile as compile_helpers
from foldjax._openfold3_compile import (
    TRIANGLE_BACKEND_ENV,
    cache_scope_changed,
    observe_cache_scope,
    resolve_triangle_kernel,
    triangle_backend,
)
from foldjax.cache import compilation_cache_scope
from foldjax.models import _capture
from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.representative_atoms import (
    RepresentativeAtomTable,
    token_representative_atoms,
)
from tests.models.cp_probe_env import inherited_environment
from tests.models.openfold3.feature_fixture import minimal_features


def _config(**changes) -> inference.InferenceConfig:
    config = inference.InferenceConfig(
        n_token=4,
        n_atom=4,
        n_query=1,
        n_key=1,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=1,
        max_relative_chain=1,
        num_cycles=1,
        num_samples=1,
        max_atoms_per_token=1,
        plddt_bins=1,
        pae_bins=1,
        pae_bin_max=1.0,
        no_rollout_steps=1,
    )
    return config._replace(**changes)


def _table(offset: float = 0.0) -> RepresentativeAtomTable:
    zeros = np.zeros(32, dtype=np.float32)
    ones = np.ones(32, dtype=np.float32)
    offsets = np.full(32, offset, dtype=np.float32)
    return RepresentativeAtomTable(
        cb_offset=offsets,
        cb_mask=ones,
        ca_offset=zeros,
        ca_mask=ones,
        c4_offset=zeros,
        c4_mask=ones,
        c2_offset=zeros,
        c2_mask=ones,
        glycine=zeros,
        purine_dna=zeros,
        purine_rna=zeros,
        pyrimidine_dna=zeros,
        pyrimidine_rna=zeros,
    )


def test_repeated_factories_share_one_trace_and_keep_chemistry_dynamic(
    monkeypatch,
) -> None:
    traces: list[tuple[object, ...]] = []

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        *,
        n_chain=None,
        noise_fn=None,
        noise_tape=None,
        noise_mask=None,
        augment=True,
        use_trunk_pair_embedding=True,
    ):
        del key, noise_fn, noise_tape, noise_mask
        traces.append(
            (
                config.cp_layout,
                n_chain,
                os.environ[TRIANGLE_BACKEND_ENV],
                augment,
                use_trunk_pair_embedding,
            )
        )
        return batch["x"] + params + representative_atoms.cb_offset[0]

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    try:
        args = (
            jax.random.key(0),
            {"x": jnp.ones((4,), dtype=jnp.float32)},
            jnp.asarray(2.0, dtype=jnp.float32),
        )
        first = inference.compile_predict(
            _config(), _table(0), n_chain=2, cache_scope="scope-a"
        )(*args)
        second = inference.compile_predict(
            _config(cp_layout="1d"),
            _table(1),
            n_chain=2,
            cache_scope="scope-a",
        )(*args)
        jax.block_until_ready((first, second))

        assert len(traces) == 1
        assert inference._compiled_predict._cache_size() == 1
        np.testing.assert_array_equal(first, np.full(4, 3.0, dtype=np.float32))
        np.testing.assert_array_equal(second, np.full(4, 4.0, dtype=np.float32))
        assert traces[0] == ("1d", 2, "cueq", True, True)
    finally:
        inference._compiled_predict.clear_cache()


def test_every_python_graph_choice_partitions_the_stable_jit(monkeypatch) -> None:
    traces: list[tuple[object, ...]] = []

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        *,
        n_chain=None,
        noise_fn=None,
        noise_tape=None,
        noise_mask=None,
        augment=True,
        use_trunk_pair_embedding=True,
    ):
        del key, params, representative_atoms, noise_fn
        traces.append(
            (
                config,
                n_chain,
                noise_tape is not None,
                noise_mask is not None,
                os.environ[TRIANGLE_BACKEND_ENV],
                augment,
                use_trunk_pair_embedding,
            )
        )
        return batch["x"]

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    args = (
        jax.random.key(0),
        {"x": jnp.ones((4,), dtype=jnp.float32)},
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def run(config=None, **kwargs) -> None:
        value = inference.compile_predict(config or _config(), _table(), **kwargs)(
            *args
        )
        jax.block_until_ready(value)

    try:
        run(cache_scope="a")
        run(cache_scope="a")
        assert len(traces) == 1

        run(n_chain=2, cache_scope="a")
        run(cache_scope="b")
        run(triangle_kernel="xla", cache_scope="a")
        run(config=_config(returned_representations=("pair",)), cache_scope="a")
        run(
            config=_config(returned_representations=("pair",), stop_after_trunk=True),
            cache_scope="a",
        )
        run(augment=False, cache_scope="a")
        run(use_trunk_pair_embedding=False, cache_scope="a")

        masked = inference.compile_predict(_config(), _table(), cache_scope="a")
        jax.block_until_ready(masked(*args, noise_mask=jnp.ones((1, 4), dtype=bool)))
        taped = inference.compile_predict(_config(), _table(), cache_scope="a")
        jax.block_until_ready(
            taped(*args, noise_tape=jnp.ones((2, 1, 4, 3), dtype=jnp.float32))
        )

        assert len(traces) == 10
        assert (
            inference._compiled_predict._cache_size()
            == inference._MAX_RETAINED_EXECUTABLES
        )

        # The first identity is least-recently used and has been evicted. A
        # later request recompiles it instead of retaining an unbounded list of
        # full-model executables for every shape/profile seen by the process.
        run(cache_scope="a")
        assert len(traces) == 11
        assert (
            inference._compiled_predict._cache_size()
            == inference._MAX_RETAINED_EXECUTABLES
        )
        assert traces[-3][2:4] == (False, True)
        assert traces[-2][2:4] == (True, False)
    finally:
        inference._compiled_predict.clear_cache()


def test_ambient_triangle_kernel_is_resolved_into_the_graph_identity(
    monkeypatch,
) -> None:
    traces: list[str] = []

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        *,
        n_chain=None,
        noise_fn=None,
        noise_tape=None,
        noise_mask=None,
        augment=True,
        use_trunk_pair_embedding=True,
    ):
        del (
            key,
            params,
            config,
            representative_atoms,
            n_chain,
            noise_fn,
            noise_tape,
            noise_mask,
            augment,
            use_trunk_pair_embedding,
        )
        kernel = os.environ[TRIANGLE_BACKEND_ENV]
        traces.append(kernel)
        return batch["x"] + (1 if kernel == "cueq" else 2)

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    args = (
        jax.random.key(0),
        {"x": jnp.zeros((1,), dtype=jnp.float32)},
        jnp.asarray(0.0),
    )
    run = inference.compile_predict(_config(), _table())
    try:
        monkeypatch.setenv(TRIANGLE_BACKEND_ENV, "cueq")
        cueq = run(*args)
        monkeypatch.setenv(TRIANGLE_BACKEND_ENV, "xla")
        xla = run(*args)
        jax.block_until_ready((cueq, xla))

        assert traces == ["cueq", "xla"]
        np.testing.assert_array_equal(cueq, [1])
        np.testing.assert_array_equal(xla, [2])
        assert inference._compiled_predict._cache_size() == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_default_kernel_resolution_waits_for_temporary_explicit_context(
    monkeypatch,
) -> None:
    monkeypatch.delenv(TRIANGLE_BACKEND_ENV, raising=False)
    entered = threading.Event()
    release = threading.Event()
    resolved: list[str] = []

    def hold_explicit_context() -> None:
        with triangle_backend("xla"):
            entered.set()
            assert release.wait(timeout=5)

    def resolve_default() -> None:
        assert entered.wait(timeout=5)
        resolved.append(resolve_triangle_kernel(None, cp_shards=1))

    holder = threading.Thread(target=hold_explicit_context)
    resolver = threading.Thread(target=resolve_default)
    holder.start()
    assert entered.wait(timeout=5)
    resolver.start()
    # The patched resolver waits on the same lock. The historical lock-free
    # read completes here and records the transient "xla" value.
    resolver.join(timeout=0.1)
    release.set()
    holder.join(timeout=5)
    resolver.join(timeout=5)

    assert not holder.is_alive()
    assert not resolver.is_alive()
    assert resolved == ["cueq"]
    assert TRIANGLE_BACKEND_ENV not in os.environ


def test_lowered_compiled_callable_preserves_public_arguments(monkeypatch) -> None:
    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        del key, config, kwargs
        return batch["x"] + params + representative_atoms.cb_offset[0]

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    args = (
        jax.random.key(0),
        {"x": jnp.ones((4,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(1))
    try:
        expected = run(*args)
        compiled = run.lower(*args).compile()
        actual = compiled(*args)
        jax.block_until_ready((expected, actual))
        np.testing.assert_array_equal(expected, actual)

        mask = jnp.ones((1, 4), dtype=bool)
        masked = inference.compile_predict(_config(), _table(1))
        expected_masked = masked(*args, noise_mask=mask)
        compiled_masked = masked.lower(*args, noise_mask=mask).compile()
        actual_masked = compiled_masked(*args, noise_mask=mask)
        jax.block_until_ready((expected_masked, actual_masked))
        np.testing.assert_array_equal(expected_masked, actual_masked)
        with pytest.raises(ValueError, match="RNG route"):
            compiled_masked(*args)
    finally:
        inference._compiled_predict.clear_cache()


def test_compile_factory_does_not_shadow_an_external_capture(monkeypatch) -> None:
    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        del key, params, config, representative_atoms, kwargs
        return _capture.capture("external", batch["x"])

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    args = (
        jax.random.key(0),
        {"x": jnp.ones((4,), dtype=jnp.float32)},
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    try:
        with _capture.capturing(("external",)) as captured:
            value = inference.compile_predict(_config(), _table())(*args)
        jax.block_until_ready(value)
        assert "external" in captured
    finally:
        inference._compiled_predict.clear_cache()


def test_deleted_persistent_scope_is_repopulated_on_memory_hit(
    tmp_path, monkeypatch
) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    cache = tmp_path / "jax-cache"
    args = (
        jax.random.key(0),
        {"x": jnp.ones((1024,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(cache))

    def populate() -> None:
        with compilation_cache_scope(cache):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            value = run(*args)
            jax.block_until_ready(value)

    try:
        populate()
        assert any(cache.iterdir())
        for entry in tuple(cache.iterdir()):
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        assert not any(cache.iterdir())

        populate()

        assert any(cache.iterdir())
        assert traces == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_deletion_during_memory_hit_is_not_absorbed(tmp_path, monkeypatch) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    cache = tmp_path / "racy-cache"
    args = (
        jax.random.key(0),
        {"x": jnp.ones((1024,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(cache))
    original_observe = inference.observe_cache_scope
    observations = 0

    def delete_before_observe(
        scope, *, token=None, require_payload=False, require_atime=False
    ) -> None:
        nonlocal observations
        observations += 1
        if observations == 2:
            for entry in tuple(cache.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        original_observe(
            scope,
            token=token,
            require_payload=require_payload,
            require_atime=require_atime,
        )

    monkeypatch.setattr(inference, "observe_cache_scope", delete_before_observe)

    def populate() -> None:
        with compilation_cache_scope(cache):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            value = run(*args)
            jax.block_until_ready(value)

    try:
        populate()
        assert any(cache.iterdir())
        populate()
        assert not any(cache.iterdir())
        assert traces == 1

        populate()

        assert any(cache.iterdir())
        assert traces == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_retargeted_cache_symlink_is_repopulated_on_memory_hit(
    tmp_path, monkeypatch
) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"
    first.mkdir()
    second.mkdir()
    scope = tmp_path / "cache"
    scope.symlink_to(first, target_is_directory=True)
    args = (
        jax.random.key(0),
        {"x": jnp.ones((1024,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(scope))

    def populate() -> None:
        with compilation_cache_scope(scope):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            value = run(*args)
            jax.block_until_ready(value)

    try:
        populate()
        assert any(first.iterdir())
        assert not any(second.iterdir())
        scope.unlink()
        scope.symlink_to(second, target_is_directory=True)

        populate()

        assert any(second.iterdir())
        assert traces == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_unrelated_persistent_cache_addition_does_not_invalidate(tmp_path) -> None:
    scope = tmp_path / "shared-cache"
    scope.mkdir()
    (scope / "existing-cache").write_bytes(b"first")
    observe_cache_scope(str(scope))

    (scope / "other-process-cache").write_bytes(b"second")

    assert not cache_scope_changed(str(scope))


def test_cache_access_metadata_does_not_invalidate(tmp_path) -> None:
    scope = tmp_path / "bounded-cache"
    scope.mkdir()
    payload = scope / "jit_program-cache"
    access = scope / "jit_program-atime"
    lock = scope / "writer.lockfile"
    payload.write_bytes(b"program")
    access.write_bytes(b"1")
    lock.write_bytes(b"locked")
    observe_cache_scope(str(scope))

    access.write_bytes(b"2")
    lock.unlink()

    assert not cache_scope_changed(str(scope))

    payload.write_bytes(b"corrupt")

    assert cache_scope_changed(str(scope))
    observe_cache_scope(str(scope))

    access.unlink()

    assert cache_scope_changed(str(scope))


def test_bounded_repair_waits_for_a_concurrent_jax_writer(tmp_path) -> None:
    fcntl = pytest.importorskip("fcntl")
    scope = tmp_path / "writer-cache"
    scope.mkdir()
    payload_written = threading.Event()
    finish_write = threading.Event()
    repair_started = threading.Event()
    repair_done = threading.Event()

    def write_pair() -> None:
        with (scope / ".lockfile").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            (scope / "program-cache").write_bytes(b"payload")
            payload_written.set()
            assert finish_write.wait(2)
            (scope / "program-atime").write_bytes(b"access")
            fcntl.flock(lock, fcntl.LOCK_UN)

    def repair() -> None:
        repair_started.set()
        snapshot = compile_helpers._cache_scope_snapshot(str(scope))
        compile_helpers._repair_cache_scope_atime_pairs(str(scope), snapshot)
        repair_done.set()

    writer = threading.Thread(target=write_pair)
    writer.start()
    assert payload_written.wait(1)
    repairer = threading.Thread(target=repair)
    repairer.start()
    assert repair_started.wait(1)
    assert not repair_done.wait(0.1)

    finish_write.set()
    writer.join(timeout=2)
    repairer.join(timeout=2)

    assert not writer.is_alive()
    assert not repairer.is_alive()
    assert (scope / "program-cache").read_bytes() == b"payload"
    assert (scope / "program-atime").read_bytes() == b"access"


def test_bounded_repair_anchors_symlink_target_before_mutation(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"
    first.mkdir()
    second.mkdir()
    (first / "orphan-cache").write_bytes(b"orphan")
    (second / "valid-cache").write_bytes(b"valid")
    (second / "valid-atime").write_bytes(b"access")
    scope = tmp_path / "cache"
    scope.symlink_to(first, target_is_directory=True)
    original_entries = compile_helpers._cache_scope_entries_from_fd
    retargeted = False

    def retarget_after_anchored_list(directory_fd):
        nonlocal retargeted
        entries = original_entries(directory_fd)
        if not retargeted:
            retargeted = True
            scope.unlink()
            scope.symlink_to(second, target_is_directory=True)
        return entries

    monkeypatch.setattr(
        compile_helpers,
        "_cache_scope_entries_from_fd",
        retarget_after_anchored_list,
    )
    snapshot = compile_helpers._cache_scope_snapshot(str(scope))

    compile_helpers._repair_cache_scope_atime_pairs(str(scope), snapshot)

    assert not (first / "orphan-cache").exists()
    assert (second / "valid-cache").read_bytes() == b"valid"
    assert (second / "valid-atime").read_bytes() == b"access"


def test_bounded_repair_rejects_directory_and_dangling_symlink_pairs(
    tmp_path,
) -> None:
    scope = tmp_path / "malformed-cache"
    scope.mkdir()
    (scope / "directory-cache").mkdir()
    (scope / "directory-atime").write_bytes(b"access")
    (scope / "link-cache").write_bytes(b"payload")
    (scope / "link-atime").symlink_to(scope / "missing-atime")
    snapshot = compile_helpers._cache_scope_snapshot(str(scope))

    repaired = compile_helpers._repair_cache_scope_atime_pairs(str(scope), snapshot)

    assert not compile_helpers._cache_scope_population_ready(
        repaired, require_atime=True
    )
    assert not any(path.name.endswith(("-cache", "-atime")) for path in scope.iterdir())
    assert any(path.name.startswith(".foldjax-invalid-") for path in scope.iterdir())


def test_flock_error_is_nonfatal_and_does_not_leak_a_descriptor(
    tmp_path, monkeypatch
) -> None:
    fcntl = pytest.importorskip("fcntl")
    descriptors = tmp_path / "descriptors"
    descriptors.mkdir()
    directory_fd = os.open(
        descriptors,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    proc_fds = os.path.exists("/proc/self/fd")
    before = len(os.listdir("/proc/self/fd")) if proc_fds else 0

    def unsupported(*args, **kwargs):
        del args, kwargs
        raise OSError("unsupported")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    try:
        with compile_helpers._cache_scope_file_lock(
            descriptors, directory_fd=directory_fd
        ) as acquired:
            assert not acquired
        if proc_fds:
            assert len(os.listdir("/proc/self/fd")) == before
    finally:
        os.close(directory_fd)


def test_unrepairable_bounded_cache_retries_only_once(tmp_path, monkeypatch) -> None:
    scope = tmp_path / "unrepairable-cache"
    scope.mkdir()
    (scope / "orphan-cache").write_bytes(b"payload")
    monkeypatch.setattr(
        compile_helpers,
        "_repair_cache_scope_atime_pairs",
        lambda _scope, snapshot: snapshot,
    )
    compile_helpers.reset_cache_scope_tracking()

    try:
        first = compile_helpers.inspect_cache_scope(str(scope), repair_atime=True)
        assert first is not None and not first.invalidated
        compile_helpers.observe_cache_scope(
            str(scope), token=first, require_payload=True, require_atime=True
        )

        retry = compile_helpers.inspect_cache_scope(str(scope), repair_atime=True)
        assert retry is not None and retry.invalidated
        compile_helpers.observe_cache_scope(
            str(scope), token=retry, require_payload=True, require_atime=True
        )

        stable = compile_helpers.inspect_cache_scope(str(scope), repair_atime=True)
        assert stable is not None and not stable.invalidated
    finally:
        compile_helpers.reset_cache_scope_tracking()


def test_cache_lock_symlink_is_not_followed(tmp_path) -> None:
    pytest.importorskip("fcntl")
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose O_NOFOLLOW")
    scope = tmp_path / "symlink-lock-cache"
    scope.mkdir()
    target = tmp_path / "outside-lock"
    target.write_bytes(b"outside")
    (scope / ".lockfile").symlink_to(target)
    directory_fd = os.open(
        scope,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )

    try:
        with compile_helpers._cache_scope_file_lock(
            scope, directory_fd=directory_fd
        ) as acquired:
            assert not acquired
    finally:
        os.close(directory_fd)

    assert target.read_bytes() == b"outside"


def test_first_compile_deletion_is_not_accepted_as_an_empty_baseline(
    tmp_path, monkeypatch
) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    cache = tmp_path / "first-race-cache"
    args = (
        jax.random.key(0),
        {"x": jnp.ones((1024,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(cache))
    original_observe = inference.observe_cache_scope
    observations = 0

    def delete_before_observe(
        scope, *, token=None, require_payload=False, require_atime=False
    ) -> None:
        nonlocal observations
        observations += 1
        if observations == 1:
            for entry in tuple(cache.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        original_observe(
            scope,
            token=token,
            require_payload=require_payload,
            require_atime=require_atime,
        )

    monkeypatch.setattr(inference, "observe_cache_scope", delete_before_observe)

    def populate() -> None:
        with compilation_cache_scope(cache):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            value = run(*args)
            jax.block_until_ready(value)

    try:
        populate()
        assert not any(cache.iterdir())
        assert traces == 1

        populate()

        assert any(cache.iterdir())
        assert traces == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_missing_persistent_payload_retries_once_then_accepts_unavailable(
    tmp_path, monkeypatch
) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    cache = tmp_path / "threshold-cache"
    args = (
        jax.random.key(0),
        {"x": jnp.ones((16,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(cache))

    def populate() -> None:
        with compilation_cache_scope(cache):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1_000_000.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.block_until_ready(run(*args))

    try:
        populate()
        populate()
        populate()

        assert traces == 2
        assert not any(path.name.endswith("-cache") for path in cache.iterdir())
    finally:
        inference._compiled_predict.clear_cache()


def test_disabled_and_zero_size_persistent_caches_do_not_request_population(
    tmp_path,
) -> None:
    cache = tmp_path / "population-disabled-cache"
    scope = inference.canonical_cache_scope(str(cache))
    previous_enabled = jax.config.jax_enable_compilation_cache
    previous_max_size = jax.config.jax_compilation_cache_max_size

    try:
        with compilation_cache_scope(cache):
            jax.config.update("jax_enable_compilation_cache", False)
            assert not inference._persistent_cache_matches(scope)

            jax.config.update("jax_enable_compilation_cache", True)
            jax.config.update("jax_compilation_cache_max_size", 0)
            assert not inference._persistent_cache_matches(scope)

            jax.config.update("jax_compilation_cache_max_size", 1_000_000)
            assert inference._persistent_cache_matches(scope)
            assert inference._persistent_cache_is_bounded(scope)

            jax.config.update("jax_compilation_cache_max_size", -1)
            assert inference._persistent_cache_matches(scope)
            assert not inference._persistent_cache_is_bounded(scope)
    finally:
        jax.config.update("jax_enable_compilation_cache", previous_enabled)
        jax.config.update("jax_compilation_cache_max_size", previous_max_size)


def test_bounded_cache_retries_an_orphan_payload_after_first_compile(
    tmp_path, monkeypatch
) -> None:
    traces = 0

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        **kwargs,
    ):
        nonlocal traces
        del key, config, representative_atoms, kwargs
        traces += 1
        return batch["x"] + params

    monkeypatch.setattr(inference, "predict", tiny_predict)
    inference._compiled_predict.clear_cache()
    cache = tmp_path / "bounded-first-race-cache"
    args = (
        jax.random.key(0),
        {"x": jnp.ones((1024,), dtype=jnp.float32)},
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    run = inference.compile_predict(_config(), _table(), cache_scope=str(cache))
    original_observe = inference.observe_cache_scope
    observations = 0

    def delete_atime_before_observe(
        scope, *, token=None, require_payload=False, require_atime=False
    ) -> None:
        nonlocal observations
        observations += 1
        if observations == 2:
            for entry in cache.iterdir():
                if entry.name.endswith("-cache"):
                    entry.with_name(
                        f"{entry.name[: -len('-cache')]}-atime"
                    ).write_bytes(b"access")
        original_observe(
            scope,
            token=token,
            require_payload=require_payload,
            require_atime=require_atime,
        )

    monkeypatch.setattr(inference, "observe_cache_scope", delete_atime_before_observe)
    monkeypatch.setattr(inference, "_persistent_cache_is_bounded", lambda scope: True)

    def populate() -> None:
        with compilation_cache_scope(cache):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.block_until_ready(run(*args))

    try:
        populate()
        assert not any(path.name.endswith("-cache") for path in cache.iterdir())

        populate()
        populate()

        payloads = {
            path.name[: -len("-cache")]
            for path in cache.iterdir()
            if path.name.endswith("-cache")
        }
        atimes = {
            path.name[: -len("-atime")]
            for path in cache.iterdir()
            if path.name.endswith("-atime")
        }
        assert payloads
        assert payloads <= atimes
        assert traces == 2

        orphan = cache / "other-program-cache"
        orphan.write_bytes(b"orphan")
        populate()

        assert not orphan.exists()
        assert traces == 2
    finally:
        inference._compiled_predict.clear_cache()


def test_noise_sources_remain_mutually_exclusive() -> None:
    compiled = inference.compile_predict(_config(), _table())
    with pytest.raises(ValueError, match="mutually exclusive"):
        compiled(
            jax.random.key(0),
            {"x": jnp.ones((4,), dtype=jnp.float32)},
            jnp.asarray(0.0),
            noise_tape=jnp.ones((1,)),
            noise_mask=jnp.ones((1,), dtype=bool),
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(31, dtype=np.float32),
        np.full(32, np.nan, dtype=np.float32),
        np.full(32, "x", dtype="U1"),
    ],
)
def test_representative_atom_table_is_validated_before_tracing(bad) -> None:
    fields = list(_table())
    fields[0] = bad
    with pytest.raises(ValueError, match="representative atom field"):
        inference.compile_predict(_config(), RepresentativeAtomTable(*fields))


def test_dynamic_representative_table_is_exact_for_the_real_gather_kernel() -> None:
    batch = {
        name: jnp.asarray(value)
        for name, value in minimal_features(tokens=3, atoms=6).items()
    }
    positions = jnp.arange(18, dtype=jnp.float32).reshape(1, 6, 3) / 7
    mask = batch["atom_mask"]
    table = _table()

    constant = jax.jit(
        lambda features, x, atom_mask: token_representative_atoms(
            features, x, atom_mask, table
        )
    )(batch, positions, mask)
    dynamic = jax.jit(token_representative_atoms)(batch, positions, mask, table)

    for expected, actual in zip(constant, dynamic, strict=True):
        np.testing.assert_array_equal(expected, actual)


_FOUR_DEVICE_STABLE_JIT_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from foldjax._openfold3_compile import triangle_backend
    from foldjax.models._cp import (
        context_parallel,
        cp_layout,
        replicate_tree,
        shard_pair_rows,
    )
    from foldjax.models.openfold3 import inference
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    traces = []

    def tiny_predict(
        key,
        batch,
        params,
        config,
        representative_atoms,
        *,
        n_chain=None,
        noise_fn=None,
        noise_tape=None,
        noise_mask=None,
        augment=True,
        use_trunk_pair_embedding=True,
    ):
        del key, noise_fn, noise_tape, noise_mask
        traces.append(
            (
                cp_layout(),
                config.cp_layout,
                n_chain,
                os.environ["OPENFOLD3_TRIANGLE_BACKEND"],
                augment,
                use_trunk_pair_embedding,
            )
        )
        return (
            shard_pair_rows(batch["pair"])
            + params
            + representative_atoms.cb_offset[0]
        )

    inference.predict = tiny_predict
    inference._compiled_predict.clear_cache()
    zeros = np.zeros(32, dtype=np.float32)
    table = RepresentativeAtomTable(*([zeros] * 13))
    base = inference.released_config(
        n_token=4,
        n_atom=4,
        num_cycles=1,
        num_samples=1,
        no_rollout_steps=1,
        pair_chunk_size=None,
        msa_depth=1,
    )
    row = base._replace(cp_shards=4, cp_layout="1d")
    grid = base._replace(cp_shards=4, cp_layout="2d")
    args = (
        jax.random.key(0),
        {"pair": jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)},
        jnp.asarray(0.25, dtype=jnp.float32),
    )

    serial_run = inference.compile_predict(base, table, cache_scope="probe")
    row_run = inference.compile_predict(row, table, cache_scope="probe")
    grid_run = inference.compile_predict(grid, table, cache_scope="probe")
    serial = serial_run(*args)
    row_value = row_run(*args)
    grid_value = grid_run(*args)
    jax.block_until_ready((serial, row_value, grid_value))
    np.testing.assert_array_equal(serial, row_value)
    np.testing.assert_array_equal(serial, grid_value)
    assert len(traces) == 3, traces
    assert inference._compiled_predict._cache_size() == 3

    # A fresh factory with an identical graph identity must hit the same trace.
    repeated = inference.compile_predict(row, table, cache_scope="probe")(*args)
    jax.block_until_ready(repeated)
    assert len(traces) == 3, traces

    lowered_serial = serial_run.lower(*args)
    lowered_row = row_run.lower(*args)
    lowered_grid = grid_run.lower(*args)
    assert len(traces) == 3, traces
    hlo_serial = lowered_serial.compiler_ir(dialect="hlo").as_hlo_text()
    hlo_row = lowered_row.compiler_ir(dialect="hlo").as_hlo_text()
    hlo_grid = lowered_grid.compiler_ir(dialect="hlo").as_hlo_text()
    assert hlo_serial != hlo_row
    assert hlo_row != hlo_grid
    assert "sharding" in hlo_row.lower()
    assert "sharding" in hlo_grid.lower()

    # The public compiled callable must retain the factory's three-argument
    # API and re-enter the same mesh before placing hidden dynamic inputs.
    compiled_row = lowered_row.compile()
    compiled_row_value = compiled_row(*args)
    jax.block_until_ready(compiled_row_value)
    np.testing.assert_array_equal(row_value, compiled_row_value)

    # Physical order is part of the identity even when logical mesh shape and
    # layout match; otherwise a prior executable retains stale device bindings.
    devices = list(jax.devices())
    forward = Mesh(np.asarray(devices), ("cp",))
    reverse = Mesh(np.asarray(devices[::-1]), ("cp",))
    assert inference._cp_topology_identity(
        forward, layout="1d"
    ) != inference._cp_topology_identity(reverse, layout="1d")

    # Exercise those ordered identities as executable cache keys. Reusing the
    # first executable here would retain its NamedSharding on devices 0/1 and
    # fail (or misplace data) when the second call is placed on devices 2/3.
    inference._compiled_predict.clear_cache()
    subset_config = base._replace(cp_shards=2, cp_layout="1d")
    subset_values = []
    subset_trace_start = len(traces)
    for chosen in (devices[:2], devices[2:]):
        with (
            triangle_backend("xla"),
            context_parallel(2, layout="1d", devices=chosen) as mesh,
        ):
            identity = inference._PredictGraphIdentity(
                config=subset_config,
                n_chain=None,
                augment=True,
                use_trunk_pair_embedding=True,
                rng_route="native",
                triangle_kernel="xla",
                cp_topology=inference._cp_topology_identity(mesh, layout="1d"),
                cache_scope="probe",
            )
            value = inference._compiled_predict(
                replicate_tree(args[0]),
                replicate_tree(args[1]),
                replicate_tree(args[2]),
                replicate_tree(table),
                None,
                None,
                identity=identity,
            )
            jax.block_until_ready(value)
            subset_values.append(np.asarray(value))
    np.testing.assert_array_equal(subset_values[0], subset_values[1])
    assert len(traces) - subset_trace_start == 2, traces
    assert inference._compiled_predict._cache_size() == 2
    print(
        "OPENFOLD3_STABLE_JIT_4_DEVICE_OK "
        f"initial_traces=3 ordered_device_traces=2 "
        f"hlo_bytes={len(hlo_serial)}/{len(hlo_row)}/{len(hlo_grid)}"
    )
    """
)


def test_stable_jit_serial_and_forced_four_device_parity_hlo() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _FOUR_DEVICE_STABLE_JIT_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_SKIP_MESH_CHECK": "1",
            **inherited_environment(),
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_STABLE_JIT_4_DEVICE_OK" in completed.stdout
