from __future__ import annotations

import dataclasses
import gc
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax
import foldjax.backends.boltz2 as backend_module
import foldjax.models.boltz2.api as native_api
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.manifest import MANIFEST_NAME
from foldjax.schema import PredictionError, PredictionRequest
from tests.models.cp_probe_env import inherited_environment

_FORCED_CP_SESSION_PROBE = textwrap.dedent(
    r"""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import jax
    import jax.numpy as jnp
    import numpy as np

    import foldjax.models.boltz2.api as api
    import foldjax.models.boltz2.bridge.native as native
    import foldjax.models.boltz2.models.predict as predict_module
    from foldjax.backends.boltz2 import Boltz2Backend
    import foldjax.models._cp as cp_module
    from foldjax.schema import PredictionRequest

    assert jax.device_count() == 4, jax.devices()
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        weights = root / "model.npz"
        weights.write_bytes(b"weights")
        mols = root / "mols"
        mols.mkdir()
        input_path = root / "job.yaml"
        input_path.write_text("{}\n")
        features = {
            "atom_pad_mask": np.ones((1, 4), np.float32),
            "token_pad_mask": np.ones((1, 4), np.float32),
            "mol_type": np.zeros((1, 4), np.int32),
            "affinity_token_mask": np.zeros((1, 4), np.float32),
        }
        api.featurize = lambda **kwargs: (features, "job", root)
        counts = {"loads": 0, "traces": 0}

        def load_params(path):
            counts["loads"] += 1
            return {"trunk": {"weight": jnp.asarray([1.0])}}

        def predict(params, feats, key, **kwargs):
            counts["traces"] += 1
            value = jax.random.uniform(key, ())
            return {
                "sample_atom_coords": jnp.full((1, 4, 3), value),
                "plddt": jnp.ones((1, 4)),
                "iptm": jnp.asarray([value]),
            }

        native.load_params = load_params
        predict_module.boltz2_predict = predict
        original_replicate_tree = cp_module.replicate_tree
        placements = {"params": 0}

        def counted_replicate_tree(tree, **kwargs):
            if isinstance(tree, dict) and "trunk" in tree:
                placements["params"] += 1
            return original_replicate_tree(tree, **kwargs)

        cp_module.replicate_tree = counted_replicate_tree
        request = PredictionRequest(
            model="boltz2",
            input=input_path,
            input_format="native",
            weights=weights,
            output_dir=root / "out",
            num_seeds=2,
            options={"mols": mols},
        )
        backend = Boltz2Backend()
        values = []
        with backend.session((request,)):
            for seed in request.resolved_seeds:
                output = api.predict(
                    seq=["AAAA"],
                    weights=weights,
                    mols=mols,
                    out_dir=root,
                    seed=seed,
                    write_fmt=None,
                    _runtime=backend,
                    cp_devices=4,
                    cp_layout="1d",
                    cp_atom_windows=False,
                    attention_backend="xla",
                    triangle_backend="xla",
                    glu_backend="xla",
                )
                values.append(float(output["coords"][0, 0]))
            assert counts == {"loads": 1, "traces": 1}, counts
            assert placements == {"params": 1}, placements
            output = api.predict(
                seq=["AAAA"],
                weights=weights,
                mols=mols,
                out_dir=root,
                seed=2,
                write_fmt=None,
                _runtime=backend,
                cp_devices=4,
                cp_layout="2d",
                cp_atom_windows=False,
                attention_backend="xla",
                triangle_backend="xla",
                glu_backend="xla",
            )
            assert counts == {"loads": 2, "traces": 2}, counts
            assert placements == {"params": 2}, placements
            assert len(set(values + [float(output["coords"][0, 0])])) == 3
        assert not backend._params
        assert not backend._runners
    print("BOLTZ2_SESSION_CP_OK")
    """
)


def _features(*, affinity: bool = False) -> dict[str, np.ndarray]:
    return {
        "atom_pad_mask": np.ones((1, 3), dtype=np.float32),
        "token_pad_mask": np.ones((1, 2), dtype=np.float32),
        "mol_type": np.asarray([[0, 0]], dtype=np.int32),
        "affinity_token_mask": np.asarray(
            [[0.0, 1.0 if affinity else 0.0]], dtype=np.float32
        ),
    }


def _request(
    tmp_path: Path,
    *,
    weights: Path | None = None,
    affinity: bool = False,
    on_error: str = "stop",
) -> PredictionRequest:
    input_path = tmp_path / "job.yaml"
    input_path.write_text("affinity: {}\n" if affinity else "{}\n")
    weights = weights or tmp_path / "boltz2_conf.npz"
    if not weights.exists():
        weights.write_bytes(b"weights")
    mols = tmp_path / "mols"
    mols.mkdir(exist_ok=True)
    return PredictionRequest(
        model="boltz2",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        num_seeds=2,
        cache_dir=tmp_path / "cache",
        on_error=on_error,
        options={"mols": mols, "write_fmt": None},
    )


def _patch_primary_runtime(monkeypatch, tmp_path: Path, counts: dict[str, int]) -> None:
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: (_features(), "job", tmp_path),
    )

    def fake_load_params(path: Path):
        counts["loads"] += 1
        return {"trunk": {"weight": jnp.asarray([1.0])}}

    def fake_predict(params, feats, key, **kwargs):
        counts["traces"] += 1
        draw_key, _ = jax.random.split(key)
        value = jax.random.uniform(draw_key, ()) + params["trunk"]["weight"][0]
        return {
            "sample_atom_coords": jnp.full((1, 3, 3), value),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.asarray([value]),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )


def test_native_session_reuses_primary_params_and_jit_across_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    counts = {"loads": 0, "traces": 0}
    _patch_primary_runtime(monkeypatch, tmp_path, counts)
    backend = Boltz2Backend()

    with backend.session((request,)):
        first = native_api.predict(
            seq=["AA"],
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            seed=0,
            write_fmt=None,
            _runtime=backend,
        )
        second = native_api.predict(
            seq=["AA"],
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            seed=1,
            write_fmt=None,
            _runtime=backend,
        )

        assert counts == {"loads": 1, "traces": 1}
        assert not np.array_equal(first["coords"], second["coords"])
        assert set(backend._params) == {"primary"}
        assert set(backend._runners) == {"primary"}

    assert backend._params == {}
    assert backend._runners == {}
    scalar = native_api.predict(
        seq=["AA"],
        weights=request.weights,
        mols=request.options["mols"],
        out_dir=tmp_path,
        seed=1,
        write_fmt=None,
    )
    np.testing.assert_array_equal(second["coords"], scalar["coords"])
    np.testing.assert_array_equal(second["plddt"], scalar["plddt"])


def test_public_batch_reuses_primary_params_and_jit_across_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    counts = {"loads": 0, "traces": 0}
    _patch_primary_runtime(monkeypatch, tmp_path, counts)

    report = foldjax.predict_batch(request)

    assert counts == {"loads": 1, "traces": 1}
    assert not report.failures
    assert [sample.seed for sample in report.results[0].samples] == [0, 1]
    assert not np.array_equal(
        report.results[0].samples[0].coordinates,
        report.results[0].samples[1].coordinates,
    )


def test_native_session_reuses_primary_and_affinity_stages(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path, affinity=True)
    affinity_weights = tmp_path / "boltz2_aff.npz"
    affinity_weights.write_bytes(b"affinity")
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: (_features(affinity=True), "job", tmp_path),
    )
    monkeypatch.setattr(
        native_api,
        "_prepare_affinity_features",
        lambda **kwargs: _features(affinity=True),
    )
    loads: list[str] = []
    traces: list[str] = []

    def fake_load_params(path: Path):
        role = "affinity" if Path(path) == affinity_weights else "primary"
        loads.append(role)
        if role == "affinity":
            return {
                "trunk": {"weight": jnp.asarray([2.0])},
                "affinity": {"weight": jnp.asarray([3.0])},
            }
        return {"trunk": {"weight": jnp.asarray([1.0])}}

    def fake_predict(params, feats, key, **kwargs):
        role = "affinity" if "affinity" in params else "primary"
        traces.append(role)
        samples = kwargs["multiplicity"]
        output = {
            "sample_atom_coords": jnp.zeros((samples, 3, 3)),
            "plddt": jnp.ones((samples, 2)),
        }
        if role == "affinity":
            output["affinity_pred_value"] = jnp.asarray([2.0])
        else:
            output["iptm"] = jnp.arange(samples, dtype=jnp.float32)
        return output

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )
    backend = Boltz2Backend()

    with backend.session((request,)):
        for seed in request.resolved_seeds:
            native_api.predict(
                seq=["AA"],
                weights=request.weights,
                affinity_weights=affinity_weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=seed,
                write_fmt=None,
                _runtime=backend,
            )
        assert loads == ["primary", "affinity"]
        assert traces == ["primary", "affinity"]
        assert set(backend._params) == {"primary", "affinity"}
        assert set(backend._runners) == {"primary", "affinity"}


def test_continued_failure_reloads_and_retraces_the_next_seed(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path, on_error="continue")
    counts = {"loads": 0, "traces": 0}
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: (_features(), "job", tmp_path),
    )

    def fake_load_params(path: Path):
        counts["loads"] += 1
        return {"trunk": {"weight": jnp.asarray([1.0])}}

    def fake_predict(params, feats, key, **kwargs):
        counts["traces"] += 1
        if counts["traces"] == 1:
            raise ValueError("synthetic first-seed failure")
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.asarray([0.5]),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", fake_load_params
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", fake_predict
    )

    report = foldjax.predict_batch(request)

    assert counts == {"loads": 2, "traces": 2}
    assert len(report.results) == 1
    assert len(report.results[0].samples) == 1
    assert len(report.failures) == 1
    assert report.failures[0].seed == 0


def test_all_resumed_seeds_do_not_import_the_native_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    calls: list[int] = []

    def fake_predict(**kwargs):
        calls.append(kwargs["seed"])
        output_dir = Path(kwargs["out_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        structure = output_dir / f"native-{kwargs['seed']}.cif"
        structure.write_text("data_job\n")
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.ones((2,)),
            "iptm": np.asarray([0.5]),
            "out_path": structure,
        }

    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda name: SimpleNamespace(predict=fake_predict),
    )
    first = foldjax.predict_batch(request)
    assert calls == [0, 1]
    assert not first.failures

    def fail_import(name: str):
        raise AssertionError(f"resumed batch imported {name}")

    monkeypatch.setattr(backend_module, "import_module", fail_import)
    resumed = foldjax.predict_batch(dataclasses.replace(request, resume=True))

    assert calls == [0, 1]
    assert not resumed.failures
    assert resumed.skipped == (
        request.output_dir / "seed_0",
        request.output_dir / "seed_1",
    )


def test_partial_resume_cannot_mix_checkpoint_generations(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)

    def fake_predict(**kwargs):
        output_dir = Path(kwargs["out_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        structure = output_dir / f"native-{kwargs['seed']}.cif"
        structure.write_text("data_job\n")
        return {
            "coords": np.zeros((2, 3)),
            "plddt": np.ones((2,)),
            "iptm": np.asarray([0.5]),
            "out_path": structure,
        }

    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda name: SimpleNamespace(predict=fake_predict),
    )
    assert not foldjax.predict_batch(request).failures
    (request.output_dir / "seed_1" / MANIFEST_NAME).unlink()

    original = Boltz2Backend.observe_resumed
    mutated = False

    def replace_after_resume(self, scalar_request):
        nonlocal mutated
        original(self, scalar_request)
        if not mutated:
            mutated = True
            request.weights.write_bytes(b"replacement generation")

    monkeypatch.setattr(Boltz2Backend, "observe_resumed", replace_after_resume)

    def fail_import(name: str):
        raise AssertionError(f"mixed-generation batch imported {name}")

    monkeypatch.setattr(backend_module, "import_module", fail_import)
    resumed = foldjax.predict_batch(
        dataclasses.replace(request, resume=True, on_error="continue")
    )

    assert resumed.skipped == (request.output_dir / "seed_0",)
    assert len(resumed.failures) == 1
    assert resumed.failures[0].seed == 1
    assert "weights changed" in resumed.failures[0].error


def test_higher_priority_weight_appearing_poisons_an_active_session(
    tmp_path: Path,
) -> None:
    base = tmp_path / "boltz2_conf"
    base.touch()
    npz = base.with_suffix(".npz")
    npz.write_bytes(b"npz")
    request = _request(tmp_path, weights=base)
    backend = Boltz2Backend()

    with backend.session((request,)):
        backend.validate_session(request)
        params = backend.load_params(
            "primary",
            base,
            lambda path: {"selected": Path(path).with_suffix(".npz")},
            placement=("cpu",),
        )
        assert params["selected"] == npz
        base.with_suffix(".safetensors").write_bytes(b"preferred")

        with pytest.raises(PredictionError, match="weights changed"):
            backend.validate_session(request)
        assert backend._params == {}
        assert backend._runners == {}


def test_runtime_identity_splits_dtype_and_ambient_triangle_modes(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    backend = Boltz2Backend()
    previous_x64 = jax.config.jax_enable_x64
    loads = 0

    def load_params(path):
        nonlocal loads
        loads += 1
        return jax.device_put(np.asarray([1.123456789012345], dtype=np.float64))

    try:
        with backend.session((request,)):
            jax.config.update("jax_enable_x64", False)
            narrow_identity = native_api._parameter_runtime_identity(
                jax,
                cp_devices=1,
                cp_layout="1d",
            )
            narrow = backend.load_params(
                "primary",
                request.weights,
                load_params,
                placement=narrow_identity,
            )
            jax.config.update("jax_enable_x64", True)
            wide_identity = native_api._parameter_runtime_identity(
                jax,
                cp_devices=1,
                cp_layout="1d",
            )
            wide = backend.load_params(
                "primary",
                request.weights,
                load_params,
                placement=wide_identity,
            )
            assert narrow_identity != wide_identity
            assert loads == 2
            assert narrow.dtype == jnp.float32
            assert wide.dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    xla_identity = native_api._runtime_identity(
        jax, cp_devices=1, cp_layout="1d", compile_cache=None
    )
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    cueq_identity = native_api._runtime_identity(
        jax, cp_devices=1, cp_layout="1d", compile_cache=None
    )
    assert xla_identity != cueq_identity
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "invalid")
    with pytest.raises(ValueError, match="must be 'cueq' or 'xla'"):
        native_api._runtime_identity(
            jax, cp_devices=1, cp_layout="1d", compile_cache=None
        )

    previous_prng = jax.config.jax_default_prng_impl
    try:
        monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
        jax.config.update("jax_default_prng_impl", "rbg")
        rbg_identity = native_api._runtime_identity(
            jax, cp_devices=1, cp_layout="1d", compile_cache=None
        )
        jax.config.update("jax_default_prng_impl", "unsafe_rbg")
        unsafe_identity = native_api._runtime_identity(
            jax, cp_devices=1, cp_layout="1d", compile_cache=None
        )
        assert rbg_identity != unsafe_identity
    finally:
        jax.config.update("jax_default_prng_impl", previous_prng)


def test_replacement_drops_old_params_before_loading_the_new_tree(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = Boltz2Backend()

    class Params:
        pass

    with backend.session((request,)):
        first = backend.load_params(
            "primary",
            request.weights,
            lambda path: Params(),
            placement=("one",),
        )
        old = weakref.ref(first)
        del first

        def replacement(path):
            gc.collect()
            assert old() is None
            return Params()

        second = backend.load_params(
            "primary",
            request.weights,
            replacement,
            placement=("two",),
        )
        assert isinstance(second, Params)


def test_session_retraces_when_the_prng_implementation_changes(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    counts = {"loads": 0, "traces": 0}
    _patch_primary_runtime(monkeypatch, tmp_path, counts)
    backend = Boltz2Backend()
    previous_prng = jax.config.jax_default_prng_impl
    try:
        with backend.session((request,)):
            jax.config.update("jax_default_prng_impl", "rbg")
            rbg = native_api.predict(
                seq=["AA"],
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=0,
                write_fmt=None,
                _runtime=backend,
            )
            jax.config.update("jax_default_prng_impl", "unsafe_rbg")
            unsafe = native_api.predict(
                seq=["AA"],
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=0,
                write_fmt=None,
                _runtime=backend,
            )
        scalar = native_api.predict(
            seq=["AA"],
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            seed=0,
            write_fmt=None,
        )
    finally:
        jax.config.update("jax_default_prng_impl", previous_prng)

    assert counts == {"loads": 2, "traces": 3}
    assert not np.array_equal(rbg["coords"], unsafe["coords"])
    np.testing.assert_array_equal(unsafe["coords"], scalar["coords"])


def test_keyboard_interrupt_is_preserved_and_cleans_session_state(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: (_features(), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda path: {"trunk": {"weight": jnp.asarray([1.0])}},
    )

    def interrupt(params, feats, key, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", interrupt
    )
    backend = Boltz2Backend()

    with pytest.raises(KeyboardInterrupt), backend.session((request,)):
        native_api.predict(
            seq=["AA"],
            weights=request.weights,
            mols=request.options["mols"],
            out_dir=tmp_path,
            seed=0,
            write_fmt=None,
            _runtime=backend,
        )

    assert backend._params == {}
    assert backend._runners == {}
    assert backend._session_open is False


def test_session_bounds_the_number_of_retained_jit_executables(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = Boltz2Backend()

    class FakeJit:
        def __init__(self) -> None:
            self.entries = 0
            self.clears = 0

        def __call__(self, value):
            self.entries += 1
            return value

        def _cache_size(self) -> int:
            return self.entries

        def clear_cache(self) -> None:
            self.entries = 0
            self.clears += 1

    native_runner = FakeJit()
    with backend.session((request,)):
        runner = backend.jit_runner(
            "primary",
            ("one-graph",),
            lambda value: value,
            lambda function: native_runner,
        )
        assert [runner(value) for value in range(8)] == list(range(8))
        assert native_runner.entries == 0
        assert native_runner.clears == 1

    assert native_runner.clears == 2


def test_session_reuses_one_cp_topology_and_splits_another() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _FORCED_CP_SESSION_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_SKIP_MESH_CHECK": "1",
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "BOLTZ2_SESSION_CP_OK" in completed.stdout
