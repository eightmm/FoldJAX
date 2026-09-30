"""The Boltz outer executable must preserve native BF16 materialization."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models.boltz2 import api, compile_policy
from foldjax.models.boltz2.models.trunk_blocks import trunk
from foldjax.schema import PaddingConfig
from tests.test_boltz2_session import _features, _request


@pytest.mark.parametrize("dtype", ["float32", jnp.float32, np.dtype("float32")])
def test_fp32_preserves_unconfigured_jit(dtype, monkeypatch):
    calls = []
    monkeypatch.setattr(jax, "jit", lambda function, **kw: calls.append(kw) or function)
    fn = lambda x: x  # noqa: E731
    assert compile_policy.jit(fn, compute_dtype=dtype) is fn
    assert calls == [{}]
    assert compile_policy.compiler_options(dtype) == {}


@pytest.mark.parametrize("dtype", ["bfloat16", jnp.bfloat16, np.dtype(jnp.bfloat16)])
def test_bf16_options_are_per_jit_and_fresh(dtype, monkeypatch):
    calls = []
    monkeypatch.setattr(jax, "jit", lambda function, **kw: calls.append(kw) or function)
    fn = lambda x: x  # noqa: E731
    assert compile_policy.jit(fn, compute_dtype=dtype) is fn
    assert calls == [{"compiler_options": {"xla_allow_excess_precision": False}}]
    options = compile_policy.compiler_options(dtype)
    options.clear()
    assert compile_policy.compiler_options(dtype) == {
        "xla_allow_excess_precision": False
    }


@pytest.mark.parametrize("dtype", [None, "float16", "bf16", "float64"])
def test_unsupported_dtype_fails_before_compile(dtype):
    with pytest.raises(ValueError, match="unsupported Boltz compute dtype"):
        compile_policy.jit(lambda x: x, compute_dtype=dtype)


def test_fp32_weights_are_rounded_inside_outer_selector_jit():
    weights = jnp.asarray([[1.003], [0], [0.001], [0], [0.001], [0], [0]], jnp.float32)
    features = {
        key: jnp.zeros((1, 1), jnp.int32)
        for key in (
            "asym_id",
            "residue_index",
            "entity_id",
            "token_index",
            "sym_id",
            "cyclic_period",
        )
    }

    def forward(kernel, feats):
        params = trunk._cast_trunk_params(
            {"rel_pos": {"linear_layer": {"kernel": kernel}}}, jnp.bfloat16
        )
        return trunk.relative_position_forward(
            params["rel_pos"], feats, r_max=0, s_max=0
        )

    executable = compile_policy.jit(forward, compute_dtype="bfloat16")
    actual = executable.lower(weights, features).compile()(weights, features)
    np.testing.assert_array_equal(np.asarray(actual, np.float32), [[[[1.0]]]])
    # This discriminates against preserving original FP32 weights then rounding
    # only the output; host-narrowed inputs would hide the regression.
    unrounded = (np.float32(1.003) + np.float32(0.001) * 2).astype(jnp.bfloat16)
    assert np.float32(unrounded) != 1.0


def test_runner_identity_separates_compiler_policy():
    kwargs = dict(
        predict_function=object(),
        predict_kwargs={"multiplicity": 5},
        noise_mode="none",
        runtime=(),
    )
    original = api._runner_identity(**kwargs)
    protected = api._runner_identity(
        **kwargs, compiler_options={"xla_allow_excess_precision": False}
    )
    assert original != protected
    assert protected[-1] == (
        "compiler_options",
        (("xla_allow_excess_precision", False),),
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("affinity", [False, True])
def test_high_level_primary_and_affinity_factories_and_metadata(
    tmp_path, monkeypatch, dtype, session, affinity
):
    request = _request(tmp_path, affinity=affinity)
    affinity_path = tmp_path / "boltz2_aff.npz"
    affinity_path.write_bytes(b"test-only-affinity")
    monkeypatch.setattr(
        api, "featurize", lambda **kw: (_features(affinity=affinity), "job", tmp_path)
    )
    monkeypatch.setattr(
        api, "_prepare_affinity_features", lambda **kw: _features(affinity=affinity)
    )

    def load(path):
        params = {"trunk": {"weight": jnp.ones(1)}}
        if path == affinity_path:
            params["affinity"] = {"weight": jnp.ones(1)}
        return params

    def predict(params, feats, key, **kwargs):
        assert kwargs["compute_dtype"] == getattr(jnp, dtype)
        samples = kwargs["multiplicity"]
        result = {
            "sample_atom_coords": jnp.zeros((samples, 3, 3)),
            "plddt": jnp.ones((samples, 2)),
            "iptm": jnp.zeros(samples),
        }
        if "affinity" in params:
            result["affinity_pred_value"] = jnp.ones(1)
        return result

    monkeypatch.setattr("foldjax.models.boltz2.bridge.native.load_params", load)
    monkeypatch.setattr("foldjax.models.boltz2.models.predict.boltz2_predict", predict)
    actual_jit = api._boltz_jit
    factories = []

    def observe(function, *, compute_dtype, deterministic=False):
        factories.append((function.__name__, compute_dtype))
        return actual_jit(
            function, compute_dtype=compute_dtype, deterministic=deterministic
        )

    monkeypatch.setattr(api, "_boltz_jit", observe)
    backend = Boltz2Backend() if session else None
    context = backend.session((request,)) if session else nullcontext()
    with context:
        for seed in range(2 if session else 1):
            output = api.predict(
                seq=["AA"],
                weights=request.weights,
                affinity_weights=affinity_path,
                mols=request.options["mols"],
                out_dir=tmp_path,
                write_fmt=None,
                compute_dtype=dtype,
                seed=seed,
                _runtime=backend,
            )
    expected = {
        "primary": {
            "mode": "jit",
            "scope": "outer_jit",
            "compute_dtype": dtype,
            "compiler_options": compile_policy.compiler_options(dtype),
        }
    }
    assert factories == [("run_model", dtype)] + (
        [("run_affinity", dtype)] if affinity else []
    )
    if affinity:
        expected["affinity"] = expected["primary"].copy()
    assert output["execution_policy"] == expected


def test_eager_steering_is_explicitly_outside_outer_jit_policy(tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.setattr(api, "featurize", lambda **kw: (_features(), "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict",
        lambda *args, **kw: {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.zeros(1),
        },
    )
    monkeypatch.setattr(
        api, "_boltz_jit", lambda *a, **kw: pytest.fail("unexpected jit")
    )
    result = api.predict(
        seq=["AA"],
        weights=request.weights,
        mols=request.options["mols"],
        out_dir=tmp_path,
        steering_args={"contact_guidance_update": True},
    )
    assert result["execution_policy"]["primary"] == {
        "mode": "eager",
        "compute_dtype": "bfloat16",
        "compiler_options": {},
        "scope": "eager_steering_not_covered",
    }


def _constrained(kind: str | None) -> dict[str, np.ndarray]:
    """`_features()` with what the featurizer emits for each constraint kind."""
    features = _features()
    features["contact_pair_index"] = np.zeros(
        (1, 2, 4 if kind == "contact" else 0), dtype=np.int64
    )
    features["template_force"] = np.asarray(
        [[kind == "template"]], dtype=bool
    )
    return features


def _run_guided(tmp_path, monkeypatch, features, **kwargs):
    request = _request(tmp_path)
    seen = {}
    monkeypatch.setattr(api, "featurize", lambda **kw: (features, "job", tmp_path))
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params", lambda path: {"trunk": {}}
    )

    def predict(*args, **kw):
        seen["steering_args"] = kw["steering_args"]
        return {
            "sample_atom_coords": jnp.zeros((1, 3, 3)),
            "plddt": jnp.ones((1, 2)),
            "iptm": jnp.zeros(1),
        }

    monkeypatch.setattr("foldjax.models.boltz2.models.predict.boltz2_predict", predict)
    result = api.predict(
        seq=["AA"],
        weights=request.weights,
        mols=request.options["mols"],
        out_dir=tmp_path,
        **kwargs,
    )
    return result, seen


@pytest.mark.parametrize("kind", ["contact", "template"])
def test_a_forced_constraint_turns_on_upstreams_contact_guidance(
    tmp_path, monkeypatch, kind
):
    """`boltz predict` passes contact_guidance_update=True (main.py:156)."""
    result, seen = _run_guided(tmp_path, monkeypatch, _constrained(kind))
    assert seen["steering_args"] == {
        "fk_steering": False,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": True,
        "num_gd_steps": 20,
    }
    assert result["execution_policy"]["primary"]["mode"] == "eager"


def test_an_unconstrained_job_keeps_the_compiled_sampler(tmp_path, monkeypatch):
    result, seen = _run_guided(tmp_path, monkeypatch, _constrained(None))
    assert seen["steering_args"] is None
    assert result["execution_policy"]["primary"]["mode"] == "jit"


def test_explicit_steering_args_opt_out_of_the_guidance(tmp_path, monkeypatch):
    off = {
        "fk_steering": False,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
    }
    result, seen = _run_guided(
        tmp_path, monkeypatch, _constrained("contact"), steering_args=off
    )
    assert seen["steering_args"] == off
    assert result["execution_policy"]["primary"]["mode"] == "jit"


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"deterministic": True}, "deterministic=true"),
        ({"cp_devices": 2}, "context parallelism"),
        ({"padding": PaddingConfig()}, "padding"),
    ],
)
def test_guidance_that_cannot_run_is_refused_not_skipped(
    tmp_path, monkeypatch, kwargs, named
):
    with pytest.raises(ValueError) as caught:
        _run_guided(tmp_path, monkeypatch, _constrained("contact"), **kwargs)
    message = str(caught.value)
    assert named in message
    assert "contact_guidance_update" in message
    assert '"contact_guidance_update": false' in message
