"""The guided sampler's denoiser attention.

The guided sampler is eager, and an eager `pallas_call` binds a fresh `jit`
closure per call (`jax/_src/pallas/pallas_call.py`, `_pallas_call_impl`), so the
released tokamax attention compiled once per denoiser call there: on one RTX PRO
6000 the TFG run timed out at 60 minutes against 2.5 for the unguided one, with
4,147 unreadable `jit__jit_run` cache entries. Under `use_tfg_guidance` an
omitted option now resolves to this port's jitted XLA attention and a spelled
`tokamax` is refused; the unguided release is untouched.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax._src import dispatch

from foldjax.api import resolve_cache_dir
from foldjax.backends import protenix as backend_impl
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.protenix.models.model import protenix_infer_static
from foldjax.models.protenix.models.primitives import attention as attention_impl

from .test_model import _toy_features, _toy_params
from .test_tfg_guidance_option import _guidance_reaching_the_sampler, _request


@pytest.fixture
def _on_a_gpu(monkeypatch) -> None:
    """The released (GPU) resolution; nothing here executes the fused kernel."""

    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: True)


@pytest.mark.usefixtures("_on_a_gpu")
def test_guidance_resolves_an_omitted_denoiser_attention_to_xla_jit(
    tmp_path: Path,
) -> None:
    backend = ProtenixBackend()
    guided = _request(tmp_path, use_tfg_guidance=True)
    explicit = _request(
        tmp_path, use_tfg_guidance=True, diffusion_attention_backend="xla_jit"
    )

    assert backend.apply_sampling(guided)["diffusion_attention_backend"] == "xla_jit"
    # Recorded where the program is named, and shared with the explicit spelling.
    profile = backend.cache_profile(guided)
    assert profile["diffusion_attention_backend"] == "xla_jit"
    assert profile == backend.cache_profile(explicit)
    assert resolve_cache_dir(guided, backend) == resolve_cache_dir(explicit, backend)
    argv = backend._native_invocation(guided).argv
    assert argv[argv.index("--diffusion-attention-backend") + 1] == "xla_jit"


@pytest.mark.usefixtures("_on_a_gpu")
@pytest.mark.parametrize("spelling", [None, False, "false"])
def test_without_guidance_the_released_kernel_and_namespace_stay(
    tmp_path: Path, spelling
) -> None:
    backend = ProtenixBackend()
    options = {} if spelling is None else {"use_tfg_guidance": spelling}
    request = _request(tmp_path, **options)

    assert "diffusion_attention_backend" not in backend.apply_sampling(request)
    assert "diffusion_attention_backend" not in backend.cache_profile(request)
    assert "--diffusion-attention-backend" not in (
        backend._native_invocation(request).argv
    )
    assert resolve_cache_dir(request, backend) == resolve_cache_dir(
        _request(tmp_path, diffusion_attention_backend="tokamax"), backend
    )


@pytest.mark.usefixtures("_on_a_gpu")
@pytest.mark.parametrize(
    "option",
    [
        # The trunk is eager under guidance too; only its `*_jit` backends trace.
        *backend_impl._TOKAMAX_ATTENTION_OPTIONS,
        # The neutral knob, which spells the trunk single attention.
        "attention_kernel",
    ],
)
def test_guidance_refuses_a_spelled_tokamax_attention(
    tmp_path: Path, option: str
) -> None:
    request = _request(tmp_path, use_tfg_guidance=True, **{option: "tokamax"})
    with pytest.raises(ValueError, match="recompiles tokamax's fused attention"):
        ProtenixBackend().validate_request(request)
    # Unguided, the same spelling is the user's to make.
    ProtenixBackend().validate_request(_request(tmp_path, **{option: "tokamax"}))


def test_the_native_switch_runs_the_denoiser_attention_on_xla_jit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # The parser defaults to tokamax and cannot tell that from a spelled one,
    # so the run resolves it, beside the sampler-scan override.
    guided = _guidance_reaching_the_sampler(tmp_path, monkeypatch, "--use-tfg-guidance")
    assert guided["diffusion_attention_backend"] == "xla_jit"
    assert "diffusion attention runs xla_jit" in capsys.readouterr().out

    unguided = _guidance_reaching_the_sampler(tmp_path, monkeypatch)
    assert unguided["diffusion_attention_backend"] == "tokamax"


def _guided_toy_run(steps: int) -> jax.Array:
    """The toy model through the eager guided sampler.

    `rho > 0`, so `TFGEngine.step` also takes the denoiser under `jax.grad`.
    """

    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    return protenix_infer_static(
        _toy_features(),
        _toy_params(),
        jnp.linspace(1.0, 0.0, steps + 1, dtype=jnp.float32),
        key=None,
        num_samples=1,
        init_noise=init_noise,
        step_noises=tuple(jnp.zeros_like(init_noise) for _ in range(steps)),
        num_recycles=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        run_confidence=False,
        use_sampler_scan=False,
        diffusion_attention_backend=backend_impl._GUIDED_DIFFUSION_ATTENTION_BACKEND,
        guidance_config={
            "enable": True,
            "rho": 0.5,
            "mu": 0.1,
            "steps": {"tfg_inner": 1, "projection_outer": 0},
            "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
        },
        guidance_features={
            "interchain_bond_index": jnp.asarray([[0], [1]], dtype=jnp.int32)
        },
    )["coordinate"]


def _repeat_run_compiles(steps: int) -> int:
    """Backend compiles in a repeat of an identical guided run."""

    seen = [0]
    counting = [False]

    def listener(event: str, _duration: float, **_kwargs: object) -> None:
        if counting[0] and event == dispatch.BACKEND_COMPILE_EVENT:
            seen[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        _guided_toy_run(steps).block_until_ready()
        counting[0] = True
        _guided_toy_run(steps).block_until_ready()
    finally:
        counting[0] = False
        jax.monitoring.unregister_event_duration_listener(listener)
    return seen[0]


def test_the_guided_denoiser_attention_compiles_once_per_shape(monkeypatch) -> None:
    """A repeat of a guided run builds no program at all.

    Measured against a repeat rather than against "the steps after the first",
    because a schedule-gated guidance term may legitimately add a program
    mid-run; the first run warms every step's. The tripwire arm reproduces the
    mechanism tokamax hit -- a fresh `jit` closure per attention call, which is
    what `_pallas_call_impl` binds -- and must show compiles growing with the
    step count, or the zero certifies nothing.
    """

    assert _repeat_run_compiles(2) == 0

    compiled_attention = attention_impl._compiled_attention

    def fresh_jit_per_call(q_x, kv_x, params, num_heads, attn_bias=None, **kwargs):
        def _jit_run(q_x, kv_x, params, attn_bias):
            return compiled_attention(q_x, kv_x, params, num_heads, attn_bias, **kwargs)

        return jax.jit(_jit_run)(q_x, kv_x, params, attn_bias)

    monkeypatch.setattr(attention_impl, "_compiled_attention", fresh_jit_per_call)
    two, four = _repeat_run_compiles(2), _repeat_run_compiles(4)
    assert 0 < two < four
