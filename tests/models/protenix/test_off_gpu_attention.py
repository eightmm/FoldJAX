"""Protenix's released tokamax denoiser attention, off a GPU.

tokamax implements its fused attention for GPUs only, so on a CPU host the
released defaults raised ``NotImplementedError: Not supported on cpu`` from the
first denoiser trace. An omitted option now resolves to the traced XLA path
there; a spelled ``tokamax`` is refused before any work, in one line.
"""

import json
from pathlib import Path

import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends import protenix as backend_impl
from foldjax.backends.protenix import ProtenixBackend
from foldjax.schema import PredictionRequest


def _request(tmp_path: Path, **options: object) -> PredictionRequest:
    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "protenix.jax"
    weights.write_bytes(b"native fixture")
    return PredictionRequest(
        model="protenix",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        seed=0,
        options={"model_name": "protenix_base_default_v1.0.0", **options},
    )


def test_an_omitted_option_resolves_to_xla_jit_off_a_gpu(tmp_path: Path) -> None:
    backend = ProtenixBackend()
    omitted = _request(tmp_path)
    explicit = _request(tmp_path, diffusion_attention_backend="xla_jit")

    assert not backend_impl._gpu_process()
    assert backend.apply_sampling(omitted)["diffusion_attention_backend"] == "xla_jit"
    # Recorded where the program is named, and shared with the explicit spelling.
    profile = backend.cache_profile(omitted)
    assert profile["diffusion_attention_backend"] == "xla_jit"
    assert profile == backend.cache_profile(explicit)
    assert resolve_cache_dir(omitted, backend) == resolve_cache_dir(explicit, backend)
    argv = backend._native_invocation(omitted).argv
    assert argv[argv.index("--diffusion-attention-backend") + 1] == "xla_jit"


def test_an_omitted_option_keeps_the_released_kernel_on_a_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(backend_impl, "_gpu_process", lambda: True)
    backend = ProtenixBackend()

    options = backend.apply_sampling(_request(tmp_path))

    assert "diffusion_attention_backend" not in options
    assert "diffusion_attention_backend" not in backend.cache_profile(
        _request(tmp_path)
    )


@pytest.mark.parametrize(
    "option",
    [
        "diffusion_attention_backend",
        "trunk_single_attention_backend",
        "trunk_triangle_attention_backend",
    ],
)
def test_a_spelled_tokamax_is_refused_in_one_line_off_a_gpu(
    tmp_path: Path, option: str
) -> None:
    backend = ProtenixBackend()

    with pytest.raises(ValueError, match="needs a GPU") as caught:
        backend.predict(_request(tmp_path, **{option: "tokamax"}))

    assert f"--option {option}=xla_jit" in str(caught.value)
    assert not (tmp_path / "out").exists()
