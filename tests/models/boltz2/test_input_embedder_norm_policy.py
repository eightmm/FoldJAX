import jax.numpy as jnp
import pytest

from foldjax.models.boltz2.models.trunk_blocks import input_embedder


@pytest.mark.parametrize(
    "dtype,expected", [(jnp.float32, "fp32"), (jnp.bfloat16, "amp")]
)
def test_atom_bias_norm_follows_projection_precision(monkeypatch, dtype, expected):
    p = jnp.zeros((1, 1, 32, 128, 16), jnp.float32)
    monkeypatch.setattr(
        input_embedder, "atom_encoder_forward", lambda *a, **kw: (p, p, p)
    )

    def norm_route(name):
        def call(value, scale, bias, eps):
            assert value is p
            assert eps == 1e-5
            raise RuntimeError(name)
        return call

    monkeypatch.setattr(input_embedder, "_layer_norm", norm_route("fp32"))
    monkeypatch.setattr(input_embedder, "amp_layer_norm", norm_route("amp"))
    params = {
        "atom_encoder": {},
        "atom_enc_proj_z": {
            "norm": {"scale": jnp.ones(16), "bias": jnp.zeros(16)},
            "linear": {"kernel": jnp.zeros((16, 12), dtype)},
        },
    }
    with pytest.raises(RuntimeError, match=f"^{expected}$"):
        input_embedder.input_embedder_forward(params, {})
