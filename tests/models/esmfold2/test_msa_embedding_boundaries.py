import jax
import jax.numpy as jnp
import pytest

from foldjax.models.esmfold2.models import embedders


@pytest.mark.parametrize("native", [False, True])
def test_native_embedding_outputs_are_materialized_before_add(monkeypatch, native):
    calls = []

    def linear(x, params, prefix):
        calls.append(prefix)
        return jnp.ones((*x.shape[:-1], 2), dtype=jnp.bfloat16)

    def barrier(x):
        calls.append("barrier")
        return x

    def block(msa, pair, *args, **kwargs):
        assert bool(jnp.all(msa == 2))
        calls.append("block")
        return msa, pair

    monkeypatch.setattr(embedders, "linear", linear)
    monkeypatch.setattr(embedders, "msa_encoder_block", block)
    monkeypatch.setattr(jax.lax, "optimization_barrier", barrier)
    embedders.msa_encoder(
        jnp.zeros((1, 2, 2, 2)),
        jnp.zeros((1, 2, 3)),
        jnp.zeros((1, 2, 4, 33)),
        jnp.zeros((1, 2, 4)),
        jnp.zeros((1, 2, 4)),
        jnp.ones((1, 2, 4)),
        {},
        "msa_encoder",
        n_layers=1,
        native_opm_params={} if native else None,
    )
    assert calls == ["msa_encoder.embed", "msa_encoder.project_inputs"] + (
        ["barrier", "barrier"] if native else []
    ) + ["block"]
