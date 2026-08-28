"""Model-level CPU proof for Boltz-2 atom-window context parallelism.

The primitive CP tests prove each routing operation in isolation.  This probe
keeps the test Torch- and checkpoint-free while exercising the real
conditioning -> atom encoder -> token transformer -> atom decoder composition
with the native JAX parameter-tree contract.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from tests.models.cp_probe_env import inherited_environment

_FEATURES = Path(__file__).parent / "fixtures/1UBQ_A.npz"


def _native_params() -> dict[str, object]:
    """Return a small deterministic tree with the released native schema."""

    import jax.numpy as jnp

    rng = np.random.default_rng(20260823)
    single_in = 6
    single = 8
    pair_in = 4
    pair = 4
    atom = 8
    heads = 2

    def weight(rows: int, cols: int, scale: float = 0.05):
        return jnp.asarray(
            rng.normal(scale=scale, size=(rows, cols)), dtype=jnp.float32
        )

    def bias(size: int, scale: float = 0.01):
        return jnp.asarray(rng.normal(scale=scale, size=(size,)), dtype=jnp.float32)

    def norm(size: int, *, with_bias: bool = True):
        result = {"scale": jnp.ones((size,), dtype=jnp.float32)}
        if with_bias:
            result["bias"] = jnp.zeros((size,), dtype=jnp.float32)
        return result

    def adaln(conditioning_dim: int, activation_dim: int):
        return {
            "s_norm": norm(conditioning_dim, with_bias=False),
            "s_scale": {
                "kernel": weight(conditioning_dim, activation_dim),
                "bias": bias(activation_dim),
            },
            "s_bias": {"kernel": weight(conditioning_dim, activation_dim)},
        }

    def transformer_layer(dim: int, conditioning_dim: int):
        hidden = 2 * dim
        return {
            "adaln": adaln(conditioning_dim, dim),
            "pair_bias_attn": {
                "num_heads": heads,
                "proj_q": {"kernel": weight(dim, dim), "bias": bias(dim)},
                "proj_k": {"kernel": weight(dim, dim)},
                "proj_v": {"kernel": weight(dim, dim)},
                "proj_g": {"kernel": weight(dim, dim)},
                "proj_o": {"kernel": weight(dim, dim)},
            },
            "output_projection": {
                "kernel": weight(conditioning_dim, dim),
                "bias": bias(dim),
            },
            "transition": {
                "adaln": adaln(conditioning_dim, dim),
                "swish_gate": {"kernel": weight(dim, 2 * hidden)},
                "a_to_b": {"kernel": weight(dim, hidden)},
                "b_to_a": {"kernel": weight(hidden, dim)},
                "output_projection": {
                    "kernel": weight(conditioning_dim, dim),
                    "bias": bias(dim),
                },
            },
        }

    def transformer(dim: int, conditioning_dim: int):
        return {"layers": [transformer_layer(dim, conditioning_dim)]}

    def projection():
        return {
            "norm": norm(pair),
            "linear": {"kernel": weight(pair, heads)},
        }

    diffusion_conditioning = {
        "pairwise_conditioner": {
            "dim_pairwise_init_proj": {
                "norm": norm(2 * pair_in),
                "linear": {"kernel": weight(2 * pair_in, pair)},
            },
            "transitions": [],
        },
        "atom_encoder": {
            "embed_atom_features": {
                "kernel": weight(3 + 1 + 128 + 4 * 64, atom, scale=0.02),
                "bias": bias(atom),
            },
            "embed_atompair_ref_pos": {"kernel": weight(3, pair)},
            "embed_atompair_ref_dist": {"kernel": weight(1, pair)},
            "embed_atompair_mask": {"kernel": weight(1, pair)},
            "s_to_c_trans": {
                "norm": norm(single_in),
                "linear": {"kernel": weight(single_in, atom)},
            },
            "z_to_p_trans": {
                "norm": norm(pair),
                "linear": {"kernel": weight(pair, pair)},
            },
            "c_to_p_trans_q": {"kernel": weight(atom, pair)},
            "c_to_p_trans_k": {"kernel": weight(atom, pair)},
            "p_mlp": [{"kernel": weight(pair, pair)} for _ in range(3)],
        },
        "atom_enc_proj_z": [projection()],
        "atom_dec_proj_z": [projection()],
        "token_trans_proj_z": [projection()],
    }

    score_model = {
        "single_conditioner": {
            "norm_single": norm(2 * single_in),
            "single_embed": {
                "kernel": weight(2 * single_in, single),
                "bias": bias(single),
            },
            "fourier_embed": {"proj": {"kernel": weight(1, 4), "bias": bias(4)}},
            "norm_fourier": norm(4),
            "fourier_to_single": {"kernel": weight(4, single)},
            "transitions": [],
        },
        "atom_attention_encoder": {
            "r_to_q_trans": {"kernel": weight(3, atom)},
            "atom_encoder": {"diffusion_transformer": transformer(atom, atom)},
            "atom_to_token_trans": {"kernel": weight(atom, single)},
        },
        "s_to_a_linear": {
            "norm": norm(single),
            "linear": {"kernel": weight(single, single)},
        },
        "token_transformer": transformer(single, single),
        "a_norm": norm(single),
        "atom_attention_decoder": {
            "a_to_q_trans": {"kernel": weight(single, atom)},
            "atom_decoder": {"diffusion_transformer": transformer(atom, atom)},
            "atom_feat_to_atom_pos_update": {
                "norm": norm(atom),
                "linear": {"kernel": weight(atom, 3)},
            },
        },
    }
    return {
        "diffusion_conditioning": diffusion_conditioning,
        "score_model": score_model,
    }


def _model_inputs(*, compact_atom_ownership: bool = False) -> dict[str, object]:
    """Use real featurizer output at CP-aligned atom/window boundaries."""

    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_features_npz
    from foldjax.models.boltz2.data.ownership import compact_atom_to_token_storage

    fixture = load_features_npz(_FEATURES)
    atoms = 256
    tokens = int(fixture["token_pad_mask"].shape[1])
    atom_features = (
        "ref_pos",
        "atom_pad_mask",
        "ref_space_uid",
        "ref_charge",
        "ref_element",
        "ref_atom_name_chars",
        "atom_to_token",
    )
    feats = {name: np.asarray(fixture[name][:, :atoms]) for name in atom_features}
    feats["token_pad_mask"] = np.asarray(fixture["token_pad_mask"])
    if compact_atom_ownership:
        feats = compact_atom_to_token_storage(feats)
    feats = {name: jnp.asarray(value) for name, value in feats.items()}

    def values(shape: tuple[int, ...], start: float, stop: float):
        values_1d = jnp.linspace(
            start,
            stop,
            num=int(np.prod(shape)),
            dtype=jnp.float32,
        )
        return values_1d.reshape(shape)

    return {
        "s_inputs": values((1, tokens, 6), -0.2, 0.2),
        "s_trunk": values((1, tokens, 6), 0.25, -0.25),
        "z_trunk": values((1, tokens, tokens, 4), -0.1, 0.1),
        "relative_position_encoding": values((1, tokens, tokens, 4), 0.15, -0.15),
        "r_noisy": values((1, atoms, 3), 0.12, -0.12),
        "times": jnp.asarray([0.17], dtype=jnp.float32),
        "feats": feats,
    }


def test_trunk_input_embedder_accepts_compact_atom_ownership_exactly() -> None:
    """Execute the real trunk consumer so ownership cannot be dropped there."""

    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.bridge.native import load_features_npz
    from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
        input_embedder_forward,
    )

    dense_inputs = _model_inputs()
    compact_inputs = _model_inputs(compact_atom_ownership=True)
    fixture = load_features_npz(_FEATURES)
    for name in (
        "res_type",
        "profile",
        "deletion_mean",
        "method_feature",
        "modified",
        "cyclic_period",
        "mol_type",
    ):
        value = jnp.asarray(fixture[name])
        dense_inputs["feats"][name] = value
        compact_inputs["feats"][name] = value

    native = _native_params()
    conditioning = native["diffusion_conditioning"]
    score = native["score_model"]
    channels = 8
    params = {
        "atom_encoder": conditioning["atom_encoder"],
        "atom_enc_proj_z": conditioning["atom_enc_proj_z"][0],
        "atom_attention_encoder": score["atom_attention_encoder"],
        "res_type_encoding": {
            "kernel": jnp.zeros((33, channels), dtype=jnp.float32)
        },
        "msa_profile_encoding": {
            "kernel": jnp.zeros((34, channels), dtype=jnp.float32)
        },
        "method_conditioning_init": jnp.zeros((2, channels), dtype=jnp.float32),
        "modified_conditioning_init": jnp.zeros((1, channels), dtype=jnp.float32),
        "cyclic_conditioning_init": {
            "kernel": jnp.zeros((1, channels), dtype=jnp.float32)
        },
        "mol_type_conditioning_init": jnp.zeros((1, channels), dtype=jnp.float32),
    }
    run = jax.jit(input_embedder_forward)

    expected = run(params, dense_inputs["feats"])
    actual = run(params, compact_inputs["feats"])

    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


_MODEL_PROBE = textwrap.dedent(
    r"""
    import jax
    import numpy as np
    import sys
    from jax.sharding import NamedSharding, PartitionSpec

    from foldjax.models._cp import (
        context_parallel,
        pair_spec,
        replicate_tree,
        single_spec,
    )
    from foldjax.models._cp_atom import atom_spec
    from foldjax.models.boltz2.models.diffusion.diffusion import (
        conditioned_diffusion_score_forward,
    )
    from foldjax.models.boltz2.data.ownership import (
        ATOM_TO_TOKEN_INDEX,
        COMPACT_ATOM_TO_TOKEN,
    )
    from tests.models.boltz2.test_atom_cp_model_integration import (
        _model_inputs,
        _native_params,
    )

    assert jax.device_count() == 4
    params = _native_params()
    dense_inputs = _model_inputs()
    inputs = _model_inputs(compact_atom_ownership=True)
    assert "atom_to_token" in dense_inputs["feats"]
    assert "atom_to_token" not in inputs["feats"]
    assert COMPACT_ATOM_TO_TOKEN in inputs["feats"]
    assert ATOM_TO_TOKEN_INDEX in inputs["feats"]
    assert "torch" not in sys.modules

    def model(p, x):
        return conditioned_diffusion_score_forward(
            p,
            s_inputs=x["s_inputs"],
            s_trunk=x["s_trunk"],
            z_trunk=x["z_trunk"],
            relative_position_encoding=x["relative_position_encoding"],
            r_noisy=x["r_noisy"],
            times=x["times"],
            feats=x["feats"],
            token_layers=1,
            multiplicity=1,
            use_scan=True,
            attention_backend="xla",
            atom_context_parallel=True,
        )

    dense_reference = jax.device_get(jax.jit(model)(params, dense_inputs))
    reference = jax.device_get(jax.jit(model)(params, inputs))
    np.testing.assert_array_equal(dense_reference, reference)
    assert reference.shape == (1, 256, 3)
    assert np.isfinite(reference).all()
    assert float(np.std(reference)) > 1e-5

    hlos = {}
    for layout in ("1d", "2d"):
        with context_parallel(4, layout=layout) as mesh:
            placed_params = replicate_tree(params, shard_pair_features=False)
            placed_inputs = dict(inputs)
            placed_inputs["feats"] = replicate_tree(
                inputs["feats"], shard_atom_features=True
            )
            owner_ids = placed_inputs["feats"][ATOM_TO_TOKEN_INDEX]
            owner_marker = placed_inputs["feats"][COMPACT_ATOM_TO_TOKEN]
            owner_shard_atoms = 64 if layout == "1d" else 128
            assert owner_ids.sharding.shard_shape(owner_ids.shape) == (
                1,
                owner_shard_atoms,
            )
            assert owner_marker.sharding.shard_shape(owner_marker.shape) == ()
            placed_inputs["s_inputs"] = jax.device_put(
                inputs["s_inputs"], NamedSharding(mesh, single_spec(3, token_axis=1))
            )
            placed_inputs["s_trunk"] = jax.device_put(
                inputs["s_trunk"], NamedSharding(mesh, single_spec(3, token_axis=1))
            )
            pair_sharding = NamedSharding(
                mesh, pair_spec(4, row_axis=1, col_axis=2)
            )
            placed_inputs["z_trunk"] = jax.device_put(
                inputs["z_trunk"], pair_sharding
            )
            placed_inputs["relative_position_encoding"] = jax.device_put(
                inputs["relative_position_encoding"], pair_sharding
            )
            placed_inputs["r_noisy"] = jax.device_put(
                inputs["r_noisy"], NamedSharding(mesh, atom_spec(3, atom_axis=1))
            )
            placed_inputs["times"] = jax.device_put(
                inputs["times"], NamedSharding(mesh, PartitionSpec())
            )

            compiled = jax.jit(model)
            result = compiled(placed_params, placed_inputs)
            assert result.sharding.shard_shape(result.shape) != result.shape
            hlos[layout] = compiled.lower(
                placed_params, placed_inputs
            ).compiler_ir(dialect="hlo").as_hlo_text().lower()

        actual = jax.device_get(result)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(reference, actual, atol=5e-5, rtol=5e-5)

    one_d_hlo = hlos["1d"]
    assert "collective-permute" in one_d_hlo or "collective_permute" in one_d_hlo
    assert "reduce-scatter" in one_d_hlo or "reduce_scatter" in one_d_hlo
    assert "all-gather" not in one_d_hlo and "all_gather" not in one_d_hlo, one_d_hlo

    two_d_hlo = hlos["2d"]
    assert "collective-permute" in two_d_hlo or "collective_permute" in two_d_hlo
    assert "reduce-scatter" in two_d_hlo or "reduce_scatter" in two_d_hlo
    assert "all-reduce" in two_d_hlo or "all_reduce" in two_d_hlo
    assert "all-gather" not in two_d_hlo and "all_gather" not in two_d_hlo, two_d_hlo
    print("ATOM_MODEL_CP_OK")
    """
)


def test_conditioning_and_score_model_match_serial_under_atom_cp() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _MODEL_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ATOM_MODEL_CP_OK" in completed.stdout
