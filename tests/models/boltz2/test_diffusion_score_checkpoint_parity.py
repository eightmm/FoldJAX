import sys
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_mapping import (
    map_diffusion_score_model_state_dict,
)
from foldjax.models.boltz2.models.diffusion.atom import (
    get_indexing_matrix,
    single_to_keys,
)
from foldjax.models.boltz2.models.diffusion.diffusion import (
    diffusion_score_model_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "structure_module.score_model"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_diffusion_score_model_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    token_layers = 2
    torch_module = _load_torch_diffusion_module(checkpoint_state, token_layers)
    params = map_diffusion_score_model_state_dict(
        checkpoint_state,
        PREFIX,
        num_token_layers=token_layers,
        token_transformer_heads=16,
    )
    inputs = _score_inputs(token_layers)
    torch_to_keys_fn, jax_to_keys_fn = _to_keys_fns()
    torch_conditioning = {
        "q": inputs["q"],
        "c": inputs["c"],
        "to_keys": torch_to_keys_fn,
        "atom_enc_bias": inputs["atom_enc_bias"],
        "atom_dec_bias": inputs["atom_dec_bias"],
        "token_trans_bias": inputs["token_trans_bias"],
    }
    jax_conditioning = {
        "q": jnp.asarray(inputs["q"].numpy()),
        "c": jnp.asarray(inputs["c"].numpy()),
        "to_keys": jax_to_keys_fn,
        "atom_enc_bias": jnp.asarray(inputs["atom_enc_bias"].numpy()),
        "atom_dec_bias": jnp.asarray(inputs["atom_dec_bias"].numpy()),
        "token_trans_bias": jnp.asarray(inputs["token_trans_bias"].numpy()),
    }

    with torch.no_grad():
        expected = torch_module(
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            r_noisy=inputs["r_noisy"],
            times=inputs["times"],
            feats=inputs["feats"],
            diffusion_conditioning=torch_conditioning,
            multiplicity=1,
        )
    actual = diffusion_score_model_forward(
        params,
        s_inputs=jnp.asarray(inputs["s_inputs"].numpy()),
        s_trunk=jnp.asarray(inputs["s_trunk"].numpy()),
        r_noisy=jnp.asarray(inputs["r_noisy"].numpy()),
        times=jnp.asarray(inputs["times"].numpy()),
        feats=_jax_feats(inputs["feats"]),
        diffusion_conditioning=jax_conditioning,
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_diffusion_module(
    state: dict[str, torch.Tensor],
    token_layers: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        AtomAttentionDecoder,
        AtomAttentionEncoder,
        SingleConditioning,
    )
    from boltz.model.modules.transformersv2 import DiffusionTransformer
    from boltz.model.modules.utils import LinearNoBias

    class ScoreModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.single_conditioner = SingleConditioning(
                sigma_data=16,
                token_s=384,
                dim_fourier=256,
                num_transitions=2,
            )
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=128,
                token_s=384,
                atoms_per_window_queries=32,
                atoms_per_window_keys=128,
                atom_encoder_depth=3,
                atom_encoder_heads=4,
                structure_prediction=True,
            )
            self.s_to_a_linear = torch.nn.Sequential(
                torch.nn.LayerNorm(768),
                LinearNoBias(768, 768),
            )
            self.token_transformer = DiffusionTransformer(
                dim=768,
                dim_single_cond=768,
                depth=token_layers,
                heads=16,
            )
            self.a_norm = torch.nn.LayerNorm(768)
            self.atom_attention_decoder = AtomAttentionDecoder(
                atom_s=128,
                token_s=384,
                attn_window_queries=32,
                attn_window_keys=128,
                atom_decoder_depth=3,
                atom_decoder_heads=4,
            )

        def forward(
            self,
            s_inputs,
            s_trunk,
            r_noisy,
            times,
            feats,
            diffusion_conditioning,
            multiplicity=1,
        ):
            s, _ = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
            a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
                feats=feats,
                q=diffusion_conditioning["q"].float(),
                c=diffusion_conditioning["c"].float(),
                atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
                to_keys=diffusion_conditioning["to_keys"],
                r=r_noisy,
                multiplicity=multiplicity,
            )
            a = a + self.s_to_a_linear(s)
            mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            a = self.token_transformer(
                a,
                mask=mask.float(),
                s=s,
                bias=diffusion_conditioning["token_trans_bias"].float(),
                multiplicity=multiplicity,
            )
            a = self.a_norm(a)
            return self.atom_attention_decoder(
                a=a,
                q=q_skip,
                c=c_skip,
                atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
                feats=feats,
                multiplicity=multiplicity,
                to_keys=to_keys,
            )

    module = ScoreModel().eval()
    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local_key = key.removeprefix(f"{PREFIX}.")
        if local_key.startswith("token_transformer.layers."):
            index = int(local_key.split(".")[2])
            if index >= token_layers:
                continue
        module_state[local_key] = value
    module.load_state_dict(module_state)
    return module


def _to_keys_fns():
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    torch_keys = torch_index(K=2, W=32, H=128, device="cpu")
    torch_to_keys_fn = partial(torch_to_keys, indexing_matrix=torch_keys, W=32, H=128)
    jax_keys = get_indexing_matrix(k=2, w=32, h_keys=128)
    return torch_to_keys_fn, lambda x: single_to_keys(x, jax_keys, w=32, h_keys=128)


def _score_inputs(token_layers: int) -> dict[str, object]:
    atoms = 64
    tokens = 8
    atom_to_token = torch.zeros(1, atoms, tokens)
    atom_to_token[0, torch.arange(atoms), torch.arange(atoms) % tokens] = 1.0
    feats = {
        "ref_pos": torch.linspace(-0.3, 0.3, steps=atoms * 3).reshape(1, atoms, 3),
        "atom_pad_mask": torch.ones(1, atoms, dtype=torch.float32),
        "atom_to_token": atom_to_token,
        "token_pad_mask": torch.ones(1, tokens, dtype=torch.float32),
    }
    feats["atom_pad_mask"][:, -3:] = 0.0
    feats["token_pad_mask"][:, -1:] = 0.0

    return {
        "s_inputs": torch.linspace(-0.2, 0.2, steps=tokens * 384).reshape(
            1, tokens, 384
        ),
        "s_trunk": torch.linspace(0.25, -0.25, steps=tokens * 384).reshape(
            1, tokens, 384
        ),
        "r_noisy": torch.linspace(0.15, -0.15, steps=atoms * 3).reshape(1, atoms, 3),
        "times": torch.tensor([0.17], dtype=torch.float32),
        "feats": feats,
        "q": torch.linspace(-0.2, 0.2, steps=atoms * 128).reshape(1, atoms, 128),
        "c": torch.linspace(0.25, -0.25, steps=atoms * 128).reshape(1, atoms, 128),
        "atom_enc_bias": torch.linspace(-0.1, 0.1, steps=2 * 32 * 128 * 12).reshape(
            1, 2, 32, 128, 12
        ),
        "atom_dec_bias": torch.linspace(0.1, -0.1, steps=2 * 32 * 128 * 12).reshape(
            1, 2, 32, 128, 12
        ),
        "token_trans_bias": torch.linspace(
            -0.05, 0.05, steps=tokens * tokens * token_layers * 16
        ).reshape(1, tokens, tokens, token_layers * 16),
    }


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
