import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_mapping import (
    map_conditioned_diffusion_model_state_dict,
    map_diffusion_conditioning_state_dict,
)
from foldjax.models.boltz2.models.diffusion.diffusion import (
    conditioned_diffusion_score_forward,
)
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    diffusion_conditioning_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "diffusion_conditioning"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_diffusion_conditioning_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    token_layers = 2
    torch_module = _load_torch_diffusion_conditioning(
        checkpoint_state, token_layers
    )
    params = map_diffusion_conditioning_state_dict(
        checkpoint_state,
        PREFIX,
        num_token_layers=token_layers,
    )
    inputs = _conditioning_inputs()

    with torch.no_grad():
        (
            expected_q,
            expected_c,
            torch_to_keys,
            expected_enc,
            expected_dec,
            expected_tok,
        ) = torch_module(
            inputs["s_trunk"],
            inputs["z_trunk"],
            inputs["relative_position_encoding"],
            inputs["feats"],
        )
    actual = diffusion_conditioning_forward(
        params,
        s_trunk=jnp.asarray(inputs["s_trunk"].numpy()),
        z_trunk=jnp.asarray(inputs["z_trunk"].numpy()),
        relative_position_encoding=jnp.asarray(
            inputs["relative_position_encoding"].numpy()
        ),
        feats=_jax_feats(inputs["feats"]),
        token_layers=token_layers,
    )

    np.testing.assert_allclose(
        np.asarray(actual["q"]),
        expected_q.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["c"]),
        expected_c.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["atom_enc_bias"]),
        expected_enc.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["atom_dec_bias"]),
        expected_dec.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual["token_trans_bias"]),
        expected_tok.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    assert callable(torch_to_keys)


def test_checkpoint_conditioning_plus_score_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    token_layers = 2
    torch_conditioning = _load_torch_diffusion_conditioning(
        checkpoint_state, token_layers
    )
    torch_score = _load_torch_score_model(checkpoint_state, token_layers)
    params = map_conditioned_diffusion_model_state_dict(
        checkpoint_state,
        num_token_layers=token_layers,
        token_transformer_heads=16,
    )
    inputs = _conditioning_inputs()
    inputs["feats"]["token_pad_mask"] = torch.ones(1, 8, dtype=torch.float32)
    inputs["feats"]["token_pad_mask"][:, -1:] = 0.0
    s_inputs = torch.linspace(0.2, -0.2, steps=8 * 384).reshape(1, 8, 384)
    r_noisy = torch.linspace(0.15, -0.15, steps=64 * 3).reshape(1, 64, 3)
    times = torch.tensor([0.17], dtype=torch.float32)

    with torch.no_grad():
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            torch_conditioning(
                inputs["s_trunk"],
                inputs["z_trunk"],
                inputs["relative_position_encoding"],
                inputs["feats"],
            )
        )
        expected = torch_score(
            s_inputs=s_inputs,
            s_trunk=inputs["s_trunk"],
            r_noisy=r_noisy,
            times=times,
            feats=inputs["feats"],
            diffusion_conditioning={
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            },
            multiplicity=1,
        )
    actual = conditioned_diffusion_score_forward(
        params,
        s_inputs=jnp.asarray(s_inputs.numpy()),
        s_trunk=jnp.asarray(inputs["s_trunk"].numpy()),
        z_trunk=jnp.asarray(inputs["z_trunk"].numpy()),
        relative_position_encoding=jnp.asarray(
            inputs["relative_position_encoding"].numpy()
        ),
        r_noisy=jnp.asarray(r_noisy.numpy()),
        times=jnp.asarray(times.numpy()),
        feats=_jax_feats(inputs["feats"]),
        token_layers=token_layers,
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_diffusion_conditioning(
    state: dict[str, torch.Tensor],
    token_layers: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning

    module = DiffusionConditioning(
        token_s=384,
        token_z=128,
        atom_s=128,
        atom_z=16,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=token_layers,
        token_transformer_heads=16,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=388,
    ).eval()
    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local_key = key.removeprefix(f"{PREFIX}.")
        if local_key.startswith("token_trans_proj_z."):
            index = int(local_key.split(".")[1])
            if index >= token_layers:
                continue
        module_state[local_key] = value
    module.load_state_dict(module_state)
    return module


def _load_torch_score_model(
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
            self.single_conditioner = SingleConditioning(16, 384, 256, 2)
            self.atom_attention_encoder = AtomAttentionEncoder(128, 384, 32, 128)
            self.s_to_a_linear = torch.nn.Sequential(
                torch.nn.LayerNorm(768), LinearNoBias(768, 768)
            )
            self.token_transformer = DiffusionTransformer(
                dim=768,
                dim_single_cond=768,
                depth=token_layers,
                heads=16,
            )
            self.a_norm = torch.nn.LayerNorm(768)
            self.atom_attention_decoder = AtomAttentionDecoder(128, 384, 32, 128)

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
        if not key.startswith("structure_module.score_model."):
            continue
        local_key = key.removeprefix("structure_module.score_model.")
        if local_key.startswith("token_transformer.layers."):
            index = int(local_key.split(".")[2])
            if index >= token_layers:
                continue
        module_state[local_key] = value
    module.load_state_dict(module_state)
    return module


def _conditioning_inputs() -> dict[str, object]:
    atoms = 64
    tokens = 8
    atom_to_token = torch.zeros(1, atoms, tokens)
    atom_to_token[0, torch.arange(atoms), torch.arange(atoms) % tokens] = 1.0
    ref_element = torch.zeros(1, atoms, 128)
    ref_element[0, torch.arange(atoms), torch.arange(atoms) % 128] = 1.0
    chars = torch.zeros(1, atoms, 4, 64)
    chars[0, torch.arange(atoms), 0, torch.arange(atoms) % 64] = 1.0
    chars[0, torch.arange(atoms), 1, (torch.arange(atoms) + 1) % 64] = 1.0
    chars[0, torch.arange(atoms), 2, (torch.arange(atoms) + 2) % 64] = 1.0
    chars[0, torch.arange(atoms), 3, (torch.arange(atoms) + 3) % 64] = 1.0
    feats = {
        "ref_pos": torch.linspace(-0.3, 0.3, steps=atoms * 3).reshape(
            1, atoms, 3
        ),
        "atom_pad_mask": torch.ones(1, atoms, dtype=torch.float32),
        "ref_space_uid": (torch.arange(atoms) // 8).reshape(1, atoms),
        "ref_charge": torch.linspace(-0.5, 0.5, steps=atoms).reshape(1, atoms),
        "ref_element": ref_element,
        "ref_atom_name_chars": chars,
        "atom_to_token": atom_to_token,
    }
    feats["atom_pad_mask"][:, -3:] = 0.0
    return {
        "s_trunk": torch.linspace(-0.2, 0.2, steps=tokens * 384).reshape(
            1, tokens, 384
        ),
        "z_trunk": torch.linspace(-0.1, 0.1, steps=tokens * tokens * 128).reshape(
            1, tokens, tokens, 128
        ),
        "relative_position_encoding": torch.linspace(
            0.15, -0.15, steps=tokens * tokens * 128
        ).reshape(1, tokens, tokens, 128),
        "feats": feats,
    }


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
