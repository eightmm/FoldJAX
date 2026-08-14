"""End-to-end inference on synthetic parameters.

No real weights are involved, so this cannot claim scientific correctness. What
it does establish is that the full path runs: trunk -> noise conditioning ->
denoiser -> sampler -> confidence heads, with shapes that agree and finite
outputs. Every stage's numerics are gated against upstream elsewhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.inference import InferenceConfig, InferenceParams, predict

pytestmark = pytest.mark.torch_parity

C_ATOM, C_ATOM_PAIR, C_TOKEN = 8, 6, 8
C_S, C_Z = 8, 4
C_ELEMENT, C_CHARS = 12, 5
C_RESTYPE, C_PROFILE, C_MSA = 32, 32, 32
C_S_INPUT = C_TOKEN + C_RESTYPE + C_PROFILE + 1
C_FOURIER = 16
MAX_RELATIVE_IDX = 4
MAX_RELATIVE_CHAIN = 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN, N_MSA = sum(COUNTS), len(COUNTS), 3
MAX_ATOMS_PER_TOKEN = 5
PLDDT_BINS, PAE_BINS = 50, 8
# The confidence heads' distance one-hot; small here, 39 in the released config.
CONF_BINS = 6


def _config() -> InferenceConfig:
    return InferenceConfig(
        n_token=N_TOKEN,
        n_atom=N_ATOM,
        n_query=4,
        n_key=8,
        atom_heads=2,
        token_heads=2,
        no_heads_msa=2,
        no_heads_pair=2,
        no_heads_pair_bias=2,
        max_relative_idx=MAX_RELATIVE_IDX,
        max_relative_chain=MAX_RELATIVE_CHAIN,
        num_cycles=1,
        num_samples=2,
        max_atoms_per_token=MAX_ATOMS_PER_TOKEN,
        plddt_bins=PLDDT_BINS,
        pae_bins=PAE_BINS,
        pae_bin_max=32.0,
        confidence_no_bin=CONF_BINS,
        no_rollout_steps=3,
    )


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _params(torch, randomized) -> InferenceParams:
    from ml_collections import ConfigDict
    from openfold3.core.model.feature_embedders.input_embedders import (
        FourierEmbedding,
        InputEmbedderAllAtom,
        MSAModuleEmbedder,
    )
    from openfold3.core.model.heads.prediction_heads import (
        DistogramHead,
        ExperimentallyResolvedHeadAllAtom,
        PairformerEmbedding,
        PerResidueLDDTAllAtom,
        PredictedAlignedErrorHead,
        PredictedDistanceErrorHead,
    )
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    from openfold3.core.model.latent.pairformer import PairFormerStack
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        AtomAttentionDecoder,
        AtomAttentionEncoder,
    )
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.primitives import LayerNorm, Linear

    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_atom_attention_decoder,
        map_atom_attention_encoder,
        map_atom_logit_head,
        map_diffusion_conditioning,
        map_diffusion_transformer,
        map_input_embedder,
        map_layer_norm,
        map_linear,
        map_msa_embedder,
        map_msa_module_stack,
        map_pair_head,
        map_pairformer_embedding,
        map_pairformer_stack,
    )
    from foldjax.models.openfold3.models.denoiser import DenoiserParams
    from foldjax.models.openfold3.models.trunk import TrunkParams

    def enc_cfg(noisy: bool) -> dict:
        cfg = {
            "c_atom_ref": ConfigDict(
                {"element": C_ELEMENT, "name_chars": 4 * C_CHARS}
            ),
            "c_atom": C_ATOM,
            "c_atom_pair": C_ATOM_PAIR,
            "c_token": C_TOKEN,
            "c_hidden": 4,
            "no_heads": 2,
            "no_blocks": 1,
            "n_transition": 2,
            "n_query": 4,
            "n_key": 8,
            "use_ada_layer_norm": noisy,
        }
        if noisy:
            cfg |= {"add_noisy_pos": True, "c_s": C_S, "c_z": C_Z}
        return cfg

    embedder = randomized(
        InputEmbedderAllAtom(
            c_s_input=C_S_INPUT, c_s=C_S, c_z=C_Z,
            max_relative_idx=MAX_RELATIVE_IDX,
            max_relative_chain=MAX_RELATIVE_CHAIN, atom_attn_enc=enc_cfg(noisy=False),
        )
    )
    msa_emb = randomized(
        MSAModuleEmbedder(
            c_m_feats=C_MSA + 2, c_m=C_ATOM, c_s_input=C_S_INPUT,
            subsample_main_msa=False, subsample_all_msa=False,
            min_subsampled_all_msa=1, max_subsampled_all_msa=N_MSA,
        )
    )
    msa_stack = randomized(
        MSAModuleStack(
            c_m=C_ATOM, c_z=C_Z, c_hidden_msa_att=4, c_hidden_opm=4,
            c_hidden_mul=6, c_hidden_pair_att=4, no_heads_msa=2, no_heads_pair=2,
            no_blocks=1, transition_type="swiglu", transition_n=2,
            msa_dropout=0.0, pair_dropout=0.0, opm_first=True,
            fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9, eps=1e-3,
        )
    )
    pf = randomized(
        PairFormerStack(
            c_s=C_S, c_z=C_Z, c_hidden_pair_bias=4, no_heads_pair_bias=2,
            c_hidden_mul=6, c_hidden_pair_att=4, no_heads_pair=2, no_blocks=1,
            transition_type="swiglu", transition_n=2, pair_dropout=0.0,
            fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9,
        )
    )
    trunk_params = TrunkParams(
        input_embedder=map_input_embedder(dict(embedder.state_dict())),
        msa_module_embedder=map_msa_embedder(dict(msa_emb.state_dict())),
        msa_module=map_msa_module_stack(dict(msa_stack.state_dict())),
        pairformer_stack=map_pairformer_stack(dict(pf.state_dict())),
        layer_norm_z=map_layer_norm(dict(randomized(LayerNorm(C_Z)).state_dict())),
        linear_z=map_linear(
            dict(randomized(Linear(C_Z, C_Z, bias=False)).state_dict()), bias=False
        ),
        layer_norm_s=map_layer_norm(dict(randomized(LayerNorm(C_S)).state_dict())),
        linear_s=map_linear(
            dict(randomized(Linear(C_S, C_S, bias=False)).state_dict()), bias=False
        ),
    )

    # The pair conditioning path concatenates the relpos encoding onto z, so its
    # layer norm and projection span num_relpos_dims + C_Z. The bin counts follow
    # upstream: 2*clip+2 for each of residue/token/chain, plus one same-entity
    # feature.
    num_relpos_dims = (
        (2 * MAX_RELATIVE_IDX + 2) * 2 + (2 * MAX_RELATIVE_CHAIN + 2) + 1
    )
    cond_state = {}
    for name, module in (
        ("layer_norm_s", randomized(LayerNorm(C_S + C_S_INPUT, create_offset=False))),
        ("linear_s", randomized(Linear(C_S + C_S_INPUT, C_S, bias=False))),
        ("fourier_emb", FourierEmbedding(c=C_FOURIER, seed=42)),
        ("layer_norm_n", randomized(LayerNorm(C_FOURIER, create_offset=False))),
        ("linear_n", randomized(Linear(C_FOURIER, C_S, bias=False))),
        (
            "layer_norm_z",
            randomized(LayerNorm(num_relpos_dims + C_Z, create_offset=False)),
        ),
        ("linear_z", randomized(Linear(num_relpos_dims + C_Z, C_Z, bias=False))),
        ("transition_z.0", randomized(SwiGLUTransition(c_in=C_Z, n=2))),
        ("transition_z.1", randomized(SwiGLUTransition(c_in=C_Z, n=2))),
    ):
        for key, value in module.state_dict().items():
            cond_state[f"{name}.{key}"] = value

    denoiser = DenoiserParams(
        atom_attn_enc=map_atom_attention_encoder(
            dict(randomized(AtomAttentionEncoder(**enc_cfg(noisy=True))).state_dict())
        ),
        layer_norm_s=map_layer_norm(
            dict(randomized(LayerNorm(C_S)).state_dict())
        ),
        linear_s=map_linear(
            dict(randomized(Linear(C_S, C_TOKEN, bias=False)).state_dict()),
            bias=False,
        ),
        diffusion_transformer=map_diffusion_transformer(
            dict(
                randomized(
                    DiffusionTransformer(
                        c_a=C_TOKEN, c_s=C_S, c_z=C_Z, c_hidden=4, no_heads=2,
                        no_blocks=1, n_transition=2, use_ada_layer_norm=True,
                        n_query=None, n_key=None, inf=1e9,
                    )
                ).state_dict()
            )
        ),
        layer_norm_a=map_layer_norm(dict(randomized(LayerNorm(C_TOKEN)).state_dict())),
        atom_attn_dec=map_atom_attention_decoder(
            dict(
                randomized(
                    AtomAttentionDecoder(
                        c_token=C_TOKEN, c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR,
                        c_hidden=4, no_heads=2, no_blocks=1, n_transition=2,
                        n_query=4, n_key=8, use_ada_layer_norm=True,
                    )
                ).state_dict()
            )
        ),
    )

    # The confidence heads read representations re-embedded with the predicted
    # geometry, so this group is required, not optional.
    pairformer_embedding_module = randomized(
        PairformerEmbedding(
            pairformer=ConfigDict(
                {
                    "c_s": C_S,
                    "c_z": C_Z,
                    "c_hidden_mul": 4,
                    "c_hidden_pair_att": 4,
                    "c_hidden_pair_bias": 4,
                    "no_heads_pair": 2,
                    "no_heads_pair_bias": 2,
                    "no_blocks": 1,
                    "transition_n": 2,
                    "pair_dropout": 0.0,
                    "inf": 1e9,
                    "transition_type": "swiglu",
                    "fuse_projection_weights": False,
                    "blocks_per_ckpt": None,
                }
            ),
            c_s_input=C_S_INPUT,
            c_z=C_Z,
            min_bin=3.25,
            max_bin=50.75,
            no_bin=CONF_BINS,
            inf=1e9,
        )
    )

    return InferenceParams(
        trunk=trunk_params,
        diffusion_conditioning=map_diffusion_conditioning(cond_state),
        denoiser=denoiser,
        plddt_head=map_atom_logit_head(
            dict(
                randomized(
                    PerResidueLDDTAllAtom(
                        c_s=C_S, c_out=PLDDT_BINS,
                        max_atoms_per_token=MAX_ATOMS_PER_TOKEN,
                    )
                ).state_dict()
            )
        ),
        pae_head=map_pair_head(
            dict(
                randomized(
                    PredictedAlignedErrorHead(c_z=C_Z, c_out=PAE_BINS)
                ).state_dict()
            )
        ),
        pde_head=map_pair_head(
            dict(
                randomized(
                    PredictedDistanceErrorHead(c_z=C_Z, c_out=PAE_BINS)
                ).state_dict()
            )
        ),
        distogram_head=map_pair_head(
            dict(randomized(DistogramHead(c_z=C_Z, c_out=PAE_BINS)).state_dict()),
            layer_norm=False,
        ),
        pairformer_embedding=map_pairformer_embedding(
            dict(pairformer_embedding_module.state_dict())
        ),
        experimentally_resolved_head=map_atom_logit_head(
            dict(
                randomized(
                    ExperimentallyResolvedHeadAllAtom(
                        c_s=C_S,
                        c_out=2,
                        max_atoms_per_token=MAX_ATOMS_PER_TOKEN,
                    )
                ).state_dict()
            )
        ),
    )


def _representative_atoms():
    """A table shaped for this test's restype width; all lookups unused."""
    import numpy as np

    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    zeros = np.zeros(C_RESTYPE, dtype=np.float32)
    return RepresentativeAtomTable(*([zeros] * len(RepresentativeAtomTable._fields)))


def _batch(torch) -> dict:
    atom_to_token = torch.cat(
        [torch.full((c,), i, dtype=torch.long) for i, c in enumerate(COUNTS)]
    )
    atom_slots = torch.zeros(1, N_TOKEN, MAX_ATOMS_PER_TOKEN)
    for index, count in enumerate(COUNTS):
        atom_slots[:, index, :count] = 1.0
    raw = {
        "ref_pos": torch.randn(1, N_ATOM, 3),
        "ref_charge": torch.randn(1, N_ATOM),
        "ref_mask": torch.ones(1, N_ATOM),
        "ref_element": torch.rand(1, N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": torch.rand(1, N_ATOM, 4, C_CHARS),
        "ref_space_uid": (torch.arange(N_ATOM).float() // 4).reshape(1, N_ATOM),
        "atom_mask": torch.ones(1, N_ATOM),
        "atom_to_token_index": atom_to_token.reshape(1, N_ATOM),
        "token_mask": torch.ones(1, N_TOKEN),
        "num_atoms_per_token": torch.tensor([COUNTS], dtype=torch.float32),
        "restype": torch.rand(1, N_TOKEN, C_RESTYPE),
        "profile": torch.rand(1, N_TOKEN, C_PROFILE),
        "deletion_mean": torch.rand(1, N_TOKEN),
        "token_bonds": torch.zeros(1, N_TOKEN, N_TOKEN),
        "residue_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "asym_id": torch.tensor([[0, 0, 1]]),
        "entity_id": torch.tensor([[0, 0, 0]]),
        "token_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "sym_id": torch.tensor([[0, 0, 1]]),
        "msa": torch.rand(1, N_MSA, N_TOKEN, C_MSA),
        "has_deletion": torch.zeros(1, N_MSA, N_TOKEN),
        "deletion_value": torch.rand(1, N_MSA, N_TOKEN),
        "msa_mask": torch.ones(1, N_MSA, N_TOKEN),
        # Everything is atomized here, so the representative atom of each token is
        # its first atom and no chemistry column is consulted. The chemistry
        # itself is gated in test_torch_parity_representative_atoms.
        "is_atomized": torch.ones(1, N_TOKEN),
        "is_protein": torch.zeros(1, N_TOKEN),
        "is_dna": torch.zeros(1, N_TOKEN),
        "is_rna": torch.zeros(1, N_TOKEN),
        "start_atom_index": torch.tensor(
            [[sum(COUNTS[:i]) for i in range(N_TOKEN)]], dtype=torch.float32
        ),
        "max_atom_per_token_mask": atom_slots.reshape(
            1, N_TOKEN * MAX_ATOMS_PER_TOKEN
        ),
    }
    return {k: jnp.asarray(v.numpy()) for k, v in raw.items()}


def test_full_inference_path_runs(openfold3_source, randomized, monkeypatch) -> None:
    import foldjax.models.openfold3.inference as inference

    torch = _torch()
    config = _config()
    batch = _batch(torch)
    # An archive mask describes reference geometry, not each predicted sample.
    # Keep it present to prove inference still derives frames from coordinates.
    batch["has_frame"] = jnp.zeros((1, N_TOKEN), dtype=jnp.float32)
    calls = []
    original_token_frame_atoms = inference.token_frame_atoms

    def tracked_token_frame_atoms(*args, **kwargs):
        calls.append(True)
        return original_token_frame_atoms(*args, **kwargs)

    monkeypatch.setattr(inference, "token_frame_atoms", tracked_token_frame_atoms)
    out = predict(
        jax.random.key(0),
        batch,
        _params(torch, randomized),
        config,
        _representative_atoms(),
    )
    assert out.coordinates.shape == (config.num_samples, N_ATOM, 3)
    # Confidence is per sample: the heads read representations re-embedded with
    # each sample's predicted geometry, so these carry the sample axis. Only the
    # distogram, which reads the trunk pair embedding, does not.
    assert out.plddt.shape == (config.num_samples, N_ATOM)
    assert out.ptm.shape == (config.num_samples,)
    assert out.pae_logits.shape == (config.num_samples, N_TOKEN, N_TOKEN, PAE_BINS)
    assert out.pde_logits.shape == (config.num_samples, N_TOKEN, N_TOKEN, PAE_BINS)
    assert out.distogram_logits.shape == (1, N_TOKEN, N_TOKEN, PAE_BINS)
    assert out.experimentally_resolved_logits.shape == (config.num_samples, N_ATOM, 2)
    assert out.iptm.shape == (config.num_samples,)
    assert out.chain_pair_iptm is None
    assert calls == [True]
    for name, array in (
        ("coordinates", out.coordinates),
        ("plddt", out.plddt),
        ("ptm", out.ptm),
        ("iptm", out.iptm),
        ("pae", out.pae_logits),
        ("pde", out.pde_logits),
        ("distogram", out.distogram_logits),
        ("experimentally_resolved", out.experimentally_resolved_logits),
    ):
        assert np.isfinite(np.asarray(array)).all(), name


def test_iptm_outputs_appear_when_chains_are_declared(
    openfold3_source, randomized
) -> None:
    torch = _torch()
    out = predict(
        jax.random.key(0),
        _batch(torch),
        _params(torch, randomized),
        _config(),
        _representative_atoms(),
        n_chain=2,
    )
    assert out.iptm is not None
    assert out.chain_pair_iptm.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(out.chain_pair_iptm)).all()


def test_monomer_iptm_is_zero_and_has_no_chain_pair_matrix(
    openfold3_source, randomized
) -> None:
    torch = _torch()
    batch = _batch(torch)
    batch["asym_id"] = jnp.zeros_like(batch["asym_id"])
    out = predict(
        jax.random.key(0),
        batch,
        _params(torch, randomized),
        _config(),
        _representative_atoms(),
        n_chain=1,
    )
    np.testing.assert_allclose(np.asarray(out.iptm), 0.0, atol=1e-7)
    assert out.chain_pair_iptm is None


def test_prediction_is_deterministic_for_a_fixed_key(
    openfold3_source, randomized
) -> None:
    torch = _torch()
    batch, params, config = _batch(torch), _params(torch, randomized), _config()
    table = _representative_atoms()
    first = predict(jax.random.key(5), batch, params, config, table)
    again = predict(jax.random.key(5), batch, params, config, table)
    other = predict(jax.random.key(6), batch, params, config, table)
    np.testing.assert_allclose(
        np.asarray(first.coordinates), np.asarray(again.coordinates)
    )
    assert not np.allclose(
        np.asarray(first.coordinates), np.asarray(other.coordinates)
    )
    # Confidence is re-embedded from the sampled coordinates, so unlike the
    # trunk outputs it does depend on the sampler key.
    np.testing.assert_allclose(
        np.asarray(first.pae_logits), np.asarray(other.pae_logits)
    )
