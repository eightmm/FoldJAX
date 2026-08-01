"""Torch-vs-JAX parity gate for the full feature_embedding port.

Two gates per the established pattern:
  (a) real-component gate: run the REAL exported feature_embedding.pt
      forward_256 with synthetic features and assert the JAX embed_features dict
      matches the component's bf16 output (<= ~1 bf16 ULP per modality);
  (b) precision-matched fp32 gate: JAX fp32 cat+linear vs torch fp32 (< 1e-4).

The component computes in bfloat16; cross-backend bf16 matmul reduction order
differs, so the real-component gate is at bf16 ULP, and exact-numeric parity is
asserted in fp32 (PROJECT.md precision rule).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.bridge.component_io import (  # noqa: E402
    load_component_state_dict,
)
from foldjax.models.chai.models import feature_embedding as fe  # noqa: E402
from foldjax.models.chai.models.primitives import linear  # noqa: E402

_MODALITIES = ("ATOM", "ATOM_PAIR", "TOKEN", "TOKEN_PAIR", "MSA", "TEMPLATES")
_BF16_ULP = 0.0625  # one bf16 ULP at this output magnitude


def _build_inputs(b=1, n=256, msa=4, t=2, seed=7) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def ri(hi, *s):
        return torch.randint(0, hi, s, generator=g)

    def rb(*s):  # random binary float
        return torch.randint(0, 2, s, generator=g).float()

    def rest(hi, *s):  # restraint: mix of -1 (masked) and real distances
        masked = torch.rand(*s, generator=g) < 0.5
        real = torch.rand(*s, generator=g) * hi
        return torch.where(masked, -torch.ones(*s), real)

    return dict(
        TemplateDistogram=ri(39, b, t, n, n),
        TemplateMask=rb(b, t, n, n, 2),
        TemplateResType=ri(33, b, t, n, 1),
        TemplateUnitVector=torch.randn(b, t, n, n, 3, generator=g),
        ChainIsCropped=rb(b, n),
        ESMEmbeddings=torch.randn(b, n, 2560, generator=g),
        IsDistillation=ri(2, b, n),
        MSADeletionMean=torch.randn(b, n, generator=g),
        MSAProfile=torch.randn(b, n, 33, generator=g),
        MissingChainContact=rb(b, n),
        ResidueType=ri(33, b, n),
        TokenBFactor=ri(3, b, n),
        TokenPLDDT=ri(4, b, n),
        BlockedAtomPairDistogram=ri(12, b, n, 1, 1),
        InverseSquaredBlockedAtomPairDistances=torch.randn(b, n, 1, 1, 2, generator=g),
        IsPairedMSA=rb(b, msa, n),
        MSADataSource=ri(6, b, msa, n),
        MSADeletionValue=torch.randn(b, msa, n, generator=g),
        MSAHasDeletion=rb(b, msa, n),
        MSAOneHot=ri(33, b, msa, n),
        DockingConstraintGenerator=ri(6, b, n, n),
        RelativeChain=ri(6, b, n, n),
        RelativeEntity=ri(3, b, n, n),
        RelativeSequenceSeparation=ri(67, b, n, n),
        RelativeTokenSeparation=ri(67, b, n, n),
        TokenDistanceRestraint=rest(30.0, b, n, n, 1),
        TokenPairPocketRestraint=rest(20.0, b, n, n, 1),
        AtomNameOneHot=ri(65, b, n, 4),
        AtomRefCharge=torch.randn(b, n, generator=g),
        AtomRefElement=ri(130, b, n),
        AtomRefMask=rb(b, n),
        AtomRefPos=torch.randn(b, n, 3, generator=g),
    )


def test_feature_embedding_full_real_component_bf16(official_asset_path) -> None:
    path = official_asset_path("feature_embedding.pt")
    module = torch.jit.load(str(path), map_location="cpu").eval()
    state = load_component_state_dict(path)

    inp = _build_inputs()
    with torch.no_grad():
        ref = module.forward_256(**inp)

    feats = {k: jnp.asarray(v.numpy()) for k, v in inp.items()}
    out = fe.embed_features(feats, state)

    for k in _MODALITIES:
        o = np.asarray(out[k].astype(jnp.float32))
        r = ref[k].float().numpy()
        assert o.shape == r.shape, f"{k}: shape {o.shape} != {r.shape}"
        diff = float(np.max(np.abs(o - r)))
        assert diff <= _BF16_ULP, f"{k}: bf16 diff {diff} > {_BF16_ULP}"


_CATS = {
    "ATOM": fe._atom_cat,
    "ATOM_PAIR": fe._atom_pair_cat,
    "TOKEN": fe._token_cat,
    "TOKEN_PAIR": fe._token_pair_cat,
    "MSA": fe._msa_cat,
}


def test_feature_embedding_full_fp32_parity(official_asset_path) -> None:
    path = official_asset_path("feature_embedding.pt")
    state = load_component_state_dict(path)
    inp = _build_inputs(seed=11)
    feats = {k: jnp.asarray(v.numpy()) for k, v in inp.items()}

    restype_w = jnp.asarray(
        state["feature_embeddings.TEMPLATES.TemplateResType.embedding.weight"]
    )

    for k in _MODALITIES:
        if k == "TEMPLATES":
            xj = fe._templates_cat(feats, restype_w)
        else:
            xj = _CATS[k](feats)
        w_np = state[f"input_projs.{k}.0.weight"]
        b_np = state[f"input_projs.{k}.0.bias"]

        out = np.asarray(linear(xj, jnp.asarray(w_np), jnp.asarray(b_np)))
        ref = torch.nn.functional.linear(
            torch.from_numpy(np.asarray(xj)),
            torch.from_numpy(w_np),
            torch.from_numpy(b_np),
        ).numpy()
        assert out.shape == ref.shape
        diff = float(np.max(np.abs(out - ref)))
        assert diff < 1e-4, f"{k}: fp32 diff {diff} >= 1e-4"


def test_combined_embedding_msa_crop_matches_full_slice_exactly(
    official_asset_path,
) -> None:
    state = load_component_state_dict(official_asset_path("feature_embedding.pt"))
    inp = _build_inputs(n=4, msa=8, seed=12)
    features = {key: jnp.asarray(value.numpy()) for key, value in inp.items()}
    msa_names = {
        "IsPairedMSA",
        "MSADataSource",
        "MSADeletionValue",
        "MSAHasDeletion",
        "MSAOneHot",
    }
    cropped_features = {
        key: value[:, :4] if key in msa_names else value
        for key, value in features.items()
    }

    full = fe.embed_features(features, state)
    cropped = fe.embed_features(cropped_features, state)

    np.testing.assert_array_equal(
        np.asarray(cropped["MSA"]), np.asarray(full["MSA"][:, :4])
    )
    for modality in set(full) - {"MSA"}:
        np.testing.assert_array_equal(
            np.asarray(cropped[modality]), np.asarray(full[modality])
        )
