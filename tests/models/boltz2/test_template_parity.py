"""Torch-parity gate for the JAX TemplateV2Module port.

Feeds identical random template features + matched weights to the torch
``TemplateV2Module`` and the JAX ``template_module_forward``, then compares the
z-contribution. Also asserts the no-template gate returns no update.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", False)

_BOLTZ_SRC = Path(
    os.environ.get("BOLTZ_SRC", Path(__file__).resolve().parents[4] / "boltz" / "src")
)


def _ensure_boltz() -> bool:
    """Make the upstream torch ``boltz`` package importable (bridge-only)."""

    if importlib.util.find_spec("boltz") is not None:
        return True
    boltz_src = str(_BOLTZ_SRC)
    if _BOLTZ_SRC.is_dir() and boltz_src not in sys.path:
        sys.path.insert(0, boltz_src)
    return importlib.util.find_spec("boltz") is not None


def _build_feats(rng: np.random.Generator, b: int, t: int, n: int, num_tokens: int):
    import torch

    restype_idx = rng.integers(0, num_tokens, size=(b, t, n))
    restype = np.eye(num_tokens, dtype=np.float32)[restype_idx]
    frame_rot = rng.standard_normal((b, t, n, 3, 3)).astype(np.float32)
    frame_t = rng.standard_normal((b, t, n, 3)).astype(np.float32)
    cb = rng.standard_normal((b, t, n, 3)).astype(np.float32) * 10.0
    ca = rng.standard_normal((b, t, n, 3)).astype(np.float32) * 10.0
    cb_mask = (rng.random((b, t, n)) > 0.2).astype(np.float32)
    frame_mask = (rng.random((b, t, n)) > 0.2).astype(np.float32)
    # template_mask: [B, T, N] real-template signal (>=1 nonzero).
    tmask = (rng.random((b, t, n)) > 0.3).astype(np.float32)
    tmask[:, 0, 0] = 1.0
    vis = rng.integers(0, 3, size=(b, t, n)).astype(np.float32)

    np_feats = {
        "template_restype": restype,
        "template_frame_rot": frame_rot,
        "template_frame_t": frame_t,
        "template_cb": cb,
        "template_ca": ca,
        "template_mask_cb": cb_mask,
        "template_mask_frame": frame_mask,
        "template_mask": tmask,
        "visibility_ids": vis,
    }
    torch_feats = {k: torch.from_numpy(v) for k, v in np_feats.items()}
    jax_feats = {k: jnp.asarray(v) for k, v in np_feats.items()}
    return np_feats, torch_feats, jax_feats


def test_template_module_parity():
    torch = pytest.importorskip("torch")
    if not _ensure_boltz():
        pytest.skip("upstream torch boltz package not available")
    from boltz.data import const as bconst
    from boltz.model.modules.trunkv2 import TemplateV2Module

    from foldjax.models.boltz2.bridge.torch_mapping import (
        map_template_module_state_dict,
    )
    from foldjax.models.boltz2.models.trunk_blocks.template import (
        template_module_forward,
    )

    token_z = 128
    b, t, n = 1, 3, 12
    rng = np.random.default_rng(0)

    torch.manual_seed(0)
    module = TemplateV2Module(
        token_z=token_z,
        template_dim=64,
        template_blocks=2,
        dropout=0.0,
    ).eval()

    z_np = rng.standard_normal((b, n, n, token_z)).astype(np.float32)
    pm_np = (rng.random((b, n, n)) > 0.1).astype(np.float32)
    _, torch_feats, jax_feats = _build_feats(rng, b, t, n, bconst.num_tokens)

    with torch.no_grad():
        u_torch = (
            module(torch.from_numpy(z_np), torch_feats, torch.from_numpy(pm_np))
            .cpu()
            .numpy()
        )

    sd = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    sd = {f"template_module.{k}": v for k, v in sd.items()}
    params = map_template_module_state_dict(sd)

    u_jax = np.asarray(
        template_module_forward(
            params,
            jnp.asarray(z_np),
            jax_feats,
            jnp.asarray(pm_np),
        )
    )

    max_abs = float(np.max(np.abs(u_jax - u_torch)))
    denom = float(np.max(np.abs(u_torch))) + 1e-8
    rel = max_abs / denom
    assert max_abs < 1e-4 or rel < 1e-4, f"max_abs={max_abs} rel={rel}"


def test_no_template_gate():
    """The trunk gate skips the template path for dummy (all-zero) masks."""

    from foldjax.models.boltz2.models.trunk_blocks.template import has_template_feats

    dummy = {"template_mask": jnp.zeros((1, 1, 8), dtype=jnp.float32)}
    assert has_template_feats(dummy) is False

    real = jnp.zeros((1, 1, 8), dtype=jnp.float32).at[0, 0, 0].set(1.0)
    assert has_template_feats({"template_mask": real}) is True
