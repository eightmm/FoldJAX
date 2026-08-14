"""End-to-end smoke test: full JAX structure sampling on real features.

Runs trunk -> diffusion conditioning -> sampling loop via boltz2_sample_forward
on a real processed feature bundle and a real checkpoint. This is stochastic
(random augmentation + noise), so it validates shape / finiteness / basic
geometry rather than bit-parity with PyTorch.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.bridge.checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.native import load_features_npz
from foldjax.models.boltz2.bridge.torch_mapping import map_boltz2_graph_state_dict
from foldjax.paths import downloads_dir

# Both used to point into sibling checkouts -- the released checkpoint at
# `../boltz/.cache/`, the features at `../boltz_jax/outputs/`. The checkpoint now
# comes from the FoldJAX download store, which is where
# `foldjax weights fetch --model boltz2` puts the same released file, and the
# features are the fixture carried beside this test. The `.npz` holds the same
# 1UBQ_A bundle the `.pt` did; it is read here rather than the torch file so the
# fixture is one format the whole suite shares.
CHECKPOINT = downloads_dir("boltz2") / "boltz2_conf.ckpt"
FEATURES = Path(__file__).parent / "fixtures/1UBQ_A.npz"


@pytest.fixture(scope="module")
def graph_params():
    if not CHECKPOINT.exists():
        pytest.skip(
            f"released checkpoint not found: {CHECKPOINT}. "
            "Run `foldjax weights fetch --model boltz2`."
        )
    state = load_checkpoint_state_dict(CHECKPOINT)
    return map_boltz2_graph_state_dict(state, token_transformer_heads=16)


def _load_feats() -> dict[str, jnp.ndarray]:
    return load_features_npz(FEATURES)


@pytest.mark.slow
def test_full_sample_runs_and_is_finite(graph_params) -> None:
    if not FEATURES.exists():
        pytest.skip(f"features not found: {FEATURES}")
    from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_sample_forward

    feats = _load_feats()
    n_atoms = int(feats["atom_pad_mask"].shape[1])

    out = boltz2_sample_forward(
        graph_params,
        feats,
        jax.random.PRNGKey(0),
        recycling_steps=0,
        num_sampling_steps=5,
    )
    coords = np.asarray(out["sample_atom_coords"])

    assert coords.shape == (1, n_atoms, 3)
    assert np.isfinite(coords).all()
    # Real (unmasked) atoms must spread out, not collapse to a point.
    mask = np.asarray(feats["atom_pad_mask"]).astype(bool)[0]
    real = coords[0][mask]
    assert real.std() > 0.1


@pytest.mark.slow
def test_alignment_reverse_diff_toggle_is_finite(graph_params) -> None:
    if not FEATURES.exists():
        pytest.skip(f"features not found: {FEATURES}")
    from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_sample_forward

    feats = _load_feats()
    out = boltz2_sample_forward(
        graph_params,
        feats,
        jax.random.PRNGKey(1),
        num_sampling_steps=3,
        augmentation=False,
        alignment_reverse_diff=False,
    )
    coords = np.asarray(out["sample_atom_coords"])
    assert np.isfinite(coords).all()
