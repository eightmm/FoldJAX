"""``map_inference_params`` against a real ``OpenFold3`` module.

The prefixes in ``INFERENCE_PREFIXES`` are the one part of the bridge that cannot
be checked by a per-layer parity test: each sub-mapper is gated against its own
upstream module, but nothing proves those modules sit where the assembler looks
for them. So this builds the actual released model -- block counts reduced to one
each, every channel width left alone -- and maps its ``state_dict``.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import jax
import numpy as np
import pytest

from foldjax.models.openfold3.bridge import torch_mapping
from foldjax.models.openfold3.bridge.torch_mapping import (
    INFERENCE_PREFIXES,
    find_model_prefix,
    map_inference_params,
)

pytestmark = pytest.mark.torch_parity


def _mapped_digest(params) -> str:
    """Hash one mapped PyTree without changing its array dtypes or values."""
    digest = hashlib.sha256(str(jax.tree.structure(params)).encode())
    for leaf in jax.tree.leaves(params):
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode())
        digest.update(repr(array.shape).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _alias_state(*, prefix: str = "wrapper") -> dict[str, np.ndarray]:
    root = f"{prefix}." if prefix else ""
    return {
        f"{root}diffusion_module.a": np.asarray([1.0, -0.0], dtype=np.float32),
        f"{root}diffusion_module.b": np.asarray([2, 3], dtype=np.int32),
        f"{root}sample_diffusion.diffusion_module.a": np.asarray(
            [1.0, -0.0], dtype=np.float32
        ),
        f"{root}sample_diffusion.diffusion_module.b": np.asarray(
            [2, 3], dtype=np.int32
        ),
        "other.sample_diffusion.diffusion_module.a": np.asarray(
            [9.0], dtype=np.float32
        ),
    }


def test_prunes_only_a_complete_alias_group_at_the_resolved_nested_prefix() -> None:
    state = _alias_state()

    removed = torch_mapping.prune_sample_diffusion_aliases(state, prefix="wrapper")

    assert removed == 2
    assert set(state) == {
        "wrapper.diffusion_module.a",
        "wrapper.diffusion_module.b",
        "other.sample_diffusion.diffusion_module.a",
    }


def test_preserves_a_partial_sample_diffusion_group() -> None:
    state = _alias_state()
    del state["wrapper.sample_diffusion.diffusion_module.b"]

    assert torch_mapping.prune_sample_diffusion_aliases(state, prefix="wrapper") == 0
    assert "wrapper.sample_diffusion.diffusion_module.a" in state


def test_preserves_a_nonmatching_sample_diffusion_group_bitwise() -> None:
    state = _alias_state()
    state["wrapper.sample_diffusion.diffusion_module.a"] = np.asarray(
        [1.0, 0.0], dtype=np.float32
    )

    assert torch_mapping.prune_sample_diffusion_aliases(state, prefix="wrapper") == 0
    assert "wrapper.sample_diffusion.diffusion_module.a" in state


def test_compares_scalar_aliases_bitwise_without_aborting() -> None:
    identical = {
        "diffusion_module.scalar": np.asarray(-0.0, dtype=np.float32),
        "sample_diffusion.diffusion_module.scalar": np.asarray(-0.0, dtype=np.float32),
    }
    assert torch_mapping.prune_sample_diffusion_aliases(identical, prefix="") == 1
    assert set(identical) == {"diffusion_module.scalar"}

    different = {
        "diffusion_module.scalar": np.asarray(-0.0, dtype=np.float32),
        "sample_diffusion.diffusion_module.scalar": np.asarray(0.0, dtype=np.float32),
    }
    assert torch_mapping.prune_sample_diffusion_aliases(different, prefix="") == 0
    assert "sample_diffusion.diffusion_module.scalar" in different


def test_object_array_preserves_the_complete_alias_group() -> None:
    marker = object()
    state = {
        "diffusion_module.numeric": np.asarray([1.0], dtype=np.float32),
        "diffusion_module.extra_state": np.asarray(marker, dtype=object),
        "sample_diffusion.diffusion_module.numeric": np.asarray(
            [1.0], dtype=np.float32
        ),
        "sample_diffusion.diffusion_module.extra_state": np.asarray(
            marker, dtype=object
        ),
    }

    assert torch_mapping.prune_sample_diffusion_aliases(state, prefix="") == 0
    assert set(state) == {
        "diffusion_module.numeric",
        "diffusion_module.extra_state",
        "sample_diffusion.diffusion_module.numeric",
        "sample_diffusion.diffusion_module.extra_state",
    }


def _shrink(node) -> None:
    """Set every ``no_blocks`` to 1, leaving widths at their released values."""
    if hasattr(node, "keys"):
        for key in list(node.keys()):
            if key == "no_blocks":
                node[key] = 1
            else:
                _shrink(node[key])


@pytest.fixture(scope="module")
def released_state(openfold3_source: Path) -> dict:
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3

    config = copy.deepcopy(model_config)
    _shrink(config.architecture)
    # Upstream only builds the PAE head when this is set; the released inference
    # config enables it, and predict() needs it for PAE and ipTM.
    config.architecture.heads.pae.enabled = True
    return dict(OpenFold3(config).state_dict())


def test_maps_a_real_model_state_dict(released_state: dict) -> None:
    params = map_inference_params(released_state)

    assert len(params.trunk.pairformer_stack.blocks) == 1
    assert len(params.trunk.msa_module.blocks) == 1
    assert len(params.denoiser.diffusion_transformer.blocks) == 1
    # The distogram head is the one without a layer norm.
    assert params.distogram_head.layer_norm is None
    assert params.pae_head.layer_norm is not None
    assert params.plddt_head.layer_norm is not None
    # Fourier buffers must come from the checkpoint; JAX cannot reproduce the
    # seeded torch stream that generated them.
    assert params.diffusion_conditioning.fourier_emb.w.shape[-1] > 0
    # The pair conditioning path must be mapped, not silently skipped.
    assert params.diffusion_conditioning.linear_z.weight.shape
    assert len(params.diffusion_conditioning.transition_z) == 2


def test_openbind_layout_is_the_only_inference_contract(
    released_state: dict,
) -> None:
    params = map_inference_params(released_state, prestack=False)
    transformer = params.denoiser.diffusion_transformer
    assert transformer.layer_norm_z.weight.shape
    assert all(
        not hasattr(block.attention_pair_bias, "layer_norm_z")
        for block in transformer.blocks
    )


def test_rejects_a_legacy_checkpoint_without_the_openbind_marker(
    released_state: dict,
) -> None:
    state = dict(released_state)
    del state["diffusion_module.diffusion_transformer.layer_norm_z.weight"]

    with pytest.raises(KeyError, match="v0.5.0/OpenBind checkpoints only"):
        map_inference_params(state)


def test_rejects_a_non_openbind_model_version(released_state: dict) -> None:
    state = dict(released_state)
    state["version_tensor"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match=r"model version \[2, 0, 0\] only"):
        map_inference_params(state)


def test_pruning_keeps_the_mapped_tree_bitwise_exact(released_state: dict) -> None:
    baseline = map_inference_params(released_state)
    pruned_state = dict(released_state)
    removed = torch_mapping.prune_sample_diffusion_aliases(
        pruned_state,
        prefix=torch_mapping.resolve_model_prefix(pruned_state),
    )
    pruned = map_inference_params(pruned_state)

    assert removed > 0
    assert jax.tree.structure(pruned) == jax.tree.structure(baseline)
    baseline_leaves = jax.tree.leaves(baseline)
    pruned_leaves = jax.tree.leaves(pruned)
    for expected, actual in zip(baseline_leaves, pruned_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _mapped_digest(pruned) == _mapped_digest(baseline)


def test_every_declared_prefix_exists_in_the_real_model(released_state: dict) -> None:
    """A renamed upstream attribute must fail here, not at load time."""
    for group, prefix in INFERENCE_PREFIXES.items():
        if prefix == "":
            continue
        matches = [key for key in released_state if key.startswith(prefix + ".")]
        assert matches, f"{group}: nothing under {prefix!r} in the real model"


def test_model_prefix_detection_on_a_real_model(released_state: dict) -> None:
    """The trunk stack is at the top level, and aux_heads' copy is not mistaken
    for it."""
    assert find_model_prefix(released_state) == ""
    nested = {f"wrapper.{key}": value for key, value in released_state.items()}
    assert find_model_prefix(nested) == "wrapper"


def test_absent_pae_head_is_reported_as_a_checkpoint_property(
    released_state: dict,
) -> None:
    """Upstream skips the PAE head unless enabled; that must not read as a bug."""
    without_pae = {
        key: value
        for key, value in released_state.items()
        if not key.startswith("aux_heads.pae.")
    }
    with pytest.raises(KeyError, match="config.pae.enabled"):
        map_inference_params(without_pae)


def test_a_non_openfold3_checkpoint_is_rejected() -> None:
    with pytest.raises(KeyError, match="not an OpenFold3"):
        map_inference_params({"some.other.weight": 0})


def test_explicit_empty_prefix_skips_detection(released_state: dict) -> None:
    """Passing "" must not be treated as "detect it for me"."""
    nested = {f"wrapper.{key}": value for key, value in released_state.items()}
    with pytest.raises(KeyError):
        map_inference_params(nested, "")


def _set_all(node, name: str, value) -> None:
    if hasattr(node, "keys"):
        for key in list(node.keys()):
            if key == name:
                node[key] = value
            else:
                _set_all(node[key], name, value)


@pytest.fixture(scope="module")
def fused_state(openfold3_source: Path) -> dict:
    """The same model with every stack's tri-mul fused."""
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3

    config = copy.deepcopy(model_config)
    _shrink(config.architecture)
    config.architecture.heads.pae.enabled = True
    _set_all(config.architecture, "fuse_projection_weights", True)
    return dict(OpenFold3(config).state_dict())


def test_the_released_config_is_unfused(released_state: dict) -> None:
    """Recorded so a change upstream shows up as a failure here.

    ``fuse_projection_weights`` is false in the released config, so real
    checkpoints are expected unfused -- but the fused path must still work,
    because it is a per-stack config choice, not a version.
    """
    assert not any("linear_ab_p" in key for key in released_state)
    assert any("linear_a_p" in key for key in released_state)


def test_a_fused_model_maps_through_the_composite_mappers(fused_state: dict) -> None:
    """The fused splitter has to be reachable from map_inference_params.

    It is only useful if the composite mappers route to it; calling it by hand
    per tri-mul prefix is not a load path.
    """
    assert any("linear_ab_p" in key for key in fused_state)
    params = map_inference_params(fused_state)
    tri = params.trunk.pairformer_stack.blocks[0].pair_stack.tri_mul_out
    assert tri.linear_a_p.weight.shape == tri.linear_b_p.weight.shape
    assert tri.linear_a_g.weight.shape == tri.linear_b_g.weight.shape


def test_both_layouts_produce_the_same_parameter_shapes(
    released_state: dict, fused_state: dict
) -> None:
    """Downstream code must not be able to tell the layouts apart."""
    import jax

    unfused_shapes = jax.tree.map(
        lambda x: tuple(x.shape), map_inference_params(released_state)
    )
    fused_shapes = jax.tree.map(
        lambda x: tuple(x.shape), map_inference_params(fused_state)
    )
    assert unfused_shapes == fused_shapes


class _Tracking(dict):
    """A state dict that records which keys the mapper actually reads."""

    def __init__(self, state: dict) -> None:
        super().__init__(state)
        self.used: set[str] = set()

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:
        present = super().__contains__(key)
        if present:
            self.used.add(key)
        return present


def test_every_parameter_of_the_real_model_is_consumed(released_state: dict) -> None:
    """No parameter group may be silently ignored.

    This is the check that caught the port shipping a template tower, a
    confidence re-embedding and a diffusion pair-conditioning path that no code
    ever read: each was individually correct and individually tested, and
    together they left 31% of the model's parameters unused. Shape assertions
    cannot see that, because an unused group changes no shape.

    Two exclusions, both verified rather than assumed:

    * ``sample_diffusion.*`` mirrors ``diffusion_module.*`` -- upstream's sampler
      holds a reference to the same module, so those tensors share storage.
    * ``version_tensor`` is metadata, not a parameter.
    """
    tracked = _Tracking(released_state)
    map_inference_params(tracked)

    unconsumed = sorted(set(released_state) - tracked.used)
    aliases = [key for key in unconsumed if key.startswith("sample_diffusion.")]
    for key in aliases:
        mirrored = key[len("sample_diffusion.") :]
        assert mirrored in released_state, f"{key} is not an alias after all"
        assert released_state[key].data_ptr() == released_state[mirrored].data_ptr(), (
            f"{key} does not share storage with {mirrored}"
        )

    remaining = [
        key
        for key in unconsumed
        if not key.startswith("sample_diffusion.") and key != "version_tensor"
    ]
    assert not remaining, (
        f"{len(remaining)} parameters are never read by the inference path: "
        f"{remaining[:12]}"
    )
