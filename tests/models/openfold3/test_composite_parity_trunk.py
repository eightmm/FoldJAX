"""Whole-trunk parity against upstream's ``OpenFold3.run_trunk``.

The 30 per-layer parity files each gate one module in isolation, which cannot see
how the modules are wired together: a module that is correct but never called, or
called with the wrong input, passes every one of them. This test closes that hole
for the trunk by running upstream's own ``run_trunk`` -- recycling, template
embedding, MSA module and Pairformer -- against this port's ``trunk`` on the same
batch and the same weights.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_trunk
from foldjax.models.openfold3.models.trunk import trunk

from .of3_features import make_batch

pytestmark = pytest.mark.torch_parity

# A whole-model forward pass amplifies weight magnitude; see the randomized
# fixture for why the scale is lowered instead of the tolerance.
COMPOSITE_SCALE = 0.05

RTOL = 1e-4
ATOL = 1e-4

N_TOKEN, N_MSA, N_TEMPL = 12, 6, 2


def _reduced_config(blocks: int = 1):
    """The released config with ``blocks`` blocks per stack; widths untouched.

    More than one block matters: with a single block, a stack mapper cannot get the
    block *order* wrong, so nothing detects a reversed or off-by-one index.
    """
    from openfold3.projects.of3_all_atom.config.model_config import model_config

    config = copy.deepcopy(model_config)
    architecture = config.architecture
    architecture.pairformer.no_blocks = blocks
    architecture.msa.msa_module.no_blocks = blocks
    architecture.template.template_pair_stack.no_blocks = blocks
    architecture.diffusion_module.diffusion_transformer.no_blocks = blocks
    architecture.heads.pairformer_embedding.pairformer.no_blocks = blocks
    architecture.heads.pae.enabled = True
    # The released eval settings turn on the Triton triangle kernels, which cannot
    # run on CPU tensors. This port has no kernel paths at all -- they are a
    # memory optimization upstream asserts is numerically equivalent -- so tests
    # compare against the unfused reference implementation.
    for mode in ("train", "eval"):
        memory = config.settings.memory[mode]
        memory.use_triton_triangle_kernels = False
        memory.use_cueq_triangle_kernels = False
        memory.use_deepspeed_evo_attention = False
        memory.use_lma = False
        memory.chunk_size = None
        memory.tune_chunk_size = False
    return config


NUM_CYCLES = (1, 3)


@pytest.fixture(scope="module", params=NUM_CYCLES, ids=lambda n: f"cycles{n}")
def reference(request, openfold3_source: Path, randomized):
    """Upstream's model, its trunk outputs, and the batch they came from.

    The weights are randomized: OpenFold3 zero-initializes output and gate
    projections, so a default-initialized model leaves whole sub-paths at exactly
    zero and the comparison below would be 0 against 0 for those paths.
    """
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    torch.manual_seed(0)
    model = randomized(OpenFold3(_reduced_config()), scale=COMPOSITE_SCALE)
    batch = make_batch(n_token=N_TOKEN, n_msa=N_MSA, n_templ=N_TEMPL)
    # Recycling is itself a composition: each cycle re-projects the previous
    # output, so a single cycle would not exercise the recycling projections.
    num_cycles = request.param
    with torch.no_grad():
        s_input, s, z = model.run_trunk(batch=batch, num_cycles=num_cycles)
    return model, batch, (s_input, s, z), num_cycles


def _as_jax(batch: dict) -> dict:
    return {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }


def test_upstream_trunk_consumes_this_batch(reference) -> None:
    """The batch is built here, so the reference run is what validates it."""
    _model, _batch, (s_input, s, z), _cycles = reference
    import torch

    for name, tensor in (("s_input", s_input), ("s", s), ("z", z)):
        assert bool(torch.isfinite(tensor).all()), name
        assert tensor.abs().sum() > 0, f"{name} is all zeros"


def test_trunk_matches_upstream(reference) -> None:
    model, batch, (s_input_ref, s_ref, z_ref), num_cycles = reference
    config = _reduced_config().architecture

    params = map_trunk(dict(model.state_dict()))
    s_input, s, z = trunk(
        _as_jax(batch),
        params,
        num_cycles=num_cycles,
        n_query=config.input_embedder.atom_attn_enc.n_query,
        n_key=config.input_embedder.atom_attn_enc.n_key,
        atom_heads=config.input_embedder.atom_attn_enc.no_heads,
        n_token=N_TOKEN,
        max_relative_idx=config.input_embedder.max_relative_idx,
        max_relative_chain=config.input_embedder.max_relative_chain,
        no_heads_msa=config.msa.msa_module.no_heads_msa,
        no_heads_pair=config.pairformer.no_heads_pair,
        no_heads_pair_bias=config.pairformer.no_heads_pair_bias,
    )

    for got, want, name in (
        (s_input, s_input_ref, "s_input"),
        (s, s_ref, "s"),
        (z, z_ref, "z"),
    ):
        assert got.shape == tuple(want.shape), name
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            want.detach().numpy().astype(np.float64),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"trunk {name} diverged from upstream run_trunk",
        )


def test_masked_msa_rows_are_a_no_op(reference) -> None:
    """Appending fully-masked MSA rows must not change the trunk at all.

    This is what bounds the MSA-subsampling divergence. Upstream subsamples to
    1024 rows at inference (``subsample_all_msa`` is on in the released config and
    has no training guard); when there are fewer than 1024 valid rows it keeps all
    of them in order and pads with masked rows. If padding were not a no-op, this
    port would differ from upstream for *every* MSA rather than only for those with
    more than 1024 valid rows.
    """
    import torch

    model, batch, _out, num_cycles = reference
    params = map_trunk(dict(model.state_dict()))
    config = _reduced_config().architecture

    def run(features: dict):
        return trunk(
            _as_jax(features),
            params,
            num_cycles=num_cycles,
            n_query=config.input_embedder.atom_attn_enc.n_query,
            n_key=config.input_embedder.atom_attn_enc.n_key,
            atom_heads=config.input_embedder.atom_attn_enc.no_heads,
            n_token=N_TOKEN,
            max_relative_idx=config.input_embedder.max_relative_idx,
            max_relative_chain=config.input_embedder.max_relative_chain,
            no_heads_msa=config.msa.msa_module.no_heads_msa,
            no_heads_pair=config.pairformer.no_heads_pair,
            no_heads_pair_bias=config.pairformer.no_heads_pair_bias,
        )

    padded = dict(batch)
    extra = 4
    for key, block in (
        ("msa", torch.ones(1, extra, N_TOKEN, 32).int()),
        ("has_deletion", torch.ones(1, extra, N_TOKEN)),
        ("deletion_value", torch.ones(1, extra, N_TOKEN)),
        ("msa_mask", torch.zeros(1, extra, N_TOKEN)),
    ):
        padded[key] = torch.cat([batch[key], block], dim=1)

    for name, without, with_padding in zip(
        ("s_input", "s", "z"), run(batch), run(padded), strict=True
    ):
        # Compared to float32 tolerance rather than bitwise. The two calls have
        # different MSA row counts, so they are separate programs, compiled and
        # autotuned separately -- and autotuning depends on how much device memory
        # is free, which depends on what else has run. Exact equality holds when
        # this test runs alone and fails inside the full suite, which makes it an
        # assertion about XLA's scheduler rather than about the model. The claim
        # being tested is that masked rows contribute nothing, and at these
        # magnitudes (~1.7) one float32 ulp is ~1e-7.
        np.testing.assert_allclose(
            np.asarray(without, dtype=np.float64),
            np.asarray(with_padding, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"masked MSA rows changed {name}",
        )
