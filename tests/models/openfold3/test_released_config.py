"""``released_config`` against upstream's config object, field by field.

An earlier version of this test parsed upstream's ``model_config.py`` as text,
which only checked the numbers that happened to be written as literals there. It
missed ``num_cycles``, which is not a literal at all: upstream stores
``shared.num_recycles`` and runs ``num_recycles + 1`` cycles, so the released value
is 4 and this preset claimed 10.

Reading the config object instead makes every field checkable, including derived
ones, and makes a field added upstream fail loudly here rather than silently keep
a stale default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.models.openfold3.inference import RELEASED_BLOCK_COUNTS, released_config

pytestmark = pytest.mark.torch_parity

N_TOKEN, N_ATOM = 16, 200


@pytest.fixture(scope="module")
def upstream(openfold3_source: Path):
    from openfold3.projects.of3_all_atom.config.model_config import model_config

    return model_config


def _released_msa_depth(embedder) -> int:
    """Upstream's fixed subsample depth, or ``None`` if it does not subsample."""
    assert not embedder.subsample_main_msa, (
        "upstream switched to per-chain main-MSA subsampling; the port's single "
        "depth no longer describes it"
    )
    if not embedder.subsample_all_msa:
        return None
    low = embedder.min_subsampled_all_msa
    high = embedder.max_subsampled_all_msa
    assert low == high, (
        f"upstream now draws the depth from a range ({low}..{high}); the port "
        "assumes a fixed count"
    )
    return low


def _expected(config) -> dict:
    architecture = config.architecture
    encoder = architecture.input_embedder.atom_attn_enc
    conditioning = architecture.diffusion_module.diffusion_conditioning
    heads = architecture.heads
    schedule = architecture.noise_schedule
    diffusion = architecture.shared.diffusion
    expected = {
        "n_query": encoder.n_query,
        "n_key": encoder.n_key,
        "atom_heads": encoder.no_heads,
        "token_heads": architecture.diffusion_module.diffusion_transformer.no_heads,
        "no_heads_msa": architecture.msa.msa_module.no_heads_msa,
        "no_heads_pair": architecture.pairformer.no_heads_pair,
        "no_heads_pair_bias": architecture.pairformer.no_heads_pair_bias,
        "max_relative_idx": conditioning.max_relative_idx,
        "max_relative_chain": conditioning.max_relative_chain,
        # Derived, not a literal: upstream runs num_recycles + 1 cycles.
        "num_cycles": architecture.shared.num_recycles + 1,
        "num_samples": diffusion.no_full_rollout_samples,
        "no_rollout_steps": diffusion.no_full_rollout_steps,
        "max_atoms_per_token": heads.max_atoms_per_token,
        "plddt_bins": heads.lddt.c_out,
        "pae_bins": heads.pae.c_out,
        "sigma_data": architecture.diffusion_module.diffusion_module.sigma_data,
        "s_max": schedule.s_max,
        "s_min": schedule.s_min,
        "p": schedule.p,
        "confidence_min_bin": heads.pairformer_embedding.min_bin,
        "confidence_max_bin": heads.pairformer_embedding.max_bin,
        "confidence_no_bin": heads.pairformer_embedding.no_bin,
        "experimentally_resolved_bins": heads.experimentally_resolved.c_out,
        "opm_first": architecture.msa.msa_module.opm_first,
        # The PAE bin edges live with the loss/metric config rather than the head,
        # since the head only emits logits.
        "pae_bin_max": architecture.loss_module.confidence.pae.bin_max,
        # Token count above which upstream runs the confidence re-embedding one
        # diffusion sample at a time (its ``apply_per_sample``). This is a memory
        # strategy rather than an architecture dimension, but it is a *default* --
        # 750 -- and a port that batched the samples unconditionally would be
        # running a different configuration from the released one at any realistic
        # target size, so it belongs in this comparison.
        "per_sample_token_cutoff": heads.per_sample_token_cutoff,
        # Alignment rows the trunk sees. Upstream draws this from
        # ``randint(min_subsampled_all_msa, max_subsampled_all_msa + 1)``, and the
        # released config sets both to 1024, so the count is fixed even though the
        # mechanism is a draw. Checking ``max`` alone would pass a config that had
        # widened the range downward, so assert they are equal first.
        "msa_depth": _released_msa_depth(architecture.msa.msa_module_embedder),
    }
    for name in ("gamma_0", "gamma_min", "noise_scale", "step_scale"):
        expected[name] = architecture.sample_diffusion[name]
    return expected


def test_every_field_matches_upstream(upstream) -> None:
    config = released_config(n_token=N_TOKEN, n_atom=N_ATOM)
    mismatched = {}
    for name, want in _expected(upstream).items():
        got = getattr(config, name)
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(got - want) > 1e-9:
                mismatched[name] = (got, want)
        elif got != want:
            mismatched[name] = (got, want)
    assert not mismatched, f"released_config disagrees with upstream: {mismatched}"


def test_shapes_are_the_caller_s(upstream) -> None:
    config = released_config(n_token=7, n_atom=99)
    assert (config.n_token, config.n_atom) == (7, 99)


def test_no_field_is_left_unchecked(upstream) -> None:
    """Every ``InferenceConfig`` field is either checked above or a shape.

    Without this, adding a field would silently escape the comparison.
    """
    config = released_config(n_token=N_TOKEN, n_atom=N_ATOM)
    # Shapes come from the caller; pair_chunk_size is a memory/speed knob with no
    # upstream counterpart -- upstream's equivalent is settings.memory.chunk_size,
    # which is a different mechanism (it chunks more than the pair attention).
    # returned_pair_logits is likewise FoldJAX's own: which [.., N, N, bins]
    # logits stay entry outputs, decided by the npz writer's budget.
    shapes = {
        "n_token",
        "n_atom",
        "pair_chunk_size",
        "returned_pair_logits",
        "has_atomized_tokens",
    }
    unchecked = set(config._fields) - set(_expected(upstream)) - shapes
    assert not unchecked, f"fields not compared against upstream: {sorted(unchecked)}"


def test_block_counts_match_upstream(upstream) -> None:
    architecture = upstream.architecture
    assert RELEASED_BLOCK_COUNTS["pairformer_stack.blocks"] == (
        architecture.pairformer.no_blocks
    )
    assert RELEASED_BLOCK_COUNTS["msa_module.blocks"] == (
        architecture.msa.msa_module.no_blocks
    )
    assert RELEASED_BLOCK_COUNTS["diffusion_transformer.blocks"] == (
        architecture.diffusion_module.diffusion_transformer.no_blocks
    )
