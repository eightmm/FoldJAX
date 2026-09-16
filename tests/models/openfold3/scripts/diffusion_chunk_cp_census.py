"""Census the five-wide pair conditioning the OpenFold3 rollout carries.

Under context parallelism the diffusion pair conditioning is the largest value
in the shipped program: ``pair_conditioning`` is hoisted out of the rollout at a
leading axis of one, and ``predict``'s ``denoise_fn`` widens it to the sample
width *at the point of use*, which is what makes the sample width a property of
the coordinates being denoised rather than of the config. Measured at 6,568
tokens on four devices that widening is ``f32[5, N/4, N, 128]`` -- 25.7 GiB per
rank, held by both the rollout and the 24-block diffusion transformer.

The question this probe answers is whether ``diffusion_chunk_size`` removes it,
and the answer has to be read off the program rather than argued from the
source: a chunk that narrowed the coordinates but rebuilt the conditioning
outside the loop would still spell ``_expand_samples`` inside ``denoise_fn``
and would still hold every copy. So the census counts values whose shape is
``[samples, local_rows, tokens, channels]`` -- a sample axis times a *sharded*
pair tile -- in the per-device SPMD text and in the pre-optimisation HLO, for
one arm with the rollout whole and one with it chunked to a single sample.

**The SPMD half is the authoritative one.** ``compiler_ir(dialect="hlo")`` is
pre-partition, so a value outside a ``shard_map`` body still carries its global
shape and cannot match a local tile: that census only has power where the
program is already sharded in the source, which here is the atom-CP denoiser.
Both are printed because the pair of numbers says which it is.

Verified to fail, rather than assumed to: retaining the widened conditioning
outside the loop behind a ``lax.optimization_barrier`` -- the one mutation XLA
cannot fold away, since a plain ``broadcast`` sliced back down *is* folded and
leaves the program unchanged -- puts 11 of these tiles in the chunked arm's
SPMD text and 0 in its HLO, and this script exits non-zero.

Three things hold together, and each of the first two is the other's tripwire:

* the unchunked arm must show such a value, or the bound below has no power and
  would pass a program that never widened anything;
* the chunked arm must show none;
* both arms must produce the same coordinates, to the port's own chunk
  tolerance (``models/sampler.py`` narrows every noise draw from the full
  sample width rather than redrawing it, so the difference is float32
  fusion order, not a different random stream).

The trunk and the confidence tail are stubbed, in the same places
``tests/models/openfold3/test_pae_metric_sinking.py`` stubs them: the property
is about the diffusion rollout, and the diffusion rollout -- ``predict``'s own
``denoise_fn``, ``pair_conditioning``, ``single_conditioning``, the real
denoiser and the real ``sample_diffusion`` chunk loop -- is what runs here. The
entry point is ``inference._predict_from_trunk``, the shared tail ``predict``
itself calls, so the closure under test is the shipped one and not a
reconstruction of it.

Run under a forced device count with ``FOLDJAX_CP_PROBE_DEVICES`` and
``FOLDJAX_CP_PROBE_LAYOUT``; also usable on a GPU with real devices.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import context_parallel, cp_layout
from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    FourierEmbeddingParams,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
)
from foldjax.models.openfold3.models.relpos import relpos_complex
from tests.models.openfold3.atom_cp_fixtures import (
    ATOM_HEADS,
    C_ATOM_PAIR,
    C_S,
    C_Z,
    N_ATOM,
    N_KEY,
    N_QUERY,
    N_TOKEN,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_case,
    build_params,
)

#: The released sample count, and the number of copies the unchunked rollout
#: holds. Five rather than the fixture's own two so the census shape cannot be
#: confused with anything the fixture ships.
NUM_SAMPLES = 5

#: Rollout steps. Two is enough for the chunk loop to sit inside the diffusion
#: scan and for a redrawn noise stream to diverge visibly.
NUM_STEPS = 2

#: Relative-position clips. ``layer_norm_z`` spans the relpos channels plus
#: ``C_Z``, and that channel count is read off :func:`relpos_complex` itself
#: rather than recomputed from the clips: a formula here would be a second copy
#: of the encoding's width to get wrong.
MAX_RELATIVE_IDX, MAX_RELATIVE_CHAIN = 2, 1

#: Width of the input-embedder single representation the stub hands back.
C_S_INPUT = 8

#: Fourier features of the log noise level.
C_FOURIER = 8

#: Confidence-head bin counts. The stubs read the first; `compute_plddt` pins
#: the second at upstream's 50 bins, which is not a knob. ``PAE_BINS`` is
#: deliberately none of the diffusion channel widths below: the confidence
#: heads stack their own five-sample pair logits, which is a returned output
#: rather than the conditioning, and a shared width lets the census count one
#: for the other -- it did, at four bins against ``C_Z``. ``main`` asserts the
#: separation.
PAE_BINS, PLDDT_BINS = 7, 50

#: Channel width of the stubbed confidence Pairformer's pair output, held apart
#: from the conditioning widths for the same reason.
STUB_PAIR_CHANNELS = 3

#: Channel widths the diffusion pair conditioning can wear: ``C_Z`` is the
#: conditioned pair representation itself -- the ``f32[5, N/4, N, 128]`` of the
#: deployment measurement -- and ``C_ATOM_PAIR`` is the atom encoder's
#: projection of it. A sample axis must be laid over neither.
CONDITIONING_CHANNELS = (C_Z, C_ATOM_PAIR)

#: The port's own chunk tolerance: `models/sampler.py`'s chunked rollout agrees
#: with the whole one to float32 fusion order, and this is the pair
#: `tests/models/openfold3/test_sampler.py` asserts it at.
CHUNK_RTOL, CHUNK_ATOL = 1e-5, 1e-4


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260917)


def _array(rng: np.random.Generator, *shape: int) -> jnp.ndarray:
    return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)


def _linear(rng: np.random.Generator, out_features: int, in_features: int):
    return LinearParams(
        weight=_array(rng, out_features, in_features) / np.sqrt(in_features),
        bias=_array(rng, out_features),
    )


def _layer_norm(rng: np.random.Generator, channels: int) -> LayerNormParams:
    # Scale-only, the way `DiffusionConditioning` builds every one of its norms.
    return LayerNormParams(weight=_array(rng, channels) * 0.1 + 1.0, bias=None)


def _transition(rng: np.random.Generator, channels: int):
    wide = 2 * channels
    return SwiGLUTransitionParams(
        layer_norm=_layer_norm(rng, channels),
        swiglu=SwiGLUParams(
            linear_a=_linear(rng, wide, channels),
            linear_b=_linear(rng, wide, channels),
        ),
        linear_out=_linear(rng, channels, wide),
    )


def _conditioning_params(relpos_dims: int) -> DiffusionConditioningParams:
    """A ``DiffusionConditioningParams`` at the fixture's channel widths."""

    rng = _rng()
    return DiffusionConditioningParams(
        layer_norm_s=_layer_norm(rng, C_S + C_S_INPUT),
        linear_s=_linear(rng, C_S, C_S + C_S_INPUT),
        fourier_emb=FourierEmbeddingParams(
            w=_array(rng, C_FOURIER), b=_array(rng, C_FOURIER)
        ),
        layer_norm_n=_layer_norm(rng, C_FOURIER),
        linear_n=_linear(rng, C_S, C_FOURIER),
        transition_s=(_transition(rng, C_S),),
        layer_norm_z=_layer_norm(rng, C_Z + relpos_dims),
        linear_z=_linear(rng, C_Z, C_Z + relpos_dims),
        transition_z=(_transition(rng, C_Z),),
    )


def _batch() -> dict[str, jnp.ndarray]:
    """The features the diffusion path reads, at a leading axis of one.

    ``build_case`` ships its features pre-broadcast to its own sample count;
    ``predict`` expands a batch-1 feature itself, and expanding is the step
    under test, so the leading axis is narrowed back to one here.
    """

    case = build_case()
    batch = {name: value[:1] for name, value in case.batch.items()}
    tokens = np.arange(N_TOKEN, dtype=np.int32)
    # Two chains, so `relpos_complex`'s chain terms are not all one value.
    chains = (tokens >= N_TOKEN // 2).astype(np.int32)
    batch.update(
        {
            "residue_index": jnp.asarray(tokens)[None],
            "token_index": jnp.asarray(tokens)[None],
            "asym_id": jnp.asarray(chains)[None],
            "entity_id": jnp.asarray(chains)[None],
            "sym_id": jnp.zeros((1, N_TOKEN), dtype=jnp.int32),
            # Read only by the stubbed atom head.
            "max_atom_per_token_mask": jnp.ones((1, N_ATOM), dtype=jnp.float32),
        }
    )
    return batch


def _trunk_output() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng = _rng()
    return (
        _array(rng, 1, N_TOKEN, C_S_INPUT),
        _array(rng, 1, N_TOKEN, C_S),
        _array(rng, 1, N_TOKEN, N_TOKEN, C_Z),
    )


def _config(devices: int, layout: str) -> inference.InferenceConfig:
    return inference.InferenceConfig(
        n_token=N_TOKEN,
        n_atom=N_ATOM,
        n_query=N_QUERY,
        n_key=N_KEY,
        atom_heads=ATOM_HEADS,
        token_heads=TOKEN_HEADS,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=MAX_RELATIVE_IDX,
        max_relative_chain=MAX_RELATIVE_CHAIN,
        num_recycles=1,
        num_samples=NUM_SAMPLES,
        max_atoms_per_token=N_ATOM // N_TOKEN,
        plddt_bins=PLDDT_BINS,
        pae_bins=PAE_BINS,
        pae_bin_max=32.0,
        num_steps=NUM_STEPS,
        sigma_data=SIGMA_DATA,
        # Released defaults for both: the confidence pass runs one sample at a
        # time and the atom graph is distributed, which is the shape of the
        # program a deployment actually compiles.
        per_sample_token_cutoff=0,
        cp_atom_windows=True,
        has_atomized_tokens=False,
        cp_shards=devices,
        cp_layout=layout,
    )


def _relpos_dims(batch: dict[str, jnp.ndarray]) -> int:
    """The relative-position encoding's own channel count, from the encoding."""

    return int(
        relpos_complex(
            batch,
            max_relative_idx=MAX_RELATIVE_IDX,
            max_relative_chain=MAX_RELATIVE_CHAIN,
        ).shape[-1]
    )


def _install_stubs(params_holder: dict[str, object], relpos_dims: int) -> None:
    """Replace the trunk-side and confidence-side layers, and only those.

    Everything between the trunk output and the sampled coordinates stays real:
    the noise schedule, both conditioning paths, the denoiser, the rollout and
    its chunk loop. What is stubbed is what needs a parameter tree this fixture
    does not carry and cannot influence the sample axis of the diffusion pair
    conditioning.
    """

    def fake_representatives(batch, coordinates, atom_mask, table):
        del batch, atom_mask, table
        return coordinates[..., :N_TOKEN, :], jnp.ones(
            (*coordinates.shape[:-2], N_TOKEN), dtype=coordinates.dtype
        )

    def fake_frames(*args, **kwargs):
        del args, kwargs
        return (None, None, None), jnp.ones(
            (NUM_SAMPLES, N_TOKEN), dtype=jnp.bool_
        )

    def fake_pairformer(si_input, si, zij, x_pred, params, **kwargs):
        del si_input, zij, params, kwargs
        signal = x_pred[..., 0]
        single = si + signal[..., None] / 19.0
        delta = signal[..., :, None] - signal[..., None, :]
        # `STUB_PAIR_CHANNELS` wide, for the same reason `PAE_BINS` is what it
        # is: a stub value must not be able to wear a conditioning shape.
        pair = jnp.stack([delta, delta**2, -delta], axis=-1)
        return single, pair

    def fake_pair_head(pair, params):
        del params
        bins = jnp.arange(PAE_BINS, dtype=pair.dtype)
        return pair[..., :1] + bins / 23.0

    def fake_distogram(z, params):
        del params
        return jnp.broadcast_to(
            jnp.arange(PAE_BINS, dtype=z.dtype), (1, N_TOKEN, N_TOKEN, PAE_BINS)
        )

    def fake_atom_head(single, params, mask, *, c_out, n_atom, **kwargs):
        del params, mask, kwargs
        bins = jnp.arange(c_out, dtype=single.dtype)
        return single[..., :1, :1] + jnp.zeros((n_atom, c_out), single.dtype) + bins

    inference.token_representative_atoms = fake_representatives
    inference.token_frame_atoms = fake_frames
    inference.pairformer_embedding = fake_pairformer
    inference.predicted_aligned_error_head = fake_pair_head
    inference.predicted_distance_error_head = fake_pair_head
    inference.distogram_head = fake_distogram
    inference.atom_logit_head = fake_atom_head
    params_holder["params"] = inference.InferenceParams(
        trunk=None,
        diffusion_conditioning=_conditioning_params(relpos_dims),
        denoiser=build_params(),
        pairformer_embedding=None,
        plddt_head=None,
        pae_head=None,
        pde_head=None,
        distogram_head=None,
        experimentally_resolved_head=None,
    )


def _pair_shape_pattern(
    samples: int, local_rows: int, channels: tuple[int, ...]
) -> re.Pattern[str]:
    """Values laying a sample axis over a *sharded* pair tile.

    ``[samples, local_rows, tokens, channel]``. The local row count differs
    from the token count in this fixture, so nothing unsharded and nothing
    atom-shaped can spell the same thing, and the channel is named rather than
    wildcarded so the confidence heads' own five-sample output stack cannot be
    counted as the conditioning.
    """

    widths = "|".join(str(channel) for channel in channels)
    return re.compile(
        rf"\b(?:f32|bf16)\[{samples},{local_rows},{N_TOKEN},(?:{widths})\]"
    )


def _census(text: str, pattern: re.Pattern[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for shape in pattern.findall(text):
        found[shape] = found.get(shape, 0) + 1
    return found


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ.get("FOLDJAX_CP_PROBE_LAYOUT", "1d")
    assert jax.device_count() == devices, jax.devices()
    assert N_TOKEN % devices == 0, (N_TOKEN, devices)
    local_rows = N_TOKEN // devices

    # `foldjax` is installed editable, so a child whose `PYTHONPATH` does not
    # reach this tree silently imports the install instead -- a different
    # checkout inside a worktree, and a census of code nobody edited.
    source = Path(__file__).resolve().parents[4] / "src"
    print("foldjax", inference.__file__)
    assert inference.__file__.startswith(str(source)), (inference.__file__, source)
    batch, trunk_output = _batch(), _trunk_output()
    holder: dict[str, object] = {}
    _install_stubs(holder, _relpos_dims(batch))
    params = holder["params"]
    config = _config(devices, layout)
    traced: list[object] = []

    def build(chunk: int | None):
        """One fresh closure per arm; `jit` keys its cache on the callable."""

        def run(batch_in, trunk_in):
            traced.append(cp_layout())
            return inference._predict_from_trunk(
                jax.random.key(11),
                batch_in,
                params,
                config._replace(diffusion_chunk_size=chunk),
                None,
                trunk_output=trunk_in,
            )

        return run

    # The chunked arm's width comes from the resolver the shipped default goes
    # through, not from a literal: a census against `1` would keep passing if
    # the default stopped resolving there, which is the whole claim.
    resolved = inference.resolve_diffusion_chunk_size(
        "auto", num_samples=NUM_SAMPLES, cp_shards=devices
    )
    print(f"  resolved default under {devices} shards: {resolved!r}")
    assert resolved == inference.CP_DIFFUSION_CHUNK_SIZE, resolved
    assert isinstance(resolved, int) and 1 <= resolved < NUM_SAMPLES, resolved
    assert (
        inference.resolve_diffusion_chunk_size(
            "auto", num_samples=NUM_SAMPLES, cp_shards=1
        )
        is None
    ), "the serial default stopped being the unchunked rollout"

    for stub_width in (PAE_BINS, STUB_PAIR_CHANNELS):
        assert stub_width not in CONDITIONING_CHANNELS, stub_width
    pattern = _pair_shape_pattern(NUM_SAMPLES, local_rows, CONDITIONING_CHANNELS)
    # The confidence heads' own five-sample pair logit stack, which chunking
    # the rollout neither removes nor should. It is the control that makes the
    # census specific to the conditioning rather than to any wide value, and
    # it is where "all five samples still reach the output" is read.
    returned = _pair_shape_pattern(NUM_SAMPLES, local_rows, (PAE_BINS,))
    arms: dict[str | int, dict[str, object]] = {}
    for chunk in (None, resolved, 2):
        jax.clear_caches()
        with context_parallel(devices, layout=layout):
            program = jax.jit(build(chunk))
            # Traced arguments rather than closed-over constants: a folded
            # feature can make a value appear or vanish for reasons the real
            # program, whose features arrive as arguments, does not share.
            lowered = program.lower(batch, trunk_output)
            spmd = lowered.compile().as_text()
            hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
            prediction = program(batch, trunk_output)
            coordinates = np.asarray(
                jax.device_get(prediction.coordinates), dtype=np.float64
            )
        arm = {
            "spmd": _census(spmd, pattern),
            "hlo": _census(hlo, pattern),
            "returned": _census(spmd, returned),
            "coordinates": coordinates,
        }
        arms["whole" if chunk is None else chunk] = arm
        label = "whole" if chunk is None else f"chunk={chunk}"
        print(
            f"  {label:<9} samples={coordinates.shape[0]} "
            f"conditioning spmd={arm['spmd']} hlo={arm['hlo']} "
            f"returned_logits={arm['returned']}"
        )

    # Positive evidence that every arm was traced under the mesh rather than
    # replayed from an unsharded jaxpr: without this the census is a census of
    # the serial program, where no sharded tile exists in either arm and the
    # chunked side passes for the wrong reason.
    assert traced == [layout] * 3, traced

    whole, one, two = arms["whole"], arms[resolved], arms[2]

    # The defect, still present with the rollout whole. Without this the bound
    # below would pass a program that widened nothing anywhere.
    assert whole["spmd"], ("no sharded five-wide pair tile with the rollout whole")
    assert whole["hlo"], ("no sharded five-wide pair tile in the unoptimised HLO")
    # The property.
    assert not one["spmd"], one["spmd"]
    assert not one["hlo"], one["hlo"]
    # The control: a wide value the chunk is not meant to touch stays put in
    # every arm. Without it, "no five-wide tile" could be satisfied by a
    # program that had lost the sample axis altogether.
    for label, arm in (("whole", whole), (f"chunk={resolved}", one), ("chunk=2", two)):
        assert arm["returned"], (label, "the five-sample pair logits vanished")

    # Sample count and sample identity: the same five samples in the same
    # order, to the tolerance the port documents for its chunk loop, with no
    # cross-sample coupling -- a chunk that leaked one sample's coordinates
    # into another's would move them by whole Angstrom.
    # Absolute, not "equal to the other arm": two arms that had both lost the
    # sample axis would agree with each other.
    expected_shape = (NUM_SAMPLES, N_ATOM, 3)
    assert whole["coordinates"].shape == expected_shape, whole["coordinates"].shape
    for label, arm in ((f"chunk={resolved}", one), ("chunk=2", two)):
        assert arm["coordinates"].shape == expected_shape, (
            label,
            arm["coordinates"].shape,
        )
        delta = np.abs(arm["coordinates"] - whole["coordinates"])
        print(
            f"  {label} vs whole: max {delta.max():.3e} "
            f"rms {np.sqrt((delta**2).mean()):.3e} "
            f"scale {np.abs(whole['coordinates']).max():.3f}"
        )
        np.testing.assert_allclose(
            arm["coordinates"],
            whole["coordinates"],
            rtol=CHUNK_RTOL,
            atol=CHUNK_ATOL,
            err_msg=f"{label} moved the coordinates",
        )

    # Non-vacuity: the samples genuinely differ from one another, so the
    # agreement above is a statement about per-sample identity and not about
    # five copies of one structure.
    reference = whole["coordinates"]
    for index in range(1, NUM_SAMPLES):
        assert not np.allclose(reference[0], reference[index]), index

    print("OPENFOLD3_DIFFUSION_CHUNK_CP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
