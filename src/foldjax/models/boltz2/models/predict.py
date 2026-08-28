"""Single Boltz2.forward-equivalent JAX inference wrapper.

Mirrors ``boltz.model.models.boltz2.Boltz2.forward`` (trunk -> structure
sampling -> distogram -> (bfactor) -> confidence -> (affinity)) and returns
ONE result dict. The individual head functions are reused unchanged so the
wrapper does not alter numerics.

Key mapping (foldjax.models.boltz2 key -> Boltz2.forward key):
    sample_atom_coords -> sample_atom_coords  (structure sampler)
    pdistogram         -> pdistogram          (distogram head)
    pbfactor           -> pbfactor            (bfactor head, optional)
    plddt/pae/pde/ptm/iptm/complex_*/... -> confidence_module outputs
    affinity_*         -> affinity_module outputs (only if affinity_params given)

Training-only branches and miniformer are outside this inference wrapper. The
template path and the complete two-member affinity ensemble are supported; the
``s``/``z`` trunk tensors are kept internal.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models import _capture
from foldjax.models._cp import cp_mesh
from foldjax.models.boltz2.models.heads.affinity import affinity_module_forward
from foldjax.models.boltz2.models.heads.bfactor import bfactor_forward
from foldjax.models.boltz2.models.heads.confidence import confidence_module_forward
from foldjax.models.boltz2.models.heads.distogram import distogram_forward
from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
    input_embedder_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    _cast_float_feats,
    _cast_params,
    boltz2_sample_forward,
    boltz2_trunk_forward,
)

Params = Mapping[str, object]

#: Stream id folded into the run key for the MSA subsample, so those draws are
#: independent of the sampler's noise and adding them moved nothing else.
_MSA_SUBSAMPLE_STREAM = 0xB012


def boltz2_predict(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    *,
    recycling_steps: int = 3,
    num_sampling_steps: int = 200,
    augmentation: bool = True,
    steering_args: Mapping[str, object] | None = None,
    run_confidence: bool = True,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    run_distogram: bool = True,
    run_bfactor: bool = False,
    #: Whether the confidence head's full-bin logits stay in the result. The
    #: summaries (plddt, pae, complex_* scalars, ptm family) are computed here
    #: either way; the pae/pde logits alone are f32[num_samples, N, N, 64] -- 10.8
    #: GiB each at 3,012 tokens, 5 samples -- and as jit outputs they stay
    #: resident alongside the temp arena for the whole run. Same finding as
    #: Protenix (EXPERIMENT_LOG: the 84.1 vs 57.3 attribution).
    return_confidence_logits: bool = True,
    affinity_params: Params | None = None,
    affinity_mw_correction: bool = False,
    eps: float = 1e-5,
    # `boltz predict` does NOT subsample. Its `--subsample_msa` is a click flag
    # (`is_flag=True`, so `Option.default is False`), and click passes its own
    # value regardless of the `= True` in the callback's Python signature --
    # which is what an earlier reading of `main.py` took for the default.
    # Verified at runtime: no `torch.randperm` reaches `MSAModule.forward` on
    # the shipped CLI path. Upstream reads the whole alignment, so this does
    # too.
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    **sample_kwargs: object,
) -> dict[str, object]:
    """Run the full Boltz-2 inference graph and return one result dict.

    Computes the deterministic trunk once, then reuses it for structure sampling
    and downstream heads. Head numerics are identical to calling each head
    function directly with the same trunk/sample tensors.
    """
    multiplicity = int(sample_kwargs.pop("multiplicity", 1))
    if cp_mesh() is not None and (
        str(sample_kwargs.get("triangle_backend", "cueq")) == "cueq"
    ):
        # Under context parallelism the fused kernel default resolves to the
        # blocked XLA path -- the cueq FFI call cannot be partitioned. An
        # explicitly requested pallas/tokamax backend still fails loudly in
        # the triangle modules.
        sample_kwargs["triangle_backend"] = "xla"
    if affinity_params is None and "affinity" in params:
        affinity_params = params["affinity"]
    return_pair_chains_iptm = bool(sample_kwargs.pop("return_pair_chains_iptm", True))
    confidence_sequentially = bool(sample_kwargs.pop("confidence_sequentially", False))
    #: Diffusion samples denoised at once. The neutral name every port in this
    #: repository uses for this knob; `None` denoises all of them together.
    diffusion_chunk_size = sample_kwargs.pop("diffusion_chunk_size", None)
    if diffusion_chunk_size is not None:
        diffusion_chunk_size = int(diffusion_chunk_size)
        if diffusion_chunk_size < 1:
            raise ValueError("diffusion_chunk_size must be at least 1")
    recompute_nonpolymer_frames = bool(
        sample_kwargs.pop("recompute_nonpolymer_frames", True)
    )
    # Decided by the caller from concrete features: under jit the template mask
    # is a tracer, so the trunk cannot make this call itself.
    use_template = sample_kwargs.pop("use_template", None)
    trunk_use_scan = sample_kwargs.get(
        "trunk_use_scan", sample_kwargs.get("use_scan", True)
    )
    compute_dtype = jnp.dtype(sample_kwargs.get("compute_dtype", jnp.float32))
    trunk_params = params["trunk"]
    trunk_feats = feats
    if compute_dtype != jnp.float32:
        # boltz2_sample_forward applies this mixed-precision policy when it
        # owns the trunk calculation.  The full predict wrapper computes the
        # trunk separately for reuse by the heads, so it must apply the same
        # casts here rather than handing an already-fp32 trunk to the sampler.
        trunk_params = _cast_params(trunk_params, compute_dtype)
        trunk_feats = _cast_float_feats(feats, compute_dtype)

    # Upstream's MSA subsample is drawn per forward pass from the run's own RNG,
    # so it belongs on the run's key: one seed still reproduces one result.
    # Derived by `fold_in` rather than `split` on purpose -- splitting would
    # hand the sampler a different key and move every coordinate, making an
    # MSA-selection change look like a diffusion change. This way the only
    # thing that moves is which alignment rows the trunk sees.
    msa_key = jax.random.fold_in(key, _MSA_SUBSAMPLE_STREAM)
    trunk = boltz2_trunk_forward(
        trunk_params,
        trunk_feats,
        msa_key=msa_key if subsample_msa else None,
        recycling_steps=recycling_steps,
        eps=eps,
        use_scan=bool(trunk_use_scan),
        chunk_size=int(sample_kwargs.get("chunk_size", 128)),
        triangle_attention_chunk=sample_kwargs.get("triangle_attention_chunk"),
        triangle_attention_q_chunk=sample_kwargs.get("triangle_attention_q_chunk"),
        transition_hidden_chunk=sample_kwargs.get("transition_hidden_chunk"),
        matmul_precision=str(sample_kwargs.get("matmul_precision", "highest")),
        attention_backend=str(sample_kwargs.get("attention_backend", "xla")),
        triangle_backend=str(sample_kwargs.get("triangle_backend", "cueq")),
        glu_backend=str(sample_kwargs.get("glu_backend", "xla")),
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_template=use_template,
    )
    s_inputs, s, z = trunk["s_inputs"], trunk["s"], trunk["z"]
    s_inputs = _capture.capture("single_inputs", s_inputs)
    s = _capture.capture("single", s)
    z = _capture.capture("pair", z)

    if stop_after_trunk:
        # The representations exist; the sampler, the confidence module
        # and the affinity path are the rest of the run.
        return {
            name: value
            for name, value in (
                ("single_inputs", s_inputs),
                ("single", s),
                ("pair", z),
            )
            if name in return_representations
        }

    def sample(key, sample_multiplicity=None):
        return boltz2_sample_forward(
            params,
            feats,
            key,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            augmentation=augmentation,
            steering_args=steering_args,
            multiplicity=multiplicity
            if sample_multiplicity is None
            else sample_multiplicity,
            eps=eps,
            trunk=trunk,
            **sample_kwargs,
        )

    # The denoiser holds its activations for every sample it is handed, and
    # this port's conditioning is `jnp.repeat`-ed rather than broadcast, so both
    # grow with the sample count: at 1,003 tokens the peak went 6.7 GiB at one
    # sample to 20.8 at thirty-two. Denoising a chunk at a time bounds both.
    #
    # Unlike a blocked matmul this is not the same arithmetic: each chunk draws
    # its noise at its own width, so the structures are different draws from
    # the same distribution rather than the same draws computed differently.
    # That is what Protenix's `diffusion_chunk_size` already does, and why the
    # rule below leaves the released five-sample run on the single unchunked
    # call it has always taken.
    if diffusion_chunk_size is not None and multiplicity > diffusion_chunk_size:
        starts = range(0, multiplicity, diffusion_chunk_size)
        chunk_keys = jax.random.split(key, len(list(starts)))
        sample_atom_coords = jnp.concatenate(
            [
                sample(
                    chunk_key,
                    min(diffusion_chunk_size, multiplicity - start),
                )["sample_atom_coords"]
                for chunk_key, start in zip(
                    chunk_keys,
                    range(0, multiplicity, diffusion_chunk_size),
                    strict=True,
                )
            ],
            axis=0,
        )
    else:
        sample_atom_coords = sample(key)["sample_atom_coords"]

    out: dict[str, object] = {"sample_atom_coords": sample_atom_coords}
    # The trunk's own outputs, for downstream work that wants what the
    # model saw rather than only where it put the atoms. Off by default:
    # the pair representation is quadratic in token count and an entry
    # output stays resident for the whole run.
    for name, value in (
        ("single_inputs", s_inputs),
        ("single", s),
        ("pair", z),
    ):
        if name in return_representations:
            out[name] = value

    pdistogram = None
    if run_distogram or run_confidence:
        pdistogram = distogram_forward(params, z)
        if run_distogram:
            out["pdistogram"] = pdistogram

    if run_bfactor:
        out["pbfactor"] = bfactor_forward(params, s)

    if run_confidence:
        # Boltz2.forward feeds the first distogram's logits: pdistogram[:,:,:,0].
        pred_distogram_logits = pdistogram[:, :, :, 0]
        confidence_kwargs = dict(
            eps=eps,
            use_scan=bool(sample_kwargs.get("use_scan", True)),
            chunk_size=int(sample_kwargs.get("chunk_size", 128)),
            triangle_attention_chunk=sample_kwargs.get("triangle_attention_chunk"),
            triangle_attention_q_chunk=sample_kwargs.get("triangle_attention_q_chunk"),
            transition_hidden_chunk=sample_kwargs.get("transition_hidden_chunk"),
            matmul_precision=str(sample_kwargs.get("matmul_precision", "highest")),
            attention_backend=str(sample_kwargs.get("attention_backend", "xla")),
            triangle_backend=str(sample_kwargs.get("triangle_backend", "cueq")),
            glu_backend=str(sample_kwargs.get("glu_backend", "xla")),
            return_pair_chains_iptm=return_pair_chains_iptm,
            recompute_nonpolymer_frames=recompute_nonpolymer_frames,
            atom_context_parallel=bool(
                sample_kwargs.get("atom_context_parallel", False)
            ),
        )

        def confidence_call(x_pred: jnp.ndarray, conf_multiplicity: int):
            return confidence_module_forward(
                params["confidence"],
                s_inputs=s_inputs,
                s=s,
                z=z,
                x_pred=x_pred,
                feats=feats,
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=conf_multiplicity,
                **confidence_kwargs,
            )

        if confidence_sequentially and multiplicity > 1:
            # Match upstream production inference: keep diffusion samples
            # parallel, but process the O(N^2) confidence stack one sample at
            # a time. lax.map preserves full JIT compatibility without the
            # multiplicity-sized pair activation peak.
            conf = jax.lax.map(
                lambda coords: confidence_call(coords[None], 1),
                sample_atom_coords,
            )
            conf = jax.tree.map(lambda value: jnp.squeeze(value, axis=1), conf)
        else:
            conf = confidence_call(sample_atom_coords, multiplicity)
        if not return_confidence_logits:
            conf = {
                key: value for key, value in conf.items() if not key.endswith("_logits")
            }
        out.update(conf)

    if affinity_params is not None and "modules" in affinity_params:
        pad_mask = feats["token_pad_mask"][0].astype(z.dtype)
        rec_mask = (feats["mol_type"][0] == 0).astype(z.dtype) * pad_mask
        lig_mask = (feats["affinity_token_mask"][0] != 0).astype(z.dtype) * pad_mask
        cross_pair_mask = (
            lig_mask[:, None] * rec_mask[None, :]
            + rec_mask[:, None] * lig_mask[None, :]
            + lig_mask[:, None] * lig_mask[None, :]
        )
        z_affinity = z * cross_pair_mask[None, :, :, None]
        if multiplicity > 1:
            if "iptm" not in out:
                raise ValueError(
                    "multi-sample affinity requires confidence output to select "
                    "the best ipTM sample"
                )
            best_idx = jnp.argmax(out["iptm"])
        else:
            best_idx = jnp.asarray(0, dtype=jnp.int32)
        coords_affinity = jnp.take(sample_atom_coords, best_idx, axis=0)[None, None]
        affinity_s_inputs = input_embedder_forward(
            params["trunk"]["input_embedder"],
            feats,
            eps=eps,
            attention_backend=str(sample_kwargs.get("attention_backend", "xla")),
            affinity=True,
        )
        module_outputs = [
            affinity_module_forward(
                module_params,
                s_inputs=affinity_s_inputs,
                z=z_affinity,
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
                eps=eps,
            )
            for module_params in affinity_params["modules"]
        ]
        values = [module_out["affinity_pred_value"] for module_out in module_outputs]
        probabilities = [
            jax.nn.sigmoid(module_out["affinity_logits_binary"])
            for module_out in module_outputs
        ]
        affinity_value = sum(values) / len(values)
        if affinity_mw_correction:
            affinity_value = (
                1.03525938 * affinity_value
                - 0.59992683 * feats["affinity_mw"][0] ** 0.3
                + 2.83288489
            )
        out["affinity_pred_value"] = affinity_value
        out["affinity_probability_binary"] = sum(probabilities) / len(probabilities)
        for index, (value, probability) in enumerate(
            zip(values, probabilities, strict=True), start=1
        ):
            out[f"affinity_pred_value{index}"] = value
            out[f"affinity_probability_binary{index}"] = probability
    elif affinity_params is not None:
        # Backward compatibility for native bundles exported before ensemble
        # support. New exports always use the complete ``modules`` structure.
        aff = affinity_module_forward(
            affinity_params,
            s_inputs=s_inputs,
            z=z,
            x_pred=sample_atom_coords,
            feats=feats,
            multiplicity=multiplicity,
            eps=eps,
        )
        out.update(aff)

    return out
