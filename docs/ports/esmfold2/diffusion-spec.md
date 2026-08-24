# ESMFold2 structure head: what the torch source actually does

Companion to `trunk-spec.md`, read off the same vendored source. These are the
parts where a plausible implementation is a wrong one.

## The sampler is not the EDM ODE it looks like

`DiffusionStructureHead.sample`:

* schedule is Karras with `p=8`, `sigma_data * base^p`, then a terminal 0 --
  **and then clipped**: rows above `max_inference_sigma=256` are dropped and
  256 is prepended. 68 nominal steps become **48 denoiser calls**.
* solver is **first-order Euler**, one denoiser call per step, no Heun
  correction.
* `t_hat = sigma * (1 + gamma)` with `gamma = 0.605` above `sigma > 1.107`.
  With `noise_scale = 0` the churn adds no noise, yet the network is still
  queried at the inflated sigma and the step is taken from it.
* between the denoiser call and the step, `x_noisy` is **Kabsch-aligned onto
  `x_denoised`** (`_weighted_rigid_align`, fp32, gesvd). Easy to miss, and
  nothing downstream reveals its absence.
* every step re-centres and applies a **random rotation and translation** to
  the state.

RNG order per step, for anyone chasing step-for-step parity: rotation
`randn(B*S,4)`, translation `randn(B*S,1,3)`, churn `randn_like(x)` -- the
last is drawn even though it is multiplied by zero.

## Sample batching has no sample axis

`num_diffusion_samples` is `repeat_interleave(S, dim=0)` on the batch axis,
laid out `[b0s0, b0s1, ..., b1s0, ...]`. The pair tensor is expanded lazily
inside `AttentionPairBias`.

## Traps

* `F.rms_norm(..., eps=None)` resolves to `finfo(dtype).eps` -- 7.8e-3 in
  bfloat16, 1.2e-7 in float32. `qk_norm` and every atom-block adaLN use it, so
  the trunk dtype changes the arithmetic, not just the rounding.
* atom-block adaLN chunks its modulation as `shift, scale, gate` per branch
  (attention then feed-forward), not `scale, shift`.
* `AdaptiveLayerNorm` normalises the conditioning with `s_scale` and no bias,
  but the output gates consume **raw `s`**.
* RoPE uses the half-split convention: `cos` is `concat([c, c])`, pairing
  `(j, j+16)`, not interleaved. `build_3d_rope`'s signature defaults are dead
  -- the call site passes `(2, 10, 20.0, 10000.0)`, which fills `half_dim=16`
  exactly, and the axes are the *reference conformer* coordinates plus
  `ref_space_uid`.
* the sliding window is `swa_window_size // 2 = 64` each side, measured in
  **packed rank** (padding atoms do not consume window), with the diagonal
  force-allowed.
* `AttentionPairBias` softmaxes over `dim=-2`; `beta` is inert on the
  reference path; `pred_r1` is always zeros so `coords_linear`'s last three
  columns are dead weight.
* `ConfidenceHead` computes `pair = pair + folding_trunk(pair)` where the
  trunk output *already* contains `pair` -- the effective result is
  `2*pair_in + updates`. Reproduce it, do not fix it.
* `ConfidenceHead.s_norm`, `s_inputs_to_single`, `s_input_to_s` and
  `s_trunk` throughout the diffusion path are constructed, keyed in the
  checkpoint, and never used.
* pLDDT bins span **0 to 1**, not 0 to 100. pTM/ipTM come from the PAE
  logits, not a head of their own.
* `FourierEmbedding.w/b` and `ConfidenceHead.boundaries` are persistent
  buffers: load them, do not redraw them.

## Managed assets

Upstream ESMFold2 supports proteins, DNA, RNA, ligands, and modified residues.
FoldJAX currently exposes only its protein input path because the complete
upstream all-biomolecule feature builder has not yet been ported to NumPy/JAX.
Those supported protein features get their canonical atom names and reference
coordinates from the chemistry tables carried in the package, so neither
featurization nor prediction reads the publisher's 417 MB `ccd.pkl`.

The managed store has two explicit profiles:

* `released` (default) stages the structure checkpoint and complete ESMC-6B;
* `structure-only` stages `model.safetensors` and `config.json` only (940 MB).
  It supports either `no_language_model=true` or a released-model run whose
  ESMC checkpoint is supplied explicitly with `esmc_weights`.

A complete release satisfies both profiles; a structure-only install cannot
satisfy a default prediction. `foldjax predict --profile structure-only` is the
first-class structure-network-only request and sets `no_language_model=true`
automatically. Request resolution also selects the smaller managed bundle from
the legacy native option, or from an explicit external `esmc_weights`, when no
structure weights were supplied. The default path fails with the complete-bundle
fetch instruction instead of silently changing the model.
Porting upstream ligand and nucleic-acid support remains a separate
feature-building task; merely downloading the old CCD pickle would not make
those inputs valid.
