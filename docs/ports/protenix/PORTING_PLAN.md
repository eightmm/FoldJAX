# Protenix JAX Porting Plan

## Reference

- Source implementation: `../protenix/protenix`
- JAX port: `../protenix_jax/src/protenix_jax`
- First reference model class: `protenix.model.protenix.Protenix`
- First inference modules:
  - `InputFeatureEmbedder`
  - `RelativePositionEncoding`
  - `TemplateEmbedder`
  - `MSAModule`
  - `PairformerStack`
  - `DiffusionModule`
  - `DistogramHead`
  - `ConfidenceHead`

## Port Order

1. Checkpoint inspection and state_dict mapping helpers.
2. Primitives: linear, layer norm, transition, attention helpers.
3. Heads with small contracts: distogram first, confidence later.
4. Embedders and relative position encoding.
5. Pairformer blocks and MSA stack.
6. Diffusion conditioning, atom attention, diffusion transformer.
7. Static-shape sampling loop.
8. Full inference wrapper and benchmark scripts.

## Gates

- Every ported module needs a focused parity test.
- Checkpoint key and shape assumptions must be explicit.
- Use static shapes in JAX benchmarks and separate compile time from steady
  latency.
- Record peak VRAM after warmup once full sampling exists.
- Structure-level checks must include RMSD and clash sanity.

## 2026-06-17 CPU-only bootstrap

Completed the first dependency-light JAX port layer:

- checkpoint/state_dict inspection CLI
- PyTorch-layout `linear`
- `LayerNorm`
- `Transition`
- `AdaptiveLayerNorm`
- `FourierEmbedding`
- `RelativePositionEncoding.generate_relp` equivalent
- `DistogramHead`

All coverage is CPU-only and checkpoint-free so far. The tests lock reference
formulas and state_dict key contracts before real checkpoint parity is added.

## 2026-06-17 Checkpoint cache

Downloaded and CPU-verified:

- `../protenix/checkpoint/protenix_base_default_v1.0.0.pt`
- size: 1,475,950,125 bytes
- payload: top-level `model`
- state keys: 4,174 after stripping DDP `module.` prefix

The now-removed checkpoint loader handled Protenix `model` payloads, Lightning
`state_dict` payloads, plain state_dict payloads, and all-key `module.` prefix
normalization.

## 2026-06-19 Checkpoint mapping gate

Added the now-removed checkpoint mapping gate. It loaded the real checkpoint on CPU and
checks all currently ported mapping prefixes:

- `distogram_head`
- `relative_position_encoding`
- `linear_no_bias_sinit`
- `input_embedder.atom_attention_encoder` cache projections
- `input_embedder.atom_attention_encoder.atom_transformer.diffusion_transformer`
- root trunk initialization/recycling projections
- `confidence_head.pairformer_stack`
- `confidence_head.pairformer_stack.blocks.0`
- `confidence_head`
- `confidence_head.pairformer_stack.blocks.0.pair_transition`
- `diffusion_module.diffusion_conditioning.fourier_embedding`
- `diffusion_module.diffusion_transformer.blocks.0.conditioned_transition_block.adaln`
- `confidence_head.pairformer_stack.blocks.0.attention_pair_bias`
- `confidence_head.pairformer_stack.blocks.0.tri_mul_out`
- `confidence_head.pairformer_stack.blocks.0.tri_mul_in`
- `confidence_head.pairformer_stack.blocks.0.tri_att_start`
- `confidence_head.pairformer_stack.blocks.0.tri_att_end`
- `diffusion_module.diffusion_transformer.blocks.0.conditioned_transition_block`

Current real-checkpoint output shape summary:

- `distogram.linear.weight`: `(64, 128)`
- `relative_position.linear.weight`: `(128, 139)`
- `sinit.weight`: `(384, 449)`
- `input_atom_cache.linear_ref_pos.weight`: `(128, 3)`
- `input_atom_cache.linear_f.weight`: `(128, 385)`
- `input_atom_cache.linear_d.weight`: `(16, 3)`
- `input_atom_encoder.linear_cl.weight`: `(16, 128)`
- `input_atom_encoder.small_mlp.linear_1.weight`: `(16, 16)`
- `input_atom_encoder.linear_q.weight`: `(384, 128)`
- `input_feature_embedder.atom_linear_q.weight`: `(384, 128)`
- `input_feature_embedder.has_esm`: `(0,)`
- `input_atom_transformer_block.attention.linear_q.weight`: `(128, 128)`
- `input_atom_transformer_block.attention.linear_z.weight`: `(4, 16)`
- `input_atom_transformer_block.transition.linear_a1.weight`: `(256, 128)`
- `input_atom_transformer_stack.num_blocks`: `(3,)`
- `trunk.initial.zinit1.weight`: `(128, 384)`
- `trunk.initial.token_bond.weight`: `(128, 1)`
- `trunk.recycling.linear_z.weight`: `(128, 128)`
- `trunk.recycling.linear_s.weight`: `(384, 384)`
- `confidence_head.linear_s1.weight`: `(128, 449)`
- `confidence_head.distance.lower_bins`: `(39,)`
- `confidence_head.distance.linear_d.weight`: `(128, 39)`
- `confidence_head.output.linear_pae.weight`: `(64, 128)`
- `confidence_head.output.plddt_weight`: `(24, 384, 50)`
- `confidence_head.output.resolved_weight`: `(24, 384, 2)`
- `confidence_pair_transition.linear_a.weight`: `(512, 128)`
- `confidence_pairformer_block.tri_mul_out.linear_a_p.weight`: `(128, 128)`
- `confidence_pairformer_block.tri_att_start.linear.weight`: `(4, 128)`
- `confidence_pairformer_block.single_transition.linear_out.weight`: `(384, 1536)`
- `confidence_pairformer_stack.num_blocks`: `(4,)`
- `confidence_pairformer_stack.block0.tri_mul_out.linear_a_p.weight`: `(128, 128)`
- `confidence_attention_pair_bias.attention.linear_q.weight`: `(384, 384)`
- `confidence_attention_pair_bias.linear_z.weight`: `(16, 128)`
- `confidence_tri_mul_out.linear_a_p.weight`: `(128, 128)`
- `confidence_tri_mul_in.linear_z.weight`: `(128, 128)`
- `confidence_tri_att_start.linear.weight`: `(4, 128)`
- `confidence_tri_att_end.mha.linear_q.weight`: `(128, 128)`
- `diffusion_fourier.w`: `(256,)`
- `diffusion_transition_adaln.linear_s.weight`: `(768, 384)`
- `diffusion_conditioned_transition.linear_a1.weight`: `(1536, 768)`
- `diffusion_conditioned_transition.linear_b.weight`: `(768, 1536)`

## 2026-06-19 Attention leaf path

Added CPU-tested JAX implementations for:

- `Attention.prepare_qkv`
- standard full attention with pair bias and gating
- standard/global `AttentionPairBias` for the Pairformer/Confidence path

This intentionally excludes local atom attention/windowing for now. The global
path is the next useful base for Pairformer and confidence module parity.

## 2026-06-19 Diffusion transformer leaf path

Added CPU-tested JAX implementation and checkpoint mapping for:

- `ConditionedTransitionBlock`
- Protenix keys: `linear_nobias_a1`, `linear_nobias_a2`,
  `linear_nobias_b`, `linear_s`

The real v1 and v2 checkpoint gates confirm the expected diffusion transformer
block shapes. `protenix-v2.pt` was downloaded from a Hugging Face mirror because
the upstream ByteDance CDN returned HTTP 403 from this environment.

Downloaded and CPU-verified:

- `../protenix/checkpoint/protenix-v2.pt`
- size: 1,859,785,497 bytes
- sha256:
  `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`
- payload: top-level `model`
- state keys: 4,174

The v2 checkpoint keeps the same state key count but increases several pair
widths, for example `distogram.linear.weight` is `(64, 256)` and
`relative_position.linear.weight` is `(256, 139)`.

## 2026-06-19 Triangle multiplication leaf path

Added CPU-tested JAX implementation and checkpoint mapping for:

- `TriangleMultiplicationOutgoing`
- `TriangleMultiplicationIncoming`
- Protenix keys: `layer_norm_in`, `layer_norm_out`, `linear_a_p`,
  `linear_a_g`, `linear_b_p`, `linear_b_g`, `linear_z`, `linear_g`

The implementation follows the AF3/Boltz-JAX split of projection/gating and
triangle contraction, with an optional output-row chunk path for lower peak
memory. The real v1 and v2 checkpoint gates confirm confidence Pairformer
shapes:

- v1: `confidence_tri_mul_out.linear_a_p.weight` = `(128, 128)`
- v2: `confidence_tri_mul_out.linear_a_p.weight` = `(256, 256)`

## 2026-06-19 Triangle attention leaf path

Added CPU-tested JAX implementation and checkpoint mapping for:

- `TriangleAttentionStartingNode`
- `TriangleAttentionEndingNode`
- Protenix keys: `layer_norm`, triangle bias `linear`, and `mha.linear_*`

The implementation follows Protenix's dense torch path and keeps an optional
query-axis chunk path for lower attention logits memory. The real v1 and v2
checkpoint gates confirm confidence Pairformer shapes:

- v1: `confidence_tri_att_start.linear.weight` = `(4, 128)`
- v2: `confidence_tri_att_start.linear.weight` = `(8, 256)`

## 2026-06-19 Pairformer block composition

Added CPU-tested JAX composition and checkpoint mapping for one inference-mode
`PairformerBlock` in the confidence stack:

- `tri_mul_out`
- `tri_mul_in`
- `tri_att_start`
- transposed `tri_att_end`
- `pair_transition`
- optional single path: `attention_pair_bias`, `single_transition`

Dropout and in-place mutation are intentionally omitted for inference. The block
uses the same AF3/Boltz-JAX memory hooks already present in the leaves:
triangle multiplication output-row chunking and triangle attention query-axis
chunking. The real v1 and v2 checkpoint gates confirm block-level mapping.

## 2026-06-19 Pairformer stack scan path

Added CPU-tested JAX composition and checkpoint mapping for the full confidence
`PairformerStack` module list. Both Protenix v1 and v2 checkpoints contain
four confidence pairformer blocks.

The stack supports:

- `use_scan=False`: Python-unrolled blocks for easier debugging and lower
  steady serving overhead on small stacks.
- `use_scan=True`: stacked parameter pytree with `jax.lax.scan`, matching the
  compile-time reduction pattern used in Boltz-JAX and AF3.

Static config leaves such as `AttentionPairBias.has_s` are handled separately
from array leaves so `lax.scan` only scans tensors.

## 2026-06-19 Confidence head single-sample path

Added CPU-tested JAX implementation and checkpoint mapping for the infer-only
single-sample confidence head path:

- distance one-hot embedding with checkpoint `lower_bins` / `upper_bins`
- scalar distance projection
- initial pair update from `s_inputs`
- confidence `PairformerStack`
- PAE and PDE logits
- atom-level pLDDT and resolved logits

The implementation follows the Protenix `memory_efficient_forward` path and
keeps the outer sample loop out of scope for now. Real v1 and v2 checkpoint
gates confirm both width variants:

- v1: `confidence_head.distance.linear_d.weight` = `(128, 39)`
- v2: `confidence_head.distance.linear_d.weight` = `(256, 39)`
- v1/v2: `confidence_head.output.plddt_weight` = `(24, 384, 50)`

## 2026-06-19 Trunk initialization path

Added CPU-tested JAX implementation and checkpoint mapping for the root-level
initialization in `Protenix.get_pairformer_output`:

- `s_init = linear_no_bias_sinit(s_inputs)`
- `z_init` from `linear_no_bias_zinit1/2`
- relative position encoding addition
- token bond projection addition
- optional constraint addition
- no-dropout recycling projections for `s` and `z`

Real v1 and v2 checkpoint gates confirm both pair-width variants:

- v1: `trunk.initial.zinit1.weight` = `(128, 384)`
- v2: `trunk.initial.zinit1.weight` = `(256, 384)`
- v1/v2: `trunk.recycling.linear_s.weight` = `(384, 384)`

## 2026-06-19 Atom local layout and cache path

Added CPU-tested JAX implementation and checkpoint mapping for the first
AtomAttentionEncoder layer:

- local q/k trunking for atom windows
- token-to-atom broadcast
- atom-to-token sum/mean aggregation
- token-pair to local atom-pair gather
- base AtomAttentionEncoder cache preparation for `p_lm` and `c_l`
- atom pair conditioning helper before the atom transformer

The implementation keeps local window tensors in blocked dense form, matching
the Protenix/Boltz-JAX memory pattern for local atom attention. The atom
transformer blocks themselves are still out of scope for this checkpoint.

Real v1 and v2 checkpoint gates confirm shared input atom cache shapes:

- `input_atom_cache.linear_ref_pos.weight` = `(128, 3)`
- `input_atom_cache.linear_f.weight` = `(128, 385)`
- `input_atom_cache.linear_d.weight` = `(16, 3)`

## 2026-06-19 AtomTransformer local block path

Added CPU-tested JAX implementation and checkpoint mapping for the local
AtomTransformer block path:

- local blocked attention over atom windows
- adaptive local `AttentionPairBias`
- inference-mode `DiffusionTransformerBlock`
- `DiffusionTransformerStack` with Python loop and optional `jax.lax.scan`
- mapper for the input AtomTransformer's three local diffusion blocks

The local attention keeps q/k/v in blocked dense form and applies the window
mask before softmax, matching the Protenix local-cross-attention path while
leaving optimized Pallas/Triton-style kernels for a later backend pass.

Real v1 and v2 checkpoint gates confirm shared input atom transformer shapes:

- `input_atom_transformer_block.attention.linear_q.weight` = `(128, 128)`
- `input_atom_transformer_block.attention.linear_z.weight` = `(4, 16)`
- `input_atom_transformer_block.transition.linear_a1.weight` = `(256, 128)`
- `input_atom_transformer_stack.num_blocks` = `(3,)`

## 2026-06-19 Input AtomAttentionEncoder path

Added CPU-tested JAX implementation and checkpoint mapping for
`AtomAttentionEncoder(has_coords=False)`, the path used by `InputFeatureEmbedder`:

- atom cache preparation for `p_lm` and `c_l`
- atom-pair conditioning from local `c_l` q/k windows
- three-layer small atom-pair MLP
- local AtomTransformer stack
- atom q projection
- atom-to-token mean aggregation

This is still a tensor-level port, not the full `InputFeatureEmbedder` wrapper.
The wrapper needs the surrounding token feature concatenation and optional ESM
projection next.

Real v1 and v2 checkpoint gates confirm shared full input atom encoder shapes:

- `input_atom_encoder.linear_cl.weight` = `(16, 128)`
- `input_atom_encoder.small_mlp.linear_1.weight` = `(16, 16)`
- `input_atom_encoder.linear_q.weight` = `(384, 128)`

## 2026-06-19 InputFeatureEmbedder wrapper

Added CPU-tested JAX implementation and checkpoint mapping for the
`InputFeatureEmbedder` wrapper:

- calls `AtomAttentionEncoder(has_coords=False)`
- concatenates token features: atom encoder output, `restype`, `profile`,
  `deletion_mean`
- supports optional ESM projection when `linear_esm` is present in checkpoint
- maps the full `input_embedder` prefix

Real v1 and v2 checkpoint gates confirm the default checkpoints do not include
ESM weights:

- `input_feature_embedder.atom_linear_q.weight` = `(384, 128)`
- `input_feature_embedder.has_esm` = `(0,)`

## 2026-06-19 ConstraintEmbedder optional path

Added CPU-tested JAX implementation and checkpoint mapping for the optional
`ConstraintEmbedder` path used before trunk recycling:

- pocket pair projection
- token contact pair projection
- atom contact pair projection
- inference-mode substructure MLP projection without dropout
- absent-weight-safe state_dict mapper for default checkpoints

The real v1 and v2 checkpoints currently contain no `constraint_embedder.*`
weights, so the mapping gate records this explicitly instead of failing:

- `constraint_embedder.has_pocket` = `(0,)`
- `constraint_embedder.has_contact` = `(0,)`
- `constraint_embedder.has_contact_atom` = `(0,)`
- `constraint_embedder.has_substructure` = `(0,)`

## 2026-06-19 Root PairformerStack mapping

Extended the real checkpoint gate to cover the root `pairformer_stack` used by
the main trunk. This reuses the existing inference-mode Pairformer block/stack
implementation and verifies that Protenix v1/v2 expose the same block layout as
the confidence stack:

- v1: `root_pairformer_block.tri_mul_out.linear_a_p.weight` = `(128, 128)`
- v2: `root_pairformer_block.tri_mul_out.linear_a_p.weight` = `(256, 256)`
- v1/v2: `root_pairformer_block.single_transition.linear_out.weight` =
  `(384, 1536)`
- v1/v2: `root_pairformer_stack.num_blocks` = `(48,)`

## 2026-06-19 MSAModule trunk path

Added CPU-tested JAX implementation and checkpoint mapping for the MSA trunk
path used inside Protenix recycling:

- `OuterProductMean`
- `MSAPairWeightedAveraging`
- MSA transition
- `MSABlock` composition with Pairformer pair stack
- `MSAModule` feature embedding from `msa`, `has_deletion`, `deletion_value`
- full `msa_module` state_dict mapper over all four MSA blocks

The implementation keeps Protenix as the numerical/key reference while using
the Boltz-JAX style pure-function parameter tree. MSA feature sampling is not
duplicated here; this path expects already materialized/static inference
features.

Real v1 and v2 checkpoint gates confirm both width variants:

- v1: `msa_module.linear_m.weight` = `(64, 34)`
- v2: `msa_module.linear_m.weight` = `(128, 34)`
- v1/v2: `msa_module.num_blocks` = `(4,)`
- v1/v2: `msa_module.last_block.has_msa_stack` = `(0,)`

## 2026-06-19 TemplateEmbedder trunk path

Added CPU-tested JAX implementation and checkpoint mapping for Protenix
`TemplateEmbedder`:

- template pair-feature construction in Protenix order:
  distogram, pseudo-beta mask, residue type broadcasts, unit vector, backbone
  frame mask
- multichain and pair-mask application
- single-template pair-only Pairformer stack path
- averaging over templates
- final ReLU and projection back to trunk pair width
- full `template_embedder` state_dict mapper

The pair stack reuses the existing `PairformerStack(has_s=False)` path. Real v1
and v2 checkpoint gates confirm the configured two template blocks:

- v1: `template.linear_z.weight` = `(64, 128)`
- v2: `template.linear_z.weight` = `(64, 256)`
- v1/v2: `template.linear_a.weight` = `(64, 108)`
- v1/v2: `template.pairformer_stack.num_blocks` = `(2,)`

## 2026-06-19 Pairformer output wrapper

Added CPU-tested JAX orchestration for Protenix `get_pairformer_output` after
`InputFeatureEmbedder`:

- builds `s_init` and `z_init`
- applies optional constraint pair features
- runs no-dropout recycling projections
- adds template updates
- runs MSA updates
- runs the root 48-block Pairformer stack
- returns `(s_inputs, s, z)`

This is the first trunk-level composition point. It still expects `s_inputs`
from the already ported `InputFeatureEmbedder`; a later wrapper will connect
the atom input embedder directly once full feature fixtures are available.

Real v1 and v2 checkpoint gates now cover the grouped trunk mapper:

- `pairformer_output.trunk.sinit.weight` = `(384, 449)`
- v1: `pairformer_output.template.linear_u.weight` = `(128, 64)`
- v2: `pairformer_output.template.linear_u.weight` = `(256, 64)`
- `pairformer_output.msa.num_blocks` = `(4,)`
- `pairformer_output.root_pairformer.num_blocks` = `(48,)`

## 2026-06-19 DiffusionConditioning

Added CPU-tested JAX implementation and checkpoint mapping for
`DiffusionConditioning`, the first stage of Protenix diffusion scoring:

- pair conditioning cache from trunk `z` plus relative position encoding
- pair transition updates
- single conditioning from trunk `s` plus `s_inputs`
- Fourier noise embedding with Protenix `log(sigma / sigma_data) / 4`
- single transition updates
- reusable `pair_z` cache path

Real v1 and v2 checkpoint gates confirm both pair-width variants:

- v1: `diffusion_conditioning.linear_z.weight` = `(128, 256)`
- v2: `diffusion_conditioning.linear_z.weight` = `(256, 512)`
- v1/v2: `diffusion_conditioning.linear_s.weight` = `(384, 833)`
- v1/v2: `diffusion_conditioning.linear_n.weight` = `(384, 256)`

## 2026-06-19 Diffusion AtomAttentionEncoder coordinates path

Extended the atom encoder from the input-only `has_coords=False` path to the
diffusion `has_coords=True` path:

- trunk single injection through `layernorm_s` and `linear_no_bias_s`
- trunk pair injection through `layernorm_z`, `linear_no_bias_z`, and local
  atom-pair broadcast
- noisy coordinate injection through `linear_no_bias_r`
- sample-leading atom layout support for pair conditioning and local atom
  transformer calls
- checkpoint mapping for `diffusion_module.atom_attention_encoder`

The base input encoder mapping remains unchanged. The diffusion path follows
the Protenix module contract while keeping the JAX implementation function-only
and static-windowed like the Boltz-JAX/AF3-style local attention path.

Real v1 and v2 checkpoint gates confirm:

- v1: `diffusion_atom_encoder.linear_z.weight` = `(16, 128)`
- v2: `diffusion_atom_encoder.linear_z.weight` = `(16, 256)`
- v1/v2: `diffusion_atom_encoder.linear_s.weight` = `(128, 384)`
- v1/v2: `diffusion_atom_encoder.linear_r.weight` = `(128, 3)`
- v1/v2: `diffusion_atom_encoder.linear_q.weight` = `(768, 128)`
- v1/v2: `diffusion_atom_encoder.transformer.num_blocks` = `(3,)`

## 2026-06-19 DiffusionModule denoising step wrapper

Added CPU-tested JAX implementation and checkpoint mapping for the infer-only
one-step Protenix denoising network:

- `AtomAttentionDecoder`
- diffusion module single-to-token projection
- 24-block token `DiffusionTransformer` stack mapping
- final token layer norm
- one-step EDM coordinate rescale in `diffusion_module_forward`
- shared-cache path for precomputed `pair_z` and atom encoder `p_lm/c_l`

The wrapper still expects static/precomputed features and does not yet include
the outer stochastic sampling schedule. This is the final model-side denoising
step needed before wiring the sampler loop.

Real v1 and v2 checkpoint gates confirm:

- `diffusion_module.linear_s.weight` = `(768, 384)`
- `diffusion_module.token_transformer.num_blocks` = `(24,)`
- `diffusion_module.layernorm_a.weight` = `(768,)`
- `diffusion_module.atom_decoder.linear_a.weight` = `(128, 768)`
- `diffusion_module.atom_decoder.linear_out.weight` = `(3, 128)`
- `diffusion_module.atom_decoder.transformer.num_blocks` = `(3,)`

## 2026-06-19 Global attention query chunking

Added CPU-tested query-axis chunking for dense/global attention paths:

- `attention(..., q_chunk_size=...)`
- `attention_pair_bias(..., q_chunk_size=...)`
- token/global `DiffusionTransformer` through `global_q_chunk_size`
- root/confidence Pairformer single attention through `single_att_q_chunk_size`
- diffusion denoising wrapper through `token_q_chunk_size`

This reduces peak logits/probability memory for dense token attention while
preserving unchunked numerical output. Atom attention remains local-windowed
through `n_queries/n_keys`; triangle attention and triangle multiplication keep
their existing chunk hooks.

Still not implemented:

- Protenix `blocks_per_ckpt` activation checkpointing
- `enable_efficient_fusion`
- full dynamic chunk policy from Protenix inference configs

## 2026-06-19 Diffusion sampling loop

Added CPU-tested JAX implementation for Protenix inference diffusion sampling:

- `inference_noise_schedule` matching Protenix `InferenceNoiseScheduler`
- Algorithm 18 predictor-corrector Euler loop
- stochastic `gamma0/gamma_min`, `noise_scale_lambda`, and `step_scale_eta`
  controls
- per-step coordinate centering
- `diffusion_chunk_size` sample chunking
- deterministic testing path through injected initial and per-step noise
- `sample_diffusion_with_module` helper for static feature dictionaries and
  `DiffusionModuleParams`

This connects the already ported one-step denoiser to the outer sampler. It
does not yet include training-free guidance (`TFGEngine`) or Protenix's dynamic
chunk-size policy from runtime config.

Still not implemented:

- Protenix `blocks_per_ckpt` activation checkpointing
- `enable_efficient_fusion`
- full dynamic chunk policy from Protenix inference configs
- training-free guidance (`TFGEngine`)

## 2026-06-19 Static top-level inference wrapper

Added CPU-tested orchestration for the currently ported infer-only static
feature path:

- `ProtenixInferenceParams` checkpoint grouping
- `protenix_infer_static` wrapper
- input feature embedding
- trunk/template/MSA/root Pairformer recycling path
- precomputed diffusion conditioning cache
- diffusion sampler with `DiffusionModuleParams`
- distogram logits
- confidence logits: `plddt`, `pae`, `pde`, `resolved`
- basic confidence postprocessing: `contact_probs`, `atom_plddt`,
  `token_pair_pae`, `token_pair_pde`, `summary_plddt`, `summary_gpde`,
  `summary_ptm`, `summary_iptm`, `summary_ranking_score`
- chain confidence postprocessing: `chain_plddt`, `chain_pair_plddt`,
  `chain_gpde`, `chain_pair_gpde`, `chain_pair_pae_mean`,
  `chain_pair_pae_min`
- AF3-style clash postprocessing: `has_clash` and `summary_ranking_score`
  penalty using Protenix's inter-chain count/relative-clash rule
- VDW clash postprocessing: `has_vdw_clash` and
  `summary_ranking_score_vdw_penalized` using Protenix/RDKit radii and
  same-molecule pair skipping
- native weight load path so static inference can run without torch
- `protenix-jax-static-infer` CLI for checkpoint + precomputed static feature
  `.npz` inference, including chunk/scan knobs and compressed `.npz` outputs

The chunk controls now pass through the top-level wrapper:

- atom local-window controls: `n_queries`, `n_keys`
- root Pairformer: `triangle_mul_chunk_size`,
  `triangle_att_q_chunk_size`, `single_att_q_chunk_size`
- diffusion token attention: `token_q_chunk_size`
- diffusion sample batching: `diffusion_chunk_size`
- confidence sample loop: one predicted sample at a time, matching Protenix's
  memory-efficient inference path
- root Pairformer scan: `use_pairformer_scan`
- confidence Pairformer scan: `use_confidence_scan`
- diffusion transformer scan: `use_diffusion_scan`

This follows the Boltz-JAX/AF3-style static parameter grouping and scan/chunk
knobs.

## 2026-06-19 Automatic chunk policy

Added a Protenix-style automatic chunk resolver for static inference:

- token thresholds from `configs_base.py`:
  `<=1024: None`, `<=1536: 512`, `<=2048: 256`, `<=2560: 128`,
  `>2560: 32`
- the resolved token chunk size feeds JAX triangle multiplication,
  triangle attention, root/confidence single attention, and diffusion token
  attention chunks
- diffusion sample chunking defaults to Protenix's inference chunk size of `5`
  when `n_sample > 5`
- explicit CLI chunk arguments override automatic values
- `--chunk-policy manual` preserves raw user-provided values, and
  `--chunk-policy off` disables chunking for debugging

## 2026-06-19 Runtime model limits

Added static-inference runtime checks matching Protenix's public inference
guard:

- `protenix-v2` rejects `n_token > 2560` before checkpoint/native weight load
- `--model-name auto` infers known model names from checkpoint or native weight
  filenames
- explicit `--model-name protenix-v2` handles renamed native weight files

## 2026-06-19 Sequence-only JSON featurizer

Added a first raw-input path:

- `protenix-jax-featurize-json --input input.json --out static_features.npz`
- supports sequence-only `proteinChain` jobs with standard 20AA plus `X`
- supports `pairedMsaPath` and `unpairedMsaPath` A3M inputs for chain-level
  profile, deletion-mean, and row-level `msa`/`has_deletion`/`deletion_value`
  features with strict aligned-length validation
- MSA rows are merged across protein chains by token-axis concatenation with
  gap padding for shorter per-chain MSA depths, and `--max-msa-rows` controls
  inference-time row depth for memory
- `protenix-jax-static-infer` can consume sequence-only JSON directly through
  `--input-json`, or run from a prebuilt static `.npz` through `--features`
- expands `count` into separate chain copies with shared `entity_id` and
  distinct `asym_id`/`sym_id`
- emits the current static wrapper fields: token metadata, restype/profile/MSA,
  dummy reference atom features, local atom pair geometry, relative-position
  features, representative atom masks, and `mol_id`
- rejects unsupported ligands, nucleic acids, covalent bonds, templates,
  modified residues, bad MSA alignment lengths, and unsupported residue letters
  instead of silently fabricating invalid features
- CPU smoke covers both prebuilt JSON-derived `.npz` features and direct JSON
  input flowing into `protenix-jax-static-infer` with toy native weights

It is not yet a full Protenix runtime clone: ligand/nucleic-acid featurization,
species-based paired-MSA row assembly, template ingestion, TFG, and
`enable_efficient_fusion` remain out of scope.

## 2026-06-19 Boltz-JAX-style predict wrapper

Added `protenix_jax.models.predict.protenix_predict_static`, a torch-free
library entry point mirroring Boltz-JAX's `models/predict.py` pattern. The
static inference CLI now calls this wrapper instead of building the noise
schedule and calling the lower-level model function directly. A unit test locks
the wrapper output to the direct `protenix_infer_static` call.

## 2026-06-19 Standalone runtime CLI

Added `protenix-jax-predict`, a native-weight-only inference CLI. It accepts
static `.npz` features or direct sequence-only JSON, and intentionally has no
checkpoint option. The standalone runtime test monkeypatches Python imports and
fails if the native prediction path imports `torch` or upstream `protenix`.

## 2026-06-19 Remove checkpoint conversion from package

Removed the packaged checkpoint loader/exporter path.
The default project now contains the standalone runtime only: JSON/static
features, native weight loading, and prediction output writing. The legacy
`protenix-jax-static-infer` command is now a native-only alias for
`protenix-jax-predict`.

## Chai Note

Chai is not started in this phase. Its public package loads downloaded
TorchScript `.pt` components, so its first phase must inspect TorchScript graph
and weight extractability before a JAX module port can be planned.
