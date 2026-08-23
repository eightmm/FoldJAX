# OpenFold3-JAX

> **Archived port record.** This port now lives inside FoldJAX, at
> `foldjax.models.openfold3`, and is installed by FoldJAX's own `uv sync` —
> there is no separate package or sibling checkout to install. Use the current
> [OpenFold3 guide](../../openfold3.md) and top-level
> [README](../../../README.md) for production instructions; the parity and
> performance discussion below is retained as porting evidence.

A JAX inference port of [OpenFold3](https://github.com/aqlaboratory/openfold-3).

**Status: end-to-end inference runs on the released weights.** The whole latent trunk
— Pairformer stack (AF3 Alg. 17), MSA module stack (AF3 Alg. 8) and template pair
stack (AF2 Alg. 16) — plus the diffusion sampler and confidence heads are ported and
gated against the real upstream torch modules. The released `of3_ft3_v1` checkpoint
loads and predicts: real 1UBQ folds to 2.5 Å CA RMSD from its deposition using the
query sequence alone, and real targets up to 2076 tokens run. Raw-job
featurization and inference use FoldJAX's NumPy/JAX path and import neither
torch nor the publisher runtime.

`PROJECT.md`'s milestone numbering predates all of that and understates where the
port is; the tables below are what has actually been gated.

## Why OpenFold3

Of the co-folding models ported in this workspace, OpenFold3 is the only one that
is Apache 2.0 for **code and weights**, with its training data published. It is
also the same AF3 architecture already ported in `../protenix_jax` and
`../chai_jax`, so the module inventory overlaps heavily and those ports serve as
working references for the shared algorithms.

The model is ~11k lines across 35 files under `openfold3/core/model`. The
remaining ~76k lines of the upstream package are data pipeline, training, loss,
metrics, and tooling that an inference port does not need.

## Process-level settings

Importing `foldjax.models.openfold3` applies allocator defaults with `setdefault`,
so an operator's explicit process settings survive. Numerical precision is scoped
around this model's `predict` call rather than changing JAX globally:

- `jax_default_matmul_precision = "high"` inside prediction — TF32, which is what upstream sets for
  itself (`entry_points/import_utils.py:33`,
  `torch.set_float32_matmul_precision("high")`). This was `"highest"` until
  2026-08-10 on the strength of a GPU comparison against torch — max |error|
  `1.9e-6` pinned against `7.8e-3` unpinned, 78x past the 1e-4 gate — but that
  reference torch ran at *its* default, not at the "high" upstream selects at
  inference, so the two sides were being held to different settings and the port
  was paying for the stricter one. Re-measured end to end at 1,003 tokens: -15%
  warm, no memory change, CA RMSD 0.011 A against a 0.005 A run-twice floor, and
  TM 0.9937 against the deposited structure either way. Re-running that torch
  comparison needs the reference captured under `"high"` on both sides.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MEM_FRACTION=0.90` via `setdefault`,
  so an operator's explicit choice survives.

## Setup

```bash
uv sync --extra cuda13 --extra openfold3-preprocess --group dev
```

The preprocessing extra contains chemistry and data libraries only. No
FoldJAX extra installs PyTorch, Lightning, or TorchMetrics.

## Tests

```bash
# production and NumPy/JAX preprocessing paths
JAX_PLATFORMS=cpu uv run --extra openfold3-preprocess --group dev \
  pytest -q tests/models/openfold3
```

Publisher-reference parity is intentionally run in a separately provisioned
external environment; it is not an installation profile of FoldJAX.

## The parity gate

Every ported module is checked numerically against the real OpenFold3 torch
module at `rtol=atol=1e-4`. `protenix_jax` shipped a genuine model bug that this
kind of test would have caught, so the gate exists from the first module rather
than being retrofitted.

Parameters are **randomized** before every comparison. This matters more than it
sounds: OpenFold3 zero-initializes output and gate projections
(`init="final"`, `init="gating"`), so a default-constructed `Attention`,
`SwiGLUTransition` or `TriangleMultiplicativeUpdate` returns exactly zero. A
parity test on such a module compares 0 against 0 and passes for any port,
correct or not — which is exactly what happened here before it was caught.
`tests/test_parity_gate_is_meaningful.py` now guards against that recurring.

Random values are otherwise sufficient: the gate verifies forward math and
checkpoint key mapping, not trained weights. Tests also pin the upstream
`state_dict` key names the mappers depend on, so an upstream rename fails a test
instead of silently producing wrong parameters.

`torch` belongs only to that external publisher-reference gate. Production
featurization and inference must never import it.

Parity alone is not sufficient: a literal translation of upstream can pass every
numerical test and still be untraceable under `jax.jit` (boolean indexing on a
traced mask is the case that bit this port). Ported code that must run inside a
compiled step carries a jit test as well.

The parity tests put `../openfold3` on `sys.path` rather than installing it. The
import closure of the model leaves is 16 files needing only torch, ml_collections,
gemmi, biotite, scipy and numpy, whereas installing the package would pull in
lmdb, pdbeccdutils, awscli and the rest of the data stack.

## Ported so far

| module | upstream | parity |
|---|---|---|
| `Linear` | `core/model/primitives/linear.py` | yes, with and without bias |
| `LayerNorm` | `core/model/primitives/normalization.py` | yes, all scale/offset combinations |
| `AdaLN` (AF3 Alg. 26) | `core/model/primitives/normalization.py` | yes |
| `SwiGLU` | `core/model/primitives/activations.py` | yes |
| `SwiGLUTransition` (AF3 Alg. 11) | `core/model/layers/transition.py` | yes, masked and unmasked |
| `Attention` | `core/model/primitives/attention.py` | yes, gated/ungated, with biases and extra batch dims |
| `TriangleMultiplicativeUpdate` (AF3 Alg. 12/13) | `core/model/layers/triangular_multiplicative_update.py` | yes, outgoing and incoming, masked and unmasked |
| `TriangleAttention` (AF3 Alg. 14/15) | `core/model/layers/triangular_attention.py` | yes, starting and ending node, masked and unmasked |
| `AttentionPairBias` (AF3 Alg. 24, trunk) | `core/model/layers/attention_pair_bias.py` | yes |
| `PairBlock` | `core/model/latent/base_blocks.py` | yes, incl. the `_mask_trans` option |
| `PairFormerBlock` / `PairFormerStack` (AF3 Alg. 17) | `core/model/latent/pairformer.py` | yes, 1 and 3 blocks |
| `OuterProductMean` (AF3 Alg. 9) | `core/model/layers/outer_product_mean.py` | yes, incl. fully-masked normalization |
| `MSAPairWeightedAveraging` (AF3 Alg. 10) | `core/model/layers/msa.py` | yes, masked and unmasked |
| `MSAModuleBlock` / `MSAModuleStack` (AF3 Alg. 8) | `core/model/latent/msa_module.py` | yes, both `opm_first` orderings, 1 and 3 blocks, incl. the final block's skipped MSA update |
| `TemplatePairBlock` / `TemplatePairStack` (AF2 Alg. 16) | `core/model/latent/template_module.py` | yes, both `tri_mul_first` orderings, 1 and 2 blocks |
| Sequence-local atom blocking (AF3 Alg. 7) | `core/utils/atom_attention_block_utils.py` | yes, incl. underflow/overflow window shifts and ragged batches |
| `RefAtomFeatureEmbedder` (AF3 Alg. 5) | `core/model/layers/sequence_local_atom_attention.py` | yes, incl. arcsinh charge and cross-conformer pair gating |
| `AtomAttentionEncoder` (AF3 Alg. 5) | `core/model/layers/sequence_local_atom_attention.py` | yes, input and diffusion paths, all four outputs |
| `AtomAttentionDecoder` (AF3 Alg. 6) | `core/model/layers/sequence_local_atom_attention.py` | yes |
| `atom_transformer` (cross-attention stack) | `core/model/layers/diffusion_transformer.py` | yes, via the encoder and decoder gates |
| Atom pair conditioning | `core/model/layers/sequence_local_atom_attention.py` | yes, incl. the leading-ReLU pair MLP and its odd Sequential indices |
| `broadcast_token_feat_to_atoms` | `core/utils/atomize_utils.py` | yes, against upstream's `repeat_interleave` path |
| `convert_pair_rep_to_blocks` | `core/utils/atom_attention_block_utils.py` | yes, several token/atom count splits |
| `NoisyPositionEmbedder` (AF3 Alg. 5) | `core/model/layers/sequence_local_atom_attention.py` | yes, all three outputs |
| `ConditionedTransitionBlock` (AF3 Alg. 25) | `core/model/layers/transition.py` | yes, masked and unmasked |
| `AttentionPairBias` (AdaLN/diffusion variant) | `core/model/layers/attention_pair_bias.py` | yes |
| `DiffusionTransformerBlock` / `DiffusionTransformer` | `core/model/layers/diffusion_transformer.py` | yes, self-attention config, 1 and 3 blocks |
| `CrossAttentionPairBias` (sequence-local) | `core/model/layers/attention_pair_bias.py` | yes, plain and AdaLN forms, ragged atom counts |
| Trunk with recycling (AF3 Alg. 1) | `projects/of3_all_atom/model.py` | composition and recycling behaviour, against the gated sub-modules |
| `InputEmbedderAllAtom` (AF3 Alg. 2) | `core/model/feature_embedders/input_embedders.py` | yes, all three outputs, incl. the asymmetric pair construction |
| MSA input embedding (AF3 Alg. 8) | `core/model/feature_embedders/input_embedders.py` | yes, projection path; subsampling excluded as training-only |
| `relpos_complex` (AF3 Alg. 3) | `core/utils/relpos.py` | yes, several chain/entity layouts, incl. the out-of-condition bin |
| Noise/single conditioning (AF3 Alg. 21) | `core/model/layers/diffusion_conditioning.py` | yes, incl. the Fourier buffers and concat order |
| `centre_random_augmentation` (AF3 Alg. 19) | `core/model/structure/augmentation.py` | `quat_to_rot` against upstream, plus rigid-transform invariants |
| EDM sampler rollout (AF3 Alg. 18) | `core/model/structure/diffusion_module.py` | schedule arithmetic, gamma gating, determinism |
| Denoiser body (AF3 Alg. 20) | `core/model/structure/diffusion_module.py` | composition order, against the gated sub-modules |
| EDM noise schedule + conditioning (AF3 p. 24) | `core/model/structure/diffusion_module.py` | yes, four `(steps, s_max, s_min, p)` settings plus limit behaviour |
| PAE / PDE / distogram heads | `core/model/heads/prediction_heads.py` | yes, incl. the symmetrization and layer-norm differences between them |
| pLDDT / experimentally-resolved heads | `core/model/heads/prediction_heads.py` | yes, several per-token atom-count splits |
| `aggregate_atom_feat_to_tokens` | `core/utils/atomize_utils.py` | yes, mean and sum, masked atoms and empty tokens |
| `max_atom_per_token_masked_select` | `core/utils/atomize_utils.py` | yes, against upstream's `masked_select` path |
| Confidence metrics: pLDDT, pTM, ipTM (AF3 SI 5.7/5.9.1) | `core/metrics/confidence.py` | yes, incl. bin-center convention and the ipTM chain exclusion |
| Inter-chain clash detection | `core/metrics/sample_ranking.py` | yes, incl. mask/polymer filtering and both violation criteria |
| Per-chain pTM (AF3 SI 5.9.3) | `core/metrics/sample_ranking.py` | yes, incl. absent-chain handling, jit-checked |
| Chain-pair ipTM matrix (AF3 SI 5.9.3) | `core/metrics/sample_ranking.py` | yes, incl. symmetry, zero diagonal and absent chains |
| Chain-mean ipTM (AF3 SI 5.9.3) | `core/metrics/sample_ranking.py` | yes, incl. the asymmetric row/column frame filtering |
| Bespoke (ligand-aware) ipTM (AF3 SI 5.9.3) | `core/metrics/sample_ranking.py` | yes, incl. the ligand-majority rule and both branches |
| Sample ranking score | `core/metrics/sample_ranking.py` | formula and ordering behaviour (upstream's own function is entangled with RASA/featurization) |

## End-to-end inference

`foldjax.models.openfold3.inference.predict` runs trunk -> noise conditioning -> denoiser ->
sampler -> confidence heads:

```python
from foldjax.models.openfold3.data import load_features
from foldjax.models.openfold3.inference import InferenceConfig, predict

batch, chemistry = load_features("features.npz")
prediction = predict(key, batch, params, config, chemistry, n_chain=2)
prediction.coordinates   # [num_samples, N_atom, 3]
prediction.plddt, prediction.ptm, prediction.iptm
```

`released_config()` supplies the released architecture settings — transcribed from
upstream's `model_config.py` and gated by reading that file back — so only the
input sizes and sampling knobs are arguments:

```python
from foldjax.models.openfold3.inference import RELEASED_BLOCK_COUNTS, released_config

config = released_config(n_token=384, n_atom=3000, num_samples=5)
```

Check `RELEASED_BLOCK_COUNTS` against the CLI's block counts before trusting it; a
mismatch means the checkpoint uses a different config. Featurization
(building `batch`) and output writing are out of scope; see below.

The OpenFold3 suite runs on CPU with GPU-only kernel tests skipped, and the full
FoldJAX gate exercises it alongside every other backend. Running the GPU gate is
still essential — it caught a scatter-add nondeterminism that CPU testing could
not.

### What has actually been run

The released `of3_ft3_v1` checkpoint (4945 tensors, 48/4/24/3 blocks, unfused
triangular multiplication) loads and predicts. On real 1UBQ from its sequence alone,
with no alignment and no template, the five samples land **2.52–2.79 Å CA RMSD** from
the deposition, with CA–CA spacing of 3.74 Å and a radius of gyration of 11.4 Å
against the crystal's 11.7 Å. pLDDT ~0.57 and pTM ~0.52 are what single-sequence
input should give. That is the port folding a protein, not a shape check.

Two defects surfaced only under real weights, and neither could have been found with
parameters built from upstream's module definitions:

- the released checkpoint stores `fourier_emb.w` as `(c, 1)` though the module
  declares `(c,)`, which turned a scale into an outer product;
- `predict` was importing upstream to rebuild the representative-atom table, quietly
  breaking the featurize/predict split this package is organized around. The table
  now travels inside the feature `.npz`.

Sizes measured end to end with the released weights and settings (5 samples, 200
rollout steps, 4 cycles), on real depositions:

| target | tokens | atoms | cold | warm | peak | pLDDT |
|---|---:|---:|---:|---:|---:|---:|
| 1UBQ | 76 | 601 | 170 s | 2.5 s | 2.3 GiB | 0.574 |
| 3QEX | 966 | 8029 | 292 s | 108 s | 26.1 GiB | 0.200 |
| 5WLH | 1494 | 13071 | 451 s | 273 s | 37.4 GiB | 0.203 |

Longer targets need `pair_chunk_size`, which `released_config` selects from the token
count. Triangle attention's scores are `[rows, heads, N, N]` -- cubic in token count
-- so one *unchunked* pair block needs 267 GiB of temporaries at 2076 tokens against
41 GiB at 256 rows. Because the chunk shrinks as the target grows, peak memory rises
more slowly than the token count squared: 966 to 1494 tokens is 1.55x the tokens and
only 1.44x the peak, where quadratic would be 2.39x. What is bounded is the score
tensor, not the sequence length.

pLDDT levelling off at ~0.20 is what single-sequence input should give on targets this
size, not a size-dependent defect -- a defect would not level off.

## Deliberately out of scope

Not every upstream function belongs in a JAX inference port, and guessing wrong
here would waste effort:

- `compute_disorder` needs Biotite structure arrays and a SASA calculation — data
  pipeline, not array math. `sample_ranking_score` defaults its `disorder` term to
  zero, which is what upstream does when RASA is unavailable.
- `has_frame` comes from the featurizer/structure module, so the confidence
  functions take it as an argument rather than deriving it.
- MSA subsampling (`subsample_main_msa` / `subsample_all_msa`) is random
  training-time augmentation, disabled for inference.
- The AF2-only MSA attention variants (`MSARowAttentionWithPairBias`,
  `MSAColumnAttention`, `MSAColumnGlobalAttention`) are not on the AF3 path.
- The cuEquivariance triangular-multiplication kernel layout is not mapped; the
  plain and fused layouts both are.

## Checkpoints

`foldjax.models.openfold3.bridge.checkpoint` loads a checkpoint into a flat `{key: array}`
mapping and can summarize one before any layout is assumed:

```python
from foldjax.models.openfold3.bridge.checkpoint import (
    describe, detect_fused_tri_mul, load_checkpoint,
)

state = load_checkpoint("openfold3.safetensors")
print(describe(state, depth=2))        # top-level structure
print(detect_fused_tri_mul(state))     # True / False / None
```

One command surveys a checkpoint before anything is mapped:

```bash
openfold3-jax-inspect-checkpoint weights.safetensors --depth 2
openfold3-jax-inspect-checkpoint weights.pt --grep pairformer --limit 20
```

It reports tensor and parameter counts, the top-level structure, per-stack block
counts, and the triangular-multiplication layout.

**Every parameter of the released model is read by the inference path** (416 of
416 unique tensors; the other 131 in a `state_dict` are `sample_diffusion`
aliases that share storage with `diffusion_module`, plus `version_tensor`). A test
measures this against a real `OpenFold3` module and fails if any group goes
unread, which is how three unwired subsystems were found: the template tower, the
confidence re-embedding, and the diffusion pair-conditioning path.

Parity is gated at three levels. Per-layer tests compare each module against its
upstream twin. **Composite** tests run upstream's own composed code -- `run_trunk`
(1 and 3 recycling cycles), `DiffusionModule.forward`, `SampleDiffusion.forward`
and `AuxiliaryHeadsAllAtom.forward` -- which catches a module that is correct but
never called or fed the wrong input. And `predict()` itself is compared with
`OpenFold3.forward` as a single call, at 1 and 3 blocks per stack: coordinates,
PAE, PDE, distogram, pLDDT and experimentally-resolved.

The rollout is compared by injecting the same noise into both sides and making
augmentation the identity, since the two PRNG streams cannot be matched -- hence
`predict`'s `noise_fn` and `augment` arguments, which exist for that comparison
rather than for production use.

**MSA subsampling.** The released config subsamples the MSA to 1024 rows at
inference, choosing them with `torch.randperm`. `subsample_msa` reproduces
upstream's ordering and accepts an explicit permutation; without one it agrees
exactly at or below 1024 valid rows, and above that the row *choice* cannot be
reproduced from JAX. Selection happens outside the embedder, so it also keeps
shapes static.

## Structures and scores out

```python
from foldjax.models.openfold3 import write_prediction_outputs

written = write_prediction_outputs(prediction, batch, "out/", name="ubq")
# out/ubq_sample_0.cif ... , out/ubq_confidences.json, out/ubq_raw.npz
```

One mmCIF per diffusion sample with pLDDT in the B-factor column, a confidence JSON
with per-sample pTM/ipTM/mean-pLDDT ranked by the AF3 ranking score, and the raw
arrays. Atom identity is decoded back out of the features rather than tracked
alongside them, so it cannot drift from what the model was given. `openfold3-jax-predict`
does the same from a feature `.npz` and a checkpoint.

## From a sequence

```bash
uv sync --extra openfold3-preprocess    # NumPy/JAX raw-job featurization
openfold3-jax-featurize openfold3-query.json -o ubq.npz
```

`openfold3-query.json` is upstream's native `{"queries": ...}` format, not
FoldJAX's common job schema. Alignments and templates ride along per chain:

```json
{"queries": {"ubq": {"chains": [{"molecule_type": "protein", "chain_ids": ["A"],
  "sequence": "MQIFVK...", "main_msa_file_paths": ["aln/colabfold_main.a3m"]}]}}}
```

Search alignments automatically with `--msa-server` (public ColabFold MMseqs2)
or supply your own. The client lives in `foldjax.search`, shared with the other
JAX ports rather than copied into each.

```bash
uv sync --extra openfold3-preprocess
openfold3-jax-featurize openfold3-query.json -o ubq.npz --msa-server
```

`--paired-msa` is off by default: pairing needs taxonomy-annotated headers, and an
alignment that cannot be paired collapses the MSA to the query sequence alone with
no error at all.

Alignment files are selected by **stem** -- `colabfold_main`, `uniref90_hits`, and
the other database names -- because that is how upstream selects them; an
unrecognized name is refused with the accepted list rather than silently ignored.
With no alignments the MSA is the query alone and accuracy drops sharply, so both
the featurizer and the CLI say so.

Featurization uses FoldJAX's one-query NumPy/JAX orchestration while reusing the
vendored publisher chemistry and parsing routines. It does not instantiate the
publisher Dataset or Lightning DataModule and does not create PyTorch tensors.
The self-contained `.npz` can be reused without the preprocessing extra. No CCD
download is required -- biotite's bundled copy is the default.

```python
from foldjax.models.openfold3 import compile_predict
from foldjax.models.openfold3.data import load_features
from foldjax.models.openfold3.inference import released_config

batch, chemistry = load_features("ubq.npz")
config = released_config(n_token=76, n_atom=601)
prediction = compile_predict(config, chemistry)(key, batch, params)
```

Use `--pad-tokens/--pad-atoms` (or `pad_features`) to bucket several targets onto one
compiled function.

**Run it compiled.** Eager execution dispatches 4 recycling cycles over 48
Pairformer blocks one operation at a time -- ~69 s per call at 16 tokens. The
rollout is a `lax.scan`, so compiled the released 5-sample 200-step rollout runs in
**0.89 s** after a one-time 249 s compile (RTX PRO 6000).

```python
from foldjax.models.openfold3 import compile_predict
from foldjax.models.openfold3.inference import released_config

config = released_config(n_token=..., n_atom=...)
run = compile_predict(config, chemistry)
prediction = run(key, batch, params)   # params may be swapped without recompiling
```

`predict` itself stays an ordinary uncompiled function, which is what the parity
tests use. Repeated `compile_predict` factories share a process-wide bounded JIT
pool when their graph identity matches, so multi-seed and repeated backend calls
no longer retrace the same program. The pool retains at most eight executable
variants and evicts least-recently-used entries, bounding long-lived unpadded
batch memory. The identity names the full model configuration, resolved CP
layout and ordered devices, effective triangle kernel, RNG route, requested
representations/stop boundary, chain count, and persistent-cache scope. The
representative-atom chemistry table remains a validated dynamic pytree: changing
its values changes the result without changing the graph. `run.lower(...).compile()`
keeps the same public `(key, batch, params)` call shape and recreates the exact CP
placement selected during lowering.

**Cache the compile.** The FoldJAX backend and `openfold3-jax-predict` enable a
persistent cache by default because every uncached process pays this dominant
cost again. Direct library callers opt in explicitly:

```python
from foldjax.models.openfold3 import enable_compilation_cache

cache = enable_compilation_cache()  # or pass a directory / set $OPENFOLD3_JAX_CACHE
run = compile_predict(config, chemistry, cache_scope=str(cache))
```

Passing the returned cache path as `cache_scope` also lets the in-process pool
notice when that namespace is deleted or replaced. It then discards the matching
memory cache and recompiles, so a cache-warming request repopulates the directory
instead of reporting success after an in-memory-only hit.

Measured on an RTX PRO 6000, released architecture, 5 samples x 20 steps:

| tokens | atoms | compile | run | peak GPU |
|--------|-------|---------|-----|----------|
| 32 | 394 | 292 s | 0.22 s | 1.6 GiB |
| 128 | 1460 | 342 s | 0.66 s | 2.9 GiB |
| 384 | 4480 | 646 s | 5.80 s | 16.1 GiB |

Run time and memory scale gently; compile time does not -- at 384 tokens XLA prints
its own slow-compile warning. Parity was also checked at 128 tokens and released
depths: coordinates 2.9e-5 against a 6.45 maximum.

Parity at the released settings is gated end to end: 48/4/24/2/4 blocks, 4
recycling cycles, 368M parameters, 5 samples x 200 steps, worst disagreement 2.1e-5
on coordinates whose maximum is 35.0.

**Measured on real depositions.** Parity against upstream runs on real RCSB entries
(DNA duplex, protein, tRNA with metals, protein+ligand), not hand-built batches --
which is how a one-off residue-numbering bug in the output layer was found. At 574
tokens (4HHB) a prediction takes 11.8 s after a 234 s compile; 849 tokens does not
yet fit on a 96 GiB card.

See `PROJECT.md` for what the gates still do not cover (released depths, bf16,
upstream's memory-optimization and kernel paths, real weights).

Once weights are available, a second command runs every check that needs them, in
the order that fails cheapest first — block counts against the released
architecture, then the trunk mapper, then optionally the forward pass:

```bash
openfold3-jax-verify-checkpoint of3_ft3_v1.pt
openfold3-jax-verify-checkpoint of3_ft3_v1.pt --batch batch.npz --tokens 384 --atoms 3072
```

It exits non-zero on a block-count mismatch rather than mapping a checkpoint the
released config does not describe, and reports the shape and finiteness of every
`Prediction` field. It does **not** check structural accuracy — that needs a
reference structure, so RMSD and clashes are still unverified. `--batch` takes a featurized batch as `.npz`;
build one with `openfold3-jax-featurize` or pass a raw job through the common
FoldJAX prediction API.

`safetensors` uses its NumPy loader; a `.pt`/`.ckpt` file uses FoldJAX's
restricted archive reader and does not import torch. Publisher-reference parity
tooling stays outside FoldJAX's install metadata. Wrapper prefixes (`module.`,
`_orig_mod.`, `ema.`) are stripped on load.

**Both triangular-multiplication layouts load, through the ordinary mappers.**
The fused variant is one `2*c_hidden` projection split in half — the same function
as two `c_hidden` projections — so `map_fused_triangle_multiplication` converts it
into the unfused parameters the forward pass uses, gated against upstream's fused
module. `map_tri_mul` picks the layout per tri-mul prefix, because
`fuse_projection_weights` is a per-stack config choice, so `map_pair_block` and
everything above it accept either. `detect_fused_tri_mul` is diagnostic, not a
blocker.

The released config sets `fuse_projection_weights: false`, so real checkpoints
are expected unfused; a test records that, and another builds a fully fused model
from upstream's config and maps it, so neither path can rot unnoticed.

Weights: `huggingface.co/OpenFold/OpenFold3`. Never committed here. **The repo is
gated** — even the example config returns HTTP 401 unauthenticated, so loading
real weights needs a HuggingFace account with access granted.
