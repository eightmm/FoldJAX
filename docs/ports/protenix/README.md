![Protenix-JAX hero: a protein, DNA, and ligand emerging from a luminous JAX-inspired tensor field](assets/protenix-jax-hero.png)

# Protenix-JAX

> **Archived port record.** Protenix-JAX now ships as
> `foldjax.models.protenix`. Use the top-level [README](../../../README.md) and
> common `foldjax` API for current installation and prediction. The native
> entry points and performance evidence below remain useful for advanced
> inspection.

An independent JAX inference port of
[ByteDance Protenix](https://github.com/bytedance/Protenix) for biomolecular
structure prediction.

Protenix-JAX converts trusted upstream Protenix checkpoints into a native JAX
weight format, then runs prediction without importing PyTorch or the upstream
`protenix` package. It targets compiled, persistent-cache inference on NVIDIA
GPUs, while retaining CPU/XLA paths for testing and development.

> [!IMPORTANT]
> Protenix-JAX is an independent community port. It is not an official
> ByteDance or Protenix project. Model weights are not distributed in this
> repository; obtain them from the upstream Protenix project and follow the
> upstream model terms.

## Highlights

- Base-v0.5/v1, applied-base, constraint, mini, tiny, ESM, ISM, and
  Protenix-v2 checkpoint structures.
- Native JAX trunk, diffusion sampler, distogram, confidence head, and complete
  chain-level confidence summaries.
- cuEquivariance triangle multiplication and attention on CUDA 13, with native
  XLA attention paths where appropriate.
- Protein, DNA, RNA, ion, CCD, SMILES, PDB, SDF, MOL, and MOL2 inputs.
- CCD modifications, covalent bonds with leaving-group removal, contact,
  atom-contact, pocket, and substructure constraints.
- Protein/RNA MSA preprocessing boundaries with content-addressed caches.
- JSON, A3M, and HHR templates backed by a local mmCIF database.
- ESM/ISM preprocessing and all eight Protenix training-free-guidance (TFG)
  potentials.
- Multi-job, multi-seed, ranked CIF, confidence JSON, and optional raw NPZ
  output.

Training is not implemented. This repository is focused on inference,
checkpoint conversion, preprocessing, parity, and performance.

## Installation

FoldJAX currently targets Python 3.13. The recommended environment manager is
[uv](https://docs.astral.sh/uv/).

```bash
# NVIDIA CUDA 13 runtime and development tools
uv sync
uv run foldjax weights fetch --model protenix
uv run foldjax predict --model protenix --input job.json
```

Optional extras:

| Extra | Purpose |
|---|---|
| `cuda12` | JAX CUDA 12 runtime; select XLA triangle backends explicitly |
| `cuda13` | JAX CUDA 13 plus cuEquivariance JAX kernels |
| `gpu` group | the same pins as `cuda13`, enabled by default so a bare `uv sync` is a GPU install |
| base install | RDKit/NumPy preprocessing, restricted checkpoint reader, and JAX ESM/ISM |
| `dev` group | pytest and Ruff |

The default production triangle backend is `cueq_jit`, so CUDA 13 inference
should install the `cuda13` extra. CPU and CUDA 12 users should choose the XLA
triangle backend explicitly.

## Convert Protenix Weights

Checkpoint conversion is a one-time trusted operation. Prediction uses only
the resulting native JAX file.

```bash
uv run protenix-jax-export-weights \
  --checkpoint /path/to/protenix_base_default_v1.0.0.pt \
  --out weights/protenix_base_default_v1.0.0-jax.pkl \
  --no-compress
```

Only convert checkpoints obtained from a trusted source. Publisher `.pt`
archives are decoded by FoldJAX's restricted NumPy reader; no PyTorch runtime
is imported. Native `.jax` weights use FoldJAX's restricted container.

## Predict

Run directly from a Protenix inference JSON file:

```bash
uv run protenix-jax-predict \
  --weights weights/protenix_base_default_v1.0.0-jax.pkl \
  --input-json examples/input.json \
  --model-name protenix_base_default_v1.0.0 \
  --output-format protenix \
  --out outputs/predictions
```

`--model-name auto` infers a known model name from the weight filename. Model
selection supplies the original recycle count, diffusion-step count, `gamma0`,
and `eta`; explicit CLI values override those defaults.

Multiple top-level jobs, JSON `modelSeeds`, CLI `--seeds`, and multiple samples
are supported. `--output-format protenix` writes ranked CIF and confidence JSON
files. Use `both` to also retain the raw NPZ output.

### Static feature bundles

Precompute features once:

```bash
JAX_PLATFORMS=cpu uv run protenix-jax-featurize-json \
  --input examples/input.json \
  --out outputs/features.npz \
  --max-msa-rows 16384
```

Then predict from the bundle:

```bash
uv run protenix-jax-predict \
  --weights weights/protenix-v2-jax.pkl \
  --features outputs/features.npz \
  --model-name protenix-v2 \
  --output-format both \
  --out outputs/predictions
```

Nested feature dictionaries are serialized as dotted NPZ keys and restored
without enabling pickle.

### ESM and ISM variants

`protenix_mini_esm_v0.5.0` and `protenix_mini_ism_v0.5.0` use the exact
36-layer, 2560-wide ESM2 encoder in JAX. Place the publisher checkpoint beside
the Protenix native weights, or use the matching managed asset profile. The
official zip-based `.pt`, representation-only `.safetensors`, and `.npz`
formats are accepted without importing PyTorch or fair-esm.

The common API configures the full bundle from one profile name:

```bash
foldjax weights fetch --model protenix --profile mini-esm-v0.5.0
foldjax predict --model protenix --input job.json --profile mini-esm-v0.5.0
```

This selects `protenix_mini_esm_v0.5.0` and the staged ESM2 checkpoint
automatically. `mini-ism-v0.5.0` does the same for the ISM pair. Explicit
conflicting `model_name` or `esm_checkpoint_dir` options are rejected.

A static feature archive containing a finite `[num_tokens, 2560]`
`esm_token_embedding` bypasses the 3B language-model checkpoint entirely. This
is useful when several structure predictions reuse one sequence embedding.

## MSA and Template Search

Search is disabled by default. Protenix-JAX never downloads multi-gigabyte
databases implicitly.

- Protein MSA: local wrapper or a ColabFold-compatible remote MMseqs2 endpoint.
- RNA MSA: local nhmmer-style wrapper with user-managed Rfam, RNAcentral, and
  NT-RNA databases.
- Template search: local wrapper producing A3M or HHR plus a user-managed mmCIF
  coordinate database.

All search results use sequence, backend, version, options, and input-file
hashes in their cache identity. Run `protenix-jax-predict --help` for the
complete search CLI.

MSA and template database snapshots affect scientific reproducibility. Record
the database release, wrapper version, and search options with every result.

## Training-Free Guidance

Pass the original Protenix guidance mapping as JSON:

```bash
uv run protenix-jax-predict \
  --weights weights/protenix_base_default_v1.0.0-jax.pkl \
  --input-json examples/input.json \
  --guidance-config configs/guidance.json \
  --output-format protenix \
  --out outputs/guided
```

The JAX implementation includes InterchainBond, PairwiseDistance, StereoBond,
ChiralAtom, PlanarImproper, LinearBond, ExperimentalTorsion, and VinaSteric
potentials, Monte-Carlo smoothing, and constraint projection. Unsupported
RDKit geometry fails before guided sampling rather than silently disabling an
active term.

## Runtime and Memory

The automatic chunk policy keeps the upstream large-input thresholds and adds a
measured memory-bounded policy for the 769–1024 range:

| Tokens | Triangle chunk |
|---:|---:|
| `<= 768` | disabled |
| `<= 1024` | 256 |
| `<= 1536` | 512 |
| `<= 2048` | 256 |
| `<= 2560` | 128 |
| `> 2560` | 32 |

Protenix-v2 rejects more than 2560 tokens, matching the upstream runtime guard.
Explicit chunk arguments override the automatic policy.

The 769–1024 policy is intentionally more memory-bounded than upstream. On the
measured 952-token production workload, chunk 256 reduced peak memory by 48–54%
without a material warm-latency regression.

Persistent compilation caching is enabled by default under
`outputs/compile_cache`. Compilation time is reported separately from warm
execution; changing tensor shapes or static runtime options can create a new
executable.

### Compilation-cache prewarming

The default lazy cache is sufficient for interactive use. A serving deployment
can prewarm its representative inputs on the same GPU and software image that
will serve requests:

```bash
uv run protenix-jax-predict \
  --weights weights/protenix_base_default_v1.0.0-jax.pkl \
  --features outputs/features_768.npz \
  --model-name protenix_base_default_v1.0.0 \
  --out outputs/prewarm-unused.npz \
  --compile-cache outputs/compile_cache \
  --prewarm-only
```

Repeat for the exact token, atom, and MSA shapes and for profiles whose sample
count, steps, cycles, backends, chunk policy, confidence, or guidance mode
differ. Multiple seeds do not require separate executables; prewarm mode runs
the first seed for each input job. It executes once to finish GPU kernel
autotuning and persistent XLA cache writes, but writes no prediction files. Set
`JAX_LOG_COMPILES=1` when auditing whether a serving call compiled a new
executable.

## Verified Performance

The production comparison below uses an RTX PRO 6000 Blackwell, base-v1,
10 recycles, 200 diffusion steps, five samples, confidence output, and the
same 245-token/2529-atom protein-DNA input.

| Runtime | Warm median | Peak VRAM |
|---|---:|---:|
| Protenix-JAX, BF16 trunk | **4.163 s** | **2.724 GB** |
| Upstream Protenix Torch, BF16 | 4.908 s | 4.037 GB |

Under that production-default comparison, JAX is 1.18x faster and uses 32.5%
less peak GPU memory. See the controlled methodology and parity details:

- [Torch/JAX parity benchmark](PARITY_BENCHMARK_2026-07-13.md)
- [Warm performance profile](PERFORMANCE_PROFILE_2026-07-13.md)
- [Production inference benchmark](PRODUCTION_INFERENCE_BENCHMARK_2026-07-13.md)

Matched-noise multimodal checks reached all-atom Kabsch errors of
`2.54e-5 Å` for protein-DNA and `1.85e-4 Å` for protein-ATP. TFG potential
checks reached maximum energy and gradient differences of approximately
`4.8e-7` and `2.9e-6` against the Torch implementation.

## Development

```bash
uv sync
JAX_PLATFORMS=cpu uv run pytest -q tests/models/protenix
uv run ruff check src tests
uv build
```

Publisher-reference parity and production benchmarks use separately
provisioned environments. Model weights, captured tensors, compile caches, and
benchmark outputs are intentionally excluded from Git.

## Scope and Scientific Use

Structure prediction confidence is not experimental validation. Record model
name, checkpoint checksum, input JSON, seeds, MSA/template database snapshots,
search settings, and Protenix-JAX revision for reproducible use. Avoid mixing
database cutoffs when comparing models or making temporal generalization
claims.

## License and Attribution

Protenix-JAX is distributed under the Apache License 2.0. This repository
contains an independent JAX port derived from the Apache-2.0-licensed
[Protenix](https://github.com/bytedance/Protenix) implementation. See
[LICENSE](../../../LICENSE) and [NOTICE](../../../NOTICE).

If you use the models or architecture in research, cite the corresponding
Protenix publications listed by the upstream project.
