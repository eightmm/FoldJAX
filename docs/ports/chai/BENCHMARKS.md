# Chai-JAX benchmark record

All accelerator results below were measured on an NVIDIA RTX PRO 6000 Blackwell
96 GB. Cold time includes JAX compilation; warm time uses a compiled executable.
Unless a section says otherwise, memory is the process-wide accelerator
high-water mark with JAX preallocation disabled. Chai-JAX now defaults
preallocation off and the allocator fraction to `0.90`; caller-provided
environment values take precedence. These results do not replace the final
completion gates.

## Standalone inference, official padded no-MSA smoke

Configuration: one alanine, 256-token/5,888-atom bucket, MSA tensor
`(1, 16384, 256)`, no ESM/templates/restraints, one recycle, two diffusion
timesteps, one sample, seed 0, full-FP32 products plus explicit BF16 component
casts. Torch uses upstream `low_memory=True`; JAX keeps parameters on the host,
uses staged cached executables, and runs with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

| metric | Torch | JAX/XLA |
|---|---:|---:|
| warm latency | 1.58493 s | 0.61667 s |
| peak accelerator memory | 6.333 GiB allocated | 2.671 GiB in use |

JAX is 2.57x faster and uses 57.8% less peak accelerator memory on this smoke
configuration. Two same-process JAX calls produced identical fingerprints at
feature initialization, trunk, diffusion, confidence, and final coordinates.
This is preliminary performance evidence, not the default 3-recycle,
200-timestep, 5-sample release gate.

A representative public-default comparison using three recycles, 200 diffusion
steps, five samples, and realistic input remains an open release gate. The
short smoke above must not be presented as that result.

Reproduce with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/benchmark_inference.py \
  --fasta input.fasta --bundle "$CHAI_JAX_BUNDLE" \
  --conformers "$CHAI_JAX_CONFORMERS" \
  --recycles 1 --timesteps 2 --samples 1 --iterations 2 \
  --compile-cache .cache/chai-jax

"$CHAI_UPSTREAM_PYTHON" scripts/benchmark_torch_inference.py \
  --fasta input.fasta --output torch-benchmark \
  --recycles 1 --timesteps 2 --samples 1 --iterations 2
```

## Trunk `forward_256`

Configuration: batch 1, 256 tokens, MSA depth 16,384, four templates, 48
Pairformer blocks, BF16 trunk math, official Chai TorchScript weights and equal
inputs. JAX uses staged cached executables and bounded `lax.scan` MSA chunks.

| metric | Torch | JAX |
|---|---:|---:|
| warm latency | 1.853 s | 0.455 s |
| peak accelerator memory | 6.805 GB | 4.274 GB |
| cold compile + first run | n/a | 28.513 s |

Single output: NRMSE 0.01962, correlation 0.999822. Pair output: NRMSE
0.01566, correlation 0.999880, maximum absolute error 16.0. The pair result
includes the exported graph's outer residual around the complete MSA module;
omitting that residual produced the earlier 0.07 NRMSE. JAX is 4.08x faster
warm and uses 37.2% less peak accelerator memory, so both the accuracy and
performance gates are green.

Reproduce with:

```bash
uv run python scripts/benchmark_trunk_parity.py
```

## Public static-size buckets

The isolated bucket runner starts a fresh process for each public size, disables
ESM/MSA/templates, and performs one recycle, two diffusion steps, one sample,
and two identical calls. The second call records warm executable latency. All
successful repeats had an exact coordinate fingerprint match.

Bounded XLA results:

| bucket | warm | peak in use | status |
|---:|---:|---:|---|
| 256 | 0.61667 s | 2.671 GiB | GREEN |
| 384 | 1.22349 s | 3.318 GiB | GREEN |
| 512 | 2.24246 s | 5.436 GiB | GREEN |
| 768 | 5.70239 s | 12.147 GiB | GREEN |
| 1024 | 11.87848 s | 21.309 GiB | GREEN |
| 1536 | 48.01176 s | 35.364 GiB | GREEN |
| 2048 | >610 s | approximately 79 GB | RED: timed out |

CUDA 13/cuEquivariance results:

| bucket | cold compile + first | warm | peak in use | status |
|---:|---:|---:|---:|---|
| 256 | 104.9128 s | 0.56193 s | 2.127 GiB | GREEN |
| 384 | 109.6687 s | 0.92547 s | 3.330 GiB | GREEN |
| 512 | 117.9526 s | 1.48358 s | 5.436 GiB | GREEN |
| 768 | 128.7772 s | 3.22239 s | 12.144 GiB | GREEN |
| 1024 | 148.2080 s | 6.10861 s | 21.309 GiB | GREEN |
| 1536 | 185.5707 s | 16.15410 s | 47.733 GiB | GREEN |
| 2048 | not established | 52.80157 s | 56.675 GiB | GREEN, cached serving |

The 2048 row is deliberately different from the smaller rows. Building the
2048 executable in the serving process exceeded the bounded cold-cache budget.
After populating the persistent cache, a fresh process completed its first call
in `87.10559 s`, its warm call in `52.80157 s`, and reproduced the exact
coordinate fingerprint. Its allocator pool was 65 GiB and sampled NVIDIA
process high-water mark was 66.475 GiB. Thus 2048 is verified only for CUDA
13/cuEquivariance cached serving; it is not verified for bounded XLA or cold
cache construction.

For an additional cross-framework point, the 768 short configuration took
`8.69124 s` warm in Torch with 13.366 GiB allocated, versus `5.70239 s` and
12.147 GiB in XLA JAX: 1.52x faster with 9.1% less peak memory. These static
bucket smokes remain separate from the open representative production-default
comparison.

Reproduce the isolated matrix with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/benchmark_buckets.py \
  --bundle "$CHAI_JAX_BUNDLE" \
  --conformers "$CHAI_JAX_CONFORMERS"
```

## Live MSA and template preprocessing

A public ColabFold/RCSB integration run used the 76-residue ubiquitin sequence
with template search enabled. The remote response contained 300 template hits;
Chai-JAX materialized the combined content-addressed m8 file, downloaded the
selected CIF records, aligned them with Kalign 3.3.5, and produced four
76-token template features with 304 valid template-token mask entries. A cache
repeat produced the same source SHA-256
`95bd87960360e99ca57cc38658da2a4da0d5eeef38139db7d853992262d93737`.

## Native ESM2 preprocessing

The official 36-layer ESM2 t36 3B TorchScript artifact was exported to a
checksummed, 37-shard native bundle and executed without importing Torch at
runtime. For `<cls>ACDE<eos>`, the full-model JAX/Torch output NRMSE was
0.005554 with correlation 0.9999846 and maximum absolute error 0.02734. JAX
cold compile plus execution took 2.978 s and warm median latency was 12.058 ms.
Peak accelerator memory was 6.137 GB. Content-cached generation for the same
four-residue sequence improved from 11.560 s including model load to 0.348 ms,
with bit-identical cached FP32 embeddings.

The source TorchScript SHA-256 is
`074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a`;
the native manifest SHA-256 is
`9dafefbedbb0817108d921754c2226c307f894b3d3d0bf7b01ec9d6f0b5a376f`.

## Matched-randomness 200-step sampler

On the real protein fixture (32 valid atoms, 256-token bucket, one recycle,
one sample), the Torch and JAX paths used their own preprocessing and trunk
representations but replayed the same initial noise, every churn-noise tensor,
rotation, and translation for all 200 diffusion steps. Coordinate-component
RMSE was
0.02590 Å, maximum absolute drift was 0.1100 Å, and correlation was 0.9999567.
JAX warm sampler time was 2.462 s versus 7.442 s for Torch, a 3.02x speedup.

A shared-static-input control reduced component RMSE to 3.37e-6 Å. The
comparison also exposed an atom-reference-space mask mismatch; before that
correction the framework-native coordinate-component RMSE was 2.386 Å. This was
the first noise-matched protein gate; the expanded modality and branch matrix
below supersedes its earlier protein-only scope.

## Matched 200-step scientific matrix

Each row uses framework-native preprocessing and trunk representations, one
sample, released weights, and the same complete NumPy random tape in both
samplers. The fixtures are single-seed scientific gates, not a multi-seed
quality study.

| fixture/branch | component RMSE | raw atom RMSD | max component drift | correlation | JAX warm speedup |
|---|---:|---:|---:|---:|---:|
| protein, no ESM | 0.02590 Å | 0.04486 Å | 0.11003 Å | 0.9999567 | 3.02x |
| protein, native/default ESM | 0.02520 Å | 0.04365 Å | 0.08715 Å | 0.9999596 | 2.92x |
| RNA + DNA | 0.06999 Å | 0.12123 Å | 0.21871 Å | 0.999873 | 3.01x |
| protein + ligand | 0.31838 Å | 0.55145 Å | 1.95927 Å | 0.993536 | 2.97x |
| deep MSA | 0.12119 Å | 0.20991 Å | see parity artifact | 0.999893 | 2.94x |
| four templates | 0.15800 Å | 0.27366 Å | see parity artifact | 0.999722 | 3.02x |
| contact restraint | 0.04735 Å | 0.08201 Å | 0.15595 Å | 0.9998545 | 2.94x |
| recycle-2 | 0.02518 Å | 0.04361 Å | 0.07555 Å | 0.9999591 | 2.94x |
| modified residue + glycan + covalent bond | 0.68226 Å | 1.18168 Å | 2.06412 Å | 0.97445 | 2.94x |

`component RMSE` is `sqrt(mean((jax_xyz - torch_xyz)^2))` over scalar xyz
components. Conventional raw all-atom RMSD is
`sqrt(mean(sum((jax_xyz - torch_xyz)^2, axis=-1)))`, exactly `sqrt(3)` times
the component RMSE for the same joined atoms. Neither metric is Kabsch aligned.

The deep-MSA fixture contains 33,066 aligned-Parquet rows; both backends retain
9,308 joined valid rows and a 12,909-row profile depth. Its single/pair trunk
NRMSE is `0.00973/0.01266`. The four-template fixture has single/pair trunk
NRMSE `0.00905/0.01216`. Their active-versus-baseline effects, as well as the
restraint, covalent and recycle effects, are non-trivial and directionally
matched; exact feature and effect metrics are in
[`SCIENTIFIC_PARITY.md`](SCIENTIFIC_PARITY.md).

## Diffusion denoiser `forward_256`

Configuration: batch 1, one diffusion sample, 256 tokens, 5,888 atoms, FP32
activations, BF16 storage for the 16-block diffusion transformer weights, FP32
for conditioning/atom encoder/atom decoder/top-level weights, query chunk 256,
official Chai TorchScript weights and equal deterministic inputs.

| metric | Torch | JAX |
|---|---:|---:|
| warm latency | 0.380 s | 0.00677 s |
| peak accelerator memory | 1.360 GB | 1.284 GB |
| cold compile + first run | n/a | 19.00 s |

Output parity: NRMSE 0.002608, correlation 0.99999645, maximum absolute error
0.08307. This component gate is green and is joined to the matched 200-step
matrix above.

Reproduce with:

```bash
uv run python scripts/benchmark_diffusion_parity.py \
  --repeats 3 --query-chunk-size 256 --bf16-component transformer
```

## Confidence and public score outputs

A committed real protein-ligand fixture runs the official 256-crop confidence
head into the public JAX confidence/ranking implementation and writes the same
score-NPZ schema. PAE/PDE/pLDDT logit NRMSE is
`0.003926/0.005541/0.002428`. Converted PAE, PDE and pLDDT scores pass the
declared mixed absolute/relative tolerance; aggregate score, pTM, ipTM,
per-chain pTM, per-chain-pair ipTM, and clash fields pass the public output gate.
Total and inter-chain clash counts match exactly.

Confidence triangle attention uses exact FP32 softmax with bounded query
chunks. That change removed the cubic materialized-logit memory failure while
preserving the official fixture output. The successful 768/1024 bucket results
above include this implementation.
