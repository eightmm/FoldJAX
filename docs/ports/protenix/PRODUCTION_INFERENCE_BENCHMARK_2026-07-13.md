# Protenix production inference benchmark — 2026-07-13

## Contract

Both implementations use `protenix_base_default_v1.0.0`, seed 101, 10
recycling cycles, 200 diffusion steps, 5 samples, MSA enabled, templates/RNA
MSA/TFG disabled, diffusion shared-variable caches, and the complete
distogram/confidence/ranking output path. Timings exclude feature/checkpoint
loading and compilation. The final JAX follow-up reports five synchronized
warm runs; the Torch reference reports three.

Torch uses BF16 autocast, TF32, fast CUDA LayerNorm, and cuEquivariance 0.9.0
CUDA 13 triangle kernels. JAX uses a persistent compilation cache, BF16 for the
input/trunk, FP32 diffusion and output-head islands, `lax.scan` for the complete
200-step sampler, and reusable `xla_jit` attention/block-part boundaries.

Torch retains upstream random per-recycle MSA sampling. JAX deliberately uses
the complete 703-row MSA with one static shape: this performs more MSA work but
is faster on XLA than fragmented random-depth executables. The comparison is
therefore conservative for JAX rather than workload-reduced.

## Final results

| Input | Torch warm median | JAX warm median | JAX latency | Torch peak | JAX peak | JAX memory |
|---|---:|---:|---:|---:|---:|---:|
| 7r6r protein+DNA | 4.892 s | 4.729 s | 3.3% faster | 4.037 GB | 2.761 GB | 31.6% lower |
| 7r6r protein+ATP | 4.378 s | 4.157 s | 5.1% faster | 3.940 GB | 2.534 GB | 35.7% lower |

Warm samples were:

- DNA Torch: 4.856, 4.892, 4.920 s; JAX: 4.735, 4.722, 4.729, 4.678, 4.774 s.
- ATP Torch: 4.362, 4.378, 4.429 s; JAX: 4.157, 4.121, 4.068, 4.191, 4.161 s.

Raw final metrics are `outputs/production_benchmark/final_torch_{dna,atp}_3x.json`
and `outputs/production_benchmark/ablation_msa704_{dna,atp}.json`.

## What changed

The original production JAX path took 11.454 s on DNA and 11.092 s on ATP.
The useful changes were:

1. Put the complete 200-step diffusion sampler in one `lax.scan` executable.
   This removed repeated Python/XLA dispatch and reduced the FP32 DNA run to
   about 6.19 s.
2. Cast input-embedder and trunk parameters/features to BF16 once while keeping
   diffusion, confidence, and output computations in FP32. This mirrors the
   useful part of upstream autocast without lowering coordinate integration
   precision.
3. Keep confidence samples as a short host loop. For this shape, scanning five
   confidence samples was slower.
4. Preserve the proven small `xla_jit` attention/block-part boundaries and
   persistent cache policy from the earlier Boltz-JAX-style optimization.
5. Align an almost-full MSA to a 64-row boundary when this adds at most eight
   rows. The measured 703-row inputs become 704 rows with an exact OPM mask.
   This improved DNA by 2.2% and ATP by 0.2% relative to the previous JAX
   production path without reducing MSA content.

Two plausible optimizations were measured and rejected:

- Exact random per-cycle MSA shapes took 5.839 s on DNA and required many
  shape-specific compilations. Masked padding to one 384-row bucket still took
  5.797 s. Full-depth 703-row MSA is faster because it retains efficient GEMM
  shapes and one executable.
- The ported efficient-fusion normalization path took 4.824 s in a single DNA
  run versus 4.732 s without it. XLA already hoists the shared work sufficiently,
  so production defaults leave this optional path disabled.

Exact commands and dirty-tree fingerprints are recorded in
`docs/EXPERIMENTS.jsonl`.

## 2026-07-14 follow-up ablations

- Nesting a separately jitted denoiser inside the sampler scan was abandoned
  after more than three minutes of compilation and about 75 GB of device
  allocation.
- Casting the complete denoiser to BF16 reduced DNA latency to 3.831 s, but
  failed correctness: the matched-noise 20-step all-atom Kabsch error was
  4.93 Å and the first denoise output differed by 1.41 Å.
- FP32 LayerNorm statistics reduced that BF16 error but still left 1.46 Å
  end-to-end error. Casting only the central 24-block token transformer also
  failed the accuracy gate and did not improve the parity-harness latency.
- Computing BF16 LayerNorm statistics in FP32 was neutral overall for serving:
  DNA improved 1.2%, while ATP regressed 1.2% on cache-hit five-run medians.
  It was therefore not retained as a speed optimization.
