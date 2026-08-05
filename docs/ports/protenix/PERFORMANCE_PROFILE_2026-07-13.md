# Protenix JAX warm-performance profile — 2026-07-13

## Contract

The profile uses the same correctness contract as the parity benchmark:
7r6r protein+DNA, 245 tokens, 2,529 atoms, real 703-row MSA, FP32/highest
precision, 10 recycling cycles, 20 diffusion steps, one sample, and matched
Torch noise. Every timing blocks on the device. Torch and JAX run on the same
RTX PRO 6000 Blackwell Max-Q.

## Default warm breakdown

| Stage | JAX | Share | Torch reference | JAX/Torch |
|---|---:|---:|---:|---:|
| Input atom embedder | 0.071 s | 0.2% | included in trunk | — |
| 10-cycle trunk | 25.56 s | 86.2% | 5.21 s | 4.91x |
| Conditioning cache | 0.005 s | <0.1% | included | — |
| 20-step diffusion | 4.00 s | 13.5% | 0.423 s | 9.45x |
| Stage sum | 29.63 s | 100% | 5.63 s | 5.26x |

The end-to-end warm benchmark is 31.01 s; the small difference is process and
orchestration noise around the separately synchronized stage measurements.

## Trunk breakdown

A representative recycle takes 2.598 s:

| Component | Time per cycle | Share |
|---|---:|---:|
| Recycling projection | 0.004 s | 0.2% |
| Template module | 0.078 s | 3.0% |
| MSA module (703 rows) | 0.065 s | 2.5% |
| Root Pairformer (48 blocks) | 2.450 s | 94.3% |

The deep MSA is not the performance problem. Repeating the root Pairformer ten
times accounts for roughly 24.5 s of the 31 s warm latency.

## Why the Pairformer is slow

The default root stack uses `jax.lax.scan` over 48 parameter blocks. On this
shape and GPU, the same stack measured:

| Pairformer execution | Warm time |
|---|---:|
| `use_scan=True` | 2.382 s |
| `use_scan=False` | 0.560 s |

The scan path is 4.26x slower in steady state. It reduces graph expansion and
compile pressure, but dynamic parameter slicing and the sequential scan loop
prevent the faster per-layer execution seen in the unrolled Python path. A
single block's separately synchronized primitive groups total only 17.9 ms;
triangle attention is the largest individual group at about 9.6 ms for its two
directions, but no single primitive explains the 2.38 s scan stack.

Disabling only Pairformer scan gives:

| Metric | Default scan | Pairformer loop |
|---|---:|---:|
| Cold | 160.82 s | 119.80 s |
| Warm | 31.01 s | 11.25 s |
| Peak allocated | 2.98 GB | 2.78 GB |
| All-atom Kabsch RMSD | 2.54e-5 Å | 2.54e-5 Å |
| CA Kabsch RMSD | 2.93e-5 Å | 2.79e-5 Å |

This is a 2.76x end-to-end warm speedup without a measurable parity regression.

## Remaining 2x gap after Pairformer scan is disabled

The scan-off stage profile is 7.09 s trunk plus 3.85 s diffusion, or 11.01 s.
Torch is 5.63 s total. The remaining causes are:

1. The sampler is a Python `for` loop over all 20 steps. It is not a single
   jitted graph or a `lax.scan`, unlike the Boltz JAX full-graph sampler.
2. The denoiser's 3-block atom encoder, 24-block token transformer, and 3-block
   atom decoder all default to Python layer loops. Across 20 steps this creates
   600 transformer-block invocations plus their individual JAX dispatches.
3. Before the follow-up, the `use_diffusion_scan=True` path was unusable: it raised
   `TracerBoolConversionError` because `AttentionPairBiasParams.has_s` becomes a
   traced boolean inside `lax.scan` and is consumed by a Python `if`.
4. JAX precomputed the diffusion pair-conditioning cache, but the top-level
   model did not materialize and pass the atom encoder's reusable `p_lm/c_l`
   cache. The upstream default cache path does, although this is smaller than
   the transformer/dispatch cost.
5. The model is orchestrated as eager JAX operations rather than a stable,
   bucketed full-graph executable. Warm means individual XLA executables are
   already compiled; it does not remove Python dispatch or scan-loop runtime.

## Recommended order

1. Make Pairformer loop the latency-oriented path (retain scan as a compile-time
   option) and validate DNA/ATP/RNA/template buckets before changing defaults.
2. Repair static boolean handling in diffusion transformer scan and benchmark
   scan versus unrolled execution; scan may reduce dispatch but can also hurt
   steady latency, as the Pairformer result demonstrates.
3. Implement a sampler-level `lax.scan` and wrap the stable sampling graph in
   `jax.jit`, with matched-noise RMSD checked after each step.
4. Add the reusable atom encoder cache and re-profile diffusion.
5. Only then evaluate fused triangle kernels; triangle attention is not the
   dominant explanation for the current 31 s result.

Raw profiles are under `outputs/parity/profile/`; the optimized parity result
is under `outputs/parity/jax_pairloop/7r6r_dna/`.

## Implementation follow-up

The Boltz-style execution changes were implemented and measured rather than
enabled based on analogy alone. The reusable upstream-equivalent atom cache is
now always prepared by the top-level inference path, but broad graph capture
was not beneficial on this workload:

| Experiment | Warm diffusion | Result |
|---|---:|---|
| Python loops, no atom cache | 3.852 s | baseline |
| Python loops, atom cache | 3.645 s | retained |
| Sampler `lax.scan`, layer loops | 8.809 s | rejected |
| Sampler and layer `lax.scan` | 4.946 s | rejected |
| Diffusion-layer scan only | 91.993 s | rejected |
| Whole denoiser `jax.jit` | 24.523 s | rejected |
| Built-in XLA SDPA | 4.612 s | rejected |
| Small reusable attention JIT boundaries | 1.230 s | retained |
| Attention and transition JIT boundaries | 0.824 s | retained |

Tokamax triangle attention was also rejected: it repeatedly autotuned distinct
calls for more than seven minutes and did not complete the benchmark. cuDNN
SDPA did not support the benchmark's FP32 inputs. The winning approach keeps
Python orchestration but adds stable, reusable JIT boundaries around global,
local, and triangle attention, triangle multiplication, and transition blocks.
This is closer to Boltz's reuse of compiled kernels than to compiling the whole
sampler as one graph.

The latency defaults are now Pairformer loops plus the `xla_jit` attention and
block-part paths. The final full matched-noise results are:

| Input | Torch warm | JAX cold | JAX warm | Peak allocated | All-atom Kabsch RMSD | CA Kabsch RMSD |
|---|---:|---:|---:|---:|---:|---:|
| 7r6r protein+DNA | 5.63 s | 151.70 s | 4.47 s | 2.62 GB | 4.72e-5 Å | 5.23e-5 Å |
| 7r6r protein+ATP | 5.31 s | 148.76 s | 4.08 s | 2.60 GB | 2.21e-4 Å | 2.22e-4 Å |

Compared with the previous optimized JAX path, warm latency fell from 11.03 s
to 4.47 s for DNA and from 9.80 s to 4.08 s for ATP. JAX is 20.6% and 23.1%
faster than the matched Torch warm measurements, while peak allocation is
34.1% and 32.9% lower respectively. Cold compilation is deliberately excluded
from the serving objective because production uses the persistent compilation
cache. Raw final results are under `outputs/parity/jax_cached_final/`. RNA and
template matched-noise fixtures were not available in this profiling set, so
those modalities were not re-benchmarked.

The last retained improvement was routing `triangle_attention_backend` through
the template and MSA pair stacks as well as the root Pairformer stack. On the
DNA stage profile this reduced trunk time from 4.171 s to 3.580 s and total
stage time from 5.054 s to 4.441 s. A larger whole-Pairformer-block JIT boundary
was re-tested under the cache-hit objective, but total stage time was 4.448 s;
the isolated 4% block microbenchmark did not survive end-to-end measurement,
so that candidate was removed.

## Boltz-JAX cross-port audit

The two ports benefit from different execution boundaries:

| Area | Boltz-JAX | Protenix-JAX | Transfer decision |
|---|---|---|---|
| Layer/sampler control | `lax.scan` compiles homogeneous stacks and sampling loops | Python loops are faster for the 48-block root trunk and denoiser | Keep model-specific defaults |
| Kernel reuse | Large scan/full-graph executables | Small reusable attention/transition JIT executables | Do not copy Protenix JIT boundaries into Boltz scan graphs |
| Diffusion conditioning | Preprojects per-layer pair biases | Fuses pair projection into each compiled attention call | Keep Protenix fusion; a materialized bias cache is slower |
| Q/K/V projection | Combines Q+gate and K+V GEMMs | Separate dynamic parameter pytrees | Combining at runtime is slower in Protenix; only reconsider with prepacked weights |
| Serving | Persistent compile cache, feature cache, shape buckets | Persistent compile cache, static feature bundles | Token bucketing needs a real token-pad mask before it is safe to port |
| Precision | Optional low-precision trunk with an FP32 diffusion island | Validated FP32/highest path | Treat mixed precision as a separate parity study |

Three direct transfers were tested and rejected:

| Experiment | Baseline | Candidate | Conclusion |
|---|---:|---:|---|
| Protenix primitive JIT boundaries in Boltz, 1UBQ 20-step smoke | 12.689 s warm | 12.808 s warm | 0.9% slower; Boltz scan already provides the compile boundary |
| Boltz per-layer diffusion-bias cache in Protenix DNA profile | 0.824 s diffusion | 1.197 s diffusion | 45% slower due to materialization/load cost and lost fusion |
| Boltz combined Q+gate/K+V GEMMs in Protenix DNA profile | 5.135 s stage sum | 5.877 s stage sum | 14% slower because dynamic weight concatenation dominates |

The failed candidates were removed. The useful shared concepts already retained
are persistent compilation caching, static conditioning/atom caches, exact
query chunking, and backend-specific compiled-kernel reuse. Further exchange
should focus on prepacked native weights or a masked Protenix bucket ABI, not
on copying scan/JIT switches mechanically.
