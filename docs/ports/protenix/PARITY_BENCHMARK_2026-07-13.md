# Protenix JAX parity benchmark — 2026-07-13

## Contract

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q
- checkpoint: `protenix_base_default_v1.0.0`
- precision: Torch FP32 with TF32 disabled; JAX FP32 with `highest` matmul precision
- inference: 10 recycling cycles, 20 diffusion steps, one sample
- augmentation: center-only
- parity inputs: identical static features and captured init/per-step diffusion noise
- MSA: real 7r6r MSA, 703 rows, full depth in fixed order in both frameworks
- checkpoint SHA-256: `2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04`

The two cases exercise protein+DNA (245 tokens, 2,529 atoms) and protein+ATP
(234 tokens, 1,697 atoms). The JAX cold time includes compilation; warm time is
the second execution in the same process. Torch time is a warmed trunk plus
diffusion and excludes checkpoint loading.

## Results

| Case | Torch warm | JAX cold | JAX warm | JAX/Torch | all-atom Kabsch | CA Kabsch | first-denoiser Kabsch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7r6r DNA | 5.63 s | 160.82 s | 31.01 s | 5.51x slower | 2.54e-5 Å | 2.93e-5 Å | 5.14e-5 Å |
| 7r6r ATP | 5.31 s | 152.30 s | 26.19 s | 4.93x slower | 1.85e-4 Å | 1.87e-4 Å | 5.19e-5 Å |

| Case | `s_inputs` corr | `s_trunk` corr / RMSE | `z_trunk` corr / RMSE | JAX peak allocated | Torch peak allocated |
|---|---:|---:|---:|---:|---:|
| 7r6r DNA | 1.000000000000 | 0.9999999999999 / 1.24e-4 | 0.9999999999998 / 2.77e-5 | 2.98 GB | 3.97 GB |
| 7r6r ATP | 1.000000000000 | 1.0000000000000 / 1.14e-4 | 0.9999999999953 / 1.18e-4 | 2.80 GB | 3.87 GB |

JAX's device allocator reported about 2.8–3.0 GB peak bytes in use, while the
default preallocating process held about 73.5 GiB according to `nvidia-smi`.
Those are different quantities and should not be conflated.

### Cached-serving optimized result

After adding reusable primitive JIT boundaries, static diffusion/atom caches,
and routing the compiled triangle backend through the root, template, and MSA
pair stacks, the cache-hit serving result is:

| Case | Torch warm | JAX warm | Speedup | JAX peak | Memory reduction | all-atom / CA Kabsch |
|---|---:|---:|---:|---:|---:|---:|
| 7r6r DNA | 5.63 s | 4.47 s | 1.26x | 2.62 GB | 34.1% | 4.72e-5 / 5.23e-5 Å |
| 7r6r ATP | 5.31 s | 4.08 s | 1.30x | 2.60 GB | 32.9% | 2.21e-4 / 2.22e-4 Å |

Cold compilation is not part of this serving comparison because deployed
shapes are precompiled into JAX's persistent compilation cache. The raw final
metrics are under `outputs/parity/jax_cached_final/`.

## Findings

1. Upstream inference samples a random MSA depth and random row permutation on
   every recycle. The current JAX static path always consumes all materialized
   rows. Without pinning this contract, `z_trunk` correlation fell to 0.99854
   (DNA) and 0.99677 (ATP), so matched-noise RMSD was not a valid model-parity
   measurement.
2. With both sides pinned to all 703 rows in order, the trunk is effectively
   numerically matched. This exposed the earliest remaining mismatch in atom
   cross-attention: upstream computes `q = norm_a(a, s)` followed by
   `kv = norm_kv(q, s)`, while JAX normalized both branches from the original
   `a`. Before the fix, local block-0 RMSE was 0.059 and first-denoiser RMSE was
   1.344 Å.
3. Applying the upstream sequential normalization contract reduced local
   block-0 RMSE to 4.51e-7 and first-denoiser RMSE to 3.28e-5 Å in the stage
   capture. The full 20-step all-atom Kabsch errors are now 2.54e-5 Å (DNA) and
   1.85e-4 Å (ATP), so the correctness gap is closed.
4. JAX repeatability (`max_abs` 9.7e-5–3.2e-4 Å) is on the same scale as the
   remaining coordinate residual and accounts for the practical parity floor.

## Comparison with Boltz JAX

The existing Boltz fair benchmark on the same GPU reports a 2,528-atom case at
200 diffusion steps and three recycles: JAX 5.25 s versus Torch 8.19 s (1.56x
speedup), with 25-step matched-noise Kabsch RMSD 1.51e-5 Å on the smaller parity
case. Protenix uses a different architecture and 10 recycles, so absolute
runtime is not an apples-to-apples model comparison. After the cross-attention
fix, both ports are numerically matched. The remaining difference is
performance: Boltz JAX is faster than its Torch reference, while Protenix JAX
is about 5x slower than its Torch reference.

Raw metrics are under `outputs/parity/torch_staticmsa/` and
`outputs/parity/jax_fixed/`.
