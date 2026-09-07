# OpenDDE BF16 LayerNorm correction

## Confirmed mismatch and change

Native OpenDDE's default `OpenFoldLayerNorm` disables autocast for BF16 input,
casts affine operands to BF16, and calls native LayerNorm. Preserving the
original FP32 affine values is therefore not the right correction.

The shared FoldJAX Protenix/OpenDDE primitive instead performed intermediate
normalization and affine operations in BF16. Individual intermediate results
were rounded before subsequent arithmetic. FP32 affine parameters could also
promote the output to FP32, unlike the native BF16-input path.

The correction applies only to BF16 inputs: quantize affine operands to BF16,
widen those values and inputs to FP32, perform normalization and affine math,
then narrow the final output once. FP32 input behavior is unchanged. Pinned
Protenix has the same native BF16 affine-cast policy, justifying the shared
primitive correction. Production default dtype is unchanged.

## Evidence

Captured the first 12 distinct BF16 LayerNorm module calls during native
3GCA inference (five samples, 200 steps, 10 cycles), without replacing native
arithmetic. The observed class was `OpenFoldLayerNorm`, not the optional fast
extension. This isolates a real operator mismatch; it does not locate the
first divergence across the entire network or prove that this operator alone
explains the final coordinate drift.

On the same captured native CUDA inputs and quantized affine operands,
CPU-JIT FoldJAX comparison over 2,776,192 values:

| Measure | Old primitive | Corrected primitive |
|---|---:|---:|
| Values not bit-equal to native | 991,643 | 52 |
| Maximum absolute difference | 0.0625 | 0.015625 |

Residual BF16 rounding differences remain. Neither bitwise parity nor the
fixed 1e-4 numerical tolerance is claimed. CPU versus CUDA reduction ordering
is an additional variable; the queued GPU operator probe tests this separately.

[Per-operator CPU evidence](../bench/experiments/opendde-norm-cpu-2026-09-06.json)
includes native capture hashes and metrics. Native capture job 175 succeeded;
job 174 failed before inference due to a missing `src` entry in `PYTHONPATH`.
That failed attempt is retained and not counted as a scientific result.

## Verification and remaining work

Eight new BF16 eager regression cases failed before the correction and pass
after it; all eight were also checked under JIT. They cover BF16/FP32 affine
storage, absent scale/bias, native affine quantization and output dtype.
Focused primitive, trunk, realized-dtype, OpenDDE real-weight trunk, tape and
entity checks: **97 passed**, with one
JAX int64-to-int32 warning from the existing feature path. Ruff and diff checks
passed. Full-package CPU tests, all-model GPU gates, commit and push were not run.

Frozen-source jobs 176 (3GCA BF16 full replay against both native modes) and
177 (same-input GPU operator comparison) were submitted. Their completion and
metrics must be verified separately. Matching LayerNorm does not yet match
the whole native AMP policy, and coordinate improvement is not assumed.
