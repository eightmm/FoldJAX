# OpenBind ordinary TF32 projection boundary: 2026-09-08

## Scope and observations

This is a teacher-forced operator diagnosis, not a production/default change or
full-model admission. The pinned native revision is
`c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`. Real weights belong to
`pairformer_stack.blocks.0.pair_stack.tri_att_start`; inputs are the previously
frozen synthetic FP32 activations, not real trunk activations.

Native full-operator job 448 repeated all 52 cases bitwise. Candidate job 449
passed only 1/52 strict comparisons. Teacher-forced boundary jobs 452/453 isolate
ordinary LayerNorm and q/k/v/g/pair-bias projections. Native LayerNorm output is
the sole projection input; candidate LayerNorm output is measured separately.
The latter passes, with maximum errors `4.77e-7` at N16/17 and `9.54e-7` at N437.
Neither `high` nor explicit TF32 alone closes the projection boundary; higher
accuracy F32/TF32-X3 is further from native. These failures remain recorded.

## CPU discriminating calculation

Quantize both **the actual saved native LN output** and **the unchanged actual
projection weight** to a TF32 grid. Multiply in NumPy FP64, then cast the result
to FP32. This tests input representation, not native GPU accumulation order.
Use all 256 rows at N16, all 289 rows at N17, and the 1,024 actual N437 rows at
`np.unique(np.linspace(0, 190969 - 1, 1024, dtype=int))`. No mask removal or
candidate-normalized input is used. The N437 CPU result is explicitly a row
sample, not a full-shape closure.

For finite FP32 values viewed as `uint32` bits, the tested quantizers were:

```python
rtz = bits & np.uint32(0xffffe000)
rne = (bits + np.uint32(0xfff) + ((bits >> 13) & 1)) & np.uint32(0xffffe000)
rna = (bits + np.uint32(0x1000)) & np.uint32(0xffffe000)
# Interpret quantized bits as float32, then:
result = (quantized_x.astype(np.float64)
          @ quantized_weight.astype(np.float64).T).astype(np.float32)
```

All 15 case/projection combinations pass the frozen `atol=rtol=1e-4` gate for
RNE (nearest, ties-to-even) against native and RNA (nearest, ties-away) against
JAX `high`. Maximum remaining error is `1.33514404296875e-5` in either matching
direction. Opposite tie rounding gives errors up to approximately `0.00118`;
RTZ gives errors up to `0.024013519287109375` against native. This is strong
evidence for ordinary GEMM tie rounding, not proof that all full-module drift
has this single cause.

Do not transfer this RNE hypothesis to the separate native Triton triangle-mul
linear kernel: its earlier same-operand GPU controls established RTZ. Kernel
contracts must remain distinct.

## Immutable evidence identities

Artifact labels contain no machine-specific paths. Full-operator native capture:
`runtime-operators-20260908-suRDO0/openbind-native`; boundary native/candidate:
`boundary-controls-20260908-zFMCaW/openbind-{native,candidate}`.

| Artifact | SHA256 |
|---|---|
| Full-operator native manifest | `874f10a64a493c27fafa4bfdb640080e91b3bbe31a688b7720508dce94c4463e` |
| Boundary native manifest | `83a7726dee84a70e30521433885aad2b5ed4507f897fa5f916a06ddc20b396fd` |
| Boundary candidate manifest | `19d3446698c9af85b4a09cf9b98231780896a42fa44004053ff641485491c59c` |
| Original selected weight archive | `bfa9ae16376cd5c6a34587c8cb4ea0e7235557182ea64987d548208b403a72a1` |
| N16 native LN bytes | `285f88cda6cdefcf3f8788fa972d9f5d58612fd680ab7a1409f0a34e2e56c99b` |
| N17 native LN bytes | `d55088b507043813a8fbca3370e05cfa9aca87ca85e0a7548a97c25cd9f56d93` |
| N437 native LN bytes | `c864563098a5ea80ff905941a05673601efb729cd7d0bb0d3cd4284fa8eccf0a` |
| N437 selected row indices, int64 | `b5dc3738aca6e6e6bab71c7adb0a7801353b450baed2adfcc5ebb4b75fbfa91e` |

## Full-shape GPU rounding control

The optional `--operand-rounding rne` in
[`openbind_projection_boundary_probe.py`](../bench/openbind_projection_boundary_probe.py)
prequantizes the complete captured teacher and all five projection weights on
the host, then uses **unchanged current `high` GEMMs only**, at all three full
shapes. Original and effective operand hashes are both recorded. Native captures,
candidate LayerNorm diagnostics, baseline tolerances and production code are
unchanged. RNE-grid operands should be invariant under subsequent RNA rounding;
GPU job 458 establishes this for the captured projection operands:

| Full token shape | Strict passing projections | Maximum absolute error |
|---|---:|---:|
| N16 | 5/5 | `6.198883056640625e-6` |
| N17 | 5/5 | `5.7220458984375e-6` |
| N437 | 5/5, all bitwise | `0` |

The frozen gate remains `atol=rtol=1e-4`. Unlike the earlier CPU row sample,
this comparison covers every projection element at each full shape. Artifact
label: `openbind-rne-control-20260908-kDNapj/candidate`; manifest SHA256:
`a1718c2149f039b91d332ca0eae9c8de5b4d61ddc29488206e8737aaf3938eff`.
This confirms the teacher-forced host-prequantization control, not the pending
private runtime implementation, full triangle modules, or full-model parity.

The host helper preserves signed zero, infinities and NaN payload bits; the
benchmark's admitted operands still must be finite FP32, including after
quantization. Exact positive/negative ties, grid idempotence and nonfinite bit
preservation have dedicated CPU regressions.

## Private runtime follow-through: full operator panel still fails

The local private wrapper now applies exact RNE at ordinary dot operands only.
Separate custom Triton RTZ linear and attention cores are unchanged. Formula
interpretation remains explicitly distinct from actual GPU execution; default
`interpret=False` and its RNE placement have independent tests. Parent affected
OpenBind and replay checks passed 152 tests; the writer's broader scoped gate
passed 428. Neither substitutes for GPU numerical evidence.

Job 464 reruns all 52 full modules against unchanged native job 448. It still
passes only **1/52** strict comparisons. Maximum errors by family are:

| Operator | Passing / total | Maximum absolute error |
|---|---:|---:|
| Triangle multiplication outgoing | 1/13 | `0.053714752197265625` |
| Triangle multiplication incoming | 0/13 | `0.04686546325683594` |
| Triangle attention starting | 0/13 | `0.6337738037109375` |
| Triangle attention ending | 0/13 | `0.5737466812133789` |

Artifact: `openbind-runtime-rne-20260908-rSMiFD/candidate`; runtime SHA256:
`37ab01948ad523a6af626e10994a27455001053b645651356e1bf7b54cb49e19`.
Teacher-forcing native LN into projections removed an upstream boundary from
that earlier test. It does not prove that candidate LN values followed by RNE,
or the rest of either full module, match. Actual native dispatch and the first
remaining full-module boundary require capture before further changes. No
public backend connection, model admission, or tolerance change is made.

### Norm instruction follow-up

Saved native PTX uses `sqrt.approx.ftz.f32` followed by `div.full.f32` for
the custom multiplication norm's reciprocal standard deviation. The private
width64/128 GPU path now spells those instructions explicitly; interpretation,
other widths, mean/second-moment division and affine order are unchanged.
Parent norm/ESM probe checks passed 84 tests, including actual unpatched
Triton lowering. This is distinct from attention's ordinary Torch LayerNorm.

Job 474 still passes 38/38 strict norm cases, with 24 bitwise. Every output
archive SHA256 matches the prior job 417 exactly: **no measured numerical
improvement** follows from this instruction change on this panel. The N437
width128 case retains 7,532,666 unequal values, max `2.384185791015625e-7`.
Tiny norm differences cannot be dismissed before RTZ/RNE quantization; actual
remaining arithmetic boundaries need isolation. No full-module rerun or
default connection is justified by unchanged norm outputs alone.
Artifact: `native-norm-boundaries-20260908-3TfgmJ/openbind-norm`; runtime
SHA256 `7aebbd5d2b73bc0025c81d43b32aba8aa33fa78cdc9076bb9feb2b03f40d81f9`.

### Final affine FMA closes the custom norm panel

Native PTX ends with `sub → mul → fma.rn.f32(normalized, weight, bias)`.
The candidate TTIR instead expressed separate final multiply/add. On the
width64 seven-row example, CPU fused affine reproduced all native outputs
while separate rounding reproduced all candidate outputs, including every
one of the 138 differences. The private width64/128 GPU path now explicitly
uses that final FMA; statistics, input normalization and other paths are
unchanged. Parent actual-lowering/regression checks passed 63 tests.

GPU job 478 then passes **38/38 bitwise**, including every value in the full
N437 width128 case. This isolates the final affine rounding for the measured
norm panel; it is not a claim of full-module, ordinary-LN or model parity.
Artifact: `runtime-affine-split-20260908-VLMhj2/openbind-norm`. Full-module
job 481 is the next separately recorded validation.

Job 481 completes but full-module parity remains open: 6/52 strict cases pass,
up from 1/52. Multiplication outgoing and incoming each pass 3/13 (one bitwise
each); both attention families still pass 0/13. Family maximum errors are
unchanged from job 464. The standalone norm panel is not an exhaustive set of
actual intermediate values. The next boundary must capture the actual full
module's first norm and following projection, while preserving the native
module output/repeat, rather than infer full-module closure from isolated
kernels. Artifact: `runtime-affine-split-20260908-VLMhj2/openbind-full`.

### Ordinary attention LayerNorm: exact boundary, incomplete module

Job 494 tests the existing CUDA Welford helper on the full native ordinary
attention LayerNorm inputs at N16, N17 and N437. All three norm outputs and
same-executable repeats are bitwise equal. This norm is distinct from the
custom multiplication norm above. The following projection controls still
consume the native norm output and are not evidence of an integrated wrapper.
Artifact: `openbind-ordinary-norm-result-20260908-TG4yap/candidate`.

The private native triangle attention wrapper now uses that CUDA helper for
its ordinary first norm. CPU and interpretation retain the previous formula;
custom multiplication norms and public backend defaults are unchanged. Source
provenance now includes the shared norm helper and context-parallel module.

Integrated job 496 completes all 52 full-module cases with finite outputs,
but still passes only 6/52 strict and 2/52 bitwise. The N437 pairformer starting
attention remains nonmatching: RMSE 0.00007095149046210782, max
0.0194854736328125, 99,831 unequal entries. These are internal representation
errors, not structural RMSD. Exact standalone ordinary norm does not close the
full attention wrapper or model. Actual fused projection/layout boundaries
remain to be isolated. Artifact:
`openbind-norm-integrated-result-20260908-dtudjq/candidate`; source
`openbind-norm-integrated-source-20260908-doyvtp`.
