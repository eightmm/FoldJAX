# Boltz-2 mixed-precision boundary audit — 2026-09-05

Status: partial numerical-policy repair, **not upstream-equivalence acceptance**.
The subsequent [5SAK boundary follow-up](boltz-5sak-boundary-followup-2026-09-05.md)
repairs the relative-position accumulation and complete-trunk diagnostic;
measurements below retain their original frozen source identity.
The numerical reference is pinned Boltz commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`, with an unmodified source checkout.
The starting FoldJAX commit is `3c9e5e8c57c66dbd7975da812c0b9c6502abc9de`.
No experimental/crystal coordinates or confidence-ranking comparison is used.

[Full per-sample results, input audits and source/artifact hashes](../bench/experiments/boltz-mixed-policy-2026-09-05.json)
record 14 completed n=5 FoldJAX replay cells: six corrected cells and one
5SAK single-track-only ablation for each of two multiplication backends.
At completion of these replays, the live source was verified equal to the frozen replay source, SHA-256
`f4c3d560f119e3ea6b2979401a77a04eaa1cbb902a141f237329208f1fbd721f`.

## Corrected boundaries

- Preserve the original Pairformer `pre_norm_s`, `attention`, and `transition_s`
  weights before narrowing the rest of the trunk. The native single track
  disables autocast. FP32 activations do not recover already-rounded weights.
  Both ordinary layer lists and prestacked parameter containers are covered.
- Diffusion conditioning is not one uniform FP32 island. Native
  `PairwiseConditioning` and final atom/token bias projections execute under
  mixed autocast; the entire `AtomEncoder` executes with autocast disabled.
  The port now narrows the former at Linear boundaries, retaining FP32
  LayerNorm affine and atom geometry/encoder calculations.
- Lazy token bias retains its conditioning projection dtype across the sampler
  boundary. It computes the low-precision projection and widens its result to
  FP32, just as native precomputation followed by score-model `.float()` does.
- The explicit mixed conditioning transition evaluates SiLU internally in FP32
  before narrowing the activation output. This avoids an extra low-precision
  sigmoid rounding inside the activation. The ordinary FP32/default primitive
  path is retained.
- The standalone tape replay uses the same trunk-weight preparation as the
  production predictor and sampler, rather than the previous blanket helper.

This is not a global change to the shared LayerNorm policy. Other models,
checkpoint selection, public precision defaults and scientific tolerances are
unchanged.

The native runtime used PyTorch `2.12.0+cu130`; its
[CUDA SiLU implementation](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)
uses the opmath accumulator before returning the activation dtype.

## Measured native dtype boundaries

An instrumented native 3GCA run (n=5, 200 steps, 3 configured recycles, seed 101,
`bf16-mixed`, native kernels enabled) recorded these actual forward dtypes:

| Boundary | Native dtype |
|---|---|
| Trunk `s` and `z` entering conditioning | FP32 |
| Relative-position encoding | BF16 |
| Pairwise conditioner output | BF16 |
| Atom encoder `q`, `c`, `p` | FP32 |
| Final atom encoder/decoder and token bias projections | BF16 |

Hooks only observe results; they do not replace arithmetic or consume RNG.
The FP32 atom encoder boundary was checked from the entire enclosing context,
not inferred from individual `.float()` expressions.

## Isolated conditioning control

Feed the **same captured native trunk and native features** to both conditioning
policies. This isolates conditioning; it is not independent-preprocessing
evidence. Under JAX `highest` matmul precision, `q` and `c` pass the fixed
`atol=rtol=1e-4` leaf check. Mixed pairwise/bias outputs still fail it.
Matching dtype boundaries does not guarantee bitwise-identical GPU reductions,
fusion or elementary functions. Residuals must not be converted into a pass by
loosening tolerances.

The initial whole-conditioning GPU control precedes the SiLU adjustment and
retains its own source hash. The later whole-conditioning CPU control includes
that adjustment. The per-operator GPU probes below and all 14 structural
replays use the final candidate; the earlier control is not relabeled as a
final-source measurement.

### Where a sub-tolerance difference becomes a larger one

The follow-up captures each native PairwiseConditioning operator's input and
tests that operator separately, without feeding accumulated port errors into
it. On the same GPU, all 12 inspected operator outputs pass `atol=rtol=1e-4`:
the seven Linear outputs and two SiLU outputs are exactly equal; three
LayerNorm outputs have small FP32 residuals.

| Native-input LayerNorm probe | FP32 max absolute difference | Different elements after BF16 cast | Max difference after BF16 cast |
|---|---:|---:|---:|
| Initial pair projection norm | 2.861023e-6 | 32 / 541,696 | 0.00390625 |
| Transition 0 norm | 7.152557e-7 | 9 / 270,848 | 0.000244140625 |
| Transition 1 norm | 5.960464e-7 | 7 / 270,848 | 0.000030517578125 |

Thus a locally passing FP32 leaf comparison does not imply equal inputs after
the next quantization boundary. This is directly measured, not a conclusion
drawn from the final RMSD. It does **not** identify LayerNorm as the sole cause
of the final structure difference, nor excuse the remaining trunk-policy
mismatches. An earlier CPU-versus-native-GPU probe had small Linear differences;
the matched-GPU probe eliminated those, so they are not attributed to the GPU
port's Linear implementation.

## End-to-end protocol

Use the previously independently constructed FoldJAX inputs and the exact
captured preprocessing/sampler tapes. Re-audit raw feature keys, shapes,
dtypes and values before each replay. All common values must match except
the explicitly reported reference-coordinate arithmetic residual (<=1e-4).
Native-only `disto_target` is a classified training target, not a copied input.

Five paired samples use the native 200-step schedule and 3 configured recycles
(4 trunk passes), no MSA subsampling. Atom identity and masks must match; apply
one proper unweighted Kabsch transform to all valid system atoms per sample,
then measure each entity instance without a second alignment or rematching.
The legacy raw-coordinate/correlation exit remains diagnostic, not scientific
acceptance. A benchmark job completing is not a parity pass.

Frozen source snapshots and source/input/tape/weight hashes bind each replay.
XLA multiplication and cuEquivariance multiplication are separate arms. The
historical baseline used cuEquivariance multiplication with XLA attention;
only the corresponding new arm is a same-kernel before/after comparison.
Native mixed references enable fused kernels, so native/FoldJAX attention
implementation remains an explicitly different backend.

### Same-kernel mixed-precision comparison

Both historical and new FoldJAX arms use cuEquivariance multiplication and XLA
attention, against the same native mixed reference and actual RNG tape.
Values are maxima of five entity RMSDs, in angstroms.

| Complex | Historical port | Corrected single weights + conditioning |
|---|---|---|
| 5SAK | protein 16.2093; ligand 13.1694 | protein 5.68975; ligand 0.857441 |
| 1URN | protein 0.070164; RNA 0.313566 | protein 0.114505; RNA 0.0160273 |
| 3GCA | RNA 0.026244; ligand 0.028175 | RNA 0.0700688; ligand 0.0852626 |

The repair is not monotonically improving: 5SAK and the 1URN RNA improve, but
the 1URN protein and both 3GCA entities worsen. None of these measurements
establish native mixed-precision equivalence. XLA multiplication controls are
retained separately; comparing one of them directly with the historical cueq
arm would confound the policy change with the kernel change.

The same-kernel 5SAK ablation with only original single-track weights preserved
gives protein **16.0354 Å**, ligand **4.59407 Å**. Adding the conditioning repair
gives **5.68975 / 0.857441 Å**. This is a controlled policy contrast, not proof
that the remaining gap has one cause.

### FP32 regression controls

The corrected candidate, retaining the historical cuEquivariance multiplication
and XLA attention configuration, gives:

| Complex | Maximum entity RMSDs over five samples (Å) |
|---|---|
| 5SAK | protein 0.00566715; ligand 0.000779158 |
| 1URN | protein 0.0000654075; RNA 0.0000435657 |
| 3GCA | RNA 0.0000453977; ligand 0.0000126714 |

5SAK lies within the previously measured fixed-tape FP32 variability, rather
than demonstrating bitwise reproducibility. The historical fresh 5SAK protein
maximum was 0.029999 Å and its repeat was 0.002405 Å. The small FP32 controls
do not validate the mixed-precision path.

## Remaining differences — not fixed by this patch

1. Trunk features and non-single-track parameters are still broadly narrowed.
   Native atom-input geometry and embedding/normalization parameters have more
   selective FP32 retention. Do not claim the raw-input audit also proves
   equality after these internal casts.
2. Trunk LayerNorm affine, residual promotion, and MSA/triangle contraction
   output policies still require operator-level alignment. Native `z` is FP32
   in the captured run; the port's low-precision pair carry is not equivalent.
3. Sparse relative-position and MSA embedding rewrites split a native single
   Linear into multiple operations, introducing additional BF16 rounding sites.
4. Fused triangle calls bypass ordinary Linear/LayerNorm helpers. Their actual
   input/weight/accumulator policy needs independent native kernel captures.
5. Confidence/affinity heads and other supported input families are outside
   these coordinate-only three-complex measurements.

Accordingly, FP32 remains the appropriate numerical debugging control here;
this work does not establish better crystal accuracy or justify changing every
model's default dtype. Complete native mixed-precision parity is still open.

## Verification

- New precision regressions: 8 passed (original parameter values with both
  list/prestacked storage; eager/JIT bias projection; scan/unrolled full score
  with eager versus lazy mixed conditioning).
- Focused Boltz-2 run before the final two score tests: 280 passed, 3 skipped,
  9 slow/checkpoint tests deselected. The final full gate includes the added
  tests.
- Full isolated CPU gate with a fresh model-store directory:
  `JAX_PLATFORMS=cpu FOLDJAX_HOME=<fresh-empty-directory> OMP_NUM_THREADS=4
  python -m pytest -q -m 'not network' --cov=foldjax
  --cov-report=term-missing --cov-fail-under=80`:
  **3,572 passed, 411 skipped, 8 deselected**, 66 warnings; coverage **87.37%**.
  Skips retain optional asset/runtime requirements; network tests are excluded
  by the existing CI policy. An earlier local-asset-inclusive CPU run was
  terminated after entering checkpoint-heavy inference, not counted as a pass.
- Ruff, dependency lock and `git diff --check` passed.
- Read-only peer review found no must-fix issue in the limited patch; that
  review does not endorse universal numerical parity.
- Hosted CI, release wheel verification and all-six-model GPU release gates
  were not run for this unlanded, Boltz-only candidate. No main push is claimed.
