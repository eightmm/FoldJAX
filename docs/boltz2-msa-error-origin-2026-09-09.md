# Where Boltz-2's 5SAK error is born: the MSA module, on identical inputs

Status: a localization from existing captures. No fix, no admission. It narrows
the one cell that fails the cross-model gate.

## The cell this is about

The [cross-model gate](common-error-standard-2026-09-09.md) is crossed exactly
once, by Boltz-2 on 5SAK at 1.1761 Å. That cell is already decomposed:
substituting upstream trunk tensors collapses all five sampler trajectories from
`[1.18, 0.17, 0.06, 1.04, 0.48]` to below 0.001 Å, so the sampler and the scan
lowering are excluded. What remains is a trunk `s_rmse` of 0.0065 and `z_rmse`
of 0.0135 that this target's long diffusion trajectory turns into an angstrom.

This narrows *where in the trunk*.

## The stage table

Native capture `boltz-amp-closure-20260907/native-5sak` against a JAX capture
built from it (`--upstream-capture`), so the two are paired by construction.
437 tokens, four recycles.

| Stage | RMSE | Relative |
| --- | ---: | ---: |
| `rel_pos` (pair initialisation) | **0.000000** | 0.000e+00 |
| `input_embedder` (single) | 5.901610e-06 | 2.460e-05 |
| cycle-00 MSA `input_z` | 6.158335e-04 | 9.765e-05 |
| **cycle-00 MSA `delta_z`** | **3.238075e-02** | 2.073e-03 |
| cycle-00 pairformer `output_z` | 7.860285e-02 | 3.295e-03 |
| cycle-03 MSA `input_z` | 1.369353e-02 | 1.877e-03 |
| cycle-03 MSA `delta_z` | 3.427136e-02 | 2.035e-03 |
| cycle-03 pairformer `output_z` | 7.577320e-02 | 3.050e-03 |

The pair representation starts **bitwise identical**. The single embedding is
2.5e-5 relative. The pair entering the first MSA module is 6.2e-4. The update
that module returns is 3.2e-2 -- **a factor of 53 across one module**, and the
largest single step anywhere in the trunk. Everything after it is that error
being carried and roughly doubled by the pairformer, then recycled.

## The control that makes it arithmetic

Every MSA input the JAX side consumed was compared against the native feature
file:

| Feature | |
| --- | --- |
| `msa`, `msa_mask`, `msa_paired` | bitwise equal |
| `deletion_value`, `has_deletion` | bitwise equal |
| `token_pad_mask` | bitwise equal |

Six of six identical. Combined with a pair input already matched to 6.2e-4 and
a bitwise-identical `rel_pos`, the 53x is not a different input being fed to the
same arithmetic. It is the same input through different arithmetic.

## What this does and does not say

It says the search for Boltz-2's remaining trunk residual belongs inside the MSA
module at the first recycle, not in the pair initialisation, the input embedder,
the pairformer, the sampler or the scan -- all of which are now either exact or
downstream carriers.

It does not name the operation inside that module. The native capture stores
only `input_z` and `delta_z` for this stage, so no finer boundary exists in the
artifacts to compare; the JAX side's extra keys are that module's inputs, which
is what made the control above possible but adds no internal boundary. Isolating
further needs a native capture with sub-module boundaries, which is a capture
change rather than an analysis one.

It also does not establish that closing this closes the cell. The decomposition
says the trunk residual is what the trajectory amplifies; it does not say this
stage is the whole trunk residual.

## Teacher-forcing the module: the input contributes nothing

Feeding the JAX MSA module native's own `input_z`, with the bitwise-identical
features, and comparing its `delta_z` against native's:

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| in-run captured `delta_z` | 3.238075e-02 |
| JAX `input_z` -> JAX module | 3.375386e-02 |
| **native `input_z` -> JAX module** | **3.375255e-02** |

The two teacher-forced arms agree to five significant figures. The 6.2e-4
difference in the pair entering the module contributes essentially nothing to
the 3.2e-2 leaving it. The residual is the module's arithmetic, on identical
inputs.

(The standalone arms sit 4% above the in-run capture, because a module run
outside the whole graph does not get the same chunking and autotuning context.
Read the arms against each other, not against the in-run absolute.)

## Every exposed knob is excluded, and two make it worse

Same teacher-forced setup, one knob at a time:

| Arm | RMSE | |
| --- | ---: | --- |
| shipped (cueq triangle, xla glu, `highest`) | 3.375651e-02 | |
| `triangle_backend=xla` | 3.375084e-02 | unchanged |
| `use_scan=False` | 3.374830e-02 | unchanged |
| `matmul_precision=tensorfloat32` | 4.059741e-02 | **20% worse** |
| parameters cast to bfloat16 | 4.308405e-02 | **28% worse** |
| parameters cast to float32 | 3.375281e-02 | unchanged; they already are |

The fused cuEquivariance triangle kernel was the obvious suspect and moves the
result by 0.02%. The scan lowering moves it by 0.02%. The two knobs that do
move it both move it the wrong way, which is itself a result: `highest` is the
right precision for this module, and a blanket bfloat16 is not what native
does.

The parameters load as 58 float32 leaves and the module already runs fp32.

## What that leaves

Not a knob. The port is at its best available configuration and still 3.375e-2
from native, so closing this is an implementation difference rather than a
setting.

The shape of it is constrained by the two failed arms. A blanket bfloat16 cast
is *worse* than fp32, but upstream runs this trunk under autocast -- so native
is materialising bfloat16 roundings at particular points while keeping others
in fp32, and the port is in fp32 throughout. Matching means reproducing where
those boundaries fall, not choosing a dtype. That is per-operation work inside
the MSA module and it is what the native AMP boundary investigation already
underway is about.
