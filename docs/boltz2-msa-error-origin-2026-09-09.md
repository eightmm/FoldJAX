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
