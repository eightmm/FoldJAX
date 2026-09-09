# Protenix precision probe: inconclusive, and the baseline says why

The OpenDDE result -- fp32 matmul cuts coordinate error against native by 1.7x
to 4.5x, because native runs torch TF32 and the port at `high` runs a different
JAX TF32 -- raised the obvious question for Protenix-v2, whose port carries the
same default (`models/protenix/models/predict.py`, `matmul_precision: str =
"high"`) and whose upstream is likewise torch.

## What was run

Source snapshot `protenix-highest-20260909`, copied from the running panel's
tree with that one default changed to `"highest"`. Two arms on
`protein_protein_7st3` against the panel's completed native capture, five
samples, whole-system Kabsch on `cli-output.npz` coordinates.

| Arm | max | median |
| --- | ---: | ---: |
| `high` (baseline) | 0.26194 | 0.13271 |
| `highest` | 26.43498 | 24.12924 |

## Why neither number is usable

The **baseline** is the tell. The cross-model standard records Protenix-v2 on
7st3 at 0.0335 A. My `high` arm reports 0.262 A on the same target -- roughly
8x that, with no intended difference. A baseline that misses the known value by
8x means the arm is not the configuration the standard measured, so the
`highest` comparison is anchored to nothing.

And 26 A is not a precision effect. A matmul-precision change does not move a
structure by 26 A across all five samples uniformly; that is a different
structure, i.e. a broken run, and the most likely cause is the snapshot or the
harness wiring rather than arithmetic.

Recording both numbers rather than the second one alone is the point. Taken by
itself, "fp32 is 100x worse for Protenix" is a publishable-looking result and it
would be wrong.

## What the next attempt needs

Reproduce the standard's 0.0335 A on 7st3 with an unmodified arm **before**
changing anything. That is the positive control this probe skipped. Concretely:
the panel's own foldjax arm for 7st3 had not been produced when this ran -- only
`native` existed -- so there was no in-panel baseline to check against, and I
substituted one of my own without first validating it.

The underlying question stays open and stays worth asking: Protenix-v2's 0.1611
A on 5SAK is the second-largest cell in the panel, its port defaults to `high`,
and on OpenDDE that exact default was the reducible error.
