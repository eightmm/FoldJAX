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

---

# Correction: the baseline was fine; the `highest` run is incoherent

The section above invalidates the probe because my `high` baseline read 0.262 A
where the cross-model standard records 0.0335 A. That reasoning is wrong, and
the control that shows it was available without running anything.

The panel's **own** unmodified foldjax arm, produced by the other session on
`protein_1ubq`, scores against the panel's own native capture at:

```
astra panel 1ubq foldjax vs native: max 0.11022  median 0.01995
standard table protenix 1ubq      : 0.0024
```

So the panel's untouched arm is off the standard table by the same kind of
factor my arm was. The gap is not a broken baseline; it is that this panel
measures against a different native capture than the matched-tape artifact the
standard table was built from. Comparing a panel number to a table number was
never a positive control, and I should not have treated it as one.

## What actually disqualifies the `highest` arm

A property of the run itself, measured after the fact. Within-arm spread of the
five diffusion samples, each against sample 0:

| Arm | sample spread | coordinate range | NaN |
| --- | --- | --- | --- |
| `high` | 3.57, 0.91, 1.39, 0.81 | -47.0 .. 44.9 | none |
| `highest` | 15.44, 17.01, 16.58, 16.60 | -55.5 .. 52.0 | none |

The baseline's five samples agree with each other to about 1-3.6 A, which is a
normal diffusion ensemble. The `highest` arm's samples are 15-17 A apart. The
coordinates are finite and in range, so nothing crashed -- the model simply is
not producing a converged ensemble.

## The finding, stated at the strength the evidence supports

**Protenix-v2's port does not run correctly with the global matmul default
forced to `highest`.** That is a behavioural claim about the port, evidenced by
the ensemble incoherence, and it is independent of any comparison to native.

It also means the OpenDDE lever does not transfer. On OpenDDE, `highest`
produced a coherent ensemble that landed closer to native. Here it produces an
ensemble that does not hold together, so the question of whether fp32 would
reduce Protenix's error against native remains genuinely unanswered -- the
experiment that would answer it needs a narrower intervention than the global
default, and finding out why the global one breaks inference is the prerequisite.

That is consistent with this port's known accidental fp32 island in the
diffusion module: a global precision change is not a small perturbation here.
