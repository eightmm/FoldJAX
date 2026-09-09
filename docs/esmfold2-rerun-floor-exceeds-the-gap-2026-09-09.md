# ESMFold2's rerun floor is larger than its distance to native

The cross-model standard records ESMFold2 as `n/a` because "neither side exposes
a complete injectable tape". That reason is wrong, and the real one is worse for
any single-run comparison.

## An end-to-end pair already exists

`esmfold2-native-full-tape-20260907-5sak-a` and
`esmfold2-jax-full-tape-20260907-5sak-a` hold five-sample coordinates from both
sides on the same case, same `input_sha256`, same `seed` (101), same sample
count. Their metadata records `core_only_shared_features: true`, and the port's
output directory carries no `lm_schema`, no `native_shim_control`, no
`injection.npz` -- the interchange and shim paths in `bench/esmfold2_tape.py`
were not taken. Both sides computed their own language model, trunk and
diffusion; only the preprocessed input features are shared.

So an end-to-end comparison was available without running anything:

```
port vs native: max 1.8991  median 0.1731
```

## The control that voids it

The port rerun with the identical command, same seed, same tape, only the output
directory changed:

| Comparison | max | median | per sample |
| --- | ---: | ---: | --- |
| **port rerun vs port (floor)** | **2.8295** | **0.1637** | 2.8295, 0.1670, 0.1620, 0.1637, 0.0885 |
| port vs native | 1.8991 | 0.1731 | 1.8991, 0.1164, 0.1343, 0.1841, 0.1731 |
| port rerun vs native | 1.3095 | 0.1940 | 1.3095, 0.1783, 0.1940, 0.2979, 0.1686 |

The port cannot reproduce itself to better than 2.83 A max and 0.16 A median.
Its distance to native is 1.90 A max and 0.17 A median -- **smaller than its own
rerun floor on the max, and indistinguishable on the median**.

Note also that the two port runs land at different distances from native
(1.8991 and 1.3095). Two runs of the same code with the same seed disagree about
the answer by more than either disagrees with native.

## What this means

ESMFold2's row cannot be filled by a single-run comparison, and no amount of
tape work changes that. The blocker is not missing infrastructure; it is that
this model's deliberate stochasticity produces run-to-run spread larger than the
port-versus-native effect being measured. A number like "1.8991" placed in the
standard's table beside OpenFold3's 0.0010 would be reporting noise.

The seed does not pin it. Both runs passed `--seed 101` through the same
harness; the spread survives.

## What would fill the row

A distribution comparison, not a run comparison: many seeds per side, and a test
that the two sample distributions agree, with the single-side rerun spread as
the null. That is a different experiment from every other row in the table,
which is the honest reason ESMFold2 does not share their column.

Until then the row should say what is measured here -- that the port sits within
its own sampling spread of native -- rather than `n/a`, and rather than a
single number.

Artifacts: `esmfold2-jax-full-tape-RERUN-20260909` (this session),
`esmfold2-{jax,native}-full-tape-20260907-5sak-a`.

## The distribution comparison, run

The section above named a distribution comparison as what would fill the row.
Two more port runs were added, giving four port runs against the one native run.

| Distribution | n | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| port vs port (floor) | 30 | 0.1305 | 0.4101 | 0.7120 | 2.8665 |
| port vs native | 20 | 0.1711 | 0.4643 | 1.5637 | 1.9298 |

**Every one of the twenty port-versus-native distances falls inside the range of
the port-versus-port distances** (fraction 1.00). The port's largest disagreement
with native, 1.93 A, is smaller than the port's largest disagreement with
itself, 2.87 A.

A Mann-Whitney U on the two samples returns p = 0.029, with port-versus-native
shifted slightly higher.

## Why that p-value should not be reported as a difference

The twenty cross values are not twenty independent draws. They are four port
runs measured against **one** native run, so every one of them shares a single
native sample, and the thirty floor values reuse the same four port runs
pairwise. The test's independence assumption fails in both samples.

With one native draw there is no way to separate "the port's distribution
differs from native's" from "this particular native run sits slightly off-centre
in its own ensemble". A shift of 0.04 A in median is exactly the size that one
off-centre draw would produce.

So the measurement stands where the previous section left it, now with 30 floor
values instead of 5: the port is inside its own sampling spread of native, and
nothing sharper can be said from one native run.

## The blocker, named precisely

More native runs need the upstream torch model, and that path is broken in the
one environment that has it:

```
$ foldjax-bench/jctc-matrix-20260904/upstream-root/esmfold2-venv/bin/python
  -c "import transformers.models.esmfold2"
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

`torch` and `transformers` are both installed there; `huggingface_hub` is a
version ahead of what this `transformers` expects. No other virtualenv on the
machine has `transformers` at all.

That is a dependency repair, not a numerics question, and it changes an
environment other work may depend on -- so it is recorded here rather than done.
Repairing it makes the ESMFold2 row answerable: run native at several seeds, and
test the two distributions with both sides properly replicated.

## The blocker was repaired, and the row is now answerable

The import failure above was not one version skew but a chain, and none of it
needed the shared virtualenv changed:

1. `transformers.utils.hub` imports `is_offline_mode` from `huggingface_hub`,
   which no released version exports (checked 0.35.3, 0.36.0, 0.36.1, 0.36.2,
   1.0.0, 1.1.0). The name is a plain read of the offline flag, so a
   `sitecustomize.py` in an overlay directory restores the alias.
2. The *installed* transformers (5.16.1) does not contain `ESMFold2Model` at
   all -- its class is `EsmFold2Model`. The harness never used the installed
   package: `bench/esmfold2_tape.py` takes `--upstream-source-root` and checks
   `inspect.getfile(ESMFold2Model)` against it. The tree is
   `jctc-matrix-20260904/upstream-root/transformers-esmfold2` (transformers
   4.57.6), and it matches the venv's own `tokenizers` 0.22.2.

With the overlay on `PYTHONPATH` the reference model loads and the native arm
runs. The overlay is at `foldjax-bench/esm-hfhub-overlay` and contains one file.

### The overlay did not contaminate the reference

A second native capture (`esmfold2-native-full-tape-N2-20260909`) lands inside
the first one's own ensemble: max 2.1838, median 0.1364 -- the same scale as
every other pairing here. Had the shim changed behaviour, this is where it would
show.

### Both sides replicated

Four port runs and two native runs, median whole-system Kabsch per run pair:

| Pairing | pairs | values |
| --- | --- | --- |
| within port | 6 | 0.1637, 0.0779, 0.1029, 0.1621, 0.1391, 0.1218 |
| within native | 1 | 0.1364 |
| cross | 8 | 0.1731, 0.1739, 0.1940, 0.1943, 0.1605, 0.1312, 0.1375, 0.1612 |

Exact permutation test over the six run labels -- the correct treatment, since
under the null "port and native draw from the same distribution" the labels are
exchangeable and the individual sample distances are not independent:

```
observed cross-minus-within: +0.0366 A
exact permutation, 15 arrangements -> p = 0.067
```

The observed labelling is the most extreme of all fifteen, and 1/15 = 0.067 is
the smallest p this design can produce. So the separation is real in direction
and the test is at its floor; it does not reach conventional significance with
six runs.

## What ESMFold2's row should say

**Port-versus-native separation ~0.037 A, once the sampling spread is quotiented
out.** That is the same band as the other passing models -- OpenDDE's 0.02-0.07,
Protenix-v2's 0.0335 on 7st3 -- and nowhere near the 0.5 A contract threshold.

ESMFold2 is not outside the standard. It needed a different estimator, because
its stochasticity is larger than the quantity being estimated, and a single-run
number reports noise. With runs replicated on both sides the model sits with the
others.

More native runs would sharpen the p-value; they would not change the estimate
much, since the six cross pairs already agree to within 0.06 A.

---

# Retraction: the "rerun floor" was XLA autotuning, not the model

Everything above rests on the port's replay being irreproducible -- a measured
"floor" of 2.8295 A max that swamped the 1.8991 A distance to native, from which
this document concluded ESMFold2 needs a distribution comparison rather than a
matched one. That conclusion is wrong.

## What the tape actually contains

Eight arrays, and they cover every stochasticity source this model has:

```
initial_pair_state          lm_dropout_masks
msa_column_keep             msa_row_choices
diffusion_initial_normal    diffusion_rotation_quaternions
diffusion_translations      diffusion_churn_normals
```

LM dropout, MSA sampling and all four diffusion draws are captured. The tape is
complete, which is the opposite of the "no complete injectable tape" the
cross-model standard gives as the reason for ESMFold2's `N/A`.

## The probe

Four replays of that tape had differed on **every element**, with max diffs of
1.67, 4.04 and 20.10 A. `--xla_gpu_deterministic_ops=true` is not usable here --
it disables autotuning and a Triton gemm fusion then has no default config
(`INTERNAL: No supported config found for HLO ... __triton_gemm`).

Pinning the compiled executable instead: one warm run populating a shared
`JAX_COMPILATION_CACHE_DIR`, then two measured runs reusing it.

```
shared-cache replays: bitwise=True  max|diff| 0.000000  mean|diff| 0.000000
```

**Bitwise identical.** The port's replay is fully deterministic given the same
compiled program. The entire 1.67-20.10 A spread was XLA picking different
kernels between runs, amplified by 5SAK's trajectory into angstroms.

## What this changes

- ESMFold2's tape is complete and is consumed. A matched comparison is available
  now, not blocked on infrastructure.
- The `~0.037 A` permutation estimate above is measuring autotune variance in
  both the cross and within distributions, not model stochasticity. It should
  not be quoted.
- The port-versus-native numbers (1.8991 max, 0.1731 median) were taken without
  a pinned cache on either side and inherit the same variance.

## The general point, which is larger than this model

Any comparison in this project run without a pinned compilation cache carries
kernel-selection variance, and on an amplifying target that variance reaches
tens of angstroms. Several of this session's own measurements -- the OpenDDE
precision arms, the Protenix precision arms, the free-running cross-model table
-- were run without pinning it. Their conclusions rest on effects far larger
than the variance seen here, and the OpenDDE and Boltz-2 arms carried explicit
rerun controls that would have exposed it, but the caveat belongs on record.

The correct next measurement for this row is both arms replayed with pinned
kernels, which is now a small job rather than a research problem.
