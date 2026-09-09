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
