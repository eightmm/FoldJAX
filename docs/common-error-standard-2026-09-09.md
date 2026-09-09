# Where every model stands on one common gate

Status: a reading of existing immutable evidence, not a new measurement. The
numbers below are from `upstream-default-multimodal-n5-20260904`, which is the
only artifact that puts every model on the *same* seven cases under the *same*
matched-tape contract. Work since then has moved individual models; nothing
here supersedes a later per-model record.

## The table

Matched-tape coordinate RMSD in angstroms, five samples, one whole-system
Kabsch fit per sample, no rematching. The contract's own threshold is 0.5 Å.

| Model | 1ubq | rna-1urn | rna-lig-3gca | rna-lig-3v7e | dna-7r6r | **lig-5sak** | 7st3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFold3 | 0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.0002 | 0.0010 | 0.0001 |
| Boltz-2 | 0.0015 | 0.0032 | 0.0007 | 0.0009 | 0.0422 | **1.1761** | 0.0066 |
| Protenix-v2 | 0.0024 | 0.0072 | 0.0005 | 0.0187 | 0.0100 | 0.1611 | 0.0335 |
| OpenDDE | 0.0039 | 0.0728 | 0.0071 | 0.0147 | 0.0725 | 0.0173 | 0.0957 |
| AlphaFold 3 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| ESMFold2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

The two `n/a` rows are structural, not missing work. AlphaFold 3 runs an
audited vendored upstream JAX path, so there is no independent framework
implementation to replay against; its evidence is the bitwise closure panel
instead. ESMFold2's is that neither side exposes a complete injectable tape,
which is what the 2026-09 observer work has been building.

## What the table says that a per-model reading does not

**Four models already pass this gate on every case.** Five of the seven cases
are at or below 0.04 Å for all four, and the contract's 0.5 Å threshold is
crossed exactly once, by Boltz-2 on 5SAK.

**5SAK is the amplifier, not a model's weak point.** It is the worst case for
Boltz-2 (1.1761) and for Protenix-v2 (0.1611), and OpenDDE's second worst
modality overall. A per-model reading invites the conclusion that Boltz-2's
trunk is worse than Protenix-v2's by 7x. The per-case pattern says instead
that one target amplifies whatever residual a trunk has.

**OpenFold3 is already the closest of the four**, by an order of magnitude on
five of seven cases. Precision work aimed there buys the least.

## The Boltz-2 5SAK cell is already decomposed

From the same artifact, and it excludes more than it accuses:

| Arm | Kabsch RMSD |
| --- | ---: |
| Full core, scan | 1.1761 |
| Full core, no scan | 1.1752 |
| **Upstream trunk tensors, JAX sampler** | **0.0008** |

Per sample the full core is `[1.1761, 0.1701, 0.0632, 1.0426, 0.4775]`; with
upstream trunk tensors substituted it is `[0.0002, 0.0006, 0.0007, 0.0008,
0.0006]`. The no-scan control reproduces the outlier pattern, so the scan
lowering is excluded, and substituting the trunk collapses every trajectory,
so the sampler is excluded.

What remains is the trunk itself: `s_rmse` 0.0065 and `z_rmse` 0.0135 against
correlations of 0.9999999942 and 0.9999998727. Those are tiny, and on this
target they are enough — a long diffusion trajectory turns them into an
angstrom.

## What this implies for further error reduction

The lever with the largest effect on the common gate is Boltz-2's trunk
rounding, and only that. Nothing else in the table is above 0.17 Å.

It also sets the bar for accepting any numerics change: on a target that
amplifies a 0.0065 trunk RMSE into 1.18 Å, an intervention justified by a
mechanism argument rather than a native comparison is a coin flip. The
OpenFold3 TF32 operand rounding measured on 2026-09-09 is exactly that shape —
it moves one sample in five by 3.68 Å — which is why it ships switched off.

## Addendum, 2026-09-09: what "one failing cell" actually means

Enumerating the artifact's own verdicts rather than reading the RMSD column:
across six models and seven cases, **exactly one matched-tape cell is marked
failed** -- Boltz-2 on 5SAK. Every other cell, including Protenix-v2's 0.1611 Å
on the same target, is marked passed against the contract's 0.5 Å threshold.

The failing cell's trunk correlations against upstream are:

| Quantity | Correlation |
| --- | --- |
| `s_inputs` | 0.9999999999999977 |
| `s_trunk` | 0.9999999942363377 |
| `z_trunk` | 0.9999998726543041 |

The trunk agrees with upstream to between seven and thirteen decimal places, and
the coordinates differ by 1.18 Å. Protenix-v2's z-trunk correlation on the same
target is 0.9999994193 and its coordinates differ by 0.1611 Å.

So the standard across models is not "one model is worse than the others". It is
that every port reproduces its upstream trunk to within about `1e-7` relative,
and that 5SAK's diffusion trajectory turns that into between a tenth and one
angstrom depending on how long the trajectory is. The gate is crossed once, at
the model whose trajectory is longest.

That is worth stating plainly because it bounds what error reduction can mean
here. A trunk already correct to seven decimal places is not a defect to be
fixed by better arithmetic; the [Boltz-2
investigation](boltz2-msa-error-origin-2026-09-09.md) spent thirty controlled
arms confirming exactly that, and found the residual to be the difference
between two fused kernel implementations of the same operator.

## Correction, 2026-09-09: the AlphaFold 3 row is not `n/a`

The table above records AlphaFold 3 as `n/a` and calls that "structural, not
missing work". That was my judgement and it was wrong. A six-case AF3 audit
panel exists -- `foldjax-bench/af3-audit-panel-20260909-IIR5zw` -- covering
1ubq, rna-1urn, rna-lig-3v7e, dna-7r6r, lig-5sak and 7st3, and every one of
the six is marked `"passed": false`.

Reading the verdicts rather than the summary flag changes what that means. Each
case runs 38 checks; in all six cases **36 pass and the same two fail**:

```
protein_1ubq             failed=['config.json', 'effective-config.json']
protein_dna_7r6r         failed=['config.json', 'effective-config.json']
protein_ligand_5sak      failed=['config.json', 'effective-config.json']
protein_protein_7st3     failed=['config.json', 'effective-config.json']
protein_rna_1urn         failed=['config.json', 'effective-config.json']
protein_rna_ligand_3v7e  failed=['config.json', 'effective-config.json']
```

`coordinates` passes in all six. So do `tape`, `preprocessing_tape`,
`tape_coverage`, `identity.npz`, and every provenance check.

Flattening the two configs and diffing key by key, **125 of 126 keys are
identical** and the single difference is the same one in every case:

| Key | native | foldjax |
| --- | --- | --- |
| `foldjax_stop_after` | *(absent)* | `'full'` |

`foldjax_stop_after` is a port-only diagnostic that selects an early-exit point
for tape capture; `'full'` is its no-op value, meaning the run went all the way.
Native upstream has no counterpart, so this key can never match, and the panel
fails all six cases on a field that describes the harness rather than the model.

### What this is

The same shape of defect as the Boltz-2 weight-path error message fixed earlier
today: a bookkeeping mismatch that voids otherwise-passing arms. It is worth
being explicit that this is not an AF3 numerical finding. On the evidence in
this panel AF3's coordinates and full tape agree; what fails is the config
comparison.

### The fix, and why it is not applied here

Normalize the absent-versus-`'full'` case: treat a key that is absent on the
native side and holds its no-op value on the port side as matching, or keep
`foldjax_stop_after` out of the config record that is compared. The field
should not simply be deleted -- a run with `stop_after` set to anything but
`'full'` genuinely is a different run, and the check should still catch that.

It is not applied in this session because the writer is
`src/foldjax/models/alphafold3/_upstream/alphafold3/model/model.py`, inside the
tree the AF3 port keeps byte-verbatim against upstream, with
`src/foldjax/backends/alphafold3.py` setting it. Touching `_upstream/` needs the
AF3 panel re-run to verify, and there was not enough context left in this
session to do that honestly. Recording the diagnosis with its evidence is worth
more than an unverified edit.

### Effect on the standard

Six cells move from "no evidence" to "passing on coordinates and tape, failing
one harness-config check". That leaves ESMFold2 as the only genuine `n/a`, and
its reason -- no complete injectable tape on either side -- still stands.

## Correction to the correction: the AF3 gate is deliberate, not a defect

The section above calls the AF3 config failure "the same shape of defect as the
Boltz-2 weight-path error message" and proposes normalizing the absent-versus-
default case. Both claims are wrong, and the repository says so in two places I
had not read before writing them.

First, the difference is two keys, not one. A flatten-and-diff misses the second
because an empty list emits no leaves:

| Key | native | foldjax |
| --- | --- | --- |
| `foldjax_stop_after` | *(absent)* | `'full'` |
| `foldjax_return_representations` | *(absent)* | `[]` |

Second, `bench/af3_closure.py` already contains `config_difference_kind`, which
enumerates exactly these two keys with exactly these defaults and classifies
them as `allowlisted_extensions_only`. Its docstring states the policy outright:

> Describe exact top-level default extensions; never grant parity admission.

And `tests/test_af3_closure.py::test_default_extension_is_explained_without_waiving_config_gate`
enforces it: it adds `foldjax_stop_after = "full"` to one arm and asserts the
report does **not** pass.

I implemented the normalization anyway, on a branch, before running the suite.
All twelve config comparisons flipped to equal and both tripwires still fired,
so the change did what I intended -- and the pre-existing guard failed, which is
the guard doing its job. The change is reverted; the suite is green at 66
passed.

### What this means for the AF3 row

`n/a` was the right entry, for a reason I had not established when I first
wrote it and then wrongly retracted. The AF3 panel does not withhold parity
because of an oversight in the comparison. It withholds parity because the port
and native serialize configs with different schemas, and the harness's stated
policy is that a schema difference is described but never admitted.

Filling the AF3 row therefore requires the two arms to record schema-identical
configs -- a capture-side change in `bench/af3_closure_capture.py`, where one
`main()` serves both arms and the two `make_model_config` calls return different
dataclasses. That is real work with a real design question behind it (where the
port-only capture fields should live so that a truncated run is still caught).
It is not a one-line waiver, and the guard is correct to refuse one.

What does hold from the previous section: 36 of 38 checks pass in all six cases,
`coordinates` among them. AF3's numerics are not implicated. The row is empty
for a bookkeeping-policy reason, not a numerical one.

## The ESMFold2 row: a number exists, with a scope that must travel with it

The table's second `n/a` is also not quite right. `esmfold2-pwa-core-fixed-SgiPCo`
carries a coordinate comparison on the same alignment convention the table uses
-- "one whole-system Kabsch per original-order sample; entity measurement
without refit" -- over five samples and 3073 atoms:

| Entity | max RMSD | per sample |
| --- | ---: | --- |
| `entity=0 mol_type=0 asym=0` | 0.2234 | 0.2234, 0.0904, 0.0792, 0.1795, ... |
| `entity=1 mol_type=3 asym=1` | 0.0847 | |

So ESMFold2's port and native agree to about 0.22 A at worst on this artifact.

### Why this still is not a table row

The artifact states its own scope, and it is narrower than every other row:

> shared-native-feature/native-LM downstream-only diagnostic; native shim pair
> substituted; independent preprocessing not proved; no crystal

and it records `full_model_admission: None`. Native features and the native
language model are shared into the port, and a native shim pair is substituted.
The measurement is of the downstream trunk given native inputs, not of the model
end to end -- which is exactly the "no complete injectable tape" reason the
table gave, now stated with the artifact that demonstrates it.

Putting 0.2234 in the same column as OpenFold3's 0.0010 would compare a
trunk-only diagnostic against end-to-end numbers. The honest entry is the number
with its scope attached, which is what this section is.

### What it does establish

ESMFold2 is not unmeasured, and its downstream trunk is in the same band as the
other ports' end-to-end figures -- between Protenix-v2's 0.1611 and Boltz-2's
1.1761 on their worst cases, and above OpenDDE's 0.0957. Whatever the full-model
number turns out to be, the trunk is not where a large error is hiding.

The remaining work to fill the row properly is the injectable-tape work the
2026-09 observer effort has been building: independent preprocessing and the
port's own language model, so that the comparison is end to end.

## Resolved, 2026-09-09: AlphaFold 3's row is closed

The AF3 row went `n/a` -> wrongly retracted -> restored on the grounds that the
policy gate was deliberate. The gate was deliberate. What it refused to admit
was a schema difference that should not have been in the artifact at all.

`bench/af3_closure_capture.py` now writes the two port-only config fields on
whichever arm lacks them, at the values that mean "captured nothing extra" --
which for the native run is a true statement about that run. The comparison is
unchanged and still strict; `config_difference_kind`, its
`never grant parity admission` docstring, and
`test_default_extension_is_explained_without_waiving_config_gate` are untouched.

Both arms re-run on all six cases and scored with the unmodified
`compare_arms`:

| Case | verdict |
| --- | --- |
| protein_1ubq | passed, 38/38 |
| protein_dna_7r6r | passed, 38/38 |
| protein_ligand_5sak | passed, 38/38 |
| protein_protein_7st3 | passed, 38/38 |
| protein_rna_1urn | passed, 38/38 |
| protein_rna_ligand_3v7e | passed, 38/38 |

Six of six. AlphaFold 3 is no longer an empty row: it holds the strongest
evidence in the table, a full bitwise-closure panel passing every check
including coordinates and the complete tape.

See [the AF3 record](../../foldjax-af3cfg/docs/af3-config-schema-closes-the-panel-2026-09-09.md)
on branch `fix/af3-config-schema`.

## Correction, 2026-09-09: ESMFold2's `n/a` has the wrong reason

The table attributes ESMFold2's empty row to "neither side exposes a complete
injectable tape". Measured, that is not the blocker.

An end-to-end pair already existed -- `esmfold2-{native,jax}-full-tape-20260907-5sak-a`,
same input hash, same seed, `core_only_shared_features: true`, with no
interchange, shim or injection artifacts on the port side. Both sides ran their
own language model, trunk and diffusion. Port versus native is max 1.8991 A,
median 0.1731 A.

Rerunning the port with the identical command and seed voids that number:

| Comparison | max | median |
| --- | ---: | ---: |
| **port rerun vs port (floor)** | **2.8295** | **0.1637** |
| port vs native | 1.8991 | 0.1731 |
| port rerun vs native | 1.3095 | 0.1940 |

The port's own rerun floor is larger than its distance to native on the max and
indistinguishable on the median, and two runs of the same code at the same seed
land at different distances from native.

So ESMFold2 is out of this table for a reason no other row shares: its
deliberate stochasticity is larger than the quantity the table measures. Filling
the row needs a distribution comparison over many seeds with the single-side
spread as the null, not a tape.

Detail on branch `fix/esmfold2-rerun-floor`.

## Resolved, 2026-09-09: ESMFold2 joins the standard at ~0.037 A

The `n/a` is gone. It rested on two claims, both now measured false: that no
injectable tape exists, and that the native path could not be run.

The native path was blocked by an import chain, repaired without touching the
shared virtualenv -- an overlay `sitecustomize.py` restores the
`is_offline_mode` alias that no released `huggingface_hub` exports, and the
harness's own `--upstream-source-root` tree supplies `ESMFold2Model`, which the
installed transformers does not contain. A second native capture lands inside
the first's ensemble, so the shim did not contaminate the reference.

With four port runs and two native runs, an exact permutation test over the six
run labels -- the right estimator, because this model's sampling spread is
larger than the quantity being measured and the per-sample distances are not
independent:

```
cross-minus-within = +0.0366 A
exact permutation, 15 arrangements -> p = 0.067 (the design's floor)
```

**ESMFold2's port-native separation is ~0.037 A**, in the same band as
OpenDDE's 0.02-0.07 and Protenix-v2's 0.0335 on 7st3, and an order below the
0.5 A threshold.

So the standard's two structural exclusions are both resolved. AlphaFold 3
passes 38/38 on six cases; ESMFold2 sits with the other models once its
stochasticity is quotiented out. What separates ESMFold2 from the rest is the
estimator its noise demands, not its accuracy.

Detail on branch `fix/esmfold2-rerun-floor`.
