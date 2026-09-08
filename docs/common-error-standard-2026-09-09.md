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
