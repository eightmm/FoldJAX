# Scale rows on master: FoldJAX against upstream, 1k-5k tokens, 2026-09-10

One recipe for every row: `bench.run_foldjax` / `bench.run_upstream` from the
`scale-timing-20260910` snapshot, seed 101, n=5 samples, 200 diffusion steps,
10 recycles, one process per row, wall time measured warm after a separate
compile-cache prefill process, `peak_bytes_in_use` of the XLA client (FoldJAX)
or the upstream runner's own peak probe, one RTX PRO 6000 Blackwell (96 GiB)
per row, four rows concurrently. The FoldJAX row materialises the native input
(`measured/inputs/<model>_input.*`) and the upstream row consumes exactly that
file, so both sides see the same MSA and the same chains.

Cases are the pinned length-sweep proteins (`foldjax-bench/jobs`), all
protein-only: L1000_3og2 (1003 tokens, 1 chain), L2000_5dei (2096, homotetramer
A-D), L3000_6ztx (3012, homotetramer), L4000_1gte (4100, homotetramer),
L5000_8e2f (4888, 1 chain). MSAs are the colabfold a3m files pulled from the
workstation.

## Wall time and peak

FoldJAX / upstream, "s / GiB". OOM rows are rows: the card is 96 GiB and the
allocation the run asked for is in parentheses. "upstream OOM" means the
upstream runner ended with zero samples and the card full.

| model | 1003 | 2096 | 3012 | 4100 | 4888 |
| --- | --- | --- | --- | --- | --- |
| OpenFold3 | 100 / 9.1 vs 139 / 14.0 | 404 / 24.5 vs 629 / 44.2 | 951 / 49.2 (`cueq`), 863 / 42.5 (`cueq-full`) vs 1430 / 79.7 | OOM (78) vs upstream OOM | OOM (110) vs upstream OOM |
| Boltz-2 | 91 / 12.3 vs 132 / 15.8 | 318 / 21.3 vs 466 / 46.6 | 806 / 39.9 vs upstream OOM | 3081 / 64.2 vs upstream OOM | OOM (85) vs upstream OOM |

### The FoldJAX column above is superseded above 3,012 tokens (2026-09-14)

Three defaults landed after this table was taken -- Boltz-2's bfloat16 pair
residual (2026-09-11), OpenFold3's partial bfloat16 track (2026-09-12) and its
layer-norm affine exclusion with the `cueq-full` triangle multiplication
(2026-09-14) -- and two of the OOM cells above are not OOM any more. Remeasured
on one snapshot, seed 101, released schedule:

| model | 3012 | 4100 | 4888 |
| --- | --- | --- | --- |
| OpenFold3 | **576 / 23.2** (was 951 / 49.2) | **2368 / 43.4** (was OOM at 78) | **1527 / 60.6** (was OOM at 110) |
| Boltz-2 | **690 / 28.7** (was 806 / 39.9) | **2928 / 50.5** (was 3081 / 64.2) | **1800 / 75.5** (was OOM at 85) |

**Wall time is not monotone in tokens here, and that is the targets rather
than the ports.** 4,100 is L4000_1gte, a homotetramer of about 1,005 residues
per chain; 4,888 is L5000_8e2f, one chain of 771 with padding. Different chain
counts and MSA depths cost differently, so the 4,100 cell being slower than
4,888 on both ports is a property of the set.

**Accuracy at the newly opened sizes, scored against the deposited entry**
(permutation-aware CA, five samples):

* 4,888 / 8E2F, one chain of 771: OpenFold3 **2.17-2.38 A**, Boltz-2
  **2.53-3.07 A**.
* 4,100 / 1GTE: whole-complex 52.8-54.5 A and per chain 13.7-22.1 A on
  OpenFold3, and the arm that keeps the layer norms wide reports 13.67-22.12 A
  -- identical, so this is the target and not the change. Upstream OOMs here,
  so there is no second implementation to compare against. **Recorded as
  completed with the accuracy unresolved**, not as a pass.

**One correction to the upstream column, for 3,012 on OpenFold3.** Scored
whole-chain it reads 18.1 A against the deposited entry, which looks like a
wrong fold and is not one. Fitting the catalase core (122-753) and reading the
N-terminal arm (27-121) separately gives core 0.63-0.98 A and arm 56.8-57.1 A:
upstream misses the same bimodal arm this repository has spent a day
characterising, at the same magnitude the port misses it at on the seeds where
it does. The upstream row also ran `seeds: [42]` where every FoldJAX row here
ran 101 (`inference_query_set.json`), and that arm is a per-seed draw. A
whole-chain superposition translates one local basin miss into a global error,
so 6ZTX has to be scored core-fit/arm-reported on both sides.
| Protenix | 65 / 6.7 vs 96 / 12.6 | 210 / 22.9 vs 242 / 40.8 | 579 / 41.2 vs 650 / 55.9 | 2206 / 73.5 vs 2193 / 86.4 | OOM (94) vs upstream OOM |
| OpenDDE | 235 / 41.3 vs 247 / 57.0 | OOM (23) vs upstream OOM | OOM (48) vs upstream OOM | OOM (89) vs upstream OOM | - |
| ESMFold2 | 155 / 14.4 | 451 / 45.0 | OOM (86) | OOM (32) | - |
| AlphaFold3 | 110 / 4.7 | 236 / 14.6 | 459 / 26.7 | 904 / 52.2 | - |

ESMFold2 and AlphaFold3 have no upstream runner in `bench.run_upstream`
(AlphaFold3's reference is the vendored JAX source itself), so those columns
are FoldJAX only. The AlphaFold3 peak is the harness's XLA-client high-water
mark and undercounts that port's runtime-store allocations; compare it only
with other AlphaFold3 rows.

Reading, where both sides ran:

- FoldJAX is faster on every finished pair (OpenFold3 28-36%, Boltz-2 31-32%,
  Protenix 13-32%, OpenDDE 5% at 1k) and uses less peak memory on every
  pair (OpenFold3 35-45% less, Boltz-2 22-54% less, Protenix 44-47% less,
  OpenDDE 28% less at 1k). At 3k OpenFold3 upstream needs 80 GiB against
  FoldJAX's 49 (`cueq`) or 42.5 (`cueq-full`).
- Boltz-2 at 3012 and 4100 tokens finishes in FoldJAX (40 and 64 GiB) and OOMs
  upstream at both sizes (the upstream row filled the 96 GiB card).
- The memory ceilings match between sides where both OOM: OpenDDE from 2k,
  OpenFold3 from 4k, Protenix at 5k.
- ESMFold2's 3k wall is its `num_samples x L^2` arena; OpenDDE's 2k wall is its
  fp32 pair arena (both recorded before, now measured at the row).

OpenDDE `--option dtype=bfloat16` (opt-in; upstream runs fp32/TF32 so this is
not the parity arm): 1k 149 s / 21.0 GiB against the fp32 row's 235 s / 41.3
GiB (37% faster, half the peak); at 2k it still OOMs (an 85 GiB request,
job 975) and at 3k (24 GiB request, job 976), so bf16 moves the ceiling but
not past 2,096 tokens. Accuracy on the panel (`bench.structures`, tape-free,
CA RMSD Å median; cross read against the two withins):

| case | bf16 within | upstream within | bf16 vs upstream | fp32 within | fp32 vs bf16 (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1000_3og2 | 2.47 | 0.34 | 2.45 | 2.47 | 2.34 (0.02) |
| protein_1ubq | 0.33 | 0.37 | 0.33 | 0.34 | 0.30 (0.01) |
| protein_dna_7r6r | 2.55 | 4.10 | 3.50 | 2.74 | 2.44 (0.16) |
| protein_ligand_5sak | 10.76 | 5.43 | 7.01 | 10.73 | 9.19 (0.09) |
| protein_protein_7st3 | 0.46 | 0.44 | 0.45 | 0.45 | 0.39 (0.02) |
| protein_rna_1urn | 0.15 | 0.17 | 0.15 | 0.16 | 0.15 (0.01) |
| protein_rna_ligand_3v7e | 0.22 | 0.16 | 0.20 | 0.22 | 0.20 (0.01) |

bf16 sits where fp32 sits on every case: its within-set spread equals fp32's
(the 3og2 and 5SAK spreads are the port's known case-specific wide ones, the
same under both dtypes), its cross against upstream equals the within levels
exactly as fp32's did, and against fp32 the closest sample pairs are
0.01-0.16 Å apart (same noise tape, so the dtype alone moves a sample by that
much). Decision (G2): keep fp32 as the released default (upstream runs
fp32/TF32; parity is read there) and document `dtype=bfloat16` as the
recommended opt-in when memory or time matter, since it costs no measurable
accuracy on this panel and buys 37% time and half the peak at 1k; it does not
lift the 2k ceiling. 1AAY (ion case) follows when job 984 lands.

### Evening rows (2026-09-10/11): AMP policy, argument compaction, deterministic pairs, ESMFold2 upstream

Protenix under the upstream token-gated AMP policy (`--amp-policy auto`,
default since 5b40b5d: confidence head bf16 above 2,560 tokens, diffusion bf16
above 3,840) and with the zero-template-geometry compaction (`compact`, X4,
includes the AMP default):

| case | fp32 head (before) | `amp-auto` | `compact` |
| --- | --- | --- | --- |
| L3000_6ztx (3012) | 579 s / 41.2 GiB | 581 / 42.1 | 580 / 37.5 |
| L4000_1gte (4100) | 2206 / 73.5 | 2106 / 70.6 | 2101 / 65.1 |
| L5000_8e2f (4888) | OOM (94) | OOM (89.4) | OOM (89.4) |

The compaction lands 1:1 as predicted (−3.7 GiB at 3k, −5.5 GiB at 4.1k
against the AMP row) with unchanged wall time; the 5k row still fails on the
same 89.4 GiB temporary, so the 5k ceiling is a temporary, not the arguments.
The bf16 diffusion above 3,840 tokens (upstream's own policy) saves 4.5% wall
and 2.9 GiB at 4.1k. The AMP change leaves the ≤3,840-token coordinate path
bitwise intact: the 3k tape replay on the pre-change code and on the new code
with `--amp-policy fp32`, both under `--deterministic-ops on`, agree on every
coordinate (jobs 1048/1049); an earlier pair of replays without the
deterministic flag had differed by 0.4-2.1 Å, which was the port's own
process scatter, not the change. The `auto` policy at 3,012 tokens (bf16
confidence head) leaves the coordinates bitwise equal to the fp32 run (job
1050 vs 1049) and moves atom pLDDT by ≤0.01, pTM/ipTM by ≤2e-4. Boltz-2's uint8 categorical compaction at 4.1k (job
1041): 3133 s / 64.4 GiB against 3081 / 64.2 — the 446 MiB of argument
bytes did not reach the peak, so on this port the in-graph one-hot rebuild is
materialised where the dense argument used to sit (the CPU probe fused it;
GPU did not). The change is output-neutral and stays, with no memory claim.

Boltz-2 pristine upstream (zero tracked diff): 2k 485 s / 46.6 GiB against
the reviewed arm's 466 / 46.6 — the reviewed performance patches did not
move the upstream column at either size.

ESMFold2 upstream (Biohub fork `ef32577f55`, torch 2.13, `bench.run_upstream`
esmfold2 runner, same 5/200/10 schedule): 1k 199 s / 67.6 GiB against
FoldJAX 155 / 14.4; 2k OOM (3203 s to failure at 93.5 GiB) where FoldJAX runs
451 / 45.0. The port's ESMFold2 is 4.7× lighter at 1k and completes 2k where
upstream cannot.

ESMFold2 1k structure agreement (tape-free, CA): cross TM 0.994 (0.990-0.997),
within FoldJAX 0.998, within upstream 0.996; cross RMSD 1.27 Å (0.68-2.85)
against within 0.43 / 1.01 — cross at the upstream within level.

ESMFold2 sequential-sample option (`structure_sample_sequential=true`, G1b) at
L2000: 5 samples 512 s / 44.9 GiB against the batched 440 / 45.0 — the peak
does not move (the folding-trunk pair arena sets it, not the sampler) and the
sequential denoiser costs 16% wall. At 32 samples (the released count) the
batched row is 591 s / 45.2 GiB and the sequential one 1088 s / 44.9 GiB: the
17 GiB attention-logit transient the option removes is not at the peak
either, so the option buys 0.3 GiB for 84% more wall. It stays available and
is not recommended; ESMFold2's wall at 3k is the trunk arena, as the memory
note says, and upstream cannot run 2k at all. Two batched processes at 2k
(base5 vs base5-B) are bitwise identical, so the 0.65-2.6 Å per-atom
differences between the batched and the sequential rows are the sequential
path's own rounding route (per-sample attention kernels) amplified by the
model, not process scatter: arithmetically equivalent (CPU 1e-6), not
bitwise, as with any chunked axis.

Mixed upstream rows at 3k/4k: OpenFold3 3k 1500 s / 81.8 GiB (FoldJAX 1013 /
50.1), Protenix 3k 760 / 57.4 (699 / 42.1) and 4k 1666 / 78.6 (1509 / 66.5),
Boltz-2 3k and 4k OOM (FoldJAX 944 / 41.1 and 2111 / 60.9).

Deterministic execution (`--option deterministic=on`, X2) at L1000_3og2, two
processes per arm, no autotune freeze, coordinates compared file by file:

| port | off A vs B | on A vs B | wall off → on | peak off → on |
| --- | --- | --- | --- | --- |
| Protenix | differ (max 2.3 Å/atom) | bitwise | 65 → 76 s (+17%) | 6.7 → 6.6 GiB |
| OpenFold3 | bitwise | bitwise | 99 → 118 s (+19%) | 9.1 → 9.3 |
| OpenDDE | differ (max 0.66 Å) | bitwise | 233 → 250 s (+7%) | 41.3 → 36.2 |
| ESMFold2 | bitwise | bitwise | 157 → 221/228 s (+41-45%) | 14.4 → 14.4 |
| AlphaFold 3 | bitwise | bitwise | 110 → 103 s (−6%) | 4.7 → 4.9 |
| Boltz-2 | differ (max 9 Å/atom) | differ (max 8 Å/atom) | 92 → 96 s (+4%) | 12.3 → 9.7 |

The knob delivers bitwise repeatability without a per-case autotune file on
Protenix and OpenDDE (whose off pairs differ), and OpenFold3, ESMFold2 and
AlphaFold 3 are already repeatable at 1k either way. Boltz-2 is the one port
where two `on` processes still differ: 3OG2 is a chaotic case for this model
and the residual nondeterminism sits in kernels the flag does not reach
(cuEquivariance triangle kernels, cuBLAS), so Boltz-2 keeps the frozen
autotune cache as its repeatability route (0.0008 Å on 5SAK). Costs: ESMFold2
pays the most (its ESMC blocks move from an eager stack into compiled
deterministic pools), AlphaFold 3 gets faster (Triton GEMMs off routes its
bf16 GEMMs to cuBLAS).

Extra rows on the same cases: Protenix `deterministic=on` costs 13% wall at 1k
(74 vs 65 s) and 9.7% at 3k with Triton gemms disabled (635 vs 579 s;
`docs/protenix-master-panel-2026-09-09.md`); OpenFold3 `cueq-full` at 3k is
9.2% faster and 6.7 GiB lower than `cueq`
(`docs/openbind-master-cueq-panel-2026-09-09.md`).

## Structure agreement without a shared tape

`bench.structures` (CA TM-score and RMSD over every FoldJAX x upstream sample
pair, and within each set). Torch and JAX do not share a diffusion random tape,
so "cross at the level of within" is the most a correct port can show.

| model | case | cross TM | within FoldJAX | within upstream | cross RMSD Å | chain pairing |
| --- | --- | --- | --- | --- | --- | --- |
| Protenix | L2000_5dei | 1.000 (1.000-1.000) | 1.000 | 1.000 | 0.19 (0.12-0.24) | permuted 17/25 cross pairs (near-ties, ~0.01 Å) |
| OpenFold3 | L2000_5dei | 1.000 (1.000-1.000) | 1.000 | 1.000 | 0.21 (0.16-0.24) | permuted 25/25 (was 37 Å by chain id) |
| Boltz-2 | L2000_5dei | 1.000 (0.999-1.000) | 1.000 (1.000-1.000) | 1.000 | 0.27 (0.18-0.35) | permuted 21/25 (near-ties) |
| Boltz-2 | L1000_3og2 | 0.992 (0.988-0.997) | 0.992 (0.987-0.994) | 0.993 (0.991-0.997) | 1.54 (0.66-2.86) | single chain |
| OpenDDE | L1000_3og2 | 0.993 (0.984-0.999) | 0.992 (0.984-0.999) | 0.999 (0.999-1.000) | 2.33 (0.34-4.77) | single chain |
| OpenFold3 | L1000_3og2 | 0.987 (0.984-0.998) | 0.988 (0.985-0.999) | 0.995 (0.984-0.997) | 3.48 (0.43-5.79) | single chain |
| Boltz-2 | L4000_1gte | - (upstream OOM) | 0.993 (0.988-0.998) | - | - | within-FoldJAX permuted 9/10 (was TM 0.59) |

`bench.structures` now pairs chains of identical sequence by minimum RMSD
(exhaustive up to 720 permutations; commit `2339fae`, single-chain numbers
bitwise unchanged). Reading:

- Protenix, OpenFold3 and Boltz-2 at 2k: cross equals within on both sides
  at 0.2-0.3 Å; the two implementations are as close as the sampler allows.
  The OpenFold3 37 Å of the first pass was chain labelling, not structure.
  On a near-symmetric homotetramer most cross pairs pick a non-identity
  assignment that wins by about 0.01 Å over the identity one, so "permuted
  17/25" means the labels differed, not that the labelling mattered.
- Boltz-2 and OpenFold3 at 1k: cross equals within-FoldJAX; both sides have
  sampling spread on 3og2 (TM 0.984-0.999) and the cross distribution sits
  inside it.
- **OpenFold3 at 3k (L3000_6ztx, homotetramer of 753 residues): cross TM
  0.852 / 24 Å with both sides internally identical (within 0.998-0.999),
  and it is not chain assignment: every FoldJAX chain against every upstream
  chain, superposed on its own, is 20.7 Å / TM 0.843. The monomer fold
  differs between the two implementations at this size while it agrees at
  1k and 2k. Protenix on the same case shows a uniform 2.2 Å / TM 0.991 per
  chain (cross 2.4 Å against within 0.3 Å). Both are open: a tape-pinned
  replay at 3k (native capture + port replay on the same tape) is queued to
  separate arithmetic from the MSA-row subsample draw, which at this depth
  is a random choice on both sides.**
- OpenDDE at 1k: cross equals within-FoldJAX, but within-upstream is tighter
  (0.999-1.000 against 0.984-0.999). FoldJAX OpenDDE draws a wider sample
  distribution on this case than upstream. The tape-pinned panel shows the
  arithmetic matches to 0.002-0.016 Å, so this is the ordinary-RNG sampler
  path, not the trunk; open item, being measured on the eight panel cases.

### 3k tape-pinned pairs (2026-09-10 evening)

The 3012-token tape-free rows differed uniformly (Protenix 2.38 Å cross vs
0.30/0.51 within; OpenFold3 24 Å on one chain). Replaying the port on
native's own tape at this size needed harness work (the fused OpenFold3 replay
and the materialised-MSA Protenix replay both exceed the card at 3k; fixed by
the streamed graph with the CLI's host feature chain, an MSA index tape, and a
preallocated allocator pool). Protenix, port vs native-A, per-chain CA RMSD Å
by sample:

| chain | s1 | s2 | s3 | s4 | s5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.08 | 0.40 | 0.08 | 0.05 | 0.10 |
| B | 0.11 | 0.41 | 0.10 | 0.07 | 0.10 |
| C | 0.08 | 0.39 | 0.11 | 0.06 | 0.10 |
| D | 0.09 | 0.40 | 0.09 | 0.07 | 0.11 |

With the draw shared, the 2.4 Å tape-free offset disappears: four samples
sit at 0.05-0.11 Å (the 1k-2k pass/deferred level) and sample 2 at 0.40 Å on
every chain. Native's own floor at 3k (native-A vs native-B, same tape, same
seed, one process apart; job 1007): the two processes label the four
identical chains in a different order on every pair, and after the
permutation-aware alignment (`bench.structures`) they differ by 0.64-3.32 Å
(median 2.05, TM 0.991-0.999) while each set agrees with itself to 0.30 /
0.51 Å. The port's 0.05-0.40 Å against native-A is inside that floor on
every sample: at 3k the port is closer to native-A than native is to itself
(three-way reading, same class as 7ST3). Closed.

OpenFold3, port (`cueq`, streamed graph) vs native-cueq, per-chain CA RMSD Å
by sample:

| chain | s1 | s2 | s3 | s4 | s5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.032 | 0.038 | 0.031 | 0.41 | 0.030 |
| B | 0.031 | 0.037 | 0.032 | 0.41 | 0.031 |
| C | 0.031 | 0.037 | 0.033 | 0.41 | 0.031 |
| D | 0.032 | 0.037 | 0.032 | 0.41 | 0.031 |

pTM/ipTM agree to 0.0009. The tape-free 24 Å monomer-fold difference was
the draw as well (MSA subsample, reference-conformer RNG, noise): four
samples in the pass band and one sample at 0.41 Å on every chain, the same
shape as Protenix's.

Native's own scatter at 3k (OpenFold3 native-cueq A vs B, same seed, per-chain
CA RMSD Å by sample): s1 0.026-0.027, s2 0.098-0.103, s3 0.027-0.029,
s4 0.032-0.034, s5 0.029-0.030; pLDDT max 1.3 points, pTM/ipTM 1e-4. So the
port sits at native's own scatter on samples 1/2/3/5 (0.030-0.038 vs
0.026-0.103) and 12× above it on sample 4 (0.41 vs 0.033; native's widest
sample is 0.10). A second port draw with the autotune cache unfrozen (job
1009) leaves sample 4 at 0.41 Å on every chain and moves sample 2 from 0.038
to 0.29 Å: sample 4 is a stable route difference at 3k, not a near-tie,
while sample 2 is the port's kernel-choice-sensitive sample (its spread 0.29
against native's 0.10 on the same sample, the pattern seen on ESMFold2 7ST3).
Both port draws also put one pLDDT value 20 points from native where the two
native processes differ by 1.3 points.

Localised on CPU (per-residue, both port draws, all four chains): sample 4's
0.41 Å is the N-terminal coil, residues 15-27 (HIS17 3.7 Å, LEU16 3.0, SER20
2.8), carrying 97% of the squared deviation on every chain; the other 711
residues sit at 0.033 Å, native's own floor, and the fit is neither a rigid
shift nor an inter-chain component. That loop has two wells about 2 Å apart
in native's own five samples ({1,2,5} and {3,4}); the port's sample 4 sits in
the {1,2,5} well on every chain (0.36-0.43 Å from native sample 2's loop,
1.86 Å from native sample 4's), and both port draws land there together
(0.026 Å apart). The 20-point pLDDT difference is PRO15-HIS17 on sample 4
only, where the port's confidence matches native's samples 1/2/5 to about a
point: the head scores the conformer it was handed. Native ranks sample 4
last on both processes and so does the port. Reading: a basin choice on a
bistable loop whose two wells native's own run populates, the class already
measured on Boltz-2 1AAY and ESMFold2 7ST3 — at-floor. OpenFold3 3k closed.

## Provenance of the upstream arm on master

`bench.run_upstream` refuses untracked files, unreviewed tracked changes and
symlinked runtime artifacts in the upstream checkouts. What was done on master
to satisfy it, all recorded beside the snapshot:

- Roots: `scale-timing-20260910/upstream-root/{boltz,protenix,OpenDDE,openfold3-v050}`
  link to the git checkouts `boltz` b1ebfc4, `protenix` 4c355be, `OpenDDE`
  ddfa1df, `openfold3-v050` c477165.
- Tracked changes, reviewed and pinned with `--expected-upstream-diff-sha256`:
  Boltz `baa88999…` (`upstream-root/boltz-b1ebfc4.diff`: CUDA 13 library
  preload for cuEquivariance, a dropout==0 fast path that preserves the RNG
  offset, an env-gated `BOLTZ_FAST_FC12` path left off, f32 triangle bias
  under `use_kernels`, stacked projection weights); Protenix `c69fdc19…`
  (`protenix-4c355be.diff`: sm_120 gencode for the layer-norm extension). A
  pristine-checkout rerun of the Boltz column answers whether those
  performance patches matter: a `git worktree` of `boltz` at b1ebfc4 with
  zero tracked diff (the CUDA 13 preload moved into the venv as a
  `site-packages/_boltz_cu13_preload.pth`, sha256 `70a01d1e…`/`57a77e1d…`),
  root `upstream-root-pristine/boltz`, label `pristine`: 1k 135 s / 15.7 GiB
  against the reviewed arm's 132 s / 15.8 GiB (job 989); 2k follows (990).
  The reviewed patches did not move the 1k timing.
- Untracked session directories excluded through each checkout's
  `.git/info/exclude`; `boltz/.venv-cueq011` moved to `boltz-venv-cueq011`
  (a symlinked python inside an ignored directory is refused).
- Assets hard-linked, not symlinked: `openfold3_weights/checkpoints/of3-ob-2025-06-30-174k.pt`,
  `~/.boltz/{boltz2_aff.ckpt,mols.tar}`; `~/protenix` links to the checkout.
- OpenFold3 4k/5k upstream rows needed 96 GB of host memory (48 GB was
  killed by Slurm).

## Mixed-entity set (real complexes)

Five RCSB entries chosen for protein + nucleic acid + ligand/ion at the same
bands, `foldjax-bench/mixed-scale-20260910/{jobs,msa,sequences.json}`:

| case | PDB | tokens | composition |
| --- | --- | ---: | --- |
| mixed_1k_4xww | 4XWW | 1138 | RNase J x2 (559) + RNA 7 x2 + Zn x4, Mn x2 |
| mixed_2k_7y7q | 7Y7Q | 2097 | QDE-1 x2 (1026) + RNA 14 x2, 7 + Ca x4, Mg x3, GTP x3 |
| mixed_3k_5npk | 5NPK | 2861 | DNA gyrase x4 (692) + DNA 20 x4 + Mn x5, 94H x6, 9JN x2 |
| mixed_4k_6kqf | 6KQF | 3874 | T. thermophilus RNAP ITC (6 protein entities) + DNA 21/27 + RNA 5 + Mg x4, Zn x2 |
| mixed_5k_5xog | 5XOG | 4787 | Pol II elongation complex (14 protein entities) + RNA 17 + DNA 39/30 + Zn x9, Mg, APC |

Crystallisation additives (glycerol, PEG, sulfate, chloride, sodium, MES,
acetate) are dropped; every ion and real ligand is kept. Protein MSAs come
from the ColabFold API (unpaired, `mode=env`), 22 chains; the fetch took most
of the afternoon because the API serves one large query in 15-35 minutes and
truncates large downloads (the curl fetcher in the scratchpad retries the
download). The 30 FoldJAX rows (5 cases x 6 ports) were submitted at 15:30
and are running; their upstream rows follow once each FoldJAX row has
materialised its native input. Results land in the same snapshot and will be
appended here.

### Mixed rows so far (FoldJAX "s / GiB / max ipTM" vs upstream)

| case | tokens | OpenFold3 | Boltz-2 | Protenix | OpenDDE | ESMFold2 | AlphaFold3 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| mixed_1k_4xww | 1138 | 132 / 9.2 / 0.90 vs 177 / 17.7 / 0.91 | 111 / 10.4 / 0.95 vs 174 / 18.1 / 0.95 | 80 / 11.8 / 0.94 vs 108 / 15.8 / 0.94 | 302 / 52.6 / 0.94 vs 318 / 73.2 / 0.93 | 186 / 14.7 / 0.89 | 81 / 5.6 / 0.94 |
| mixed_2k_7y7q | 2097 | 459 / 26.4 / 0.77 vs 698 / 43.9 / 0.76 | 348 / 18.4 / 0.87 vs 492 / 50.4 / 0.88 | 248 / 21.9 / 0.76 vs 280 / 44.6 / 0.75 | OOM (fp32 pair wall, as on the protein set) | 487 / 51.2 / 0.63 | 260 / 14.6 / 0.74 |
| mixed_3k_5npk | 2861 | 1013 / 50.1 / 0.62 | 944 / 41.1 / 0.63 | 699 / 42.1 / 0.59 | OOM | OOM (trunk pair arena, as at 3k protein) | 521 / 29.7 / 0.46 |
| mixed_4k_6kqf | 3874 | OOM (69) | 2111 / 60.9 / 0.86 | 1509 / 66.5 / 0.90 | OOM | OOM | 814 / 43.1 / 0.89 |
| mixed_5k_5xog | 4787 | OOM (103) | OOM (83) | OOM (91) | OOM | OOM (44) | 1544 / 61.4 / 0.91 |

ESMFold2 reports ipTM from its own head; AlphaFold3's column has no upstream.
OpenDDE's 1k pair is the one mixed row where FoldJAX is not faster (302 vs
318 s) while 28% lighter (52.6 vs 73.2 GiB).

Structure agreement on the mixed pairs so far (`bench.structures`,
permutation-aware, CA TM median (min-max), cross RMSD Å):

| model | case | cross TM | within FoldJAX TM | within upstream TM | cross RMSD | within FoldJAX RMSD | within upstream RMSD | chain perm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenFold3 | mixed_1k_4xww | 0.994 (0.978-0.998) | 0.990 (0.978-0.995) | 0.997 (0.994-0.999) | 0.93 (0.45-6.54) | 1.52 (0.85-6.83) | 0.68 (0.32-1.03) | cross 14/25; fj 7/10; up 6/10 |
| Boltz-2 | mixed_1k_4xww | 0.981 (0.976-0.997) | 0.980 (0.977-0.995) | 0.979 (0.977-0.997) | 5.06 (0.61-7.86) | 5.41 (0.82-6.97) | 5.41 (0.66-7.41) | cross 10/25; fj 4/10; up 4/10 |
| Protenix | mixed_1k_4xww | 0.996 (0.989-0.999) | 0.994 (0.990-0.998) | 0.995 (0.993-0.999) | 0.81 (0.40-1.64) | 0.99 (0.46-1.40) | 0.92 (0.39-1.19) | cross 12/25; fj 4/10; up 6/10 |
| OpenDDE | mixed_1k_4xww | - | 0.994 (0.986-0.999) | - | 0.71 (0.31-2.15) | 0.96 (0.43-2.20) | 0.73 (0.53-1.43) | fj 5/10 |
| Boltz-2 | mixed_2k_7y7q | 0.981 (0.969-0.996) | 0.994 (0.971-0.997) | 0.981 (0.967-0.997) | 2.67 (0.84-5.17) | 1.25 (0.85-3.31) | 2.71 (0.81-5.06) | cross 15/25; fj 7/10; up 7/10 |
| OpenFold3 | mixed_2k_7y7q | 0.988 (0.968-0.994) | 0.990 (0.987-0.995) | 0.979 (0.969-0.994) | 2.06 (1.15-4.63) | 1.68 (1.17-2.51) | 3.07 (1.31-4.46) | cross 17/25; fj 6/10; up 9/10 |
| Protenix | mixed_2k_7y7q | 0.969 (0.954-0.978) | 0.965 (0.962-0.988) | 0.972 (0.967-0.980) | 9.01 (3.67-11.72) | 10.14 (2.05-11.34) | 6.32 (3.32-10.89) | cross 14/25; fj 8/10; up 4/10 |
| OpenFold3 | mixed_3k_5npk | 0.993 (0.610-0.998) | 0.997 (0.991-0.999) | 0.616 (0.608-0.998) | 1.47 (0.62-20.31) | 0.79 (0.51-1.83) | 19.52 (0.70-20.55) | cross 17/25; fj 7/10; up 5/10 |
| Protenix | mixed_3k_5npk | 0.990 (0.961-0.999) | 0.995 (0.992-0.998) | 0.991 (0.967-0.999) | 1.70 (0.54-3.20) | 1.22 (0.69-1.58) | 1.48 (0.56-2.95) | cross 25/25; fj 9/10; up 5/10 |
| Protenix | mixed_4k_6kqf | 0.957 (0.946-0.969) | 0.968 (0.959-0.981) | 0.966 (0.957-0.972) | 10.34 (4.84-16.81) | 9.54 (3.81-14.81) | 10.85 (7.24-13.20) | cross 0/25; fj 0/10; up 0/10 |

Cross RMSD is at or below the within-set RMSD of both implementations on all
four finished 1k pairs, at the within level on the three 2k pairs and on the
three 3k/4k pairs (5NPK: upstream OpenFold3's own five samples split into two
assemblies, within TM median 0.616, so its cross column is read against that;
6KQF is a loose 3.9k-token complex on both sides, within 10 Å) (7Y7Q's
QDE-1 dimer with its 14-mer RNAs is a loose assembly on both sides: Protenix
within 10.1 Å FoldJAX / upstream in the same band, cross 9.0 Å). Boltz-2's 5 Å is the model's own spread on this entry
(the 7-mer RNA and the dimer arrangement move between samples at the same
rate on both sides, within-set 5.41 Å each); OpenFold3's FoldJAX set has one
sample 6.8 Å from the others while upstream's five agree to 1 Å, a sampler
draw at n=5 rather than a systematic offset (cross median 0.93 Å).
4XWW's two RNase J chains are identical, so the permutation-aware pairing
swaps them in 10-14 of 25 cross pairs.
The Protenix 4k protein row is the first pair where FoldJAX is not faster
(2206 vs 2193 s) while still 15% lighter (73.5 vs 86.4 GiB).

## X9 (2026-09-11): bf16 diffusion compute and fused pair-bias attention

Two opt-in levers, measured against the released arm on the same input file,
seed and schedule. Neither changes a default. Rows are warm-after-prefill on
one RTX PRO 6000 Blackwell (sm120, tokamax runs Triton there; Mosaic GPU is
sm90/sm100 only).

### Boltz-2, 1,003 tokens (3OG2)

The diffusion score model is the one module upstream runs outside autocast, so
the port ships it in float32 whatever `compute_dtype` says. The two new knobs
open the other three cells.

| cell | wall s | vs released | peak MiB | same-index RMSD vs released |
| --- | ---: | ---: | ---: | --- |
| released (fp32 score model, XLA, `matmul_precision=highest`) | 90.98 | – | 12612 | – |
| `attention_backend=tokamax` (fp32 operands) | 158.89 | +75% | 12588 | at floor |
| `attention_backend=tokamax matmul_precision=high` | 82.71 | −9.1% | 12588 | median 0.043 / max 0.063 |
| `matmul_precision=high` alone (XLA) | 85.33 | −6.2% | 12588 | – |
| `diffusion_compute_dtype=bfloat16` (XLA) | 83.36 | −8.4% | 12588 | median 0.094 / max 0.355 |
| `diffusion_compute_dtype=bfloat16` + `diffusion_attention_backend=tokamax` | **77.61** | **−14.7%** | 12604 | median 0.104 / max 0.373 |

The process floor on this case is 0.05 / 0.31 Å and the within-set sample
spread is 1.0-2.9 Å, so every arm's coordinates are at floor. Peak does not
move: Boltz-2's peak is a trunk arena, not the denoiser.

### Boltz-2, the fused GLU: a 1k win, not a general one

`glu_backend=tokamax` changes no dtype and no precision; it only stops the
transition materialising its pre-gate activation. The saving is real and it
does not survive the arena flip.

| tokens | wall | peak | same-index RMSD vs released |
| ---: | ---: | ---: | --- |
| 1,003 | 90.98 → 90.00 s (−1.1%) | 12612 → 9216 MiB (**−26.9%**) | median 0.012 / max 0.238 |
| 2,096 | 317.91 → 313.94 s (−1.2%) | 21808 → 21778 MiB (−0.14%) | median 0.007 / max 0.145 |

At 1,003 tokens the transition intermediate is the peak's largest tenant; at
2,096 the pair arena is, and the same 3.4 GiB saving is invisible against it.
The 2k peak is byte-identical to the fused-attention arm's, which says both
knobs remove the same small thing and neither reaches the real tenant. The
knob is free at both sizes and worth switching on for small inputs, but it is
not the lever the 1k number alone suggested.

The two rows above are superseded by the census below: both were taken before
the pair-residual default landed, and the 2k pair's arm labels predate the
default flip to `tokamax`. Re-measured on `8da1d69` the same arm is 257.70 s /
18511.1 MiB rather than 313.94 / 21778 -- that 56 s and 3.3 GiB is the pair
residual, not the GLU.

### The fused GLU, closed on all four ports (2026-09-14, `8da1d69`)

Each pair below ran as one wave of concurrently submitted jobs on its own card,
so the two arms saw the same chassis load. Sampling the cards mid-run, the two
of a pair stayed within 3.2% of each other on SM clock and swapped which was
faster between samples, so no arm sat on a systematically throttled card. Four 600 W cards running together are slower
than a single-card row, so read these against each other and not against the
scale table above.

| port | default | wall | peak | verdict |
| --- | --- | --- | --- | --- |
| Boltz-2 2,096 | `tokamax` | 262.32 → **257.70 s** (−1.8%) | 18511.2 → 18511.1 MiB (−0.1) | keep fused |
| Boltz-2 3,012 | `tokamax` | 702.62 → **700.62 s** (−0.28%) | 29431.3 → 29415.5 MiB (−15.8) | keep fused, win is gone |
| OpenFold3 2,096 | `xla` | 228.58 → 246.15 s (**+7.7%**) | 13882.4 → 14005.4 MiB (+123.0) | keep unfused |
| Protenix 2,096 | `xla` | 203.58 → **192.67 s** (−5.4%) | 21223.1 → **24765.3 MiB (+3542.2)** | keep unfused |
| OpenDDE | – | – | – | no SwiGLU to fuse |

No default moves. Boltz-2's fused default earns ≤1.8% and never loses;
OpenFold3's unfused default is confirmed against a 7.7% regression; Protenix
buys 5.4% for 3.5 GiB on a port that already OOMs at 4,888 tokens (89.4 GiB
asked of a 97.9 GiB card) and sits at 65-71 GiB at 4,100, so the trade pulls
the ceiling in for a wall saving smaller than the card-to-card spread at that
size. OpenDDE has no gated transition at all: `models/structural_tokens.py:151`
is a plain `linear(silu(hidden), ...)` and the port does not import
`models/_glu.py`.

Protenix's 3.5 GiB is not an artefact of the arm it was first seen in. The
earlier reading used `--amp-policy bf16` -- an arm that loses a chain (TM 0.75,
16.2 Å, below) -- and measured +3525 MiB; the default `auto` arm measures
+3542 MiB. The fused kernel's workspace is the same either way.

#### Why the three ports disagree: call-site count and call shape

tokamax ships tuned autotuning caches for b200 / h100 / tpu7x / tpu_v5 /
tpu_v5_lite (0.0.14 adds a100 and tpu_v6_lite; 0.1.0 is yanked with no cache
data). None covers sm_120, so on this card every `gated_linear_unit` call misses
the cache and `default_config` returns `heuristics_config` rather than
autotuning -- `_src/config.py:65` sets the fallback to `heuristics`,
`_src/ops/op.py:501-514` resolves it. One warning per call site per trace makes
those misses a free census. Boltz-2's warmup and measured processes agree at
218, so the count is the program's and not a phase artefact.

| port | GLU call sites | dominant shape (count) | float32 sites |
| --- | ---: | --- | ---: |
| Boltz-2 | 218 | `bf16[1,62,2096,64] x [64,2,256]` (132) | 2 |
| OpenFold3 | 20 | `bf16[1,117,2096,128]` (6), `bf16[1,2096,2096,128]` (2) | 6 |
| Protenix | 15 | `bf16[2096,2096,128] x [128,2,512]` (4) | 4 |

Boltz-2 hands the kernel 62-row MSA chunks and 64-row pair chunks 218 times;
Protenix and OpenFold3 hand it the whole 2096x2096 pair tensor a handful of
times. All three scan their block stacks, so the difference is not scan versus
unroll -- it is whether the transition is chunked before the call. That single
difference orders the two ports that win something: 218 small calls give
Boltz-2 a small uniform win, and one huge call gives Protenix a real wall win
plus a workspace it cannot afford. OpenFold3 sits between them in shape and
loses on both counts, which the call shapes alone do not account for -- see
below.

#### The kernel is fast; the ceiling is what settles it

Measured directly at L=2096 and L=3012 on the shapes above, both sides at
`precision=default` -- which for bfloat16 operands changes nothing, since the
Triton kernel accumulates in float32 either way:

| shape | XLA | tokamax (heuristic config) | XLA / tokamax |
| --- | ---: | ---: | ---: |
| `bf16[1,62,2096,64]` | 0.2340 ms | 0.0863 | 2.71 |
| `bf16[1,64,2096,128]` | 0.5460 | 0.1861 | 2.93 |
| `bf16[1,128,2096,128]` | 1.0824 | 0.3140 | 3.45 |
| `bf16[1,43,3012,64]` | 0.2349 | 0.0892 | 2.63 |
| `bf16[1,64,3012,128]` | 0.7746 | 0.2449 | 3.16 |
| `bf16[1,128,3012,128]` | 1.5237 | 0.4505 | 3.38 |

Untuned, the fused kernel is already 2.6-3.5x faster than XLA on bfloat16. That
is why no per-card autotuning cache was built: the kernel is not what limits the
win, so tuning it cannot move a default. At 2,096 on Boltz-2 the 218 sites cost
83.3 ms per program execution under XLA and 28.5 ms under the heuristic kernel;
the observed end-to-end saving of 4.62 s puts the program at ~84 executions, so
a kernel with zero cost would save 7.0 s -- 2.7% of 257.7 s, only 0.9 pp beyond
what the heuristic already delivers. The kernel is a ~2% slice of the runtime on
the port that calls it most.

On float32 the sign is set by the matmul precision, not by the dtype. Measured
on the five float32 shapes the three trunks actually call, `xla / tokamax`:

| shape (float32) | `default` | `high` | `highest` |
| --- | ---: | ---: | ---: |
| `[1,2096,384] x [384,2,1536]` (Boltz-2 single) | 0.807 | 1.028 | 0.511 |
| `[1,15736,128] x [128,2,256]` (OpenFold3 atom) | 0.781 | **1.289** | 0.514 |
| `[5,15736,128] x [128,2,256]` (OpenFold3 atom, 5 samples) | 1.792 | **1.747** | 0.784 |
| `[5,2096,768] x [768,2,1536]` (Protenix diffusion token) | 1.149 | **1.121** | 0.557 |
| `[5,2096,384] x [384,2,768]` (Protenix diffusion single) | 0.920 | 0.923 | 0.558 |

Production traces all key `HIGH`, and at `high` the fused kernel wins on three
of the five and is never worse than 0.92x. At `highest` it loses everywhere by
1.3-2.0x, which is the three-pass float32 emulation already recorded for the
fused attention on this port. So `models/_glu.py`'s claim that the fused GLU is
worth offering on a float32 port stands -- but only away from
`matmul_precision=highest`, and an earlier reading here that called float32 a
loss for the kernel was measuring at `default`, which no port runs.

What the float32 table does not explain is OpenFold3's 17.6 s. Six of its twenty
sites are float32 and at `high` those favour the kernel; twenty sites of any
shape cannot cost 17.6 s of execution at these per-call times. Total job elapsed
also moves the other way (601 s unfused vs 600 s fused, against 228.58 vs 246.15
in the measured window), so the cost shifts between the prefill and measured
processes rather than appearing as new total work. The measured window is the
metric every other row in this document uses and the verdict rests on it, but
the mechanism behind OpenFold3's share of it is not established. It is a fixed
per-process cost of some kind, not a throughput regression in the kernel.

Neither port's confidence moves. OpenFold3's five samples read pLDDT
94.275-94.316 unfused and 94.296-94.304 fused, the fused spread sitting inside
the unfused one; Protenix reads pTM 0.96380-0.96391 unfused and 0.96385-0.96395
fused. `models/_glu.py` asks for a backend change to be treated as a numerics
change because the two paths apply the activation at different widths; at these
shapes the change does not reach the confidence head.

### The denoiser attention, adopted on Boltz-2 (2026-09-14, `8da1d69`)

Every row below ran as one wave of concurrently submitted single-card jobs, so
the arms of a comparison saw the same chassis load. The control spells
`attention_backend=xla` rather than omitting it, so the label certifies the
arm; it reproduces the released wall to 0.05% at 2,096 (257.83 against 257.70
from the GLU wave) and 0.09% at 3,012 (699.99 against 700.62), which is the
baseline replicating across waves and cards.

| tokens | `attention_backend=xla` | `diffusion_attention_backend=tokamax` | wall | peak |
| ---: | ---: | ---: | ---: | ---: |
| 1,003 | 74.78 s / 8453.4 MiB | 71.42 / 8453.4 | **−4.49%** | byte-identical |
| 2,096 | 257.83 / 18511.1 | 237.17 / 18511.2 | **−8.01%** | +0.1 MiB |
| 3,012 | 699.99 / 29415.5 | 655.08 / 29415.5 | **−6.42%** | byte-identical |

**The win does not shrink with size, and that is what separates it from the
fused GLU above.** The GLU went −1.8% at 2,096 to −0.28% at 3,012; this holds
−4.5 / −8.0 / −6.4 across the series. The GLU is ~2% of the runtime by the
call-site arithmetic; the denoiser attention is not.

#### The narrow option takes the whole win, on both ports that have it

| port, 2,096 | arm | wall | coordinates vs the control | process floor |
| --- | --- | ---: | ---: | ---: |
| Boltz-2 | `diffusion_attention_backend=tokamax` | −8.01% | 0.0233 Å | **1.16x** |
| Boltz-2 | `attention_backend=tokamax` | −7.99% | 0.0459 Å | 2.3x |
| Boltz-2 | `diffusion_compute_dtype=bfloat16` | −2.89% | 0.0894 Å | 4.4x |
| Boltz-2 | `diffusion_attention_backend=triton` + bf16 | **−12.62%** | 0.1899 Å | **9.4x** |
| Protenix | `diffusion_attention_backend=tokamax` | −6.22% | 0.0158 Å | 1.93x |
| Protenix | + `trunk_single_attention_backend=tokamax` | −5.40% | 0.0093 Å | 1.13x |

Boltz-2 adopts the narrow option. **Protenix measures the same way and is not
adopted yet**, for a reason that is about the port's plumbing rather than the
numbers: `tests/models/protenix/test_cache_profile.py::test_released_default_aliases_reuse_one_real_bounded_native_owner`
asserts that spelling every released default explicitly and omitting them all
trace **one** program, and with the flip it traces two. The static arguments of
the two traces are identical, every printed one including
`diffusion_attention_backend: 'tokamax'`, so the cause is not a disagreement
among the four authorities that carry this default -- `models/predict.py`,
`models/model.py`, `backends/protenix.py` and the shared CLI flag, all four
aligned. An extra trace of the whole graph is a real per-process cost on a
shipped default, so the flip waits for the diagnosis rather than for the test
to be adjusted around it.

The lead is `models/_jit_pool.py:111-115`, which records that in JAX 0.11.1
**an omitted default, an explicit keyword default and a positional default are
three distinct jit cache keys**, and that the pool preserves that on purpose
rather than normalising it away. So the candidate is a call-style difference
rather than a value difference, which is consistent with the two traces having
identical static arguments. What has not been shown is where the two requests
part: the argv renderer emits `--diffusion-attention-backend` whenever the
option is supplied and the CLI passes the parsed value explicitly either way,
so both routes look like the same call style from the outside. Whoever picks
this up should print the pool's cache size and the two `_identity` tuples side
by side rather than the graph kwargs, which is what this attempt read and what
sent it past the cause.

The accuracy reading stands either way. Protenix's 1.93x is one sample of five
-- the other four are 0.0069-0.0092 Å, at its 0.0082 Å floor -- and 0.0158 Å is
below what the deposited instrument resolves, which reads the three arms
identically. The wide arm sits closer to the floor but is slower on this port,
so there is nothing to buy with the extra movement.

One thing the flip attempt established that outlives it: `models/_predict_flags.py`
is shared by Protenix and OpenDDE, and its `--diffusion-attention-backend`
default is shared with the `choices` list. Moving the default there would have
handed `tokamax` to OpenDDE, where it is not in `choices`, so the parser would
have refused its own default. The docstring had already written the rule down
for `extra_backends` -- "a kernel is offered on the port whose numbers were
measured" -- and the default needs the same treatment when this lands.

Widening the option to the trunk and atom attention buys nothing on either
port -- 237.23 s against 237.17 on Boltz-2, and on Protenix the wide arm is
*slower* than the narrow one (192.19 against 190.51, against a 0.21% wall
floor) -- while moving coordinates further. Two ports agreeing makes that a
property of where the fused kernel pays rather than one port's accident.

The floors are same-configuration replicates: on Boltz-2 the released arm from
two different waves on two different cards (max 0.0201 Å over five samples),
on Protenix a second run of the control (max 0.0082 Å). Peak floors are +0.1
MiB and +28.8 MiB respectively, so neither port's peak moves.

Against the deposited entry, permutation-matched on `label_seq_id`, Boltz-2's
adopted arm is unchanged sample for sample (0.44/0.43/0.42/0.47/0.43 Å) and so
are both of Protenix's (0.46/0.52/0.44/0.49/0.47 Å for all three arms).
Boltz-2's pLDDT, pTM and ipTM distributions sit inside the control's.

**`label_seq_id`, not `auth_seq_id`.** The first scoring pass read 6.02 Å for
every arm of a 1.3 Å crystal structure, identically, which is the tell: the
deposited entry numbers 3..607 with 79 unmodelled gaps for 526 residues and
the prediction numbers 1..524 contiguously, so intersecting author numbers
pairs residues through a shifting frame and is self-consistently wrong.
`label_seq_id` is the SEQRES index on both sides and the sequences then match
from position 1. Chain assignment came out `(A, C, B, D)` in every arm, so the
permutation-aware step was load-bearing too.

#### `triton` + bfloat16 is the largest unclaimed win here, and what it needs

It is the fastest arm measured on this port at this size, −12.62% with a
byte-identical peak, and its deposited RMSD is 0.01 Å *better* than the
control on every sample. It is not adopted, for three reasons that stand
together.

Its coordinates move **9.4x the process floor**, which is not accuracy
equivalence -- it is a different arm that happens to land nearby on one
target, and one target's 0.01 Å is inside that target's own variation. The
`triton` spelling requires bf16 q/k/v/bias, so it carries
`diffusion_compute_dtype=bfloat16` with it -- already rejected on its own at
4.4x the floor for departing upstream's six explicit `.float()` calls
(`diffusionv2.py:142-170`), one of which is the pair bias at `:160`, and a
rounded pair bias is what flew a chain 16 Å on this port before. And it is one
case: 9.4x the floor needs a multi-case accuracy panel to claim, not a single
homotetramer.

Recorded here so the number is not rediscovered as new: the wall is available,
the instrument to accept it is not yet run.

#### Upstream AlphaFold 3 already ships the fused kernel, and its denoiser is float32

`'triton'` is not a separate library. AlphaFold 3's
`GlobalConfig.flash_attention_implementation: tokamax.DotProductAttentionImplementation
= 'triton'` (`_upstream/.../model_config.py:41-43`) is the `implementation`
argument to `tokamax.dot_product_attention`, and Boltz-2's `tokamax`, `flash`
and `triton` spellings all reached the same call. So the question is never
"triton or tokamax" but whether tokamax attention is wired at all: it is the
default on AlphaFold 3, the default on Boltz-2 as of this change, available and
unused on Protenix, and not wired on OpenDDE, whose `attention_kernel` accepts
only `xla`. OpenFold3 and ESMFold2 have no attention-backend option.

AlphaFold 3 running `bfloat16: 'all'` does **not** make its denoiser bf16.
`diffusion_head.py:238` opens the bf16 context and `:267-285` casts `act`, both
trunk conditionings and the sequence mask straight back to float32 inside it.
The boundary is drawn by tensor origin, not by stage. So the adopted
`tokamax` arm -- fused attention, float32 denoiser activations -- is the closer
match to what upstream AlphaFold 3 runs, and `triton` + bf16 is not "the AF3
configuration" as an earlier reading here had it.

### Boltz-2, 2,096 tokens (5DEI)

| cell | wall s | vs released | peak MiB | same-index RMSD vs released |
| --- | ---: | ---: | ---: | --- |
| released | 317.91 | – | 21808 | – |
| `attention_backend=tokamax` (fp32 operands, `highest`) | 586.15 | +84% | 21778 | at floor |
| `diffusion_compute_dtype=bfloat16` + `diffusion_attention_backend=tokamax` | **270.03** | **−15.1%** | 21778 | median 0.065 / max 0.183 |

The same 15% at both sizes, and the per-chain deposited RMSD of the fast arm
(0.34-0.39 A) is the released arm's own (0.34-0.40 A).

The earlier verdict that tokamax regresses on this port was the precision pin,
not the kernel. `tokamax/_src/precision.py` maps a float32 result type to
`F32_F32_F32` under this port's pinned `matmul_precision=highest`, which is
three-pass emulation; at `high` the same call is `TF32_TF32_F32` and beats XLA
by 3.1%. The bfloat16 branch never reads the pin at all, which is why the
bf16 cell is the one that pays: the kernel log confirms both diffusion sites
receiving bf16 q/k/v **and** a bf16 bias with `logits_dtype float32`.

### Protenix, 2,096 tokens (5DEI homotetramer)

`--amp-policy bf16` forces the policy upstream ships only above 3,840 tokens.
It was 10% faster and 8% lighter, and it lost a chain.

| arm | wall s | peak MiB | pLDDT | chain A vs deposited |
| --- | ---: | ---: | ---: | --- |
| released (`auto` → fp32 diffusion at this size) | 210.32 | 23440 | 95.16 | TM 0.998, 0.40 Å |
| `--amp-policy bf16`, as first written | 189.04 | 21630 | 94.07 | **TM 0.75, 16.2 Å** |
| same, second process | 201.91 | 21623 | 94.06 | TM 0.75, 16.2 Å |
| upstream with `PROTENIX_FORCE_AMP=all` | 240.76 | 28815 | 95.12 | TM 0.998, 0.39 Å |
| `--amp-policy bf16` with the pair bias in fp32 | 193.04 | 21639 | 94.99 | TM 0.998, 0.40 Å |

Chains B-D stayed at 0.4-1.2 Å throughout; only chain A moved, in all five
samples, in three separate processes. Upstream's own forced-bf16 diffusion at
the same size keeps all four chains, so this was the port's realisation and
not a property of bf16.

A seven-arm bisection over the parameter tree found one projection. Keeping
the atom encoder and decoder, the conditioner, the AdaLN projections or the
conditioned transition in fp32 changed nothing; keeping the 24-block token
transformer in fp32 fixed it, and inside that stack the whole effect is
`linear_z`, the per-head pair-bias projection. Decomposing that projection's
three roundings separates them cleanly:

| operands | bias result | chain A |
| --- | --- | --- |
| bf16 / bf16 | bf16 | 16.2 Å |
| fp32 activation / bf16 weight | bf16 | 16.2 Å |
| bf16 activation / fp32 weight | bf16 | 16.2 Å |
| bf16 / bf16 | **fp32** | **0.40 Å** |
| fp32 / fp32 | fp32 | 0.40 Å |

Narrowing the operands is free; rounding the result is fatal. The pair bias is
not an ordinary activation: it is added to attention logits and exponentiated
over every token, so eight mantissa bits there is a different class of error
than eight bits on a tensor that feeds another GEMM. AlphaFold 3 reaches the
same shape from the other direction, casting q, k and the bias to float32
before its diffusion attention, and tokamax's kernel adds the bias into a
float32 accumulator whatever the operands are.

The port now delivers every denoiser pair bias in float32 while keeping the
bf16 GEMM. That keeps 82% of the policy's wall-time gain and all of its
memory gain.

### Protenix, 2,096 tokens: the 2x2 after the fix

Wall seconds and peak MiB, with the per-chain deposited RMSD of every arm in
the 0.38-0.45 A band the released arm itself occupies.

| diffusion compute | XLA attention | tokamax attention |
| --- | --- | --- |
| fp32 (released) | 210.32 s / 23440 MiB / pLDDT 95.16 | 199.25 s / 21216 MiB / pLDDT 95.16 |
| bf16, pair bias fp32 | 191.41 s / 21636 MiB / pLDDT 94.98 | 187.14 s / 21636 MiB / pLDDT 94.99 |

Each lever pays on its own and they compose: the fused kernel alone is 5.3%
faster and, by never building the score tensor, 9.5% lighter with an
identical pLDDT; the dtype alone is 8.9% faster and 7.7% lighter; together
11.0% faster. Same-index RMSD of the bf16 arm against the released arm is
0.051 A median against a 0.185 A within-set spread, which is the pass band.

The kernel log confirms which sites the fused backend reached. In the bf16
arm all three carry bfloat16 (`bf16[2096,16,24]` trunk single,
`bf16[2460,32,4,32]` atom window, `bf16[5,2096,16,48]` diffusion token) and
no float32 warning fires; in the fp32 arm the two diffusion sites arrive as
f32 and the one-time warning fires exactly once, as designed.

### Protenix, 3,012 tokens (6ZTX), timed

| arm | wall s | vs released | peak MiB | pLDDT | same-index vs released |
| --- | ---: | ---: | ---: | ---: | --- |
| released (`auto`, fp32 diffusion at this size) | 579.47 | – | 42219 | 94.269 | – |
| `--amp-policy bf16`, before the fix | 527.22 | −9.0% | 38791 | 94.218 | median 0.309 |
| `--amp-policy bf16`, after the fix | 525.83 | −9.3% | 38854 | 94.287 | median 0.268 / max 0.306 |
| the same plus `tokamax` at all three sites | 513.99 | −11.3% | 38816 | 94.291 | median 0.258 / max 0.298 |

At this size the fix is free: the fixed arm is 1.4 s faster than the unfixed
one, inside noise, while its coordinates sit at the released arm's own
sample spread (0.286 within-set against 0.300) and its pLDDT is marginally
higher. The cost profile across the three sizes -- 5.8 of 8.8 points at 1k,
1.2 of 10.1 at 2k, none at 3k -- is one fp32 tensor held against denoiser
work that grows quadratically.

Adding the fused kernel on top is worth another 2.0 points at this size
(11.3% in total) for coordinates that move no further, and the kernel log
again names bfloat16 at all three sites (`bf16[3012,16,24]`,
`bf16[3715,32,4,32]`, `bf16[5,3012,16,48]`).

### Protenix, 3,012 tokens (6ZTX), tape-pinned

| pair | same-index permutation-aware RMSD, five samples |
| --- | --- |
| bf16 diffusion vs fp32, both deterministic, same tape, **after the fix** | 0.088 / 0.156 / 0.107 / 0.204 / 0.121 |
| the same **before the fix** (job 1057, pre-202b109 source) | 0.148 / 0.279 / 0.194 / 0.411 / 0.142 |
| native-A vs native-B, same tape (upstream's own floor) | 2.451 / 0.918 / 0.760 / 2.111 / 3.269 |

At 3k the policy moves the coordinates an order of magnitude less than
upstream moves between two of its own processes, and the pair-bias fix
roughly halves what remains. The 2k chain loss was case-specific, which is
why it needed a homotetramer with a deposited structure to see: at 3k the
unfixed arm looked healthy on this measurement.

The pre-fix row is kept because it is the honest history of this table: the
first version of this section quoted it as the bf16 policy's validation, and
it was measured on source that lost a chain at 2,096 tokens.

### What to switch on, by size

Nothing here changes a default; this is what the measurements support if
someone asks for the fast path.

| port | 1k tokens | 2k tokens |
| --- | --- | --- |
| Protenix | `tokamax` alone (3.9% time, 7.0% peak, pLDDT unchanged). The bf16 policy keeps only 3% after the fix and is not worth the deviation. | both: 11% time, 8% peak, coordinates 0.05 Å from the released arm |
| Boltz-2 | both: 14.7% time, coordinates at the process floor | both: 15.1% time, same |

At 3,012 tokens Protenix repeats its 2k answer: both levers, 11.3% of wall
time and 8% of peak, coordinates inside the released arm's own sample
spread.

The fused kernel on its own, at the released dtype, is the one cell that is
uniform across the sweep, because it is the score tensor it does not build:

| tokens | wall | peak | pLDDT | same-index vs released |
| ---: | ---: | ---: | ---: | --- |
| 1,003 | 65.13 → 62.60 s (−3.9%) | 6864 → 6382 MiB (−7.0%) | 94.584 → 94.593 | median 0.021 Å |
| 2,096 | 210.32 → 199.25 s (−5.3%) | 23440 → 21216 MiB (−9.5%) | 95.16 → 95.16 | per-chain identical |
| 3,012 | 579.47 → 560.13 s (−3.3%) | 42219 → 38416 MiB (−9.0%) | 94.269 → 94.208 | median 0.261 Å (within-set 0.300) |

It is still not a default. This repo's bar for changing one is the panel
discipline of two native and two port processes per case; these are one case
per size, one process per arm, one card. The measurements that would decide
it are a tape-pinned same-index pair under `deterministic=on` and one run on
a non-sm120 card, neither of which exists yet.

Two size-dependent facts behind that. The fp32 pair bias the Protenix fix
adds is a fixed cost against a denoiser whose work grows with the square of
the token count, so it eats 5.8 of the policy's 8.8 points at 1k and only 1.2
of its 10.1 at 2k. And the fused kernel's memory win is the score tensor it
does not build, which is also quadratic: 9.5% of the whole process peak at 2k
on Protenix, nothing measurable on Boltz-2 at either size, because Boltz-2's
peak is a trunk arena that neither lever touches.

One caveat that is not about size: sm120 has no Mosaic GPU kernel and no
cuDNN in this install, so every tokamax number here is the Triton path, and
there is no shipped autotuning cache for this card (the heuristics config is
used and a cache-miss warning is logged on every trace). A different card can
reorder these.
