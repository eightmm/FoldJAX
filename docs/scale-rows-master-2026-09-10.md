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
- ESMFold2's 3k wall is its folding-trunk pair arena, which has no sample axis
  in it (46,041.8 MiB at 5 samples against 46,284.8 at the released 32, at
  2,096 tokens); OpenDDE's 2k wall is its pair arena in either dtype (both
  recorded before, now measured at the row). This line used to call the
  ESMFold2 wall a `num_samples x L^2` arena, which is the confidence head's
  own term and is divided away by `confidence_sample_sequential`.

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

#### Diagnosed: the second request hits the pool and JAX retraces anyway

Instrumenting `BoundedJitPool` rather than the graph's kwargs settles where the
second trace comes from. Per pool invocation, with all four authorities that
carry this default set to `tokamax` -- `models/predict.py`, `models/model.py`,
`backends/protenix.py` and the shared CLI flag:

| invocation | pool | owner's jit cache |
| --- | --- | --- |
| request 1 | miss | absent → 1 |
| request 2 | **hit** | **1 → 2** |

So both requests get the **same** owner and JAX traces twice inside it. What the
pool compares is identical across the two: `jax.config.values`, every static
keyword by `(type, repr)`, all three positional arguments by
`(type, repr, committed)`, the keyword names, and the argument counts. The
pool's identity is therefore **coarser than JAX's own dispatch key** here, which
is the failure its docstring names -- "hide multiple executables inside one
owner instead of deduping them" -- seen from the inside.

Two findings fall out on the way.

**The test's own instrumentation compares the wrong pair.** It asserts
`owner_identities[0] == owner_identities[1]`, and the test's `RecordingPool`
computes `_identity` once itself and once through `super().__call__`, so indices
0 and 1 are the *same request* twice. Those two are equal whatever the flip
does. The comparison that matters is request 1 against request 2.

**Before the four authorities were aligned, the probe named the fifth one
directly**: with only `backends/protenix.py` flipped, the two requests differed
by `diffusion_attention_backend: 'xla_jit' vs 'tokamax'`, because the omitted
request renders no flag and takes the *CLI parser's* default. That is a real
disagreement the test is built to catch, and it is why the option needs all four
sites moved together.

#### Retracted: the `_jit` boundary is not the mechanism

`JAX_EXPLAIN_CACHE_MISSES=1` counts JAX's own "function is being re-defined
repeatedly, preventing caching?" reports. Over one run of the test that catches
this:

| arm | reports | test |
| --- | ---: | --- |
| released `xla_jit` | **0** | passes |
| `tokamax` | **37** | fails, 2 traces |
| `tokamax_jit` (added for this) | **37** | fails, 2 traces |

Three of the 37 land on our own
`models/protenix/models/diffusion/transformer.py:245`, the `lax.scan` over the
diffusion transformer stack whose `body` is defined at `:220`; four are inside
tokamax's `_src/batching.py` and `_src/ops/attention/pallas_triton.py`, and
thirty report no location.

An earlier reading here said the cause was that `xla_jit` routes through
`_compiled_attention` -- an inner `jax.jit` -- while `tokamax` traces inline, so
choosing the fused kernel removed a graph boundary. **That was wrong.** Giving
the fused kernel its own `_jit` variant is a small change --
`attention_backend.removesuffix("_jit")` at `attention.py:123` and `:244`, the
context-parallel refusal, the CLI choices -- and it works: the probe runs it at
rank 4 and rank 5 operands through both compiled wrappers. It also changes
nothing. `tokamax_jit` produces the same 37 reports at the same sites as bare
`tokamax`, so the boundary is not what the tracing cache was losing. The kernel
itself is, wherever it sits.

**The Boltz-2 control that reading leaned on was vacuous**, which is worth
stating plainly because it is the same trap this repository has hit before. Its
end-to-end smoke test reports zero -- and logs **zero** autotuning cache misses
and zero `PallasTriton` entries, so the fused kernel never fired in it. A test
that does not reach the code certifies nothing about it. Two further Boltz-2
files that spell `attention_backend="tokamax"` also fired the kernel zero times,
so the question of whether the landed Boltz-2 default shows this is **still
open**.

What is not in doubt is the Boltz-2 default's measured case: -4.49 / -8.01 /
-6.42% at the three sizes, the peak byte-identical at two of them, coordinates
at 1.16x the process floor, the deposited RMSD unchanged sample for sample, and
`tests/parity --run-cpu-parity` 59 passed. Those are end-to-end wall and
structure measurements and they do not depend on this trace accounting. What a
defeated tracing cache would cost is compile time across repeated requests in
one process, which a warm-after-prefill wall measurement is designed not to see.

#### Answered: it is tokamax's own closures, on both ports

Firing the kernel deliberately settles it. One port, one backend, a `lax.scan`
body traced twice in one process, at token-transformer shapes rather than unit
test shapes -- the earlier control was vacuous precisely because small shapes
let tokamax pick an XLA implementation and the Triton kernel never fires, so the
autotuning-miss count is carried here as proof that it did.

| Boltz-2 arm | kernel fired | reports | where they land |
| --- | ---: | ---: | --- |
| `xla` | 0 | 2 | both in the probe's own closures |
| `tokamax` | 2 | **15** | the same 2, plus 3 in tokamax's `_src/batching.py:210`, 1 in `_src/ops/attention/pallas_triton.py:299`, and 9 unlocated |

**Every added report is inside tokamax.** None lands on our code here; the three
that landed on `diffusion/transformer.py:245` under Protenix are the same thing
seen from the scan body that happens to enclose the call. So this is a property
of `tokamax.dot_product_attention` at 0.0.13 rebuilding closures per trace, not
a Protenix defect, and not the `_jit` boundary.

That changes what the Protenix blocker is. The failing test asserts that two
requests share **one** traced program, and tokamax's per-trace closures make
that unreachable while the fused kernel is in the graph. The assertion and the
kernel are incompatible by construction, so the choice is to accept two traces
with the reason recorded, or to keep the default off -- not to fix something in
this repository.

And the consistency point: **the landed Boltz-2 default already has this
property.** It ships the fused denoiser attention and would report the same way.
It measures -4.49 / -8.01 / -6.42% at three sizes with the peak byte-identical
at two of them, so on the evidence that decides defaults here the property costs
nothing that shows.

#### Priced: the retrace costs nothing the fused kernel can be charged for

Two predictions inside one interpreter, 1,003 tokens on 3OG2, the two arms
submitted as one wave on their own cards, reading the **second** call -- the one
a serving process pays repeatedly and no bench row in this document contains:

| arm | first | second | second/first |
| --- | ---: | ---: | ---: |
| `diffusion_attention_backend=tokamax` | 72.07 s / 8453.4 MiB | 82.63 / 10427.0 | **1.1465** |
| `diffusion_attention_backend=xla` | 74.93 / 8453.4 | 87.33 / 10427.0 | **1.1655** |

**Both arms pay a second-call penalty and the unfused one pays more.** The peak
grows identically in both (8453.4 -> 10427.0 MiB), so whatever the second call
retains, it retains the same way with either kernel. The penalty belongs to a
second prediction in one process, not to the fused kernel, and tokamax's
per-trace closures cannot be charged for it.

The fused kernel's advantage survives and widens slightly: -3.8% on the first
call (72.07 against 74.93) and **-5.4% on the second** (82.63 against 87.33).

So the 37 "re-defined repeatedly" reports are a bookkeeping fact with no wall
consequence, and the Protenix test's two traces are the same fact. **Adopted on
that basis**: `diffusion_attention_backend` is `tokamax` on this port, and the
test asserts the two traces the kernel forces with the pricing above written
beside the assertion.

The guard must not simply be loosened to `traces <= 2`, though: that is exactly
the count a real default disagreement produces, and this same test caught one --
the CLI parser's `--diffusion-attention-backend`, the fifth authority, while the
other four had been moved. The assertions that carry that job are
`owner_identities[0] == owner_identities[1]` and `runner._entry_count() == 1`,
because a fifth authority makes the identities differ and the pool open a second
owner. Both stay; only the trace count and the owner's dispatch-cache size move
to what the kernel forces.

(An earlier reading here claimed that comparison was itself broken -- that
`RecordingPool` computes `_identity` twice per request so
`owner_identities[0] == owner_identities[1]` reads request one against itself.
That was wrong, and it came from reading the probe's own instrumentation back
into the test: the probe patched `BoundedJitPool._identity`, which both
`RecordingPool.__call__` and `BoundedJitPool.__call__` reach, so the probe saw
four calls. The test appends in `__call__` only, once per request, so indices 0
and 1 are the two requests and the comparison was always the right one.)

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

### Four open items, run the same evening (2026-09-14, snapshot `8da1d69`)

After the denoiser attention landed on both ports, four things were still
open: the chunk budgets that were tuned before the fused kernels changed the
peak composition, the `triton` + bf16 arm that had only one case behind it,
the OpenDDE blocked-multiplication verdict that had never been tried on
Protenix, and the ESMFold2 trunk that the closing plan still listed as
realised f32. All four are closed below. One draw per arm unless a redraw is
named; wall in seconds, peak in MiB, same card class, released schedule.

#### Chunk budgets are inert on the fused paths

OpenFold3 `pair_chunk_size` (auto is 117 rows at 2,096 and 58 at 3,012, sized
from a `[rows, heads, N, N]` fp32 score tensor that `cueq-full` never forms):

| tokens | chunk | wall | peak | mean pLDDT |
| ---: | ---: | ---: | ---: | ---: |
| 2,096 | 117 (auto) | 231.28 / 229.84 | 13,882.4 / 13,882.4 | 94.29 |
| 2,096 | 233 | 226.44 / 228.42 | 13,868.5 / 13,868.5 | 94.29 |
| 2,096 | 466 | 227.63 | 14,705.1 | 94.29 |
| 2,096 | 0 (off) | 222.68 | 22,967.4 | 94.29 |
| 3,012 | 58 (auto) | 575.15 | 23,773.9 | 92.187 |
| 3,012 | 116 | 577.06 | 23,774.0 | 92.188 |
| 3,012 | 232 | 575.17 | 23,774.0 | 92.187 |
| 3,012 | 0 (off) | 568.49 | 45,363.2 | 92.188 |

Inside the plateau the chunk size moves nothing on either axis: the peak
repeats to the MiB across draws and across sizes, and the wall differences
(230.6 vs 227.4 mean at 2k, −1.4%) sit inside the 1.4–2.0 s within-arm
spread. Only switching chunking off moves anything, and it buys 1–4% of wall
for 9–22 GiB. The auto formula's reasoning is wrong — it budgets a tensor that
no longer exists — but the value it picks lands on the flat part, so the wrong
reasoning is harmless. No default changes.

Protenix `chunk_policy` (auto sets all five trunk chunk knobs to one size:
128 at 2,096, 32 at 3,012; off is no chunking):

| tokens | policy | wall | peak |
| ---: | --- | ---: | ---: |
| 2,096 | auto | 204.15 | 21,253.7 |
| 2,096 | off | 207.29 | 21,253.7 |
| 3,012 | auto | 569.63 (earlier draw 560.13) | 37,539.1 |
| 3,012 | off | 549.27 | 37,542.0 |

At 2k the two arms peak at the same MiB and the wall favours auto by 1.5%; at
3k the peak is again identical and the one off draw is 3.6% faster than the
auto draw, with the auto arm's two draws 1.7% apart. The memory reason for
the policy is gone at both sizes; whether the 3k wall reading survives a
redraw decides if `off` becomes the 3k resolution, and that redraw is
recorded when it lands.

#### `triton` + bf16, five cases: deposited-equal on 5/5, coordinates 0.8–2.9× the floor

The arm is the Boltz-2 `diffusion_attention_backend=triton` spelling, which
the port ties to `diffusion_compute_dtype=bfloat16`, against today's default
(`tokamax`, fp32 denoiser); ctl and ctlB are two processes of the default, and
their same-index movement is the floor. The kernel fires at every size
(cache-miss census: `q f32[5,254,16,48]` at 254 tokens).

| case | tokens | ctl | ctlB | triton | wall Δ | peak Δ | floor (max) | triton vs ctl (max) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DHA | 254 | 17.83 | 17.85 | 16.82 | −5.7% | 0 | 0.070 Å | 0.056 Å | 0.8× |
| 4REK | 500 | 29.48 | 29.42 | 27.78 | −5.8% | +450 MiB | 0.030 | 0.050 | 1.7× |
| 3OG2 | 1,003 | 74.33 | 73.84 | 67.56 | −9.1% | 0 | 0.275 | 0.382 | 1.4× |
| 3LXU | 1,350 | 103.42 | 106.04 | 95.99 | −7.2% | 0 | 0.635 | 1.811 | 2.9× |
| 5DEI | 2,096 | 239.69 | 237.87 | 227.62 | −5.0% | 0 | 0.141 | 0.258 | 1.8× |

Deposited CA RMSD (permutation-aware, `label_seq_id`) is the same on the
three arms of every case to within the floor's own movement (0.01–0.06 Å):
3DHA 0.45–0.55 on all arms, 4REK 0.46–0.55, 3OG2 0.92–1.47, 3LXU 5.20–5.40,
5DEI per chain identical. The earlier 5DEI draw that read −12.62% and 9.4×
had a floor of 0.020 Å; this draw's floor on the same case is 0.141 Å, seven
times larger, so the ratio column is a random variable of the floor draw as
much as of the arm, and 3LXU is a 5.3 Å target where a 1.8 Å displacement
between two equally wrong structures is invisible to the deposited score.

What the panel lacks is a weak-interface multimer, which is the exact target
class where rounding the pair bias to bf16 moved a chain 16 Å on another port,
and this arm rounds that bias: the `triton` guard requires bf16 q, k, v *and*
bias, and reaching it requires the whole-denoiser knob (`trunk.py:625-629`),
which narrows every score-model Linear except two islands. So the arm is two
variables. It is not adopted as-is. The split — bf16 q/k/v cast at the kernel
call, bias and denoiser fp32 — is the next measurement, and the audit that
priced it found that tokamax's Triton kernel also casts the softmax
probabilities to `v.dtype` before P·V and stores the output in `q.dtype`, so
the split is not operand rounding alone and gets the same floor test.

#### OpenDDE's blocked multiplication does not transfer to Protenix

OpenDDE routes its triangle multiplication to the blocked XLA path under a
bf16 trunk (`opendde/models/model.py:373`, measured 2026-08-23 as −13.4% wall
and −11% peak against cueq at 1,003 residues). Protenix runs the same
triangle code on the same bf16 trunk and defaults to cueq. Tried on Protenix:

| tokens | arm | wall | peak |
| ---: | --- | ---: | ---: |
| 1,003 | cueq (default) | 60.96 | 6,375.6 |
| 1,003 | blocked (`PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla`) | 60.59 | 6,658.3 |
| 2,096 | cueq (default, same wave) | 204.15 | 21,253.7 |
| 2,096 | blocked | 224.38 | 22,347.1 |

A tie at 1k with 283 MiB against blocked, and +9.9% wall with +1.1 GiB at 2k.
The difference is shape, not code: OpenDDE's pair is 384 channels on
structural tokens (about twice the residues), Protenix's is 128 channels on
tokens, and the fused path's `2·(2c)` concatenated projections against the
blocked path's padded copies scale differently in `c`. Protenix keeps cueq.

#### ESMFold2's trunk is bf16; one f32 buffer remains, and it is not the peak

The closing plan's "realised f32 trunk" was stale: the promotion it referred
to (mask multiply and an explicit `astype`) was removed in `8e5a246`
(2026-09-11), and the realised trunk and the persistent pair are bf16 today
(2,145 MiB at 2,096 tokens). A read-only audit found one long-lived f32
buffer left: `normalized` at `esmfold2/models/trunk.py:117`, the layer-norm
output at `[N², 256]`, live across the O(N³) einsum because both consumers
(`proj_bundle`, `proj_gate`) are separate GEMMs. Narrowing it once after the
norm is bitwise identical on CPU (both consumers cast to bf16 internally; a
×2 tripwire fired), upstream-faithful in output and a deliberate divergence in
realised dtype. Upper bound 982 / 4,290 / 8,860 MiB at 1,003 / 2,096 / 3,012
tokens, and possibly zero if XLA CSEs the two converts. It is not scheduled:
ESMFold2's peak is the per-sample diffusion + confidence arena
(`num_samples · L² · 4c_z`), not the trunk, so a trunk-only narrowing cannot
move the peak unless the trunk becomes the maximum tenant.

#### sm120 autotuning, priced and not built

AlphaFold 3 persists tokamax autotuning results under `FOLDJAX_HOME`
(`backends/_tokamax_autotune.py`); the store is model-agnostic in all but
four seams. The Triton attention search space is `block_q`/`block_k` in
{16..128}, `num_warps` {1,2,4,8}, `num_stages` {1..4}; split-k is never
explored, but a tuned config is still not bit-identical to the heuristic one
because `block_k` sets the online-softmax order. Ceiling: with the 8.01% the
xla→tokamax move saved at 2k, the kernel's share of the new wall is
0.0871/(r−1) for an XLA/tokamax speed ratio r — 8.7 / 4.4 / 2.9% at r = 2 / 3
/ 4 — so a 1.5× tuned-over-heuristic kernel is worth at most ~1.5% of wall,
against a whole-program warm that costs AF3 326 s at 2k, longer than a
Boltz-2 prediction at that size. Arithmetic closes it; if ever built it is an
opt-in `kernel_autotuning`, never a default.

#### The per-step bias work, priced

Both ports run the sampler as one `lax.scan` with the transformer blocks in a
nested scan (Protenix forces `use_diffusion_scan=True` under `graph_jit`,
`models/predict.py:217-225`), so the per-block pair-bias projection is
re-done every step on both ports and XLA's loop-invariant code motion cannot
reach it. On Boltz-2 that is deliberate (`lazy_token_trans_bias=True`; the
eager alternative holds `[N, N, 24·16]` f32, 6.7 GB at 2,096 and 36 GB at
4,888, and bf16 storage is ruled out by the pair-bias rounding finding). The
priced traffic is ~2.2 GB of `z` reads per layer-step, ~7 s of 257 s at
2,096 if bandwidth-bound and unfused. Left lazy. What is not deliberate is
`bias = jnp.repeat(bias, multiplicity, axis=0)` per layer per step
(`diffusion_transformer.py:236`): five copies of a bias that is identical
across samples, ~4 s at 2,096 by the same arithmetic, and bit-identical to
remove if the kernel broadcasts a batch-1 bias. That patch is in flight.

### The second batch of the evening: two landings, two rejections, one lever found (2026-09-14, later)

The gap hunt continued with a compile-only arena probe and four Opus
patches, each measured on its own snapshot with two control draws.

#### Landed: the per-layer bias repeat on Boltz-2 (`5c064d3`)

| arm | wall | peak |
| --- | ---: | ---: |
| ctl (main) | 239.47 | 18,511.1 |
| ctlB (main) | 236.85 | 18,511.1 |
| bias broadcast | 231.22 | 18,511.1 |

−2.4..−2.9% at 2,096 tokens, coordinates 0.028 Å from the control against a
0.066 Å control-vs-control floor, deposited CA RMSD identical on all five
samples. tokamax squeezes a size-1 batch axis and maps that operand with
`in_axes=None`, so the kernel reads one `[H, N, N]` buffer for every sample;
the xla branch is a broadcasting add. The 2-D context-parallel path keeps
the repeat because its bias `PartitionSpec` names the batch axis.

#### Landed: Protenix `chunk_policy=auto` stops chunking the fused trunk (`cf4360f`)

The 3,012 redraws confirmed the first draw: auto 569.63 / 568.86 s against
off 549.27 / 549.19 s (−3.5% twice), peak identical to the mebibyte, and the
off-vs-auto coordinate movement (0.069–0.072 Å) sits at the auto-vs-auto
floor (0.070 Å). Protenix's `auto` now resolves the five trunk knobs to no
chunking up to 3,012 tokens; above that the upstream value stays because
that side is unmeasured. OpenDDE names upstream's table explicitly and keeps
it, because its blocked triangle path honours the widths, and two tests hold
the two callers to the tables they measured.

#### Rejected: bf16 q/k/v at the kernel call, on both ports

The split of the `triton` + bf16 arm that the audit priced: q, k, v cast to
bfloat16 at the tokamax call, pair bias and the rest of the denoiser fp32.
tokamax's Triton kernel adds the f32 bias into f32 logits unrounded, but it
also casts the softmax probabilities to `v.dtype` before P·V and writes the
output at `q.dtype`, so the split is three roundings, not one.

| port | tokens | ctl | ctlB | bf16 q/k/v | wall Δ | floor | move | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protenix | 2,096 | 192.25 | 192.07 | 190.63 | −0.8% | 0.011 Å | 0.254 Å | 23× |
| Protenix | 1,003 | 60.23 | 59.87 | 58.46 | −2.6% | 0.323 | 0.318 | at floor |
| Boltz-2 | 2,096 | 231.28 | 231.32 | 230.39 | −0.4% | – | – | – |
| Boltz-2 | 1,003 | 70.33 | 70.90 | 68.40 | −3.0% | – | – | – |

Under one percent at 2,096 on both ports, and 23× the floor on Protenix.
The 5–9% of the `triton` + bf16 arm is the bf16 denoiser, not the kernel's
operand dtype. Neither `tokamax_bf16` spelling is landed.

#### Closed by measurement: OpenDDE's blocked multiplication on Protenix, the sm120 autotuner, ESMFold2's norm

Recorded in the previous section. The blocked path loses at 2,096
(+9.9% wall, +1.1 GiB); the autotuner is priced at ≤1.5% of wall against a
326 s warm; ESMFold2's remaining f32 buffer is not the peak tenant.

#### The Protenix arena, read from the compiled program

A compile-only probe (lower and compile the one executable, never run it;
buffer assignment joined to the loop nest by each `while`'s proved trip
count) at 2,096 tokens, released defaults:

* temp arena 19.33 GiB, argument 1.29 GiB; 2,604 allocations, parsed bytes
  equal to `memory_analysis()` exactly.
* The 24 diffusion blocks are a nested `lax.scan` on the CLI path
  (`models/predict.py:217-225` forces `use_diffusion_scan=True` under
  `graph_jit`), so the per-block pair bias is recomputed per step on this
  port too and nothing pair-shaped is hoisted above the sampler. The K=200
  loop carries exactly one pair tensor: the pre-normalised `z_norm`,
  f32[1, 2096, 2096, 128], 2.1 GiB.
* The two largest values are f32[2096, 2096, 256] at 4.19 GiB each, at entry
  level: the layer norm of the diffusion pair-conditioning concat
  (`[z_trunk bf16 | relpe f32]` is f32 by promotion), centered and normalised
  materialised separately. This is the deliberate fp32 diffusion island;
  left alone.
* The top of the arena is the MSA stack: bf16[13267·2096, 64] at 3.3 GiB
  three times, bf16[8, 13267, 8, 2096] at 3.3 GiB, two 1.66 GiB tenants.
  n_msa is 13,267 rows here; upstream's policy draws a uniform-random subset
  size in [1, n_msa] per recycle and pads to the maximum over cycles.

#### The lever that follows: Protenix MSA depth

| tokens | `max_msa_depth` | wall | peak |
| ---: | ---: | ---: | ---: |
| 2,096 | 16384 (default) | 190.51 | 21,225.4 |
| 2,096 | 4096 | 179.06 | 13,754.2 |
| 2,096 | 2048 | 175.62 | 13,726.8 |
| 3,012 | 16384 (default, auto chunk) | 569.63 | 37,539.1 |
| 3,012 | 4096 | 496.55 | 27,058.9 |

−35% peak and −6% wall at 2,096; −28% peak and −13% wall at 3,012. Below
4096 the peak stops moving, so the next tenant is about 13.7 GiB at 2k. Of
the ladder, 3DHA (10.2k rows), 4REK (12.9k), 3OG2 (8.8k), 5DEI (13.3k) and
6ZTX (17.5k) bind at 4096; 3LXU (2.7k) does not.

This is not a kernel or a dtype: it changes what the model reads, so the
rerun floor is the wrong control and the 2026-08-02 "confidence inside the
MSA-draw noise band" is not an admission. The admission test (agreed with
the spec partner) is per-case seed blocks — A uncapped with MSA seed a, B
uncapped with MSA seed b, P capped with MSA seed b, diffusion RNG held
fixed — comparing d(A, P) against d(A, B) on deposited CA RMSD, complex and
per chain, with a frozen 10% margin on the one-sided 95% upper bound over
blocks. It needs an MSA seed separate from the diffusion seed, which the
port did not have (`cli/predict.py:1071/1083/1190` share one); that option
and the harness are the next patch. The cap stays opt-in until the panel
passes.

#### The MSA-cap admission test: FAIL at the frozen margin; the cap stays opt-in

Design as agreed with the spec partner, run on `x12-msacap-20260914` (main
`754134f`): per case, four seed blocks of three arms — A uncapped with MSA
seed a, B uncapped with MSA seed b, P `max_msa_depth=4096` with MSA seed b —
all three under upstream's per-cycle row sampling (`--sample-msa-per-cycle`,
because FoldJAX's released path is full-depth and there `msa_seed` is inert,
so A and B would be the same program), diffusion RNG fixed at seed 101. The
statistic is mean d(A, P) − mean d(A, B) over blocks on permutation-aware CA
RMSD after superposition, with a one-sided 95% block-bootstrap upper bound
against a margin of 10% of the control distance, frozen before the rows ran.

| case | control mean d(A,B) | statistic | 95% upper bound | verdict | deposited A / B / P |
| --- | ---: | ---: | ---: | --- | --- |
| 4REK (500) | 0.300 Å | +0.135 (+45%) | +0.371 (+124%) | FAIL | 1.12 / 1.05 / 1.08 |
| 3OG2 (1,003) | 0.502 | −0.045 (−9%) | +0.190 (+38%) | FAIL (power) | 0.82 / 0.76 / 0.80 |
| 5DEI (2,096) | 0.261 | +0.056 (+21%) | +0.099 (+38%) | FAIL | 0.53 / 0.47 / 0.47 |

Read plainly: against the deposited entries the capped arm is as good as the
uncapped arms on all three cases, but the cap moves the coordinates more than
an MSA re-draw does on two of the three (by 21% and 45% of the draw distance),
and the third is a power failure rather than a pass (the point estimate is
inside the margin; the bound is not). The rule was frozen at 10% and it is not
met, so the default stays `max_msa_depth=16384`; the cap remains the first
opt-in knob for a memory-bound Protenix job (`--option max_msa_depth=4096`:
2k peak 21,225 → 13,754 MiB, 3k 37,539 → 27,059 MiB). On the released
full-depth path the cap reads 1.03–1.06 Å on 4REK and 0.76–0.86 Å on 3OG2
against 0.80–0.88 Å uncapped, one draw each.

Two things a later panel would change. More blocks: the bound on 3OG2 is a
dispersion problem, and the harness prints the margin the observed spread
would have passed (38%). And the block-level floor is bimodal — some A/B pairs
land at the process floor (0.03–0.07 Å) because two draws of a uniform size
in [1, n] can coincide in what they cover — which inflates the ratio on those
blocks; a design that draws sizes without that coincidence would tighten the
control. The harness is `tmp/msacap/admit.py` in the session's job directory;
the option it needs (`msa_seed`) is on main.

### Closing comparison against upstream, on the defaults as landed (2026-09-14, `0d26624`)

Every row below is one process of the released defaults on main as of
`0d26624` (snapshot `x11-final-20260914`), seed 101, released schedule, one
card class; the upstream column is the 2026-09-10 upstream arm on the same
host (`scale-timing-20260910`, provenance in "Provenance of the upstream arm
on master"). Wall in seconds, peak in MiB.

| model | tokens | FoldJAX wall | upstream wall | Δ | FoldJAX peak | upstream peak | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boltz-2 | 1,003 | 71.0 | 132.5 | −46% | 8,453 | 16,162 | −48% |
| Boltz-2 | 2,096 | 230.9 | 465.6 | −50% | 18,511 | 47,722 | −61% |
| Boltz-2 | 3,012 | 649.5 | OOM | – | 29,416 | OOM | – |
| OpenFold3 | 1,003 | 71.0 | 139.1 | −49% | 5,336 | 14,345 | −63% |
| OpenFold3 | 2,096 | 228.9 | 629.1 | −64% | 13,882 | 45,256 | −69% |
| OpenFold3 | 3,012 | 580.6 | 1,430.2 | −59% | 23,774 | 81,589 | −71% |
| Protenix | 1,003 | 59.4 | 96.2 | −38% | 6,404 | 12,932 | −51% |
| Protenix | 2,096 | 191.0 | 241.9 | −21% | 21,225 | 41,796 | −49% |
| Protenix | 3,012 | 526.2 | 650.2 | −19% | 37,537 | 57,238 | −34% |
| OpenDDE | 1,003 | 146.0 | 246.9 | −41% | 21,492 | 58,346 | −63% |

The upstream wall column is four days older than the FoldJAX column, and
this card has moved 36% under throttle before, so three of its 2,096-token
rows were re-run the same evening on the final rows' own native inputs
(`uprow-recheck.sbatch`, results `*-upstream-recheck.json`): Boltz-2 465.24 s
/ 47,721.7 MiB, OpenFold3 624.75 / 45,256.2, Protenix 242.49 / 41,795.8 —
within 1% of the 2026-09-10 walls and identical peaks, so the deltas above
stand as same-day comparisons at 2k and the 1k/3k walls carry the four-day
caveat only.

Against the deposited entries (permutation-aware CA RMSD on `label_seq_id`,
five samples):

| model | 3OG2 (1,003) | 5DEI (2,096) | 6ZTX (3,012) |
| --- | --- | --- | --- |
| Boltz-2 | 1.42 0.96 1.22 0.93 0.98 | 0.44 0.43 0.42 0.47 0.42 | 0.62 0.62 0.63 0.66 0.64 |
| OpenFold3 | 0.81 0.91 0.79 0.85 0.88 | 0.52 0.53 0.54 0.57 0.52 | 0.87 0.64 0.66 0.88 0.90 |
| Protenix | 0.88 0.80 0.88 0.86 0.85 | 0.46 0.52 0.44 0.49 0.47 | 0.66 0.63 0.61 0.57 0.67 |
| OpenDDE | 0.91 0.73 0.95 1.02 1.00 | – | – |

Protenix's mean pLDDT on the same rows is 94.58 / 95.17 / 94.28 against
upstream's 94.71 / 95.12 / 93.55, and OpenDDE's 94.74 against 94.72. The
structure-level agreement with upstream on these cases is the "Structure
agreement without a shared tape" section above: cross TM 0.99–1.00 with
within-arm spread of the same size.

What moved today, in the order it landed: Boltz-2 and Protenix denoiser
attention on tokamax (`98c5ebd`, `845971e`), one shared call into the fused
triangle multiplication (`027ff11`), the per-layer bias broadcast on Boltz-2
(`5c064d3`), Protenix's auto chunk policy off the fused trunk (`cf4360f`), and
the Protenix `msa_seed` option (`754134f`) that the MSA-cap admission test
needs. Rejected with numbers: Protenix blocked multiplication, bf16 q/k/v on
both ports, the chunk sizes inside the plateau, the sm120 autotuner, ESMFold2's
last f32 norm. Open with a measured lever and a test in flight:
`max_msa_depth=4096` on Protenix (−35% peak, −6% wall at 2k; −28%, −13% at 3k).

The interface and the program were checked against each other on the final
snapshot: the tokamax cache-miss census shows the denoiser attention firing at
three signatures on Boltz-2 and Protenix and none on OpenFold3 (cueq-full), the
Boltz-2 token bias reaching the kernel as `f32[1, 16, N, N]` where it was
`[5, 16, N, N]` before `5c064d3`, and `docs/cli.md` now names the Protenix chunk
default, the Boltz-2 denoiser default and `--msa-seed` (`0f4f3ee`).

### VRAM-aware admission and the memory laws (2026-09-15, `e610cab`)

The chunk and memory defaults above were decided on 96 GiB cards. The
policy that adapts them to the card is admission, not knob-turning: each
port estimates its peak from a law fitted on this ledger's rows, compares
the upper estimate with 0.9 × the allocator's ceiling (`bytes_limit`, which
is the memory fraction × the card with preallocation on or off), and reports
`fits`, `over_budget` or `unknown` before anything large is allocated.
`over_budget` refuses with the estimate, the threshold, the pool and the
levers named (`--memory-check=warn` downgrades it); `unknown` (no budget, or
a size outside the fitted domain) proceeds with one warning; nothing that
changes the input is ever applied automatically — the MSA cap that failed
its admission test stays opt-in. `--memory-budget-gib` plans against an
explicit budget (min with the pool when the pool is known), and the run
manifest records the decision.

| law | form (MiB) | allowance | domain | rows fitted |
| --- | --- | ---: | --- | --- |
| Boltz-2 (5 samples) | 3910 + 4.58·N + 4.33e-7·N³ | 1,006 | 1,003–4,888 | 8,453/18,511/29,416/51,712/77,312 |
| Protenix | max(1235 + 2.85e-3·N², 7.32e-4·M·N), M = processed MSA rows | 1,018 | 1,003–3,012 | see the two-phase note below |
| OpenFold3 chunked (128 rows) | 1460 + 0.875·N + 2.23e-3·N² | 783 | 1,003–4,888 | 4,316.7/13,882/23,774/42,468.6/59,214.8 |
| OpenFold3 unchunked | 1438 + 4.85e-3·N² | 215 | 1,003–3,012 | 6,194.6/22,967/45,363 |
| OpenDDE bf16 trunk (5 samples) | 4016 + 4.83e-3·S², S = structural tokens | 1,171 | 1,902–7,876 S | 21,492 at S=1,902 (1,003 res); 46,858.2 at S=2,978 (1,531 res) |
| OpenDDE fp32 trunk (5 samples) | 1.180e-2·S² | 1,199 | 1,902–7,876 S | 43,090 and 42,291.2, both at S=1,902 |
| ESMFold2 (5–32 samples) | 5435 + 9.24e-3·N² | 295 | 1,003–2,096 | 14,733.3/46,041.8 |

The two laws added 2026-09-16 (`memory admission for OpenDDE and ESMFold2`)
are the two ports whose `--memory-check` and `--memory-budget-gib` were inert
before it, and each has a shape the four above do not:

- **OpenDDE is keyed on the structural token count**, not the residue count:
  S/residues is 1.896 at 1,003 residues and 1.945 at 1,531 and 4,100, so a
  residue-keyed law would carry that drift squared. Two arms, one per realised
  trunk dtype, because at S=1,902 float32 costs twice the peak (42,690 against
  21,492 MiB fitted) and the dtype is therefore a lever the refusal names.
  The bf16 arm is an exact two-point fit whose parameters are separately
  corroborated rather than free: the intercept lands at 4,016 MiB, inside the
  3.0–4.6 GiB of arguments this port carries across these sizes, and the
  square term at 0.883 of the fused-arm arena coefficient 5.4724e-3 against
  0.866 measured (the bf16 trunk routes its triangle multiplication to the
  blocked XLA path, which took 2,650 MiB off the 19,797 MiB fused arena at
  S=1,902). Its allowance is the snapshot spread: the same configuration read
  20,326.4 MiB on 2026-08-23 and 21,492 on 2026-09-10. The fp32 arm has one
  fitted size measured twice, so it is a square term with no intercept; its
  coefficient lands at 1.092 of the measured fp32 arena coefficient, i.e. an
  arena that is 91.6% of the peak, the share the bf16 arm shows too.
  Both domains end at 7,876 S, above everything fitted and with no failure
  fitted to anything: what the top rests on is that the verdict has been
  checked against an outcome there — S≈4,034 (2,096 residues) asked an 87 GiB
  arena and died serial and on both 1-D layouts, S=7,876 (4,100 residues)
  asked 331.5 GiB and died, and the fp32 arm's own censored point is S=2,978
  (93.40 GiB requested, and `PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla`
  asked 80.72 and died too). Read the top of the domain as a verdict and not
  as a value: at S≈4,034 the bf16 law estimates 80.7 GiB, below the ~86 GiB
  the run actually exhausted, so above S=2,978 it reads *low* and what carries
  the refusal is the 0.9 admission fraction rather than the estimate alone. This law replaces the arena preflight that used
  to warn here, which estimated the temp arena alone.
- **ESMFold2's law has no sample term, and that is measured.** Three places in
  this repository still describe its peak as `num_samples · L² · 4c_z`; that is
  the confidence head's term, which `confidence_sample_sequential` (on by
  default) divides away, and what remains is the folding trunk, which has no
  sample axis. At 2,096 tokens the peak is 46,041.8 MiB at 5 samples and
  46,284.8 at the released 32 — 0.53% — so the law is fitted at 5, declared
  valid to 32, and its allowance (243.0 + the 51.2 the 45.2 GiB row is quoted
  to) carries the difference. Without that widening a refusal would never fire
  on a default run, because this port reads its sample count off the
  checkpoint and the released value is 32. Both rows come from the control arm
  of GPU rows 1111/1112, which is where this port's peaks are recorded to the
  tenth of a MiB. The domain stops at 2,096, the largest size it has
  completed; 3,012 is censored and outside — the law reads 89,364 MiB = 87.3
  GiB there against the 86 GiB its allocator asked for before failing on this
  95.6 GiB card, which is a check and not a fit.

AlphaFold 3 still has no law, and now says so: both flags are accepted and
answered `unknown` once, naming the port, for a caller who passes one of them
(a run that passes neither stays silent, the way a run that fits does). Its peaks here are the harness's own
XLA-client high-water marks, which undercount its vendored runtime's
allocations, so there is nothing to fit. Before this it rejected the flags
outright, which reads like a misspelling rather than a missing measurement.

The allowance is the largest in-sample underestimate plus the measured
repeat spread; leave-one-size-out folds are printed by the calibration
script as an extrapolation diagnostic and not used, because the rule that
used them refused a run that had completed (Boltz-2 at 4,888, 77.3 GiB
measured). Every fit point is admitted at this host's threshold; the
tightest is that same row with 900 MiB to spare. Protenix's peak is the
maximum of two phases, not their sum: at 2,096 tokens the MSA stack sets it
at full depth (21.2 GiB with 13,267 rows) and the pair phase takes over
below about 8,000 rows (14.7 GiB at 8,192, 13.75 at 4,096, 13.73 at 2,048).

OpenFold3 resolves one configuration rather than selecting between two
(2026-09-16): inside the validated domain the automatic answer is the
fixed 128-row pair chunk at every size and whatever the card reports, and
admission is then the same verdict-only comparison the other two laws get.
The unchunked loop was the automatic answer whenever its own upper estimate
fit, and on a 96 GiB card that bought seconds for gigabytes: at 2,096
tokens (5DEI, same snapshot) unchunked is 220.85 s / 22,967 MiB against
blocked-128's 228.5 s / 13,990 MiB — +65% peak for −3.4% wall — and at
1,003 tokens the wall is equal (71.6 against 72.0 s) for 6,195 against
4,317 MiB; at 4,100 and 4,888 only the blocked arm runs at all. Memory is
what these defaults are judged on, so the seconds do not buy the
gigabytes. `OPENFOLD3_UNCHUNKED_PEAK` stays as the estimate for a run that
asks for the unblocked loop by name (`--option pair_chunk_size=0`); it is
no longer a candidate, so `--memory-check` now reaches this port like the
other two and an over-budget estimate is a refusal. Measured on the
landed rule (`8a84569`, same 5DEI row): 230.4 s / 13,990 MiB, coordinates
bitwise identical to the unchunked row it replaced, deposited identical.

The MSA padding ladder moved the same day (`f0683e7`): 2,048-row steps
above 2,048 rows, so a padded Protenix job pays at most one step on the
axis that sets its peak — 13,267 stored rows now pad to 14,336, 25,281 MiB
against 28,202 on the 16,384 rung (+19% over the exact run instead of
+33%), coordinates 0.007 Å from the exact run, deposited identical.

128 replaces the score-tensor formula because it was never worse over five
sizes and won where the formula lost: 1,003 tokens 4,317 MiB against the
formula's 5,336 (−19%) and off's 6,195, wall equal; 4,100 tokens 2,267 s /
42,469 MiB against 2,368 / 44,442; 4,888 tokens 1,418 s / 59,215 against
1,527 / 62,054; the 2k and 3k plateaus contain it. Below 1,003 tokens the
old unblocked program stays (the CPU parity captures pin it, and the
blocked arm has no measurement there). Coordinates at 1k: 128 vs the
formula 0.019 Å, off vs the formula 0.0175 Å, deposited identical.

Checked on the card with explicit budgets (5DEI, 2,096 tokens):

| port | budget | decision | outcome |
| --- | ---: | --- | --- |
| Protenix | 18 GiB | over_budget | refused before compile: "19.9 GiB + 1.0 GiB allowance against a 16.2 GiB threshold" |
| Protenix | 18 GiB, `--memory-check=warn` | over_budget, warned | ran, 191.9 s / 21,223 MiB |
| Protenix | 30 GiB | fits | ran, 192.9 s / 21,225 MiB |
| OpenFold3 | 20 GiB | fits, chunked (128) | 228.5 s / 13,990 MiB |
| OpenFold3 | 40 GiB | fits, chunked (128) — unchunked before 2026-09-16 | same program as the row above; the unchunked arm this budget used to select ran 225.0 s / 22,967 MiB |
| Boltz-2 | 18 GiB | over_budget | refused before compile |

Checked on the CPU with `--memory-budget-gib 1` on the two ports added
2026-09-16 (padding used to reach each law's domain, tiny examples otherwise
falling below it and answering `unknown`):

| port | shape | `refuse` | `warn` |
| --- | --- | --- | --- |
| OpenDDE | `--pad-structural-tokens 2048` | refused before the first trace: "2048 structural tokens … bf16 trunk 23.7 GiB + 1.1 GiB allowance against a 0.9 GiB threshold", naming `--cp-devices 4` | warned with the same text, then the host allocator refused 853 GB — "the allocator will answer", exactly as the message says |
| ESMFold2 | `--pad-tokens 1024` | refused before the language model: "1024 tokens … released 14.8 GiB + 0.3 GiB allowance against a 0.9 GiB threshold" | warned with the same text and ran on into ESMC and the trunk |

AlphaFold 3's `unknown` warning was not exercised end to end: the vendored
`alphafold3` package is not installed in this environment, so the adapter's
`predict` cannot be reached. What is checked is that both flags are in its
option set, that `foldjax plan` validates them there, and that
`memory_policy.admit_unmeasured` warns once and records the decision.

Two knobs measured for Boltz-2 and found inert for memory (its peak is the
trunk pair arena): `token_attention_chunk=256` (234.0 s / 18,511 MiB) and
`diffusion_chunk_size=1` (306.7 s / 18,511 MiB, +33% wall). No automatic
memory lever exists on that port; the admission only refuses or warns.

### Context parallelism on the deployment cards (2026-09-15)

The purpose of context parallelism here is to run targets that do not fit
one card; wall time is secondary. Both collective probes passed on this
node (row-sharded f32[N, 256] gathered by replication and reduced, 8 rows
to 64 MiB, on GPU0–1 (PCIe-switch pair) and on all four cards across the
host bridge). Released defaults, 5 samples, seed 101, coordinates against
the serial row of the same snapshot (the serial rerun floors are 0.011 Å
Protenix, 0.066 Å Boltz-2, ~0.03 Å OpenFold3, 0.52 Å OpenDDE):

| port | tokens | cards, layout | wall | peak per device | serial peak | move vs serial |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Protenix | 2,096 | 2, 1-D | 230.2 | 13,179 | 21,225 | 0.023 Å |
| Protenix | 2,096 | 4, 1-D | 356.7 | 10,663 | 21,225 | 0.016 Å |
| OpenDDE | 1,003 | 2, 1-D | 154.8 | 14,515 | 21,492 | 0.098 Å |
| Boltz-2 | 2,096 | 2, 1-D | 430.1 | 17,086 | 18,511 | 0.035 Å |
| Boltz-2 | 2,096 | 4, 1-D | 545.6 | 19,040 | 18,511 | – |
| OpenFold3 | 2,096 | 2, 1-D | 395.6 | 13,566 | 13,882 | 0.030 Å |
| OpenFold3 | 2,096 | 4, 1-D | 442.7 | 11,489 | 13,882 | – |

Accuracy holds on every completed row. Per-device memory falls with the
card count on Protenix and OpenDDE, whose peaks are pair-shaped; it barely
moves on Boltz-2 and OpenFold3 under 1-D because the 1-D triangle
multiplication all-gathers a full-width operand and the fused kernels
resolve to XLA under a mesh.

Two defects were found and fixed the same day. Protenix refused to start
under CP with no attention option because its released `tokamax` default is
not partitionable; it now resolves to `xla_jit` under a mesh, the rule
Boltz-2 already applied (`834c55b`). And the 2-D layout hung a 4-card
Boltz-2 run at 2,096 tokens: GPU 0 asked for 56.78 GiB and the other three
ranks waited forever at the NCCL clique rendezvous. That size is not an atom
tensor; it is the 2-D ring triangle attention forming its whole local score
tile, f32[B, N/s, H, N/s, N/s], 17.15 GiB at 2×2 with about three co-live,
where the serial and 1-D paths chunk the query axis. The ring now computes
its tile in query blocks (`dcfe0f0`; single-block lowering byte-identical to
the old ring; 8.38 GiB at the default block of 512, and smaller with the
per-port query-chunk option). The 2-D rows are re-run on that fix below.

Distributed atom graph on Protenix (`a750737`): the diffusion atom
encoder/decoder windows, atom-pair conditioning, atom↔token routing and the
token transformer's pair bias are sharded under a mesh (option
`cp_atom_windows`, default on; the serial denoiser lowering is pinned
byte-identical to the previous program). It needs the padded atom count to
divide 32 × the CP rows and the token count to divide the rows, which
`--padding` (automatic) satisfies; a misaligned shape warns once and runs the
replicated graph. The same graph landed on OpenDDE (`b2cac84`, aligned on
structural tokens) and OpenFold3 (`74fdb6e`, index-driven ring gather).
Measured at 3,012 tokens on 4 cards, 1-D, `--padding` (6ZTX):

| port | atom graph | wall | peak per device | serial padded |
| --- | --- | ---: | ---: | ---: |
| Protenix | distributed | 858.2 | 11,413 | 27,163 |
| Protenix | replicated | 1,453.8 | 12,012 | 27,163 |
| OpenFold3 | distributed | 1,059.3 | 28,024 | 37,414 |
| OpenFold3 | replicated | 1,076.7 | 28,024 | 37,414 |

On Protenix the distributed graph is 41% faster than the replicated one and
closer to the serial coordinates (0.026 Å against 0.853 Å; deterministic
ops on all three arms). On OpenFold3 at this size the pair stack sets the
peak, so the atom graph moves neither number; its value there is
structural. OpenFold3's 6ZTX rows moved 0.77–3.54 Å same-index against a
serial that reproduces itself bitwise, with the deposited scores landing in
the same two basins (0.63 and 0.90 Å): a chaotic target where CP's
reduction order flips one sample's basin, not a floor and not a defect.

**The 2-D layout, three fixes deep.** The query-block ring (`dcfe0f0`) kept
its blocks co-live because an unrolled loop lets XLA schedule them
together (GPU 0 asked 68.18 GiB); `bef3922` made the blocks a `lax.scan`,
and Boltz-2 2,096 on 2×2 then completed at 18,437 MiB per device (block
512) and 16,659 (block 128, −10% against serial). OpenDDE 2,096 still ran
out of memory, 106.46 GiB on every rank padded and 98.6 GiB unpadded, with
the arena at half of serial where a 2×2 grid should quarter it. A CPU probe
with four fake devices read the post-SPMD HLO (sharding is decided before
backend fusion, so per-device shapes transfer even though bytes do not): no
pair tensor was replicated. The ring's second-pass `while` carried five
unblocked f32[N/2, H, N/2, 32] tensors — q, k, v, the accumulator and its
correction — because the 2-D path had given up the row loop the serial and
1-D paths run (24 rows for 12 heads); 67.5 of the 106 GiB. And OpenDDE's
role-conditioned pair projection gathered its weights as [N/2, 7, 384, 384]
under a mesh, 5.9 GiB where 0.84 would do. `c26c998` makes the local row
block the ring's outermost loop (projection inside the block; carry 3.14×
smaller at two sizes, bit-identical across 120 arms including 3×3 meshes),
and `356f404` indexes both roles inside a `lax.scan` over the seven roles
(indexing alone is not enough: unrolled, XLA co-schedules the seven
gathers). Rows on `7bb0838`, 2,096 tokens, 4 cards, 2×2, no padding:

| port | wall | peak per device | serial | 1-D on 4 cards | move vs serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDDE | 1,770.2 | 32,068 | OOM (87 GiB arena) | OOM | – (deposited 0.59–0.75 Å) |
| Boltz-2 (block 128) | 973.0 | 16,642 | 18,511 | 19,040 | 0.112 Å, deposited identical |
| Protenix | 559.9 | 11,572 | 21,225 | 10,663 | 0.014 Å, deposited identical |

OpenDDE at 2,096 residues is the first target on this node that no other
layout runs: serial, 1-D on two cards and 1-D on four all die on the
k·N_st² arena plus the 1-D triangle multiplication's full-width gather.
That row is the reason `cp_layout=auto` now resolves to the 2-D grid on
square device counts for OpenDDE and Boltz-2 (the two ports where 2-D wins
or is the only fit) and stays 1-D for Protenix (1-D 10.7 GiB against 2-D
11.6) and OpenFold3 (2-D unmeasured). 2-D is slower — 3.5× serial wall on
Boltz-2 — and is chosen for the ceiling, which is what CP is for here.

**Ceilings.** Protenix 8E2F at 4,888 tokens asks 94 GiB serial and dies on
a 96 GiB card. On four cards, 1-D: unpadded, the atom graph replicated
(37,422 atoms are not 128-aligned), the warmup completed in about 75
minutes and the measured pass timed out; padded to the 5,120 bucket
(122,880 atoms, atom graph distributed) the measured pass took 2,286 s at
26,511 MiB per device. Deposited CA RMSD 4.41/4.35/3.59/4.05/4.68 Å against
4.17/4.14/3.67/3.86/4.54 unpadded — the same band; the two-process floor at
this size under bf16 diffusion without deterministic ops is 1.2–2.7 Å, so
same-index numbers are not read there. OpenFold3 serial on the same target
scores 2.17–2.38 and Boltz-2 2.53–3.07.

A rank that runs out of memory used to leave the other ranks waiting
forever at the NCCL clique rendezvous (XLA's terminate timeout defaults to
−1). `foldjax predict` and `cache warm` now compose
`--xla_gpu_nccl_termination_timeout_seconds=600` into `XLA_FLAGS` before
JAX is imported when `cp_devices` > 1 (`1f23f1b`; `FOLDJAX_CP_RENDEZVOUS_TIMEOUT`
overrides, a negative value declines); the bench harness does the same
(`5849258`, verified: the first row on it ended on its OOM within a minute
instead of at the time cap). A process that already has a backend can only
warn.

**Padding on the 256-token grid.** The old bucket ladder (2,048 → 3,072)
made a 2,096-token job pay +88–112% wall and +44–156% peak on three ports;
the ladder is now every 256 tokens to 8,192 (`7631016`), and CP-aware
targets round to the mesh (`37eeec8`). Same-snapshot cost at 2,096 → 2,304
(5DEI): OpenFold3 +30% wall, +12.7% peak (25,885 against 22,967
unchunked), coordinates 0.02 Å, deposited identical. Protenix and Boltz-2's
first padded rows looked better than that (−19% peak; +0.05 Å deposited on
every Boltz-2 sample) because `--padding` was silently capping the MSA
depth at the 1,024-row profile — the cap that failed its seed-block
admission at 4,096 — on Protenix, Boltz-2, OpenDDE (one of two places) and
AlphaFold 3, while OpenFold3 and ESMFold2 select their rows before padding
either way. `09b361a` pads the MSA axis to the bucket at or above the rows
the unpadded run would process; the profile depth is a floor, never a cap;
an explicit `msa` target below storage is refused naming `--max-msa-depth`
as the input-changing knob. Re-measured on that fix, Protenix 235.4 s / 28,202 MiB (+23% wall, +33%
peak: its MSA axis went 13,267 → 16,384 rows, the ladder's coarsest step,
and the MSA stack is its 2k peak) and Boltz-2 277.2 s / 21,715 MiB (+20%,
+17%: the token step plus the fixed 24× atom axis), both with coordinates
at the serial floor (0.013 and 0.033 Å) and deposited scores identical to
the exact runs. Padding stays opt-in — a compile-cache benefit is not worth
a tax on every single-shape run — and `foldjax cache warm` bakes the
bucketed executables for deployments that want them. A finer MSA ladder
above 8,192 rows would bound Protenix's padded cost to one step.


**Boltz-2 under the 2-D grid, second look (2026-09-16).** A matched-kernel
serial control (triangle, GLU and diffusion attention all on XLA, the family
every CP arm resolves to) runs 5DEI at 2,096 tokens in 554.9 s at 17,455 MiB
against the released kernels' 230.95 s / 18,511. Two readings follow. The CP
arms' wall penalty is mostly that resolution, not communication: the 2-D arm
at 973 s is 1.75× the like-for-like serial and 4.2× the released one, so
bringing the fused kernels inside `shard_map` is the next wall lever. And the
2-D memory win against the same family is only 5%. A CPU four-device probe
named the transition's pre-gate GLU (built unblocked under a mesh, 15 ×
f32[1, 1056, 1056, 1024] per device at 2,112 tokens where serial keeps 64
rows) and the trunk's row-only pair constraint; both were fixed (`93b3c52`
blocks local rows inside the shard, `d6ada16` shards the trunk's pair tensors
on both axes; serial program byte-identical, CP outputs bit-identical), and
the 2-D row moved 16,642 → 16,056 MiB, −3.5%, at unchanged wall and
deposited scores. The buffer the CPU probe ranked first was not in the card's
peak-live set — the arena-cover lesson again — so Boltz-2's 2-D peak owner
on the card is still unattributed and the next step is a GPU-side buffer
assignment, not another CPU probe.

### The ceiling wave and the fixes it forced (2026-09-16/17)

The question the whole feature exists to answer: does a four-card node run
what one 96 GiB card cannot? A 6,568-token case (6NYF's 821-residue chain
in eight copies, its own alignment reused) and the 3,012-residue 6ZTX put
every port against the wall.

| port | one card | four cards | per device | pass wall | verdict |
| --- | --- | --- | ---: | ---: | --- |
| Protenix, 6,568 | OOM (99.6 GiB) | 1-D completes | 44,432 MiB | 1 h 56 min | ceiling opened |
| OpenFold3, 6,568 | OOM | 1-D OOM (101 GiB on every rank); 2-D completes | 42,209 MiB | 2 h 56 min | ceiling opened on the grid |
| OpenDDE, 3,012 res | 180 GiB estimate | 2-D completes | 64,322 MiB | 1 h 14 min | ceiling opened; 6.5k needs 828 GiB and is out of reach |
| Boltz-2, 6,568 | OOM (130 GiB) | 2-D: one pass OOMed (46.9 GiB arena), the other ran 3 h 45 min without finishing | ~70 GB/card | > 4 h | not established |
| ESMFold2, 3,012 | OOM | 1-D warmup completes | – | > 1 h | opened; wall-bound |

OpenDDE's 3,012 residues score 0.51–0.56 Å against 6ZTX, the best any port
has on that target; Protenix's warmup pass had the first-invocation OOM
(41.5 GiB) that the second pass does not, the same pattern the 3,012-token
OpenFold3 serial run has always shown. The two ports that did not open
their ceiling were attributed, on the card and on CPU, and fixed:

**Boltz-2.** A compile-only attribution on the four cards (XLA's own
peak-live section, 2,096 tokens, 2×2) named the MSA stack, not the pair
stack: the f32 residual stream `f32[1, 8192, 1048, 64]` twice (2,096 MiB
each), the MSA transition's intermediates `bf16[8585216, 64]` eight times
co-live (8.4 GiB, half the peak), pair-weighted averaging's output once.
Two things followed from that. The MSA transition had lost its row block
under a mesh (`transition.py` zeroed the chunk for every rank-4 tensor,
though the MSA depth axis is replicated under 1-D and is exactly the axis a
serial block slices); `dff6709` keeps the block, and the same rule reached
the diffusion-conditioning and template pair transitions (`e0a9c4d`). And a
2×2 grid only halved that stack because it split the token axis alone:
`021b240` shards the alignment depth over the grid's column axis
(`P(None, cp_col, cp_row, None)`, a quarter of the stack per card), with
explicit bridges to the pair layout — pair-weighted averaging gathers its
small projected logits along the column axis and streams values around a
row-axis ring, the outer product mean streams its key operand and reduces
numerator and mask count along the column axis before taking the mean once.
Its reductions are chained through `optimization_barrier`, because XLA
otherwise merged 272 of them into two all-reduces whose results were 8.7
GiB co-live. On CPU the 2-D arena fell 29.5% at 2,112 tokens; the two
GPU rows on that code are recorded below when they land. The processed
depth itself is upstream's: `subsample_msa` is off by default there too.

That probe also found what CPU probes had missed twice, and one CPU probe
found what the GPU could not show: a real miscompile. Under the 1-D
layout the float32 outer product mean assembled its output with a chain of
`out.at[:, a:b].set(block)`; when the pairformer pinned that result on the
CP row axis one statement later, the SPMD partitioner (Shardy and GSPMD
alike, jax 0.11.1) returned the last row of every shard but the last
wrong — 37.9 on a scale of 74 at 848 tokens with the released chunk of 77,
irregular in token count and chunk, and flipping with layer count and
`lax.scan`. The released configuration sat in the clean column (four scanned
layers, and the bf16 path assembles by concatenation, which is the whole
f32/bf16 asymmetry), so no released run was wrong; `9df6826` assembles by
concatenation on both paths, bitwise equal in serial, and pins the 848/77
geometry.

**OpenFold3.** Its 1-D failure at 6,568 tokens is two phase-local sets per
rank, not one tenant: the transition phase held the widened SwiGLU as one
whole local tile (`f32[1115136, 512]` eighteen times, 20.6 GiB each at that
size, in both layouts) because `pair_block.py` disabled the row chunk under
a mesh — `5ec4a91` chunks local rows inside the shard, serial byte-identical
— and the triangle multiplication's 1-D full-width operand all-gather
(`f32[1, 128, N, N]` twelve times, 20.6 GiB) that the Cannon path replaces
with half-width tiles; that gather is the entire 1-D/2-D gap and is inherent
to 1-D (the pair bias's full width under 1-D is inherent too: axis −2 is the
query axis, refuted as a lever). The diffusion pair conditioning
(`f32[5, N/4, N, 128]`, 25.7 GiB) is the third; `67ddd48` resolves an
omitted `diffusion_chunk_size` to 1 under a mesh (5.1 GiB; chunk against
unchunked 1.6e-7 relative; the resolved width is in the compile profile).
With the grid completing at 6,568 tokens where 1-D and one card both die,
`368c190` makes `cp_layout=auto` the square grid for OpenFold3 too; the
2,096-token grid row measured 6,955 MiB per device against 13,990 serial
and 11,489 for 1-D (−50% / −39%; coordinates 0.022 Å from serial, deposited
identical).

**The wall.** A matched-kernel Boltz-2 serial control (triangle, GLU and
diffusion attention all on XLA, the family every CP arm resolves to) runs
5DEI at 2,096 tokens in 554.9 s at 17,455 MiB against the released kernels'
230.95 s / 18,511. Most of the CP arms' wall penalty is therefore that
resolution, not communication (the 2-D arm is 1.75× the like-for-like
serial, 4.2× the released one), and the fused kernels are the next lever.
`bee0cef` adds `--option triangle_attention_ring_kernel=tokamax` (Boltz-2,
Protenix): the ring's per-step tile — a local attention — runs tokamax's
fused attention with `normalize_output=False, return_residuals=True` and the
tiles combine by a softmax-statistics merge with Neumaier compensation on
numerator and denominator. It is a different program (one rotation, tiles
normalised against their own maximum; rows the model masks away come out as
zeros), opt-in, GPU-only, refused rather than downgraded; the default path is
byte-identical. Its GPU experiment is queued; nothing is measured yet.

**ESMFold2** gained the 2-D layout (`880a99d`: Cannon triangle multiplication,
pair transition blocked inside the shard; 2×2 and 3×3 within 2e-5 of serial
on CPU, serial and 1-D byte-identical). On the card its 1-D arm completes
3,012 tokens — the size one card cannot — but a pass exceeds an hour, and
4,100 tokens did not finish one pass in two hours: this port's ceiling is
wall, not memory, until the fused kernels return under a mesh.

**Admission** now covers every port but AlphaFold 3 (`bb4aae8`): OpenDDE
`4016 + 4.83e-3·S²` MiB on structural tokens for the bf16 trunk
(`1.18e-2·S²` fp32), ESMFold2 `5435 + 9.24e-3·N²`, both from the ledger's
rows with OOMs treated as censored; AlphaFold 3 warns `unknown` instead of
ignoring the flags.

### After the ceiling wave: the rows the fixes were waiting for (2026-09-17)

The fixes above were measured on the 2,096-token 5DEI rows as they landed,
one snapshot per commit, all on the 2×2 grid with the ring's query block at
128. Per-device peak, wall of the measured pass, and the same-index distance
from the released serial run (its rerun floor is 0.066 Å on Boltz-2):

| snapshot | port | what changed | per device | pass wall | vs serial |
| --- | --- | --- | ---: | ---: | --- |
| `7bb0838` | Boltz-2 | row-blocked ring | 16,642 MiB | 973 s | 0.013–0.112 Å, deposited identical |
| `d6ada16` | Boltz-2 | transition local rows, pair spec on both axes | 16,056 MiB | 971 s | 0.013–0.046 Å |
| `e0a9c4d` | Boltz-2 | MSA transition keeps its row block | 15,111 MiB | 975 s | 0.013–0.039 Å |
| `021b240` | Boltz-2 | MSA depth over the column axis | **8,296 MiB** | 943 s | 0.013–0.018 Å on four samples, 0.112 on the fifth |
| `2eb9e15` | OpenFold3 | first grid row | 6,955 MiB | 603 s | 0.021–0.023 Å |
| `67ddd48` | OpenFold3 | transition local rows, `diffusion_chunk_size=1` | 7,250 MiB | 607 s | 0.024–0.029 Å; deposited 0.51–0.56 against 0.52–0.57 |

Boltz-2's grid now sits at −55% of the released serial peak (18,511 MiB)
and −56% of its own 1-D arm (19,040), the same class of gain Protenix's
grid showed first; the sample that moves 0.11 Å is the same sample on every
2-D arm, and its deposited distance is unchanged. The OpenFold3 fixes were
costed at 6,568 tokens (20.6 GiB per SwiGLU copy there) and are neutral at
2,096, where the removed tile was not in the peak-live set and the
row-blocked restacking adds a small tenant. On the 6,568-token target the
post-fix warmup pass completed without the first-invocation OOM the
pre-fix row had, but serialising the five samples' diffusion pushed the
measured pass past its 4-hour cap; that row (6 h) and Boltz-2's 6,568-token
grid row (9 h) were cancelled on 2026-09-17 when the node was handed over
for external use, so **Boltz-2's four-card ceiling above 4,888 tokens and
OpenFold3's post-fix 6,568-token peak are still unmeasured.**

**ESMFold2 on the grid.** The 2-D layout is where this port's ceiling
actually opens: 3,012 tokens (6ZTX) completed both passes in 30 minutes,
772 s / 23,430 MiB per device, where the 1-D arm had not finished one pass
in two hours; 4,100 tokens (1GTE) 1,375 s / 37,380 MiB per device (its
warmup had the first-invocation OOM once). One card dies at 3,012.
`9fb9185` makes `cp_layout=auto` the grid for ESMFold2 on square counts.
The paper-artefact pass over the CIFs then raised a flag against that
default: on 6ZTX the 2-D row's best sample is 3.88 Å from the deposited
structure with all four chains near 4.0 Å, while the 1-D warmup structures
score 0.95 and 1.04 Å — and 6ZTX is the target on which every port's CP arm
has flipped a sample's basin (0.77–3.54 Å on OpenFold3). Whether that is
the port's sampling spread, the target's chaos, or a 2-D defect that the
CPU parity probes (2e-5 at small sizes) cannot see is decided by a
same-seed serial / 2-D / 1-D comparison. The CPU half of that ran on
2026-09-20 (four fake devices, 3DHA at 254 tokens, released schedule,
seed 101 — 127-row shards, so two 64-row blocks per shard and the ring's
row blocks engaged): against the deposited structure serial scores
0.77 / 0.87 / 0.87 / 0.60 / 0.83 Å, the grid 0.77 / 0.86 / 0.81 / 0.60 /
0.82, the row mesh 0.77 / 0.87 / 0.81 / 0.60 / 0.82; same-index the grid is
0.008–0.082 Å from serial, the row mesh 0.007–0.086, and the two CP arms
0.012 from each other (they share the XLA-kernel resolution that serial
does not). A serial run against itself is 0.0000, so the sharded programs
did run. A second geometry (4REK, 499 tokens: 250-row shards, four
64-row blocks per shard) reads the same way: the grid 0.018–0.043 Å from
serial, deposited 1.13–1.18 Å on both arms. No defect at either geometry;
the 6ZTX reading stays a GPU same-index question. On 1GTE (4,100 tokens, four chains of 1,005) the
grid's ESMFold2 scores 1.35–1.91 Å per chain, which also settles an older
note: OpenFold3's 13.7–22 Å per chain there is a misplaced C-terminal
domain (its first 600 residues sit at 0.7–1.1 Å, the last 400 at 12–19 Å
in 200-residue windows), a property of that port on that target, not of
the target.

**ESMFold2 on one card.** Its 3,012-token wall was attributed on the card
(compile-only, 2,096 tokens, released schedule; peak-live 41,733 MiB against
46,042 measured): the triangle prologue's packed projection at full width
(`bf16[N², 1024]` and its transposed copy, 8,580 MiB each, 41% of the peak),
the f32 layer norm of the language-model pair (4,290 MiB), six full pair
state copies (2,145 MiB each), and the alignment one-hot
(`bf16[1, N, 13280, 33]`, 1,752 MiB). `3531fb4`, `999d1e8` and `fb9328b`
block the prologue, the conditioning and transition layers, and the MSA
profile over rows (CPU parity worst case 0.0193 → 0.0228 Å against a 0.030
tolerance). On the card that landed 2,717 MiB: 43,325 against 46,042 at
2,096 tokens, and 3,012 tokens still asks for a single 82.45 GiB arena — the
trunk program's arena, not the `jit_dynamic_slice` executable the OOM names
(9,759 bytes per pair element against the admission law's 9,692). The
interval reading explains the shortfall: removing the 1024-wide class moves
the maximal co-live moment from 43,322 MiB to a 34,775 MiB moment owned by
the f32 `normalized` tensor, its transpose and the f32 language-model pair
(4,290 MiB each), and the blocked prologue itself added three full-width
concatenation destinations (3 × 2,145 MiB) — the third time this wave a
named removal moved the peak to the next co-live set instead of landing
whole. `1814601` streams the triangle so that only one full-width bf16
operand per direction remains and every other intermediate lives per
64-row block, and `a70f261` stores the normalised operand in bf16 past the
layer norm (f32 kept at the transition norm, the outer product mean, the
single norm and the Parcae injection). The census at 2,112 tokens goes from
29 full-width values (67,654 MiB) to 2 (4,356 MiB), with no f32 full-width
value left; bitwise on both fake meshes; parity worst case 0.0258 Å. Two
blocked changes have now used 0.0065 Å of the 0.0107 Å margin, so a third
recalibrates the tolerance first. Neither commit has run on a card yet.

**The fused ring, measured.** The experiment behind `bee0cef` ran once the
Pallas call inside the ring's `shard_map` was allowed (`ae896e1`,
`check_vma=False` on the fused body only): one Boltz-2 triangle-attention
layer's ring at 2,096 tokens on the 2×2 grid takes 146.7 ms median with the
tokamax tile against 281.1 ms on XLA (−48%; compile 1.7 s against 12.2 s),
and a cold five-sample prediction 786 s against 1,119 s (−30%). Rows with
valid keys differ by at most 0.125 on a range of 46 (mean 5.2e-6); rows the
model masks come out as zeros, as documented. Coordinates were not compared
by the experiment, so the option stays opt-in until a bench row with
`--option triangle_attention_ring_kernel=tokamax` is read against the
deposited structure.

**The miscompile census.** After `9df6826` every traced write on a sharded
axis was enumerated: 41 sites across the five ports and the shared CP
code, of which the outer product mean was the only member of the failing
class; 1,072 arms over two and four devices, both partitioners and both
layouts came back clean, with the fixed defect as a positive control firing
20 of 20. The grid is affected as well as the row mesh. The two Protenix
sites that looked similar are unreachable under a mesh.

Gates on `fb9328b`: 7,464 passed, 0 failed, CPU parity 59/59; on
`a70f261` the worker's full suite 7,458 passed and parity 59/59 with the
worktree on `PYTHONPATH` (the shared editable install otherwise measures
`main`).

**Still to run on the node** (in this order, once Slurm accepts the
account again): the ESMFold2 attribution re-read on `main` (one card,
45 min) and its 2,096 / 3,012-token serial rows, then the admission law
refit; the ESMFold2 same-seed serial / 1-D / 2-D comparison on 5DEI at
2,096 tokens and the 6ZTX 3,012-token 2-D against 1-D same-index (revert
`auto → 2d` if the grid sits above the 1-D floor); the two cancelled
6,568-token rows (OpenFold3 post-fix, 6 h; Boltz-2 grid, 9 h); one
deposited bench row on the fused ring.

### The node under a second account: the rows of 2026-09-21/22

Slurm came back for a different Unix account (the lab account, capped at
four cards and 128 GB of host memory per group, so rows ran one at a time);
the snapshots of this wave are `x34` (`10b3fd1`), `x35` (`742d158`),
`x36` (`7436f87`). The measured pass of every row below is warm and on
the released schedule.

**The fused ring tile, measured against the deposited structure.**
Boltz-2 5DEI at 2,096 tokens on the 2×2 grid with
`triangle_attention_ring_kernel=tokamax`: 617.9 s / 8,289 MiB per device
against the XLA grid's 943.4 s / 8,296 — the wall falls by a third at the
same peak, the deposited distances are identical (0.44 0.43 0.42 0.47
0.42), and same-index the fused arm sits 0.013–0.030 Å from the released
serial run on every sample, inside the 0.066 Å rerun floor where the XLA
grid's fifth sample sits at 0.112. The two diffusion-attention sites the
reviewer's spec put next — `cp_fused_attention=atom`, `token`, and both
(`301301a`, opt-in) — fire on the card and change nothing: 964 / 989 /
953 s against 943, and 611 s with the ring tile on top of both against
618 for the ring tile alone. On Boltz-2 at this size the triangle-attention
ring is where the grid's time was; the diffusion attention is not. The
option stays opt-in and documented as measured-neutral. Protenix on the
same grid with the fused tile: 339.4 s / 11,593 MiB per device against
559.9 s / 11,572 on XLA (−39%), deposited 0.46 0.45 0.52 0.51 0.46 against
serial's 0.46 0.52 0.44 0.49 0.47 — the same band — but same-index
0.11–0.25 Å from serial where the XLA grid sits at 0.006–0.014: this
port's triangle attention runs f32 on XLA and the tile is a bf16 kernel,
so on Protenix the fused ring is equivalent by the deposited instrument
only and stays opt-in; any default is Boltz-2's alone, and waits on its
6,568-token grid rows.

**OpenFold3 at 6,568 tokens, post-fix.** The grid row on `10b3fd1`
(transition local rows, `diffusion_chunk_size=1`, `auto → 2d`) completed
both passes without the first-invocation OOM the pre-fix row had, at
10,554 s / 42,230 MiB per device — the same wall and peak as the pre-fix
row (10,554 s / 42,209). The fixes are neutral at this size and are kept
for the OOM they removed; the four-card ceiling number stands.

**ESMFold2's compile.** The blocking commits of the previous wave
unrolled every 64-row block into the trace: at 2,096 tokens the trunk
program was 969,694 HLO lines, the GPU compile took 62 minutes and the
measured pass 805 s against the released 451 s. `2f5d825` rolls each
block loop into one traced body (`fori_loop` over the full blocks, the
tail traced once, writing into a buffer the caller lends — a fresh
destination inside a `while` is a colocated allocation XLA never reuses);
the 2,096-token 24-layer trunk goes from 442,494 to 32,914 StableHLO lines
and from 96 s to 3.8 s of CPU compile, and on the card the 2k serial row
is 452.6 s again with compile in minutes. Its memory price was two lent
full-width pairs per `folding_trunk` call — four calls, eight buffers —
which `7436f87` threads as one workspace through the recycle scan, the
coda and the confidence head (bitwise on the full program). Peak history
at 2,096 tokens, like for like (`peak_bytes`): released 46,042 MiB →
blocked 43,325 → streamed 34,702 → rolled 39,149 → threaded 39,290
(neutral at 2k, bitwise). At 3,012 tokens the arena request went 82.45 → 74.87 → 70.72
GiB across those steps; with ~13 GB of weights resident the pool needs
the request near 70 GiB, and the next lever is the three f32 full-width
pair values the attribution found at upstream's `z.float()` boundary
(4,290 MiB each at 2k, in flight). `confidence_dtype=bfloat16` was
re-measured at both sizes now that the peak sits in the confidence
phase: −1,203 MiB on the unrolled program, nothing on the rolled one, and
the identical 74.87 GiB request at 3k — it is not part of this story.

**ESMFold2 under the mesh, and the 3,012-token flag.** On `742d158`,
5DEI at 2,096 tokens: serial 452.6 s / 39,149 MiB; the grid 400.8 s /
17,337 MiB per device; the row mesh 701.6 s / 17,337 — the same
per-device peak under both layouts, because this port's CP peak is not
the pair stack. The two CP arms are 0.026–0.042 Å from each other on all
five samples and both sit 0.16–0.36 Å from serial on the three folded
samples (chains permuted; `tmp/esm-cpu-cp/same_index_perm.py`), 22–24 Å
on the two samples where every arm fails the homotetramer's assembly
differently. 6ZTX at 3,012 tokens: the grid 770 s / 22,252 MiB, the row
mesh 1,384 s / 23,071; deposited the grid read 4.11 30.15 4.00 4.19 3.88
Å against the row mesh's 14.50 32.46 1.04 0.95 29.95 — the flag raised by
the paper-artefact pass, reproduced to the hundredth across two
snapshots. The CPU diagnosis found no defect: all four of the port's
`shard_map` regions are clean at the 1,506-row tile (reference trunk
3.1e-6 of scale, `condition_pair` bitwise, prologue rows 0 ULP, RNG
bitwise; pinned by `106e8be` at the 2,096/3,012 geometries and
mutation-tested against four planted faults), and the two layouts
separate only in the language-model encoder's autocast stack, by 38–44
bf16 ULP, flat from 2,048 to 3,072 tokens. What decided it was a second
seed: at seed 202 the grid folds 6ZTX at 1.24 1.00 1.16 Å on three
samples (pTM 0.93) where the row mesh at seed 101 had two. The 4 Å was a
basin the grid's rounding tape reached on a chaotic target, not a fault
the grid introduces at that size; `auto → 2d` stays, and ESMFold2's CP
arms on 6ZTX are recorded across two seeds and two layouts.

**Boltz-2 at 6,568 tokens, completed.** The grid row on `10b3fd1`
(MSA depth over the column axis and every Boltz-2 grid fix above)
completed both passes: 17,930 s (4 h 59 min) / 50,904 MiB per device for
five structures, where one card asks 130 GiB and the earlier attempt
timed out at 4 h. With that, every port completes its largest case on
the four-card node: Protenix 6,568 tokens (1-D, 44.4 GiB), OpenFold3
6,568 (2-D, 42.2 GiB), Boltz-2 6,568 (2-D, 50.9 GiB), OpenDDE 3,012
residues (2-D, 64.3 GiB), ESMFold2 4,100 tokens (2-D, 37.4 GiB). The
fused-ring arm of the same row, and the per-family wall split at 2,096
and 6,568 tokens that says where Boltz-2's five hours go, are queued.

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
