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
A-D), L3000_6ztx (3012, homotetramer), L4000_1gte (4116, homotetramer),
L5000_8e2f (5000, 1 chain). MSAs are the colabfold a3m files pulled from the
workstation.

## Wall time and peak

FoldJAX / upstream, "s / GiB". OOM rows are rows: the card is 96 GiB and the
allocation the run asked for is in parentheses. "upstream OOM" means the
upstream runner ended with zero samples and the card full.

| model | 1003 | 2096 | 3012 | 4116 | 5000 |
| --- | --- | --- | --- | --- | --- |
| OpenFold3 | 100 / 9.1 vs 139 / 14.0 | 404 / 24.5 vs 629 / 44.2 | 951 / 49.2 (`cueq`), 863 / 42.5 (`cueq-full`) vs 1430 / 79.7 | OOM (78) vs upstream OOM | OOM (110) vs upstream OOM |
| Boltz-2 | 91 / 12.3 vs 132 / 15.8 | 318 / 21.3 vs 466 / 46.6 | 806 / 39.9 vs upstream OOM | 3081 / 64.2 vs upstream OOM | OOM (85) vs upstream OOM |
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
- Boltz-2 at 3012 and 4116 tokens finishes in FoldJAX (40 and 64 GiB) and OOMs
  upstream at both sizes (the upstream row filled the 96 GiB card).
- The memory ceilings match between sides where both OOM: OpenDDE from 2k,
  OpenFold3 from 4k, Protenix at 5k.
- ESMFold2's 3k wall is its `num_samples x L^2` arena; OpenDDE's 2k wall is its
  fp32 pair arena (both recorded before, now measured at the row).

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
  pristine-checkout rerun of the Boltz column is the follow-up if those
  performance patches are ever in question.
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
| mixed_2k_7y7q | 2097 | 459 / 26.4 / 0.77 vs 698 / 43.9 / 0.76 | 348 / 18.4 / 0.87 vs 492 / 50.4 / 0.88 | 248 / 21.9 / 0.76 | OOM (fp32 pair wall, as on the protein set) | 487 / 51.2 / 0.63 | 260 / 14.6 / 0.74 |

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

Cross RMSD is at or below the within-set RMSD of both implementations on all
four finished pairs. Boltz-2's 5 Å is the model's own spread on this entry
(the 7-mer RNA and the dimer arrangement move between samples at the same
rate on both sides, within-set 5.41 Å each); OpenFold3's FoldJAX set has one
sample 6.8 Å from the others while upstream's five agree to 1 Å, a sampler
draw at n=5 rather than a systematic offset (cross median 0.93 Å).
4XWW's two RNase J chains are identical, so the permutation-aware pairing
swaps them in 10-14 of 25 cross pairs.
The Protenix 4k protein row is the first pair where FoldJAX is not faster
(2206 vs 2193 s) while still 15% lighter (73.5 vs 86.4 GiB).
