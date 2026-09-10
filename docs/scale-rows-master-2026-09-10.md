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
| OpenFold3 | 100 / 9.1 vs 139 / 14.0 | 404 / 24.5 vs 629 / 44.2 | 951 / 49.2 (`cueq`), 863 / 42.5 (`cueq-full`) vs (running) | OOM (78) vs upstream OOM | OOM (110) vs upstream OOM |
| Boltz-2 | 91 / 12.3 vs 132 / 15.8 | 318 / 21.3 vs 466 / 46.6 | 806 / 39.9 vs (running) | 3081 / 64.2 vs upstream OOM | OOM (85) vs (running) |
| Protenix | 65 / 6.7 vs (running) | 210 / 22.9 vs 242 / 40.8 | 579 / 41.2 vs (running) | 2206 / 73.5 vs (running) | OOM (94) vs upstream OOM |
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
  Protenix 13% at 2k, OpenDDE 5% at 1k) and uses less peak memory on every
  pair (OpenFold3 35-45% less, Boltz-2 22-54% less, Protenix 44% less at 2k,
  OpenDDE 28% less at 1k).
- Boltz-2 at 4116 tokens finishes in FoldJAX (64 GiB) and OOMs upstream.
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

| model | case | cross TM | within FoldJAX | within upstream | cross RMSD Å |
| --- | --- | --- | --- | --- | --- |
| Protenix | L2000_5dei | 1.000 (1.000-1.000) | 1.000 | 1.000 | 0.19 (0.13-0.24) |
| Boltz-2 | L2000_5dei | 1.000 (0.999-1.000) | 1.000 (0.999-1.000) | 1.000 | 0.28 (0.21-0.35) |
| Boltz-2 | L1000_3og2 | 0.992 (0.988-0.997) | 0.992 (0.987-0.994) | 0.993 (0.991-0.997) | 1.54 (0.66-2.86) |
| OpenDDE | L1000_3og2 | 0.993 (0.984-0.999) | 0.992 (0.984-0.999) | 0.999 (0.999-1.000) | 2.33 (0.34-4.77) |
| OpenFold3 | L1000_3og2 | 0.987 (0.984-0.998) | 0.988 (0.985-0.999) | 0.995 (0.984-0.997) | 3.48 (0.43-5.79) |
| OpenFold3 | L2000_5dei | 0.569 (0.569-0.569) | 1.000 | 1.000 | 37.4 |

- Protenix and Boltz-2 at 2k: cross equals within on both sides; the two
  implementations are as close as the sampler allows (0.2-0.3 Å).
- Boltz-2 and OpenFold3 at 1k: cross equals within-FoldJAX; both sides have
  sampling spread on 3og2 (TM 0.984-0.999) and the cross distribution sits
  inside it.
- OpenDDE at 1k: cross equals within-FoldJAX, but within-upstream is tighter
  (0.999-1.000 against 0.984-0.999). FoldJAX OpenDDE draws a wider sample
  distribution on this case than upstream. The tape-pinned panel shows the
  arithmetic matches to 0.002-0.016 Å, so this is the ordinary-RNG sampler
  path, not the trunk; open item, one case so far.
- OpenFold3 at 2k (37 Å, TM 0.569 on a homotetramer while both sides are
  internally identical) is chain assignment, not structure: `bench.structures`
  pairs residues by chain id and the two implementations label the four
  identical chains in a different order. The same shows in Boltz-2's own
  within-set spread at 4k (0.59-1.00). Multi-chain cases need a
  permutation-aware alignment before their cross numbers mean anything; the
  1k single-chain rows and the 2k Protenix/Boltz-2 rows (where the orders
  happened to agree) are the ones to read.

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
