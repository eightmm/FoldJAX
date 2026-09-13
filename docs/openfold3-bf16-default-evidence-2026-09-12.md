# OpenFold3 BF16 default evidence (2026-09-12)

FoldJAX defaults OpenFold3's partial token/pair track to `bfloat16`.
`--option dtype=float32` restores the publisher's FP32 inference precision.
This note records the measurement the default rests on, and what it does and
does not establish.

## Protocol and identity

The 28 timed rows used the same input, checkpoint, 10 recycles, 200 diffusion
steps, and five samples per seed across the two dtypes. The source SHA256 was
`85a1ee17a35adcea26fb064f87a8e51eea57c2f57251ee804afcbacca505b7ee`.
The panel covered 3OG2, 5DEI, and 6ZTX: seeds 101, 202, 303, and 404 at
1,003 and 2,096 tokens, plus 505 and 606 at 3,012. Every row completed with
finite outputs.

| target | tokens | warm FP32 wall | warm BF16 wall | FP32 peak | BF16 peak | CA deposited `fit_rmsd` |
| --- | --- | --- | --- | --- | --- | --- |
| 3OG2 | 1,003 | 98.51–98.84 s | 75.48–76.87 s | 9,272.4–9,273.9 MiB | 5,552.2–5,552.6 MiB | FP32 0.72–0.92 Å; BF16 0.70–0.92 Å |
| 5DEI | 2,096 | 402.34–405.06 s | 261.82–263.36 s | 25,066.2–25,066.8 MiB | 19,107.1–19,107.3 MiB | both dtypes 0.45–0.58 Å |
| 6ZTX | 3,012 | 948.89–958.24 s | 657.97–664.98 s | 50,411.5–50,413.8 MiB | 34,892.8–34,893.5 MiB | core 122–753, both dtypes 0.55–1.03 Å |

BF16 is faster and smaller at every size, and per-chain accuracy against the
deposited coordinates is the same on both arms at every size. That pair of
observations is what the default change rests on.

## The 3,012-token outlier belongs to the target, not to the dtype

6ZTX is catalase HPII, one sequence in four chains, so the scoring is
permutation-aware: a homotetramer scored chain-for-chain by label reads a
relabelling as a large uniform displacement. Superposing on the catalase core
(residues 122–753) and reading the N-terminal arm (27–121) separately, over
twelve arms of five samples each:

| arm | core 122–753 | arm 27–121, median displacement |
| --- | --- | --- |
| FP32 seeds 101, 303, 404, 505, 606 | 0.57–1.03 Å | 0.40–0.50 Å |
| **FP32 seed 202** | 0.68–0.71 Å | **56.41–56.66 Å** |
| BF16 seeds 101, 202, 303, 505, 606 | 0.55–0.95 Å | 0.39–0.48 Å |
| **BF16 seed 404** | 0.70–0.99 Å | **56.67–56.79 Å** |

The core folds correctly in all twelve arms. Each dtype misses the arm's basin
at one seed in six, and neither misses it at a seed the other one makes. The
miss is reproducible within a seed — all five samples of that seed land in the
alternative basin — and the arm's own pLDDT reports it locally, 83.5 against
90.8–92.8 elsewhere with no overlap, where the whole-complex mean barely moves
(92.0 → 89.2).

This retires the earlier reading, which measured the bfloat16 arm as a
same-index residual against a float32 arm at the same seed and found 25 Å at
seed 202. A residual between two arms cannot say which of them moved, and the
float32 arm's healthy within-set spread at that seed (0.569 Å) could not see a
basin the whole arm was sitting in.

## Limits

- Six seeds per arm at 3,012 tokens do not estimate a failure rate, and
  nothing here claims the two dtypes fail at equal rates. What the panel rules
  out is that the miss is a property of `bfloat16`.
- The mechanism is not measured. It is consistent with the per-recycle MSA
  resubsampling this port performs (`backends/openfold3.py:636` into
  `data/featurize.py:1183-1190`, 1,024 of 17,542 rows redrawn on each of ten
  recycles), which would make the arm's basin a draw the seed decides.
- The structure recheck is CA-only across 140 outputs and is a retrospective
  accuracy read, not an all-atom release gate. Values labelled `report_median`
  or `report_mean` elsewhere are per-atom displacement summaries, not fitted
  RMSD; only `fit_rmsd` is an RMSD measurement.
- The FoldJAX-versus-upstream benchmark matrix pins `dtype=float32` on this
  port (`bench/spec.py`), so those columns compare two implementations at one
  precision rather than two precisions.
