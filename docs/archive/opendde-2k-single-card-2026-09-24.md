# OpenDDE at 2,096 residues on one card

L2000_5dei (4 x 524 residues, 4,040 structural tokens) at the released
defaults -- bf16 trunk, 5 samples, 200 steps, 10 recycles, seed 101 -- on one
96 GiB RTX PRO 6000 Blackwell (CC 12.0). Main `d57e1bb` OOMs there asking for
an 84.98 GiB arena on an 85.5 GiB pool. Every number below comes from
`foldjax-bench/opendde-2k-20260924/` (probe, snapshots, Slurm logs, per-run
JSON); each job ran a `git archive` snapshot of the commit it names.

## Why a compile-only probe read 10 GiB less than the failing run

XLA's GPU compiler plans against the client's pool. `foldjax predict` sets
`XLA_CLIENT_MEM_FRACTION=0.9` (`cli._apply_mem_fraction`); `foldjax.api` does
not, so a probe driving the API compiled at JAX's default 0.75. Same code, same
case, compile only (gap job 2339):

| pool fraction | preallocate | remat limit | temp arena |
| --- | --- | --- | --- |
| 0.9 | on | 81.20 GiB | 91,251,297,264 B = 84.98 GiB |
| 0.9 | off | 81.20 GiB | 84.98 GiB (same to 1 KB) |
| 0.75 | off | 67.67 GiB | 78,800,196,336 B = 73.39 GiB |

The 0.9 arena is the failing row's allocation to 14 KB. Preallocation does
not move it. A compile-only attribution of the shipped program has to set
the CLI's fraction.

## The peak on main, at the shipped 0.9

Distinct-buffer occupancy (aliases counted once) of main's program: 75,500
MiB at the structural refiner's blocked triangle multiplication, six bf16
`[4040, 4040, 384]` tensors (11.67 GiB each) plus the residue pair tensor
(3.1 GiB). Two of the six exist only because of how the loop was written:

* `pad(z_norm)`: the row axis padded to whole 43-row blocks (4040 % 43 = 41).
* the zero destination: XLA merged the outgoing and incoming calls' identical
  constant inits into one buffer; the first loop ran on a copy while the
  original stayed live across it as the second loop's init.

The rest: the carry destination, `z_norm`, `b` transposed channel-major and
the refiner's pair tensor. Next ceiling: 65,123 MiB at the cuEq triangle
attention (q, k, v transposed to `[N, heads, N, d]`, the kernel output, the
pair tensor, the residue pair tensor, LSE/max). Packing loss above the worst
moment: 11.5 GiB.

## What shipped: `blocked_form="overlap"` (822083f)

The refiner's multiplication starts a ragged last block at `n - chunk`
instead of padding, and derives its destination init from the call's own
input. Opt-in; only OpenDDE's structural refiner takes it; Protenix keeps the
padded program (a shared default moved Protenix's CPU replay parity, 1UBQ
sample 4, past its tolerance).

* GPU, deterministic ops, L1000_3og2 (1,902 structural tokens, ragged for
  64-row blocks): main and 822083f wrote byte-identical coordinates for all 5
  samples (job 2372).
* L2000_5dei, shipped CLI, 0.9 pool preallocated, `memory_check=warn` (job
  2367): completes. `peak_bytes_in_use` 78,607.5 MiB cold / 78,599.4 warm;
  1,037.9 s cold / 687.2 s warm; CA RMSD to 5DEI 0.586-0.746 A, the 2x2
  context-parallel run's range; cold vs warm 0.004-0.006 A.
* L1000_3og2: peak 19,342.7 MiB (campaign row on main 21,497.8), warm 151.4 s
  (149.0), CA RMSD to 3OG2 0.725-1.018 A (unchanged).

Admission is still the fitted law's (`OPENDDE_BF16_PEAK`, 80.9 GiB predicted
at 4,040) and refuses. Against the measured peak the margin is small: the
threshold is 0.9 of the 87,528 MiB pool, 78,775 MiB, and the measured peak is
78,607.5 MiB -- 168 MiB below it before any allowance. Whether a law refit on
measured peaks admits it depends on the allowance the refit derives.

## Levers measured and not shipped

Both were opt-in and can return as memory levers. Neither is in this branch's
history: the measured commits (`2ae00ee`, `9d23c1a`, on `d57e1bb`) survive as
whole-tree snapshots in `foldjax-bench/opendde-2k-20260924/probe/code-2ae00ee`
and `code-9d23c1a`. The SHAs in this note are those pre-rebase commits; every
measurement here was taken on a `d57e1bb`-based snapshot, before main's bf16
cuEq attention padding (`a176340`).

**Row-blocked cuEq triangle attention in the refiner (2ae00ee,
`triangle_attention(row_blocks=True)`).** The shared bias is built once; each
block of rows is normalised, projected, attended, gated and projected out on
its own. L2000_5dei: arena 61.65 GiB compile-only (packing loss fell to 0.2
MiB), measured peak 66,525.3 MiB, 689.8 s warm, CA RMSD to 5DEI unchanged
(0.586-0.746 A), 0.007 A from 822083f. Not shipped: the projections become
`rows * N` GEMMs, and at L1000_3og2 sample 0 moved 0.026-0.042 A against four
main runs whose own pairwise draws of that sample stay at or below 0.027 A
(0.0126 A among main runs alone), and 0.051 A under deterministic ops (jobs
2382, 2385). Samples 1-4 stay inside the floor; deposited accuracy is
unchanged to 0.001 A.

**Streamed multiplication (9d23c1a, `blocked_form="stream"`).** `b`, each
`a` block and the epilogue computed one block of rows at a time, so `z_norm`
is never whole. On top of 2ae00ee it cut peak occupancy from 63.1 to 51.7
GiB, but the arena stayed at 61.65 GiB: XLA's heap placement spread the four
co-live pair tensors over five slot offsets. No arena gain at 4,040.

## Tools

`foldjax-bench/opendde-2k-20260924/probe/attribute.py` compiles the shipped
program through the API with the recorder in place of `model._infer_pool`
(set `PROBE_MEM_FRACTION`, default 0.9 in `probe.sbatch`) and reports the
arena, the distinct-buffer occupancy with its separated maxima (the
replacement ladder), and each tenant's source line and stage.

## Admission after the rebase onto the aligned-attention main (2026-09-24)

Peaks of the shipped command on main 77e89e9 + this change (cold/warm, 0.9 pool,
released defaults, one 95.6 GiB card): 489 st 3,003-3,007 MiB; 945 st 6,027;
1,902 st 19,208-19,210; 2,620 st (L1350_3lxu) 34,811-34,815; 4,040 st (two jobs)
78,588-78,616. The 4,040-token runs complete with `--memory-check=warn`.

A refit of `OPENDDE_BF16_PEAK` on these points does not admit 4,040 under the
repository's allowance rule with any basis tried (`1+n2` over by 478 MiB,
`1+n+n2` by 74 MiB): the allowance is the largest in-sample underestimate over
all sizes (196 MiB, at 2,620) and the measured 4,040 peak sits only 159 MiB
under the 78,775 MiB threshold. Shipping that refit would break
`test_no_measured_completed_run_is_refused`, so the law was left as it was
(fitted on the earlier program; it overestimates this one at 1,902 by about
2.3 GiB and refuses 4,040 as before). Admitting 2,096 residues at released
defaults needs a policy change -- a per-size allowance for measured sizes, or a
further memory lever such as the row-blocked attention above -- not a refit.
