# FoldJAX closing plan, 2026-09-10 evening window

Scope: the remaining gaps after the master parity night
(`master-parity-summary-2026-09-09.md`) and the scale rows
(`scale-rows-master-2026-09-10.md`). Every goal below has an owner, its
inputs, the command or artifact that verifies it, and a **terminal state**:
the condition under which the goal is closed even when the number itself
cannot move. Three gaps cannot be closed by construction (Boltz-2 bitwise,
OpenDDE fp32 wall, chaotic-sample residuals); for those the terminal state
is a documented ceiling, not a smaller number.

Split: the parent (this session) owns every GPU submission, every merge,
docs synthesis, the ledger and the final CI. Workers run CPU-only in their
own worktrees under `foldjax-bench/wt/` and report a branch, never a merge.
Cancel jobs by explicit id only.

## Goals, weakest model first

### G1 ESMFold2 (open: 7ST3 chain B; 3k memory wall; no upstream column)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G1a chain B close | parent | job 964 (`esm-7st3-port-inj`, port on native LM injection) vs native-C/D captures | per-key relative diff of `injection.npz` / trunk / coda against the native pair; `bench/master_three_way.py` at the coordinates | **Closed 18:10 (job 964)**: coordinates inside native scatter on every sample (chain B max 0.85 vs 1.09 Å), coda at 1.1x native scatter, LM-origin offset 1.6e-2 through the loop; at-floor with the kernel-choice spread recorded. |
| G1b sequential samples (option merged `5645ec0`; L2000 rows 991-994 base/seq × 5/32 samples) | worker `esm-seq` → parent GPU rows | the diffusion token transformer's `[batch*S, L, L, heads]` f32 logits (2.9 GiB per transient at 3k/5 samples, 18.6 GiB at the released 32); the folding-trunk pair arena `bf16[L², 4c_z]` is sample-independent (`esmfold2-peak-is-one-arena`, post-`5fab8ae`) so no sample option moves the 3k trunk wall | CPU equivalence test (batched vs sequential coordinates bitwise or ≤1e-5 rel); GPU rows L2000 at 5 and 32 samples with `--option structure_sample_sequential=on` | option merged and its transient saving measured at 32 samples; the 3k wall is attributed to the trunk arena and its ceiling is read against upstream's own 3k outcome from G1c (upstream OOM too → documented ceiling; upstream fits → trunk chunking task opens, as OpenFold3's OPM did). |
| G1c upstream column (runner merged `4ed1994`; rows 995/996 submitted 18:40) | worker `esm-upstream` → parent rows | Biohub/transformers fork at `ef32577f55` as a real git checkout, `esmfold2-venv` | `bench.run_upstream --model esmfold2` on protein_1ubq (CPU smoke), then L1000/L2000 Slurm rows | ESMFold2 has upstream time/VRAM at 1k/2k and a tape-free structure comparison, or the runner is merged and the rows are recorded as OOM. |

### G2 OpenDDE (fp32 wall at 2k, no speed gain over upstream)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G2a bf16 rows | parent | jobs 974-984 (`--option dtype=bfloat16`, L1000/L2000/L3000 + 8 panel cases) | peak/time table; `bench.structures` bf16 vs fp32 vs upstream on the panel | bf16 rows tabulated; a documented decision (recommend / opt-in / reject) based on the cross-vs-within spread. No arena-reduction work: `opendde-arena-law-and-fp32-ceiling` already excludes every backend lever. |
| G2b spread doc | parent | 8-case tape-free spread already measured (cross at within levels, 3og2 case-specific) | section in `scale-rows-master-2026-09-10.md` | written. |

### G3 OpenFold3 / Protenix at 3k (6ztx: 20.7 Å monomer fold / 2.2 Å uniform)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G3a tape-pinned pairs (both first port replays OOM'd at 3k for harness reasons: OpenFold3 973 fused graph materialises the [L, L, c_msa²] outer product, 35 GiB → `--streamed` flag added f88c38e, resubmitted as 997; Protenix 971 passes native's per-cycle MSA rows as materialised features, 190 GiB → worker `pn-index-tape` switches the harness to the trunk's index tape, then resubmit) | parent | jobs 970→971 (Protenix native → port), 972→973 (OpenFold3 native cueq → port replay) | per-sample CA RMSD, three-way where a second native exists | residual at the 1k-2k pass/at-floor level → the tape-free gap is the MSA-row / sampler draw, documented, closed. Residual large on the same tape → bisect task opens (trunk boundary capture at 3k). |

### G4 Boltz-2 (best; only the reviewed-diff asterisk on the upstream column)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G4a pristine upstream (root ready, zero tracked diff; rows 989/990 submitted 18:27) | worker `boltz-pristine` (setup) → parent rows | `git worktree` of `boltz` at b1ebfc4 with only the cu13 preload shim, `.venv` reachable, `boltz.__file__` proof | 1k/2k upstream rows from the pristine root; times within the reviewed-diff rows' scatter | upstream column rows carry no diff sha, or the shim itself is recorded as the only tracked change with its sha. |

### G5 Mixed-entity set (5 real complexes, 1.1k-4.8k tokens)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G5a FoldJAX rows | parent | jobs 934-963 | 30 rows in `results/`, OOM rows recorded from the slurm log | table in `scale-rows-master-2026-09-10.md`. |
| G5b upstream rows | parent (`mixed_upstream_submitter.sh`) | rows 965-968, 985, and the rest as FoldJAX rows land | 20 upstream rows or OOM | same table, upstream columns filled. |
| G5c structures | parent | `compare/` pairs | `bench.structures --markdown` (permutation-aware) | cross-vs-within per case in the doc. |
| G5d tooling | workers `scale-table`, `mixed-cases` | `results/*.json`, `logs/*.slurm`, `compare-structures.json`; the afternoon's scratch fetchers | `tests/test_bench_scale_table.py`, `tests/test_bench_mixed_cases.py` | the four hand-rolled tables come from one command; the next mixed set is one command. |

### G6 Close-out

Full CI on main (`ZLIB_ROOT=$HOME/.local CMAKE_PREFIX_PATH=$HOME/.local
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q -m 'not network'
--cov=foldjax --cov-fail-under=80 tests`), `ruff check .`, `uv lock --check`,
ledger rows copied from the snapshot, memory notes, push to origin main,
final status mapping each goal to its terminal state (a goal still gated on
a job names the job and what each outcome means).

## Queue policy for the window

Diagnostics (964, 970-973) gate decisions and run before collection: the
3k-5k mixed rows and the OpenDDE bf16 rows are held (`scontrol hold`) until
the diagnostics are running, then released. New rows stay at 48 GB host
memory; only the OpenFold3/Protenix 4k-5k upstream rows and the 3k tape
pairs need 96 GB (three cards, not four).

## Worker briefs (common constraints)

- Worktree under `foldjax-bench/wt/<name>` from main, CPU only
  (`JAX_PLATFORMS=cpu`), never touch `src/` on main, never submit Slurm.
- Verification is the tests you add plus the modules you touch. Never the
  full suite; the parent runs it once at the end.
- Report: branch name, files, test command and its output, what was not
  done. No commit trailers.
