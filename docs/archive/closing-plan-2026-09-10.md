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
| G1b sequential samples — **closed (option not recommended)**: L2000 5 samples 512 vs 440 s at the same 45 GiB peak; 32 samples 1088 vs 591 s at 44.9 vs 45.2 GiB — the trunk pair arena sets the peak at every sample count, so the option cannot lift the 3k wall; upstream itself OOMs at 2k (G1c). (option merged `5645ec0`; base rows: L2000 5 samples 440 s / 45.0 GiB, 32 samples 591 s / 45.2 GiB; the seq rows 992/994 failed on the bench's string-valued `--option` (fixed d882dd5) and rerun as 1043/1044) | worker `esm-seq` → parent GPU rows | the diffusion token transformer's `[batch*S, L, L, heads]` f32 logits (2.9 GiB per transient at 3k/5 samples, 18.6 GiB at the released 32); the folding-trunk pair arena `bf16[L², 4c_z]` is sample-independent (`esmfold2-peak-is-one-arena`, post-`5fab8ae`) so no sample option moves the 3k trunk wall | CPU equivalence test (batched vs sequential coordinates bitwise or ≤1e-5 rel); GPU rows L2000 at 5 and 32 samples with `--option structure_sample_sequential=on` | option merged and its transient saving measured at 32 samples; the 3k wall is attributed to the trunk arena and its ceiling is read against upstream's own 3k outcome from G1c (upstream OOM too → documented ceiling; upstream fits → trunk chunking task opens, as OpenFold3's OPM did). |
| G1c upstream column — **done**: 1k 199 s / 67.6 GiB vs FoldJAX 155 / 14.4; 2k upstream OOM vs FoldJAX 451 / 45.0 (runner merged `4ed1994`; rows 995/996) | worker `esm-upstream` → parent rows | Biohub/transformers fork at `ef32577f55` as a real git checkout, `esmfold2-venv` | `bench.run_upstream --model esmfold2` on protein_1ubq (CPU smoke), then L1000/L2000 Slurm rows | ESMFold2 has upstream time/VRAM at 1k/2k and a tape-free structure comparison, or the runner is merged and the rows are recorded as OOM. |

### G2 OpenDDE (fp32 wall at 2k, no speed gain over upstream)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G2a bf16 rows — **closed 22:20** | parent | jobs 974-984 | 1k 149 s / 21.0 GiB vs fp32 235 / 41.3; 2k and 3k still OOM; panel accuracy: bf16 within = fp32 within on all 7 cases, cross vs upstream at the within level, fp32-vs-bf16 closest pairs 0.01-0.16 Å | Decision: fp32 stays the released default (parity arm); `dtype=bfloat16` documented as the recommended opt-in for memory/time (no measurable accuracy cost on the panel, 37% faster, half the peak at 1k, ceiling unchanged at 2k). |
| G2b spread doc | parent | 8-case tape-free spread already measured (cross at within levels, 3og2 case-specific) | section in `scale-rows-master-2026-09-10.md` | written. |

### G3 OpenFold3 / Protenix at 3k (6ztx: 20.7 Å monomer fold / 2.2 Å uniform)

| package | owner | inputs | verify | terminal state |
| --- | --- | --- | --- | --- |
| G3a tape-pinned pairs — **both closed 2026-09-10 evening** (Protenix: port 0.05-0.40 Å inside native's 0.64-3.32 Å process floor; OpenFold3: four samples at native's 0.03 Å floor, sample 4 a bistable-loop basin choice; details in X1) | parent | jobs 970→971 (Protenix native → port), 972→973 (OpenFold3 native cueq → port replay) | per-sample CA RMSD, three-way where a second native exists | residual at the 1k-2k pass/at-floor level → the tape-free gap is the MSA-row / sampler draw, documented, closed. Residual large on the same tape → bisect task opens (trunk boundary capture at 3k). |

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

## 12-hour extension (2026-09-10 19:40 → 2026-09-11 07:30)

Ultracode window. Anchored on the two consistency items that are still open
and on the levers that keep every closed item closed. Each package has a
terminal state; nothing below reopens an at-floor verdict.

| package | owner | terminal state |
| --- | --- | --- |
| X1 3k offsets (OpenFold3 6ztx 24 Å uniform, Protenix 6ztx 2.4 Å uniform) | parent (GPU 1002/1005) + workflow readers (CPU) | OpenFold3 input audit at 3012 (CPU, port featurizer vs native input.npz, 35 leaves): everything equal except ref_pos (independent RDKit-conformer/augmentation RNG on the two sides, same magnitude at the 1k control where folds agree) and a 1-ulp deletion_value rounding; MSA rows 16366/16366 identical as a multiset and in order. Config diff done on CPU and adversarially checked by two refuters (source lens, capture lens): the coordinate path carries no size-gated input or arithmetic knob at 3012 on either model, so the tape-pinned replay was the right discriminator; the refuters did find Protenix's token-gated confidence-head/diffusion AMP switch (now X7). Neither upstream carries a size-gated *input* knob at 3012 (Protenix `chunk_size_thresholds` and `msa_chunk_size` are memory-only, input audit at 3k passes leaf-by-leaf; OpenFold3 `offload_inference.token_cutoff 2800` and `per_sample_token_cutoff 750` are memory/batching, native subsamples 1024 rows per cycle like the port). The tape-pinned replay is therefore the necessary test. **Protenix landed (job 1002, index tape + preallocated pool, 45 min / 24 GB output)**: port vs native-A on the shared tape, per-chain RMSD Å by sample 1-5 — A 0.08/0.40/0.08/0.05/0.10, B 0.11/0.41/0.10/0.07/0.10, C 0.08/0.39/0.11/0.06/0.10, D 0.09/0.40/0.09/0.07/0.11. The tape-free 2.4 Å uniform offset was the draw (MSA rows, dropout masks, noise), not arithmetic; four samples sit in the pass/deferred band and sample 2 at 0.40 Å on every chain; native-B queued to read sample 2 against native's own 3k scatter. **OpenFold3 landed (job 1005, streamed graph + CLI host feature chain + preallocated pool, 694 s)**: port vs native-cueq on the shared tape, per-chain RMSD Å by sample 1-5 — A 0.032/0.038/0.031/0.41/0.030, B 0.031/0.037/0.032/0.41/0.031, C 0.031/0.037/0.033/0.41/0.031, D 0.032/0.037/0.032/0.41/0.031; pTM/ipTM max abs error 0.0009. The tape-free 24 Å monomer-fold difference was the draw (MSA subsample, ref-conformer RNG, noise): four samples in the pass band, sample 4 at 0.41 Å on every chain, the same shape as Protenix's sample 2. Both 3k offsets are closed as draws. OpenFold3 native A-vs-B at 3k (job 1008): 0.026-0.034 Å on four samples, 0.10 Å on sample 2 → the port is at native scatter on samples 1/2/3/5 and 12× above it on sample 4 (0.41 vs 0.033); a second port draw (job 1009, autotune unfrozen) keeps sample 4 at 0.41 Å (stable route difference, not a near-tie) and moves sample 2 to 0.29 Å (kernel-choice sensitivity, native 0.10 there); pLDDT differs by 20 points in both draws vs 1.3 between natives. OpenFold3 3k closed: sample 4's 0.41 Å is one loop (residues 15-27, all chains, 97% of the deviation) sitting in the other of two wells that native's own five samples populate ({1,2,5} vs {3,4}); both port draws agree (0.026 Å), the +20 pLDDT on PRO15-HIS17 matches native's {1,2,5} confidence for that well, and the sample is ranked last on both sides. Basin choice on a bistable loop → at-floor (worker of3-3k-sample4 report in the scratchpad). Protenix native-B (1007) landed: native A-vs-B on the same tape differ 0.64-3.32 Å after permutation-aware chain alignment (they relabel the identical chains on every pair), so the port's 0.05-0.40 Å sits inside native's own process floor on every sample — Protenix 3k closed at-floor (port closer to native-A than native to itself). Large → bisect from the captured trunk. Two more OOMs → one lever each (packed dropout tape without `--include-trunk`; template-collapsed streamed replay) then the ceiling is recorded. |
| X2 unified `deterministic` execution knob (merged 8451ffe 21:40: shared `foldjax.models._compile_policy`, six ports, 49 files, integration suite 3726 passed; adversarial review workflow running; GPU pairs 1014-1037 queued: per port at 1k, on×2 and off×2 without autotune freeze) | workflow (CPU implement, worktrees) → parent GPU verify | **GPU verdict (rows 1014-1037)**: on-pairs bitwise for Protenix, OpenFold3, OpenDDE, ESMFold2, AlphaFold 3; Boltz-2's on-pair still differs (kernels outside the flag; frozen autotune stays its route — documented ceiling). Costs +4…+45% (ESMFold2 highest), AlphaFold 3 −6%. Review workflow: no confirmed blocker; one reproduced major (ESMFold2 LM-embedding off-key spelled unconditionally) fixed in d882dd5; minors noted (Protenix late eager refusal, annotations). |
| X3 parity regression suite on CPU | workflow | The gated upstream-parity suites (never collected in CI) get a CPU-runnable subset driven by small stored captures, so a rounding-route regression fails CI instead of the next GPU night. |
| X4 device-argument compaction for the 4k wall (merged 23cd07e/b7b3473 22:35: Protenix zero-template-geometry marker with in-graph rebuild, bit-exact at the embedder under f32/bf16/CP; Boltz-2 uint8 categorical storage for contact_conditioning, type/token bonds, ref_element, ref_atom_name_chars, 251 MiB at 3k / 446 MiB at 4.1k, byte-identical output on CPU; GPU rows 1038-1041 queued: Protenix 3k/4k/5k and Boltz-2 4k label `compact`) | workflow inventory (CPU, done 20:20) → implement (workflow, worktrees) → parent GPU rows | Inventory at 4100 tokens (CPU hooks on the real backend handoff): arguments are 0.48 GiB for OpenFold3 (0.6% of its 78 GiB OOM), 0.80 GiB for Boltz-2 (1.2% of 64 GiB), 6.08 GiB for Protenix (8.3% of 73.5 GiB). The 4k wall is temporaries; the argument side cannot close OpenFold3's. One lever worth a commit: Protenix ships 5.5 GiB of all-zero template geometry (7.8 GiB at 4888 tokens) that a bit-exact in-graph-zeros flag removes; Boltz-2's pair one-hots (contact_conditioning int32, type/token bonds) are 0.44 GiB. Both are being implemented bit-exact with CPU equality tests; the GPU rows then measure the landed peak. **Measured**: Protenix 3k 42.1 → 37.5 GiB, 4.1k 70.6 → 65.1 GiB (the argument bytes, 1:1), wall unchanged; 5k still OOM on the same 89.4 GiB temporary → the 5k ceiling is a temporary and is recorded as such. Boltz-2 4.1k compact: 64.4 vs 64.2 GiB — no gain (the rebuilt one-hot is materialised on GPU); output-neutral, kept, no memory claim. |
| X7 Protenix token-gated AMP policy (found by the refute stage; merged 5b40b5d 21:10 — `--amp-policy auto|fp32|bf16`, 39 CPU tests incl. realised-dtype hooks and a two-snapshot bitwise check at ≤2560; GPU rows 1010-1012 (3k/4k/5k `amp-auto`) and the 3k tape replay 1013 queued) | worker `x7-protenix-amp` (CPU) → parent GPU rows | Upstream `runner/inference.py:492-522 update_inference_configs` runs the confidence head under bf16 autocast above 2560 tokens and the diffusion sampler under bf16 above 3840 (captures: `skip_amp.confidence_head` false at 3012, true at 76); the port ran both in fp32 at every size. Terminal (met): the port resolves the same policy from n_token by default (`--amp-policy auto|fp32|bf16`), ≤2560 stays bitwise, the realised dtype is tested on CPU and recorded in provenance; GPU: 4.1k 2206 → 2106 s and 73.5 → 70.6 GiB (bf16 diffusion above 3,840); 3k unchanged in time; **open**: the 3k tape replay under `auto` (job 1013) differs from the fp32 replay by 0.4-2.1 Å per sample after chain-permutation alignment although the diffusion is fp32 at 3,012 tokens — the probe with `--amp-policy fp32` on the same code (job 1042) also differs from the pre-X7 replay by 0.4-2.1 Å with the closest-sample mapping scrambled — but both replays ran without deterministic ops, and Protenix's own two-process spread without the knob is up to 2.3 Å per atom at 1k, so the probe cannot separate a code change from process scatter. Three deterministic replays (`--deterministic-ops on`) are queued: pre-X7 code (1048), X7 code with fp32 policy (1049), X7 code with auto (1050); E == F bitwise settles that the fp32 path is intact, and F vs G isolates what the head autocast touches. **Settled (03:30)**: E == F bitwise on all 5 × 23,768 × 3 coordinates — X7 leaves the fp32 coordinate path intact, and the earlier 0.4-2.1 Å deltas were the port's own process scatter without deterministic ops (lesson recorded in memory: a same-tape port A/B needs `deterministic=on`). The released default stays `auto`; G (auto, deterministic on, job 1050) == F bitwise on every coordinate; the bf16 confidence head moves atom pLDDT by at most 0.0099 (on the 0-1 scale, about one point), chain pTM/ipTM by ≤1.9e-4 and PAE means by ≤0.005 — the released `auto` default is verified: upstream's policy, coordinates untouched, confidence within its own rounding. One more reading from the same runs: the deterministic port (E, F) sits 5.1-6.8 Å (permutation-aware) from native-A and from the non-deterministic port replay that had matched native-A at 0.05-0.40 Å — `deterministic=on` also routes every Triton GEMM to cuBLAS, so it is a different rounding route, and at this 3k case it lands 5.1-6.8 Å from native-A and 5.1-6.8 Å from native-B alike (the two natives are 0.76-0.91 Å apart on their closest samples, permutation-aware), i.e. outside native's own floor — the option is for repeatability, not a parity arm, and every panel number was read with it off. Whether the deterministic route's 3k conformer is a worse structure is not measured (next step: TM/pLDDT of E against the deposited 6ZTX beside native's). |
| X8 Boltz-2 upstream MSA-deletion regression (found by the X4 inventory, verified by `boltz2-deletion-check`) | worker (CPU opt-in option) → parent GPU A/B | Upstream v2.2.0+ zeroes has_deletion/deletion_value/deletion_mean for every real MSA (commit 04d27c71); the port reproduces it bitwise. Terminal: documented (`boltz2-upstream-msa-deletion-regression-2026-09-10.md`); default stays upstream-faithful; opt-in `msa_deletions=restored` with a guard test; one GPU A/B at 1k (restored vs released) with TM to the deposited structure recorded; regression reported upstream. |
| X5 collection | parent | mixed 4k/5k rows, OpenDDE bf16, ESMFold2 seq/upstream, Boltz-2 pristine 2k; tables and structures updated. |
| X6 close-out (CI on 5e40334 at 02:40: 6546 passed / 0 failed / 421 skipped, coverage 87.76%; ruff clean; `uv lock --check` clean; ledgers merged (139 rows); memory notes written; the three deterministic 3k replays 1048-1050 decide X7's default before the final status) | parent | full CI, ruff, `uv lock --check`, ledger, memory, push; final status maps every goal and package to its terminal state (~07:30). |

## X9 (2026-09-11 morning): fused attention where cuEq does not reach, and bf16 compute with fp32 residuals

Question from the user: AlphaFold 3 is 1.3-2× faster and 1.5-2× lighter than
the three torch-faithful ports at the same schedule. Two levers it uses and
they do not: bf16 compute in the diffusion transformer and confidence head,
and fused (tokamax/Pallas) attention on every attention site. Both are
experiments, never defaults; the parity defaults stay upstream-faithful.

Design: a 2×2 matrix per port (compute dtype of diffusion/confidence:
released vs bf16-with-fp32-residuals) × (pair-bias attention kernel: XLA vs
tokamax `dot_product_attention`), because the earlier Boltz-2 verdict
(`boltz-jax-kernel-backends-verdict`: tokamax/flash regressed 9-11% on sm120)
was measured with fp32 q/k/v in the diffusion island, exactly the cell the
dtype lever removes. Existing starting points: Protenix `--amp-policy bf16`
(X7 realisation: autocast projections, ten `precision=torch.float32` layers
exempt, sampler state f32); Boltz-2 `compute_dtype=bfloat16` default with
deliberate fp32 islands (diffusion attention, confidence) and an existing
`attention_backend=tokamax|flash|xla` knob; OpenFold3 `32-true` upstream, bf16
known to break the input embedder, diffusion/confidence-only bf16 untested;
OpenDDE mirrors Protenix. Rows queued first with no code: Protenix
`amp_policy=bf16` at 1k/2k/3k (1051-1053), Boltz-2 `attention_backend=tokamax`
at 1k/2k (1054-1055, the fp32-island cell).

Acceptance for a recommended opt-in: wall/peak gain at 1k-3k, AND the
tape-pinned 1k replay residual of the bf16 arm stays inside native's process
floor with `deterministic=on` on both arms, AND TM to the deposited structure
unchanged within sample scatter. sm120 (this host) is Triton-only for
tokamax; a win here is not a fleet-wide win and is recorded as such.

### X9 verdict (2026-09-11)

Both levers work, and the dtype lever found a real defect on the way.

**Boltz-2** gained two opt-in options scoped to the diffusion score model:
`diffusion_compute_dtype` and `diffusion_attention_backend`. Together at
1,003 tokens they are 14.7% faster than the released arm with coordinates at
the process floor; the dtype alone buys 8.4% and the fused kernel another
6.9%. Peak does not move, because this port's peak is a trunk arena. The
older "tokamax regresses 9-11%" verdict is retired: that was the port's
pinned `matmul_precision=highest` selecting tokamax's three-pass
`F32_F32_F32`, not the kernel. At `high` the same call beats XLA, and the
bfloat16 branch never reads the pin.

**Protenix** gained `tokamax` at the trunk single attention and both
diffusion attention sites, the three pair-bias attentions cuEquivariance does
not cover. Its `--amp-policy bf16` was 10% faster and 8% lighter at 2,096
tokens and misfolded one chain of the 5DEI homotetramer in every sample; the
cause was one projection rounding one tensor, and the fix (pair bias
delivered in float32, GEMM still bfloat16) keeps 82% of the gain. See the X9
section of `scale-rows-master-2026-09-10.md` for the bisection.

Two things the acceptance criterion asked for and this work supplies: the
tape-pinned 3k replay puts the bf16 arm 0.14-0.41 Å from the fp32 arm against
a 0.76-3.27 Å native floor, and the per-chain deposited TM is what caught the
2k defect that whole-complex TM and pLDDT alone would have missed.

Against the acceptance criterion, cell by cell: wall and peak gain at 1k-3k
is met for both ports except Protenix's dtype lever at 1k, where the fix
leaves only 3%; the tape-pinned residual with `deterministic=on` is inside
native's floor by five to ten times at 3k; and the deposited TM is unchanged
on every arm that ships. The recommendation is per size and per port, in the
X9 section of `scale-rows-master-2026-09-10.md`. sm120 is Triton-only for
tokamax and has no shipped autotuning cache, so none of the kernel numbers
generalise to another card without a re-run.

What the exercise cost and returned: it found a real accuracy defect in a
shipped opt-in (`--amp-policy bf16` lost a chain at 2k), retired a
three-month-old verdict that had been keeping fused attention out of Boltz-2
on the strength of one mis-attributed measurement, and left both ports with
opt-in levers worth 11-15% of wall time. The defect was only visible through
per-chain deposited RMSD on a homomer; whole-complex TM, pLDDT and the
sample-to-sample spread all looked healthy.

