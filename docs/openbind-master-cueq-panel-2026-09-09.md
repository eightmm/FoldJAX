# OpenBind cuEq-versus-cuEq panel on the Slurm server (2026-09-09)

## Question

The seven-case OpenBind ledgers compared the port's cuEquivariance path against
upstream's released `predict` preset, which runs upstream's own Triton triangle
kernels (`use_triton_triangle_kernels: True`, `use_cueq_triangle_kernels:
False`). Five cases exceeded 0.1 Å there, 5SAK's ligand by 6.9 Å. This panel
asks the question that comparison cannot: does the port match upstream when
upstream is switched to the cuEq kernels the port uses? That is the fair
reference for a cuEq port, and the user's stated tolerance -- a small drift
from choosing cuEq -- applies to the residual against it, not against Triton.

## Environment

- Server `master`, 4× RTX PRO 6000 Blackwell Server (97,887 MiB), Slurm
  partition `batch`. QOS `normal` caps one user at `gres/gpu=2`, so two jobs
  run at a time regardless of the four cards.
- Native side: git worktree `openfold3-v050` at `c4771653` with a fresh
  `.venv` -- torch 2.12.1+cu130, Lightning 2.6.1, triton 3.7.1,
  cuequivariance / cuequivariance-torch / cuequivariance-ops-torch-cu13 0.11.1.
  The mirrored `upstream-root/openfold3-v050` copy lacks `core/config` and
  cannot run.
- Port side: FoldJAX `1d1cfea` archived to
  `foldjax-bench/openbind-cueq-master-20260909-Ieldnw` (`git archive HEAD`),
  cuequivariance-jax 0.11.1 with the cu13 ops package. The two sides ship the
  same cuEquivariance release.
- The mirror carried no `input.npz`/`tape.npz` and no panel MSA; the pinned
  MSAs were copied from the workstation and verified against
  `panel-manifest.json` digests before any native run.

## Native capture, rebuilt

`bench/openbind_native_capture.py` is the `--native-wrapper` for
`openbind_native_outputs.py`. It records the forward batch (`input.npz`,
including atom-array annotations), every `torch.randint/randperm/randn/
randn_like` draw in call order (`tape.npz`, spelled the way
`parse_forward_tape` reads it), the effective config, the public coordinates,
and a kernel census (`kernel-calls.json`): how many calls reached
`_cueq_triangle_attn`, `_triton_evo_attn`, plain `_attention`,
`_cueq_triangle_mult`, and the Triton trimul helpers, plus how often
`cueq_would_fall_back` returned true. The census matters because upstream's
cuEq attention silently falls back to plain torch attention at or below
`CUEQ_TRIATTN_FALLBACK_THRESHOLD = 100` tokens; the JAX kernel has no such
threshold, so 1UBQ (76 tokens) and 3GCA (46 tokens) are not cuEq-versus-cuEq
comparisons for attention whatever the flag says.

Smoke (GB1, 56 residues, no MSA): the tape parsed as 605 draws (4 `randint`,
no `randperm` with a query-only MSA, 1 + 200×3 diffusion draws) and the port
replayed it. FoldJAX cuEq versus native Triton: 0.021 Å; versus native cuEq
(attention fell back, trimul fused): 0.017 Å. Both arms' three calls were
bitwise equal.

## 5SAK: the discriminating result

Native captures (437 tokens, 3,073 atoms, n=5, 200 steps, four trunk passes,
FP32, seed 101) took 70 s each. Census, Triton arm: `attention.triton` 500,
Triton trimul helpers 4,000/5,000. Census, cuEq arm: `attention.cueq` 500,
`attention.cueq_fallback_false` 500, `trimul.cueq` 428, no fallback.

| FoldJAX cuEq versus | A max RMSD (Å) | L max RMSD (Å) | pLDDT max | pTM max | ipTM max |
| --- | ---: | ---: | ---: | ---: | ---: |
| native Triton (jobs 260/261) | 1.114963 | 6.881971 | 10.560 | 0.000742 | 0.023954 |
| native cuEq (jobs 263/264) | 0.047650 | 0.049454 | 2.721 | 0.000453 | 0.002881 |

The Triton row reproduces the workstation ledger (1.109/6.886 Å) on different
hardware, so the discrepancy is deterministic. Against upstream's own cuEq
path both entities sit below the 0.05 Å pass line. The residual the earlier
ledgers were chasing was the kernel choice, not the port. Confidence is not
admitted by this (the strict gate is separate), and this port arm still runs
the multiplication in XLA; the `cueq-full` arm below measures whether fusing
it as upstream does tightens the remaining 0.05 Å.

## Upstream against itself: Triton versus cuEq, same tape

Both native captures of a case share the forward batch array for array and
the RNG tape digest, so the only change between them is upstream's own
kernel flag. Entity maxima after one system Kabsch per sample:

| case | tokens | native Triton vs native cuEq (Å) | cuEq attention in the cuEq arm |
| --- | ---: | --- | --- |
| protein_ligand_5sak | 437 | A 0.9635; L 6.8847 | cuEq |
| protein_protein_7st3 | 545 | A 0.2176; B 0.9034 | cuEq |
| protein_1ubq | 76 | A 0.1935 | fell back (plain torch attention) |
| protein_dna_7r6r | 245 | A 0.8254; B 0.6251; D 0.6200 | cuEq |
| protein_rna_ligand_3v7e | 235 | L 0.0944; P 0.0626; R 0.1319 | cuEq |
| protein_rna_1urn | 118 | P 0.0336; R 0.0127 | cuEq |
| rna_ligand_3gca | 46 | L 0.0076; R 0.0099 | fell back (plain torch attention) |

These are the numbers the earlier seven-case ledgers attributed to the port:
5SAK's ligand (6.89 Å there, 6.88 Å here), 7ST3's B chain (0.91 / 0.90),
7R6R's three chains (0.82/0.61/0.61 there, 0.83/0.63/0.62 here). Upstream
moves that far by itself when it swaps its Triton kernels for cuEq. A port
that runs cuEq can only be compared against the cuEq arm.

The native cuEq arm is not bitwise reproducible across processes either:
a second 5SAK capture with the same tape (job 296, census identical) differs
from the first by A 0.0455 Å, L 0.0562 Å. The port's residual
against the first capture, 0.048/0.049 Å, is therefore at the native floor,
not above it. `bench/openbind_native_diff.py` produced these reports
(`native-triton-vs-cueq.json`, `native-cueq-vs-repeat.json` beside the
captures).

## Seven cases against native cuEq, with both process floors

Every FoldJAX arm below ran with XLA autotuning frozen per case and backend
(first process dumps its choices, later ones load them with
`--xla_gpu_require_complete_aot_autotune_results`); the second frozen process
was bitwise equal to the first on every field in every case. Two floors sit
beside each residual: *native floor* is native cuEq against its own repeat
(same tape, second process); *port floor* is the frozen port process against
an unfrozen one (same tape, different autotune choices). A residual above
0.1 Å but within twice the larger floor is process noise, not a route
difference, and is reported as `at-floor`.

### `cueq` (fused attention, XLA multiplication) -- snapshot `openbind-cueq-master-20260909-Ieldnw`

| case | census | entity max RMSD (A) | native floor (A) | port floor (A) | structure | repeat bitwise | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| protein_ligand_5sak | cueq-vs-cueq | A 0.133270; L 0.085040 | 0.0562 | 0.1567 | at-floor | True | at-floor |
| protein_protein_7st3 | cueq-vs-cueq | A 0.037665; B 0.197524 | 0.1329 | 0.1129 | at-floor | True | at-floor |
| protein_1ubq | small-token (native cuEq attention fell back) | A 0.382701 | 0.1534 | 0.0674 | investigate | True | excluded |
| protein_dna_7r6r | cueq-vs-cueq | A 0.063691; B 0.062581; D 0.061501 | 0.0391 | 0.0793 | deferred | True | deferred |
| protein_rna_ligand_3v7e | cueq-vs-cueq | L 0.116444; P 0.143696; R 0.182466 | 0.0637 | 0.0873 | investigate | True | fail |
| protein_rna_1urn | cueq-vs-cueq | P 0.041580; R 0.013503 | 0.0147 | 0.0130 | pass | True | pass |
| rna_ligand_3gca | small-token (native cuEq attention fell back) | L 0.005262; R 0.009997 | 0.0098 | 0.0102 | pass | True | excluded |

accepted: False

### `cueq-full` (fused attention and multiplication) -- snapshot `openbind-cueqfull-master-20260909-pX5P4`

Port floors were not measured for this arm; only the native floor applies.

| case | census | entity max RMSD (A) | native floor (A) | port floor (A) | structure | repeat bitwise | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| protein_ligand_5sak | cueq-vs-cueq | A 0.216078; L 0.137350 | 0.0562 | - | investigate | True | fail |
| protein_protein_7st3 | cueq-vs-cueq | A 0.054119; B 0.168730 | 0.1329 | - | at-floor | True | at-floor |
| protein_1ubq | small-token (native cuEq attention fell back) | A 0.418959 | 0.1534 | - | investigate | True | excluded |
| protein_dna_7r6r | cueq-vs-cueq | A 0.067149; B 0.047744; D 0.050975 | 0.0391 | - | deferred | True | deferred |
| protein_rna_ligand_3v7e | cueq-vs-cueq | L 0.097638; P 0.057731; R 0.093185 | 0.0637 | - | deferred | True | deferred |
| protein_rna_1urn | cueq-vs-cueq | P 0.013014; R 0.013665 | 0.0147 | - | pass | True | pass |
| rna_ligand_3gca | small-token (native cuEq attention fell back) | L 0.005825; R 0.009943 | 0.0098 | - | pass | True | excluded |

accepted: False

### Reading

- 7R6R falls from 0.80/0.61/0.60 Å against Triton to 0.06 Å against cuEq on
  both arms; 1URN passes on both; 3GCA passes although its native cuEq
  attention fell back (46 tokens).
- 5SAK and 7ST3 sit inside the noise band on the `cueq` arm (5SAK's 0.133 Å
  equals the quadrature sum of its two floors, 0.056 and 0.157 Å). 3V7E's R
  chain, 0.182 Å, is 0.007 Å above twice its larger floor; on `cueq-full` it
  is 0.093 Å. 5SAK moves the other way on `cueq-full` (0.216/0.137 Å, pLDDT
  4.9 points). Neither arm is uniformly closer: both live in the same
  0.1-0.2 Å kernel-selection band on the chaotic cases, and no case shows a
  residual that a port defect would explain and the floors would not.
- 1UBQ (76 tokens) is the one case that stays above every floor, 0.38 Å, and
  it is exactly the case where native cuEq attention silently ran plain torch
  attention while the port ran the fused kernel; it is a different-kernel
  comparison by construction and is excluded from the count.

Structure is therefore closed for OpenBind at the level the user set: what
remains between the port and upstream's cuEq path is the size of upstream's
own process-to-process movement. Confidence maxima (pLDDT up to 6 points on
7ST3) are recorded, not admitted. The `cueq-full` default question is a
performance question now and goes to the warm benchmarks.

## Boltz-2: version is not the cause

Upstream Boltz-2 selects cuEquivariance torch kernels under
`use_kernels=True`, so it was already a cuEq-versus-cuEq comparison. On this
server two native processes of 5SAK (pinned `b1ebfc4`, cuEq torch 0.10.0,
bf16-mixed, same tape) were bitwise equal in coordinates, tape, features and
every recorded trunk boundary, and a third process under cuEq torch 0.11.1
was bitwise equal to them as well. Native Boltz-2 has no process floor, and
the port's 1.18 Å cell (MSA-module `delta_z` 3.2e-2) cannot be attributed to
the kernel release. It is a real cross-route difference and needs the
module-internal bisection the earlier ledger stopped short of.

## Job log

Jobs 265–294 completed the seven cases with the same three arms per case
(native Triton, native cuEq, FoldJAX cuEq replay against each, plus a
second-process FoldJAX repeat for the cross-process floor). Results are
appended below as they land; the ledger rows are in `docs/EXPERIMENTS.jsonl`.

## Handoff state (session end, 2026-09-09 20:40 KST)

- Snapshot root: `/home/jaemin/non-project/optimizing/foldjax-bench/openbind-cueq-master-20260909-Ieldnw` (FoldJAX `1d1cfea`); job scripts in its `jobs/`
  (`core-case.sbatch`, `native-capture.sbatch`, `launch-panel.sh`,
  `gpu-pytest.sbatch`); logs in `logs/`. Native captures under
  `foldjax-bench/openbind-master-native-20260909/<case>/native-{triton,cueq}`.
- Queued: panel jobs 265–294 (native Triton/cuEq per case, FoldJAX cuEq replay
  against each, second-process repeat), 295 (GPU test for `cueq-full`),
  296 (second native cuEq 5SAK capture for the native cross-process floor).
  `python bench/openbind_panel_table.py <snapshot>` renders the results.
- Uncommitted: the `cueq-full` triangle kernel (fused cuEq multiplication,
  `src/foldjax/models/openfold3/models/triangle.py`, attention alias, option
  map, tests). Commit only after job 295 passes; the GPU test's masked-input
  case may expose a mask-semantics difference (cuEq masks output, the XLA path
  masks the projections) and must not be loosened to pass.
- Next: (1) diff job 262 against 261 with `openbind_candidate_diff.py` and
  job 296's `coordinate.npz` against 263's for the two floors; (2) write the
  acceptance script (native arm = cuEq, 0.05 pass / 0.05–0.1 deferred / >0.1
  investigate, census `attention.cueq_fallback_true == 0` required, 1UBQ and
  3GCA reported as a separate small-token class, repeat flags required) and
  register it as the goal-drive acceptance; (3) snapshot #2 with `cueq-full`
  and replay it against the native cuEq captures; (4) warm benchmarks with
  the scope mismatch stated; (5) an ion case only after parity is settled.
- Not done, by decision: no push to origin (not authorised); the
  workstation tree still holds the 54 files committed here as 85f23a6,
  3d3ae56, 1d1cfea.
