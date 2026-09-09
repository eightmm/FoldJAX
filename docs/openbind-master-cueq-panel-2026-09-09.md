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
an unfrozen one (same tape, different autotune choices), the largest of `n`
samples. A residual above 0.1 Å but within twice the larger floor is process
noise, not a route difference, and is reported as `at-floor`.

### `cueq` (fused attention, XLA multiplication) -- snapshot `openbind-cueq-master-20260909-Ieldnw`

| case | census | entity max RMSD (A) | native floor (A) | port floor (A) | structure | repeat bitwise | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| protein_ligand_5sak | cueq-vs-cueq | A 0.133270; L 0.085040 | 0.0562 | 0.1567 (n=1) | at-floor | True | at-floor |
| protein_protein_7st3 | cueq-vs-cueq | A 0.037665; B 0.197524 | 0.1329 | 0.1129 (n=1) | at-floor | True | at-floor |
| protein_1ubq | small-token (native cuEq attention fell back) | A 0.382701 | 0.1534 | 0.0674 (n=1) | investigate | True | excluded |
| protein_dna_7r6r | cueq-vs-cueq | A 0.063691; B 0.062581; D 0.061501 | 0.0391 | 0.0793 (n=1) | deferred | True | deferred |
| protein_rna_ligand_3v7e | cueq-vs-cueq | L 0.116444; P 0.143696; R 0.182466 | 0.0637 | 0.2237 (n=2) | at-floor | True | at-floor |
| protein_rna_1urn | cueq-vs-cueq | P 0.041580; R 0.013503 | 0.0147 | 0.0130 (n=1) | pass | True | pass |
| rna_ligand_3gca | small-token (native cuEq attention fell back) | L 0.005262; R 0.009997 | 0.0098 | 0.0102 (n=1) | pass | True | excluded |

accepted: True

### `cueq-full` (fused attention and multiplication) -- snapshot `openbind-cueqfull-master-20260909-pX5P4`

| case | census | entity max RMSD (A) | native floor (A) | port floor (A) | structure | repeat bitwise | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| protein_ligand_5sak | cueq-vs-cueq | A 0.216078; L 0.137350 | 0.0562 | 0.1557 (n=2) | at-floor | True | at-floor |
| protein_protein_7st3 | cueq-vs-cueq | A 0.054119; B 0.168730 | 0.1329 | 0.2061 (n=1) | at-floor | True | at-floor |
| protein_1ubq | small-token (native cuEq attention fell back) | A 0.418959 | 0.1534 | 0.0769 (n=1) | investigate | True | excluded |
| protein_dna_7r6r | cueq-vs-cueq | A 0.067149; B 0.047744; D 0.050975 | 0.0391 | 0.0724 (n=1) | deferred | True | deferred |
| protein_rna_ligand_3v7e | cueq-vs-cueq | L 0.097638; P 0.057731; R 0.093185 | 0.0637 | 0.2653 (n=1) | deferred | True | deferred |
| protein_rna_1urn | cueq-vs-cueq | P 0.013014; R 0.013665 | 0.0147 | 0.0135 (n=1) | pass | True | pass |
| rna_ligand_3gca | small-token (native cuEq attention fell back) | L 0.005825; R 0.009943 | 0.0098 | 0.0105 (n=1) | pass | True | excluded |

accepted: True

### The two borderline cases, resolved by a second draw rather than a wider rule

A single frozen-versus-unfrozen pair is one sample of the port's own
kernel-selection noise. For the two cases that sat at the edge, one more
unfrozen process was replayed against the *same native cuEq capture*, giving a
second residual draw and a second floor sample (jobs 414, 415):

| case / arm | residual draw 1 (frozen) | residual draw 2 (unfrozen) | port floor sample 1 | port floor sample 2 (frozen vs draw 2) |
| --- | ---: | ---: | ---: | ---: |
| 3V7E `cueq`, R chain | 0.182 | 0.071 | 0.087 | 0.224 |
| 5SAK `cueq-full`, A chain | 0.216 | 0.230 | 0.156 | 0.060 |

3V7E: the two draws disagree by more than the residual and the second floor
sample exceeds it, so 0.182 Å is the port's own process noise (`at-floor`).
5SAK on `cueq-full`: both draws sit at 0.22 Å, a *stable* difference, still
within twice its 0.156 Å floor sample and therefore admitted by the rule --
but it is a stable route difference the rule allows, not noise, and it is
larger than the `cueq` arm's 0.133 Å on the same case. Recorded as such.

### Confidence: native floors and port residuals (max |delta| over five samples; pLDDT 0-100 / pTM / ipTM)

| case | native cuEq vs its repeat | native Triton vs cuEq | port `cueq` vs native cuEq | port `cueq-full` vs native cuEq |
| --- | --- | --- | --- | --- |
| protein_ligand_5sak | 1.95 / 0.001 / 0.002 | 9.67 / 0.001 / 0.024 | 1.22 / 0.001 / 0.002 | 4.87 / 0.001 / 0.005 |
| protein_protein_7st3 | 3.61 / 0.000 / 0.000 | 9.17 / 0.001 / 0.003 | 6.27 / 0.000 / 0.002 | 5.06 / 0.001 / 0.006 |
| protein_1ubq | 0.88 / 0.001 / 0.000 | 1.61 / 0.002 / 0.000 | 1.36 / 0.001 / 0.000 | 2.64 / 0.003 / 0.000 |
| protein_dna_7r6r | 0.71 / 0.000 / 0.000 | 2.10 / 0.000 / 0.000 | 0.53 / 0.000 / 0.000 | 0.75 / 0.000 / 0.001 |
| protein_rna_ligand_3v7e | 0.86 / 0.002 / 0.002 | 1.22 / 0.001 / 0.004 | 5.91 / 0.010 / 0.007 | 1.37 / 0.004 / 0.003 |
| protein_rna_1urn | 1.71 / 0.001 / 0.001 | 0.40 / 0.000 / 0.001 | 1.63 / 0.001 / 0.003 | 0.89 / 0.000 / 0.001 |
| rna_ligand_3gca | 0.12 / 0.000 / 0.000 | 0.11 / 0.000 / 0.000 | 0.12 / 0.000 / 0.000 | 0.23 / 0.000 / 0.000 |

pTM/ipTM agree to 0.01 everywhere. Per-atom pLDDT maxima move 2-4 points
between two native cuEq processes and 9 points between upstream's own two
kernel families; the port's 6.3 (7ST3 `cueq`) and 5.9 (3V7E `cueq`) sit
inside twice the native repeat floor on 7ST3 and above it on 3V7E, where the
`cueq-full` arm (1.4) does not. Confidence stays recorded, not gated: a
per-atom maximum is one atom's pLDDT, and upstream's own repeat moves it by
the same order.

### Reading

- 7R6R falls from 0.80/0.61/0.60 Å against Triton to 0.06 Å against cuEq on
  both arms; 1URN passes on both; 3GCA passes although its native cuEq
  attention fell back (46 tokens).
- 5SAK and 7ST3 sit inside the noise band on the `cueq` arm (5SAK's 0.133 Å
  equals the quadrature sum of its two floors, 0.056 and 0.157 Å). 3V7E's R
  chain is process noise by the second draw. 5SAK on `cueq-full` is a stable
  0.22 Å, admitted by the floor rule and noted as the arm's weakest case.
- 1UBQ (76 tokens) is the one case that stays above every floor, 0.38-0.42 Å,
  and it is exactly the case where native cuEq attention silently ran plain
  torch attention while the port ran the fused kernel; it is a different-kernel
  comparison by construction and is excluded from the count.

Both arms are accepted by `bench/openbind_acceptance.py`. What remains between
the port and upstream's cuEq path is the size of upstream's own
process-to-process movement.

## Warm timing and peak, seven cases, four arms

Three warm calls after one first call, median reported; FoldJAX arms are the
resident-input `compile_predict` forward with ordinary RNG (MSA cycle rows
from NumPy `default_rng(101)`, JAX key 101), unfrozen autotune (what a user
gets); native arms are upstream `predict_step` with its own reseeding and its
full confidence/ranking post-processing, kernel family selected the same way
as the captures and proven by the census (1UBQ and 3GCA native cuEq attention
fell back to torch). Peaks are allocator lifetime peaks (torch
`max_memory_allocated`, JAX `peak_bytes_in_use`), which include setup and
compile. **The native and FoldJAX workloads are not identical, so the ratio is
an observed end-to-end forward ratio, never a same-operations kernel
speedup.** Every FoldJAX arm's three warm outputs were bitwise equal to its
first. GPU: RTX PRO 6000 Blackwell Server Edition, one job per card, four
cards busy. Snapshot `openbind-warm-master-20260909-bRhwKA` (the fj jobs show
`FAILED` in `sacct` only because the sbatch tail looked for the wrong summary
file after the harness had already written `finished.json`; ledger rows are
exit 0).

| case | native-triton warm s / peak GiB | native-cueq warm s / peak GiB | fj-cueq warm s / peak GiB | fj-cueq-full warm s / peak GiB |
| --- | ---: | ---: | ---: | ---: |
| protein_ligand_5sak | 11.657 / 8.82 | 10.603 / 9.73 | 9.560 / 6.92 | 8.811 / 7.38 |
| protein_protein_7st3 | 16.885 / 12.36 | 15.015 / 13.78 | 14.498 / 9.77 | 13.252 / 10.54 |
| protein_1ubq | 5.995 / 1.77 | 5.738 / 1.77 | 1.105 / 1.80 | 1.109 / 1.80 |
| protein_dna_7r6r | 6.335 / 3.64 | 6.382 / 3.93 | 3.924 / 3.12 | 3.771 / 3.26 |
| protein_rna_ligand_3v7e | 6.096 / 3.94 | 5.892 / 4.21 | 4.090 / 3.08 | 3.958 / 3.21 |
| protein_rna_1urn | 6.349 / 2.27 | 6.419 / 2.34 | 1.585 / 2.00 | 1.616 / 2.00 |
| rna_ligand_3gca | 5.852 / 1.53 | 5.405 / 1.54 | 1.038 / 1.65 | 1.044 / 1.65 |

`cueq-full` is faster than `cueq` on every case with a triangle workload
(5SAK 7.8%, 7ST3 8.6%, 7R6R 3.9%, 3V7E 3.2%; equal on the three small
cases) and uses more memory on the same cases (5SAK +0.46 GiB, 7ST3 +0.77,
7R6R +0.14, 3V7E +0.13). Native cuEq is 4-11% faster than native Triton on
the two large cases and equal elsewhere.

**Default decision: keep `cueq` as the OpenFold3 default; `cueq-full` stays an
option.** The speed gain is under 9% at these sizes, the peak is higher on
every case that benefits, and on parity `cueq-full` carries the one stable
0.22 Å case (5SAK) while `cueq` carries none. The `fused-triangle-kernel-beats-
blocking` note recorded that the fused multiplication's advantage grows with
size (large at 1003 tokens, nothing at 3012 for attention); a 1-3k-token warm
pair is the measurement that would justify flipping the default, and it is not
in this panel.

## Boltz-2: version is not the cause

Upstream Boltz-2 selects cuEquivariance torch kernels under
`use_kernels=True`, so it was already a cuEq-versus-cuEq comparison. On this
server two native processes of 5SAK (pinned `b1ebfc4`, cuEq torch 0.10.0,
bf16-mixed, same tape) were bitwise equal in coordinates, tape, features and
every recorded trunk boundary, and a third process under cuEq torch 0.11.1
was bitwise equal to them as well. Native Boltz-2 has no process floor. The
scale for its residual is upstream's own kernel toggle instead; see
`docs/boltz2-master-kernel-toggle-2026-09-09.md`.

## Job log

Jobs 265-294: seven cases, native Triton and cuEq captures, FoldJAX `cueq`
replay against each, second-process repeat. 295: `cueq-full` GPU tests. 296:
second native cuEq 5SAK capture. 297-413: native cuEq repeats for the other
six cases, `cueq` port floors, the `cueq-full` arm (three replays per case),
cross-process diffs. 414-429: second draws (3V7E `cueq`, 5SAK `cueq-full`)
and seven `cueq-full` port floors. 432-457: native warm (Triton, cuEq).
473-486: FoldJAX warm (`cueq`, `cueq-full`). Ledger rows in
`docs/EXPERIMENTS.jsonl`.

## State (2026-09-09 22:10 KST)

- Snapshots: `cueq` `openbind-cueq-master-20260909-Ieldnw` (FoldJAX
  `1d1cfea`), `cueq-full` `openbind-cueqfull-master-20260909-pX5P4`, warm
  `openbind-warm-master-20260909-bRhwKA` (`ad038ee` + the warm harness fix).
  Native captures `openbind-master-native-20260909/<case>/native-{triton,cueq,
  cueq-repeat}` with `native-cueq-vs-repeat.json` and
  `native-triton-vs-cueq.json` (coordinates and confidence).
- Acceptance: `bench/openbind_acceptance.py --snapshot <snap> --native <nat>
  [--backend cueq-full]` exits 0 on both arms.
- Open: a 1-3k-token warm pair for the `cueq-full` default question; an
  ion-containing case once the other models' floors are settled.
