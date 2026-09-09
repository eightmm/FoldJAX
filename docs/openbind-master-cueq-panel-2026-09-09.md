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

## Still running

Jobs 265–294 complete the seven cases with the same three arms per case
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
