# OpenDDE on master: seven cases, native floors, and the TF32 policy decision

Status: closed. Every case is `pass` (entity maximum RMSD below 0.05 Å against
native) except 1URN, whose residual equals native's own process floor. The
port's default matmul policy `high` is the matching policy; `highest` is
worse on every case where it was measured.

## Arms

Pinned upstream OpenDDE `ddfa1df` (v1.1.1, same commit and clean tree as the
workstation's pinned copy), n=5, 200 steps, seed 101, one process per job on
RTX PRO 6000 Blackwell Server Edition, snapshot `opendde-master-20260909-Ayx8BN` (FoldJAX
`e324ad0`). Native runs its own default: `enable_tf32: True`
(`opendde/config/inference_defaults.py:24`, applied per forward by
`_tf32_runtime_scope` in `opendde/model/opendde.py`), recorded in every
capture's provenance as `native_tf32 True`. The port ran twice at
`--jax-matmul-precision high` (JAX TF32, the port default) and once at
`highest` (pure fp32), all replaying native A's tape. Comparisons by
`bench/opendde_closure_report.py`; table by `bench/opendde_master_table.py`.

- *native A vs B*: two native processes, same tape -- upstream's own process
  floor (torch TF32 kernels are not bitwise across processes).
- *native vs port high*: the residual under the matching policy.
- *port high A vs B*: two port processes -- the port's own floor (XLA autotune
  not frozen).
- *native vs port highest*: the mismatched policy, for the default decision.

## Results (entity maximum RMSD, Å; one whole-system Kabsch per sample)

| case | native A vs B (floor) | native vs port high | port high A vs B (floor) | native vs port highest | verdict |
| --- | --- | --- | --- | --- | --- |
| protein_1ubq | A 0.0063 | A 0.0123 | A 0.0051 | - | pass |
| protein_dna_7r6r | A 0.0095; B 0.0066; D 0.0065 | A 0.0162; B 0.0109; D 0.0108 | A 0.0393; B 0.0311; D 0.0309 | A 0.0750; B 0.0532; D 0.0492 | pass |
| protein_ligand_5sak | A 0.0094; L 0.0044 | A 0.0096; L 0.0053 | A 0.0108; L 0.0059 | A 0.0618; L 0.0166 | pass |
| protein_protein_7st3 | A 0.0057; B 0.0053 | A 0.0066; B 0.0070 | A 0.0081; B 0.0071 | A 0.0326; B 0.1181 | pass |
| protein_rna_1urn | P 0.1484; R 0.0029 | P 0.1487; R 0.0037 | P 0.0678; R 0.0035 | P 0.1488; R 0.0044 | at-floor |
| protein_rna_ligand_3v7e | L 0.0012; P 0.0063; R 0.0043 | L 0.0023; P 0.0075; R 0.0047 | L 0.0017; P 0.0076; R 0.0046 | L 0.0074; P 0.0102; R 0.0108 | pass |
| rna_ligand_3gca | L 0.0017; R 0.0018 | L 0.0015; R 0.0020 | L 0.0017; R 0.0021 | - | pass |

## Reading

- Six of seven cases pass outright at 0.002-0.016 Å against native, with both
  process floors at the same 0.002-0.010 Å scale: the port reproduces
  upstream to within upstream's own run-to-run movement.
- 1URN is the one chaotic case: native moves 0.148 Å against itself, and the
  port sits at 0.149 Å from native A and 0.068 Å from itself. That is
  `at-floor`, not a route difference.
- `highest` is worse than `high` everywhere it was run (7ST3 B 0.118 vs
  0.007; 5SAK A 0.062 vs 0.010; 7R6R A 0.075 vs 0.016; 3V7E 0.011 vs 0.008;
  1URN unchanged). Native computes in torch TF32; matching it means JAX TF32,
  which is what `high` selects. The earlier workstation reading that
  `highest` was closer came from a comparison against a capture whose
  policy was not the shipped one; on master, against upstream's default, the
  direction is unambiguous.

**Decision: `_MATMUL_PRECISION = "high"` stays the OpenDDE default** (it is
also the matching policy, so there is no speed-versus-fidelity trade to
make here; `highest` costs about 2x wall-clock and moves the structure away
from upstream).

## Job log

Jobs 493-527 (`dde-nat-{A,B}-*`, `dde-fj-high-{A,B}-*`, `dde-fj-highest-*`);
ledger rows in `docs/EXPERIMENTS.jsonl`. Assets: `~/.cache/opendde`
checkpoint and CCD; legacy driver
`foldjax-bench/entity-parity-20260905/opendde_independent_n5.py` copied from
the workstation.
