# ESMFold2 on master: seven cases, native floors, and the port's own kernel-selection noise

Status: six of seven cases closed at the user's tolerance (one pass, two
deferred, three at-floor by basin sharing); 7ST3 chain B carries a residual
above upstream's own scatter on two samples and stays open. The port's
apparent process noise was XLA kernel selection: with autotune frozen the
port is bitwise repeatable and, on 1UBQ, sits at 0.02 Å from native.

## Arms

Upstream: the `transformers-esmfold2` fork (Biohub, `ef32577f55`) imported
from its source tree by `bench/esmfold2_tape.py capture`, torch 2.13.0+cu130,
`float32_matmul_precision("highest")`, TF32 off, ordinary (non-deterministic)
CUDA algorithms, n=5, seed 101; the environment was rebuilt on master from the
workstation's pinned package list and matches it file for file. Port: FoldJAX
ESMFold2 replaying the native tape (`bench/esmfold2_tape.py replay --tape`),
independent JAX language model, shared core features
(`entity-parity-20260905/inputs/<case>/esmfold2/upstream-biohub-full.npz`).
Snapshot `esmfold2-master-20260909-RbVYiU` (FoldJAX `e324ad0`). Native A/B share the tape and
the LM output bitwise (`tape.npz`, `upstream_lm.npz` equal); their
coordinates differ because native's CUDA path is not deterministic. Port A/B
are two processes with XLA autotune unfrozen; port C/D are two processes with
autotune dumped by C and loaded by D.

Comparisons by `bench/esmfold2_tape_report.compare_coordinates` (one
whole-system Kabsch per sample, entity maxima); table by
`bench/master_three_way.py --model esmfold2`.

## Seven cases (entity maximum RMSD, Å; entity 0 = protein or first polymer, 1 = ligand or second polymer, 2 = third)

| case | native A vs B (floor) | native A vs port | native B vs port | port A vs B (floor) | verdict |
| --- | --- | --- | --- | --- | --- |
| protein_1ubq | 0 0.006 | 0 0.122 | 0 0.125 | 0 0.109 | at-floor |
| protein_dna_7r6r | 0 0.731; 1 0.100; 2 0.094 | 0 0.121; 1 0.035; 2 0.040 | 0 0.761; 1 0.118; 2 0.110 | 0 0.217; 1 0.044; 2 0.040 | at-floor |
| protein_ligand_5sak | 0 1.112; 1 0.373 | 0 1.302; 1 0.436 | 0 0.370; 1 0.134 | 0 0.749; 1 0.469 | at-floor |
| protein_protein_7st3 | 0 0.186; 1 1.088 | 0 0.273; 1 1.530 | 0 0.378; 1 1.788 | 0 0.929; 1 4.536 | at-floor |
| protein_rna_1urn | 0 0.032; 1 0.018 | 0 0.054; 1 0.018 | 0 0.064; 1 0.018 | 0 0.038; 1 0.022 | deferred |
| protein_rna_ligand_3v7e | 0 0.042; 1 0.027; 2 0.023 | 0 0.070; 1 0.025; 2 0.017 | 0 0.054; 1 0.028; 2 0.019 | 0 0.038; 1 0.029; 2 0.020 | deferred |
| rna_ligand_3gca | 0 0.013; 1 0.009 | 0 0.016; 1 0.008 | 0 0.015; 1 0.005 | 0 0.022; 1 0.010 | pass |

## Frozen autotune: the port's floor is kernel selection

| case | port C vs D (frozen) | native A vs port C, per sample | native A vs native B, per sample |
| --- | --- | --- | --- |
| 1UBQ | bitwise equal | 0.022, 0.022, 0.017, 0.018, 0.016 | 0.002-0.006 |
| 7ST3 chain A | bitwise equal | 0.40, 0.04, 0.23, 0.03, 0.58 | 0.10, 0.05, 0.08, 0.03, 0.19 |
| 7ST3 chain B | bitwise equal | 1.07, 0.06, 0.53, 0.04, 3.58 | 0.24, 0.02, 0.08, 0.03, 1.09 |

1UBQ's 0.122 Å in the table above was one unfrozen draw on sample 2; frozen,
the port matches native at 0.02 Å on every sample. On 7ST3 samples 2 and 4
the frozen port is at 0.03-0.06 Å, native scatter level. Samples 1, 3 and 5
of chain B are above native's own scatter by 2-4x (1.07 vs 0.24, 0.53 vs
0.08, 3.58 vs 1.09 Å), against both native draws (native B vs port C: 1.30,
0.46, 2.55 Å). That is the one ESMFold2 residual this panel leaves open: a
smaller chain in a two-chain complex where the port's trajectory departs
from upstream's on the chaotic samples. It is consistent with the
workstation's earlier 0.42 Å finding on 5SAK and with the known port-side
autocast placement work in `docs/esmfold2-native-tape-2026-09-07.md`.

Frozen pairs for the other five cases (jobs 619-628): every pair is bitwise
equal. Frozen port C against native A, entity maxima: 5SAK 1.62 (sample 1,
the shared basin; native B vs port C 0.64) / ligand 0.48; 7R6R 0.28 / 0.05 /
0.05 (native A vs B is 0.73 on sample 5, and the port sits with native A);
3V7E 0.056 / 0.022 / 0.017; 1URN 0.026 / 0.020; 3GCA 0.022 / 0.008. With
kernel selection removed, 1URN and 3GCA are in the pass band and 3V7E stays
deferred at 0.056 Å.

## Reading

- 3GCA passes (0.016 Å against a 0.013 Å native floor). 1URN (0.054/0.018)
  and 3V7E (0.070/0.025/0.017) are in the deferred band against native
  floors of 0.03-0.04 Å.
- 5SAK, 7R6R and 1UBQ are basin sharing or kernel selection: 5SAK sample 1
  native-A vs native-B 1.11, native-A vs port 1.30, native-B vs port 0.37;
  7R6R sample 1 native B is 0.73 Å from native A while the port is 0.12 Å
  from native A.
- 7ST3 chain B stays open as described above.

## Job log

Jobs 533-536, 564-591, 619-628; ledger rows in `docs/EXPERIMENTS.jsonl`.
Upstream environment `foldjax-bench/jctc-matrix-20260904/upstream-root/esmfold2-venv`
(uv, `--no-deps` from the workstation's freeze) and the fork source tree
copied from the workstation.

## 7ST3 chain B: the language model is excluded as the cause (2026-09-10)

Two CPU reads and one GPU arm on the open item.

**LM outputs are deterministic on both sides and differ by a bf16-order
drift.** `upstream_lm.npz` (native A = native B bitwise) versus `jax_lm.npz`
(port A = port B bitwise), `lm_hidden_states` `[1, L, 81, 2560]`, relative
RMSE per layer: 0 at the embedding, 1.6-2.6e-3 at layer 1, growing smoothly to
1.4-2.2e-2 at layer 80 on 1UBQ / 7ST3 / 5SAK alike. A uniform per-layer band
that starts at one bf16 rounding is accumulation-order arithmetic in ESMC-6B,
the same class as Boltz-2's MSA band; it is not case-specific and 1UBQ
passes at 0.02 Å with the same drift.

**Native LM injected into the port (job 707, `--native-lm`, the port's
trunk and structure head on native-A's exact LM output and tape):**

| pair | chain A samples 1-5 | chain B samples 1-5 |
| --- | --- | --- |
| native-A vs native-B | 0.10, 0.05, 0.08, 0.03, 0.19 | 0.24, 0.03, 0.08, 0.03, 1.09 |
| native-A vs port-C (frozen, own LM) | 0.40, 0.04, 0.23, 0.03, 0.58 | 1.07, 0.06, 0.53, 0.04, 3.58 |
| native-A vs port-nativelm | 0.97, 0.08, 0.54, 0.05, 0.84 | 3.16, 0.13, 1.27, 0.08, 4.69 |
| native-B vs port-nativelm | 1.06, 0.06, 0.51, 0.04, 0.73 | 3.37, 0.14, 1.20, 0.08, 3.69 |
| port-C vs port-nativelm | 0.59, 0.05, 0.31, 0.04, 0.34 | 2.22, 0.08, 0.74, 0.05, 1.66 |

Giving the port native's LM does not bring chain B toward native; it moves
samples 1/3/5 further (3.2/1.3/4.7 Å). So the residual is not the LM drift.
It sits in the downstream core (pair trunk + structure head) on the three
samples that native itself cannot repeat to better than 1.09 Å; samples 2 and
4 agree to 0.03-0.08 Å in every pairing including this one. Reading: 7ST3
chain B samples 1/3/5 are near-chaotic for this model, the port's downstream
core lands outside native's own scatter on them, and the cause is a rounding
route inside the trunk or structure head, not the language model. Stays
open, narrower: one deterministic-array bisect inside the downstream core
(trunk output, then structure-module output) is the next step, not more
coordinate draws.
