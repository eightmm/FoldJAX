# Protenix current-source 3GCA comparison

CPU-only aggregation of completed tsp 802. No GPU job or model source edit
was made during this aggregation; the concurrent Claude Boltz MSA job 831
and its source snapshot were left untouched.

Candidate: external `public-other-models-20260909-4S8RAo/protenix-3gca`.
Reference: `protenix-closure-20260907/3GCA/native-r0b`.
Checkpoint: `protenix_base_default_v1.0.0.jax`; the input directory's legacy
`protenix-v2` name is not the checkpoint selection.

The existing `native_record`, `check_inputs`, `canonical_foldjax`,
`compare_entity_parity` and `compare_arrays` adapters were used directly.
Independent input comparison passes. Five samples retain their original order;
one whole-system alignment is used per sample with no entity refit.

| Entity | Per-sample RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| RNA R | 0.06410044, 0.02767160, 0.01152783, 0.01629140, 0.02169107 | 0.06410044 |
| Ligand L | 0.01533241, 0.01963973, 0.00721899, 0.03073523, 0.01011131 | 0.03073523 |

Strict confidence fails overall. Maximum absolute differences in selected
canonical fields are atom pLDDT 0.001530766487121582 (native 0–1 scale),
summary pTM 0.00008690357208251953 and summary ipTM 0.000024974346160888672.
Summary agreement does not substitute for all raw confidence leaves.

RNA remains deferred gray, not admitted. This aggregation does not establish
performance, a full current-panel pass, candidate artifact rehash completion,
or independently observed device tape consumption. No commit/push or release
is claimed.
