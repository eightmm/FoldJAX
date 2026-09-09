# One ion-containing case (1AAY) added to the multimodal panel, run on master

Status: complete. All four models ran the case; the tables below are the
fair-reference parity recipe the seven existing panel cases were run under on
2026-09-09.

## The case

`protein_dna_ion_1aay` -- PDB 1AAY, the Zif268 three-finger zinc-finger peptide
bound to its DNA site, with the three structural Zn(II) ions the fingers fold
around. It is the panel's first target with ions and its first entity carried
in more than one copy.

| entity | job chain | content | residues / atoms |
| --- | --- | --- | ---: |
| protein | A | `MERPYACPVESC...HTKIHLRQKD` (Zif268) | 90 |
| DNA | B | `AGCGTGGGCGT` | 11 |
| DNA | C | `TACGCCCACGC` | 11 |
| ligand | D, E, F | CCD `ZN`, one entity in three copies | 3 |

115 tokens. The three ions are one ligand entity with an id list, which
`foldjax/job.py` reads as a tuple id and every backend translation expands into
three single-atom chains.

Reference `data/references/1AAY.cif` (RCSB), label-asym to job-chain map
`{C: A, A: B, B: C, D: D, E: E, F: F}` -- the deposition's three Zn asyms are
already spelled D, E and F, and its protein is label asym C.

**Length field.** `sequences.json` records `length: 115`, the token count the
task specified. The panel's own generator counts polymer residues only and
would write 112 (5SAK is recorded as 419 with its ligand excluded, though the
OpenBind panel calls the same case 437 tokens). Nothing downstream reads this
except the shortest-first sort in `bench/spec.py`, which is identical either
way. Both numbers are stated here so the discrepancy is not silent.

## MSA provenance

The protein chain went through the panel's own path,
`foldjax.search.msa.MsaSearchPipeline` over `RemoteMMseqs2Client
("https://api.colabfold.com", version="colabfold-mmseqs2")` with options
`{"host": "https://api.colabfold.com", "pairing": "paircomplete"}`, cached under
`data/msa-cache`. The two DNA chains and the ions carry no alignment.

| field | value |
| --- | --- |
| file | `data/msa/1aay_A_unpaired.a3m` |
| depth | 18,684 sequences |
| bytes | 2,859,344 |
| sha256 | `b7cc301e34c23237a8e600ce9a782e175b6029d69df4c41493fc2b7e4d1238f0` |
| cache key | `9c16637897cb8b00d7fecadd18ff13c7c081a27d4e2a946d2c9c72d896487d6d` |
| unpaired job id | `9sd1AHPZD7KmI34nPk9dgHUVv0Pw1HMuxegEeg` |
| paired job id | `ZhWMOd-9zek1v2stRy7S8zcVDHNPouWyS0aPdQ` |

The seven pinned panel MSAs were verified byte-for-byte against
`panel-manifest.json` before and after the write; none changed.

## Registration, and why it is not in git

`foldjax-bench` is not a git repository and the FoldJAX repository tracks no
`sequences.json`, `jobs/`, `panel-manifest.json` or `*.a3m`, so the panel data
below cannot be committed. Its digests are recorded here instead.

| artifact | sha256 |
| --- | --- |
| `data/jobs/protein_dna_ion_1aay.json` | `e05209e17629bc4ded233e07d261c17a1c6610be5ade426cd690d938924d2d8f` |
| `data/sequences.json` (all eight cases) | `11054b7e54612373c652df282c8ebad0afed7a5759a22f4f837fea0196102ca6` |
| `panel-manifest.json` (eight targets) | `d61d8cf13fb70489b8c817306f84dd57dbe50647bb7efa70215a19d834621774` |
| `data/references/1AAY.cif` | `41b8a72426810c08997e89928f8fb8dc897120dabff4011760bc0864aa8afe28` |
| `data/msa/1aay_A_unpaired.a3m` | `b7cc301e34c23237a8e600ce9a782e175b6029d69df4c41493fc2b7e4d1238f0` |
| `prepare_panel.py` (with the new `CASES` entry) | `0c137dc726a8a52e6d05336018de3677b1dce419ee139e9bcc3b1c290f7a69b0` |

The case was appended to the three data files in place, not by re-running
`prepare_panel.py`: its `copy_legacy()` reads a `jctc-matrix-20260904/data`
tree that does not exist on this server, and `launch-panel.sh` gates on the
manifest's a3m digests, so a regenerate would break the other seven. The case
was added to `prepare_panel.CASES` for the record, with that reason in a
comment beside it.

## Ion translation, before any GPU time

Every backend's input translation accepts the three-copy `ZN` entity. Checked
on the login node with `foldjax.input.materialize_native_input` alone, no model
execution:

| model | native input | bytes |
| --- | --- | ---: |
| openfold3 | `openfold3_input.json`, `ccd_codes: ["ZN"]` over `chain_ids: [D, E, F]` | 1,015 |
| boltz2 | `boltz2_input.yaml`, `ligand.ccd: ZN`, `id: [D, E, F]` | 722 |
| protenix | `protenix_input.json`, `ligand` with the same three ids | 1,020 |
| opendde | `opendde_input.json` | 1,020 |

No model rejected the ion.

## FoldJAX side, one run per model (the arms' input source)

`bench.run_foldjax` with the panel's own argv, environment and options
(`run_suite.py`): seed 101, 5 samples, 200 steps, recycles per the panel
schedule (OpenFold3 and Boltz-2 3, Protenix and OpenDDE 10), OpenFold3 at its
default precision, Boltz-2 and Protenix `dtype=bfloat16`, OpenDDE
`dtype=float32`. Its side effect is the backend-native input each parity arm
below consumes.

| model | job | outcome |
| --- | ---: | --- |
| openfold3 | 592 | ran; 115 tokens, 5 samples, 143.9 s, peak 1,837.8 MiB, mean pLDDT 89.94-93.07, ipTM 0.935-0.952, no clash |
| boltz2 | 593 | prediction ran; **exited 1** at 206 s on `ArtifactFingerprintError` |
| protenix-v2 | 594 | prediction ran and wrote `seed_101/predictions`; **exited 1** at 201 s, same error |
| opendde | 595 | prediction ran and wrote `seed_101/predictions`; **exited 1** at 208 s, same error |

Three of the four exits are fingerprint kills, not the ion and not a prediction
failure. `bench/provenance.py` fingerprints `src/foldjax` and aborts when that
tree changes mid-prediction; a merge landed three refactor commits in it on
`main` while these jobs ran, moving the repository HEAD from `5987229` to
`4902a7a`. The trigger is that merge, not the input. OpenFold3 ran before those
commits and is clean. In every case the backend-native input was written before
the guard fired, and the Protenix and OpenDDE predictions completed and were
written, so what the guard rejected is the provenance record, not the result.
Every parity arm below reads a snapshot source tree rather than the live
checkout and is insulated from the same churn. Not re-run, at the task owner's
direction: these rows are the arms' input source, not a measurement this note
reports.

## OpenFold3 / OpenBind: the native side

Pinned upstream `openfold3-v050` at `c4771653` with its own `.venv`
(torch 2.12.1+cu130, triton 3.7.1, cuEquivariance 0.11.1), released `predict`
preset, n=5, 200 steps, seed 101, FP32, RNG tape and kernel census recorded.
Three captures: `native-triton`, `native-cueq`, `native-cueq-repeat`
(jobs 596, 597, 598; 34, 33 and 31 s).

Census. 115 tokens is above upstream's `CUEQ_TRIATTN_FALLBACK_THRESHOLD` of
100, so unlike 1UBQ and 3GCA this is a genuine cuEq-versus-cuEq comparison:

| arm | calls |
| --- | --- |
| `native-triton` | `attention.torch` 6200, `attention.triton` 500, `trimul.triton_layernorm` 3500, `trimul.triton_linear_fused` 4000 |
| `native-cueq` | `attention.cueq` 500, `attention.cueq_fallback_false` 500, `attention.torch` 6200, `trimul.cueq` 428 |
| `native-cueq-repeat` | identical to `native-cueq` |

No `attention.cueq_fallback_true`. The Triton census matches 7R6R's exactly.

Upstream against itself, same tape, entity maximum RMSD in Å (the three ions
carry their own rows, D/E/F):

| pair | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native cuEq vs its repeat (native floor) | 0.0640 | 0.0097 | 0.0100 | 0.0318 | 0.0232 | 0.0368 |
| native Triton vs native cuEq | 0.0416 | 0.0102 | 0.0109 | 0.0239 | 0.0212 | 0.0403 |

For comparison, the same Triton-versus-cuEq pair measured 0.83/0.63/0.62 Å on
the panel's other protein-DNA case (7R6R) and 6.88 Å on 5SAK's ligand.

## OpenFold3 / OpenBind: the port against the fair reference

FoldJAX `cueq` arm, snapshot `openbind-cueq-master-20260909-Ieldnw`,
cuequivariance-jax 0.11.1, replaying each native capture's fixed tape three
times. XLA autotuning is frozen per case and backend as the panel does: job 599
dumped `autotune/protein_dna_ion_1aay-cueq.textproto`, and jobs 600 and 601
loaded it with `--xla_gpu_require_complete_aot_autotune_results`. Job 602 is the
unfrozen process that gives the port's own floor.

Entity maximum RMSD in Å, one whole-system Kabsch per sample:

| pair | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| port vs native Triton (599, frozen) | 0.043872 | 0.011258 | 0.011037 | 0.022848 | 0.015006 | 0.056618 |
| port vs native Triton (600, second process) | 0.043872 | 0.011258 | 0.011037 | 0.022848 | 0.015006 | 0.056618 |
| **port vs native cuEq (601, the fair reference)** | **0.029929** | **0.010012** | **0.009701** | **0.023284** | **0.017918** | **0.033700** |
| port vs native Triton (602, unfrozen) | 0.043749 | 0.009882 | 0.010612 | 0.017796 | 0.015123 | 0.063092 |
| port floor: 599 vs 602 | 0.027941 | 0.010807 | 0.010415 | 0.028954 | 0.017657 | 0.036899 |

The two frozen processes are bitwise equal on all eight recorded fields
(coordinates, pLDDT, pTM, ipTM, PAE, PDE, distogram and
experimentally-resolved logits); their entity RMSD is 1e-15 Å, i.e. float noise.

`bench/openbind_acceptance.py --snapshot <snap> --native <nat> --case
protein_dna_ion_1aay`:

| case | census | entity max RMSD (A) | native floor (A) | port floor (A) | structure | repeat bitwise | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| protein_dna_ion_1aay | cueq-vs-cueq | A 0.029929; B 0.010012; C 0.009701; D 0.023284; E 0.017918; F 0.033700 | 0.0640 | 0.0369 (n=1) | pass | True | pass |

accepted: True

Confidence, maximum absolute difference over the five samples
(pLDDT 0-100 / pTM / ipTM):

| pair | pLDDT | pTM | ipTM |
| --- | ---: | ---: | ---: |
| native cuEq vs its repeat | 2.466 | 0.003637 | 0.002639 |
| native Triton vs native cuEq | 2.428 | 0.001260 | 0.001117 |
| port vs native cuEq | 2.284 | 0.002739 | 0.002253 |

## Boltz-2

Pinned upstream `b1ebfc4`, cuEquivariance torch kernels (`use_kernels=True`),
bf16-mixed, n=5, 200 steps, 3 recycles, seed 101; two native processes on the
same tape (jobs 606, 607) and two port replays of `native-A`'s tape and features
(`boltz2-master-port-20260909-IcGOFo`, jobs 608, 609, XLA autotune not frozen).
Comparisons by `bench.boltz_amp_report.compare_coordinates` over `native-A`'s
`features.npz`, one whole-system Kabsch per sample.

Boltz labels entities `entity=N|mol_type=M|asym=K`. Read from that
`features.npz` in token order, the mapping to job chains is:

| label | chain | tokens |
| --- | --- | ---: |
| `entity=0\|mol_type=0\|asym=0` | A (protein) | 90 |
| `entity=1\|mol_type=1\|asym=1` | B (DNA) | 11 |
| `entity=2\|mol_type=1\|asym=2` | C (DNA) | 11 |
| `entity=3\|mol_type=3\|asym=3` | D (Zn) | 1 |
| `entity=3\|mol_type=3\|asym=4` | E (Zn) | 1 |
| `entity=3\|mol_type=3\|asym=5` | F (Zn) | 1 |

The three ions share one `entity_id`, which is how they were declared, and the
115 tokens are Boltz's own count.

Entity maximum RMSD, Å:

| pair | A | B | C | D | E | F | bitwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native A vs native B (native floor) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True |
| native A vs port A | 0.174617 | 0.004053 | 0.004741 | 0.005161 | 0.006293 | 0.003292 | False |
| native B vs port A | 0.174617 | 0.004053 | 0.004741 | 0.005161 | 0.006293 | 0.003292 | False |
| port A vs port B (port floor) | 0.018087 | 0.003088 | 0.003314 | 0.002618 | 0.004233 | 0.003865 | False |

The two native processes are bitwise equal in coordinates, as they were on
5SAK, so Boltz-2 has no native process floor on this case either.

`bench.boltz_amp_report` gates, `port-A` against `native-A`:

| gate | threshold | result |
| --- | --- | --- |
| coordinate | 0.05 Å entity max | not passed; the protein's 0.174617 Å is the only entity above it |
| public confidence, strict | atol 1e-4, rtol 1e-4 | not passed; 48 leaves, shapes and dtypes all equal |

Confidence maxima over the five samples: `complex_iplddt` 0.000183,
`complex_plddt` 0.000321, `confidence_score` 0.000684, `iptm` 0.003235,
`complex_pde` 0.003462, `complex_ipde` 0.003719.

### Upstream's own kernel toggle on this case

Native Boltz-2 has no process floor here (A and B are bitwise), so the scale for
the residual is upstream's own alternate implementation, as
`docs/boltz2-master-kernel-toggle-2026-09-09.md` established for 5SAK. A third
native capture ran the same tape with `use_kernels=False` (job 634, 42 s;
submitted by another agent, folded in here because this note is the case's
results record).

| pair | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native A vs native kernels-off | 0.019571 | 0.003529 | 0.003501 | 0.003377 | 0.009497 | 0.006592 |
| native kernels-off vs port A | 0.174574 | 0.002391 | 0.003288 | 0.002878 | 0.004964 | 0.004655 |
| native A vs port A | 0.174617 | 0.004053 | 0.004741 | 0.005161 | 0.006293 | 0.003292 |

Protein chain per sample, Å:

| pair | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| native A vs kernels-off | 0.010 | 0.012 | 0.006 | 0.009 | 0.020 |
| kernels-off vs port A | 0.021 | 0.015 | 0.175 | 0.010 | 0.015 |
| native A vs port A | 0.028 | 0.016 | 0.175 | 0.006 | 0.011 |

The whole of the port's 0.1746 Å is sample 3; the other four samples sit at
0.006-0.028 Å, and both port pairings carry the same sample-3 movement.
Upstream's own kernel toggle moves the protein 0.020 Å at most, on sample 5.
Unlike 5SAK, where the toggle moved the coordinates further than the port did
(2.912 Å against 2.421 Å), here it moves them less.

## Protenix

Pinned upstream `4c355be` plus the pinned `layer_norm/torch_ext_compile.py`
edit, `protenix_base_default_v1.0.0`, n=5, 200 steps, 10 recycles, seed 101,
default policy (bf16 storage, torch TF32). Two native processes (jobs 610, 611)
and two port replays of `native-A`'s RNG and MSA tapes (jobs 612, 613), port
weights `.foldjax/weights/protenix/protenix_base_default_v1.0.0.jax` with the
`foldjax-compare-002` weight audit. Table by `bench/master_three_way.py --root
<PN> --model protenix`.

The input is the one the `protenix-v2` materialization wrote, but both arms run
the `protenix_base_default_v1.0.0` checkpoint, which is the same pairing the
panel's other seven cases use; the directory name names the materialization, not
the weights.

Entity maximum RMSD, Å, one whole-system Kabsch per sample:

| case | native A vs B (floor) | native A vs port | native B vs port | port A vs B (floor) | verdict |
| --- | --- | --- | --- | --- | --- |
| protein_dna_ion_1aay | A 0.054; B 0.005; C 0.007; D 0.008; E 0.005; F 0.007 | A 0.063; B 0.041; C 0.009; D 0.005; E 0.009; F 0.007 | A 0.055; B 0.039; C 0.007; D 0.007; E 0.006; F 0.010 | A 0.055; B 0.035; C 0.010; D 0.012; E 0.010; F 0.010 | deferred |

Verdict rule, unchanged from the Protenix master panel: `pass` below 0.05 Å,
`deferred` below 0.1 Å, `at-floor` when the worst native-A-versus-port entity is
within twice the larger of the two floors, else `investigate`.

## OpenDDE

Pinned upstream `ddfa1df` (v1.1.1) at its own default `enable_tf32: True`,
n=5, 200 steps, 10 recycles, seed 101. Two native processes (jobs 614, 615),
two port processes at `--jax-matmul-precision high` (the port default, JAX TF32,
jobs 616, 617) and one at `highest` (pure fp32, job 618), all replaying
`native-A`'s tape. Table by `bench/opendde_master_table.py --root <DS>`.

Entity maximum RMSD, Å, one whole-system Kabsch per sample:

| case | native A vs B (floor) | native vs port high | port high A vs B (floor) | native vs port highest | verdict |
| --- | --- | --- | --- | --- | --- |
| protein_dna_ion_1aay | A 0.0696; B 0.0018; C 0.0020; D 0.0013; E 0.0024; F 0.0172 | A 0.0696; B 0.0019; C 0.0021; D 0.0028; E 0.0027; F 0.0168 | A 0.0360; B 0.0017; C 0.0019; D 0.0033; E 0.0015; F 0.0117 | A 0.0692; B 0.0022; C 0.0025; D 0.0035; E 0.0021; F 0.0168 | deferred |

Wall clock: native 34 and 33 s, port `high` 171 and 170 s, port `highest` 350 s.

## The four models side by side

Worst entity, in Å, against each model's own fair reference, beside the floors
that reference carries. Verdicts are the ones each panel tool emits; Boltz-2 has
no verdict column because its panel reports gates rather than a verdict.

| model | reference | worst residual | worst entity | native floor | port floor | verdict |
| --- | --- | ---: | --- | ---: | ---: | --- |
| OpenFold3 / OpenBind | native cuEq, same tape | 0.0337 | F (Zn) | 0.0640 | 0.0369 | pass |
| Boltz-2 | native A, same tape | 0.1746 | A (protein) | 0 (bitwise); kernels-off 0.0196 | 0.0181 | coordinate gate 0.05 Å not passed |
| Protenix | native A, same tape | 0.063 | A (protein) | 0.054 | 0.055 | deferred |
| OpenDDE | native A, same tape, `high` | 0.0696 | A (protein) | 0.0696 | 0.0360 | deferred |

Which entity is worst splits by model. On OpenBind, chain F (one Zn) is the
worst entity in all five port arms and in the port floor, while the protein is
worst in both native-only comparisons. On Boltz-2, Protenix and OpenDDE the
protein is the worst entity in every arm that has a nonzero value. Across all
19 comparison reports the largest ion number is 0.063092 Å (chain F, OpenBind's
unfrozen port process against native Triton), and the largest number of any
kind is Boltz-2's 0.174617 Å on the protein.

No model rejected the ion, in input translation or at run time.

## Job log

All on `master`, Slurm partition `batch` for GPU arms and `cpu` for the
diff and acceptance jobs, RTX PRO 6000 Blackwell Server Edition, one card per
job. Ledger rows in `docs/EXPERIMENTS.jsonl`.

| jobs | what |
| --- | --- |
| 592-595 | FoldJAX materialization, one per model (`openfold3`, `boltz2`, `protenix-v2`, `opendde`) |
| 596-598 | OpenBind native captures: Triton, cuEq, cuEq repeat |
| 599-602 | OpenBind port replays: vs Triton (autotune dump), vs Triton second process, vs cuEq, vs Triton unfrozen |
| 603-605 | OpenBind native floor, port floor, cross-process diff and acceptance |
| 633 | OpenBind native Triton vs native cuEq |
| 606-609 | Boltz-2 native A/B and port A/B |
| 610-613 | Protenix native A/B and port A/B |
| 614-618 | OpenDDE native A/B, port `high` A/B, port `highest` |
| 634 | Boltz-2 third native capture, `use_kernels=False` (submitted by another agent) |

## State

- Panel: `foldjax-bench/upstream-default-multimodal-n5-20260904`, now eight
  targets. Case data under `data/{jobs,sequences.json,msa,references}`; native
  inputs under `work/protein_dna_ion_1aay/foldjax/<model>/inputs/`.
- OpenBind: snapshot `openbind-cueq-master-20260909-Ieldnw` (comparison JSONs,
  `autotune/protein_dna_ion_1aay-cueq.textproto`), native captures
  `openbind-master-native-20260909/protein_dna_ion_1aay/native-{triton,cueq,cueq-repeat}`
  with `native-cueq-vs-repeat.json` and `native-triton-vs-cueq.json`.
- Boltz-2: `boltz2-master-native-20260909/protein_dna_ion_1aay/native-{A,B,nokernels}`,
  port `boltz2-master-port-20260909-IcGOFo/protein_dna_ion_1aay-port-{A,B}`,
  reports `-three-way.json`, `-port-A-vs-native-A.json` and `-kernel-toggle.json`.
- Protenix: `protenix-master-native-20260909-9yET4f/protein_dna_ion_1aay/{native-A,native-B,port-A,port-B}`.
- OpenDDE: `opendde-master-20260909-Ayx8BN/protein_dna_ion_1aay/{native-A,native-B,fj-high-A,fj-high-B,fj-highest}`.
- Job scripts and launchers for this case: `foldjax-bench/ion-1aay-master-20260909/jobs/`.

## Not run, and what the default sweep now covers

1AAY is deliberately kept in the default sweep: the mixed-modality benchmark is
meant to carry an ion case, so the manifest's eighth target is not an opt-in.

- **AlphaFold3 and ESMFold2 were not run on this case tonight.** The task scoped
  it to the four models above. A bare `run_suite.py` will now run both here.
- **Three `results/protein_dna_ion_1aay/foldjax/*.json` records are marked
  failed, and all three are fingerprint kills** (`boltz2`, `protenix-v2`,
  `opendde`), not input or model failures: see the materialization section
  above. `run_suite.py` will archive and rerun them, which is the right
  behaviour for a record whose provenance could not be sealed.
