# Outputs

What a run writes, file by file, and how to read it back. Every model writes
the same layout; the field-by-field contract is in
[cli.md](cli.md#outputs), and the JSON Schemas ship in the package as
`foldjax/schemas/run.schema.json` and `foldjax/schemas/confidence.schema.json`
(`foldjax.summary.load_schema`).

## The output tree

```
<run>/
├── foldjax_run.json                                   the run manifest; written last, so it also marks the run finished
├── seed-<seed>_sample-<nn>/
│   ├── <job>_seed-<seed>_sample-<nn>.cif              the structure (mmCIF), pLDDT 0-100 in B_iso_or_equiv
│   ├── <job>_seed-<seed>_sample-<nn>.pdb              only with --structure-format pdb|both
│   ├── confidence.json                                scores for this sample
│   ├── confidence_full.npz                            per-sample confidence arrays, where the model returns them
│   └── predicted_aligned_error.json                   PAE for viewers, where the model returned PAE
├── seed_<seed>/                                       with several seeds: that seed's native files and manifest
├── inputs/                                            the job as translated for the model (a common-schema input only)
├── .foldjax.lock                                      held while a run writes here; kept on purpose (see below)
└── ...                                                whatever the model's own writer produced, left where it wrote it
                                                       (Boltz-2: msa/, predictions/, processed/)
```

A single run's default `<run>` is `foldjax-outputs/<input stem>`;
`--output-dir` replaces it. A batch (several models, several inputs, a
directory, or a `{"jobs": [...]}` file) puts each pair in its own run:

```
<output-dir>/<model>/<input stem or job name>/...
<output-dir>/foldjax_failures.json                     only when a pair failed under --keep-going
```

`.foldjax.lock` is how a second process pointed at the same directory is
refused instead of interleaving its files with the first's. The file is never
deleted, because removing a lock file lets a third process lock a fresh one
while the second still holds the old. When no run holds it, it is inert and
safe to delete.

A run into a directory that already holds a finished run replaces that run's
`foldjax_run.json` before it writes anything, and deletes nothing else. When
the earlier run was another model or seed, its sample directories (and a
`foldjax_report.html` or `compare/` made from it) stay beside the new ones,
and a warning says so: give each run its own `--output-dir` to keep runs
apart. `--resume` reuses a finished run that matches the request instead.

`<nn>` is the diffusion sample index, zero-padded, counting from `00` within
each seed. It is never a rank: Protenix and OpenDDE name their native files by
rank, and that rank is kept as `native_rank`. A native input holding several
jobs (an AlphaFold 3 or Protenix job list) nests each job one level down,
under `<run>/<job>/`.

## The structure file

The mmCIF is the record: the manifest's SHA-256 and the confidence arrays'
atom axis refer to it. It always carries the model's pLDDT (0-100) per atom in
`B_iso_or_equiv`; FoldJAX checks that column against `confidence_full.npz`
and fills it only where a native writer left something else. It also carries
ModelCIF confidence records: `_ma_qa_metric` pLDDT `global` (the
whole-structure pLDDT of the common summary) and `local` (each residue's mean
per-atom pLDDT), and `_software` rows for FoldJAX and the upstream model with
their versions. Records a native writer already wrote, such as AlphaFold 3's
own metrics, are kept and only the missing ones added.

`--structure-format pdb` or `both` writes a `.pdb` beside the mmCIF. PDB
cannot hold more than 99,999 atoms, multi-character chain ids, residue names
longer than three characters or residue numbers outside -999..9999; such a
structure is refused rather than written in a non-standard dialect.

## `confidence.json`

One per sample. Schema version `1.0`.

| field | meaning |
|---|---|
| `schema_version` | `"1.0"`; a reader of `1.x` accepts any `1.y` |
| `model`, `seed`, `sample`, `native_rank`, `job` | which structure this is |
| `scores` | the model's own scalar scores, under the model's own names, unchanged (flags such as `has_clash` as 0/1) |
| `summary` | `plddt`, `ptm`, `iptm` and `ranking` on one name and one scale |
| `execution` | how the run executed, where the native summary carries it (`num_recycles`) |
| `score_notes` | a native writer's own note on its scores, verbatim |
| `pocket_sampling` | only under `--option pocket_sampling=select`: whether this sample satisfied every pocket of the job, with each listed residue's smallest heavy-atom distance to the binder ([cli.md](cli.md#--option-pocket_samplingselect)) |

Each `summary` field is `{value, scale, source, transform, granularity,
population}`, saying which native number it came from and what was done to it,
or `{value: null, reason}` when the model does not report it or it does not
apply. A missing value is never written as 0: a single-chain structure's
`iptm` is null even where the model wrote 0.0. `plddt` is 0-100; `ptm` and
`iptm` are 0-1. `ranking` names the score the model ranks its own samples by:

| model | `ranking.key` | `plddt` source |
|---|---|---|
| AlphaFold 3 | `ranking_score` | mean of the per-atom pLDDT in the structure |
| Boltz-2 | `confidence_score` | `complex_plddt` ×100 |
| ESMFold2 | `plddt` (FoldJAX's choice; upstream ranks nothing) | `complex_plddt` ×100 |
| OpenDDE | `ranking_score` | `plddt` |
| OpenFold3 | `sample_ranking_score` (needs biotite for proteins; else null, with the reason) | `mean_plddt` |
| Protenix | `ranking_score` | `plddt` |

> Common fields standardize names and numerical scales. They retain
> model-specific definitions and calibration and do not establish comparable
> accuracy probabilities or authorize pooled cross-model ranking.

A pLDDT of 80 from one model and 80 from another are not the same statement,
and FoldJAX never ranks or averages confidence across models. Accuracy against
a deposited structure *is* comparable across models: `foldjax compare DIR
--reference X.cif` ([cli.md](cli.md#analysis-and-workflow-commands)).

## `confidence_full.npz` and `predicted_aligned_error.json`

`confidence_full.npz` holds the per-sample arrays the model's compiled program
returns, with their index maps (token and atom to chain and residue), units and
axes in a `_meta` record; FoldJAX computes no confidence quantity of its own.
Read it with:

```python
import foldjax

arrays = foldjax.load_confidence_arrays("out/seed-42_sample-00")
pae = arrays["pae"].astype("float32")   # [token, token] in angstroms, stored as float16
print(arrays.meta["arrays"]["pae"]["axes"], arrays.unavailable)
```

What a default run writes, per model (`foldjax capabilities --model M` lists
the same under `confidence_arrays`):

| model | default arrays | to add or drop PAE/PDE |
|---|---|---|
| AlphaFold 3 | `pae`, `pde`, `contact_probs`, `atom_plddt`, `chain_ptm`, `chain_iptm`, `chain_pair_iptm`, `chain_pair_pae_min`, `chain_pair_pde_min`, `chain_pair_pde_mean` | — |
| Boltz-2 | `pae`, `pde`, `token_plddt`, `chain_ptm`, `chain_pair_iptm` | — |
| ESMFold2 | `pae`, `pde`, `token_plddt`, `atom_plddt`, `chain_pair_iptm` | drop: `--option return_expected_errors=false` |
| OpenFold3 | `pae`, `pde`, `atom_plddt`, `chain_ptm`, `chain_pair_iptm`, `chain_pair_iptm_bespoke` | drop: `--option return_expected_errors=false` |
| Protenix | `atom_plddt`, per-chain and chain-pair pTM/ipTM/pLDDT/gPDE, chain-pair PAE mean and minimum | add: `--option output_format=both` |
| OpenDDE | `atom_plddt`, per-chain and chain-pair pTM/ipTM/pLDDT/gPDE | add: `--option include_raw=true` |

The manifest's `confidence_arrays` says, per run and per sample, which arrays
are present and why any other is absent. Dropping PAE/PDE on ESMFold2 or
OpenFold3 also drops `predicted_aligned_error.json` and, on OpenFold3, the
`gpde` score; it compiles a separate program
([`docs/cli.md`](cli.md#--option-return_expected_errorsfalse-esmfold2-openfold3)).

Wherever a sample has PAE, `predicted_aligned_error.json` sits beside it in
AlphaFold DB's schema (`predicted_aligned_error`,
`max_predicted_aligned_error`, 31.75 Å for all six models), which PAE viewers,
Mol\* and ChimeraX read as they read an AlphaFold DB entry. Interface scores
computed from PAE (ipSAE, pDockQ, pDockQ2, LIS) come from
`foldjax interfaces DIR` and are labelled `derived`, apart from the model's own
chain-pair ipTM (`native.chain_pair_iptm`).

## `foldjax_run.json`

The run manifest, schema version `1.0`. Two version fields: `schema` (an
integer, what makes a run safe to resume) and `schema_version` (this file
contract, shared with `confidence.json`). `msa_search`, `weights.kind`,
`weights.stat_signature`, AlphaFold 3's per-sample `metadata.native_sample`,
its `kernel_tuning`, and the `pocket_sampling=select` fields (`pocket_sampling`,
`constraints[].route`, `best.pocket_satisfied`, per-sample
`metadata.pocket_sampling`) are declared as optional fields, so a manifest written before the schema named
them still validates and still resumes.

| field | what it records |
|---|---|
| `foldjax`, `foldjax_source` | the FoldJAX version, and `git describe` of the checkout it was imported from |
| `finished` | when the run finished (UTC) |
| `model` | the canonical model name |
| `input` | the input path, format and SHA-256; `input.source` names the multi-job file and job it came from |
| `input_dependencies` | identities of the files the job names (alignments, templates), which `--resume` compares |
| `weights` | path, profile, label and the checkpoint's identity |
| `seeds`, `seed_source` | the seeds run, and where they came from: `user`, `upstream`, `job`, `random` or `foldjax` |
| `msa`, `msa_search` | the alignment policy, and per searched chain the files it got or the error that left it on its single sequence |
| `msa_pairing` | how a searched complex was paired: `requested`, `resolved`, the ColabFold `mode`, and `paired_by` (`row`, `species` or null; `species` means no row beyond the query is paired) |
| `msa_stats` | per chain, the depth (rows, query included) and Neff at 80% identity of each alignment the model read, with the definition |
| `templates`, `template_max_date`, `template_search` | the template policy, the cutoff, and per chain where hits came from, what was kept and what was skipped |
| `template_dir` | with `--templates DIR`: the folder's path, file count and a SHA-256 over its files |
| `preset` | with `--preset`: its name, the sampling it set and the publisher's source |
| `ignored_msas`, `ignored_templates`, `ignored_constraints` | inputs the job named that the model never read, as its upstream does not |
| `constraints` | pocket and contact constraints as the native input carries them, with `max_distance_source`; under `pocket_sampling=select` each pocket's `route`, `native` or `selection` |
| `pocket_sampling` | only under `--option pocket_sampling=select`, FoldJAX's own route: how many samples were scored against the pocket and satisfied it, whether the native input conditioned on the pocket as well, and the pocket records; `best.selection` then says the satisfied samples were ranked first and `best.pocket_satisfied` whether one was found |
| `sampling`, `options`, `options_verifiable` | the schedule knobs and native options as run |
| `padding`, `shape_profile` | the padding request and the concrete shapes executed |
| `kernel_tuning` | AlphaFold 3 only: the `kernel_autotuning` value, whether the persistent Tokamax store was installed, and per source (`store`, `measured`, `tokamax`) how many model calls took their kernel configurations from it -- the option is a cache-miss policy, so it alone does not say which program ran |
| `runtime` | JAX, jaxlib, platform and device identity |
| `cost` | what the run cost (below) |
| `memory` | what the memory policy predicted before the run, beside `cost.peak_bytes` |
| `samples` | every sample: scores, structure path and SHA-256, metadata |
| `confidence_arrays` | the per-sample confidence arrays written, and why others are absent |
| `best` | the sample this model ranks first, by its own ranking score |

### `cost`

```json
"cost": {
  "seconds": 381.2,
  "peak_bytes": 19756849152,
  "phases": {"prepare input": 1.2, "predict": 378.0, "write": 0.3},
  "breakdown": {
    "seconds": {"weight load": 6.1, "featurize": 2.4, "trace": 9.8, "lower": 7.5,
                "compile": 61.0, "cache restore": 0.0, "execute and host": 291.2},
    "counts": {"programs": 3, "cache hits": 0, "cache misses": 3}
  }
}
```

(The numbers above are illustrative.) `seconds` is the whole run;
`peak_bytes` is the device allocator's high-water mark, not the preallocated
pool `nvidia-smi` shows, and is null where the device reports no allocator
statistics (a CPU). `phases` are the stages printed
on stderr and sum to the run. `breakdown` splits the time *inside* them:
`weight load`, `featurize` and `language model` where a backend marks them,
the compiler's `trace`, `lower`, `compile` (XLA compile time with the
persistent-cache reads taken out) and `cache restore` (those reads), and
`execute and host`, which is the `predict` phase less every part recorded
inside it: device execution plus unnamed host work, derived rather than
measured. `counts` gives the programs compiled and the persistent-cache hits
and misses. A second run of the same shape should show `cache hits` and almost
no `compile`; if it does not, the shape, the weights or a compile-relevant
option changed.

## `foldjax_failures.json`

Written beside the runs of a batch run with `--keep-going` when any pair
failed, and removed again by a later pass in which nothing failed. One record
per failure: `model`, `input`, `seed`, `output_dir`, `error_type`, `error`,
and `source` for a job from a multi-job file. The exit status is 3.

## Reading a directory back

```bash
foldjax show out/                                  # per-run summary, for reading
foldjax show out/ --format csv > runs.csv          # one row per model/input/seed/sample, failures included
foldjax show out/ --format json --aggregate        # count, median, spread per input/model/configuration
foldjax show out/ --rank-by iptm                   # samples ordered within each model only
foldjax compare out/                               # pairwise RMSD and coverage within each input
foldjax report out/                                # one static HTML page: scores, pLDDT plot, PAE heatmap
```

`foldjax.load_results(root)` and `foldjax.results_table(...)` return the rows
`show --format csv|json` prints. All of these read only the canonical files
above, never a backend's native side files.
