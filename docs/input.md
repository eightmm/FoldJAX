# Input

Every format FoldJAX reads, what each backend accepts, and how alignments,
templates and binding affinity are supplied.

One file, JSON or YAML, in the common schema:

```yaml
name: example
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    unpaired_msa: msa.a3m
    modifications:
      - {ccd: SEP, position: 3}
  - {type: ligand, id: [L], ccd: ATP}
bonds:
  - [[A, 3, OG], [L, 1, PA]]
```

`--input-format auto` decides on content, so native dialects (a Boltz YAML, a
Protenix job list) pass through untouched. Modifications, covalent bonds and
`unpaired_msa` have one neutral spelling and are translated where the backend
supports them. **A field a backend cannot express is refused, with one
exception: an input the upstream model itself never reads.** A nucleic-acid
alignment or a structural template that the backend's upstream ignores (see
[alignments](#alignments) and
[templates](#templates-and-binding-affinity) below) is dropped as upstream drops it, but never
silently: FoldJAX warns and records it in `foldjax_run.json` under
`ignored_msas` or `ignored_templates`, and `ignore_nucleic_msa=false` /
`ignore_templates=false` refuse the job instead. A native OpenDDE job's
`constraint`, which upstream's inference build never reads, follows the same
rule (below). Everything else -- chemistry, bonds, modifications, affinity, a
template form the backend cannot take -- is refused, because discarding it
would change the science without changing the exit code.

The same document can be built in Python instead of written by hand:

```python
from foldjax import Job, Ligand, Modification, PredictionRequest, Protein

job = Job(
    "example",
    [
        Protein("A", "ACDEFG", unpaired_msa="msa.a3m",
                modifications=[Modification("SEP", 3)]),
        Ligand("L", ccd="ATP"),
    ],
)
request = PredictionRequest(model="protenix", input=job.write("job.json"))
```

`Job.write` emits exactly the document above; `Job.read` takes either format
back.

Sequences are normalized where the document is validated: whitespace is
removed, not trimmed, so a YAML block scalar (`sequence: |`) works; letters are
upper-cased; a nucleic-acid sequence is checked against IUPAC and points at the
offending position. Chain ids may be omitted entirely and are assigned `A`,
`B`, ... in document order — an id that is *present but blank* is still an
error, because that is a typo rather than an omission.

**Or skip the file.** A sequence is the smallest thing anyone has, and it is
enough:

```bash
uv run foldjax predict --model boltz2 --sequence MKTAYIAKQRQISFVK --ligand ATP
uv run foldjax predict --model protenix --input target.fasta   # one chain per record
uv run foldjax predict --model boltz2   --input 1abc.pdb       # re-fold a deposition
uv run foldjax predict --model boltz2   --input jobs/          # a directory is a batch
uv run foldjax predict --model boltz2   --input jobs.yaml      # so is a {jobs: [...]} file
```

A `.pdb` or `.mmcif` file is read for its **chemistry, never its coordinates**:
polymer chains keep their sequences and names, non-water heteroatoms become CCD
ligands, and the point is to predict the positions again. `.cif` needs the
explicit `structure:1abc.cif` spelling, because that suffix is also a perfectly
good name for a job document. A residue with no one-letter code is refused by
name rather than dropped — express it as a `modifications` entry instead.

### Several jobs in one file

A batch can also be one file. A mapping with a single `jobs` key holds a list
of ordinary common-schema jobs, in JSON or YAML:

```yaml
jobs:
  - name: kinase
    entities:
      - {type: protein, id: A, sequence: MKTAYIAKQR, unpaired_msa: kinase.a3m}
  - name: kinase_atp
    entities:
      - {type: protein, id: A, sequence: MKTAYIAKQR, unpaired_msa: kinase.a3m}
      - {type: ligand, id: L, ccd: ATP}
```

```bash
uv run foldjax predict --model boltz2 protenix --input jobs.yaml --msa none
```

It runs exactly as a directory holding `kinase.json` and `kinase_atp.json`
would: each job into `<out>/<model>/<job name>`, the same native input, the
same `--resume` and `--keep-going` behavior, one `foldjax_run.json` per run.
Each job is written to `$FOLDJAX_HOME/runtime/jobs/split/<digest>/<name>.json`
and run from there, with its relative `unpaired_msa`, `paired_msa` and template
paths made absolute against the jobs file's directory, so they name the files
they name in the jobs file. The digest is of that job alone: editing or moving
one job reruns that job under `--resume`, not the others.

Every job needs a `name`, and the names must be unique within the file, also
after they are made safe for a directory name (`a/b` and `a_b` collide); the
file is refused otherwise, naming both jobs. A job's own content is checked
when it runs, per model, as a file of its own would be, and the error names the
job by file, 0-based index and name (`jobs.yaml jobs[1] ('kinase_atp'): ...`).
`foldjax_run.json` records the generated document under `input.path` and the
file you wrote under `input.source` (`path`, `resolved_path`, `sha256`,
`index`, `name`); `foldjax_failures.json` and `foldjax plan` carry the same
`source`. It is provenance only and not part of the resume identity.

The container is a mapping on purpose. A top-level list is already a native
shape -- AlphaFold Server's job list and the Protenix/OpenDDE list of jobs --
and `--input-format auto` treats every top-level list as native. No native
dialect has a top-level `jobs` key (AlphaFold 3: `name`, `sequences`,
`modelSeeds`; Boltz: `version`, `sequences`; OpenFold3: `seeds`, `queries`),
so the one document cannot be mistaken for another; any other top-level key is
refused. `--input-format native` leaves the file whole. Because a jobs file is
several runs, the Python API takes it as `inputs=("jobs.yaml",)`, the spelling
a directory takes; `input=` refuses it and says so. `foldjax models
--for` reads one job and does not accept a jobs file.

`--sequence`/`--dna`/`--rna`/`--ligand`/`--ligand-smiles` and FASTA files are
turned into an ordinary common-schema job under `$FOLDJAX_HOME/runtime/jobs/`
and run through the same path as any other input, so `plan` prints the
generated file and the manifest hashes it. FASTA records become protein chains
unless `Job.from_fasta(path, kind="rna")` says otherwise: `ACGT` is a valid
protein as well as valid DNA, and guessing would fold the wrong polymer
silently. `--ligand` takes CCD codes and `--ligand-smiles` takes SMILES,
separately, because `CCO` is both a plausible CCD code and ethanol. `--name`
sets what that generated job is called in output file names; it defaults to
`job`.

### Alignments

A protein chain with no `unpaired_msa` is refused by default (`--msa none`).
No upstream but ESMFold2 folds such a chain from its sequence alone: Boltz-2
refuses the job (`boltz/main.py:581-583`), and Protenix, OpenDDE, OpenFold3
and AlphaFold 3 search for an alignment. The search is not FoldJAX's default
because it sends the sequence to a server, so the error names the three ways
on: `--msa auto`, an `unpaired_msa` path on the entity, or `--msa single` to
fold from the single sequence on purpose, which says so once per run.
ESMFold2 keeps its upstream behavior and folds the chain alone with the same
warning. Until 2026-09-30 every model folded it alone by default.

```bash
uv run foldjax predict --model openfold3 --input job.yaml --msa auto
```

`--msa auto` searches the ColabFold MMseqs2 server for every protein chain that
arrived without an alignment and caches the result under
`$FOLDJAX_HOME/msa/`, keyed by sequence and search provenance. **It sends the
sequence to a third-party server** — that is why it is opt-in and why nothing
is searched by default; point `FOLDJAX_MSA_SERVER_URL` at your own instance for
sequences that must not leave. The cache is not
per model, so running one target through three backends searches once.
`--msa required` fails instead of falling back, which is what a batch script
wants: the silent fallback it guards against is a *successful* single-sequence
run. `FOLDJAX_MSA_SERVER_URL` points at a different server, and the URL is part
of the cache identity, so two servers never read each other's alignments.
For sequences that must not leave the machine — and for RNA, which no public
endpoint answers — point FoldJAX at a locally installed search instead:

```bash
export FOLDJAX_MSA_COMMAND="/opt/msa/run.sh"          # protein
export FOLDJAX_RNA_MSA_COMMAND="/opt/msa/rna.sh"      # RNA
export FOLDJAX_MSA_LOCAL_VERSION="uniref-2026-06"     # part of the cache identity
```

Each wrapper is called as `<command> --input query.fasta --output DIR` and
writes `pairing.a3m` and `non_pairing.a3m` (`rna_msa.a3m` for the RNA one).
Nothing is sent anywhere on this path. The version string is part of the cache
key, so upgrading a database invalidates the alignments it produced instead of
mixing generations. `foldjax doctor` prints which search is configured.

Protenix 2.0.0 and OpenDDE both release `use_rna_msa=False`
(Protenix `configs/configs_inference.py:37`, `--use_rna_msa` default false).
With it false, upstream never opens an RNA chain's alignment and says nothing
about it. A FoldJAX request keeps that default: an RNA `unpaired_msa` is left
out of the native input, as upstream leaves it unread, with a warning and a
manifest record (below). An RNA `paired_msa` is refused for either model, since
neither dialect can carry one. Set `options={"use_rna_msa": true}` (CLI:
`--option use_rna_msa=true`) to read the RNA alignment, as upstream's flag
does; protein MSAs are unaffected. Protenix, like upstream, allows the flag
only for `protenix-v2` and the two v1.0.0 base models. Native Protenix and
OpenDDE input follow the same rule for `unpairedMsaPath` (and the
inline `unpairedMsa`): without the option the alignment is ignored with a
warning. The native `protenix-jax-predict` spells it `--use-rna-msa`, which
`--rna-msa-local-command` now requires.

The same rule applies to every nucleic-acid alignment a backend would discard.
RNA `unpaired_msa` is read by AlphaFold 3 and OpenFold3, and by Protenix and
OpenDDE with `use_rna_msa=true`. No backend reads a DNA one. Boltz-2 and
ESMFold2 read neither. A common-schema job that gives such an alignment to a backend
that ignores it folds the chain without it, as that upstream does, but not
silently: the alignment is left out of the native input, a `UserWarning` names
the chain, the field and the file, and the run manifest lists it under
`ignored_msas`. `ignore_nucleic_msa` (default `true`) governs this; set
`--option ignore_nucleic_msa=false` to refuse such a job instead, with the
entity and the backend named. AlphaFold 3's own parser already refuses a DNA
alignment, so the option does not apply there. `--msa auto` searches RNA
chains only for the backends that read the result.

A nucleic-acid `paired_msa` follows the same rule. OpenFold3 reads an RNA one
and no backend reads a DNA one. Protenix, OpenDDE and OpenFold3 would discard
a DNA one, so it is dropped with a warning and recorded under `ignored_msas`,
or refused under `ignore_nucleic_msa=false`. Boltz-2 and ESMFold2 cannot
express a `paired_msa` at all, Protenix and OpenDDE refuse an RNA one (above),
and AlphaFold 3's parser refuses either, so the option does not turn those
refusals into a drop.

### Templates and binding affinity

```yaml
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    templates:
      - mmcif: templates/5xyz.cif
        chain_id: B                # the template's author chain
        query_indices: [1, 2, 3]   # 0-based, as AlphaFold 3's queryIndices
        template_indices: [7, 8, 9]
  - {type: ligand, id: [L], ccd: ATP}
properties:
  - affinity: {binder: L}
```

The two forms of a template are different inputs, not one input in two
spellings. AlphaFold 3, Protenix and OpenDDE require the query→template residue
map and refuse a bare file; **Boltz-2 aligns the mmCIF itself** and refuses a
map it would have to ignore; OpenFold3 takes either form, one form per chain.

Both index lists are **0-based**, exactly AlphaFold 3's `queryIndices` and
`templateIndices`, and reach AlphaFold 3 and Protenix verbatim. A query index
counts residues of the entity's sequence; one past its end -- what a 1-based
map has at its last position -- or a negative one is refused rather than
shifted. A template index counts the template chain's full polymer sequence
(`_entity_poly_seq`), unresolved residues included, as AlphaFold 3 and
OpenFold3 read it. `chain_id` is the template's author chain
(`auth_asym_id`); Boltz-2 and OpenFold3 address chains by `label_asym_id`, so
FoldJAX looks the label id up in the file for them. Protenix and OpenDDE read
the *first* chain of the file and count only its observed residues (upstream
`parse_simple_cif`), ignoring `chain_id`; the two counts agree only for a
single-chain file in which every residue is resolved. Give those two such a
file -- it is what `--templates auto` writes for them.

Protenix and OpenDDE have native template machinery,
but both released inference configurations set `use_template=False` (Protenix
`configs/configs_inference.py:36`, OpenDDE `config/inference_defaults.py:28`)
and then ignore a job's templates. FoldJAX preserves that default and folds
without common-schema templates as upstream does, but not silently: a
`UserWarning` names the chain and the file, and the run manifest lists each
dropped template under `ignored_templates` (null for native input, which is not
inspected). Set `options={"use_template": true}` (CLI:
`--option use_template=true`) to materialize the mapped templates and run the
template path, or `--option ignore_templates=false` to refuse such a job
instead. `use_template=true` with an explicit `ignore_templates=true` is
refused as contradictory. Native `templatesPath` follows the same opt-in rule
and is ignored with a warning without it. Upstream Protenix allows
`use_template` only for `protenix-v2` and the two v1.0.0 base models, and so
does this port; `--template-search-command` requires it.
Exact checked parity uses native Kalign 3.3.5; newer wrapper builds are not
assumed alignment-equivalent. Affinity
reaches Boltz-2 alone — it is the only carried model with that head. OpenFold3
has no per-template residue-map field in its query, so a mapped template is
written as the template cache its reader loads (`template_alignment_file_path`
plus `template_entry_chain_ids`, `idx_map` against `label_seq_id`), and a bare
file goes to `template_cif_paths`, which OpenFold3 aligns and ranks itself
(upstream's CIF-direct mode). A chain mixing the two forms, or a template on a
nucleic-acid chain, is refused: OpenFold3 reads one source per protein chain.
`foldjax capabilities --model MODEL
[--json]` reports both `common_schema_features` and `native_only_features` for
exactly this reason, generated from the same translation table the writer uses.
`native_only_features` also names what only a native input can reach: ligands
of several CCD components (glycans; not OpenFold3 or ESMFold2), AlphaFold 3's
user-defined CCD entries, ligands read from a file (Protenix, OpenDDE), pocket and contact
constraints (Boltz-2, Protenix) and cyclic polymers (Boltz-2, OpenFold3). The
common schema has no field for any of them; pass the model's native file
instead.

OpenDDE is absent from that constraint list on purpose. It shares Protenix's
native dialect and featurizer, but its model has no constraint embedder, and
upstream's inference build warns and ignores a job's `constraint` (OpenDDE
1.1.1 `opendde/data/inference/json_to_feature.py:28-32`, and its
`docs/infer_json_format.md`, "Unsupported `constraint`"). A native OpenDDE job
that carries one is therefore folded without it, as upstream folds it, but not
silently: the featurizer drops the field before the shared Protenix code can
build a `constraint_feature` from it, warns, and the run manifest lists the job
under `ignored_constraints` (an empty list when no job had one, null for every
other backend and for common-schema input, which cannot carry a constraint).
`--option ignore_constraints=false` refuses such a job instead, at `plan` as
well as `predict`. The option governs native input only: `true` on a
common-schema job is refused as meaningless. Covalent links reach OpenDDE
through `covalent_bonds` (the common `bonds`), as upstream says.

From a bare sequence the same affinity request is `--affinity-binder CHAIN`,
naming which chain of the generated job to score. It reaches Boltz-2 alone, for
the same reason the `properties` block does: the others have no such head and
refuse it rather than dropping it.

### Searching for templates

```bash
uv run foldjax predict --model alphafold3 --input job.yaml --msa auto --templates auto
```

Upstream AlphaFold 3 and OpenFold3 search for templates by default; Protenix
and OpenDDE ship a search that is off by default; FoldJAX searched for none
until `--templates auto` (`PredictionRequest(templates="auto")`). The default
stays `none` for the reason `--msa auto` is opt-in: **the search sends the
sequence to the ColabFold MMseqs2 server** (`FOLDJAX_MSA_SERVER_URL` points at
your own). For every protein chain that names no `templates`, it reads the
PDB70 hits (`pdb70.m8`) of the server's ordinary MSA job, as OpenFold3 v0.5.0
does, downloads each hit's mmCIF from RCSB by PDB id, realigns the query to
the hit chain with Kalign (the `openfold3-preprocess` extra's `kalign-python`),
and attaches what the selected model's released inference would keep:

| model | release-date cutoff (default) | selection | form |
|---|---|---|---|
| `alphafold3` | 2021-09-30, a hit released after it dropped (`run_alphafold.py:295-297`) | AlphaFold 3's filters: subsequence > 0.95, < 10 residues, alignment ≤ 0.1 of the query, no resolved aligned residue, duplicates; first 4 (`data/pipeline.py:456-462`) | mapped |
| `protenix` | 2021-09-30 (`infer_dataloader.py:107`) | the same filters, 20 candidates, first 4 | mapped, one observed-residue chain per file |
| `opendde` | 2021-09-30 (`infer_dataloader.py:196`) | as Protenix | as Protenix |
| `openfold3` | none (`TemplatePreprocessorSettings.max_release_date=None`); a set cutoff drops a hit released on or after it (`template.py:2698`) | e-value order, no sequence filters, first 4 (`n_templates=4`) | mapped, as a template cache |
| `boltz2` | none: upstream Boltz-2 has no template search | first 4 hits whose structure has the hit chain -- a FoldJAX convenience, not parity | the file; Boltz aligns it |

Hits are taken in e-value order. `--template-max-date YYYY-MM-DD`
(`template_max_date`) replaces the default cutoff, compared the way that
upstream compares it; it is refused without `--templates auto`. Protenix and
OpenDDE refuse `--templates auto` unless `--option use_template=true` is set,
since at their released default the result would be discarded, and ESMFold2,
which has no template input, refuses it outright -- both at `foldjax plan`
already. So is `--templates auto` on a native document, which is passed to
the backend untouched and so would search nothing; the search applies to
FoldJAX-format jobs. A chain that names its own templates keeps exactly those.
AlphaFold 3 also skips a hit whose author chain spans several polymer chains,
since its template file must hold exactly one.

The Protenix and OpenDDE selection approximates upstream's rather than
reproducing it: upstream ranks hmmsearch or HHsearch hits by `sum_probs` and
applies its filters to that search's own alignment, while these hits come
from MMseqs2 in e-value order and are filtered on the Kalign realignment. A
hit whose release date is unknown is dropped, as upstream's prefilter drops
one missing from its release-date table (`template_utils.py:369-372`); here
the date is the mmCIF's earliest `_pdbx_audit_revision_history` revision. The
same filters on the Kalign span stand in for AlphaFold 3's hmmsearch hit.

The hits are cached under `$FOLDJAX_HOME/templates/hits/`, keyed by sequence
and server like the alignment cache, and the structures under
`$FOLDJAX_HOME/templates/mmcif/`, one directory per download source so two
servers never share a file. Only PDB ids go to RCSB.
`FOLDJAX_TEMPLATE_MMCIF_DIR` is read first: a flat or wwPDB-divided
(`ab/1abc.cif.gz`) mirror of `.cif` or `.cif.gz` files, a compressed one
unpacked once into the cache. `FOLDJAX_TEMPLATE_STRUCTURE_URL` changes the
download source and an empty value turns downloading off. For sequences that must not leave the machine,
`FOLDJAX_TEMPLATE_COMMAND` names a local search, called as
`<command> --input query.fasta --output DIR` and writing `DIR/pdb70.m8` (hits
named `<pdb id>_<author chain>`); `FOLDJAX_TEMPLATE_LOCAL_VERSION` is part of
its cache identity. `foldjax doctor` prints what a search would use.

A search that cannot run -- no server, no Kalign -- warns and folds that
chain without searched templates, and so does one whose hits were all dropped
(by the cutoff, the filters, or a download or realignment that failed); the
warning and the record name the reasons with their counts. It never fails the
job. Kalign is checked before anything is sent.
`foldjax_run.json` records `templates`, `template_max_date` and
`template_search`: per chain, where the hits came from, the cutoff applied and
the line it comes from, the selection rule, the hits file, every template kept
(PDB id, author and label chain, release date, e-value, file) and what was
skipped, or the error. The same record is written as `template_search.json`
beside the generated native input. With `--msa auto` as well, the same
sequence is submitted to the server twice, once per search.
