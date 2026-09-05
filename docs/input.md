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
supports them; **a field a backend cannot express is rejected, not dropped** — silently
discarding an alignment would change the science without changing the exit
code.

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
```

A `.pdb` or `.mmcif` file is read for its **chemistry, never its coordinates**:
polymer chains keep their sequences and names, non-water heteroatoms become CCD
ligands, and the point is to predict the positions again. `.cif` needs the
explicit `structure:1abc.cif` spelling, because that suffix is also a perfectly
good name for a job document. A residue with no one-letter code is refused by
name rather than dropped — express it as a `modifications` entry instead.

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

A protein chain with no `unpaired_msa` is folded from its single sequence. That
is unchanged and still the default — but it now says so, once per run, instead
of being the invisible difference between a good prediction and a poor one.

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

OpenDDE's released inference configuration sets `use_rna_msa=False`. A
FoldJAX request preserves that default and rejects an RNA `unpaired_msa` or
`paired_msa` instead of materializing a file the model will discard. Set
`options={"use_rna_msa": true}` (CLI:
`--option use_rna_msa=true`) to opt into OpenDDE 1.1.1's RNA-MSA path; protein
MSAs remain supported independently. Native OpenDDE input follows the same
rule for `unpairedMsaPath`.

### Templates and binding affinity

```yaml
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    templates:
      - mmcif: templates/5xyz.cif
        query_indices: [1, 2, 3]
        template_indices: [7, 8, 9]
  - {type: ligand, id: [L], ccd: ATP}
properties:
  - affinity: {binder: L}
```

The two forms of a template are different inputs, not one input in two
spellings. AlphaFold 3 and Protenix require the query→template residue map and
refuse a bare file; **Boltz-2 aligns the mmCIF itself** and refuses a map it
would have to ignore. OpenDDE has native template machinery, but its released
inference configuration sets `use_template=False`: FoldJAX preserves that
default and rejects common-schema templates instead of silently dropping them.
Set `options={"use_template": true}` (CLI:
`--option use_template=true`) to materialize the mapped templates and run the
v1.1.1 template path. Native `templatesPath` follows the same opt-in rule.
Exact checked parity uses native Kalign 3.3.5; newer wrapper builds are not
assumed alignment-equivalent. Affinity
reaches Boltz-2 alone — it is the only carried model with that head. OpenFold3
builds template features from its own pipeline and has no per-job field, so a
template addressed to it is refused. `foldjax capabilities --model MODEL`
reports both `common_schema_features` and `native_only_features` for exactly
this reason.

From a bare sequence the same affinity request is `--affinity-binder CHAIN`,
naming which chain of the generated job to score. It reaches Boltz-2 alone, for
the same reason the `properties` block does: the others have no such head and
refuse it rather than dropping it.
