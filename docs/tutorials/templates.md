# Tutorial: structural templates

Give a model a known structure to start from: a file you already have, the
result of a template search, or a private folder of structures that must not
leave your machine.

**Files you supply.** The jobs below name `target.a3m`, an alignment for the
protein, and `templates/1abc.cif`, a template structure whose chain B and
residue map stand in for yours; none of them ships with FoldJAX. Put your own
beside the job file, or drop the `unpaired_msa` line and run with `--msa auto`
(searches, sending the sequence to a server) or `--msa single` (folds from the
sequence alone) to try the commands first.

Which models read templates at all, and in which form, decides the rest:

| model | template input | by default |
|---|---|---|
| AlphaFold 3 | a file plus a query→template residue map | read |
| OpenFold3 | either a mapped file or a bare file it aligns itself, one form per chain | read |
| Boltz-2 | a bare file it aligns itself; a residue map is refused | read |
| Protenix, OpenDDE | a mapped file | **not read**: their released configurations set `use_template=False`; FoldJAX drops the template with a warning and records it under `ignored_templates` unless you pass `--option use_template=true` |
| ESMFold2 | none | — |

## Your own template file

A mapped template, for AlphaFold 3, OpenFold3, Protenix and OpenDDE:

```yaml
name: templated
entities:
  - type: protein
    id: A
    sequence: MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKV
    unpaired_msa: target.a3m
    templates:
      - mmcif: templates/1abc.cif
        chain_id: B                       # the template's author chain
        query_indices: [0, 1, 2, 3, 4]    # 0-based, as AlphaFold 3's queryIndices
        template_indices: [10, 11, 12, 13, 14]
```

Both index lists are **0-based** and mean the same residues for every model. A
query index counts residues of this entity's sequence; a template index counts
the template chain's full polymer sequence (`_entity_poly_seq`), unresolved
residues included. A map that runs one past the end (what a 1-based map has at
its last position) is refused rather than shifted. FoldJAX restates the map in
each model's own terms, including Protenix's and OpenDDE's count of observed
residues only.

For Boltz-2, give the file without the map; Boltz-2 aligns it itself:

```yaml
    templates:
      - mmcif: templates/1abc.cif
        chain_id: B
```

```bash
foldjax predict --model alphafold3 openfold3 --input templated.yaml --output-dir out
foldjax predict --model protenix --input templated.yaml --option use_template=true
```

Upstream Protenix allows `use_template` only for `protenix-v2` and the two
v1.0.0 base checkpoints, and so does FoldJAX.

## Searching for templates

```bash
foldjax predict --model alphafold3 --input target.yaml --msa auto --templates auto
```

`--templates auto` searches templates for every protein chain that names
none. It reads the PDB70 hits of the ColabFold MMseqs2 server's MSA job, so
**it sends the sequence to that server**; downloads each hit's mmCIF from RCSB
by PDB id; realigns the query with Kalign (`--extra templates`); and keeps what
the model's released inference would keep:

| model | release-date cutoff (default) | selection |
|---|---|---|
| AlphaFold 3 | 2021-09-30 | AlphaFold 3's filters, first 4 |
| Protenix, OpenDDE | 2021-09-30 | the same filters, 20 candidates, first 4; needs `--option use_template=true` |
| OpenFold3 | none | e-value order, first 4 |
| Boltz-2 | none (upstream Boltz-2 has no template search) | the first 4 hits whose structure has the hit chain: a FoldJAX convenience, not parity |

`--template-max-date YYYY-MM-DD` replaces the cutoff, which matters whenever
you benchmark against structures released after it. `--templates required`
searches the same way but fails the run when the search cannot run or keeps
nothing, where `auto` warns and folds without. The manifest's
`template_search` records, per chain, where the hits came from, the cutoff and
its source, every template kept and every one skipped, with the reason.

For sequences that must not leave the machine, point `FOLDJAX_TEMPLATE_COMMAND`
at a local search, or use a private folder (next). A local mirror of the PDB
(`FOLDJAX_TEMPLATE_MMCIF_DIR`) avoids the RCSB downloads
([configuration.md](../configuration.md#environment-variables)).

## A private folder of structures

```bash
foldjax predict --model alphafold3 --input target.yaml --msa auto \
    --templates path/to/my_structures/
```

`--templates DIR` searches a folder of mmCIF files (`.cif`, `.mmcif`,
`.cif.gz`, nested folders included) instead of PDB70, on this machine only,
and hands the hits to the same per-model selection as `--templates auto`. Hits
come from `mmseqs easy-search` when `mmseqs` is on `PATH`, otherwise from a
Kalign alignment of every chain (kept at 25% identity or more over at least
10 aligned residues). Without the aligner it needs, the run is refused before
anything loads, `plan` included.

No release-date cutoff applies unless you give `--template-max-date`, because
private structures usually carry no release date. The manifest records the
folder by content (`template_dir`: path, file count and a SHA-256 over every
file's name and bytes), so `--resume` reruns after the folder changes. In
Python: `PredictionRequest(templates="auto", template_dir="path/to/my_structures")`.

## Check before running

```bash
foldjax plan --model protenix --input target.yaml --templates auto \
    --option use_template=true
```

`plan` refuses `--templates auto` for ESMFold2, and for Protenix or OpenDDE
without `--option use_template=true` (drop it above to see the refusal),
before anything is searched. Without `kalign-python` (the `templates`
extra), which realigns every hit, it refuses `--templates required` and warns
that `--templates auto` will fold without templates, as the run would.
