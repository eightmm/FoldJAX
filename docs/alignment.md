# Structure alignment

FoldJAX can place predictions from several models in one rigid coordinate frame
without making a second copy of every structure. Alignment is post-processing:
it never changes predictions, run manifests, scores, or resume identities.

```python
import foldjax

report = foldjax.align_predictions(
    batch.results,
    reference="openfold3/seed-0/sample-00",
)

for item in report.alignments:
    print(item.key, item.rmsd, item.matched_atoms, item.coverage)
```

An external deposited structure can be the reference instead:

```python
report = foldjax.align_predictions(
    batch.results,
    reference="target.cif",  # .pdb and .mmcif are accepted too
    chain_map={"A": "C", "B": "D"},
)
```

`chain_map` is optional. Matching prefers compatible identical chain ids, then
unique exact sequence or ligand matches. A homomer permutation is not guessed:
provide a map when several reference chains are equally valid.

## What is fitted

The default `selection="representative"` uses protein `CA` and DNA/RNA `C4'`
atoms. A ligand-only structure uses common named heavy atoms. Mixed complexes
are fitted on their polymers and carry ligands along in the same transform, so
the complex is never distorted merely to improve ligand overlap.

`selection="backbone"` uses polymer backbone atoms and `selection="heavy"`
uses common named heavy atoms. Every mode performs one reflection-free Kabsch
fit for the entire structure. There is no outlier rejection or independent
per-chain movement.

RMSD is therefore a property of the reported correspondence, not a cross-model
accuracy or confidence score. Always inspect `matched_atoms`, `coverage`, and
`chain_map` with it.

## Storage

Calling an alignment function writes nothing. The returned report contains one
3x3 rotation, one translation, provenance hashes, and fit metrics per structure.
Persisting it is explicit and normally takes only a few kilobytes:

```python
report.write_json("alignment.json")
restored = foldjax.AlignmentReport.read_json("alignment.json")
```

Transformed mmCIF is generated in memory when needed:

```python
cif_text = report.aligned_cif("protenix/seed-0/sample-00")
```

Materializing coordinates is also explicit:

```python
report.write_aligned(
    "protenix/seed-0/sample-00",
    "protenix-aligned.cif",
)
```

For an mmCIF source, FoldJAX edits only `_atom_site.Cartn_x/y/z` in a copied CIF
document. ModelCIF and model-specific categories remain present. Before applying
a saved transform, the source SHA-256 is checked so the transform cannot be
silently applied to a different structure.
