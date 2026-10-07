# Tutorial: a glycoprotein

Fold a protein carrying an N-linked glycan: a ligand chain made of several
CCD components, bonded to an asparagine and to each other.

**Files you supply.** The jobs below name `glycoprotein.a3m`, an alignment for
the protein; none of them ships with FoldJAX. Put your own beside the job file,
or drop the `unpaired_msa` line and run with `--msa auto` (searches, sending
the sequence to a server) or `--msa single` (folds from the sequence alone) to
try the commands first.

## The job

`glycoprotein.yaml`:

```yaml
name: glycoprotein
entities:
  - type: protein
    id: A
    sequence: MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE
    unpaired_msa: glycoprotein.a3m
  - type: ligand
    id: G
    ccd: [NAG, NAG, BMA]
bonds:
  - [[A, 41, ND2], [G, 1, C1]]   # Asn 41 (an N-Y-T sequon) to the first GlcNAc
  - [[G, 1, O4], [G, 2, C1]]     # GlcNAc beta-1,4 GlcNAc
  - [[G, 2, O4], [G, 3, C1]]     # GlcNAc beta-1,4 Man
```

- A ligand's `ccd` may be a **list**: one chain of several components, in
  order. A single code (`ccd: NAG`) is an ordinary one-component ligand.
- In `bonds`, the glycan's residues are numbered from 1 by their position in
  that list, as AlphaFold 3 numbers them; protein residues are numbered from 1
  along the sequence. Each bond end is `[chain, residue, atom name]`, with the
  atom named as in the component's CCD definition.
- The bonds inside the glycan and the one to the protein are ordinary common
  `bonds`. Where the shared CCD is installed (`foldjax setup` fetches it),
  FoldJAX checks each code and each bonded atom against it before any model
  loads, and refuses a code or an atom that does not exist.

```bash
foldjax models --for glycoprotein.yaml
foldjax predict --model boltz2 protenix alphafold3 --input glycoprotein.yaml --output-dir out
```

## Which models take it

| model | what the glycan becomes |
|---|---|
| AlphaFold 3 | `ccdCodes` plus `bondedAtomPairs`, its own spelling |
| Boltz-2 | a `ccd` list plus `bond` constraints |
| Protenix, OpenDDE | one ligand `CCD_NAG_NAG_BMA` plus `covalent_bonds` (a code containing `_` is refused, since their ligand string splits on it) |
| ESMFold2 | Biohub's own CCD-list ligand input, one token per atom; a covalently bonded chain drops every residue's leaving atoms |
| OpenFold3 | refused: OpenFold3 v0.5.0 builds a ligand chain from one code and raises `NotImplementedError` for more |

The exact translation per model is in
[input.md](../input.md#multi-residue-ligands-glycans).

## AlphaFold 3's standalone-glycan option

AlphaFold 3's training and evaluation removed leaving atoms from glycan
ligands even when they were bonded to nothing ("standalone" glycans).
Upstream's `--fix_standalone_glycans`, off by default, stops that, and its own
help says this moves away from the regime AlphaFold 3 was trained and
evaluated in. FoldJAX passes it through:

```bash
foldjax predict --model alphafold3 --input glycans_only.yaml \
    --option fix_standalone_glycans=true
```

## Reading the result

The glycan is one ligand chain in the output mmCIF, with its residues in the
order of the `ccd` list. Look at the bonded geometry, not only the confidence:

```bash
foldjax check out/                 # PoseBusters, with the extra installed
```

Per-atom pLDDT for the glycan is in the B-factor column and, for models that
return it, in `confidence_full.npz` (`atom_plddt`); like every confidence
value it is calibrated per model and not comparable across them.
