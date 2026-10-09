# Tutorial: a protein-ligand complex with constraints

Fold a protein with a ligand, tell the model where the ligand binds, ask
Boltz-2 for a binding affinity, and check the poses. The residue numbers below
are placeholders: use your own pocket.

**Files you supply.** The jobs below name `target.a3m`, an alignment for the
protein (and the last section a deposited structure, `deposited.cif`); none of
them ships with FoldJAX. Put your own beside the job file, or drop the
`unpaired_msa` line and run with `--msa auto` (searches, sending the sequence
to a server) or `--msa single` (folds from the sequence alone) to try the
commands first.

## The job

`target_atp.yaml`:

```yaml
name: target_atp
entities:
  - type: protein
    id: A
    sequence: MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKV
    unpaired_msa: target.a3m
  - {type: ligand, id: L, ccd: ATP}          # or smiles: "..." for a ligand with no CCD code
constraints:
  - pocket: {binder: L, contacts: [[A, 38], [A, 41], [A, 45]], max_distance: 6.0}
```

A **pocket** says the `binder` chain should sit near the listed residues.
`contacts` are `[chain, residue]` pairs, 1-based like `bonds` and
`modifications`, on chains other than the binder. `max_distance` (Å) is
optional; omitted, each model runs its own upstream default and the manifest
records which (`max_distance_source: job` or `upstream`).

```bash
foldjax models --for target_atp.yaml        # which models read it, and why the others do not
foldjax models --for target_atp.yaml --profile base-constraint-v0.5.0   # Protenix's constraint checkpoint
foldjax predict --model boltz2 openfold3 --input target_atp.yaml --output-dir out
```

## What each model does with a pocket

The same field does not mean the same thing to every model:

| model | what the pocket becomes | omitted `max_distance` |
|---|---|---|
| Boltz-2 | a conditioning input: the trunk is told the binder sits within that distance of the contacts; several pockets allowed | 6.0 |
| Protenix | the `pocket` channel of its constraint embedder; only the `base-constraint-v0.5.0` checkpoint has one, and every other checkpoint refuses the job | required |
| OpenFold3 | pocket-guided sampling: the distance ranks the sampler's ligand proposals; nothing conditions the network on it ([openfold3.md](../openfold3.md#pocket-constraints)); one pocket, and the binder must be a ligand | 4.0 |
| OpenDDE | dropped with a warning, as upstream OpenDDE drops every constraint; recorded under `ignored_constraints`; `--option ignore_constraints=false` refuses the job instead | — |
| AlphaFold 3, ESMFold2 | refused: no such input upstream | — |

To run the constrained Protenix checkpoint:

```bash
foldjax weights fetch --model protenix --profile base-constraint-v0.5.0
foldjax predict --model protenix --profile base-constraint-v0.5.0 \
    --input target_atp.yaml --output-dir out
```

## Ranking samples by the pocket on every model

The table above is what each *upstream* does with a pocket. FoldJAX adds one
route of its own that every model takes:

```bash
foldjax predict --model alphafold3 esmfold2 opendde protenix boltz2 openfold3 \
    --input target_atp.yaml --output-dir out --option pocket_sampling=select
```

`pocket_sampling=select` does not change sampling. After prediction every
sample is scored with Boltz-2's pocket rule -- residue 38 is satisfied when a
heavy atom of `L` lies within 6.0 Å of one of its heavy atoms, the pocket when
residues 38, 41 and 45 all are -- and the run's `best` prefers a sample that
satisfied the pocket, still ordered by the model's own ranking score. When no
sample did, the model's ranking stands and `best.pocket_satisfied` is
`false`. Boltz-2, OpenFold3 and the Protenix constraint checkpoint keep their
native conditioning and gain the selection; AlphaFold 3, ESMFold2, OpenDDE
and the released Protenix checkpoint, which refuse or drop the pocket today,
run with the pocket read by the selection alone -- for them `max_distance` is
required, since none has a default of its own. Each sample's
`confidence.json` lists the per-residue distances under `pocket_sampling`,
and `foldjax_run.json` records the route and whether the native input
conditioned on the pocket as well
([cli.md](../cli.md#--option-pocket_samplingselect)).

## Contacts between residues

A **contact** asks for two residues to lie within `max_distance` of each
other, for example across an interface:

```yaml
constraints:
  - contact: {token1: [A, 83], token2: [B, 35], max_distance: 8.0}
```

Boltz-2 reads contacts (several, within one chain too); Protenix reads them
with `base-constraint-v0.5.0`, between residues of different chains that are
not modified; OpenDDE drops and records them; AlphaFold 3, ESMFold2 and
OpenFold3 refuse them. A contact on a ligand atom, and Boltz-2's `force`
steering, are native-input features: write them in the model's own format
([input.md](../input.md#contact-constraints)). Pockets and contacts may share
one `constraints` list.

## Binding affinity (Boltz-2)

Boltz-2 is the only carried model with an affinity head. Name the ligand
chain in `properties`:

```yaml
properties:
  - affinity: {binder: L}
```

Every other model refuses a job that asks for affinity, rather than fold it
and drop the request, so keep affinity jobs in their own files or run them
with `--model boltz2` alone. From a bare sequence the same request is
`--affinity-binder CHAIN`, naming a chain of the generated job. Here the
protein becomes `A` and the ligand `B`; `foldjax plan` with the same flags
prints the generated job (`generated_input`) when you are unsure:

```bash
foldjax predict --model boltz2 --msa auto --affinity-binder B \
    --sequence MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKV \
    --ligand-smiles "CC(=O)Oc1ccccc1C(=O)O"
```

The affinity outputs (`affinity_pred_value`, `affinity_probability_binary`
and the per-member values) are among the sample's `scores` in
`confidence.json`. For a screen of many ligands against one target, generate
the jobs with `foldjax jobs expand --affinity` and read them back with
`foldjax show out/ --screen` ([batches](batch-slurm.md#a-ligand-screen)).

## Check the poses

```bash
uv sync --inexact --extra posebusters
foldjax check out/                  # PoseBusters on every predicted ligand
foldjax compare out/ --reference deposited.cif --metrics lddt,lig_rmsd
```

`check` reports `pb_valid` and one `pb.<check>` column per PoseBusters test;
a check PoseBusters could not compute counts as not passed. `compare
--reference` scores each structure against a deposited one, including ligand
RMSD after a pocket fit, which, unlike confidence, is comparable across
models.
