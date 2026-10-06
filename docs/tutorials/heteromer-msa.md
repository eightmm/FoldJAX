# Tutorial: a heteromer with an MSA search

Fold a two-chain complex, barnase with its inhibitor barstar, with alignments
searched for you, paired across the chains the way each model's upstream
pairs them, and read back how deep each alignment was.

## The job

`barnase_barstar.yaml`:

```yaml
name: barnase_barstar
entities:
  - type: protein
    id: A
    sequence: AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
  - type: protein
    id: B
    sequence: KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS
```

Neither chain names an alignment, so without an `--msa` choice the job is
refused. Check which models can take it under a search before anything runs:

```bash
foldjax models --for barnase_barstar.yaml --msa auto
```

## Search, fold, and pair

```bash
foldjax predict --model openfold3 protenix boltz2 \
    --input barnase_barstar.yaml --msa auto --output-dir out
```

**`--msa auto` sends both sequences to the public ColabFold MMseqs2 server**
(`https://api.colabfold.com`). For sequences that must not leave the machine,
point `FOLDJAX_MSA_SERVER_URL` at your own server or `FOLDJAX_MSA_COMMAND` at
a local search ([below](#searching-on-your-own-databases)). `--msa required`
does the same but fails the run rather than folding a chain from its single
sequence when its search fails, which is what a batch script wants.

Alignments are cached under `$FOLDJAX_HOME/msa/`, keyed by sequence, server
and pairing, and the cache is shared by every model: each chain's unpaired
alignment is searched once for all three models above, and a complex pairing
search once per distinct search mode.

### How the chains are paired

A heteromer's paired alignment puts rows from the same organism side by side
across chains, which is where much of the inter-chain signal comes from.
`--msa-pairing` chooses how a searched complex is paired; the default,
`model`, is what each model's upstream does:

| model | `--msa-pairing model` (default) | `greedy` / `complete` |
|---|---|---|
| OpenFold3, Boltz-2 | one ColabFold `pairgreedy-env` search over the complex's distinct sequences | that search with ColabFold's greedy or complete strategy |
| Protenix | one `pairgreedy` search, each chain's block written so that row *i* of every chain is paired, as Protenix's ColabFold mode writes it | the same with that strategy |
| OpenDDE | the same search, passed as the server wrote it: OpenDDE's species re-pairing then pairs only the query row, exactly as upstream OpenDDE does | opts into Protenix's row pairing; a departure from OpenDDE's released behaviour, not measured for accuracy |
| AlphaFold 3 | each chain's own alignment; AlphaFold 3 re-pairs by UniProt species, which ColabFold headers do not carry | refused |
| ESMFold2 | no paired alignment | refused |

`--msa-pairing none` delivers no paired alignment and skips the pairing
search. A monomer or homomer is never complex-paired. A complex search whose
per-chain blocks come back with different depths is refused before it is
cached (`required` fails; `auto` folds without the pairing and warns).

```bash
foldjax predict --model openfold3 --input barnase_barstar.yaml \
    --msa auto --msa-pairing complete --output-dir out-complete
```

`foldjax plan` shows the resolution without searching anything:

```bash
foldjax plan --model openfold3 --input barnase_barstar.yaml --msa auto --msa-pairing complete
```

```json
"msa_pairing": {"mode": "paircomplete-env", "paired_by": "row", "requested": "complete", "resolved": "complete"}
```

`paired_by` is `row` when row *i* of each chain is paired, `species` when
pairing is left to a species re-pairing that, on ColabFold headers, pairs only
the query row. The pairing mode is part of the MSA cache key and of the
`--resume` identity, so a run asked under another pairing reruns.

## Search ahead of time

A GPU node often has no network, or should not wait on a public server. Search
and cache every alignment first, wherever the network is:

```bash
foldjax msa prefetch barnase_barstar.yaml --model openfold3 protenix boltz2
```

`msa prefetch` takes job files, multi-job files, FASTA, structures or
directories. With `--model` it runs exactly the search `predict` would run for
that model (complex pairing included); without it, the per-chain search every
model shares. It loads no weights and writes no outputs, prints one JSON record
per chain, and exits 3 if any chain's search failed. A later `predict --msa
auto` with the same store reads every alignment from the cache.

## Read the alignment depth

Each run records, per chain, how many rows the model read and the effective
number of sequences (Neff, at 80% identity):

```bash
foldjax show out/                 # an "alignment" line per chain
foldjax show out/ --format csv    # an msa_stats column
```

In `foldjax_run.json`, `msa_stats` holds the same per chain: `depth` counts
rows of the alignment the model reads, query included, before the model's
own `--max-msa-depth`, deduplication or cap; `neff` weights each row by one
over the number of rows (itself included) within 80% identity of it over the
query's match columns, and uses at most the first 20,000 rows. A shallow
alignment (a Neff in the tens) is a reason to read the structure with care,
whatever the confidence says.

## Compare the models

```bash
foldjax show out/ --rank-by iptm       # samples ordered by ipTM within each model only
foldjax compare out/                   # pairwise RMSD between every structure of the input
foldjax interfaces out/                # ipSAE, pDockQ, pDockQ2, LIS per chain pair
```

Ranks never cross models: each model's ipTM has its own calibration. To score
against a deposited structure, which is comparable across models, add
`--reference 1brs.cif` to `compare` ([cli.md](../cli.md#analysis-and-workflow-commands)).

## Searching on your own databases

`foldjax msa wrapper` prints the path of a reference `FOLDJAX_MSA_COMMAND`
that runs ColabFold's own MMseqs2 steps against databases built with
ColabFold's `setup_databases.sh`. It imports nothing from FoldJAX, so it runs
under the interpreter that has ColabFold installed:

```bash
export FOLDJAX_MSA_COMMAND="/opt/colabfold/bin/python $(foldjax msa wrapper) \
    --db /data/colabfold_db --threads 16"            # add --gpu for MMseqs2-GPU
export FOLDJAX_MSA_LOCAL_VERSION="uniref30_2302+envdb_202108"
foldjax msa prefetch jobs/
```

`FOLDJAX_MSA_LOCAL_VERSION` names the databases and is part of the cache key,
so new databases never reuse old alignments. A local wrapper searches one
chain at a time and cannot pair a complex, so under it no model receives a
complex pairing alignment.
