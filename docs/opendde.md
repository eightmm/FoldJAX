# OpenDDE

OpenDDE is carried as a JAX reimplementation at `foldjax.models.opendde`,
tracking OpenDDE 1.1.1 at commit `ddfa1df`. It shares Protenix's input dialect
and much of its featurizer, and folds in a structural-token space of its own.
Its native command line is kept as `opendde-jax-predict`.

```bash
foldjax weights fetch --model opendde
foldjax predict --model opendde --input job.yaml --msa auto
```

## Weights and profiles

| profile | checkpoint | what it is |
|---|---|---|
| `released` (default) | `opendde.pt` | the general-purpose release |
| `abag` | `opendde_abag.pt` | the antibody-antigen-optimized release: the same 655,791,538-parameter graph, a different parameter set, chosen explicitly |

```bash
foldjax weights fetch --model opendde --profile abag
foldjax predict --model opendde --profile abag --input antibody_antigen.yaml --msa auto
```

Both come from one pinned Hugging Face revision, a 2.6 GB checkpoint each
(3.3 GB with the chemistry assets shared with Protenix), checked against their
SHA-256 ([model-versions.md](model-versions.md#opendde)); code
and checkpoints are Apache-2.0. `foldjax setup` fetches `released`; `abag` is
never substituted implicitly.

## What a run does by default

| knob | default |
|---|---|
| samples / steps / recycles | 5 / 200 / 10 |
| MSA | up to 16,384 rows |
| seed | a native job's `modelSeeds`; for a common-schema job none upstream, so FoldJAX draws one and records it |
| precision | bfloat16 trunk and bfloat16 confidence Pairformer |

Upstream runs float32: `--option dtype=float32 --option confidence_dtype=fp32`
reproduces it. The float32 trunk costs about twice the peak memory of the
default ([cli.md](cli.md#a-bfloat16-trunk---option-trunk_dtypebf16),
[cli.md](cli.md#the-bfloat16-opendde-confidence-head---confidence-dtype)).

## Inputs

- **Alignments.** Under `--msa auto` FoldJAX submits the same `pairgreedy`
  ColabFold search upstream OpenDDE submits -- its protein entities sorted by
  sequence, whenever there is more than one, even two entities with one
  sequence -- and passes the blocks as the server wrote them. OpenDDE's
  species re-pairing then pairs only the query row, as upstream does, and the
  block's rows join each chain's unpaired alignment (`paired_by: species` in
  the manifest). `--msa-pairing greedy` or `complete` opts into the row
  pairing Protenix's opt-in uses instead: a departure from
  OpenDDE's released behaviour that has not been measured for accuracy
  ([heteromer tutorial](tutorials/heteromer-msa.md)). RNA alignments are read
  only with `--option use_rna_msa=true`.
- **Templates** are read only with `--option use_template=true`, as upstream's
  released configuration leaves them off; otherwise they are dropped with a
  warning and recorded under `ignored_templates`
  ([templates tutorial](tutorials/templates.md)).
- **Constraints are never read.** OpenDDE's model has no constraint embedder,
  and upstream's inference build warns and ignores a job's `constraint`. A
  common job's pocket or contact constraints, or a native job's `constraint`,
  are dropped the same way, with a warning and an `ignored_constraints` record;
  `--option ignore_constraints=false` refuses such a job instead. Covalent
  bonds do reach the model.
- **Native only**: ligands read from a file need a native OpenDDE job.

## Options worth knowing

| option | default | effect |
|---|---|---|
| `include_raw=true` | `false` | also write PAE, PDE and contact probabilities to `confidence_full.npz` (and `predicted_aligned_error.json`) |
| `dtype=float32`, `confidence_dtype=fp32` | bfloat16 | upstream's precision |
| `use_tfg_guidance=true` | `false` | upstream's training-free guidance, on Protenix's guidance port; eager, so refused with `--padding`, `deterministic=on` and context parallelism |
| `cp_devices=N` | `1` | context parallelism over N GPUs: the only way this port has run 2,096 residues on one node's cards |

## Memory

OpenDDE's peak grows with the square of its **structural** token count, which
is about 1.9 times the residue count, so it reaches a card's limit earlier than
the residue count suggests. Its memory law is keyed on structural tokens, and
a refusal names the levers that apply: the bfloat16 trunk if a float32 one was
asked for, and context parallelism (`--option cp_devices=4`) above about 2,000
residues on a 96 GB card ([cli.md](cli.md#memory),
[context_parallel.md](context_parallel.md)).

## State of the port

Stage-level parity against upstream is in
[model-versions.md](model-versions.md#stage-level-parity-current); speed and
memory in [benchmark.md](benchmark.md). The porting record is
[ports/opendde/](ports/opendde/README.md), history rather than instructions.
