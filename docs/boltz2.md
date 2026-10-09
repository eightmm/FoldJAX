# Boltz-2

Boltz-2 is carried as a JAX reimplementation at `foldjax.models.boltz2`,
tracking upstream commit `b1ebfc4` (package metadata 2.2.1). It is the one
carried model with a binding-affinity head, and it reads pocket and contact
constraints from the common job.

```bash
foldjax weights fetch --model boltz2
foldjax predict --model boltz2 --input job.yaml --msa auto
```

## Weights

The managed bundle comes from `boltz-community/boltz-2` at a pinned revision:
the structure and confidence checkpoint, the affinity checkpoint and the
molecule archive, about 6.2 GB, each checked against its pinned SHA-256
([model-versions.md](model-versions.md#boltz-2)). Code and weights are MIT.
`foldjax setup` fetches it with the other public models. FoldJAX reads the
publisher's checkpoints with its restricted NumPy archive reader and converts
them once; nothing imports PyTorch. The molecule archive is unpacked beside
the weights and passed to the featurizer automatically.

## What a run does by default

| knob | default |
|---|---|
| samples / steps / recycles | 1 / 200 / 3, upstream's `boltz predict` defaults |
| MSA depth | up to 16,384 rows (the featurizer's cap) |
| seed | none upstream, so FoldJAX draws one, prints it and records it |
| precision | bfloat16 trunk with a bfloat16 pair residual; float32 matmuls at `high` |

Upstream's precision is `--option pair_residual_dtype=float32 --option
matmul_precision=highest`. Why the defaults differ, and what was measured:
[README, precision](../README.md#precision) and
[cli.md](cli.md#a-bfloat16-boltz-2-pair-residual---option-pair_residual_dtype-on-by-default).

One sample is upstream's default and FoldJAX keeps it. For a population to
choose from, ask for more: `--num-samples 5`, or several seeds with `--seeds`.

## Alignments

A protein chain without an alignment is refused, as upstream Boltz-2 refuses
it; pass `--msa auto`, an `unpaired_msa` file, or `--msa single` on purpose.
Under `--msa auto` FoldJAX runs the search upstream Boltz runs: one ColabFold
`pairgreedy-env` search over the complex's distinct sequences, delivered as
the keyed CSV `boltz/main.py` builds (paired rows keep their row number as
`key`, unpaired rows follow with -1). `--msa-pairing complete` uses
ColabFold's complete strategy instead, and `none` delivers no pairing
([heteromer tutorial](tutorials/heteromer-msa.md)). The common `paired_msa`
field is not available for Boltz-2; supply pairing in a native Boltz YAML.

`--option msa_deletions=restored` reinstates the MSA deletion features that
upstream v2.2.0 and later zero for every real alignment; the default
`released` reproduces upstream as released, regression included
([cli.md](cli.md#--option-msa_deletionsreleasedrestored-boltz-2)).

## Templates, constraints and affinity

- **Templates** are mmCIF files Boltz-2 aligns itself; a template residue map
  is refused. With `--templates auto` FoldJAX keeps the first four hits whose
  structure has the hit chain, a convenience: upstream Boltz-2 has no template
  search ([templates tutorial](tutorials/templates.md)).
- **Pocket constraints** condition the trunk (default `max_distance` 6.0 Å),
  several per job; **contact constraints** likewise, within one chain or across
  chains ([protein-ligand tutorial](tutorials/protein-ligand.md)). With
  `--option pocket_sampling=select` FoldJAX also scores every sample with this
  model's own pocket rule after the run and ranks a satisfying sample first;
  the conditioning is unchanged
  ([cli.md](cli.md#--option-pocket_samplingselect)).
- **`force`** (bool, default `false`) on a pocket or contact asks upstream's
  own steering: a forced job turns on `contact_guidance_update` automatically
  (`main.py:156`), the same `BoltzSteeringParams()` `boltz predict` passes
  without `--use_potentials`, and runs the eager sampler for it. That sampler
  builds no outer executable, so it refuses `--padding`, `cp_devices > 1` and
  `deterministic=true`; `foldjax plan` refuses the same combination before
  anything runs. Pass `--option steering_args='{"fk_steering": false,
  "physical_guidance_update": false, "contact_guidance_update": false}'` to
  keep `force` for the record and skip the guidance on purpose. Every other
  model is refused: none has a steering potential to turn on. Off (the
  default), a job runs exactly as it always has.
- **Affinity**: `properties: [{affinity: {binder: L}}]`, or
  `--affinity-binder CHAIN` for a generated job. The affinity stage follows
  upstream's two-stage contract (rank the structure samples, crop the
  receptor-ligand complex, run the separate affinity checkpoint and average its
  ensemble members), and `confidence.json` reports `affinity_pred_value`,
  `affinity_probability_binary` and the per-member values.
- **Native only**: cyclic polymers need a native Boltz YAML
  (`--input-format boltz` or auto-detection).

## Upstream run options

Upstream `boltz predict` options FoldJAX passes through `--option`; an omitted
option runs upstream's value:

| option | default | effect |
|---|---|---|
| `step_scale=FLOAT` | `1.5` | sampler step size; lower values diversify the samples |
| `subsample_msa=true` | `false` | redraw a subset of MSA rows every trunk pass (refused with `--padding` and context parallelism) |
| `num_subsampled_msa=N` | `1024` | rows per pass when subsampling |
| `method=NAME` | none | method conditioning, one of upstream's method types (for example `x-ray diffraction`, `electron microscopy`, `solution nmr`), case-insensitive |
| `use_potentials=true` | `false` | Feynman-Kac steering and physical guidance; eager, so refused with `steering_args`, `--padding`, `deterministic=on` and context parallelism |

Each of these joins the compile-cache identity. The full table, with upstream
line references: [cli.md](cli.md).

## Kernels and scale

On a GPU an omitted setting runs fused kernels: Pallas kernels for the
triangle multiplication and the pair transitions, and cuEquivariance triangle
attention. `BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND=cueq` (or `xla`) and
`--option glu_backend=tokamax` (or `xla`) select the released paths, and
`--option triangle_kernel=xla` plain XLA attention
([cli.md](cli.md#pallas-pair-kernels-gpu-default-boltz-2-and-openfold3-both-protenix-the-multiplication)).
Off a GPU the released kernels run. Context parallelism splits one prediction over
several GPUs (`--option cp_devices=N`), including the atom-window diffusion
path ([context_parallel.md](context_parallel.md)). A fitted peak-memory law
refuses a run that will not fit before it compiles
([cli.md](cli.md#memory)).

## State of the port

Feature, weight, trunk, sampler, confidence and final-structure parity against
upstream are recorded per stage in
[model-versions.md](model-versions.md#stage-level-parity-current); speed and
memory against upstream in [benchmark.md](benchmark.md). The porting record is
[ports/boltz2/](ports/boltz2/README.md), history rather than instructions.
