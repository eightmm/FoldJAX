# Protenix

Protenix is carried as a JAX reimplementation at `foldjax.models.protenix`,
tracking upstream commit `4c355be` (package metadata 2.0.0). One backend runs
every public Protenix checkpoint, selected with `--profile`, and its native
command line is kept as `protenix-jax-predict`.

```bash
foldjax weights fetch --model protenix
foldjax predict --model protenix --input job.yaml --msa auto
```

## Weights and profiles

| profile | checkpoint | what it is |
|---|---|---|
| `released` (default) | `protenix_base_default_v1.0.0` | the 368M base model, 2021-09-30 training cutoff (the cutoff benchmarks against AlphaFold 3 need) |
| `base-20250630` | `protenix_base_20250630_v1.0.0` | the same architecture trained to a 2025-06-30 cutoff; upstream recommends it for practical use |
| `v2` | `protenix-v2.pt`, **supplied by you** | Protenix v2, 464M parameters ([below](#protenix-v2)) |
| `mini-esm-v0.5.0`, `mini-ism-v0.5.0` | `protenix_mini_{esm,ism}_v0.5.0` | small models conditioned on an ESM2-3B or ISM language model, staged beside them; upstream's 5-step, 4-cycle sampler |
| `base-constraint-v0.5.0` | `protenix_base_constraint_v0.5.0` | the base model with pocket, contact, substructure and atom-contact constraint embedders: the checkpoint that reads constraints |
| `mini-default-v0.5.0`, `tiny-default-v0.5.0` | `protenix_{mini,tiny}_default_v0.5.0` | small models without a language model, on the 5-step, 4-cycle sampler |

```bash
foldjax weights fetch --model protenix --profile base-20250630
foldjax predict --model protenix --profile base-20250630 --input job.yaml
```

The code and the v1.x and v0.5.0 checkpoints are Apache-2.0. Every
downloadable checkpoint is pinned by SHA-256
([model-versions.md](model-versions.md#protenix)); `foldjax setup` fetches the
`released` profile, and `foldjax setup --all` the rest that can be downloaded.

### Protenix v2

Upstream declares the v2 weights proprietary and not to be transferred
without the rights holder's written consent, so FoldJAX never downloads them.
Put a `protenix-v2.pt` you obtained with that consent in the profile's
directory and convert it:

```bash
v2="$(foldjax home --path weights)/protenix-v2"
mkdir -p "$v2" && cp protenix-v2.pt "$v2/"
foldjax weights fetch --model protenix --profile v2
foldjax predict --model protenix --profile v2 --input job.yaml
```

Only the file with the pinned SHA-256 is accepted. Upstream refuses v2 above
2,560 tokens as a memory precaution; FoldJAX warns and runs past it (measured
to 3,012 tokens), and `--option strict_token_limit=true` restores the refusal.
v2 costs up to about twice the base model's memory at large sizes
([cli.md](cli.md#weights-and-setup)).

## What a run does by default

| knob | default |
|---|---|
| samples / steps / recycles | 5 / 200 / 10 for the base models; the small models run upstream's 5 steps and 4 recycles |
| MSA | up to 16,384 rows; every recycle reads a fresh random subset of them, as upstream does |
| seed | 101, upstream's default |
| precision | bfloat16 trunk; the confidence head bfloat16 at every size |

Upstream keeps the confidence head float32 up to 2,560 tokens;
`--option amp_policy=upstream` reproduces that
([cli.md](cli.md#a-bfloat16-protenix-confidence-head---amp-policy-on-by-default)).
`--preset fast` selects the publisher's reduced schedule (5 steps, 4
recycles) and is accepted with the `mini-esm-v0.5.0` and `mini-ism-v0.5.0`
profiles, whose released default it already is; elsewhere it is refused with
the reason.

Protenix's own randomness has three sources: the seed (diffusion noise),
`--option msa_seed=N` (the per-recycle MSA row draw, `--msa-seed` on the
native CLI; follows `--seed` when omitted),
and MC dropout on the recycle pair update, which upstream applies to about 40%
of predictions (`--option mc_dropout_apply_rate=0` turns it off).
`--option full_depth_msa=true` reads every MSA row in every recycle instead
of a random subset ([cli.md](cli.md#--msa-seed-protenix)).

## Inputs

- **Alignments.** Under `--msa auto` each chain gets its unpaired ColabFold
  alignment, environmental hits first, and a heteromer's chains are not
  paired: that is what a run of upstream's ColabFold mode does, whose runner
  never reads the complex search it writes. Upstream's default mode pairs by
  NCBI taxonomy on Protenix's own MSA server, which FoldJAX does not use, and
  a heteromer run prints a note saying so. `--msa-pairing greedy` (or
  `complete`) opts into the `pairgreedy` search with each chain's block
  written so that row *i* of every chain is paired, as upstream's ColabFold
  mode writes its `pairing.a3m` -- a FoldJAX choice, not measured for
  accuracy against either upstream mode
  ([heteromer tutorial](tutorials/heteromer-msa.md)). RNA
  alignments are read only with `--option use_rna_msa=true` (upstream's
  released default is false; allowed for `v2` and the two v1.0.0 base
  checkpoints).
- **Templates** are read only with `--option use_template=true`, as upstream's
  released configuration leaves them off; without it a job's templates are
  dropped with a warning and recorded under `ignored_templates`. Allowed for
  `v2` and the two v1.0.0 base checkpoints
  ([templates tutorial](tutorials/templates.md)).
- **Constraints.** Pocket and contact constraints need the
  `base-constraint-v0.5.0` profile; every other checkpoint refuses them, because
  it has no constraint embedder to read them
  ([protein-ligand tutorial](tutorials/protein-ligand.md)). On that profile
  every run adds the constraint embedder's substructure term to the initial
  pair representation, constraint or not, as upstream does: upstream attaches
  an all-zero substructure map to every job (it never parses
  `constraint.structure`), and its transformer embedder turns that map into
  one constant pair vector, which FoldJAX computes directly. A nonzero
  substructure feature is refused. A conversion made before FoldJAX mapped
  that embedder is reported as stale; `foldjax weights fetch --model protenix
  --profile base-constraint-v0.5.0` rebuilds it.
- **Native only**: ligands read from a file need a native Protenix job.

## Options worth knowing

| option | default | effect |
|---|---|---|
| `output_format=both` | | also write PAE, PDE and contact probabilities to `confidence_full.npz` (and `predicted_aligned_error.json`) |
| `use_tfg_guidance=true` | `false` | upstream's training-free guidance with its default mapping, on an eager sampler: geometry terms on ligands, ions and modified residues only, never on standard residues ([what it constrains](cli.md#upstream-run-options---option)); the denoiser attention runs `xla_jit`, and a spelled `tokamax` attention is refused, because the eager sampler recompiles it on every call; refused with `--padding`, `deterministic=on` and context parallelism; expect an order of magnitude more wall time ([why](cli.md#upstream-run-options---option)) |
| `attention_kernel=tokamax` | | fused pair-bias attention |
| `amp_policy=upstream\|fp32\|bf16` | `auto` | which stages narrow to bfloat16 |
| `deterministic=on` | `off` | repeatable reductions, at a cost in wall time |

Every option and its measurements: [cli.md](cli.md). Context parallelism
splits one prediction over several GPUs
([context_parallel.md](context_parallel.md)).

## State of the port

Stage-level parity against upstream is in
[model-versions.md](model-versions.md#stage-level-parity-current); speed and
memory in [benchmark.md](benchmark.md). The porting record is
[ports/protenix/](ports/protenix/README.md), history rather than instructions.
