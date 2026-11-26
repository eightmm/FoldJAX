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
| seed | a native job's `modelSeeds`; when `--seed` and `modelSeeds` are absent or empty, both `foldjax predict` and `opendde-jax-predict` draw one integer in `1..65536` per job (matching upstream) and record it (`run.json` and `seed_<seed>/seed.json`) |
| precision | bfloat16 trunk and bfloat16 confidence Pairformer |
| diffusion attention | `tokamax` fused attention on a serial GPU (`foldjax predict`); `xla_jit` under context parallelism, under `use_tfg_guidance`, off a GPU, and on `opendde-jax-predict` |
| diffusion pair-bias cache | on for a serial unguided GPU (`foldjax predict`); off under context parallelism, under `use_tfg_guidance`, off a GPU, and on `opendde-jax-predict` |

Upstream runs float32: `--option dtype=float32 --option confidence_dtype=fp32`
reproduces it. The float32 trunk costs about twice the peak memory of the
default ([cli.md](cli.md#a-bfloat16-trunk---option-trunk_dtypebf16),
[cli.md](cli.md#the-bfloat16-opendde-confidence-head---confidence-dtype)).

On a serial GPU, omitting `diffusion_attention_backend` runs OpenDDE's
diffusion token and windowed atom attentions on tokamax's fused kernel
(measured on one RTX PRO 6000 vs `xla_jit`: 1,003 tokens warm 130.1 -> 119.8 s,
0.921x; 1,993 tokens 676.4 -> 580.6 s, 0.858x; device peak unchanged at both;
CA same-index residual 0.005-0.006 Å at 1,993 tokens, equal to the
cross-process floor with `deterministic=off`, and per-chain deposited CA RMSD
unchanged to 1e-3). Use `--option diffusion_attention_backend=xla_jit` to
restore the blocked XLA path on a serial GPU; under context parallelism,
`use_tfg_guidance`, or off a GPU an omitted option resolves to `xla_jit`, while
an explicit `tokamax` keeps its refusal under context parallelism and
`use_tfg_guidance`.

On a serial unguided GPU, omitting `diffusion_pair_bias_cache` computes every
diffusion-transformer block's token pair bias once per prediction instead of
at every denoising step, with the efficient pair fusion the cache requires
(measured on one RTX PRO 6000 on top of tokamax attention, five samples, warm:
1,003 tokens 119.7 -> 101.4 s, 0.847x; 1,993 tokens 581.0 -> 496.2 s, 0.854x;
device peak unchanged at both; per-chain deposited CA RMSD unchanged to
1e-3 and the same-index CA residual 0.004-0.062 Å, the cross-process floor
with `deterministic=off`). The cached biases take `24 x 16 x S^2 x 4` bytes
for S structural tokens; above the 4,040 padded structural tokens measured,
the memory check treats a fitting estimate as a lower bound. Use
`--option diffusion_pair_bias_cache=false` to project per step on a serial
GPU; an explicit `true` keeps its refusal under context parallelism and
`use_tfg_guidance`.

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
  bonds do reach the model. `--option pocket_sampling=select` is the one way a
  common pocket counts here: a FoldJAX-only route that scores every sample
  against the pocket after prediction and ranks a satisfying sample first
  (`max_distance` required; the model itself still sees no constraint)
  ([cli.md](cli.md#--option-pocket_samplingselect)).
- **Native only**: ligands read from a file need a native OpenDDE job.
- **Polymer modifications and terminal caps.** Inline `modifications` on
  `proteinChain`, `dnaSequence` and `rnaSequence` require a CCD component with a
  canonical linking polymer type (`protein`, `dna` or `rna`). Non-polymer caps
  such as `CCD_ACE` and `CCD_NME` are refused as inline `modifications`
  (upstream reclassifies them inside the chain and leaves the preceding
  residue's C-terminal `OXT` attached); supply terminal caps as separate
  `ligand` entities joined to the terminal polymer residue via `covalent_bonds`.
- **Sequence alphabets and job validation.** Polymer sequences accept only the
  letters upstream's input specification documents (`ARNDCQEGHILKMFPSTWYVX` for
  protein, `ACGTNX` for DNA, `ACGUNX` for RNA; undocumented parser aliases such
  as DNA `I`/`U` and RNA `I` are refused). Top-level native jobs are validated
  before weights load for non-empty, NUL-free, non-`"err"`, collision-free
  names (both raw and after `safe_job_name` directory sanitization) and unique
  integer `modelSeeds` in `0..4294967295` (booleans and fractional numbers are
  refused rather than truncated; direct `featurize_opendde_json` calls with
  neither `seed` nor `modelSeeds` default to `101`).

## Upstream behavioural notes

- **Job-name path sanitization and shared featurizer normalization.** Where
  upstream `validate_sample_name` rejects separators and `.`/`..` outright,
  FoldJAX sanitizes job names into a single safe directory component via
  `safe_job_name` (still rejecting raw or post-sanitization directory
  collisions and `"err"` before weights load). In the shared Protenix/OpenDDE
  featurizer, A3Ms whose first aligned row differs in ungapped length from the
  query sequence are truncated or gap-padded, and `covalent_bonds` accepts
  0-based numeric atom indices on `CCD_` ligands as well as `FILE_`/SMILES
  ligands (with SMILES ligands also accepting explicit nonzero atom-map
  numbers).
- **Sample ranking tie-break.** Samples with identical `ranking_score` break
  ties deterministically in ascending source sample order (`stable` sort),
  whereas upstream delegates to PyTorch's unstable `torch.argsort`.
- **Zero-contact `gPDE`.** When every contact probability in a structure or
  chain pair is zero, `gpde`, `chain_gpde` and `chain_pair_gpde` report `0.0`
  rather than dividing by zero.

## Options worth knowing

| option | default | effect |
|---|---|---|
| `include_raw=true` | `false` | also write PAE, PDE and contact probabilities to `confidence_full.npz` (and `predicted_aligned_error.json`) |
| `dtype=float32`, `confidence_dtype=fp32` | bfloat16 | upstream's precision |
| `diffusion_attention_backend=B` | `tokamax` (serial GPU) / `xla_jit` (CP, TFG, off-GPU) | attention kernel for the diffusion token and windowed atom attentions (`xla`, `xla_jit`, `xla_sdpa`, `tokamax`); explicit `tokamax` is refused under context parallelism and `use_tfg_guidance` |
| `use_tfg_guidance=true` | `false` | upstream's training-free guidance, on Protenix's guidance port: geometry terms on ligands, ions and modified residues only, never on standard residues ([what it constrains](cli.md#upstream-run-options---option)); eager, so refused with explicit padding (the default padding steps aside for it), explicit `deterministic=on` (the default steps aside for it) and context parallelism, and about 12x the unguided wall time ([measured](cli.md#upstream-run-options---option)) |
| `deterministic=on` | `off` | compile this run for repeatable reductions and fixed GEMM kernels, so two cold processes agree bit for bit; off by default here because the port's large GEMMs lose Triton under it: 1.34x the warm wall at 1,000 tokens (GPU-active time 1.40x), peak unchanged ([the measurement](cli.md#--option-deterministicon)) |
| `cp_devices=N` | `1` | context parallelism over N GPUs: the only way this port has run 2,096 residues on one node's cards |
| `structural_triangle_att_row_chunk_size=N` | unset | experimental outer-row chunks for structural-refiner cuEq/Tokamax attention; preserves full column softmax and dtype; unsupported with context parallelism |

The row-chunk candidate bounds Q/K/V, gate and output projections inside each
block. It has CPU contract coverage, but fused GPU accuracy, speed and capacity
remain unmeasured. It selects a separate compilation namespace: use the same
width for warming and prediction. Default profiles need no rewarm. See the
[cluster measurement panel](cluster-memory-handoff.md#b-opendde-projection-row-chunking-candidate).

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
