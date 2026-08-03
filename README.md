<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.png">
  <img alt="FoldJAX — biomolecular structure prediction, compiled" src="docs/banner-dark.png">
</picture>

# FoldJAX

Biomolecular structure prediction in JAX. Five models behind one interface —
four carried complete inside the package:

| model | vendored at | licence | upstream |
|---|---|---|---|
| `boltz2` | `foldjax.models.boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `chai` | `foldjax.models.chai` | Apache-2.0 | [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

— plus `alphafold3`, which FoldJAX drives rather than carries, because its
parameters may not be redistributed. It takes the same job file as the rest
once installed; see [docs/alphafold3.md](docs/alphafold3.md).

Give it one job file and a model name. Weights, the input dialect, the output
directory, and the XLA compile cache are all resolved for you.

```bash
uv sync --extra cuda13 --extra torch-bridge --group dev
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

`torch-bridge` is only for that one conversion step: the upstream checkpoints
are torch files. **Prediction never imports torch for any model**, so once the
weights exist, `--extra cuda13` is the whole runtime.

```json
{
  "model": "boltz2",
  "output_dir": "foldjax-outputs/job",
  "samples": [
    {
      "seed": 0,
      "structure_path": "foldjax-outputs/job/boltz2_input.cif",
      "scores": {"mean_plddt": 0.9589, "iptm": 0.0}
    }
  ]
}
```

Every port keeps its own featurizer, sampler, checkpoint format, defaults, and
output writer, so results match running that project directly. FoldJAX supplies
the common request, the translation, and the plumbing.

## Input

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

`--input-format` defaults to `auto` and decides on content, not extension: only
the common schema is a mapping with `entities`, and every native dialect is
passed through untouched. So a Boltz YAML, a Protenix job list, or a Chai FASTA
all work as-is.

Modifications and covalent bonds have one model-neutral spelling here and are
translated into each backend's own names — Protenix and OpenDDE want
`CCD_`-prefixed types and entity-indexed `covalent_bonds`, Boltz wants
`{ccd, position}` and a `bond` constraint, AlphaFold 3 wants bare `ptmType`.

**A field a backend cannot express is rejected, not dropped.** Neither Boltz nor
Chai has a place for a separate `paired_msa`, and Chai addresses ligands by
SMILES only. Silently discarding an MSA would change the science without
changing the exit code.

`unpaired_msa` reaches every model, Chai included. Chai's own input is a FASTA
with nowhere to put an alignment path, so FoldJAX converts the A3M into the
`.aligned.pqt` form it reads and writes that beside the FASTA. That matters more
than it sounds: otherwise a job that pins an alignment pins it everywhere except
Chai, which quietly runs on a different one — and any comparison across models
becomes a comparison of MSAs.

## Choosing what to run

```bash
uv run foldjax predict \
  --model protenix \
  --input job.yaml \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10 \
  --seed 42
```

`--num-samples`, `--num-steps`, and `--num-recycles` are model-neutral. Each
backend calls them something different, and FoldJAX translates:

| neutral | alphafold3 | boltz2 | chai | opendde / protenix |
|---|---|---|---|---|
| `--num-samples` | `heads.diffusion.eval.num_samples` | `diffusion_samples` | `num_diffusion_samples` | `n_sample` |
| `--num-steps` | `heads.diffusion.eval.steps` | `steps` | `num_diffusion_timesteps` | `n_step` |
| `--num-recycles` | `num_recycles` | `recycling` | `num_trunk_recycles` | `n_cycle` |
| `--max-msa-depth` | `evoformer.num_msa` | `max_msa_depth` | `max_msa_depth` | `max_msa_rows` |

All four reach all five models. AlphaFold 3's step count and MSA depth are not
arguments of `make_model_config`, but the config it returns is an ordinary
mutable object and upstream already sets the sample count on it by assignment;
FoldJAX sets these the same way. That matters beyond convenience — a model that
cannot be held to the same schedule as the others cannot be compared with them.

Setting both a neutral knob and its native name is an error rather than a silent
preference, and so is asking for a knob a backend does not have: OpenFold3
exposes no MSA-depth argument, so `--max-msa-depth` is refused for it rather
than quietly ignored. `--option KEY=VALUE` still passes anything native straight
through.

`--seeds 1 2 3` runs the job once per seed, into a `seed_<n>` directory each,
and returns every structure together. The samples from one seed are correlated,
so this is the usual way to get independent predictions. Three of the models
take a seed list natively and three do not, so the loop lives in FoldJAX and
every model gets it identically.

Every run writes `foldjax_run.json` beside its structures: the model, the input
and its SHA-256, the resolved weights, the seeds and knobs actually used, the
JAX runtime, and each structure's confidence. A directory of `.cif` files
otherwise cannot say what produced it.

### Memory

For what these models cost against their own upstream repositories, under one
schedule and one measurement, see
[FoldJAX against upstream](#foldjax-against-upstream). This section is about
*why* the peaks landed where they did.

OpenDDE is the example worth keeping: it would not run at all before this work,
asking for 76 GiB on a 488-token job and dying. It now runs that job in about
10 GiB, and at 490 residues under the released schedule it peaks at 10.5 GiB
against upstream torch's 18.6 GiB.

Each of the changes below came from reading XLA's buffer assignment for the
peak and fixing whatever it named — a habit worth more than any individual fix
in the list.

**The transition was the largest buffer in three of the four ports.** It widens
its input before narrowing it again and holds three copies of the wide form at
once — `a`, `b`, and `silu(a) * b`. On OpenDDE's structural pair representation
those were three `f32[946, 946, 768]` buffers, 2,622 MiB apiece: 7,866 MiB of a
10,914 MiB arena, for an operation that reduces only over the channel axis.
Blocking the leading axis is mathematically exact, so it needs no flag: there
is no accuracy to trade, only kernel launches, and only on tensors already big
enough for that to be cheap.

**Triangle multiplication widened its projections before contracting them.**
Both are `[N, N, c_hidden]` — 1,311 MiB each on OpenDDE's structural branch —
and they were cast to float32 first, so a bfloat16 trunk still paid float32 for
its two largest tensors. They now keep their own dtype and accumulate through
`preferred_element_type`: identical float32 accumulation, half the operand
bytes. It is what AlphaFold 3's `BF16_BF16_F32` algorithm already does, and it
is most of why AF3's peak sits where it does — its released trunk runs bfloat16
weights *and* activations.

**Triangle attention is a batch of independent attentions, one per row.**
`q`, `k` and `v` are `[N_row, heads, N_col, d]` and nothing in the softmax
crosses a row, so blocking the row axis bounds all three projections *and* the
score tensor together — for the same block count and the same arithmetic, since
each row is projected exactly once either way. Blocking the query axis, which
is the obvious reading of "chunk the attention", bounds only the scores and
`q`; `k` and `v` stay whole at 1,311 MiB apiece. Row blocking is also
mathematically exact — it divides a batch, not a reduction — where query
blocking splits the softmax's own axis. Exact is not the same as bit-identical:
changing the batch extent changes the GEMM shape, and XLA may tile and
accumulate it differently, the same caveat the transition's row chunk carries.

The block size is a row count, not a byte budget, because that is what the
sweep found: 25 rows is the optimum at 946 structural tokens *and* at 1,892,
where the equivalent byte budgets differ fourfold. Bytes stay on as a ceiling.

Measured as they landed, those took Protenix from 12,264 to 6,043 MiB and
OpenDDE from 32,185 to 9,761 MiB on a 488-token job, with confidence flat
(75.08 → 75.07 and 87.49 → 87.49). Both have moved again since — the default
triangle-attention backend changed and the block rule was re-derived — so read
[FoldJAX against upstream](#foldjax-against-upstream) for what they cost now.
These figures are the size of each step, not a current reading.

**Eager arithmetic between compiled blocks is not free.** Chai runs its trunk
as many small programs on purpose, so XLA can release MSA intermediates between
them — but the residual `a + b` joining those programs dispatches its own
executable, with all three buffers live. On the MSA representation that is
three copies of the largest tensor in the trunk, for an add. Routing them
through a compiled add that donates its update buffer is the same arithmetic
and took Chai from 8,637 to 7,576 MiB with scores unchanged to five decimals.

### A bfloat16 trunk (`--option trunk_dtype=bf16`)

This is a backend option, not a FoldJAX flag, and only Protenix and OpenDDE
have it — the two Protenix-family ports:

```bash
foldjax predict --model opendde --input job.yaml --option trunk_dtype=bf16
```

Protenix already defaults to `bf16`; OpenDDE defaults to `fp32`, so it is
OpenDDE the option is worth setting on.

AlphaFold 3 runs its trunk in bfloat16 — weights *and* activations — and that
is most of why its peak sits where it does. Protenix ships the same default;
OpenDDE does not, and the flag that was supposed to give it to you only
narrowed the weights. Every activation stayed float32, because three arrays
entered the trunk wide and promoted everything downstream of them: `s_inputs`
(the MSA one-hot is allocated with *its* dtype, which is how one array decided
the dtype of the whole MSA module), `relp` and the token bonds (built in the
trunk from integer features), and the per-cycle alignments (sampled from the
raw features rather than the trunk's own narrowed dict).

All three are now cast where they cross the boundary, which is what AF3 does
(`prev['pair'].astype(pair_activations.dtype)`) rather than trying to stop every
internal op from widening. It matters more the larger the job, because the
activations are quadratic and the weights are not:

| opendde | fp32 | bf16 | |
|---|---|---|---|
| 488 tokens | 9,837 MiB, 10.5 s | **7,533 MiB, 8.6 s** | pLDDT 87.490 → 87.404 |
| 976 tokens | 32,714 MiB, 46.3 s | **18,091 MiB, 28.9 s** | pLDDT 79.235 → 78.859 |

Nearly half the memory and a third off the wall clock at 976 tokens, for 0.5%
of pLDDT. It stays off by default because that cost is real; turn it on when
the job would not otherwise fit.

**Dead outputs cost twice.** OpenDDE returned six representation tensors that
nothing reads; the structural pair representation alone was 1,311 MiB of a
1,869 MiB output, and returning it kept the refiner's working buffer live to
the end of the program. They are still available from the Python API —
`foldjax.models.opendde.models.model` takes `return_representations=True` —
but no CLI path asks for them, because no CLI output uses them.

Two things that sound like memory knobs and were not, both measured before the
fixes above: blocking the triangle *attention* moved Protenix's peak by 0.4%,
and switching its trunk to bfloat16 moved it by 0.3% — the latter because
Protenix's CLI already defaults to `--trunk-dtype bf16`, so it was measuring
nothing at all.

### `--max-msa-depth`

Every trunk holds a `[depth, tokens, channels]` MSA representation, and on a
488-token job with a 13,254-row alignment that tensor is 3,158 MiB. Capping the
depth used to halve Protenix. It no longer does — the transition fix took the
same memory without touching the alignment:

| protenix, 488 tokens | peak | pLDDT | pTM |
|---|---|---|---|
| default (13,254 rows) | 6,043 MiB | 75.07 | 0.749 |
| `--max-msa-depth 1024` | 5,989 MiB | 79.88 | 0.915 |

A 0.9% saving now, so treat it as an accuracy knob rather than a memory one.
Confidence moves with it and not monotonically, because which rows get sampled
changes; that spread is the alignment's own variance, not evidence that a
shallower MSA predicts better. The default keeps each port's own depth.

**Chai has the same knob but not the same bargain.** It embeds the selected
rows once into a `[1, depth, tokens, 64]` tensor — 2,048 MiB at its 16,384-row
bucket — and the trunk reads that, so the cap has to act at the embedding, not
at the per-recycle crop. It works, and it is the largest single lever there,
but unlike Protenix it costs accuracy rather than trading noise:

| chai, 488 tokens | peak | mean pLDDT | pTM |
|---|---|---|---|
| default | 7,576 MiB | 0.849 | 0.744 |
| `--max-msa-depth 4096` | 5,126 MiB | 0.838 | 0.735 |
| `--max-msa-depth 2048` | 4,856 MiB | 0.829 | 0.697 |
| `--max-msa-depth 1024` | **4,707 MiB** | 0.830 | 0.705 |

A third of the memory for one point of pLDDT. It stays off by default. Boltz-2
is the least sensitive of the four — 4,766 → 4,688 MiB at 1024, for 0.891 →
0.880 pLDDT — which is not worth spending, so it is off there too.

Chai's cap has two crop sites and only one of them counts: the selection in
`_embed_and_initialize` decides what gets embedded and therefore how big that
tensor is. Capping the per-recycle crop alone leaves it at full depth and moves
nothing — which is exactly what the first attempt did.

**OpenDDE is the opposite case, and it is worth knowing why.** Capping its MSA
changes nothing; blocking its attention changes everything. It is dual-branch,
and the structural refiner runs on sub-residue tokens with three times the
heads of the trunk, so its score tensor — `[946, 12, q, 946]` on a 488-residue
job — is what sizes the peak. Protenix's chunk policy maps a token count to a
chunk size and was written for that four-head trunk, so its 256 still left
10.5 GiB materialised, twice over. FoldJAX blocks that branch by bytes instead.
The budget was then swept rather than guessed:

| structural score budget | peak | warm | pLDDT |
|---|---|---|---|
| 2048 MiB | 15,639 MiB | 11.1 s | 87.4899 |
| **1024 MiB** (default) | **14,443 MiB** | 12.3 s | 87.4907 |
| 512 MiB | 14,465 MiB | 13.9 s | 87.4901 |
| 256 MiB | 14,463 MiB | 15.4 s | 87.4918 |

(Swept before the transition and triangle fixes above, which later took the
same configuration to 12,530 MiB; the shape of the curve is what matters here.)

The knee is at 1 GiB: past it the score tensor is no longer what sizes the
peak, so tightening buys nothing and costs 13% wall time per halving.
Confidence is flat across the whole sweep, because blocking the query axis is
exact — the softmax still reduces over the entire key axis inside each block.
Before any of this, OpenDDE asked for 76 GiB and died.

OpenDDE is still the heaviest of the five, and some of that is the model rather
than the implementation: its structural refiner runs on 946 sub-residue tokens
for a 488-residue job, with a 384-channel pair representation. That tensor is
1,311 MiB in float32 where AlphaFold 3's `[488, 488, 128]` bfloat16 pair is
61 MiB, and the refiner keeps several of them live at once. `--trunk-dtype
bf16` narrows the element width for 11,722 MiB at a cost of 0.10 pLDDT; it is
off by default because that cost is real, small as it is. Protenix runs bf16 by
default, where it is worth 8,616 -> 6,043 MiB.

The same lesson every time, and never in the same direction twice: **attribute
the peak, then turn the knob it points at.** Capping the MSA halved Protenix
and did nothing for OpenDDE; blocking the attention halved OpenDDE and did
nothing for Protenix; and blocking the transition then halved Protenix again,
which made the MSA cap it had started with almost irrelevant. Every one of
those was read out of XLA's buffer assignment. None was guessable from the
model description, and two of our confident predictions were wrong.

`foldjax plan` shows exactly what a request resolves to before anything loads:

```console
$ uv run foldjax plan --model protenix --input job.yaml --num-samples 2
{
  "cache_dir": "/home/you/.cache/foldjax/compile",
  "input_format": "foldjax",
  "model": "protenix",
  "output_dir": "foldjax-outputs/job",
  "sampling": {"num_samples": 2},
  "weights": "/home/you/.cache/foldjax/weights/protenix/protenix_base_default_v1.0.0.jax"
}
```

## Weights

Every upstream project publishes torch checkpoints; FoldJAX predicts from
torch-free JAX weights. `foldjax weights` owns that gap — it downloads from each
publisher, verifies against the published hash where one exists, converts once,
and stores the result under the FoldJAX home.

```bash
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
```

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or $XDG_CACHE_HOME/foldjax
├── downloads/<model>/     # released files, exactly as published
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # converted JAX weights — what prediction loads
└── compile/               # XLA persistent compilation cache
```

Conversion needs torch, which is why it is a separate step behind
`--extra torch-bridge`. Prediction never imports it.

No weights are redistributed. Each file is downloaded from its own publisher
under that project's terms, and nothing is fetched implicitly during prediction —
a missing model names the command that would get it.

Chai's conformer archive is an `antipickle` file; upstream reads it by
importing `chai_lab` for its private adapters, which needs torch and pins an
rdkit this environment cannot hold. FoldJAX decodes the format directly
instead, so `--model chai` is one command like the rest.

Weight files exported before these ports were vendored recorded
`protenix_jax.*` / `opendde_jax.*` as their module names. The loader maps those
onto the vendored classes, so **existing weight files keep working**, and the
restricted unpickler still rejects arbitrary globals.

## GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

The old comparison that lived here -- a 35-residue job at one sample and
20 steps, with three of its four upstream memory figures read off
`nvidia-smi` -- has been replaced by
[FoldJAX against upstream](#foldjax-against-upstream), which runs a real
schedule at three sizes and measures both sides the same way.

`--no-cache` turns it off for benchmark or ephemeral runs; `--cache-dir` moves
the root. Every model also runs on CPU, slowly.

## FoldJAX against upstream

Every model here is a JAX reimplementation of a published torch model, so the
only comparison worth making is against the repository it came from, on the
same job, under the same schedule, measured the same way.

That is what [`bench/`](bench/) does, and it is reproducible: one command per
row, one process per measurement, results written as they land.

**The schedule is AlphaFold 3's released default — 5 diffusion samples, 200
diffusion steps, 10 recycles, seed 101 — not a number chosen here.** Protenix's
base model ships the same three. Chai and Boltz-2 default to 3 recycles rather
than 10, so this asks more of them than their own default does; it asks the
same of their upstream, which is what makes the comparison mean anything.

| tokens | model | FoldJAX s | upstream s | FoldJAX MiB | upstream MiB | speed | memory | confidence | FoldJAX | upstream |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | alphafold3 | 256 | - | 1,588 | - | - | - | - | - | - |
| 132 | boltz2 | 14 | 27 | 2,621 | 2,947 | 1.95x | 1.12x | complex_plddt | 0.5750 | 0.7084 |
| 132 | chai | 41 | 51 | 2,322 | 6,486 | 1.25x | 2.79x | aggregate_score | 0.1233 | 0.1232 |
| 132 | opendde | 70 | 18 | 3,655 | 3,784 | 0.25x | 1.04x | ranking_score | 0.0947 | 0.0957 |
| 132 | protenix | 15 | 52 | 1,777 | 2,910 | 3.39x | 1.64x | ranking_score | 0.0996 | 0.0997 |
| 490 | boltz2 | 40 | 53 | 4,585 | 8,128 | 1.34x | 1.77x | complex_plddt | 0.3594 | 0.3629 |
| 490 | chai | 69 | 97 | 5,257 | 9,027 | 1.40x | 1.72x | aggregate_score | 0.0587 | 0.0584 |
| 490 | opendde | 181 | 71 | 10,552 | 18,580 | 0.39x | 1.76x | ranking_score | 0.0973 | 0.0915 |
| 490 | protenix | 33 | 60 | 4,668 | 4,481 | 1.82x | 0.96x | ranking_score | 0.0670 | 0.0659 |
| 970 | boltz2 | 132 | 127 | 9,820 | 14,211 | 0.96x | 1.45x | complex_plddt | 0.5312 | 0.5408 |
| 970 | chai | 180 | 262 | 17,185 | 15,080 | 1.46x | 0.88x | aggregate_score | 0.0790 | 0.0789 |
| 970 | opendde | 537 | 265 | 34,411 | 59,755 | 0.49x | 1.74x | ranking_score | 0.0963 | 0.0966 |
| 970 | protenix | 98 | 94 | 10,807 | 12,315 | 0.96x | 1.14x | ranking_score | 0.0839 | 0.0834 |

Read it with the method in mind:

- **The same job.** The upstream side's input is generated by FoldJAX's own
  translator, so neither side is running a hand-written approximation of the
  other's job file, and both name the same alignment.
- **The same measurement.** Peak memory is the live-bytes high-water mark on
  both sides — `peak_bytes_in_use` under JAX, `max_memory_allocated` under
  torch — never `nvidia-smi`, which reports the caching allocator's reserved
  pool and so tracks its doubling schedule rather than the model.
- **FoldJAX is timed warm.** Each case runs once to fill the XLA compile cache
  and that run is discarded. A cold JAX run is mostly compilation, which the
  torch side has no equivalent of: at 132 tokens it was 174 s against 29 s,
  almost all of it compiling. Compilation is paid once per shape and replayed
  from disk, which is what a second prediction costs.
- **The confidence columns are the same field**, chosen as the best one *both*
  implementations report. They are still different samples — the torch and JAX
  PRNG streams differ, so a shared seed is not a shared random tape. Same-seed
  numerical parity was established per port against a matched tape and is a
  separate exercise from this table.

**Every upstream runs on its own fast path.** Two of them did not at first, and
the differences were large enough to change conclusions rather than shade them,
so they were fixed rather than disclosed:
[`bench/upstream-environments.md`](bench/upstream-environments.md) records what
each virtualenv needed and how to verify it. Boltz-2 and Chai needed nothing.

### What the table says

- **FoldJAX uses less memory almost everywhere.** The exceptions are Chai at
  970 tokens (17,185 against 15,080, 14% more) and Protenix at 490 (4,647
  against 4,481, 4% more). Elsewhere it is 1.04x to 2.79x lower, and the
  advantage grows with size for Boltz-2 and OpenDDE.
- **On speed it wins at small and mid sizes and converges at large.** Boltz-2
  and Protenix are 1.3x-3.3x faster at 132 and 490 tokens and level with
  upstream at 970. Chai is faster at every size and pulls further ahead as the
  job grows (1.46x at 970).
- **OpenDDE is the one model slower than its upstream**, by 2x-4x at every
  size, while using 1.7x less memory at 490 and 970. That is the remaining gap,
  and it is now located: at 132 tokens the whole job is 70 s, of which 31 s is
  the nine extra trunk recycles (~3.4 s each) and just 1 s is the 199 extra
  diffusion steps. Diffusion is effectively free since the sampler was rolled
  into `lax.scan`; the pairformer is what is left. Unrolling that stack instead
  is not the answer -- it does not finish compiling in ten minutes.
- **Confidence agrees for Chai, OpenDDE and Protenix.** Chai matches to four
  decimals at all three sizes, OpenDDE and Protenix to within 1-2%. Protenix
  used to be the exception, and finding out why turned up a real defect --
  see below.
- **Boltz-2 scores systematically lower than its upstream, and that is an open
  defect.** It is not sample variance: 20 samples a side (four seeds, five
  samples each) at 132 tokens are completely disjoint, FoldJAX 0.481-0.566
  against upstream 0.632-0.731. The same sign holds at 490 and 970, where it
  is small enough to have read as noise.

  It is located but not fixed. On the real 132-token features the two trunks
  disagree -- correlation 0.9948 on `s` and 0.9868 on `z` against upstream's
  final trunk state -- and the disagreement is already there after two passes
  and barely grows by eleven, so it is a per-pass difference rather than
  recycling drift. The features entering both trunks are bit-identical, the
  recycle counts mean the same thing on both sides, and the trunk, confidence
  module and diffusion score network each match torch in their own parity
  tests. Those tests run on synthetic features with a handful of MSA rows and
  one to four layers; production is 2,371 rows and 64, which is the only
  regime where the chunked paths engage. That is where to look next.

  Read the Boltz-2 confidence columns as this model scoring its own output
  lower than upstream scores its own. The timing and memory columns are
  unaffected.

### What the Protenix confidence gap turned out to be

Protenix reported pTM 0.61 / 0.64 / 0.87 where upstream reported 0.50 / 0.33 /
0.42. Two things said this was a defect and not a difference of opinion: at 490
tokens FoldJAX's pLDDT was *lower* while its pTM was higher, so it was not
simply predicting better structures; and AlphaFold 3, which this whole family
descends from, reported 0.37-0.39 on the same sequence -- next to upstream
Protenix, far from FoldJAX.

The sharpest symptom was that **FoldJAX's confidence responded the wrong way to
recycling**: 1 cycle to 10 took its pTM from 0.745 down to 0.612, where
upstream went from 0.116 up to 0.498. All three heads (pLDDT, PAE, PDE)
disagreed together, which points at their shared input -- the trunk -- rather
than at any head.

MSA depth, MSA policy, trunk dtype, upstream's fused layer norm, the recycling
initialisation and the entire pTM computation were each ruled out by
measurement without finding it. What found it was running both trunks stage by
stage on the same feature tensors:

| stage | upstream | FoldJAX, before |
|---|---|---|
| `s_inputs` | 0.06949 | 0.06949 |
| `z_init` | 7.9353 | 7.9348 |
| after recycle | 8.8778 | 8.8772 |
| after template | 24.5211 | 24.5211 |
| **after MSA** | **36.70** | **737.32** |

Every stage before the MSA module agreed to five decimals. **The MSA block ran
its two sub-updates in the wrong order**: Protenix does the communication first
(`z += outer_product_mean(m)`) and then hands the MSA stack the pair
representation that mean just produced, while the port did the MSA stack first
and fed the outer product mean an already-updated `m`. Both halves were reading
the other's output.

Nothing failed, because nothing could: the two arrangements have identical
shapes and dtypes, and produce a plausible number either way. The trunk simply
converged to the wrong state -- and since it recycles that state, the error
compounded, which is why more recycling made the confidence worse. `z` came out
of the first cycle at std 428 against upstream's 44 and then oscillated between
107 and 143 for ten cycles instead of settling at 46.8.

Fixed, the trunk tracks upstream cycle for cycle and the confidence lands on it
at all three sizes, at unchanged time and memory. Boltz-2's own order is
different and matches *its* upstream; Chai already matched; OpenDDE has no such
module.

Two things had to be wrong at once for this to survive:

- The block-composition test built its expectation by calling the same
  sub-functions in the same order as the implementation, so it agreed with
  whichever order the implementation used. It now records what each sub-update
  is *handed*, which is the only thing that tells the two arrangements apart.
- `parity_matched_noise.py` compared the trunk against a captured upstream run
  and *printed* the correlation without checking it. It now takes a threshold
  and exits non-zero.

Protenix was also the only port with no torch-anchored test at all -- Boltz-2
has a family of `*_checkpoint_parity.py` that load the real torch modules, and
Chai has its own. `tests/models/protenix/scripts/trunk_stage_parity.py` is the
attribution tool that found it, kept as a runnable check.

### Two earlier claims retracted

Both came from an upstream that was not running its own fast path, and both
were corrected by fixing the environment rather than by reinterpreting the
numbers. See [`bench/upstream-environments.md`](bench/upstream-environments.md).

- **"OpenDDE at 970 tokens runs in FoldJAX and cannot run upstream at all."**
  Not true. Upstream had no `cuequivariance`, so its `auto` triangle kernels
  fell back to plain torch and it ran out of memory. With the cu13 build
  installed it completes in 265 s at 59,755 MiB — its 490-token peak also fell
  from 91,191 MiB to 18,580. FoldJAX still uses 1.7x less memory there, but the
  "cannot run" claim was about the environment, not the model.
- **"Protenix is 1.9x-3.4x faster."** At 970 tokens it is level. Upstream's
  fused layer norm is a CUDA extension this host could not build, and its
  architecture list stopped one generation before this GPU; with both fixed,
  upstream's 970-token time went from 235 s to 94 s against FoldJAX's 98 s.

`alphafold3` has a FoldJAX column but no upstream one: FoldJAX drives the
AlphaFold 3 installation you provide rather than reimplementing the model, so
both columns would be the same code. Its row is there because it is the
reference this architecture family descends from, and because it confirms the
neutral knobs reach it -- 5 samples, 200 steps and 10 recycles all applied.
It also has the lowest peak of any model here, 1,588 MiB, which is what a
trunk that is bfloat16 throughout costs.

## Python API

```python
from foldjax import PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="chai",
        input="job.yaml",
        num_samples=5,
        seed=42,
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

`foldjax.resolve_request` applies every default without running anything, and
`foldjax.capabilities(model)` reports the input formats, entity types, and
sampling knobs a backend honours. Backend-specific output stays on
`PredictionResult.raw`.

## Environment

One Python 3.12 environment, no sibling checkouts:

- JAX / CUDA 13 plugin 0.10.1, cuequivariance-jax 0.10.0 with CUDA 13 ops 0.9.0,
  Tokamax 0.0.12

```bash
uv sync --extra cuda13 --group dev                       # predict with any model
uv sync --extra cuda13 --extra torch-bridge --group dev  # + convert checkpoints
```

**The prediction environment is torch-free for every model.** Boltz-2's
featurizer is vendored from upstream Boltz and written against torch; it runs
here on a NumPy array layer (`foldjax.models.boltz2.data._numpy_torch`) that
reproduces the torch featurizer exactly. `--extra boltz-preprocess` installs
real torch so the two can be compared, which is what
`tests/models/boltz2/test_featurize_torch_parity.py` does.

Two further backends stay external because they cannot be vendored.

**AlphaFold 3** is fully wired and runs from the same job file as the other
four — `foldjax predict --model alphafold3 --input job.yaml` — but FoldJAX
drives an installation you provide rather than carrying one. Its parameters are
not redistributable, and upstream pins `jax[cuda12]`, which cannot co-resolve
with a CUDA 13 cluster; the right JAX plugin differs per site. It is therefore
deliberately absent from the dependency set.
[docs/alphafold3.md](docs/alphafold3.md) has the install for either CUDA
generation, where the weights go, and what changes on a device whose shared
memory the upstream Triton kernels do not fit.

**OpenFold3** is registered the same way and for the same reason, but the port
behind it (`openfold3-jax`) is still in progress and is published on no index.
It is therefore not a dependency either — naming it would break `uv sync` for
everyone without that checkout, since uv resolves every extra to build the
lockfile. [docs/openfold3.md](docs/openfold3.md) has the install and the
current state of the port.

## Tests

```bash
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q
uv run --extra cuda13 --group dev ruff check .

# One file, or one test, with no coverage gate in the way:
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q tests/test_cache.py

# The 80% orchestration-layer gate, which only means anything on a full run.
# This is what CI enforces.
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q \
    --cov=foldjax --cov-report=term-missing --cov-fail-under=80
```

1,020 tests pass with the optional extras installed, 85% coverage on the
orchestration layer. Each vendored port brought its own suite to
`tests/models/<name>/`, including the torch-parity gates it was built against.
Those are excluded from collection unless the extra they need is installed — so
the torch-free environment runs the subset that does not need one — and
`tests/models/test_optional_suite_gate.py` keeps that exclusion list honest
rather than letting a suite go quietly uncollected.

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Model weights are covered by their own licences and are never redistributed
here. AlphaFold 3 goes further — its parameters may not be redistributed at
all, so it is the one backend FoldJAX drives rather than carries. See
[docs/alphafold3.md](docs/alphafold3.md).
