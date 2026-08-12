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
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `openfold3` | `foldjax.models.openfold3` | Apache-2.0 | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

— plus `alphafold3`, which FoldJAX drives rather than carries, because its
parameters may not be redistributed. It takes the same job file as the rest
once installed; see [docs/alphafold3.md](docs/alphafold3.md).

OpenFold3 is carried twice over: the JAX port is a reimplementation, and
upstream's own *data pipeline* is vendored beside it, so featurization is exact
rather than approximately right and still needs nothing outside this repository.
Building features needs `--extra openfold3-preprocess`, because that pipeline is
upstream's torch code; predicting needs neither it nor torch. See
[docs/openfold3.md](docs/openfold3.md).

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
passed through untouched. So a Boltz YAML or a Protenix job list works as-is.

Modifications and covalent bonds have one model-neutral spelling here and are
translated into each backend's own names — Protenix and OpenDDE want
`CCD_`-prefixed types and entity-indexed `covalent_bonds`, Boltz wants
`{ccd, position}` and a `bond` constraint, AlphaFold 3 wants bare `ptmType`.

**A field a backend cannot express is rejected, not dropped.** Boltz has no place
for a separate `paired_msa`, because it derives pairing from one per-chain a3m.
Silently discarding an alignment would change the science without changing the
exit code.

`unpaired_msa` reaches every model. That matters more than it sounds: a job that
pins an alignment must pin it everywhere, or any comparison across models becomes
a comparison of MSAs.

The same document can be built in Python instead of written by hand:

```python
from foldjax import Job, Ligand, Modification, Protein

job = Job(
    "example",
    [
        Protein("A", "ACDEFG", unpaired_msa="msa.a3m",
                modifications=[Modification("SEP", 3)]),
        Ligand("L", ccd="ATP"),
    ],
)
request = PredictionRequest(model="protenix", input=job.write("job.json"))
```

`Job` is a constructor, not a second schema: `to_document()` returns exactly the
mapping above, and every rule about what a document may say stays where the
model is known, so the class cannot drift from what the CLI accepts. `Job.read`
takes either file format back the other way.

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

| neutral | alphafold3 | boltz2 | openfold3 | opendde / protenix |
|---|---|---|---|---|
| `--num-samples` | `heads.diffusion.eval.num_samples` | `diffusion_samples` | `num_samples` | `n_sample` |
| `--num-steps` | `heads.diffusion.eval.steps` | `steps` | `no_rollout_steps` | `n_step` |
| `--num-recycles` | `num_recycles` | `recycling` | `num_cycles` | `n_cycle` |
| `--max-msa-depth` | `evoformer.num_msa` | `max_msa_depth` | *feature cut* | `max_msa_rows` |

All four reach all five models. AlphaFold 3's step count and MSA depth are
not arguments of `make_model_config`, but the config it returns is an ordinary
mutable object and upstream already sets the sample count on it by assignment;
FoldJAX sets these the same way. That matters beyond convenience — a model that
cannot be held to the same schedule as the others cannot be compared with them.

Setting both a neutral knob and its native name is an error rather than a silent
preference. OpenFold3 has no MSA-depth argument at all, so there the knob is
applied to the features: the first *n* alignment rows are kept, in order, and
the per-token `profile` and `deletion_mean` are left as computed over the whole
alignment — which is what the other backends' native caps also do. It used to be
refused, which left the one model that could not be held to the same MSA budget
as the rest, and that budget is the dominant memory knob.
`--option KEY=VALUE` still passes anything native straight through.

Seeds work the same way. `--seed` runs one, `--seeds 0 1 2` runs those three,
and `--num-seeds 3` is the same as the latter — counting up from `--seed`, so
`--seed 7 --num-seeds 3` is 7, 8 and 9. Each seed runs the whole job and every
structure comes back in one result.

`--seeds 1 2 3` runs the job once per seed, into a `seed_<n>` directory each,
and returns every structure together. The samples from one seed are correlated,
so this is the usual way to get independent predictions. Only some backends take
a seed list natively, so the loop lives in FoldJAX and every model gets it
identically.

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
10 GiB, and at 490 residues under the released schedule it peaks at 13.3 GiB
against upstream torch's 18.6 GiB. (That 13.3 is fp32, matching upstream's own
default. The 10.5 GiB this line used to quote was a bfloat16 run being compared
against an fp32 upstream, which is not the same measurement.)

Each of the changes below came from reading XLA's buffer assignment for the
peak and fixing whatever it named — a habit worth more than any individual fix
in the list.

**The transition was the largest buffer in three of the four ports** carried at
the time this was measured; OpenFold3 was vendored later and has not been
through this pass. It widens
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

**Eager arithmetic between compiled blocks is not free.** Measured on the Chai
port, which has since been removed, but the shape generalises to any trunk run
as many small programs so XLA can release MSA intermediates between them: the
residual `a + b` joining those programs dispatches its own executable, with all
three buffers live. On an MSA representation that is three copies of the largest
tensor in the trunk, for an add. Routing them through a compiled add that donates
its update buffer is the same arithmetic and took that port from 8,637 to
7,576 MiB with scores unchanged to five decimals.

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

**Where the cap has to act is model-specific, and getting it wrong moves
nothing.** The now-removed Chai port embedded the selected rows once into a
`[1, depth, tokens, 64]` tensor and the trunk read *that*, so capping the
per-recycle crop left the embedding at full depth and changed neither memory nor
scores. The lever was the selection feeding the embedding, and it was worth a
third of the peak for one point of pLDDT — a real accuracy cost, unlike
Protenix, where the cap only trades noise.

Boltz-2 is the least sensitive — 4,766 → 4,688 MiB at 1024, for 0.891 → 0.880
pLDDT — which is not worth spending, so the cap is off there too.

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

OpenDDE is still the heaviest model in the table above, and some of that is the
model rather
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
uv run foldjax setup                            # everything public, in one go
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
```

`setup` is the one to run on a new machine. It fetches and converts every model
whose weights are published — Boltz-2, OpenDDE, Protenix — along with the CCD
assets and the template metadata they share, and then reports what it could not
do for you: AlphaFold 3 and OpenFold3, whose publishers release parameters on
request, and whatever the template modality is still missing. MSA needs nothing
installed; it searches against the ColabFold MMseqs2 server.

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or a .foldjax/ in the checkout,
├── downloads/<model>/     #   or $XDG_CACHE_HOME/foldjax — `foldjax home` says
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # converted JAX weights — what prediction loads
└── compile/               # XLA persistent compilation cache
```

`mkdir .foldjax` in a checkout and everything above lands there instead, which
keeps weights, assets and caches beside the source they belong to. `.gitignore`
already covers the directory, and `$FOLDJAX_HOME` still outranks it.

Conversion needs torch, which is why it is a separate step behind
`--extra torch-bridge`. Prediction never imports it.

No weights are redistributed. Each file is downloaded from its own publisher
under that project's terms, and nothing is fetched implicitly during prediction —
a missing model names the command that would get it.

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
base model ships the same three. Boltz-2 defaults to 3 recycles rather than 10,
so this asks more of it than its own default does; it asks the same of its
upstream, which is what makes the comparison mean anything.

| tokens | model | FoldJAX s | upstream s | FoldJAX MiB | upstream MiB | speed | memory | confidence | FoldJAX | upstream |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | alphafold3 | 256 | - | 1,588 | - | - | - | - | - | - |
| 132 | boltz2 | 14 | 27 | 2,981 | 2,947 | 1.93x | 0.99x | complex_plddt | 0.7260 | 0.7084 |
| 132 | opendde | 25 | 17 | 3,651 | 3,784 | 0.66x | 1.04x | ranking_score | 0.0947 | 0.0957 |
| 132 | protenix | 15 | 52 | 1,777 | 2,910 | 3.39x | 1.64x | ranking_score | 0.0996 | 0.0997 |
| 490 | boltz2 | 47 | 53 | 6,222 | 8,128 | 1.13x | 1.31x | complex_plddt | 0.3619 | 0.3629 |
| 490 | opendde | 80 | 73 | 13,345 | 18,580 | 0.91x | 1.39x | ranking_score | 0.0967 | 0.0915 |
| 490 | protenix | 33 | 60 | 4,668 | 4,481 | 1.82x | 0.96x | ranking_score | 0.0670 | 0.0659 |
| 970 | boltz2 | 150 | 127 | 11,570 | 14,211 | 0.85x | 1.23x | complex_plddt | 0.5391 | 0.5408 |
| 970 | opendde | 272 | 271 | 44,519 | 59,755 | 1.00x | 1.34x | ranking_score | 0.0964 | 0.0966 |
| 970 | protenix | 98 | 94 | 10,807 | 12,315 | 0.96x | 1.14x | ranking_score | 0.0839 | 0.0834 |
| 1531 | boltz2 | 382 | 302 | 21,174 | 26,491 | 0.79x | 1.25x | complex_plddt | 0.7172 | 0.7195 |
| 1531 | opendde | failed | failed | failed | failed | - | - | - | - | - |
| 1531 | protenix | 254 | 176 | 24,764 | 27,918 | 0.69x | 1.13x | ranking_score | 0.1200 | 0.1193 |

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
each virtualenv needed and how to verify it. Boltz-2 needed nothing.

### What the table says

- **FoldJAX uses less memory almost everywhere.** Two rows are level or slightly
  worse: Boltz-2 at 132 tokens (2,981 against 2,947, 1% more) and Protenix at 490
  (4,668 against 4,481, 4% more). Elsewhere it is 1.04x to 1.64x lower.
- **On speed it wins at small and mid sizes and loses at large.** Boltz-2 and
  Protenix are 1.1x-3.4x faster at 132 and 490 tokens, level at 970, and slower
  at 1,531.
- **OpenDDE is the slowest relative to its upstream**, and the gap is smaller
  than it looked. Earlier readings of 2x-4x were taken at bfloat16 against an
  fp32 upstream and were not a matched comparison: upstream runs `dtype: fp32`
  with `enable_tf32: True`, not bf16. Re-measured with the dtypes agreeing, it
  is 0.66x at 132 tokens, 0.91x at 490 and level at 970, while using 1.04x to
  1.39x less memory. What remains is the trunk: at 132 tokens the whole job is
  70 s, of which 31 s is the nine extra recycles (~3.4 s each) and just 1 s is
  the 199 extra diffusion steps. Diffusion is effectively free since the sampler
  was rolled into `lax.scan`; the pairformer is what is left. Unrolling that
  stack is not the answer -- it does not finish compiling in ten minutes.
- **At 1,531 tokens OpenDDE runs out of memory on both sides**, in fp32 on a
  96 GiB card. That row is recorded as a failure rather than dropped, and
  upstream's carries `"exited 0 but produced no structures"` -- its runner
  swallows the allocator error and exits zero, so the row exists only because
  the harness checks for structures instead of an exit code. `trunk_dtype=bf16`
  completes it in 479.9 s at 58,747 MiB, kept under its own name because
  upstream has no bf16 counterpart to compare against.
- **Confidence agrees for every model.** OpenDDE and Protenix match to within
  1-2%, Boltz-2 to within 1-2% except at 132 tokens. Two of those took real
  fixes, both found the same way and both described below.
- **OpenFold3 has no row.** It is vendored and runnable, but no upstream
  OpenFold3 environment has been provisioned here, so there is nothing to
  compare a FoldJAX number against. See [docs/openfold3.md](docs/openfold3.md).

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
different and matches *its* upstream; OpenDDE has no such module.

Two things had to be wrong at once for this to survive:

- The block-composition test built its expectation by calling the same
  sub-functions in the same order as the implementation, so it agreed with
  whichever order the implementation used. It now records what each sub-update
  is *handed*, which is the only thing that tells the two arrangements apart.
- `parity_matched_noise.py` compared the trunk against a captured upstream run
  and *printed* the correlation without checking it. It now takes a threshold
  and exits non-zero.

Protenix was also the only port with no torch-anchored test at all, where
Boltz-2 has a family of `*_checkpoint_parity.py` that load the real torch
modules. `tests/models/protenix/scripts/trunk_stage_parity.py` is the
attribution tool that found it, kept as a runnable check.

### What the Boltz-2 confidence gap turned out to be

Boltz-2 scored its own structures systematically lower than upstream scored
upstream's. It was not sample variance, which is what it had first been closed
as: 20 samples a side at 132 tokens -- four seeds, five samples each -- were
completely disjoint, 0.481-0.566 against 0.632-0.731, and the same sign held at
490 and 970 where it was small enough to read as noise.

The trunk, confidence module and diffusion score network each matched torch in
their own parity tests, the features entering both models were bit-identical,
and a fan-out comparing every trunk subsystem against upstream raised 21
candidate divergences of which **none survived refutation** -- one of them
measuring `corr(s) = corr(z) = 1.00000000` for the 64-layer pairformer on the
real checkpoint at production shape.

The cause was not in the arithmetic at all. `boltz predict` overrides its own
`MSAModuleArgs` default and passes `subsample_msa=True`, so `MSAModule.forward`
draws a fresh `torch.randperm(n_msa)[:1024]` **on every forward pass** -- eleven
recycles see eleven different slices and ensemble over the whole alignment. The
port subsampled to the same depth but always took the top 1,024, and took the
same 1,024 again every pass. Its docstring described this as following
AlphaFold 3's deterministic truncation, which is defensible in isolation and is
not what Boltz-2 computes.

| 132 tokens, 20 samples | min | max | mean |
|---|---|---|---|
| before, fixed top 1,024 | 0.481 | 0.566 | 0.530 |
| after, fresh draw per pass | 0.590 | 0.724 | 0.645 |
| upstream | 0.632 | 0.731 | 0.673 |

Two distributions that shared no values now largely coincide, and the mean gap
falls from 0.143 to 0.028. The residue is expected rather than outstanding:
torch's `randperm` and JAX's `permutation` draw *different* subsets from the
same alignment, so these are two different draws, not the same one.

The key is derived with `fold_in` rather than `split`, so the sampler keeps the
exact noise stream it had; otherwise every coordinate would have moved too and
an MSA-selection change would have read as a diffusion change.

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

One request type, every model. The knobs a request carries are model-neutral:
the same spelling means the same thing whichever backend runs it, and a knob a
backend cannot honour raises instead of silently doing something else.

```python
from foldjax import PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="protenix",          # boltz2 / opendde / openfold3 / alphafold3
        input="job.yaml",          # FoldJAX job, or the model's native format
        output_dir="out/",
        seed=42,
        # Sampling, in one vocabulary. None keeps that backend's default.
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=8192,
        # Execution, likewise: dtype, matmul_precision, triangle_kernel,
        # attention_kernel. "auto" is the fastest path this build can run --
        # never a silent fallback to a slower one.
        options={"dtype": "bfloat16", "triangle_kernel": "auto"},
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

`sample.scores` carries that model's released confidence summaries (pLDDT,
pTM/ipTM, ranking) per structure. The native option spellings each port grew
up with (`compute_dtype`, `trunk_dtype=bf16`, ...) still work and warn once;
`foldjax.execution.KNOBS` lists the neutral vocabulary.

Programs return summaries, not raw tensors, by default: the full-bin PAE/PDE
logits are `[n_samples, N, N, 64]` -- tens of GiB at long sequences -- so they
stay out of the outputs unless asked for (`return_confidence_logits=True` on
the Boltz-2 library API, `--include-raw` on the Protenix/OpenDDE CLIs, the raw
`.npz` formats elsewhere).

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

One backend stays external because it cannot be vendored.

**AlphaFold 3** is fully wired and runs from the same job file as the other
five — `foldjax predict --model alphafold3 --input job.yaml` — but FoldJAX
drives an installation you provide rather than carrying one. Its parameters are
not redistributable, and upstream pins `jax[cuda12]`, which cannot co-resolve
with a CUDA 13 cluster; the right JAX plugin differs per site. It is therefore
deliberately absent from the dependency set.
[docs/alphafold3.md](docs/alphafold3.md) has the install for either CUDA
generation, where the weights go, what changes on a device whose shared memory
the upstream Triton kernels do not fit — and why you want `uv sync --inexact`
afterwards, since an exact sync removes the one package that cannot be locked.

**OpenFold3 used to be the second**, for a reason vendoring removed: the port
was published on no index, so naming it would have broken `uv sync` for everyone
without that checkout. It is now carried at `foldjax.models.openfold3` and needs
no extra to predict — it added no base dependency, because its inference path
wanted only `jax`, `numpy`, `safetensors` and `gemmi`, all already here.

Its *featurization* used to be the one thing in FoldJAX that was not
self-contained: it delegated to upstream OpenFold3's data pipeline through a
sibling checkout, so building features required a second repository. That
pipeline is vendored now, at `models/openfold3/_upstream/` — upstream's own code,
verbatim, under its own licence and in its own style. Featurization needs
`--extra openfold3-preprocess` for the torch and lightning it imports;
prediction needs neither.
[docs/openfold3.md](docs/openfold3.md) has the detail and the current state of
the port.

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

1,669 tests pass with the optional extras installed, 88 skipped, 87% coverage on
the orchestration layer. Each vendored port brought its own suite to
`tests/models/<name>/`, including the torch-parity gates it was built against.

Those gates need an extra the torch-free environment deliberately omits, and the
suites handle it two different ways. Boltz-2's import torch at module scope, so
they have to be excluded at *collection* time;
`tests/models/test_optional_suite_gate.py` keeps that exclusion list honest
rather than letting a suite go quietly uncollected. OpenFold3's import torch
inside the test bodies instead, so they collect anywhere and skip at run time —
the better arrangement of the two, since a skipped test is reported and an
uncollected one is not.

Two things the suite used to need from outside are now in the repository, so
nothing skips for want of a package that is on no index. The shared ColabFold
MMseqs2 client is `foldjax.search` — it has no third-party dependency, so
carrying it costs nothing and `--msa-server` works out of the box. The real-PDB
target loader is `tests/_foldbench`, kept under `tests/` rather than `src/`
because it fetches entries from RCSB over the network, which the installed
package has no business being able to do.

The tests built on real depositions still skip without `biotite`, which parses
them:

```bash
uv sync --extra cuda13 --extra openfold3-preprocess   # adds biotite
```

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Every port began as its own repository, and each carried notes the code alone
does not: an experiment log, a porting plan, the gates it had to pass, and the
benchmarks those gates were argued from. Those are in
[docs/ports/](docs/ports/), so nothing is left behind in a checkout that no
longer exists. They are history rather than instructions — the module names and
commands in them predate vendoring — but they are the only record of why a
given default was chosen and what was tried and rejected first.

Model weights are covered by their own licences and are never redistributed
here. AlphaFold 3 goes further — its parameters may not be redistributed at
all, so it is the one backend FoldJAX drives rather than carries. See
[docs/alphafold3.md](docs/alphafold3.md).
