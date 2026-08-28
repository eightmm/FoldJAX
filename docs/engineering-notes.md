# Engineering notes

The long-form record of how the ports got to their current peaks and parity --
moved out of the README when it grew past reading length. Each section is kept
verbatim from when it was written; `EXPERIMENT_LOG.md` in the repository root
carries the day-by-day version with the measurements inline.

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

### Which precision each model runs

Two independent axes. **Element width** is what the trunk's weights and
activations are stored and computed in; **matmul precision** is how XLA
executes the float32 matmuls that remain. A model can be bfloat16 and
`highest`, or float32 and TF32, and Boltz-2 upstream is the first of those.

| model | element width | matmul precision | knob |
|---|---|---|---|
| AlphaFold 3 | bfloat16 | — | none; upstream is `bfloat16: 'all'` |
| Boltz-2 | bfloat16 | `highest` | `--option dtype=float32` |
| Protenix / v2 | bfloat16 | `high` (TF32) | `--option dtype=float32` |
| OpenDDE | **bfloat16** | `high` (TF32) | `--option dtype=float32` |
| OpenFold3 | float32 | `high` (TF32) | none |
| ESMFold2 | float32 | — | none |

Read against upstream, every row matches what its publisher ships **except
OpenDDE**, which upstream runs in float32 (`opendde/config/model_base.py:37`).
The others: AlphaFold 3 `model_config.py:34`, Boltz-2 `main.py:1262`
(`precision="bf16-mixed"`) with `main.py:1096` asking for `highest` matmuls,
Protenix `configs_base.py:135`, OpenFold3 `import_utils.py:33` — its
`bf16-mixed` YAMLs are training configs, not inference — and ESMFold2's
released `config.json`, whose autocast fires only when the parameters are
already bfloat16.

Only OpenDDE and Protenix expose the width as an option, and only OpenDDE has
anywhere to go with it, because the other four are either already narrow or
have no knob at all.

### What the width costs, measured

All three ports with the knob, at 1,003 tokens (`L1000_3og2`), five samples,
two runs per arm, both arms named explicitly — a default is not an arm:

| model | float32 | bfloat16 | confidence |
|---|---|---|---|
| boltz2 | 164.8 s, 10.14 GiB | **94.6 s, 6.87 GiB** | `complex_plddt` 0.9486 → 0.9490 |
| protenix | 88.7 s, 11.64 GiB | **66.5 s, 6.73 GiB** | `ranking_score` 0.1921 → 0.1921 |
| opendde | 280.4 s, 42.96 GiB | **174.4 s, 19.18 GiB** | `ranking_score` 0.1926 → 0.1926 |

Faster *and* smaller on every one, with the score unchanged. That is the whole
case for the narrow default, and for OpenDDE it is why the default moved on
2026-08-28. It also moves the wall: at 1,531 tokens OpenDDE float32 asks for
86.7 GiB and dies where bfloat16 finishes at 44.2 GiB.

**What the width costs is repeatability, not accuracy — and it is small.**
Read any of these numbers without the ladder they sit in and they look like
something. Here is the whole ladder, same job, same tool, median CA RMSD:

| what varies | boltz2 | opendde |
|---|---|---|
| the seed | **1.908 Å** | **2.431 Å** |
| the diffusion sample, one seed | 1.335 Å | 2.469 Å |
| nothing — the same arm run twice, bfloat16 | 0.4154 Å | 0.0135 Å |
| **float32 → bfloat16** | **0.3630 Å** | **0.0301 Å** |
| nothing — the same arm run twice, float32 | 0.0439 Å | 0.0142 Å |

The width sits at the bottom of that ladder. On Boltz-2 it moves the structure
about a fifth as far as changing the seed does; on OpenDDE, a hundredth.
Choosing bfloat16 is a smaller decision than choosing which seed to run.

Two things follow, and they are different claims. **Scientifically** the width
is not a result-changing knob at these sizes. **As a measurement**, Boltz-2 and
Protenix become roughly ten times less repeatable in bfloat16 (0.0439 → 0.4154
Å, 0.0118 → 0.0995 Å), which puts their float32↔bfloat16 gap below the noise of
either arm: at five samples it cannot be resolved, only bounded. OpenDDE's
floor does not move at all, which is the row its default rests on — the gap
sits at about twice a floor that stayed put, with one sample of five moving
0.6140 Å.

The asymmetry follows where the cast lands. OpenDDE narrows the embedder and
both trunks and leaves the diffusion sampler in float32; Boltz-2's
`compute_dtype` reaches the sampler itself, which is the stochastic part. A
width that touches sampling buys memory and spends repeatability.

One more thing the ladder shows: OpenDDE's seeds are not worth much. Changing
the seed moves it 2.431 Å where its own five samples already sit 2.469 Å apart
— the sampling is doing all the exploring. Boltz-2's seeds add 43% on top of
its sample spread. `--seed` on `bench/run_foldjax` is how these were measured;
the table itself stays pinned to one seed so its rows compare.

`bench/spec.py` pins `dtype=float32` for OpenDDE's benchmark row, because a
row that compared two precisions would not be a comparison.

### The other axis, measured

Matmul precision is independent of width, and moves structures far less. Boltz-2
at 499 tokens (`L250_3dha`), the `highest` it pins against the `high` its
neighbours use:

| | median CA RMSD |
|---|---|
| `highest` rerun floor | 0.0093 Å |
| `highest` ↔ `high` | 0.0218 Å |

About twice the floor, cleanly separated — every cross-setting pair exceeded
every within-setting pair. For reference, OpenFold3 recorded the same shape
from the other direction: 0.011 Å against a 0.005 Å floor, for −15% wall.

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
of pLDDT. That 0.5% is why this stayed opt-in for as long as it did. It became
the default on 2026-08-28, once the same comparison had a rerun floor beside it
and the confidence score came back unchanged rather than 0.5% down — see
"Why OpenDDE moved" above. `--option dtype=float32` is still one flag away.

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
