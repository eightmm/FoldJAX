# Engineering notes

The long-form record of how the ports got to their current peaks and parity --
moved out of the README when it grew past reading length. Each section is kept
verbatim from when it was written; `EXPERIMENT_LOG.md` in the repository root
carries the day-by-day version with the measurements inline.

### Memory

For what these models cost against their own upstream repositories, under one
schedule and one measurement, see
[FoldJAX against upstream](benchmark.md). This section is about
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
the time this was measured. OpenFold3 was vendored later, and this line used to
say it had not been through the pass; it has since -- its `outer_product_mean`
and its pair transition both block their rows (`models/msa.py`,
`models/pair_block.py`). ESMFold2, vendored later still, had not, which is what
[Three operations ESMFold2 never had blocked](#three-operations-esmfold2-never-had-blocked)
is about. The transition widens
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
[FoldJAX against upstream](benchmark.md) for what they cost now.
These figures are the size of each step, not a current reading.

**Eager arithmetic between compiled blocks is not free.** Measured on the Chai
port, which has since been removed, but the shape generalises to any trunk run
as many small programs so XLA can release MSA intermediates between them: the
residual `a + b` joining those programs dispatches its own executable, with all
three buffers live. On an MSA representation that is three copies of the largest
tensor in the trunk, for an add. Routing them through a compiled add that donates
its update buffer is the same arithmetic and took that port from 8,637 to
7,576 MiB with scores unchanged to five decimals.

### Three operations ESMFold2 never had blocked

Every other port went through the pass above. ESMFold2 was vendored last and
did not, so it still carried three shapes this repository had already fixed
elsewhere -- and XLA's arena accounting had named all three without anyone
reading it back to the code that owned them.

| what | where | the buffer, at 1,003 tokens |
|---|---|---|
| SwiGLU widened whole | `models/primitives.py` `swiglu` | `bf16[N², 2048]` = **3,930 MiB**, seven of them |
| outer product built whole | `models/trunk.py` `outer_product_mean` | `bf16[N², 1024]` = **1,965 MiB**, four of them |
| the contraction's operands promoted | `models/trunk.py` `triangle_multiplicative` | `[N², 512]` doubled, 982 → **1,965 MiB** |

The sizes are not estimates. `pair_transition.ffn.w12` is `[2048, 256]` in the
checkpoint, and `outer_product_mean.W` is `[64, 128]` giving a 32×32 outer that
flattens to 1,024 -- so the arithmetic on the released widths reproduces the
arena's own figures exactly, and the five `w12` sites (folding trunk, confidence
head, LM encoder, MSA encoder, parcae coda) explain why there were seven.

**The third one was a comment that had inverted its source.** It read "Upstream
forces float32 here", and upstream does the opposite, with its own comment
saying why (`modeling_esmfold2.py:1098`): it casts the fp32 visibility mask
*down* to `routed.dtype` "so masking does not promote the O(N^3) contraction to
fp32". The port reproduced that down-cast faithfully and then promoted the whole
tensor on the next line. Removing it narrows the operands and keeps the float32
accumulation explicitly, through the `preferred_element_type` the einsum already
carried -- `(bf16, bf16) -> f32` in the lowered program.

Measured at 1,003 tokens on the released schedule, two runs per arm, one process
each:

| | wall | peak | pLDDT | pTM |
|---|---|---|---|---|
| before | 17.06 s, 17.07 s | 23.47 GiB | 0.9492 | 0.9798 |
| **after** | **15.88 s, 15.79 s** | **21.08 GiB** | 0.9492 | 0.9797 |

**-2.39 GiB and -1.23 s, for arithmetic that did not change.** The peak is
deterministic -- 23.47 twice, 21.08 twice. Switching the outer product's chunk
back off, with the other two in place, gives 21.87 GiB, so it is worth 0.79 GiB
and the SwiGLU and the dtype are worth 1.60 together.

**A tighter block is worse, which is worth knowing before anyone tunes it.**
The budget is 512 MiB, the same figure Protenix uses for the transition it
shares this shape with. At 128 MiB the peak is 22.55 GiB and at 32 MiB it is
21.76 -- both *above* the 21.08 that 512 gives. More blocks means more
concatenation temporaries, and an arena is a packing rather than a sum, so
bounding the individual buffers harder does not bound their arrangement. That
the independently chosen 512 lands at the optimum here is luck worth recording
rather than a rule.

Blocking a leading axis is exact -- nothing reduces across it -- but not
bit-identical, because the blocked shape draws a different GEMM tiling. The
structures moved 0.0209 Å all-atom paired by sample, against a rerun floor of
0.0205 Å before and 0.0368 Å after, and a 0.9192 Å spread between this model's
own diffusion samples. That is at the floor: the change is not distinguishable
from running the same arm twice.

### What raising the sample count costs

The benchmark runs five diffusion samples because that is AlphaFold 3's
released default. Asking for more is a thing users do, and until 2026-08-28 how
well it went depended entirely on which port you asked. Measured at 1,003
tokens, peak GiB against `--num-samples`:

| model | 1 | 5 | 16 | 32 | 1 → 32 |
|---|---|---|---|---|---|
| protenix | 6.6 | 6.6 | 7.3 | 7.5 | 1.13x |
| esmfold2 | 20.9 | 21.1 | 21.1 | — | ~1.0x |
| opendde | 19.0 | 19.1 | — | 31.3 | 1.65x |
| boltz2 | 6.7 | 6.8 | 11.1 | 20.8 | **3.13x** |
| openfold3 | 9.0 | 11.0 | 20.9 | 38.2 | **4.24x** |

**The ranking is exactly which ports had a lever.** Protenix resolves a
`diffusion_chunk_size` from the sample count under its default `auto` policy,
and maps its confidence head over the samples; ESMFold2 does the second. Boltz-2
had the confidence half — its `api.py` turns `confidence_sequentially` on
whenever there is more than one sample — and nothing for the rollout. OpenFold3
runs its confidence re-embedding per sample above 750 tokens and had nothing for
the rollout either. OpenDDE's CLI resolved a `diffusion_chunk_size` and then
passed four of the six knobs it had resolved, dropping this one.

So the fix was not a new idea, it was the one Protenix already had, given to
the ports that lacked it under the same name:

| model | 32 samples, before | after |
|---|---|---|
| boltz2 | 20.84 GiB | **6.79** — its own five-sample peak, flat |
| openfold3 | 38.24 GiB | **14.68** — 4.24x down to 1.63x |
| opendde | 31.34 GiB | 31.36 — no change, see below |

The released five-sample runs are untouched to a hundredth of a GiB, because
the rule resolves to no chunk until there is more than one chunk to run. That
is what `> DIFFUSION_CHUNK_SIZE` rather than `>=` buys, and it is why this
could land without re-measuring the benchmark.

**Boltz-2 grew fastest per sample for a reason worth keeping.** Its
conditioning is `jnp.repeat`-ed where OpenFold3's is `broadcast_to`, so the
sample axis costs real copies before the denoiser allocates anything.
OpenFold3's expansion is a view, which is why its width could be made to follow
the coordinates -- `denoise_fn` now widens to `xl_noisy.shape[0]` instead of to
the config -- at no copying cost.

**Unlike a blocked matmul, this is not the same arithmetic.** Each chunk draws
its noise at its own width, so the coordinates are different draws from the same
distribution rather than the same draws computed differently. OpenFold3 is the
exception: its noise is drawn at the full sample width and narrowed, which costs
nothing worth saving -- the draw is `[S, N_atom, 3]`, megabytes against the
denoiser's gigabytes -- so its chunking agrees with the unchunked rollout to
9.5e-06 absolute, which is float32 round-off rather than a different sample.

**OpenDDE's growth was not the rollout, and the chunk it got did not move it**
-- 31.34 GiB before and 31.36 after. The wiring landed anyway, because a CLI
that resolves a setting and drops it is a defect whether or not the setting
would have helped, and XLA's own accounting then said where the growth really
was: at 1,003 tokens the one consolidated graph's **temp arena** went 19,306
MiB at one sample to 29,273 at thirty-two, with arguments flat at 2,069 and
outputs at nothing. Not the rollout, not the outputs -- an intermediate.

It was the confidence head's logits. The head already loops samples one at a
time; the scoring ran *after* that loop, so every sample's
`f32[N, N, 64]` PAE and PDE were stacked, reduced once, and then dropped --
7,859 MiB per stack at this size, which is most of the 9,967 the arena grew.
Protenix had already fixed exactly this and its `_confidence` docstring names
it: "what kept three `f32[num_samples, N, N, 64]` stacks live at once was
stacking every sample's logits and only then reducing them". OpenDDE now scores
inside the loop, the same shape of fix, and against a control taken on the same
commit:

| samples | before | after |
|---|---|---|
| 1 | 20.96 GiB | 20.96 |
| 5 | 20.96 | 20.96 |
| 16 | 22.04 | **20.23** |
| 32 | 31.10 | **20.23** |

Flat, and identical where the released schedule sits. Best `ranking_score`
moved 0.192628 → 0.192674, which is the map boundary rather than a different
answer.

**Three times in this campaign a "regression" was a stale baseline.** The
per-sample route looked 1.9 GiB worse at five samples and again at one, both
against numbers measured several commits earlier, and both vanished the moment
a control was taken on the same commit as the change. On a repository this
active the control is not optional and it is not reusable.

**One name, one constant.** `diffusion_chunk_size` is what the CLI, the Python
API, the backend option set and each port's own signature call it, and
`foldjax.execution.DIFFUSION_CHUNK_SIZE` is the single place the width lives --
Protenix's `PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE` now reads from it. Two ports
grew this knob on the same day; without one home they would have grown two
spellings and two constants, which is the failure `foldjax.execution` exists to
prevent.

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
| ESMFold2 | **bfloat16** trunk, float32 sampler | — | none |

The matmul column names the model-level scope. A fused backend can own the
precision of its internal contractions: OpenFold3's cuEquivariance triangle
path explicitly pins `lax.Precision.DEFAULT`, which is why its shipped GPU
route has a separate numerical envelope from the strict `highest`/XLA gate.

Read against upstream, **two rows diverge from what their publisher ships**,
and only one of them was a decision. The four that match: AlphaFold 3
`model_config.py:34`, Boltz-2 `main.py:1262` (`precision="bf16-mixed"`) with
`main.py:1096` asking for `highest` matmuls, Protenix `configs_base.py:135`,
and OpenFold3 `import_utils.py:33` — its `bf16-mixed` YAMLs are training
configs, not inference.

**OpenDDE** is the declared one: upstream runs float32
(`opendde/config/model_base.py:37`), FoldJAX ships bfloat16 since 2026-08-28,
and the rest of this section is the measurement that justified it.

**ESMFold2 is the undeclared one, found 2026-08-28.** Upstream has exactly one
autocast in the whole model — `transformers/models/esmfold2/modeling_esmfold2.py:2021`:

    use_amp = next(self.esmc.parameters()).dtype == torch.bfloat16
    with torch.autocast(device_type=..., dtype=torch.bfloat16, enabled=use_amp):
        esmc_out = self.esmc(...)

It wraps **the ESM-C language model call and nothing else**, and it is gated on
ESM-C's own parameters already being bfloat16. Everything outside that call —
the inputs embedder, `z_init`, the relative-position and token-bond encoders,
the MSA encoder, the folding trunk, the parcae coda — runs at the checkpoint's
parameter dtype, and the released `config.json` says `dtype = float32`.

FoldJAX casts all ten sub-trees in `TRUNK_PREFIXES`
(`models/esmfold2/models/model.py:320`) to `trunk_dtype`, which is
`"bfloat16"` and is **not reachable from any knob**: `settings_from_config`
never reads it, so no checkpoint can set it, and `with_overrides` exposes only
recycles, samples, steps and MSA depth. So the overlap with upstream is the
language model; the trunk itself is bfloat16 here and float32 there.

The difference is real, it was undocumented, and the harness that would have
caught it asserted the opposite; see
[What `bench/esmfold2_compare.py` was not controlling](#what-benchesmfold2_comparepy-was-not-controlling).
**It is also small, and it was measured rather than assumed** — see
[What ESMFold2's width costs](#what-esmfold2s-width-costs) below, which is why
the default stays where it is.

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

### What ESMFold2's width costs

Measured 2026-08-28 at 1,003 tokens (`L1000_3og2`), ESMFold2's own released
schedule (3 loops, 14 sampling steps) and four diffusion samples, one process
per arm so no peak is another arm's high-water mark, and `trunk_dtype` replaced
on the frozen settings because no knob reaches it.

| arm | wall | peak | pLDDT | pTM |
|---|---|---|---|---|
| float32 trunk | 27.38 s | 32.84 GiB | 0.9492 | 0.9797 |
| **bfloat16 trunk (shipped)** | **16.78 s** | **23.47 GiB** | 0.9492 | 0.9797 |

1.63x the speed on 1.40x less memory, with both confidence figures identical to
four decimals. The structures, as superposed **all-atom** RMSD paired by sample
index — a different metric from the CA numbers above, so read the column
against itself and not against Boltz-2's:

| what varies | ESMFold2 |
|---|---|
| the diffusion sample, one seed | **0.9135 Å** (bf16), 0.9199 Å (fp32) |
| **float32 → bfloat16** | **0.0992 Å** |
| nothing — the same arm run twice, float32 | 0.0291 Å |
| nothing — the same arm run twice, bfloat16 | 0.0230 Å |

**This is the best-resolved width measurement in this document.** On Boltz-2 and
Protenix the bfloat16 arm's own rerun floor swallowed the gap, so the width
could only be bounded. Here the floor is 0.023–0.029 Å and the gap is 0.099 Å —
three to four times the noise, so it is a value rather than a ceiling — while
the model's own sampling moves the structure 0.91 Å, nine times further. The
floor is not zero despite an explicit `jax.random.key`, which is XLA
nondeterminism and the reason a rerun is always the control.

So the divergence from upstream is genuine and it is worth keeping: it costs a
tenth of an Angstrom on a model whose own samples disagree by nine tenths, and
buys back a third of the wall clock and a quarter of the peak. What was wrong
was never the choice, only that three files described it as upstream's.

A seed row is absent because it was not measured here; ESMFold2 has three
deliberate randomness sources at inference, so its seed spread is at least its
sampling spread and the row would not change the reading.

### What `bench/esmfold2_compare.py` was not controlling

The ESMFold2 width divergence above survived because the one harness that
compares the port against torch **asserted the control it did not have**. Its
docstring listed, as the second of three things that make the comparison mean
something:

> **the same precision.** Upstream runs its trunk under bfloat16 autocast on
> CUDA and the port now does too, so this measures implementation rather than
> dtype.

The torch side is built by `_torch_row`, and it does two things that together
make that false. It loads the model with
`ESMFold2Model.from_pretrained(str(weights), load_esmc=False)` — **no dtype
argument**, so every parameter outside ESM-C arrives at the config's
`float32` — and it then calls the model inside nothing but `torch.no_grad()`.
The only autocast in upstream is the one inside `self.esmc`, and
`model.load_esmc(..., precision="bf16")` does make it fire, so the language
model really is bfloat16 on both sides. The trunk is not: float32 there,
bfloat16 here.

The row it emitted then labelled itself `"trunk_dtype": "bfloat16 (autocast)"`
— a **hardcoded string**, not a reading of the model it had just built. So the
harness reported a controlled variable by writing down the value it expected
rather than the one it had, which is the failure mode that makes a comparison
worse than no comparison: it does not merely miss the difference, it certifies
its absence.

Both are now fixed to describe what the process does, and the label is derived
from the loaded parameters instead of asserted. The lesson is the same one
`tests/` keeps relearning — **assert the property, not a proxy for it**, and a
string literal is the weakest proxy there is.

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

### Where this card's ceiling actually is, and what does not move it

Measured 2026-09-01 on current main, at 4,100 tokens (`L4000_1gte`, 1GTE,
4 x 1025 at 1.65 A, a 23,037-row alignment), released schedule, seed 101:

| model | result | peak | of the 85.5 GiB pool |
|---|---|---:|---:|
| alphafold3 | 1,232.8 s | 52.1 GiB | 61% |
| protenix | 2,329.4 s | 75.3 GiB | 88% |
| boltz2 | 3,194.5 s | 76.4 GiB | 89% |
| openfold3 | failed by default; 3,290.6 s with `diffusion_chunk_size=1` | 76.4 GiB | 89% |
| opendde | failed, asked for 88.75 GiB in one block | -- | past the pool, inside the card |

At 4,926 tokens all six fail, five of them asking for a single block larger
than the card (105.65 to 170.01 GiB). So the ceiling on this hardware sits
between those two sizes, and the surviving models' own memory exponents place
it: peak grows as `N^1.77` (Protenix) to `N^1.86` (Boltz-2), which reaches the
pool at **about 4,350-4,400 tokens**.

**Time stops tracking the model once the arena approaches the pool.** The
memory exponents stay near-quadratic all the way up, but the time exponent
jumps, and it jumps at the same occupancy in every model that reaches it:

| model, segment | occupancy at the top | time exponent |
|---|---:|---:|
| boltz2 2,096 -> 3,012 | 51% | 2.96 |
| protenix 2,096 -> 3,012 | 51% | 3.04 |
| protenix-v2 2,096 -> 3,012 | 91% | 4.29 |
| boltz2 3,012 -> 4,100 | 89% | 3.71 |
| protenix 3,012 -> 4,100 | 88% | 3.79 |

Around half the pool the exponent is ~3.0; near 90% it is 3.7-4.3.

**This note first said that was XLA rematerialising. That was asserted rather
than measured, and it is wrong.** Protenix was re-run at both occupancies under
`TF_CPP_VMODULE=hlo_rematerialization=2`, which makes the pass report itself on
stderr: **zero lines at 49% and zero at 88%**. The detector is not vacuous --
the same string appears in this project's own OOM logs, where
`hlo_rematerialization.cc:3297` reports the budget it could not meet. A first
attempt to count rematerialisation by grepping the dumped HLO returned zero for
a worse reason: the string never appears in the dump at all, so that counter
could not tell "did not happen" from "is not named that".

**The obvious alternative is dead too, and in the opposite direction.** This is
a 300 W Max-Q part whose SM clock falls to about half its 3,090 MHz ceiling
under load, so the first thing a reviewer proposed was that longer runs sit hot
longer and manufacture an exponent. Logged per run: the 3,012-token run
averaged 1,358.8 MHz, the 4,100-token run **1,658.3 MHz**. The larger job ran at
the *higher* clock. Normalising wall time by mean clock therefore raises the
exponent rather than lowering it, from 3.84 to **4.48**.

So the correlation with occupancy is real and reproducible, and the mechanism
is unidentified. It is not rematerialisation and not throttling. The untested
candidate is that the working set outruns the cache hierarchy at that
footprint, which would surface as effective bandwidth rather than as anything
XLA reports. Do not restate the original claim without measuring it.

Protenix-v2's 4.29 still needs no appeal to architecture: its checkpoint is the
same 4,174 arrays as the base model with one width doubled (`c_z` 128 -> 256,
visible as the dominant trailing dimension in each file), so it reaches any
given occupancy a size earlier. Where both have room -- 1,354 -> 2,096, at 46%
-- their exponents are 1.87 and 1.94. That much survives; only the named cause
does not.

**Four levers were measured against this. One works, on one model.**

*Serialising the diffusion sample axis rescues OpenFold3 and nothing else.* At
4,100 tokens its default route asks for a single 107.85 GiB block against a
95.0 GiB card and fails in 336 s. With `diffusion_chunk_size=1` it completes:
3,290.6 s at 76.4 GiB, 89% of the pool. The same option moved Protenix's peak
from 77,158.2 MiB to 77,157.2 -- one mebibyte -- and cost 16% in time. The two
models' peaks sit in different places: Protenix's is the trunk's pair stack,
which the sample axis does not touch, and OpenFold3's is on the
sample-expanded path, where serialising deletes the offending block outright.
A lever measured as inert on one model is not evidence about the next.

OpenFold3's other kernel knob does nothing here. `OPENFOLD3_TRIANGLE_BACKEND=xla`
selects the blocked triangle *attention* and left the failing request at exactly
107.85 GiB -- the same number to two decimal places, which is what a lever that
does not touch the responsible buffer looks like.

*The blocked triangle multiplication is a trade, not a saving.* The warning in
`triangle.py` said the blocked path is "about 24% cheaper on memory at every
size tried". That 24% is the ratio for that operation's own two projections,
not for a run. End to end on Protenix at 4,100 tokens: peak 77,158.2 MiB fused
against 71,360.6 blocked, **7.5%**, for 2,329 s against 2,715, **+17%**. It
buys 6.6 points of occupancy (88.1% to 81.5%) and rescues nothing.

**Two levers were measured against this, and neither is one.**

*The pool fraction is real but does not rescue these runs.* `bytes_limit`
tracks `XLA_CLIENT_MEM_FRACTION` exactly -- 85.47 GiB at 0.90 and 90.22 GiB at
0.95, read off the device rather than computed. Both failures were retried at
0.95 and both still failed: Protenix at 4,926 needs one contiguous 87.93 GiB
block, and OpenDDE at 4,100 has a program whose own rematerialisation
report -- one of the few places the pass does speak -- puts it at 88.77 GiB
against a 90.22 GiB pool. A pool with 1.45 GiB of slack cannot lay
out that program's co-live set whatever its total says.

*Capping the MSA depth moves almost nothing, at this size too.* The arithmetic
argues loudly that it should: one `[depth, tokens, 64]` copy at 4,100 tokens is
0.5 GiB at AlphaFold 3's 1,024 rows and 8.0 GiB at Protenix's 16,384, while
their pair terms are identical. Measured at `--max-msa-depth 1024`:

| model | shipped default | capped to 1,024 | change |
|---|---:|---:|---:|
| protenix | 77,158.2 MiB | 76,793.6 MiB | -0.5% |
| boltz2 | 78,282.5 MiB | 78,058.2 MiB | -0.3% |
| opendde | fails | still fails | -- |

Half a percent, against an arithmetic prediction of several gigabytes. The
alignment tensor is simply not co-live with the peak: the peak is the pair
stack, and the MSA representation is gone by the time it forms. This is the
same result [`--max-msa-depth`](#--max-msa-depth) recorded at 488 tokens
(0.9%), and the reason to record it again is that the earlier note left room
for the cap to matter more at a size where the MSA term is large and the pool
is nearly full. It does not. **Treat the cap as an accuracy knob at every
size**, and note that this also disposes of the tempting explanation for why
AlphaFold 3 completes the same 4,100-token job at 61% of the pool while
Protenix needs 88%: it is not the 16x deeper alignment.

### Protenix's peak moves with the seed, through the alignment

The depth cap is not the depth the trunk reads. Protenix draws a fresh row
sample every recycle, and the size of that sample is itself random --
`sample_msa_cycle_index_tape` (`trunk_blocks/msa.py`) does
`rng.integers(1, n_msa + 1)` per cycle, which is upstream's own choice, then
takes that many rows out of a permutation. The compiled program has to hold the
largest of those draws, rounded up to a 64-row bucket, so one number decides the
MSA tensor for the whole run: the maximum over `num_recycles` uniform draws.

With the released 16,384-row cap and ten recycles that maximum is usually close
to the cap but not reliably so:

| seed | rows the program is sized for | of the cap |
|---|---:|---:|
| 101 (the benchmark's) | 15,488 | 94.5% |
| 0 | 13,952 | 85.2% |
| 2 | 13,760 | 84.0% |
| over 2,000 seeds | 8,448 min / 15,296 median / 16,384 max | 51.6% - 100% |

So two seeds of the same job can differ by nearly a factor of two in the MSA
tensor alone. Three consequences worth keeping:

- **A benchmark row is one draw.** The table's Protenix numbers are seed 101's
  94.5%, not an average, and re-running at another seed moves the peak for a
  reason that has nothing to do with the code.
- **Near a memory limit, the seed decides whether a job runs.** An OOM that
  disappears when the seed changes is this, not flakiness.
- **`--max-msa-depth` scales the whole distribution**, because it lowers
  `n_msa` and every draw is uniform on `[1, n_msa]`. That is why capping
  Protenix is close to free on accuracy: it does not remove rows the model was
  relying on, it narrows a range upstream was already sampling from.

The other five do not have this. AlphaFold 3 and OpenFold3 take a fixed 1,024,
ESMFold2 redraws a fixed 1,024 per loop, OpenDDE samples per cycle but sizes to
its cap, and Boltz-2 subsamples only behind a flag that is off by default in
both implementations -- so Boltz-2 carries all 8,192 parsed rows into every
pass, the only one of the six that does.

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
numbers. See [`bench/upstream-environments.md`](../bench/upstream-environments.md).

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
