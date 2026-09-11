# Command line

The complete `foldjax` command surface: prediction, batches, padding, memory
knobs, weights and the compile cache.

```bash
uv run foldjax predict \
  --model protenix \
  --input job.yaml \
  --profile mini-esm-v0.5.0 \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10 \
  --seed 42
```

With one model and one input, the compatible default remains
`foldjax-outputs/<input-stem>`. An explicit `--output-dir` replaces that scalar
default. Batch requests add canonical model namespaces below their output root
so different models cannot overwrite one another.

`--num-samples`, `--num-steps`, `--num-recycles` and `--max-msa-depth` are
model-neutral, and every port now spells them that way inside as well. Each
port used to keep whatever its upstream called them, so this table listed six
columns of synonyms -- `diffusion_samples`, `n_sample`, `num_loops`,
`no_rollout_steps`, `max_msa_rows` -- and the adapters translated between them.
The names are FoldJAX's own now, so there is nothing left to translate:

| neutral | every port's own name | where it lands |
|---|---|---|
| `--num-samples` | `num_samples` | AlphaFold 3 `heads.diffusion.eval.num_samples` |
| `--num-steps` | `num_steps` | AlphaFold 3 `heads.diffusion.eval.steps` |
| `--num-recycles` | `num_recycles` | AlphaFold 3 `num_recycles` |
| `--max-msa-depth` | `max_msa_depth` | AlphaFold 3 `evoformer.num_msa`; OpenFold3 cuts the feature |

The right-hand column is upstream's own config path, which is a different
thing and stays upstream's. What changed is FoldJAX's vocabulary, not the
checkpoints or the configs it writes into.

Omit `--num-recycles` to use the selected model's managed default, including
with `--padding` and `cache warm`. Padding does not impose a shared recycle count.
See [recycling defaults](token-padding-profiles.md#recycling-defaults) for the
model-specific settings, paper evidence and actual trunk pass counts. OpenFold3 translates the
common additional-recycle count to its internal total count (3 to 4).

Protenix's and OpenDDE's native CLIs kept their old flags as aliases:
`--n-sample`, `--n-step`, `--n-cycle` and `--max-msa-rows` still work and mean
what they always meant.

All four reach all six models; setting both a neutral knob and its native
name is an error, and `--option KEY=VALUE` passes anything native straight
through. Seeds fan out the same way — `--seeds 0 1 2` (or `--num-seeds 3`)
runs the job once per seed into `seed_<n>` directories and returns every
structure together. Every run writes `foldjax_run.json` beside its structures:
model, input SHA-256, resolved weights and their stat/tree identity, the knobs
actually used, and each structure's confidence.

A run reports its stages on stderr and its results as a table on stdout:

```
[foldjax] protenix · job.yaml · seed 101
  prepare input        1.2s
  predict             6m18s
  write                0.3s

model     protenix        weights  protenix-v0.5.0
samples   15              time     6m21s     peak  18.4 GiB
seeds     101, 102, 103   msa      auto
best      seed 102 / sample 01     ranking_score 0.873
          seed-102_sample-01/job_seed-102_sample-01.cif
```

stdout stays JSON whenever it is not a terminal, so pipes and scripts are
unchanged; `--json` forces it and `--quiet` silences the stage lines
(`FOLDJAX_PROGRESS=0` does the same). The same stage timings are kept in
`foldjax_run.json` under `cost.phases`, because "it took eleven minutes" cannot
be acted on until it is split into a search, a compile and a schedule.
`foldjax show <output-dir>` renders that table again later, for one run or a
whole batch, from the manifests alone.

Batches get two flags that only make sense per model/input pair:

```bash
uv run foldjax predict --model boltz2 --input jobs/ --resume --keep-going
```

`--resume` skips a run only when its finished `foldjax_run.json` matches the
exact request, input, checkpoint and referenced-local-asset stat/tree
identities, and the persisted structure/representation artifacts. Any relevant
metadata or directory-tree change triggers a conservative rerun; FoldJAX does
not reread large checkpoint or asset payloads merely to write or check a
manifest. Checkpoint trees containing symlinks or special entries are likewise
treated as non-resumable. Legacy or incomplete manifests rerun conservatively;
a manifest's presence alone is not evidence that today's request produced it.
Matching works at **seed** granularity: every seed writes
its own manifest, so a five-seed job that died on the fourth repeats only what
is missing rather than all five. Trunk archives are restored lazily, including
their logical BF16 dtype, without loading a quadratic pair array just to decide
whether it is reusable.

Inputs, checkpoints, and referenced local assets must remain immutable while a
prediction is running. Resume detects changes observed between runs; it does
not lock files against concurrent replacement during backend execution.
`--keep-going` runs the rest of the batch when one run fails and exits 3 if any
did, instead of losing seventeen good predictions to the third one's OOM; what
failed is recorded in `foldjax_failures.json` beside the runs that did not,
because a successful run leaves a manifest behind and a failed one used to
leave nothing at all.

Both are request fields, not CLI-only flags:
`PredictionRequest(resume=True, on_error="continue")`. `foldjax.predict_batch`
returns a `BatchReport` with the results, the reused runs and the failures
together; `foldjax.predict` keeps its original return type.

Two commands answer the questions that used to need a failed run:

```bash
uv run foldjax models --for job.yaml   # which models can run this, and why not
uv run foldjax doctor                  # install, accelerator, weights, templates
uv run foldjax cache gc --older-than 30 --max-size 20G   # reports; --apply deletes
```

`models --for` is answered from the input translation table, so it needs no
weights, no GPU and no network. `cache gc` reports by default and deletes only
with `--apply`: cache entries are pure derived data, but they are still someone's
disk.

### Optional shape padding

JAX reuses a compiled executable only when all dynamic feature axes have the
same shape. FoldJAX therefore offers one opt-in padding switch across all six
models:

```bash
uv run foldjax predict --model boltz2 --input job.yaml --padding
```

With no padding flag, the exact historical model path and shapes are unchanged.
With `--padding`, each backend selects a token bucket and derives a complete
shape profile from it. Atom storage is `24 * tokens`, rounded up to a multiple
of 32. MSA capacity is fixed by the backend's active inference limit rather
than the observed alignment depth; template storage retains the native depth.
OpenDDE structural tokens use `2 * tokens`. ESMFold2 language-model storage
reserves `3 * tokens` to include BOS/EOS for every possible protein chain;
Protenix ESM/ISM uses `min(tokens, provider maximum length)` so its terminal
native sequence length remains supported even when the token bucket is larger.
AlphaFold 3 retains its native token-derived atom and fixed-MSA profile.

With the common padded defaults, OpenDDE uses 1,280 MSA rows and the other
five models use 1,024. AF3's input featurizer and trunk both use 1,024; OpenDDE
selects 1,280 rows before
padding sampled cycles. Deeper alignments use the existing native crop/selection
rules, and shallow inputs receive masked rows. This changes the inference input
relative to a deeper native MSA. Explicit `--max-msa-depth` and `--pad-msa`
overrides retain backend-specific semantics; padding OFF retains native defaults.
MSA capacity stays at the model's fixed default unless explicitly overridden;
`cache warm` compiles only the requested profile, not all MSA sizes.
Protenix padding continues to require its full-depth MSA path. ESMFold2
variants without active MSA selection require an explicit `--pad-msa` for inputs
above the automatic capacity; padding does not silently sample those inputs.

Explicit `--pad-*` targets take precedence. Inputs exceeding a derived capacity
fail rather than being truncated; use a larger token profile or an explicit
axis target. `--padding-overflow exact` only changes token-grid overflow and
does not bypass capacity validation. Padding can increase memory and computation,
particularly for shallow MSAs and language-model inputs. Masks keep padded
entries out of the structure and confidence output, and the run summary and
`foldjax_run.json` report the concrete profile that ran. Matching this profile
is necessary for executable reuse; dtype, static options and runtime must also
match.

For a deployment profile or an exact cache warm, pin only the axes you need;
the remaining model axes still use safe automatic buckets:

```bash
uv run foldjax cache warm \
  --model openfold3 \
  --input job.yaml \
  --pad-tokens 512 \
  --pad-atoms 4096 \
  --pad-msa 128
```

The advanced axes are `--pad-templates`, `--pad-structural-tokens` (OpenDDE),
and `--pad-language-model-tokens` (ESMFold2/ESMC and Protenix ESM/ISM). A
target smaller than the materialized input is an error—padding never truncates
tokens or atoms.
Automatic profiles fail before compilation above their standard grid; use
`--padding-overflow exact` only when an unplanned exact-shape compile is
intentional. Padding can substantially increase quadratic token work, so a
4096-token profile is explicit capacity planning, not a good universal default.

`--profile` is the prediction-side spelling of `foldjax weights ... --profile`.
For Protenix, `mini-esm-v0.5.0` and `mini-ism-v0.5.0` select the matching mini
structure model and the ESM/ISM checkpoint staged beside it, so callers do not
need to know `model_name` or `esm_checkpoint_dir`. The Python API uses the same
field: `PredictionRequest(..., profile="mini-esm-v0.5.0")`.

To choose a model before downloading anything, use `foldjax models --json`.
It reports each canonical name, accepted inputs and execution knobs, whether
its weights are ready, their location and download size when known, and the
next setup step. Each accepted format also has an `input_requirements` entry:
the prediction and preprocessing runtimes, required install extras, whether
preprocessing imports PyTorch, and a short explanation. `foldjax capabilities
--model MODEL` remains the compact capability-only view with the same format
requirements.

### Memory

The peaks in the [benchmark](benchmark.md) are the defaults at
work; nothing below needs to be set to reach them.

- **Row-blocked pair stack.** The pair transition and triangle attention are
  evaluated a block of rows at a time -- mathematically exact, so there is no
  flag and no accuracy trade. Block sizes resolve from the token count.
- **Native-dtype triangle projections**, accumulating in float32 through
  `preferred_element_type`: float32 results at half the operand bytes.
- **Summaries, not logits.** Confidence is reduced in-graph; the full-bin
  PAE/PDE logits (`[samples, N, N, 64]` -- tens of GiB at long sequences) are
  returned only on request.

Past 2,560 tokens Protenix and OpenDDE resolve one chunk width for five knobs
at once, and upstream's value there is 32. Measured at 3,012 tokens, that is
the most expensive of the widths tried: `--option token_q_chunk_size=512` and
its four siblings take 2.1 GiB off the peak and cost no time. The default
stays upstream's, so this is a knob to reach for when the pool is the binding
constraint, not a setting that is wrong.
[Details](engineering-notes.md#the-chunk-width-above-2560-tokens-costs-memory-rather-than-saving-it).

`--mem-fraction` is the one memory flag that is not about the model. It sets
how much of the device JAX preallocates, and FoldJAX defaults it to 0.9 rather
than JAX's own 0.75: one prediction owns the process, and a quarter of the card
held in reserve is what stops jobs that would otherwise fit. Lower it to share
the device with another process.

Design notes: [docs/engineering-notes.md](engineering-notes.md).

### Input and trunk arrays without a structure

`--representations` hands back the trunk's own arrays alongside the prediction:
a comma-separated list, or `all`. `foldjax capabilities --model MODEL` lists
what each model produces. They are the largest arrays a run makes -- a pair
representation is quadratic in token count -- so nothing is written unless you
ask.

`--stop-after trunk` stops once those representations exist, skipping the
diffusion sampler and the confidence heads. It writes no structure, so it only
makes sense together with `--representations`; the default is `full`.

`--stop-after inputs --representations single_inputs` stops before trunk
recycling. All six carried models support this stage. Here `all` selects only
input-stage arrays; requesting a trunk array is an error. The Python
[`get_model` interface](model-interface.md) exposes these stages as `embed`,
`encode` and `predict`.

### A bfloat16 trunk (`--option trunk_dtype=bf16`)

Protenix defaults to `bf16` (its upstream ships bf16-mixed). OpenDDE shipped
`fp32` with its upstream until 2026-08-28 and now defaults to `bf16` too;
`--option dtype=float32` restores upstream's precision -- at 1,531
tokens it is the difference between completing and OOM on both sides. Boltz-2's
equivalent is `compute_dtype`, default `bfloat16`; OpenFold3 has no trunk
dtype: upstream runs `32-true` and a bf16 trunk destroys its prediction.
Details and measurements: [docs/engineering-notes.md](engineering-notes.md).

### A bfloat16 Boltz-2 diffusion (`--option diffusion_compute_dtype=bfloat16`)

Boltz-2's trunk dtype above stops at the trunk. Upstream wraps
`structure_module.sample` in `torch.autocast(enabled=False)`, so this port runs
the whole diffusion score model in float32 whatever `compute_dtype` says, and
that stays the default. Two opt-in experiment knobs open the AlphaFold 3-shaped
cell instead: `--option diffusion_compute_dtype=bfloat16` narrows the score
model's Linear kernels while the residual stream, the sampler's `atom_coords`
and the two coordinate projections at its edges stay float32, and `--option
diffusion_attention_backend={xla,tokamax,triton}` selects the score model's
attention alone, leaving the float32 trunk and confidence Pairformers on the
global `attention_backend`. Neither changes any released default, and both are
part of the compilation-cache identity, so a non-default value never receives
the executable built without it. Both also reach the affinity re-embedding
run, which uses the same score model.

The two go together. Under the released float32 island a fused kernel takes
Tokamax's slow `F32_F32_F32` preset, which is slower than plain XLA; with the
bf16 knob on, q, k, v and the pair bias all reach the kernel in bfloat16 and
Tokamax selects `BF16_BF16_F32` regardless of this port's pinned
`matmul_precision=highest`, which it consults only for float32 operands.
`triton` pins that kernel and fails loudly rather than falling back, so it
requires `diffusion_compute_dtype=bfloat16` and an NVIDIA GPU of compute
capability 8.0 or newer; on any card older than sm80 Tokamax would take
`F32_F32_F32` for bfloat16 operands too. Tokamax ships Mosaic GPU kernels only
up to sm100, so on an sm120 card `tokamax` skips Mosaic and lands on the same
Triton kernel `triton` pins -- the two spellings differ there in whether a
missing kernel is silent. Context parallelism refuses every non-`xla` value,
because the 2-D grid routes pair-bias attention through its own collective and
never reaches the backend switch.

The fused path is not upstream's arithmetic. Tokamax rounds the attention
probabilities to bfloat16 before the P@V contraction, where upstream's
autocast-disabled core keeps that contraction in float32; the `xla` spelling
keeps the float32 score core and is the upstream-faithful shape. Treat the
combination as a measurement, not a recommended default.

### Fused pair-bias attention (`--option attention_kernel=tokamax`, Protenix)

cuEquivariance covers Protenix's triangle attention; the global token
attention and the windowed atom attention of the diffusion encoder and decoder
still form their score tensor in XLA. `tokamax` routes those two through a
fused Triton kernel instead. It is opt-in and the default is unchanged
(`xla_jit` at both sites). The neutral spelling reaches the trunk single
attention; the diffusion sites take the port's own
`--diffusion-attention-backend tokamax`, and the native CLI spells the trunk
one `--trunk-single-attention-backend tokamax`. The kernel is a bfloat16
lever, so it is meant together with `--amp-policy bf16` for the diffusion
sites and a bfloat16 trunk for the single-attention site; float32 inputs are
passed through to tokamax's float32 path and warn once rather than being
upcast or refused. There is no fallback: the implementation is pinned to
Triton (sm120 here), so a device that cannot run it raises instead of quietly
running XLA under the tokamax name. A requested query chunk size does not
reach a kernel that takes the whole query axis and is reported as unused, and
the backend is refused under context parallelism, which it has not been
validated against. OpenDDE reaches the same two sites through Protenix's
primitives but does not offer the value: it has not been measured there.

### Fused gated linear unit (`--option glu_backend=tokamax`, OpenFold3)

Every transition in OpenFold3 is a SwiGLU: two projections widened to four
times the channel count, multiplied together and projected back. XLA writes
both widened tensors out before multiplying them. `tokamax` runs the same
product in one fused Triton kernel, which never materializes them. The saving
is the intermediate, not the precision, which is why the value is offered on a
port that runs float32 throughout. Boltz-2 spells the same option the same
way; this adds it to OpenFold3.

It reaches every SwiGLU the model has: the Pairformer's pair and single
transitions, the MSA module, the template pair stack, the confidence
re-embedding, the diffusion conditioning, and the conditioned transitions of
the token and atom transformers inside the rollout.

The default is `xla` and no released run changes. That default is also
upstream's: `SwiGLU` in `core/model/primitives/activations.py` ships
`use_kernel: bool = False` and `SwiGLUTransition` never passes the flag, so
asking for the fused kernel is a deliberate deviation from the architecture
the released weights were produced under, not a faster spelling of it. The two
arms do not round identically either -- the fused kernel applies the
activation at its own width -- so treat the switch as a numerics change and
read it against the port's own rerun floor.

There is no fallback: the implementation is pinned to Triton, so a card that
cannot run the kernel raises rather than running XLA under the fused name.
Context parallelism refuses the value, because a fused kernel cannot be
partitioned, and an unrecognized value is refused with the two that exist. The
choice is part of the compilation-cache identity, so a fused run never
receives the executable built without it, and spelling the default out names
the same namespace an omitted option does.

### `--option deterministic=on`

Compiles this run's executables for reduction orders that repeat, so two
processes given the same input return the same structure bit for bit instead
of the same structure to within the model's own rerun scatter. Measured on
Protenix it costs 13% of the wall time at 1,003 tokens (9.7% at 3,012 with
Triton GEMMs off); the other ports' costs are measured per port on GPU before
the setting is recommended anywhere. It is off by default, which is what
every measurement in this repository was taken under.

The setting rides on the executable rather than on the process. The equivalent
XLA environment variable is read once at start-up, so it cannot distinguish one
prediction from the next and it reaches every other model sharing a benchmark
process. Being part of the compiled program also makes it part of the
compilation-cache identity: a deterministic run never receives the executable
built without it. One shared module (`foldjax.models._compile_policy`) owns
the two XLA flags for all six ports.

Every port takes it through the shared execution vocabulary
(`--option deterministic=on`); the argv ports (Protenix, OpenDDE) also expose
`--deterministic-ops {off,on}` on their native CLIs. What it covers per port:
Protenix's consolidated graph and, on the ESM/ISM variants, the language-model
encoder; OpenDDE's inference graph and the shape-complementarity executable;
OpenFold3's fused program and all four host-streamed stage executables;
Boltz-2's primary and affinity runners (the bf16 arm keeps its
`xla_allow_excess_precision` policy underneath); ESMFold2's structure program,
the language-model embedding, and the ESMC blocks, which run eagerly under
`off` and through a pool of compiled blocks under `on` (the two policies are
therefore not expected to agree bit for bit with each other); AlphaFold 3's
model runner, where the promise is XLA-emitted operations plus the pinned
tokamax kernel store (the attention kernel has an XLA fallback, the GLU does
not). Eager routes have no executable to put it on, so `--no-graph-jit`,
`--no-compile`, Boltz-2 steering and the like are refused with the option
rather than run without it. Custom-call kernels (cuEquivariance, tokamax,
Pallas) sit outside the flag's reach; their repeatability is observed, not
documented. The option changes rounding routes (Triton GEMMs move to cuBLAS),
so a deterministic run is repeatable but not the run the parity panel was
measured on: on a chaotic 3,012-token Protenix case the deterministic port
landed 5-7 Å from the non-deterministic port and from native, the size of
native's own basin choices. Read parity residuals with the option off.

### `--max-msa-depth`

Selects the model's native MSA depth control. Candidate assembly, profile
statistics and per-cycle subsampling differ by model, so this is not a guarantee
of identical rows or an exact common tensor size. See the
[model-specific semantics](model-interface.md#msa-selection-and-execution-capacity).
The legacy request/CLI padding presets supply a serving depth when omitted;
the new model handle requires an explicit depth when enabling padding.

### `--option msa_deletions={released,restored}` (Boltz-2)

Selects which MSA deletion loop Boltz-2's featurizer runs. `released` is the
default and reproduces the released upstream exactly, regression included:
upstream v2.2.0+ slices each sequence's deletion records out of the previous
sequence's slice rather than the chain's, and because the first MSA row is the
query and carries no deletions, `has_deletion`, `deletion_value` and
`deletion_mean` come out zero for every real alignment. `restored` reinstates
the pre-`04d27c71` loop, which predates the regression and, by release date,
matches the pipeline the published weights were trained with -- an inference
from release dates, not something checked against the training code. Jobs
without an MSA are unaffected either way.

The two modes compile the same executable and differ only in three feature
arrays, but they are separate compile-cache and feature-cache namespaces, so a
`restored` run never answers out of a `released` run's cached features. The
effect on coordinates has not been measured; treat `restored` as an experiment
until it has been. Background and the reproduction:
[docs/boltz2-upstream-msa-deletion-regression-2026-09-10.md](boltz2-upstream-msa-deletion-regression-2026-09-10.md).
The native Boltz-2 predict script spells the same choice `--msa-deletions`.

### `--option diffusion_chunk_size=N`

Denoises the diffusion samples N at a time instead of all of them at once, on
Boltz-2, OpenDDE, OpenFold3 and Protenix. Unlike `--num-samples` it narrows the
sample axis without dropping a prediction, so it is the first thing to try when
a run is close to fitting. AlphaFold 3 and ESMFold2 do not take it: AlphaFold 3
already evaluates more than five samples in groups of five, and ESMFold2 spells
its two equivalents separately -- its confidence head runs one sample at a time
by default, and `structure_sample_sequential` below does the same for its
denoiser.

Whether it helps depends on where the model's peak lives, and that is not the
same place in every model. Measured at 4,100 tokens: OpenFold3's default route
asks for a single 107.85 GiB block and fails, and `diffusion_chunk_size=1`
completes the same five samples at 76.4 GiB; Protenix at that size moved by one
mebibyte, because its peak is the trunk's pair stack, which no sample axis
touches. Chunking is arithmetically the same prediction and not a bitwise one --
the rollout runs on a different array shape per chunk -- which measures 9.5e-06
on coordinates of magnitude 10-30.

The default is unchunked for the released five-sample schedules; the automatic
width engages only above five samples. Per-model measurements:
[docs/engineering-notes.md](engineering-notes.md).

### `--option structure_sample_sequential=true` (ESMFold2)

Denoises ESMFold2's diffusion samples one at a time rather than together, and
narrows the diffusion cache to match. Off by default. It divides the token
transformer's `[samples, tokens, tokens, heads]` float32 attention logits by
the sample count -- 2.7 GiB at 3,012 tokens and five samples, 17.3 GiB at the
released thirty-two -- and little else, because ESMFold2's pair trunk carries
no sample axis at all. Its measured 45 GiB peak at 2,096 residues and its failure at
3,012 are trunk tensors, so this is a control for raising the sample count and
not for reaching a longer input. Each sample draws the noise it would have
drawn batched, from the same key; the result is arithmetically the same
prediction on a narrower array and not a bitwise one. Details and the cost:
[docs/esmfold2.md](esmfold2.md#denoising-the-samples-one-at-a-time).

### Weights and setup

Upstreams publish several formats. `foldjax weights` downloads public files,
checks their pinned size and SHA-256, converts Boltz-2/OpenDDE/Protenix once,
stages ESMFold2 and OpenFold3's publisher-native checkpoints, and locates
manually supplied AlphaFold 3 parameters. Managed checkpoint loading remains
PyTorch-free.

```bash
uv run foldjax setup                            # default public models, in one go
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
uv run foldjax home --path runtime              # one script-friendly location
```

`--download-only` fetches the released files and skips the JAX conversions, on
both `setup` and `weights fetch`: useful on a machine that has the bandwidth
but not the accelerator to convert on. `--weights PATH` on `predict` and `plan`
points at a model-native checkpoint or asset directory directly, bypassing the
store; omit it and the store resolves the file.

Weight preparation keeps its lifecycle and progress on stderr. `weights fetch`
retains a human-readable summary on stdout; use `weights path` when a script
needs path-only stdout. A fresh conversion shows `resolve`, per-file `download`,
`convert` (or `stage`/`direct`), `validate`, and `ready`, with elapsed time and
sizes; a repeat run explicitly reports that the verified native bundle was
skipped. Library callers can receive the same structured `AssetEvent` objects
through `foldjax.assets.fetch(..., on_event=callback)` instead of parsing text.

`setup` fetches and converts every model whose weights are published — Boltz-2,
OpenDDE, OpenFold3, Protenix — plus shared CCD assets. Protenix has a second
published base checkpoint behind `--profile base-20250630`: the same 368M
architecture trained to a 2025-06-30 wwPDB cutoff instead of the release's
2021-09-30. Upstream recommends it for practical use and keeps the release for
benchmarks, since comparing against AlphaFold 3 needs AlphaFold 3's cutoff, and
FoldJAX follows that split.

**Protenix ships two supported models and `setup` fetches both.**
`--profile v2` (or `model_name=protenix-v2`) selects Protenix v2 — 464M
parameters against the release's 368M, announced 2026-04-08, with clear gains
on antibody-antigen targets. The port needs no architecture change for it: its
blocks read their widths from the parameters they are handed, so v2's c_z=256
and its eight triangle heads arrive with the checkpoint.

Its provenance is worth stating. ByteDance's own CDN key stopped serving
`protenix-v2.pt` the day it was announced and upstream says accessibility is
under internal review
([bytedance/Protenix#295](https://github.com/bytedance/Protenix/issues/295),
still open), so FoldJAX fetches a mirror of that object and pins its SHA-256.
The archive was checked before the hash was recorded: 464,442,431 parameters,
matching the 464.44 M upstream documents, with Pairformer weights of
`tri_mul_out.linear_a_p (256, 256)` and `tri_att_start.linear (8, 256)` —
exactly what `hidden_scale_up` produces at c_z=256. A substituted file fails
on the hash before it is loaded.

**FoldJAX runs v2 at 3,012 tokens; upstream refuses at 2,561.** That limit is
a memory budget, not an architectural bound — upstream's own assertion says
"It might cause OOM", and the relative position encoding buckets residue
separation with `clip(delta, 0, 2 * r_max)` at r_max=32, so nothing in the
model is bounded by token count. A budget calibrated on one implementation
does not transfer to another that uses less, so FoldJAX warns with the
expected size and runs. Measured: 3,012 tokens, five completed samples,
**78.2 GiB**. Pass `--option strict_token_limit=true` for upstream's refusal
instead. Neither implementation validates the model at that length, and the
warning says so.

Capacity is not free, and the arithmetic is simple enough to plan with. v2
doubles the pair channel, and the trunk's dominant tensors are pair tensors of
shape `[tokens, tokens, c_z]`, so its cost approaches twice the release's as
the square term takes over:

| tokens | release | v2 | ratio |
|---|---|---|---|
| 499 | 4.6 GiB | 6.5 | 1.40x |
| 1,003 | 6.7 | 10.1 | 1.50x |
| 2,096 | 23.2 | 39.0 | 1.68x |
| 3,012 | 43.7 | 78.2 | 1.79x |

None of that is overhead this port could remove — the same doubling costs
upstream 1.08x to 1.32x *more* than it costs FoldJAX at each size it will run.
It is what the extra capacity weighs. Above about a thousand tokens the peak
tracks `2.1 GiB + 0.0086 MiB × tokens²` closely enough to size a card before
starting, which is what the warning prints. OpenFold3's 2.29 GB v0.5.0 OpenBind
checkpoint comes from the publisher's own S3 bucket, pinned by SHA-256, so it
needs no account and no request. It is the sole managed profile; legacy p1 and
p2 files are rejected. The exact identity is recorded in the
[model version ledger](model-versions.md#openfold3).

Two models are outside that, for two different reasons. ESMFold2 is public but
held back from the default because its full structure+ESMC+chemistry bundle is
about 26.8 GB — `foldjax setup --all` takes it with the rest, `foldjax weights
fetch --model esmfold2` takes it alone, and `--profile structure-only` takes
the 1.36 GB structure+chemistry profile instead. **AlphaFold 3 is the one set of
weights you have to supply yourself**, because DeepMind releases its parameters
only to applicants who accept their terms and they may not be redistributed; no
flag changes that, and `foldjax setup` says which directory `af3.bin` goes in.
Once it is there nothing else is asked of you: the ABI-specific extension and
CCD tables build themselves on first use. MSA search needs nothing installed
(ColabFold MMseqs2 server).

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or a .foldjax/ in the checkout,
├── downloads/<model>/     #   or $XDG_CACHE_HOME/foldjax — `foldjax home` says
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # prediction-ready weights/assets
├── compile/               # XLA persistent compilation cache
└── runtime/<model>/       # generated native binaries/data (AlphaFold 3)
```

`mkdir .foldjax` in a checkout keeps all of it beside the source. Where a `.pt`
archive needs conversion or direct loading, FoldJAX's restricted reader returns
NumPy arrays without importing PyTorch. No weights are redistributed — each
file comes from its own publisher under that project's terms, and nothing is
fetched implicitly during prediction.

`runtime/` is the one directory that grows on its own. An AlphaFold 3 tree is
keyed by the vendored source and the interpreter ABI, so editing a vendored
file or moving Python mints a new one — about a gigabyte each, nearly all of it
the chemical component dictionary rather than code. `foldjax runtime gc --model
alphafold3` reports old candidates while always keeping the tree selected by
the current source. It is a dry run by default: the store is shared between
checkouts, and age cannot prove that another checkout has stopped selecting its
own generation. Review the list and add `--apply` to remove it; by default,
anything prepared in the last week is not even listed -- `--keep-days N` moves
that window. `--all` includes those recent candidates but still requires
`--apply` to delete them.

### GPU and the compile cache

These graphs can take minutes to compile and seconds to execute after a cache
hit, so the persistent XLA cache is on by default and namespaced per model,
weight identity, runtime profile, and the options that actually change the
compiled program. Different models and weight sets never share a FoldJAX cache
namespace, and options that only affect output formatting never fragment one.
An eligible, readable entry may be reused across processes; a missing, invalid,
or runtime-rejected entry recompiles normally.

Warm the exact request before a production run on its deployment GPU:

```bash
uv run foldjax cache warm \
  --model opendde \
  --input job.yaml \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10
```

This deliberately uses an `execute_once` strategy: FoldJAX runs the first
requested seed through the normal prediction path, including GPU kernel
autotuning, then discards its prediction files. Pass `--output-dir` if that
warm-up prediction should be kept. The command reports the exact cache
namespace, elapsed time, device allocator peak, and files/bytes before and
after. Persistent entries are accelerator-, JAX-runtime-, weight-, input-shape-,
profile-, and static-option-specific, so warm representative production shapes
with the same options on the machine that will run them. One small input cannot
prewarm arbitrary larger shapes. A padded warm reports the one concrete full
shape profile it selected; it does not claim that every bucket or every
token/atom/MSA combination was compiled.

A zero cache delta is reported neutrally as `no_new_entry_observed`: directory
contents alone cannot prove an XLA cache hit, and FoldJAX does not pretend that
they can.

The cache setting is scoped to one FoldJAX request even though JAX exposes it
as process-wide configuration: FoldJAX serializes its own predictions, applies
the resolved directory while the backend runs, and restores the embedding
application's previous setting afterwards, including on failure. An embedding
application must not compile unrelated JAX programs concurrently in another
thread; use a separate process for mixed workloads because JAX has no
thread-local cache configuration.
`--no-cache` therefore disables reads and writes for that request rather than
merely omitting a directory argument; `use_compile_cache=False` is the Python
spelling. It cannot be combined with `--cache-dir`/`cache_dir`, because one asks
to disable the cache while the other asks where to use it. Every model also
runs on CPU, slowly.
