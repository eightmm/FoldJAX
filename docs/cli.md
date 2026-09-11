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
equivalent is `compute_dtype`, default `bfloat16`.
Details and measurements: [docs/engineering-notes.md](engineering-notes.md).

### The bfloat16 OpenDDE confidence head (`--confidence-dtype`)

OpenDDE's trunk dtype above stops at the trunk: `cast_trunk_params` narrows the
embedder and both trunks and leaves the diffusion module, the distogram and the
confidence head in float32. This flag narrows the confidence head alone, and
only the part of it AlphaFold 3 narrows. It defaults to `bf16` since
2026-09-11; `--confidence-dtype fp32` keeps the whole head wide.

AF3's released confidence head is the boundary. With `global_config.bfloat16 ==
'all'`, its default, the pair and single activations are cast to bfloat16 on
entry and the whole re-embedding Pairformer runs narrow; the pair
representation is widened again before the distance-error logits and the single
representation before the pLDDT logits. This flag reproduces that: narrow
re-embedding stack, float32 pLDDT, PAE, PDE and resolved logits. It is not a
blanket cast of the subtree.

It is the safest dtype change in the repo, and the reason is structural. The
confidence head runs after the sampler, reads its finished coordinates, and
emits scores; nothing downstream of it is a coordinate, so narrowing it cannot
move a structure. Protenix measured the same boundary at 3,012 tokens:
coordinates bitwise unchanged, atom pLDDT moved at most 0.0099, chain pTM and
ipTM at most 1.9e-4, PAE means at most 0.005.

Three things stay wide, each for its own reason. The four output projections
stay float32 because their logits feed softmaxes, which is the one rounding
this port has measured as harmful elsewhere. The distance bins stay float32
because they are compared against float32 distances, never multiplied by them.
The two distance projections and the outer-sum initialiser `linear_s1`/
`linear_s2` narrow their own operands instead of inheriting a dtype, because
what reaches them is float32 geometry rather than a trunk representation.

`bf16` is the default because the argument above has no counterweight: the
head cannot move a structure, the same shared code was measured at 3,012
tokens on Protenix, and the narrow arm is the shape AlphaFold 3 ships. What
does not yet exist is an OpenDDE-specific GPU row for it, so `fp32` remains a
supported pin rather than a fallback. The value joins the compilation-cache
identity, so a run that narrows the head never receives the executable built
without it, and an unrecognised width is refused naming `fp32` and `bf16`
rather than falling back. It is independent of `--trunk-dtype`: OpenDDE
widens every trunk output to float32 before its heads, so this flag casts the
head's activations itself and `--trunk-dtype fp32
--confidence-dtype bf16` is a real combination. Protenix's nearest equivalent,
`--amp-policy`, works the other way -- it reproduces a torch autocast context,
so its confidence stage inherits the trunk's dtype and a float32 trunk leaves
it nothing to narrow. Context parallelism is supported: the narrowing is three
casts and a parameter tree, with no kernel or collective of its own, and the
released bfloat16 trunk already runs the same Pairformer code under a mesh.

### A bfloat16 Protenix confidence head (`--amp-policy`, on by default)

Protenix's trunk is bfloat16 by default and two stages beside it are resolved
separately, because upstream resolves them separately: it rewrites the policy
from the token count before the model is built
(`runner/inference.py:492 update_inference_configs`). `--amp-policy` is where
the port spells that resolution, and it takes four values.

| value | confidence head | diffusion sampler |
| --- | --- | --- |
| `auto` (default) | bfloat16 at every size | bfloat16 above 3,840 tokens |
| `upstream` | bfloat16 above 2,560 tokens | bfloat16 above 3,840 tokens |
| `fp32` | float32 at every size | float32 at every size |
| `bf16` | bfloat16 at every size | bfloat16 at every size |

**`auto` is not upstream's table below 2,560 tokens, deliberately.** Upstream
keeps the confidence head float32 there; this port narrows it. Upstream's
2,560 is an OOM heuristic for the configuration it ships, not a measured
accuracy boundary, and the port does not inherit it as one. `--amp-policy
upstream` reproduces the native gate at every size and is the spelling a
parity run against a native capture wants; it is a separate cache namespace,
so it never receives the executable built for the default.

The diffusion half of the gate did not move. That stage owns the coordinates,
so every job at or below 3,840 tokens runs the sampler on exactly the
arithmetic it ran on before this default existed -- one process running both
policies on the same inputs produces bitwise-identical coordinates, which the
test suite pins.

What narrows is the head's re-embedding: `input_strunk_ln`, the Pairformer
blocks, the two distance projections and the outer-sum initialiser
`linear_s1`/`linear_s2`. The four output projections and the distance bins
stay float32 under every value, because upstream runs that block inside
`autocast(enabled=False)` at 76 tokens and at 3,012 alike. `s_inputs` reaches
the head float32 whatever the policy -- the input embedder concatenates raw
reference features that autocast does not narrow -- which is why
`linear_s1`/`linear_s2` are built to narrow their own operands rather than
inherit a dtype; a merely narrowed weight would promote that matmul back to
float32 and carry the pair tensor, and every block after it, along.

The measured evidence sits either side of the change. At 3,012 tokens, where
upstream already narrows the head, it moved atom pLDDT by at most 0.0099,
chain pTM and ipTM by at most 1.9e-4 and PAE means by at most 0.005 with the
coordinates bitwise unchanged. At 2,096 tokens -- below upstream's
threshold -- `--amp-policy bf16`, which narrows the head *and* the sampler and
so does strictly more than `auto`, lands 0.39-0.45 Å per chain from the
deposited structure for 11.9% less wall time and 7.7% less peak memory. Those
savings are the two-stage arm's; `auto` narrows one stage and takes the
smaller share. **There is no GPU row for the confidence-only change below
2,560 tokens yet.**

The policy is realised only under a bfloat16 trunk. Upstream's `skip_amp`
flags choose whether a stage *leaves* the ambient autocast context, and under
`--trunk-dtype fp32` there is no context to leave, so both stages run float32
whatever the policy says and `--amp-policy bf16` is not a way around it. This
is the opposite of OpenDDE's `--confidence-dtype`, which casts the head's
activations itself and combines with a float32 trunk.

### A partial bfloat16 OpenFold3 (`--option dtype=bfloat16`)

OpenFold3 defaults to `float32`, which is upstream's inference precision
(`openfold3/entry_points/validator.py:127`). `--option dtype=bfloat16` is
opt-in, and **whether it is a good idea depends on the size of your target.**
Measured on GPU 2026-09-11, each arm against a same-source control:

| tokens | wall f32 -> bf16 | peak f32 -> bf16 | structure vs the f32 arm |
| --- | --- | --- | --- |
| 1,003 | 98.65 -> 75.70 s (-23.3%) | 9,274 -> 5,553 MiB (-40.1%) | — |
| 2,096 | 403.29 -> 262.33 s (-35.0%) | 25,067 -> 19,107 MiB (-23.8%) | identical to 0.01 A |
| 3,012 | 950.84 -> 664.61 s (-30.1%) | 50,412 -> 34,893 MiB (-30.8%) | **4.65-5.80 A drift** |

The speed and the memory hold at every size. The structure does not. At 2,096
tokens on 5DEI, four chains by five samples, per-chain deposited RMSD is
0.45-0.51 A on both arms chain for chain, TM 0.996-0.997 on both, and sample 3
picks the same alternative basin on both. At 3,012 tokens on 6ZTX, same-index
RMSD against the float32 arm is 4.65 / 4.68 / 5.64 / 5.80 / 5.26 A against a
float32 within-set spread of 0.619 A -- 8.5x the control's own spread -- and
the bfloat16 arm's internal spread is inflated 2.8x to 1.729 A.

**So: take it at or below roughly 2,000 tokens, and do not take it above
that.** The 3k failure is uniform drift rather than a lost region -- on 6ZTX
sample 0 gives A 4.50 / B 4.63 / C 4.51 / D 4.54 and sample 3 gives 5.63 /
5.65 / 5.64 / 5.63, with complex TM holding at 0.978-0.980, so all four chains
move together and the fold survives. That is the signature of accumulation
over 48 Pairformer blocks times 10 cycles, not of one region breaking.

"Partial" is the other load-bearing word. A whole-trunk bfloat16 cast, input
embedder included, destroys the prediction outright -- pLDDT 0.858 to 0.466,
with the error already the size of `s_input` before a Pairformer block runs --
so the option narrows a named set of subtrees and no others.

What the option narrows is the token/pair representation track; everything
atom- or coordinate-shaped stays float32. The split is
`inference.cast_narrow_params` (`models/openfold3/inference.py:281`), and it
is a list of named subtrees rather than a rule, so a new parameter group is
wide until someone classifies it:

**bfloat16** -- `trunk.msa_module_embedder`, `trunk.msa_module`,
`trunk.pairformer_stack`, `trunk.layer_norm_z`/`linear_z`,
`trunk.layer_norm_s`/`linear_s`, `trunk.template_embedder`; the diffusion
conditioning's *pair* branch (`layer_norm_z`, `linear_z`, `transition_z`);
and `pairformer_embedding.pairformer_stack`.

**float32** -- `trunk.input_embedder` (this is the island that collapsed the
prediction when it was narrowed, and upstream pins it too at
`feature_embedders/input_embedders.py:129-131`); the entire `denoiser`, atom
encoder, atom decoder and 24-block token diffusion transformer alike; the
diffusion conditioning's *single* branch including its Fourier noise
embedding; the confidence head's geometry re-embedding
(`pairformer_embedding.linear_i`/`linear_j`/`linear_distance`); and every
output head. So every output logit -- distogram, PAE, PDE, pLDDT,
experimentally-resolved -- is float32 in both profiles.

The attention softmaxes *inside* the narrowed regions do run bfloat16, with a
bfloat16 pair bias. That is deliberate and it is what both upstreams do:
OpenFold3's `softmax_no_cast` disables autocast specifically so bfloat16
softmax stays bfloat16, and AlphaFold 3's evoformer runs the same way. It is
also the shape that cost Protenix a chain when it was applied to the *diffusion
token transformer's* pair bias -- which is why that transformer is float32
here. The trunk's bfloat16 softmax is the region the 2026-08-10 island already
measured at pLDDT -0.001; **the confidence Pairformer's is not measured on this
port at all**, and it is the first thing a GPU run should look at. The nearest
evidence is Protenix, where the same change at 3,012 tokens left coordinates
bitwise unchanged and moved atom pLDDT by at most 0.0099, chain pTM and ipTM by
at most 1.9e-4, and PAE means by at most 0.005. That is a sibling's number, not
this port's.

That shape is AlphaFold 3's released inference shape, read out of its source
rather than borrowed by analogy: AF3's bfloat16 context narrows only what
enters it already narrow, and its atom cross-attention is driven by float32
reference positions. Its two float32 islands are also the two that upstream
OpenFold3 pins itself while training this checkpoint under `bf16-mixed`, which
all four released training configs do: the input embedder's atom encoder
(`feature_embedders/input_embedders.py:129-131`), which this port keeps wide
too, and the confidence head's Pairformer stack
(`heads/prediction_heads.py:224`, whose `pairformer_dtype` defaults to
`torch.float32`), which `--option dtype=bfloat16` narrows here unless
`confidence_dtype=float32` says otherwise.

### Separating OpenFold3's confidence head (`--option confidence_dtype=...`)

The confidence head has its own knob, and it is **not** a second thing to
turn on: `confidence_dtype` follows `dtype` when unset, so `--option
dtype=bfloat16` already narrows the head, and narrowing the head alone
measures as nothing (+0.4% wall / -0.1% peak at 1,003 tokens, -1.1% / -0.0%
at 2,096). Under a narrowed trunk it is fully subsumed: `dtype=bfloat16` and
`dtype=bfloat16 confidence_dtype=bfloat16` reported byte-identical peaks
(5,553 MiB at 1,003 tokens, 19,107 MiB at 2,096) and walls within 0.3%. What
the knob buys is separating the two regions after the fact --

* `--option confidence_dtype=float32` keeps the head wide against the
  narrowed trunk, which is upstream's shape for that region;
* `--option dtype=float32 --option confidence_dtype=bfloat16` does the
  reverse.

The region is separable because it consumes predicted coordinates and emits
scores, never coordinates, so narrowing it cannot move a structure -- which
is also why the 3,012-token drift above is a property of `dtype` and not of
this knob. Upstream
pins it wide with a dedicated `pairformer_dtype` defaulting to
`torch.float32` (`openfold3/core/model/heads/prediction_heads.py:192`, used at
`:224`, restoring the incoming dtype at `:241`) on top of running 32-true
overall, so `confidence_dtype=float32` is the closest single option to
upstream's arrangement of this head. Boltz-2's `diffusion_compute_dtype` is
the same idea of a region-scoped knob.

The nearest direct evidence for narrowing this region is a sibling's:
Protenix at 3,012 tokens left coordinates bitwise unchanged and moved atom
pLDDT by at most 0.0099, chain pTM/ipTM by at most 1.9e-4 and PAE means by at
most 0.005. Both knobs are part of the compilation-cache identity, so a
narrowed run never receives the float32 executable, an explicit value equal to
the one the request would have resolved to names the same namespace rather
than forking a second, and the option is accepted under context parallelism.

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

The two go together. Under the float32 island a fused kernel inherits this
port's `matmul_precision` -- Tokamax resolves an unset precision from
`jax_default_matmul_precision` -- so before 2026-09-11 it took the three-pass
`F32_F32_F32`, which was measured slower than plain XLA; since the default
moved to `high` it takes `TF32_TF32_F32` there instead, and that comparison
has not been re-measured. `--option matmul_precision=highest` restores the
arm the recorded figure describes. With the bf16 knob on, q, k, v and the pair
bias all reach the kernel in bfloat16 and Tokamax selects `BF16_BF16_F32`
whatever `matmul_precision` says, because it consults it only for float32
operands.
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

### A bfloat16 Boltz-2 pair residual (`--option pair_residual_dtype`, on by default)

`compute_dtype=bfloat16` narrows every trunk GEMM. It does not decide what the
pair representation is *stored* in between them, and above roughly 2,000
tokens that stored pair tensor -- `[1, N, N, 128]` -- is the largest tenant of
the peak. At 3,012 tokens one copy of it is 4.326 GiB in float32 and 2.163 GiB
in bfloat16; at 2,096 tokens, 2.095 and 1.047 GiB. FoldJAX stores it in
bfloat16, which is what the released bfloat16 trunk now runs.

Measured on one RTX PRO 6000 Blackwell, warm after prefill, same input and
schedule per pair, each arm against a control built from the same source:

| tokens | wall (float32 -> bfloat16) | peak (float32 -> bfloat16) |
| --- | --- | --- |
| 2,096 | 313.94 -> 273.77 s (-12.8%) | 21,778 -> 18,511 MiB (-15.0%) |
| 3,012 | 781.45 -> 717.50 s (-8.2%) | 40,748 -> 29,416 MiB (-27.8%) |

This is the only lever measured on this port that moves the peak at all. The
precision pin's peak contribution is zero at every size, and the fused GLU's
-26.9% exists only at 1,003 tokens. This one narrows the pair arena itself --
the tenant that dominates the peak above roughly 760 tokens -- so its saving
*grows* with size instead of shrinking away, which is why it is a default and
they are not.

The two levers do not add up. With the bfloat16 pair residual and the
`matmul_precision` default both applied, the pair is wall -16.1% and peak
-15.0% at 2,096 tokens, and wall -10.6% and peak -27.8% at 3,012. The peak is
entirely this option's: the precision pin contributes none of it at any size.
The wall almost is not. At 2,096 the two are -12.8% here and -4.5% there,
which come to -17.3% added and -16.7% composed as if independent, against the
-16.1% measured together -- a small overlap and no more. Read the measured
row, not the arithmetic.

The two also point opposite ways as the input grows. The precision pin shrinks
with length (-7.2% wall at 1,003 tokens, -4.5% at 2,096); this option grows
(-15.0% peak at 2,096, -27.8% at 3,012).

Accuracy at 2,096 tokens on 5DEI, a homotetramer, five samples, per-chain
RMSD to the deposited chain: 0.34-0.40 A on both arms, TM 0.998 on both, and
19 of 20 chain-sample cells agree to two decimals. The one that does not is
sample 4 selecting a different basin on one chain, inside this case's own
0.238 A within-set spread.

#### Asking for the other arm

`--option pair_residual_dtype=float32` keeps the wide stream. It emits no cast
at all, so it is the same lowered program, byte for byte, that the default was
before this change -- unchanged by construction rather than by a
numerically-equal cast.

Three spellings are accepted and a fourth is refused:

| spelling | meaning |
| --- | --- |
| omitted, or `auto` | follow the trunk: bfloat16 on the released trunk, float32 on `compute_dtype=float32` |
| `bfloat16` | narrow outright; requires `compute_dtype=bfloat16` |
| `float32` | wide outright, legal on either trunk |
| `null` | **refused** |

`float32` used to be refused, on the reasoning that omitting the option was
already its spelling and a second spelling would leave a provenance record
unable to say which arm ran. Flipping the default inverts that: omission now
spells bfloat16, so the wide arm needs a name of its own and gets one. Null is
refused for the mirror-image reason. It spelled the float32 stream while that
was the default, and in a signature where every other `null` means "inherit"
it would now read as "the default", which is the opposite arm. A spelling
whose meaning inverted is worth an error, not a silent reinterpretation; the
error names both replacements.

The width is part of the compilation-cache identity, and the identity records
the width the trunk *realised*, never the spelling that asked for it -- so
omitting the option, `auto` and `bfloat16` are one namespace and one retained
runner on the released trunk. It is recorded even when it is float32, because
runs recorded before this change omit the key whichever arm they ran, and
absence has to keep meaning "recorded before the width was recorded" rather
than quietly becoming a third name for one of the two arms. Boltz-2 cache
directories from before this change are therefore not reused.

#### What it does not change

Which program each block runs. Every block is told that a narrowed residual is
still the autocast configuration, instead of inferring the precision policy
from the activation width: without that, triangle multiplication reads a
bfloat16 pair as "not autocast" and runs its contraction in float32, an arm
that fires and measures a *wider* program than either arm intends. Told, it
keeps the bfloat16 contraction on both the released cuEquivariance backend and
the XLA one, and triangle attention keeps its float32 query scale.

Neither pair bias that enters a softmax picks up a rounding it did not have.
The single track's is float32 on both arms -- its projection is inside the
autocast-disabled parameter island and the layer hands it an explicitly
widened pair -- and triangle attention's is bfloat16 on both, because that
projection's kernel is bfloat16 in the released run already. This is the rule
that cost a whole chain on Protenix (a rounded pair bias moved a chain 16 A),
so it is asserted on the *omitted* arm, not only on the explicit one.

The single track itself does not move: upstream runs it inside
`torch.autocast(enabled=False)` (`boltz/model/layers/pairformer.py:105`) and at
3,012 tokens it is 4.4 MiB against the pair tensor's 4,430 MiB. The trunk also
hands its pair representation back in float32, so the diffusion conditioner,
the confidence module and the affinity head see exactly the dtype they always
saw, and the template path is unaffected: its `a_proj` is in the float32
exemption list, so the per-template pair stays float32 whatever the trunk
carry is.

#### Two backends that now diverge further

Three normalisations do change width, all of them because upstream's own code
is written to follow the input dtype at exactly those points. Triangle
attention's entry LayerNorm takes the bfloat16-input exception upstream writes
for itself (`boltz/model/layers/triangular_attention/primitives.py:139-147`)
and returns bfloat16. cuEquivariance's fused input norm does the same by
construction -- measured here, `layer_norm_transpose` returns the dtype it is
given -- so on the released `triangle_backend=cueq` the pair normalisation
inside triangle multiplication runs in bfloat16. The plain XLA triangle
multiplication does *not* follow suit: it uses `nn.LayerNorm` semantics, which
autocast excludes, so it promotes to float32 whatever it is handed.

The two triangle backends therefore compute different things at that point,
and they now do so **by default** rather than only under an opt-in. If you are
comparing `triangle_backend=cueq` against `triangle_backend=xla`, that
normalisation is one of the differences you are measuring, and
`--option pair_residual_dtype=float32` removes it from both.

#### Upstream, and the way this could have lost

Upstream stores this residual in float32, for a reason of its own: eval-mode
`get_dropout_mask` returns a float32 tensor
(`boltz/model/layers/dropout.py:41-43`) and multiplies all four Pairformer
pair updates (`boltz/model/layers/pairformer.py:77,82,87,95`), so torch
promotes every residual sum. In this port the float32 came from two unrelated
places: `z_init` picks it up from ContactConditioning's
`encoding_unspecified`, an `nn.Parameter` rather than a Linear kernel, so the
bfloat16 cast skipped it; and above 384 tokens OuterProductMean re-promoted on
every MSA layer. Storing it narrow is a deviation from upstream's AMP
placement, in the one port where AMP placement has already been shown to move
coordinates -- upstream's own kernels-off toggle moves 5SAK by 2.91 A. That is
why it needed the 5DEI row above before it became a default, and it is why
`float32` stays exactly reachable. AlphaFold 3 stores its trunk pair
activations in bfloat16 too.

The named way this could have lost, and did not: nothing the program receives
or returns changes width, so every byte it saves is a temp, and temps get
repacked. Against that, one widening that is free in float32 becomes real in
bfloat16 -- PairWeightedAveraging widens the pair tensor at `msa.py:464` and
normalises it through a pinned CUDA kernel (`amp_layer_norm`), and a custom
call's operand cannot be fused, so each MSA layer was expected to materialise
a float32 pair copy that the wide arm gets for nothing. That could have made the knob a net loss in the MSA stack
even while it paid in the Pairformer stack. It did not bite at either size
measured. The mechanism is written down because it is the reason this could
have failed, and the next port trying the same thing has to check the same
site.

#### The CPU parity replay runs the other arm

`tests/parity/test_boltz2.py` pins `pair_residual_dtype="float32"`. That suite
asks one question -- does the port round where upstream rounds -- and its
tolerances are calibrated against the single policy its captures were taken
under, which is upstream's float32 residual. Inheriting the released default
there is not a stricter test, it is a different one: tier A's `z` lands at
relative RMSE 1.7405e-02 against a 5.0e-03 tolerance, 3.5x over, next to a
2.7360e-03 calibration. Pinned, the replay lowers to the program those
residuals were calibrated on, byte for byte.

**That pin routes around a hole in the tier-A tripwire; it does not close
it.** The tripwire reads `cyclic_pos_enc` and `fix_sym_check` and nothing
else, so a trunk default that changes the output -- as this one does --
passes it silently and surfaces only as a residual over tolerance, if the
tolerance happens to be tight enough to catch it. The next trunk default to
move will meet the same gap. Whether the released default is close enough to
upstream is a GPU-panel question, answered there (5DEI above), not by
widening a tolerance here.

Still unverified: context parallelism (`cp_devices>1`) inherits the default
and was measured only on one card, and the pinned `(1, 437, 437, 128)` norm
shape inside the cuEquivariance branch has not been exercised narrow.


### Boltz-2's matmul precision (`--option matmul_precision=highest`)

Boltz-2 ships TF32 float32 matmuls since 2026-09-11. It is the one model here
whose upstream asks for true float32 -- `main.py:1096` is
`torch.set_float32_matmul_precision("highest")`, where OpenFold3, Protenix and
OpenDDE all select TF32 -- and this port followed it until that date, on the
reasoning that matching upstream's configuration is what faithfulness means.

It is a deliberate departure, not an oversight. The criterion for a default
here is accuracy equivalence, and TF32 meets it. Measured on GPU over the
released schedule, warm after prefill, one RTX PRO 6000 Blackwell:

| tokens | wall, `highest` | wall, `high` | peak |
|---|---|---|---|
| 1,003 | 88.31 s | 82.17 s (**-7.0%**) | 9,216 MiB, both arms |
| 2,096 | ~313.94 s | 287.62 s (**~-8.4%**) | 21,778 MiB, both arms |

**This buys wall clock and no memory at all**, and the peak column is worth a
paragraph because an earlier reading of these rows got it wrong. That reading
put 1,003 tokens at 90.98 -> 82.17 s with peak 12,612 -> 9,216 MiB, a -26.9%
memory saving, and attributed it here. The 90.98 / 12,612 baseline was source
`72116ac3`, which predates the fused-GLU default: the whole 12,612 -> 9,216
belongs to `glu_backend=tokamax` and is recorded under that option, along
with the 317.91 -> 313.94 s the 2,096-token baseline moved at the same time.
Against same-source controls the precision change moves no bytes at any size.
The one control actually run beside this change is the 1,003-token 88.31 s;
the 2,096-token figure is the GLU note's post-flip number rather than a
control run beside 287.62, so read that row as approximate.

Accuracy at 2,096 tokens on 5DEI, a homotetramer, five samples: per-chain RMSD
to the deposited chain is 0.34-0.40 Å on both arms, chain for chain, TM 0.998
on both, and sample 4 selects the same alternative basin in both arms. The
same-index residual between the arms is 0.038 Å median / 0.135 Å max against a
0.238 Å spread *within* either set -- the arms are closer to each other than
five samples of one arm are to each other.

`--option matmul_precision=highest` selects exactly the program this port
shipped before. It stays reachable because the parity harnesses compare
against upstream's rounding and need it; they all pin it explicitly, so none
of them moved. The two values are separate compilation-cache namespaces, and
spelling `high` explicitly selects the same namespace as omitting it.

#### The second precision surface, which this default does not reach

Triangle attention takes a `matmul_precision` *string*
(`models/predict.py:284`, `:425`), turns it into a `jax.lax.Precision`, and
passes it explicitly to its four projections -- q/g, k/v, the triangle bias
and the output linear. An explicit `precision=` beats
`jax.default_matmul_precision`, so those four dots do not follow the option
above. `api.predict` sets no such string, so they sit at their signature
default, `highest`. Inside one triangle-attention layer the two surfaces are
visible side by side: the four projections stay `HIGHEST` while the score and
P@V contractions beside them, which carry no `precision=`, follow the option.

That asymmetry is not new and the flip did not create it; it is the state the
measurement above was taken in, which is why the shipped default is
bit-for-bit that arm.

**On the released `dtype=bfloat16` it is inert, and the reason is not
obvious** -- the pin is still spelled `HIGHEST` on those matmuls, so the
natural assumption is that it is doing something. Two narrowings meet there:

* `_cast_trunk_params` (`models/trunk_blocks/trunk.py:246`) narrows every
  `*/kernel` in the trunk except four subtrees --
  `input_embedder/atom_encoder`, `template_module/a_proj`,
  `input_embedder/atom_attention_encoder/atom_to_token_trans`, and each
  Pairformer layer's `pre_norm_s`/`attention`/`transition_s`. Triangle
  attention's `tri_att_start`/`tri_att_end` are in none of them, so their
  kernels are bfloat16. (The exempt `attention` key is the *single*
  representation's `AttentionPairBias`, not this one.)
* `triangle_attention._linear` then casts the activation to the kernel's
  width before the matmul, so the float32 pair residual never meets a float32
  kernel there either.

Both operands are therefore bfloat16, and the attribute has no float32
accumulation to choose between. `cuequivariance_ops_jax` agrees
independently: its `use_tf32` returns False for any non-float32 dtype
(`_triangle_attention.py:51`) before it reads the precision at all.

#### What unifying the two surfaces would be worth, and what it would cost

Unifying them was built and measured: at 1,003 tokens the unified default is
81.96 s against the hybrid's 82.17 s and at 2,096 it is 287.71 s against
287.62 s -- the same to within noise at both sizes, which is what the
inertness above predicts. So there is no performance case for it at the
shipped dtype, and it is not shipped: a unified default would be a program no
accuracy row describes.

Under `--option dtype=float32` the pin is live and worth something. From the
released checkpoint the trunk Pairformer is 64 layers at `c_z=128` with 4
heads of 32, plus 4 more inside the MSA module and an 8-layer stack in the
confidence head. Per Pairformer layer, in MACs:

* the pinned projections, 164,864 × N²
* scope-governed N² in the same layer (`tri_mul` p/g in and out,
  `transition_z`), 393,216 × N²
* scope-governed N³ (triangle-attention scores and P@V 512, `tri_mul`
  contraction 256), 768 × N³

| tokens | pinned share of one Pairformer layer |
|---|---|
| 1,003 | 12.4% |
| 2,096 | 7.6% |
| 3,012 | 5.7% |

It is N² against N³ competition, so it shrinks with length. Layer-passes
carrying it per released prediction: 64 × 4 recycles, plus 16 in the MSA
module, plus 8 × 5 samples in the confidence head. Treat these as a FLOP
share, not a wall prediction.

If anyone does want that row, the recipe is not "edit one constant". Both
snapshots need `--option dtype=float32`, or the pin has no float32 operand
and the measurement is of nothing; and the changed snapshot needs **two**
edits, not one -- `predict_kwargs["matmul_precision"]` added in
`api.predict`, *and* a `"high" -> Precision.HIGH` arm in
`triangle_attention.resolve_matmul_precision`, which takes
`highest`/`float32`/`fp32` and `default`/`tensorfloat32`/`tf32` and raises on
anything else. That refusal is deliberate: it makes a quiet wiring of the two
surfaces fail on the first prediction rather than compile a program nothing
has measured.

#### Harnesses pinned to the old value

Every Boltz-2 parity harness pins `highest` on both surfaces and was left
alone: `tests/parity/test_boltz2.py`,
`tests/models/boltz2/scripts/parity_matched_tape.py`,
`bench/boltz_foldjax_capture.py` (which refuses a capture recorded at
anything else), `bench/boltz_pair_stage_probe.py`,
`bench/boltz_atom_attention_probe.py` and
`bench/boltz_atom_projection_probe.py`. Their question is "do we match
upstream", so pinning is correct and nothing drifted.

**Four benchmark and profiling scripts now measure a configuration the
product no longer ships**: `tests/models/boltz2/scripts/predict.py`,
`profile_warm_stages.py`, `compare_trunk_backends.py` and
`benchmark_warm_predict.py` all latch `jax_default_matmul_precision` to
`highest`. They were left pinned on purpose, so their recorded numbers stay
comparable with each other -- but a figure they produce from today on is a
`highest` figure, not a default-configuration one. A recorded number whose
configuration has quietly stopped being the default is the failure mode the
stale 12,612 MiB baseline above is an instance of; say which arm a row
describes when you quote one.

The torch-gated checkpoint-parity modules (`tests/models/conftest.py`) are
immune for a different reason: they call the forward functions directly and
never enter the scope, so they take the signature defaults, which stay
`highest`. That is what keeps all 24 comparing upstream's rounding, and it is
why those defaults are not a tidy-up target.

#### What is still float32 in this trunk, with the scope at `high`

Asked after the flip: is anything left running genuine float32 arithmetic at
the released `compute_dtype=bfloat16`?

**No matmul runs at `HIGHEST` any more.** The only explicit `precision=` in
the whole Boltz-2 tree is the triangle-attention one above -- every other dot
inherits the scope -- and at the released dtype its operands are bfloat16. So
nothing in the trunk asks XLA for float32 accumulation.

**Float32 weights do survive, and now run TF32.** Applying
`_cast_trunk_params` at bfloat16 to the released checkpoint leaves 612.92 MiB
of float32 `kernel` parameters, and essentially all of it is the *single*
representation's path in each of the 64 trunk Pairformer layers, which that
function exempts on purpose:

| site | float32 | why it is exempt |
|---|---|---|
| `transition_s` fc1/fc2/fc3 | 432.00 MiB | the `transition_s` exemption |
| `attention` proj q/k/v/g/o | 180.00 MiB | the `attention` exemption |
| `attention` proj_z | 0.50 MiB | same |
| `input_embedder/atom_encoder` (10 sites) | 0.21 MiB | named subtree |
| `atom_attention_encoder/atom_to_token_trans` | 0.19 MiB | named subtree |
| `template_module/a_proj` | 0.03 MiB | named subtree |

A further 1.06 MiB over 84 sites is float32 norm affine, biases and
embeddings, which are not GEMM operands.

As arithmetic this is small and shrinking. Per Pairformer layer the exempt
path is 2,506,752 × N (the `transition_s` and attention projections, linear
in tokens) plus 2,816 × N² (proj_z and the single attention's own scores and
P@V), against the pair path's 393,216 × N² + 768 × N³ -- **0.45% of the
layer's MACs at 1,003 tokens, 0.20% at 2,096, 0.13% at 3,012.** It is not a
lever.

**What is left is storage, and it already has a knob.** The float32 pair
representation `[1, N, N, 128]` is 4.326 GiB at 3,012 tokens against 2.163
GiB in bfloat16 (2.095 / 1.047 at 2,096), and from roughly 2,000 tokens up it
is the peak's largest tenant. `--option pair_residual_dtype=bfloat16` narrows
it; it ships off and its own section above says why. After that the trunk has
nothing else wide: the remaining float32 activations are the single
representation `[1, N, 384]` and the norms, both negligible, and the
diffusion module's float32 island is upstream's `autocast(enabled=False)`
rather than anything this port chose -- it has its own
`diffusion_compute_dtype` knob.

#### What the flip does at the fused call sites

The cuEquivariance triangle-*multiplication* FFI reads the scope
(`models/_cueq.py:67`), so it moves -- but only where its operands are
float32: it overrides any float32 policy to `TriMulPrecision.DEFAULT` for
half-precision operands, so the released bfloat16 trunk saw `DEFAULT` before
and sees `DEFAULT` now, and only `--option dtype=float32` goes `IEEE` ->
`TF32`. The triangle-*attention* FFI does **not** read the scope on this
port: Boltz-2 hands it the op-level `precision` argument, and its TF32 flag
is dtype-gated as above. Tokamax does read the scope and resolves an unset
precision through it, so its float32 branch now takes `TF32_TF32_F32` where
it took the three-pass `F32_F32_F32`.

### A bfloat16 OpenDDE denoising network (`--option diffusion_dtype=bf16`)

OpenDDE's `trunk_dtype` above stops at the trunk: the diffusion module keeps
float32 weights and the three trunk representations are widened on their way
into the sampler, whatever the trunk and the confidence head are set to.
`--option diffusion_dtype=bf16` narrows the denoising network instead. It is a
sibling of the trunk dtype rather than a second spelling of it -- `trunk_dtype`
casts four whole parameter subtrees, while this one reproduces a boundary and
narrows only what upstream's autocast narrows -- and it is independent of
`--confidence-dtype`, which owns a different stage and reached its own bf16
default separately.

**Upstream runs float32 here and this is a deviation.** Upstream OpenDDE's
released `dtype` is `fp32` (`opendde/config/model_base.py:37`), so its autocast
context never opens and the denoiser is float32 at every size, whatever the
`skip_amp.sample_diffusion` token gate at `runner/inference.py:1501-1509`
selects. **It is unmeasured for accuracy.** No OpenDDE GPU row exists for it;
every published OpenDDE coordinate number was taken with the denoiser in
float32. Treat it as an experiment, not a recommendation.

The boundary follows a tensor's origin rather than its stage, which is where
AlphaFold 3 and upstream Protenix both draw it. Eleven projections stay
float32 because upstream constructs them `precision=torch.float32`: the atom
encoder's reference position and pair distance, its three coordinate
conditioning projections, the decoder's coordinate update, Algorithm 20's
single projection, and all four conditioner projections. The eleventh is
OpenDDE's own -- the projection that compresses the 384-channel trunk pair
representation to 128 channels before conditioning -- and Protenix has no
equivalent, so OpenDDE does not simply reuse Protenix's realisation. Every
per-head pair bias delivers its result in float32 while still multiplying in
bfloat16, inherited from a Protenix measurement on 5DEI and unmeasured here.
The sampler's state, its noise schedule and its rigid augmentation stay
float32, and a guard restores the denoiser's prediction to that width at the
boundary.

`--option diffusion_dtype=bf16` requires `dtype=bfloat16` (the released
OpenDDE default) and is refused on a float32 trunk: upstream opens one
autocast context from the global dtype, so "float32 everywhere except the
denoiser" is not a configuration it can run. It is refused under context
parallelism as well. **Context-parallel support is deliberately deferred**:
single-GPU comes first, and the sharded denoiser would need its own evidence.
The value is part of the compilation-cache identity, so a non-default value
never receives the executable built without it.

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
is the intermediate, not the precision. Note that it does not compose with
`--option dtype=bfloat16` on this port: measured 2026-09-11, adding
`glu_backend=tokamax` to the bfloat16 arm turns -23.3% wall into -1.3% and
raises peak from 5,553 to 5,775 MiB at 1,003 tokens, and turns -35.0% into
-30.8% at 2,096. The fused kernel loses at both sizes under bfloat16. Boltz-2
spells the same option the same way; this adds it to OpenFold3.

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
### Fused gated linear unit (`--option glu_backend`, Boltz-2, on by default)

Boltz-2's transitions and its triangle-multiplication gate compute
`activation(x @ w_gate) * (x @ w_value)`. Written as two matmuls and a
product, XLA writes the widened gate and value tensors out before multiplying
them; the fused Triton kernel runs the same arithmetic and never materialises
them. This is the released default. `--option glu_backend=xla` restores the
previous arithmetic exactly and gets its own compile-cache namespace.

Measured on one RTX PRO 6000 Blackwell, warm after prefill, the same input
file and schedule on both arms, released `xla` -> `tokamax`:

| tokens | wall (s) | peak (MiB) |
| ------ | -------------- | ---------------- |
| 1,003 | 90.98 -> 90.00 | 12,612 -> 9,216 |
| 2,096 | 317.91 -> 313.94 | 21,808 -> 21,778 |
| 3,012 | 806.01 -> 803.94 | 40,844 -> 40,749 |

Read the memory column honestly. The 26.9% saving at 1,003 tokens is a
small-input effect and does not generalise: what the kernel removes is the
transition's pre-gate intermediate, which stops being the peak's largest
tenant once the pair arena dominates above roughly 760 tokens, which is why
the two larger sizes save 0.1% and 0.2%. The durable claim is the wall-time
column -- never slower at any size.

Coordinates move, because the two paths do not round identically: the XLA path
evaluates the activation in float32 and casts back, the way the torch model
this ports from does, while the fused kernel applies the activation at the
kernel's own width. Same-index RMSD against the released arm is median 0.012 /
maximum 0.238 A at 1,003 tokens and median 0.007 / maximum 0.145 A at 2,096,
against per-case sample spreads of 1.0-2.9 A and 0.24 A -- far inside the
model's own scatter. The 3,012-token structures were not compared.

The kernel is pinned to Triton with no fallback, so a card that cannot run it
says so instead of running XLA under the tokamax name. A fused kernel cannot
be partitioned either, so under context parallelism the default resolves to
`xla` and the run proceeds; naming `--option glu_backend=tokamax` together
with `--option cp_devices=N` for N greater than 1 is refused rather than
silently downgraded.

### Fused gated linear unit (`--glu-backend tokamax`, Protenix)

Protenix has two gated transitions -- the `Transition` block the trunk, the
MSA stack, the diffusion conditioner and the confidence head all share, and
the diffusion transformer's `ConditionedTransitionBlock` -- and both compute
`silu(x @ Wa) * (x @ Wb)`. Spelled as two matmuls and an elementwise product,
XLA writes both widened branches and their product before narrowing the
result: three copies of the wide form, 7,866 MiB of a 10,914 MiB temp arena on
OpenDDE's structural pair tensor at 946 tokens, which is the measurement the
blocking in the shared `Transition` was added for. `--glu-backend tokamax`
runs the same arithmetic in one fused Triton kernel that writes none of them,
at both sites at once.

Blocking and fusing are alternatives, not complements: blocking *bounds* that
tensor by doing it a few rows at a time, fusing *removes* it. So a run that
passes both a transition chunk size and the fused backend gets the fused one
and is told once that the chunk size is unused -- the same shape of warning
the triangle and attention stacks give for their own fused kernels, and for
the same reason.

The default is `xla` and a released run is unchanged, bit for bit -- the
released value keeps the port's existing code, rather than routing it through
a shared function that rounds differently. The fused route is a numerics
change as well as a memory one, in two places: the kernel applies its
activation at its own width, and it is handed `jax.nn.silu` where the port
spells the same function as `x * (1 + exp(-x))^-1`. On float32 activations
that is an ulp; on a bfloat16 stage it need not be. There is no
fallback -- the implementation is pinned to Triton, so a card that cannot run
the kernel raises instead of running XLA under the tokamax name -- and the
backend is refused under context parallelism, where the kernel cannot be
partitioned and the widened tensor is already divided across devices. The
FP32-exempt projections of the bfloat16 diffusion policy are refused too:
their widen-then-narrow is not arithmetic the kernel can express. No
transition is on that exemption list today, so `--amp-policy bf16` and
`--glu-backend tokamax` combine.

**Neither port has a GPU measurement for this kernel.** The arena numbers
above are what the *unfused* transition costs, not what fusing saves. OpenDDE
reaches these same two transitions through Protenix's primitives and is
deliberately not offered the value -- asking for it is an error there, not a
silent no-op -- because a kernel is offered on the port whose numbers were
measured. OpenDDE is the natural next port to measure: the arena that
motivated the blocking in the first place is OpenDDE's.

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

### `--option token_attention_chunk=N` (Boltz-2)

Sets the query block the Boltz-2 diffusion token transformer's pair-bias
attention runs in. Unset takes the built-in rung, which is what production
wants; pass an integer to pin one width for every shape, and `0` to restore
the unblocked score buffer.

The buffer is what the knob is about. The token transformer keeps its logits
and their softmax in float32, so the pair costs `2 * samples * heads * N^2 * 4`
bytes: at the released five samples and sixteen heads that is 640 MiB at 1,024
tokens and 2.50 GiB at 2,048. Above 2,048 the long-sequence policy already
blocked it at 64, which brings the same buffer down to 80 MiB, so a
2,048-token input sat at the worst point of the curve and a 2,049-token input
did not. The rung removes that cliff: above 1,024 tokens the query axis is
blocked at 128, the same width the trunk Pairformer's own single-attention has
always used, and the buffer grows linearly rather than quadratically -- 160 MiB
at 2,048 instead of 2.50 GiB. Under bucketed padding the rung reaches exactly
two token buckets, 1,536 and 2,048; 1,024 and everything below it is on the
same program as before.

Blocking splits independent query rows and each row's softmax still reduces
over the whole key axis, so the arithmetic is exact -- but it is not bitwise,
because XLA reschedules a narrow query axis. Measured on CPU at a 128-wide
block, 200 and 256 tokens differ from the single-shot path in about 90% of
output words, at 3.1e-07 on values of order 1. Inputs at or below the block
width are untouched: that path short-circuits to the single-shot program and
is bit-identical.

Only the XLA attention path honours the block. A fused attention kernel
ignores it -- `attention_kernel=tokamax`, or a `diffusion_attention_backend`
that resolves to one -- because those kernels never materialise the score
buffer in the first place, and so does the 2-D context-parallel layout, which
splits both pair axes across devices instead. `null` is the same as leaving
the option unset.

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

### `--option glu_backend=tokamax` (ESMFold2)

Runs the twelve transitions in ESMFold2's diffusion token transformer through
`tokamax.gated_linear_unit`, one fused Triton kernel, instead of the widened
matmul and split the port takes by default. `xla` is the default and is what
every released number describes; the option is opt-in on this port for the
same reason it is on Boltz-2, which spells it the same way.

What it removes is the `2 * hidden` projection each of those blocks
materialises before gating -- temporary traffic, once per block per denoising
step. What it does not touch is ESMFold2's peak, which is a folding-trunk
arena quadratic in tokens and carrying no sample axis at all; expect this
option to buy time, not headroom, and do not reach for it to fit a longer
input. `structure_sample_sequential` above divides the sample axis, which is
a different term and not this peak.

Three limits. The kernel is Triton, so it needs a GPU and there is no
fallback: a card that cannot run it says so rather than running XLA under a
name that claims otherwise. Context parallelism refuses it, because a Pallas
custom call carries no partitioner for the sharded pair state. And a backend
change is a numerics change -- read a switched run against this port's own
rerun floor, not against the default as if it were exact.

### `--option confidence_dtype=bfloat16` (ESMFold2)

Narrows the confidence head's re-embedding to bfloat16, following AlphaFold
3's boundary: its confidence head casts the pair and single activations at
`confidence_head.py:121-127`, runs the whole re-embedding Pairformer narrow,
and returns to float32 at `:163` before the distogram-error logits and at
`:244` before pLDDT. `float32` is the default and no released number changes.

**Measured, and on its own it buys nothing.** GPU rows 1111/1112, this port,
against the released default:

| tokens | wall, float32 -> bfloat16 | peak, float32 -> bfloat16 |
|---|---|---|
| 1,003 | 155.16 -> 160.92 s (+3.7%) | 14,733.3 -> 14,733.3 MiB |
| 2,096 | 450.88 -> 450.05 s (-0.2%) | 46,041.8 -> 46,042.3 MiB |

The arm fires -- the 2,096-token peak moves 0.5 MiB, so it is not a dead
branch -- and it moves nothing else: no peak change at either size, and no
wall change outside noise. The reason is structural. This port's peak is a
single folding-trunk temp arena, and the confidence head contributes no term
to it, so a knob inside that head has nothing to shrink. **Do not reach for
this option alone.** It is here for combination arms, where it is the one
region of this port that can be narrowed without risking a structure.

**float32 here is upstream's own width, not an accidental island.** Upstream's
`ConfidenceHead.forward` runs outside every autocast region --
`modeling_esmfold2.py:172-221`, called at `:1061`, after the model-level
bfloat16 context opened at `:936` has closed at `:1030`. The only autocast
inside the head wraps its own `folding_trunk` alone, at `:223`, and it
receives a float32 pair and gives back `pair.add_(pair_delta.float())`. So
upstream's re-embedding and residual stream are float32, and the bfloat16
Linear operands inside the head's trunk are what `:223` already gives under
either value of this option.

The rows above are wall and peak. **Accuracy is the part still unmeasured
here**: every recorded ESMFold2 pLDDT, pTM and PAE number describes the
float32 default, and no row on this port reads the narrowed one back. What
exists is a sibling's measurement of the same narrowing: Protenix at 3,012
tokens moved coordinates not at all, atom pLDDT by at most 0.0099, chain pTM
and ipTM by at most 1.9e-4, and PAE means by at most 0.005. Read that as the
scale to expect, not as this port's number. The confidence head makes scores
and never coordinates, so nothing narrowed inside it can move a structure --
which is why this is the safest dtype change the port offers.

What it narrows: the five projections that build the pair from the single
input, the distance-bin embedding gather, and the residual stream those feed
into the head's own trunk. What stays float32: both entry normalisations, the
representative coordinates and the distances built from them, the comparison
that picks a distance bin, every mask, and everything past the trunk -- the
row-attention pooling and all four output heads. Two of those are load-bearing
rather than cautious: the pooling projection's output is the score tensor of a
softmax, and the PAE head's output is what pTM and ipTM are read off, through
a softmax with no float32 guard of its own. Nothing that feeds a softmax or an
exponential is rounded, and every returned score is float32 in both arms.

**Read off the trace, not off the setting.** A jaxpr census of
`confidence_head` on CPU, counting `dot_general` operand dtypes and casts
through every sub-jaxpr including the `platform_dependent` branches:

| | `(f32,f32)->f32` | `(bf16,bf16)->bf16` | `(bf16,bf16)->f32` | branch points |
|---|---|---|---|---|
| `float32` | 17 | 8 | 10 | 8 |
| `bfloat16` | 12 | 13 | 15 | 13 |

Exactly five float32 matmuls disappear and exactly five branch points appear:
the five `s_to_z*` projections, routed through `_autocast_linear`. Read those
two counts as the signal. The bfloat16 dots gain ten rather than five because
the tracer keeps both platform branches of each narrowed Linear -- the CUDA
one accumulating to bfloat16 and the fallback to float32 -- so that column
double-counts. The stored parameters are not touched either way:
`_autocast_linear` narrows its operands inside the call, and the checkpoint
arrays stay float32 and uncopied.

It is separate from `trunk_dtype`, and deliberately so. `trunk_dtype` targets
the region upstream's own autocast covers, and it already reaches *inside*
this head: the head reopens that autocast for its own trunk the way upstream
does, so that stack runs with bfloat16 Linear operands at the released
default. What is still float32 is the re-embedding in front of it. Setting
`confidence_dtype=bfloat16` under `trunk_dtype=float32` narrows the
re-embedding and leaves the head's trunk in float32.

Unlike the fused kernels above, it composes with context parallelism: this is
arithmetic under a sharding constraint rather than a custom call GSPMD has no
partitioner for. Under a mesh the head's trunk takes its storage-cast branch,
which the option does not select, so there it narrows the re-embedding only.

The value joins the compilation-cache identity, so a narrowed run never
receives the executable built without it, and the two spellings are exact:
`bf16` is refused, naming `float32` and `bfloat16`.

Only the diffusion token transformer is fused. The atom encoder and decoder
feed-forwards inside the same denoiser, the trunk's SwiGLU and transition
layers, and the ESMC language model's MLP all stay on XLA.

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
