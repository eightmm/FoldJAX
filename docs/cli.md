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
With `--padding`, each backend selects the smallest conservative *complete*
shape profile: tokens plus the atom, MSA, template, structural-token, and/or
language-model axes that actually enter that model. Masks keep padded entries
out of the structure and confidence output, and the run summary and
`foldjax_run.json` report the concrete profile that ran.

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

Design notes: [docs/engineering-notes.md](engineering-notes.md).

### A bfloat16 trunk (`--option trunk_dtype=bf16`)

Protenix defaults to `bf16` (its upstream ships bf16-mixed); OpenDDE defaults
to `fp32` (so does its upstream) and takes the flag as an opt-in -- at 1,531
tokens it is the difference between completing and OOM on both sides. Boltz-2's
equivalent is `compute_dtype`, default `bfloat16`; OpenFold3 has no trunk
dtype: upstream runs `32-true` and a bf16 trunk destroys its prediction.
Details and measurements: [docs/engineering-notes.md](engineering-notes.md).

### `--max-msa-depth`

Caps the `[depth, tokens, channels]` MSA representation, the dominant memory
term after the pair stack. What the cap costs in accuracy differs by model --
free on Protenix (its upstream subsamples anyway), pointless on OpenDDE, and
potentially harmful when a model embeds the complete alignment before recycling
-- so FoldJAX translates the knob but never imposes it. Per-model semantics and
the measurements:
[docs/engineering-notes.md](engineering-notes.md).

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
starting, which is what the warning prints. OpenFold3's 2.29 GB p1
checkpoint comes from the publisher's own unsigned S3 key, pinned by SHA-256,
so it needs no account and no request; upstream p2 and OpenBind v0.5 target
newer incompatible architectures and are not substituted.

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
alphafold3` removes the ones nothing can select any more; it always keeps the
live tree, and by default keeps anything prepared in the last week too, since
this store is shared between checkouts and another commit's tree may be live
for someone else. `--dry-run` shows what would go.

### GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

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
