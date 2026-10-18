# Warm a compilation cache

An exact-input warm still runs one representative seed through prediction:

```bash
foldjax cache warm --model boltz2 --input job.json --padding --msa single
```

To prepare several serving sizes on the selected device, supply token buckets
instead of input/sequence arguments:

```bash
foldjax cache warm --model boltz2 protenix --token-buckets 256,512,768 \
  --msa single --fixed-msa 2048
foldjax cache warm --model esmfold2 --token-buckets 256..1536 --json
foldjax cache warm --model alphafold3 --token-buckets auto
```

The core fixed-MSA preset requires an explicit positive capacity. The example uses
the recommended 2048 to match the Slurm scripts. The user's 1UBQ/2LZM/4HHB
comparison on the four supported models found lDDT within ±0.01 across
1024/2048/4096 rows; this is finite-panel evidence, not a general accuracy
guarantee. Choose the same cap for warming and prediction.
The four MSA presets and native selection limits are explained in
[fixed-msa.md](fixed-msa.md). AlphaFold 3 and ESMFold2 do not accept that preset.

A list is sorted and deduplicated. A range includes the model's existing bucket
grid entries between its inclusive endpoints; it does not enumerate every
integer. `auto` selects the full grid. The ordinary grid is
`padding.TOKEN_BUCKETS`: 128-token steps below 1024, then 256-token steps through
8192. AlphaFold 3 uses its carried runner's native list: 256, 512, 768, 1024,
1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120. An explicit native AF3
`--option buckets='[...]'` selects that native route and retains its cache
namespace; do not combine it with neutral `--padding` options. Native intervals
must contain two distinct positive lengths to be verified this way.

Each bucket executes **two** nearby, shorter single-chain protein lengths under
one padded token target. Synthetic warming uses single-sequence input without
searching an external server; `--msa auto/required` belongs to an exact real-job
warm. The normal sampling schedule is retained, and weights must already be
installed. This runs real prediction and kernel tuning, can be expensive, and
should be submitted to a GPU compute node on a cluster. It does not download
weights or claim that the synthetic structure has scientific value.

Protenix, OpenDDE, OpenFold3 and ESMFold2 keep the actual chain count out of the
compiled model's static arguments. Confidence scores reserve 64 chain slots with
runtime masks/counts and cropped output axes. Inputs with up to 64 chains therefore share
the same body executable when the other padded shapes and execution options
agree. More chains grow the capacity to the next power of two and emit a warning;
that overflow needs another executable. The per-sample confidence schedule
remains sequential; chain pairs do not stack full-bin PAE tensors.

Warming opens the same backend session as managed prediction, including its
weight/feature staging. ESMFold2 warms the compact language-model embedding
route used by prediction, rather than the older all-layer-state route. Earlier
warm memory limits may therefore be conservative; measure the updated path
before choosing larger serving buckets. See [cache compatibility](cache-compatibility.md)
for the ESMFold2 revision that requires warming again.

The CLI defaults to `--isolation process`. Each model/bucket starts a fresh
Python interpreter, so model parameters, compiled executables, and the CUDA
context from an earlier bucket cannot accumulate in the next one. Both lengths
of one bucket run together in that child to check reuse. Exact-input warming
uses a fresh child per model; multiple inputs for that model stay together.
Device/budget queries and request resolution also happen in the child. Children
inherit the current interpreter, FoldJAX source, environment, and request
options; no scheduler job or second GPU allocation is created.

Use `--isolation in-process` to retain the previous single-process behavior.
Python `warm_cache(request)` and `warm_token_buckets(request, spec)` retain that
default for compatibility; pass `isolation="process"` explicitly to isolate
them. Process mode accepts JSON-compatible native options. Ambient
`jax.default_device` contexts, changes made through `jax.config.update`, custom
backend registrations, callbacks and device objects do not cross an exec
boundary. Express device/platform selection with request options and the
environment, or use in-process mode for those Python-only settings.

Buckets run in ascending order for each model. By default a CUDA allocation
failure or proactive memory-admission refusal stops that model's larger buckets;
the next model can still run. `--keep-going` attempts later buckets too.
`--stop-on-oom` explicitly selects the default. A failed bucket appears in the
report and makes the CLI exit with status 3.

The default output is a table. `--json` includes device kind and card/pool byte
capacities when readable, requested and successful buckets, first failure and
reason, elapsed seconds, and both ordinary warm reports for each bucket. The
two runs must select the same compilation namespace and concrete padded targets
when the backend reports them. The report also counts JAX compilation requests
on the second length. A request can restore an existing executable, so this
count is diagnostic and **is not a persistent-cache miss count**. Cache file
deltas likewise do not prove an executable hit. Exact-input warming retains its
existing JSON output and `execute_once` behavior.
Isolated reports include the worker PID and exit code for each bucket or exact
result. Native worker progress goes to stderr. A worker terminated without a
report is a failure, with its signal/exit status; an unexplained `SIGKILL` is
not automatically classified as OOM. Normal exceptions preserve the bucket
failure/stop-on-OOM policy. `SystemExit` and keyboard cancellation stop the warm.
After successful warming, the CLI prints a prediction command template to
stderr. JSON stdout remains machine-readable. Keep the other execution options
and weights/profile the same; the template does not infer the real job's MSA
search policy.

Prediction results are discarded unless `--output-dir` is supplied. Retained
bucket results go under `OUT/MODEL/BUCKET/REAL_LENGTH`; generated directories
cannot escape through a child symlink. Synthetic job documents are temporary.
The parent owns each worker's discarded-output directory and removes it even
when the worker dies. On keyboard cancellation it terminates and reaps its own
active worker before cleaning that directory. Main-thread warming also handles
an ordinary SIGTERM this way. Existing custom SIGTERM handlers retain their
application-defined behavior; SIGKILL of the parent cannot run cleanup.

To inspect the current expected namespaces without compiling, predicting,
loading model parameters, or searching for MSAs, add `--verify-only`:

```bash
foldjax cache warm --verify-only --model protenix --token-buckets 128..1024 \
  --fixed-msa 2048 --msa single
foldjax cache warm --verify-only --model esmfold2 --input job.json --padding
```

This always prints JSON with the expected namespace and graph revision, each
trusted shared/private tier's entry count, and `entries_present` or `missing`.
Exit status is 0 when every requested namespace contains a non-empty regular
JAX executable entry and 3 when any does not. It creates no cache directories or
entries, ignores tuning-only files and symlinks, and applies prediction's cache
trust rules. Installed weight locations and the selected device/runtime are
still queried to derive the same namespaces as prediction. Bucket checks use
the same two synthetic lengths and padding/options as warming.

This is a **namespace inventory**, not a cache-hit or complete-coverage check.
Entries are not deserialized and their individual HLO keys are not computed.
Code or concrete input changes can require compilation within an existing
namespace, and the presence of one entry does not prove that every executable
or Tokamax tuning result is available. Announced graph-ABI changes use a new
revision, so this check can flag the resulting missing namespace. See
[cache compatibility notes](cache-compatibility.md) for models that need warming
again after an update.

To check a real input's executable key rather than just its namespace, use:

```bash
foldjax cache verify --model esmfold2 --input job.yaml --padding --msa single
foldjax cache verify --model boltz2 --input job.yaml --fixed-msa 2048 --msa none
```

The second example expects local alignments in the input. Verification never
searches an MSA or template service: choose `--msa single`, or preserve supplied
local alignments with `--msa none`. Use the same input, weights, device and
execution options as prediction. This can read checkpoints and preprocess the
input, but it lowers the model graph without compiling or executing it. JAX's
own HLO/options/device key calculation is used to inspect trusted shared entries
first, then private entries. No executable is deserialized and no cache entry is
written.

The JSON report includes each graph's `cache_key`, `key_present` or `missing`,
and its shared/private source. Exit status is 0 when every checked key is present,
and 3 for missing keys or unsupported paths. Presence does not validate the
entry's contents or guarantee successful execution. Current adapters cover the
managed ESMFold2 structure graph and the Boltz-2 primary graph on one device;
they do not check LM staging graphs, auxiliary JITs, or kernel tuning. Boltz-2
affinity, eager steering, context parallelism, PGLE and other model adapters
are reported as unsupported, as is an ESMFold2 installation that cannot establish
the managed compact-LM route. A matching namespace containing only another
shape's executable is reported as missing by this check.

Boltz-2 also records the resolved trunk, diffusion and confidence chunk widths
in `foldjax_run.json` under `resolved_chunking`, including calls that reuse a
compiled runner. Its automatic widths follow token-size rungs, independently
of the card's memory budget. A 22/24 GiB budget alone therefore does not explain
the reported ECC/non-ECC 4090 mismatch; compare these values and the executable
keys on the actual cards before attributing it to chunk selection. JAX's device
topology and runtime settings also participate in its executable key.

For prediction reuse, select the same model/weights, device, runtime, token and
other padded capacities, MSA cap, and compile-relevant native options.
Bucket-warmed AF3/ESMFold2 profiles require `predict --padding`; the four-model
fixed-MSA profiles require the same `predict --fixed-msa N` (recommended 2048).
An unpadded exact-length prediction does not match those warmed profiles.
`--padding` chooses the token bucket; explicit non-grid targets require the same
`--pad-tokens` value in prediction. Templates, extra modalities, language-model
providers, context parallelism, or other shape axes can require additional
executables. A warm report does not promise portability across GPU kinds or
that every different job will share every small JIT.

See [shared installation](shared-install.md) for administrator/reader cache
permissions and [Slurm examples](../examples/slurm/README.md) for submission and
a separate ordinary-user verification run. Those scripts enable fixed MSA 2048
for its four supported models and padding for AF3/ESMFold2 in warm, verification,
plan and prediction. Use matching overrides, or `--exact` with an exact input,
at every stage. The core CLI does not enable a preset implicitly.
