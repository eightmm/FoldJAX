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
After successful warming, the CLI prints a prediction command template to
stderr. JSON stdout remains machine-readable. Keep the other execution options
and weights/profile the same; the template does not infer the real job's MSA
search policy.

Prediction results are discarded unless `--output-dir` is supplied. Retained
bucket results go under `OUT/MODEL/BUCKET/REAL_LENGTH`; generated directories
cannot escape through a child symlink. Synthetic job documents are temporary.

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
