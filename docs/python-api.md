# Python API

Driving FoldJAX from Python instead of the command line: requests, results,
sessions and the structured events the CLI renders.

Everything the CLI does, as one request type — and the same neutral
vocabulary, so nothing below has a CLI-only or API-only spelling.

```python
from foldjax import PaddingConfig, PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="protenix",          # boltz2 / esmfold2 / opendde / openfold3 / alphafold3
        input="job.yaml",          # FoldJAX job, or the model's native format
        output_dir="out/",
        seed=42,
        # Sampling, in one vocabulary. None keeps that backend's default.
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=8192,
        # Opt-in shape normalization. `padding=True` uses standard buckets;
        # pinning axes defines one exact serving/cache profile.
        padding=PaddingConfig(tokens=512, atoms=4096, msa=128),
        # Execution, likewise: dtype, matmul_precision, triangle_kernel,
        # attention_kernel. "auto" is the fastest path this build can run --
        # never a silent fallback to a slower one.
        options={"dtype": "bfloat16", "triangle_kernel": "auto"},
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

The same prewarming operation is available without the CLI:

```python
from foldjax import PredictionRequest, warm_cache

report = warm_cache(
    PredictionRequest(model="opendde", input="job.yaml", num_steps=200)
)
print(report.summary())
```

**One declaration, several runs.** Every axis that can name one thing can name
several -- the plural spelling fans out, nothing else changes. `models` x
`inputs` runs the cross product, each prediction into its own
`output_dir/<model>/<input stem>` subtree; `seeds` reruns each under every
seed. Model aliases are canonicalized before the path is chosen, and two input
files with the same stem are refused if they would collide. One result comes
back per model/input run, in declaration order:

```python
results = predict(
    PredictionRequest(
        models=("boltz2", "protenix", "openfold3"),
        inputs=("kinase.yaml", "complex.yaml"),
        seeds=(101, 102, 103),
        num_samples=5,
    )
)  # 3 models x 2 inputs -> 6 results, each holding 3 seeds x 5 samples
```

Representation capture is single-seed: `PredictionResult` carries one
`Representations` archive handle, so combining `representations=...` with more
than one resolved seed is rejected during request resolution instead of
silently retaining only one archive. Run one seed per request when capturing
trunk arrays. Empty selectors such as `representations=(",",)` are also
rejected; name at least one array or use `"all"`.

The CLI spells it the same way: `foldjax predict --model boltz2 protenix
--input kinase.yaml complex.yaml`.

For multiple different models, omit `weights` so each resolves its managed
checkpoint. One explicit `weights` path is accepted only when every run uses
the same canonical model; this prevents handing one model another model's
checkpoint by accident.

`resolve_request(request)` resolves one scalar model/input pair.
`resolve_requests(request)` handles either spelling and returns the complete
tuple of scalar requests, including the plural cross product and final output
paths, without executing a model. `foldjax plan` exposes the latter behavior in
the CLI and prints one object for a scalar request or a list for a plural one.

`sample.scores` carries that model's released confidence summaries. Native
option spellings (`compute_dtype`, `trunk_dtype=bf16`, ...) still work and
warn once; `foldjax.execution.KNOBS` lists the neutral vocabulary. Predictions
return summaries, not raw tensors -- the full-bin PAE/PDE logits are tens of
GiB at long sequences and come back only on request
(`--option return_confidence_logits=true` on Boltz-2, `--option
include_raw=true` on OpenDDE, or the raw `.npz` output formats). Both are
native option names passed through `--option`, not FoldJAX flags of their own.

Leave a sampling knob unset and each backend runs its own upstream's released
default -- which differ, deliberately: matching each upstream is the whole
point of the ports. What `None` means per model:

| model | samples | steps | recycles | MSA depth |
|---|---|---|---|---|
| `alphafold3` | 5 | 200 | 10 | 1,024 rows (`evoformer.num_msa`) |
| `boltz2` | **1** | 200 | **3** | uncapped -- the whole alignment |
| `esmfold2` | **32** | **14** | **3** | 1,024 rows, resubsampled per loop |
| `opendde` | 5 | 200 | 10 | 16,384 rows |
| `openfold3` | 5 | 200 | **3** | 1,024 rows, subsampled |
| `protenix` | 5 | 200 | 10 | 16,384 rows |

ESMFold2's row is not a typo: its released `config.json` asks for 32 samples
off a 14-step schedule, and the schedule is then clipped at `sigma = 256`, so
it runs fewer denoiser calls than the number suggests. Its own source's
dataclass defaults say 68 steps and 20 loops; the file wins.

The benchmark pins the five AlphaFold-3-shaped models to one schedule
(5 / 200 / 10) precisely because these defaults differ; set `num_recycles=10`
on Boltz-2 and you are asking more of it than `boltz predict` does. ESMFold2 is
not on that chart: it has no evolutionary trunk and no comparable schedule, so
putting it on the same axes would compare two different questions.

`foldjax.capabilities(model)` reports the runnable backend's scientific input
and sampling surface. A dormant code path disabled by the shipped inference
configuration is not a capability; for example, OpenDDE reports
`supports_templates=false` because its fixed `use_template=False` path drops
template input. Two fields further describe the *common schema* route:
`common_schema_features` is what this backend's dialect can carry from a
FoldJAX job document, and `native_only_features` names abilities the model has
but the common adapter cannot safely reach. Templates are the case that
matters: backend support does not imply that every input route honours a
per-job template.
Its `input_requirements` mapping distinguishes
dependencies by input format — for example, OpenFold3's `openfold3-features` archive is JAX-only,
while its raw `native`, `openfold3`, and `foldjax` formats require the
non-Torch `openfold3-preprocess` chemistry extra for JAX featurization.
`foldjax.model_info(model)` adds execution choices and managed-weight
readiness, path, provenance, download size, and setup guidance; it is the
Python counterpart of `foldjax models --json`. Backend-specific output stays
on `PredictionResult.raw`.

A backend returning normally is not enough to count as success. FoldJAX checks
that it returned a `PredictionResult` for the selected model, at least one
sample, and for every sample either a non-empty structure file or non-empty
coordinates. Violations raise `PredictionOutputError`. A vendored native CLI
that calls `SystemExit` is converted to `PredictionError` instead of terminating
the host Python process; diagnosed allocator failures remain `MemoryError`.
Copied native structures retain their real `.pdb`, `.cif`, or `.mmcif` suffix,
and multi-job AlphaFold 3 inputs keep separate job names instead of overwriting
one another. Secret-looking option values and URL credentials are redacted from
plans and run manifests.
