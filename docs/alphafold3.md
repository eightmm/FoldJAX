# Running AlphaFold 3 through FoldJAX

FoldJAX carries AlphaFold 3's Apache-2.0 source and runner, but never its model
parameters. The parameters are released directly by Google under separate
terms that forbid redistribution. Everything else uses the same interface as
the other models — one job file, one model name.

```bash
foldjax predict --model alphafold3 --input job.yaml
```

## Installing the AlphaFold 3 runtime

Install FoldJAX's AlphaFold 3 extra for its non-JAX runtime dependencies. Keep
the CUDA extra that matches the machine, then inspect or
prepare the generated runtime explicitly:

```bash
uv sync --extra alphafold3
foldjax runtime status --model alphafold3
foldjax runtime prepare --model alphafold3
```

On the first AlphaFold 3 prediction, FoldJAX builds upstream's small
ABI-specific extension and the generated CCD lookup tables. These files go to
`$FOLDJAX_HOME/runtime/alphafold3`, never into the installed wheel or
site-packages, so a read-only container installation works. `foldjax home`
prints the exact runtime location. `runtime prepare` makes this potentially
long step explicit; prediction still prepares it automatically on first use if
you skip that command.

The first build needs a C++ compiler, Git, network access, and either `uv` or
Python's `pip`. The `alphafold3` extra supplies the Python *runtime*
dependencies. CMake and Ninja are isolated build requirements declared by the
vendored AlphaFold 3 project, so `uv`/`pip` obtains them for the temporary build
environment; the extra does not install them directly. CMake also fetches
pinned C++ sources from their upstream Git repositories. Later predictions
reuse the source-, Python-ABI-, platform-, and NumPy-version-keyed runtime.

FoldJAX deliberately ignores unmarked `cpp*.so` files and generated CCD
pickles left in a source checkout. They do not prove which source or NumPy ABI
created them, so reusing them after an update could fail much later inside the
model. Generated artifacts are published only through the keyed runtime store.

FoldJAX does not automatically adopt an independently installed
`alphafold3` distribution. The default is always the vendored,
source-and-ABI-keyed managed runtime, even when another distribution appears
earlier on `sys.path`. This makes the code provenance and JAX-only dependency
boundary reproducible.

If an external `alphafold3` package was already imported in the current Python
process, FoldJAX refuses to replace its native extension in place. Restart the
process before using the managed runtime.

For development against an upstream checkout, opt in explicitly with the
backend's `source` option. Place that checkout's `src` first on `PYTHONPATH` so
the selected runner and package have the same provenance; FoldJAX rejects a
mismatched pair:

```bash
PYTHONPATH=/path/to/alphafold3/src:$PYTHONPATH \
  foldjax predict --model alphafold3 --input job.yaml \
  --option source=/path/to/alphafold3
```

Without that explicit option FoldJAX uses the runner matching its vendored
source. The former `ALPHAFOLD3_SOURCE` ambient environment switch is not
honoured. Do not install upstream's JAX dependency over FoldJAX's CUDA-specific
JAX environment.

## Weights

Request the parameters from DeepMind
([terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)),
then drop the file into the FoldJAX weight store:

```bash
weights_store=$(foldjax home --path weights)
mkdir -p "$weights_store/alphafold3"
cp af3.bin "$weights_store/alphafold3/"
```

`foldjax weights list` will then show `alphafold3` as ready. FoldJAX never
downloads, converts, or redistributes these parameters — it only knows where to
look. `--weights` still works if you keep them somewhere else.

## MSAs

AlphaFold 3 refuses to featurise a protein or RNA chain whose MSA is *absent*:
it assumes its own genetic-search pipeline will produce one, which needs
hundreds of gigabytes of databases. FoldJAX therefore writes an **empty** MSA
when your job carries no alignment, which AlphaFold 3 accepts as single-sequence
mode — the same fallback the other MSA-capable backends already have. Predictions will
be much worse than with a real MSA; supply one for anything you care about:

```yaml
entities:
  - type: protein
    id: A
    sequence: LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF
    unpaired_msa: msas/A.a3m
```

To use AlphaFold 3's own search pipeline instead, run its `run_alphafold.py`
directly with `--db_dir` — that path is outside FoldJAX.

## GPU kernels on an unfamiliar device

AlphaFold 3's transition blocks call `tokamax.gated_linear_unit`, which prefers
a Triton kernel and, on a cache miss, sizes it from heuristics. Those heuristics
are not right for every device: on an RTX PRO 6000 Blackwell they ask for
108 KiB of shared memory against the 99 KiB the device has, and the launch fails
with `RESOURCE_EXHAUSTED` from inside the diffusion transformer — after the
kernel has compiled, so Tokamax's own per-implementation fallback never sees it.

FoldJAX therefore runs AlphaFold 3 with Tokamax's autotuner enabled, which
benchmarks the candidate configurations on whatever device is in front of it and
keeps one that fits. This costs time on the first compile of each shape and is
cached afterwards. Two knobs if you want something else:

| option | effect |
|---|---|
| `--option kernel_autotuning=heuristics` | upstream default; faster first run, may not fit your device |
| `--option kernel_autotuning=error` | fail loudly on a cache miss instead of guessing |
| `--option attention_backend=xla` | skip the Triton attention kernel entirely |

## Options

`--num-samples` and `--num-recycles` map onto AlphaFold 3's own
`diffusion_samples` and `recycles`; its defaults (5 samples, 10 recycles) apply
otherwise. Native names are still reachable with `--option`:

```bash
foldjax predict --model alphafold3 --input job.yaml \
  --option diffusion_samples=5 --option recycles=10 \
  --option buckets='[256,512,1024,2048,5120]'
```
