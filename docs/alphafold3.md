# Running AlphaFold 3 through FoldJAX

AlphaFold 3 is the one backend FoldJAX does not carry itself. Its parameters are
released under terms that forbid redistribution, and upstream pins a CUDA 12
JAX plugin that cannot co-resolve with a CUDA 13 cluster. So FoldJAX does not
list it as a dependency: you install it for your own hardware, and FoldJAX finds
it. Everything after that is the same as any other model — one job file, one
model name.

```bash
foldjax predict --model alphafold3 --input job.yaml
```

## Installing AlphaFold 3 into the FoldJAX environment

AlphaFold 3 from v3.0.5 onwards targets the same stack FoldJAX runs on — JAX
0.10.x, Tokamax 0.0.12, Python 3.12 — so it installs into the same environment.
Earlier releases pinned `jax==0.4.34` and `triton==3.1.0` and do not.

```bash
git clone https://github.com/google-deepmind/alphafold3
uv sync --extra cuda13 --extra alphafold3    # its deps, minus JAX
uv run build_data                            # builds the CCD tables, ~5 min
uv pip install --no-deps ./alphafold3        # --no-deps keeps your JAX plugin
```

Install it **non-editable**, and build the CCD tables first. `build_data` writes
a 512 MB `ccd.pickle` next to the source it is run against, and AlphaFold 3 reads
it back from its own package directory — there is no environment variable for it.
An editable install therefore leaves that half-gigabyte in the checkout and needs
it at run time; a plain install copies it into the environment, and the checkout
becomes disposable. The runner does not come along either, which is why FoldJAX
carries a copy: see `backends/_alphafold3_upstream/`.

`--no-deps` is the important part. AlphaFold 3 requires `jax[cuda12]`; letting
that resolve would install a second JAX plugin next to the one your cluster
actually uses. Everything else it needs is generation-independent, which is what
the `alphafold3` extra carries — `absl-py`, `dm-haiku`, `etils[epath]` and
`zstandard`, with `rdkit` and `tqdm` already in the base set.

**Afterwards, sync with `--inexact`:**

```bash
uv sync --inexact --extra cuda13 --extra alphafold3
```

`uv sync` is an *exact* sync by default: it removes anything not in the
lockfile, and the `alphafold3` package itself cannot be in the lockfile, so a
plain `uv sync` uninstalls it every time. The extra keeps its dependencies from
being removed alongside it, but only `--inexact` keeps the package. This is the
one manual step in the "one environment runs every model" story, and it exists
because AlphaFold 3 is the one backend FoldJAX cannot carry.

If you would rather keep it in its own environment, point FoldJAX at the
checkout instead and the backend will load the runner from there:

```bash
export ALPHAFOLD3_SOURCE=/path/to/alphafold3
```

Without that variable the backend looks for `run_alphafold.py` beside the
imported package — which is what an editable install gives you — and otherwise
uses its vendored copy.

## Weights

Request the parameters from DeepMind
([terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)),
then drop the file into the FoldJAX weight store:

```bash
mkdir -p "$(foldjax home)/weights/alphafold3"
cp af3.bin "$(foldjax home)/weights/alphafold3/"
```

`foldjax weights list` will then show `alphafold3` as ready. FoldJAX never
downloads, converts, or redistributes these parameters — it only knows where to
look. `--weights` still works if you keep them somewhere else.

## MSAs

AlphaFold 3 refuses to featurise a protein or RNA chain whose MSA is *absent*:
it assumes its own genetic-search pipeline will produce one, which needs
hundreds of gigabytes of databases. FoldJAX therefore writes an **empty** MSA
when your job carries no alignment, which AlphaFold 3 accepts as single-sequence
mode — the same fallback the other four backends already have. Predictions will
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
