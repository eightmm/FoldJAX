# Installation details

Mixed CUDA generations, the per-model extras, and what each one is for.

Mixed clusters pick a CUDA generation per node from the same checkout and
lockfile — the extras conflict by construction, so each node type syncs its
own venv; weights, caches and sources are shared:

```bash
uv sync --extra cuda13 --group dev                                    # CUDA 13 nodes
UV_PROJECT_ENVIRONMENT=.venv-cu12 uv sync --extra cuda12 --group dev  # CUDA 12 nodes
```

Managed weight conversion and all six inference paths use NumPy/JAX only.
OpenFold3 can predict from a self-contained feature `.npz`; turning raw
JSON/YAML into that archive uses the Torch-free in-package featurizer and the
non-Torch chemistry packages in `--extra openfold3-preprocess`. Protenix
ESM/ISM conditioning runs the exact ESM2 architecture in JAX.

**AlphaFold 3** needs `--extra alphafold3`; its compiled half and generated CCD
tables are built under `$FOLDJAX_HOME/runtime/alphafold3` on first use — a
one-time step that also works from a read-only wheel install. FoldJAX uses its
versioned managed runtime by default; external source is explicit opt-in:
[docs/alphafold3.md](alphafold3.md).
**OpenFold3** predicts from an embedded-chemistry `.npz` with no extra; raw job
featurization needs `--extra openfold3-preprocess`:
[docs/openfold3.md](openfold3.md).
