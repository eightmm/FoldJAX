# Installation details

Mixed CUDA generations, the per-model extras, and what each one is for.

`uv sync` with no arguments is the CUDA 13 development environment. The `gpu`
dependency group carries that runtime and is on by default, as is `dev`; only a
departure from it needs a flag:

```bash
uv sync                                  # CUDA 13, plus pytest and Ruff
uv sync --no-default-groups --group dev  # CPU-only machine: no CUDA wheels
```

A CUDA 12 node switches the default group off through the environment rather
than a flag, and leaves it set for the whole shell. `uv run` re-syncs the
default groups before it runs anything, so a bare `uv run` on such a node
reinstalls the CUDA 13 wheels beside the CUDA 12 ones:

```bash
export UV_NO_GROUP=gpu
export UV_PROJECT_ENVIRONMENT=.venv-cu12
uv sync --extra cuda12
```

Mixed clusters pick a generation per node from the same checkout and lockfile.
The two CUDA extras conflict by construction, and so does `--extra cuda12` with
the default `gpu` group, so a forgotten `UV_NO_GROUP` stops with a conflict
error instead of installing both generations. Each node type syncs its own
venv; weights, caches and sources are shared.

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
