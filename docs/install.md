# Installation details

Mixed CUDA generations, the per-model extras, and what each one is for.

`uv sync` with no arguments is the CUDA 13 development environment. The `gpu`
dependency group carries that runtime and is on by default, as is `dev`; only a
departure from it needs a flag:

```bash
uv sync                                  # CUDA 13, plus pytest and Ruff
uv sync --no-default-groups --group dev  # CPU-only machine: no CUDA wheels
```

The CPU-only line holds for one command only. `uv run` syncs the default
groups before it runs anything, so the next `uv run pytest` reinstalls the CUDA
wheels. Make the opt-out last for the shell (or put it in your profile):

```bash
export UV_NO_DEFAULT_GROUPS=1            # every uv command now skips dev and gpu
uv sync --group dev
uv run pytest -q                         # stays CPU-only
```

**A `uv sync` removes every extra it is not given.** It syncs exactly, so a
bare `uv sync` after `uv sync --extra alphafold3` uninstalls AlphaFold 3's
dependencies again. Name every extra you use, every time:

```bash
uv sync --extra alphafold3 --extra openfold3-preprocess --extra templates
```

`uv run` syncs inexactly and leaves extras in place; `uv sync --inexact` does the
same when adding one. That is the form `foldjax doctor` prints in a checkout.

| extra | what it adds |
|---|---|
| `cuda13` / `cuda12` | the CUDA JAX plugin, cuEquivariance ops and Triton (`cuda13` is also the default `gpu` group) |
| `alphafold3` | AlphaFold 3's runtime dependencies (Haiku and friends) |
| `openfold3-preprocess` | OpenFold3 raw-job featurization; includes `kalign-python` |
| `templates` | `kalign-python` alone: the realignment behind `--templates auto`/`required` for AlphaFold 3, Protenix, OpenDDE and OpenFold3 |

`kalign-python` is an extra rather than a base dependency because it publishes
x86-64 Linux and macOS wheels only; in the base set every aarch64 Linux install
would compile it from source.

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

## Version of an editable install

In a checkout, `foldjax.__version__` comes from the source and is the version
that ran. The installed distribution metadata (`importlib.metadata.version`,
`pip show foldjax`) is written when the checkout is synced and is not updated
by a pull, so it can name an older release: one development venv reported
0.3.0 against a 0.1.0 source tree. `foldjax doctor` prints both and flags the
mismatch; `uv sync --inexact` (with your usual extras) refreshes it. A run
manifest records the source version and, in a checkout, `git describe
--always --dirty` under `foldjax_source`.

## pip

```bash
pip install 'foldjax[cuda13]'                    # [cuda12], or no extra for CPU
pip install 'foldjax[cuda13,alphafold3,openfold3-preprocess,templates]'
```

pip resolves from the ranges in `pyproject.toml`, not from `uv.lock`, so its
environment can differ from the validated one in unpinned packages. FoldJAX is
not on PyPI yet; until it is, install a release from GitHub:
`pip install 'foldjax[cuda13] @ git+https://github.com/eightmm/FoldJAX@v0.1.0'`.
In a pip install `foldjax doctor` prints `pip install 'foldjax[...]'` hints
rather than `uv sync` ones.

## Docker

The repository's `Dockerfile` builds a CUDA image from the frozen lockfile
(`uv sync --frozen --no-default-groups`); no weights are baked in. Mount a
FoldJAX store at `/foldjax`, which the image sets as `FOLDJAX_HOME`: weights,
chemistry assets, the compile cache and AlphaFold 3's generated runtime all
live there and survive the container.

```bash
docker build -t foldjax .                                   # CUDA 13
docker build -t foldjax:af3 --build-arg ALPHAFOLD3=1 .      # + AlphaFold 3
docker build -t foldjax:cu12 --build-arg CUDA_EXTRA=cuda12 \
    --build-arg CUDA_BASE=nvidia/cuda:12.9.1-base-ubuntu24.04 .

docker run --rm --gpus all -v "$HOME/.cache/foldjax:/foldjax" foldjax doctor
docker run --rm --gpus all -v "$HOME/.cache/foldjax:/foldjax" foldjax setup
docker run --rm --gpus all -v "$HOME/.cache/foldjax:/foldjax" -v "$PWD:/work" \
    foldjax predict --model boltz2 --input job.yaml --msa auto
```

Build arguments: `CUDA_EXTRA` (`cuda13`, default, or `cuda12`, with a matching
`CUDA_BASE`), `ALPHAFOLD3=1`, `OPENFOLD3_PREPROCESS` and `TEMPLATES` (both `1`
by default; `0` leaves the extra out). The working directory is `/work`.

The `ALPHAFOLD3=1` variant also installs a C++ toolchain, git and zlib headers:
AlphaFold 3 compiles its extension on first use, and CMake fetches abseil,
pybind11, libcifpp and dssp from GitHub while doing so. Do that once against
the mounted store, with network access, before the first prediction:

```bash
docker run --rm -v "$HOME/.cache/foldjax:/foldjax" foldjax:af3 \
    runtime prepare --model alphafold3
```

AlphaFold 3's parameters are placed in the store by hand, as outside a
container ([alphafold3.md](alphafold3.md)). To keep files in the store owned by
you rather than root, add `--user "$(id -u):$(id -g)"`.
