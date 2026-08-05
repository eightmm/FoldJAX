# OpenFold3

OpenFold3 is registered as a FoldJAX backend but is **not** installed by
`uv sync`, and its port is **not finished**. Read this before selecting it.

## Why it is not a dependency

`openfold3-jax` is published on no package index. Listing it in `pyproject.toml`
— even behind an optional extra — makes `uv sync` fail for everyone who does not
already have the checkout, because uv resolves every extra to build the
lockfile. That is exactly what happened: the extra pointed at a sibling
directory (`../openfold3_jax`) that existed only on the machine it was
developed on, so the first documented command in the README failed on every
fresh clone.

AlphaFold 3 is excluded for a related but different reason (see
[alphafold3.md](alphafold3.md)): its parameters are not redistributable and
upstream pins an incompatible CUDA generation.

## State of the port

`openfold3_jax`'s own `PROJECT.md` says "milestones 1-5 of 9 ... nothing runs
end to end yet, and no checkpoint has been loaded". That is out of date. What
was verified here, by running it:

| step | result |
|---|---|
| `openfold3_jax` imports | all six modules the adapter needs |
| every symbol the adapter calls | present, with matching signatures |
| `load_checkpoint(of3_ft3_v1.pt)` | 4,945 tensors |
| `map_inference_params` | returns `InferenceParams` |
| FoldJAX job -> OpenFold3 query | translated, alignment linked (see below) |
| `featurize_query` | 132 tokens, 1,013 atoms, MSA `(1, 2373, 132, 32)` |
| `foldjax.predict(model="openfold3")` | one `.cif` written, with confidence scores |

So the port is far further along than its own notes claim: it runs end to end.

What that does **not** establish is that it runs *correctly*. The first
end-to-end run used a deliberately short schedule -- 1 recycle, 20 steps, 1
sample -- and returned `mean_plddt 0.400`, `ptm 0.305`. That is low, and a
schedule that short cannot distinguish a hard target from a wrong weight
mapping: `t0128` is an unconverged membrane protein on which Boltz-2's own
samples share TM 0.06. Nothing here has been compared against upstream
OpenFold3, and the model is still absent from `bench/spec.py::MODELS`.

Read the table as "the path is connected", not "the port is validated".

OpenFold3 is the highest-value remaining port: Apache-2.0 for code *and*
weights with published training data, on the same AF3 architecture already
ported twice in this repository.

## Weights

The weight store has no OpenFold3 entry, so `foldjax weights fetch --model
openfold3` reports that it has no managed assets. Pass the checkpoint
explicitly with `--weights`; `bridge.checkpoint.load_checkpoint` reads the
released `.pt` directly.

## Installing it

Two checkouts, because featurization and inference have different dependencies
on purpose: inference needs only JAX and a checkpoint, while featurization runs
on upstream OpenFold3's data stack.

```bash
uv pip install --no-deps -e /path/to/openfold3_jax
```

`--no-deps` matters: `openfold3-jax` pins its own JAX and cuEquivariance
versions, and resolving them would fight the ones FoldJAX already has pinned.

Featurization additionally imports upstream `openfold3`, which pulls a chain of
packages that are not FoldJAX dependencies:

```bash
uv pip install --no-deps biotite lmdb func_timeout click pdbeccdutils \
    memory_profiler boto3 botocore s3transfer jmespath kalign-python
```

`kalign-python` is the distribution that provides `import kalign`; installing
`kalign` fails, since no such project exists on PyPI. Upstream's own
`pyproject.toml` is the authority for any name not obvious from the import.

None of these are declared as a FoldJAX extra, for the same reason
`openfold3-jax` is not: uv resolves every extra when building the lockfile, so
naming a package that is only needed by one optional backend makes `uv sync`
slower and more fragile for everyone else. The backend resolves its package by
import, and selecting `--model openfold3` without it raises `ModuleNotFoundError`
naming `openfold3_jax`.

## Alignments are selected by filename

OpenFold3 identifies an alignment's source from the file's **stem** and ignores
a name it does not recognise, and the stem also chooses the row cap:
`colabfold_main` keeps 16,384 rows, `colabfold_paired` 8,192, `uniref90_hits`
10,000, `mgnify_hits` 5,000 (`dataset_config_components.py:80`).

A job that names its alignment after the target — `t0128.a3m`, which is what
every other backend here accepts — would therefore be silently dropped. The
translation links it into the generated inputs directory under an accepted
stem instead, one directory per chain so two chains cannot claim the same name:

```
<output>/inputs/msa/A/colabfold_main.a3m -> /path/to/t0128.a3m
```

An alignment that already carries a recognised stem is passed through
untouched, since upstream then knows its row cap. `tests/test_openfold3_input.py`
covers both, plus the two-chain collision and the suffix.
