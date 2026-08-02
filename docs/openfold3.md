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

The port is at increment 1 of 9: primitives are ported behind a torch-parity
gate; the trunk and the diffusion sampler do not exist yet. FoldJAX carries the
adapter, the input translation, and their tests, so the backend is ready for the
port rather than the other way round. **No end-to-end OpenFold3 prediction has
been run through FoldJAX.**

OpenFold3 is nonetheless the highest-value remaining port: it is Apache-2.0 for
code *and* weights with published training data, and its model is the same AF3
architecture already ported twice in this repository.

## Weights

The weight store has no OpenFold3 entry, so `foldjax weights fetch --model
openfold3` reports that it has no managed assets. That is deliberate, not an
oversight — there is nothing to convert until the port can consume it. Pass the
checkpoint explicitly with `--weights`.

## Installing it anyway

With a local checkout:

```bash
uv pip install -e /path/to/openfold3_jax
```

The backend resolves the package by import, so nothing else needs configuring.
Selecting `--model openfold3` without it installed raises `ModuleNotFoundError`
naming `openfold3_jax`.
