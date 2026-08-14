# OpenFold3

OpenFold3 is carried inside FoldJAX, at `foldjax.models.openfold3`; no sibling
checkout is needed. Raw jobs use FoldJAX's Torch-free NumPy/JAX featurizer:

```bash
uv sync --extra openfold3-preprocess
foldjax predict --model openfold3 --input job.yaml
```

The same Torch-free NumPy/JAX featurizer is also available as a split step when
you want to reuse an archive and its embedded chemistry table:

```bash
openfold3-jax-featurize openfold3-query.json -o features.npz
foldjax predict --model openfold3 --input features.npz \
  --input-format openfold3-features
```

`openfold3-query.json` is OpenFold3's native upstream `{"queries": ...}`
document, not FoldJAX's common job schema. Common JSON/YAML jobs use the
one-shot `foldjax predict` command above; the split featurizer intentionally
exposes the exact upstream preprocessing contract.

The distinction is available before running anything:

```bash
foldjax capabilities --model openfold3
foldjax models --json
```

Both JSON views expose `input_requirements` per accepted format. For
`openfold3-features`, it reports `preprocessing_runtime: "precomputed"`, no
required extra, and `requires_torch: false`. The raw `native`, `openfold3`, and
`foldjax` formats report `preprocessing_runtime: "jax"`, the non-Torch
`openfold3-preprocess` chemistry extra, and `requires_torch: false`.

It was external until it was vendored, and this file used to explain how to
install it separately. That is no longer necessary — the reason it was excluded
was that `openfold3-jax` is published on no package index, which vendoring
removes. AlphaFold 3 source is vendored too; only its separately licensed
parameters remain external, while its isolated first-use build avoids
upstream's environment pin. See [alphafold3.md](alphafold3.md).

## What it costs to add

Nothing, at the dependency level. The inference path needs `jax`, `numpy`,
`safetensors` and `gemmi`, all of which the carried model stack already required,
so vendoring OpenFold3 added no base dependency.

## Weights

The published checkpoint is `of3_ft3_v1.pt`, and the code *and* weights are
Apache-2.0 with the training data published — this is the most permissively
licensed model FoldJAX carries. The **repository is access-gated**, though: even
its example config returns HTTP 401 unauthenticated, so FoldJAX cannot fetch it
for you.

```bash
# Request access at https://huggingface.co/OpenFold/OpenFold3, then:
store=$(foldjax home | python -c 'import json,sys; print(json.load(sys.stdin)["weights"])')
mkdir -p "$store/openfold3"                         # `foldjax home` prints the rest
cp of3_ft3_v1.pt "$store/openfold3/"
foldjax weights list                                # openfold3 now reads `ready`
```

`foldjax weights fetch --model openfold3` reports that the parameters are
released only on request and names the directory to put them in, rather than
failing with a bare 401.

Nothing is converted: the port reads the released checkpoint directly with
FoldJAX's restricted torch-archive reader. It reconstructs `.pt`/`.ckpt`
tensors as NumPy arrays without importing PyTorch; a `safetensors` export can
also be passed to `--weights`.

## Torch-free featurization

FoldJAX owns the one-query orchestration and array construction used for
OpenFold3 prediction. It reuses upstream's non-tensor chemistry, structure and
MSA parsing routines, but does not import its Dataset, Lightning DataModule, or
PyTorch tensor feature layer. Reference conformer augmentation is driven by the
request seed, so repeated featurization is reproducible.

Until 2026-08-06 that meant a second repository: building features required a
sibling checkout that `uv sync` could not provide. The required data subset is
vendored now:

```
src/foldjax/models/openfold3/_upstream/openfold3/
├── core/data/        # chemistry, structure, MSA, and template data helpers
├── core/config/      # MSA/data settings
└── projects/of3_all_atom/config/  # inference query schema
```

The 51-module subset remains recognisably upstream and stays excluded from
automatic reformatting so it can be reviewed against its Apache-2.0 source,
but it is not byte-for-byte identical. FoldJAX carries a small compatibility
patch set: a data-only package root, a plain-array/pickle-free MSA cache reader,
and local NumPy/data helper imports. Model, training, Lightning, and tensor
feature modules are not shipped. FoldJAX-owned orchestration and NumPy feature
construction live outside this tree.

Raw featurization still needs its chemistry/data extra:

```bash
uv sync --extra cuda13 --extra openfold3-preprocess
```

The extra contains no training framework or second tensor runtime. `deepspeed`,
PyTorch, Lightning and TorchMetrics are deliberately absent from every
FoldJAX install profile.

### Pickle-free preprocessing caches

FoldJAX exposes public writers for reusable MSA and template inputs. Both
formats contain only numeric or Unicode arrays and are always read with
`allow_pickle=False`:

```python
import numpy as np

from foldjax.models.openfold3.data import (
    save_preparsed_msas,
    save_template_cache,
)

msa_cache = save_preparsed_msas(
    {
        "colabfold_main": {
            "msa": ["ACDE", "AAAA"],
            "deletion_matrix": np.zeros((2, 4), dtype=np.int32),
            "metadata": ["query", "hit"],
        }
    },
    "query-msas.npz",
)

template_cache = save_template_cache(
    {
        "1ubq_A": {
            "idx_map": [[1, 1], [2, 2], [3, 3]],
            "release_date": "1987-01-02",
            "cif_path": "structures/1ubq.cif",
        }
    },
    "template_cache/query.npz",
)
```

Relative `cif_path` values are resolved from the template cache directory. If
`cif_path` is omitted, the loader expects
`<cache-parent-parent>/template_structures/<entry_id>.cif`. Requested template
IDs are considered in order: unknown, duplicated, malformed, or locally
missing candidates are warned about and skipped until four valid entries have
been selected; accepted entries are reindexed from zero. A request for which no
entry is valid raises a detailed error, and FoldJAX never downloads a CIF.

Legacy upstream object-valued NPZ files require Python pickle and are therefore
rejected. To migrate trusted legacy data, extract its primitive arrays in an
isolated conversion environment and pass those values to the writers above;
the production FoldJAX process never needs to enable pickle.

### What the vendoring was checked against

The port keeps a feature-level parity gate against the publisher implementation
for development. Production tests additionally block every `torch` import and
exercise raw structure, MSA and template inputs through the FoldJAX path.

The test suite's `conftest.py` locates an optional external checkout only for
torch-parity gates, which compare this port against upstream's own *model*
modules. Those are the thing being ported, so carrying a copy would mean diffing
the port against a copy of its own source. Production code has no checkout
locator, and parity tests skip when their external reference environment is
absent.

## Sampling knobs

`--num-samples`, `--num-steps` and `--num-recycles` all reach it, translated to
`num_samples`, `no_rollout_steps` and `num_cycles`. `--max-msa-depth` narrows the
released `msa_depth` setting and subsamples on the host before device transfer.

Backend options: `query_id`, `ccd_file_path`, `pair_chunk_size`, `prefix`,
`no_compile`, and the three native sampling spellings. `ccd_file_path` applies
only while preprocessing raw input; a feature archive already contains fixed
chemistry and rejects that option rather than ignoring it.

`pair_chunk_size` is worth knowing about. Triangle attention's scores are
`[rows, heads, N, N]` — cubic in token count — so one *unchunked* pair block
needs 267 GiB of temporaries at 2076 tokens against 41 GiB at 256 rows.
`released_config` picks a chunk from the token count, and because the chunk
shrinks as the target grows, peak memory rises more slowly than the square of
the token count: 966 to 1494 tokens is 1.55x the tokens and only 1.44x the peak.

## Alignments are selected by filename

OpenFold3 identifies an alignment's source from the file's **stem** and ignores
a name it does not recognise, and the stem also chooses the row cap:
`colabfold_main` keeps 16,384 rows, `colabfold_paired` 8,192, `uniref90_hits`
10,000, `mgnify_hits` 5,000 (`dataset_config_components.py:80`).

A job that names its alignment after the target — `t0128.a3m`, which is what
every other backend here accepts — would therefore be silently dropped. The
translation links it into the generated inputs directory under an accepted stem
instead, one directory per chain so two chains cannot claim the same name:

```
<output>/inputs/msa/A/colabfold_main.a3m -> /path/to/t0128.a3m
```

An alignment that already carries a recognised stem is passed through untouched,
since upstream then knows its row cap. `tests/test_input_openfold3.py` covers
both, plus the two-chain collision and the suffix.

## The compile cache

Compiling this architecture is the dominant cost and grows with length — minutes
at a few hundred tokens, where XLA prints its own slow-compile warning. FoldJAX's
persistent cache is on by default and namespaced per model, weight identity and
compile-relevant options, so it is paid once per shape rather than once per
process.

The standalone `openfold3-jax-predict` command also enables the persistent
cache by default. Use `--cache-dir` to choose its location or `--no-cache` for
an intentionally ephemeral run.

The backend used to ignore that cache, which made it the one model least able to
afford doing so. It no longer does.

## State of the port

The whole latent trunk — Pairformer stack (AF3 Alg. 17), MSA module stack
(AF3 Alg. 8) and template pair stack (AF2 Alg. 16) — plus the diffusion sampler
and confidence heads are ported and gated against the real upstream torch
modules at `rtol=atol=1e-4`. Parity is checked at three levels: per layer,
against upstream's own *composed* code (`run_trunk`, `DiffusionModule.forward`,
`SampleDiffusion.forward`, `AuxiliaryHeadsAllAtom.forward`), and `predict()`
against `OpenFold3.forward` as a single call. At the released settings the worst
disagreement is 2.1e-5 on coordinates whose maximum is 35.0.

Every parameter of the released model is read by the inference path — 416 of 416
unique tensors — measured by a test that fails if any group goes unread, which is
how three unwired subsystems were originally found.

On real 1UBQ, from the query sequence alone with no alignment and no template,
the five samples land 2.52–2.79 Å CA RMSD from the deposition. Real targets up
to 2076 tokens run.

### What is not established

- **No comparison against upstream OpenFold3 as a program.** `bench/spec.py`
  lists `openfold3` in `MODELS` but not in `REIMPLEMENTED`, because
  `run_upstream.command` has no branch for it and no upstream virtualenv has
  been provisioned here. The FoldJAX side can be measured today; the column that
  would make it a comparison cannot.
- **Structural accuracy beyond 1UBQ** is unmeasured — no RMSD or clash gate
  across a target set.
- **Precision** is fp32 storage with matmul precision pinned to `high` (TF32),
  which is what upstream sets for itself. Upstream also runs `bf16-mixed` in
  places, which is numerically different by design.
- **Upstream's memory-optimization and kernel paths** (chunked forward,
  `forward_offload`, DeepSpeed/cuEq/Triton/LMA) are not ported. Upstream asserts
  they are numerically equivalent; that is upstream's claim, not a measurement
  here.
- **MSA row choice above 1024 rows** cannot be reproduced without an explicit
  permutation, because upstream draws it with `torch.randperm`.
- **Batch size** is always 1. Nothing assumes it; nothing measures it either.
- **Template corpus coverage.** The local Torch-free path is gated on the
  supported raw template formats, but broad structural accuracy still needs a
  larger deposition set than the current smoke targets.
