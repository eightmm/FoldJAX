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

The only supported checkpoint family is OpenFold3 `v0.5.0` OpenBind. Upstream
names `openbind-2025-06-30-174k` as the release default and publishes
`of3-ob-2025-06-30-174k.pt` from its `openfold3-data` bucket. FoldJAX pins its
2,287,872,989-byte size and SHA-256
`bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`
before staging it.

Because that key needs no account and no request, **`foldjax setup` fetches it
along with Boltz-2, OpenDDE and Protenix** — 2.29 GB. Fetching it on its own,
or asking where it landed, still works:

```bash
foldjax weights fetch --model openfold3
foldjax weights path --model openfold3
```

The mapper requires OpenBind's model version `[2, 0, 0]`, stack-level diffusion
pair norm, and corrected ending-node triangle bias. p1 and p2 checkpoints are
rejected with a compatibility error; there is no legacy fallback or silent
substitution. See the
[model version ledger](model-versions.md#openfold3) for the exact source commits,
checkpoint hashes, and support boundary.

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
uv sync --extra openfold3-preprocess
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
compile-relevant options. An eligible, readable entry can reuse that compilation
in a later process; a missing or rejected entry recompiles it.

The standalone `openfold3-jax-predict` command also enables the persistent
cache by default. Use `--cache-dir` to choose its location or `--no-cache` for
an intentionally ephemeral run.

Its raw-array output has an explicit 4 GiB default budget. Before compilation,
that same budget removes PAE, PDE, or distogram per-bin logits that the writer
would discard, so large unused arrays do not cross the JIT boundary. Pass
`--all-arrays` to disable both limits and retain all three distributions. This
does not include trunk representations, which remain controlled separately by
`--representations`. At the released five samples and 64 bins, the three pair
distributions total more than 10 GiB at 2,000 tokens, so the override can raise
device and host memory as well as output size substantially.

The default `per_sample_token_cutoff=0` runs the geometry-dependent confidence
Pairformer one diffusion sample at a time whenever `num_samples > 1`. This caps
sample-expanded pair state even for short targets. At 490 tokens and the
released five samples, a direct A/B moved the peak from 9,045.3 to 4,263.5 MiB
while wall time changed from 34.72 to 35.14 s. A positive numeric cutoff retains the
token threshold within the released five-sample width; wider requests retain
the existing memory guard. `None` selects an explicitly always-batched graph.
The schedule does not change the sampler or sample order;
the final exact-source A/B had minimum paired TM-score 0.99986 and maximum CA
RMSD 0.0924 Å, while confidence remained within the existing inference
tolerance.

The sampler itself stays batched, and at long targets that is what runs out of
memory. `--option diffusion_chunk_size=1` denoises the samples one at a time:
at 4,100 tokens the default route asks for a single 107.85 GiB block and fails,
where the chunked one completes the same five samples in 3,290.6 s at 76.4 GiB.
This is not free the way the confidence schedule is — chunking a free axis is
arithmetically the same prediction but not a bitwise one, measured at 9.5e-06
on coordinates of magnitude 10–30 — and 4,100 tokens is past anything either
implementation validates. What comes out says so itself: all five samples
reported a clash at a mean pLDDT near 42. Fitting is not the same as working.

The common `foldjax.predict()` backend additionally has a narrower managed
default: its `*_raw.npz` keeps coordinates, pLDDT, pTM, ipTM, chain-pair ipTM,
and the experimentally-resolved logits, but omits `pae_logits`, `pde_logits`,
and `distogram_logits`. Those quadratic native diagnostics are not fields of
`PredictionResult`; excluding them before tracing lets XLA remove the unused
PDE/distogram heads and avoids retaining their output buffers. With multiple
samples, serial confidence activates the existing compact PAE-metric route;
pTM, ipTM, and chain-pair ipTM keep its tested `max_abs <= 1e-3` contract.

The native pair distributions remain an explicit common-API opt-in with
`options={"all_arrays": True}` (or `--option all_arrays=true` on the common
CLI). Their retained output payload necessarily scales with `num_samples`, as do
any pair arrays retained under the standalone command's or direct API's existing
budgets, even though the confidence Pairformer's transient is serial across
multiple samples. The full-prediction opt-in has its own compilation-cache
identity. Both
output choices are irrelevant to `stop_after="trunk"`, which returns before
confidence and shares the historical trunk cache.

On the 490-token `t0488` GPU benchmark, serializing the released five-sample
confidence path reduced its peak from 9,045.3 to 4,263.5 MiB while wall time
changed from 34.72 to 35.14 s. Earlier provenance-limited measurements of the same bounded schedule
reduced the 10-sample peak from 16,674.3 to 4,263.5 MiB and the 20-sample peak
from 31,930.0 to 4,263.7 MiB, while wall time changed from 49.17 to 48.74 s and
from 76.30 to 75.67 s, respectively. In the final exact-source five-sample A/B,
matched structures had minimum TM-score 0.99986 and maximum CA RMSD 0.0924 Å;
maximum score changes were `7.736e-3` for mean pLDDT, `1.102e-4` for pTM, and
`2.203e-5` for ranking score. The earlier 10/20-sample A/Bs produced bitwise-identical
coordinates and maximum pLDDT/pTM changes of `1.668e-4`/`5.812e-6`. These
figures are XLA allocator live-byte peaks from separate warm and measured
processes, not the driver's reserved-memory reading.

The backend used to ignore that cache, which made it the one model least able to
afford doing so. It no longer does.

## State of the port

The whole latent trunk — Pairformer stack (AF3 Alg. 17), MSA module stack
(AF3 Alg. 8) and template pair stack (AF2 Alg. 16) — plus the diffusion sampler
and confidence heads are ported and gated against the real v0.5 upstream torch
modules. Layer gates use `rtol=atol=1e-4` on the full-precision CPU/XLA route;
the randomized composite atom-head sink uses `atol=3e-3` because a preceding
sub-`1e-4` activation difference is amplified by LayerNorm.
Parity is checked at three levels: per layer,
against upstream's own *composed* code (`run_trunk`, `DiffusionModule.forward`,
`SampleDiffusion.forward`, `AuxiliaryHeadsAllAtom.forward`), and `predict()`
against `OpenFold3.forward` as a single call. With the actual OpenBind checkpoint,
identical prepared 1BNA features and a shared random tape, the model-core
Kabsch RMSD is `5.78e-6 Å`. The shipped cuEquivariance GPU route has the
separate production envelope described below.

Every parameter group of the v0.5 model is read by the inference path, measured
by a test that fails if any non-alias tensor goes unread. The official checkpoint
contains 4,890 tensors; 740 are verified byte-identical sampler aliases before
they are pruned.

Checkpoint mapping is not conditional on one triangular-multiplication layout.
The released OpenBind config sets `fuse_projection_weights: false`, and a gate
records that unfused layout. The same composite mapper is also tested against an
upstream model with every stack fused: it splits each `2*c_hidden` projection
into the two parameters consumed by the forward pass, and both layouts produce
the same mapped shapes. `detect_fused_tri_mul` is therefore an inspection aid,
not an unsupported-checkpoint gate.

The checked-in benchmark table still contains historical OpenFold3 0.3.1/p1
rows with a 5 / 200 / 10 schedule. They remain real time/memory records rather
than being relabelled as v0.5. The current harness now drives the v0.5.0 worktree
and OpenBind file, so new measurements cannot accidentally extend that old
series. The historical rows have a hardware qualification: 0.3.1's accelerated
attention paths do not run on this host's sm_120 GPU, so the timing uses plain
Torch attention. Those historical wall times also include avoidable
template/CCD initialization, while the retained 3,012-token row explicitly
disables confidence-head host offload (a measured +2.2 GiB/-60 s harness
variant). Their confidence has a separate seed caveat: an audited CLI flag
replaced the requested 101 with a seed generated from 42; the current harness
is fixed and the old confidence cells are marked until re-measured.
The exact runner and environment are documented in
[`bench/README.md`](../bench/README.md) and
[`bench/upstream-environments.md`](../bench/upstream-environments.md).

Storage remains fp32 and prediction runs inside the shipped
`openfold3_precision` scope at matmul precision `high` (TF32), matching
upstream's `torch.set_float32_matmul_precision("high")` for ordinary matmuls.
The fused cuEquivariance triangle contractions explicitly pin
`lax.Precision.DEFAULT`, so the surrounding scope does not make every
accelerator contraction identical to upstream Torch. Pairformer accelerator
verification has two deliberately different gates on a small randomized
two-block stack. The strict mapping/algebra gate compares Torch and JAX at
`highest` through XLA triangle attention with normalized maximum error at most
`1e-5`; a three-seed production gate compares upstream `high` against FoldJAX
`high` with the shipped cuEquivariance route under a measured `5e-3` normalized
envelope. A device gate applies the strict XLA budget across CPU and GPU. Its
CPU-safe companion asserts that the production scope is exactly `high` and
restores the caller's setting when it exits. This is real-kernel Pairformer
coverage, not an end-to-end GPU parity claim.

On real 1UBQ, from the query sequence alone with no alignment and no template,
the five samples land 2.52–2.79 Å CA RMSD from the deposition. Real targets up
to 3,012 tokens have completed on the benchmark card; the 4,926-token case does
not fit, as recorded in the scaling table.

### What is not established

- **Structural accuracy beyond 1UBQ** is unmeasured — no RMSD or clash gate
  across a target set.
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
