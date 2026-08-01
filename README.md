<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.png">
  <img alt="FoldJAX — biomolecular structure prediction, compiled" src="docs/banner-dark.png">
</picture>

# FoldJAX

Biomolecular structure prediction in JAX. Five models behind one interface —
four carried complete inside the package:

| model | vendored at | licence | upstream |
|---|---|---|---|
| `boltz2` | `foldjax.models.boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `chai` | `foldjax.models.chai` | Apache-2.0 | [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

— plus `alphafold3`, which FoldJAX drives rather than carries, because its
parameters may not be redistributed. It takes the same job file as the rest
once installed; see [docs/alphafold3.md](docs/alphafold3.md).

Give it one job file and a model name. Weights, the input dialect, the output
directory, and the XLA compile cache are all resolved for you.

```bash
uv sync --extra cuda13 --extra torch-bridge --group dev
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

`torch-bridge` is only for that one conversion step: the upstream checkpoints
are torch files. **Prediction never imports torch for any model**, so once the
weights exist, `--extra cuda13` is the whole runtime.

```json
{
  "model": "boltz2",
  "output_dir": "foldjax-outputs/job",
  "samples": [
    {
      "seed": 0,
      "structure_path": "foldjax-outputs/job/boltz2_input.cif",
      "scores": {"mean_plddt": 0.9589, "iptm": 0.0}
    }
  ]
}
```

Every port keeps its own featurizer, sampler, checkpoint format, defaults, and
output writer, so results match running that project directly. FoldJAX supplies
the common request, the translation, and the plumbing.

## Input

One file, JSON or YAML, in the common schema:

```yaml
name: example
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    unpaired_msa: msa.a3m
    modifications:
      - {ccd: SEP, position: 3}
  - {type: ligand, id: [L], ccd: ATP}
bonds:
  - [[A, 3, OG], [L, 1, PA]]
```

`--input-format` defaults to `auto` and decides on content, not extension: only
the common schema is a mapping with `entities`, and every native dialect is
passed through untouched. So a Boltz YAML, a Protenix job list, or a Chai FASTA
all work as-is.

Modifications and covalent bonds have one model-neutral spelling here and are
translated into each backend's own names — Protenix and OpenDDE want
`CCD_`-prefixed types and entity-indexed `covalent_bonds`, Boltz wants
`{ccd, position}` and a `bond` constraint, AlphaFold 3 wants bare `ptmType`.

**A field a backend cannot express is rejected, not dropped.** Neither Boltz nor
Chai has a place for a separate `paired_msa`, and Chai addresses ligands by
SMILES only. Silently discarding an MSA would change the science without
changing the exit code.

`unpaired_msa` reaches every model, Chai included. Chai's own input is a FASTA
with nowhere to put an alignment path, so FoldJAX converts the A3M into the
`.aligned.pqt` form it reads and writes that beside the FASTA. That matters more
than it sounds: otherwise a job that pins an alignment pins it everywhere except
Chai, which quietly runs on a different one — and any comparison across models
becomes a comparison of MSAs.

## Choosing what to run

```bash
uv run foldjax predict \
  --model protenix \
  --input job.yaml \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10 \
  --seed 42
```

`--num-samples`, `--num-steps`, and `--num-recycles` are model-neutral. Each
backend calls them something different, and FoldJAX translates:

| neutral | alphafold3 | boltz2 | chai | opendde / protenix |
|---|---|---|---|---|
| `--num-samples` | `diffusion_samples` | `diffusion_samples` | `num_diffusion_samples` | `n_sample` |
| `--num-steps` | — | `steps` | `num_diffusion_timesteps` | `n_step` |
| `--num-recycles` | `recycles` | `recycling` | `num_trunk_recycles` | `n_cycle` |
| `--max-msa-depth` | — | `max_msa_depth` | — | `max_msa_rows` |

Setting both a neutral knob and its native name is an error rather than a silent
preference, and so is asking for a knob a backend does not have — AlphaFold 3
exposes no diffusion-step count, so `--num-steps` is refused for it rather than
quietly ignored. `--option KEY=VALUE` still passes anything native straight
through.

### Memory: `--max-msa-depth`

One knob dominates peak memory, and it is not the one you would guess. Every
trunk here holds an `[depth, tokens, channels]` MSA representation. On a
488-token job with a 13,254-row alignment that single tensor is 3,158 MiB, and
three are live at once — 9.5 GiB of Protenix's 12.2 GiB peak.

| protenix, 488 tokens | peak | warm | pLDDT | pTM |
|---|---|---|---|---|
| default (13,254 rows) | 12,264 MiB | 5.55 s | 75.1 | 0.749 |
| `--max-msa-depth 4096` | 6,008 MiB | 5.44 s | 70.3 | 0.856 |
| `--max-msa-depth 2048` | 5,991 MiB | 5.40 s | 76.6 | 0.902 |
| `--max-msa-depth 1024` | **5,986 MiB** | **5.40 s** | 79.9 | 0.915 |

Half the memory, marginally *faster*, and the cap barely matters below ~4096 —
once the MSA stops being the largest buffer, something else is, so there is no
reason to cut harder than you have to. Five samples instead of one costs 11%
more (5,986 → 6,666 MiB): the sample axis is already sequential.

Two things that sound like memory knobs and are not, both measured on the same
job: a bfloat16 trunk moved the peak by 0.3% (12,264 → 12,223 MiB), and
blocking the triangle attention moved it by 0.4% (→ 12,217 MiB). Neither is
where the memory is. **Attribute the peak before turning knobs** — these were
found by reading XLA's buffer assignment, not by guessing.

Confidence moves with the cap, but not monotonically (79.9 at 1024, 70.3 at
4096), because which rows get sampled changes. Treat that spread as the
alignment's own variance, not as evidence that a shallower MSA predicts better.
The default keeps each port's own depth; nothing changes unless you ask.

Chai has no reachable cap — it reads alignments through its own context objects
rather than a featurizer argument — so `foldjax capabilities --model chai` does
not report the knob rather than reporting one it cannot honour.

`foldjax plan` shows exactly what a request resolves to before anything loads:

```console
$ uv run foldjax plan --model protenix --input job.yaml --num-samples 2
{
  "cache_dir": "/home/you/.cache/foldjax/compile",
  "input_format": "foldjax",
  "model": "protenix",
  "output_dir": "foldjax-outputs/job",
  "sampling": {"num_samples": 2},
  "weights": "/home/you/.cache/foldjax/weights/protenix/protenix_base_default_v1.0.0.jax"
}
```

## Weights

Every upstream project publishes torch checkpoints; FoldJAX predicts from
torch-free JAX weights. `foldjax weights` owns that gap — it downloads from each
publisher, verifies against the published hash where one exists, converts once,
and stores the result under the FoldJAX home.

```bash
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
```

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or $XDG_CACHE_HOME/foldjax
├── downloads/<model>/     # released files, exactly as published
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # converted JAX weights — what prediction loads
└── compile/               # XLA persistent compilation cache
```

Conversion needs torch, which is why it is a separate step behind
`--extra torch-bridge`. Prediction never imports it.

No weights are redistributed. Each file is downloaded from its own publisher
under that project's terms, and nothing is fetched implicitly during prediction —
a missing model names the command that would get it.

Chai's conformer archive is an `antipickle` file; upstream reads it by
importing `chai_lab` for its private adapters, which needs torch and pins an
rdkit this environment cannot hold. FoldJAX decodes the format directly
instead, so `--model chai` is one command like the rest.

Weight files exported before these ports were vendored recorded
`protenix_jax.*` / `opendde_jax.*` as their module names. The loader maps those
onto the vendored classes, so **existing weight files keep working**, and the
restricted unpickler still rejects arbitrary globals.

## GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

Measured on an RTX PRO 6000 Blackwell Max-Q, same 35-residue job, one sample,
20 diffusion steps, one recycle. Upstream is each model's own repository on
torch, given the same input and settings.

| model | FoldJAX | of which model | upstream | FoldJAX peak | upstream peak | programs |
|---|---|---|---|---|---|---|
| `boltz2` | **10.4 s** | 4.4 s | 19.7 s | **2183 MiB** | 2019 MiB | 140 |
| `opendde` | 13.0 s | **1.8 s** | 9.1 s | **2731 MiB** | 5915 MiB* | 238 |
| `chai` | 20.4 s | 5.2 s | 28.4 s | **1646 MiB** | 8065 MiB* | 523 |
| `protenix` | **12.1 s** | 3.0 s | 49.4 s | **1832 MiB** | 3663 MiB* | 159 |

**Read the memory column carefully, and do not read it off `nvidia-smi`.** That
tool reports what the CUDA allocator has *reserved*, not what the program is
using, and XLA's BFC allocator grows by doubling: three of these models parked
on exactly 4094 MiB of reserved pool no matter whether they needed 1.6 GB or
4.0 GB of it. The column above is `peak_bytes_in_use` from
`jax.local_devices()[0].memory_stats()`, which is the real high-water mark;
`nvidia-smi` sits 0.9-2.6 GB above it. The upstream column is
`torch.cuda.max_memory_allocated()`, the same quantity. Starred entries are
still `nvidia-smi` peaks, so they flatter FoldJAX by whatever torch's caching
allocator over-reserved; `boltz2` is the one measured both ways on both sides.

Two things made the difference. **Layer weights were being copied inside the
traced graph**: every port stacks its per-layer parameters onto one axis so
`lax.scan` can walk them, and doing that with `jnp.stack` inside `jit` makes
XLA emit a `concatenate` over the weights -- 2,290 MiB of them in Boltz-2's
optimized HLO, for a 35-residue job. Stacking on the host at load time instead
(`foldjax.models._stacking`) cut the temp arena from 845 to 118 MiB for
`boltz2` and from 1,438 to 88 MiB for `opendde`, and left the numbers
unchanged. **And the eager ports were dispatching the model op by op**: OpenDDE
once ran 1,824 programs per prediction, Protenix 666. Tracing each whole graph
once -- `opendde_infer_compiled`, `protenix_infer_compiled` -- collapsed those
to 238 and 159. For Protenix that is 30.9 s down to 12.1 s, with the predicted
structure identical to the op-by-op path within its own run-to-run noise
(0.0753 A RMSD against a 0.0738 A floor between two identical eager runs).
`--no-graph-jit` restores the op-by-op path on either model.

The "of which model" column is the second prediction in the same process, so it
excludes weight loading, featurization, and reading the compiled program back
from the cache. Those one-off costs dominate a single short job -- OpenDDE
spends 1.8 s in the model and about 12 s getting ready to -- which is worth
knowing before optimizing the wrong thing. Chai has the most room left: 523
programs and the slowest steady state of the four.

Confidence agrees with upstream where the sampler is converged enough to
compare — OpenDDE pLDDT 92.75 vs 93.16, pTM 0.819 vs 0.828, gPDE 0.354 vs 0.357;
Chai's aggregate score 0.16815 vs 0.1681. These are still *different samples*:
torch and JAX PRNG streams differ, so the diffusion noise differs, and at a
20-step schedule that moves the numbers. Same-seed numerical parity was
established separately per port with a matched random tape.

Two caveats on the upstream side: Protenix's fused CUDA layer-norm needs
`ninja` to JIT-compile, so it ran with `LAYERNORM_TYPE=torch`; and OpenDDE pins
`torch==2.7.1+cu126`, which has no kernels for this GPU and fails with "no
kernel image is available" while its runner still reports success, so it was
upgraded to `torch==2.12.0+cu130` to run at all.

`--no-cache` turns it off for benchmark or ephemeral runs; `--cache-dir` moves
the root. Every model also runs on CPU, slowly.

## Python API

```python
from foldjax import PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="chai",
        input="job.yaml",
        num_samples=5,
        seed=42,
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

`foldjax.resolve_request` applies every default without running anything, and
`foldjax.capabilities(model)` reports the input formats, entity types, and
sampling knobs a backend honours. Backend-specific output stays on
`PredictionResult.raw`.

## Environment

One Python 3.12 environment, no sibling checkouts:

- JAX / CUDA 13 plugin 0.10.1, cuequivariance-jax 0.10.0 with CUDA 13 ops 0.9.0,
  Tokamax 0.0.12

```bash
uv sync --extra cuda13 --group dev                       # predict with any model
uv sync --extra cuda13 --extra torch-bridge --group dev  # + convert checkpoints
```

**The prediction environment is torch-free for every model.** Boltz-2's
featurizer is vendored from upstream Boltz and written against torch; it runs
here on a NumPy array layer (`foldjax.models.boltz2.data._numpy_torch`) that
reproduces the torch featurizer exactly. `--extra boltz-preprocess` installs
real torch so the two can be compared, which is what
`tests/models/boltz2/test_featurize_torch_parity.py` does.

Two further backends stay external because they cannot be vendored.

**AlphaFold 3** is fully wired and runs from the same job file as the other
four — `foldjax predict --model alphafold3 --input job.yaml` — but FoldJAX
drives an installation you provide rather than carrying one. Its parameters are
not redistributable, and upstream pins `jax[cuda12]`, which cannot co-resolve
with a CUDA 13 cluster; the right JAX plugin differs per site. It is therefore
deliberately absent from the dependency set.
[docs/alphafold3.md](docs/alphafold3.md) has the install for either CUDA
generation, where the weights go, and what changes on a device whose shared
memory the upstream Triton kernels do not fit.

**`openfold3-jax`** is an in-progress sibling port, registered and enabled by
`--extra openfold3`.

## Tests

```bash
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q
uv run --extra cuda13 --group dev ruff check .
```

1,020 tests pass with the optional extras installed, 85% coverage on the
orchestration layer. Each vendored port brought its own suite to
`tests/models/<name>/`, including the torch-parity gates it was built against.
Those are excluded from collection unless the extra they need is installed — so
the torch-free environment runs the subset that does not need one — and
`tests/models/test_optional_suite_gate.py` keeps that exclusion list honest
rather than letting a suite go quietly uncollected.

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Model weights are covered by their own licences and are never redistributed
here. AlphaFold 3 goes further — its parameters may not be redistributed at
all, so it is the one backend FoldJAX drives rather than carries. See
[docs/alphafold3.md](docs/alphafold3.md).
