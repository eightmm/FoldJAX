# Chai-JAX

Chai-JAX is a standalone, inference-only JAX port of
[Chai-1](https://github.com/chaidiscovery/chai-lab). It follows the same design
goal as `boltz_jax` and `protenix_jax`: convert the released model assets once,
then run preprocessing, folding, confidence scoring, ranking, and ModelCIF
output without importing PyTorch.

The functional pipeline and the tested scientific branches are implemented,
but the local technical release gate is still in hardening. The strict
multi-seed scientific matrix, a representative production-default Torch/JAX
comparison, and an installed-wheel native fold smoke remain open. See
[the completion gates](docs/COMPLETION_GATES.md),
[benchmark record](docs/BENCHMARKS.md), and
[scientific parity report](docs/SCIENTIFIC_PARITY.md) for the exact scope.

## Installation

Python 3.11 and 3.12 are supported by the package metadata. With `uv`:

```bash
# CPU development and portable tests
uv sync --extra dev

# NVIDIA CUDA 13; includes the optional cuEquivariance triangle-attention path
uv sync --extra cuda13 --extra dev

# NVIDIA CUDA 12; uses the XLA triangle-attention implementation
uv sync --extra cuda12 --extra dev
```

On CUDA 13, triangle attention selects cuEquivariance automatically when its
wheel is usable and otherwise falls back to the bounded XLA implementation.
Override selection before process startup with
`CHAI_JAX_TRIANGLE_ATTENTION_BACKEND=cueq` or `=xla`. The two choices compile
different executables, so benchmark and cache them separately. ESM attention
is independently selected with `xla` (default) or `cudnn`.

PyTorch and `antipickle` are conversion/test dependencies only:

```bash
# Keep the active accelerator extra in the same resolution.
uv sync --extra cuda13 --extra dev --extra torch-bridge
```

Running `uv sync --extra torch-bridge` alone may remove CUDA 13/cuEquivariance
packages from the environment. Substitute `cuda12` when that is the active
platform.

## Convert the released assets

Runtime requires a strict native bundle containing all six released Chai-1
components, a native reference-conformer archive, and—by default—the pinned
ESM2 t36 3B model. The installed exporters checksum every source and record
bundle provenance. Point the variables below at assets you obtained under the
upstream project's terms:

```bash
chai-jax assets bundle \
  --components-directory "$CHAI_COMPONENTS" \
  --output "$CHAI_JAX_CACHE/models/chai1" \
  --chai-source chai-lab/chai-1 \
  --chai-release v0.6.1

chai-jax assets conformers \
  --chai-source-directory "$CHAI_SOURCE" \
  --source "$CHAI_CONFORMERS" \
  --output "$CHAI_JAX_CACHE/conformers.npz"

chai-jax assets esm2 \
  --source "$CHAI_ESM2" \
  --output "$HOME/.cache/chai_jax/models/esm2_t36_3B"
```

`chai-jax assets --help` lists the commands. The aliases
`chai-jax-export-bundle`, `chai-jax-export-conformers`, and
`chai-jax-export-esm2` expose the same exporters. Conversion requires the
`torch-bridge` extra; normal inference does not.

## Input format

The FASTA header is `>TYPE|name=UNIQUE_NAME`, where `TYPE` is `protein`, `rna`,
`dna`, `ligand`, or `glycan`. Ligands use SMILES, modified polymer residues use
parenthesized CCD codes, and glycans use Chai's linkage notation:

```text
>protein|name=receptor
AC(SEP)DEG
>rna|name=rna_a
ACGU
>dna|name=dna_b
ACGT
>ligand|name=ligand_a
CS
>glycan|name=glycan_a
NAG(4-1 NAG)
```

Entity names must be unique. `--fasta-names-as-cif-chains` preserves them as
output chain identifiers; because Chai's model features encode chain IDs in
four ASCII bytes, names used with this option must be at most four ASCII
characters. Otherwise synthetic chain IDs are assigned and longer descriptive
entity names such as those above remain valid.

## Fold from the CLI

The public defaults match Chai's 3 trunk recycles, 200 diffusion timesteps,
5 diffusion samples, one trunk sample, and enabled ESM embeddings:

```bash
chai-jax fold input.fasta \
  --output-dir prediction \
  --bundle-path "$CHAI_JAX_CACHE/models/chai1" \
  --conformer-path "$CHAI_JAX_CACHE/conformers.npz" \
  --compilation-cache-dir "$CHAI_JAX_CACHE/xla"
```

Before JAX is imported, Chai-JAX defaults
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`. Explicit environment values always win.
The 2048-token cuEquivariance path is intended for a fresh serving process with
an already populated persistent compilation cache.

Set `--seed` for repeatable preprocessing, trunk MSA subsampling, and diffusion
within Chai-JAX. If it is omitted, one nonzero random seed is generated and used
for the complete call. Equal integer seeds do not make Torch and JAX draw the
same random tensors; scientific comparisons use the explicit matched random
tape described in [the parity report](docs/SCIENTIFIC_PARITY.md).

Useful modality options are:

- `--no-use-esm-embeddings` disables ESM. `--esm-embeddings-path` reads native
  precomputed embeddings; `--esm-model-path` and `--esm-cache-directory` run
  and content-cache the native ESM2 model.
- `--msa-directory DIR` reads Chai-compatible, SHA256-named
  `*.aligned.pqt` files. `--use-msa-server` queries the configured ColabFold
  endpoint and stores content-addressed results under
  `--msa-cache-directory` (or `OUTPUT_DIR/msas`).
- `--use-templates-server` enables templates returned by the MSA server. It
  requires `--use-msa-server`. For local templates, pass an m8 file with
  `--template-hits-path`, its structures with `--template-cif-directory`, and
  a writable `--template-cache-directory`. Native template NPZ input is also
  accepted. Raw m8/CIF alignment requires Kalign 3; select it with
  `--kalign-executable`.
- `--constraint-path restraints.csv` accepts Chai-compatible contact, pocket,
  and covalent rows. Docking constraints are currently a programmatic prepared
  feature rather than a public CSV row type.

`--recycle-msa-subsample`, `--num-trunk-recycles`,
`--num-diffn-timesteps`, `--num-diffn-samples`, and `--num-trunk-samples`
control inference. Run `chai-jax fold --help` for the complete interface.

## Python API

```python
from pathlib import Path

from chai_jax.inference import InferenceConfig, run_inference

cache = Path.home() / ".cache" / "chai_jax"
result = run_inference(
    "input.fasta",
    output_dir="prediction",
    bundle_path=cache / "models" / "chai1",
    conformer_path=cache / "conformers.npz",
    config=InferenceConfig(
        seed=0,
        use_msa_server=True,
        compilation_cache_dir=cache / "xla",
    ),
)

print(result.cif_paths)
print(result.pae.shape, result.pde.shape, result.plddt.shape)
```

Each trunk produces `pred.model_idx_N.cif` and `scores.model_idx_N.npz`; multiple
trunk samples use `trunk_N/` subdirectories. The score archives contain Chai's
aggregate score, pTM, ipTM, per-chain scores, and clash fields. The returned
`StructureCandidates` also contains unpadded PAE, PDE, pLDDT, and ranking data.
An active MSA additionally writes `msa_depth.pdf`. Output directories must be
new or empty so an existing prediction is never overwritten silently.

## Verification

The portable suite needs neither an upstream checkout nor official weights:

```bash
JAX_PLATFORMS=cpu uv run --extra dev pytest -q
```

Official Torch/JAX parity is deliberately opt-in:

```bash
CHAI_JAX_OFFICIAL_ASSET_DIR="$CHAI_COMPONENTS" \
CHAI_JAX_UPSTREAM_DIR="$CHAI_SOURCE" \
CHAI_JAX_UPSTREAM_PYTHON="$CHAI_SOURCE/.venv/bin/python" \
CHAI_JAX_CONFORMER_PATH="$CHAI_JAX_CACHE/conformers.npz" \
JAX_PLATFORMS=cpu uv run --extra cuda13 --extra dev --extra torch-bridge \
  pytest -q --run-official-parity
```

`CHAI_JAX_UPSTREAM_PYTHON` overrides the default Python resolved from
`CHAI_JAX_UPSTREAM_DIR`. Every ported model component retains an official
Torch-vs-JAX parity gate. With `--run-official-parity`, every required missing
or invalid configured path fails closed instead of silently skipping.

## Current evidence and limitations

The corrected official 256-token trunk has pair NRMSE `0.01566`, is `4.08x`
faster warm, and uses `37.2%` less peak accelerator memory than the released
TorchScript component on the recorded RTX PRO 6000 Blackwell run. Matched
200-step tests cover protein, RNA+DNA, protein-ligand, deep MSA, templates,
contact restraints, recycle-2, and a modified-residue/glycan/covalent complex.
The official confidence-head-to-score pipeline is also gated on a real
protein-ligand crop. Exact measurements and the single-seed/small-fixture
caveats are recorded in the linked benchmark documents.

Fresh-process XLA execution is verified through 1536 tokens. CUDA 13 with
cuEquivariance is verified through 1536 from a cold cache and at 2048 from a
populated persistent cache in a fresh serving process. The bounded XLA 2048
attempt did not complete within 610 seconds and approached 79 GB, so 2048 is
not claimed for XLA or for cold cache construction. The portable and official
test suites, wheel build, isolated install, and installed CLI help checks are
green; the installed-wheel native fold, strict multi-seed matrix, and a
representative production-default Torch comparison remain open.

GitHub CI, license/notice review, repository artwork, and publication metadata
are intentionally deferred until repository publication is authorized. They
are publication gates, not blockers for the local technical verdict.
