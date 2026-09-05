# OpenDDE JAX

> **Archived port record.** OpenDDE now ships inside FoldJAX. Use the
> top-level [README](../../../README.md) for current installation and the
> common `foldjax` CLI. This file retains the port's model-specific evidence
> and native entry-point details.

Inference-only JAX port of
[Aureka AI Research OpenDDE](https://github.com/aurekaresearch/OpenDDE), pinned
to release `v1.1.1`, source revision
`ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`.

The native prediction path now runs an OpenDDE JSON job through
featurization, the released 655,791,538-parameter model, confidence scoring,
and CIF/summary writing without importing PyTorch or executing upstream
OpenDDE/Protenix. The optional one-time checkpoint conversion also uses
FoldJAX's restricted NumPy archive reader; PyTorch is not installed or imported.

The checked tiny protein example produces a finite 71-atom CIF and summary
through both the native CLI and the FoldJAX backend. A second actual-weight
smoke with the official `P4G` CCD component produces finite `[1, 31, 3]`
coordinates. A FoldJAX end-to-end smoke for atom-tokenized `SEP` writes a
finite 16-atom `SEP, ALA` CIF and the exact 16-key summary. These are CPU
wiring smokes with one recycle and one diffusion step, not full-schedule
quality claims. Tiny-example features match upstream OpenDDE exactly, and an
actual-checkpoint comparison using the same random tape measured raw coordinate
RMSE `0.000189 Å`, all-atom RMSD `0.000327 Å`, and maximum absolute error
`0.000977 Å`.

The current 1.1.1 closure adds the parts that example did not cover. The
released checkpoint completed protein, DNA, RNA, CCD ligand, SMILES ligand,
ion, and a mixed all-entity input; the ABAG checkpoint completed the mixed
input as well. Ten upstream/FoldJAX preprocessing cases, including RNA MSA and
the 9FM7 template, had all 60 shared arrays bit-identical per case. The template
tensors were nonzero and the mapped 9FM7 real-weight GPU run completed with
native Kalign 3.3.5. Finally, ABAG driven by the same upstream MSA and diffusion
tape measured `0.081714 Å` raw all-atom RMSD. The short two-step executions are
port gates rather than accuracy measurements; the released-schedule evidence
below remains the scientific runtime comparison.

The full 200-step, 10-cycle schedule is also verified for one FP32 sample on an
RTX PRO 6000 Blackwell Max-Q. Both pinned upstream OpenDDE and native JAX
completed the official protein/DNA `7R6R` job with the official weights, MSA
enabled, seed 101, finite scores, and no clashes. Detailed cold/warm timings,
memory, and score comparisons are below.

## Checkpoint provenance and conversion

Checkpoints are not bundled, and prediction never downloads them implicitly.
The explicit weight command downloads and verifies one selected checkpoint
from Hugging Face revision
`eddd563ce96571f784012edd8f045181c8f8627d`:

| Asset | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `opendde.pt` | 2,625,249,069 | `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc` |
| `opendde_abag.pt` (`abag` profile) | 2,625,271,509 | `5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d` |

A verified local conversion contains 655,791,538 parameters:

| Derived asset | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `opendde.jax` | 2,623,328,201 | `bc3e1400acb3b213ae9c466c518f6f14f4322d2913102aaa11fd56233cf409ee` |
| `opendde_abag.jax` | 2,623,328,201 | `c5eec319ed9c8f445c93bcce92d9aca82dc745d0f01b67f39c3bc85e31e4f5ca` |

After obtaining that asset from the
[official OpenDDE Hugging Face repository](https://huggingface.co/aurekaresearch/OpenDDE/tree/eddd563ce96571f784012edd8f045181c8f8627d),
verify it before conversion:

```bash
printf '%s  %s\n' \
  '7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc' \
  'opendde.pt' | sha256sum --check
```

The common managed-weight command downloads, verifies, and converts the pinned
publisher checkpoint with the NumPy reader:

```bash
foldjax weights fetch --model opendde

# Opt-in antibody-antigen checkpoint, stored separately from the default:
foldjax weights fetch --model opendde --profile abag
```

To convert a trusted local copy explicitly instead, use the native exporter;
no additional tensor-runtime extra is required:

```bash
uv run opendde-jax-export-weights \
  --checkpoint /path/to/opendde.pt \
  --out /path/to/opendde.jax
```

The reader accepts only the storage and reconstruction primitives needed by
the released archive and has no arbitrary-deserialization fallback. The
resulting native file is loaded by the same PyTorch-free prediction runtime.
Its loader accepts only NumPy arrays and the port's parameter NamedTuples;
arbitrary pickle globals are rejected. The actual 2.6 GB native archive loads
all 4,612 leaves and 655,791,538 parameters without importing `torch`.

## Chemistry and template assets

Prediction never downloads databases or runtime assets. The managed setup
commands stage shared public chemistry assets; native-entry-point callers may
also supply the assets a job needs explicitly:

| CLI option | Purpose |
| --- | --- |
| `--components-cif PATH` | official `components.cif` for exact arbitrary-CCD atom identity |
| `--ccd-rdkit-cache PATH` | trusted, hash-verified `components.cif.rdkit_mol.pkl` reference conformers |
| `--template-mmcif-dir DIR` | local PDB mmCIF files referenced by precomputed template hits |
| `--template-release-dates PATH` | official `release_date_cache.json` for the model cutoff |
| `--template-obsolete-map PATH` | official `obsolete_to_successor.json` mapping |
| `--kalign-binary PATH` | native Kalign 3.3.5 executable for the verified query-to-mmCIF template realignment path |

The RDKit cache is a pickle and must be treated as trusted input; use only the
pinned official SHA-256
`d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35`.
The loader snapshots the file, verifies that immutable byte string, and
deserializes that exact snapshot, so neither path replacement nor in-place file
mutation can change the bytes after verification. Kalign is an
external GPL-3+ preprocessing executable; it is version-gated and neither
linked into nor redistributed by this Apache-2.0 project.
Exact sizes, hashes, source URLs, the selected template records used in
validation, and the fixed `2021-09-30` template cutoff are recorded in
[`OFFICIAL_ASSETS.json`](OFFICIAL_ASSETS.json). These files are not
redistributed by this project.

## Prediction

Install the runtime and reproduce the short arbitrary-CCD CPU smoke:

```bash
uv sync
uv run opendde-jax-predict \
  --input-json examples/tiny_with_ccd.json \
  --weights /path/to/opendde.jax \
  --out outputs/tiny_with_ccd \
  --components-cif /data/common/components.cif \
  --ccd-rdkit-cache /data/common/components.cif.rdkit_mol.pkl \
  --n-sample 1 \
  --n-step 1 \
  --n-cycle 1 \
  --cpu-only
```

For a job containing precomputed template-hit A3M files, add:

```bash
  --template-mmcif-dir /data/pdb_mmcif \
  --template-release-dates /data/common/release_date_cache.json \
  --template-obsolete-map /data/common/obsolete_to_successor.json \
  --kalign-binary /usr/bin/kalign
```

The release defaults remain `n_sample=5`, `n_step=200`, and `n_cycle=10`.
The full 200-step, 10-cycle schedule has been checked on GPU both for the FP32
one-sample publisher-parity run below and for five-sample BF16 performance runs
at 490 and 970 tokens.

The CLI accepts only native JAX weights; passing a `.pt`, `.pth`, or `.ckpt`
file fails with a conversion instruction. `opendde_infer_static` remains the
lower-level entry point for an already featurized, unbatched example.

## Supported native input

- Standard protein, DNA, and RNA; arbitrary CCD ligands with exact official
  atom identity, plus SMILES ligands.
- OpenDDE-compatible modified polymers, including parity cases for `MSE`,
  `SEP`, `6MA`, and `5MC`, with their released residue/atom tokenization rules.
- Explicit standard CCD substitutions stay residue-tokenized, ligand `MSE` is
  normalized to `MET` identity while remaining atom-tokenized, and ambiguous
  leaving groups follow the upstream seed-dependent selection order.
- Entity copies and explicit covalent bonds, including `_struct_conn` records
  in output CIF.
- User-supplied protein/RNA MSA files, legacy precomputed-MSA directories, and
  precomputed template-hit A3M plus local mmCIF coordinates. JSON-relative
  paths resolve from the input file first. OpenDDE defaults both
  `use_rna_msa` and `use_template` to false; FoldJAX preserves those defaults
  and exposes the two paths only when the matching option is true.
- Exact template coordinate parity is pinned to the native Kalign 3.3.5 CLI.
  `kalign-py` 3.6.0 is executable through its different `--format` spelling,
  but produced a different alignment in the checked 9FM7 case. Upstream
  OpenDDE itself sends that wrapper the unsupported native `-format` spelling
  and yields no usable template rows. Therefore “3.3.5 or newer” is not a
  reproducibility claim; use the recorded 3.3.5 binary and hash for parity.
- OpenDDE's exact 16-key summary contract: `plddt`, `gpde`, `ptm`, `iptm`,
  `chain_gpde`, `chain_pair_gpde`, `chain_ptm`, `chain_iptm`,
  `chain_pair_iptm`, `chain_pair_iptm_global`, `chain_plddt`,
  `chain_pair_plddt`, `has_clash`, `disorder`, `ranking_score`, and
  `num_recycles`.

Model execution is JAX-only. Native preprocessing uses NumPy, Gemmi, and RDKit,
plus an external Kalign (3.3.5 or newer) only for template realignment, but
never PyTorch.
When the caller has not selected Protenix triangle kernels explicitly, the
OpenDDE model call uses dependency-free XLA triangle attention/multiplication
and restores the process environment afterward; the base runtime therefore
does not require cuEquivariance.
Ligands and ions are deliberately written as
standards-correct non-polymer entities with `HETATM` and an unset label
sequence ID; this fixes the pinned upstream writer's polymer representation
instead of copying that output bug.

## Verification

The current CPU verification gate is:

```bash
uv sync
JAX_PLATFORMS=cpu uv run pytest -q tests/models/opendde
uv run ruff check .
```

The final gate is intentionally expressed as commands rather than a fixed test
count, which changes as chemistry and intake cases are added. Actual-checkpoint
inference has been verified on CPU and on the single GPU described below.

The released-example preprocessing matrix can be reproduced with
`opendde-jax-verify-inputs` after setting the six asset environment variables
corresponding to the CLI options above. The checked matrix covers 8 JSON files
and 10 jobs: all features are finite and `torch_imported=false`. OpenDDE 1.1.1
performs a second sequence deduplication after Kalign realignment, so 5A9R is a
duplicate of 5A9S in 9FM7 and 1QPZ is a duplicate of 1JFT in
MGYP004658859411; each case has three coordinate-backed rows plus one zero
padding row, not four distinct templates. The current 9FM7 native-Kalign-3.3.5
capture has shape `[4,644,24,3]`, mask sums
`[13700,13472,13162,0]`, and all 60 shared FoldJAX/upstream arrays are bitwise
equal.

### Historical full-schedule GPU gate (OpenDDE 1.0.2, 2026-08-01)

This result predates the v1.1.1 source lift and is retained as a long-schedule
performance/scientific-behavior record, not as the current source-parity gate.
The real official `7R6R` input contains 245 residue tokens, 475 structural
tokens, 2,529 atoms, and protein plus two DNA chains. With MSA enabled, the
released fixed-depth, duplicate-preserving path assembled 703 rows from 702
unique sequences; the duplicate is retained to match OpenDDE rather than being
silently deduplicated. The official weights were run in FP32 with one sample
and seed 101. Both upstream and JAX completed a 20-step/2-cycle gate before
passing the full 200-step/10-cycle run.

| Full run | Internal/model (s) | Whole process (s) | GPU peak | CPU max RSS |
| --- | ---: | ---: | ---: | ---: |
| Upstream OpenDDE | 21.14 | 29.91 | 15,019.48 MiB stage peak | 4,386,912 KiB |
| JAX cold | 173.90 | 175.57 | 15,013,159,424 active bytes | 5,106,192 KiB |
| JAX warm | 49.58 | 51.17 | 15,003,035,648 active bytes | 3,608,556 KiB |

The GPU columns preserve the two runners' native measurement semantics, so the
upstream stage peak and JAX peak-active bytes should not be interpreted as a
more precise cross-runtime allocation comparison than they are.

| Full-run score | Upstream | JAX warm |
| --- | ---: | ---: |
| pLDDT | 85.2963 | 85.3314 |
| gPDE | 0.401731 | 0.400875 |
| pTM | 0.867095 | 0.868908 |
| iPTM | 0.920755 | 0.920760 |
| ranking | 0.910023 | 0.910390 |
| clash | false | false |

The serialized structures have the same 2,529 unique atom identities and all
coordinates are finite. After Kabsch alignment, the upstream/JAX-warm all-atom
RMSD is `2.267 Å`; independently aligned protein `CA` RMSD is `2.222 Å`, and
the two DNA-chain `C4'` RMSDs are `0.589 Å` and `0.495 Å`. JAX cold versus warm
all-atom RMSD is `0.0229 Å`.

This run also closes the large-input structural-role projection failure. The
old broadcast accidentally materialized `[475,475,384,384]` (about 124 GiB);
the implementation now evaluates the same projection row-wise with bounded
memory.

The diffusion-side structural relative-position projection is bounded too. Its
released 139-channel feature has four active values per token pair, so FoldJAX
gathers those four weight rows directly instead of creating the dense one-hot.
At 1,886 structural tokens this reduced XLA temporary bytes from 2,011,244,544
to 10,672,144; the 128-channel output allocation remains 1,821,181,952 bytes.
Full five-sample `t0488` and `t0976` runs stayed near 6,466 and 21,081 MiB
because the surrounding model, not this leaf, sets their process peaks.

Torch and JAX do not generate the same random draws from a shared numeric seed.
The full-run score and fold agreement therefore validate scientific behavior
and execution, but the `2.267 Å` value is a sample-to-sample comparison, **not**
a raw-coordinate numerical parity claim. The tiny shared-random-tape comparison
above remains the coordinate-level parity gate.

## Current limitations

This is a working inference port, but it is not a bundled data pipeline:

- database search, MSA/template generation, and asset downloads are not
  orchestrated; provide precomputed files and local databases explicitly;
- multi-device execution is available as `--cp-devices P` (context
  parallelism, the JAX form of upstream's Fold-CP: pair representations
  row-sharded across `P` local devices, XLA triangle kernels only). Numerical
  parity against the single-device program is verified on a simulated
  four-device CPU mesh; real multi-GPU memory and speed are not yet measured
  on this hardware;
- GPU validation covers one FP32 publisher-parity sample and five-sample BF16
  memory A/Bs on one RTX PRO 6000 Blackwell Max-Q. Other accelerators and
  multi-GPU scaling are not yet measured.

Missing assets and unsupported input contracts fail explicitly rather than
silently changing chemistry.

The source port retains the upstream Apache-2.0 attribution in `NOTICE`.
Checkpoint/common-asset licensing is not stated unambiguously in the pinned
Hugging Face tree, so verify redistribution rights separately; this project
does not redistribute those assets.
