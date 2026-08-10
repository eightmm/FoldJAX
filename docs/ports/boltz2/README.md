![Boltz2-JAX hero: a protein-ligand complex in a luminous JAX-inspired energy field](docs/assets/boltz2-jax-hero.png)

# Boltz2-JAX

A JAX/XLA inference port of **[Boltz-2](https://github.com/jwohlwend/boltz)**,
the open-source biomolecular structure & affinity model. This project
reimplements the Boltz-2 inference graph (trunk, pairformer, MSA module,
diffusion sampler, and confidence/affinity heads) in pure JAX, loading the
original Boltz-2 PyTorch checkpoints unchanged.

It also ports the AlphaFold3-style efficiency path: fused attention/GLU kernels
via [tokamax](https://github.com/openxla/tokamax) (Triton), an optional custom
Pallas triangle-attention kernel, length-dependent chunking, and fp16/bf16
low-precision inference — selectable per backend without changing weights.

> Boltz-2 only. Boltz-1 checkpoints/features are not supported.

> [!IMPORTANT]
> Boltz2-JAX is an independent community port. It is not an official Boltz
> project. Model weights are not bundled in this repository; the setup script
> obtains them from the upstream Boltz community distribution.

## Status

- Inference only (no training).
- Weight-compatible: the same Boltz-2 checkpoints load and run; no checkpoint
  key/shape or feature ABI change.
- Data pipeline (raw input → features) runs without `import boltz` (vendored
  under `src/boltz_jax/data/`). Supported inputs: proteins, ligands (SMILES and
  CCD codes), templates (CIF/PDB), and MSAs from `msa: empty`, precomputed
  a3m/csv, or the colabfold MSA server.
- Affinity follows the complete upstream two-stage contract: rank the initial
  structure samples, crop the predicted receptor/ligand complex to the affinity
  limits, run the separate `boltz2_aff` trunk/diffusion checkpoint, and average
  both affinity ensemble members. The value, binary probability, and individual
  member outputs are returned.
- **Numerically faithful to torch Boltz-2.** With matched precision, seed, and
  injected diffusion noise (init + per-step churn + augmentation), boltz-jax fp32
  reproduces torch fp32 coordinates to **1.9e-4 Å RAW RMSD** — the XLA/cuBLAS
  floor. The trunk is bit-exact (corr 1.0) and a single denoiser step matches to
  1e-4 Å. A normal jax run differs from a torch run only by framework RNG (each
  draws its own diffusion noise), not by any model error.
- Performance depends strongly on sample count and kernel policy. In the
  production-size cached benchmark below (five samples), the Torch-compatible
  cuEquivariance JAX profile is **1.41× faster** and uses **2.25 GiB less peak
  allocation** than upstream torch. MSA is subsampled to 1024 by both
  frameworks, so it is not a cross-framework differentiator.
- MSA subsampling: boltz-jax takes the **first 1024** rows (deterministic);
  upstream torch takes a **random** 1024 (`randperm`, unseeded). Identical fold,
  but exact coordinates can't bit-match at MSA depth > 1024 by construction. Use
  `subsample_msa=False` or cap MSA ≤ 1024 for bit-level cross-framework tests.

## Quickstart

Two commands ([uv](https://docs.astral.sh/uv/) required):

```bash
# 1. One-time setup: install deps + download Boltz-2 weights/mols + convert
bash scripts/setup.sh

# 2. Predict: raw YAML -> structure
uv run python scripts/predict.py --input job.yaml --fmt cif
```

`scripts/setup.sh` auto-detects your CUDA major version from the driver
(override with `CUDA=cuda12 bash scripts/setup.sh`), runs `uv sync` with the GPU
+ torch-bridge extras, downloads the weights + molecule DB, and converts the
checkpoints. It is idempotent. Because the env is synced once with the extras,
`uv run python …` afterwards needs **no extra flags** — the GPU JAX plugin and
the torch-side featurizer are already in the environment.

`predict.py` defaults match Boltz-2 (`--steps 200 --recycling 3`, step scale
1.5, fp32) and read the setup paths, so step 2 needs no path flags.

All artifacts stay inside the project and are git-ignored: downloads + native
weights under `.cache/`, and predictions, compile cache, and feature cache under
`outputs/`. Nothing is written outside the repo.

## Inference

`scripts/predict.py` turns a raw YAML job into a structure file. Full form (all
defaults shown; override only what you need):

```bash
uv run python scripts/predict.py \
  --input job.yaml \
  --weights outputs/native_weights/boltz2_conf \
  --mols .cache/boltz/mols \
  --out-dir outputs/predictions \
  --fmt cif
```

Affinity jobs are identified by the upstream YAML `properties.affinity` entry.
The CLI automatically loads `boltz2_aff` next to `--weights`; override it with
`--affinity-weights`. Its upstream defaults are five diffusion samples, 200
steps, and five recycles; use `--diffusion-samples-affinity` and
`--sampling-steps-affinity` to override the first two. `--use-potentials`
enables the upstream FK/physical/contact steering configuration.

### Input examples

Protein + ligand (CCD code; SMILES also works via `smiles:`):

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
  - ligand:
      id: B
      ccd: ATP
```

With a structural template:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
      msa: empty
templates:
  - cif: path/to/template.cif
```

MSAs: set `msa: empty` (single sequence), point to a precomputed `.a3m`/`.csv`,
or omit `msa` and generate one from the colabfold server with `--use-msa-server`:

```bash
uv run python scripts/predict.py \
  --input job.yaml --use-msa-server --fmt cif
# [--msa-server-url https://api.colabfold.com] [--msa-pairing-strategy greedy|complete]
```

The MMseqs2 client is self-contained in `boltz_jax.data.msa`: it submits and
polls monomer or paired searches, downloads and safely extracts the result,
converts A3M blocks to Boltz CSV/NPZ features, and reuses completed archives.
Transient HTTP failures are retried and malformed responses, partial basic
credentials, corrupt archives, and unsafe tar entries fail explicitly.

Private-compatible servers may use either basic authentication or an API-key
header (the two methods are mutually exclusive). Prefer environment variables
so secrets do not appear in shell history or the process list:

```bash
BOLTZ_MSA_USERNAME=user BOLTZ_MSA_PASSWORD=secret \
  uv run python scripts/predict.py --input job.yaml --use-msa-server

MSA_API_KEY_VALUE=secret uv run python scripts/predict.py \
  --input job.yaml --use-msa-server --msa-api-key-header X-API-Key
```

The equivalent Python API arguments are `msa_server_username`,
`msa_server_password`, `msa_api_key_header`, and `msa_api_key_value`.
`BOLTZ_MSA_USERNAME`, `BOLTZ_MSA_PASSWORD`, and `MSA_API_KEY_VALUE` are used
when explicit values are absent. Authentication values are not written to the
feature cache. Public ColabFold database snapshots are controlled by that
service; retain the generated raw archive/output directory for provenance and
change or clear `--feature-cache` when a fresh search is required.

### Optimization knobs

The CUDA 13 install defaults both triangle kernels to Torch-compatible cuEq;
`use_scan` and the persistent compile cache remain enabled.

| Knob | Options | Notes |
|------|---------|-------|
| `--compute-dtype` | `bfloat16` (default) / `float32` | bf16-mixed — trunk bf16, diffusion and confidence an fp32 island — is what upstream's predict path runs (`main.py:1262`). At 1,003 tokens over an 8,192-row alignment, five samples of the released schedule: 65.0 s against 87.3 s warm, 10,218 MiB against 13,035 MiB peak, TM to the deposited structure 0.9931 against 0.9932. Reported pLDDT is ~0.002 lower, four times fp32's own run-to-run scatter and not matched by anything in the structures. fp16 is range-unstable in the sampler |
| `--diffusion-samples` | integer | generate multiple structures in one call; confidence runs sequentially when >1 to cap peak VRAM |
| `--compile-cache` | dir (default on) | persistent XLA cache; reuses compiles across runs |
| `--feature-cache` | dir (default on) | memoize features by input digest; cache hit is bit-identical and skips featurization |
| `--bucket` | flag (default off) | schema-aware token/atom/MSA bucketing for shared serving executables; MSA is capped at 1024 before JIT |
| `--prewarm-only` | flag | execute one input/profile to populate the persistent cache, then skip prediction-file writing |
| `matmul_precision` | `highest` (default) / `default` | `default` = TF32 (GPU). Unlike the other three ports, `highest` here **matches** upstream: `main.py:1096` is `set_float32_matmul_precision("highest")`, not the `"high"`/`enable_tf32` that OpenFold3, Protenix and OpenDDE select. Asking for TF32 is a divergence from Boltz-2, not a convergence on it |
| `attention_backend` | `xla` / `tokamax` | fused tokamax attention |
| `triangle_backend` | `xla` / `tokamax` / `pallas` / `cueq` | triangle-attention kernel |
| `--triangle-multiplication-backend` | `xla` / `cueq` | triangle-multiplication kernel |
| `glu_backend` | `xla` / `tokamax` | transition & triangle-mult GLU |

### Compilation-cache prewarming

The default lazy cache is sufficient for interactive use. A serving deployment
can prewarm only the shapes and static profiles it expects, on the same GPU and
software image that will serve requests:

```bash
uv run python scripts/predict.py \
  --input representative_768.yaml \
  --bucket \
  --compute-dtype bfloat16 \
  --compile-cache outputs/compile_cache \
  --prewarm-only
```

Repeat for required token/atom/MSA buckets and for profiles whose sample count,
steps, recycles, backends, confidence, guidance, or affinity mode differ. The
command executes once so GPU kernel autotuning and persistent XLA cache writes
finish, but it does not write final PDB/CIF predictions. Featurization/prep cache
artifacts may still be created. Set `JAX_LOG_COMPILES=1` when auditing whether a
serving call compiled a new executable.

MSA depth uses `1, 128, 256, 512, 768, 1024` buckets. The single-sequence case
stays at 1; padding every shallow/no-MSA target to 1024 would make MSA compute
and activation memory unnecessarily large. Inputs deeper than 1024 are
deterministically truncated before entering JIT, matching production inference.

The CUDA 13 extra installs NVIDIA cuEquivariance as part of the environment;
the CLI and top-level inference API select both cuEq triangle kernels by
default:

```bash
uv sync --extra cuda13
uv run --extra cuda13 python scripts/predict.py INPUT.yaml \
  --compute-dtype bfloat16 --diffusion-samples 5
```

The adapter consumes the mapped upstream Torch weights without changing their
meaning. At the production attention shape (952 tokens, BF16), JAX cuEq
attention was bit-identical to Torch cuEq; Pallas differed by 0.276% relative
RMSE. JAX cuEq triangle multiplication reduced relative RMSE against Torch from
0.724% (XLA) to 0.022%. On an RTX PRO 6000 Blackwell at integrin9 (952 tokens,
deep MSA, 200 steps, recycling 3, BF16-mixed, five samples), the full cuEq path
measured 43.26 s / 12.10 GiB warm versus upstream Torch 61.14 s / 14.41 GiB.
These are cached warm timings; re-measure on each target GPU and sequence-shape
distribution.

CPU/CUDA 12 installs do not include the CUDA 13 cuEq operations package; select
both XLA backends explicitly there. A missing or broken cuEq CUDA 13 install
raises instead of silently changing numerical kernels.

## Python API

After `setup.sh`, call inference from Python (heavy deps load lazily, so
`import boltz_jax` stays cheap):

```python
import boltz_jax

out = boltz_jax.predict(
    seq=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"],
    ligand_ccd=["ATP"],                 # also dna=, rna=, ligand_smiles=, or input="job.yaml"
    weights="outputs/native_weights/boltz2_conf",
    mols=".cache/boltz/mols",
    write_fmt="cif",                    # omit to skip writing a structure file
)
out["coords"]   # (n_atom, 3) np.ndarray;  out["plddt"], out["raw"], out["out_path"]
```

For multiple structures, set `diffusion_samples=N`; `coords` and `plddt` retain
the leading sample axis and structure writing returns `out_paths`. For affinity,
pass an upstream affinity YAML via `input=`. `affinity_weights` defaults to the
`boltz2_aff` sibling of `weights`, and `affinity_pred_value`,
`affinity_probability_binary`, plus the two member values/probabilities are
available in the result.

For composing with other JAX models, use the low-level pure-JAX function and the
weight loader directly:

```python
from boltz_jax import boltz2_predict, load_params, build_job_yaml  # all lazy
```

`boltz2_predict(params, feats, key, ...)` is a jit-friendly function over a
parameter + feature pytree; `boltz_jax.featurize(...)` produces `feats`.

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q  # 132 unit/parity/raw-input/E2E tests
uv run ruff check .
```

The suite covers both real checkpoints, all trunk/diffusion/confidence/affinity
blocks, both affinity ensemble members, steering potentials, cuEq dispatch,
templates, precomputed and live-compatible unpaired/paired MSA paths, MSA/cache
determinism, and raw protein/DNA/RNA/CCD/SMILES inputs.
See [the release gates](docs/RELEASE_GATES.md) for exact scope and GPU smokes.

## Attribution & license

Released under the [MIT License](LICENSE).

Derivative work of **Boltz** (© 2024 Jeremy Wohlwend, Gabriele Corso, Saro
Passaro et al.), used under the MIT License. The preprocessing/featurization
code under `src/boltz_jax/data/` is adapted from the Boltz repository; the
original copyright and license are retained in [`NOTICE`](NOTICE). Model weights
are distributed by the Boltz community under MIT.

If you use this work, please cite the Boltz-2 technical report
(https://doi.org/10.1101/2025.06.14.659707).
