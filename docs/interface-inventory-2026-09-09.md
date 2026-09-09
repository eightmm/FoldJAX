# FoldJAX interface inventory (2026-09-09, survey for the unification task)

Evidence-only inventory of entry points, duplicated helpers, the shared layer and its bypasses, pinning tests, and ranked unification candidates. Produced by a read-only survey agent on master; nothing was edited.

Evidence-based survey of user-facing and cross-port interface code, for unification and de-duplication. Read-only survey; nothing was edited.

Repository: `/home/jaemin/non-project/optimizing/foldjax`

**Scope correction: there are six ports, not five.** `alphafold3` has a full backend at `src/foldjax/backends/alphafold3.py:738` and shares most of the duplication patterns below. It is included throughout.

All paths below are relative to `/home/jaemin/non-project/optimizing/foldjax/`.

Sizes, for calibration:

| Layer | Lines |
|---|---|
| Shared `src/foldjax/*.py` | 14,529 |
| `src/foldjax/backends/*.py` | 6,690 |
| Shared `src/foldjax/models/_*.py` | 3,599 |
| Per-port `cli/` + `bridge/` | 8,779 |

---

## 1. Entry points

### 1.1 The dispatch asymmetry

The single most important structural fact is that the six backends dispatch three different ways.

**Route A, function API** (openfold3, boltz2, esmfold2)

- openfold3 imports port modules at `src/foldjax/backends/openfold3.py:383-390` and calls `inference.compile_predict` (`:637`), `inference.predict` (`:652`, `:662`), `output.write_prediction_outputs` (`:716`). Module docstring states the rationale at `backends/openfold3.py:1-12`: "this drives the port's Python API rather than its CLI."
- boltz2 calls `native.predict(**native_options)` at `backends/boltz2.py:745`, resolving through `_native_module()` (`:106-122`) to `src/foldjax/models/boltz2/api.py:480`.
- esmfold2 calls `inference.predict(...)` at `backends/esmfold2.py:1037` and `output_module.write_prediction_outputs(...)` at `:1097`.

**Route B, in-process argv into the port's own CLI** (protenix, opendde)

- protenix assembles argv at `backends/protenix.py:410-468`, imports the CLI module at `:470`, and calls `module.main(argv, ...)` at `:498-502` and `:510-518`. Not a subprocess.
- opendde assembles argv at `backends/opendde.py:263-300`, imports at `:303`, calls `native.main(argv, ...)` at `:327-330`, `:332`, `:344-349`, `:351-355`.
- Both negotiate a private kwarg through a capability flag `PREPARED_PARAMS_LOADER_API`, declared at `src/foldjax/models/protenix/cli/predict.py:25`, checked at `backends/protenix.py:471-473` and `backends/opendde.py:304-305`.

**Route C, module exec of upstream's absl runner** (alphafold3)

- `_runner_path(options)` at `backends/alphafold3.py:187` picks the vendored runner `VENDORED_RUNNER` (`:50`) or an explicit checkout; `_load_runner` at `:252-277` uses `spec.loader.exec_module(module)` (`:269`) with content-hash caching in `sys.modules` (`:260-262`).
- `_settle_absl_flags()` at `:318-333` then calls `flags.FLAGS.mark_as_parsed()` (`:333`) so absl does not reject the real process argv, which is `foldjax predict --model ...`.
- Upstream functions called: `runner.make_model_config` (`:1120`), `runner.ModelRunner` (`:958`), `model_runner.run_inference` (`:545`, `:617`), `runner.predict_structure` (`:1245`), `runner.write_outputs` (`:1249`).

**Cost of Route B.** `src/foldjax/models/opendde/cli/predict.py` writes seven process-global environment variables at `:647`, `:653-655`, `:660`, `:662`. Because the call is in-process, the backend must snapshot and restore them: `_EXPORTED_ENVIRONMENT` at `backends/opendde.py:28-36` and `_restored_environment` at `:386-403`. The test suite carries the same fixture independently at `tests/conftest.py:57` with the same seven names at `tests/conftest.py:41-49`.

**Consequence for planning: `models/protenix/cli/predict.py` (1,231 lines) and `models/opendde/cli/predict.py` (917 lines) are the live `foldjax predict` execution path, not legacy entry points.** Any refactor there is a production change, not a cleanup of dead code.

### 1.2 Console scripts

`pyproject.toml:171-186` registers one shared entry point and 13 per-port scripts:

| Script | Target |
|---|---|
| `foldjax` | `foldjax.cli:entrypoint` |
| `boltz-jax-export-weights` | `models.boltz2.bridge.export_weights:main` |
| `boltz-jax-inspect-checkpoint` | `models.boltz2.bridge.checkpoint:main` |
| `opendde-jax-export-weights` | `models.opendde.bridge.export_weights:main` |
| `opendde-jax-predict` | `models.opendde.cli.predict:main` |
| `opendde-jax-verify-inputs` | `models.opendde.cli.verify_inputs:main` |
| `openfold3-jax-featurize` | `models.openfold3.cli.featurize:entrypoint` |
| `openfold3-jax-inspect-checkpoint` | `models.openfold3.cli.inspect_checkpoint:entrypoint` |
| `openfold3-jax-predict` | `models.openfold3.cli.predict:entrypoint` |
| `openfold3-jax-verify-checkpoint` | `models.openfold3.cli.verify_checkpoint:entrypoint` |
| `protenix-jax-export-weights` | `models.protenix.bridge.export_weights:main` |
| `protenix-jax-featurize-json` | `models.protenix.data.featurize_json:main` |
| `protenix-jax-predict` | `models.protenix.cli.predict:main` |
| `protenix-jax-static-infer` | `models.protenix.cli.static_infer:main` |

**esmfold2 and alphafold3 register no console script.** Their only entry surface is the shared `foldjax` CLI and `foldjax.api`. esmfold2 has no `cli/` package at all.

### 1.3 Public Python API surface

Shared, `src/foldjax/__init__.py:16-63`: `predict`, `predict_batch`, `resolve_request`, `resolve_requests`, `detect_input_format`, `get_model`, `Model`, `ModelConfig`, `ExecutionConfig`, `available_models`, `capabilities`, `model_info`, `normalize_model_name`, `warm_cache`, plus the `PredictionRequest` / `PredictionResult` schema types and the `Job` builders.

Per-port:

| Port | Entry point | Config builder | Checkpoint load | Params map |
|---|---|---|---|---|
| openfold3 | `compile_predict` `models/openfold3/inference.py:1665`, `predict` `:668` | `released_config` `:1130` | `bridge/checkpoint.py:65` | `bridge/torch_mapping.py:942` |
| boltz2 | `predict` `models/boltz2/api.py:480`, `featurize` `:224`, `boltz2_predict` `models/predict.py:89` | none, defaults inline in `predict` | `bridge/native.py:208` | `bridge/native.py:208` |
| protenix | `main(argv)` `cli/predict.py:69`, `protenix_predict_static` `models/predict.py:24` | `runtime_policy.model_inference_defaults:58` | `bridge/weights_io.py:104` | `bridge/torch_mapping.py:1176` |
| opendde | `main(argv)` `cli/predict.py:416` | none | reuses protenix `weights_io.py:104` via `bridge/weights_io.py:10-15` | `bridge/weights_io.py:57` |
| esmfold2 | `load` `inference.py:93`, `predict` `:482`, `compiled_predict` `:738` | checkpoint `config.json` | `bridge/checkpoint.py:45` | `bridge/checkpoint.py:45` |
| alphafold3 | upstream `predict_structure` via `backends/alphafold3.py:1245` | `make_model_config` `:1120` | upstream lazy `params.get_model_haiku_params` | n/a, haiku native |

Naming is not unified: `compile_predict` / `compiled_predict` / none; `released_config` / `model_inference_defaults` / none; `load_checkpoint` / `load_params` / `load_native_weights`. Four of six ports have no `released_config` equivalent at all.

Re-export discipline differs sharply. `models/openfold3/__init__.py:44-58` exports 14 names. `models/boltz2/__init__.py:13-20` lazily exports 5. `models/opendde/__init__.py:5-88` exports about 30. `models/protenix/__init__.py:3-5` exports only `__version__`. `models/esmfold2/__init__.py` exports nothing, it is a docstring only. `models/alphafold3/__init__.py:11-16` exports two provenance constants.

### 1.4 Inputs

| Port | Native dialect | Parser |
|---|---|---|
| openfold3 | JSON spec, or `.npz` feature archive | `json.loads` at `backends/openfold3.py:785`; `data.load_feature_archive` `data/featurize.py:1037` |
| boltz2 | FASTA or YAML | `check_inputs` `data/preprocess.py:31`, `parse_fasta` / `parse_yaml` `:144-146` |
| protenix | JSON list, or `.npz` features | `json.load` at `cli/predict.py:657`; `load_static_feature_npz` `data/static_io.py:12` |
| opendde | JSON list | `load_jobs` `data/featurize_json.py:126` |
| esmfold2 | FoldJAX job document, JSON only | `_job_document` `backends/esmfold2.py:1124` |
| alphafold3 | upstream JSON | `folding_input.load_fold_inputs_from_path` at `backends/alphafold3.py:1065`, validated by `_validated_fold_jobs` `:695` |

All model-neutral dialect conversion happens once, upstream of the backends, in `src/foldjax/input.py`. `materialize_native_input` dispatches at `input.py:1405`; per-port translators are `_boltz` `:710`, `_protenix` `:804` (also serving opendde, `:1402`), `_openfold3` `:1220`. `read_job_document` at `input.py:1323` accepts JSON and YAML.

### 1.5 Checkpoint location and loading

**Already uniform.** All six receive `request.weights` pre-resolved by `assets.resolve_weights(backend.name, profile=asset_profile)` at `src/foldjax/api.py:157-163`. No backend resolves its own store path. Registry entries: openfold3 `assets.py:1627`, boltz2 `:1662`, opendde `:1432`, alphafold3 `:1379`.

alphafold3 differs only in that `assets.REGISTRY["alphafold3"].downloads = ()` at `assets.py:1391`, with `ready_check=_alphafold3_ready` at `:966`. Its parameters are placed manually under the publisher's terms, and `resolve_weights` raises pointing at manual-placement notes at `:2801-2808` rather than suggesting `foldjax weights fetch` as it does for the other five at `:2811-2819`.

Session-scoped weight reuse goes through the shared `PreparedWeightSession` at `backends/_weight_session.py:32` for openfold3, protenix and opendde. boltz2, esmfold2 and alphafold3 hand-roll it. See section 2.2.

### 1.6 Config and precision

**Matmul precision.** One shared option table `MATMUL_PRECISION_OPTION` at `backends/base.py:26-31`, merged into every backend's `execution_options`. Popped by `Backend.matmul_precision` at `base.py:107-123`, which returns `functools.partial(execution.matmul_precision_scope, value)`. The scope sets a ContextVar at `execution.py:193-219`; `resolved_matmul_precision` reads it at `:222-229`.

Only three ports read it inside model code:

| Port | Local default | Read site |
|---|---|---|
| openfold3 | `_MATMUL_PRECISION = "high"` `inference.py:386` | `inference.py:412-414` |
| protenix | `matmul_precision: str = "high"` `models/predict.py:89` | `models/predict.py:122-125` |
| boltz2 | `MATMUL_PRECISION = "highest"` `api.py:61` | `api.py:181-184` |

**opendde, esmfold2 and alphafold3 never call `resolved_matmul_precision`.** Grep across `src/` returns only the three sites above plus the comment at `base.py:23`. For those three ports the neutral knob takes effect only through the outer scope and is otherwise inert; by default no `jax.default_matmul_precision` scope is opened at all.

**Kernel and attention backend selection.** Six `execution_options` dicts, values genuinely per-port:

- `openfold3.py:215` — `triangle_kernel` to `{auto: cueq, cueq: cueq, cueq-full: cueq-full, xla: xla}`
- `boltz2.py:330` — `dtype`, `triangle_kernel` to `{auto: cueq, ...}`, `attention_kernel` to `{auto: xla, tokamax: tokamax, xla: xla}`
- `protenix.py:231` — `dtype`, `triangle_kernel` to `{auto: cueq_jit, cueq: cueq_jit, xla: xla_jit}`, `attention_kernel` to `{auto: xla_jit, xla: xla_jit}`
- `opendde.py:109` — `dtype`, `attention_kernel` only; no `triangle_kernel` by design, comment at `:106-108`
- `esmfold2.py:379` — matmul precision only; comment at `:390-392` explains the port's attention is XLA's
- `alphafold3.py:768` — `attention_kernel` to `{auto: triton, xla: xla}`

**Sampling knobs.** `sampling_options` maps neutral names to native spellings at `openfold3.py:201`, `boltz2.py:322`, `protenix.py:223`, `opendde.py:100`, `esmfold2.py:368`, `alphafold3.py:760`. Three backends override `apply_sampling` to inject a managed default: openfold3 adds one recycle at `:333-334`, boltz2 sets 5 at `:395-400`, esmfold2 sets 9 at `:736-741`, alphafold3 sets 3 at `:797-802`.

**Seed.** `PredictionRequest.seed` at `schema.py:381`, multi-seed fan-out at `schema.py:600`. Reaches the ports as `jax.random.PRNGKey`/`key` at `backends/openfold3.py:607`, `models/boltz2/api.py:826`, `models/protenix/cli/predict.py:986-989`, `models/opendde/cli/predict.py:310`, `models/esmfold2/inference.py:938`, `backends/alphafold3.py:703`.

### 1.7 Outputs

**Already uniform at the top.** No backend calls `foldjax.output.normalize`. It runs once for everyone at `api.py:1073`, relocating files into the canonical `seed-<seed>_sample-<nn>/` layout and writing `confidence.json` via `output._write_confidence` at `output.py:120-139`.

Per-port native writers:

| Port | Structure | Scores | Arrays |
|---|---|---|---|
| openfold3 | `output.write_structure` `models/openfold3/output.py:643` | `write_scores` `:493` | `write_arrays` `:594` |
| boltz2 | `write_prediction` `data/write/structure.py:21` | in-memory `_sample_scores` `backends/boltz2.py:143` | `models/_representations.save` |
| protenix | `_write_cif` `data/output.py:420` | `write_protenix_outputs` `:154-167` | `save_output_npz` `data/static_io.py:147` |
| opendde | reuses protenix `write_protenix_outputs` via `cli/predict.py:377-386` | same | same |
| esmfold2 | `to_mmcif` `data/pdb.py:136` via `output.py:227-237` | `json.dumps` `output.py:246-247` | `crop_prediction` `:156` |
| alphafold3 | upstream `write_outputs` `backends/alphafold3.py:1249` | upstream | upstream |

`models/esmfold2/data/pdb.py` defines real PDB writers `to_pdb:62` and `to_pdb_models:234`, but the live path calls `to_mmcif:136` exclusively. The PDB writers are reachable only from tests.

---

## 2. Duplicated helpers across ports

### 2.1 Byte-identical: the protenix and opendde session block

`backends/protenix.py:272-322` and `backends/opendde.py:125-175` are **51 byte-identical lines**, verified by `diff`. They cover `session`, `_ccd_memory_scope`, `invalidate_session`, `validate_session`, `observe_resumed`. Same lease name `"protenix_external_ccd"`, same release function `_release_external_ccd_cache`, same import from `models.protenix.data.featurize_json`.

`backends/esmfold2.py:449-465` is the same shape with two identifiers changed: lease name `"esmfold2_ccd"` and release function `_release_ccd_cache`.

```python
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        memory = ExitStack()
        try:
            with self._weights.session(requests):
                self._managed_memory = memory
                try:
                    yield self
                finally:
                    self._managed_memory = None
                    self._ccd_memory_leased = False
        finally:
            try:
                memory.close()
            except BaseException:
                pass
```

`backends/openfold3.py:233-247` is the reduced form, `session` without the memory ExitStack, then the same three hooks verbatim.

### 2.2 Same algorithm, three hand-rolled copies: weight anchoring

`PreparedWeightSession._anchor` at `backends/_weight_session.py:72-96` implements a sentinel-based anchor with poison-on-mismatch. Three backends reimplement it line for line:

| Site | Lines | Snapshot helper | Source-key helper |
|---|---|---|---|
| `_weight_session.py:72-96` (shared) | 25 | `_file_snapshot:15` | `str(path)` |
| `backends/boltz2.py:479-504` | 26 | `_weight_bundle_snapshot:31` | `_weight_source_key:74` |
| `backends/esmfold2.py:493-528` | 36 | `_model_asset_snapshot:105` | `_model_source_key:177` |
| `backends/alphafold3.py:879-908` | 30 | `_managed_asset_snapshot:91` | `_managed_source_key:127` |

All four share the identical control flow: `missing = object()` sentinel, `expected is missing` stores, `expected is None` degrades to unverifiable, mismatch calls `_poison`, then `require_verifiable and snapshot is None` calls `_poison` again. The only real divergence is the snapshot payload type: a multi-file tuple instead of a single-path triple.

`_poison` is byte-identical in three places and already exists shared at `_weight_session.py:66-70`:

```python
    def _poison(self, message: str) -> None:
        self.invalidate_session()
        self._session_poisoned = message
        raise PredictionError(message)
```

Sites: `backends/boltz2.py:459-462`, `backends/esmfold2.py:473-476`, `backends/alphafold3.py:864-867`. `_raise_if_poisoned` is identical at `boltz2.py:454-457` and `alphafold3.py:860-863`.

Total hand-rolled: roughly 290 lines across `boltz2.py:31-78,454-504`, `esmfold2.py:105-192,473-528`, `alphafold3.py:91-136,860-908`, against a 131-line shared module three other backends already use.

### 2.3 Same loop, five sites: the released-default strip

Every backend strips released defaults from its cache profile with the same six lines:

```python
for name, default in _RELEASED_COMPILE_DEFAULTS.items():
    if name not in profile:
        continue
    value = profile[name]
    if type(value) is type(default) and value == default:
        profile.pop(name)
```

| Site | Lines | Note |
|---|---|---|
| `backends/boltz2.py:423-431` | 9 | |
| `backends/protenix.py:349-356` | 8 | plus a second pass over `MODEL_INFERENCE_DEFAULTS` at `:359-372` |
| `backends/opendde.py:193-200` | 8 | |
| `backends/esmfold2.py:754-760` | 7 | over `_FIXED_COMPILE_DEFAULTS` |
| `backends/alphafold3.py:813-824` | 12 | plus a managed-route guard using `_MANAGED_CONFIG_DEFAULTS:361` |
| `backends/openfold3.py:270-279` | 10 | **the one variant**: coerces to `int` and keys on `options`, not `profile` |

The `bool` is an `int` subclass comment is repeated verbatim in four of them.

A second smaller repetition, the `cp_layout == "1d"` strip: `boltz2.py:432-433`, `protenix.py:377-379`, `opendde.py:202-204`.

The `_RELEASED_COMPILE_DEFAULTS` tables themselves are per-model value sets and are **not** duplication: `openfold3.py:83` has 4 keys, `boltz2.py:273` has 14, `protenix.py:123` has 8, `opendde.py:72` has 15, `esmfold2.py:72` has 3 under the name `_FIXED_COMPILE_DEFAULTS`, `alphafold3.py:351` has 8.

### 2.4 Byte-identical: the weight-export CLI

`models/protenix/bridge/export_weights.py:13-33` and `models/opendde/bridge/export_weights.py:15-35` have a **byte-identical 21-line `main`**, verified by `diff`. Only the module docstring and the import source differ.

```python
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument("--compress", dest="compress", action="store_true")
    compression.add_argument("--no-compress", dest="compress", action="store_false")
    parser.set_defaults(compress=True)
    args = parser.parse_args(argv)
    ...
```

`models/boltz2/bridge/export_weights.py:96-125` is the same shape with a different flag set: `--conf-ckpt`, `--aff-ckpt`, `--out-dir`, `--dtype`.

### 2.5 Byte-identical: the score reader

`backends/protenix.py:552-557` and `backends/opendde.py:407-412` are the **same six lines**, verified by `diff`:

```python
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )
```

`_CONFIDENCE_INFIX = "_summary_confidence_sample_"` is declared twice, `protenix.py:76` and `opendde.py:89`.

The shared helper `scalar_scores` at `src/foldjax/scores.py:15-31` is used by protenix, opendde and alphafold3 (`alphafold3.py:1316`). `backends/openfold3.py:798-809` reimplements JSON parsing instead, because its payload nests a `"samples"` list rather than being a flat scalar dict; no thin adapter over `scalar_scores` was written.

### 2.6 Duplicated argv rendering

Both Route-B backends walk `_CLI_OPTIONS` and emit dash-cased flags:

- `backends/protenix.py:439-447` — sorted walk, `f"--{key.replace('_', '-')}"`, with `_FLAG_OPTIONS` switch handling via `_render_switch:93`
- `backends/opendde.py:275-277` — sorted walk, same expression, no switch handling

`_CLI_OPTIONS` at `protenix.py:26-52` has 21 keys, at `opendde.py:37-64` has 26 keys. **15 keys are shared**: `cp_devices`, `cp_layout`, `num_samples`, `num_steps`, `num_recycles`, `max_msa_depth`, `trunk_dtype`, `diffusion_attention_backend`, `trunk_single_attention_backend`, `chunk_policy`, `triangle_mul_chunk_size`, `triangle_att_q_chunk_size`, `single_att_q_chunk_size`, `token_q_chunk_size`, `diffusion_chunk_size`.

### 2.7 Duplicated argparse flag sets

| Parser | Span | Flags |
|---|---|---|
| `models/protenix/cli/predict.py` | `:75-380`, 306 lines | 88 |
| `models/opendde/cli/predict.py` | `:423-594`, 172 lines | 41 |
| `models/openfold3/cli/predict.py` | `:24-154`, 131 lines | 18 |
| shared `src/foldjax/cli.py` `_add_predict_arguments` | `:35-293`, 259 lines | 49 |

**32 flag strings are shared by protenix and opendde**: `--chunk-policy`, `--compile-cache`, `--cp-devices`, `--cp-layout`, `--cpu-only`, `--diffusion-attention-backend`, `--diffusion-chunk-size`, `--input-json`, `--max-msa-depth`, `--max-msa-rows`, `--n-cycle`, `--n-keys`, `--no-graph-jit`, `--n-queries`, `--n-sample`, `--n-step`, `--num-recycles`, `--num-samples`, `--num-steps`, `--out`, `--representations`, `--representations-dir`, `--seed`, `--single-att-q-chunk-size`, `--stop-after`, `--template-mmcif-dir`, `--token-q-chunk-size`, `--triangle-att-q-chunk-size`, `--triangle-mul-chunk-size`, `--trunk-dtype`, `--trunk-single-attention-backend`, `--weights`.

6 are shared by all three port CLIs: `--cp-devices`, `--cp-layout`, `--representations`, `--representations-dir`, `--seed`, `--stop-after`.

The chunk-knob block is near-verbatim. `protenix/cli/predict.py:198-203` and `opendde/cli/predict.py:505-509` declare the same five `type=int` chunk flags; `--chunk-policy` and `--trunk-dtype` follow in both with the same `choices` tuples and different help text and defaults, `bf16` for protenix and `fp32` for opendde.

### 2.8 Two parallel profile protocols

`managed_asset_profile` and `apply_managed_profile` exist at `backends/protenix.py:135,143` (52 lines) and `backends/esmfold2.py:302,323` (57 lines). The bodies are semantically different: protenix maps a model-variant name and stages an ESM checkpoint directory; esmfold2 decides whether the 25 GB ESM-C bundle is required.

They are a candidate not for body merging but because **the shared layer reaches them by hard-coded backend name** in three separate blocks at `src/foldjax/api.py:127-175`:

```python
        if backend.name == "esmfold2":
            from foldjax.backends.esmfold2 import apply_managed_profile
            options = apply_managed_profile(options, requested_profile)
        elif backend.name == "protenix":
            from foldjax.backends.protenix import apply_managed_profile
            ...
    if backend.name == "esmfold2":
        ...
    elif backend.name == "protenix" and (...):
        ...
    if backend.name == "protenix" and asset_profile is not None:
        ...
```

### 2.9 Structurally different, not candidates

Listed so they are not re-surveyed:

- **`execution_options` dicts** (six sites, section 1.6). Translation tables whose values differ per port by construction. The same neutral `triangle_kernel` maps to `cueq`, `cueq_jit` and `xla`, and opendde omits it deliberately. `MATMUL_PRECISION_OPTION` is the one entry that was identical and is already shared at `base.py:26`.
- **`_RELEASED_COMPILE_DEFAULTS` tables**. Per-model values. The *loop* over them is the candidate; the tables are not.
- **`_padding_plan`**: `backends/openfold3.py:103-161` reads `token_mask`/`atom_mask`/`msa_mask` plus template compaction; `backends/esmfold2.py:210-300` reads `token_attention_mask`/`atom_attention_mask`/`msa_attention_mask` plus a 32-atom block constraint. Different keys, different axes.
- **`_shape_profile`**: `backends/opendde.py:415-431` guards on `padded` and keys `per_run`; `backends/alphafold3.py:568-575` has no guard and keys `per_job`; `backends/boltz2.py:254-266` `_padding_shape_profile` splits primary and affinity stages. Three different shapes.
- **`_openfold3_compile.py`** (484 lines) is port-specific code in the shared layer. Its docstring at `:1-7` gives the reason: importing the model submodule would execute the package `__init__` and pull in the JAX runtime, which the backend must avoid during cache planning and capability discovery. A deliberate inversion, not an accident.

---

## 3. The shared layer, and who bypasses it

### 3.1 What is already shared and universally used

| Module | Lines | Provides | Users |
|---|---|---|---|
| `schema.py` | 809 | `PredictionRequest`/`Result`/`Sample`, `ModelCapabilities`, seed fan-out `:600` | all six |
| `api.py` | 1,377 | `predict`, `resolve_request`, dispatch, output normalization `:1073` | all six |
| `cli.py` | 2,155 | the `foldjax` CLI, 49 predict flags | all six |
| `assets.py` | 2,840 | weight store, `resolve_weights:2792`, per-model registry | all six |
| `paths.py` | 97 | `foldjax_home`, `weights_dir`, `compile_cache_dir` | all six |
| `execution.py` | 229 | `KNOBS:56`, `ALIASES:66`, `matmul_precision_scope:193`, `resolved_matmul_precision:222` | all six via `base.py` |
| `backends/base.py` | 297 | `Backend` ABC, `apply_sampling:70`, `matmul_precision:107`, `validate_request:125`, `cache_profile:286` | all six |
| `padding.py` | 179 | buckets, `resolve_axis:77`, `PaddingPlan:147` | all six |
| `output.py` | 279 | `normalize:163`, `_write_confidence:120`, `safe_job_name:65` | shared layer, once |
| `cache.py` | 235 | `cache_namespace:85`, `compilation_cache_scope:195`, `_device_identity:116` | shared layer |
| `manifest.py` | 1,288 | `path_stat_identity:129`, `file_content_digest:187` | shared layer + 3 backends |
| `scores.py` | 31 | `scalar_scores:15` | protenix, opendde, alphafold3 |
| `input.py` | 1,409 | dialect translation, `read_job_document:1323` | all six |
| `backends/_weight_session.py` | 131 | `PreparedWeightSession` | openfold3, protenix, opendde |
| `models/_representations.py` | 462 | representation archive I/O | all six |
| `models/_random.py` | 55 | masked-prefix RNG | protenix, opendde |
| `torch_archive.py` | 275 | restricted checkpoint reader | openfold3, boltz2, protenix, opendde |

Matmul precision is the clearest existing success: one option table, one resolver, no per-port constant duplication in the option layer.

### 3.2 Real bypasses

1. **Compile cache is configured twice.** `api.py:1030-1034` already opens `cache.compilation_cache_scope(request.cache_dir)`, which takes `_CONFIG_LOCK`, calls `compilation_cache.reset_cache()` at `cache.py:219` and restores prior config in a `finally`. Inside that scope, two ports mutate JAX config again:
   - `models/openfold3/compilation.py:48-80` `enable_compilation_cache`, called from `backends/openfold3.py:605`
   - `models/protenix/cli/predict.py:451-456`, raw `jax.config.update` calls

   Neither restores on exit and neither calls `reset_cache()`, which the shared helper's docstring says is required because JAX constructs its file-cache object at most once. Both exist because the port CLIs must work standalone.

   Related: `models/openfold3/compilation.py:21-45` `default_cache_dir` reimplements a `foldjax_home()`-aware resolver with its own `OPENFOLD3_JAX_CACHE` env var, overlapping `paths.compile_cache_dir:62`. `protenix/cli/predict.py:331-335` defaults `--compile-cache` to `Path("outputs/compile_cache")` rather than the shared path, and `backends/protenix.py:422-431` works around this by forcing `--no-compile-cache`.

2. **Three job-name sanitizers, three policies.**

   | Site | Character class | Truncation |
   |---|---|---|
   | `output.py:65` `safe_job_name` | `[^\w.-]+` Unicode | SHA-based, limit 120 |
   | `models/openfold3/output.py:67` `_safe_output_name` | hard reject | none |
   | `models/protenix/data/output.py:190` `sanitize_job_name` | `[^A-Za-z0-9_.-]+` ASCII | none |

   The protenix one documents itself as intentionally matching upstream's ASCII naming. Flag as a decision to make, not an automatic merge.

3. **Three device-identity readers.** `cache.py:116-142` returns a dict from five JAX device attributes. `backends/alphafold3.py:139-150` returns a tuple from the same five. `models/boltz2/api.py:357-367` is a third. Same inputs, three shapes, three cache mechanisms.

4. **esmfold2 reads job documents itself.** `backends/esmfold2.py:1124` `_job_document` does `json.loads` only; `input.read_job_document:1323` accepts JSON and YAML. `_chains_from_document:1135` walks the raw dict where `job.Job.from_document:275` is the typed, validated reader. Neither shared function is imported in that file. **Unverified:** whether a YAML job actually reaches this reader or is materialized to JSON by `api.py` first.

5. **The AF3 parameter filename contract exists three times.** Upstream's `select_model_files`, FoldJAX's `assets.py:949-956` `_AF3_PARAMETER_PATTERNS`, and `backends/alphafold3.py:57-88` `_PARAMETER_PATTERNS` / `_selected_parameter_files`. Kept in sync by comment.

6. **Smaller ones.** `models/boltz2/api.py:1317` prints to stdout, bypassing `progress.py:44-88`. `models/boltz2/data/featurize.py:73-75` inlines chunked SHA-256 where `manifest.file_content_digest:187` exists. `models/openfold3/cli/predict.py:117-149` reads `peak_bytes_in_use` directly, duplicating `manifest.py:167` and `oom.py:118`. `backends/esmfold2.py:79` `_esmc_asset_paths` mirrors `models/esmfold2/bridge/esmc.py:39` `shard_paths`, with an admitting comment.

---

## 4. Tests that pin the current interfaces

**What runs.** CI is `.github/workflows/ci.yml`, job `check`: `pytest -q -m 'not network' --cov=foldjax --cov-report=term-missing --cov-fail-under=80` on `ubuntu-latest`, `JAX_PLATFORMS=cpu`, installed with `--no-default-groups --group dev --extra alphafold3 --extra openfold3-preprocess`.

**The only collection gate** is `tests/models/conftest.py:20-52`, which drops 24 boltz2 torch-parity files when `find_spec("torch")` is `None`. torch is in no install profile, so those 24 files never run in CI. A second job `imports` builds a wheel and smoke-tests `foldjax --help` and `foldjax models --json`.

`tests/conftest.py:25-40` registers `--run-official-parity`; tests marked `official_parity` skip inside without it.

Every file below **collects and runs** in CI. Counts verified by `pytest --collect-only`.

| Test file | Tests | Pins |
|---|---|---|
| `tests/test_backends.py` | 216 | session hooks, `_scores`, argv assembly, `invalidate_session` |
| `tests/test_api.py` | 123 | managed-profile dispatch, session lifecycle |
| `tests/test_assets.py` | 107 | profile resolution, `managed_asset_profile` |
| `tests/test_cache.py` | 96 | `cache_profile`, `compilation_cache_scope` |
| `tests/test_execution_knob_coverage.py` | 53 | knob translation across all six |
| `tests/models/protenix/test_cache_profile.py` | 51 | released-default strip, `_ccd_memory_scope` |
| `tests/test_esmfold2_backend.py` | 51 | anchoring, ccd scope, `_job_chains` |
| `tests/test_cli.py` | 46 | shared CLI flags, `--option` validation |
| `tests/test_cli_ergonomics.py` | 45 | shared CLI flag ergonomics |
| `tests/models/openfold3/test_predict_cli.py` | 44 | port CLI flags |
| `tests/models/protenix/test_padding.py` | 29 | port CLI argv, padding flags |
| `tests/models/protenix/test_static_infer_cli.py` | 27 | port CLI flags |
| `tests/test_output_layout.py` | 27 | `output.normalize` layout |
| `tests/models/opendde/test_padding.py` | 18 | port CLI argv |
| `tests/test_execution_vocabulary.py` | 11 | alias vocabulary |
| `tests/test_precision_policy.py` | 11 | precision resolution |
| `tests/models/opendde/test_predict_cli.py` | 9 | port CLI flags |
| `tests/test_weight_session.py` | 8 | `PreparedWeightSession` contract |
| `tests/test_managed_memory.py` | 8 | lease semantics |
| `tests/models/openfold3/test_verify_cli.py` | 7 | checkpoint CLI |
| `tests/models/openfold3/test_inspect_cli.py` | 6 | checkpoint CLI |
| `tests/models/openfold3/test_featurize_cli.py` | 4 | featurize CLI |
| `tests/models/opendde/test_cache_profile.py` | 3 | released-default strip |
| `tests/models/boltz2/test_predict_cli.py` | 1 | export CLI |

Additionally pinning the hand-rolled anchors: `tests/test_boltz2_session.py` (1,185 lines) and `tests/test_alphafold3_session.py` (1,001 lines).

Literal port-CLI flag strings also appear in `tests/test_backends.py:415,568`, `tests/models/protenix/test_output_feature_projection.py:253,340,454,510`, `tests/models/opendde/test_predict_cli.py:242,345`, `tests/models/opendde/test_parity_matched_tape.py:31,44`.

`tests/test_distribution.py` (698 lines) pins `pyproject.toml` itself, including that the `gpu` group and `cuda13` extra stay equal and that no published dependency set installs torch.

---

## 5. Ranked unification candidates

**Coverage constraint applying throughout.** `pyproject.toml` `[tool.coverage.run]` omits `*/foldjax/models/*`, `*/foldjax/search/*`, `*/foldjax/backends/_alphafold3_upstream/*`. So `src/foldjax/*.py` **and** `src/foldjax/backends/*.py` are both measured. Moving code from a backend into the shared layer is denominator-neutral. Lifting anything out of `models/*/cli/` or `models/*/bridge/` **adds uncovered measured lines** and can move the 80% gate. Candidates 6 and 8 grow the denominator; the rest do not.

### 1. The 51-line protenix and opendde session block

- **Removes:** 51 duplicated lines, one copy.
- **Files:** `src/foldjax/backends/protenix.py:272-322`, `src/foldjax/backends/opendde.py:125-175`. Consider folding in `src/foldjax/backends/esmfold2.py:449-465`, which differs only in the lease name and release callable, and `src/foldjax/backends/openfold3.py:233-247`, the reduced form.
- **Shape:** a `_CcdMemorySessionMixin` or two `Backend` hook defaults parameterized by `(lease_name, release_callable)`.
- **Gating tests:** `tests/test_backends.py` (216), `tests/test_managed_memory.py` (8), `tests/models/protenix/test_cache_profile.py` (51), `tests/models/opendde/test_cache_profile.py` (3), `tests/test_esmfold2_backend.py` (51).
- **Risk:** very low. `diff` proves the text is identical.
- **Coverage:** neutral.

### 2. Three hand-rolled anchors against `PreparedWeightSession`

- **Removes:** roughly 290 lines, of which about 90 are the anchor and poison bodies and about 200 are snapshot and source-key helpers that would become injected callables.
- **Files:** `src/foldjax/backends/boltz2.py:31-78,454-504`; `src/foldjax/backends/esmfold2.py:105-192,473-528`; `src/foldjax/backends/alphafold3.py:91-136,860-908`. Target: `src/foldjax/backends/_weight_session.py:66-100`.
- **Shape:** generalize `PreparedWeightSession` to accept a snapshot callable and a source-key callable, so a multi-file tuple works alongside the single-path triple.
- **Gating tests:** `tests/test_weight_session.py` (8), `tests/test_boltz2_session.py`, `tests/test_alphafold3_session.py`, `tests/test_esmfold2_backend.py` (51), `tests/test_backends.py` (216).
- **Risk:** low mechanically, but this is failure-isolation code. The poison paths and the unverifiable-source degradation need explicit coverage before the change, not after. `tests/test_weight_session.py` has only 8 tests for the shared implementation.
- **Coverage:** neutral.

### 3. The released-default strip loop at five sites

- **Removes:** about 44 duplicated lines, 35 in the main loop plus 9 in the `cp_layout` strip.
- **Files:** `backends/boltz2.py:423-433`, `backends/protenix.py:349-356,377-379`, `backends/opendde.py:193-204`, `backends/esmfold2.py:754-760`, `backends/alphafold3.py:813-824`. Leave `backends/openfold3.py:270-279` alone or migrate it last; it is the int-coercing variant keyed on `options`.
- **Shape:** a `Backend._strip_released_defaults(profile, defaults)` helper plus a `released_defaults` class attribute, preserving the exact-type check.
- **Gating tests:** `tests/models/protenix/test_cache_profile.py` (51), `tests/models/opendde/test_cache_profile.py` (3), `tests/test_cache.py` (96), `tests/test_alphafold3_session.py`, `tests/test_boltz2_session.py`, `tests/models/openfold3/test_stable_compile.py`.
- **Risk:** low to medium. Every behavior change here is cache-key-visible. A wrong strip silently splits or merges compile namespaces, which shows up as a recompile or a wrong-graph reuse, not as a test failure, unless the cache-profile suites cover the case.
- **Coverage:** neutral.

### 4. The managed-profile name dispatch at `api.py:127-175`

- **Removes:** 49 lines of `if backend.name == ...` from the shared layer. Small line count, high architectural value.
- **Files:** `src/foldjax/api.py:127-175`; `src/foldjax/backends/protenix.py:135,143`; `src/foldjax/backends/esmfold2.py:302,323`; `src/foldjax/backends/base.py`.
- **Shape:** two default no-op `Backend` methods, `managed_asset_profile(options)` and `apply_managed_profile(options, profile, *, weights=None)`. The two implementations stay per-port; only the dispatch collapses.
- **Gating tests:** `tests/test_api.py` (123), `tests/test_assets.py` (107), `tests/test_esmfold2_backend.py` (51).
- **Risk:** low. Note the protenix call is made twice, once before and once after weight resolution, at `api.py:130` and `api.py:169`, with different arguments. The protocol must preserve both call sites.
- **Coverage:** neutral.

### 5. The compile-cache double-set

- **Removes:** about 39 lines, and removes a correctness hazard rather than only duplication.
- **Files:** `src/foldjax/models/openfold3/compilation.py:21-80`, `src/foldjax/models/protenix/cli/predict.py:451-456`, `src/foldjax/backends/openfold3.py:605`, `src/foldjax/backends/protenix.py:422-431`. Target: `src/foldjax/cache.py:195-227`.
- **Shape:** both port CLIs call `cache.compilation_cache_scope` instead of mutating config. This is delete-and-redirect, not extract.
- **Gating tests:** `tests/test_cache.py` (96), `tests/models/openfold3/test_stable_compile.py`, `tests/models/protenix/test_static_infer_cli.py` (27).
- **Risk:** low under the managed path, medium for direct CLI callers. The standalone CLIs are the reason this code exists, so the redirect must keep them working without `foldjax.api`.
- **Coverage:** neutral for the backend edits; the `models/` deletions shrink an unmeasured tree.

### 6. The identical weight-export CLI

- **Removes:** 21 duplicated lines, one copy.
- **Files:** `src/foldjax/models/protenix/bridge/export_weights.py:13-33`, `src/foldjax/models/opendde/bridge/export_weights.py:15-35`. `src/foldjax/models/boltz2/bridge/export_weights.py:96-125` is the same shape with a different flag set and could adopt the same builder.
- **Shape:** one `weight_export_cli(load, save, description)` helper.
- **Gating tests:** none directly; `tests/test_distribution.py` (698) pins that both console scripts exist and resolve. Add a smoke test before changing.
- **Risk:** very low.
- **Coverage:** if the helper lands in `src/foldjax/`, it **adds roughly 25 measured lines** to the 80% denominator. Cover it or place it under `models/`.

### 7. `_scores` and `_CONFIDENCE_INFIX`

- **Removes:** 6 duplicated lines plus one duplicated constant.
- **Files:** `src/foldjax/backends/protenix.py:76,552-557`, `src/foldjax/backends/opendde.py:89,407-412`. Target: beside `src/foldjax/scores.py:15`, as something like `sample_summary_scores(structure_path)`.
- **Shape:** move the filename-derivation and the infix constant into `scores.py`.
- **Gating tests:** `tests/test_backends.py` (216), `tests/models/test_confidence_detail_elision.py`.
- **Risk:** very low.
- **Coverage:** neutral, `scores.py` is already measured.
- **Optional follow-on:** give `backends/openfold3.py:798-809` a thin adapter over the same helper for its nested-samples payload.

### 8. The protenix and opendde port parsers

- **Removes:** potentially 90 to 110 lines of overlapping flag declarations, plus the two argv render loops. Highest value and highest risk.
- **Files:** `src/foldjax/models/protenix/cli/predict.py:75-380` (306 lines, 88 flags), `src/foldjax/models/opendde/cli/predict.py:423-594` (172 lines, 41 flags), `src/foldjax/backends/protenix.py:26-52,439-447`, `src/foldjax/backends/opendde.py:37-64,275-277`. 32 flags and 15 option keys are shared.
- **Gating tests:** `tests/test_backends.py` (216), `tests/models/opendde/test_predict_cli.py` (9), `tests/models/protenix/test_static_infer_cli.py` (27), `tests/models/protenix/test_padding.py` (29), `tests/models/opendde/test_padding.py` (18), `tests/models/protenix/test_output_feature_projection.py`, `tests/models/opendde/test_parity_matched_tape.py`.
- **Risk:** high. These parsers are the live `foldjax predict` path for two of six models, not legacy code. They are also what forces the environment save-and-restore in both `backends/opendde.py:386-403` and `tests/conftest.py:57`.
- **Coverage:** grows the denominator if any shared builder lands in `src/foldjax/`.
- **Recommendation: split this.** Phase one is a shared flag-group builder for the 32 common flags, which is mechanical and testable against the existing CLI suites. Phase two, converting either port from `module.main(argv)` to a function API so the environment dance can be deleted, is separate work deserving its own plan and its own review.

---

## Open items

Two claims I could not close within this survey:

1. Whether a YAML job document actually reaches esmfold2's JSON-only reader at `backends/esmfold2.py:1124`, or is materialized to JSON by `api.py` first. If YAML does reach it, that is a user-visible inconsistency; if not, the duplication is cosmetic.
2. Whether the three job-name sanitizers (`output.py:65`, `models/openfold3/output.py:67`, `models/protenix/data/output.py:190`) diverge in practice on any real job name. They differ on Unicode input and on length, but no test exercises a name that separates them.

Nothing in the repository was edited.
