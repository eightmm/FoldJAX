# Common model interface

FoldJAX exposes common jobs, inference choices and execution stages. Native
features, parameter trees, atom indices and intermediate trunk state stay inside
each backend. A common representation name describes its role, not an
interchangeable embedding space across models.

```python
from foldjax import ExecutionConfig, ModelConfig, get_model

model = get_model(
    "alphafold3",
    config=ModelConfig(msa_depth=1024, trunk_passes=4),
    execution=ExecutionConfig(padding=True, use_compile_cache=True),
)
request = model.plan("job.json", stage="inputs")  # validate without inference
result = model.embed("job.json")
inputs = result.representations["single_inputs"]
trunk = model.encode("job.json", outputs=("single", "pair"))
prediction = model.predict("job.json")
```

The same calls support `alphafold3`, `boltz2`, `protenix`, `opendde`,
`openfold3` and `esmfold2`. Existing `Job` objects are accepted as well as paths.
Relative MSA/template references inside a Python `Job` use `base_dir` (current
working directory by default); file inputs retain document-relative references.
Planning stores Python jobs in the managed job store and validates paths.

## Public and private boundaries

| Public interface | Responsibility |
| --- | --- |
| `Job` | Sequences, chemistry, raw MSA and template references |
| `ModelConfig` | `msa_depth`, total `trunk_passes`, `samples`, `steps`, `msa_search` |
| `ExecutionConfig` | Padding, persistent compile cache location/use, resume |
| `model.capabilities` | Supported inputs, sampling bindings and representations |
| `embed` | Native input representation, zero trunk passes |
| `encode` | Trunk representations, before structure sampling |
| `predict` | Structure prediction and optional representations |
| `PredictionResult` | Common artifacts and lazy representation access |

`encode` returns only `single` by default; request `pair` explicitly because its
storage scales quadratically with tokens. Input and trunk stages return no
structure samples. Representation archives record shape, dtype, axes and native
space; they do not define a universal chain/residue mapping or feature format.

`get_model` constructs a lazy configuration handle without loading weights.
`with_config(config=..., execution=...)` returns a new handle. Backends own weight
loading and runtime lifetimes; this API does not promise that successive calls
retain one loaded checkpoint or reuse results from an earlier stage. Input-only
execution may still prepare native features and load a whole parameter tree.
ESMFold2's input-only path skips the language model.

`native_options` is an explicit advanced escape hatch, validated by the backend.
Ordinary callers do not construct native feature dictionaries. Unsupported
settings fail instead of silently becoming common settings.

## Defaults and counting

Unset `ModelConfig` fields retain the selected **FoldJAX model/checkpoint**
defaults, including the agreed [recycling policy](recycling-defaults.md).
They do not restore every publisher CLI default. `trunk_passes=P` always requests
P total main-trunk evaluations, including the initial evaluation:

| Backend | Compatible `PredictionRequest.num_recycles` |
| --- | --- |
| AF3, Boltz2, ESMFold2, OpenFold3 | P - 1 |
| Protenix, OpenDDE | P |

OpenFold3's adapter then translates the common additional-recycle count to its
native total count. A single total pass is supported. Input extraction executes
zero trunk passes regardless of the configured prediction schedule.

The existing `predict(PredictionRequest(...))` API remains available with its
established counting and padding policy. The new handle writes
`model_config.json` alongside the existing run manifest and exposes the same
record through `result.configuration`. It records requested values and adapter
bindings. `adapter_value` means a translated option, which may include FoldJAX
policy defaults; `backend_default` means the value is resolved inside the backend
or checkpoint, and is not claimed to have been inspected here. The run manifest
remains the authority for resume validation.

## MSA selection and execution capacity

Reuse the same raw MSA files through `Job`; each model retains its pairing,
alphabet, profile calculation and row selection. `msa_depth` names the native
depth control, not a guarantee of identical selected rows or identical tensors.

| Model | Native depth behavior |
| --- | --- |
| AF3 | Model MSA crop; padded preprocessing also has a feature capacity |
| Boltz2 | Candidate-row cap; trunk subsampling has a separate limit |
| Protenix | Paired/unpaired assembly cap; profile statistics precede cropping |
| OpenDDE | Candidate-row cap; padded cycle capacity is separate |
| OpenFold3 | Per-cycle cap, bounded by the adapter at 1024; host union may be larger |
| ESMFold2 | Per-cycle sampling cap when the checkpoint uses an MSA encoder; profiles remain model-native |

The new handle requires an explicit `ModelConfig(msa_depth=...)` when enabling
padding, so execution configuration does not silently choose scientific input
depth. Use 1024 for the agreed common profile, or 1280 for OpenDDE. This requirement
does not imply that changing padding is scientifically neutral for every native
preprocessing pipeline. See [padding profiles](token-padding-profiles.md) for
capacity and masking behavior. The legacy request API retains its existing
implicit serving-depth defaults.

## Input representation semantics

All six backends advertise `single_inputs` through `input_representations`.
Only these input-stage names are accepted by `embed`; `all` selects this subset.

- AF3: assembled target representation including learned atom encoding.
- Boltz2, Protenix, OpenDDE and OpenFold3: each model's native input embedding
  before the main trunk. OpenDDE's later `structural_single_inputs` is excluded.
- ESMFold2: native `x_inputs`, before LM-conditioned pair initialization; this is
  not an ESMC embedding.

The carried AF3 runtime supports these stages. An external AF3 runtime does not.
An AF3 native document containing multiple jobs cannot request one common
representation archive; submit separate jobs through the batch API instead.

Default output paths are `foldjax-outputs/<model>/<job-stem>/<stage>`.
Representation files are validated on resume, including input-only runs.
Compilation remains backend-owned and uses existing persistent cache handling;
this interface is not a cross-model executable cache.

## Validation boundary

CPU tests cover common option translation, early-return boundaries, archive
selection/cropping, resume and model-specific synthetic execution. Real-checkpoint
GPU accuracy, speed and memory measurements are separate acceptance gates; the
new interface alone makes no such performance or parity claim.

Validation on 2026-09-08: the complete CPU run reported 5,727 passed, 413
skipped and four failures. The failures were an ESMFold2 static-argument ordering
mismatch, a concurrently updated MSA test double, and two test expectations for
the former recycling defaults. After correction, all 105 tests across the four
affected test files and the new model-interface file passed. The separate common
API/configuration/resume suite passed 549 tests. Ruff, lock and diff checks passed.
The complete suite was not repeated after these focused corrections; GPU and
real-checkpoint validation were not run for this change.
