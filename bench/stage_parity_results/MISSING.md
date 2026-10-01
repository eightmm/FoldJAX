# Stages the stored captures cannot support

`bench/stage_parity.py` writes a stage as `not_captured` when the native
capture holds nothing to compare against, rather than filling it from the
port. Six stage cells are in that state, all for the same reason: the
ESMFold2 and OpenDDE master captures did not save the native trunk boundary,
so there is no native trunk to compare against (S3) or to inject (S4, S5).
Each section below names the upstream hook (pinned source, file:function),
the tensors a new GPU capture must save, and where the port takes the
injection, plus a capture sketch that reuses the existing tape harness, so
the noise, dropout and MSA draws stay on the same tape the stored captures
already carry.

| model | S3 trunk | S4 diffusion | S5 confidence | cause |
| --- | --- | --- | --- | --- |
| ESMFold2 `protein_1ubq` | not captured | not captured | not captured | `tape.npz` holds `initial_pair_state` (a trunk *input*); no trunk output saved |
| OpenDDE `protein_1ubq` | not captured | not captured | not captured | `raw.npz` holds heads + coordinates only; `torch/tape.json` records `"stages": {}` |

Protenix, Boltz-2 and OpenFold3 have every stage. Three measured stages carry
a caveat that a re-capture would also close; they are listed at the end.

## ESMFold2 (Biohub `transformers` snapshot `ef32577f`)

Upstream source: `jctc-matrix-20260904/upstream-root/transformers-esmfold2/src/transformers/models/esmfold2/modeling_esmfold2.py`.
ESMFold2 has no single-representation trunk: the folding trunk produces a
pair tensor `z`, and the sampler and confidence head read `x_inputs` (the
input embedding) as their single input.

| stage | upstream hook | tensors to save |
| --- | --- | --- |
| S3 | `ESMFold2Model.forward`, the `z` after `z = self.parcae_coda(z, pair_attention_mask=pair_mask)` and `z = z.float()` (lines 1027-1029), observed as the `z_trunk` keyword of `self.structure_head.sample(...)` (line 1032) | `z_trunk` `[1, N, N, 256]` f32; `s_inputs` (= `x_inputs`) `[1, N, 451]`; `relative_position_encoding`; `distogram_logits` from line 1030 |
| S4 | `DiffusionStructureHead.sample` (called at line 1032) | its keyword arguments above plus the returned `sample_atom_coords` `[5, A, 3]`; the diffusion draws are already in `tape.npz` |
| S5 | `ConfidenceHead.forward` (class at line 96, forward at line 172; called at line 1061) | keyword inputs `s_inputs`, `z`, `x_pred`, `relative_position_encoding`, `token_bonds_encoding`, `distogram_atom_idx`, masks; outputs `plddt_logits`, `pae_logits`, `pde_logits`, `resolved_logits` and the reduced scores already in `upstream_confidence.npz` |

Port injection seams (`src/foldjax/models/esmfold2/models/model.py`,
`predict`): the `folding_trunk(z, ..., "parcae_coda", ...)` call (line 1669)
for the trunk (patch that call site, not the module-global `folding_trunk`,
which the recycling loop at line 1031 also uses); `diffusion.sample` (line
1736) for S5's coordinates; `confidence_head` (line 1815) is downstream of
both.

Capture sketch -- add to `bench/esmfold2_tape.py:_capture` before the
`model(**device_features, num_diffusion_samples=SAMPLES)` call (line 681), so
the `TorchRecorder` that writes `tape.npz` is still the only RNG observer:

```python
stages: dict[str, torch.Tensor] = {}
original_sample = model.structure_head.sample

def sample(**kwargs):
    for name in ("z_trunk", "s_inputs", "relative_position_encoding"):
        stages[f"trunk.{name}"] = kwargs[name].detach().float().cpu()
    output = original_sample(**kwargs)
    stages["diffusion.sample_atom_coords"] = output["sample_atom_coords"].detach().cpu()
    return output

model.structure_head.sample = sample

def on_confidence(module, args, kwargs, output):
    for name, value in kwargs.items():
        if torch.is_tensor(value):
            stages[f"confidence_in.{name}"] = value.detach().float().cpu()
    for name, value in output.items():
        stages[f"confidence_out.{name}"] = value.detach().float().cpu()

handle = model.confidence_head.register_forward_hook(on_confidence, with_kwargs=True)
...  # existing forward
handle.remove()
_save_npz(args.output_dir / "stages.npz", {k: v.numpy() for k, v in stages.items()})
```

Then S3 compares the port's `z` at the `parcae_coda` seam with
`trunk.z_trunk`; S4 injects `trunk.*` there and replays `tape.npz`; S5
additionally injects `diffusion.sample_atom_coords` at `diffusion.sample`.

## OpenDDE 1.1.1 (commit `ddfa1df8`)

Upstream source: `/home/jaemin/non-project/optimizing/OpenDDE/opendde/model/opendde.py`.
OpenDDE runs a residue-level Pairformer trunk and then expands to structural
tokens; the sampler and confidence head read the *structural* tensors.

| stage | upstream hook | tensors to save |
| --- | --- | --- |
| S3 | `OpenDDE.get_pairformer_output` (line 952; called from `_main_inference_loop` at line 1869) and `OpenDDE.expand_to_structural_tokens` (line 422; called at line 1899) | residue `s_inputs`, `s`, `z` (`[N_res, 449]`, `[N_res, 384]`, `[N_res, N_res, 128]`) and structural `s_inputs`, `s`, `z` (`[N_struct, ...]`; 146 structural tokens on 1UBQ) |
| S4 | `OpenDDE.run_sample_diffusion_stage` (line 1195, called at line 1998) / `OpenDDE.sample_diffusion` (line 1299) | the structural `s_inputs`, `s_trunk`, `z_trunk` it receives and the returned `coordinate`; the sampler draws are already in `torch/tape.npz` |
| S5 | `OpenDDE.run_confidence_head_stage` (line 1509, called at line 2033) | keyword inputs `s_inputs`, `s_trunk`, `z_trunk`, `pair_mask`, `x_pred_coords`; returned `plddt`, `pae`, `pde`, `resolved` logits (the reduced scores are already in `raw.npz`) |

Port injection seams (`src/foldjax/models/opendde/models/model.py`,
`opendde_infer_static`): `pairformer_output_from_s_inputs` (line 1022) for the
residue trunk, `structural_token_expand` (line 1049) for the structural
boundary, `sample_diffusion` (line 1213) for S5's coordinates; the
`confidence_head` call (line 1330) is downstream. All three are module globals, so
the same counted-patch-plus-pool-clear pattern `run_protenix` uses applies.

Capture sketch -- `tests/models/opendde/scripts/capture_upstream_tape.py`
already installs forward hooks on `input_embedder`, `msa_module` and
`pairformer_stack` (lines 320-433), but the master capture's
`torch/tape.json` records `"stages": {}` -- what `--skip-trunk-stages`
produces, a flag that exists because the per-block stage tensors are ~26 GiB.
The boundaries themselves are small (about 15 MB at 76 residue / 146
structural tokens), so wrap the methods instead of the blocks:

```python
saved = {}

def wrap(name, method, pick):
    def wrapped(*args, **kwargs):
        result = method(*args, **kwargs)
        for key, value in pick(kwargs, result).items():
            saved[f"{name}.{key}"] = value.detach().float().cpu().numpy()
        return result
    return wrapped

model.get_pairformer_output = wrap(
    "residue_trunk", model.get_pairformer_output,
    lambda kw, out: dict(zip(("s_inputs", "s", "z"), out)))
model.expand_to_structural_tokens = wrap(
    "structural_trunk", model.expand_to_structural_tokens,
    lambda kw, out: dict(zip(("s_inputs", "s", "z"), out[1:])))
model.run_confidence_head_stage = wrap(
    "confidence", model.run_confidence_head_stage,
    lambda kw, out: {
        **{k: kw[k] for k in ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords")},
        **dict(zip(("plddt", "pae", "pde", "resolved"), out)),
    })
...  # existing tape-recorded forward, unchanged, with --skip-trunk-stages
np.savez_compressed(out_dir / "stages.npz", **saved)
```

## Caveats on measured stages that a re-capture would close

* **Boltz-2 S1 -- reference-conformer augmentation not replayed.** The
  featurizer rotates and translates each reference conformer at random.
  Upstream's draws are on disk (`native-A/preprocessing-tape.npz`, recorded by
  patching `boltz.data.feature.featurizerv2.center_random_augmentation` in
  `bench/boltz_closure_capture.py`), but the port featurizer has no injection
  point for them (`src/foldjax/models/boltz2/data/_external.py:center_random_augmentation`,
  called from `data/feature/featurizerv2.py:1561`). `ref_pos` differs by up to
  13 A raw and by 1.4e-6 A after one Kabsch fit per `ref_space_uid`, so the
  difference is that rigid draw and nothing else; every categorical feature
  is exact. Closing it is a port change (accept the tape), not a capture.
* **OpenFold3 S1 -- reference conformers not reproduced.** OpenFold3 builds
  each reference conformer with RDKit ETKDGv3 embedding
  (`openfold3/core/data/primitives/structure/conformer.py:_compute_conformer`,
  vendored at `src/foldjax/models/openfold3/_upstream/...`). The capture's
  tape holds only the MSA draws and the sampler draws, not these
  coordinates, and the port's seed-101 featurization does not reproduce them:
  all 118 conformers differ in internal geometry (intra-conformer distances
  up to 3.9 A apart; seeds 42 and 0 behave the same), so `ref_pos` stays
  3.9 A off even after a per-conformer fit. Every categorical feature is
  exact and the other float features agree to 1.2e-7. A re-capture should
  save the per-residue conformer coordinates returned by
  `_compute_conformer` / `multistrategy_compute_conformer` (they are what
  upstream writes into `ref_pos`), and the port featurizer needs a seam to
  take them. S3-S6 are unaffected: they run on the native `input.npz`.
* **ESMFold2 S1 -- input document inferred.** The capture ran a shared feature
  archive (`metadata.json: core_only_shared_features = true`) and records only
  its sha256, not the document or MSA it was built from. S1 re-featurizes the
  JCTC-matrix 1UBQ job with the bench's 1UBQ alignment (sha256 `4bf7b489...`,
  the alignment OpenDDE's capture recorded for the same sequence); the result
  agrees with the archive to 1.2e-7 on every shared array, which is what
  supports the inference. A re-capture should record the input document and
  alignment digests next to `features.npz`.
* **Every S3-S6 stage runs CPU XLA against a GPU capture.** Triangle kernels
  are XLA where the native run used cuEquivariance, and bf16 is CPU bf16. The
  residuals are therefore a CPU-vs-GPU number on top of port-vs-native (see
  `docs/parity-cpu.md`); a GPU run of this script would separate the two.
