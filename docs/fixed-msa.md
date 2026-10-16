# Fixed MSA rows

`--fixed-msa N` is available for Boltz-2, Protenix, OpenFold3, and OpenDDE in
`predict`, `plan`, and `cache warm`. It means exactly:

```text
--padding --max-msa-depth N --pad-msa N
```

`N` is required and must be a positive integer in the core CLI; padding and
MSA caps remain opt-in there. The Slurm submission wrapper defaults to **2048**
for these four models, following the user-reported cluster comparison that
found no accuracy difference among 1024/2048/4096 rows. The underlying results
are not included in this repository. This deployment choice does not change
native selection limits or establish a timing/memory ranking. Choose the same
value when warming and predicting:

```bash
foldjax cache warm --model boltz2 --token-buckets 256,512 --msa single --fixed-msa 2048
foldjax predict --model boltz2 --input job.json --msa auto --fixed-msa 2048
```

`examples/slurm/foldjax-submit` accepts `--fixed-msa N` or the site variable
`FIXED_MSA_DEPTH` to override 2048; `--exact` disables its automatic preset.
AlphaFold 3 and ESMFold2 never receive that wrapper flag.

The preset caps alignment rows through the model's existing input selection,
then pads the compiled MSA axis to exactly `N`. A shallower alignment gets
masked rows; a deeper alignment goes through the native cap and sampling.
Explicit `N` takes precedence over the ordinary 1024-row profile floor and
OpenDDE's 1280-row floor, including when `N` is smaller. Padding alone never
crops alignment rows. Choosing a smaller cap can change the prediction, so the
preset is opt-in and does not change unpadded or ordinary `--padding` defaults.

| Model | Input selection and compiled rows |
| --- | --- |
| Boltz-2 | The existing featurizer limits assembled rows to `N`; padding then supplies exactly `N` rows. |
| Protenix | The existing featurizer caps assembled rows at `N`. Native per-recycle sampling draws only from those real rows; the masked feature axis and selection tape both have width `N`. |
| OpenFold3 | Native per-recycle selection reads at most `min(N, 1024)` rows. The host streaming scheduler pads each selected cycle to exactly `N`; it keeps the variable-sized union of source rows outside JIT. |
| OpenDDE | The existing featurizer caps assembled rows at `N`. Native per-recycle sampling retains its released maximum of 1280 rows; padding gives each cycle exactly `N` rows. |

OpenFold3's `max_msa_depth=1024` in `plan` is the model-side selection ceiling,
separate from the requested compiled capacity. For example,
`--fixed-msa 2048` preserves the released maximum of 1024 selected rows per
recycle and adds masked rows up to 2048. `--fixed-msa 128` selects at most 128
and pads to 128. Raising this preset above 1024 does not make OpenFold3 consume
more real rows than its released model configuration. OpenDDE similarly keeps
its native 1280-row per-cycle selection ceiling even if a larger input cap is
requested.

`--msa single` warm-up and `--msa auto` prediction use the same compile
namespace and MSA shapes when model, weights, device/runtime, token bucket,
other padded capacities, and execution options match. Alignment contents,
search provenance, and the number of real rows do not select a namespace.
Other shape axes, optional features, and model options must also match; fixed
MSA rows alone do not guarantee that every different job shares every kernel.
Templates or a different Protenix language-model provider, for example, can
change the executed program. Search uses the same external-server configuration
as an ordinary prediction.

CPU regression tests exercise native cap/sampling before padding, zero masks
for synthetic rows, model-bound input signatures and abstract StableHLO
lowering for one, three, and seventeen source rows at caps of one and seven,
and warm/predict namespace identity. They do not claim GPU latency or independently
validate the user-reported accuracy comparison between MSA caps.
