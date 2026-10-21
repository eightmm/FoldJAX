# OpenDDE compact serving profile

`--opendde-padding-profile compact` fixes atom and structural-token capacities
from the residue-token bucket. Use it with the same MSA/precision/sampling
options in warming and prediction. It enables padding and is opt-in; ordinary
OpenDDE padding still follows the input's actual stored auxiliary dimensions.

```bash
foldjax cache warm --model opendde --token-buckets 128..1024 \
  --msa single --fixed-msa 2048 --opendde-padding-profile compact
foldjax predict --model opendde --input job.yaml --msa auto \
  --fixed-msa 2048 --opendde-padding-profile compact
```

The versioned policy is `PaddingConfig(auxiliary_profile="opendde-compact-v1")`
in Python. Its serial capacities are:

| Token bucket | Atom slots | Structural-token slots |
| ---: | ---: | ---: |
| 128 | 1024 | 256 |
| 256 | 2048 | 512 |
| 384 | 3072 | 768 |
| 512 | 4096 | 1024 |
| 640 | 6144 | 1280 |
| 768 | 6144 | 1536 |
| 896 | 8192 | 1792 |
| 1024 | 8192 | 1856 |
| 1280 | 12288 | 2304 |
| 1536 | 12288 | 2752 |

These are bounded serving slots, **not maximum molecular sizes or guaranteed
GPU capacities**. The 1024/1536 structural bounds retain the reported successful
GFP4/GFP6 shapes. For example, the reported GFP4 input has 7584 stored atoms
and 1816 structural tokens, so it selects 8192/1856 at bucket 1024. GFP6 has
11376/2724 and selects 12288/2752 at bucket 1536. Neither result establishes
coverage for arbitrary sequence composition or nucleic-acid/ligand content.
The other rows are selected capacities awaiting cluster admission, not new
measured card limits. The existing 24 GB / A6000 published operating limits
must not be raised merely because a profile exists.

If an unpinned stored axis exceeds its capacity, prediction refuses it before
compilation. It reports the axis, actual storage, bound and explicit `--pad-*`
alternative. It never truncates or silently enlarges an auxiliary axis. For
example, 1008 all-alanine residues need 2016 structural slots and cannot fit the
compact 1024 bucket. Dense RNA/DNA or modified chemistry can exceed the atom
bound as well. Choose a larger token bucket, matching sufficient explicit
`--pad-atoms`/`--pad-structural-tokens` pins, or omit the compact profile and
warm the actual job. Explicit pins retain their existing no-truncation checks.
Standard compact buckets stop at 1536; `auto` warms this finite grid.

Automatic compact targets retain context-parallel alignment: atoms divide
`32 * mesh_rows`, structural capacity divides both 64 and `mesh_rows`, and
tokens keep the existing token alignment. Explicit pins keep their original
semantics. Always use the same CP configuration when warming and predicting.

Compact synthetic warming uses alternating glycine/alanine rather than the
ordinary all-alanine sequence. Its native structural storage is about 1.5 times
its length, so both representative lengths fit these bounds. Nothing is removed
from the features: existing masks, native MSA sampling, padding and output
cropping apply. CPU tests compare the complete normalized model-input and
sampled-MSA PyTree shape signatures across both warm lengths and another
sequence. Full GPU numerical accuracy and actual cache-hit/capacity admission
remain cluster checks.

Host padding policies continue to share the existing outer cache namespace.
JAX's concrete input/HLO keys distinguish auxiliary shapes. Warm any missing
compact shapes; a previously compiled entry with the same concrete shapes and
options remains eligible. Other five model keys and ordinary OpenDDE targets
are unchanged. The run record retains the policy and actual resolved targets.

## Slurm example and cluster checks

Both example scripts accept `--opendde-padding-profile compact` or the optional
`OPENDDE_PADDING_PROFILE=compact` environment setting. They apply it only to
OpenDDE in a mixed-model submission. `--exact` clears an environment preset and
conflicts with an explicitly supplied profile flag. The setting is disabled by
default until the site's real-input validation is complete.

```bash
OPENDDE_PADDING_PROFILE=compact examples/slurm/foldjax-submit \
  -m opendde -o results --msa auto -- --input job.yaml
```

Keep each size in a fresh allocated process. On the 24 GB cards, compare 896
and 1024; on A6000 compare 1280 and 1536 before testing another explicit
profile. Keep native sampling, actual alignments and fixed MSA 2048. Run the
site's four actual validation jobs and record any capacity refusal instead of
silently changing their chemistry. Return actual/target shapes, successful JAX
loads versus compiles, raw device peak bytes, elapsed time, model outputs and
native-repeat accuracy comparisons. Synthetic warming alone does not prove
those jobs fit or that an executable loaded from the shared tier.

The [2026-10-21 memory policy](memory-capacity.md#reports-and-admission-changes-from-2026-10-21)
uses the effective pool ceiling for measured OpenDDE device classes. An estimate
whose uncertainty overlaps that ceiling warns; a nominal estimate over it still
refuses. A successful warning-mode run is not a guarantee for another profile.
