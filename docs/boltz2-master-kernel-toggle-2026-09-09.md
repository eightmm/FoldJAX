# Boltz-2 on master: upstream's own kernel toggle is the scale for the 5SAK residual

Status: 5SAK closed at the user's tolerance -- the port's residual sits inside
the movement upstream shows against itself when only its fused-kernel choice
changes, on the same tape, at every recorded trunk boundary and at the
coordinates. **This scale is 5SAK-specific.** On the ion case 1AAY
(`docs/ion-case-1aay-master-2026-09-09.md`) upstream's own toggle moves the
protein by 0.020 Å while the port sits 0.175 Å away on one sample (stable
across port processes, 0.018 Å), so there the difference is a genuine route
difference about 9x upstream's implementation scatter, and Boltz-2 stays
open on that case. Bitwise agreement is unreachable without matching torch's
fusion order and is not claimed.

## Question

The Boltz-2 5SAK cell was the one residual above 0.5 Å in the common gate:
1.18 Å (later 1.90 Å on the pinned source) against native, born in the first
MSA module (`delta_z` RMSE 3.2e-2, relative 2.07e-3), with every exposed knob
excluded and every stage inside the module at bf16-grid level. Native Boltz-2
is bitwise deterministic across processes and across cuEq torch 0.10.0/0.11.1
(`docs/openbind-master-cueq-panel-2026-09-09.md`), so there is no process
floor to read the residual against. The OpenBind lesson applies instead: how
far does upstream move against *itself* when an equivalent implementation of
the same operators is swapped in?

## Arm

Native capture on master, pinned `b1ebfc4`, cuEq torch 0.10.0, bf16-mixed,
n=5, 200 steps, 3 recycles, seed 101, identical RNG tape (`tape.npz` equal
array by array; `tape.json` differs only in `kernels` and the output path):

| arm | `use_kernels` | triangle attention / multiplication |
| --- | --- | --- |
| `native-A` (= `native-B`, bitwise) | True | cuEquivariance torch fused kernels |
| `native-nokernels` | False | upstream's unfused torch path |

Directory: `foldjax-bench/boltz2-master-native-20260909/protein_ligand_5sak/`.
Ledger rows in `docs/EXPERIMENTS.jsonl` (jobs 327, 328, 458).

## Trunk boundaries: kernels-on versus kernels-off, same tape

RMSE and Pearson correlation over the whole tensor, float64.

| boundary | RMSE | relative | correlation |
| --- | ---: | ---: | ---: |
| cycle-00 `msa_module.input_z` | 0 | 0 | 1 (bitwise) |
| cycle-00 `msa_module.delta_z` | 3.334e-2 | 2.134e-3 | 0.9999977 |
| cycle-00 `pairformer_module.output_s` | 5.673e-2 | 9.22e-4 | 0.9999996 |
| cycle-00 `pairformer_module.output_z` | 9.346e-2 | 3.918e-3 | 0.9999923 |
| cycle-03 `msa_module.delta_z` | 3.598e-2 | 2.136e-3 | 0.9999977 |
| cycle-03 `pairformer_module.output_z` | 9.341e-2 | 3.760e-3 | 0.9999929 |
| final `trunk/z` | 9.341e-2 | 3.760e-3 | 0.9999929 |
| final `trunk/s` | 5.999e-2 | 9.96e-4 | 0.9999995 |

The port, judged by the same metrics against `native-A`'s lineage on the
workstation (`docs/boltz2-msa-error-origin-2026-09-09.md`): first
`msa_module.delta_z` RMSE 3.227e-2 relative 2.07e-3; `z_trunk` correlation
0.9999998727.

So at the module where the residual is born, upstream's own alternate
implementation is farther from `native-A` (3.33e-2) than the port is
(3.23e-2), and at the end of the trunk the port's correlation (0.99999987)
is higher than upstream's own (0.9999929). The port is closer to upstream
than upstream is to itself under an equivalent implementation.

## Coordinates: entity maximum RMSD, one whole-system Kabsch per sample

| pair | protein (asym 0) per sample | max | ligand (asym 1) max |
| --- | --- | ---: | ---: |
| `native-A` vs `native-nokernels` | 2.809, 0.272, 0.137, 2.912, 2.760 | 2.912 | 0.397 |
| port vs native (workstation ledger, pinned source) | 1.897, 0.082, 0.049, 1.477, 0.636 | 1.897 | - |

The same three samples (1, 4, 5) carry both movements; samples 2 and 3 stay
under 0.3 Å in both. Sample 1 moves 2.8 Å when upstream swaps its own kernel
path and 1.9 Å when the port is substituted. This is a chaotic target: a
below-source rounding difference in the MSA module (the 3e-2 relative 2e-3
band both arms share) is amplified to ångströms by the diffusion sampler on
three of five samples, in the same direction of sensitivity for both.

## Same-machine port replay on this tape (jobs 490/491)

FoldJAX Boltz-2 (`boltz2-master-port-20260909-IcGOFo`, HEAD `e324ad0`)
replayed `native-A`'s tape and features on master.

Trunk boundaries, port versus `native-A`, beside upstream's own toggle:

| boundary | port RMSE / relative / corr | kernels-off RMSE / relative / corr |
| --- | --- | --- |
| cycle-00 `msa_module.input_z` | 6.16e-4 / 9.8e-5 / 0.99999999 | 0 / 0 / 1 |
| cycle-00 `msa_module.delta_z` | 3.262e-2 / 2.088e-3 / 0.9999978 | 3.334e-2 / 2.134e-3 / 0.9999977 |
| cycle-03 `pairformer_module.output_s` | 5.383e-2 / 8.94e-4 / 0.9999996 | 5.999e-2 / 9.96e-4 / 0.9999995 |
| cycle-03 `pairformer_module.output_z` | 7.056e-2 / 2.840e-3 / 0.9999960 | 9.341e-2 / 3.760e-3 / 0.9999929 |

The port is closer to `native-A` than upstream's own unfused path at every
boundary, including the one where the residual is born (the port's `input_z`
already differs by 1e-4 relative because its input embedder is its own; the
toggle arm shares upstream's embedder bitwise).

Coordinates, three-way, entity RMSD per sample (protein chain; ligand in
parentheses as the maximum):

| pair | samples 1-5 | max |
| --- | --- | ---: |
| `native-A` vs kernels-off | 2.809, 0.272, 0.137, 2.912, 2.760 | 2.912 (0.397) |
| `native-A` vs port | 2.421, 0.166, 0.019, 2.273, 0.345 | 2.421 (0.141) |
| kernels-off vs port | 0.875, 0.211, 0.127, 1.860, 2.773 | 2.773 (0.296) |

The three implementations are mutually equidistant at 2.4-2.9 Å on samples
1 and 4, and samples 2 and 3 agree to 0.2 Å in every pair: a common chaotic
floor, not a port-specific offset.

Port versus port, two processes on master (jobs 490/491, XLA autotune not
frozen): protein 0.828, 0.002, 0.001, 0.141, 0.110 Å; ligand max 0.082 Å.
Sample 1 alone moves 0.83 Å between two runs of the identical port source on
the identical tape, which is the same sample that moves 2.4-2.9 Å between any
two of the three implementations. The chaos is in the target, and the port's
own process floor on it is already 0.8 Å.

## The port's 0.83 Å process floor was XLA kernel selection (jobs 631/632)

Two more port processes on the same tape with XLA autotune frozen
(`port-C` dumped `protein_ligand_5sak-port-autotune.textproto`, `port-D`
loaded it with `--xla_gpu_require_complete_aot_autotune_results=true`;
`protein_ligand_5sak-frozen-pair.json`):

| pair | protein samples 1-5 (Å) | ligand max |
| --- | --- | ---: |
| `port-C` vs `port-D` (frozen) | 0.0008, 0.0001, 0.0001, 0.0002, 0.0002 | 0.0001 |
| `port-A` (unfrozen) vs `port-C` | 0.827, 0.002, 0.001, 0.141, 0.110 | 0.081 |
| `native-A` vs `port-C` | 2.219, 0.166, 0.020, 2.277, 0.338 | 0.150 |
| `native-A` vs `port-D` | 2.219, 0.166, 0.020, 2.277, 0.338 | 0.151 |

With kernel selection pinned the port is repeatable to 1e-3 Å (not bitwise:
the remaining 1e-4 band is the same trunk-level nondeterminism Protenix
shows, `docs/protenix-master-panel-2026-09-09.md`). The 0.83 Å "port floor"
of jobs 490/491 was therefore autotune choosing different kernels per
process, and the frozen port sits at the same 2.2-2.3 Å from `native-A` on
samples 1 and 4 as the unfrozen `port-B` did. Nothing about the native
residual changes; only its attribution: the chaos is amplified from the
kernel-family rounding band, and the port's own contribution to the scatter
is 1e-3 Å once its kernels are pinned.

## Reading

- The residual is not attributable to a port defect: no exposed knob moved it
  (earlier ledger), every stage inside the module sits at bf16-grid level, and
  upstream's own equivalent implementation moves the same boundaries by the
  same amount and the same samples by more.
- Kernel family already matches (cuEq torch vs cuEq JAX, same CUDA kernels);
  the remaining difference is fusion/accumulation order inside torch autocast
  versus XLA, which no port option reaches.
- Closure at the user's tolerance ("수치가 조금 변하는 건 용인"): recorded as a
  sensitivity band, not a pass at 0.05 Å. Anyone needing bitwise Boltz-2 must
  reproduce torch's autocast fusion order, which is implementation work
  outside this port's option space.
