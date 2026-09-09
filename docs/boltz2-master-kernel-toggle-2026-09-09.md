# Boltz-2 on master: upstream's own kernel toggle is the scale for the 5SAK residual

Status: structure closed at the user's tolerance. The port's 5SAK residual sits
inside the movement upstream shows against itself when only its fused-kernel
choice changes, on the same tape, at every recorded trunk boundary and at the
coordinates. Bitwise agreement is unreachable without matching torch's fusion
order and is not claimed.

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

A same-machine port replay on this tape (jobs 488/489, two processes for the
port's own floor) is appended below when it lands.

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
