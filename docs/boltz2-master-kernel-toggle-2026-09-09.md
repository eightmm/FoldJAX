# Boltz-2 on master: upstream's own kernel toggle is the scale for the 5SAK residual

Status: 5SAK closed at the user's tolerance -- the port's residual sits inside
the movement upstream shows against itself when only its fused-kernel choice
changes, on the same tape, at every recorded trunk boundary and at the
coordinates. **This scale is 5SAK-specific.** On the ion case 1AAY
(`docs/ion-case-1aay-master-2026-09-09.md`) upstream's own toggle moves the
protein by 0.020 Å while the port sits 0.175 Å away on one sample (stable
across port processes, 0.018 Å), so at the coordinates the port sits about 9x
upstream's implementation scatter there. The 1AAY section below places it:
the port's trunk is inside upstream's own implementation band at every
boundary and the port's sampler reproduces native to 1e-5 Å given native's
trunk, so it is one near-tie sample amplifying a same-size trunk band, not a
route the port can change. Bitwise agreement is unreachable without matching torch's
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

With kernel selection pinned the port is repeatable to 1e-3 Å, not bitwise:
the two runs' trunk boundaries agree bitwise at every recorded array except
`msa_module.input_emb`, which differs by one f32 ulp (2.4e-7), so the
remaining 1e-3 Å arises downstream of the trunk (sampler or that one ulp;
not separated). The 0.83 Å "port floor"
of jobs 490/491 was therefore autotune choosing different kernels per
process, and the frozen port sits at the same 2.2-2.3 Å from `native-A` on
samples 1 and 4 as the unfrozen `port-B` did. Nothing about the native
residual changes; only its attribution: the chaos is amplified from the
kernel-family rounding band, and the port's own contribution to the scatter
is 1e-3 Å once its kernels are pinned.

## 1AAY: where the 0.175 Å lives (jobs 635/636)

The ion case is the one Boltz-2 residual above upstream's own toggle scale
(port 0.175 Å on sample 3 against 0.020 Å for kernels-off). Two
counterfactuals on `native-A`'s tape with `bench.boltz_downstream_probe`
(native trunk `s/z/s_inputs/rel_pos` substituted, FoldJAX sampler and
confidence heads skipped; snapshot `boltz2-master-port-20260909-IcGOFo`):

| arm | protein samples 1-5 (Å) | DNA / Zn max |
| --- | --- | ---: |
| native trunk + FoldJAX conditioning + FoldJAX sampler | 1.6e-5, 6.6e-5, 5.1e-5, 4.8e-5, 3.7e-5 | 6e-5 |
| native trunk + native conditioning + FoldJAX sampler | 5.2e-5, 3.1e-5, 5.6e-5, 3.3e-5, 7.0e-5 | 8e-5 |

Given native's trunk, the port's diffusion conditioning and sampler
reproduce native's coordinates to 1e-5 Å on every sample and every entity,
sample 3 included. The whole 0.175 Å is therefore born in the trunk and
amplified by sample 3's diffusion path.

Trunk boundaries on 1AAY, `native-A` versus the port and versus upstream's
own kernels-off arm (RMSE / relative / correlation, float64):

| boundary | port | kernels-off |
| --- | --- | --- |
| cycle-00 `msa_module.input_z` | 7.8e-4 / 1.1e-4 / 0.99999999 | 0 (bitwise) |
| cycle-00 `msa_module.delta_z` | 2.567e-2 / 1.314e-3 / 0.99999914 | 2.841e-2 / 1.454e-3 / 0.99999894 |
| cycle-00 `pairformer.output_s` | 6.104e-2 / 8.42e-4 / 0.99999965 | 6.735e-2 / 9.29e-4 / 0.99999958 |
| cycle-00 `pairformer.output_z` | 8.847e-2 / 2.937e-3 / 0.99999569 | 9.975e-2 / 3.311e-3 / 0.99999452 |
| cycle-03 `msa_module.delta_z` | 3.315e-2 / 1.518e-3 / 0.99999885 | 3.458e-2 / 1.583e-3 / 0.99999875 |
| cycle-03 `pairformer.output_s` | 7.381e-2 / 1.022e-3 / 0.99999948 | 8.140e-2 / 1.127e-3 / 0.99999937 |
| cycle-03 `pairformer.output_z` | 8.508e-2 / 2.698e-3 / 0.99999636 | 9.548e-2 / 3.028e-3 / 0.99999542 |

As on 5SAK, the port's trunk is closer to `native-A` at every boundary than
upstream's own unfused path is. The two perturbations have the same size and
different directions; sample 3's diffusion path amplifies the port's
direction to 0.175 Å and the kernels-off direction to 0.020 Å. So the 1AAY
item is not a defect the port can fix by matching operators (it already
matches them better than upstream matches itself); it is one near-tie sample
on a tame target whose outcome depends on which sub-bf16 rounding direction
the trunk takes. Recorded as an accepted sensitivity, same class as 5SAK,
with the coordinate figure kept in the ion document as measured.

### The near-tie, measured (jobs 647-650)

"Near-tie" was an assertion until this control. `bench/boltz_trunk_perturb.py`
adds Gaussian noise to native-A's own trunk at the port's measured band
(relative RMSE `s` 1.02e-3, `z` 2.70e-3, `s_inputs` 7.8e-5; `rel_pos`
untouched) and runs the FoldJAX sampler on the same tape. Four seeds,
protein entity RMSD against native-A:

| draw | samples 1-5 (Å) | DNA / Zn max |
| --- | --- | ---: |
| seed 1001 | 0.012, 0.015, 0.009, 0.008, 0.067 | 0.009 |
| seed 1002 | 0.016, 0.006, 0.007, 0.007, 0.013 | 0.007 |
| seed 1003 | 0.020, 0.013, **0.1745**, 0.006, 0.015 | 0.008 |
| seed 1004 | 0.007, 0.009, **0.1744**, 0.005, 0.007 | 0.006 |
| port (for reference) | 0.028, 0.006, **0.1746**, 0.010, 0.018 | 0.018 |

Two of four random perturbations of the port's own size land sample 3 at
0.1745 Å, the same distance as the port's 0.1746 Å, while the other two stay
at 0.007-0.009 Å; the other samples stay at 0.005-0.02 Å except one 0.067 Å
on sample 5. Sample 3 has two outcomes 0.17 Å apart and a random same-norm
trunk perturbation picks the far one about half the time. The port's 0.175 Å
is that outcome, not a route of its own. Verdict for 1AAY: at-floor, with
the floor now measured as this bistability rather than inferred. (The
kernels-off-trunk sanity arm, job 651, did not run: the probe refuses a
kernels-off capture by policy.)

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
