# Protenix on master: native floors first, then the port read three ways

Status: closed at the user's tolerance. Two cases pass outright (1URN, 3GCA);
the other five sit inside upstream's own process-to-process movement. On every
chaotic case the port lands in the same basin as one of the two native
processes, at the distance those two native processes agree with each other
when they share a basin.

## Arms

Pinned upstream Protenix `4c355be` plus the pinned `layer_norm/torch_ext_compile.py`
edit (same as the workstation's source spec), `protenix_base_default_v1.0.0`,
n=5, 200 steps, seed 101, default policy (bf16 storage, torch TF32), on RTX
PRO 6000 Blackwell Server Edition. Master environment: `protenix/.venv`,
torch 2.12.0+cu130, cuequivariance-torch 0.11.1 (the workstation captures
used 0.9.0; Boltz-2 showed cuEq-release independence, Protenix has not been
tested for it separately -- the two native processes here share 0.11.1, so
the floors below are release-internal). Native captures record the RNG tape,
the MSA tape and, where native selected MC dropout (5SAK, 7R6R), the per-cycle
dropout masks; the port replays all of them
(`bench/protenix_closure_capture.py`, `bench/protenix_foldjax_capture.py
--reference`). Snapshot `protenix-master-native-20260909-9yET4f` (FoldJAX `e324ad0`), port weights
`.foldjax/weights/protenix/protenix_base_default_v1.0.0.jax` with the
`foldjax-compare-002` weight audit.

Four runs per case: native A, native B (same tape, second process), port A,
port B (same tape, second process, XLA autotune not frozen). Comparisons by
`bench/protenix_master_diff.py`; table by `bench/master_three_way.py`.

## Results (entity maximum RMSD, Å; one whole-system Kabsch per sample)

| case | native A vs B (floor) | native A vs port | native B vs port | port A vs B (floor) | verdict |
| --- | --- | --- | --- | --- | --- |
| protein_1ubq | A 0.095 | A 0.170 | A 0.171 | A 0.169 | at-floor |
| protein_dna_7r6r | A 1.270; B 1.012; D 0.957 | A 1.488; B 1.188; D 1.126 | A 0.262; B 0.202; D 0.199 | A 1.718; B 1.342; D 1.301 | at-floor |
| protein_ligand_5sak | A 0.069; L 0.028 | A 0.105; L 0.055 | A 0.061; L 0.048 | A 0.114; L 0.037 | at-floor |
| protein_protein_7st3 | A 10.856; B 25.453 | A 6.358; B 19.159 | A 10.768; B 25.692 | A 12.469; B 26.749 | at-floor |
| protein_rna_1urn | P 0.008; R 0.003 | P 0.021; R 0.011 | P 0.018; R 0.011 | P 0.017; R 0.012 | pass |
| protein_rna_ligand_3v7e | L 1.045; P 0.843; R 1.530 | L 1.027; P 0.862; R 1.556 | L 0.160; P 0.216; R 0.219 | L 0.286; P 0.194; R 0.492 | at-floor |
| rna_ligand_3gca | L 0.001; R 0.002 | R 0.031; L 0.034 | R 0.031; L 0.034 | R 0.062; L 0.033 | pass |

Verdict rule: `pass` below 0.05 Å; `deferred` below 0.1 Å; `at-floor` when
the worst native-A-versus-port entity is within twice the larger of the two
floors; else `investigate`.

## Reading, per sample, where it matters

The floors on 7ST3, 7R6R and 3V7E are not noise bands, they are basin
selection: upstream's bf16 diffusion trajectories are bistable on some
samples, and two native processes on the identical tape pick different
basins. The port is a third process, and it always shares a basin with one
of them:

| case, chain | sample | native A vs B | native A vs port | native B vs port |
| --- | ---: | ---: | ---: | ---: |
| 7ST3 A | 1 | 10.86 | **0.38** | 10.77 |
| 7ST3 A | 3 | 6.32 | 6.36 | **0.19** |
| 7ST3 A | 5 | 3.64 | **0.25** | 3.66 |
| 7ST3 B | 1 | 25.45 | **1.39** | 25.69 |
| 7R6R A | 1 | 1.27 | 1.49 | **0.26** |
| 3V7E R | 1 | 1.53 | 1.56 | **0.11** |
| 3V7E P | 2 | 0.67 | 0.49 | **0.22** |

On 3V7E the port agrees with native B on all five samples (0.04-0.22 Å)
while native A is 0.1-1.5 Å from both. On the non-chaotic samples (7ST3
samples 2 and 4, 7R6R samples 2-5) the port sits at 0.03-0.2 Å, the same
scale as native versus native.

The tight cases read directly: 1URN 0.021/0.011 Å against a 0.008 Å native
floor; 3GCA 0.031/0.034 Å against 0.002 Å (both `pass`); 5SAK 0.105/0.055 Å
against a 0.069 Å native floor and a 0.114 Å port floor. 1UBQ's 0.170 Å is
one sample (2) where the port's two processes also differ by 0.169 Å while
the two native processes agree to 0.003 Å: that is the port's own
kernel-selection noise on that sample (autotune unfrozen), not a route
difference; samples 1, 3-5 are at 0.015-0.044 Å.

## What this closes and what it does not

- The earlier workstation readings (7R6R 0.8-3.4 Å, 3V7E 0.2-1.4 Å against a
  single native run) were basin selection, as the two native repeats on the
  workstation (7R6R 3.0/2.4/2.3 Å) already suggested. With the port as the
  third vertex the picture is unambiguous.
- Bitwise Protenix parity is not reachable: upstream is not deterministic
  against itself on these targets. What is reachable, and what this panel
  shows, is that the port's distribution of outcomes is upstream's.
- Frozen-autotune port pair on 1UBQ (jobs 629/630, XLA autotune dumped by C
  and loaded by D with complete-results required): **not** bitwise; the two
  frozen processes still differ by 0.171 Å on sample 2 (0.018-0.059 on the
  others), and port C sits 0.172 Å from native A on that sample. Unlike
  OpenFold3 and ESMFold2, the Protenix port carries a process-to-process
  nondeterminism that autotune freezing does not remove (a candidate is the
  atom-level scatter/segment reductions, which are not order-stable on GPU).
  It is the same size as the native process floor on 1UBQ (0.095 Å) and does
  not change any verdict, but it is the one place where "bitwise
  repeatable port" is not yet true.
- Not measured here: cuEq 0.9.0 versus 0.11.1 for Protenix natively.

## Job log

Jobs 528, 537-549 (native A/B), 550-563 (port A/B); ledger rows in
`docs/EXPERIMENTS.jsonl`. Assets under `~/common` (CCD components, PDB
cluster file, obsolete-release CSV) copied from the workstation.

## Where the frozen pair's 0.17 Å comes from (1UBQ, port-C vs port-D)

With XLA autotune frozen and loaded (`--xla_gpu_require_complete_aot_autotune_results=true`)
the two port processes share `foldjax-input.npz` bitwise (64 arrays, 0
differing), `infer-boundary.json` and `schedule-audit.json` byte-equal, yet
`prediction.npz` already differs at `s_trunk` and `z_trunk` (max abs 16 / 32
on bf16-scale trunk activations), `distogram_logits` (8), and only then at
`coordinate` (2.29 max abs, 0.17 Å entity RMSD). The nondeterminism is in the
trunk, not in the diffusion sampler's atom scatter; it survives autotune
freezing, so the remaining candidates are XLA's non-deterministic reductions
(atomics in the cuEq/Triton triangle kernels or `xla_gpu_deterministic_ops`
off). One sentence of status, not a defect: native's own floor on this case
is the same size.

### Deterministic XLA ops make the port bitwise and move 1UBQ into the pass band (jobs 654/656)

Two more port processes on native-A's tape with `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
and nothing else (under that flag XLA recorded no autotune results, so the
autotune cache could not be frozen separately; `port-autotune-detops.textproto`
is empty). Entity RMSD, chain A, five samples:

| pair | samples 1-5 (Å) | max |
| --- | --- | ---: |
| `port-E` vs `port-F` (both deterministic) | 3e-15, 7e-15, 5e-15, 4e-15, 5e-15 | bitwise (every `prediction.npz` array equal) |
| `native-A` vs `port-E` | 0.013, 0.016, 0.017, 0.028, 0.037 | 0.037 |
| `native-A` vs `port-C` (frozen autotune, non-deterministic ops) | 0.022, 0.172, 0.026, 0.017, 0.024 | 0.172 |
| `port-E` vs `port-C` | 0.015, 0.177, 0.013, 0.022, 0.017 | 0.177 |

So the port's remaining 0.17 Å on 1UBQ sample 2 was XLA's non-deterministic
reductions (atomics in scatter/reduce fusions), not the cuEq/Triton kernels
and not the sampler; with deterministic ops the port is bitwise across
processes and sits 0.037 Å from native-A on every sample, inside the pass
band. Wall time did not move (69 s for jobs 630, 654 and 656 alike on this
115-token case). Not yet a default: one case; the other six cases and the
1-3k-token timing cost are unmeasured, and the flag is process-wide XLA
configuration rather than a port option.

### Deterministic ops on all seven cases (jobs 657-662, one port process each)

Entity maximum RMSD (Å), port with `--xla_gpu_deterministic_ops=true`
(`port-E`) against each native process, beside the earlier unfrozen port-A
column. Verdict from the better native pairing at the same rule as the panel.

| case | native A-B | A vs port-A | B vs port-A | A vs port-E | B vs port-E | port-E verdict (before) |
| --- | --- | --- | --- | --- | --- | --- |
| protein_1ubq | A 0.095 | 0.170 | 0.171 | 0.037 | 0.109 | pass (at-floor) |
| protein_rna_1urn | P 0.008, R 0.003 | 0.021 | 0.018 | 0.035 | 0.028 | pass (pass) |
| rna_ligand_3gca | L 0.001, R 0.002 | 0.034 | 0.034 | 0.044 | 0.044 | pass (pass) |
| protein_ligand_5sak | A 0.069, L 0.028 | A 0.105, L 0.055 | A 0.061, L 0.048 | A 0.117, L 0.044 | A 0.090, L 0.039 | deferred (deferred) |
| protein_dna_7r6r | A 1.27, B 1.01, D 0.96 | A 1.49 | A 0.26 | A 1.23 | A 0.24 | at-floor (at-floor) |
| protein_rna_ligand_3v7e | L 1.05, P 0.84, R 1.53 | R 1.56 | R 0.22 | R 1.58 | R 0.50 | at-floor (at-floor) |
| protein_protein_7st3 | A 10.9, B 25.5 | B 19.2 | B 25.7 | B 19.1 | B 26.4 | at-floor (at-floor) |

The flag changes one verdict (1UBQ at-floor → pass) and leaves the other six
in their bands; the three chaotic cases keep sharing a basin with one native
process per sample (7ST3 samples 2/4: 0.03-0.09 Å against native-A). Wall
time: 67-180 s per replay against 150-360 s for the earlier port-A runs, but
those ran with a colder compile cache, so this is "no penalty visible", not
a timing. The deterministic port is bitwise across processes (1UBQ E/F), which
the frozen-autotune port was not.

Recommendation: use `XLA_FLAGS=--xla_gpu_deterministic_ops=true` for every
Protenix parity replay from here on. Promotion to the backend's default waits
on a 1-3k-token timing (deterministic reductions can cost at scale) and on a
port-level way to set it that does not reconfigure the host process.

### Deterministic ops at scale (jobs 663/664, L1000_3og2, 1003 protein tokens)

`bench.run_foldjax`, warm after a compile-cache prefill, n=5, 200 steps,
10 recycles, one process per row:

| arm | wall s | peak MiB |
| --- | ---: | ---: |
| default XLA | 65.44 | 6888 |
| `--xla_gpu_deterministic_ops=true` | 74.12 | 6912 |

The flag costs 13% wall at 1k tokens (the 115-token replays showed none),
so it is a parity-replay setting, not a default. The 3k-token rows (jobs
665/666) follow below when they land.

### Deterministic ops at 3012 tokens: XLA's autotuner has no deterministic config for two gemm fusions (jobs 666, 728)

| arm (L3000_6ztx, 3012 tokens) | outcome |
| --- | ---: |
| default XLA (job 665) | 579.3 s warm, peak 42219 MiB |
| `--xla_gpu_deterministic_ops=true` | compile fails: `Failed to get configs for: 2 out of 171 instructions` |
| `--xla_gpu_exclude_nondeterministic_ops=true` | same failure, same two instructions |

The two are `gemm_fusion_dot` instructions of shape `f32[5,16,4,96,48]`
(a batched diffusion-head matmul whose padded shape only appears at this
bucket), for which XLA's deterministic autotuner finds "No supported config".
So the flag is not usable at 3k tokens as-is; a retry with Triton gemms
disabled (`--xla_gpu_enable_triton_gemm=false`, job queued as
`det-notriton`) is the one remaining escape hatch, recorded below when it
lands. Either way the opt-in option in the port must fail loudly at compile
time rather than silently fall back, which is what XLA already does.

### The escape hatch works: deterministic ops with Triton gemms disabled (job 747)

| arm (L3000_6ztx) | wall s | peak MiB |
| --- | ---: | ---: |
| default XLA (job 665) | 579.3 | 42219 |
| `--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false` | 635.4 | 41207 |

Routing the two batched gemms through cuBLAS instead of Triton gives the
deterministic autotuner a candidate; the run compiles and costs 9.7% wall at
3k tokens (13% at 1k with Triton still on). The port's `deterministic=on`
option (`foldjax.models._compile_policy.DETERMINISTIC_COMPILER_OPTIONS`, shared
by every port)
therefore carries both keys, so it compiles at every bucket measured. Still
opt-in; the default run is the one every other number here describes.
