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
- Not measured here: cuEq 0.9.0 versus 0.11.1 for Protenix natively; the
  port's frozen-autotune floor (would tighten 1UBQ sample 2 and 5SAK).

## Job log

Jobs 528, 537-549 (native A/B), 550-563 (port A/B); ledger rows in
`docs/EXPERIMENTS.jsonl`. Assets under `~/common` (CCD components, PDB
cluster file, obsolete-release CSV) copied from the workstation.
