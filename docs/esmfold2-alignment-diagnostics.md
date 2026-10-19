# ESMFold2 alignment diagnostics

The experimental Jacobi path has not met the full-model default-change
criterion. That is a result of the paired experiment, not an attribution of
its residual to insufficient Jacobi iterations. SVD remains the default.

## What the current evidence distinguishes

On the 254/499-token GPU panel, Jacobi's maximum entity RMSDs against SVD were
0.0887/0.0627 Å; independent SVD repeats differed by 0.0754/0.0571 Å. Both
comparisons exceeded the fixed 0.05 Å diagnostic. They used the same actual
diffusion tape and identical compact LM embeddings. Other random consumers
used unchanged keys but their individual draws were not independently
captured. Fresh compilations can also choose different kernels. These results
cannot identify the alignment solver as the cause of the whole-model residual.
See [the measured scope](performance-diagnostics.md#esmfold2-solver-scope).

The CPU-only diagnostic separates three questions:

- **Convergence:** does another sweep reduce the eigen residual or improve the
  proper-rotation objective on the same covariance?
- **FP32 arithmetic and conditioning:** does FP64 arithmetic improve the result
  on exactly the same FP32-rounded covariance, and is the largest eigenvector
  well separated from the next one?
- **Centering and input precision:** do both SVD and Jacobi differ from an FP64
  alignment of the same rounded point clouds when scale or translation changes?

The initial deterministic synthetic panel contained 416 matrices, including
ordinary, rank-deficient, reflected and scaled covariances. Its maximum
normalized eigen residual was:

| Jacobi arithmetic | Sweeps | Maximum normalized eigen residual |
| --- | --- | --- |
| FP32 | 1 | 0.457 |
| FP32 | 3 | 0.00143 |
| FP32 | 5 | 2.22e-7 |
| FP32 | 10 | 2.22e-7 |
| FP32 | 20 | 2.72e-7 |
| FP32 | 40 | 2.22e-7 |
| FP64, same rounded inputs | 10 | 6.94e-16 |

This panel gives no evidence that the production ten sweeps failed to
converge. Adding sweeps did not improve its FP32 residual floor. The
instrumented ten-sweep rotations matched the actual production helper bit
for bit on this CPU panel. Instrumentation can affect fusion, so the tool
also evaluates the production helper separately.

For well-conditioned random matrices, the largest rotation-component error
against an FP64 SVD oracle was 5.54e-7. Near an unresolved largest-eigenvalue
gap, different rotations can have nearly the same optimum: rank-one and
reflection-degenerate cases differed by up to 1.95 in a rotation component,
while the panel's normalized objective deficit stayed below 1.81e-7. Rotations
in a nonunique or weakly determined subspace need objective and eigen-residual
checks; a large rotation difference alone is not evidence of nonconvergence.
A small negative deficit can reflect finite-precision loss of orthogonality,
not an improvement on the exactly constrained optimum.
Such choices can still affect later sampler steps, so they are not irrelevant
to a full prediction.

A deliberately extreme translation stress at 1e6 coordinate units produced
roughly 0.3–0.57 units of error against FP64 centering even when SVD and Jacobi
returned identical FP32 coordinates. That identifies shared FP32 arithmetic
in that synthetic stress, not the cause of the measured GPU-model difference.
No covariances from those differing full-model samples have yet been examined
by this diagnostic. There is no proven production-solver bug to fix from the
synthetic panel, and it does not authorize more sweeps, FP64 inference or a
relaxed accuracy threshold as a new default.

## Reproduce or inspect captured covariances

```bash
python -m bench.esmfold2_alignment_probe \
  --samples-per-family 32 --seed 17 --output /tmp/esm-alignment-synthetic
python -m bench.esmfold2_alignment_probe \
  --input captured-covariances.npz --output /tmp/esm-alignment-captured
```

The command forces CPU execution. Output must be a new directory.
`report.json` records convergence, objective, determinant/orthogonality errors,
the eigenvalue gap, source hashes and the scale/translation panel.
`arrays.npz` retains each covariance, rotation, label and per-case metric.
Reports never approve a full-model default.

A capture NPZ must contain `correlation[..., 3, 3]` in FP32 or FP64, optionally
with one string `labels` entry per matrix. Leading axes are flattened in order;
labels can retain diffusion step and sample identity. The oracle consumes the
same FP32-rounded covariance as production and cannot recover lost input bits.
Loading an NPZ does not verify that its values came from a real model run;
retain the capture's input, checkpoint, source and step/sample provenance.
The tool does not install a GPU capture hook or execute a model. Obtain actual
covariances immediately before the rotation solve in a separate diagnostic
build; an instrumented capture is not an ordinary timing measurement.

For a differing sample, replay each captured covariance through SVD, the
production Jacobi helper and the sweep/precision panel. If convergence is
already at the FP32 floor, increasing sweeps is not a supported fix. Inspect
small eigenvalue gaps, then replay the same centered source/target arrays
with a fixed denoiser output to localize the first alignment difference.
Compare native input, initial-pair, dropout, MSA and diffusion draws explicitly
before attributing the later trajectory to the solver. Keep within-baseline
reruns and confidence outputs alongside the candidate residuals.

## Cluster handoff: 1024–2048 tokens

Use actual 24 GB and 48 GB cards to investigate their capacities. A fraction
on a 96 GB card does not reproduce a smaller card's module/context demands.
Record device kind, usable capacity/ECC state, allocator environment, input and
weight/source identities, shapes, precision and resolved kernels. Keep one GPU
job per device and a fresh process for each card, token bucket and solver arm.
Use the same explicit fraction for both arms of a comparison, starting from
the product's [allocator policy](configuration.md#allocator-policy). A separate
fraction comparison is a separate experiment.

Inside the cluster GPU allocation, run the capacity ladder independently for
both solvers, retaining a nonzero status and full log when one arm fails:

```bash
mkdir capacity-results
for bucket in 1024 1280 1536 1792 2048; do
  mkdir "capacity-results/$bucket"
  for alignment in svd jacobi; do
    rc=0
    foldjax predict --model esmfold2 --input "jobs/$bucket.json" \
      --msa single --padding --pad-tokens "$bucket" --seed 101 --num-samples 5 \
      --option "rigid_align_backend=$alignment" \
      --output-dir "capacity-results/$bucket/$alignment" \
      >"capacity-results/$bucket/$alignment.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\n' "$bucket" "$alignment" "$rc" \
      >>capacity-results/status.tsv
  done
done
```

The jobs must contain the intended real inputs and fit their padded axes;
never truncate an input to satisfy a bucket. Keep native recycling, diffusion,
precision and other execution settings. Five samples preserve the existing
paired panel; the checkpoint's 32-sample default needs its own capacity check
before claiming a default-run token limit. Record both cold compile/load and
fresh-process warmed-cache execution. A failed baseline does not establish
that the candidate also fails, and a missing result is not a successful run.
Measure total process GPU memory separately from the JAX allocator peak,
recording the sampling interval; post-failure usage cannot prove the peak.

For sizes where both solvers complete, use the existing five-sample paired
probe without `--smoke`:

```bash
python -m bench.esmfold2_performance_probe \
  --input jobs/1024.json jobs/1280.json --experiment alignment \
  --output paired-1024-1280
python -m bench.esmfold2_performance_probe \
  --input jobs/1536.json jobs/1792.json --experiment alignment \
  --output paired-1536-1792
python -m bench.esmfold2_performance_probe \
  --input jobs/1024.json jobs/2048.json --experiment alignment \
  --output paired-1024-2048
```

Each size has baseline, candidate and baseline-repeat workers plus a separate
shared-tape preparation worker. Repeating 1024 in the last pair is an explicit
control imposed by the tool's two-size interface. This core probe uses native
unpadded features and an explicit diffusion tape; that tape cannot be combined
with serving prefix-RNG padding. Its capacity is therefore not interchangeable
with the padded CLI capacity ladder. Report actual shapes for both, rather
than calling the two graphs identical.

Keep per-entity same-index RMSDs after one global proper fit, raw confidence
errors and repeat floors. Separately evaluate the normal output artifacts
against deposited structures with `foldjax compare OUTPUT_DIR --reference reference.cif
--metrics lddt,rmsd_ca`, auditing atom/entity matching first. Reference lDDT is
a different question from same-index arm agreement; a high lDDT cannot erase a
failed paired diagnostic. If no matched reference exists, record lDDT as
unmeasured. Keep the fixed criteria and report failures without selecting a
new default from a successful bucket alone.
