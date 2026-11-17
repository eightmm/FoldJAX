# ESMFold2

The one carried model that is not an evolutionary-trunk predictor, and the
one whose two runs of the same job differ on purpose.

ESMFold2 is the odd one out and worth knowing about before you use it. It has
no evolutionary trunk: it is a diffusion structure head on a linear-recurrence
pair trunk, folding the representations of **ESMC-6B**, a 25 GB protein
language model that upstream distributes separately (`foldjax weights fetch
--model esmfold2` puts the complete bundle where the loader looks). It is also
*random at inference by design* —
random initial pair state, per-loop language-model dropout, sampler noise — so
two seeds give genuinely different structures rather than the same structure
twice. **[Upstream ESMFold2](https://github.com/Biohub/esm/blob/main/esm/models/esmfold2/prepare_input.py)
is an all-biomolecule model**: its released input pipeline supports proteins,
DNA, RNA, small-molecule ligands, and modified residues. FoldJAX ports that
input contract without Torch: standard polymers, CCD and SMILES ligands,
modified residues, and explicit covalent bonds are featurized in NumPy from
the verified Biohub CCD snapshot (`data/all_atom.py`), which `foldjax predict`
uses for every job including plain protein chains. The standalone
`foldjax.models.esmfold2.inference.build_job_features` (`data/features.py`)
helper remains available as a CCD-free in-package protein-only API using
`PROTEIN_REF_POS` conformers.

The complete released ESMFold2 profile is the compatible default. To run the
explicit structure-network-only variant without downloading ESMC-6B, fetch its
1.36 GB structure+chemistry profile and select the same profile for prediction:

```bash
foldjax weights fetch --model esmfold2 --profile structure-only
foldjax predict --model esmfold2 --input protein.json \
  --profile structure-only
```

`--profile structure-only` sets `no_language_model=true` itself; conflicting
native options are rejected instead of silently overriding the profile.
The older explicit option still selects that managed profile automatically
when weights are omitted. Without either spelling, FoldJAX still requires the complete
structure+ESMC release and never silently falls back to a different model.
An explicit `esmc_weights=/path/to/esmc` also selects the 1.36 GB managed
profile: the released language-model branch still runs, but its weights come
from that path rather than a redundant managed ESMC copy.

## Buffered language-model states (experimental)

`--option lm_state_storage=buffer` stores each ESMC layer directly into one
donated buffer in structure-token layout. It avoids retaining the full layer
list and then allocating a full stack, transpose and token gather. Every layer,
the original dtype promotion, block arithmetic and final softmax-weighted layer
combination remain unchanged. It does not enable compiled ESMC blocks or change
SVD alignment. The default remains `stack` pending full-model GPU comparison.

```bash
foldjax predict --model esmfold2 --input job.yaml \
  --option lm_state_storage=buffer
```

In the reported 4090 GFP complex experiment, both SVD and Jacobi failed before
alignment, while allocating the stacked language-model states. This explains
why changing the solver did not raise that experiment's capacity. It does not
identify the peak of every input or of the buffered candidate. CPU component
tests preserve layer and compact-embedding bits; actual GPU peak, full-model
accuracy and 1280/1536/2048-token success remain unmeasured.

Storage is outside the structure JIT's ABI, so the option retains its existing
compilation namespace and model executable key. The buffer-update helper has
its own JAX key. Default executions, including buckets up to 1024, retain the
old stack path. The later cluster comparison found exact cold/warm agreement
within every process, but large compact-embedding differences between some
fresh processes in both storage modes. This does not isolate a buffer defect
or establish a kernel cause. Neither storage mode is promoted from that panel.
See [the repeated-process controls and structural assessment](esmc-process-variation-2026-10-21.md),
[the earlier process diagnostic](cluster-memory-2026-10-20.md) and
the earlier [measurement follow-up](cluster-memory-followup.md)
for capacity tests and the separate five-sample `lm-storage` paired experiment.

## Denoising the samples one at a time

`--option structure_sample_sequential=true` runs the diffusion denoiser once
per sample instead of once for all of them. It is off by default.

What it narrows is one tensor. The diffusion token transformer builds attention
logits of shape `[samples, tokens, tokens, heads]` in float32, inside every
block of every denoiser call, and at the released 16 heads and 3,012 tokens
that is 2.7 GiB at five samples and 17.3 GiB at thirty-two. Sequencing divides
it by the sample count, and narrows the diffusion cache's per-atom conditioning
by the same factor. The blocked atom attention carries a sample axis as well,
but it is linear in atoms rather than quadratic in tokens, so this one tensor
is most of the saving.

What it does not narrow is the trunk, and that matters more than the saving
does. ESMFold2's pair trunk has no sample axis in this port at all: the
recurrence and the coda return one `[tokens, tokens, d_pair]` state, and the
sample count first appears in the structure head. The 45 GiB peak measured at
2,096 residues and the failure at 3,012 are trunk pair tensors, so a run on the
released five-sample schedule should not expect this option to move them. It
earns its place when a caller raises the sample count, not when a long input
will not fit.

The cost is one launch per sample per step in place of one batched launch, and
arithmetic that is the same arithmetic on a narrower array rather than a
bitwise-identical one: a reduction over one row may order itself differently
than over thirty-two. The random stream is untouched. Every draw the sampler
makes -- the initial noise, the per-step augmentation and translation, the
churn normal -- still happens once per step at the full batched width from the
same key, so each sample gets the same noise it would have got batched. The
confidence head's own `confidence_sample_sequential` is a separate stage and
stays on; either may be set without the other. A request that puts more than
one input in a single call is refused rather than run, because telling a
rollout which input it belongs to would mean slicing the pair conditioning
inside the loop this option exists to narrow.

## Expected errors

Every sample's expected PAE and PDE matrices, `[tokens, tokens]` in angstroms,
are returned by default and written to its `confidence_full.npz` and
`predicted_aligned_error.json`. `--option return_expected_errors=false` stops
them being outputs of the compiled program: the head still computes them, so
coordinates, pLDDT, pTM, ipTM and chain-pair ipTM come from the same model, and
the npz records why `pae` and `pde` are absent. They do not come from the same
program, though, and are not bitwise equal: the two programs round differently
and ESMFold2's structure head is chaotic enough to amplify that. On a GPU at
3,012 tokens each mode reproduced itself bitwise, while the same seed's samples
moved 0.55-3.54 A CA RMSD between the modes (0.015-0.36 A at 2,096 tokens) and
the median TM-score to the deposited structure by at most 0.007. Compare the two
as two draws, not as one result. That saves `2 x samples x tokens^2 x 4` bytes of
program output -- 2.2 GiB at 3,012 tokens and the checkpoint's 32 samples,
derived rather than measured -- and the host copies the writers make. It is a
separate compiled program with its own cache namespace; an omitted option and
`true` share the released one. See
[`docs/cli.md`](cli.md#--option-return_expected_errorsfalse-esmfold2-openfold3).

## Solver-free rigid alignment (experimental)

`--option rigid_align_backend=jacobi` replaces the denoiser's 3-by-3 SVD and
determinant with a fixed-size quaternion Jacobi solver. It uses scalar JAX
arithmetic, keeps float32 centering and coordinate products, and returns a
proper rotation, including reflection correction, without a cuSolver call.

The default remains `svd`. Component tests compare against the previous FP32
implementation and an FP64 SVD oracle, but do not establish full-model accuracy
or a higher token limit. Degenerate inputs can have multiple optimal rotations.
The [paired GPU measurements](performance-diagnostics.md#esmfold2-solver-scope)
at two sizes did not support changing the default; external CUDA workspace and
24 GB capacity still need measurement. Use the same explicit option for warming
and prediction; Jacobi has its own compilation profile.

## Managed memory staging

For one semantic input, including a multi-seed sweep of that input, FoldJAX's
managed predictor does not need ESMC-6B and the structure network resident at
the same time. It first loads ESMC with only the small structure-side projection
that combines its hidden layers, synchronizes the resulting compact embedding,
releases ESMC's device arrays, and then loads the structure network. This changes
buffer ownership only; the input, seed, model values, and generated structures
are unchanged.

A batch containing different inputs deliberately keeps the complete model
resident and reuses it. Staging such a batch would trade the memory saving for
re-reading the 25.4 GB ESMC checkpoint for every input. Direct calls to
`foldjax.models.esmfold2.inference.load` likewise still return the complete
model; staging is confined to FoldJAX's managed request session.

## Pocket constraints

Upstream ESMFold2 has no pocket or restraint input, so a common job's pocket
constraint is refused. `--option pocket_sampling=select` is the one way it
counts: FoldJAX's own route, which scores every finished sample against the
pocket with Boltz-2's rule (heavy atoms of the binder within `max_distance`
of each listed residue) and ranks a satisfying sample first by this port's
pLDDT ordering. The sampler and the structures are unchanged; `max_distance`
is required, since upstream has no default of its own
([cli.md](cli.md#--option-pocket_samplingselect)). With ESMFold2's sampling
spread, several samples or seeds give the selection something to choose from.

## Deviations from upstream

* **Release checkpoints only (`type="release"`).** Upstream dispatches
  `type="experimental"` in `config.json` to a separate
  `ESMFold2ExperimentalModel` architecture. FoldJAX ports `ESMFold2Model`
  (`type="release"`) only and rejects non-release checkpoint configs at load
  time.
* **Context-parallel pair-stack mixed precision.** On single-device runs with
  `trunk_dtype="bfloat16"`, `native_pair_autocast` keeps `float32` residual
  streams and `LayerNorm` statistics across `inputs_embedder`, `msa_encoder`,
  `folding_trunk`, `parcae` discretization, `parcae_coda`, and
  `confidence_head.folding_trunk`. When `cp_devices > 1`, `native_pair_autocast`
  is disabled on those sharded pair stacks so distributed pair highways stay in
  `bfloat16` rather than carrying full `float32` pair residuals; `lm_encoder`
  retains `native_autocast=True`, Cannon triangle contractions accumulate in
  `float32`, and `bfloat16` Linear / SwiGLU blocks accumulate GEMMs and `silu`
  in `float32`.
* **Shape-specialized CUDA pair-attention softmax.** Under single-device CUDA
  `native_pair_autocast`, `models/native_softmax.py` dispatches to a custom
  Pallas/Triton softmax kernel only for the exact logit shape `(1, 437, 437, 8)`
  and uses `jax.nn.softmax` for every other token count (~1 FP32 ULP rounding
  difference).
* **Blocked atom-attention prefix precondition.** The blocked sliding-window
  atom attention (`atom._windowed_attention`) assumes valid atoms form a
  contiguous prefix of `atom_attention_mask`. `inference.predict` checks that
  invariant and automatically falls back to the exact dense packed-rank window
  (`atom_rows_per_block=0`) when internal mask holes are present; direct
  low-level callers of `model.predict` or `atom.swa_attention` with non-prefix
  atom masks must pass `rows_per_block=0` (or
  `ESMFOLD2_ATOM_ATTENTION_BACKEND=dense`).
* **ESMC-6B execution and FP8.** Upstream supports `esmc_precision="fp8"` via
  TransformerEngine and dispatches ESMC self-attention through
  TransformerEngine, xFormers, or FlashAttention2. FoldJAX is Torch- and
  TransformerEngine-free and runs ESMC-6B in pure XLA (`bfloat16` default,
  `float32` optional). Across ESMC's 80 unnormalized `bfloat16` residual
  blocks, ~1 BF16 ULP kernel differences yield valid alternative ensemble
  draws within natural multi-seed spread rather than bitwise Torch identity.
* **Structure B-factors (per-atom `0–100` vs upstream token `0–1`).**
  Upstream's protein-only `output_to_pdb` helper writes token-level `plddt` on
  `0–1` into every atom of a residue. FoldJAX writes per-atom
  `plddt_per_atom * 100` (`0–100`) into both `.cif`
  (`_atom_site.B_iso_or_equiv`) and `.pdb` B-factors in accordance with the
  cross-model output contract (`docs/outputs.md`).
* **Sample ranking vs summary `plddt`.** Upstream `output_to_pdb` formats a
  single sample without ranking. FoldJAX orders multi-sample outputs by
  valid-token mean `plddt` (`scores.plddt`), while `summary.json` records
  atom-weighted `complex_plddt * 100` as `plddt` alongside `ranking_score`
  (`docs/outputs.md`).
* **Modified nucleic-acid featurization details.** Three `data/all_atom.py`
  choices for CCD-built modified DNA/RNA tokens are retained pending direct
  comparison against Biohub's `prepare_input.py`: modified DNA/RNA tokens map
  to `res_type = 22` (`UNK` / `"DX"` slot) rather than `"DN"` (`26`) or `"N"`
  (`31`); 5'-terminal phosphate atoms (`P`, `OP1`, `OP2`, `OP3`) are stripped
  at `residue_index == 0` while `OP3` is dropped only when
  `residue_index != len(sequence) - 1`; and implicit polymer-backbone
  `token_bonds` entries are added only between adjacent protein tokens (plus
  any explicit covalent bonds).
