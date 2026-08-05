# OpenFold3

OpenFold3 is carried inside FoldJAX, at `foldjax.models.openfold3`. `uv sync`
installs it with everything else, and predicting needs no extra and no sibling
checkout:

```bash
foldjax predict --model openfold3 --input job.yaml
```

It was external until it was vendored, and this file used to explain how to
install it separately. That is no longer necessary — the reason it was excluded
was that `openfold3-jax` is published on no package index, which vendoring
removes. AlphaFold 3 stays external for a different reason that vendoring cannot
remove: its parameters may not be redistributed, and upstream pins an
incompatible CUDA generation. See [alphafold3.md](alphafold3.md).

## What it costs to add

Nothing, at the dependency level. The inference path needs `jax`, `numpy`,
`safetensors` and `gemmi`, all of which the other four ports already required,
so vendoring OpenFold3 added no base dependency.

## Weights

The published checkpoint is `of3_ft3_v1.pt`, and the code *and* weights are
Apache-2.0 with the training data published — this is the most permissively
licensed model FoldJAX carries. The **repository is access-gated**, though: even
its example config returns HTTP 401 unauthenticated, so FoldJAX cannot fetch it
for you.

```bash
# Request access at https://huggingface.co/OpenFold/OpenFold3, then:
foldjax home                                        # where FoldJAX keeps things
mkdir -p ~/.cache/foldjax/weights/openfold3         # or $FOLDJAX_HOME/weights/...
cp of3_ft3_v1.pt ~/.cache/foldjax/weights/openfold3/
foldjax weights list                                # openfold3 now reads `ready`
```

`foldjax weights fetch --model openfold3` reports that the parameters are
released only on request and names the directory to put them in, rather than
failing with a bare 401.

Nothing is converted: the port reads the released checkpoint directly. A `.pt`
is a torch file, so loading one needs `--extra torch-bridge`. A `safetensors`
export loads without torch and can be passed to `--weights` instead.

## Featurization is the one part that is not self-contained

Every other port in FoldJAX featurizes in-package. OpenFold3 delegates to
upstream's own data pipeline, on purpose: it is exact where a reimplementation
would be a guess, and it now serves as the reference a future torch-free
featurizer would be gated against.

That needs two things prediction does not:

```bash
uv sync --extra cuda13 --extra openfold3-preprocess    # upstream's data-side deps
```

and upstream's own code, which is a **checkout rather than a package** — it is
on no index either, and unlike the port it cannot be vendored without carrying
its whole data stack. `foldjax.models.openfold3.upstream` looks for it beside
the FoldJAX repository and honours `$OPENFOLD3_SOURCE`:

```bash
git clone https://github.com/aqlaboratory/openfold-3 ../openfold3
# or
export OPENFOLD3_SOURCE=/path/to/openfold-3
```

Missing either one raises with both remedies named rather than a bare
`ModuleNotFoundError`.

## Sampling knobs

`--num-samples`, `--num-steps` and `--num-recycles` all reach it, translated to
`num_samples`, `no_rollout_steps` and `num_cycles`.

**`--max-msa-depth` is refused**, and that is deliberate: OpenFold3 exposes no
MSA-depth argument, so accepting the flag would mean silently ignoring it. It is
the only sampling knob any backend refuses.

Backend options: `query_id`, `ccd_file_path`, `pair_chunk_size`, `prefix`,
`no_compile`, and the three native sampling spellings.

`pair_chunk_size` is worth knowing about. Triangle attention's scores are
`[rows, heads, N, N]` — cubic in token count — so one *unchunked* pair block
needs 267 GiB of temporaries at 2076 tokens against 41 GiB at 256 rows.
`released_config` picks a chunk from the token count, and because the chunk
shrinks as the target grows, peak memory rises more slowly than the square of
the token count: 966 to 1494 tokens is 1.55x the tokens and only 1.44x the peak.

## Alignments are selected by filename

OpenFold3 identifies an alignment's source from the file's **stem** and ignores
a name it does not recognise, and the stem also chooses the row cap:
`colabfold_main` keeps 16,384 rows, `colabfold_paired` 8,192, `uniref90_hits`
10,000, `mgnify_hits` 5,000 (`dataset_config_components.py:80`).

A job that names its alignment after the target — `t0128.a3m`, which is what
every other backend here accepts — would therefore be silently dropped. The
translation links it into the generated inputs directory under an accepted stem
instead, one directory per chain so two chains cannot claim the same name:

```
<output>/inputs/msa/A/colabfold_main.a3m -> /path/to/t0128.a3m
```

An alignment that already carries a recognised stem is passed through untouched,
since upstream then knows its row cap. `tests/test_input_openfold3.py` covers
both, plus the two-chain collision and the suffix.

## The compile cache

Compiling this architecture is the dominant cost and grows with length — minutes
at a few hundred tokens, where XLA prints its own slow-compile warning. FoldJAX's
persistent cache is on by default and namespaced per model, weight identity and
compile-relevant options, so it is paid once per shape rather than once per
process.

The backend used to ignore that cache, which made it the one model least able to
afford doing so. It no longer does.

## State of the port

The whole latent trunk — Pairformer stack (AF3 Alg. 17), MSA module stack
(AF3 Alg. 8) and template pair stack (AF2 Alg. 16) — plus the diffusion sampler
and confidence heads are ported and gated against the real upstream torch
modules at `rtol=atol=1e-4`. Parity is checked at three levels: per layer,
against upstream's own *composed* code (`run_trunk`, `DiffusionModule.forward`,
`SampleDiffusion.forward`, `AuxiliaryHeadsAllAtom.forward`), and `predict()`
against `OpenFold3.forward` as a single call. At the released settings the worst
disagreement is 2.1e-5 on coordinates whose maximum is 35.0.

Every parameter of the released model is read by the inference path — 416 of 416
unique tensors — measured by a test that fails if any group goes unread, which is
how three unwired subsystems were originally found.

On real 1UBQ, from the query sequence alone with no alignment and no template,
the five samples land 2.52–2.79 Å CA RMSD from the deposition. Real targets up
to 2076 tokens run.

### What is not established

- **No comparison against upstream OpenFold3 as a program.** `bench/spec.py`
  lists `openfold3` in `MODELS` but not in `REIMPLEMENTED`, because
  `run_upstream.command` has no branch for it and no upstream virtualenv has
  been provisioned here. The FoldJAX side can be measured today; the column that
  would make it a comparison cannot.
- **Structural accuracy beyond 1UBQ** is unmeasured — no RMSD or clash gate
  across a target set.
- **Precision** is fp32 with matmul precision pinned to `highest`. Upstream also
  runs `bf16-mixed`, which is numerically different by design.
- **Upstream's memory-optimization and kernel paths** (chunked forward,
  `forward_offload`, DeepSpeed/cuEq/Triton/LMA) are not ported. Upstream asserts
  they are numerically equivalent; that is upstream's claim, not a measurement
  here.
- **MSA row choice above 1024 rows** cannot be reproduced without an explicit
  permutation, because upstream draws it with `torch.randperm`.
- **Batch size** is always 1. Nothing assumes it; nothing measures it either.
- **A torch-free featurizer**, which is what would make this port as
  self-contained as the other four.
