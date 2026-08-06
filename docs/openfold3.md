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
store=$(foldjax home | python -c 'import json,sys; print(json.load(sys.stdin)["weights"])')
mkdir -p "$store/openfold3"                         # `foldjax home` prints the rest
cp of3_ft3_v1.pt "$store/openfold3/"
foldjax weights list                                # openfold3 now reads `ready`
```

`foldjax weights fetch --model openfold3` reports that the parameters are
released only on request and names the directory to put them in, rather than
failing with a bare 401.

Nothing is converted: the port reads the released checkpoint directly. A `.pt`
is a torch file, so loading one needs `--extra torch-bridge`. A `safetensors`
export loads without torch and can be passed to `--weights` instead.

## Featurization runs upstream's own pipeline, vendored

Every other port in FoldJAX featurizes with code written here. OpenFold3 runs
upstream's own data pipeline instead, on purpose: it is exact where a
reimplementation would be a guess, and porting it is a separate exercise that
this now serves as the reference for.

Until 2026-08-06 that meant a second repository — `upstream.ensure_importable()`
put a sibling checkout on `sys.path`, so building features required a clone that
`uv sync` could not provide. The pipeline is vendored now:

```
src/foldjax/models/openfold3/_upstream/openfold3/
├── core/data/        # the pipeline: 100 modules
├── core/utils/       # geometry, atomize, logging
├── core/config/
└── projects/of3_all_atom/config/
```

~46k lines of upstream code, verbatim, under upstream's Apache-2.0 and in
upstream's style — `ruff` skips it for the same reason it skips Boltz-2's
vendored featurizer: reformatting would destroy the property that makes
vendoring defensible, which is that the tree stays diffable against the
repository it came from.

**One edit was necessary**, and it is marked in the file.
`core/data/framework/__init__.py` imported its sibling modules by rebuilding each
name out of *path components*:

```python
__import__(".".join(list(path.parts[-6:-1]) + [stem]))
```

That reconstructs `openfold3.core.data.framework.single_datasets.<mod>` only
while the package sits exactly six components deep. Vendored, the last six
components are unchanged but the prefix is not, so every name it built pointed at
a top-level `openfold3` that does not exist here. It now derives the name from
`__package__`, which is equivalent upstream and correct anywhere.

Featurization still needs one thing prediction does not — the extra, because
this is upstream's torch code:

```bash
uv sync --extra cuda13 --extra openfold3-preprocess
```

Everything listed in that extra is a *direct* import of the vendored tree,
checked against it rather than guessed. `deepspeed` is deliberately absent:
upstream guards it with `find_spec` and only uses it to disable a Triton
autotune.

### What the vendoring was checked against

Featurizing 1UBQ both ways — through the vendored tree and through the upstream
checkout, in one process — agrees on **33 of 34 features bit-for-bit**. The
thirty-fourth is `ref_pos`, and the control that matters is that running the
*vendored* pipeline twice disagrees with itself by the same order (8.7 Å against
8.9 Å): the reference conformers are generated per run, so that difference is
nondeterminism inside one implementation rather than a difference between two.

`models/openfold3/upstream.py` survives, but only for the torch-parity gates,
which compare this port against upstream's own *model* modules. Those are the
thing being ported, so carrying a copy would mean diffing the port against a copy
of its own source. Boltz-2, Protenix and OpenDDE make the same trade, and their
parity suites skip without their checkouts too.

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
- **A torch-free featurizer.** The pipeline is in-repository now, so the port is
  as self-contained as the other three; what it is not is torch-free on the
  featurization side, because the vendored pipeline is upstream's torch code.
  Porting it would drop the `openfold3-preprocess` extra entirely, and the
  vendored tree is the reference such a port would be gated against.
