# FoldJAX documentation

Start with the quickstart, then the tutorial closest to your job. The
reference pages are complete but long; the guides link into them where a
detail matters.

## Getting started

| page | what it covers |
|---|---|
| [Quickstart](quickstart.md) | install, `foldjax doctor`, weights, a first fold, and reading what it wrote |
| [Installation](install.md) | CUDA 12/13, CPU-only machines, pip, the container image, extras |
| [Which model?](faq.md#which-model-should-i-use) | the six models side by side, and the FAQ around them |

## Tutorials

| page | what it covers |
|---|---|
| [Heteromer with an MSA search](tutorials/heteromer-msa.md) | `--msa auto`, complex pairing, `foldjax msa prefetch`, alignment depth and Neff |
| [Protein-ligand with constraints](tutorials/protein-ligand.md) | pocket and contact constraints, Boltz-2 affinity, which model reads what |
| [Glycans](tutorials/glycan.md) | multi-residue ligands and their covalent bonds |
| [Templates](tutorials/templates.md) | your own template files, `--templates auto`, private template folders |
| [Batches and Slurm](tutorials/batch-slurm.md) | multi-job files, screens, padding, `--shard`, array jobs |
| [Resuming and failures](tutorials/resume.md) | `--resume`, `--keep-going`, what makes a run rerun |

## Guides

| page | what it covers |
|---|---|
| [Outputs](outputs.md) | the output tree, `foldjax_run.json`, `confidence.json`, confidence arrays, cost |
| [Configuration](configuration.md) | the store under `FOLDJAX_HOME`, every `FOLDJAX_*` variable, compile-cache trust |
| [FAQ](faq.md) | out of memory, CPU-only and macOS, AlphaFold 3 weights, rerun differences |
| [Colab](colab.md) | what the notebook does and does not do |
| [Structure alignment](alignment.md) | placing several models' structures in one frame |

## Models

| page | model |
|---|---|
| [alphafold3.md](alphafold3.md) | AlphaFold 3: the runtime build, user-supplied weights, kernels |
| [boltz2.md](boltz2.md) | Boltz-2: affinity, constraints, MSA and run options |
| [esmfold2.md](esmfold2.md) | ESMFold2: language-model folding, random by design |
| [opendde.md](opendde.md) | OpenDDE: the general and antibody-antigen checkpoints |
| [openfold3.md](openfold3.md) | OpenFold3: OpenBind weights, Torch-free featurization, pockets |
| [protenix.md](protenix.md) | Protenix: weight profiles, v2, constraints, templates |

## Reference

| page | what it covers |
|---|---|
| [Command line](cli.md) | every `foldjax` command and flag, the output contract, memory, native options, weights, the compile cache |
| [Input](input.md) | every input format, alignments, templates, constraints, glycans, affinity |
| [Python API](python-api.md) | requests, results, sessions, events |
| [Common model interface](model-interface.md) | `foldjax.get_model(name)`: `embed`, `encode`, `predict` |
| [Padding profiles](token-padding-profiles.md) | token and MSA buckets for shape reuse |
| [Context parallelism](context_parallel.md) | running one prediction over several GPUs |
| [Model versions](model-versions.md) | exact upstream targets, checkpoint hashes, parity evidence |
| [Licences](licences.md) | parameter terms per model |
| [Benchmark](benchmark.md) | FoldJAX against each upstream: time and memory |
| [Engineering notes](engineering-notes.md) | how the memory, speed and precision defaults were reached |
| [CPU parity subset](parity-cpu.md) | what the stored-capture replay certifies |
| [Recycling defaults](recycling-defaults.md) | the superseded paper-schedule record |

[Porting records](ports/README.md) and the [archive of dated engineering
notes](archive/README.md) are history: the evidence behind a default, not
instructions.
