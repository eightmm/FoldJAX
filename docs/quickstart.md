# Quickstart

From a clean machine to a predicted structure you can open, with Boltz-2:
install, check the install, fetch one model's weights, fold one sequence, read
what was written. The download (about 6.2 GB) and the first compile take most
of the time; everything else is seconds.

## 1. Install

FoldJAX needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/eightmm/FoldJAX
cd FoldJAX
uv sync                      # a CUDA 13 GPU machine
```

On a machine without an NVIDIA GPU, opt out of the default `gpu` group for the
whole shell, because every `uv run` re-syncs the default groups:

```bash
export UV_NO_DEFAULT_GROUPS=1
uv sync
```

Without a checkout, pip installs the same package (no extra is the CPU build;
add `[cuda13]` or `[cuda12]` on a GPU machine); every `uv run foldjax` below
is then just `foldjax`:

```bash
pip install 'foldjax @ git+https://github.com/eightmm/FoldJAX@v0.1.0'
```

CUDA 12, a container image and the optional extras are in
[install.md](install.md).

## 2. Check the install

```bash
uv run foldjax doctor
```

`doctor` prints the FoldJAX and JAX versions, the accelerator JAX sees, where
the store is, which weights are present, and which MSA and template search is
configured. On a GPU machine the `jax` line should name a `cuda` device; if it
says `cpu`, see [the FAQ](faq.md#foldjax-runs-on-the-cpu-although-the-machine-has-a-gpu).
Every model still runs on a CPU, slowly.

## 3. Fetch weights

```bash
uv run foldjax weights fetch --model boltz2
```

This downloads the released Boltz-2 checkpoints, checks their size and
SHA-256, and converts them once into `$(foldjax home --path weights)/boltz2`.
`uv run foldjax setup` does the same for every model whose weights are public
(Boltz-2, OpenDDE, OpenFold3, Protenix); ESMFold2 is opt-in for its size, and
AlphaFold 3's parameters you request from Google yourself
([alphafold3.md](alphafold3.md#weights)). Where the files go is
[configuration.md](configuration.md#the-store).

## 4. Fold a sequence

```bash
uv run foldjax predict --model boltz2 --name first_fold --msa single \
    --sequence MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE
```

- `--msa single` folds from the sequence alone. Without an alignment the job
  would be refused, because every model but ESMFold2 is meant to run with one.
  `--msa auto` searches for an alignment instead, and **sends the sequence to
  the public ColabFold MMseqs2 server**; for anything you care about, use it or
  give the chain an `unpaired_msa` file
  ([heteromer tutorial](tutorials/heteromer-msa.md)).
- `--name first_fold` names the job, and so the output directory:
  `foldjax-outputs/first_fold`.
- Boltz-2 seeds nothing by default upstream, so FoldJAX draws a seed, prints
  it and records it.

Stage timings go to stderr and the result to stdout: a summary table in a
terminal, JSON when stdout is a pipe or a file (`--json` asks for JSON in a
terminal too). The compiled program is cached under the store, so the same
shape does not compile again. On a GPU the first run of a shape spends most of
its time compiling, and a rerun takes a fraction of it; on a CPU executing the
model dominates, and a rerun takes about as long.

To check a request without running it (the weights must be installed, but
nothing loads and nothing is written):

```bash
uv run foldjax plan --model boltz2 --name first_fold --msa single \
    --sequence MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE
```

## 5. Read the outputs

```bash
uv run foldjax show foldjax-outputs/first_fold
```

```
foldjax-outputs/first_fold/
├── foldjax_run.json                                  the run manifest: input, weights, seed, options, cost
├── seed-<seed>_sample-00/
│   ├── first_fold_seed-<seed>_sample-00.cif          the structure; pLDDT in the B-factor column
│   ├── confidence.json                               the model's scores, plus a common summary
│   ├── confidence_full.npz                           PAE, PDE, per-token pLDDT, chain-pair ipTM
│   └── predicted_aligned_error.json                  PAE in AlphaFold DB's format, for viewers
├── inputs/                                           the job as translated for Boltz-2
├── msa/, predictions/, processed/                    Boltz-2's own working files, where it wrote them
└── .foldjax.lock                                     held while a run writes here; left in place on purpose
```

Open the `.cif` in PyMOL, ChimeraX or Mol\*; colouring by B-factor shows
pLDDT. In `confidence.json`, `summary.plddt`, `summary.ptm` and
`summary.ranking` are the numbers to read first; `scores` keeps Boltz-2's own
names. Everything in these files is described in [outputs.md](outputs.md).

The same run from Python, into a directory of its own. Boltz-2 draws a new
seed each time, so writing into `foldjax-outputs/first_fold` again would put a
second run beside the first (FoldJAX warns when that happens):

```python
from foldjax import Job, PredictionRequest, predict

sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE"
job = Job.from_sequences(protein=(sequence,), name="first_fold")
result = predict(
    PredictionRequest(
        model="boltz2",
        input=job.store(),
        msa="single",
        output_dir="foldjax-outputs/first_fold_python",
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

## Next

- Put the job in a file: [input.md](input.md) has the format; `examples/`
  has three ready-made jobs.
- Another model: `--model protenix`, `--model openfold3`, or several at once
  (`--model boltz2 protenix openfold3`); [which model](faq.md#which-model-should-i-use).
- A complex, a ligand, a glycan, templates or a screen: [the tutorials](README.md#tutorials).
