# Google Colab

What the notebook does for you, and what it deliberately does not: cache
policy, checkpoint handling, and the tutorial schedule it keeps separate
from every model's released defaults.

[Open the notebook](../notebooks/FoldJAX_Colab.ipynb)

The [Google Colab workflow](../notebooks/FoldJAX_Colab.ipynb) detects the active
accelerator runtime and configures its matching JAX stack and kernels around
FoldJAX's common input API.
A compact form accepts protein,
DNA, RNA, CCD ligands, and SMILES ligands. Independent checkboxes expose all six
carried models; the default sends one compact synthetic protein+RNA+ATP job to
Protenix, OpenDDE, Boltz-2, and OpenFold3. ESMFold2 accepts the same protein, DNA, RNA,
ligand, modification, and covalent-bond schema; its complete public
structure+ESMC+chemistry bundle is 26.77 GB. OpenFold3 is selected by default
and always downloads or reuses its managed public v0.5.0 OpenBind checkpoint. AlphaFold 3
accepts a user-supplied parameter directory.
Checkpoint availability is handled first; input compatibility is checked after
the biomolecule form and before prediction. The notebook keeps large model
sessions sequential, records per-model failures without losing successful runs,
and presents each setup, execution, and validation stage as a guided status
card or comparison table rather than a stream of raw notebook logs. Its
interactive structure selector keeps every model, seed, and sample in one
viewer. The final ZIP contains the common input, batch report, score table,
structures, confidence artifacts, and reproducibility manifests for every run.

The notebook installs and verifies the stack for the detected runtime before
showing the biomolecule form and prediction stages. A matching install marker,
verified managed-weight state, and finished prediction manifests make **Run
all** idempotent: later runs skip work that is already complete. Its cache
inventory reports checkpoints, shared assets, generated runtimes, MSA entries,
compilation entries, and predictions without echoing sequence text. The MSA
preflight groups identical chains and shows whether each sequence is disabled,
served by a verified sequence-and-search-provenance cache entry, or will require
a search. Protein MSA results are therefore reused across selected models and
repeated runs. Cache persistence to Drive covers weights, MSA results, and
generated runtime assets; output persistence separately keeps prediction trees
and their resume manifests across VM replacement. Compilation writes only to a
4 GiB local working cache, while selected model/JAX/device/kernel packs are
restored from and synchronized to a private Drive archive capped globally at
2.5 GiB with 30-day LRU retention. TPU execution remains experimental until
every model has real-TPU parity evidence. The notebook shows checkpoint
source/licence/size before downloading and keeps its short
1-sample/20-step/1-recycle tutorial schedule clearly separated from every
model's released defaults.
