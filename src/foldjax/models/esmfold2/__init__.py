"""ESMFold2 in JAX.

A reimplementation of Biohub's ESMFold2 (MIT), carried the same way as the
other ports here: upstream's released weights, this project's JAX code, and
per-module parity gates against the torch modules that define correctness.

Both halves are ported. `models/` holds the 235M-parameter structure network --
the atom stack, the linear-recurrence pair trunk, the diffusion head and its
sampler, and the confidence head -- and `models/esmc.py` holds ESMC-6B, the
protein language model whose 81 hidden states the trunk folds. `data/` builds
features and writes structures, and `bridge/` reads the two checkpoints.
Publisher-runtime feature conversion lives with its parity tests, outside the
installable package; the production tree contains only the NumPy/JAX path.

Where the port and upstream diverge, it is stated at the point of divergence
rather than here; `docs/ports/esmfold2/` records what the torch source does
and where it is easy to read wrongly.
"""
