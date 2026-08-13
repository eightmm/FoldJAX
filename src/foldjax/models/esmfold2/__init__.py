"""ESMFold2 in JAX.

A reimplementation of Biohub's ESMFold2 (MIT), carried the same way as the
other ports here: upstream's released weights, this project's JAX code, and
per-module parity gates against the torch modules that define correctness.

The port is in progress. `foldjax.backends.esmfold2` currently drives the torch
model; this package replaces it module by module, and each one lands only once
`tests/models/esmfold2/` shows it matching torch on real weights.
"""
