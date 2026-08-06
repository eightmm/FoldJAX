"""Upstream AlphaFold 3 files that its wheel does not install.

`run_alphafold.py` is what the AlphaFold 3 backend actually drives: an absl
program at the repository root that builds the data pipeline, loads the
parameters and writes the structure. It is not part of the `alphafold3` package,
so `pip install alphafold3` leaves it behind and a checkout was the only way to
reach it -- which made the backend depend on a sibling directory rather than on
its own installation.

It is Apache-2.0 (the licence beside this file), so carrying a copy is allowed.
The model parameters are a separate matter and are not here: they are obtained
from Google under their own terms and read from the FoldJAX weight store.

Vendored from google-deepmind/alphafold3 at
85c4d20505fd5cef05eac22b534d4e793971ae69. A checkout still wins when one is
present, and `test_the_vendored_runner_matches_upstream` fails if the two drift.
"""
