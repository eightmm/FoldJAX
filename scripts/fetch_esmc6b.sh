#!/usr/bin/env bash
# Fetch ESMC-6B into the ESMFold2 weight store.
#
# The 25.4 GB language model ESMFold2 folds representations from. Upstream
# distributes it apart from the 940 MB structure weights, so the store's
# `esmfold2` entry does not pull it; `<weights>/esmc` is where the loader looks.
set -euo pipefail

REPO="biohub/ESMC-6B"
TARGET="${1:-$(uv run python -c 'from foldjax.paths import weights_dir; print(weights_dir("esmfold2") / "esmc")')}"

mkdir -p "$TARGET"
# `hf download` resumes, verifies each file's hash, and parallelises the six
# shards; a curl loop would do none of those.
uv run hf download "$REPO" \
  --local-dir "$TARGET" \
  --include "config.json" "model.safetensors.index.json" "*.safetensors"

echo "ESMC-6B in $TARGET"
du -sh "$TARGET"
