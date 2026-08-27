"""Pull the atom-encoder tensors out of `transformers`' re-exported ESMFold2.

The published checkpoint FoldJAX loads and the one `transformers` loads hold
bit-identical weights under different names, so the parity test drives each
side with its own artifact and never maps a name onto another. This fetches the
library's side.

The re-export is 24.5 GB, almost all of it the language model. safetensors puts
byte offsets in a header, so the 28 atom-encoder tensors come out with range
requests: 3.6 MiB rather than the 4.7 GiB shard they live in.

    python tests/models/esmfold2/fetch_hf_atom_weights.py <destination>
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = "biohub/ESMFold2-hf"
SHARD = "model-00001-of-00006.safetensors"
PREFIX = "input_embedder.atom_encoder."
_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def _range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "foldjax-parity"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def fetch(destination: Path) -> Path:
    url = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
    length = struct.unpack("<Q", _range(url, 0, 8))[0]
    header = json.loads(_range(url, 8, 8 + length))
    base = 8 + length
    tensors = {}
    for name, entry in sorted(header.items()):
        if not name.startswith(PREFIX):
            continue
        start, end = entry["data_offsets"]
        raw = _range(url, base + start, base + end)
        tensors[name] = np.frombuffer(raw, dtype=_DTYPES[entry["dtype"]]).reshape(
            entry["shape"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **tensors)
    return destination


if __name__ == "__main__":
    out = fetch(Path(sys.argv[1] if len(sys.argv) > 1 else "atom_encoder.npz"))
    print(f"wrote {out}")
