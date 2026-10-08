"""Regenerate ``fixtures/substructure_upstream.npz`` from upstream Protenix.

Run with the upstream checkout's own interpreter (it needs torch, which the
FoldJAX environment does not have), from the FoldJAX checkout root::

    CUDA_VISIBLE_DEVICES= PROTENIX_ROOT_DIR=../protenix \\
        ../protenix/.venv/bin/python \\
        tests/models/protenix/scripts/substructure_upstream_fixture.py

One upstream ``ConstraintEmbedder`` in the base-constraint configuration --
all four embedders enabled, the substructure one in transformer mode
(``configs_model_type.py:117-135``, ``configs_base.py:293``) -- at small
widths, its zero-initialised weights replaced by seeded random ones, on CPU
in eval mode. Records its ``state_dict`` (``param/<key>``) and two forwards:

* ``off_*``: the all-zero feature maps upstream's inference featurizer
  attaches to a job without a constraint (``json_to_feature.py:363-378``);
* ``pocket_*``: pocket and token-contact maps set, the substructure map still
  zero, as every inference job has it (``constraint_featurizer.py:378-390``).

``substructure_out`` is the bare ``SubstructureEmbedder`` on the zero map.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

PROTENIX = Path(os.environ.get("PROTENIX_ROOT_DIR", "../protenix")).resolve()
sys.path.insert(0, str(PROTENIX))

from protenix.model.modules.embedders import ConstraintEmbedder  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "substructure_upstream.npz"

N_TOKEN = 5
C_Z = 4
HIDDEN = 8  # nn.TransformerEncoderLayer is built with nhead=4
N_LAYERS = 2  # the checkpoint has one; two exercises the layer recursion


def main() -> None:
    torch.manual_seed(0)
    module = ConstraintEmbedder(
        pocket_embedder={"enable": True, "c_z_input": 1},
        contact_embedder={"enable": True, "c_z_input": 2},
        contact_atom_embedder={"enable": True, "c_z_input": 2},
        substructure_embedder={
            "enable": True,
            "n_classes": 4,
            "architecture": "transformer",
            "hidden_dim": HIDDEN,
            "n_layers": N_LAYERS,
        },
        c_constraint_z=C_Z,
        initialize_method="zero",
    )
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) * 0.5
            )
    module.eval()

    zeros = {
        "pocket": torch.zeros(N_TOKEN, N_TOKEN, 1),
        "contact": torch.zeros(N_TOKEN, N_TOKEN, 2),
        "contact_atom": torch.zeros(N_TOKEN, N_TOKEN, 2),
        "substructure": torch.zeros(N_TOKEN, N_TOKEN, 4),
    }
    pocket = dict(zeros)
    pocket["pocket"] = torch.zeros(N_TOKEN, N_TOKEN, 1)
    pocket["pocket"][0:2, 3, 0] = 6.0
    pocket["contact"] = torch.zeros(N_TOKEN, N_TOKEN, 2)
    pocket["contact"][1, 4] = pocket["contact"][4, 1] = torch.tensor([0.0, 8.0])

    record: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, features in (("off", zeros), ("pocket", pocket)):
            for channel, value in features.items():
                record[f"{name}_{channel}"] = value.numpy()
            record[f"{name}_out"] = module(features).numpy()
        record["substructure_out"] = module.substructure_z_embedder(
            zeros["substructure"]
        ).numpy()
    for key, value in module.state_dict().items():
        record[f"param/{key}"] = value.numpy()
    np.savez_compressed(OUT, **record)
    print(f"wrote {OUT} ({len(record)} arrays, torch {torch.__version__})")


if __name__ == "__main__":
    main()
