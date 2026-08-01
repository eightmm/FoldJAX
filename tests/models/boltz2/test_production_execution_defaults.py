from __future__ import annotations

import inspect

from foldjax.models.boltz2.models.diffusion.atom import diffusion_transformer_forward
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_module_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.trunk import boltz2_sample_forward


def test_production_layer_stacks_default_to_memory_stable_scan() -> None:
    assert inspect.signature(pairformer_module_forward).parameters["use_scan"].default
    diffusion_scan = inspect.signature(diffusion_transformer_forward).parameters[
        "use_scan"
    ]
    assert diffusion_scan.default
    assert inspect.signature(boltz2_sample_forward).parameters["use_scan"].default
