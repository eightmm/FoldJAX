from __future__ import annotations

import inspect

from foldjax.models.chai.models.trunk import _pairformer_stack, trunk_forward


def test_production_trunk_defaults_to_memory_stable_scan() -> None:
    assert inspect.signature(_pairformer_stack).parameters["use_scan"].default
    assert inspect.signature(trunk_forward).parameters["use_scan"].default
