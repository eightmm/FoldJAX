"""ESMFold2's trunk width is FoldJAX's own choice, and these lock in the facts.

Found 2026-08-28: three places in this repository asserted that upstream
ESMFold2 runs its trunk under a bfloat16 autocast and that the port therefore
matched it. Upstream has exactly one autocast, it wraps the ESM-C call alone,
and the released checkpoint is float32 -- so the port's bfloat16 trunk is a
divergence, and `bench/esmfold2_compare.py` was labelling its torch row with a
hardcoded "bfloat16 (autocast)" string that no run ever produced.

These tests assert the properties rather than the prose: what the settings
actually carry, that nothing can quietly change it, and that the comparison
harness reads its label off the model instead of asserting one.
"""

from __future__ import annotations

import pytest

from foldjax.models.esmfold2.models import model as structure_model


def test_the_trunk_is_bfloat16_and_that_is_a_divergence():
    """Upstream's released `config.json` is float32; this port is not.

    Kept as bfloat16 because that is what the port was measured and validated
    at, not because upstream does it. Flipping this value is a deliberate act
    -- it changes every published ESMFold2 number -- so it gets a test rather
    than a comment that the next reader can skim past.
    """
    assert structure_model.ModelSettings().trunk_dtype == "bfloat16"


def test_no_checkpoint_can_set_the_trunk_width():
    """`settings_from_config` does not read `trunk_dtype`.

    The docs say the width is unreachable from any knob, which is only true
    while this holds. A config that could set it would make the shipped width
    depend on which checkpoint was loaded.
    """
    from_empty = structure_model.settings_from_config({})
    from_config = structure_model.settings_from_config(
        {"trunk_dtype": "float32", "dtype": "float32"}
    )
    assert from_empty.trunk_dtype == from_config.trunk_dtype == "bfloat16"


def test_no_override_can_set_the_trunk_width():
    """`with_overrides` exposes four knobs and the width is not one of them."""
    with pytest.raises(TypeError):
        structure_model.with_overrides(
            structure_model.ModelSettings(),
            trunk_dtype="float32",  # type: ignore[call-arg]
        )


def test_the_comparison_harness_reads_its_label_off_the_model():
    """The defect this file exists for: a reported dtype that was a literal.

    A stub is enough -- the point is that the function looks at parameters. If
    someone replaces this with a constant again, the fake float32 trunk below
    stops being reported and this fails.
    """
    from bench.esmfold2_compare import _torch_trunk_dtype

    class FakeDtype:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return f"torch.{self.name}"

    class FakeParameter:
        def __init__(self, name: str) -> None:
            self.dtype = FakeDtype(name)

    class FakeModel:
        def named_parameters(self):
            yield "esmc.layers.0.weight", FakeParameter("bfloat16")
            yield "folding_trunk.blocks.0.weight", FakeParameter("float32")

    assert _torch_trunk_dtype(FakeModel()) == "float32"


def test_the_harness_says_so_when_there_is_no_trunk_to_read():
    """Silence would be indistinguishable from agreement."""
    from bench.esmfold2_compare import _torch_trunk_dtype

    class Empty:
        def named_parameters(self):
            return iter(())

    assert "unknown" in _torch_trunk_dtype(Empty())
