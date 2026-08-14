"""Compatibility imports for the renamed NumPy checkpoint reader.

New code should import :mod:`foldjax.models.boltz2.bridge.checkpoint`.  This
module remains so callers of the original default, NumPy-backed API do not
break; publisher-runtime tensor conversion lives only in the parity tests.
"""

from foldjax.models.boltz2.bridge.checkpoint import (
    checkpoint_state_dict,
    describe_checkpoint,
    load_checkpoint_state_dict,
    main,
)

__all__ = [
    "checkpoint_state_dict",
    "describe_checkpoint",
    "load_checkpoint_state_dict",
    "main",
]


if __name__ == "__main__":
    main()
