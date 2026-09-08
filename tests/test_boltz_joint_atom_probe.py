import pytest

from bench.boltz_joint_atom_probe import validate_embedding_calls


@pytest.mark.parametrize("enabled,observed,recycles", [(True, 4, 3), (False, 0, 3)])
def test_embedding_capture_count_matches_runtime(enabled, observed, recycles):
    validate_embedding_calls(enabled, observed, recycles)


@pytest.mark.parametrize(
    "enabled,observed", [(True, 0), (True, 3), (True, 5), (False, 1)]
)
def test_embedding_capture_rejects_missing_or_extra_callbacks(enabled, observed):
    with pytest.raises(ValueError, match="incomplete MSA embedding"):
        validate_embedding_calls(enabled, observed, 3)
