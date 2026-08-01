"""The MSA depth cap, which is the dominant memory knob.

The trunk of every one of these models holds an ``[depth, tokens, channels]``
MSA representation. Measured on a 488-token job with a 13,254-row alignment,
that single tensor was 3,158 MiB and three of them were live at once -- 9.5 GiB
of a 12.2 GiB peak, against 2,891 MiB of activations for the same job in
AlphaFold 3. Capping the depth to 1024 halved Protenix's peak and let OpenDDE
run at all, where lowering the compute dtype and blocking the triangle
attention had each moved it by less than 0.5%.

The knob is model-neutral because the tensor exists in every trunk; each
backend translates it to its own name.
"""

from __future__ import annotations

import pytest

from foldjax.registry import capabilities, get_backend
from foldjax.schema import PredictionRequest


def _request(tmp_path, model, **kwargs):
    job = tmp_path / "job.json"
    job.write_text('{"entities": [{"type": "protein", "id": ["A"], '
                   '"sequence": "ACDEFG"}]}')
    return PredictionRequest(model=model, input=job, **kwargs)


@pytest.mark.parametrize("model", ["protenix", "opendde", "boltz2"])
def test_the_cap_is_a_neutral_knob(model: str) -> None:
    assert "max_msa_depth" in capabilities(model).sampling


@pytest.mark.parametrize(
    ("model", "native"),
    [("protenix", "max_msa_rows"), ("opendde", "max_msa_rows"),
     ("boltz2", "max_msa_depth")],
)
def test_the_cap_reaches_each_backend_under_its_own_name(
    tmp_path, model: str, native: str
) -> None:
    options = get_backend(model).apply_sampling(
        _request(tmp_path, model, max_msa_depth=1024)
    )
    assert options[native] == 1024


def test_an_unset_cap_leaves_the_backend_default_alone(tmp_path) -> None:
    """Omitting it must not pin a depth, or the port's own default is gone."""
    options = get_backend("protenix").apply_sampling(_request(tmp_path, "protenix"))
    assert "max_msa_rows" not in options


def test_the_cap_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_msa_depth must be at least 1"):
        _request(tmp_path, "protenix", max_msa_depth=0)


def test_setting_both_spellings_is_an_error(tmp_path) -> None:
    """Silently preferring one would change the depth without changing the
    exit code -- the same rule the other neutral knobs follow."""
    request = _request(
        tmp_path, "protenix", max_msa_depth=1024,
        options={"max_msa_rows": 4096},
    )
    with pytest.raises(ValueError, match="both set"):
        get_backend("protenix").apply_sampling(request)


def test_the_cap_changes_the_compile_cache_namespace(tmp_path) -> None:
    """It changes the compiled program's shapes, so it cannot share an entry."""
    backend = get_backend("protenix")
    assert "max_msa_rows" in backend.compile_options


def test_a_chunk_size_the_kernel_cannot_honour_is_not_silent() -> None:
    """cuEquivariance has no chunked triangle-attention kernel.

    A chunk size that reaches it bounds nothing, and the score tensor it exists
    to bound is materialised whole. That was silent, which is how a knob comes
    to look like it works.
    """
    import warnings

    from foldjax.models.protenix.models.triangle import triangle

    triangle._WARNED_UNCHUNKABLE = False
    with pytest.warns(RuntimeWarning, match="ignored by the 'cueq' backend"):
        triangle._warn_unchunkable_backend(128)
    # Once per process: the trunk calls this per block per layer, and warning
    # every time would bury the run in identical noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triangle._warn_unchunkable_backend(128)
    assert caught == []
