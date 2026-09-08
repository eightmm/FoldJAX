"""Native recycle options keep their model-specific zero-pass boundary."""

import pytest

from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest


@pytest.mark.parametrize(
    ("model", "accepts_zero"),
    [
        ("alphafold3", True),
        ("boltz2", True),
        ("esmfold2", True),
        ("openfold3", False),
        ("opendde", False),
        ("protenix", False),
    ],
)
def test_native_zero_recycles_validation(tmp_path, model, accepts_zero):
    source = tmp_path / "job.json"
    source.write_text('{"entities": []}')
    request = PredictionRequest(
        model=model,
        input=source,
        input_format="foldjax",
        options={"num_recycles": 0},
    )
    backend = get_backend(model)
    # Explicit native counts must never receive the common request's +1.
    assert backend.apply_sampling(request)["num_recycles"] == 0
    if accepts_zero:
        backend.validate_request(request)
    else:
        with pytest.raises(ValueError, match="num_recycles"):
            backend.validate_request(request)
