"""Every model the docs call two-dimensional must accept `cp_layout`.

`docs/context_parallel.md` marks Boltz-2, Protenix, OpenDDE and OpenFold3 as
supporting the square grid, and each of their native command lines takes
`--cp-layout`. The unified CLI reaches those through a per-backend option set,
and OpenDDE's was missing the entry: `opendde-jax-predict --cp-layout 2d` ran
while `foldjax --model opendde --option cp_layout=2d` was refused as an
unsupported option, so the documented layout was unreachable through the
interface the project puts first.

ESMFold2 is the deliberate exception -- it has no triangle attention and no
square-grid path -- so its refusal is asserted rather than tolerated.
"""

import pytest

from foldjax import PredictionRequest, resolve_request

SQUARE_GRID_MODELS = ("boltz2", "opendde", "openfold3", "protenix")


def _request(model: str, job) -> PredictionRequest:
    return PredictionRequest(
        model=model,
        input=job,
        options={"cp_devices": 4, "cp_layout": "2d"},
    )


@pytest.fixture
def job(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(
        '{"name": "t", "entities": [{"type": "protein", "id": ["A"], '
        '"sequence": "GRISMTVKKLYFIPAGRCMLDHSSVNSALTPGK"}]}'
    )
    return path


@pytest.mark.parametrize("model", SQUARE_GRID_MODELS)
def test_the_square_grid_is_reachable_through_the_common_option(model, job) -> None:
    resolve_request(_request(model, job))


def test_esmfold2_refuses_a_layout_it_does_not_have(job) -> None:
    with pytest.raises(ValueError, match="cp_layout"):
        resolve_request(_request("esmfold2", job))
