"""Every model the docs call two-dimensional must accept `cp_layout`.

`docs/context_parallel.md` marks Boltz-2, Protenix, OpenDDE, OpenFold3 and
ESMFold2 as supporting the square grid, and each of their native command lines
takes `--cp-layout`. The unified CLI reaches those through a per-backend option
set, and OpenDDE's was missing the entry: `opendde-jax-predict --cp-layout 2d`
ran while `foldjax --model opendde --option cp_layout=2d` was refused as an
unsupported option, so the documented layout was unreachable through the
interface the project puts first.

AlphaFold 3 is the deliberate exception -- the vendored publisher runtime is
not rewritten for FoldJAX context parallelism at all -- so its refusal is
asserted rather than tolerated. ESMFold2 was that exception until it gained
the grid; what it still refuses is a device count the grid cannot be built
from, which is a different sentence and has its own case below.
"""

import pytest

from foldjax import PredictionRequest, resolve_request
from foldjax.registry import get_backend

SQUARE_GRID_MODELS = ("boltz2", "esmfold2", "opendde", "openfold3", "protenix")

#: The ports whose backend declares `triangle_attention_ring_kernel`. The
#: other three square-grid models run the same ring object and could take the
#: same option; what they lack is a measurement, and their adapters refuse the
#: scope rather than dropping it (`models/openfold3/models/
#: triangle_attention_cp.py`), so an unmeasured port cannot report a fused run
#: it did not make.
RING_KERNEL_MODELS = ("boltz2", "protenix")


def _request(model: str, job) -> PredictionRequest:
    weights = job.parent / f"{model}.weights"
    weights.touch()
    return PredictionRequest(
        model=model,
        input=job,
        # This gate exercises cheap option validation, not managed checkpoint
        # discovery.  An explicit path keeps a clean CI runner independent of
        # whichever multi-gigabyte model weights happen to be installed.
        weights=weights,
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


def test_a_model_without_context_parallelism_refuses_the_layout(job) -> None:
    """AlphaFold 3 carries neither option, and says so rather than ignoring it."""

    with pytest.raises(ValueError, match="cp_layout"):
        resolve_request(_request("alphafold3", job))


@pytest.mark.parametrize("devices", [1, 2, 3, 8])
def test_the_grid_is_refused_on_a_count_it_cannot_be_built_from(devices, job) -> None:
    """A square mesh needs a perfect-square count, and the refusal is early.

    ESMFold2 is asked because it is the port that most recently gained the
    option; the message comes from its adapter, before any checkpoint opens.
    """

    request = _request("esmfold2", job)
    options = dict(request.options)
    options["cp_devices"] = devices
    with pytest.raises(ValueError, match="perfect-square"):
        resolve_request(
            PredictionRequest(
                model=request.model,
                input=request.input,
                weights=request.weights,
                options=options,
            )
        )


def _ring_request(model: str, job, kernel: str, **extra) -> PredictionRequest:
    request = _request(model, job)
    options = dict(request.options)
    options["triangle_attention_ring_kernel"] = kernel
    options.update(extra)
    return PredictionRequest(
        model=model,
        input=request.input,
        weights=request.weights,
        options=options,
    )


def test_the_two_backend_vocabularies_are_the_ring_s_own() -> None:
    """Three copies of one tuple, compared rather than trusted.

    Both backends must stay import-time JAX-free so `foldjax plan` can
    validate a request without initialising a device, so neither can import
    the ring module for its vocabulary. This is what stops the copies drifting
    -- a value added to the ring and not to a backend would advertise a body
    no request can reach, and the reverse would accept a word the ring
    refuses at trace time, after featurization.
    """

    from foldjax.backends import boltz2, protenix
    from foldjax.models._cp_attention import RING_TILE_KERNELS

    assert boltz2._RING_TILE_KERNELS == RING_TILE_KERNELS
    assert protenix._RING_TILE_KERNELS == RING_TILE_KERNELS


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
@pytest.mark.parametrize("kernel", ["xla", "tokamax"])
def test_the_ring_kernel_is_reachable_on_the_ports_that_declare_it(
    model: str,
    kernel: str,
    job,
) -> None:
    """Accepted while planning, on a 2-D request, without a GPU in sight.

    Whether this machine can *run* the fused tile is a trace-time question and
    deliberately not asked here; planning must not initialise a backend.
    """

    resolve_request(_ring_request(model, job, kernel))


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
def test_a_kernel_outside_the_vocabulary_is_refused_while_planning(
    model: str,
    job,
) -> None:
    with pytest.raises(ValueError, match="triangle_attention_ring_kernel"):
        resolve_request(_ring_request(model, job, "triton"))


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
def test_the_fused_tile_is_refused_without_the_grid_it_is_a_body_of(
    model: str,
    job,
) -> None:
    """A serial run has no ring, so there is no body to pick.

    Accepting it there would compile the shipped program under a name that
    says a fused kernel ran.
    """

    with pytest.raises(ValueError, match="cp_layout=2d"):
        resolve_request(
            _ring_request(model, job, "tokamax", cp_devices=1, cp_layout="auto")
        )
    with pytest.raises(ValueError, match="cp_layout=2d"):
        resolve_request(_ring_request(model, job, "tokamax", cp_layout="1d"))


@pytest.mark.parametrize(
    "model",
    [name for name in SQUARE_GRID_MODELS if name not in RING_KERNEL_MODELS],
)
def test_a_port_without_the_option_refuses_it_rather_than_ignoring_it(
    model: str,
    job,
) -> None:
    with pytest.raises(ValueError, match="triangle_attention_ring_kernel"):
        resolve_request(_ring_request(model, job, "tokamax"))


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
def test_the_shipped_body_shares_the_namespace_omitting_it_selects(
    model: str,
    job,
    tmp_path,
) -> None:
    """`xla` is the released value, so spelling it must not fork the cache.

    `tokamax` is a different ring body and different arithmetic, so it must.
    """

    backend = get_backend(model)

    def profile(**options):
        request = _request(model, job)
        merged = {**request.options, **options}
        return backend.cache_profile(
            PredictionRequest(
                model=model,
                input=request.input,
                weights=request.weights,
                output_dir=tmp_path / "out",
                options=merged,
            )
        )

    assert profile(triangle_attention_ring_kernel="xla") == profile()
    assert profile(triangle_attention_ring_kernel="tokamax") != profile()
