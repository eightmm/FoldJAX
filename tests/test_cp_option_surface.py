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
#:
#: The two share the option and not its default: on a GPU grid Boltz-2 omits
#: its way onto the fused tile and Protenix does not, which is a difference in
#: what was measured rather than in what either can run. The cases below
#: assert both halves of that.
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


def _ring_profile(model: str, job, tmp_path, **options):
    request = _request(model, job)
    return get_backend(model).cache_profile(
        PredictionRequest(
            model=model,
            input=request.input,
            weights=request.weights,
            output_dir=tmp_path / "out",
            options={**request.options, **options},
        )
    )


def _pin_fused_tile(monkeypatch, available: bool) -> None:
    """Answer "can this process run the fused tile" without owning a card.

    Boltz-2 resolves an omitted `triangle_attention_ring_kernel` against this
    question on the host, so a test that left it to the runner would assert
    one thing on a CPU box and another on a GPU one. Every case below pins it
    and every pinned value is asserted in both directions, so neither arm can
    be a branch that never fires.
    """

    monkeypatch.setattr(
        "foldjax.models._cp_attention.ring_tile_kernel_available",
        lambda: available,
    )


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
def test_the_shipped_body_shares_the_namespace_omitting_it_selects(
    model: str,
    job,
    tmp_path,
    monkeypatch,
) -> None:
    """`xla` is what an omitted option realises where the tile cannot run.

    Spelling it there must not fork the cache -- it is one program under two
    names. `tokamax` is a different ring body and different arithmetic, so it
    must fork. The arm where the tile *can* run is the next test.
    """

    _pin_fused_tile(monkeypatch, False)

    def profile(**options):
        return _ring_profile(model, job, tmp_path, **options)

    assert profile(triangle_attention_ring_kernel="xla") == profile()
    assert profile(triangle_attention_ring_kernel="tokamax") != profile()


def test_boltz2_omitting_the_ring_kernel_is_the_fused_tile_on_a_gpu_grid(
    job,
    tmp_path,
    monkeypatch,
) -> None:
    """The flip, read off the identity: omission means the tile, `xla` does not.

    An omitted option and an explicit `xla` were one namespace on the grid and
    are now two, because they are now two programs. The recorded word is the
    body the run realises, so an omitted request lands in the namespace the
    opt-in `tokamax` already warmed rather than in a third one, and an
    explicit `xla` keeps the entry -- absence -- that an omitted option wrote
    before the flip.
    """

    def profile(**options):
        return _ring_profile("boltz2", job, tmp_path, **options)

    _pin_fused_tile(monkeypatch, True)
    fused = profile()
    assert fused["triangle_attention_ring_kernel"] == "tokamax"
    assert fused == profile(triangle_attention_ring_kernel="tokamax")
    assert fused != profile(triangle_attention_ring_kernel="xla")
    # The other half of the arm, so the pin above is not measuring a branch
    # that would have been taken anyway: without the card, the same request is
    # the XLA namespace and spelling `xla` is the same entry again.
    _pin_fused_tile(monkeypatch, False)
    assert "triangle_attention_ring_kernel" not in profile()
    assert profile() == profile(triangle_attention_ring_kernel="xla")
    assert profile() != fused


@pytest.mark.parametrize(
    ("options", "why"),
    [
        ({"cp_devices": 1, "cp_layout": "auto"}, "serial"),
        ({"cp_layout": "1d"}, "one-dimensional"),
    ],
)
def test_boltz2_keeps_the_xla_namespace_where_there_is_no_ring(
    options: dict,
    why: str,
    job,
    tmp_path,
    monkeypatch,
) -> None:
    """The tile is a body of the 2-D ring, so off the grid the flip is inert.

    Asserted with the card *present*, which is the only arm that can fail: the
    resolver must read the layout, not just the machine.
    """

    _pin_fused_tile(monkeypatch, True)
    profile = _ring_profile("boltz2", job, tmp_path, **options)
    assert "triangle_attention_ring_kernel" not in profile, why
    assert profile == _ring_profile(
        "boltz2", job, tmp_path, triangle_attention_ring_kernel="xla", **options
    )


def test_protenix_keeps_the_xla_tile_when_the_card_could_run_the_fused_one(
    job,
    tmp_path,
    monkeypatch,
) -> None:
    """The flip is Boltz-2's alone, and the reason is a measurement.

    Protenix's triangle attention is float32 on the XLA ring and the tile is a
    bfloat16 kernel, so on the same four cards its samples move 0.11-0.25 A
    from the serial run against a 0.066 A rerun floor -- outside the band the
    Boltz-2 rows sit inside. Until that is closed the option stays opt-in
    there, and this is what says so.
    """

    _pin_fused_tile(monkeypatch, True)
    profile = _ring_profile("protenix", job, tmp_path)
    assert "triangle_attention_ring_kernel" not in profile
    assert profile == _ring_profile(
        "protenix", job, tmp_path, triangle_attention_ring_kernel="xla"
    )
    assert profile != _ring_profile(
        "protenix", job, tmp_path, triangle_attention_ring_kernel="tokamax"
    )


#: The one port whose backend declares `cp_fused_attention`. It names two
#: *Boltz-2* diffusion attention sites -- the halo-exchanged atom windows and
#: the grid-transposed token tile -- and the other square-grid ports reach
#: neither through this port's adapters, so offering them the word would
#: advertise a site no request could reach.
FUSED_ATTENTION_MODELS = ("boltz2",)


def _fused_request(model: str, job, value: str, **extra) -> PredictionRequest:
    request = _request(model, job)
    options = dict(request.options)
    options["cp_fused_attention"] = value
    options.update(extra)
    return PredictionRequest(
        model=model,
        input=request.input,
        weights=request.weights,
        options=options,
    )


def test_the_fused_attention_vocabulary_is_the_model_s_own() -> None:
    """Two copies of one tuple, compared rather than trusted.

    The same invariant as the ring kernel's above and for the same reason: the
    backend must stay import-time JAX-free, so it cannot import the module
    that owns the vocabulary.
    """

    from foldjax.backends import boltz2
    from foldjax.models._cp_attention import CP_FUSED_ATTENTION_REQUESTS

    assert boltz2._CP_FUSED_ATTENTION_REQUESTS == CP_FUSED_ATTENTION_REQUESTS


@pytest.mark.parametrize("model", FUSED_ATTENTION_MODELS)
@pytest.mark.parametrize("value", ["off", "atom", "token", "atom+token"])
def test_every_fused_attention_site_is_reachable_while_planning(
    model: str,
    value: str,
    job,
) -> None:
    """Accepted while planning, on a 2-D request, without a GPU in sight.

    Whether this machine has tokamax at all is a trace-time question and
    deliberately not asked here; planning must not initialise a backend.
    """

    resolve_request(_fused_request(model, job, value))


@pytest.mark.parametrize("model", FUSED_ATTENTION_MODELS)
def test_a_fused_attention_site_outside_the_vocabulary_is_refused(
    model: str,
    job,
) -> None:
    with pytest.raises(ValueError, match="cp_fused_attention"):
        resolve_request(_fused_request(model, job, "tokamax"))
    with pytest.raises(ValueError, match="cp_fused_attention"):
        resolve_request(_fused_request(model, job, "token+atom"))


@pytest.mark.parametrize("model", FUSED_ATTENTION_MODELS)
def test_the_fused_sites_are_refused_without_the_mesh_they_live_on(
    model: str,
    job,
) -> None:
    """A serial run has neither site; a 1-D one has no token tile.

    Accepting either there would compile the shipped program under a name that
    says a fused kernel ran.
    """

    with pytest.raises(ValueError, match="cp_devices greater than 1"):
        resolve_request(
            _fused_request(model, job, "atom", cp_devices=1, cp_layout="auto")
        )
    with pytest.raises(ValueError, match="cp_layout=2d"):
        resolve_request(_fused_request(model, job, "token", cp_layout="1d"))
    with pytest.raises(ValueError, match="cp_layout=2d"):
        resolve_request(_fused_request(model, job, "atom+token", cp_layout="1d"))
    # The atom windows do exist on the 1-D mesh, so that half is not refused
    # for the layout.
    resolve_request(_fused_request(model, job, "atom", cp_layout="1d"))


@pytest.mark.parametrize(
    "model",
    [name for name in SQUARE_GRID_MODELS if name not in FUSED_ATTENTION_MODELS],
)
def test_a_port_without_the_fused_sites_refuses_the_option(model: str, job) -> None:
    with pytest.raises(ValueError, match="cp_fused_attention"):
        resolve_request(_fused_request(model, job, "atom"))


@pytest.mark.parametrize("model", FUSED_ATTENTION_MODELS)
def test_the_released_fused_value_shares_the_namespace_omitting_it_selects(
    model: str,
    job,
    tmp_path,
) -> None:
    """`off` is what a context-parallel run ships, so spelling it must not fork
    the cache. Each site is a different program, so each of the others must."""

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

    assert profile(cp_fused_attention="off") == profile()
    digests = {
        value: profile(cp_fused_attention=value)
        for value in ("atom", "token", "atom+token")
    }
    for value, digest in digests.items():
        assert digest != profile(), value
    assert len({str(sorted(d.items())) for d in digests.values()}) == 3


# --- triangle_attention_grid (Boltz-2, Protenix, OpenFold3) ------------------

#: The ports whose backend declares `triangle_attention_grid`: the three whose
#: 2-D entries dispatch on the gather scope. OpenDDE's trunk is Protenix's
#: Pairformer and reads the scope too, but its adapter offers no option naming
#: it, so its model entry refuses the scope; ESMFold2 has no triangle
#: attention at all and refuses it the same way
#: (`_cp_attention.refuse_triangle_attention_grid`).
GRID_MODELS = ("boltz2", "protenix", "openfold3")


def _grid_request(model: str, job, grid: str, **extra) -> PredictionRequest:
    request = _request(model, job)
    return PredictionRequest(
        model=model,
        input=request.input,
        weights=request.weights,
        options={**request.options, "triangle_attention_grid": grid, **extra},
    )


@pytest.mark.parametrize("model", GRID_MODELS)
def test_the_grid_algorithm_vocabulary_is_the_model_s_own(model: str) -> None:
    """One tuple per JAX-free backend, compared rather than trusted."""

    from importlib import import_module

    from foldjax.models._cp_attention import TRIANGLE_ATTENTION_GRIDS

    backend = import_module(f"foldjax.backends.{model}")
    assert backend._TRIANGLE_ATTENTION_GRIDS == TRIANGLE_ATTENTION_GRIDS


@pytest.mark.parametrize("model", GRID_MODELS)
@pytest.mark.parametrize("grid", ["ring", "gather"])
def test_the_grid_is_reachable_on_a_2d_request(model: str, grid: str, job) -> None:
    resolve_request(_grid_request(model, job, grid))


@pytest.mark.parametrize("model", GRID_MODELS)
def test_the_gather_grid_is_refused_where_it_cannot_be_honoured(
    model: str, job
) -> None:
    """Misspelled, or off the grid: a serial run and a 1-D mesh have no ring."""

    with pytest.raises(ValueError, match="triangle_attention_grid"):
        resolve_request(_grid_request(model, job, "rotate"))
    for serial in ({"cp_devices": 1, "cp_layout": "auto"}, {"cp_layout": "1d"}):
        with pytest.raises(ValueError, match="cp_layout=2d"):
            resolve_request(_grid_request(model, job, "gather", **serial))


@pytest.mark.parametrize("model", RING_KERNEL_MODELS)
def test_the_gather_grid_is_refused_beside_a_ring_body(model: str, job) -> None:
    """`gather` runs no ring, so a ring body named beside it describes nothing."""

    with pytest.raises(ValueError, match="runs no ring"):
        resolve_request(
            _ring_request(model, job, "tokamax", triangle_attention_grid="gather")
        )
    resolve_request(_ring_request(model, job, "xla", triangle_attention_grid="gather"))


def test_protenix_s_omitted_layout_is_not_the_grid(job) -> None:
    """Protenix's `auto` is the 1-D mesh on every count, four devices included.

    So an omitted layout builds no ring, and both ring options are refused
    there rather than recorded for a program that never reads them. (The
    shared `square_grid_cp_layout` reads `auto` as the grid on a square count,
    which is the other three ports' rule; asking it here accepted the fused
    tile on `cp_devices=4` and ran the 1-D program under the tile's name.)
    OpenFold3 and Boltz-2 do build the grid there, and accept it.
    """

    for extra in (
        {"triangle_attention_ring_kernel": "tokamax"},
        {"triangle_attention_grid": "gather"},
    ):
        request = _request("protenix", job)
        with pytest.raises(ValueError, match="cp_layout=2d"):
            resolve_request(
                PredictionRequest(
                    model="protenix",
                    input=request.input,
                    weights=request.weights,
                    options={"cp_devices": 4, **extra},
                )
            )
    for model in ("boltz2", "openfold3"):
        request = _request(model, job)
        resolve_request(
            PredictionRequest(
                model=model,
                input=request.input,
                weights=request.weights,
                options={"cp_devices": 4, "triangle_attention_grid": "gather"},
            )
        )


@pytest.mark.parametrize(
    "model",
    [name for name in SQUARE_GRID_MODELS if name not in GRID_MODELS],
)
def test_a_port_without_the_grid_option_refuses_it(model: str, job) -> None:
    with pytest.raises(ValueError, match="triangle_attention_grid"):
        resolve_request(_grid_request(model, job, "gather"))


@pytest.mark.parametrize("model", ["opendde", "esmfold2"])
def test_a_port_without_the_grid_option_refuses_the_model_scope(model: str) -> None:
    """The model-level scope a harness enters is refused, not run or ignored.

    OpenDDE's trunk is Protenix's Pairformer, so without this the scope would
    run the gather under a namespace that never names it; ESMFold2 has no
    triangle attention, so it would report a gather that never happened.
    Asserted on each port's host entry, which refuses before reading a single
    argument -- the placeholders below would fail anything else first -- and
    lets the same call through to that failure under `ring`.
    """

    from foldjax.models._cp_attention import triangle_attention_grid_scope

    if model == "opendde":
        from foldjax.models.opendde.models.model import opendde_infer_compiled

        def entry():
            return opendde_infer_compiled({}, None, None)
    else:
        from foldjax.models.esmfold2.inference import predict

        def entry():
            return predict(None, {}, None)

    with triangle_attention_grid_scope("gather"), pytest.raises(
        ValueError, match="no triangle_attention_grid option"
    ):
        entry()
    for grid in (None, "ring"):
        with triangle_attention_grid_scope(grid), pytest.raises(Exception) as error:
            entry()
        assert "triangle_attention_grid" not in str(error.value)


@pytest.mark.parametrize("available", [True, False])
def test_boltz2_gather_forks_the_namespace_and_drops_the_ring_body(
    available: bool, job, tmp_path, monkeypatch
) -> None:
    """`ring` spelled is omission; `gather` is its own program.

    Under `gather` there is no ring, so an omitted ring kernel must not
    record the fused tile even on a GPU grid (the ``available=True`` arm,
    which is the one that can fail).
    """

    _pin_fused_tile(monkeypatch, available)

    def profile(**options):
        return _ring_profile("boltz2", job, tmp_path, **options)

    assert "triangle_attention_grid" not in profile()
    assert profile(triangle_attention_grid="ring") == profile()
    gather = profile(triangle_attention_grid="gather")
    assert gather["triangle_attention_grid"] == "gather"
    assert "triangle_attention_ring_kernel" not in gather
    assert gather != profile()
    assert gather == profile(
        triangle_attention_grid="gather", triangle_attention_ring_kernel="xla"
    )


@pytest.mark.parametrize("model", ["protenix", "openfold3"])
def test_gather_forks_the_namespace_and_ring_is_omission(
    model: str, job, tmp_path, monkeypatch
) -> None:
    """The same two halves on the two ports that gained the option.

    Pinned with the fused tile available, the arm in which a ring body could
    leak into the record; Protenix's omitted ring body is `xla` regardless.
    """

    _pin_fused_tile(monkeypatch, True)

    def profile(**options):
        return _ring_profile(model, job, tmp_path, **options)

    assert "triangle_attention_grid" not in profile()
    assert profile(triangle_attention_grid="ring") == profile()
    gather = profile(triangle_attention_grid="gather")
    assert gather["triangle_attention_grid"] == "gather"
    assert "triangle_attention_ring_kernel" not in gather
    assert gather != profile()
    if model in RING_KERNEL_MODELS:
        assert gather == profile(
            triangle_attention_grid="gather", triangle_attention_ring_kernel="xla"
        )
