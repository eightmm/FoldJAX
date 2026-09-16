"""`cp_layout` on ESMFold2: what `auto` means, and what each spelling records.

Three things can disagree here and none of them is visible in the other two.
The port resolves the alias (`models/esmfold2/inference._resolve_cp_layout`),
the adapter resolves it again while a request is still being planned on the
host, because that path must not import JAX
(`backends/esmfold2._resolved_cp_layout`), and the compile profile records what
runs. A copy of a rule is a place it can drift, so the two resolvers are read
against each other rather than each against a table of strings.

`auto` is the row mesh on this port. The square grid is implemented and
checked against the serial trunk in `test_context_parallel.py`, but nothing
has measured a per-device peak or a wall time for it on a card; OpenDDE and
Boltz-2 resolve `auto` to the grid because they have that measurement. The
test below is where that decision is pinned, so flipping it means coming here
with a number.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from foldjax.backends.esmfold2 import (
    _FIXED_COMPILE_DEFAULTS,
    ESMFold2Backend,
    _resolved_cp_layout,
)
from foldjax.padding import cp_mesh_rows
from foldjax.schema import PredictionRequest
from tests.models.cp_probe_env import inherited_environment

#: Counts the rules are compared on, including ones no CPU process here can
#: build: the resolution is arithmetic and does not need the devices.
_COUNTS = (1, 2, 3, 4, 8, 9, 16)


@pytest.fixture
def profile_of(tmp_path):
    job = tmp_path / "job.json"
    job.write_text("{}")

    def build(**options):
        return ESMFold2Backend().cache_profile(
            PredictionRequest(
                model="esmfold2",
                input=job,
                output_dir=tmp_path / "out",
                options=options,
            )
        )

    return build


def test_auto_is_the_row_mesh_on_every_device_count() -> None:
    """The port's own resolver, not a copy of the rule.

    Including the perfect squares, which is the whole content of the decision:
    on four devices this port answers `1d` where OpenDDE and Boltz-2 answer
    `2d`.
    """

    from foldjax.models.esmfold2.inference import _resolve_cp_layout

    for devices in _COUNTS:
        assert _resolve_cp_layout("auto", devices) == "1d", devices
        assert _resolve_cp_layout("1d", devices) == "1d", devices


def test_the_grid_is_refused_on_a_count_it_cannot_use() -> None:
    from foldjax.models.esmfold2.inference import _resolve_cp_layout

    for devices in (1, 2, 3, 8):
        with pytest.raises(ValueError, match="perfect-square"):
            _resolve_cp_layout("2d", devices)
    for devices in (4, 9, 16):
        assert _resolve_cp_layout("2d", devices) == "2d", devices


def test_the_adapter_and_the_port_resolve_one_rule() -> None:
    """The plan-time copy must answer what the run-time resolver builds.

    The adapter cannot call the port's resolver -- that would pull JAX into
    option planning -- so it carries the rule, and this is what keeps the two
    from drifting. The adapter answers `None` wherever there is no grid to
    name, which is every case the port answers `1d` for.
    """

    from foldjax.models.esmfold2.inference import _resolve_cp_layout

    for devices in _COUNTS:
        for layout in ("auto", "1d", "2d"):
            try:
                resolved = _resolve_cp_layout(layout, devices)
            except ValueError:
                # A refusal the adapter reports through
                # `validate_native_options` instead; checked below.
                continue
            adapter = _resolved_cp_layout(
                {"cp_devices": devices, "cp_layout": layout}
            )
            expected = "2d" if resolved == "2d" and devices > 1 else None
            assert adapter == expected, (devices, layout, adapter)


def test_a_malformed_device_count_keeps_its_own_namespace() -> None:
    """The adapter names no grid for a count it cannot read as one.

    Borrowing a resolved layout for a malformed request would put it in the
    namespace of a well-formed one; the port reports the value itself.
    """

    for devices in (True, "4", 4.0, None):
        assert _resolved_cp_layout({"cp_devices": devices, "cp_layout": "2d"}) is None


def test_the_row_mesh_keeps_the_namespace_every_recorded_run_has(
    profile_of,
) -> None:
    """Omitted, `auto` and `1d` are one program and one cache namespace.

    They have to be: `auto` resolves to the row mesh here, so the three
    spellings build the same mesh and trace the same graph. Splitting them
    would move every namespace recorded before `cp_layout` existed.
    """

    baseline = profile_of(cp_devices=4)
    assert "cp_layout" not in baseline, baseline
    assert profile_of(cp_devices=4, cp_layout="auto") == baseline
    assert profile_of(cp_devices=4, cp_layout="1d") == baseline
    assert "cp_layout" not in profile_of(), "a serial run names no layout"


def test_the_grid_is_recorded_resolved(profile_of) -> None:
    """The grid is a different program, so it is a different namespace."""

    grid = profile_of(cp_devices=4, cp_layout="2d")
    assert grid["cp_layout"] == "2d", grid
    assert grid != profile_of(cp_devices=4)
    assert profile_of(cp_devices=9, cp_layout="2d")["cp_layout"] == "2d"


def test_the_layout_is_part_of_the_compile_identity() -> None:
    """Named on both surfaces: accepted by a request, and cache-forking.

    A compile option absent from `native_options` is rejected by
    `validate_request`; a native option absent from `compile_options` hands
    two programs one namespace. `tests/test_option_census.py` holds this for
    every port, and it is repeated here because this is the port the name was
    added to.
    """

    backend = ESMFold2Backend()
    assert "cp_layout" in backend.native_options
    assert "cp_layout" in backend.compile_options
    assert _FIXED_COMPILE_DEFAULTS["cp_layout"] == "auto"


def test_an_unbuildable_or_misspelled_layout_is_refused_while_planning() -> None:
    """Before the 939 MB checkpoint is opened, and where `plan` sees it."""

    backend = ESMFold2Backend()
    with pytest.raises(ValueError, match="cp_layout must be one of"):
        backend.validate_native_options({"cp_layout": "3d"})
    with pytest.raises(ValueError, match="perfect-square"):
        backend.validate_native_options({"cp_layout": "2d", "cp_devices": 2})
    with pytest.raises(ValueError, match="perfect-square"):
        backend.validate_native_options({"cp_layout": "2d"})
    backend.validate_native_options({"cp_layout": "2d", "cp_devices": 4})
    backend.validate_native_options({"cp_layout": "auto", "cp_devices": 3})


def test_the_grid_pads_tokens_to_one_side_of_the_mesh() -> None:
    """Both pair axes are divided, so the multiple is the side, not the count.

    The adapter passes the resolved layout to `cp_aligned_padding` for this
    reason; padding a 2x2 run to a multiple of four would over-pad, and
    padding a 3x3 run to a multiple of nine would too.
    """

    assert cp_mesh_rows(4, "1d") == 4
    assert cp_mesh_rows(4, "2d") == 2
    assert cp_mesh_rows(9, "2d") == 3
    # An omitted layout is the row mesh in that helper as well, which is what
    # makes passing this port's `auto` through it correct rather than lucky.
    assert cp_mesh_rows(4, "auto") == 4


_MESH_PROBE = textwrap.dedent(
    r"""
    import jax

    from foldjax.models._cp import (
        CP_AXIS,
        CP_COL_AXIS,
        CP_ROW_AXIS,
        context_parallel,
        cp_identity,
    )
    from foldjax.models.esmfold2.inference import _resolve_cp_layout, _run

    assert jax.device_count() == 4, jax.devices()

    # The mesh each answer builds, not just the string it is.
    with context_parallel(4, layout=_resolve_cp_layout("auto", 4)):
        assert cp_identity() == ("1d", 4, (4, 1), (CP_AXIS,)), cp_identity()
    with context_parallel(4, layout=_resolve_cp_layout("2d", 4)):
        assert cp_identity() == (
            "2d", 4, (2, 2), (CP_ROW_AXIS, CP_COL_AXIS),
        ), cp_identity()

    # The static argument and the ambient mesh are one decision, and the guard
    # is what says so. Both directions: a grid executable served from a row
    # mesh, and a row executable served from a grid. Garbage inputs are fine,
    # the guard runs before the model touches anything.
    with context_parallel(4, layout="2d"):
        try:
            _run(None, {}, {}, None, None, 1, False, 4)
        except RuntimeError as error:
            assert "cp_layout='1d'" in str(error), error
        else:
            raise AssertionError("a row executable ran on the grid")
    with context_parallel(4, layout="1d"):
        try:
            _run(None, {}, {}, None, None, 1, False, 4, cp_layout="2d")
        except RuntimeError as error:
            assert "cp_layout='2d'" in str(error), error
        else:
            raise AssertionError("a grid executable ran on the row mesh")

    print("CP_LAYOUT_MESH_OK")
    """
)


def test_the_layout_argument_and_the_active_mesh_are_one_decision() -> None:
    """A forced four-device CPU process: the two meshes, and both mismatches."""

    completed = subprocess.run(
        [sys.executable, "-c", _MESH_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CP_LAYOUT_MESH_OK" in completed.stdout
