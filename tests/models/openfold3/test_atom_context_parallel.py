"""Gates for OpenFold3's distributed diffusion atom graph.

Two questions, and they need different machinery. *Does a serial run still
execute the program it executed before?* is answered by lowering the denoiser
on a single device and comparing the text -- against this same tree's ``main``,
and against a recorded pin so a later change is visible once ``main`` has moved
on. *Does the mesh variant compute the same thing?* needs a forced host device
count, which has to be set before JAX initialises, so those arms run in
subprocesses driven by the scripts under ``scripts/``.

The scripts hold the assertions and print their own evidence; the tests here
choose the arms and insist on the marker line. That split is deliberate: the
same scripts run on a GPU with real devices, where the memory claim can be
measured rather than inferred from a lowering.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.models.cp_probe_env import inherited_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: SHA-256 of the normalised HLO of one serial denoiser step, taken from a
#: ``git archive main`` tree at the commit this branch left (see
#: ``test_the_serial_lowering_is_unchanged_by_this_patch``, which recomputes
#: the ``main`` side rather than trusting this number). Its job is to outlive
#: that comparison: once this branch is on ``main`` the dynamic test compares
#: ``main`` with itself and guards nothing, while this pin still fails the
#: moment the serial diffusion program changes. A change here is a claim about
#: the serial program and needs the measurement that justifies it; recompute
#: with ``python -m tests.models.openfold3.scripts.serial_diffusion_lowering``.
SERIAL_DIFFUSION_LOWERING_SHA256 = (
    "aaff4a4644143102d8a18b2f39bcc0908930a438f7514e3d21ba328f578407a0"
)

#: Mesh arms. 96 atoms over 12 tokens with a 4/8 block divides a 1-D four-way
#: split, a 2x2 grid and a 3x3 grid, and the 1-D layout is exercised as well
#: because it is the one ``cp_layout="auto"`` selects.
MESH_ARMS = ((4, "1d"), (4, "2d"), (9, "2d"))

#: Real-atom arms. 90 of 96 leaves the last two query blocks shifted off their
#: own atoms; 60 of 96 leaves nine blocks reading atoms 52..59, three whole
#: blocks away. The second is the arm a halo exchange cannot satisfy at any
#: static width, which is why the key side is an index-driven ring gather --
#: with all 96 atoms real, a halo would pass this file.
REAL_ATOM_ARMS = (90, 60)


def _run(
    module: str,
    *,
    devices: int,
    layout: str | None = None,
    real_atoms: int | None = None,
    foldjax_source: Path | None = None,
    timeout: int = 900,
) -> str:
    """Run one probe module in a child with a forced device count."""

    source = REPOSITORY_ROOT / "src" if foldjax_source is None else foldjax_source
    environment = {
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        **inherited_environment(),
        # After `inherited_environment`, so the tree under test wins over
        # whatever the parent's PYTHONPATH points at -- in a worktree those are
        # different checkouts, and a probe that reports on the installed tree
        # reports on code nobody edited.
        "PYTHONPATH": f"{source}:{REPOSITORY_ROOT}",
    }
    if layout is not None:
        environment["FOLDJAX_CP_PROBE_LAYOUT"] = layout
    if real_atoms is not None:
        environment["FOLDJAX_OF3_CP_REAL_ATOMS"] = str(real_atoms)
    completed = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPOSITORY_ROOT,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _lowering_report(source: Path | None = None) -> dict[str, str]:
    output = _run(
        "tests.models.openfold3.scripts.serial_diffusion_lowering",
        devices=1,
        foldjax_source=source,
    )
    report = {}
    for line in output.strip().splitlines():
        key, _, value = line.partition(" ")
        report[key] = value
    return report


@pytest.fixture(scope="module")
def serial_lowering() -> dict[str, str]:
    return _lowering_report()


def test_the_serial_lowering_carries_no_collectives_or_sharding(
    serial_lowering: dict[str, str],
) -> None:
    assert serial_lowering["COLLECTIVES"] == "0", serial_lowering
    assert serial_lowering["SHARDING_CALLS"] == "0", serial_lowering
    assert serial_lowering["SHARDING_ANNOTATIONS"] == "0", serial_lowering


def test_the_serial_lowering_matches_its_recorded_pin(
    serial_lowering: dict[str, str],
) -> None:
    assert serial_lowering["HASH"] == SERIAL_DIFFUSION_LOWERING_SHA256, (
        "the serial diffusion lowering changed; see the constant's docstring"
    )


def test_the_serial_lowering_is_unchanged_by_this_patch(
    tmp_path: Path,
    serial_lowering: dict[str, str],
) -> None:
    """Compare against ``main`` in this interpreter, not against a memory.

    The pin above cannot tell a deliberate change from a JAX upgrade. This can:
    both sides run with the same JAX, the same parameters and the same probe
    script, and only the ``foldjax`` tree differs.
    """

    revision = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--verify", "main"],
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        pytest.skip("no `main` in this checkout to compare against")
    archive = tmp_path / "main"
    archive.mkdir()
    export = subprocess.run(
        f"git -C {REPOSITORY_ROOT} archive main | tar -x -C {archive}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert export.returncode == 0, export.stderr

    reference = _lowering_report(archive / "src")
    assert reference["HASH"] == serial_lowering["HASH"], (
        "this patch changed the program a serial OpenFold3 denoiser runs: "
        f"main {reference['HASH']}, here {serial_lowering['HASH']}"
    )
    assert reference["COLLECTIVES"] == "0", reference


@pytest.mark.parametrize(("devices", "layout"), MESH_ARMS)
def test_the_distributed_atom_graph_matches_serial(devices: int, layout: str) -> None:
    output = _run(
        "tests.models.openfold3.scripts.atom_cp_parity",
        devices=devices,
        layout=layout,
    )
    assert "OPENFOLD3_ATOM_CP_OK" in output, output
    assert f"layout={layout}" in output, output


@pytest.mark.parametrize("real_atoms", REAL_ATOM_ARMS)
def test_the_distributed_atom_graph_matches_serial_with_a_shifted_tail(
    real_atoms: int,
) -> None:
    """The arm that separates the ring gather from a halo exchange.

    ``atom_blocks.block_indices`` slides a key window left to end on the last
    real atom, so with 60 of 96 atoms real, nine query blocks read atoms
    52..59 -- outside any halo whose width is a function of ``n_query`` and
    ``n_key``. The shift comes from ``jnp.sum(atom_mask)``, so there is no
    static width that would cover it either.
    """

    output = _run(
        "tests.models.openfold3.scripts.atom_cp_parity",
        devices=4,
        layout="1d",
        real_atoms=real_atoms,
    )
    assert "OPENFOLD3_ATOM_CP_OK" in output, output
    assert f"real_atoms={real_atoms}/96" in output, output


def test_a_mesh_does_not_leak_into_the_serial_program_around_it() -> None:
    output = _run(
        "tests.models.openfold3.scripts.atom_cp_invariance",
        devices=4,
    )
    assert "OPENFOLD3_ATOM_INVARIANCE_OK" in output, output
    assert "COLLECTIVES 0 SHARDING 0" in output, output
    # The hash equality alone would hold for any mesh: every guard reads
    # `cp_atom_windows and cp_mesh() is not None`, so with the option off the
    # mesh is never consulted. These two lines are what make the claim about the
    # mesh rather than about the option -- the blocking site fired on both arms
    # and saw no plan on either.
    assert "BLOCK_SITES 4 4" in output, output
    assert "PLANS_SEEN 0 0" in output, output


def test_the_atom_window_option_is_part_of_the_compile_identity(tmp_path) -> None:
    """Two atom-graph programs must not share one cache namespace.

    And an explicit ``true`` must share the namespace an omitted option
    selects, which is what the released-defaults strip is for.
    """

    from foldjax import PredictionRequest
    from foldjax.backends.openfold3 import OpenFold3Backend

    job = tmp_path / "job.json"
    job.write_text(
        '{"name": "t", "entities": [{"type": "protein", "id": ["A"], '
        '"sequence": "GRISMTVKKLYFIPAGRCMLDHSSVNSALTPGK"}]}'
    )
    weights = tmp_path / "openfold3.weights"
    weights.touch()
    backend = OpenFold3Backend()

    def profile(options: dict[str, object]) -> dict[str, object]:
        return backend.cache_profile(
            PredictionRequest(
                model="openfold3",
                input=job,
                weights=weights,
                options=options,
            )
        )

    assert "cp_atom_windows" in backend.native_options
    assert "cp_atom_windows" in backend.compile_options
    omitted = profile({})
    explicit_on = profile({"cp_atom_windows": True})
    off = profile({"cp_atom_windows": False})
    assert "cp_atom_windows" not in omitted, omitted
    assert omitted == explicit_on
    assert off.get("cp_atom_windows") is False, off
    assert off != omitted


def test_the_option_reaches_the_model_configuration() -> None:
    """The CLI flag and the managed option must both land on the config field.

    Without this, ``--no-cp-atom-windows`` would parse, validate, reach
    ``released_config`` -- and be dropped by a signature that never grew the
    parameter, which is the shape of gap ``tests/test_option_census.py`` was
    written after.
    """

    from foldjax.models.openfold3.inference import released_config

    default = released_config(n_token=8, n_atom=16)
    assert default.cp_atom_windows is True
    off = released_config(n_token=8, n_atom=16, cp_atom_windows=False)
    assert off.cp_atom_windows is False
