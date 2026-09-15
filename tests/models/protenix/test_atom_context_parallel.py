"""Gates for Protenix' distributed diffusion atom graph.

Two questions, and they need different machinery. *Does a serial run still
execute the program it executed before?* is answered by lowering the denoiser
on a single device and comparing the text -- against this same tree's
``main``, and against a recorded pin so a later change is visible once ``main``
has moved on. *Does the mesh variant compute the same thing?* needs a forced
host device count, which has to be set before JAX initialises, so those arms
run in subprocesses driven by the scripts under ``scripts/``.

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
#: the ``main`` side rather than trusting this number).  Its job is to outlive
#: that comparison: once this branch is on ``main`` the dynamic test compares
#: ``main`` with itself and guards nothing, while this pin still fails the
#: moment the serial diffusion program changes.  A change here is a claim
#: about the serial program and needs the measurement that justifies it;
#: recompute with
#: ``python -m tests.models.protenix.scripts.serial_diffusion_lowering``.
SERIAL_DIFFUSION_LOWERING_SHA256 = (
    "eea41b130670b7b70f4c6b79d619a3e267244b75aa78caa61d5ec3b92000573d"
)

#: Mesh arms. 96 atoms over 24 tokens with a 4/8 window divides both a 2x2 and
#: a 3x3 grid, and the 1-D layout is exercised as well because it is the one
#: ``cp_layout="auto"`` selects.
MESH_ARMS = ((4, "1d"), (4, "2d"), (9, "2d"))

#: Attention arms. ``xla_jit`` is what the adapter resolves an omitted
#: ``diffusion_attention_backend`` to under a mesh, so it is the released CP
#: path and not an exotic setting: inside a sharded body it falls back to
#: ``xla``, because the window plan holds that body's tracers and cannot cross
#: an inner ``jit``. Pinning only ``xla`` here would have tested the arm
#: nobody runs.
ATTENTION_ARMS = ("xla", "xla_jit")


def _run(
    module: str,
    *,
    devices: int,
    layout: str | None = None,
    attention: str | None = None,
    scan: bool = False,
    token_chunk: int | None = None,
    foldjax_source: Path | None = None,
    timeout: int = 600,
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
        # whatever the parent's PYTHONPATH points at -- in a worktree those
        # are different checkouts, and a probe that reports on the installed
        # tree reports on code nobody edited.
        "PYTHONPATH": f"{source}:{REPOSITORY_ROOT}",
    }
    if layout is not None:
        environment["FOLDJAX_CP_PROBE_LAYOUT"] = layout
    if attention is not None:
        environment["FOLDJAX_CP_PROBE_ATTENTION"] = attention
    if scan:
        environment["FOLDJAX_CP_PROBE_SCAN"] = "1"
    if token_chunk is not None:
        environment["FOLDJAX_CP_PROBE_TOKEN_CHUNK"] = str(token_chunk)
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
        "tests.models.protenix.scripts.serial_diffusion_lowering",
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

    The pin above cannot tell a deliberate change from a JAX upgrade. This
    can: both sides run with the same JAX, the same parameters and the same
    probe script, and only the ``foldjax`` tree differs.
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
        "this patch changed the program a serial Protenix denoiser runs: "
        f"main {reference['HASH']}, here {serial_lowering['HASH']}"
    )
    assert reference["COLLECTIVES"] == "0", reference


@pytest.mark.parametrize("attention", ATTENTION_ARMS)
@pytest.mark.parametrize(("devices", "layout"), MESH_ARMS)
def test_the_distributed_atom_graph_matches_serial(
    devices: int,
    layout: str,
    attention: str,
) -> None:
    output = _run(
        "tests.models.protenix.scripts.atom_cp_parity",
        devices=devices,
        layout=layout,
        attention=attention,
    )
    assert "PROTENIX_ATOM_CP_OK" in output, output
    assert f"attention={attention}" in output, output


@pytest.mark.parametrize(("devices", "layout"), MESH_ARMS)
def test_the_distributed_atom_graph_matches_serial_under_the_released_scan(
    devices: int,
    layout: str,
) -> None:
    """The configuration a real context-parallel run actually compiles.

    ``cp_shards > 1`` requires ``graph_jit``, and ``graph_jit`` sets
    ``use_diffusion_scan=True``, so the block stack runs inside ``lax.scan``
    *inside* the ``shard_map`` body: the halo's ``ppermute`` lives in the scan
    body and the window mask is a scan constant. Nothing about the unrolled
    arm above says that composes.
    """

    output = _run(
        "tests.models.protenix.scripts.atom_cp_parity",
        devices=devices,
        layout=layout,
        attention="xla_jit",
        scan=True,
    )
    assert "PROTENIX_ATOM_CP_OK" in output, output
    assert "scan=True" in output, output


def test_the_distributed_atom_graph_matches_serial_with_a_token_query_chunk(
    tmp_path,
) -> None:
    """A query chunk slices the token axis the scatter now delivers sharded.

    ``chunk_policy=auto`` emits a token query chunk at real sizes, and the
    axis it slices is the one ``shard_single`` newly constrains, so the
    composition is not hypothetical.
    """

    output = _run(
        "tests.models.protenix.scripts.atom_cp_parity",
        devices=4,
        layout="2d",
        attention="xla_jit",
        scan=True,
        token_chunk=8,
    )
    assert "PROTENIX_ATOM_CP_OK" in output, output
    assert "token_chunk=8" in output, output


@pytest.mark.parametrize(("devices", "layout"), MESH_ARMS)
def test_the_atom_token_routing_adapters_match_serial(
    devices: int,
    layout: str,
) -> None:
    output = _run(
        "tests.models.protenix.scripts.atom_cp_routing",
        devices=devices,
        layout=layout,
    )
    assert "PROTENIX_ATOM_ROUTING_OK" in output, output


def test_a_mesh_does_not_leak_into_the_serial_program_around_it() -> None:
    output = _run(
        "tests.models.protenix.scripts.atom_cp_invariance",
        devices=4,
    )
    assert "PROTENIX_ATOM_INVARIANCE_OK" in output, output


def test_the_atom_window_option_is_part_of_the_compile_identity(tmp_path) -> None:
    """Two atom-graph programs must not share one cache namespace.

    And an explicit ``true`` must share the namespace an omitted option
    selects, which is what the released-defaults strip is for.
    """

    from foldjax import PredictionRequest
    from foldjax.backends.protenix import ProtenixBackend

    job = tmp_path / "job.json"
    job.write_text(
        '{"name": "t", "entities": [{"type": "protein", "id": ["A"], '
        '"sequence": "GRISMTVKKLYFIPAGRCMLDHSSVNSALTPGK"}]}'
    )
    weights = tmp_path / "protenix.weights"
    weights.touch()
    backend = ProtenixBackend()

    def profile(options: dict[str, object]) -> dict[str, object]:
        return backend.cache_profile(
            PredictionRequest(
                model="protenix",
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
