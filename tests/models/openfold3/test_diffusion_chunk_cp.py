"""OpenFold3's diffusion sample width under context parallelism.

A mesh is asked for because a target does not otherwise fit, and under one the
sample axis is the axis sharding does not touch: the rollout's widest value is
the diffusion pair conditioning, which ``denoise_fn`` widens to the sample
width, and at 6,568 tokens on four devices that is ``f32[5, N/4, N, 128]`` --
25.7 GiB per rank. So an omitted ``diffusion_chunk_size`` resolves to
:data:`~foldjax.models.openfold3.inference.CP_DIFFUSION_CHUNK_SIZE` under a
mesh and keeps resolving from the sample count without one.

Three kinds of gate, and they answer different questions:

* the **resolution rule** -- which automatic answer an omitted option gets,
  that an explicit one always wins, and that a serial run is untouched;
* the **identity rule** -- the compile profile records the width the run
  resolves to, so the chunked mesh program and the unchunked rollout cannot
  share one cache namespace and an explicitly spelled width that *is* the
  resolved one shares the namespace it names;
* the **program**, in a subprocess with four forced CPU devices, because a
  device count has to be set before JAX initialises. That is the census in
  ``scripts/diffusion_chunk_cp_census.py``, which holds its own assertions --
  including the tripwire that the unchunked arm really does carry the five-wide
  tile, without which "no five-wide tile" says nothing, and which was checked
  to fail against a conditioning deliberately retained at full width outside
  the loop.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from foldjax.api import resolve_cache_dir
from foldjax.backends.openfold3 import OpenFold3Backend
from foldjax.models.openfold3.inference import (
    CP_DIFFUSION_CHUNK_SIZE,
    released_config,
    resolve_diffusion_chunk_size,
)
from foldjax.schema import PredictionRequest
from tests.models.cp_probe_env import inherited_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Four devices: enough for the 4-shard 1-D mesh ``cp_layout="auto"`` selects,
#: and the census depends on the count through the local row width it looks for.
_DEVICES = 4

#: Token and atom counts for the configuration checks. Inside the pair chunk
#: law's validated domain, so ``released_config`` resolves a real row width and
#: the sample-axis question is not entangled with an ``unknown`` admission.
_N_TOKEN, _N_ATOM = 1200, 9000


def _width(**changes) -> int | None:
    return released_config(
        n_token=_N_TOKEN, n_atom=_N_ATOM, **changes
    ).diffusion_chunk_size


def test_an_omitted_width_resolves_from_the_shard_count() -> None:
    """One sample at a time under a mesh; the sample count's width without one.

    Both directions stated. A change that made the mesh default unconditional
    would fail the serial rows, and one that dropped it would fail the mesh
    rows -- neither can pass by loosening the other.
    """

    assert _width() is None
    assert _width(cp_shards=1) is None
    assert _width(num_samples=6) == 5
    for shards in (2, 4, 9):
        assert _width(cp_shards=shards) == CP_DIFFUSION_CHUNK_SIZE
        # The mesh answer does not read the sample count: it is the capacity
        # the shard count implies, not a width scaled off the schedule.
        assert _width(cp_shards=shards, num_samples=6) == CP_DIFFUSION_CHUNK_SIZE
        assert _width(cp_shards=shards, num_samples=1) == CP_DIFFUSION_CHUNK_SIZE


def test_an_explicit_width_wins_everywhere() -> None:
    """Including ``None``, which is how the unchunked rollout is asked for.

    ``models/sampler.py`` takes the unchunked branch for ``None`` and for any
    width at or above the sample count, so under a mesh a caller with the room
    has two spellings and neither is the default.
    """

    for shards in (1, 4):
        assert _width(cp_shards=shards, diffusion_chunk_size=None) is None
        assert _width(cp_shards=shards, diffusion_chunk_size=2) == 2
        assert _width(cp_shards=shards, diffusion_chunk_size=5) == 5


def test_the_resolver_takes_the_sentinel_and_nothing_else() -> None:
    """The unit the two callers share, at the boundary that decides."""

    assert (
        resolve_diffusion_chunk_size("auto", num_samples=5, cp_shards=4)
        == CP_DIFFUSION_CHUNK_SIZE
    )
    assert resolve_diffusion_chunk_size("auto", num_samples=5, cp_shards=1) is None
    assert resolve_diffusion_chunk_size("auto", num_samples=6, cp_shards=1) == 5
    assert resolve_diffusion_chunk_size(None, num_samples=5, cp_shards=4) is None
    assert resolve_diffusion_chunk_size(3, num_samples=5, cp_shards=4) == 3
    # A width at or above the sample count is the unchunked rollout, and it is
    # passed through as written rather than folded: the sampler's own `>=`
    # branch is what makes the two one program, and folding here would make
    # the config claim a width the caller did not ask for.
    assert resolve_diffusion_chunk_size(9, num_samples=5, cp_shards=4) == 9


def _request(tmp_path: Path, **options) -> PredictionRequest:
    job = tmp_path / "job.json"
    job.write_text(
        '{"name": "t", "entities": [{"type": "protein", "id": ["A"], '
        '"sequence": "GRISMTVKKLYFIPAGRCMLDHSSVNSALTPGK"}]}',
        encoding="utf-8",
    )
    weights = tmp_path / "openfold3.weights"
    weights.touch()
    return PredictionRequest(
        model="openfold3",
        input=job,
        # An explicit path: this gate exercises cache identity, not managed
        # checkpoint discovery.
        weights=weights,
        cache_dir=tmp_path / "cache",
        options=options,
    )


def test_the_profile_records_the_width_the_run_resolves_to(tmp_path: Path) -> None:
    """Not the spelling, and never stripped.

    The mesh default is a different program from the rollout every recorded
    run before it took, so absence cannot keep meaning "unchunked": the two
    would share one namespace and one in-process JIT owner. Recording the
    resolved width also gives the aliasing back -- an explicit ``1`` under a
    mesh is the namespace an omitted option names.
    """

    backend = OpenFold3Backend()
    serial = _request(tmp_path)
    mesh = _request(tmp_path, cp_devices=_DEVICES)

    def profile(**options) -> dict:
        return backend.cache_profile(_request(tmp_path, **options))

    assert profile()["diffusion_chunk_size"] is None
    assert profile(cp_devices=_DEVICES)["diffusion_chunk_size"] == (
        CP_DIFFUSION_CHUNK_SIZE
    )
    # Spelling the resolved width is the same program and the same namespace.
    spelled = dataclasses.replace(
        mesh, options={"cp_devices": _DEVICES, "diffusion_chunk_size": 1}
    )
    assert backend.cache_profile(spelled) == backend.cache_profile(mesh)
    assert resolve_cache_dir(spelled, backend) == resolve_cache_dir(mesh, backend)
    # The unchunked rollout under a mesh is a *different* program, and stays
    # distinguishable from having said nothing.
    unchunked = dataclasses.replace(
        mesh, options={"cp_devices": _DEVICES, "diffusion_chunk_size": 5}
    )
    assert backend.cache_profile(unchunked)["diffusion_chunk_size"] == 5
    assert resolve_cache_dir(unchunked, backend) != resolve_cache_dir(mesh, backend)
    # And the serial namespace is not the mesh one, which the shard count
    # already said; asserted here so a resolver that ignored `cp_devices`
    # could not pass this file.
    assert resolve_cache_dir(serial, backend) != resolve_cache_dir(mesh, backend)
    # A spelling `validate_native_options` refuses keeps its own raw value
    # rather than inheriting a real run's entry.
    refused = {"cp_devices": _DEVICES, "diffusion_chunk_size": "wide"}
    assert profile(**refused)["diffusion_chunk_size"] == "wide"
    with pytest.raises(ValueError, match="diffusion_chunk_size"):
        backend.validate_native_options(dict(refused))


def test_the_context_parallel_rollout_holds_no_five_wide_pair_tile() -> None:
    """The program, on four forced CPU devices. See the script's own docstring."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.models.openfold3.scripts.diffusion_chunk_cp_census",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={_DEVICES}",
            "FOLDJAX_CP_PROBE_DEVICES": str(_DEVICES),
            "FOLDJAX_CP_PROBE_LAYOUT": "1d",
            **inherited_environment(),
            # After `inherited_environment`, so the tree under test wins over
            # whatever the parent's PYTHONPATH points at: `foldjax` is
            # installed editable, and in a worktree the install and this tree
            # are different checkouts -- a probe that reported on the installed
            # one would report on code nobody edited. The script prints the
            # `inference` module it imported for the same reason.
            "PYTHONPATH": f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
        },
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OPENFOLD3_DIFFUSION_CHUNK_CP_OK" in completed.stdout, completed.stdout
