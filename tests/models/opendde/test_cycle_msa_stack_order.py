"""OpenDDE's recycled MSA stack must lower the same way in every process.

OpenDDE resamples its alignment per recycle, so its trunk call carries
``cycle_msa_features`` and the shared Protenix trunk stacks them onto a cycle
axis before ``lax.scan``. Those ``stack`` equations are emitted in whatever
order the feature names are walked in, and a ``set`` walks them in hash order,
which CPython randomizes per process. The stacked arrays are identical either
way; the traced module is not, and a module that serializes differently every
process can never hit the persistent compile cache. Measured on a 132-token
job, that turned a 3.4 s warm run into a 32-57 s one -- attributed for a long
time to recycling, which actually costs 0.24 s per cycle.

The property is cross-process, so the test is too: the same stack is traced
under several ``PYTHONHASHSEED`` values and the jaxpr has to come out identical.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

from foldjax.models.opendde.models.msa_sampling import sample_opendde_msa_cycle_features
from foldjax.models.protenix.models.trunk_blocks.trunk import _stacked_cycle_msa
from tests.models.cp_probe_env import inherited_environment

# Enough names that hash order agreeing with sorted order by luck is not a
# thing that happens: 4 features leave a 1-in-24 chance, 8 leave 1 in 40,320.
_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp

    from foldjax.models.protenix.models.trunk_blocks.trunk import _stacked_cycle_msa

    names = [
        "msa", "msa_mask", "has_deletion", "deletion_value",
        "profile", "cluster_profile", "extra_msa", "extra_deletion",
    ]
    cycles = tuple(
        {name: jnp.zeros((2, 3), jnp.float32) for name in names} for _ in range(3)
    )
    print(",".join(_stacked_cycle_msa(cycles)))
    print(jax.make_jaxpr(_stacked_cycle_msa)(cycles))
    """
)


def _trace_under(hash_seed: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PYTHONHASHSEED": hash_seed,
            "JAX_PLATFORMS": "cpu",
            **inherited_environment(),
        },
    )
    return completed.stdout


def test_cycle_msa_stack_traces_identically_under_every_hash_seed() -> None:
    traces = {seed: _trace_under(seed) for seed in ("0", "1", "2", "3")}

    reference = traces["0"]
    assert reference.splitlines()[0].split(",") == sorted(
        reference.splitlines()[0].split(",")
    )
    for seed, trace in traces.items():
        assert trace == reference, f"PYTHONHASHSEED={seed} traced a different module"


def test_opendde_cycle_features_stack_in_a_fixed_order() -> None:
    msa = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    cycles = sample_opendde_msa_cycle_features(
        {
            "msa": msa,
            "has_deletion": msa.astype(np.float32),
            "deletion_value": msa.astype(np.float32),
        },
        n_cycle=4,
        seed=101,
    )

    stacked = _stacked_cycle_msa(cycles)

    assert list(stacked) == sorted(stacked)
    for name, value in stacked.items():
        assert value.shape == (4, *cycles[0][name].shape)
