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

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.models.msa_sampling import (
    drop_sampled_msa_source_features,
    sample_opendde_msa_cycle_features,
)
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    _stacked_cycle_msa,
    pairformer_output_from_s_inputs,
)
from tests.models.cp_probe_env import inherited_environment
from tests.models.protenix.test_trunk import _pairformer_output_params

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
        num_recycles=4,
        seed=101,
    )

    stacked = _stacked_cycle_msa(cycles)

    assert list(stacked) == sorted(stacked)
    for name, value in stacked.items():
        assert value.shape == (4, *cycles[0][name].shape)


def _trunk_features(raw_depth: int) -> dict[str, jax.Array]:
    tokens = 4
    return {
        "relp": jnp.zeros((tokens, tokens, 2), dtype=jnp.float32),
        "token_bonds": jnp.zeros((tokens, tokens), dtype=jnp.float32),
        "msa": jnp.zeros((raw_depth, tokens), dtype=jnp.int32),
        "has_deletion": jnp.full((raw_depth, tokens), jnp.nan),
        "deletion_value": jnp.full((raw_depth, tokens), jnp.inf),
        "msa_mask": jnp.ones((raw_depth, tokens), dtype=jnp.float32),
    }


def _sampled_cycles() -> tuple[dict[str, jax.Array], ...]:
    cycle = {
        "msa": jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        "has_deletion": jnp.zeros((3, 4), dtype=jnp.float32),
        "deletion_value": jnp.zeros((3, 4), dtype=jnp.float32),
        "msa_mask": jnp.ones((3, 4), dtype=jnp.float32),
    }
    return (cycle, cycle)


def test_sampled_msa_source_pruning_keeps_trunk_hlo_and_bounds_cache(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "xla")
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    params = _pairformer_output_params()
    s_inputs = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    cycles = _sampled_cycles()

    def trunk(features, sampled):
        return pairformer_output_from_s_inputs(
            features,
            s_inputs,
            params,
            num_recycles=2,
            cycle_msa_features=sampled,
            single_attention_backend="xla",
            triangle_attention_backend="xla",
        )

    raw_17 = _trunk_features(17)
    pruned_17 = drop_sampled_msa_source_features(raw_17, cycles)
    compiled = jax.jit(trunk)
    raw_output = compiled(raw_17, cycles)
    pruned_output = compiled(pruned_17, cycles)
    for raw_value, pruned_value in zip(raw_output, pruned_output, strict=True):
        raw_array = np.asarray(raw_value)
        np.testing.assert_array_equal(raw_array, np.asarray(pruned_value))
        assert np.isfinite(raw_array).all()

    raw_hlo = str(compiled.lower(raw_17, cycles).compiler_ir(dialect="stablehlo"))
    pruned_hlo = str(compiled.lower(pruned_17, cycles).compiler_ir(dialect="stablehlo"))
    assert raw_hlo == pruned_hlo

    def make_cached():
        return jax.jit(lambda features: trunk(features, cycles))

    raw_cached = make_cached()
    pruned_cached = make_cached()
    for raw_depth in (17, 23, 31):
        raw_features = _trunk_features(raw_depth)
        raw_cached(raw_features)
        pruned_cached(drop_sampled_msa_source_features(raw_features, cycles))

    assert raw_cached._cache_size() == 3
    assert pruned_cached._cache_size() == 1
