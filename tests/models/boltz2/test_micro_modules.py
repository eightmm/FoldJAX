import jax
import numpy as np

from foldjax.models.boltz2.models.primitives.micro_modules import (
    MicroConfig,
    init_micro_inputs,
    init_micro_params,
    jax_pairformer_forward,
    jax_structure_forward,
    to_jax_tree,
    to_torch_tree,
    torch_pairformer_forward,
    torch_structure_forward,
)


def test_micro_pairformer_and_structure_match_torch() -> None:
    pytest_import_torch()
    config = MicroConfig(
        batch=1,
        residues=8,
        token_s=32,
        token_z=16,
        heads=4,
        pairformer_blocks=1,
        structure_steps=3,
        atoms_per_residue=2,
    )
    params_np = init_micro_params(config, seed=3)
    inputs_np = init_micro_inputs(config, seed=4)
    torch_params = to_torch_tree(params_np, "cpu")
    torch_inputs = to_torch_tree(inputs_np, "cpu")
    jax_params = to_jax_tree(params_np)
    jax_inputs = to_jax_tree(inputs_np)

    torch_s, torch_z = torch_pairformer_forward(
        torch_params["pairformer"],
        torch_inputs["s"],
        torch_inputs["z"],
        torch_inputs["mask"],
        torch_inputs["pair_mask"],
    )
    jax_s, jax_z = jax_pairformer_forward(
        jax_params["pairformer"],
        jax_inputs["s"],
        jax_inputs["z"],
        jax_inputs["mask"],
        jax_inputs["pair_mask"],
    )
    jax.block_until_ready(jax_s)
    np.testing.assert_allclose(torch_s.detach().numpy(), np.asarray(jax_s), atol=1e-3)
    np.testing.assert_allclose(torch_z.detach().numpy(), np.asarray(jax_z), atol=1e-3)

    torch_coords = torch_structure_forward(
        torch_params["structure"],
        torch_s,
        torch_z,
        torch_inputs["coords"],
        torch_inputs["atom_token"],
        config.structure_steps,
    )
    jax_coords = jax_structure_forward(
        jax_params["structure"],
        jax_s,
        jax_z,
        jax_inputs["coords"],
        jax_inputs["atom_token"],
        config.structure_steps,
    )
    jax.block_until_ready(jax_coords)
    np.testing.assert_allclose(
        torch_coords.detach().numpy(),
        np.asarray(jax_coords),
        atol=2e-3,
    )


def pytest_import_torch():
    import pytest

    return pytest.importorskip("torch")
