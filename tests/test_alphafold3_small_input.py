"""Regression gates for FoldJAX's managed AlphaFold 3 source patch."""

from __future__ import annotations

import numpy as np


def test_atom_cross_attention_pads_a_shorter_than_key_window_layout() -> None:
    """Four-residue jobs work without forcing the released 256-token bucket."""
    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model import features
    from alphafold3.model.atom_layout import atom_layout

    shape = (4, 24)
    atom_name = np.full(shape, "", dtype=object)
    atom_element = np.full(shape, "", dtype=object)
    res_name = np.full(shape, "", dtype=object)
    res_id = np.zeros(shape, dtype=int)
    chain_id = np.full(shape, "", dtype=object)
    chain_type = np.full(shape, "", dtype=object)
    for residue_index in range(4):
        atom_name[residue_index, :5] = ("N", "CA", "C", "O", "CB")
        atom_element[residue_index, :5] = ("N", "C", "C", "O", "C")
        res_name[residue_index, :5] = "ALA"
        res_id[residue_index, :5] = residue_index + 1
        chain_id[residue_index, :5] = "A"
        chain_type[residue_index, :5] = "polypeptide(L)"

    layout = atom_layout.AtomLayout(
        atom_name=atom_name,
        atom_element=atom_element,
        res_name=res_name,
        res_id=res_id,
        chain_id=chain_id,
        chain_type=chain_type,
    )
    result = features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=layout,
        queries_subset_size=32,
        keys_subset_size=128,
        padding_shapes=features.PaddingShapes(
            num_tokens=4,
            msa_size=1,
            num_chains=1,
            num_templates=0,
            num_atoms=96,
        ),
    )

    assert result.tokens_to_keys.gather_idxs.shape == (3, 128)
    assert result.tokens_to_keys.gather_mask.shape == (3, 128)
    assert np.count_nonzero(result.tokens_to_keys.gather_mask) == 3 * 20


def test_diffusion_padding_preserves_multisample_noise_across_every_step(
    monkeypatch,
) -> None:
    """Serving buckets keep AF3's exact compact random stream on real tokens."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from foldjax.models.alphafold3 import build

    build.register_runtime()

    from alphafold3.model.network import diffusion_head

    monkeypatch.setattr(
        diffusion_head,
        "random_augmentation",
        lambda rng_key, positions, mask: positions * mask[..., None],
    )
    config = diffusion_head.SampleConfig(
        steps=4,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        num_samples=2,
    )

    def forward(atom_mask, token_mask, sample_key, prefix_stable_noise):
        batch = type("Batch", (), {})()
        batch.predicted_structure_info = type("Pred", (), {"atom_mask": atom_mask})()
        batch.token_features = type("Tokens", (), {"mask": token_mask})()
        return diffusion_head.sample(
            denoising_step=lambda positions, noise_level: positions,
            batch=batch,
            key=sample_key,
            config=config,
            prefix_stable_noise=prefix_stable_noise,
        )

    transformed = hk.transform(forward)
    sample_key = jax.random.PRNGKey(23)
    exact_mask = jnp.ones((3, 4), dtype=jnp.float32)
    padded_mask = jnp.pad(exact_mask, ((0, 5), (0, 0)))
    exact_token_mask = jnp.ones((3,), dtype=jnp.float32)
    padded_token_mask = jnp.pad(exact_token_mask, (0, 5))

    run_key, initial_noise_key = jax.random.split(sample_key)
    exact_initial_noise = jax.random.normal(
        initial_noise_key,
        (config.num_samples, *exact_mask.shape, 3),
    )
    padded_initial_noise = diffusion_head._prefix_stable_atom_noise(
        initial_noise_key,
        jnp.broadcast_to(
            padded_token_mask[None, :],
            (config.num_samples, padded_token_mask.shape[0]),
        ),
        trailing_shape=exact_mask.shape[1:] + (3,),
    )
    np.testing.assert_array_equal(
        np.asarray(padded_initial_noise[:, :3]),
        np.asarray(exact_initial_noise),
    )
    assert not np.any(np.asarray(padded_initial_noise[:, 3:]))

    step_keys = jax.random.split(run_key, config.num_samples)
    for _ in range(config.steps):
        split_keys = jax.vmap(lambda key: jax.random.split(key, 3))(step_keys)
        step_keys = split_keys[:, 0]
        noise_keys = split_keys[:, 1]
        exact_step_noise = jax.vmap(
            lambda key: jax.random.normal(key, (*exact_mask.shape, 3))
        )(noise_keys)
        padded_step_noise = jax.vmap(
            lambda key: diffusion_head._prefix_stable_atom_noise(
                key,
                padded_token_mask,
                trailing_shape=exact_mask.shape[1:] + (3,),
            )
        )(noise_keys)
        np.testing.assert_array_equal(
            np.asarray(padded_step_noise[:, :3]),
            np.asarray(exact_step_noise),
        )
        assert not np.any(np.asarray(padded_step_noise[:, 3:]))

    params = transformed.init(
        sample_key,
        exact_mask,
        exact_token_mask,
        sample_key,
        False,
    )
    exact = transformed.apply(
        params,
        sample_key,
        exact_mask,
        exact_token_mask,
        sample_key,
        False,
    )["atom_positions"]
    padded = transformed.apply(
        params,
        sample_key,
        padded_mask,
        padded_token_mask,
        sample_key,
        True,
    )["atom_positions"]

    # The stochastic tape above is bitwise identical. The full padded scan can
    # still choose a shape-specific fused arithmetic schedule on CPU, so compare
    # its real-token rollout at a tight float32 tolerance.
    np.testing.assert_allclose(
        np.asarray(padded[:, :3]),
        np.asarray(exact),
        rtol=3e-6,
        atol=1e-3,
    )
    assert not np.any(np.asarray(padded[:, 3:]))
