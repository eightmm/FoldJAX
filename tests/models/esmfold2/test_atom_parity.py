"""The atom stack against the creator reference and Hugging Face integration."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from foldjax import paths  # noqa: E402
from foldjax.models.esmfold2.models import atom  # noqa: E402


def _hf_atom_upstream():
    """The Hugging Face atom stack, or a skip.

    Not a module-level `importorskip`, which skipped the window test too --
    that one is pure JAX and has no business needing torch installed.

    This is deliberately separate from the Biohub creator snapshot used by the
    rotary test. The later Hugging Face integration exposes CamelCase config
    modules and a separately re-exported checkpoint.
    """
    pytest.importorskip("torch")
    return pytest.importorskip(
        "transformers.models.esmfold2.modeling_esmfold2"
    )

ATOL = 2e-5
# The rotary table needs head_dim // 2 >= 3*2 + 10 = 16, so head_dim >= 32 --
# which is what the released widths give (d_atom 128 over 4 heads). A smaller
# toy would not exercise the same code path; it would fail to build one.
N_ATOMS, D_ATOM, N_HEADS = 12, 64, 2


def _atom_inputs(seed=0, n_atoms=N_ATOMS):
    rng = np.random.default_rng(seed)
    return {
        "ref_pos": rng.standard_normal((1, n_atoms, 3)).astype(np.float32),
        "ref_space_uid": np.arange(n_atoms, dtype=np.int64)[None] // 3,
        "ref_charge": np.zeros((1, n_atoms), dtype=np.float32),
        "ref_element": np.eye(128, dtype=np.float32)[np.full(n_atoms, 6)][None],
        "ref_atom_name_chars": np.eye(64, dtype=np.float32)[
            np.tile(np.array([35, 33, 0, 0]), (n_atoms, 1))
        ][None],
        "atom_to_token": (np.arange(n_atoms, dtype=np.int64)[None] // 3),
        "atom_attention_mask": np.ones((1, n_atoms), dtype=np.float32),
    }


def test_the_rotary_tables_match_upstream() -> None:
    """Same axes, same frequency layout, same half-split pairing -- exactly.

    This asserted `atol=rtol=3e-3`, without the reason `PROJECT.md` asks for
    above 1e-3, in a file that otherwise holds itself to 2e-5. Measured, the
    two sides really did differ by 1.9e-3 in float32, so the tolerance was
    sized to a real gap -- and the gap was a dtype mismatch rather than a
    disagreement. FoldJAX returns these tables in bfloat16 because upstream
    hardcodes that, and the comparison took upstream's float32 reference.
    2e-3 is what bfloat16 costs on values in [-1, 1]: the tolerance was the
    size of the cast, and would have hidden any defect smaller than it.

    Current upstream exposes the same free ``build_3d_rope`` function that the
    port was derived from. They are bit-identical, and this asserts that rather
    than hiding the bfloat16 cast behind a tolerance.

    This one is comparable without a checkpoint because the function has no
    parameters or buffers. Everything below it needs a checkpoint, and there
    the library and the published weights no longer share names; see the skip
    on the encoder test.
    """
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip(
        "transformers.models.esmfold2.modeling_esmfold2_common"
    )

    head_dim = D_ATOM // N_HEADS
    data = _atom_inputs()

    with torch.no_grad():
        cos, sin = modeling.build_3d_rope(
            ref_pos=torch.from_numpy(data["ref_pos"]),
            ref_space_uid=torch.from_numpy(data["ref_space_uid"]),
            head_dim=head_dim,
            n_spatial_per_axis=2,
            n_uid_pairs=10,
            spatial_base_freq=20.0,
            uid_base_freq=10000.0,
        )
    got_cos, got_sin = atom.build_3d_rope(
        data["ref_pos"], data["ref_space_uid"], head_dim=head_dim
    )

    for name, expected, got in (("cos", cos, got_cos), ("sin", sin, got_sin)):
        reference = expected.float().numpy()
        produced = np.asarray(got).astype(np.float32)
        assert np.array_equal(produced, reference), (
            f"{name} table is no longer bit-identical to upstream's"
        )


def test_the_window_is_measured_in_packed_rank() -> None:
    """Padding between two atoms must not push them out of each other's window."""
    valid = np.array([[1, 0, 0, 1, 1]], dtype=np.float32)
    allowed = np.asarray(atom.sliding_window_mask(valid, half_window=1))
    assert allowed[0, 0, 3], "rank 0 and rank 1 are adjacent despite two pad atoms"
    assert not allowed[0, 0, 4], "rank 0 and rank 2 are outside a window of 1"
    assert allowed[0, 1, 1], "the diagonal is always allowed, padding included"


def _released_atom_encoder():
    """Both sides' real weights, or a skip.

    They are the same trained tensors: `atom_linear.weight`, `atom_norm.weight`,
    `atom_to_token_linear.weight` and the adaLN projection are bit-identical
    between the two artifacts, and the checkpoint's fused `Wqkv` is exactly
    `concat(q_proj, k_proj, v_proj)` from the re-export. Only the names and the
    QKV packing differ, so each side loads its own file and no name is ever
    mapped onto another -- which is the rename table that naming after the
    checkpoint was chosen to delete.

    `tests/models/esmfold2/fetch_hf_atom_weights.py` writes the library's side;
    `bench/upstream-environments.md` has the recipe.
    """
    modeling = _hf_atom_upstream()
    weights = os.environ.get("ESMFOLD2_HF_ATOM_WEIGHTS")
    if not weights or not Path(weights).is_file():
        pytest.skip(
            "set ESMFOLD2_HF_ATOM_WEIGHTS to the .npz written by "
            "tests/models/esmfold2/fetch_hf_atom_weights.py"
        )
    store = paths.weights_dir("esmfold2") / "model.safetensors"
    if not store.is_file():
        pytest.skip("the published ESMFold2 checkpoint is not in the weight store")
    return modeling, Path(weights), store


def test_the_atom_encoder_matches() -> None:
    """The released atom encoder, both sides on their own real weights.

    The embedding is exact to float32. Past it the two diverge by 1.3e-2, and
    all of it is one deliberate cast: this port drops q/k/v to bfloat16 when the
    caller is float32, which the reference it was ported from did and the
    current library does not. Removing that cast alone takes the layers from
    4.2e-3 to 4.9e-6 and the token output from 1.3e-2 to 2.8e-5, so nothing
    else in the stack disagrees.

    The cast is not a defect -- at the released 32 samples and 7,776 atoms the
    float32 scores would be 29,524 MiB -- and the released path runs bfloat16,
    where the cast does not fire at all. It is only visible to a float32 caller,
    which is what this test is. Hence one tight assertion on the embedding and
    one documented tolerance past it, rather than one loose number over both.
    """
    # After the guard, never before it: an unguarded `import torch` turns the
    # skip this is supposed to take into a failure in the shipped environment,
    # which is Torch-free on purpose.
    modeling, hf_weights, store = _released_atom_encoder()

    import torch
    from safetensors import safe_open
    from transformers.models.esmfold2.configuration_esmfold2 import EsmFold2Config

    data = _atom_inputs(1)
    mask_bool = torch.from_numpy(data["atom_attention_mask"]).bool()
    tokens = int(data["atom_to_token"].max()) + 1

    config = EsmFold2Config.from_pretrained(str(hf_weights.parent))
    encoder = modeling.EsmFold2AtomEncoder(config, config.atom_encoder).eval()
    library = np.load(hf_weights)
    encoder.load_state_dict(
        {
            name[len("input_embedder.atom_encoder.") :]: torch.from_numpy(
                np.array(library[name])
            )
            for name in library.files
        },
        strict=True,
    )
    with torch.no_grad():
        inputs = modeling.EsmFold2AtomInputs(
            torch.from_numpy(data["ref_pos"]),
            torch.from_numpy(data["ref_charge"]),
            torch.from_numpy(data["atom_attention_mask"]),
            torch.from_numpy(data["ref_element"]),
            torch.from_numpy(data["ref_atom_name_chars"]),
            torch.from_numpy(data["ref_space_uid"]),
            torch.from_numpy(data["atom_to_token"]),
        )
        embeds, position = encoder.embed_atoms(inputs)
        attention = encoder.build_attention_mask(mask_bool, embeds)
        expected_tokens, _ = encoder(
            embeds,
            embeds,
            attention,
            position,
            mask_bool,
            torch.from_numpy(data["atom_to_token"]),
            tokens,
        )

    prefix = "inputs_embedder.atom_attention_encoder"
    with safe_open(str(store), framework="numpy") as handle:
        params = {
            name: handle.get_tensor(name)
            for name in handle.keys()
            if name.startswith(f"{prefix}.")
        }
    got_tokens, _, conditioning, _ = atom.atom_encoder(
        data["ref_pos"],
        data["atom_attention_mask"],
        data["ref_space_uid"],
        data["ref_charge"],
        data["ref_element"],
        data["ref_atom_name_chars"],
        data["atom_to_token"],
        params,
        prefix,
        n_blocks=config.atom_encoder.num_hidden_layers,
        n_heads=config.atom_encoder.num_attention_heads,
        half_window=config.sliding_window // 2,
    )

    # Nothing narrows the dtype before the layers, so this one is exact.
    np.testing.assert_allclose(
        np.asarray(conditioning, dtype=np.float32),
        embeds.float().numpy(),
        atol=ATOL,
        rtol=ATOL,
    )
    # Past them, the deliberate bfloat16 cast on q/k/v. Measured at 1.3e-2 here
    # and at 2.8e-5 with the cast removed, so this bound is the cast's size and
    # is documented rather than chosen.
    np.testing.assert_allclose(
        np.asarray(got_tokens, dtype=np.float32),
        expected_tokens.float().numpy(),
        atol=2e-2,
        rtol=2e-2,
    )
