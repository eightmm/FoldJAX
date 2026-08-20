from __future__ import annotations

import re
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def _replace_once(path: str, old: str, new: str) -> None:
    content = _read(path)
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    _write(path, content.replace(old, new, 1))


def _regex_once(path: str, pattern: str, replacement: str) -> None:
    content = _read(path)
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{path}: regex target did not match exactly once")
    _write(path, updated)


def _patch_cp_runtime() -> None:
    path = "src/foldjax/models/_cp.py"
    marker = "\n\n@dataclass(frozen=True, slots=True)\nclass CPRuntime:"
    atom_registry = '''

#: Boltz-2 atom features that are linear in the atom count and may enter an
#: atom-window CP program already sharded over CP rows. Coupled atom/token
#: matrices are deliberately absent: they need a model-specific sparse
#: contract rather than independent axis sharding.
ATOM_FEATURE_AXES: dict[str, int] = {
    "atom_backbone_feat": 1,
    "atom_pad_mask": 1,
    "atom_resolved_mask": 1,
    "atom_to_token_ids_global": 1,
    "atom_to_token_valid": 1,
    "bfactor": 1,
    "coords": 2,
    "plddt": 1,
    "ref_atom_name_chars": 1,
    "ref_charge": 1,
    "ref_chirality": 1,
    "ref_element": 1,
    "ref_pos": 1,
    "ref_space_uid": 1,
}


@dataclass(frozen=True, slots=True)
class CPRuntime:'''
    _replace_once(path, marker, atom_registry)

    feature_block = '''def _atom_spec_for_feature(name: str, value: Any) -> PartitionSpec | None:
    """Return entry sharding for one explicitly safe linear atom feature."""

    if not _is_movable_array(value):
        return None
    atom_axis = ATOM_FEATURE_AXES.get(name)
    if atom_axis is None:
        return None
    resolved = _resolve_axis(atom_axis, value.ndim, what="atom axis")
    shape = tuple(int(size) for size in value.shape)
    if shape[resolved] <= 0 or shape[resolved] % cp_row_shards():
        return None
    entries: list[str | None] = [None] * value.ndim
    entries[resolved] = CP_ROW_AXIS if cp_layout() == "2d" else CP_AXIS
    return PartitionSpec(*entries)


def feature_spec(
    name: str,
    value: Any,
    *,
    shard_atom_features: bool = False,
) -> PartitionSpec | None:
    """Return a safe semantic entry sharding for one known input feature.

    Uneven inputs stay replicated. Pair features use both mesh axes when the
    layout is two-dimensional. Linear atom features are only considered when
    the caller explicitly enables the Boltz-2 atom-window contract; coupled
    atom/token matrices remain replicated.
    """

    if cp_mesh() is None:
        return None
    axes = _pair_axes_for_feature(name, value)
    if axes is not None:
        shape = tuple(int(size) for size in value.shape)
        if _axes_divide_mesh(shape, *axes):
            return pair_spec(value.ndim, row_axis=axes[0], col_axis=axes[1])
    if shard_atom_features:
        return _atom_spec_for_feature(name, value)
    return None


def _replicated_sharding'''
    _regex_once(
        path,
        r"def feature_spec\(.*?\n\n\ndef _replicated_sharding",
        feature_block,
    )

    tree_block = '''def _place_tree(
    node: Any,
    *,
    shard_pair_features: bool,
    shard_atom_features: bool,
) -> Any:
    if isinstance(node, Mapping):
        placed: dict[Any, Any] = {}
        for key, value in node.items():
            spec = (
                feature_spec(
                    str(key),
                    value,
                    shard_atom_features=shard_atom_features,
                )
                if isinstance(key, str)
                and (shard_pair_features or shard_atom_features)
                else None
            )
            if spec is not None:
                # Do not place a pair feature when only atom entry sharding was
                # requested, or vice versa. The semantic registry is explicit
                # so this remains a local decision rather than a shape guess.
                is_pair = _pair_axes_for_feature(str(key), value) is not None
                if (is_pair and shard_pair_features) or (
                    not is_pair and shard_atom_features
                ):
                    placed[key] = _place_leaf(value, spec=spec)
                    continue
            placed[key] = _place_tree(
                value,
                shard_pair_features=shard_pair_features,
                shard_atom_features=shard_atom_features,
            )
        return _rebuild_mapping(node, placed)

    if isinstance(node, tuple) and hasattr(node, "_fields"):
        return type(node)(
            *(
                _place_tree(
                    value,
                    shard_pair_features=shard_pair_features,
                    shard_atom_features=shard_atom_features,
                )
                for value in node
            )
        )
    if isinstance(node, tuple):
        return tuple(
            _place_tree(
                value,
                shard_pair_features=shard_pair_features,
                shard_atom_features=shard_atom_features,
            )
            for value in node
        )
    if isinstance(node, list):
        return [
            _place_tree(
                value,
                shard_pair_features=shard_pair_features,
                shard_atom_features=shard_atom_features,
            )
            for value in node
        ]

    if _is_movable_array(node):
        return _place_leaf(node)

    # Preserve registered custom pytrees while retaining the historical
    # "replicate every numeric leaf" behaviour for model parameter containers.
    try:
        leaves, treedef = jax.tree.flatten(node)
    except TypeError:
        return node
    if len(leaves) == 1 and leaves[0] is node:
        return node
    return jax.tree.unflatten(
        treedef,
        [_place_leaf(leaf) for leaf in leaves],
    )


def replicate_tree(
    tree: Any,
    *,
    shard_pair_features: bool = True,
    shard_atom_features: bool = False,
) -> Any:
    """Place a pytree on the active mesh using explicit semantic registries.

    Parameters and all unrecognised inputs are replicated. Exact, whitelisted
    pair features are placed on the pair mesh when divisible. Linear atom
    features are placed over CP rows only when ``shard_atom_features=True``;
    the default is false so Protenix, OpenDDE and OpenFold3 keep their current
    pair-only contract. Coupled dense atom/token maps are never independently
    sharded here.

    Identity when no mesh is active.
    """

    if cp_mesh() is None:
        return tree
    return _place_tree(
        tree,
        shard_pair_features=shard_pair_features,
        shard_atom_features=shard_atom_features,
    )
'''
    _regex_once(path, r"def _place_tree\(.*\Z", tree_block)


def _patch_atom_runtime() -> None:
    path = "src/foldjax/models/_cp_atom.py"
    marker = "\n\ndef replicate_atoms(array: jax.Array) -> jax.Array:"
    replacement = '''

def place_atoms(array: jax.Array, *, atom_axis: int = -2) -> jax.Array:
    """Place a host/replicated atom array directly on CP-row shards.

    Unlike :func:`shard_atoms`, which constrains an array inside a traced graph,
    this is an entry-placement operation. It is used for precomputed diffusion
    noise tapes so a large ``[steps, samples, atoms, 3]`` input is never first
    copied in full to every device.
    """

    mesh = cp_mesh()
    if mesh is None:
        return array
    resolved = _resolve_axis(atom_axis, array.ndim, name="atom axis")
    if array.shape[resolved] % cp_row_shards():
        raise ValueError(
            f"atom axis {array.shape[resolved]} is not divisible by "
            f"{cp_row_shards()} CP row shards"
        )
    return jax.device_put(
        array,
        NamedSharding(mesh, atom_spec(array.ndim, atom_axis=atom_axis)),
    )


def replicate_atoms(array: jax.Array) -> jax.Array:'''
    _replace_once(path, marker, replacement)


def _patch_output_cropping() -> None:
    path = "src/foldjax/models/boltz2/data/bucket.py"
    old = '''_TOKEN_OUTPUT_AXES: dict[str, tuple[int, ...]] = {
    "plddt": (1,),'''
    new = '''_TOKEN_OUTPUT_AXES: dict[str, tuple[int, ...]] = {
    # Optional trunk captures use the same padded token axes as the model.
    "single_inputs": (1,),
    "single": (1,),
    "pair": (1, 2),
    "plddt": (1,),'''
    _replace_once(path, old, new)


def _patch_boltz_api() -> None:
    path = "src/foldjax/models/boltz2/api.py"
    _replace_once(
        path,
        '    affinity_requested = bool(np.any(feats_np["affinity_token_mask"]))\n',
        '    affinity_requested = stop_after != "trunk" and bool(\n'
        '        np.any(feats_np["affinity_token_mask"])\n'
        '    )\n',
    )
    _replace_once(
        path,
        '''        if (
            padding_plan is not None
            and padding_plan.target["atoms"] > padding_plan.storage["atoms"]
        )''',
        '''        if (
            stop_after != "trunk"
            and padding_plan is not None
            and padding_plan.target["atoms"] > padding_plan.storage["atoms"]
        )''',
    )
    _replace_once(
        path,
        "    from foldjax.models._cp import context_parallel, replicate_tree\n",
        "    from foldjax.models._cp import context_parallel, replicate_tree\n"
        "    from foldjax.models._cp_atom import place_atoms\n",
    )
    old_placement = '''        if cp_devices > 1:
            # A single-device-committed checkpoint fails the multi-device
            # jit's device-assignment check; everything token-linear is
            # replicated onto the mesh and the graph's own constraints shard
            # the pair-shaped state.
            model_args = replicate_tree(model_args)
        out = runner(*model_args)
'''
    new_placement = '''        if cp_devices > 1:
            # Place each semantic input at its production ownership boundary.
            # Parameters and RNG keys are replicated, pair inputs may use the
            # pair mesh, and safe Boltz atom features/noise tapes enter already
            # sharded over CP rows. Dense coupled atom/token maps stay replicated.
            placed_args = list(model_args)
            placed_args[0] = replicate_tree(
                placed_args[0],
                shard_pair_features=False,
            )
            placed_args[1] = replicate_tree(
                placed_args[1],
                shard_atom_features=cp_atom_active,
            )
            placed_args[2] = replicate_tree(
                placed_args[2],
                shard_pair_features=False,
            )
            if cp_atom_active and len(placed_args) == 5:
                placed_args[3] = place_atoms(placed_args[3], atom_axis=-2)
                placed_args[4] = place_atoms(placed_args[4], atom_axis=-2)
            model_args = tuple(placed_args)
        out = runner(*model_args)
'''
    _replace_once(path, old_placement, new_placement)

    early_return = '''        out = runner(*model_args)
    if stop_after == "trunk":
        # The trunk-only graph intentionally has no sampler or confidence
        # outputs. Crop captured padded representations before touching any
        # coordinate field, persist them, and return immediately.
        from foldjax.models.boltz2.data.bucket import crop_prediction_outputs

        public_tokens = (
            padding_plan.actual["tokens"]
            if padding_plan is not None
            else original_tokens
        )
        public_atoms = (
            padding_plan.actual["atoms"]
            if padding_plan is not None
            else original_atoms
        )
        public_out = (
            crop_prediction_outputs(out, public_tokens, public_atoms)
            if padding_plan is not None
            else out
        )
        destination = (
            Path(representations_dir)
            if representations_dir is not None
            else (Path(out_dir) if out_dir is not None else struct_dir.parent)
        )
        archive = _representations.save(
            destination,
            {
                name: public_out[name]
                for name in wanted_representations
                if name in public_out
            },
            _representations.specs_for("boltz2"),
            model="boltz2",
        )
        result: dict[str, Any] = {
            "record_id": record_id,
            "raw": public_out,
            "representations": archive,
        }
        if padding_plan is not None:
            primary_summary = padding_plan.summary()
            primary_summary["static"] = primary_static
            result["padding"] = {"primary": primary_summary}
        return result
'''
    _replace_once(path, "        out = runner(*model_args)\n", early_return)

    _regex_once(
        path,
        r'''\n    if stop_after == "trunk":\n        # No coordinates were predicted,.*?\n\n    public_coords_batched =''',
        "\n    public_coords_batched =",
    )


def _patch_runtime_test() -> None:
    path = "tests/models/test_cp_runtime.py"
    old = '''        assert placed["relp"].sharding.shard_shape(relp.shape) == (1, 4, 4, 3)
        assert placed["ref_pos"].sharding.shard_shape(atom.shape) == atom.shape
        assert placed["odd"].sharding.shard_shape(odd.shape) == odd.shape
        assert feature_spec("relp", odd) is None

        strict = replicate_tree(
'''
    new = '''        assert placed["relp"].sharding.shard_shape(relp.shape) == (1, 4, 4, 3)
        assert placed["ref_pos"].sharding.shard_shape(atom.shape) == atom.shape
        assert placed["odd"].sharding.shard_shape(odd.shape) == odd.shape
        assert feature_spec("relp", odd) is None

        atom_map = jnp.zeros((1, 8, 8), dtype=jnp.float32)
        coords = jnp.zeros((1, 2, 8, 3), dtype=jnp.float32)
        atom_placed = replicate_tree(
            {
                "ref_pos": atom,
                "coords": coords,
                "atom_to_token": atom_map,
            },
            shard_atom_features=True,
        )
        assert atom_placed["ref_pos"].sharding.shard_shape(atom.shape) == (1, 4, 3)
        assert atom_placed["coords"].sharding.shard_shape(coords.shape) == (1, 2, 4, 3)
        # Coupled dense atom/token maps have no generic independent sharding.
        assert atom_placed["atom_to_token"].sharding.shard_shape(atom_map.shape) == atom_map.shape
        assert feature_spec(
            "ref_pos", atom, shard_atom_features=True
        ) is not None

        strict = replicate_tree(
'''
    _replace_once(path, old, new)


def _patch_atom_test() -> None:
    path = "tests/models/boltz2/test_atom_context_parallel.py"
    _replace_once(
        path,
        "    from foldjax.models._cp_atom import single_to_keys_cp\n",
        "    from foldjax.models._cp_atom import place_atoms, single_to_keys_cp\n",
    )
    old = '''    with context_parallel(4, layout="2d"):
        compiled = jax.jit(
            lambda value: single_to_keys_cp(
                value,
                query_window=32,
                key_window=128,
            )
        )
        out = compiled(x)
        got = jax.device_get(out)
        hlo = compiled.lower(x).compiler_ir(dialect="hlo").as_hlo_text().lower()
'''
    new = '''    with context_parallel(4, layout="2d"):
        x_distributed = place_atoms(x, atom_axis=1)
        assert x_distributed.sharding.shard_shape(x.shape) == (2, 128, 3)
        compiled = jax.jit(
            lambda value: single_to_keys_cp(
                value,
                query_window=32,
                key_window=128,
            )
        )
        out = compiled(x_distributed)
        got = jax.device_get(out)
        hlo = compiled.lower(x_distributed).compiler_ir(
            dialect="hlo"
        ).as_hlo_text().lower()
'''
    _replace_once(path, old, new)


def _patch_api_test() -> None:
    path = "tests/models/boltz2/test_api.py"
    content = _read(path)
    test_name = "test_stop_after_trunk_crops_and_saves_representations"
    if test_name in content:
        raise RuntimeError(f"{path}: follow-up test already exists")
    addition = '''


def test_stop_after_trunk_crops_and_saves_representations(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        api,
        "featurize",
        lambda **kwargs: (_fake_features(affinity=True), "job", tmp_path),
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda path: {"trunk": {}},
    )

    def fake_predict(params, model_feats, key, **kwargs):
        assert kwargs["stop_after_trunk"] is True
        tokens = model_feats["token_pad_mask"].shape[-1]
        return {
            "single": jnp.ones((1, tokens, 4), dtype=jnp.float32),
            "pair": jnp.ones((1, tokens, tokens, 2), dtype=jnp.float32),
        }

    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict",
        fake_predict,
    )
    destination = tmp_path / "representations"
    result = api.predict(
        seq=["AC"],
        weights=tmp_path / "boltz2_conf",
        mols=tmp_path,
        out_dir=tmp_path,
        stop_after="trunk",
        representations=("single", "pair"),
        representations_dir=destination,
        padding=PaddingConfig(tokens=8, atoms=32, msa=1),
        write_fmt=None,
    )

    assert "coords" not in result
    assert "plddt" not in result
    assert result["raw"]["single"].shape == (1, 2, 4)
    assert result["raw"]["pair"].shape == (1, 2, 2, 2)
    assert result["representations"] == destination / "representations.npz"
    assert result["representations"].is_file()
    with np.load(result["representations"]) as archive:
        assert archive["single"].shape == (1, 2, 4)
        assert archive["pair"].shape == (1, 2, 2, 2)
    assert result["padding"]["primary"]["target"] == {
        "tokens": 8,
        "atoms": 32,
        "msa": 1,
    }
'''
    _write(path, content.rstrip() + addition + "\n")


def main() -> None:
    _patch_cp_runtime()
    _patch_atom_runtime()
    _patch_output_cropping()
    _patch_boltz_api()
    _patch_runtime_test()
    _patch_atom_test()
    _patch_api_test()
    print("final Fold-CP follow-up implementation applied")


if __name__ == "__main__":
    main()
