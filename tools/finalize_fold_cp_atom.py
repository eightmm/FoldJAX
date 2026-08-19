from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:100]!r}")
    write(path, content.replace(old, new, 1))


def replace_function(path: str, name: str, source: str) -> None:
    content = read(path)
    match = re.search(rf"^def {re.escape(name)}\(", content, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{path}: function {name!r} not found")
    next_match = re.search(r"^(?:def |class |[A-Z][A-Z0-9_]*\s*=)", content[match.end():], flags=re.MULTILINE)
    end = len(content) if next_match is None else match.end() + next_match.start()
    replacement = textwrap.dedent(source).strip() + "\n\n"
    write(path, content[: match.start()] + replacement + content[end:])


def replace_in_function(path: str, name: str, old: str, new: str) -> None:
    content = read(path)
    match = re.search(rf"^def {re.escape(name)}\(", content, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{path}: function {name!r} not found")
    next_match = re.search(r"^(?:def |class |[A-Z][A-Z0-9_]*\s*=)", content[match.end():], flags=re.MULTILINE)
    end = len(content) if next_match is None else match.end() + next_match.start()
    block = content[match.start():end]
    count = block.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}:{name}: expected one occurrence, found {count}: {old[:100]!r}"
        )
    block = block.replace(old, new, 1)
    write(path, content[: match.start()] + block + content[end:])


# ---------------------------------------------------------------------------
# Common atom runtime
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/_cp_atom.py",
    "def _single_to_keys_local(\n",
    "def single_to_keys_local(\n",
)
replace_once(
    "src/foldjax/models/_cp_atom.py",
    "return _single_to_keys_local(\n",
    "return single_to_keys_local(\n",
)
replace_once(
    "src/foldjax/models/_cp_atom.py",
    "def shard_windows(array: jax.Array, *, window_axis: int = -3) -> jax.Array:\n",
    '''def replicate_atoms(array: jax.Array) -> jax.Array:\n    """Replicate a linear atom result explicitly at a post-CP boundary."""\n\n    mesh = cp_mesh()\n    if mesh is None:\n        return array\n    return jax.lax.with_sharding_constraint(\n        array,\n        NamedSharding(mesh, PartitionSpec()),\n    )\n\n\ndef shard_windows(array: jax.Array, *, window_axis: int = -3) -> jax.Array:\n''',
)

# ---------------------------------------------------------------------------
# Boltz atom utilities and transformer adapter
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/boltz2/models/diffusion/atom.py",
    "import jax.numpy as jnp\n\nfrom foldjax.models._stacking import take_layers\n",
    '''import jax.numpy as jnp\nfrom jax.sharding import PartitionSpec\n\nfrom foldjax.models._cp import cp_mesh, cp_row_shards, shard_single\nfrom foldjax.models._cp_atom import (\n    atom_axis_name,\n    atom_spec,\n    gather_tokens_to_atoms_cp,\n    scatter_atoms_to_tokens_mean_cp,\n    shard_atoms,\n    single_to_keys_local,\n    window_spec,\n)\nfrom foldjax.models._stacking import take_layers\n''',
)
replace_once(
    "src/foldjax/models/boltz2/models/diffusion/atom.py",
    '''def _repeat_index(\n''',
    '''def atom_to_token_index_from_feats(\n    feats: Mapping[str, jnp.ndarray],\n) -> tuple[jnp.ndarray, jnp.ndarray]:\n    """Read compact global token IDs, falling back to the legacy one-hot map."""\n\n    if "atom_to_token_ids_global" in feats:\n        indices = feats["atom_to_token_ids_global"].astype(jnp.int32)\n        valid = feats.get("atom_to_token_valid")\n        if valid is None:\n            valid = indices >= 0\n        return jnp.maximum(indices, 0), valid.astype(bool)\n    return atom_to_token_index(feats["atom_to_token"])\n\n\ndef _repeat_index(\n''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/atom.py",
    "atom_transformer_forward",
    r'''
    def atom_transformer_forward(
        params: Params,
        q: jnp.ndarray,
        c: jnp.ndarray,
        bias: jnp.ndarray,
        to_keys: Callable[[jnp.ndarray], jnp.ndarray],
        mask: jnp.ndarray,
        attn_window_queries: int,
        attn_window_keys: int,
        multiplicity: int = 1,
        eps: float = 1e-5,
        attention_backend: str = "xla",
        atom_context_parallel: bool = False,
    ) -> jnp.ndarray:
        """Run Boltz AtomTransformer with serial or halo-sharded windows."""

        if atom_context_parallel and cp_mesh() is not None:
            return _atom_transformer_forward_cp(
                params,
                q,
                c,
                bias,
                mask,
                attn_window_queries=attn_window_queries,
                attn_window_keys=attn_window_keys,
                multiplicity=multiplicity,
                eps=eps,
                attention_backend=attention_backend,
            )

        w = attn_window_queries
        h_keys = attn_window_keys
        batch, atoms, dim = q.shape
        num_windows = atoms // w

        q = jnp.reshape(q, (batch * num_windows, w, dim))
        c = jnp.reshape(c, (batch * num_windows, w, c.shape[-1]))
        mask = jnp.reshape(mask, (batch * num_windows, w))
        bias = jnp.repeat(bias, multiplicity, axis=0)
        bias = jnp.reshape(
            bias,
            (bias.shape[0] * num_windows, w, h_keys, bias.shape[-1]),
        )

        def to_keys_new(x: jnp.ndarray) -> jnp.ndarray:
            x = jnp.reshape(x, (batch, num_windows * w, -1))
            return jnp.reshape(to_keys(x), (batch * num_windows, h_keys, -1))

        q = diffusion_transformer_forward(
            params["diffusion_transformer"],
            a=q,
            s=c,
            bias=bias,
            mask=mask.astype(jnp.float32),
            to_keys=to_keys_new,
            multiplicity=1,
            eps=eps,
            attention_backend=attention_backend,
        )
        return jnp.reshape(q, (batch, num_windows * w, dim))


    def _atom_transformer_forward_cp(
        params: Params,
        q: jnp.ndarray,
        c: jnp.ndarray,
        bias: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        attn_window_queries: int,
        attn_window_keys: int,
        multiplicity: int,
        eps: float,
        attention_backend: str,
    ) -> jnp.ndarray:
        """Run each device's owned query windows with neighbour halo exchange."""

        mesh = cp_mesh()
        if mesh is None:
            raise RuntimeError("atom CP adapter requires an active mesh")
        if q.ndim != 3 or c.ndim != 3 or mask.ndim != 2 or bias.ndim != 5:
            raise ValueError(
                "atom CP expects q/c [B,A,C], mask [B,A], and bias [B,K,W,H,C]"
            )
        w = attn_window_queries
        h_keys = attn_window_keys
        axis_name = atom_axis_name()
        rows = cp_row_shards()

        def local(params_l, q_l, c_l, bias_l, mask_l):
            batch, local_atoms, dim = q_l.shape
            if local_atoms % w:
                raise ValueError("local atom shard is not query-window aligned")
            local_windows = local_atoms // w
            q_flat = q_l.reshape(batch * local_windows, w, dim)
            c_flat = c_l.reshape(batch * local_windows, w, c_l.shape[-1])
            mask_flat = mask_l.reshape(batch * local_windows, w)
            bias_l = jnp.repeat(bias_l, multiplicity, axis=0)
            if bias_l.shape[0] != batch:
                raise ValueError(
                    "atom bias multiplicity does not match the atom activation batch"
                )
            bias_flat = bias_l.reshape(
                batch * local_windows,
                w,
                h_keys,
                bias_l.shape[-1],
            )

            def to_keys_local(x: jnp.ndarray) -> jnp.ndarray:
                x = x.reshape(batch, local_atoms, -1)
                keys = single_to_keys_local(
                    x,
                    query_window=w,
                    key_window=h_keys,
                    axis_name=axis_name,
                    axis_size=rows,
                )
                return keys.reshape(batch * local_windows, h_keys, -1)

            out = diffusion_transformer_forward(
                params_l["diffusion_transformer"],
                a=q_flat,
                s=c_flat,
                bias=bias_flat,
                mask=mask_flat.astype(jnp.float32),
                to_keys=to_keys_local,
                multiplicity=1,
                eps=eps,
                attention_backend=attention_backend,
            )
            return out.reshape(batch, local_atoms, dim)

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                PartitionSpec(),
                atom_spec(3, atom_axis=1),
                atom_spec(3, atom_axis=1),
                window_spec(5, window_axis=1),
                atom_spec(2, atom_axis=1),
            ),
            out_specs=atom_spec(3, atom_axis=1),
        )(params, q, c, bias, mask)
    ''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/atom.py",
    "atom_attention_encoder_forward",
    r'''
    def atom_attention_encoder_forward(
        params: Params,
        feats: Mapping[str, jnp.ndarray],
        q: jnp.ndarray,
        c: jnp.ndarray,
        atom_enc_bias: jnp.ndarray,
        to_keys: Callable[[jnp.ndarray], jnp.ndarray],
        r: jnp.ndarray | None = None,
        multiplicity: int = 1,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        structure_prediction: bool = True,
        eps: float = 1e-5,
        attention_backend: str = "xla",
        atom_to_token_idx: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        atom_context_parallel: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run AtomAttentionEncoder with dense or CP-row atom ownership."""

        active = atom_context_parallel and cp_mesh() is not None
        atom_mask = feats["atom_pad_mask"].astype(bool)
        if active:
            atom_mask = shard_atoms(atom_mask, atom_axis=1)
            q = shard_atoms(q, atom_axis=1)
            c = shard_atoms(c, atom_axis=1)
            atom_enc_bias = atom_enc_bias
        if structure_prediction:
            if r is None:
                raise ValueError("r is required when structure_prediction=True")
            if active:
                r = shard_atoms(r, atom_axis=1)
            q = jnp.repeat(q, multiplicity, axis=0)
            q = q + _linear(r, params["r_to_q_trans"]["kernel"])
        c = jnp.repeat(c, multiplicity, axis=0)
        atom_mask = jnp.repeat(atom_mask, multiplicity, axis=0)

        q = atom_transformer_forward(
            params["atom_encoder"],
            q=q,
            c=c,
            bias=atom_enc_bias,
            to_keys=to_keys,
            mask=atom_mask.astype(jnp.float32),
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
            atom_context_parallel=active,
        )

        q_to_a = jax.nn.relu(
            _linear(q, params["atom_to_token_trans"]["kernel"])
        ).astype(q.dtype)
        idx = _repeat_index(
            atom_to_token_idx or atom_to_token_index_from_feats(feats),
            multiplicity,
        )
        if active:
            a = scatter_atoms_to_tokens_mean_cp(
                q_to_a,
                idx[0],
                idx[1],
                num_tokens=feats["token_pad_mask"].shape[1],
            ).astype(q.dtype)
            a = shard_single(a, token_axis=1)
        else:
            atom_to_token = jnp.repeat(
                feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
            )
            a = scatter_atoms_to_tokens_mean(
                atom_to_token,
                q_to_a,
                index=idx,
            ).astype(q.dtype)
        return a, q, c
    ''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/atom.py",
    "atom_attention_decoder_forward",
    r'''
    def atom_attention_decoder_forward(
        params: Params,
        a: jnp.ndarray,
        q: jnp.ndarray,
        c: jnp.ndarray,
        atom_dec_bias: jnp.ndarray,
        feats: Mapping[str, jnp.ndarray],
        to_keys: Callable[[jnp.ndarray], jnp.ndarray],
        multiplicity: int = 1,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        eps: float = 1e-5,
        attention_backend: str = "xla",
        atom_to_token_idx: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        atom_context_parallel: bool = False,
    ) -> jnp.ndarray:
        """Run AtomAttentionDecoder with explicit token-to-atom routing."""

        active = atom_context_parallel and cp_mesh() is not None
        idx = _repeat_index(
            atom_to_token_idx or atom_to_token_index_from_feats(feats),
            multiplicity,
        )
        projected = _linear(a.astype(q.dtype), params["a_to_q_trans"]["kernel"])
        if active:
            a = shard_single(a, token_axis=1)
            projected = shard_single(projected, token_axis=1)
            q = shard_atoms(q, atom_axis=1)
            c = shard_atoms(c, atom_axis=1)
            a_to_q = gather_tokens_to_atoms_cp(projected, idx[0], idx[1])
        else:
            atom_to_token = jnp.repeat(
                feats["atom_to_token"].astype(jnp.float32), multiplicity, axis=0
            )
            a_to_q = gather_tokens_to_atoms(
                atom_to_token,
                projected,
                index=idx,
            )
        q = q + a_to_q.astype(q.dtype)
        atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)
        if active:
            atom_mask = shard_atoms(atom_mask, atom_axis=1)

        q = atom_transformer_forward(
            params["atom_decoder"],
            q=q,
            c=c,
            bias=atom_dec_bias,
            to_keys=to_keys,
            mask=atom_mask.astype(jnp.float32),
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
            atom_context_parallel=active,
        )

        update = params["atom_feat_to_atom_pos_update"]
        q = _layer_norm(q, update["norm"]["scale"], update["norm"]["bias"], eps)
        out = _linear(q, update["linear"]["kernel"])
        return shard_atoms(out, atom_axis=1) if active else out
    ''',
)

# ---------------------------------------------------------------------------
# Diffusion conditioning and sparse pair-to-atom lookup
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_conditioning.py",
    "import jax.numpy as jnp\n\nfrom foldjax.models.boltz2.models.diffusion.atom import (\n",
    '''import jax.numpy as jnp\n\nfrom foldjax.models._cp import cp_mesh, shard_pair_rows, shard_single\nfrom foldjax.models._cp_atom import (\n    gather_token_pairs_to_atom_windows_cp,\n    gather_tokens_to_atoms_cp,\n    shard_atoms,\n    shard_windows,\n    single_to_keys_cp,\n)\nfrom foldjax.models.boltz2.models.diffusion.atom import (\n''',
)
replace_once(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_conditioning.py",
    "    atom_to_token_index,\n",
    "    atom_to_token_index,\n    atom_to_token_index_from_feats,\n",
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_conditioning.py",
    "diffusion_conditioning_forward",
    r'''
    def diffusion_conditioning_forward(
        params: Params,
        s_trunk: jnp.ndarray,
        z_trunk: jnp.ndarray,
        relative_position_encoding: jnp.ndarray,
        feats: Mapping[str, jnp.ndarray],
        token_layers: int | None = None,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        eps: float = 1e-5,
        lazy_token_trans_bias: bool = False,
        atom_context_parallel: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Run diffusion conditioning, optionally retaining CP atom ownership."""

        active = atom_context_parallel and cp_mesh() is not None
        z = pairwise_conditioning_forward(
            params["pairwise_conditioner"],
            z_trunk,
            relative_position_encoding,
            eps=eps,
        )
        if active:
            z = shard_pair_rows(z)
        atom_index = atom_to_token_index_from_feats(feats)
        if active:
            atom_index = (
                shard_atoms(atom_index[0], atom_axis=1),
                shard_atoms(atom_index[1], atom_axis=1),
            )
        q, c, p = atom_encoder_forward(
            params["atom_encoder"],
            feats,
            s_trunk,
            z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            eps=eps,
            atom_to_token_idx=atom_index,
            atom_context_parallel=active,
        )
        token_proj = params["token_trans_proj_z"]
        if token_layers is not None:
            token_proj = token_proj[:token_layers]
        atoms = feats["ref_pos"].shape[1]
        w = atoms_per_window_queries
        h_keys = atoms_per_window_keys
        indexing = None if active else get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)

        def to_keys(x: jnp.ndarray) -> jnp.ndarray:
            if active:
                return single_to_keys_cp(x, query_window=w, key_window=h_keys)
            return single_to_keys(x, indexing, w=w, h_keys=h_keys)

        out = {
            "q": q,
            "c": c,
            "atom_to_token_idx": atom_index,
            "to_keys": to_keys,
            "atom_enc_bias": _projection_list_forward(params["atom_enc_proj_z"], p, eps),
            "atom_dec_bias": _projection_list_forward(params["atom_dec_proj_z"], p, eps),
        }
        if lazy_token_trans_bias:
            out["token_trans_bias_params"] = token_proj
            out["token_trans_bias_normed_input"] = _projection_input_norm(z, eps)
        else:
            out["token_trans_bias"] = _projection_list_forward(token_proj, z, eps)
        return out
    ''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_conditioning.py",
    "atom_encoder_forward",
    r'''
    def atom_encoder_forward(
        params: Params,
        feats: Mapping[str, jnp.ndarray],
        s_trunk: jnp.ndarray | None = None,
        z: jnp.ndarray | None = None,
        structure_prediction: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        eps: float = 1e-5,
        atom_to_token_idx: tuple[jnp.ndarray, jnp.ndarray] | None = None,
        atom_context_parallel: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run Boltz AtomEncoder with sparse CP token and pair routing."""

        active = atom_context_parallel and cp_mesh() is not None

        def atom_feature(name: str) -> jnp.ndarray:
            value = feats[name]
            return shard_atoms(value, atom_axis=1) if active else value

        atom_ref_pos = atom_feature("ref_pos")
        batch, atoms, _ = atom_ref_pos.shape
        atom_mask = atom_feature("atom_pad_mask").astype(bool)
        atom_uid = atom_feature("ref_space_uid")
        atom_index = atom_to_token_idx or atom_to_token_index_from_feats(feats)
        if active:
            atom_index = (
                shard_atoms(atom_index[0], atom_axis=1),
                shard_atoms(atom_index[1], atom_axis=1),
            )

        atom_feats = jnp.concatenate(
            (
                atom_ref_pos,
                atom_feature("ref_charge")[..., None],
                atom_feature("ref_element"),
                jnp.reshape(
                    atom_feature("ref_atom_name_chars"),
                    (batch, atoms, 4 * 64),
                ),
            ),
            axis=-1,
        )
        c = _linear(
            atom_feats,
            params["embed_atom_features"]["kernel"],
            params["embed_atom_features"]["bias"],
        )
        if active:
            c = shard_atoms(c, atom_axis=1)

        w = atoms_per_window_queries
        h_keys = atoms_per_window_keys
        num_windows = atoms // w
        indexing = None if active else get_indexing_matrix(k=num_windows, w=w, h_keys=h_keys)

        def to_keys(x: jnp.ndarray) -> jnp.ndarray:
            if active:
                return single_to_keys_cp(x, query_window=w, key_window=h_keys)
            return single_to_keys(x, indexing, w=w, h_keys=h_keys)

        atom_ref_pos_queries = jnp.reshape(
            atom_ref_pos,
            (batch, num_windows, w, 1, 3),
        )
        atom_ref_pos_keys = jnp.reshape(
            to_keys(atom_ref_pos),
            (batch, num_windows, 1, h_keys, 3),
        )
        d = atom_ref_pos_keys - atom_ref_pos_queries
        d_norm = 1.0 / (1.0 + jnp.sum(d * d, axis=-1, keepdims=True))

        atom_mask_queries = jnp.reshape(atom_mask, (batch, num_windows, w, 1))
        atom_mask_keys = jnp.reshape(
            to_keys(atom_mask[..., None].astype(jnp.float32)),
            (batch, num_windows, 1, h_keys),
        ).astype(bool)
        atom_uid_queries = jnp.reshape(atom_uid, (batch, num_windows, w, 1))
        atom_uid_keys = jnp.reshape(
            to_keys(atom_uid[..., None]),
            (batch, num_windows, 1, h_keys),
        )
        valid = (
            atom_mask_queries
            & atom_mask_keys
            & (atom_uid_queries == atom_uid_keys.astype(atom_uid.dtype))
        ).astype(atom_ref_pos.dtype)[..., None]

        p = _linear(d, params["embed_atompair_ref_pos"]["kernel"]) * valid
        p = p + _linear(d_norm, params["embed_atompair_ref_dist"]["kernel"]) * valid
        p = p + _linear(valid, params["embed_atompair_mask"]["kernel"]) * valid
        if active:
            p = shard_windows(p, window_axis=1)

        q = c
        if structure_prediction:
            if s_trunk is None or z is None:
                raise ValueError("s_trunk and z are required when structure_prediction=True")
            s_to_c = params["s_to_c_trans"]
            s_to_c_out = _linear(
                _layer_norm(
                    s_trunk,
                    s_to_c["norm"]["scale"],
                    s_to_c["norm"]["bias"],
                    eps,
                ),
                s_to_c["linear"]["kernel"],
            )
            if active:
                s_to_c_out = shard_single(s_to_c_out, token_axis=1)
                c = c + gather_tokens_to_atoms_cp(
                    s_to_c_out,
                    atom_index[0],
                    atom_index[1],
                ).astype(c.dtype)
            else:
                c = c + gather_tokens_to_atoms(
                    feats["atom_to_token"].astype(jnp.float32),
                    s_to_c_out,
                ).astype(c.dtype)

            z_to_p = params["z_to_p_trans"]
            z_to_p_out = _linear(
                _layer_norm(
                    z,
                    z_to_p["norm"]["scale"],
                    z_to_p["norm"]["bias"],
                    eps,
                ),
                z_to_p["linear"]["kernel"],
            )
            if active:
                z_to_p_out = shard_pair_rows(z_to_p_out)
                query_indices = atom_index[0].reshape(batch, num_windows, w)
                query_valid = atom_index[1].reshape(batch, num_windows, w)
                key_indices = jnp.squeeze(to_keys(atom_index[0][..., None]), axis=-1)
                key_valid = jnp.squeeze(
                    to_keys(atom_index[1][..., None].astype(jnp.float32)),
                    axis=-1,
                ).astype(bool)
                p = p + gather_token_pairs_to_atom_windows_cp(
                    z_to_p_out,
                    query_indices,
                    query_valid,
                    key_indices,
                    key_valid,
                ).astype(p.dtype)
            else:
                atom_to_token_queries = jnp.reshape(
                    feats["atom_to_token"].astype(jnp.float32),
                    (batch, num_windows, w, feats["atom_to_token"].shape[-1]),
                )
                atom_to_token_keys = to_keys(feats["atom_to_token"].astype(jnp.float32))
                p = p + gather_token_pairs_to_atom_windows(
                    z_to_p_out,
                    atom_to_token_queries,
                    atom_to_token_keys,
                ).astype(p.dtype)

        p = p + _linear(
            jax.nn.relu(jnp.reshape(c, (batch, num_windows, w, 1, c.shape[-1]))),
            params["c_to_p_trans_q"]["kernel"],
        )
        p = p + _linear(
            jax.nn.relu(
                jnp.reshape(
                    to_keys(c),
                    (batch, num_windows, 1, h_keys, c.shape[-1]),
                )
            ),
            params["c_to_p_trans_k"]["kernel"],
        )
        p = p + _p_mlp_forward(params["p_mlp"], p)
        if active:
            return (
                shard_atoms(q, atom_axis=1),
                shard_atoms(c, atom_axis=1),
                shard_windows(p, window_axis=1),
            )
        return q, c, p
    ''',
)

# ---------------------------------------------------------------------------
# Diffusion graph signatures and CP token attention
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/boltz2/models/diffusion/diffusion.py",
    "import jax.numpy as jnp\n\nfrom foldjax.models.boltz2.models.diffusion.atom import (\n",
    '''import jax.numpy as jnp\n\nfrom foldjax.models._cp import cp_mesh\nfrom foldjax.models._cp_atom import single_to_keys_cp\nfrom foldjax.models.boltz2.models.diffusion.atom import (\n''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion.py",
    "diffusion_score_model_forward",
    r'''
    def diffusion_score_model_forward(
        params: Params,
        s_inputs: jnp.ndarray,
        s_trunk: jnp.ndarray,
        r_noisy: jnp.ndarray,
        times: jnp.ndarray,
        feats: Mapping[str, jnp.ndarray],
        diffusion_conditioning: Mapping[str, object],
        multiplicity: int = 1,
        eps: float = 1e-5,
        use_scan: bool = True,
        attention_backend: str = "xla",
        token_attention_chunk: int | None = None,
        token_layers: int | None = None,
        atom_context_parallel: bool = False,
    ) -> jnp.ndarray:
        """Run Boltz DiffusionModule.forward using precomputed conditioning."""

        compute_dtype = r_noisy.dtype
        s, _ = single_conditioning_forward(
            params["single_conditioner"],
            times,
            jnp.repeat(s_trunk, multiplicity, axis=0),
            jnp.repeat(s_inputs, multiplicity, axis=0),
            eps=eps,
        )

        atom_to_token_idx = diffusion_conditioning.get("atom_to_token_idx")
        a, q_skip, c_skip = atom_attention_encoder_forward(
            params["atom_attention_encoder"],
            feats=feats,
            q=diffusion_conditioning["q"].astype(compute_dtype),
            c=diffusion_conditioning["c"].astype(compute_dtype),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].astype(compute_dtype),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
            atom_to_token_idx=atom_to_token_idx,
            atom_context_parallel=atom_context_parallel,
        )

        s_to_a = params["s_to_a_linear"]
        a = a + _linear(
            _layer_norm(
                s,
                s_to_a["norm"]["scale"],
                s_to_a["norm"]["bias"],
                eps,
            ),
            s_to_a["linear"]["kernel"],
        )

        mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)
        token_bias = diffusion_conditioning.get("token_trans_bias")
        a = diffusion_transformer_forward(
            params["token_transformer"],
            a=a,
            s=s,
            bias=None if token_bias is None else token_bias.astype(compute_dtype),
            mask=mask.astype(jnp.float32),
            multiplicity=multiplicity,
            eps=eps,
            use_scan=use_scan,
            attention_backend=attention_backend,
            chunk_size=token_attention_chunk,
            layer_limit=token_layers,
            bias_params=diffusion_conditioning.get("token_trans_bias_params"),
            bias_input=diffusion_conditioning.get("token_trans_bias_input"),
            bias_normed_input=diffusion_conditioning.get("token_trans_bias_normed_input"),
        )
        a_norm = params["a_norm"]
        a = _layer_norm(a, a_norm["scale"], a_norm["bias"], eps)

        return atom_attention_decoder_forward(
            params["atom_attention_decoder"],
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].astype(compute_dtype),
            feats=feats,
            to_keys=diffusion_conditioning["to_keys"],
            multiplicity=multiplicity,
            eps=eps,
            attention_backend=attention_backend,
            atom_to_token_idx=atom_to_token_idx,
            atom_context_parallel=atom_context_parallel,
        )
    ''',
)
replace_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion.py",
    "conditioned_diffusion_score_forward",
    r'''
    def conditioned_diffusion_score_forward(
        params: Params,
        s_inputs: jnp.ndarray,
        s_trunk: jnp.ndarray,
        z_trunk: jnp.ndarray,
        relative_position_encoding: jnp.ndarray,
        r_noisy: jnp.ndarray,
        times: jnp.ndarray,
        feats: Mapping[str, jnp.ndarray],
        token_layers: int | None = None,
        multiplicity: int = 1,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        eps: float = 1e-5,
        use_scan: bool = True,
        attention_backend: str = "xla",
        token_attention_chunk: int | None = None,
        lazy_token_trans_bias: bool = False,
        atom_context_parallel: bool = False,
    ) -> jnp.ndarray:
        """Run diffusion conditioning and score model as one JAX graph."""

        active = atom_context_parallel and cp_mesh() is not None
        conditioning = diffusion_conditioning_forward(
            params["diffusion_conditioning"],
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
            token_layers=token_layers,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            eps=eps,
            lazy_token_trans_bias=lazy_token_trans_bias,
            atom_context_parallel=active,
        )
        atoms = feats["ref_pos"].shape[1]
        num_windows = atoms // atoms_per_window_queries
        indexing = None
        if not active:
            indexing = get_indexing_matrix(
                k=num_windows,
                w=atoms_per_window_queries,
                h_keys=atoms_per_window_keys,
            )

        def to_keys(x: jnp.ndarray) -> jnp.ndarray:
            if active:
                return single_to_keys_cp(
                    x,
                    query_window=atoms_per_window_queries,
                    key_window=atoms_per_window_keys,
                )
            return single_to_keys(
                x,
                indexing,
                w=atoms_per_window_queries,
                h_keys=atoms_per_window_keys,
            )

        conditioning["to_keys"] = to_keys
        return diffusion_score_model_forward(
            params["score_model"],
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            times=times,
            feats=feats,
            diffusion_conditioning=conditioning,
            multiplicity=multiplicity,
            eps=eps,
            use_scan=use_scan,
            attention_backend=attention_backend,
            token_attention_chunk=token_attention_chunk,
            token_layers=token_layers,
            atom_context_parallel=active,
        )
    ''',
)

replace_once(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_transformer.py",
    "import jax.numpy as jnp\n\nfrom foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm\n",
    '''import jax.numpy as jnp\n\nfrom foldjax.models._cp import cp_layout, shard_pair_rows, shard_single\nfrom foldjax.models._cp_atom import pair_bias_attention_2d\nfrom foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_transformer.py",
    "diffusion_transformer_layer_apply",
    '''        chunk_size=chunk_size,\n    )\n''',
    '''        chunk_size=chunk_size,\n        distributed_pair_bias=to_keys is None,\n    )\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_transformer.py",
    "_attention_pair_bias_no_proj_z_forward",
    '''    chunk_size: int | None = None,\n) -> jnp.ndarray:\n''',
    '''    chunk_size: int | None = None,\n    distributed_pair_bias: bool = True,\n) -> jnp.ndarray:\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/diffusion/diffusion_transformer.py",
    "_attention_pair_bias_no_proj_z_forward",
    '''    if attention_backend in ("tokamax", "flash"):\n''',
    '''    if distributed_pair_bias and cp_layout() == "2d":\n        q = shard_single(q, token_axis=1)\n        k = shard_single(k, token_axis=1)\n        v = shard_single(v, token_axis=1)\n        mask = shard_single(mask, token_axis=1)\n        bias = shard_pair_rows(bias, row_axis=-2, col_axis=-1)\n        out = pair_bias_attention_2d(\n            q,\n            k,\n            v,\n            bias,\n            mask,\n            scale=float(head_dim) ** -0.5,\n            inf=inf,\n        )\n    elif attention_backend in ("tokamax", "flash"):\n''',
)

# ---------------------------------------------------------------------------
# Sampler propagation
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "from foldjax.models._cp import cp_mesh as _cp_mesh\n",
    "from foldjax.models._cp import cp_mesh as _cp_mesh\nfrom foldjax.models._cp_atom import shard_atoms\n",
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_graph_score_forward",
    '''    lazy_token_trans_bias: bool = True,\n) -> jnp.ndarray:\n''',
    '''    lazy_token_trans_bias: bool = True,\n    atom_context_parallel: bool = False,\n) -> jnp.ndarray:\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_graph_score_forward",
    '''        lazy_token_trans_bias=lazy_token_trans_bias,\n    )\n''',
    '''        lazy_token_trans_bias=lazy_token_trans_bias,\n        atom_context_parallel=atom_context_parallel,\n    )\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_sample_forward",
    '''    shard_tokens: bool = True,\n    lazy_token_trans_bias: bool = True,\n''',
    '''    shard_tokens: bool = True,\n    lazy_token_trans_bias: bool = True,\n    atom_context_parallel: bool = False,\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_sample_forward",
    '''        lazy_token_trans_bias=lazy_token_trans_bias,\n    )\n    sigmas = _sample_schedule(\n''',
    '''        lazy_token_trans_bias=lazy_token_trans_bias,\n        atom_context_parallel=atom_context_parallel,\n    )\n    sigmas = _sample_schedule(\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_sample_forward",
    '''    atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)\n    shape = (*atom_mask.shape, 3)\n''',
    '''    atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)\n    if atom_context_parallel:\n        atom_mask = shard_atoms(atom_mask, atom_axis=1)\n    shape = (*atom_mask.shape, 3)\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "boltz2_sample_forward",
    '''    gammas = jnp.where(sigmas > gamma_min, gamma_0, 0.0)\n''',
    '''    if atom_context_parallel:\n        atom_coords = shard_atoms(atom_coords, atom_axis=1)\n    gammas = jnp.where(sigmas > gamma_min, gamma_0, 0.0)\n''',
)
# Two calls inside scan/eager.
for _ in range(2):
    replace_in_function(
        "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
        "boltz2_sample_forward",
        '''                token_layers=token_layers,\n            )\n''',
        '''                token_layers=token_layers,\n                atom_context_parallel=atom_context_parallel,\n            )\n''',
    )
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "_preconditioned_score_forward",
    '''    token_layers: int | None = None,\n) -> jnp.ndarray:\n''',
    '''    token_layers: int | None = None,\n    atom_context_parallel: bool = False,\n) -> jnp.ndarray:\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/trunk_blocks/trunk.py",
    "_preconditioned_score_forward",
    '''        token_layers=token_layers,\n    )\n''',
    '''        token_layers=token_layers,\n        atom_context_parallel=atom_context_parallel,\n    )\n''',
)

# ---------------------------------------------------------------------------
# Prediction/confidence boundary
# ---------------------------------------------------------------------------
replace_in_function(
    "src/foldjax/models/boltz2/models/predict.py",
    "boltz2_predict",
    '''            recompute_nonpolymer_frames=recompute_nonpolymer_frames,\n        )\n''',
    '''            recompute_nonpolymer_frames=recompute_nonpolymer_frames,\n            atom_context_parallel=bool(\n                sample_kwargs.get("atom_context_parallel", False)\n            ),\n        )\n''',
)
replace_once(
    "src/foldjax/models/boltz2/models/heads/confidence.py",
    "from foldjax.models._cp import shard_pair_rows\n",
    "from foldjax.models._cp import cp_mesh, shard_pair_rows\nfrom foldjax.models._cp_atom import replicate_atoms\n",
)
replace_in_function(
    "src/foldjax/models/boltz2/models/heads/confidence.py",
    "confidence_module_forward",
    '''    recompute_nonpolymer_frames: bool = True,\n) -> dict[str, Any]:\n''',
    '''    recompute_nonpolymer_frames: bool = True,\n    atom_context_parallel: bool = False,\n) -> dict[str, Any]:\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/models/heads/confidence.py",
    "confidence_module_forward",
    '''    s_inputs = _layer_norm(\n''',
    '''    # Confidence frame construction currently consumes a global linear\n    # atom stream. Make that O(A) gather explicit at the stage boundary; the\n    # quadratic pair state remains context-parallel throughout the head.\n    if atom_context_parallel and cp_mesh() is not None:\n        x_pred = replicate_atoms(x_pred)\n\n    s_inputs = _layer_norm(\n''',
)

# ---------------------------------------------------------------------------
# Automatic CP shape alignment and native API option
# ---------------------------------------------------------------------------
replace_once(
    "src/foldjax/models/boltz2/data/bucket.py",
    "def resolve_bucket_shape(feats: Mapping[str, object]) -> tuple[int, int, int]:\n",
    r'''def align_padding_plan_for_context_parallel(
    feats: Mapping[str, object],
    plan: PaddingPlan | None,
    *,
    cp_rows: int,
    cp_cols: int,
    query_window: int = 32,
    key_window: int = 128,
) -> PaddingPlan:
    """Align token/atom axes for pair tiles and local atom halos."""

    if cp_rows < 1 or cp_cols < 1:
        raise ValueError("CP grid dimensions must be positive")
    actual, storage = _feature_sizes(feats)
    if plan is None:
        target = dict(storage)
        base_actual = actual
        base_storage = storage
    else:
        target = dict(plan.target)
        base_actual = dict(plan.actual)
        base_storage = dict(plan.storage or plan.actual)

    token_alignment = int(np.lcm(cp_rows, cp_cols))
    target["tokens"] = (
        (target["tokens"] + token_alignment - 1) // token_alignment
    ) * token_alignment

    half_window = query_window // 2
    if query_window <= 0 or query_window % 2:
        raise ValueError("query_window must be positive and even")
    if key_window <= 0 or key_window % half_window:
        raise ValueError("key_window must divide into query half-windows")
    half_windows_per_key = key_window // half_window
    if half_windows_per_key % 2:
        raise ValueError("key_window must contain an even number of half-windows")
    halo_radius = half_windows_per_key // 2 - 1
    min_local_windows = max(1, (halo_radius + 1) // 2)
    atom_alignment = query_window * cp_rows
    target["atoms"] = (
        (target["atoms"] + atom_alignment - 1) // atom_alignment
    ) * atom_alignment
    target["atoms"] = max(
        target["atoms"],
        query_window * cp_rows * min_local_windows,
    )
    return PaddingPlan(
        actual=base_actual,
        storage=base_storage,
        target=target,
    )


def resolve_bucket_shape(feats: Mapping[str, object]) -> tuple[int, int, int]:
''',
)
replace_in_function(
    "src/foldjax/models/boltz2/api.py",
    "predict",
    '''    cp_layout: str = "auto",\n''',
    '''    cp_layout: str = "auto",\n    #: Distribute Boltz-2 atom query windows over CP rows. The required atom\n    #: and token alignment is padded automatically and cropped from outputs.\n    cp_atom_windows: bool = True,\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/api.py",
    "predict",
    '''    if cp_devices < 1:\n        raise ValueError("cp_devices must be positive")\n    resolved_cp_layout = _resolve_cp_layout(cp_layout, cp_devices)\n''',
    '''    if cp_devices < 1:\n        raise ValueError("cp_devices must be positive")\n    if not isinstance(cp_atom_windows, bool):\n        raise ValueError("cp_atom_windows must be a boolean")\n    resolved_cp_layout = _resolve_cp_layout(cp_layout, cp_devices)\n    cp_atom_active = cp_devices > 1 and cp_atom_windows and stop_after != "trunk"\n    cp_rows = (\n        int(math.isqrt(cp_devices))\n        if resolved_cp_layout == "2d"\n        else cp_devices\n    )\n    cp_cols = int(math.isqrt(cp_devices)) if resolved_cp_layout == "2d" else 1\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/api.py",
    "predict",
    '''    original_tokens = int(feats_np["token_pad_mask"].shape[-1])\n    original_atoms = int(feats_np["atom_pad_mask"].shape[-1])\n    padding_plan = None\n    if padding is not None or bucket:\n        from foldjax.models.boltz2.data.bucket import (\n            pad_feats,\n            resolve_legacy_padding_plan,\n            resolve_padding_plan,\n        )\n\n        padding_plan = (\n            resolve_padding_plan(feats_np, padding)\n            if padding is not None\n            else resolve_legacy_padding_plan(feats_np)\n        )\n        feats_np, _ = pad_feats(\n            feats_np,\n            padding_plan.target["tokens"],\n            padding_plan.target["atoms"],\n            target_msa=padding_plan.target["msa"],\n        )\n\n    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}\n''',
    '''    original_tokens = int(feats_np["token_pad_mask"].shape[-1])\n    original_atoms = int(feats_np["atom_pad_mask"].shape[-1])\n    padding_plan = None\n    if padding is not None or bucket:\n        from foldjax.models.boltz2.data.bucket import (\n            resolve_legacy_padding_plan,\n            resolve_padding_plan,\n        )\n\n        padding_plan = (\n            resolve_padding_plan(feats_np, padding)\n            if padding is not None\n            else resolve_legacy_padding_plan(feats_np)\n        )\n    if cp_atom_active:\n        from foldjax.models.boltz2.data.bucket import (\n            align_padding_plan_for_context_parallel,\n        )\n\n        padding_plan = align_padding_plan_for_context_parallel(\n            feats_np,\n            padding_plan,\n            cp_rows=cp_rows,\n            cp_cols=cp_cols,\n        )\n    if padding_plan is not None:\n        from foldjax.models.boltz2.data.bucket import pad_feats\n\n        feats_np, _ = pad_feats(\n            feats_np,\n            padding_plan.target["tokens"],\n            padding_plan.target["atoms"],\n            target_msa=padding_plan.target["msa"],\n        )\n\n    if cp_atom_active:\n        atom_to_token = np.asarray(feats_np["atom_to_token"])\n        atom_valid = np.any(atom_to_token > 0, axis=-1)\n        atom_ids = np.argmax(atom_to_token, axis=-1).astype(np.int32)\n        feats_np["atom_to_token_ids_global"] = np.where(\n            atom_valid, atom_ids, -1\n        ).astype(np.int32)\n        feats_np["atom_to_token_valid"] = atom_valid\n\n    feats = {k: jnp.asarray(v) for k, v in feats_np.items()}\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/api.py",
    "predict",
    '''        if padding is not None and padding_plan is not None\n        else None\n''',
    '''        if (\n            padding_plan is not None\n            and padding_plan.target["atoms"] > padding_plan.storage["atoms"]\n        )\n        else None\n''',
)
replace_in_function(
    "src/foldjax/models/boltz2/api.py",
    "predict",
    '''        "multiplicity": diffusion_samples,\n''',
    '''        "multiplicity": diffusion_samples,\n        "atom_context_parallel": cp_atom_active,\n''',
)

replace_once(
    "src/foldjax/backends/boltz2.py",
    '''            "cp_devices",\n            "cp_layout",\n''',
    '''            "cp_atom_windows",\n            "cp_devices",\n            "cp_layout",\n''',
)
replace_once(
    "src/foldjax/backends/boltz2.py",
    '''        "cp_devices",\n        "cp_layout",\n''',
    '''        "cp_atom_windows",\n        "cp_devices",\n        "cp_layout",\n''',
)
replace_in_function(
    "src/foldjax/backends/boltz2.py",
    "validate_native_options",
    '''            "affinity_mw_correction",\n            "bucket",\n''',
    '''            "affinity_mw_correction",\n            "bucket",\n            "cp_atom_windows",\n''',
)

# ---------------------------------------------------------------------------
# Documentation and proof gates
# ---------------------------------------------------------------------------
write(
    "tests/models/boltz2/test_atom_context_parallel.py",
    textwrap.dedent(
        r'''
        """CPU-mesh proof gates for Fold-CP atom-window communication."""

        from __future__ import annotations

        import subprocess
        import sys
        import textwrap

        import pytest


        def _run(source: str, *, devices: int) -> str:
            env = {
                "JAX_PLATFORMS": "cpu",
                "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
                "PATH": "/usr/bin",
            }
            completed = subprocess.run(
                [sys.executable, "-c", source],
                capture_output=True,
                text=True,
                env=env,
                timeout=240,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            return completed.stdout


        _HALO_PROBE = textwrap.dedent(
            r"""
            import jax
            import jax.numpy as jnp
            import numpy as np

            from foldjax.models._cp import context_parallel
            from foldjax.models._cp_atom import single_to_keys_cp

            assert jax.device_count() == 4
            rng = np.random.default_rng(20260820)
            x = jnp.asarray(rng.normal(size=(2, 256, 3)), dtype=jnp.float32)
            reference = jax.device_get(single_to_keys_cp(x, query_window=32, key_window=128))
            with context_parallel(4, layout="2d"):
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
            np.testing.assert_array_equal(reference, got)
            assert "collective-permute" in hlo or "collective_permute" in hlo, hlo
            assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
            assert out.sharding.shard_shape(out.shape) != out.shape
            print("ATOM_HALO_OK")
            """
        )


        _ROUTING_PROBE = textwrap.dedent(
            r"""
            import jax
            import jax.numpy as jnp
            import numpy as np

            from foldjax.models._cp import context_parallel
            from foldjax.models._cp_atom import (
                gather_token_pairs_to_atom_windows_cp,
                gather_tokens_to_atoms_cp,
                scatter_atoms_to_tokens_mean_cp,
            )

            assert jax.device_count() == 4
            rng = np.random.default_rng(7)
            token_values = jnp.asarray(rng.normal(size=(1, 8, 4)), dtype=jnp.float32)
            atom_indices = jnp.asarray([np.arange(32) % 8], dtype=jnp.int32)
            atom_valid = jnp.ones((1, 32), dtype=bool)
            atom_values = jnp.asarray(rng.normal(size=(1, 32, 4)), dtype=jnp.float32)

            gather_ref = gather_tokens_to_atoms_cp(token_values, atom_indices, atom_valid)
            scatter_ref = scatter_atoms_to_tokens_mean_cp(
                atom_values,
                atom_indices,
                atom_valid,
                num_tokens=8,
            )

            pair = jnp.asarray(rng.normal(size=(1, 8, 8, 3)), dtype=jnp.float32)
            q_idx = atom_indices.reshape(1, 8, 4)
            q_valid = atom_valid.reshape(1, 8, 4)
            k_idx = jnp.stack(
                [jnp.roll(q_idx, shift) for shift in range(4)], axis=-1
            )[:, :, :, 0]
            # Build [B,K,H] keys independent of W for a non-degenerate lookup.
            k_idx = jnp.asarray([[(i + j) % 8 for j in range(8)] for i in range(8)])
            k_idx = k_idx[None]
            k_valid = jnp.ones_like(k_idx, dtype=bool)
            pair_ref = gather_token_pairs_to_atom_windows_cp(
                pair,
                q_idx,
                q_valid,
                k_idx,
                k_valid,
            )

            with context_parallel(4, layout="2d"):
                gather_fn = jax.jit(gather_tokens_to_atoms_cp)
                scatter_fn = jax.jit(
                    lambda values, indices, valid: scatter_atoms_to_tokens_mean_cp(
                        values,
                        indices,
                        valid,
                        num_tokens=8,
                    )
                )
                pair_fn = jax.jit(gather_token_pairs_to_atom_windows_cp)
                gather_out = gather_fn(token_values, atom_indices, atom_valid)
                scatter_out = scatter_fn(atom_values, atom_indices, atom_valid)
                pair_out = pair_fn(pair, q_idx, q_valid, k_idx, k_valid)
                gather_hlo = gather_fn.lower(
                    token_values, atom_indices, atom_valid
                ).compiler_ir(dialect="hlo").as_hlo_text().lower()
                scatter_hlo = scatter_fn.lower(
                    atom_values, atom_indices, atom_valid
                ).compiler_ir(dialect="hlo").as_hlo_text().lower()
                pair_hlo = pair_fn.lower(
                    pair, q_idx, q_valid, k_idx, k_valid
                ).compiler_ir(dialect="hlo").as_hlo_text().lower()

            np.testing.assert_allclose(gather_ref, jax.device_get(gather_out), atol=0, rtol=0)
            np.testing.assert_allclose(scatter_ref, jax.device_get(scatter_out), atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(pair_ref, jax.device_get(pair_out), atol=0, rtol=0)
            assert "collective-permute" in gather_hlo or "collective_permute" in gather_hlo
            assert "reduce-scatter" in scatter_hlo or "reduce_scatter" in scatter_hlo
            assert "all-gather" not in pair_hlo and "all_gather" not in pair_hlo
            print("ATOM_ROUTING_OK")
            """
        )


        _PAIR_BIAS_PROBE = textwrap.dedent(
            r"""
            import jax
            import jax.numpy as jnp
            import numpy as np

            from foldjax.models._cp import context_parallel
            from foldjax.models._cp_atom import pair_bias_attention_2d

            assert jax.device_count() == 4
            rng = np.random.default_rng(11)
            q = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
            k = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
            v = jnp.asarray(rng.normal(size=(1, 8, 2, 3)), dtype=jnp.float32)
            bias = jnp.asarray(rng.normal(size=(1, 2, 8, 8)), dtype=jnp.float32)
            mask = jnp.asarray([[1, 1, 1, 1, 1, 0, 1, 1]], dtype=jnp.float32)
            scale = 3 ** -0.5

            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale + bias
            logits = logits + (1 - mask[:, None, None]) * -1e6
            probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
            reference = jnp.einsum("bhqk,bkhd->bqhd", probs, v)

            with context_parallel(4, layout="2d"):
                compiled = jax.jit(
                    lambda qv, kv, vv, bv, mv: pair_bias_attention_2d(
                        qv, kv, vv, bv, mv, scale=scale
                    )
                )
                out = compiled(q, k, v, bias, mask)
                hlo = compiled.lower(q, k, v, bias, mask).compiler_ir(
                    dialect="hlo"
                ).as_hlo_text().lower()
            np.testing.assert_allclose(reference, jax.device_get(out), atol=3e-5, rtol=3e-5)
            assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
            assert "all-reduce" in hlo or "all_reduce" in hlo, hlo
            print("PAIR_BIAS_CP_OK")
            """
        )


        @pytest.mark.parametrize("source", [_HALO_PROBE, _ROUTING_PROBE, _PAIR_BIAS_PROBE])
        def test_atom_context_parallel_primitives(source: str) -> None:
            assert "_OK" in _run(source, devices=4)
        '''
    ).lstrip(),
)

write(
    "tests/models/boltz2/test_atom_cp_padding.py",
    textwrap.dedent(
        r'''
        from __future__ import annotations

        import numpy as np

        from foldjax.models.boltz2.data.bucket import (
            align_padding_plan_for_context_parallel,
        )


        def test_cp_padding_aligns_pair_axes_and_supplies_a_complete_atom_halo() -> None:
            feats = {
                "token_pad_mask": np.ones((1, 7), dtype=np.float32),
                "atom_pad_mask": np.ones((1, 33), dtype=np.float32),
                "msa": np.ones((1, 1, 7), dtype=np.int32),
            }
            plan = align_padding_plan_for_context_parallel(
                feats,
                None,
                cp_rows=2,
                cp_cols=2,
                query_window=32,
                key_window=128,
            )
            assert plan.target["tokens"] == 8
            # Two query windows per CP row are needed for a 3-half-window halo.
            assert plan.target["atoms"] == 128
            assert plan.actual == {"tokens": 7, "atoms": 33, "msa": 1}
        '''
    ).lstrip(),
)

replace_once(
    "docs/context_parallel.md",
    '''| Boltz-2 | yes | yes | Cannon multiplication and ring triangle attention |\n''',
    '''| Boltz-2 | yes | yes | Cannon/ring pair core plus CP-row atom windows and halo exchange |\n''',
)
replace_once(
    "docs/context_parallel.md",
    '''The active CP runtime is task-local, so concurrent requests cannot overwrite\none another's mesh. Model parameters and unrecognised inputs remain\nreplicated. A small, explicit registry places known pair features directly on\nthe pair mesh when both axes divide evenly. Uneven pair features and every\natom/single-stream feature stay replicated at entry.\n\nThat last point is deliberate: FoldJAX currently distributes the quadratic\npair trunk, not the complete atom-window diffusion graph. No capability flag\nclaims otherwise. Atom-window tensors remain a linear-memory replicated stage\nuntil a model-specific halo-exchange implementation is validated.\n''',
    '''The active CP runtime is task-local, so concurrent requests cannot overwrite\none another's mesh. Model parameters and unrecognised inputs remain\nreplicated. A small, explicit registry places known pair features directly on\nthe pair mesh when both axes divide evenly.\n\nBoltz-2 additionally shards atom/query windows over CP rows. Fixed-width\nhalf-window halos use `collective-permute`; token-to-atom gathers rotate linear\nsource shards; atom-to-token means use reduce-scatter; and token-pair values are\nlooked up from rotating 2-D tiles. Inputs are padded to a complete query-window\npartition and halo width, then cropped back to the biological prefix. The\nconfidence frame stage explicitly replicates only its linear atom coordinate\nstream; the quadratic pair state remains distributed. Other models currently\nretain pair-level CP unless their atom graph has a model-specific adapter.\n''',
)
replace_once(
    "docs/context_parallel.md",
    '''  tests/models/boltz2/test_context_parallel.py \\\n''',
    '''  tests/models/boltz2/test_context_parallel.py \\\n  tests/models/boltz2/test_atom_context_parallel.py \\\n  tests/models/boltz2/test_atom_cp_padding.py \\\n''',
)
replace_once(
    "docs/context_parallel.md",
    '''The absence of atom-window CP should be included when interpreting end-to-end\nmemory scaling: the quadratic trunk scales with CP, while the later linear\natom stage does not yet scale across devices.\n''',
    '''For Boltz-2, report pair and atom stages separately: the pair trunk scales over\nboth mesh axes, while atom windows scale over CP rows and are replicated over\nCP columns. The confidence frame calculation performs one explicit O(A) gather.\nFor the other model adapters, the atom stage remains replicated until a\nmodel-specific window contract is validated.\n''',
)

print("Fold-CP atom-window implementation applied")
