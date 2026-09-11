"""The opt-in bfloat16 confidence re-embedding, at AlphaFold 3's boundary.

Every assertion here reads a realised array dtype rather than a config
spelling. A test that asserts `settings.confidence_dtype == "bfloat16"` passes
when the value never reaches an array, which is the failure this option is
most likely to have, and `test_the_narrowed_weights_reach_a_matmul` is the
tripwire that says the narrowed path is not dead code.

Everything runs on CPU. `_autocast_linear` takes its `platform_dependent`
fallback there rather than the CUDA bitwise-native path, but both narrow the
same two operands and emit bfloat16, which is what these tests read.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data.features import build_features, pad_features
from foldjax.models.esmfold2.models import embedders, heads, trunk
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.models.model import CONFIDENCE_DTYPES
from foldjax.models.esmfold2.models.segments import MAX_ATOMS_PER_TOKEN
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features

D_IN, D_PAIR, D_PROD, D_SINGLE = 7, 8, 8, 6
N_DIST_BINS, N_PAE_BINS, N_PLDDT_BINS = 4, 5, 4
TOKENS, ATOMS, TRUNK_LAYERS = 4, 4, 1

#: Every parameter the head reaches that is not inside its own trunk.
REEMBEDDING_PROJECTIONS = (
    "s_to_z",
    "s_to_z_transpose",
    "s_to_z_prod_in1",
    "s_to_z_prod_in2",
    "s_to_z_prod_out",
)
OUTPUT_PROJECTIONS = ("pae_head", "pde_head", "row_attention_pooling.out_proj")


def _array(rng: np.random.Generator, *shape: int) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32))


def _params(seed: int = 0) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, jnp.ndarray] = {
        "s_inputs_norm.weight": _array(rng, D_IN),
        "s_inputs_norm.bias": _array(rng, D_IN),
        "z_norm.weight": _array(rng, D_PAIR),
        "z_norm.bias": _array(rng, D_PAIR),
        "s_to_z.weight": _array(rng, D_PAIR, D_IN),
        "s_to_z_transpose.weight": _array(rng, D_PAIR, D_IN),
        "s_to_z_prod_in1.weight": _array(rng, D_PROD, D_IN),
        "s_to_z_prod_in2.weight": _array(rng, D_PROD, D_IN),
        "s_to_z_prod_out.weight": _array(rng, D_PAIR, D_PROD),
        # Ascending, so a rounded distance that crosses a boundary gathers a
        # visibly different row rather than a neighbouring value.
        "boundaries": jnp.asarray([0.5, 1.5, 2.5], jnp.float32),
        "dist_bin_pairwise_embed.weight": _array(rng, N_DIST_BINS, D_PAIR),
        "row_attention_pooling.attn_proj.weight": _array(rng, 1, D_PAIR),
        "row_attention_pooling.out_proj.weight": _array(rng, D_SINGLE, D_PAIR),
        "plddt_ln.weight": _array(rng, D_SINGLE),
        "plddt_ln.bias": _array(rng, D_SINGLE),
        "plddt_weight": _array(rng, MAX_ATOMS_PER_TOKEN, D_SINGLE, N_PLDDT_BINS),
        "pae_ln.weight": _array(rng, D_PAIR),
        "pae_ln.bias": _array(rng, D_PAIR),
        "pae_head.weight": _array(rng, N_PAE_BINS, D_PAIR),
        "pde_ln.weight": _array(rng, D_PAIR),
        "pde_ln.bias": _array(rng, D_PAIR),
        "pde_head.weight": _array(rng, N_PAE_BINS, D_PAIR),
        "resolved_ln.weight": _array(rng, D_SINGLE),
        "resolved_ln.bias": _array(rng, D_SINGLE),
        "resolved_weight": _array(rng, MAX_ATOMS_PER_TOKEN, D_SINGLE, 2),
    }
    for index in range(TRUNK_LAYERS):
        block = f"folding_trunk.blocks.{index}"
        for flow in ("tri_mul_out", "tri_mul_in"):
            engine = f"{block}.{flow}._engine"
            params[f"{engine}.norm_start.weight"] = _array(rng, D_PAIR)
            params[f"{engine}.norm_start.bias"] = _array(rng, D_PAIR)
            params[f"{engine}.proj_bundle.weight"] = _array(rng, 4 * D_PAIR, D_PAIR)
            params[f"{engine}.norm_mix.weight"] = _array(rng, D_PAIR)
            params[f"{engine}.norm_mix.bias"] = _array(rng, D_PAIR)
            params[f"{engine}.proj_emit.weight"] = _array(rng, D_PAIR, D_PAIR)
            params[f"{engine}.proj_gate.weight"] = _array(rng, D_PAIR, D_PAIR)
        params[f"{block}.pair_transition.norm.weight"] = _array(rng, D_PAIR)
        params[f"{block}.pair_transition.norm.bias"] = _array(rng, D_PAIR)
        params[f"{block}.pair_transition.ffn.w12.weight"] = _array(
            rng, 2 * D_PAIR, D_PAIR
        )
        params[f"{block}.pair_transition.ffn.w3.weight"] = _array(rng, D_PAIR, D_PAIR)
    return params


def _inputs(seed: int = 1) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "s_inputs": _array(rng, 1, TOKENS, D_IN),
        "z": _array(rng, 1, TOKENS, TOKENS, D_PAIR),
        "x_pred": _array(rng, 1, ATOMS, 3),
        "distogram_atom_idx": jnp.arange(TOKENS, dtype=jnp.int32)[None] % ATOMS,
        "token_mask": jnp.ones((1, TOKENS)),
        "atom_to_token": (jnp.arange(ATOMS, dtype=jnp.int32)[None] % TOKENS),
        "atom_mask": jnp.ones((1, ATOMS)),
        "asym_id": jnp.zeros((1, TOKENS), jnp.int32),
        "mol_type": jnp.zeros((1, TOKENS), jnp.int32),
    }


class Recorder:
    """Every linear, norm and exponential the head reaches, with its dtypes.

    `weight_dtypes` reads the dtype of the array that actually entered
    `jnp.matmul`, not the dtype of the stored parameter. Those differ by
    design: `_autocast_linear` narrows its operands inside the call and leaves
    the checkpoint's float32 array alone, so a test that read the stored
    parameter would report float32 for a genuinely narrowed projection.
    """

    def __init__(self) -> None:
        self.linears: list[tuple[str, str, tuple[str, ...], str]] = []
        self.matmuls: list[tuple[str, str]] = []
        self.norms: list[tuple[str, str, str]] = []
        self.exponentials: list[tuple[str, str]] = []
        self.trunk_entry: list[str] = []
        self.pooling_entry: list[str] = []

    def weight_dtypes(self, name: str) -> set[str]:
        return {dtype for row in self.linears if row[1] == name for dtype in row[2]}

    def input_dtypes(self, name: str) -> set[str]:
        return {row[3] for row in self.linears if row[1] == name}

    def output_dtypes(self, name: str) -> set[str]:
        return {row[0] for row in self.linears if row[1] == name}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    log = Recorder()
    plain, narrowed = heads.linear, heads._autocast_linear
    real_norm = heads.layer_norm
    real_pooling = heads.row_attention_pooling
    real_trunk = heads.folding_trunk
    real_matmul = jnp.matmul
    real_sigmoid, real_silu, real_softmax = jax.nn.sigmoid, jax.nn.silu, jax.nn.softmax

    def record_matmul(lhs, rhs, *args, **kwargs):
        log.matmuls.append((str(lhs.dtype), str(rhs.dtype)))
        return real_matmul(lhs, rhs, *args, **kwargs)

    monkeypatch.setattr(jnp, "matmul", record_matmul)

    def record_linear(inner, x, params, prefix):
        first = len(log.matmuls)
        out = inner(x, params, prefix)
        # The right operand of every matmul this projection issued: the
        # realised weight, whatever the checkpoint stores.
        realised = tuple(rhs for _, rhs in log.matmuls[first:])
        log.linears.append((str(out.dtype), prefix, realised, str(x.dtype)))
        return out

    monkeypatch.setattr(
        heads, "linear", lambda x, p, prefix: record_linear(plain, x, p, prefix)
    )
    monkeypatch.setattr(
        heads,
        "_autocast_linear",
        lambda x, p, prefix: record_linear(narrowed, x, p, prefix),
    )
    monkeypatch.setattr(
        embedders, "linear", lambda x, p, prefix: record_linear(plain, x, p, prefix)
    )

    def record_norm(x, weight=None, bias=None, eps=1e-5):
        out = real_norm(x, weight, bias, eps)
        log.norms.append(
            (
                str(x.dtype),
                "none" if weight is None else str(weight.dtype),
                str(out.dtype),
            )
        )
        return out

    monkeypatch.setattr(heads, "layer_norm", record_norm)

    def record_pooling(z, mask, params, prefix=""):
        log.pooling_entry.append(str(z.dtype))
        return real_pooling(z, mask, params, prefix)

    monkeypatch.setattr(heads, "row_attention_pooling", record_pooling)

    def record_trunk(pair, params, prefix, **kwargs):
        log.trunk_entry.append(str(pair.dtype))
        return real_trunk(pair, params, prefix, **kwargs)

    monkeypatch.setattr(heads, "folding_trunk", record_trunk)

    for module, name, real in (
        (trunk.jax.nn, "sigmoid", real_sigmoid),
        (trunk.jax.nn, "silu", real_silu),
        (jax.nn, "softmax", real_softmax),
    ):

        def spy(x, *args, _name=name, _real=real, **kwargs):
            log.exponentials.append((_name, str(x.dtype)))
            return _real(x, *args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return log


def _run(params, confidence_dtype, *, trunk_dtype=jnp.bfloat16, **extra):
    return heads.confidence_head(
        params=params,
        prefix="",
        n_layers=TRUNK_LAYERS,
        n_chains=1,
        trunk_dtype=trunk_dtype,
        confidence_dtype=confidence_dtype,
        **_inputs(),
        **extra,
    )


def _shipped_dtype() -> object:
    """The width the port ships, resolved rather than spelled.

    Every test that means "the default" goes through this, so the default is
    asserted by what it makes the arrays do and a later flip is one edit
    rather than a rewrite of every assertion that named a literal.
    """
    return jnp.dtype(structure_model.ModelSettings().confidence_dtype)


def test_the_shipped_default_leaves_the_confidence_tree_in_float32(recorder) -> None:
    """The default is read off arrays, not off `ModelSettings`.

    Asserting `confidence_dtype == "float32"` would pass on a port that
    carries the string and hands the head something else, which is this
    port's recorded failure mode, and it would have to be rewritten rather
    than simply fail if the default ever moved. So the default is resolved
    through `_shipped_dtype` and then read back off the operands: no
    parameter is copied and nothing narrows.

    This is also the arm every recorded ESMFold2 confidence number describes,
    and the one upstream runs -- its `ConfidenceHead.forward` sits outside
    every autocast region.
    """
    params = _params()
    original = dict(params)
    _run(params, _shipped_dtype())

    assert recorder.linears, "the recorder saw no linear at all"
    assert recorder.matmuls, "the recorder saw no matmul at all"
    for out_dtype, prefix, weights, in_dtype in recorder.linears:
        assert (out_dtype, in_dtype) == ("float32", "float32"), prefix
        assert set(weights) == {"float32"}, prefix
    assert set(recorder.trunk_entry) == {"float32"}
    assert set(recorder.pooling_entry) == {"float32"}
    # Identity, not equality: the default must not rebuild the tree either.
    for name, value in original.items():
        assert params[name] is value


def test_the_narrowed_group_is_realised_in_bfloat16(recorder) -> None:
    """AlphaFold 3's `:121-127`: the re-embedding and its residual stream."""
    _run(_params(), jnp.bfloat16)

    for name in REEMBEDDING_PROJECTIONS:
        assert recorder.weight_dtypes(name) == {"bfloat16"}, name
        assert recorder.output_dtypes(name) == {"bfloat16"}, name
    # The triple product's operands are the one quadratic tensor here.
    assert recorder.input_dtypes("s_to_z_prod_out") == {"bfloat16"}
    # A bfloat16 pair reaches the head's own trunk, so its residual stream is
    # bfloat16 for all four blocks -- AlphaFold 3's narrowed Pairformer.
    assert recorder.trunk_entry == ["bfloat16"]


def test_the_output_heads_stay_float32_under_the_option(recorder) -> None:
    """AlphaFold 3's `:163` and `:244`: the boundary holds."""
    result = _run(_params(), jnp.bfloat16)

    assert recorder.pooling_entry == ["float32"], "the float32 boundary moved"
    for name in (*OUTPUT_PROJECTIONS, "row_attention_pooling.attn_proj"):
        assert recorder.weight_dtypes(name) == {"float32"}, name
        assert recorder.input_dtypes(name) == {"float32"}, name
        assert recorder.output_dtypes(name) == {"float32"}, name
    for value in result.values():
        assert value.dtype == jnp.float32


def test_no_exponential_receives_a_rounded_input(recorder) -> None:
    """The head's trunk narrows its GEMMs and not its gates.

    `pair_update_block` has no softmax; its exponentials are four sigmoid
    gates and one SwiGLU per block, and `native_autocast` upcasts all five.
    The pooling and pTM softmaxes are the head's own and stay float32 because
    nothing narrows what feeds them.
    """
    _run(_params(), jnp.bfloat16)

    assert recorder.exponentials, "the recorder saw no exponential at all"
    assert {kind for kind, _ in recorder.exponentials} >= {"sigmoid", "softmax"}
    for kind, dtype in recorder.exponentials:
        assert dtype == "float32", kind


def test_the_entry_norms_keep_float32_statistics_and_affine(recorder) -> None:
    """Float32 statistics cannot recover a rounded affine weight."""
    _run(_params(), jnp.bfloat16)

    assert recorder.norms
    for in_dtype, weight_dtype, _ in recorder.norms:
        assert (in_dtype, weight_dtype) == ("float32", "float32")


def test_the_narrowed_weights_reach_a_matmul() -> None:
    """The tripwire: doubling a narrowed weight must move the result.

    Without it every assertion above passes on a head that computes the
    float32 answer and casts it, and on one that never reads the option.
    """
    baseline = np.asarray(_run(_params(), jnp.bfloat16)["plddt"])
    for name in REEMBEDDING_PROJECTIONS:
        doubled = _params()
        doubled[f"{name}.weight"] = doubled[f"{name}.weight"] * 2.0
        moved = np.asarray(_run(doubled, jnp.bfloat16)["plddt"])
        assert not np.allclose(moved, baseline), f"{name} never reached a matmul"


def test_narrowing_changes_the_answer_but_not_by_much() -> None:
    """Two arms that agree exactly would mean the option did nothing."""
    params = _params()
    wide = _run(params, jnp.float32)
    narrow = _run(params, jnp.bfloat16)

    assert not np.array_equal(np.asarray(wide["plddt"]), np.asarray(narrow["plddt"]))
    np.testing.assert_allclose(
        np.asarray(narrow["plddt"]), np.asarray(wide["plddt"]), atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(narrow["ptm"]), np.asarray(wide["ptm"]), atol=2e-2
    )


def test_the_distance_bins_are_chosen_in_float32(recorder) -> None:
    """A rounded distance crosses a boundary; a rounded table row does not.

    The bin index is a gather, so an off-by-one there swaps an embedding row
    outright. Both arms must select the same bins for every pair.
    """
    params = _params()
    # Two separations either side of the 0.5 boundary, one of them inside
    # bfloat16's spacing at that magnitude, so a narrowed distance would
    # select a different row.
    coords = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.50048828, 0.0, 0.0], [0.49951172, 0.0, 0.0]]],
        jnp.float32,
    )
    seen: list[np.ndarray] = []
    table = params["dist_bin_pairwise_embed.weight"]

    class Spy:
        """Record the gather index, whatever width the table itself takes."""

        def __init__(self, values):
            self._values = values

        @property
        def dtype(self):
            return self._values.dtype

        def astype(self, dtype):
            return Spy(self._values.astype(dtype))

        def __getitem__(self, index):
            seen.append(np.asarray(index))
            return self._values[index]

    arrays = {
        **_inputs(),
        "x_pred": coords,
        "atom_to_token": jnp.asarray([[0, 1, 2]], jnp.int32),
        "atom_mask": jnp.ones((1, 3)),
        "distogram_atom_idx": jnp.asarray([[0, 1, 2, 0]], jnp.int32),
    }
    for confidence_dtype in (jnp.float32, jnp.bfloat16):
        spied = dict(params)
        spied["dist_bin_pairwise_embed.weight"] = Spy(table)
        heads.confidence_head(
            params=spied,
            prefix="",
            n_layers=TRUNK_LAYERS,
            n_chains=1,
            trunk_dtype=jnp.bfloat16,
            confidence_dtype=confidence_dtype,
            **arrays,
        )
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], seen[1])
    assert seen[0].dtype == np.int32
    # The two separations really do land in different bins, so the equality
    # above is a measurement rather than a coincidence of a constant index.
    assert len(np.unique(seen[0])) > 1


def test_the_incoming_encodings_do_not_promote_the_accumulator(
    recorder, monkeypatch
) -> None:
    """The one arm where the adds at the top of the re-embedding matter.

    `relative_position_encoding` and `token_bonds_encoding` arrive in
    `trunk_dtype`, so under `trunk_dtype=float32` they are float32. Added to
    a bfloat16 accumulator without a cast they promote it on the very first
    line, and the option is inert for everything after -- with every other
    assertion in this file still green, because they run without encodings.
    """
    seen: list[str] = []
    real_shard = heads.shard_pair_rows

    def record_shard(x):
        seen.append(str(x.dtype))
        return real_shard(x)

    monkeypatch.setattr(heads, "shard_pair_rows", record_shard)
    encoding = jnp.zeros((1, TOKENS, TOKENS, D_PAIR), jnp.float32)
    _run(
        _params(),
        jnp.bfloat16,
        trunk_dtype=jnp.float32,
        relative_position_encoding=encoding,
        token_bonds_encoding=encoding,
    )

    # The pair as it leaves the re-embedding, float32 encodings and all.
    assert seen[0] == "bfloat16"
    # `trunk_dtype` still governs the trunk, which widens it back here.
    assert recorder.trunk_entry == ["float32"]


def test_context_parallelism_narrows_the_reembedding_and_not_the_trunk(
    recorder, monkeypatch
) -> None:
    """Under a mesh the trunk takes the storage-cast branch, unnarrowed here.

    That branch rounds its own gates with its parameters, which is why the
    option stops at the cast `confidence_head` already applies rather than
    handing it a bfloat16 pair of its own making.
    """
    monkeypatch.setattr(heads, "cp_mesh", lambda: object())
    monkeypatch.setattr(heads, "shard_pair_rows", lambda x: x)
    _run(_params(), jnp.bfloat16, trunk_dtype=jnp.float32)

    for name in REEMBEDDING_PROJECTIONS:
        assert recorder.weight_dtypes(name) == {"bfloat16"}, name
    # `trunk_dtype` is float32, so the cast at the trunk call widens the pair
    # back and the trunk runs exactly as it does without the option.
    assert recorder.trunk_entry == ["float32"]
    assert recorder.pooling_entry == ["float32"]


def test_the_accepted_values_are_the_two_spellings() -> None:
    assert CONFIDENCE_DTYPES == ("float32", "bfloat16")


def test_the_backend_and_the_port_default_to_the_same_width() -> None:
    """One default spelled in two layers, and they must agree.

    `_FIXED_COMPILE_DEFAULTS` is a literal so option planning stays free of
    JAX, which means nothing but this test stops it drifting from the field
    it mirrors. Drift here is silent and specific: the backend would strip an
    explicitly spelled value that the port does not actually resolve to, and
    answer that run out of the wrong compilation namespace.
    """
    from foldjax.backends.esmfold2 import _FIXED_COMPILE_DEFAULTS

    assert (
        _FIXED_COMPILE_DEFAULTS["confidence_dtype"]
        == structure_model.ModelSettings().confidence_dtype
    )


@pytest.mark.parametrize("asked", CONFIDENCE_DTYPES)
def test_the_predict_keyword_reaches_the_settings(asked: str) -> None:
    """`inference.predict(confidence_dtype=...)` -> `ModelSettings`."""
    base = structure_model.ModelSettings()
    assert (
        structure_model.with_overrides(base, confidence_dtype=asked).confidence_dtype
        == asked
    )
    # `None` is "did not ask", which is not the same as asking for the default.
    assert (
        structure_model.with_overrides(
            dataclasses.replace(base, confidence_dtype="bfloat16"),
            confidence_dtype=None,
        ).confidence_dtype
        == "bfloat16"
    )


@pytest.mark.parametrize("asked", CONFIDENCE_DTYPES)
def test_the_inference_keyword_reaches_the_compiled_identity(
    monkeypatch, asked: str
) -> None:
    """`inference.predict` -> the settings the executable is keyed on.

    The backend test proves the option reaches this keyword; this proves the
    keyword reaches the compiled program's identity rather than being
    accepted and dropped, which would answer a narrowed run out of the
    float32 executable's cache entry.
    """
    features = pad_features(
        build_features([("AG", "A", 0, 0)]), n_token=8, n_atom=64, n_msa=4
    )
    settings = structure_model.ModelSettings()
    model = SimpleNamespace(
        settings=settings,
        parameters={"token_bonds.weight": jnp.ones((settings.d_pair, 1), jnp.bfloat16)},
        esmc_parameters=None,
        esmc_settings=None,
    )
    seen: list[str] = []

    def fake_compiled_predict(*identity):
        seen.append(identity[0].confidence_dtype)
        return lambda *args: {}

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    inference.predict(jax.random.key(0), features, model, confidence_dtype=asked)

    assert seen == [asked]


def test_the_inference_keyword_refuses_an_unknown_value() -> None:
    """Named values in the message, so the caller is told what to write."""
    with pytest.raises(ValueError, match="float32.*bfloat16"):
        inference.predict(
            jax.random.key(0),
            {},
            SimpleNamespace(settings=structure_model.ModelSettings()),
            confidence_dtype="bf16",
        )


@pytest.mark.parametrize("asked", CONFIDENCE_DTYPES)
def test_the_settings_field_reaches_the_head(monkeypatch, asked: str) -> None:
    """`ModelSettings.confidence_dtype` -> `confidence_head`.

    Deleting the one line in `predict` that forwards the field leaves every
    other test in this file green, because they all pass the keyword by hand.
    This is the only test that fails.
    """
    settings = dataclasses.replace(
        structure_model.ModelSettings(),
        d_pair=2,
        d_inputs=2,
        trunk_n_layers=0,
        lm_encoder_n_layers=None,
        coda_n_layers=0,
        confidence_n_layers=0,
        msa_n_layers=None,
        num_recycles=0,
        num_samples=1,
        confidence_sample_sequential=False,
        trunk_dtype="float32",
        confidence_dtype=asked,
    )
    seen: list[object] = []

    for name, stub in (
        (
            "one_hot_atom_features",
            lambda *a, **k: (
                jnp.zeros((1, 2, 128), jnp.float32),
                jnp.zeros((1, 2, 4, 64), jnp.float32),
            ),
        ),
        ("inputs_embedding", lambda *a, **k: jnp.zeros((1, 2, 2), jnp.float32)),
        (
            "relative_position_encoding",
            lambda *a, **k: jnp.zeros((1, 2, 2, 2), jnp.float32),
        ),
        ("_token_bonds_encoding", lambda *a, **k: jnp.zeros((1, 2, 2, 2), jnp.float32)),
        ("run_loops", lambda key, z, z_init, *a, **k: z_init),
        ("folding_trunk", lambda value, *a, **k: value),
        ("linear", lambda value, params, prefix: value[..., :1]),
    ):
        monkeypatch.setattr(structure_model, name, stub)
    monkeypatch.setattr(
        structure_model.diffusion, "build_cache", lambda *a, **k: {"pair": a[7]}
    )
    monkeypatch.setattr(
        structure_model.diffusion,
        "sample",
        lambda *a, **k: (jnp.zeros((1, 2, 3), jnp.float32), None),
    )

    def record(single, pair, coords, *args, **kwargs):
        seen.append(kwargs["confidence_dtype"])
        token = jnp.zeros((1, 2), jnp.float32)
        return {
            "plddt": token,
            "plddt_per_atom": token,
            "plddt_ca": token,
            "complex_plddt": jnp.zeros((1,), jnp.float32),
            "ptm": jnp.zeros((1,), jnp.float32),
            "pair_chains_iptm": jnp.zeros((1, 1, 1), jnp.float32),
        }

    monkeypatch.setattr(structure_model, "confidence_head", record)
    structure_model.predict(
        jax.random.key(0),
        _cheap_features(),
        {"token_bonds.weight": jnp.ones((2, 1), jnp.float32)},
        settings=settings,
        initial_pair_state=jnp.zeros((1, 2, 2, 2), jnp.float32),
        n_chains=1,
        return_distogram_logits=False,
    )

    # A realised dtype, not the string the setting carries: `predict` resolves
    # the spelling, and a forward that handed the head the raw string would
    # still make `jnp.dtype(...)` agree while changing what reaches the array.
    assert seen == [jnp.dtype(asked)]
