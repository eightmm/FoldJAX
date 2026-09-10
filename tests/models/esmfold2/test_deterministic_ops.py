"""Repeatable reduction orders on ESMFold2, asked for per executable.

The mechanism is the shared one: `foldjax.models._compile_policy` owns the two
XLA keys, and a run that asks for them gets a second executable owner rather
than an option on the call. What is specific to this port is where the owners
are, because ESMFold2 is two models. The structure graph already had one owner;
ESMC-6B had none at all -- `esmc.encode` is a Python loop over 80 blocks
dispatched operation by operation, outside any executable an option can reach.

So `deterministic=on` here also puts the language model's blocks inside an
executable. That has a consequence worth stating rather than discovering:

* **`off` is byte-for-byte the run it always was.** No block is compiled, both
  block owners stay empty, and the structure graph is built with exactly the
  arguments it was built with before this existed. Every recorded ESMFold2
  number still describes it.
* **`on` and `off` are not expected to agree bitwise, and never could.** A
  reduction-order option changes reduction order; that is the whole point. On
  top of that, an eager stack and a compiled one are never bitwise equal on
  XLA to begin with -- measured on CPU in this worktree, a lone `layer_norm`
  moves 3e-8 between the two, one `esmc.block` 4.7e-10, and the three-layer
  stack below 4.5e-8, because a fused loop vectorizes its reductions and
  contracts multiply-add pairs that the op-by-op form cannot. No setting
  inside this package removes that, so the pooled stack is checked against the
  eager one at Protenix's tolerance for the same construct, not for equal
  bits.

What is checked for equal bits is the thing that matters: the default run.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._compile_policy import DETERMINISTIC_COMPILER_OPTIONS
from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.models import esmc

D_MODEL, N_HEADS, N_LAYERS, VOCAB = 8, 2, 3, 64


def _toy_parameters() -> dict[str, jnp.ndarray]:
    """One ESMC checkpoint small enough to compile in a unit test."""
    rng = np.random.default_rng(9)
    shapes: dict[str, tuple[int, ...]] = {
        "embed.weight": (VOCAB, D_MODEL),
        "transformer.norm.weight": (D_MODEL,),
    }
    for index in range(N_LAYERS):
        prefix = f"transformer.blocks.{index}."
        for name, shape in {
            "attn.layernorm_qkv.layer_norm_weight": (D_MODEL,),
            "attn.layernorm_qkv.layer_norm_bias": (D_MODEL,),
            "attn.layernorm_qkv.weight": (3 * D_MODEL, D_MODEL),
            "attn.q_ln.weight": (D_MODEL,),
            "attn.k_ln.weight": (D_MODEL,),
            "attn.out_proj.weight": (D_MODEL, D_MODEL),
            "ffn.layer_norm_weight": (D_MODEL,),
            "ffn.layer_norm_bias": (D_MODEL,),
            "ffn.fc1_weight": (4 * D_MODEL, D_MODEL),
            "ffn.fc2_weight": (D_MODEL, 2 * D_MODEL),
        }.items():
            shapes[prefix + name] = shape
    return {
        name: jnp.asarray(rng.normal(size=shape) * 0.1, jnp.float32)
        for name, shape in shapes.items()
    }


def _toy_settings() -> esmc.ESMCSettings:
    return esmc.ESMCSettings(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS
    )


def _toy_tokens() -> tuple[jnp.ndarray, jnp.ndarray]:
    ids = jnp.asarray([[0, 4, 2, 7, 1]], jnp.int32)
    return ids, jnp.zeros_like(ids)


def _eager_stack(
    ids: jnp.ndarray,
    sequence_id: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    settings: esmc.ESMCSettings,
) -> jnp.ndarray:
    """The op-by-op stack, spelled out here so the guard owns its reference."""
    x = params["embed.weight"][ids]
    rope = esmc.rotary_tables(
        ids.shape[1], settings.d_model // settings.n_heads, settings.rope_base
    )
    collected = [x]
    for index in range(settings.n_layers):
        x = esmc.block(
            x,
            params,
            f"transformer.blocks.{index}",
            n_heads=settings.n_heads,
            sequence_id=sequence_id,
            rope=rope,
            residual_scale=settings.residual_scale,
        )
        if index + 1 < settings.n_layers:
            collected.append(x)
    collected.append(esmc._norm(x, params["transformer.norm.weight"]))
    return jnp.stack(collected, axis=0)


def _same_bits(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    )


@pytest.fixture
def empty_block_owners():
    """Both ESMC block owners, emptied before and after.

    They are module-level and cache by identity, so an earlier test in the
    session running the same toy shapes would otherwise satisfy a call from
    its cache and make an entry-count assertion meaningless.
    """
    esmc._compiled_block.clear_cache()
    esmc._compiled_block_deterministic.clear_cache()
    try:
        yield
    finally:
        esmc._compiled_block.clear_cache()
        esmc._compiled_block_deterministic.clear_cache()


# ---------------------------------------------------------------------------
# The language model's blocks
# ---------------------------------------------------------------------------


def test_the_default_language_model_run_is_the_eager_stack_bit_for_bit(
    empty_block_owners,
) -> None:
    """Unasked, ESMC dispatches exactly the operations it dispatched before.

    This is the guard the whole design turns on: wrapping the blocks would
    otherwise have moved every recorded ESMFold2 number by the fusion delta
    documented in this module's docstring, on a run nobody asked to change.
    """
    ids, sequence_id = _toy_tokens()
    params, settings = _toy_parameters(), _toy_settings()

    hidden = esmc.encode(ids, sequence_id, params, settings=settings)

    assert _same_bits(hidden, _eager_stack(ids, sequence_id, params, settings))
    assert esmc._compiled_block._entry_count() == 0
    assert esmc._compiled_block_deterministic._entry_count() == 0


def test_asking_for_the_owner_alone_reaches_the_default_pool(
    empty_block_owners,
) -> None:
    """`compile_blocks` without the policy is the plain executable owner."""
    ids, sequence_id = _toy_tokens()
    params, settings = _toy_parameters(), _toy_settings()

    hidden = esmc.encode(
        ids, sequence_id, params, settings=settings, compile_blocks=True
    )

    assert esmc._compiled_block._entry_count() == 1
    assert esmc._compiled_block_deterministic._entry_count() == 0
    np.testing.assert_allclose(
        np.asarray(hidden),
        np.asarray(_eager_stack(ids, sequence_id, params, settings)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_the_policy_routes_the_blocks_through_its_own_owner(
    empty_block_owners,
) -> None:
    """The tripwire: an `on` run that fell back to eager would pass without it.

    One executable for all `n_layers` blocks, because they share a shape --
    the same property that lets the 80 real ones share one program.
    """
    ids, sequence_id = _toy_tokens()
    params, settings = _toy_parameters(), _toy_settings()

    hidden = esmc.encode(
        ids, sequence_id, params, settings=settings, deterministic=True
    )

    assert esmc._compiled_block_deterministic._entry_count() == 1
    assert esmc._compiled_block._entry_count() == 0
    np.testing.assert_allclose(
        np.asarray(hidden),
        np.asarray(_eager_stack(ids, sequence_id, params, settings)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_the_two_block_policies_are_two_owners() -> None:
    """One cache entry cannot serve both: the option is part of the build."""
    default = esmc._compiled_block
    repeatable = esmc._compiled_block_deterministic

    assert default is not repeatable
    assert esmc._block_pool(False) is default
    assert esmc._block_pool(True) is repeatable
    assert default._compiler_options is None
    assert repeatable._compiler_options == DETERMINISTIC_COMPILER_OPTIONS


def test_an_explicitly_eager_block_loop_refuses_the_policy() -> None:
    """No executable to carry it; running anyway would report a false promise."""
    ids, sequence_id = _toy_tokens()

    with pytest.raises(ValueError, match="deterministic reductions"):
        esmc.encode(
            ids,
            sequence_id,
            _toy_parameters(),
            settings=_toy_settings(),
            compile_blocks=False,
            deterministic=True,
        )


def test_the_block_owner_is_handed_one_block_s_own_subtree() -> None:
    """Whole-checkpoint arguments would make 80 programs out of one shape."""
    params = _toy_parameters()

    subtree = esmc._block_parameters(params, "transformer.blocks.1")

    assert "attn.out_proj.weight" in subtree
    assert not any(name.startswith("transformer.") for name in subtree)
    assert subtree["ffn.fc1_weight"] is params["transformer.blocks.1.ffn.fc1_weight"]
    assert esmc._block_parameters(params, "transformer.blocks.0").keys() == (
        subtree.keys()
    )


def test_the_hidden_state_helper_carries_the_policy(monkeypatch) -> None:
    """`lm_hidden_states` is the only way the port reaches `encode`."""
    seen: list[dict[str, Any]] = []

    def capture(*_args: Any, **kwargs: Any) -> jnp.ndarray:
        seen.append(kwargs)
        return jnp.zeros((N_LAYERS + 1, 1, 4, D_MODEL), jnp.float32)

    monkeypatch.setattr(esmc, "encode", capture)
    tokens = np.asarray([[0, 5, 6, 2]], np.int32)
    zeros = np.zeros_like(tokens)
    esmc.lm_hidden_states(
        tokens,
        zeros,
        zeros,
        zeros,
        np.ones_like(tokens),
        _toy_parameters(),
        settings=_toy_settings(),
        deterministic=True,
    )

    assert [call["deterministic"] for call in seen] == [True]


# ---------------------------------------------------------------------------
# The structure graph
# ---------------------------------------------------------------------------

_PREDICT_ARGUMENTS = (
    jnp.zeros((2,), dtype=jnp.uint32),
    {},
    {},
    None,
    None,  # replaced with the settings object below
    1,
    False,
    1,
    (),
    False,
    False,
    False,
    True,
    False,
    256,
)


@pytest.fixture
def recorded_jit(monkeypatch) -> list[dict[str, Any]]:
    """Record what a pool asks `jax.jit` for, compiling nothing.

    Emptied on both sides. Before, because these owners are module-level and
    cache by identity, so an earlier test that ran the same shapes would
    satisfy the call from its cache, record nothing, and let "no compiler
    options were passed" pass on an empty list. After, because what they would
    otherwise retain is the stub above -- a callable that returns `{}` for the
    identity every other ESMFold2 compile-identity test uses.
    """
    calls: list[dict[str, Any]] = []

    def fake_jit(function: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return lambda *args, **call_kwargs: {}

    def empty() -> None:
        inference.compiled_predict.cache_clear()
        inference._compiled_language_model_embedding.cache_clear()
        esmc._compiled_block.clear_cache()
        esmc._compiled_block_deterministic.clear_cache()

    empty()
    monkeypatch.setattr(jax, "jit", fake_jit)
    yield calls
    empty()


def _run_graph(deterministic: bool) -> None:
    """Drive the compiled entry point far enough to build one executable."""
    settings = inference.structure_model.ModelSettings()
    runner = inference.compiled_predict(settings, 1, deterministic=deterministic)
    runner(*(_PREDICT_ARGUMENTS[:4]), settings, *_PREDICT_ARGUMENTS[5:])


def test_off_builds_the_structure_graph_exactly_as_it_did_before(
    recorded_jit,
) -> None:
    """The default run must be the run every recorded number describes."""
    _run_graph(False)

    assert recorded_jit == [
        {"static_argnames": inference._COMPILED_PREDICT_STATIC_ARGNAMES}
    ]


def test_on_carries_the_options_into_the_structure_compile(recorded_jit) -> None:
    """And the values come from the constant, not from a literal here."""
    _run_graph(True)

    assert len(recorded_jit) == 1
    assert recorded_jit[0]["compiler_options"] == DETERMINISTIC_COMPILER_OPTIONS


def test_the_two_graph_policies_are_two_owners() -> None:
    default = inference._compiled_predict_pool
    repeatable = inference._compiled_predict_pool_deterministic

    assert default is not repeatable
    assert default._compiler_options is None
    assert repeatable._compiler_options == DETERMINISTIC_COMPILER_OPTIONS
    assert default._limit == repeatable._limit == 8


def test_the_factory_keys_on_the_policy() -> None:
    """A cached facade built without the option must not serve a run with it."""
    settings = inference.structure_model.ModelSettings()

    plain = inference.compiled_predict(settings, 1)
    repeatable = inference.compiled_predict(settings, 1, deterministic=True)

    assert plain is inference.compiled_predict(settings, 1)
    assert repeatable is inference.compiled_predict(settings, 1, deterministic=True)
    assert plain is not repeatable
    assert plain._pool is inference._compiled_predict_pool
    assert repeatable._pool is inference._compiled_predict_pool_deterministic


def test_the_eager_predict_path_refuses_instead_of_running_without_it() -> None:
    """`compile_it=False` owns no executable to carry the option."""
    with pytest.raises(ValueError, match="deterministic reductions"):
        inference.predict(
            jnp.zeros((2,), jnp.uint32),
            {},
            None,
            compile_it=False,
            deterministic=True,
        )


# ---------------------------------------------------------------------------
# The compact language-model embedding between them
# ---------------------------------------------------------------------------


def test_the_embedding_owner_keys_on_the_policy() -> None:
    """Two policies, two owners, out of an `lru_cache` that had one key."""
    inference._compiled_language_model_embedding.cache_clear()

    plain = inference._compiled_language_model_embedding("float32", (), False)
    repeatable = inference._compiled_language_model_embedding("float32", (), True)

    assert plain is inference._compiled_language_model_embedding("float32", (), False)
    assert plain is not repeatable


def test_the_embedding_owner_asks_for_nothing_when_unasked(recorded_jit) -> None:
    """An empty option map is a different compile from no option map."""
    inference._compiled_language_model_embedding("float32", (), False)

    assert recorded_jit == [{}]


def test_the_embedding_owner_carries_the_options_when_asked(recorded_jit) -> None:
    inference._compiled_language_model_embedding("float32", (), True)

    assert recorded_jit == [{"compiler_options": DETERMINISTIC_COMPILER_OPTIONS}]


def test_predict_names_the_policy_to_the_factory_only_when_asked(
    monkeypatch,
) -> None:
    """The tripwire on the hop `predict` makes; both directions.

    Unasked, the factory has to be reached in the call form it always was --
    an extra keyword is a different `lru_cache` key, so an unrequested run
    would stop hitting the entry it has been hitting.
    """
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", raising=False)
    settings = inference.structure_model.ModelSettings()
    features = {
        "asym_id": np.zeros((1, 2), dtype=np.int32),
        "token_attention_mask": np.ones((1, 2), dtype=bool),
        "token_bonds": np.ones((1, 2, 2, 1), dtype=np.float32),
    }
    loaded = inference.LoadedModel(
        parameters={"token_bonds.weight": jnp.ones((settings.d_pair, 1), jnp.bfloat16)},
        settings=settings,
    )
    seen: list[Any] = []

    def fake_compiled_predict(*_identity: Any, **kwargs: Any):
        seen.append(kwargs.get("deterministic"))
        return lambda *_args, **_kwargs: {}

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    embedding = jnp.zeros((1, 2, 256), jnp.bfloat16)

    inference.predict(
        jax.random.key(0), features, loaded, precomputed_lm_embedding=embedding
    )
    inference.predict(
        jax.random.key(0),
        features,
        loaded,
        precomputed_lm_embedding=embedding,
        deterministic=True,
    )

    assert seen == [None, True]


def _one_block_call() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Positional and keyword arguments for exactly one ESMC block."""
    params = _toy_parameters()
    ids, sequence_id = _toy_tokens()
    return (
        (
            params["embed.weight"][ids],
            esmc._block_parameters(params, "transformer.blocks.0"),
        ),
        {
            "prefix": "",
            "n_heads": N_HEADS,
            "sequence_id": sequence_id,
            "rope": esmc.rotary_tables(
                ids.shape[1], D_MODEL // N_HEADS, _toy_settings().rope_base
            ),
            "residual_scale": _toy_settings().residual_scale,
            "autocast_bfloat16": False,
        },
    )


def test_the_block_owner_asks_for_nothing_when_unasked(recorded_jit) -> None:
    """The third owner, recorded like the other two rather than inspected.

    The list is cleared after the arguments are realized: building them runs
    ordinary `jnp` work, and a scalar conversion of its own would otherwise be
    recorded here as an extra entry.
    """
    args, kwargs = _one_block_call()
    recorded_jit.clear()

    esmc._block_pool(False)(*args, **kwargs)

    assert recorded_jit == [{"static_argnames": esmc._BLOCK_STATIC_ARGNAMES}]


def test_the_block_owner_carries_the_options_when_asked(recorded_jit) -> None:
    """And the values come from the constant, not from a literal here."""
    args, kwargs = _one_block_call()
    recorded_jit.clear()

    esmc._block_pool(True)(*args, **kwargs)

    assert recorded_jit == [
        {
            "static_argnames": esmc._BLOCK_STATIC_ARGNAMES,
            "compiler_options": DETERMINISTIC_COMPILER_OPTIONS,
        }
    ]
