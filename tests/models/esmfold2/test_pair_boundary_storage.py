"""The pair keeps bfloat16 storage across `predict`'s float32 boundary.

Upstream's `z = z.float()` closes the trunk's autocast region, and this port
used to realise it as a float32 copy of the whole pair -- the tensor the
diffusion cache and the confidence head then each carried. At 2,096 tokens a
GPU peak-live attribution found that copy and the relative position encoding's
as two of the three largest tenants of the peak, 4,290 MiB apiece and 8.85 GB
apiece at 3,012 tokens.

What replaces it is *storage*, not arithmetic: widening bfloat16 is exact, so
a float32 tensor built by `astype` from a bfloat16 pair holds only values
bfloat16 can hold, and a consumer that widens before its own first float32
operation computes the same bits from either. These tests hold the two halves
of that claim apart:

* `test_the_consumers_are_handed_the_trunk_width` reads what each consumer
  actually receives, so the change cannot quietly stop happening.
* the three `..._computes_the_same_bits_...` tests compare each consumer's
  output against the same consumer fed the float32 copy, byte for byte. A
  tolerance would pass on an arithmetic change; these do not have one.
* `test_the_exported_pair_stays_float32` pins the one place the width is a
  contract rather than a buffer -- the tap and the `pair` representation a
  comparison harness reads.

`test_the_released_program_carries_no_float32_pair` is the count on the real
program, and needs the checkpoint.
"""

from __future__ import annotations

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion, heads
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.models.primitives import layer_norm
from tests.models.esmfold2.test_confidence_dtype import (
    D_PAIR,
    TOKENS,
    TRUNK_LAYERS,
    _inputs,
    _params,
)
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


def _narrow_pair(seed: int, tokens: int, width: int) -> jnp.ndarray:
    """A pair whose values are bfloat16-representable and not degenerate.

    Random rather than zeros: a storage change is invisible on a tensor whose
    values are all exactly representable anyway, and every assertion below
    would pass on one.
    """
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((1, tokens, tokens, width)).astype(np.float32)
    ).astype(jnp.bfloat16)


def _bytes(value) -> bytes:
    return np.asarray(value).tobytes()


# --- what each consumer receives -------------------------------------------


def _boundary_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Run a two-token `predict` with fake consumers that record their input.

    The scaffold is `test_distogram_output_flag`'s, with the trunk left at the
    released bfloat16 rather than pinned to float32: the widths this records
    are the ones the released configuration produces.
    """
    seen: dict[str, object] = {}
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
        trunk_dtype="bfloat16",
    )

    monkeypatch.setattr(
        structure_model,
        "one_hot_atom_features",
        lambda *args, **kwargs: (
            jnp.zeros((1, 2, 128), dtype=jnp.float32),
            jnp.zeros((1, 2, 4, 64), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "inputs_embedding",
        lambda *args, **kwargs: jnp.asarray(
            [[[1.0, -2.0], [3.0, -4.0]]], dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "relative_position_encoding",
        lambda *args, **kwargs: jnp.zeros((1, 2, 2, 2), dtype=jnp.bfloat16),
    )
    monkeypatch.setattr(
        structure_model,
        "_token_bonds_encoding",
        lambda *args, **kwargs: jnp.zeros((1, 2, 2, 2), dtype=jnp.bfloat16),
    )
    monkeypatch.setattr(
        structure_model,
        "run_loops",
        lambda key, z, z_init, *args, **kwargs: z_init,
    )
    monkeypatch.setattr(
        structure_model, "folding_trunk", lambda value, *args, **kwargs: value
    )

    def fake_linear(value, params, prefix):
        del params
        if prefix in {"z_init_1", "z_init_2", "parcae_readout"}:
            return value
        raise AssertionError(f"unexpected linear {prefix}")

    monkeypatch.setattr(structure_model, "linear", fake_linear)

    def fake_distogram(pair, params):
        del params
        seen["distogram"] = pair.dtype
        return jnp.sum(pair.astype(jnp.float32), axis=-1, keepdims=True)

    monkeypatch.setattr(structure_model, "_distogram_logits", fake_distogram)

    def fake_build_cache(*args, **kwargs):
        seen["diffusion_pair"] = args[7].dtype
        seen["diffusion_rel_pos"] = args[8].dtype
        return {"pair": args[7]}

    monkeypatch.setattr(structure_model.diffusion, "build_cache", fake_build_cache)

    def fake_sample(key, single, cache, *args, **kwargs):
        del key, single, args, kwargs
        signal = jnp.sum(cache["pair"].astype(jnp.float32))
        return jnp.broadcast_to(signal, (1, 2, 3)), None

    monkeypatch.setattr(structure_model.diffusion, "sample", fake_sample)

    def fake_confidence(single, pair, coords, *args, **kwargs):
        del single, args
        seen["confidence_pair"] = pair.dtype
        seen["confidence_rel_pos"] = kwargs["relative_position_encoding"].dtype
        seen["confidence_bonds"] = kwargs["token_bonds_encoding"].dtype
        signal = jnp.sum(pair.astype(jnp.float32)) + jnp.sum(coords)
        token = jnp.broadcast_to(signal, (1, 2))
        return {
            "plddt": token,
            "plddt_per_atom": token,
            "plddt_ca": token,
            "complex_plddt": jnp.reshape(signal, (1,)),
            "ptm": jnp.reshape(signal, (1,)),
            "pair_chains_iptm": jnp.reshape(signal, (1, 1, 1)),
        }

    monkeypatch.setattr(structure_model, "confidence_head", fake_confidence)

    output = structure_model.predict(
        jax.random.key(0),
        _cheap_features(),
        {"token_bonds.weight": jnp.ones((2, 1), dtype=jnp.float32)},
        settings=settings,
        initial_pair_state=jnp.zeros((1, 2, 2, 2), dtype=jnp.bfloat16),
        n_chains=1,
        return_representations=("pair",),
    )
    seen["exported"] = output["pair"].dtype
    return seen


def test_the_consumers_are_handed_the_trunk_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every consumer of the post-trunk pair receives bfloat16 storage.

    Read off the arrays the consumers are called with, not off the source: the
    boundary is one line, and a reinstated cast there would leave a source
    assertion about `shard_pair_rows(z)` green while restoring the buffer.
    """
    seen = _boundary_probe(monkeypatch)
    narrow = {
        name: dtype
        for name, dtype in seen.items()
        if name != "exported"
    }
    assert narrow == {
        "distogram": jnp.bfloat16,
        "diffusion_pair": jnp.bfloat16,
        "diffusion_rel_pos": jnp.bfloat16,
        "confidence_pair": jnp.bfloat16,
        "confidence_rel_pos": jnp.bfloat16,
        "confidence_bonds": jnp.bfloat16,
    }, seen


def test_the_exported_pair_stays_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """The representation and the tap keep the width the boundary used to give.

    The consumers take storage; a returned array is an output, and narrowing
    it would be a changed result rather than a changed buffer.
    """
    assert _boundary_probe(monkeypatch)["exported"] == jnp.float32


# --- bitwise, per consumer --------------------------------------------------


def test_the_distogram_computes_the_same_bits_from_bfloat16_storage() -> None:
    """`_distogram_logits` widens both addends; the sum is the same sum."""
    narrow = _narrow_pair(0, 12, 6)
    wide = narrow.astype(jnp.float32)
    rng = np.random.default_rng(1)
    params = {
        "distogram_head.weight": jnp.asarray(
            rng.standard_normal((5, 6)).astype(np.float32)
        ),
        "distogram_head.bias": jnp.asarray(
            rng.standard_normal(5).astype(np.float32)
        ),
    }
    logits = structure_model._distogram_logits(narrow, params)
    assert logits.dtype == jnp.float32
    assert _bytes(logits) == _bytes(structure_model._distogram_logits(wide, params))


def test_the_diffusion_conditioning_computes_the_same_bits_from_bfloat16_storage() -> (
    None
):
    """`condition_pair` widens inside its own row block, so storage is free here.

    Both row counts, because the blocked and unblocked spellings are different
    programs and only one of them is the one the released sizes take.
    """
    width = 6
    narrow = _narrow_pair(2, 16, width)
    narrow_rel = _narrow_pair(3, 16, width)
    wide, wide_rel = narrow.astype(jnp.float32), narrow_rel.astype(jnp.float32)
    rng = np.random.default_rng(4)

    def array(*shape: int) -> jnp.ndarray:
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32))

    params = {
        "c.z_input_norm.weight": array(2 * width),
        "c.z_input_norm.bias": array(2 * width),
        "c.z_proj.weight": array(width, 2 * width),
    }
    for index in range(2):
        block = f"c.z_transitions.{index}"
        params[f"{block}.norm.weight"] = array(width)
        params[f"{block}.norm.bias"] = array(width)
        params[f"{block}.a_proj.weight"] = array(2 * width, width)
        params[f"{block}.b_proj.weight"] = array(2 * width, width)
        params[f"{block}.out_proj.weight"] = array(width, 2 * width)

    for rows in (None, 4):
        got = diffusion._condition_pair_blocks(
            narrow, narrow_rel, params, "c.", trunk_dtype=jnp.bfloat16, rows=rows
        )
        expected = diffusion._condition_pair_blocks(
            wide, wide_rel, params, "c.", trunk_dtype=jnp.bfloat16, rows=rows
        )
        assert got.dtype == jnp.float32
        assert _bytes(got) == _bytes(expected), f"rows={rows}"


def test_the_entry_norm_computes_the_same_bits_from_bfloat16_storage() -> None:
    """`layer_norm` returns its *input's* width, which is why the head widens.

    The second assertion is the bug this widening exists to prevent: handed
    bfloat16, the same norm rounds its result, and the re-embedding's
    accumulator would start from a rounded first term.
    """
    narrow = _narrow_pair(5, 12, 6)
    wide = narrow.astype(jnp.float32)
    rng = np.random.default_rng(6)
    weight = jnp.asarray(rng.standard_normal(6).astype(np.float32))
    bias = jnp.asarray(rng.standard_normal(6).astype(np.float32))

    widened = layer_norm(narrow.astype(jnp.float32), weight, bias)
    assert widened.dtype == jnp.float32
    assert _bytes(widened) == _bytes(layer_norm(wide, weight, bias))
    assert layer_norm(narrow, weight, bias).dtype == jnp.bfloat16


@pytest.mark.parametrize("confidence_dtype", ["float32", "bfloat16"])
def test_the_confidence_head_computes_the_same_bits_from_bfloat16_storage(
    confidence_dtype: str,
) -> None:
    """Every score the head returns, from bfloat16 storage and from the copy.

    Both widths of the re-embedding option: the float32 default is the path the
    boundary change has to preserve, and the bfloat16 opt-in is the one whose
    `_as` calls now convert rather than pass through.
    """
    params = _params()
    arrays = _inputs()
    narrow = jnp.asarray(arrays["z"]).astype(jnp.bfloat16)
    encoding = _narrow_pair(7, TOKENS, D_PAIR)

    def run(pair, rel):
        return heads.confidence_head(
            params=params,
            prefix="",
            n_layers=TRUNK_LAYERS,
            n_chains=1,
            trunk_dtype=jnp.bfloat16,
            confidence_dtype=jnp.dtype(confidence_dtype),
            relative_position_encoding=rel,
            token_bonds_encoding=None,
            **{**arrays, "z": pair},
        )

    got = run(narrow, encoding)
    expected = run(narrow.astype(jnp.float32), encoding.astype(jnp.float32))
    assert set(got) == set(expected)
    for name in sorted(got):
        assert _bytes(got[name]) == _bytes(expected[name]), name


# --- the count on the real program -----------------------------------------


#: `_released_predict_text`'s own shape: three atoms a token, two structures.
#: Named here because the coordinate array is how the confidence head's
#: sample loop is told apart from every other loop that carries the pair.
_PROBE_ATOMS_PER_TOKEN = 3
_PROBE_SAMPLES = 2


@pytest.mark.slow
def test_the_released_program_carries_no_float32_pair() -> None:
    """The confidence sample loop carries the pair, and carries it narrow.

    The lowered program, so this reads what `predict` asked for rather than
    what one backend's fusion did with it afterwards. Four kinds of loop carry
    a full-width pair here and only one of them is this change's:

    * the recycle scan -- bfloat16 pair plus the block loops' float32
      workspace destination, both before this change and after;
    * the diffusion sampler's step scan -- `condition_pair`'s float32 output,
      which is that stage's own result and not a copy of anything;
    * the four rolled row-block loops inside `folding_trunk` -- one bfloat16
      and one float32 destination apiece;
    * the confidence head's sequential sample loop, which carried the float32
      copy of the trunk pair, the float32 relative position encoding and the
      float32 token-bond encoding. Those three are what this counts to zero.

    The last is identified by the sampled coordinates it maps over rather than
    by its position in the text, and the assertion is on `while` operands
    rather than on converts anywhere: the consumers still convert, which is
    the whole design, so a convert count would only reach zero if the
    arithmetic had changed with the storage.
    """
    from tests.models.esmfold2.test_blocked_trunk_ops import _released_predict_text

    tokens = 8
    text = _released_predict_text(tokens, 4)
    full = rf"tensor<1x{tokens}x{tokens}x256x(\w+)>"
    coords = (
        f"tensor<{_PROBE_SAMPLES}x{tokens * _PROBE_ATOMS_PER_TOKEN}x3xf32>"
    )
    sample_loops = [
        sorted(re.findall(full, line))
        for line in text.splitlines()
        if "stablehlo.while" in line
        and coords in line
        and "bf16" in re.findall(full, line)
    ]
    assert len(sample_loops) == 1, (
        "expected exactly one loop carrying both the sampled coordinates and "
        f"a bfloat16 pair; found {len(sample_loops)}: {sample_loops}"
    )
    assert "f32" not in sample_loops[0], (
        "the confidence sample loop carries a full-width float32 pair again: "
        f"{sample_loops[0]}"
    )
