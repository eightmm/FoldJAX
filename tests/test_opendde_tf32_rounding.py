import jax.numpy as jnp
import numpy as np
import pytest

from bench.opendde_tf32_rounding import tf32_round


@pytest.mark.parametrize(
    "mode,word", [("rne", 0x3F800000), ("rna", 0x3F802000), ("rtz", 0x3F800000)]
)
def test_tf32_ties_follow_the_declared_rule_in_both_signs(mode, word):
    value = np.asarray([0x3F801000, 0xBF801000], np.uint32).view(np.float32)
    result = np.asarray(tf32_round(jnp.asarray(value), mode)).view(np.uint32)
    np.testing.assert_array_equal(result, [word, word | 0x80000000])


def test_tf32_probe_preserves_nonfinite_payloads_and_signed_zero():
    bits = np.asarray([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345], np.uint32)
    result = np.asarray(tf32_round(jnp.asarray(bits.view(np.float32)), "rna"))
    np.testing.assert_array_equal(result.view(np.uint32), bits)
