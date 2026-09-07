"""CPU observer mechanics; deliberately no claim about CUDA TF32 numerics."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.opendde_attention_scaling import observe_native_matmuls


class Tensor(np.ndarray):
    def stride(self):
        return tuple(value // self.itemsize for value in self.strides)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def fake_torch():
    flags = SimpleNamespace(allow_tf32=True)
    calls = []

    def matmul(a, b):
        calls.append((flags.allow_tf32, a, b))
        return (a @ b + (0.125 if flags.allow_tf32 else 0)).view(Tensor)

    return SimpleNamespace(
        matmul=matmul, backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flags))
    ), calls


def test_exact_operands_flag_restoration_actual_return_and_prebias_snapshot():
    torch, calls = fake_torch()

    def attention(q, k, v):
        scores = torch.matmul(q, k.T)
        scores += 3
        return torch.matmul(scores, v)

    q = np.eye(2, dtype=np.float32).view(Tensor)
    original = torch.matmul
    ordinary = attention(q, q, q).copy()
    calls.clear()
    with observe_native_matmuls(torch, attention) as (records, outputs):
        result = attention(q, q, q)
    np.testing.assert_array_equal(result, ordinary)
    assert [flag for flag, _, _ in calls] == [True, False, True, False]
    for first, second in ((0, 1), (2, 3)):
        assert calls[first][1] is calls[second][1]
        assert calls[first][2] is calls[second][2]
    assert torch.matmul is original
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert [r["call"] for r in records] == ["qk", "pv"]
    assert all(r["max_abs"] == 0.125 and not r["bitwise_equal"] for r in records)
    assert records[0]["right_stride"] == [1, 2]
    np.testing.assert_array_equal(outputs["qk_tf32"], q + 0.125)


def test_non_target_calls_not_observed_and_missing_calls_fail():
    torch, calls = fake_torch()

    def attention():
        return None

    q = np.eye(2, dtype=np.float32).view(Tensor)
    original = torch.matmul
    with pytest.raises(ValueError, match="missing"):
        with observe_native_matmuls(torch, attention) as (records, _):
            torch.matmul(q, q)
            attention()
    assert not records and len(calls) == 1
    assert torch.matmul is original


def test_reference_failure_restores_flag_and_patch():
    torch, _ = fake_torch()

    def original(a, b):
        if not torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("reference failure")
        return a

    torch.matmul = original

    def attention(q):
        return torch.matmul(q, q)

    with pytest.raises(RuntimeError, match="reference failure"):
        with observe_native_matmuls(torch, attention):
            attention(np.eye(2, dtype=np.float32).view(Tensor))
    assert torch.matmul is original
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_disabled_actual_flag_rejected():
    torch, _ = fake_torch()
    torch.backends.cuda.matmul.allow_tf32 = False

    def attention(q):
        return torch.matmul(q, q)

    with pytest.raises(ValueError, match="TF32"):
        with observe_native_matmuls(torch, attention):
            attention(np.eye(2, dtype=np.float32).view(Tensor))
    assert torch.backends.cuda.matmul.allow_tf32 is False
