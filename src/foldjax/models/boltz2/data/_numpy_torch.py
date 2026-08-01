"""A NumPy-backed stand-in for the slice of torch Boltz-2's featurizer uses.

Boltz-2 runs inference in JAX, but its featurizer is vendored from upstream
Boltz and is written against torch. Requiring torch to build features for a
torch-free JAX model is the wrong shape, and it is the only reason the
``boltz-preprocess`` extra existed.

The featurizer touches a small, entirely mechanical slice of torch: array
construction, shape manipulation, reductions, and two ``nn.functional``
helpers. This module implements exactly that slice on NumPy.

Two rules keep it honest:

* **Never guess.** Anything not implemented raises rather than falling back to
  something approximate, so a gap shows up as a loud failure instead of a
  quietly different feature array.
* **Match torch, not NumPy**, wherever they disagree. ``transpose`` swaps two
  axes rather than reversing all of them, ``pad`` takes its pairs from the last
  dimension backwards, and integer tensors default to 64-bit. Those differences
  are where a plausible-looking shim silently changes the science.

``tests/models/boltz2/test_numpy_torch.py`` pins the disagreements, and
``test_featurize_torch_parity.py`` featurizes a real job through both this and
real torch and asserts the arrays are identical.
"""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------


class _DType:
    """A torch-style dtype name bound to its NumPy equivalent."""

    __slots__ = ("name", "numpy")

    def __init__(self, name: str, numpy_dtype: Any) -> None:
        self.name = name
        self.numpy = np.dtype(numpy_dtype)

    @property
    def is_floating_point(self) -> bool:
        return self.numpy.kind == "f"

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _DType):
            return self.numpy == other.numpy
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.numpy)


float32 = _DType("float32", np.float32)
float64 = _DType("float64", np.float64)
float16 = _DType("float16", np.float16)
int32 = _DType("int32", np.int32)
int64 = _DType("int64", np.int64)
uint8 = _DType("uint8", np.uint8)
int8 = _DType("int8", np.int8)
bool_ = _DType("bool", np.bool_)

# torch's aliases
float = float32  # noqa: A001 - mirroring torch.float
double = float64
half = float16
long = int64
int = int32  # noqa: A001 - mirroring torch.int
short = _DType("int16", np.int16)
dtype = _DType

_BOOL = bool_
globals()["bool"] = bool_  # torch.bool, without shadowing the builtin in here

pi = np.pi


def _as_numpy_dtype(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, _DType):
        return value.numpy
    return np.dtype(value)


def _default_dtype(array: np.ndarray) -> np.ndarray:
    """torch defaults python ints to int64 and floats to float32."""
    if array.dtype == np.float64:
        return array.astype(np.float32)
    return array


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------


class device:  # noqa: N801 - mirroring torch.device
    """Everything here is CPU; the type exists so signatures still type-check."""

    __slots__ = ("type",)

    def __init__(self, spec: Any = "cpu") -> None:
        self.type = str(spec).split(":")[0]

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, device) and other.type == self.type

    def __hash__(self) -> builtins.int:
        return hash(self.type)


Device = device


# --------------------------------------------------------------------------
# Tensor
# --------------------------------------------------------------------------


def _unwrap(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.array
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(item) for item in value)
    return value


def _promote(left: Any, right: Any) -> tuple:
    """Apply torch's mixed float/integer promotion rather than NumPy's.

    ``float32 / int64`` is float32 in torch and float64 in NumPy. The Boltz
    featurizer divides float32 coordinates by an integer count, so following
    NumPy here silently widens `coords` and `cyclic_period` to float64 and hands
    the model a different dtype than upstream produces.

    Only the float-meets-integer case differs; float/float and integer/integer
    promotion already agree, so those are left to NumPy.
    """
    if not isinstance(left, (np.ndarray, np.generic)):
        return left, right
    left_array = left
    right_array = right
    if not isinstance(right_array, (np.ndarray, np.generic)):
        # Python scalars already follow torch's "don't widen" rule under NEP 50.
        return left, right
    left_kind, right_kind = left_array.dtype.kind, right_array.dtype.kind
    if left_kind == "f" and right_kind in "biu":
        return left_array, right_array.astype(left_array.dtype)
    if right_kind == "f" and left_kind in "biu":
        return left_array.astype(right_array.dtype), right_array
    return left, right


def _wrap(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return Tensor(value)
    if isinstance(value, np.generic):
        return Tensor(np.asarray(value))
    return value


class Tensor:
    """A thin NumPy-backed tensor covering the featurizer's usage."""

    __slots__ = ("array",)

    def __init__(self, array: Any) -> None:
        self.array = np.asarray(array)

    # -- identity ---------------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def ndim(self) -> builtins.int:
        return self.array.ndim

    @property
    def T(self) -> Tensor:  # noqa: N802 - mirroring torch.Tensor.T
        return Tensor(self.array.T)

    @property
    def mT(self) -> Tensor:  # noqa: N802 - mirroring torch.Tensor.mT
        return Tensor(np.swapaxes(self.array, -1, -2))

    @property
    def dtype(self) -> _DType:
        return _DType(self.array.dtype.name, self.array.dtype)

    @property
    def device(self) -> device:
        return device("cpu")

    @property
    def is_cuda(self) -> builtins.bool:
        return False

    def dim(self) -> builtins.int:
        return self.array.ndim

    def size(self, axis: Any = None) -> Any:
        return self.array.shape if axis is None else self.array.shape[axis]

    def numel(self) -> builtins.int:
        return self.array.size

    def __len__(self) -> builtins.int:
        return len(self.array)

    def __repr__(self) -> str:
        return f"tensor({self.array!r})"

    def __iter__(self):
        return (_wrap(item) for item in self.array)

    def __bool__(self) -> builtins.bool:
        return builtins.bool(self.array)

    def __index__(self) -> builtins.int:
        return builtins.int(self.array)

    def __float__(self) -> builtins.float:
        return builtins.float(self.array)

    def __int__(self) -> builtins.int:
        return builtins.int(self.array)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        if dtype is None:
            return self.array
        return self.array.astype(dtype)

    # -- movement / conversion -------------------------------------------
    def numpy(self) -> np.ndarray:
        return self.array

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def clone(self) -> Tensor:
        return Tensor(self.array.copy())

    def contiguous(self) -> Tensor:
        return Tensor(np.ascontiguousarray(self.array))

    def tolist(self) -> Any:
        return self.array.tolist()

    def item(self) -> Any:
        return self.array.item()

    def to(self, *args: Any, **kwargs: Any) -> Tensor:
        target = kwargs.get("dtype")
        for value in args:
            if isinstance(value, (_DType, np.dtype, type)):
                target = value
        if target is None:
            return self
        return Tensor(self.array.astype(_as_numpy_dtype(target)))

    def type(self, target: Any = None) -> Any:
        if target is None:
            return self.dtype
        return Tensor(self.array.astype(_as_numpy_dtype(target)))

    def astype(self, target: Any) -> Tensor:
        return Tensor(self.array.astype(_as_numpy_dtype(target)))

    def long(self) -> Tensor:
        return Tensor(self.array.astype(np.int64))

    def int(self) -> Tensor:
        return Tensor(self.array.astype(np.int32))

    def float(self) -> Tensor:
        return Tensor(self.array.astype(np.float32))

    def double(self) -> Tensor:
        return Tensor(self.array.astype(np.float64))

    def bool(self) -> Tensor:
        return Tensor(self.array.astype(np.bool_))

    def byte(self) -> Tensor:
        return Tensor(self.array.astype(np.uint8))

    # -- shape ------------------------------------------------------------
    @staticmethod
    def _shape(args: tuple) -> tuple:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return tuple(args)

    def reshape(self, *shape: Any) -> Tensor:
        return Tensor(self.array.reshape(self._shape(shape)))

    def view(self, *shape: Any) -> Tensor:
        return Tensor(self.array.reshape(self._shape(shape)))

    def flatten(
        self, start_dim: builtins.int = 0, end_dim: builtins.int = -1
    ) -> Tensor:
        if start_dim == 0 and end_dim == -1:
            return Tensor(self.array.reshape(-1))
        shape = self.array.shape
        end = end_dim % self.array.ndim
        merged = builtins.int(np.prod(shape[start_dim : end + 1]))
        return Tensor(self.array.reshape(*shape[:start_dim], merged, *shape[end + 1 :]))

    def unsqueeze(self, axis: builtins.int) -> Tensor:
        return Tensor(np.expand_dims(self.array, axis))

    def squeeze(self, axis: Any = None) -> Tensor:
        if axis is None:
            return Tensor(self.array.squeeze())
        if self.array.shape[axis] != 1:
            return self  # torch leaves non-unit dims alone
        return Tensor(self.array.squeeze(axis))

    def transpose(self, dim0: builtins.int, dim1: builtins.int) -> Tensor:
        # torch swaps two axes; np.transpose would reverse all of them.
        return Tensor(np.swapaxes(self.array, dim0, dim1))

    def permute(self, *dims: Any) -> Tensor:
        return Tensor(np.transpose(self.array, self._shape(dims)))

    def repeat(self, *reps: Any) -> Tensor:
        return Tensor(np.tile(self.array, self._shape(reps)))

    def expand(self, *shape: Any) -> Tensor:
        target = [
            current if size == -1 else size
            for size, current in zip(
                self._shape(shape),
                (1,) * (len(self._shape(shape)) - self.array.ndim) + self.array.shape,
            )
        ]
        return Tensor(np.broadcast_to(self.array, tuple(target)))

    def unbind(self, axis: builtins.int = 0) -> tuple:
        return tuple(Tensor(part) for part in np.moveaxis(self.array, axis, 0))

    # -- reductions and maths ---------------------------------------------
    @staticmethod
    def _axis(dim: Any, kwargs: dict) -> Any:
        """torch accepts `axis` as an alias for `dim`; the featurizer uses both."""
        return kwargs.pop("axis", dim)

    def sum(self, dim: Any = None, keepdim: builtins.bool = False, **kw: Any) -> Tensor:
        return Tensor(self.array.sum(axis=self._axis(dim, kw), keepdims=keepdim))

    def mean(
        self, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
    ) -> Tensor:
        return Tensor(self.array.mean(axis=self._axis(dim, kw), keepdims=keepdim))

    def all(self, dim: Any = None, keepdim: builtins.bool = False, **kw: Any) -> Tensor:
        return Tensor(self.array.all(axis=self._axis(dim, kw), keepdims=keepdim))

    def any(self, dim: Any = None, keepdim: builtins.bool = False, **kw: Any) -> Tensor:
        return Tensor(self.array.any(axis=self._axis(dim, kw), keepdims=keepdim))

    def prod(
        self, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
    ) -> Tensor:
        return Tensor(self.array.prod(axis=self._axis(dim, kw), keepdims=keepdim))

    def cumsum(self, dim: Any = None, **kw: Any) -> Tensor:
        return Tensor(self.array.cumsum(axis=self._axis(dim, kw)))

    def max(self, dim: Any = None, keepdim: builtins.bool = False) -> Any:
        if dim is None:
            return Tensor(self.array.max())
        values = self.array.max(axis=dim, keepdims=keepdim)
        indices = self.array.argmax(axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, dim)
        return Tensor(values), Tensor(indices)

    def min(self, dim: Any = None, keepdim: builtins.bool = False) -> Any:
        if dim is None:
            return Tensor(self.array.min())
        values = self.array.min(axis=dim, keepdims=keepdim)
        indices = self.array.argmin(axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, dim)
        return Tensor(values), Tensor(indices)

    def argsort(
        self, dim: builtins.int = -1, descending: builtins.bool = False
    ) -> Tensor:
        order = np.argsort(self.array, axis=dim, kind="stable")
        if descending:
            order = np.flip(order, axis=dim)
        return Tensor(order)

    def sort(self, dim: builtins.int = -1, descending: builtins.bool = False) -> Any:
        order = self.argsort(dim=dim, descending=descending)
        return Tensor(np.take_along_axis(self.array, order.array, axis=dim)), order

    def nonzero(self, as_tuple: builtins.bool = False) -> Any:
        found = np.nonzero(self.array)
        if as_tuple:
            return tuple(Tensor(part) for part in found)
        return Tensor(
            np.stack(found, axis=-1) if found else np.empty((0, self.array.ndim))
        )

    def abs(self) -> Tensor:
        return Tensor(np.abs(self.array))

    def atan(self) -> Tensor:
        return Tensor(np.arctan(self.array))

    def isnan(self) -> Tensor:
        return Tensor(np.isnan(self.array))

    def add(self, other: Any, alpha: Any = 1) -> Tensor:
        return _wrap(self.array + _unwrap(alpha) * _unwrap(other))

    def argmax(self, dim: Any = None, keepdim: builtins.bool = False) -> Tensor:
        return Tensor(np.argmax(self.array, axis=dim, keepdims=keepdim))

    def argmin(self, dim: Any = None, keepdim: builtins.bool = False) -> Tensor:
        return Tensor(np.argmin(self.array, axis=dim, keepdims=keepdim))

    def unique(self, return_counts: builtins.bool = False) -> Any:
        return unique(self.array, return_counts=return_counts)

    def sqrt(self) -> Tensor:
        return Tensor(np.sqrt(self.array))

    def exp(self) -> Tensor:
        return Tensor(np.exp(self.array))

    def log(self) -> Tensor:
        return Tensor(np.log(self.array))

    def clamp(self, min: Any = None, max: Any = None) -> Tensor:  # noqa: A002
        return Tensor(np.clip(self.array, _unwrap(min), _unwrap(max)))

    def norm(
        self, p: Any = 2, dim: Any = None, keepdim: builtins.bool = False
    ) -> Tensor:
        return Tensor(np.linalg.norm(self.array, ord=p, axis=dim, keepdims=keepdim))

    def masked_fill(self, mask: Any, value: Any) -> Tensor:
        out = self.array.copy()
        out[_unwrap(mask)] = _unwrap(value)
        return Tensor(out)

    def masked_fill_(self, mask: Any, value: Any) -> Tensor:
        self.array[_unwrap(mask)] = _unwrap(value)
        return self

    def fill_(self, value: Any) -> Tensor:
        self.array.fill(_unwrap(value))
        return self

    def new_zeros(self, *shape: Any) -> Tensor:
        return Tensor(np.zeros(self._shape(shape), dtype=self.array.dtype))

    def new_ones(self, *shape: Any) -> Tensor:
        return Tensor(np.ones(self._shape(shape), dtype=self.array.dtype))

    def new_full(self, shape: Any, value: Any) -> Tensor:
        return Tensor(np.full(tuple(shape), _unwrap(value), dtype=self.array.dtype))

    # -- indexing ---------------------------------------------------------
    def __getitem__(self, key: Any) -> Tensor:
        return _wrap(self.array[_unwrap(key)])

    def __setitem__(self, key: Any, value: Any) -> None:
        self.array[_unwrap(key)] = _unwrap(value)

    # -- operators --------------------------------------------------------
    def _binary(self, other: Any, op: Any) -> Tensor:
        left, right = _promote(self.array, _unwrap(other))
        return _wrap(op(left, right))

    def __add__(self, other: Any) -> Tensor:
        return self._binary(other, np.add)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> Tensor:
        return _wrap(np.subtract(_unwrap(other), self.array))

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(other, np.true_divide)

    def __rtruediv__(self, other: Any) -> Tensor:
        return _wrap(np.true_divide(_unwrap(other), self.array))

    def __floordiv__(self, other: Any) -> Tensor:
        return self._binary(other, np.floor_divide)

    def __mod__(self, other: Any) -> Tensor:
        return self._binary(other, np.mod)

    def __pow__(self, other: Any) -> Tensor:
        return self._binary(other, np.power)

    def __matmul__(self, other: Any) -> Tensor:
        return self._binary(other, np.matmul)

    def __neg__(self) -> Tensor:
        return Tensor(-self.array)

    def __invert__(self) -> Tensor:
        return Tensor(~self.array)

    def __and__(self, other: Any) -> Tensor:
        return self._binary(other, np.logical_and)

    __rand__ = __and__

    def __or__(self, other: Any) -> Tensor:
        return self._binary(other, np.logical_or)

    __ror__ = __or__

    def __eq__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary(other, np.equal)

    def __ne__(self, other: Any) -> Tensor:  # type: ignore[override]
        return self._binary(other, np.not_equal)

    def __lt__(self, other: Any) -> Tensor:
        return self._binary(other, np.less)

    def __le__(self, other: Any) -> Tensor:
        return self._binary(other, np.less_equal)

    def __gt__(self, other: Any) -> Tensor:
        return self._binary(other, np.greater)

    def __ge__(self, other: Any) -> Tensor:
        return self._binary(other, np.greater_equal)

    def __iadd__(self, other: Any) -> Tensor:
        self.array = self.array + _unwrap(other)
        return self

    def __hash__(self) -> builtins.int:
        return id(self)


# --------------------------------------------------------------------------
# constructors and free functions
# --------------------------------------------------------------------------


def tensor(data: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    array = np.asarray(_unwrap(data), dtype=_as_numpy_dtype(dtype))
    return Tensor(array if dtype is not None else _default_dtype(array))


def as_tensor(data: Any, dtype: Any = None, device: Any = None) -> Tensor:
    return tensor(data, dtype=dtype)


def from_numpy(array: np.ndarray) -> Tensor:
    return Tensor(np.asarray(array))


def is_tensor(value: Any) -> builtins.bool:
    return isinstance(value, Tensor)


def zeros(*shape: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    return Tensor(
        np.zeros(Tensor._shape(shape), dtype=_as_numpy_dtype(dtype) or np.float32)
    )


def ones(*shape: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    return Tensor(
        np.ones(Tensor._shape(shape), dtype=_as_numpy_dtype(dtype) or np.float32)
    )


def empty(*shape: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    return Tensor(
        np.empty(Tensor._shape(shape), dtype=_as_numpy_dtype(dtype) or np.float32)
    )


def full(shape: Any, value: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    return Tensor(
        np.full(tuple(shape), _unwrap(value), dtype=_as_numpy_dtype(dtype))
        if dtype is not None
        else _default_dtype(np.full(tuple(shape), _unwrap(value)))
    )


def zeros_like(other: Any, dtype: Any = None, device: Any = None) -> Tensor:
    return Tensor(np.zeros_like(_unwrap(other), dtype=_as_numpy_dtype(dtype)))


def ones_like(other: Any, dtype: Any = None, device: Any = None) -> Tensor:
    return Tensor(np.ones_like(_unwrap(other), dtype=_as_numpy_dtype(dtype)))


def eye(n: Any, m: Any = None, dtype: Any = None, device: Any = None) -> Tensor:
    return Tensor(np.eye(n, m, dtype=_as_numpy_dtype(dtype) or np.float32))


def arange(*args: Any, dtype: Any = None, device: Any = None) -> Tensor:  # noqa: A002
    array = np.arange(*[_unwrap(value) for value in args], dtype=_as_numpy_dtype(dtype))
    return Tensor(array if dtype is not None else _default_dtype(array))


def linspace(
    start: Any, end: Any, steps: Any, dtype: Any = None, device: Any = None
) -> Tensor:
    return Tensor(
        np.linspace(_unwrap(start), _unwrap(end), _unwrap(steps)).astype(
            _as_numpy_dtype(dtype) or np.float32
        )
    )


def cat(parts: Any, dim: builtins.int = 0) -> Tensor:
    return Tensor(np.concatenate([_unwrap(part) for part in parts], axis=dim))


concat = cat


def stack(parts: Any, dim: builtins.int = 0) -> Tensor:
    return Tensor(np.stack([_unwrap(part) for part in parts], axis=dim))


def unbind(value: Any, dim: builtins.int = 0) -> tuple:
    return Tensor(_unwrap(value)).unbind(dim)


def where(condition: Any, x: Any = None, y: Any = None) -> Any:
    if x is None and y is None:
        return tuple(Tensor(part) for part in np.nonzero(_unwrap(condition)))
    return _wrap(np.where(_unwrap(condition), _unwrap(x), _unwrap(y)))


def sum(
    value: Any, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
) -> Tensor:  # noqa: A001
    return Tensor(np.sum(_unwrap(value), axis=kw.pop("axis", dim), keepdims=keepdim))


def mean(
    value: Any, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
) -> Tensor:
    return Tensor(np.mean(_unwrap(value), axis=kw.pop("axis", dim), keepdims=keepdim))


def all(
    value: Any, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
) -> Tensor:  # noqa: A001
    return Tensor(np.all(_unwrap(value), axis=kw.pop("axis", dim), keepdims=keepdim))


def any(
    value: Any, dim: Any = None, keepdim: builtins.bool = False, **kw: Any
) -> Tensor:  # noqa: A001
    return Tensor(np.any(_unwrap(value), axis=kw.pop("axis", dim), keepdims=keepdim))


def max(a: Any, b: Any = None, dim: Any = None) -> Any:  # noqa: A001
    if b is not None:
        return _wrap(np.maximum(*_promote(_unwrap(a), _unwrap(b))))
    return Tensor(_unwrap(a)).max(dim=dim)


def maximum(a: Any, b: Any) -> Tensor:
    return _wrap(np.maximum(*_promote(_unwrap(a), _unwrap(b))))


def minimum(a: Any, b: Any) -> Tensor:
    return _wrap(np.minimum(*_promote(_unwrap(a), _unwrap(b))))


def sqrt(value: Any) -> Tensor:
    return _wrap(np.sqrt(_unwrap(value)))


def abs(value: Any) -> Tensor:  # noqa: A001 - mirroring torch.abs
    return _wrap(np.abs(_unwrap(value)))


def atan(value: Any) -> Tensor:
    return _wrap(np.arctan(_unwrap(value)))


def exp(value: Any) -> Tensor:
    return _wrap(np.exp(_unwrap(value)))


def log(value: Any) -> Tensor:
    return _wrap(np.log(_unwrap(value)))


def isnan(value: Any) -> Tensor:
    return _wrap(np.isnan(_unwrap(value)))


def unique(
    value: Any, return_counts: builtins.bool = False, sorted: builtins.bool = True
) -> Any:  # noqa: A002
    found = np.unique(_unwrap(value), return_counts=return_counts)
    if return_counts:
        return Tensor(found[0]), Tensor(found[1])
    return Tensor(found)


def argmax(value: Any, dim: Any = None) -> Tensor:
    return Tensor(np.argmax(_unwrap(value), axis=dim))


def argmin(value: Any, dim: Any = None) -> Tensor:
    return Tensor(np.argmin(_unwrap(value), axis=dim))


def einsum(subscripts: str, *operands: Any) -> Tensor:
    return Tensor(np.einsum(subscripts, *[_unwrap(item) for item in operands]))


def cdist(a: Any, b: Any, p: builtins.float = 2.0) -> Tensor:
    """Pairwise distances over the last dimension, batched over the rest."""
    left, right = np.asarray(_unwrap(a)), np.asarray(_unwrap(b))
    difference = left[..., :, None, :] - right[..., None, :, :]
    if p == 2.0:
        return Tensor(np.sqrt(np.square(difference).sum(-1)))
    return Tensor(np.linalg.norm(difference, ord=p, axis=-1))


def cartesian_prod(*arrays: Any) -> Tensor:
    grids = np.meshgrid(*[np.asarray(_unwrap(item)) for item in arrays], indexing="ij")
    return Tensor(np.stack([grid.reshape(-1) for grid in grids], axis=-1))


def randn(
    *shape: Any, dtype: Any = None, device: Any = None, generator: Any = None
) -> Tensor:
    return Tensor(
        np.random.default_rng()
        .standard_normal(Tensor._shape(shape))
        .astype(_as_numpy_dtype(dtype) or np.float32)
    )


def randn_like(other: Any, dtype: Any = None, device: Any = None) -> Tensor:
    reference = np.asarray(_unwrap(other))
    return Tensor(
        np.random.default_rng()
        .standard_normal(reference.shape)
        .astype(_as_numpy_dtype(dtype) or reference.dtype)
    )


# --------------------------------------------------------------------------
# torch.nn.functional
# --------------------------------------------------------------------------


def _one_hot(value: Any, num_classes: builtins.int = -1) -> Tensor:
    indices = np.asarray(_unwrap(value))
    if num_classes == -1:
        num_classes = builtins.int(indices.max()) + 1
    out = np.zeros((*indices.shape, num_classes), dtype=np.int64)
    np.put_along_axis(out, indices[..., None].astype(np.int64), 1, axis=-1)
    return Tensor(out)


def _pad(
    input: Any,  # noqa: A002 - torch names it `input`
    pad: Any,
    mode: str = "constant",
    value: Any = 0,
) -> Tensor:
    """torch.nn.functional.pad.

    torch reads ``pad`` as (last_dim_before, last_dim_after, second_last_before,
    ...) — from the *last* dimension backwards. NumPy reads its width list from
    the first dimension forwards. Getting this backwards pads the wrong axis and
    still returns a plausible array, so it is spelled out here.
    """
    if mode != "constant":
        raise NotImplementedError(f"pad mode {mode!r} is not implemented")
    array = np.asarray(_unwrap(input))
    pairs = [(0, 0)] * array.ndim
    spec = [builtins.int(item) for item in _unwrap(pad)]
    for index in range(len(spec) // 2):
        pairs[array.ndim - 1 - index] = (spec[2 * index], spec[2 * index + 1])
    fill = 0 if value is None else _unwrap(value)
    return Tensor(np.pad(array, pairs, mode="constant", constant_values=fill))


def _log_softmax(value: Any, dim: builtins.int = -1) -> Tensor:
    array = np.asarray(_unwrap(value), dtype=np.float32)
    shifted = array - array.max(axis=dim, keepdims=True)
    return Tensor(shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True)))


class _Functional:
    one_hot = staticmethod(_one_hot)
    pad = staticmethod(_pad)
    log_softmax = staticmethod(_log_softmax)


class _NN:
    functional = _Functional()


nn = _NN()


# --------------------------------------------------------------------------
# torch.utils.data
# --------------------------------------------------------------------------


class Dataset:
    """Base class only; FoldJAX indexes datasets directly, never a DataLoader."""


class _UtilsData:
    Dataset = Dataset


class _Utils:
    data = _UtilsData()


utils = _Utils()


class _Types:
    Device = Device


types = _Types()
