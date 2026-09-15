"""Shared shape-profile selection for optional JAX input padding.

The public contract deliberately describes semantic axes rather than array
positions.  Backends remain responsible for schema-aware padding, masks, RNG
and output cropping; this module only chooses and reports conservative target
sizes in one consistent way.

Two policies live here rather than in any one port.  The first is the token
grid, whose rule is a bound on waste rather than a list of sizes: it steps by
256 from 256 to 8,192, so a bucket costs at most one step of padded work --
256 tokens, <=12.5% at 2k -- over the exact shape.  A geometric grid has no
such bound, and the gap it left between 2,048 and 3,072 was the whole cost of
padding: a 2,096-token Protenix job landed on 3,072 and paid +97% wall and
+44% peak against its exact shape for padding nobody asked for.  The price of
the constant step is more executables to bake, 32 per model rather than 11,
which ``cache warm`` amortises -- a bucket is baked once and then hit by every
job in its 256-token band, which is what makes padding the normal way to run
rather than a shape-normalising option.

The grid ends at 8,192 rather than at what one card folds because context
parallelism exists to run the targets that do not fit one card, and every
derived grid reaches that token ceiling's own derivation (24x for atoms, 2x
for structural tokens) so no axis can refuse a size the token axis accepts.
Above the last bucket ``overflow='error'`` still refuses rather than compiling
an unplanned shape.

The second policy is mesh alignment.  A distributed diffusion atom graph needs
the padded atom count to divide ``32 * rows`` and the padded token count to
divide ``rows``; an automatic target that misses those multiples costs a
replicated atom graph on every device for the sake of a few hundred padded
atoms.  Automatic targets are therefore rounded up to the mesh's multiples
when a request carries more than one context-parallel device.  Explicit pins
are left exactly as written -- a pin is a statement about the compiled shape,
and a misaligned one keeps its existing outcome of a warning and a replicated
atom graph rather than silently becoming a different shape.

Alignment is the one thing that can move an automatic target off the grid, and
it mostly does not have to: a bucket ``256 * k`` already divides every
power-of-two row count, and its derived atom target ``6,144 * k`` divides
``32 * rows`` for every ``rows`` that divides 192, so two, three, four, six
and eight rows leave a bucket's own atom target alone.  A row count with an odd
factor the bucket index lacks does round the token target past its bucket, by
at most ``rows - 1`` tokens: three rows take 2,048 to 2,049 but leave 2,304
(``256 * 9``) alone.  The one-step bound above is therefore a statement about
the bucket, not about every mesh width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

from foldjax.schema import PaddingConfig

MSA_PROFILE_DEPTH = 1024
OPENDDE_MSA_PROFILE_DEPTH = 1280

#: A constant 256-token step to 8,192, so no job pays more than one step of
#: padded work; the docstring explains why the step beats a geometric grid.
TOKEN_BUCKETS = tuple(range(256, 8192 + 1, 256))
ATOM_BUCKETS = (
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
)
MSA_BUCKETS = (1, 64, 128, 256, 512, 768, 1024, 1280, 2048, 4096, 8192, 16384)
TEMPLATE_BUCKETS = (1, 2, 4)
STRUCTURAL_TOKEN_BUCKETS = (
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
)
LANGUAGE_MODEL_TOKEN_BUCKETS = (
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    5120,
    6144,
    8192,
)

DEFAULT_BUCKETS: dict[str, tuple[int, ...]] = {
    "tokens": TOKEN_BUCKETS,
    "atoms": ATOM_BUCKETS,
    "msa": MSA_BUCKETS,
    "templates": TEMPLATE_BUCKETS,
    "structural_tokens": STRUCTURAL_TOKEN_BUCKETS,
    "language_model_tokens": LANGUAGE_MODEL_TOKEN_BUCKETS,
}


#: The query window of every port's atom attention, and therefore the number
#: of atoms one context-parallel row has to own a whole multiple of.  Boltz-2,
#: Protenix, OpenDDE and OpenFold3 all window atoms in 32s.
CP_ATOM_WINDOW = 32

#: The multiple one context-parallel mesh row imposes on each automatic axis.
#: Only the axes the distributed atom graph reads are listed: the MSA, template
#: and language-model axes are never split across the mesh, so rounding them up
#: would buy nothing and cost memory.  ``structural_tokens`` is here because
#: OpenDDE diffuses over its expanded structural tokens, whose axis -- not
#: ``tokens`` -- is the one its mesh has to divide.
_CP_AXIS_UNIT: dict[str, int] = {
    "tokens": 1,
    "atoms": CP_ATOM_WINDOW,
    "structural_tokens": 1,
}


@dataclass(frozen=True, slots=True)
class _CPAlignedPadding(PaddingConfig):
    """A padding profile that also knows how wide the mesh it feeds is.

    A private subclass rather than a public field, because the row count is
    derived from ``cp_devices`` rather than requested: it must not become a
    second, silently divergent way to ask for a mesh, and it has no place in
    the recorded request summary ``PaddingConfig`` serialises.  Every resolver
    reads it through :func:`cp_rows`, so a plain ``PaddingConfig`` keeps
    behaving exactly as it did.
    """

    cp_rows: int = 1


def square_grid_auto_layout(cp_devices: int) -> str:
    """Which layout ``cp_layout="auto"`` names on a port that picks the grid.

    ``"2d"`` on a perfect-square device count greater than one (4, 9, 16, ...)
    and ``"1d"`` on every other count, a square being the only shape the ring
    schedules accept.  OpenDDE and Boltz-2 resolve ``auto`` through this;
    Protenix and OpenFold3 keep the shared one-dimensional default.  Which
    ports read it, and the per-device measurements that decided each of those
    answers, are recorded beside the ports' own resolvers and in
    ``docs/context_parallel.md`` -- not here, so that a remeasurement moves
    the prose it belongs to.

    The rule lives in this module rather than beside the meshes because an
    adapter resolves it while a request is still being planned on the host,
    and that path must not import JAX.
    """

    if isinstance(cp_devices, bool) or not isinstance(cp_devices, int):
        raise ValueError("context-parallel device count must be an integer")
    if cp_devices < 1:
        raise ValueError("context-parallel device count must be positive")
    side = math.isqrt(cp_devices)
    return "2d" if cp_devices > 1 and side * side == cp_devices else "1d"


def cp_mesh_rows(cp_devices: int, cp_layout: str = "auto") -> int:
    """Rows of the context-parallel mesh a request would build.

    This mirrors ``foldjax.models._cp.resolve_cp_layout`` and ``cp_grid``
    deliberately instead of importing them: padding is resolved while a request
    is still being planned on the host, and that path must not import JAX.  The
    duplication is a handful of integer rules, and a test pins it against the
    real resolver.

    ``cp_layout`` is the layout the mesh will actually be built with, so
    ``auto`` here means the shared one-dimensional default.  A port whose
    ``auto`` picks the square grid resolves it -- through
    :func:`square_grid_auto_layout` -- before asking for an alignment, so that
    an omitted layout and an explicit ``2d`` pad to the same shapes on the same
    device count.
    """

    if isinstance(cp_devices, bool) or not isinstance(cp_devices, int):
        raise ValueError("context-parallel device count must be an integer")
    if cp_devices < 1:
        raise ValueError("context-parallel device count must be positive")
    if cp_layout not in {"auto", "1d", "2d"}:
        raise ValueError(
            f"context-parallel layout must be 'auto', '1d', or '2d'; got {cp_layout!r}"
        )
    if cp_layout != "2d":
        # One row per device, which is what `auto` still means for every port
        # that has not measured the grid (`square_grid_auto_layout`).
        return cp_devices
    side = math.isqrt(cp_devices)
    if cp_devices <= 1 or side * side != cp_devices:
        raise ValueError(
            "the two-dimensional layout needs a perfect-square device count "
            f"greater than one; got {cp_devices}"
        )
    return side


def cp_aligned_padding(
    config: PaddingConfig,
    *,
    cp_devices: int,
    cp_layout: str = "auto",
) -> PaddingConfig:
    """Return ``config`` with the mesh width its automatic targets must respect.

    A serial request gets its own object back, so a one-device run resolves
    byte-identically to a run made before this existed.
    """

    # A serial request has no mesh to divide, so it does not consult the
    # layout -- not even to reject one a mesh would refuse, which each port
    # validates for itself. The device count is still checked.
    rows = cp_mesh_rows(cp_devices, "auto" if cp_devices == 1 else cp_layout)
    if rows <= 1:
        return config
    return _CPAlignedPadding(
        **{field.name: getattr(config, field.name) for field in fields(PaddingConfig)},
        cp_rows=rows,
    )


def cp_rows(config: PaddingConfig) -> int:
    """Mesh rows carried by ``config``; ``1`` for any ordinary profile."""

    rows = getattr(config, "cp_rows", 1)
    return rows if isinstance(rows, int) and rows > 1 else 1


def _cp_aligned_target(target: int, config: PaddingConfig, axis: str) -> int:
    """Round one automatic target up to what the mesh can divide."""

    multiple = _CP_AXIS_UNIT.get(axis, 0) * cp_rows(config)
    if multiple <= 1:
        return target
    return ((target + multiple - 1) // multiple) * multiple


def resolve_axis(
    actual: int,
    config: PaddingConfig,
    axis: str,
    *,
    buckets: tuple[int, ...] | None = None,
    minimum: int | None = None,
) -> int:
    """Resolve one semantic axis to an exact target no smaller than input.

    ``minimum`` is the already materialized array size for an archive that may
    itself contain padding.  Such an array is never shrunk even if its real mask
    count is smaller.
    """

    if actual < 0:
        raise ValueError(f"actual {axis} must be non-negative")
    floor = max(actual, actual if minimum is None else minimum)
    requested = getattr(config, axis)
    if requested is not None:
        if requested < floor:
            raise ValueError(
                f"padding.{axis}={requested} is smaller than the input size {floor}"
            )
        return requested
    candidates = DEFAULT_BUCKETS[axis] if buckets is None else buckets
    target = next((candidate for candidate in candidates if candidate >= floor), None)
    if target is not None:
        return _cp_aligned_target(target, config, axis)
    if config.overflow == "error":
        largest = candidates[-1] if candidates else 0
        raise ValueError(
            f"input {axis} size {floor} exceeds the largest standard bucket "
            f"{largest}; pin padding.{axis} or use overflow='exact'"
        )
    return _cp_aligned_target(floor, config, axis)


def resolve_msa_axis(
    actual: int,
    config: PaddingConfig,
    *,
    minimum: int | None = None,
    profile_depth: int = MSA_PROFILE_DEPTH,
    input_depth: int | None = None,
    buckets: tuple[int, ...] = MSA_BUCKETS,
) -> int:
    """Pad the MSA axis up, and never below the rows the model would read.

    The MSA axis is resolved like the token and atom axes: the target is the
    smallest bucket that holds every stored row.  ``profile_depth`` is the
    profile's preferred floor -- a padded run pads *up* to at least that depth
    so shallow alignments in one token band still share an executable -- and
    it can never shorten a deeper alignment.  ``input_depth`` is the active
    native depth control (``--max-msa-depth``), which has already run when the
    rows counted here were materialised; it bounds the padded capacity so a
    caller who asked for fewer rows does not get a wider masked axis, and a
    storage deeper than it is a contradiction rather than something to crop.

    ``padding.msa`` is a target, not a cap: a pin below the stored rows is
    refused instead of truncating the input, because which rows a model reads
    is a scientific choice and ``--max-msa-depth`` is the option that makes it.
    """

    if actual < 0:
        raise ValueError("actual msa must be non-negative")
    if profile_depth < 1:
        raise ValueError("profile_depth must be positive")
    floor = max(actual, 0 if minimum is None else minimum)
    if input_depth is not None:
        if input_depth < 1:
            raise ValueError("input_depth must be positive")
        if floor > input_depth:
            raise ValueError(
                f"the stored MSA depth {floor} is deeper than "
                f"max_msa_depth={input_depth}, which selects rows before "
                "padding resolves its capacity"
            )
    requested = config.msa
    if requested is not None:
        if requested < floor:
            raise ValueError(
                f"padding.msa={requested} is smaller than the {floor} stored "
                "MSA rows; padding pads this axis and never truncates it, so "
                "use --max-msa-depth to read fewer rows"
            )
        return requested
    preferred = (
        profile_depth if input_depth is None else min(profile_depth, input_depth)
    )
    start = max(floor, preferred)
    target = next((candidate for candidate in buckets if candidate >= start), None)
    if target is None:
        if config.overflow == "error":
            raise ValueError(
                f"input msa size {floor} exceeds the largest standard bucket "
                f"{buckets[-1] if buckets else 0}; pin padding.msa or use "
                "overflow='exact'"
            )
        target = floor
    # The mesh never splits this axis, so no alignment rounding here.
    return target if input_depth is None else max(floor, min(target, input_depth))


def resolve_token_axis(
    actual: int,
    config: PaddingConfig,
    axis: str,
    *,
    token_target: int,
    minimum: int | None = None,
    fixed_size: int | None = None,
) -> int:
    """Tie automatic capacity to the token profile without truncating storage.

    Explicit pins retain their meaning. Model-specific MSA limits and LM
    special-token overhead belong to callers, not to a shared bucket grid.
    """
    if token_target < 1:
        raise ValueError("token_target must be positive")
    if getattr(config, axis) is None:
        target = fixed_size
        if target is None:
            if axis == "atoms":
                target = ((24 * token_target + 31) // 32) * 32
            elif axis == "structural_tokens":
                target = 2 * token_target
            elif axis == "language_model_tokens":
                target = token_target
            else:
                raise ValueError(f"fixed_size is required for {axis}")
        # Aligned before the pin, not after: below this line the axis is
        # indistinguishable from a caller's explicit pin, which is exactly what
        # alignment must leave alone.
        config = replace(config, **{axis: _cp_aligned_target(target, config, axis)})
    return resolve_axis(actual, config, axis, minimum=minimum)


@dataclass(frozen=True, slots=True)
class PaddingPlan:
    """Resolved real/storage/target dimensions for one backend call."""

    actual: dict[str, int]
    target: dict[str, int]
    storage: dict[str, int] | None = None

    @property
    def changed(self) -> bool:
        source = self.storage or self.actual
        return any(self.target.get(axis) != size for axis, size in source.items())

    def summary(self) -> dict[str, Any]:
        return {
            "actual": dict(self.actual),
            "storage": dict(self.storage or self.actual),
            "target": dict(self.target),
            "changed": self.changed,
        }

    def message(self, model: str) -> str:
        source = self.storage or self.actual
        dimensions = ", ".join(
            f"{axis} {self.actual.get(axis, source[axis])}"
            + (
                f" (stored {source[axis]})"
                if source[axis] != self.actual.get(axis, source[axis])
                else ""
            )
            + f" -> {self.target[axis]}"
            for axis in self.target
        )
        return f"{model} padding profile: {dimensions}"
