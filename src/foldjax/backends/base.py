"""Backend interface."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from foldjax import execution, memory_policy
from foldjax.padding import square_grid_auto_layout
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    _strict_boolean,
    _strict_integer,
)

#: The `matmul_precision` entry every backend's `execution_options` carries.
#:
#: It is one shared object rather than six copies because, unlike the other
#: neutral knobs, this one is not a rename: every port implements it the same
#: way, by reading `execution.resolved_matmul_precision` where it used to name
#: its own constant. Six identical literals would be six chances to spell the
#: value vocabulary differently.
MATMUL_PRECISION_OPTION: dict[str, tuple[str, dict[str, str]]] = {
    "matmul_precision": (
        "matmul_precision",
        {"highest": "highest", "high": "high"},
    )
}

#: The four neutral sampling knobs, for the ports that spell all four the
#: neutral way.
#:
#: One shared object rather than six identical literals, for the reason
#: `MATMUL_PRECISION_OPTION` gives: a port that renames one of these declares
#: its own map, so six copies of the *identity* map are six chances to drop a
#: knob from one of them instead of a record of anything. This is not the
#: `Backend.sampling_options` default, which stays empty: that default is the
#: open contract a third-party adapter inherits, and inheriting four knobs
#: would translate them into native options a backend never claimed to run
#: rather than refusing them.
SAMPLING_OPTIONS: dict[str, str] = {
    "num_samples": "num_samples",
    "num_steps": "num_steps",
    "num_recycles": "num_recycles",
    "max_msa_depth": "max_msa_depth",
}

#: The values `glu_backend` accepts, spelled here rather than imported from
#: :data:`foldjax.models._glu.GLU_BACKENDS`.
#:
#: Reading the authority imports JAX, and the adapters resolve cache
#: directories without loading a model runtime -- which is why they carry
#: copies of native defaults at all. One copy rather than three: each port
#: still names it, keeps its own check and its own message, and pins this
#: tuple to `_glu`'s in its own drift test.
GLU_BACKENDS: tuple[str, ...] = ("xla", "tokamax", "pallas")


def realised_glu_backend(value: Any, *, released: str, serial: bool) -> Any:
    """The `glu_backend` a run realises, for `value` spelled or omitted.

    An explicit value is returned as written and never rewritten, so every
    refusal downstream still sees what was asked for. An omitted one is
    `pallas` on a serial GPU process (Boltz-2 and OpenFold3 call this, and
    Protenix would once `glu` is in its `runtime_policy.PALLAS_DEFAULT`; the
    measurements are at `foldjax.models._pallas_pair.default_backend`)
    and the port's `released` value everywhere else: off a GPU the kernel
    cannot run, and a context-parallel run partitions no fused GLU, so its
    omitted option keeps the released resolution each port already has.

    `cache_profile` and `predict` call this on the same options, the way
    Boltz-2's `_realised_ring_tile_kernel` is called, so the namespace names
    the program that ran: a GPU run with the option omitted records `pallas`
    and shares the entry an explicit `pallas` already warmed, while an explicit
    released value still strips to the absence every earlier run recorded.
    `serial` is the adapter's own reading of its shard count. The probe
    imports JAX, so it is imported here rather than at module scope.
    """

    if value is not None:
        return value
    if not serial:
        return released
    from foldjax.models._pallas_pair import default_backend

    return default_backend(released)


def validate_memory_policy_options(options: Mapping[str, Any]) -> None:
    """Reject a malformed memory-policy option while planning.

    Admission itself needs a device and runs at prediction time; the mode and
    the budget are knowable without one, so the ports that accept them parse
    both here, where `foldjax plan` reaches the error. One helper rather than
    the same two calls in two adapters: the pair carries an order -- the mode
    is reported before the budget -- and two copies can disagree about it.
    """
    memory_policy.parse_check_mode(options.get("memory_check"))
    memory_policy.parse_budget_gib(options.get("memory_budget_gib"))


def square_grid_cp_layout(options: Mapping[str, Any]) -> str | None:
    """The layout a square-grid port's resolver builds for this request.

    ``None`` when there is no distributed layout to name: a serial request, and
    a ``cp_devices`` this adapter cannot read as a device count. A malformed
    count then keeps its own cache namespace rather than borrowing a resolved
    one, and the port itself reports the value.

    For OpenDDE, Boltz-2 and ESMFold2, this helper's three grid callers, an
    omitted ``cp_layout`` resolves to the square grid on a perfect-square
    count (`models/opendde/models/model.py`, `models/boltz2/api.py`,
    `models/esmfold2/inference.py`), so both the recorded namespace and the
    padding alignment have to be resolved here rather than assumed
    one-dimensional. An explicit spelling is returned as written, including
    one the port will refuse, because it is the request's own identity.

    OpenFold3's ``auto`` picks the grid on the same rule
    (`models/openfold3/inference.py`) but its adapter applies that rule
    without this helper: it records ``serial`` where this returns ``None``,
    and it reads the device count from its own already-coerced value
    (`backends/openfold3.py`).
    """
    devices = options.get("cp_devices", 1)
    if isinstance(devices, bool) or not isinstance(devices, int) or devices <= 1:
        return None
    layout = str(options.get("cp_layout", "auto"))
    return square_grid_auto_layout(devices) if layout == "auto" else layout


#: Module-level tables in which a built-in adapter states the native defaults
#: it compiles with, keyed by native option name.
_DEFAULT_TABLES = ("_RELEASED_COMPILE_DEFAULTS", "DEFAULTS")


def _released_table_value(backend: Any, native: str) -> int | None:
    """``native``'s integer default in the backend module's tables, if stated."""
    import sys

    module = sys.modules.get(type(backend).__module__)
    for name in _DEFAULT_TABLES:
        table = getattr(module, name, None)
        candidate = table.get(native) if isinstance(table, dict) else None
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


class Backend(ABC):
    name: str

    #: Explicit opt-in to dispatcher-managed instance reuse.  The historical
    #: contract constructs a fresh backend for every scalar seed; keeping that
    #: default matters for third-party adapters whose instances are stateful.
    session_reuse: bool = False

    #: Backend-specific option names accepted in addition to the native names
    #: generated by ``sampling_options`` and ``execution_options``. ``None``
    #: preserves the open-ended contract for third-party backends; FoldJAX's
    #: built-ins all declare a set so ``plan`` can reject a typo before any
    #: output directory or model runtime is touched.
    native_options: frozenset[str] | None = None

    #: Option keys that change the compiled program. Only these participate in
    #: the cache namespace, so options that merely affect output formatting do
    #: not fragment the compilation cache.
    compile_options: tuple[str, ...] = ()

    #: Model-neutral sampling knob -> this backend's own option name. Every
    #: model calls "how many structures" something different, so the request
    #: carries one spelling and the adapter translates it.
    sampling_options: dict[str, str] = {}

    #: The same idea one level down, for the knobs that were left native:
    #: neutral knob -> (this backend's name, {neutral value: its value}).
    #: See `foldjax.execution` for the vocabulary and why `auto` never falls
    #: back silently.
    execution_options: dict[str, tuple[str, dict[str, str]]] = {}

    #: Native options that are switches. `true`/`false`, `yes`/`no`, `on`/
    #: `off`, `1`/`0` (any case) and real booleans all reach the port as a
    #: `bool` (`foldjax.execution.spell_booleans`). Only a declared switch is
    #: rewritten: the value `"1"` means a count elsewhere.
    boolean_options: frozenset[str] = frozenset()

    #: Dynamic shape axes understood by this backend's padding implementation.
    #: The public capability repeats this for discovery; this class attribute is
    #: the validation authority used before any model runtime is imported.
    padding_axes: tuple[str, ...] = ()

    @classmethod
    def canonical_options(cls, options: Mapping[str, Any]) -> dict[str, Any]:
        """``options`` in this backend's own names and spellings.

        Alias keys renamed, width and switch spellings unified, neutral knobs
        translated (`auto` to nothing where it is the omitted default). Two
        option mappings that mean the same run have equal canonical forms,
        which is what resume compares. Raises like `apply_sampling` on an
        option this backend cannot express. A classmethod, because it reads
        only the class's tables and resume must not construct a backend.
        """
        return execution.translate(
            execution.spell_booleans(
                execution.spell_dtypes(
                    execution.normalize(
                        dict(options),
                        native={name for name, _ in cls.execution_options.values()},
                    ),
                    cls.execution_options,
                ),
                cls.boolean_options,
            ),
            cls.execution_options,
            model=getattr(cls, "name", cls.__name__),
        )

    def apply_sampling(self, request: PredictionRequest) -> dict[str, Any]:
        """Merge the request's sampling and execution knobs into native options.

        A knob the backend cannot express is an error, and so is setting both
        the neutral knob and its native spelling: silently preferring one would
        change how many structures come back without changing the exit code.
        """
        options = self.canonical_options(request.options)
        for knob, value in request.sampling.items():
            native = self.sampling_options.get(knob)
            if native is None:
                raise ValueError(f"{self.name} does not support {knob}")
            if native in options:
                raise ValueError(
                    f"{knob} and the native option {native!r} were both set for "
                    f"{self.name}; pass one of them"
                )
            options[native] = value
        # Padding deliberately spells no MSA depth. It selects compiled shapes,
        # and the depth selects which alignment rows the model reads: deriving
        # one from the other is how a padded run came to cap a 16,384-row
        # alignment at the profile's 1,024-row floor. Each port keeps its own
        # default depth, padded or not, and `max_msa_depth` remains the one
        # option that changes it.
        return options

    def sampling_resolution(
        self, request: PredictionRequest
    ) -> dict[str, tuple[int | None, str]]:
        """What each neutral sampling knob runs at for ``request``, and why.

        Values are in the neutral knobs' units (``num_recycles`` counts the
        recycles after the first trunk pass on every port that defines it
        that way; see `docs/model-interface.md`). The source is ``request``
        (the knob was set), ``option`` (a native option, explicit or from a
        managed profile), ``default`` (the adapter's own value) or
        ``checkpoint`` (decided by the checkpoint or its model variant). A
        value of ``None`` means it is decided where the adapter cannot see
        before loading -- a checkpoint this request names but that carries no
        readable setting, or a native runner that keeps it internal.

        Resolved through `apply_sampling`, the translation a run takes, so a
        default the adapter supplies there (ESMFold2's recycle count) is the
        value reported.
        """
        options = self.apply_sampling(request)
        omitted = self._omitted_sampling(request, options)
        resolved: dict[str, tuple[int | None, str]] = {}
        for knob, native in self.sampling_options.items():
            if native not in options:
                resolved[knob] = omitted.get(knob, (None, "checkpoint"))
                continue
            if knob in request.sampling:
                source = "request"
            elif native in request.options:
                source = "option"
            else:
                source = "default"
            # The translated value, not the one spelled: OpenFold3 narrows a
            # requested MSA depth to its released 1,024 rows.
            resolved[knob] = (
                self._neutral_sampling_value(knob, options[native]),
                source,
            )
        return resolved

    def _omitted_sampling(
        self, request: PredictionRequest, options: Mapping[str, Any]
    ) -> dict[str, tuple[int | None, str]]:
        """Values for the knobs neither the request nor `apply_sampling` set.

        Read from the module-level tables in which a built-in adapter states
        the native defaults it compiles with, so no port's defaults are stated
        twice. A port whose value comes from elsewhere overrides this.
        """
        del request, options
        found: dict[str, tuple[int | None, str]] = {}
        for knob, native in self.sampling_options.items():
            value = _released_table_value(self, native)
            if value is not None:
                found[knob] = (self._neutral_sampling_value(knob, value), "default")
        return found

    def _neutral_sampling_value(self, knob: str, native_value: Any) -> int | None:
        """A native sampling value in the neutral knob's units."""
        del knob
        if isinstance(native_value, bool) or not isinstance(native_value, int):
            return None
        return native_value

    def matmul_precision(
        self, options: dict[str, Any]
    ) -> Callable[[], AbstractContextManager[None]]:
        """Take the matmul precision out of `options`; return a scope factory.

        Popped rather than read, because the native option name exists only to
        carry the value this far -- no model takes it as an argument. Absent,
        the scope is a no-op and every port keeps the precision it pins for
        itself.

        A factory rather than one context manager: two of these backends invoke
        the model more than once per request, and a generator-based context
        manager cannot be entered twice. `with matmul_precision():` is correct
        in a loop; `with matmul_precision:` would raise on the second pass.
        """
        value = options.pop("matmul_precision", None)
        return functools.partial(execution.matmul_precision_scope, value)

    def validate_request(self, request: PredictionRequest) -> None:
        """Validate the cheap, side-effect-free part of a prediction request.

        This is shared by planning and execution. Runtime artifacts and model
        data remain prediction-time concerns, but an unsupported input dialect,
        neutral knob, execution value, or misspelled native option is knowable
        without loading weights and must have the same answer in both paths.
        """
        capabilities = self.capabilities()
        if request.input_format not in capabilities.input_formats:
            raise ValueError(
                f"{self.name} does not support input format {request.input_format!r}"
            )
        self._validate_representations(request, capabilities)
        options = self.apply_sampling(request)
        # Consumed while the common document is translated, never by the
        # native runner, so it leaves the option set before the native checks.
        from foldjax.input import (
            IGNORE_CONSTRAINTS,
            IGNORE_NUCLEIC_MSA,
            IGNORE_TEMPLATES,
            accepts_ignore_constraints,
            accepts_ignore_nucleic_msa,
            accepts_ignore_templates,
            refuse_ignored_constraints,
        )
        from foldjax.pocket_selection import POCKET_SAMPLING
        from foldjax.pocket_selection import requested as requested_pocket_sampling
        from foldjax.template_search import refuse_template_search

        # Here as well as at materialization, so `foldjax plan` refuses a
        # template search whose result the backend would discard.
        refuse_template_search(
            self.name,
            request.templates,
            options,
            input_format=request.input_format,
            template_dir=request.template_dir,
        )
        if request.msa_pairing != "model":
            from foldjax.msa_search import refuse_msa_pairing

            if request.input_format != "foldjax":
                raise ValueError(
                    f"msa_pairing={request.msa_pairing!r} pairs alignments "
                    "FoldJAX searches while translating a FoldJAX-format job; "
                    f"this {request.input_format!r} input is passed to "
                    f"{self.name} untouched"
                )
            refuse_msa_pairing(self.name, request.msa_pairing)

        # FoldJAX's own route on every backend (`foldjax.pocket_selection`):
        # read by the translation and the selection after the run, never by
        # a native runner. A native document is passed through untouched, so
        # there is no common pocket for it to score.
        if POCKET_SAMPLING in options:
            if (
                requested_pocket_sampling(options) == "select"
                and request.input_format != "foldjax"
            ):
                raise ValueError(
                    f"{POCKET_SAMPLING}=select scores the samples against a "
                    "FoldJAX common-schema job's pocket constraint; native "
                    f"{self.name} input is passed through untouched"
                )
            options.pop(POCKET_SAMPLING)

        # Governs a native ``constraint`` and a common job's pocket and contact
        # ``constraints`` alike. ``false`` on native input is checked here,
        # where `foldjax plan` sees it; on a common job the translation
        # (`foldjax.input._validate_pocket_constraints`) refuses it.
        if IGNORE_CONSTRAINTS in options and accepts_ignore_constraints(self.name):
            ignore_constraints = _strict_boolean(
                options.pop(IGNORE_CONSTRAINTS), name=IGNORE_CONSTRAINTS
            )
            if request.input_format != "foldjax" and not ignore_constraints:
                refuse_ignored_constraints(request.input, self.name)

        for option, accepts in (
            (IGNORE_NUCLEIC_MSA, accepts_ignore_nucleic_msa),
            (IGNORE_TEMPLATES, accepts_ignore_templates),
        ):
            if option not in options or not accepts(self.name):
                continue
            if (
                _strict_boolean(options.pop(option), name=option)
                and request.input_format != "foldjax"
            ):
                raise ValueError(
                    f"{option} applies to FoldJAX common-schema input; "
                    f"native {self.name} input is passed through untouched"
                )
        if request.padding is not None:
            if not self.padding_axes:
                raise ValueError(f"{self.name} does not support input padding")
            unsupported_padding = sorted(
                set(request.padding.explicit_axes) - set(self.padding_axes)
            )
            if unsupported_padding:
                raise ValueError(
                    f"{self.name} does not support explicit padding axes: "
                    f"{', '.join(unsupported_padding)}"
                )
        # Native spellings are an escape hatch, but they still mean the same
        # thing as the neutral fields.  Validate them with the public request's
        # exact integer contract so a string/float/bool cannot pass ``plan`` and
        # fail only after a model starts loading.
        for native in self.sampling_options.values():
            if native in options:
                minimum = 0 if (
                    native == self.sampling_options.get("num_recycles")
                    and self.name in {"alphafold3", "boltz2", "esmfold2"}
                ) else 1
                _strict_integer(options[native], name=native, minimum=minimum)
        if self.native_options is None:
            return
        generated = set(self.sampling_options.values())
        generated.update(native for native, _values in self.execution_options.values())
        unsupported = sorted(set(options) - generated - set(self.native_options))
        if unsupported:
            raise ValueError(
                f"unsupported {self.name} options: {', '.join(unsupported)}"
            )
        self.validate_native_options(options)

    def _validate_representations(
        self,
        request: PredictionRequest,
        capabilities: ModelCapabilities,
    ) -> None:
        """Reject invalid representation selectors and unsupported fan-out.

        Native adapters resolve these names again when they decide which arrays
        to persist.  Planning must apply the same comma-separated/``all``
        vocabulary first, though: otherwise an invalid name is discovered only
        after weights and model-specific runtime state have started loading.
        The public result owns one representation archive, so one request also
        cannot capture representations from more than one seed without losing
        every handle but one.
        """
        if not request.representations:
            return
        available = (
            capabilities.input_representations
            if request.stop_after == "inputs" else capabilities.representations
        )
        if request.stop_after == "inputs":
            for entry in request.representations:
                for part in str(entry).split(","):
                    part = part.strip()
                    if part and part != "all" and part not in available:
                        raise ValueError(
                            f"unknown input representation {part!r} for {self.name}"
                        )
        found = False
        selected_all = False
        for entry in request.representations:
            for part in str(entry).split(","):
                part = part.strip()
                if not part:
                    continue
                if part == "all":
                    if not available:
                        raise ValueError(
                            f"{self.name} does not expose trunk representations"
                        )
                    # This matches the native resolver: ``all`` selects the
                    # complete capability list and supersedes later entries.
                    found = True
                    selected_all = True
                    break
                if part not in available:
                    produced = ", ".join(available) or "none"
                    raise ValueError(
                        f"unknown representation {part!r} for {self.name}; "
                        f"this model produces: {produced}"
                    )
                found = True
            if selected_all:
                break
        if not found:
            raise ValueError(
                "representations must name at least one representation or 'all'"
            )
        if request.seed_count > 1:
            raise ValueError(
                "representations cannot be combined with multiple seeds: "
                "PredictionResult carries one representation archive; run one "
                "seed per request"
            )

    def validate_native_options(self, options: dict[str, Any]) -> None:
        """Validate built-in option values without importing a model runtime.

        Subclasses override this only for checks that are pure and cheap. The
        hook receives the translated native mapping so it can reuse the same
        helpers as ``predict`` while keeping planning free of preprocessing,
        checkpoint loading, compilation, downloads, and output writes.
        """

    def pocket_conditioning(self, request: PredictionRequest) -> bool:
        """Whether this request's native input conditions on a common pocket.

        The translation table's answer for most ports; a backend whose
        checkpoint decides (Protenix) overrides it. Under
        ``pocket_sampling=select`` a pocket this says no to reaches only
        FoldJAX's selection (`foldjax.pocket_selection`).
        """
        from foldjax.input import pocket_conditioned

        del request
        return pocket_conditioned(self.name)

    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResult: ...

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        """Keep request-scoped runtime state while executing related runs.

        Only backends with ``session_reuse = True`` are dispatched through
        this single-threaded context. Large built-ins may override it to load
        one checkpoint for several inputs or seeds. The context boundary is
        deliberate: FoldJAX must not turn multi-gigabyte model state into a
        process-global cache, retain two models while switching backends, or
        leak a failed run into the next request. Implementations must defer
        fallible loading to :meth:`predict`, where ``on_error`` applies, and
        make context cleanup non-throwing.
        """

        del requests
        yield self

    def invalidate_session(self) -> None:
        """Discard reusable state after an absorbed prediction failure.

        A backend that implements :meth:`session` may override this hook.  The
        no-op keeps third-party backends source-compatible while allowing the
        dispatcher to preserve the historical failure-isolation contract.
        """

    def observe_resumed(self, request: PredictionRequest) -> None:
        """Bind a reused artifact to request-scoped runtime state, if any."""

    def validate_session(self, request: PredictionRequest) -> None:
        """Refuse when resources used by this session changed on disk."""

    def profile_refusal(
        self, document: Mapping[str, Any], profile: str | None
    ) -> str | None:
        """Why one asset profile's weights cannot run a common-schema job.

        `foldjax models --for` asks this once the input layer has said the
        model can express the job, for refusals that depend on which managed
        checkpoint runs it rather than on the document alone. ``profile`` None
        is the released one. Answered without weights; None when nothing is
        known to refuse it.
        """
        del document, profile
        return None

    def managed_asset_profile(
        self,
        options: Mapping[str, Any],
        *,
        weights: Path | None = None,
        requested: str | None = None,
    ) -> str | None:
        """Name the managed asset bundle this request's options actually select.

        Request resolution asks every backend before it consults the weight
        store, so a model whose options change which files are required answers
        here instead of having the shared layer special-case its name. The
        default keeps whatever profile the request asked for. ``weights`` is the
        explicitly supplied path, if any, and is what lets a backend decline to
        infer a profile for weights the caller placed itself.
        """
        del options, weights
        return requested

    def apply_managed_profile(
        self,
        options: dict[str, Any],
        profile: str,
        *,
        weights: Path | None = None,
    ) -> dict[str, Any]:
        """Fold one public asset profile into this backend's native options.

        Called twice per resolution: once with the requested profile before any
        weight path is known, and once with the resolved profile and its
        ``weights`` afterwards, for backends that stage companion checkpoints
        beside the structure weights. The default owns no options and returns
        ``options`` unchanged.
        """
        del profile, weights
        return options

    @staticmethod
    def _strip_released_defaults(
        profile: dict[str, Any],
        defaults: Mapping[str, Any],
        *,
        skip: Container[str] = (),
    ) -> None:
        """Drop explicitly spelled released defaults from ``profile`` in place.

        Naming a value the native runner would have resolved anyway must not
        select a second compilation namespace. ``skip`` names defaults this
        request does not inherit, for the routes where a backend only supplies
        some of them.

        The match is on exact type as well as value because ``bool`` is an
        ``int`` subclass: a merely equal lookalike, and any malformed, extended
        or future type variant, keeps its own namespace rather than inheriting
        a released-default alias without parser proof.
        """
        for name, default in defaults.items():
            if name in skip or name not in profile:
                continue
            value = profile[name]
            if type(value) is type(default) and value == default:
                profile.pop(name)

    def cache_profile(self, request: PredictionRequest) -> dict[str, Any]:
        """Return the compile-relevant identity of ``request`` for this backend.

        Neutral sampling/execution knobs are translated first, so spelling the
        same schedule through ``num_steps`` or a backend's native ``steps``
        selects one namespace. JSON normalization happens in ``cache_namespace``.
        """
        options = self.apply_sampling(request)
        return {
            key: options[key]
            for key in self.compile_options
            if key in options
        }
