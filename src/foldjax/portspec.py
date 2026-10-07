"""What each port is, declared once as data instead of re-derived by string compare.

Three modules that know nothing about any model -- `foldjax.manifest`,
`foldjax.assets` and `foldjax.input` -- each needed a handful of per-port facts:
which implementation files a result depends on, which managed asset profiles
exist, which native dialect a common job is written into. Each asked with
`if request.model == "..."`, so the same six names appeared in forty branches
spread over three files, and adding a port meant finding all forty.

The facts live here instead, one entry per port, in the shape
`foldjax.output`'s `_RANKING_SCORE` already uses. Two rules keep that from
becoming a second dispatcher:

* **Only facts.** Behaviour that really is conditional -- a directory probe, a
  rejection with its own message, a fallback that applies to every port but one
  -- stays written out at the site that performs it. A table that hides a
  condition is worse than the branch it replaced.
* **Nothing is imported to read the table.** Code the table names is named as
  an allow-listed ``"module:attribute"`` string and resolved by
  :func:`provider` *inside* the operation that needs it. So this module imports
  only the standard library: importing it pulls in no JAX, no torch, no
  backend and no model implementation, which is the property
  `manifest.py`'s dependency ladder was written around ("Keep imports local so
  ordinary manifest use remains free of the AlphaFold runtime").

The table is keyed by canonical model ids only. It never normalises an alias --
`foldjax.registry.normalize_model_name` is the one place that does, and it
reads :data:`ALIASES` from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

#: The compatible default every model keeps.
RELEASED_PROFILE = "released"
#: ESMFold2 without ESMC-6B. Valid only with ``no_language_model=true``, and a
#: deliberately different model from the released structure+ESMC bundle.
STRUCTURE_ONLY_PROFILE = "structure-only"
#: Protenix's current release, announced 2026-04-08 with a technical report:
#: the same blocks at c_z=256 with `hidden_scale_up`, 464M parameters against
#: the base model's 368M, and clear gains on antibody-antigen targets.
PROTENIX_V2_PROFILE = "v2"
#: Protenix's other public base checkpoint: the same 368M architecture as the
#: release, trained to a 2025-06-30 wwPDB cutoff instead of AlphaFold 3's
#: 2021-09-30. Upstream recommends it "for practical application scenarios" and
#: keeps the default for benchmarks, because a fair comparison against
#: AlphaFold 3 needs the matching cutoff. That is why this is a profile rather
#: than the default: it predicts better on recent targets and worse on the one
#: question the benchmark table asks.
PROTENIX_BASE_20250630_PROFILE = "base-20250630"
PROTENIX_MINI_ESM_PROFILE = "mini-esm-v0.5.0"
PROTENIX_MINI_ISM_PROFILE = "mini-ism-v0.5.0"
#: Upstream's three other public v0.5.0 checkpoints
#: (`configs/configs_model_type.py:111-249`): the constraint-conditioned base
#: model and the two small models without a language model.
PROTENIX_BASE_CONSTRAINT_PROFILE = "base-constraint-v0.5.0"
PROTENIX_MINI_DEFAULT_PROFILE = "mini-default-v0.5.0"
PROTENIX_TINY_DEFAULT_PROFILE = "tiny-default-v0.5.0"
OPENDDE_ABAG_PROFILE = "abag"


@dataclass(frozen=True)
class SourceSpec:
    """One tracked implementation file, or one tree of them.

    ``path`` is relative to the installed `foldjax` package directory. With
    ``pattern`` set it names a directory whose matching files are tracked in
    sorted order, which is how every port's whole source tree is bound: a
    port's numerics and featurization live across dozens of files and
    enumerating them here would go stale silently, so the policy is the
    directory rather than the listing. ``recursive=False`` matches only the
    directory's own entries, for `models/`'s shared helpers, whose
    subdirectories are the ports.
    """

    path: str
    pattern: str | None = None
    recursive: bool = True

    @property
    def token(self) -> str:
        """A stable one-line spelling of this spec, for frozen fixtures."""
        if self.pattern is None:
            return self.path
        if not self.recursive:
            return f"{self.path}/{self.pattern}"
        return f"{self.path}/**/{self.pattern}"

    def resolve(self, package: Path) -> tuple[Path, ...]:
        root = package / self.path
        if self.pattern is None:
            return (root,)
        found = root.rglob(self.pattern) if self.recursive else root.glob(self.pattern)
        return tuple(sorted(path for path in found if path.is_file()))


@dataclass(frozen=True)
class StagingSpec:
    """One abandoned-staging glob under a model's weight directory.

    ``base`` is ``"root"`` (the weight directory), ``"parent"`` (the store above
    it, where Boltz-2's native conversion stages), or the name of a
    subdirectory. It is not a relative path: ``root.parent`` and ``root / ".."``
    glob the same files under different names, and these paths are unlinked.
    """

    pattern: str
    base: str = "root"

    def directory(self, root: Path) -> Path:
        if self.base == "root":
            return root
        if self.base == "parent":
            return root.parent
        return root / self.base


@dataclass(frozen=True)
class PortSpec:
    """Everything the model-neutral layer needs to know about one port."""

    #: Canonical model id -- the name every neutral module keys on.
    model: str
    #: Accepted spellings besides :attr:`model`.
    aliases: tuple[str, ...]
    #: The backend class, resolved lazily so the table stays import-free.
    backend: str
    #: Managed asset profiles, the compatible default first.
    asset_profiles: tuple[str, ...]
    #: Builder per non-default profile, ``(spec, profile) -> ModelAssets``.
    asset_profile_providers: Mapping[str, str] = field(default_factory=dict)
    #: FoldJAX-owned temporary trees a killed conversion may have left.
    asset_staging: tuple[StagingSpec, ...] = ()
    #: Implementation files recorded as run inputs, in this order. The manifest
    #: sorts by resolved path before persisting, so the order is for reading.
    manifest_sources: tuple[SourceSpec, ...] = ()
    #: Companion checkpoint files read beyond ``request.weights``.
    manifest_weight_assets: str | None = None
    #: Managed/environment CCD files featurization may read.
    manifest_ccd_assets: str | None = None
    #: Writer that turns a validated common job into this port's native input.
    input_dialect: str = ""
    #: Validator for CCD codes in a common document, when the port has one.
    input_ccd_validator: str | None = None


#: FoldJAX-owned code every port's prediction path runs through, beyond its
#: own tree: the shared `models/_*.py` helpers (RNG, compile policy, kernels
#: and their routing, stacking, context parallelism -- one flat directory, so
#: a helper added later is bound without an edit here), the adapter base whose
#: `realised_glu_backend` resolves an omitted option, and the common-job writer
#: and translation in `input.py` that turns a FoldJAX document into the native
#: one, including the MSA and template paths it writes.
_SHARED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("models", pattern="_*.py", recursive=False),
    SourceSpec("backends/base.py"),
    SourceSpec("input.py"),
)

#: Reads a user-supplied template structure for the dialects that map
#: templates through it (Boltz-2's and OpenFold3's writers in `input.py`).
_TEMPLATE_READER = SourceSpec("template_search.py")


#: Every port, keyed by canonical model id.
#:
#: `manifest_sources` is the part with a persisted consequence. Each entry is a
#: file whose *content* can change what the model predicts while the request's
#: options stay identical, so a result produced before such a change must not
#: satisfy a resume request made after it. The policy is structural, not
#: repair-driven: a port binds its whole source tree (featurizer, model,
#: runner, native-input and output writers, vendored upstream code), the
#: packaged data assets its featurizer loads, its adapter (which resolves
#: omitted options), every module in another port's tree that its code
#: imports, and `_SHARED_SOURCES`. `tests/test_portspec.py` re-derives the
#: import closure of each tree and fails when a `foldjax.models` module it
#: reaches is not bound. Package infrastructure outside `models/`
#: (`padding`, `paths`, `schema`, `memory_policy`, the search clients) is not
#: bound: what it decides is either an option already in the request identity
#: or a remote search, which makes the run unverifiable on its own.
#:
#: Binding is by stat identity (`manifest.path_stat_identity`), so a reinstall
#: or checkout that rewrites these files makes every earlier run of the port
#: unresumable. That is the intended trade -- a rerun -- against silently
#: reusing a prediction an upgraded featurizer or model would not make.
PORTS: Mapping[str, PortSpec] = {
    "alphafold3": PortSpec(
        model="alphafold3",
        aliases=("af3", "alphafold-3"),
        backend="foldjax.backends.alphafold3:AlphaFold3Backend",
        asset_profiles=(RELEASED_PROFILE,),
        # The managed AlphaFold 3 route runs a vendored runner against a
        # vendored source tree, but only when no explicit `source` is given and
        # only once the libcifpp selection resolves. That is a condition rather
        # than a list, so `manifest.py` still spells it out. What is bound here
        # is FoldJAX's own part of every route: the adapter, and the writer
        # that turns a common job (and its template chain filter) into the
        # AlphaFold 3 JSON.
        manifest_sources=(
            SourceSpec("backends/alphafold3.py"),
            *_SHARED_SOURCES,
        ),
        input_dialect="foldjax.input:_alphafold3",
    ),
    "boltz2": PortSpec(
        model="boltz2",
        aliases=("boltz", "boltz-2", "boltz-jax"),
        backend="foldjax.backends.boltz2:Boltz2Backend",
        asset_profiles=(RELEASED_PROFILE,),
        asset_staging=(
            StagingSpec(".foldjax-boltz-native-*", base="parent"),
            StagingSpec(".foldjax-mols-*"),
        ),
        manifest_sources=(
            # Native AMP/normalization repairs, the featurizer and its parsers,
            # and the structure writer. Stat only source files, not mutable
            # __pycache__ trees. The CCD `mols/` tree is bound by `manifest.py`
            # beside the checkpoint.
            SourceSpec("models/boltz2", pattern="*.py"),
            SourceSpec("backends/boltz2.py"),
            _TEMPLATE_READER,
            *_SHARED_SOURCES,
        ),
        manifest_weight_assets="foldjax.manifest:_boltz2_weight_assets",
        input_dialect="foldjax.input:_boltz",
        input_ccd_validator=(
            "foldjax.models.boltz2.data.identifiers:validate_ccd_identifier"
        ),
    ),
    "esmfold2": PortSpec(
        model="esmfold2",
        aliases=("esm-fold2", "esmfold-2"),
        backend="foldjax.backends.esmfold2:ESMFold2Backend",
        asset_profiles=(RELEASED_PROFILE, STRUCTURE_ONLY_PROFILE),
        asset_profile_providers={
            STRUCTURE_ONLY_PROFILE: "foldjax.assets:_esmfold2_structure_only_assets",
        },
        asset_staging=(
            StagingSpec(".foldjax-stage-*"),
            StagingSpec(".foldjax-stage-*", base="esmc"),
        ),
        manifest_sources=(
            # Native autocast routing, dropout opmath and the featurizer.
            SourceSpec("models/esmfold2", pattern="*.py"),
            SourceSpec("backends/esmfold2.py"),
            # The CUDA norm implementation and the primitives it shares live in
            # Boltz-2's tree and affect ESM outputs.
            SourceSpec("models/boltz2/models/primitives/_common.py"),
            SourceSpec("models/boltz2/models/primitives/native_amp_norm.py"),
            SourceSpec("models/boltz2/models/primitives/native_pwa_weights.py"),
            *_SHARED_SOURCES,
        ),
        manifest_weight_assets="foldjax.manifest:_esmfold2_weight_assets",
        input_dialect="foldjax.input:_esmfold2",
    ),
    "opendde": PortSpec(
        model="opendde",
        aliases=("open-dde", "opendde-jax"),
        backend="foldjax.backends.opendde:OpenDDEBackend",
        asset_profiles=(RELEASED_PROFILE, OPENDDE_ABAG_PROFILE),
        asset_profile_providers={
            OPENDDE_ABAG_PROFILE: "foldjax.assets:_opendde_abag_assets",
        },
        asset_staging=(StagingSpec(".foldjax-opendde-native-*"),),
        manifest_sources=(
            # Omitted options can change meaning when the native precision
            # default changes; the adapter and `runner.py` hold those defaults,
            # so a legacy BF16 result cannot satisfy an otherwise identical
            # request whose default is now FP32.
            SourceSpec("models/opendde", pattern="*.py"),
            SourceSpec("models/opendde", pattern="*.npz"),
            SourceSpec("backends/opendde.py"),
            # OpenDDE featurizes through Protenix's `featurize_protein_json`
            # (which loads its CCD tables) and runs Protenix's trunk, heads and
            # primitives, so the whole Protenix tree and its tables are bound.
            SourceSpec("models/protenix", pattern="*.py"),
            SourceSpec("models/protenix", pattern="*.npz"),
            *_SHARED_SOURCES,
        ),
        manifest_ccd_assets="foldjax.manifest:_ccd_chemistry_assets",
        input_dialect="foldjax.input:_protenix",
    ),
    "openfold3": PortSpec(
        model="openfold3",
        aliases=("of3", "openfold-3", "openfold3-jax"),
        backend="foldjax.backends.openfold3:OpenFold3Backend",
        asset_profiles=(RELEASED_PROFILE,),
        asset_staging=(StagingSpec(".foldjax-stage-*"),),
        manifest_sources=(
            # Sample-chunk/augmentation repairs, the featurizer, and the
            # vendored upstream data pipeline under `_upstream/`.
            SourceSpec("models/openfold3", pattern="*.py"),
            SourceSpec("backends/openfold3.py"),
            # Resolves an omitted triangle kernel to the fused GPU one.
            SourceSpec("_openfold3_compile.py"),
            SourceSpec("models/boltz2/models/primitives/_common.py"),
            SourceSpec("models/boltz2/models/primitives/native_amp_norm.py"),
            _TEMPLATE_READER,
            *_SHARED_SOURCES,
        ),
        manifest_ccd_assets="foldjax.manifest:_openfold3_ccd_assets",
        input_dialect="foldjax.input:_openfold3",
    ),
    "protenix": PortSpec(
        model="protenix",
        aliases=("protenix-jax",),
        backend="foldjax.backends.protenix:ProtenixBackend",
        asset_profiles=(
            RELEASED_PROFILE,
            PROTENIX_V2_PROFILE,
            PROTENIX_BASE_20250630_PROFILE,
            PROTENIX_MINI_ESM_PROFILE,
            PROTENIX_MINI_ISM_PROFILE,
            PROTENIX_BASE_CONSTRAINT_PROFILE,
            PROTENIX_MINI_DEFAULT_PROFILE,
            PROTENIX_TINY_DEFAULT_PROFILE,
        ),
        asset_profile_providers={
            PROTENIX_V2_PROFILE: "foldjax.assets:_protenix_v2_assets",
            PROTENIX_BASE_20250630_PROFILE: (
                "foldjax.assets:_protenix_base_20250630_assets"
            ),
            PROTENIX_MINI_ESM_PROFILE: "foldjax.assets:_protenix_variant_assets",
            PROTENIX_MINI_ISM_PROFILE: "foldjax.assets:_protenix_variant_assets",
            PROTENIX_BASE_CONSTRAINT_PROFILE: (
                "foldjax.assets:_protenix_plain_checkpoint_assets"
            ),
            PROTENIX_MINI_DEFAULT_PROFILE: (
                "foldjax.assets:_protenix_plain_checkpoint_assets"
            ),
            PROTENIX_TINY_DEFAULT_PROFILE: (
                "foldjax.assets:_protenix_plain_checkpoint_assets"
            ),
        },
        asset_staging=(StagingSpec(".foldjax-protenix-native-*"),),
        manifest_sources=(
            # Rigid augmentation, native mixed-precision conditioning, the AMP
            # and chunk policies, and the featurizer `data/featurize_json.py`
            # with the CCD tables it loads (`ccd_std_residues.npz`,
            # `ccd_ligands.npz`, `ccd_nucleotides.npz`).
            SourceSpec("models/protenix", pattern="*.py"),
            SourceSpec("models/protenix", pattern="*.npz"),
            SourceSpec("backends/protenix.py"),
            *_SHARED_SOURCES,
        ),
        manifest_weight_assets="foldjax.manifest:_protenix_weight_assets",
        manifest_ccd_assets="foldjax.manifest:_ccd_chemistry_assets",
        input_dialect="foldjax.input:_protenix",
    ),
}

#: Alias -> canonical id, including each canonical id as itself. Read by
#: `foldjax.registry.normalize_model_name`, which is the only normaliser.
ALIASES: Mapping[str, str] = {
    **{model: model for model in PORTS},
    **{alias: model for model, spec in PORTS.items() for alias in spec.aliases},
}


def _declared_providers() -> frozenset[str]:
    references: set[str] = set()
    for spec in PORTS.values():
        references.add(spec.backend)
        references.update(spec.asset_profile_providers.values())
        references.update(
            reference
            for reference in (
                spec.manifest_weight_assets,
                spec.manifest_ccd_assets,
                spec.input_dialect,
                spec.input_ccd_validator,
            )
            if reference
        )
    return frozenset(references)


#: Every ``"module:attribute"`` the table is allowed to reach. A reference that
#: is not in the table cannot be resolved through :func:`provider`, so the
#: indirection stays a lookup rather than a way to import arbitrary code.
PROVIDERS: frozenset[str] = _declared_providers()


def provider(reference: str) -> Any:
    """Resolve one allow-listed ``"module:attribute"`` reference.

    Call this from inside the operation that needs the code, never at import
    time: the table's whole point is that reading a port's facts costs no
    imports.
    """
    if reference not in PROVIDERS:
        raise ValueError(f"undeclared port table provider: {reference!r}")
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"provider must be spelled 'module:attribute': {reference!r}")
    return getattr(import_module(module_name), attribute)
