"""What produced this output directory.

A structure file records coordinates and nothing about the run that made it.
Six months later a directory of `.cif` files cannot answer which model wrote
them, which checkpoint, how many recycles, or on which seed -- and neither can
FoldJAX, because the request that knew all of it is gone. Two directories from
two schedules look identical.

So every prediction writes `foldjax_run.json` beside its structures: the model
and the backend, the input and its digest, the resolved weights and their
identity, the seeds and sampling knobs actually used, the JAX runtime, and the
structures with the confidence each reported. It is written after the run, so
its presence also means the run finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foldjax.redaction import public_options
from foldjax.schema import PredictionRequest, PredictionResult

MANIFEST_NAME = "foldjax_run.json"
#: Version of the fields that make a finished run safe to reuse.  Manifests
#: written before this contract existed deliberately do not qualify: a
#: structure on disk is not evidence that it came from the request now being
#: made.
MANIFEST_SCHEMA = 1
_UNVERIFIABLE = object()


def _stat_fields(info: os.stat_result) -> tuple[int, ...]:
    """Metadata that changes when a file is replaced or rewritten in place."""
    return (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _weight_stat_signature(path: Path) -> tuple[Any, ...] | None:
    """Return a hashable snapshot of a weight file or deterministic tree."""
    try:
        root = Path(path).resolve(strict=True)
        root_info = root.lstat()
        if stat.S_ISREG(root_info.st_mode):
            return ("file", str(root), _stat_fields(root_info))
        if not stat.S_ISDIR(root_info.st_mode):
            return None

        entries: list[tuple[Any, ...]] = []

        def raise_walk_error(error: OSError) -> None:
            raise error

        for parent, directories, files in os.walk(
            root, followlinks=False, onerror=raise_walk_error
        ):
            directories.sort()
            files.sort()
            parent_path = Path(parent)
            for name in [*directories, *files]:
                child = parent_path / name
                info = child.lstat()
                relative = child.relative_to(root).as_posix()
                mode = info.st_mode
                if stat.S_ISREG(mode):
                    kind = "file"
                    detail = None
                elif stat.S_ISDIR(mode):
                    kind = "directory"
                    detail = None
                else:
                    # A model loader may follow nested links or interpret a
                    # special entry. Metadata for the directory entry alone
                    # cannot bind that external content, so do not claim this
                    # tree is resumable.
                    return None
                entries.append(
                    (relative, kind, _stat_fields(info), detail)
                )
        return (
            "directory",
            str(root),
            _stat_fields(root_info),
            tuple(entries),
        )
    except (OSError, RuntimeError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_record(signature: tuple[Any, ...]) -> dict[str, Any]:
    """Compact JSON-safe identity of all stat fields in one tree snapshot."""
    record: dict[str, Any] = {
        "kind": signature[0],
        "root": list(signature[2]),
    }
    if signature[0] == "directory":
        payload = json.dumps(
            signature[3], ensure_ascii=True, separators=(",", ":")
        ).encode("ascii")
        record["entries"] = len(signature[3])
        # The digest binds every relative name, type, device/inode/size/time
        # field, and link target without bloating each run manifest by thousands
        # of checkpoint files.
        record["tree_sha256"] = hashlib.sha256(payload).hexdigest()
    return record


def path_stat_identity(path: Path) -> dict[str, Any] | None:
    """Cheap persisted identity without reading a potentially huge payload."""
    signature = _weight_stat_signature(path)
    if signature is None:
        return None
    return {
        "kind": signature[0],
        "stat_signature": _signature_record(signature),
    }


def stat_identity_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    """Whether mode/device/inode/size/mtime/ctime and tree entries are unchanged."""
    current = path_stat_identity(path)
    return current is not None and _exact_value(current, expected)


def device_peak_bytes() -> int | None:
    """The device allocator's high-water mark, or None when there is no device.

    `peak_bytes_in_use` and not the pool: JAX preallocates most of the card by
    default, so anything read from `nvidia-smi` reports the reservation and is
    the same number for a 250-token job and a 3000-token one. This is the bytes
    actually in use at the worst moment.

    It is cumulative for the process and JAX exposes no reset, so a run that does
    several passes reports the largest of them -- which is what a memory figure
    for the whole run should say anyway. It is not comparable across processes
    that did unrelated work first.
    """
    try:
        import jax

        devices = jax.devices()
    except Exception:  # noqa: BLE001 - a CPU-only or broken runtime is not an error
        return None
    if not devices:
        return None
    stats = devices[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return int(peak) if peak is not None else None


def _digest(path: Path) -> str | None:
    """SHA-256 of a file, or None if it is not one.

    The input is hashed rather than merely named because job files are edited
    in place far more often than they are renamed, and a path alone cannot tell
    two runs apart.
    """
    try:
        if not path.is_file():
            return None
        return _sha256_file(path)
    except (OSError, RuntimeError):
        return None


def file_content_digest(path: Path) -> str | None:
    """Public small-file digest used to validate persisted output artifacts."""
    return _digest(Path(path))


def _resolved_path(path: Path) -> str | None:
    """Canonical path identity, or None when the target cannot be resolved."""
    try:
        return str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError):
        return None


def _exact_value(left: Any, right: Any) -> bool:
    """Compare JSON-shaped values without conflating booleans or number types."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _exact_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _exact_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _option_identity(value: Any) -> Any:
    """Return a lossless JSON-shaped option value, or an unverifiable marker."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            return _UNVERIFIABLE
        converted = {key: _option_identity(item) for key, item in value.items()}
        if any(item is _UNVERIFIABLE for item in converted.values()):
            return _UNVERIFIABLE
        return converted
    if isinstance(value, (list, tuple)):
        converted = [_option_identity(item) for item in value]
        if any(item is _UNVERIFIABLE for item in converted):
            return _UNVERIFIABLE
        return converted
    return _UNVERIFIABLE


def _public_options_are_complete(
    request: PredictionRequest, options: Mapping[str, Any]
) -> bool:
    """Whether the public option record still proves the backend input.

    Credentials are intentionally redacted from run manifests.  Two different
    secrets would therefore have the same public value, so such a manifest is
    not sufficient evidence for reuse.  Non-JSON option objects are handled the
    same conservative way rather than storing a potentially sensitive digest.
    """
    identity = _option_identity(request.options)
    return identity is not _UNVERIFIABLE and _exact_value(identity, options)


_PATH_OPTION_WORDS = frozenset(
    {"path", "paths", "file", "files", "dir", "directory", "checkpoint", "weights"}
)
_PATH_OPTION_KEYS = frozenset(
    {
        "ccd_rdkit_cache",
        "components_cif",
        "feature_cache",
        "kalign_binary",
        "mols",
        "source",
        "template_obsolete_map",
        "template_release_dates",
    }
)
_SELF_CONTAINED_NATIVE_SUFFIXES = frozenset(
    {".cif", ".ent", ".fa", ".faa", ".fasta", ".mmcif", ".npz", ".pdb"}
)


def _path_option(key: str) -> bool:
    """Whether an open-ended backend option conventionally names a path."""
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    words = tuple(
        word
        for word in key.lower().replace("-", "_").split("_")
        if word
    )
    return key.lower() in _PATH_OPTION_KEYS or key.upper().startswith("FILE_") or bool(
        words and words[-1] in _PATH_OPTION_WORDS
    )


def _collect_option_paths(
    value: Any,
    *,
    key: str,
    path_context: bool,
    paths: list[Path],
) -> bool:
    """Collect local path-like options, refusing opaque dependency surfaces."""
    if key == "cli_args":
        # Arbitrary native argv can name files under arbitrary flags.  Without
        # understanding the native parser there is no complete dependency set.
        return value in (None, (), [])
    path_context = path_context or _path_option(key)
    if value is None:
        return True
    if isinstance(value, Path):
        paths.append(value)
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(child_key, str)
            and _collect_option_paths(
                child,
                key=child_key,
                path_context=path_context,
                paths=paths,
            )
            for child_key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(
            _collect_option_paths(
                child,
                key=key,
                path_context=path_context,
                paths=paths,
            )
            for child in value
        )
    if isinstance(value, str) and "://" in value:
        return False
    if not path_context:
        return True
    if not isinstance(value, str) or not value:
        return False
    paths.append(Path(value))
    return True


def _common_input_paths(
    request: PredictionRequest,
) -> tuple[list[Path], bool] | None:
    """Referenced MSA/template files in one common-schema job."""
    try:
        from foldjax.input import read_job_document

        document = read_job_document(request.input)
    except (OSError, ValueError):
        return None
    if not isinstance(document, Mapping):
        return None
    entities = document.get("entities")
    if not isinstance(entities, list):
        return None

    paths: list[Path] = []
    missing_alignment = False
    base = request.input.parent
    for entity in entities:
        if not isinstance(entity, Mapping):
            return None
        if entity.get("type") in {"protein", "rna"} and not entity.get(
            "unpaired_msa"
        ):
            missing_alignment = True
        for field in ("unpaired_msa", "paired_msa"):
            value = entity.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                return None
            path = Path(value.strip())
            paths.append(path if path.is_absolute() else base / path)
        templates = entity.get("templates")
        if templates is None:
            continue
        if not isinstance(templates, list):
            return None
        for template in templates:
            if not isinstance(template, Mapping):
                return None
            value = template.get("mmcif")
            if not isinstance(value, str) or not value.strip():
                return None
            path = Path(value.strip())
            paths.append(path if path.is_absolute() else base / path)
    return paths, missing_alignment


def _native_input_paths(request: PredictionRequest) -> list[Path] | None:
    """Best-effort complete local dependencies in a native JSON/YAML document.

    Plain FASTA, structure, feature-archive, and other single-file dialects are
    self-contained in the primary input digest.  Structured native dialects
    are open-ended, so inspect every local string that already names a file and
    require path-like fields to resolve rather than silently assuming they are
    inert.
    """
    try:
        from foldjax.input import read_job_document

        document = read_job_document(request.input)
    except (OSError, ValueError):
        if request.input.suffix.lower() in {".json", ".yaml", ".yml"}:
            return None
        return (
            []
            if request.input.suffix.lower() in _SELF_CONTAINED_NATIVE_SUFFIXES
            else None
        )

    paths: list[Path] = []
    base = request.input.parent

    def visit(value: Any, *, key: str = "", path_context: bool = False) -> bool:
        path_context = path_context or _path_option(key)
        if isinstance(value, Mapping):
            return all(
                isinstance(child_key, str)
                and visit(child, key=child_key, path_context=path_context)
                for child_key, child in value.items()
            )
        if isinstance(value, (list, tuple)):
            return all(
                visit(child, key=key, path_context=path_context)
                for child in value
            )
        if not isinstance(value, str) or not value.strip():
            return not path_context
        text = value.strip()
        if "://" in text:
            return False
        if text.startswith("FILE_"):
            text = text.removeprefix("FILE_").strip()
            path_context = True
            if not text:
                return False
        raw = Path(text)
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (base / raw, Path.cwd() / raw)
        )
        existing: list[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved not in existing:
                existing.append(resolved)
        # Native schemas are open-ended. A string that names any existing local
        # file or directory may be an input even when its key does not follow a
        # path naming convention, so an occasional extra dependency is safer
        # than reusing after a model-specific asset changed.
        if existing:
            paths.extend(existing)
            return True
        return not path_context

    return paths if visit(document) else None


def _uses_ccd_chemistry(request: PredictionRequest) -> bool:
    """Whether a structured job appears able to reach external CCD assets."""
    try:
        from foldjax.input import read_job_document

        document = read_job_document(request.input)
    except (OSError, ValueError):
        return False

    markers = {
        "ccd",
        "ccd_code",
        "ccd_codes",
        "ligand",
        "ligands",
        "modification",
        "modifications",
        "ptm",
        "ptms",
    }

    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            if str(value.get("type", "")).lower() == "ligand":
                return True
            for key, child in value.items():
                normalized = re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)
                ).lower()
                if normalized in markers and child not in (None, "", [], {}):
                    return True
                if visit(child):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(visit(child) for child in value)
        return False

    return visit(document)


def _implicit_ccd_assets(request: PredictionRequest) -> list[Path]:
    """Managed/environment CCD files Protenix-family featurization may read."""
    if request.model not in {"protenix", "opendde"} or not _uses_ccd_chemistry(
        request
    ):
        return []
    from foldjax.paths import assets_dir

    managed = assets_dir()
    rdkit_candidates = []
    configured_rdkit = os.environ.get("PROTENIX_CCD_RDKIT_MOL_FILE")
    if configured_rdkit:
        rdkit_candidates.append(Path(configured_rdkit))
    rdkit_candidates.append(managed / "components.cif.rdkit_mol.pkl")

    component_candidates = []
    configured_components = os.environ.get("PROTENIX_CCD_COMPONENTS_FILE")
    if configured_components:
        component_candidates.append(Path(configured_components))
    if configured_rdkit:
        component_candidates.append(Path(configured_rdkit).with_name("components.cif"))
    component_candidates.append(managed / "components.cif")

    assets: list[Path] = []
    for candidates in (rdkit_candidates, component_candidates):
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is not None:
            assets.append(selected)
    return assets


def _document_uses_key(request: PredictionRequest, wanted: str) -> bool:
    """Whether one structured input contains a non-empty semantic field."""
    try:
        from foldjax.input import read_job_document

        document = read_job_document(request.input)
    except (OSError, ValueError):
        return False

    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)
                ).lower()
                if normalized == wanted and child not in (None, "", [], {}):
                    return True
                if visit(child):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(visit(child) for child in value)
        return False

    return visit(document)


def _protenix_template_search_is_implicit(request: PredictionRequest) -> bool:
    """Whether native Protenix will resolve search hits through ambient assets.

    A ``templatesPath`` JSON is self-contained (it embeds mmCIF text), while an
    A3M/HHR hit list consults an environment/managed mmCIF database, release and
    obsolete metadata, and Kalign. Until that full open-ended search runtime is
    represented explicitly, such a run is deliberately non-resumable.
    """
    if request.model != "protenix" or request.input_format == "foldjax":
        return False
    try:
        from foldjax.input import read_job_document

        document = read_job_document(request.input)
    except (OSError, ValueError):
        return False

    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = re.sub(
                    r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)
                ).lower()
                if normalized == "templates_path" and isinstance(child, str):
                    if Path(child.strip()).suffix.lower() in {".a3m", ".hhr"}:
                        return True
                if visit(child):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(visit(child) for child in value)
        return False

    return visit(document)


def _boltz_weight_bundle(
    path: Path,
) -> tuple[list[Path], list[Path]] | None:
    """Files the native Boltz loader reads for one checkpoint base/path.

    The JSON sidecar carries scalar parameter leaves. Native exports always
    write it, but the loader tolerates its absence. Record that absence as an
    input state: an unchanged sidecar-free export can resume, while adding the
    sidecar later invalidates the manifest.
    """
    path = Path(path)
    candidates = [path]
    if path.suffix not in {".safetensors", ".npz"}:
        candidates = [path.with_suffix(".safetensors"), path.with_suffix(".npz")]
    weights = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weights is None:
        # A real Boltz run will fail before producing a manifest. Keep backend
        # overrides and third-party test doubles with unrelated formats usable.
        return [], []
    sidecar = weights.with_suffix(weights.suffix + ".json")
    if sidecar.exists():
        return ([weights, sidecar], []) if sidecar.is_file() else None
    return [weights], [sidecar]


def _implicit_weight_assets(
    request: PredictionRequest,
) -> tuple[list[Path], list[Path]] | None:
    """Companion checkpoint/config files read beyond ``request.weights``."""
    assert request.weights is not None
    paths: list[Path] = []
    missing: list[Path] = []
    if request.model == "boltz2":
        primary_bundle = _boltz_weight_bundle(request.weights)
        if primary_bundle is None:
            return None
        primary, primary_missing = primary_bundle
        try:
            recorded_weight = request.weights.resolve(strict=True)
        except (OSError, RuntimeError):
            recorded_weight = None
        paths.extend(
            path
            for path in primary
            if recorded_weight is None or path.resolve() != recorded_weight
        )
        missing.extend(primary_missing)
        affinity_requested = (
            request.stop_after != "trunk"
            and _document_uses_key(request, "affinity")
        )
        if affinity_requested:
            configured = request.options.get("affinity_weights")
            affinity = (
                Path(configured)
                if configured is not None
                else request.weights.with_name("boltz2_aff")
            )
            affinity_bundle = _boltz_weight_bundle(affinity)
            if affinity_bundle is None:
                return None
            bundle, bundle_missing = affinity_bundle
            if not bundle:
                return None
            paths.extend(bundle)
            missing.extend(bundle_missing)
    elif request.model == "esmfold2" and request.weights.is_file():
        root = request.weights.parent
        structure = root / "model.safetensors"
        config = root / "config.json"
        if not structure.is_file() or not config.is_file():
            return None
        paths.append(config)
        if structure.resolve() != request.weights.resolve():
            paths.append(structure)
        if (
            request.options.get("no_language_model") is not True
            and request.options.get("esmc_weights") is None
        ):
            esmc = root / "esmc"
            if not esmc.is_dir():
                return None
            paths.append(esmc)
    elif request.model == "protenix":
        model_name = request.options.get("model_name", "auto")
        if model_name == "auto":
            from foldjax.models.protenix.runtime_policy import (
                infer_model_name_from_path,
            )

            model_name = infer_model_name_from_path(request.weights)
        if isinstance(model_name, str) and (
            "_esm_" in model_name or "_ism_" in model_name
        ):
            # Managed profiles make this directory explicit and the generic
            # option scanner binds the complete tree. Legacy explicit weights
            # infer it from the primary checkpoint's parent instead.
            if request.options.get("esm_checkpoint_dir") is None:
                publisher_name = (
                    "esm2_t36_3B_UR50D_ism.pt"
                    if "_ism_" in model_name
                    else "esm2_t36_3B_UR50D.pt"
                )
                official = request.weights.parent / publisher_name
                stem = official.with_suffix("")
                candidates = (
                    stem.with_suffix(".safetensors"),
                    stem.with_suffix(".npz"),
                    official,
                )
                checkpoint = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    None,
                )
                if checkpoint is None:
                    return None
                paths.append(checkpoint)
    return paths, missing


def _input_dependencies(
    request: PredictionRequest,
) -> dict[str, Any]:
    """Stat/tree identities outside the primary input that preprocessing reads.

    Native dialects are intentionally conservative: FoldJAX cannot infer every
    file a model-specific document references.  Common jobs have a bounded
    schema, while OpenFold3 feature archives are self-contained in the primary
    input already hashed by the run manifest.
    """
    options = public_options(request.options)
    if not _public_options_are_complete(request, options):
        # Do not inspect raw secret-bearing options: even recording a path under
        # ``client_secret_file`` would disclose data the public manifest redacts.
        return {"verifiable": False, "artifacts": []}
    if request.options.get("use_msa_server") is True:
        return {"verifiable": False, "artifacts": []}

    paths: list[Path] = []
    if request.input_format == "foldjax":
        common = _common_input_paths(request)
        if common is None:
            return {"verifiable": False, "artifacts": []}
        common_paths, missing_alignment = common
        # A search result depends on mutable remote databases and service state,
        # neither of which a local run manifest can prove unchanged. A non-none
        # policy is still deterministic when every searchable chain supplied an
        # explicit alignment and preprocessing therefore performs no search.
        if request.msa != "none" and missing_alignment:
            return {"verifiable": False, "artifacts": []}
        paths.extend(common_paths)
    elif request.input_format != "openfold3-features":
        native_paths = _native_input_paths(request)
        if native_paths is None:
            return {"verifiable": False, "artifacts": []}
        paths.extend(native_paths)
    if _protenix_template_search_is_implicit(request):
        return {"verifiable": False, "artifacts": []}

    if not all(
        _collect_option_paths(
            value,
            key=key,
            path_context=False,
            paths=paths,
        )
        for key, value in request.options.items()
    ):
        return {"verifiable": False, "artifacts": []}
    if request.model == "boltz2" and request.options.get("mols") is None:
        # Boltz resolves this CCD tree implicitly from the checkpoint layout.
        # It changes featurization just as surely as the explicit ``mols``
        # option, so bind it whenever that conventional companion is present.
        assert request.weights is not None
        for candidate in (
            request.weights.parent / "mols",
            request.weights.parent.parent / "mols",
        ):
            if candidate.is_dir():
                paths.append(candidate)
                break
    weight_bundle = _implicit_weight_assets(request)
    if weight_bundle is None:
        return {"verifiable": False, "artifacts": []}
    weight_assets, missing_weight_assets = weight_bundle
    paths.extend(weight_assets)
    paths.extend(_implicit_ccd_assets(request))

    artifacts: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            resolved = Path(path).resolve(strict=True)
        except (OSError, RuntimeError):
            return {"verifiable": False, "artifacts": []}
        identity = path_stat_identity(resolved)
        if identity is None:
            return {"verifiable": False, "artifacts": []}
        artifact = {
            "path": str(resolved),
            **identity,
        }
        artifacts[str(resolved)] = artifact

    missing: set[str] = set()
    for path in missing_weight_assets:
        try:
            # Resolve the containing directory, while preserving the final
            # component that is intentionally absent. This keeps relative
            # checkpoint paths stable without requiring the missing file to
            # exist.
            candidate = path.parent.resolve(strict=True) / path.name
        except (OSError, RuntimeError):
            return {"verifiable": False, "artifacts": []}
        if candidate.exists():
            # It appeared between bundle discovery and manifest construction;
            # rerun rather than recording a racy dependency state.
            return {"verifiable": False, "artifacts": []}
        missing.add(str(candidate))
    return {
        "verifiable": True,
        "artifacts": [artifacts[path] for path in sorted(artifacts)],
        "missing": sorted(missing),
    }


def _artifact_path(path: Path, directory: Path | None) -> str:
    """Record an artifact relative to the manifest that owns it."""
    if directory is None:
        return str(Path(path).resolve())
    return os.path.relpath(Path(path).resolve(), Path(directory).resolve())


def _representation_record(
    result: PredictionResult, directory: Path | None
) -> dict[str, Any] | None:
    representations = result.representations
    if representations is None:
        return None
    from foldjax.models._representations import archive_headers, archive_identity

    entries = dict(representations.manifest)
    headers = archive_headers(representations.path)
    artifact = path_stat_identity(representations.path)
    if artifact is None:
        raise ValueError("representation archive has no stable stat identity")
    return {
        "model": representations.model,
        "path": _artifact_path(representations.path, directory),
        "names": list(entries),
        "identity": archive_identity(headers),
        "artifact": artifact,
        # Keep the shape/dtype/axis descriptions in the run manifest as well as
        # the representation sidecar.  A resumed result can then be rebuilt
        # without eagerly loading a quadratic pair array.
        "entries": entries,
    }


def _sample_record(sample: Any, directory: Path | None) -> dict[str, Any]:
    """Serialize one sample with a portable path and structure identity."""
    record = sample.summary()
    path = sample.structure_path
    record["structure_sha256"] = None
    if path is not None:
        record["structure_path"] = _artifact_path(Path(path), directory)
        record["structure_sha256"] = _digest(Path(path))
    return record


def matches_request(
    document: Mapping[str, Any], request: PredictionRequest, *, seed: int
) -> bool:
    """Whether ``document`` proves it is this exact scalar prediction request."""
    if document.get("schema") != MANIFEST_SCHEMA:
        return False
    if document.get("artifact_paths") != "manifest-relative":
        return False

    input_record = document.get("input")
    weights_record = document.get("weights")
    if not isinstance(input_record, Mapping) or not isinstance(
        weights_record, Mapping
    ):
        return False

    input_digest = _digest(request.input)
    if input_digest is None or request.weights is None:
        return False
    try:
        from foldjax.cache import weight_identity

        _label, weights_identity = weight_identity(request.weights)
    except (OSError, ValueError):
        return False

    options = public_options(request.options)
    if not _public_options_are_complete(request, options):
        return False
    if document.get("options_verifiable") is not True:
        return False
    recorded_dependencies = document.get("input_dependencies")
    if not isinstance(recorded_dependencies, Mapping):
        return False
    if recorded_dependencies.get("verifiable") is not True:
        return False
    current_dependencies = _input_dependencies(request)
    if current_dependencies.get("verifiable") is not True:
        return False

    expected_padding = (
        request.padding.summary() if request.padding is not None else None
    )
    expected_representations = (
        list(request.representations)
        if request.representations is not None
        else None
    )
    required = {
        "model",
        "seeds",
        "msa",
        "sampling",
        "options",
        "padding",
        "requested_representations",
        "stop_after",
        "representations",
        "samples",
        "input_dependencies",
    }
    if not required.issubset(document):
        return False
    if not {"path", "resolved_path", "format", "sha256"}.issubset(
        input_record
    ):
        return False
    if not {
        "identity",
        "profile",
        "kind",
        "stat_signature",
    }.issubset(weights_record):
        return False
    if not stat_identity_matches(
        request.weights,
        {
            "kind": weights_record["kind"],
            "stat_signature": weights_record["stat_signature"],
        },
    ):
        return False

    return all(
        (
            _exact_value(document["model"], request.model),
            _exact_value(input_record["path"], str(request.input)),
            _exact_value(
                input_record["resolved_path"], _resolved_path(request.input)
            ),
            _exact_value(input_record["format"], request.input_format),
            _exact_value(input_record["sha256"], input_digest),
            _exact_value(recorded_dependencies, current_dependencies),
            _exact_value(weights_record["identity"], weights_identity),
            _exact_value(weights_record["profile"], request.profile),
            _exact_value(document["seeds"], [seed]),
            _exact_value(document["msa"], request.msa),
            _exact_value(document["sampling"], request.sampling),
            _exact_value(document["options"], options),
            _exact_value(document["padding"], expected_padding),
            _exact_value(
                document["requested_representations"], expected_representations
            ),
            _exact_value(document["stop_after"], request.stop_after),
        )
    )


def describe_run(
    request: PredictionRequest,
    result: PredictionResult,
    *,
    native_input: Path | None = None,
    cost: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Build the manifest for one finished prediction.

    ``request`` is what the caller asked for. ``native_input`` is the backend
    dialect FoldJAX generated from it, when the job arrived in the common
    schema -- recorded alongside rather than instead, because that file lives
    inside the output directory being described and naming only it would say
    nothing about which job produced the directory.
    """
    from foldjax import __version__
    from foldjax.cache import runtime_profile, weight_identity
    from foldjax.output import best_sample

    weights = request.weights
    label = identity = None
    weights_stat: dict[str, Any] | None = None
    if weights is not None:
        label, identity = weight_identity(weights)
        weights_stat = path_stat_identity(weights)

    options = public_options(request.options)
    best = best_sample(result)
    if best is not None and best.get("structure_path"):
        best["structure_path"] = _artifact_path(
            Path(str(best["structure_path"])), directory
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "artifact_paths": (
            "manifest-relative" if directory is not None else "absolute"
        ),
        "foldjax": __version__,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": request.model,
        "input": {
            "path": str(request.input),
            "resolved_path": _resolved_path(request.input),
            "format": request.input_format,
            "sha256": _digest(request.input),
            "native": (
                _artifact_path(native_input, directory)
                if native_input is not None
                else None
            ),
        },
        "input_dependencies": _input_dependencies(request),
        "weights": {
            "path": str(weights) if weights else None,
            "profile": request.profile,
            "label": label,
            "identity": identity,
            **(weights_stat or {}),
        },
        "seeds": list(request.resolved_seeds),
        # Whether the alignments were the caller's or FoldJAX searched for them
        # changes the prediction, so it belongs with the knobs, not in a log.
        "msa": request.msa,
        "sampling": request.sampling,
        "options": options,
        # When redaction or non-JSON values erased information, the public
        # options cannot prove an exact match and resume must rerun.
        "options_verifiable": _public_options_are_complete(request, options),
        "padding": request.padding.summary() if request.padding is not None else None,
        "requested_representations": (
            list(request.representations)
            if request.representations is not None
            else None
        ),
        "stop_after": request.stop_after,
        "representations": _representation_record(result, directory),
        "runtime": runtime_profile(),
        # What the run cost. Recorded here because the alternative is measuring
        # it from outside, and from outside the only visible memory number is
        # JAX's preallocated pool -- the same figure whatever the job.
        "cost": dict(cost) if cost else None,
        "samples": [_sample_record(sample, directory) for sample in result.samples],
        # Which sample this model ranks first, by the score it ranks with. A
        # pointer rather than a second copy of the coordinates: the top-ranked
        # structure used to be written twice, and two files with one content
        # invite the question of which is authoritative.
        "best": best,
    }
    if result.shape_profile is not None:
        manifest["shape_profile"] = dict(result.shape_profile)
    return manifest


def write(
    request: PredictionRequest,
    result: PredictionResult,
    directory: Path,
    *,
    native_input: Path | None = None,
    cost: dict[str, Any] | None = None,
) -> Path | None:
    """Write the manifest, or return None if the directory cannot take it.

    A prediction that succeeded must not be turned into a failure because its
    provenance could not be recorded, so this reports rather than raises.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        with tempfile.TemporaryDirectory(
            prefix=".foldjax-manifest-", dir=directory
        ) as scratch:
            staged = Path(scratch) / MANIFEST_NAME
            staged.write_text(
                json.dumps(
                    describe_run(
                        request,
                        result,
                        native_input=native_input,
                        cost=cost,
                        directory=directory,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(staged, path)
        return path
    except Exception:  # noqa: BLE001 - provenance must never erase a valid result
        try:
            warnings.warn(
                "FoldJAX could not record run provenance; prediction output "
                "is preserved, but this run cannot be resumed safely.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:  # noqa: BLE001 - reporting cannot invalidate output
            # Warning filters may promote warnings to exceptions. Provenance
            # reporting must still never invalidate a successful prediction.
            pass
        return None
