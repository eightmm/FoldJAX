"""Build model features from an OpenFold3 query specification."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

# Every feature `predict` reads. Kept explicit so a featurizer that silently stops
# producing one fails a test rather than surfacing as a shape error much later.
MODEL_FEATURES = (
    "asym_id",
    "atom_mask",
    "atom_to_token_index",
    "deletion_mean",
    "deletion_value",
    "entity_id",
    "has_deletion",
    "is_atomized",
    "is_dna",
    "is_protein",
    "is_rna",
    "max_atom_per_token_mask",
    "msa",
    "msa_mask",
    "num_atoms_per_token",
    "profile",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_element",
    "ref_mask",
    "ref_pos",
    "ref_space_uid",
    "residue_index",
    "restype",
    "start_atom_index",
    "sym_id",
    "template_backbone_frame_mask",
    "template_distogram",
    "template_pseudo_beta_mask",
    "template_restype",
    "template_unit_vector",
    "token_bonds",
    "token_index",
    "token_mask",
)

# Optional runtime features consumed when present but not required from legacy
# archives.  ``template_padding_mask`` records storage provenance rather than
# template chemistry: rows already present in the source remain active (even
# when chemically empty, matching the released fixed-width input), while rows
# appended only for a serving bucket are excluded from the template reduction.
OPTIONAL_MODEL_FEATURES = ("template_padding_mask",)

# Produced by the dataset for bookkeeping, not consumed by the model.
_NON_FEATURES = frozenset(
    {"query_id", "seed", "repeated_sample", "valid_sample", "num_paired_seqs"}
)

_MISSING = (
    "OpenFold3 query featurization needs the chemistry/data dependencies "
    "(rdkit, pdbeccdutils, biotite, pandas, and gemmi). Install FoldJAX with "
    "`uv sync --extra openfold3-preprocess`, or featurize elsewhere and load "
    "the resulting .npz archive. PyTorch and Lightning are not required."
)


def has_atomized_tokens(features: Mapping[str, object]) -> bool:
    """Whether any active token needs a predicted-geometry neighbour search.

    Padding may carry arbitrary values outside the biological prefix, so the
    token mask is part of this host-side graph choice. Both arrays are already
    resident on the host when the inference configuration is built.
    """

    atomized = np.asarray(features["is_atomized"], dtype=bool)
    token_mask = np.asarray(features["token_mask"], dtype=bool)
    return bool(np.any(atomized & token_mask))


class OutputMetadata(NamedTuple):
    """Exact structure identity carried beside model features.

    Per-atom arrays contain real atoms only, in the same order as the nonzero
    ``atom_mask`` entries. Bond endpoints index those compact arrays. Everything
    is a NumPy scalar dtype that can be loaded with ``allow_pickle=False``.
    """

    atom_name: np.ndarray
    element: np.ndarray
    residue_name: np.ndarray
    residue_id: np.ndarray
    chain_id: np.ndarray
    entity_id: np.ndarray
    molecule_type_id: np.ndarray
    bonds: np.ndarray
    bond_type: np.ndarray


_BOND_TYPE_NAMES = frozenset(
    {
        "ANY",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "AROMATIC_SINGLE",
        "AROMATIC_DOUBLE",
        "AROMATIC_TRIPLE",
        "COORDINATION",
        "AROMATIC",
    }
)


def _query_set(spec: Mapping[str, Any] | str | Path):
    # The query schema is vendored at `.._upstream`, so this is an in-package import.
    try:
        from foldjax.models.openfold3._upstream.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
            InferenceQuerySet,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING) from error

    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    return InferenceQuerySet.model_validate(spec)


def _featurize_query(
    spec: Mapping[str, Any] | str | Path,
    *,
    query_id: str | None = None,
    seed: int = 0,
    ccd_file_path: str | Path | None = None,
    max_atoms_per_token: int = 23,
) -> tuple[dict[str, np.ndarray], OutputMetadata]:
    """Featurize one query from a specification.

    Args:
        spec: upstream's query JSON -- ``{"queries": {name: {"chains": [...]}}}`` --
            as a mapping or a path to it.
        query_id: which query to take; the only one when omitted.
        seed: deterministic seed for reference-conformer augmentation.
        ccd_file_path: a Chemical Component Dictionary file. ``None`` uses the copy
            biotite ships, so no download is required.
        max_atoms_per_token: the padded per-token atom slot count, 23 in the
            released config. Only used to build ``max_atom_per_token_mask``, which
            the dataset does not emit but the confidence heads need.

    Returns:
        Features with a leading batch axis of 1 and exact output metadata.

    Raises:
        ImportError: the OpenFold3 chemistry/data dependencies are not installed.
        KeyError: ``query_id`` is not in the specification.
    """
    if max_atoms_per_token != 23:
        raise ValueError(
            "released OpenFold3 requires max_atoms_per_token=23; "
            f"got {max_atoms_per_token}"
        )
    query_set = _query_set(spec)

    names = list(query_set.queries)
    if query_id is None:
        if len(names) != 1:
            raise KeyError(
                f"the specification holds {len(names)} queries {names}; pass "
                "query_id to choose one"
            )
        query_id = names[0]
    elif query_id not in query_set.queries:
        raise KeyError(f"{query_id!r} is not in the specification: {names}")

    query = query_set.queries[query_id]
    if query.covalent_bonds:
        raise ValueError(
            f"{query_id!r} declares covalent_bonds, but OpenFold3's vendored "
            "featurizer does not apply them; remove the bonds or use a backend "
            "that supports covalent connectivity"
        )

    try:
        from foldjax.models.openfold3._upstream.openfold3.projects.of3_all_atom.config.dataset_config_components import (  # noqa: E501
            MSASettings,
        )
        from foldjax.models.openfold3.data._numpy_featurization import (
            featurize_query_numpy,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from error

    msa_settings = MSASettings(subsample_main=False)
    _check_msa_filenames(query_id, query, msa_settings)
    try:
        raw = featurize_query_numpy(
            query,
            seed=seed,
            msa_settings=msa_settings,
            ccd_file_path=None if ccd_file_path is None else str(ccd_file_path),
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(_MISSING) from error

    features = {
        name: np.asarray(value)[None, ...]
        for name, value in raw.features.items()
        if name not in _NON_FEATURES
    }
    features["max_atom_per_token_mask"] = _max_atom_per_token_mask(
        features, max_atoms_per_token
    )

    missing = [name for name in MODEL_FEATURES if name not in features]
    if missing:
        raise RuntimeError(
            f"the OpenFold3 preprocessor did not produce {missing}; the model reads "
            "them, "
            "so this would fail later as a KeyError inside predict"
        )
    metadata = output_metadata_from_atom_array(raw.atom_array, features)
    return features, metadata


def featurize_query(
    spec: Mapping[str, Any] | str | Path,
    *,
    query_id: str | None = None,
    seed: int = 0,
    ccd_file_path: str | Path | None = None,
    max_atoms_per_token: int = 23,
) -> dict[str, np.ndarray]:
    """Featurize one query while preserving the established feature-only API."""
    features, _metadata = _featurize_query(
        spec,
        query_id=query_id,
        seed=seed,
        ccd_file_path=ccd_file_path,
        max_atoms_per_token=max_atoms_per_token,
    )
    return features


def featurize_query_with_metadata(
    spec: Mapping[str, Any] | str | Path,
    *,
    query_id: str | None = None,
    seed: int = 0,
    ccd_file_path: str | Path | None = None,
    max_atoms_per_token: int = 23,
) -> tuple[dict[str, np.ndarray], OutputMetadata]:
    """Featurize one query and retain exact atoms and covalent connectivity."""
    return _featurize_query(
        spec,
        query_id=query_id,
        seed=seed,
        ccd_file_path=ccd_file_path,
        max_atoms_per_token=max_atoms_per_token,
    )


def output_metadata_from_atom_array(
    atom_array: Any, features: Mapping[str, np.ndarray]
) -> OutputMetadata:
    """Extract the JAX-independent subset of upstream's ``AtomArray``.

    This runs only in the optional preprocessing process. The returned value has
    no Biotite objects, so archive loading and output writing remain Torch- and
    preprocessing-free.
    """
    required = (
        "atom_name",
        "element",
        "res_name",
        "res_id",
        "chain_id",
        "entity_id",
        "molecule_type_id",
    )
    missing = [name for name in required if not hasattr(atom_array, name)]
    if missing:
        raise ValueError(f"OpenFold3 atom array lacks output annotations: {missing}")

    bonds = (
        np.empty((0, 3), dtype=np.int64)
        if atom_array.bonds is None
        else np.asarray(atom_array.bonds.as_array())
    )
    if bonds.size == 0:
        endpoints = np.empty((0, 2), dtype=np.int64)
        bond_type = np.empty((0,), dtype="U20")
    else:
        if bonds.ndim != 2 or bonds.shape[1] != 3:
            raise ValueError(
                "OpenFold3 atom-array bonds must have shape (n_bond, 3)"
            )
        from biotite.structure import BondType

        endpoints = bonds[:, :2].astype(np.int64, copy=False)
        try:
            bond_type = np.asarray(
                [BondType(int(code)).name for code in bonds[:, 2]], dtype="U20"
            )
        except ValueError as error:
            raise ValueError("OpenFold3 atom array has an unknown bond type") from error

    metadata = OutputMetadata(
        atom_name=np.asarray([str(value) for value in atom_array.atom_name]),
        element=np.asarray([str(value) for value in atom_array.element]),
        residue_name=np.asarray([str(value) for value in atom_array.res_name]),
        residue_id=np.asarray(atom_array.res_id, dtype=np.int64),
        chain_id=np.asarray([str(value) for value in atom_array.chain_id]),
        entity_id=np.asarray(atom_array.entity_id, dtype=np.int64),
        molecule_type_id=np.asarray(atom_array.molecule_type_id, dtype=np.int8),
        bonds=endpoints,
        bond_type=bond_type,
    )
    return validate_output_metadata(metadata, features)


def _check_msa_filenames(query_id: str, query: Any, msa_settings: Any) -> None:
    """Reject alignment files whose name upstream would silently ignore.

    Upstream selects alignment files by *stem*: only stems present in
    ``MSASettings.max_seq_counts`` -- database names like ``colabfold_main`` or
    ``uniref90_hits`` -- are parsed, and anything else is skipped with no error.
    The result is an ``IndexError`` deep inside the MSA parser once nothing was
    collected, which gives no indication that a filename was the problem.
    """
    accepted = set(msa_settings.max_seq_counts)
    for chain in query.chains:
        for field in ("main_msa_file_paths", "paired_msa_file_paths"):
            for path in getattr(chain, field, None) or ():
                candidate = Path(path)
                if candidate.suffix == ".npz" and candidate.is_file():
                    continue
                if candidate.is_dir():
                    children = [
                        child
                        for child in candidate.iterdir()
                        if child.is_file()
                        and child.suffix in {".a3m", ".sto"}
                        and child.stem in accepted
                    ]
                    if children:
                        continue
                    raise ValueError(
                        f"{query_id}: {field} directory {str(candidate)!r} has "
                        "no supported .a3m/.sto file whose stem upstream "
                        f"accepts: {sorted(accepted)}"
                    )
                if (
                    candidate.suffix not in {".a3m", ".sto"}
                    or candidate.stem not in accepted
                ):
                    raise ValueError(
                        f"{query_id}: {field} entry {candidate.name!r} would be "
                        "ignored -- upstream parses only .a3m/.sto alignment "
                        "files by stem, and only these stems are accepted: "
                        f"{sorted(accepted)}. Rename the file, or pass a "
                        "directory containing an accepted alignment."
                    )


def _max_atom_per_token_mask(
    features: Mapping[str, np.ndarray], max_atoms_per_token: int
) -> np.ndarray:
    """Broadcast the token mask into padded per-token atom slots.

    ``[..., N_token * max_atoms_per_token]``, one block per token with the first
    ``num_atoms_per_token`` entries set. The confidence heads use it to scatter
    per-token predictions back onto atoms, and the dataset does not emit it.
    """
    token_mask = features["token_mask"]
    counts = features["num_atoms_per_token"].astype(np.int64)
    slots = np.arange(max_atoms_per_token)
    present = slots[None, None, :] < counts[..., None]
    return (present * token_mask[..., None].astype(present.dtype)).reshape(
        *token_mask.shape[:-1], -1
    ).astype(np.float32)


# Explicit shape semantics from upstream's feature schema. Negative axes keep the
# declarations independent of the leading batch dimension. Fixed-width channels
# such as the 32 residue/MSA bins, 39 distance bins and 119/128 element bins must
# never be inferred from their size: a real target can have exactly that many
# tokens or atoms.
_TOKEN_AXES: dict[str, tuple[int, ...]] = {
    "asym_id": (-1,),
    "deletion_mean": (-1,),
    "deletion_value": (-1,),
    "entity_id": (-1,),
    "has_deletion": (-1,),
    "has_frame": (-1,),
    "is_atomized": (-1,),
    "is_dna": (-1,),
    "is_ligand": (-1,),
    "is_protein": (-1,),
    "is_rna": (-1,),
    "mol_entity_id": (-1,),
    "mol_sym_component_id": (-1,),
    "mol_sym_id": (-1,),
    "mol_sym_token_index": (-1,),
    "msa": (-2,),
    "msa_mask": (-1,),
    "num_atoms_per_token": (-1,),
    "profile": (-2,),
    "residue_index": (-1,),
    "restype": (-2,),
    "start_atom_index": (-1,),
    "sym_id": (-1,),
    "template_backbone_frame_mask": (-1,),
    "template_distogram": (-3, -2),
    "template_pseudo_beta_mask": (-1,),
    "template_restype": (-2,),
    "template_unit_vector": (-3, -2),
    "token_bonds": (-2, -1),
    "token_index": (-1,),
    "token_mask": (-1,),
}

_ATOM_AXES: dict[str, tuple[int, ...]] = {
    "atom_mask": (-1,),
    "atom_to_token_index": (-1,),
    "ref_atom_name_chars": (-3,),
    "ref_charge": (-1,),
    "ref_element": (-2,),
    "ref_mask": (-1,),
    "ref_pos": (-2,),
    "ref_space_uid": (-1,),
}

_MSA_AXES: dict[str, tuple[int, ...]] = {
    "msa": (-3,),
    "msa_mask": (-2,),
    "has_deletion": (-2,),
    "deletion_value": (-2,),
}

_TEMPLATE_AXES: dict[str, tuple[int, ...]] = {
    "template_padding_mask": (-1,),
    "template_backbone_frame_mask": (-2,),
    "template_distogram": (-4,),
    "template_pseudo_beta_mask": (-2,),
    "template_restype": (-3,),
    "template_unit_vector": (-4,),
}

_UNSCHEMATIZED_MODEL_FEATURES = set(MODEL_FEATURES).difference(
    _TOKEN_AXES, _ATOM_AXES, {"max_atom_per_token_mask"}
)
if _UNSCHEMATIZED_MODEL_FEATURES:  # pragma: no cover - import-time drift guard
    raise RuntimeError(
        "OpenFold3 padding schema is missing model features: "
        f"{sorted(_UNSCHEMATIZED_MODEL_FEATURES)}"
    )


def _feature_padding(
    name: str,
    array: np.ndarray,
    *,
    tokens: int,
    atoms: int,
    msa_rows: int,
    templates: int,
    n_token: int,
    n_atom: int,
    n_msa: int,
    n_templates: int,
) -> list[tuple[int, int]]:
    """Return padding widths from the declared semantics of one feature."""
    widths = [(0, 0) for _ in array.shape]
    for kind, axes, current, target in (
        ("token", _TOKEN_AXES.get(name, ()), tokens, n_token),
        ("atom", _ATOM_AXES.get(name, ()), atoms, n_atom),
        ("MSA", _MSA_AXES.get(name, ()), msa_rows, n_msa),
        ("template", _TEMPLATE_AXES.get(name, ()), templates, n_templates),
    ):
        for declared_axis in axes:
            if not -array.ndim <= declared_axis < array.ndim:
                raise ValueError(
                    f"OpenFold3 feature {name!r} has shape {array.shape}, which "
                    f"has no declared {kind} axis {declared_axis}"
                )
            axis = declared_axis % array.ndim
            if array.shape[axis] != current:
                raise ValueError(
                    f"OpenFold3 feature {name!r} has {array.shape[axis]} entries "
                    f"on {kind} axis {axis}, expected {current}"
                )
            widths[axis] = (0, target - current)
    return widths


def pad_features(
    features: Mapping[str, np.ndarray],
    *,
    n_token: int,
    n_atom: int,
    n_msa: int | None = None,
    n_templates: int | None = None,
    max_atoms_per_token: int = 23,
) -> dict[str, np.ndarray]:
    """Pad every dynamic model axis so a compiled ``predict`` can be reused.

    Compilation is keyed on shapes, so predicting several targets without
    recompiling means padding token, atom, MSA-row and template axes to the same
    profile. Padding is zeros; token, atom, and MSA masks exclude their suffixes,
    and this helper adds a template-storage mask for newly appended rows.

    Raises:
        ValueError: the request is smaller than the features.
    """
    if max_atoms_per_token != 23:
        raise ValueError(
            "released OpenFold3 requires max_atoms_per_token=23; "
            f"got {max_atoms_per_token}"
    )
    tokens = features["token_mask"].shape[-1]
    atoms = features["atom_mask"].shape[-1]

    def declared_size(
        schemas: Mapping[str, tuple[int, ...]], *, kind: str
    ) -> int:
        sizes = {
            int(np.asarray(features[name]).shape[axes[0]])
            for name, axes in schemas.items()
            if name in features and axes
        }
        if len(sizes) > 1:
            raise ValueError(
                f"OpenFold3 {kind} features disagree on their current size: "
                f"{sorted(sizes)}"
            )
        return next(iter(sizes), 0)

    # The full model schema always contains both axes.  Keeping them optional
    # here preserves the long-standing low-level helper contract used by
    # preprocessing tools and tests that only exercise token/atom padding.
    msa_rows = declared_size(_MSA_AXES, kind="MSA")
    templates = declared_size(_TEMPLATE_AXES, kind="template")
    n_msa = msa_rows if n_msa is None else n_msa
    n_templates = templates if n_templates is None else n_templates
    if (
        n_token < tokens
        or n_atom < atoms
        or n_msa < msa_rows
        or n_templates < templates
    ):
        raise ValueError(
            "cannot pad down: have "
            f"{tokens} tokens / {atoms} atoms / {msa_rows} MSA rows / "
            f"{templates} templates, asked for {n_token} / {n_atom} / "
            f"{n_msa} / {n_templates}"
        )

    # Only declared semantic axes are padded. Shape matching is ambiguous for
    # perfectly ordinary inputs: n_token can equal n_atom, or either can equal a
    # fixed channel width such as 32, 39 or 119.
    source = dict(features)
    if templates and "template_padding_mask" not in source:
        template_name = next(name for name in _TEMPLATE_AXES if name in source)
        template_array = np.asarray(source[template_name])
        template_axis = _TEMPLATE_AXES[template_name][0] % template_array.ndim
        # Every row already stored by the caller retains the exact old
        # reduction semantics; only suffix rows introduced below receive zero
        # weight. Derive the leading shape from any declared template feature
        # so low-level/legacy feature subsets do not require template_restype.
        source["template_padding_mask"] = np.ones(
            template_array.shape[: template_axis + 1], dtype=np.float32
        )

    padded = {}
    for name, value in source.items():
        array = np.asarray(value)
        widths = _feature_padding(
            name,
            array,
            tokens=tokens,
            atoms=atoms,
            msa_rows=msa_rows,
            templates=templates,
            n_token=n_token,
            n_atom=n_atom,
            n_msa=n_msa,
            n_templates=n_templates,
        )
        padded[name] = np.pad(array, widths)

    # Rebuilt rather than padded: its length is n_token * max_atoms_per_token, so
    # padding the flat axis would interleave real and padding slots.
    padded["max_atom_per_token_mask"] = _max_atom_per_token_mask(
        padded, max_atoms_per_token
    )
    return padded


def normalize_asym_ids(
    features: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int | None]:
    """Normalize real-token chain IDs and derive the static multimer size.

    OpenFold3's interface and clash heads iterate over ``range(n_chain)``, so
    sparse or 1-based IDs silently omit chains. Padding IDs are not chemistry and
    must not increase that static bound. The returned bound is explicit for
    every non-empty input so inference can specialize the monomer confidence
    graph; ``None`` remains reserved for an unknown or empty chain set.
    """
    token_mask = np.asarray(features["token_mask"]) > 0
    asym_id = np.asarray(features["asym_id"])
    if asym_id.shape != token_mask.shape:
        raise ValueError(
            "OpenFold3 asym_id and token_mask must have the same shape, got "
            f"{asym_id.shape} and {token_mask.shape}"
        )

    normalized = np.zeros_like(asym_id)
    chain_ids = np.unique(asym_id[token_mask])
    for chain, original in enumerate(chain_ids):
        normalized[token_mask & (asym_id == original)] = chain

    result = dict(features)
    result["asym_id"] = normalized
    n_chain = int(chain_ids.size)
    return result, n_chain if n_chain > 0 else None


# The representative-atom table travels with the features under this prefix. It is
# chemistry, not a feature, but inference needs it and deriving it requires
# upstream's tables -- so writing it here is what makes the .npz self-sufficient.
CHEMISTRY_PREFIX = "chemistry."
OUTPUT_PREFIX = "output."


def validate_output_metadata(
    metadata: OutputMetadata | Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
) -> OutputMetadata:
    """Validate and normalize the exact structure-writing sidecar."""
    if isinstance(metadata, OutputMetadata):
        values = metadata._asdict()
    elif isinstance(metadata, Mapping):
        values = dict(metadata)
    else:
        raise TypeError("OpenFold3 output metadata must be a mapping")

    expected = set(OutputMetadata._fields)
    missing = sorted(expected.difference(values))
    extra = sorted(set(values).difference(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(
            "OpenFold3 output metadata is incomplete: " + "; ".join(details)
        )

    if "atom_mask" not in features:
        raise ValueError("OpenFold3 output metadata needs the atom_mask feature")
    atom_mask = np.asarray(features["atom_mask"])
    real_atoms = int(np.count_nonzero(atom_mask > 0))

    arrays = {name: np.asarray(values[name]) for name in OutputMetadata._fields}
    object_fields = [
        name for name, value in arrays.items() if value.dtype.kind == "O"
    ]
    if object_fields:
        raise ValueError(
            "OpenFold3 output metadata cannot use pickle/object arrays: "
            f"{object_fields}"
        )

    atom_fields = (
        "atom_name",
        "element",
        "residue_name",
        "residue_id",
        "chain_id",
        "entity_id",
        "molecule_type_id",
    )
    for name in atom_fields:
        if arrays[name].shape != (real_atoms,):
            raise ValueError(
                f"OpenFold3 output metadata {name} must have shape "
                f"({real_atoms},), got {arrays[name].shape}"
            )

    for name in ("atom_name", "element", "residue_name", "chain_id"):
        value = arrays[name]
        if value.dtype.kind not in "US":
            raise ValueError(f"OpenFold3 output metadata {name} must contain strings")
        if any(not str(item).strip() for item in value):
            raise ValueError(
                f"OpenFold3 output metadata {name} contains an empty value"
            )

    for name in ("residue_id", "entity_id", "molecule_type_id"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"OpenFold3 output metadata {name} must contain integers")
    if not np.isin(arrays["molecule_type_id"], (0, 1, 2, 3)).all():
        raise ValueError(
            "OpenFold3 output metadata molecule_type_id must use 0=protein, "
            "1=RNA, 2=DNA, or 3=ligand"
        )
    if np.any(arrays["entity_id"] < 1):
        raise ValueError("OpenFold3 output metadata entity_id must be positive")
    for chain_id in np.unique(arrays["chain_id"]):
        chain = arrays["chain_id"] == chain_id
        if (
            np.unique(arrays["entity_id"][chain]).size != 1
            or np.unique(arrays["molecule_type_id"][chain]).size != 1
        ):
            raise ValueError(
                f"OpenFold3 output metadata chain {chain_id!r} has "
                "inconsistent entity provenance"
            )

    def real_feature(name: str, ndim: int) -> np.ndarray:
        value = np.asarray(features[name])
        while value.ndim > ndim:
            value = value[0]
        return value

    keep = real_feature("atom_mask", 1) > 0
    chars = real_feature("ref_atom_name_chars", 3)
    if chars.shape[1:] != (4, 64) or chars.shape[0] != keep.size:
        raise ValueError(
            "OpenFold3 output metadata cannot align to ref_atom_name_chars"
        )
    decoded_names = np.asarray(
        [
            "".join(chr(int(code) + 32) for code in row).strip()
            for row in np.argmax(chars, axis=-1)[keep]
        ]
    )
    if not np.array_equal(decoded_names, arrays["atom_name"].astype(str)):
        raise ValueError(
            "OpenFold3 output metadata atom order does not match model features"
        )

    owners = real_feature("atom_to_token_index", 1).astype(np.int64)[keep]
    residue_id = real_feature("residue_index", 1).astype(np.int64)[owners]
    entity_id = real_feature("entity_id", 1).astype(np.int64)[owners]
    molecule_type = np.full(owners.shape, 3, dtype=np.int8)
    for feature_name, type_id in (("is_protein", 0), ("is_rna", 1), ("is_dna", 2)):
        typed = real_feature(feature_name, 1).astype(bool)[owners]
        molecule_type[typed] = type_id
    if not np.array_equal(residue_id, arrays["residue_id"]):
        raise ValueError(
            "OpenFold3 output metadata residue IDs do not match model features"
        )
    if not np.array_equal(entity_id, arrays["entity_id"]):
        raise ValueError(
            "OpenFold3 output metadata entity IDs do not match model features"
        )
    if not np.array_equal(molecule_type, arrays["molecule_type_id"]):
        raise ValueError(
            "OpenFold3 output metadata molecule types do not match model features"
        )

    def first_occurrence_groups(values: np.ndarray) -> np.ndarray:
        groups: dict[str, int] = {}
        return np.asarray(
            [groups.setdefault(str(value), len(groups)) for value in values]
        )

    asym_id = real_feature("asym_id", 1)[owners]
    if not np.array_equal(
        first_occurrence_groups(asym_id),
        first_occurrence_groups(arrays["chain_id"]),
    ):
        raise ValueError(
            "OpenFold3 output metadata chain grouping does not match model features"
        )

    bonds = arrays["bonds"]
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError(
            "OpenFold3 output metadata bonds must have shape (n_bond, 2)"
        )
    if not np.issubdtype(bonds.dtype, np.integer):
        raise ValueError("OpenFold3 output metadata bonds must contain integers")
    bond_type = arrays["bond_type"]
    if bond_type.shape != (bonds.shape[0],) or bond_type.dtype.kind not in "US":
        raise ValueError(
            "OpenFold3 output metadata bond_type must be one string per bond"
        )
    unknown_types = sorted(set(map(str, bond_type)).difference(_BOND_TYPE_NAMES))
    if unknown_types:
        raise ValueError(
            f"OpenFold3 output metadata has unknown bond types: {unknown_types}"
        )
    if bonds.size:
        if np.any((bonds < 0) | (bonds >= real_atoms)):
            raise ValueError(
                "OpenFold3 output metadata contains an out-of-range bond endpoint"
            )
        if np.any(bonds[:, 0] == bonds[:, 1]):
            raise ValueError("OpenFold3 output metadata contains a self bond")
        canonical = np.sort(bonds.astype(np.int64, copy=False), axis=1)
        if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
            raise ValueError("OpenFold3 output metadata contains duplicate bonds")

    return OutputMetadata(
        atom_name=arrays["atom_name"],
        element=arrays["element"],
        residue_name=arrays["residue_name"],
        residue_id=arrays["residue_id"].astype(np.int64, copy=False),
        chain_id=arrays["chain_id"],
        entity_id=arrays["entity_id"].astype(np.int64, copy=False),
        molecule_type_id=arrays["molecule_type_id"].astype(np.int8, copy=False),
        bonds=bonds.astype(np.int64, copy=False),
        bond_type=bond_type,
    )


def save_features(
    features: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    representative_atoms: object | None = None,
    output_metadata: OutputMetadata | Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write features to ``.npz`` so inference can run with no torch installed.

    The representative-atom table is written alongside them. Exact atom identity
    and bonds are included when ``output_metadata`` is supplied. Both sidecars
    live under reserved prefixes and are removed before arrays reach JAX.
    """
    target = Path(path)
    # ``numpy.savez`` only recognizes the lower-case suffix. Treating
    # ``FEATURES.NPZ`` as complete would make it create ``FEATURES.NPZ.npz``
    # while this function returned the non-existent original path.
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in features.items()}

    if representative_atoms is None:
        from foldjax.models.openfold3.bridge.chemistry import (
            representative_atom_table,
        )

        representative_atoms = representative_atom_table()
    for name, value in zip(
        type(representative_atoms)._fields, representative_atoms, strict=True
    ):
        payload[f"{CHEMISTRY_PREFIX}{name}"] = np.asarray(value)

    if output_metadata is not None:
        validated_metadata = validate_output_metadata(output_metadata, features)
        for name, value in validated_metadata._asdict().items():
            payload[f"{OUTPUT_PREFIX}{name}"] = np.asarray(value)

    # Publish atomically. Feature archives can be several GiB; a killed
    # featurizer must not leave a plausible-looking truncated file behind, and
    # replacement must replace an existing symlink rather than follow it.
    with tempfile.NamedTemporaryFile(
        prefix=".foldjax-openfold3-features-",
        suffix=".npz",
        dir=target.parent,
        delete=False,
    ) as temporary:
        staged = Path(temporary.name)
    try:
        np.savez(staged, **payload)
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target


def split_chemistry(
    loaded: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], object | None]:
    """Separate features from a chemistry table stored by :func:`save_features`.

    Returns ``(features, table_or_None)``. ``None`` means the file predates this
    format or has an incomplete chemistry payload; inference callers should
    refuse it and ask for a new self-contained archive.
    """
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    features = {
        name: value
        for name, value in loaded.items()
        if not name.startswith((CHEMISTRY_PREFIX, OUTPUT_PREFIX))
    }
    stored = {
        name[len(CHEMISTRY_PREFIX) :]: value
        for name, value in loaded.items()
        if name.startswith(CHEMISTRY_PREFIX)
    }
    if set(stored) != set(RepresentativeAtomTable._fields):
        return features, None
    if any(
        np.asarray(stored[name]).shape != (32,)
        or not np.issubdtype(np.asarray(stored[name]).dtype, np.number)
        or not np.isfinite(np.asarray(stored[name])).all()
        for name in RepresentativeAtomTable._fields
    ):
        return features, None
    return features, RepresentativeAtomTable(
        **{name: stored[name] for name in RepresentativeAtomTable._fields}
    )


def split_output_metadata(
    loaded: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
) -> OutputMetadata | None:
    """Decode the optional structure-writing sidecar from an archive payload."""
    stored = {
        name[len(OUTPUT_PREFIX) :]: value
        for name, value in loaded.items()
        if name.startswith(OUTPUT_PREFIX)
    }
    if not stored:
        return None
    return validate_output_metadata(stored, features)


def load_feature_archive(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], object, OutputMetadata | None]:
    """Load model features, chemistry, and optional exact output metadata."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    features, table = split_chemistry(payload)
    missing = [name for name in MODEL_FEATURES if name not in features]
    if missing:
        raise ValueError(f"OpenFold3 feature archive is missing: {missing}")
    if table is None:
        raise ValueError(
            "OpenFold3 feature archive has no embedded chemistry table; "
            "re-create it with openfold3-jax-featurize"
        )
    from foldjax.models.openfold3.data.validation import validate_features

    validate_features(features)
    metadata = split_output_metadata(payload, features)
    return features, table, metadata


def load_features(path: str | Path) -> tuple[dict[str, np.ndarray], object]:
    """Load one self-contained, inference-ready OpenFold3 feature archive.

    This compatibility API retains its established two-item return. Call
    :func:`load_feature_archive` when exact output metadata is also needed. It
    deliberately refuses archives without embedded chemistry: rebuilding that
    table from an archive alone would guess information fixed by the raw-job
    preprocessing run.
    """
    features, table, _metadata = load_feature_archive(path)
    return features, table


#: The four features carrying one row per alignment. ``msa`` is
#: ``[batch, rows, tokens, 32]``; the other three are ``[batch, rows, tokens]``.
MSA_ROW_FEATURES = ("msa", "msa_mask", "has_deletion", "deletion_value")


def subsample_msa_rows(
    features: Mapping[str, np.ndarray], no_subsampled: int | None
) -> dict[str, np.ndarray]:
    """Keep ``no_subsampled`` alignment rows, choosing them as upstream does.

    Upstream applies this inside the network -- ``MSAModuleEmbedder._subsample_all_msa``
    -- but selecting rows is a data-pipeline decision, so it happens here, on the host,
    before the features ever reach the device. At 3012 tokens a full 16384-row
    alignment is several GiB; subsampling on device would commit all of it to the
    memory pool first, which is the thing being avoided.

    Rows whose mask is entirely zero are "invalid" and serve only as filler. Upstream
    keeps a *random* subset of the valid rows when there are more than
    ``no_subsampled`` of them, which no PRNG choice here can reproduce -- upstream is
    not reproducible against itself either, since it draws a fresh ``randperm`` per
    call. This keeps the first ``no_subsampled`` valid rows in the order the search
    produced them, which implements the same rule deterministically. Below the
    threshold the two agree exactly: upstream keeps every valid row and pads with
    all-masked rows, and appending an all-masked row changes nothing.

    Args:
        features: batched model features. Needs the four in ``MSA_ROW_FEATURES``.
        no_subsampled: rows to keep. ``None``, or a value at or above the row count,
            returns the features unchanged.

    Returns:
        A new mapping with the four MSA features subsampled; other keys are shared.
    """
    if no_subsampled is None:
        return dict(features)
    if no_subsampled < 1:
        raise ValueError(f"msa_depth must be at least 1; got {no_subsampled}")
    mask = features["msa_mask"]
    rows = mask.shape[1]
    if rows <= no_subsampled:
        return dict(features)

    # Rank valid rows before invalid ones, preserving the search's order within each
    # group; a stable sort of the negated flag does exactly that.
    valid = np.asarray(mask).sum(axis=-1)[0] > 0
    order = np.argsort(~valid, kind="stable")[:no_subsampled]

    out = dict(features)
    for name in MSA_ROW_FEATURES:
        array = out.get(name)
        if array is not None:
            out[name] = array[:, order]
    return out


def collapse_identical_templates(
    features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Keep one row of a template axis whose rows are all the same template.

    The template stack runs once per template and the results are averaged, so
    ``n`` copies of one template cost ``n`` times the work to produce the value
    one copy already gives. That is not a hypothetical: a query with no
    templates is featurized as the released fixed-width axis of *four* empty
    ones (``_empty_template_features``, mirroring upstream's ``n_templates: 4``),
    every one of them the same all-gap restype and all-zero distogram, masks and
    unit vectors. Nothing downstream distinguishes them either --
    ``template_pair_embedder`` reads no per-template index -- so the stack
    computes the same tensor four times and divides the sum by four.

    Measured on the 1003-token bench target with no templates, the four rows
    *together* are 645.2 MiB of entry arguments, 599 MiB of it a zero
    distogram, and 5.68 GiB at 3012 tokens. Those are totals, not savings:
    collapsing to one row keeps a quarter, so it saves 484 MiB and 4.26 GiB
    respectively, and drops three of the four template-stack evaluations in
    every recycling cycle.

    End to end that is 415 MiB off the peak at 1003 tokens with no time
    difference the measurement can resolve, and 6.3 GiB (-10.4%) with 16.3 s
    (-6.8%) at 3012, the latter confirmed with the arms run in both orders so
    it is not a warming card. The written structures and confidences are
    byte-identical at both sizes, including a four-chain complex, against a
    rerun floor of exactly zero.

    This is a divergence from upstream, which does compute the duplicates. It is
    a divergence in work, not in value: with ``n`` identical rows the reduction
    is ``n * x / n``, which is ``x`` up to the rounding of the summation --
    around an ulp at the point the template update is added to the pair
    representation, far below the spread between two runs of the same code.

    Args:
        features: batched model features. Templates are found through
            ``_TEMPLATE_AXES``, so a feature subset without them is returned
            unchanged.

    Returns:
        A new mapping with every template-axis feature reduced to its first row
        when they are all equal; otherwise the same entries, shared not copied.
    """
    present = {
        name: (np.asarray(features[name]), axes[0])
        for name, axes in _TEMPLATE_AXES.items()
        if name in features
    }
    if not present:
        return dict(features)

    rows = {array.shape[axis % array.ndim] for array, axis in present.values()}
    if len(rows) != 1:
        raise ValueError(
            f"template features disagree on their row count: {sorted(rows)}"
        )
    (n_templates,) = rows
    if n_templates <= 1:
        return dict(features)

    # A padding mask records which rows are real, so rows that agree in content
    # can still differ in whether they participate. Only the all-active case is
    # collapsed; anything else keeps its storage and its weights.
    padding = present.get("template_padding_mask")
    if padding is not None:
        mask = padding[0]
        if not np.array_equal(mask, np.ones_like(mask)):
            return dict(features)

    # Smallest first, so a distinct pair of masks answers the question before
    # the hundreds of megabytes of distogram are ever compared.
    ordered = sorted(present.items(), key=lambda item: item[1][0].nbytes)
    for _name, (array, axis) in ordered:
        moved = np.moveaxis(array, axis % array.ndim, 0)
        first = moved[0]
        if not all(
            np.array_equal(first, moved[index]) for index in range(1, n_templates)
        ):
            return dict(features)

    out = dict(features)
    for name, (array, axis) in present.items():
        out[name] = np.take(array, [0], axis=axis % array.ndim)
    return out
