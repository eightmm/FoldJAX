"""Build model features from an OpenFold3 query specification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

# Produced by the dataset for bookkeeping, not consumed by the model.
_NON_FEATURES = frozenset(
    {"query_id", "seed", "repeated_sample", "valid_sample", "num_paired_seqs"}
)

_MISSING = (
    "featurization needs the 'openfold3-preprocess' extra, which installs what "
    "the vendored data pipeline imports (torch, lightning, rdkit, pdbeccdutils, "
    "lmdb, biotite). Install it with `uv sync --extra openfold3-preprocess`, or "
    "featurize elsewhere and load the resulting .npz -- the inference path "
    "itself needs none of it."
)


def _query_set(spec: Mapping[str, Any] | str | Path):
    # The pipeline is vendored at `.._upstream`, so this is an in-package import.
    # It still fails without the `openfold3-preprocess` extra, because that code
    # imports torch and lightning -- which is why the guard stays.
    try:
        from foldjax.models.openfold3._upstream.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
            InferenceQuerySet,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING) from error

    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    return InferenceQuerySet.model_validate(spec)


def featurize_query(
    spec: Mapping[str, Any] | str | Path,
    *,
    query_id: str | None = None,
    seed: int = 42,
    ccd_file_path: str | Path | None = None,
    max_atoms_per_token: int = 23,
) -> dict[str, np.ndarray]:
    """Featurize one query from a specification.

    Args:
        spec: upstream's query JSON -- ``{"queries": {name: {"chains": [...]}}}`` --
            as a mapping or a path to it.
        query_id: which query to take; the only one when omitted.
        seed: passed through to the dataset, which uses it for MSA sampling.
        ccd_file_path: a Chemical Component Dictionary file. ``None`` uses the copy
            biotite ships, so no download is required.
        max_atoms_per_token: the padded per-token atom slot count, 23 in the
            released config. Only used to build ``max_atom_per_token_mask``, which
            the dataset does not emit but the confidence heads need.

    Returns:
        Features as numpy arrays with a leading batch axis of 1.

    Raises:
        ImportError: the ``openfold3-preprocess`` extra is not installed.
        KeyError: ``query_id`` is not in the specification.
    """
    query_set = _query_set(spec)

    from foldjax.models.openfold3._upstream.openfold3.core.data.framework.single_datasets.inference import (  # noqa: E501
        InferenceDataset,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.pipelines.preprocessing.template import (  # noqa: E501
        TemplatePreprocessorSettings,
    )
    from foldjax.models.openfold3._upstream.openfold3.projects.of3_all_atom.config.dataset_configs import (  # noqa: E501
        InferenceJobConfig,
        MSASettings,
        TemplateSettings,
    )

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

    # One query at a time: the dataset orders its cache by sequence length, so
    # index 0 of a multi-query set is not necessarily the one that was asked for.
    single = type(query_set).model_validate(
        {"queries": {query_id: query_set.queries[query_id].model_dump()}}
    )
    msa_settings = MSASettings(subsample_main=False)
    _check_msa_filenames(query_set, msa_settings)

    dataset = InferenceDataset(
        InferenceJobConfig(
            query_set=single,
            seeds=[seed],
            ccd_file_path=ccd_file_path,
            msa=msa_settings,
            template=TemplateSettings(take_top_k=True),
            template_preprocessor_settings=TemplatePreprocessorSettings(),
        )
    )
    raw = dataset[0]

    features = {
        name: np.asarray(value.detach().cpu())[None, ...]
        for name, value in raw.items()
        if name not in _NON_FEATURES and hasattr(value, "detach")
    }
    # Upstream catches per-query exceptions, prints the traceback and returns a
    # sample with no tensors in it, so a failure arrives here as an empty dict
    # rather than as an exception. Without this the first missing feature surfaces
    # as ``KeyError: 'token_mask'`` several frames away from the real cause, which
    # is printed above and easy to mistake for a warning.
    if "token_mask" not in features:
        raise RuntimeError(
            f"upstream's dataset produced no features for {query_id!r}. It logs the "
            "cause above and then returns an empty sample. A common one is a ligand "
            "whose CCD entry rdkit refuses to sanitize (metal clusters such as CLF "
            "or ICS): drop that ligand from the query, or supply it as SMILES."
        )
    features["max_atom_per_token_mask"] = _max_atom_per_token_mask(
        features, max_atoms_per_token
    )

    missing = [name for name in MODEL_FEATURES if name not in features]
    if missing:
        raise RuntimeError(
            f"upstream's dataset did not produce {missing}; the model reads them, "
            "so this would fail later as a KeyError inside predict"
        )
    return features


def _check_msa_filenames(query_set: Any, msa_settings: Any) -> None:
    """Reject alignment files whose name upstream would silently ignore.

    Upstream selects alignment files by *stem*: only stems present in
    ``MSASettings.max_seq_counts`` -- database names like ``colabfold_main`` or
    ``uniref90_hits`` -- are parsed, and anything else is skipped with no error.
    The result is an ``IndexError`` deep inside the MSA parser once nothing was
    collected, which gives no indication that a filename was the problem.
    """
    accepted = set(msa_settings.max_seq_counts)
    for query_id, query in query_set.queries.items():
        for chain in query.chains:
            for field in ("main_msa_file_paths", "paired_msa_file_paths"):
                for path in getattr(chain, field, None) or ():
                    candidate = Path(path)
                    # Directories are scanned by upstream, so their contents are
                    # filtered by the same rule but not knowable from here.
                    if candidate.is_dir():
                        continue
                    if candidate.stem not in accepted:
                        raise ValueError(
                            f"{query_id}: {field} entry {candidate.name!r} would be "
                            "ignored -- upstream parses alignment files by stem, and "
                            f"only these are accepted: {sorted(accepted)}. Rename the "
                            "file, or pass a directory."
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


def pad_features(
    features: Mapping[str, np.ndarray],
    *,
    n_token: int,
    n_atom: int,
    max_atoms_per_token: int = 23,
) -> dict[str, np.ndarray]:
    """Pad token- and atom-axes so a compiled ``predict`` can be reused.

    Compilation is keyed on shapes, so predicting several targets without
    recompiling means padding them all to the same bucket. Padding is zeros, which
    the masks already exclude: ``token_mask`` and ``atom_mask`` are padded with
    zeros too.

    Raises:
        ValueError: the request is smaller than the features.
    """
    tokens = features["token_mask"].shape[-1]
    atoms = features["atom_mask"].shape[-1]
    if n_token < tokens or n_atom < atoms:
        raise ValueError(
            f"cannot pad down: have {tokens} tokens / {atoms} atoms, "
            f"asked for {n_token} / {n_atom}"
        )

    # Which axes of each feature are token- and atom-indexed cannot be guessed from
    # rank alone (token_bonds is token x token, ref_pos is atom x 3), so the axis
    # sizes are matched instead.
    padded = {}
    for name, value in features.items():
        array = np.asarray(value)
        widths = []
        for axis, size in enumerate(array.shape):
            if size == tokens and name != "max_atom_per_token_mask":
                widths.append((0, n_token - tokens))
            elif size == atoms:
                widths.append((0, n_atom - atoms))
            else:
                widths.append((0, 0))
            del axis
        padded[name] = np.pad(array, widths)

    # Rebuilt rather than padded: its length is n_token * max_atoms_per_token, so
    # padding the flat axis would interleave real and padding slots.
    padded["max_atom_per_token_mask"] = _max_atom_per_token_mask(
        padded, max_atoms_per_token
    )
    return padded


# The representative-atom table travels with the features under this prefix. It is
# chemistry, not a feature, but inference needs it and deriving it requires
# upstream's tables -- so writing it here is what makes the .npz self-sufficient.
CHEMISTRY_PREFIX = "chemistry."


def save_features(
    features: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    representative_atoms: object | None = None,
) -> Path:
    """Write features to ``.npz`` so inference can run with no torch installed.

    The representative-atom table is written alongside them. Without it the
    inference path would still have to import upstream to rebuild the table, which
    would defeat the split this file exists to maintain.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in features.items()}

    if representative_atoms is None:
        try:
            from foldjax.models.openfold3.bridge.chemistry import (
                representative_atom_table,
            )

            representative_atoms = representative_atom_table()
        except ImportError:
            representative_atoms = None
    if representative_atoms is not None:
        for name, value in zip(
            type(representative_atoms)._fields, representative_atoms, strict=True
        ):
            payload[f"{CHEMISTRY_PREFIX}{name}"] = np.asarray(value)

    np.savez(target, **payload)
    return target


def split_chemistry(
    loaded: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], object | None]:
    """Separate features from a chemistry table stored by :func:`save_features`.

    Returns ``(features, table_or_None)``. ``None`` means the file predates this
    and the caller has to rebuild the table from upstream.
    """
    from foldjax.models.openfold3.models.representative_atoms import (
        RepresentativeAtomTable,
    )

    features = {
        name: value
        for name, value in loaded.items()
        if not name.startswith(CHEMISTRY_PREFIX)
    }
    stored = {
        name[len(CHEMISTRY_PREFIX) :]: value
        for name, value in loaded.items()
        if name.startswith(CHEMISTRY_PREFIX)
    }
    if set(stored) != set(RepresentativeAtomTable._fields):
        return features, None
    return features, RepresentativeAtomTable(
        **{name: stored[name] for name in RepresentativeAtomTable._fields}
    )
