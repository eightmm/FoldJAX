"""The confidence arrays each model already computes, on disk beside its scores.

`confidence.json` keeps scalars. The arrays behind them -- predicted aligned
error, per-token or per-atom pLDDT, chain-pair ipTM -- used to stay inside
the run: AlphaFold 3 wrote PAE only into its own native JSON, and Boltz-2
transferred PAE to the host and then dropped it. Each canonical sample
directory now also holds

    <run>/seed-<seed>_sample-<nn>/confidence_full.npz

with whatever confidence arrays that model's compiled program already returns
for that sample, plus the index maps needed to read them. Nothing here
computes a confidence quantity: an array a model does not hand back is listed
under ``unavailable`` with the reason, rather than derived here.

Like `confidence.json`, names keep their model's meaning and calibration. Two
models' `pae` are both an expected aligned error in angstroms, but each over
its own tokens -- which atoms form one token differs between models, most
visibly for modified residues -- so the index maps travel with every array.

The archive is an ordinary ``.npz``. ``_meta`` is a JSON string describing
every array: its axes, unit, numerical scale, stored dtype and native name.

Storage: token-pair maps are stored as float16. At 6,568 tokens a float32
PAE is 172 MB per sample and a five-sample run writes it five times; float16
halves that, and its rounding error (at most 2**-11 relative, under 0.008 A
anywhere in PAE's 0-32 A range, under 0.0003 for a probability) is below the
two decimals AlphaFold 3 rounds its own JSON to. Every other float is stored
as float32; the stored dtype of each array is recorded in ``_meta``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from foldjax._fsutil import ordinary_file_mode

#: The file in each canonical sample directory.
FILENAME = "confidence_full.npz"
#: Suffix of the file a backend writes beside its native structure before
#: `foldjax.output.normalize` moves it into the canonical directory.
STAGED_SUFFIX = "_confidence_full.npz"
#: The npz member holding the JSON description.
META_KEY = "_meta"
#: Bumped only for an incompatible change; new arrays are additive.
SCHEMA_VERSION = 1
#: Key under `PredictionSample.metadata` (and so under each manifest sample's
#: `metadata`) and the run manifest's top-level field.
RECORD_KEY = "confidence_arrays"

#: What each axis name counts.
AXES = {
    "token": (
        "the model's own tokens, in the model's order (a standard residue is "
        "one token; ligands and, for some models, modified residues are one "
        "token per atom); index with token_chain_id/token_residue_index"
    ),
    "atom": (
        "the structure's atoms in the order they appear in the sample's mmCIF "
        "atom_site loop; atom_token_index (where present) maps each to a token"
    ),
    "chain": "chains in the order of chain_id",
}


@dataclass(frozen=True, slots=True)
class _Spec:
    axes: tuple[str, ...]
    unit: str
    description: str


_PAIR = ("token", "token")

#: Every array name a backend may write. Unlisted names are refused so the
#: vocabulary cannot drift per backend.
SPECS: dict[str, _Spec] = {
    "pae": _Spec(
        _PAIR,
        "angstrom",
        "predicted aligned error: expected error of token j's position when "
        "the prediction is aligned on token i's frame",
    ),
    "pde": _Spec(
        _PAIR, "angstrom", "predicted distance error between tokens i and j"
    ),
    "contact_probs": _Spec(
        _PAIR, "probability", "predicted probability that tokens i and j are in contact"
    ),
    "token_plddt": _Spec(("token",), "plddt", "predicted lDDT per token"),
    "atom_plddt": _Spec(("atom",), "plddt", "predicted lDDT per atom"),
    "chain_ptm": _Spec(("chain",), "score", "pTM restricted to one chain"),
    "chain_iptm": _Spec(("chain",), "score", "ipTM of one chain against the rest"),
    "chain_plddt": _Spec(("chain",), "plddt", "mean pLDDT of one chain"),
    "chain_gpde": _Spec(("chain",), "angstrom", "global PDE of one chain"),
    "chain_pair_iptm": _Spec(("chain", "chain"), "score", "ipTM per chain pair"),
    "chain_pair_iptm_global": _Spec(
        ("chain", "chain"), "score", "chain-pair ipTM normalised over the complex"
    ),
    "chain_pair_iptm_bespoke": _Spec(
        ("chain", "chain"),
        "score",
        "ligand-aware chain-pair ipTM (AF3 SI 5.9.3): a ligand chain's mean "
        "interface ipTM, else the two chains' means averaged",
    ),
    "chain_pair_plddt": _Spec(("chain", "chain"), "plddt", "pLDDT per chain pair"),
    "chain_pair_gpde": _Spec(
        ("chain", "chain"), "angstrom", "global PDE per chain pair"
    ),
    "chain_pair_pae_min": _Spec(
        ("chain", "chain"), "angstrom", "minimum PAE between two chains"
    ),
    "chain_pair_pae_mean": _Spec(
        ("chain", "chain"), "angstrom", "mean PAE between two chains"
    ),
    "chain_pair_pde_min": _Spec(
        ("chain", "chain"), "angstrom", "minimum PDE between two chains"
    ),
    "chain_pair_pde_mean": _Spec(
        ("chain", "chain"), "angstrom", "mean PDE between two chains"
    ),
    # Index maps.
    "token_chain_id": _Spec(
        ("token",), "label", "chain id of each token, as written in the mmCIF"
    ),
    "token_residue_index": _Spec(
        ("token",),
        "index",
        "residue number of each token as written in the mmCIF auth_seq_id "
        "(1-based for polymers)",
    ),
    "atom_token_index": _Spec(("atom",), "index", "0-based token of each atom"),
    "atom_chain_id": _Spec(("atom",), "label", "chain id of each atom"),
    "atom_residue_index": _Spec(
        ("atom",), "index", "residue number of each atom (mmCIF auth_seq_id)"
    ),
    "chain_id": _Spec(("chain",), "label", "chain id along every chain axis"),
}

#: Index maps rather than confidence values.
INDEX_ARRAYS = frozenset(
    {
        "token_chain_id",
        "token_residue_index",
        "atom_token_index",
        "atom_chain_id",
        "atom_residue_index",
        "chain_id",
    }
)

_DETAILS_REASON = (
    "computed inside the confidence head, but returned by the compiled program "
    "only with --option {option} (a different executable)"
)
_NO_CHAIN_PAIR_PAE = (
    "OpenDDE's confidence path runs with include_chain_pair_pae=False, so the "
    "program never computes it"
)

#: Per model: the confidence arrays a default run writes, the ones an existing
#: native option adds, and why the rest are absent. `foldjax capabilities`
#: reports `default`; the npz of each sample records exactly what it holds.
AVAILABILITY: dict[str, dict[str, Any]] = {
    "alphafold3": {
        "default": (
            "pae",
            "pde",
            "contact_probs",
            "atom_plddt",
            "chain_ptm",
            "chain_iptm",
            "chain_pair_iptm",
            "chain_pair_pae_min",
            "chain_pair_pde_min",
            "chain_pair_pde_mean",
        ),
        "opt_in": {},
        "unavailable": {
            "token_plddt": "AlphaFold 3 reports pLDDT per atom only",
            "atom_token_index": (
                "the default inference path does not return the atom-to-token "
                "layout; atom_chain_id/atom_residue_index index the atoms instead"
            ),
        },
    },
    "boltz2": {
        "default": ("pae", "pde", "token_plddt", "chain_ptm", "chain_pair_iptm"),
        "opt_in": {},
        "unavailable": {},
    },
    "esmfold2": {
        "default": ("pae", "pde", "token_plddt", "atom_plddt", "chain_pair_iptm"),
        "opt_in": {},
        "unavailable": {},
    },
    "opendde": {
        "default": (
            "atom_plddt",
            "chain_ptm",
            "chain_iptm",
            "chain_plddt",
            "chain_gpde",
            "chain_pair_iptm",
            "chain_pair_iptm_global",
            "chain_pair_plddt",
            "chain_pair_gpde",
        ),
        "opt_in": {"include_raw=true": ("pae", "pde", "contact_probs")},
        "unavailable": {
            "pae": _DETAILS_REASON.format(option="include_raw=true"),
            "pde": _DETAILS_REASON.format(option="include_raw=true"),
            "contact_probs": _DETAILS_REASON.format(option="include_raw=true"),
            "chain_pair_pae_min": _NO_CHAIN_PAIR_PAE,
            "chain_pair_pae_mean": _NO_CHAIN_PAIR_PAE,
        },
    },
    "openfold3": {
        "default": (
            "pae",
            "pde",
            "atom_plddt",
            "chain_ptm",
            "chain_pair_iptm",
            "chain_pair_iptm_bespoke",
        ),
        "opt_in": {},
        "unavailable": {},
    },
    "protenix": {
        "default": (
            "atom_plddt",
            "chain_ptm",
            "chain_iptm",
            "chain_plddt",
            "chain_gpde",
            "chain_pair_iptm",
            "chain_pair_iptm_global",
            "chain_pair_plddt",
            "chain_pair_gpde",
            "chain_pair_pae_mean",
            "chain_pair_pae_min",
        ),
        "opt_in": {"output_format=both": ("pae", "pde", "contact_probs")},
        "unavailable": {
            "pae": _DETAILS_REASON.format(option="output_format=both"),
            "pde": _DETAILS_REASON.format(option="output_format=both"),
            "contact_probs": _DETAILS_REASON.format(option="output_format=both"),
        },
    },
}


def default_arrays(model: str) -> tuple[str, ...]:
    """The confidence arrays a default run of ``model`` writes per sample."""
    entry = AVAILABILITY.get(model)
    return tuple(entry["default"]) if entry else ()


def _stored(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in "USO":
        return array.astype(str)
    if array.dtype.kind in "iub":
        return array.astype(np.int32)
    if SPECS[name].axes == _PAIR:
        return array.astype(np.float16)
    return array.astype(np.float32)


def write(
    path: str | Path,
    *,
    model: str,
    arrays: Mapping[str, Any],
    scales: Mapping[str, str] | None = None,
    sources: Mapping[str, str] | None = None,
    unavailable: Mapping[str, str] | None = None,
    sample: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write one sample's arrays atomically and return its metadata record.

    Returns None, writing nothing, when no confidence array is present: index
    maps alone describe nothing.

    ``scales`` gives the numerical range of a pLDDT-like array as the model
    reports it (``"0-1"`` or ``"0-100"``); nothing is rescaled. ``sources``
    names the native output each array came from. ``sample`` carries
    identifying fields such as ``native_rank``.
    """
    path = Path(path)
    scales = dict(scales or {})
    sources = dict(sources or {})
    stored: dict[str, np.ndarray] = {}
    described: dict[str, Any] = {}
    for name, value in arrays.items():
        if value is None:
            continue
        spec = SPECS.get(name)
        if spec is None:
            raise ValueError(f"unknown confidence array {name!r}")
        array = _stored(name, value)
        if array.ndim != len(spec.axes):
            raise ValueError(
                f"confidence array {name!r} has shape {array.shape}; "
                f"expected axes {spec.axes}"
            )
        stored[name] = array
        entry: dict[str, Any] = {
            "axes": list(spec.axes),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "unit": spec.unit,
            "description": spec.description,
        }
        if name in scales:
            entry["scale"] = scales[name]
        if name in sources:
            entry["source"] = sources[name]
        described[name] = entry
    if not set(stored) - INDEX_ARRAYS:
        return None
    missing_scale = [
        name
        for name in stored
        if SPECS[name].unit == "plddt" and name not in scales
    ]
    if missing_scale:
        raise ValueError(f"pLDDT arrays need a scale: {', '.join(missing_scale)}")
    absent = {
        str(name): str(reason)
        for name, reason in (unavailable or {}).items()
        if name not in stored
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "sample": {
            str(key): value.item() if isinstance(value, np.generic) else value
            for key, value in (sample or {}).items()
        },
        "axes": AXES,
        "arrays": described,
        "unavailable": absent,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".foldjax-confidence-", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
        try:
            np.savez(handle, **stored, **{META_KEY: np.asarray(json.dumps(meta))})
        except BaseException:
            handle.close()
            staged.unlink(missing_ok=True)
            raise
    # `NamedTemporaryFile` creates the staging file 0600; give the archive the mode an
    # ordinary write would (``0666 & ~umask``), like `confidence.json` beside it, so a
    # shared results directory stays readable to the group that reads the rest.
    os.chmod(staged, ordinary_file_mode())
    os.replace(staged, path)
    return record(path, meta)


def staged_path(structure_path: str | Path) -> Path:
    """Where a backend stages a sample's archive: beside its native structure."""
    structure_path = Path(structure_path)
    return structure_path.with_name(structure_path.stem + STAGED_SUFFIX)


def record(path: str | Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The `PredictionSample.metadata` entry for an archive at ``path``."""
    return {
        "path": str(path),
        "schema_version": meta.get("schema_version", SCHEMA_VERSION),
        "arrays": sorted(
            name for name in meta.get("arrays", {}) if name not in INDEX_ARRAYS
        ),
        "index_arrays": sorted(
            name for name in meta.get("arrays", {}) if name in INDEX_ARRAYS
        ),
        "unavailable": dict(meta.get("unavailable", {})),
    }


def staged_record(structure_path: str | Path | None) -> dict[str, Any] | None:
    """The record for the archive staged beside ``structure_path``, if any."""
    if structure_path is None:
        return None
    path = staged_path(structure_path)
    if not path.is_file():
        return None
    return record(path, read_meta(path))


def sample_metadata(structure_path: str | Path | None) -> dict[str, Any]:
    """`PredictionSample.metadata` naming the archive staged beside a structure."""
    entry = staged_record(structure_path)
    return {RECORD_KEY: entry} if entry is not None else {}


def read_meta(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive[META_KEY][()]))


def place(sample: Any, directory: Path) -> Any:
    """Move a sample's staged archive into its canonical directory.

    Called by `foldjax.output.normalize` once per placed sample. Returns the
    sample with its record pointing at ``confidence_full.npz`` by name, so a
    manifest never carries the staging path. A sample without a staged
    archive is returned unchanged.

    Unlike a structure, the staged file is always moved: only a file this
    module wrote, under its fixed staging suffix, is accepted as a source, so
    the move cannot take a user's file.
    """
    metadata = getattr(sample, "metadata", None) or {}
    entry = metadata.get(RECORD_KEY)
    if not isinstance(entry, Mapping) or "path" not in entry:
        return sample
    source = Path(str(entry["path"]))
    target = Path(directory) / FILENAME
    if (
        not source.name.endswith(STAGED_SUFFIX)
        or source.is_symlink()
        or not source.is_file()
    ):
        return sample
    if source.absolute() != target.absolute():
        try:
            os.replace(source, target)
        except OSError:
            # A different filesystem: copy beside the target, then swap in.
            with tempfile.TemporaryDirectory(
                prefix=".foldjax-confidence-", dir=target.parent
            ) as scratch:
                staged = Path(scratch) / FILENAME
                shutil.copyfile(source, staged)
                os.replace(staged, target)
            source.unlink()
    placed = {key: value for key, value in entry.items() if key != "path"}
    placed["file"] = FILENAME
    return replace(sample, metadata={**metadata, RECORD_KEY: placed})


def manifest_record(samples: Any) -> dict[str, Any] | None:
    """The run-level summary: which arrays the run's samples carry."""
    entries = [
        (getattr(sample, "metadata", None) or {}).get(RECORD_KEY)
        for sample in samples
    ]
    entries = [entry for entry in entries if isinstance(entry, Mapping)]
    if not entries:
        return None
    arrays = sorted({name for entry in entries for name in entry.get("arrays", ())})
    unavailable: dict[str, str] = {}
    for entry in entries:
        for name, reason in dict(entry.get("unavailable", {})).items():
            if name not in arrays:
                unavailable.setdefault(name, reason)
    return {
        "file": FILENAME,
        "schema_version": SCHEMA_VERSION,
        "samples": len(entries),
        "arrays": arrays,
        "unavailable": unavailable,
    }


@dataclass(frozen=True, eq=False)
class ConfidenceArrays(Mapping[str, np.ndarray]):
    """One sample's confidence arrays, read back from ``confidence_full.npz``.

    Behaves as a read-only mapping from array name to NumPy array. ``meta``
    describes each array's axes, unit, scale and stored dtype; ``unavailable``
    says why an array this model does not return is absent.
    """

    path: Path
    model: str
    arrays: Mapping[str, np.ndarray] = field(repr=False)
    meta: Mapping[str, Any] = field(repr=False)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    @property
    def unavailable(self) -> Mapping[str, str]:
        return dict(self.meta.get("unavailable", {}))

    def describe(self, name: str) -> Mapping[str, Any]:
        return dict(self.meta["arrays"][name])


def load_confidence_arrays(sample_dir: str | Path) -> ConfidenceArrays:
    """Read the confidence arrays of one sample.

    ``sample_dir`` is a canonical ``seed-<s>_sample-<nn>`` directory, or the
    ``confidence_full.npz`` file itself. Arrays are returned in the dtype they
    were stored in (float16 for token-pair maps); cast before arithmetic that
    needs more precision. Raises `FileNotFoundError` when the sample has none.
    """
    path = Path(sample_dir)
    if path.is_dir():
        path = path / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {FILENAME} at {path}")
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive[META_KEY][()]))
        if int(meta.get("schema_version", 0)) > SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses confidence-array schema {meta['schema_version']}; "
                f"this FoldJAX reads up to {SCHEMA_VERSION}"
            )
        arrays = {name: archive[name] for name in archive.files if name != META_KEY}
    return ConfidenceArrays(
        path=path, model=str(meta.get("model", "")), arrays=arrays, meta=meta
    )


__all__ = [
    "AVAILABILITY",
    "FILENAME",
    "RECORD_KEY",
    "SCHEMA_VERSION",
    "ConfidenceArrays",
    "default_arrays",
    "load_confidence_arrays",
    "manifest_record",
    "place",
    "record",
    "sample_metadata",
    "staged_path",
    "staged_record",
    "write",
]
