"""One output layout, whichever model produced it.

Six backends wrote six layouts. AlphaFold 3 nested a directory per sample and
repeated the top-ranked structure at the root; Protenix and OpenDDE wrote a flat
`<job>_sample_3.cif` whose seed is nowhere in the name; OpenFold3 wrote its
samples and confidences at one level. Reading a directory therefore meant
knowing which model had filled it, and moving a file out of it lost the only
record of which seed and sample it was.

So after a run, every structure is placed at

    <output_dir>/seed-<seed>_sample-<nn>/<job>_seed-<seed>_sample-<nn>.cif

with a `confidence.json` beside it. The name carries the whole coordinate, so a
structure mailed to someone still says what it is; the zero-padded index sorts
in the order a person means; and the directory is the same shape for all six.
Whatever else the backend wrote is left exactly where it wrote it -- the parity
scripts and upstream tooling that read those names keep working.

A run whose native input holds several jobs (an AlphaFold 3 or Protenix job
list) nests each job's directories one level down, under ``<output_dir>/<job>/``,
because the sample number restarts for every job: it is always the diffusion
index, never a rank and never a running count across jobs.

Two viewer-facing guarantees are added there too, from the sample's
`confidence_full.npz`: every mmCIF carries the model's pLDDT (0-100) in its
`B_iso_or_equiv` column, filled in only where the native writer left something
else; and a model that returns PAE gets an AlphaFold-DB-schema
`predicted_aligned_error.json` beside the structure, which PAE viewers, Mol*
and ChimeraX read as they read an AFDB entry.

**Each `confidence.json` keeps the model's own scores under the model's own
names** (``scores``), plus a ``summary`` block (`foldjax.summary`) that maps
pLDDT, pTM, ipTM and the model's ranking score onto one name and one scale,
saying for each which native number it came from and what was done to it.
Common fields standardize names and numerical scales. They retain
model-specific definitions and calibration and do not establish comparable
accuracy probabilities or authorize pooled cross-model ranking. A pLDDT from
one model and a ranking score from another are still different quantities;
averaging or ranking across models would invent a number none of them computed.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from foldjax import confidence_arrays
from foldjax.schema import PredictionOutputError, PredictionResult, PredictionSample
from foldjax.scores import EXECUTION_FIELDS
from foldjax.summary import (
    COMMON_FIELDS_NOTE,
    SCHEMA_VERSION,
    SCORE_NOTES,
    common_summary,
)

#: The score each model ranks its own samples by, best first. Used only to name
#: a `best` sample within one model's run -- never to compare two models.
#:
#: Boltz-2 ranks by `confidence_score`: upstream's own summary
#: `(4*complex_plddt + tm) / 5`, where `tm` is ipTM unless the whole sample
#: batch's ipTM is zero and then pTM. `Boltz2.predict_step` builds that value
#: and sorts the models it writes by it, and this port computes the same
#: expression in `foldjax.models.boltz2.models.predict`, so what is ranked here
#: is upstream's combination rather than a blend invented at this layer. Its
#: components -- `complex_plddt`, `iptm`, `ptm`, `mean_plddt` -- are reported
#: beside it and are still not the ranking: a run whose confidence heads did not
#: execute reports no `confidence_score` and gets no `best`.
#:
#: ESMFold2 is the one entry that is not an upstream rule. Upstream emits a
#: single structure, so it never ranks anything; drawing several samples is this
#: port's design, which leaves the ordering to FoldJAX. The choice is `plddt` --
#: the confidence head's per-token pLDDT averaged over the real tokens, on the
#: head's own 0-1 scale -- because it is the number the model itself reports
#: about the structure it just produced. It is FoldJAX's choice and is
#: documented as such; it is not something ESMFold2 publishes.
#:
#: OpenFold3 exposes the exact key only when its complete score is available.
#: Protein inputs need upstream's RASA disorder term, which the writer derives
#: with biotite (`foldjax.models.openfold3.rasa`); without biotite those runs
#: report `sample_ranking_score_no_disorder` for inspection and are
#: deliberately absent from `best_sample`.
_RANKING_SCORE = {
    "alphafold3": "ranking_score",
    "boltz2": "confidence_score",
    "esmfold2": "plddt",
    "opendde": "ranking_score",
    "openfold3": "sample_ranking_score",
    "protenix": "ranking_score",
}

_UNSAFE_NAME = re.compile(r"[^\w.-]+", flags=re.UNICODE)

#: The AlphaFold-DB-schema PAE file in each canonical sample directory.
PAE_JSON = "predicted_aligned_error.json"

#: The top of each model's PAE scale: the centre of its last bin, which is what
#: AlphaFold DB's ``max_predicted_aligned_error`` states (31.75 there). All six
#: bin 0-32 A into 64 bins of 0.5 A. AlphaFold 3 spells it as 63 breaks over
#: 0-31 A plus a catch-all bin (``confidence_head.py``, ``max_error_bin=31``);
#: Protenix, OpenDDE and OpenFold3 as ``get_bin_centers(0, 32, 64)``; Boltz-2
#: and ESMFold2 as ``arange(0.25, 32, 0.5)``. `tests/test_viewer_exports.py`
#: recomputes each from the port's own code.
MAX_PREDICTED_ALIGNED_ERROR = {
    "alphafold3": 31.75,
    "boltz2": 31.75,
    "esmfold2": 31.75,
    "opendde": 31.75,
    "openfold3": 31.75,
    "protenix": 31.75,
}

#: Rounding the writers apply to the B-factor column (two decimals at most).
_B_FACTOR_TOLERANCE = 0.01


def safe_job_name(name: str, *, limit: int = 120) -> str:
    """Return a readable filename component that cannot escape its run root."""
    original = str(name).strip()
    safe = _UNSAFE_NAME.sub("_", original.replace("/", "_").replace("\\", "_"))
    safe = safe.strip("._") or "prediction"
    if len(safe.encode("utf-8")) <= limit:
        return safe
    digest = hashlib.sha256(original.encode()).hexdigest()[:8]
    prefix_bytes = safe.encode("utf-8")[: limit - len(digest) - 1]
    prefix = prefix_bytes.decode("utf-8", errors="ignore").rstrip("._-")
    return f"{prefix or 'prediction'}-{digest}"


def sample_directory(output_dir: Path, seed: int, index: int) -> Path:
    """Where one sample's files go. Zero-padded so ten sorts after two."""
    return Path(output_dir) / f"seed-{seed}_sample-{index:02d}"


def structure_name(
    job: str, seed: int, index: int, *, suffix: str = ".cif"
) -> str:
    return f"{safe_job_name(job)}_seed-{seed}_sample-{index:02d}{suffix}"


def _index(sample: PredictionSample, fallback: int) -> int:
    """The sample number the backend reported, or its position in the result."""
    value = (sample.metadata or {}).get("sample")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _normalize_cif(path: Path, *, job: str, model: str, seed: int, index: int) -> None:
    """Give the file a data block and title that say what it is.

    Edited as a CIF *document*, not as a parsed structure: re-serializing
    coordinates would drop the categories each writer adds -- Boltz's ModelCIF
    `ma_*` blocks, AlphaFold 3's terms-of-use header -- and this only needs to
    touch what a person reads first. Atom records are not rewritten at all.
    """
    from gemmi import cif

    document = cif.read(str(path))
    block = document.sole_block()
    block.name = f"{job}_seed-{seed}_sample-{index:02d}"
    block.set_pair("_entry.id", cif.quote(block.name))
    block.set_pair(
        "_struct.title",
        cif.quote(f"{job} predicted by {model} (seed {seed}, sample {index})"),
    )
    block.set_pair("_struct.entry_id", cif.quote(block.name))
    _replace_cif(document, path)


def _replace_cif(document, path: Path) -> None:
    """Write an edited CIF document over ``path`` through a sibling and a rename.

    Written in place, a write cut short (a full disk) left a truncated
    structure that the run manifest then digested as the verified result.
    """
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-structure-", dir=path.parent
    ) as scratch:
        staged = Path(scratch) / path.name
        document.write_file(str(staged))
        shutil.copymode(path, staged)
        os.replace(staged, path)


def _atom_plddt_percent(
    arrays: confidence_arrays.ConfidenceArrays,
) -> np.ndarray | None:
    """Per-atom pLDDT on 0-100 in the mmCIF's atom order, or None."""
    if "atom_plddt" in arrays:
        name, values = "atom_plddt", np.asarray(arrays["atom_plddt"], np.float64)
    elif "token_plddt" in arrays and "atom_token_index" in arrays:
        name = "token_plddt"
        values = np.asarray(arrays["token_plddt"], np.float64)[
            np.asarray(arrays["atom_token_index"], np.int64)
        ]
    else:
        return None
    if arrays.describe(name).get("scale") == "0-1":
        values = values * 100.0
    return values


def _cif_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:  # '?' and '.' are mmCIF's unknown/inapplicable
        return float("nan")


def _ensure_plddt_b_factors(
    path: Path, arrays: confidence_arrays.ConfidenceArrays
) -> str:
    """Make the mmCIF's ``B_iso_or_equiv`` column the model's pLDDT (0-100).

    Returns ``"native"`` when the writer already put it there (to its own
    rounding), ``"filled"`` when this wrote it, or why it could not. Edited as
    a CIF document, like `_normalize_cif`, so no other category is touched;
    an already-correct file is not rewritten.
    """
    from gemmi import cif

    plddt = _atom_plddt_percent(arrays)
    if plddt is None:
        return "skipped: the confidence archive has no per-atom pLDDT"
    document = cif.read(str(path))
    block = document.sole_block()
    table = block.find_mmcif_category("_atom_site.")
    if len(table) != plddt.size:
        return (
            f"skipped: {len(table)} atom_site rows against {plddt.size} "
            "per-atom pLDDT values"
        )
    if "_atom_site.B_iso_or_equiv" in list(table.tags):
        column = table.find_column("B_iso_or_equiv")
        written = np.asarray([_cif_float(value) for value in column])
        if np.allclose(written, plddt, rtol=0.0, atol=_B_FACTOR_TOLERANCE):
            return "native"
    else:
        table.loop.add_columns(["_atom_site.B_iso_or_equiv"], "?")
        # The table is a view of the loop as it was; read the widened one.
        column = block.find_mmcif_category("_atom_site.").find_column("B_iso_or_equiv")
    for row, value in enumerate(plddt):
        column[row] = f"{value:.2f}" if np.isfinite(value) else "?"
    _replace_cif(document, path)
    return "filled"


def _write_pae_json(path: Path, pae: np.ndarray, *, maximum: float) -> None:
    """AlphaFold DB's PAE JSON: one object in a list, values to two decimals.

    Written row by row: at 3,012 tokens the matrix is 9.1 million values, and
    one ``json.dumps`` of the nested list would hold all of them as Python
    floats at once.
    """
    rows = np.round(np.asarray(pae, dtype=np.float64), 2)
    with tempfile.NamedTemporaryFile(
        "w",
        prefix=".foldjax-pae-",
        suffix=".json",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        staged = Path(handle.name)
        try:
            handle.write('[{"predicted_aligned_error": [')
            for index, row in enumerate(rows):
                if index:
                    handle.write(",")
                handle.write(json.dumps(row.tolist()))
            handle.write(f'], "max_predicted_aligned_error": {maximum}}}]')
        except BaseException:
            handle.close()
            staged.unlink(missing_ok=True)
            raise
    os.chmod(staged, 0o666 & ~confidence_arrays._umask())
    os.replace(staged, path)


def _viewer_exports(
    sample: PredictionSample, directory: Path, structure: Path, *, model: str
) -> PredictionSample:
    """Write the PAE JSON and pLDDT B-factors from the placed confidence archive.

    Their outcome is recorded under the sample's ``confidence_arrays`` record
    as ``exports``; a failure is recorded there rather than raised, because a
    viewer convenience is never worth losing the run it describes.
    """
    metadata = sample.metadata or {}
    entry = metadata.get(confidence_arrays.RECORD_KEY)
    archive = directory / confidence_arrays.FILENAME
    if not isinstance(entry, Mapping) or entry.get("file") != archive.name:
        return sample
    exports: dict[str, object] = {}
    try:
        arrays = confidence_arrays.load_confidence_arrays(archive)
    except (OSError, ValueError, KeyError) as error:
        exports["error"] = f"the confidence archive could not be read: {error}"
        arrays = None
    if arrays is not None:
        maximum = MAX_PREDICTED_ALIGNED_ERROR.get(model)
        if "pae" not in arrays:
            exports["pae_json"] = None
        elif maximum is None:
            exports["pae_json"] = None
            exports["pae_json_reason"] = f"no PAE scale is recorded for {model!r}"
        else:
            try:
                _write_pae_json(directory / PAE_JSON, arrays["pae"], maximum=maximum)
                exports["pae_json"] = PAE_JSON
            except OSError as error:
                exports["pae_json"] = None
                exports["pae_json_reason"] = str(error)
        if structure.suffix.lower() in {".cif", ".mmcif"}:
            try:
                exports["plddt_b_factor"] = _ensure_plddt_b_factors(structure, arrays)
            except Exception as error:  # noqa: BLE001 - recorded, never raised
                exports["plddt_b_factor"] = f"failed: {error}"
    return replace(
        sample,
        metadata={
            **metadata,
            confidence_arrays.RECORD_KEY: {**entry, "exports": exports},
        },
    )


def confidence_payload(
    sample: PredictionSample,
    *,
    model: str,
    index: int,
    structure: Path | None = None,
) -> dict[str, object]:
    """The `confidence.json` document for one sample (schema 1.x).

    ``scores`` is unchanged native output; ``summary`` is the common block. See
    `foldjax.summary` for the mapping and `foldjax/schemas/confidence.schema.json`
    for the contract.
    """
    metadata = sample.metadata or {}
    scores = dict(sample.scores or {})
    rank = metadata.get("native_rank")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "seed": sample.seed,
        # The diffusion index, for every model.
        "sample": index,
        # The rank a native writer encoded in its own file name (Protenix and
        # OpenDDE), so a native side file can still be matched; None elsewhere.
        "native_rank": (
            int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else None
        ),
        # Named exactly as the model reports them. See this module's docstring:
        # these do not mean the same thing from one model to the next.
        "scores": scores,
        "scores_are_model_specific": True,
        "summary": common_summary(model, scores, structure=structure),
        "summary_note": COMMON_FIELDS_NOTE,
        # How the run executed, which a native summary reported among its
        # scores (`foldjax.scores.EXECUTION_FIELDS`).
        "execution": {
            key: metadata[key] for key in sorted(EXECUTION_FIELDS) if key in metadata
        },
    }
    if metadata.get("job"):
        payload["job"] = str(metadata["job"])
    notes = SCORE_NOTES.get(model)
    if notes:
        payload["score_notes"] = dict(notes)
    return payload


def _write_confidence(
    path: Path,
    sample: PredictionSample,
    *,
    model: str,
    index: int,
    structure: Path | None = None,
) -> None:
    payload = confidence_payload(
        sample, model=model, index=index, structure=structure
    )
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-confidence-", dir=path.parent
    ) as scratch:
        staged = Path(scratch) / path.name
        staged.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(staged, path)


def _copy_atomic(source: Path, target: Path) -> None:
    """Copy through a sibling file so an existing target symlink is replaced."""
    with tempfile.TemporaryDirectory(
        prefix=".foldjax-structure-", dir=target.parent
    ) as scratch:
        staged = Path(scratch) / target.name
        shutil.copy2(source, staged)
        os.replace(staged, target)


def _move_atomic(source: Path, target: Path) -> None:
    """Replace the target itself, never follow a target symlink or directory."""
    try:
        os.replace(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        _copy_atomic(source, target)
        source.unlink()


def normalize(
    result: PredictionResult, *, job: str, root: Path | None = None
) -> PredictionResult:
    """Place every structure in the canonical layout and return the new result.

    ``root`` is where the canonical directories go, and defaults to the result's
    own output directory. A multi-seed run gives the *parent*: each seed runs
    into its own subdirectory so the backends' native files cannot overwrite one
    another, but the seed is already in the canonical name, so nesting
    `seed-3_sample-00/` inside `seed_3/` would say it twice.

    A sample without a structure (a backend that returned coordinates only) is
    passed through untouched, and so is one whose file has already been placed.
    Files produced inside the run root are moved; a backend-reported path outside
    it is copied so a malformed adapter can never delete an input or other user
    file as a side effect of normalization.
    """
    output_dir = Path(root) if root is not None else Path(result.output_dir)
    root_resolved = output_dir.resolve()
    job = safe_job_name(job)
    jobs = {
        str((sample.metadata or {}).get("job"))
        for sample in result.samples
        if (sample.metadata or {}).get("job")
    }
    samples = []
    for position, sample in enumerate(result.samples):
        index = _index(sample, position)
        sample_job_name = (sample.metadata or {}).get("job")
        # One job keeps the flat layout. Several get a level each, because the
        # sample number restarts for every job.
        parent = (
            output_dir / safe_job_name(str(sample_job_name))
            if len(jobs) > 1 and sample_job_name
            else output_dir
        )
        directory = sample_directory(parent, sample.seed, index)
        source = sample.structure_path

        if source is None or not Path(source).is_file():
            samples.append(sample)
            continue
        source = Path(source)
        suffix = source.suffix.lower()
        if suffix not in {".cif", ".mmcif", ".pdb"}:
            suffix = source.suffix or ".cif"
        sample_job = safe_job_name(str((sample.metadata or {}).get("job") or job))
        target = directory / structure_name(
            sample_job, sample.seed, index, suffix=suffix
        )
        source_resolved = source.resolve()
        if directory.is_symlink():
            raise PredictionOutputError(
                f"canonical output directory is a symlink: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise PredictionOutputError(
                f"canonical output directory is a symlink: {directory}"
            )
        if not directory.resolve().is_relative_to(root_resolved):
            raise PredictionOutputError(
                f"canonical output directory escapes run root: {directory}"
            )
        same_lexical_path = source.absolute() == target.absolute()
        try:
            link_count = source.stat().st_nlink
        except OSError:
            link_count = 0
        safe_to_move = (
            not source.is_symlink()
            and link_count == 1
            and source_resolved.is_relative_to(root_resolved)
        )
        if not (same_lexical_path and safe_to_move):
            if safe_to_move:
                _move_atomic(source, target)
            else:
                _copy_atomic(source, target)
        if suffix in {".cif", ".mmcif"}:
            try:
                _normalize_cif(
                    target,
                    job=sample_job,
                    model=result.model,
                    seed=sample.seed,
                    index=index,
                )
            except Exception as error:  # noqa: BLE001 - a header never costs a run
                # The structure is the result; a CIF this cannot parse is
                # upstream's to fix, and the file is already where it belongs,
                # unchanged, because the rewrite only ever replaces it whole.
                warnings.warn(
                    f"FoldJAX could not retitle {target} ({error}); the "
                    f"structure is kept exactly as {result.model} wrote it",
                    RuntimeWarning,
                    stacklevel=2,
                )
        _write_confidence(
            directory / "confidence.json",
            sample,
            model=result.model,
            index=index,
            structure=target,
        )
        sample = confidence_arrays.place(sample, directory)
        sample = _viewer_exports(sample, directory, target, model=result.model)
        samples.append(replace(sample, structure_path=target))
    return replace(result, samples=tuple(samples))


def best_sample(result: PredictionResult) -> dict[str, object] | None:
    """The model's own top-ranked sample, by the score that model ranks with.

    Returns ``None`` when the model reported no such score, rather than falling
    back to another one: a "best" chosen by a different quantity than the model
    ranks by would be a different claim wearing the same word.
    """
    key = _RANKING_SCORE.get(result.model)
    if key is None or not result.samples:
        return None
    ranked: list[tuple[int, PredictionSample, float]] = []
    for position, sample in enumerate(result.samples):
        if not sample.scores or key not in sample.scores:
            return None
        try:
            value = float(sample.scores[key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        ranked.append((position, sample, value))

    # ``max`` keeps the first item on a tie, preserving diffusion/sample order.
    position, winner, value = max(ranked, key=lambda item: item[2])
    best: dict[str, object] = {
        "score": key,
        "value": value,
        "seed": winner.seed,
        "sample": _index(winner, position),
        "structure_path": str(winner.structure_path) if winner.structure_path else None,
        # The top of this model's own confidence ordering within this run: not
        # the most accurate structure, and never a pick across models.
        "selection": "within-model confidence ranking",
    }
    if (winner.metadata or {}).get("job"):
        best["job"] = str(winner.metadata["job"])
    return best
