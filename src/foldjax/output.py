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

**Confidence is not comparable across models, and nothing here pretends it is.**
Each `confidence.json` records the model's own scores under the model's own
names, plus which model produced them. A pLDDT from one model and a ranking
score from another are different quantities on different scales; averaging or
ranking across models would invent a number none of them computed.
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
from dataclasses import replace
from pathlib import Path

from foldjax.schema import PredictionOutputError, PredictionResult, PredictionSample

#: The score each model ranks its own samples by, best first. Used only to name
#: a `best` sample within one model's run -- never to compare two models.
#:
#: Boltz-2 is absent on purpose. Upstream ranks by `confidence_score`
#: (0.8*complex_plddt + 0.2*iptm), which this port does not compute, and the
#: fields it does report -- `complex_plddt`, `iptm`, `ptm`, `mean_plddt` -- are
#: components rather than the ranking. Electing one of them here would publish a
#: "best" under a rule Boltz does not use, which is worse than saying nothing:
#: `best_sample` returns None and the per-sample scores are all still there.
#: OpenFold3 exposes the exact key only when its complete score is available.
#: Protein inputs need a disorder term that the torch-free writer cannot derive;
#: those runs report `sample_ranking_score_no_disorder` for inspection and are
#: deliberately absent from `best_sample`.
_RANKING_SCORE = {
    "alphafold3": "ranking_score",
    "opendde": "ranking_score",
    "openfold3": "sample_ranking_score",
    "protenix": "ranking_score",
}

_UNSAFE_NAME = re.compile(r"[^\w.-]+", flags=re.UNICODE)


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
    document.write_file(str(path))


def _write_confidence(
    path: Path, sample: PredictionSample, *, model: str, index: int
) -> None:
    payload = {
        "model": model,
        "seed": sample.seed,
        "sample": index,
        # Named exactly as the model reports them. See this module's docstring:
        # these do not mean the same thing from one model to the next.
        "scores": dict(sample.scores or {}),
        "scores_are_model_specific": True,
    }
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
    samples = []
    for position, sample in enumerate(result.samples):
        index = _index(sample, position)
        directory = sample_directory(output_dir, sample.seed, index)
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
            except Exception:  # noqa: BLE001 - a header is never worth losing a run
                # The structure is the result; a CIF this cannot parse is
                # upstream's to fix, and the file is already where it belongs.
                pass
        _write_confidence(
            directory / "confidence.json", sample, model=result.model, index=index
        )
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
    return {
        "score": key,
        "value": value,
        "seed": winner.seed,
        "sample": _index(winner, position),
        "structure_path": str(winner.structure_path) if winner.structure_path else None,
    }
