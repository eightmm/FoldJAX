"""One output layout, whichever model produced it.

Five backends wrote five layouts. AlphaFold 3 nested a directory per sample and
repeated the top-ranked structure at the root; Protenix and OpenDDE wrote a flat
`<job>_sample_3.cif` whose seed is nowhere in the name; OpenFold3 wrote its
samples and confidences at one level. Reading a directory therefore meant
knowing which model had filled it, and moving a file out of it lost the only
record of which seed and sample it was.

So after a run, every structure is placed at

    <output_dir>/seed-<seed>_sample-<nn>/<job>_seed-<seed>_sample-<nn>.cif

with a `confidence.json` beside it. The name carries the whole coordinate, so a
structure mailed to someone still says what it is; the zero-padded index sorts
in the order a person means; and the directory is the same shape for all five.
Whatever else the backend wrote is left exactly where it wrote it -- the parity
scripts and upstream tooling that read those names keep working.

**Confidence is not comparable across models, and nothing here pretends it is.**
Each `confidence.json` records the model's own scores under the model's own
names, plus which model produced them. A pLDDT from one model and a ranking
score from another are different quantities on different scales; averaging or
ranking across models would invent a number none of them computed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from foldjax.schema import PredictionResult, PredictionSample

#: The score each model ranks its own samples by, best first. Used only to name
#: a `best` sample within one model's run -- never to compare two models.
#:
#: Boltz-2 is absent on purpose. Upstream ranks by `confidence_score`
#: (0.8*complex_plddt + 0.2*iptm), which this port does not compute, and the
#: fields it does report -- `complex_plddt`, `iptm`, `ptm`, `mean_plddt` -- are
#: components rather than the ranking. Electing one of them here would publish a
#: "best" under a rule Boltz does not use, which is worse than saying nothing:
#: `best_sample` returns None and the per-sample scores are all still there.
#: OpenFold3 reports `ranking_score_no_clash`, and only for a job with more than
#: one chain -- a Prediction carries no clash term, so the name says the veto
#: that the real AF3 ranking score applies never fired. A single-chain run
#: therefore has no `best`, which is the same answer for the same reason.
_RANKING_SCORE = {
    "alphafold3": "ranking_score",
    "opendde": "ranking_score",
    "openfold3": "ranking_score_no_clash",
    "protenix": "ranking_score",
}


def sample_directory(output_dir: Path, seed: int, index: int) -> Path:
    """Where one sample's files go. Zero-padded so ten sorts after two."""
    return Path(output_dir) / f"seed-{seed}_sample-{index:02d}"


def structure_name(job: str, seed: int, index: int) -> str:
    return f"{job}_seed-{seed}_sample-{index:02d}.cif"


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize(
    result: PredictionResult, *, job: str, root: Path | None = None
) -> PredictionResult:
    """Move every structure into the canonical layout and return the new result.

    ``root`` is where the canonical directories go, and defaults to the result's
    own output directory. A multi-seed run gives the *parent*: each seed runs
    into its own subdirectory so the backends' native files cannot overwrite one
    another, but the seed is already in the canonical name, so nesting
    `seed-3_sample-00/` inside `seed_3/` would say it twice.

    A sample without a structure (a backend that returned coordinates only) is
    passed through untouched, and so is one whose file has already been placed.
    """
    output_dir = Path(root) if root is not None else Path(result.output_dir)
    samples = []
    for position, sample in enumerate(result.samples):
        index = _index(sample, position)
        directory = sample_directory(output_dir, sample.seed, index)
        target = directory / structure_name(job, sample.seed, index)
        source = sample.structure_path

        if source is None or not Path(source).is_file():
            samples.append(sample)
            continue
        source = Path(source)
        if source != target:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), target)
        try:
            _normalize_cif(
                target, job=job, model=result.model, seed=sample.seed, index=index
            )
        except Exception:  # noqa: BLE001 - a header is never worth losing a run
            # The structure is the result; a CIF this cannot parse is upstream's
            # to fix, and the file is already where it belongs.
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
    ranked = [
        sample
        for sample in result.samples
        if key is not None and sample.scores and key in sample.scores
    ]
    if not ranked:
        return None
    winner = max(ranked, key=lambda sample: sample.scores[key])
    return {
        "score": key,
        "value": winner.scores[key],
        "seed": winner.seed,
        "structure_path": str(winner.structure_path) if winner.structure_path else None,
    }
