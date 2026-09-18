"""Historical shared schedule and explicit controls for benchmark records.

The values below are the historical common schedule used by earlier matrix
records. Managed modern runs can have separate model-specific defaults and
protocols, so a record must state its effective schedule rather than treating
these values as every model's native inference default.

MSA depth is deliberately not a shared schedule field. Implementations can
consume different effective depths from the same alignment, so each result must
record the depth the model actually used alongside its peak-memory evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Historical common matrix schedule: 5 diffusion samples, 200 diffusion
#: steps, and 10 recycles. It is retained for reproducibility of those records;
#: managed current protocols record their own effective per-model settings.
SCHEDULE = {
    "num_samples": 5,
    "num_steps": 200,
    "num_recycles": 10,
}
SEED = 101

#: Data lives outside the repository: the alignments are tens of MB and the
#: weights are gigabytes, and neither belongs in git.
DATA = Path(
    __import__("os").environ.get(
        "FOLDJAX_BENCH_DATA",
        str(Path(__file__).resolve().parents[2] / "foldjax-bench"),
    )
)

#: Models admitted by ``bench.drive``'s shared benchmark matrix. ESMFold2's
#: historical point rows are outside this matrix; ``bench/esmfold2_compare.py``
#: is its separate released-configuration comparison harness.
#:
#: `protenix-v2` is Protenix's other supported checkpoint rather than another
#: port: 464M parameters against the release's 368M, same code path on both
#: sides. It is listed separately because a benchmark row is a (weights,
#: schedule) pair, and these are different weights.
#:
#: This used to say it "caps at 2,560 tokens, so it has no 3,012-token row to
#: compare". That is upstream's refusal, not this one's: `runtime_policy.py`
#: warns and runs past it, `strict_token_limit` exists to restore upstream's
#: behaviour for a caller who wants it, and the 3,012-token row has been in
#: `docs/benchmark.md` all along. Re-measured 2026-08-28: 2,058 s, 78.2 GiB.
MODELS = (
    "alphafold3",
    "boltz2",
    "opendde",
    "openfold3",
    "protenix",
    "protenix-v2",
)

#: The models with a meaningful "upstream" column, i.e. the ones `run_upstream`
#: knows how to drive.
#:
#: `alphafold3` is absent because ``run_upstream`` has no native upstream
#: adapter for AlphaFold 3. Its external installation is therefore outside this
#: shared matrix.
#:
#: OpenFold3 is included through the upstream v0.5.0 environment and the
#: OpenBind checkpoint this port implements. ``openfold3_runner.yml`` pins the
#: shared schedule and seed.
REIMPLEMENTED = (
    "boltz2",
    "opendde",
    "openfold3",
    "protenix",
    "protenix-v2",
)

#: Options the FoldJAX side pins so a row stays a comparison.
#:
#: OpenDDE and OpenFold3 ship BF16 defaults while their upstream inference
#: columns run FP32. Leaving either default here would compare two precisions,
#: not two implementations. Both pins apply to the warm-up, measured command,
#: and request identity through this one mapping. OpenDDE's neutral ``dtype``
#: alone does not set its confidence or diffusion fields, so all three are
#: explicit here.
COMPARISON_OPTIONS: dict[str, dict[str, str]] = {
    "opendde": {
        "dtype": "float32",
        "confidence_dtype": "fp32",
        "diffusion_dtype": "fp32",
    },
    "openfold3": {"dtype": "float32"},
}


@dataclass(frozen=True)
class Case:
    name: str
    length: int
    sequence: str

    @property
    def job(self) -> Path:
        return DATA / "jobs" / f"{self.name}.json"

    @property
    def a3m(self) -> Path:
        return DATA / "msa" / f"{self.name}.a3m"


def cases() -> tuple[Case, ...]:
    """The pinned sequences, shortest first."""
    document = json.loads((DATA / "sequences.json").read_text())
    return tuple(
        Case(name=name, length=int(body["length"]), sequence=body["sequence"])
        for name, body in sorted(document.items(), key=lambda kv: kv[1]["length"])
    )
