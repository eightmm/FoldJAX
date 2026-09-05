"""The one schedule every measurement in this directory runs under.

Comparing implementations only means something if both sides do the same work,
and comparing models only means something if they all do. So there is exactly
one schedule here, and it is not invented: it is AlphaFold 3's released
inference default, which Protenix's base model also ships. Every other model
here can express all four values, so nothing is left at a different setting and
quietly compared anyway.

MSA depth is deliberately *not* in the schedule, even though it is the dominant
term in peak memory. Not every upstream exposes a depth argument, so pinning one
would mean FoldJAX doing something its own reference implementation cannot --
which is the opposite of a controlled comparison. Both sides therefore read the
same alignment file and apply each implementation's own default, which is
identical between them by construction. The depth each model actually used is
reported alongside the peak, because it is what makes the peaks differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: AlphaFold 3's released default (`run_alphafold.make_model_config` plus
#: `Model.Config`): 5 diffusion samples, 200 diffusion steps, 10 recycles.
#: Protenix's base model ships the same three. Boltz-2 defaults to 3
#: recycles rather than 10, so this asks more of it than its own default
#: does -- but it asks the same of their upstream, which is what makes the
#: FoldJAX-vs-upstream column mean something.
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
#: `alphafold3` is absent because FoldJAX drives the user's own AlphaFold 3
#: installation rather than reimplementing it, so both columns would run the
#: same code.
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
#: OpenDDE's shipped default became bfloat16 on 2026-08-28, measured smaller and
#: faster and inside its own sampling spread. Upstream still ships fp32, so
#: leaving the default here would make this table compare two precisions, which
#: is not a comparison. The bfloat16 numbers live in `docs/benchmark.md` as
#: their own note, which is what a setting a user has -- rather than a result
#: these columns measured -- deserves.
COMPARISON_OPTIONS: dict[str, dict[str, str]] = {
    "opendde": {"dtype": "float32"},
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
