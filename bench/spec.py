"""The one schedule every measurement in this directory runs under.

Comparing implementations only means something if both sides do the same work,
and comparing models only means something if they all do. So there is exactly
one schedule here, and it is not invented: it is AlphaFold 3's released
inference default, which Protenix's base model also ships. Every other model
here can express all four values, so nothing is left at a different setting and
quietly compared anyway.

MSA depth is deliberately *not* in the schedule, even though it is the dominant
term in peak memory. Upstream Chai has no depth argument at all, so pinning one
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
#: Protenix's base model ships the same three. Chai and Boltz-2 default to 3
#: recycles rather than 10, so this asks more of them than their own default
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
        "FOLDJAX_BENCH_DATA", "/home/jaemin/non-project/optimizing/foldjax-bench"
    )
)

#: Every model FoldJAX can run. `alphafold3` has no "upstream" column: FoldJAX
#: drives the user's own AlphaFold 3 installation rather than reimplementing
#: it, so both columns would be the same code.
MODELS = ("alphafold3", "boltz2", "chai", "opendde", "protenix")
REIMPLEMENTED = ("boltz2", "chai", "opendde", "protenix")


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
