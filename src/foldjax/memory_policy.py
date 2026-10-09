"""Decide, before a program is built, whether this run fits the card it is on.

Every chunk and precision default in this repository was chosen on a 96 GiB
device. On a smaller one the same defaults produce an allocator failure after
the featurizer, the weight load and the first compile -- minutes of work to
learn that the job was never going to fit -- and the failure names the block
XLA could not place rather than the size of the job.

The peak of these models is not mysterious enough to need the failure. It is
dominated by pair tensors and their temporaries, so it follows the token count
closely enough to be fitted, and the allocator's own ceiling is readable from
`bytes_limit` before anything is allocated. So admission can be a comparison:
estimate the peak, compare it with the ceiling, and refuse up front when it
cannot fit -- with the levers named, because the interesting case is a job that
fits the device and not the pool.

Two things this module deliberately does not do. It never narrows an input to
make a job fit: reducing MSA depth changes the prediction, and on this
repository's own accuracy-admission test it failed. And it never claims more
than it measured -- outside a law's fitted token range the answer is
``unknown`` and the run proceeds, because a refusal from an extrapolation is
worse than no refusal at all. Below a fitted range a law may still say
``fits`` where completed runs were checked against it (``admits_from``), and
never refuses there.

The laws are frozen literals fitted by `tests/calibrate_memory_policy.py`,
which carries the measurements, the fit and the derivation of each allowance.
An allowance is the law's own worst underestimate on those measurements plus
the spread a repeated measurement shows -- deliberately not a held-out error,
because a held-out error at the largest measured size is an extrapolation, and
carrying it refused a Boltz-2 run that had completed.

Nothing here imports JAX; :func:`device_memory_budget` is the one function that
reads the device, through `foldjax.oom`, which imports JAX only when called.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal

from foldjax import oom

_MIB = 2**20
_GIB = 2**30

#: Share of the allocator's ceiling a job is admitted against.
#:
#: The pool is not everything a run needs from the device: cuBLAS, cuDNN and
#: the cuEquivariance kernels take workspaces outside it, and `foldjax.oom`
#: records a measured failure where a total that fit the pool still could not
#: be laid out in it. A job admitted at exactly the ceiling would therefore be
#: admitted into a space it cannot have. 0.9 leaves that room; it is a policy
#: number, not a measurement, and `--memory-budget-gib` is how a caller
#: overrides the budget it applies to.
ADMISSION_FRACTION = 0.9

#: Basis term name -> value at a token count and a processed MSA row count.
#: Small and closed on purpose: a law is a frozen tuple of (term, coefficient)
#: pairs, so the term names are part of the calibration record.
_TERMS = {
    "1": lambda n, m: 1.0,
    "n": lambda n, m: float(n),
    "n2": lambda n, m: float(n) ** 2,
    "n3": lambda n, m: float(n) ** 3,
    "mn": lambda n, m: None if m is None else float(m) * float(n),
}

State = Literal["fits", "over_budget", "unknown"]
BudgetSource = Literal["pool", "override", "none"]


@dataclass(frozen=True, slots=True)
class PeakLaw:
    """A fitted device-peak law for one model in one configuration.

    ``coeffs`` is one entry per *phase* -- ``(phase name, ((term,
    coefficient), ...))`` -- and the estimate is the largest phase, not their
    sum. A peak is one moment: a run whose trunk and whose diffusion each need
    20 GiB, at different times, peaks at 20 and not at 40. Most of these laws
    have a single phase; Protenix has two, because its measurements separate
    them.

    Coefficients are **MiB per unit of their term**, the unit the measurements
    were recorded in; :meth:`estimate` returns bytes. ``allowance_bytes`` is
    the largest underestimate the law makes on its own fit points, plus the
    spread the same measurement shows when it is repeated, so :meth:`upper` is
    the number to admit against -- and every measured, completed run is
    admitted by construction.

    ``profile`` names the configuration the points were taken at. A law fitted
    at five samples does not describe a one-sample run, and nothing here can
    detect that: the caller that knows the run's knobs is the one that must
    decide whether its estimate is binding. See :func:`off_profile_reason`.
    """

    model: str
    profile: str
    coeffs: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    domain_tokens: tuple[int, int]
    allowance_bytes: int
    calibration_id: str
    #: Below the fitted domain, the smallest token count from which every
    #: measured completed run is bounded by :meth:`upper`. In that band a
    #: ``fits`` is reported and nothing is refused: the law was checked against
    #: outcomes there, not fitted, so it may say "fits" but its over-estimates
    #: are not trusted to say "will not fit" (:func:`resolve_memory_policy`).
    admits_from: int | None = None
    #: ``(n_token, upper bytes)`` for sizes measured on the shipped program:
    #: there :meth:`upper` is the measured peak plus its repeat spread instead
    #: of the law's, which can read high or low at one size. Exact sizes only.
    measured_uppers: tuple[tuple[int, int], ...] = ()

    @property
    def needs_msa_rows(self) -> bool:
        """Whether :meth:`estimate` cannot be evaluated without a row count."""
        return any(term == "mn" for _phase, terms in self.coeffs for term, _c in terms)

    def covers(self, n_token: int) -> bool:
        """Whether ``n_token`` is inside the range this law was fitted over."""
        low, high = self.domain_tokens
        return low <= n_token <= high

    def bounds(self, n_token: int) -> bool:
        """Whether admission may estimate ``n_token``: the domain or the band
        below it (``admits_from``) where it was checked against measurements."""
        low, high = self.domain_tokens
        start = low if self.admits_from is None else min(low, self.admits_from)
        return start <= n_token <= high

    def measured_upper(self, n_token: int) -> int | None:
        """The measured upper bytes at exactly ``n_token``, if there is one."""
        return dict(self.measured_uppers).get(n_token)

    def estimate(self, n_token: int, msa_rows: int | None = None) -> int:
        """The fitted peak in bytes. Raises when a needed input is missing."""
        if n_token <= 0:
            raise ValueError("n_token must be positive")
        phases = []
        for _phase, terms in self.coeffs:
            total = 0.0
            for term, coefficient in terms:
                value = _TERMS[term](n_token, msa_rows)
                if value is None:
                    raise ValueError(
                        f"the {self.model} peak law needs a processed MSA row "
                        f"count for its {term!r} term"
                    )
                total += coefficient * value
            phases.append(total)
        # A fitted law is not constrained to be positive below its domain.
        return max(0, round(max(phases) * _MIB))

    def upper(self, n_token: int, msa_rows: int | None = None) -> int:
        """The estimate plus this law's allowance: what admission compares.

        At a size listed in ``measured_uppers`` it is that measured bound.
        """
        measured = self.measured_upper(n_token)
        if measured is not None:
            return measured
        return self.estimate(n_token, msa_rows) + self.allowance_bytes


# ---------------------------------------------------------------------------
# Frozen calibration. Re-derive with `python tests/calibrate_memory_policy.py`;
# `tests/test_memory_policy.py` fails if these literals drift from that fit.
# All on one 95.6 GiB device, peak = XLA peak_bytes_in_use. The four laws
# below were calibrated 2026-09-15; OpenDDE's two and ESMFold2's came later
# (2026-09-16) off ledger rows measured 2026-08-23 and 2026-09-10, so each
# law carries its own date in `calibration_id` rather than sharing one.
#
# Boltz-2, Protenix and OpenFold3 were measured on the released pair kernels.
# Since 2026-09-25 a GPU run of Boltz-2 and OpenFold3 defaults to the Pallas
# triangle multiplication and pair transitions
# (`models/_pallas_pair.default_backend`; Protenix defaults to the Pallas
# multiplication only, +24 MiB at 4,100 tokens in x53 job 2513), and the laws
# were deliberately not refitted: against a same-base control
# the Pallas peak is Boltz-2 -975 to +7 MiB, Protenix -286 to +41 MiB and
# OpenFold3 -2,302 to 0 MiB (foldjax-bench/x51-pallas-validate-20260925,
# x52-pallas-msa-pair-scope-20260925), and at 4,100 tokens Boltz-2 -16 and
# OpenFold3 -1,232 MiB (x53-pallas-4k-20260925). Both increases sit inside
# their allowances (Protenix +41 of 1,018 MiB, Boltz-2 +7 of 829). Between
# 4,100 and the 4,888 the Boltz-2 and OpenFold3 domains reach, the Pallas peak
# has not been measured against these laws.
# ---------------------------------------------------------------------------

_CALIBRATION = "2026-09-15"
#: Boltz-2, Protenix and blocked OpenFold3, refitted after bf16 cuEq triangle
#: attention started padding unaligned extents to 8 (rows re-measured on that
#: program; the aligned sizes 2,096 and 4,888 kept their points).
_CALIBRATION_ALIGNED = "2026-09-24"

#: Boltz-2 at 1003/2096/3012/4100/4888 tokens: 8454/18511/30667/55057/77312 MiB.
#: `a + b*n + d*n^3` rather than a quadratic: the local exponent of those five
#: points rises from 1.06 to 2.29, so one square term cannot hold both ends --
#: fitted as `a + c*n^2` the same points come out 12% high at 1,003 tokens and
#: the allowance more than doubles. NNLS drives the square term of a full cubic
#: to zero, which is why it is absent rather than pruned by hand.
BOLTZ2_PEAK = PeakLaw(
    model="boltz2",
    profile="bf16 trunk, fp32 diffusion, 5 samples, released schedule",
    coeffs=(
        ("whole run", (("1", 1548.02163), ("n", 6.38897827), ("n3", 3.84294776e-07))),
    ),
    domain_tokens=(1003, 4888),
    allowance_bytes=829 * _MIB,
    calibration_id=f"boltz2-{_CALIBRATION_ALIGNED}",
)

#: Protenix, in two phases, because the measurements separate them. Lowering
#: the MSA cap at 2,096 tokens stops moving the peak below about 4,096 rows
#: (13,754 MiB at 4,096 against 13,727 at 2,048 -- 27 MiB over half the
#: alignment), and above it the peak climbs with the row count (14,730 at
#: 8,192; 21,225 at full depth, 13,267 rows). So the pair phase is
#: `a + b*n^2`, fitted on the capped points (2,096 tokens at 4,096 and 2,048
#: rows, 3,012 at 4,096), and the MSA phase is `c * rows * tokens`, fitted on
#: the four that move: (1003, 8808) 6404, (2096, 8192) 14730,
#: (2096, 13267) 21225, (3012, 17542) 37537 MiB.
#:
#: An additive law was tried first and is wrong in a way worth recording: one
#: `c * rows * tokens` term cannot be both the 27 MiB the capped pair costs and
#: the 6.5 GiB the uncapped pair costs, so fitting all six points together
#: gives a coefficient that belongs to neither regime.
#:
#: The one law whose allowance carries a repeat spread: 2,096 tokens at full
#: depth came out 21,225.3 / 21,225.4 / 21,253.7 MiB across snapshots, so the
#: 989 MiB residual at the phase crossover is widened by that 28.4.
PROTENIX_PEAK = PeakLaw(
    model="protenix",
    profile="bf16 trunk, fp32 diffusion, 5 samples, released schedule",
    coeffs=(
        ("pair", (("1", 1229.45362), ("n2", 0.00284781044))),
        ("msa", (("mn", 0.000732141627),)),
    ),
    domain_tokens=(1003, 3012),
    allowance_bytes=1018 * _MIB,
    calibration_id=f"protenix-{_CALIBRATION_ALIGNED}",
)

#: OpenFold3 with the pair-stack row loop blocked at the resolved 128 rows:
#: 4316.7/13882/24820.8/47598.6/59214.8 MiB at 1003/2096/3012/4100/4888 tokens.
#: The linear term earns its place: fitted as `a + c*n^2` the same five points
#: read 14% high at 1,003 tokens -- where the whole peak is 4.3 GiB -- and the
#: allowance comes out larger rather than smaller.
#:
#: This is the law the automatic path estimates against, because from 1,003
#: tokens up the blocked loop *is* the automatic configuration
#: (`inference.resolve_pair_chunk_size`). It is the only OpenFold3 candidate
#: admission is given, so its verdict is a verdict and not a choice.
OPENFOLD3_CHUNKED_PEAK = PeakLaw(
    model="openfold3",
    profile="released schedule, 5 samples, pair stack blocked at 128 rows",
    coeffs=(
        (
            "chunked",
            (("1", 0.0), ("n", 2.5409112), ("n2", 0.00202727988)),
        ),
    ),
    domain_tokens=(1003, 4888),
    allowance_bytes=3103 * _MIB,
    calibration_id=f"openfold3-chunked-{_CALIBRATION_ALIGNED}",
    # Checked, not fitted: 146 completed serial jctc-v3 runs at 129-996 tokens
    # (released defaults, unpadded, cold and warm) all sit under `upper`, the
    # closest 576 MiB under it at 996 tokens (7,069 MiB measured). So below
    # 1,003 a run can be called a fit, and is never refused
    # (`tests/calibrate_memory_policy.OF3_BELOW_DOMAIN_CHECK`).
    admits_from=129,
)

#: OpenFold3 with the row loop unblocked: 6194.6/22967/45363 MiB at
#: 1003/2096/3012 tokens. Three sizes is what makes this arm fittable on its
#: own. Two parameters over three token counts leave a residual to take an
#: allowance from; a linear term as well would fit all three points exactly,
#: and an exact fit has no residual at all -- a zero margin rather than a
#: better law.
#:
#: **Not an automatic candidate.** Against the blocked arm at the same sizes
#: this configuration costs 44% more peak at 1,003 tokens (6,195 against
#: 4,317 MiB) for equal wall time (71.6 against 72.0 s) and 65% more at 2,096
#: (22,967 against 13,990 MiB) for 3.4% less (220.85 against 228.5 s), and at
#: 4,100 and 4,888 it has no measurement at all while the blocked arm runs.
#: Memory is what these defaults are judged on, so the automatic answer is the
#: blocked loop at every size in the validated domain and nothing selects
#: between the two. The law stays because it is still the estimate for a run
#: that spells the unblocked loop explicitly (`pair_chunk_size=0`), and
#: because `tests/calibrate_memory_policy.py` re-derives it from those three
#: measurements.
#:
#: Nothing was measured above 3,012 tokens unblocked, so the domain stops
#: there.
OPENFOLD3_UNCHUNKED_PEAK = PeakLaw(
    model="openfold3",
    profile="released schedule, 5 samples, unchunked pair stack",
    coeffs=(("unchunked", (("1", 1438.19875), ("n2", 0.00485164906))),),
    domain_tokens=(1003, 3012),
    allowance_bytes=215 * _MIB,
    calibration_id=f"openfold3-unchunked-{_CALIBRATION}",
)

#: Measured later than the four laws above, so they carry their own date.
_CALIBRATION_STRUCTURAL = "2026-09-16"
_CALIBRATION_ESMFOLD2 = "2026-09-23"

#: OpenDDE under the released bfloat16 trunk. **Keyed on the structural token
#: count, not the residue count**: this port folds in structural-token space,
#: and the ratio between the two drifts with composition -- 1.896 at 1,003
#: residues, 1.945 at 1,531 and at 4,100 -- so a law keyed on residues would
#: carry that drift squared. The caller reads the count off
#: ``structural_token_index``; :func:`admit`'s ``token_label`` is what makes
#: the message say so.
#:
#: Fitted on two completed serial rows, 21,492 MiB at 1,902 structural tokens
#: (1,003 residues) and 46,858.2 at 2,978 (1,531 residues). An exact fit over
#: two points normally means no residual to take an allowance from; here both
#: parameters are separately corroborated instead. The intercept comes out
#: 4,016 MiB, inside the 3.0-4.6 GiB of arguments this port carries across
#: these sizes, and the square term comes out 0.883 of the fused-arm arena
#: coefficient against 0.866 measured -- the bf16 trunk routes its triangle
#: multiplication to the blocked XLA path, which took 2,650 MiB off the
#: 19,797 MiB fused arena at 1,902. So the allowance is the snapshot spread:
#: the same configuration read 20,326.4 MiB one snapshot earlier.
#:
#: This law replaces the arena preflight `models/opendde/runner.py` used to
#: carry. That estimated the temp arena alone -- about 91% of the peak -- and
#: could only warn; this estimates the peak the pool has to hold, and it can
#: refuse.
#:
#: The domain ends at 7,876, above everything fitted, and no failure was
#: fitted to. What its top rests on is that the verdict has been checked
#: against an outcome there: at about 4,034 structural tokens (2,096 residues)
#: this port asked for an 87 GiB arena and died on a 95.6 GiB pool, serial and
#: on both 1-D layouts, and at 7,876 (4,100 residues) it asked for 331.5 GiB
#: and died. With both coefficients nonnegative the law bends nowhere between,
#: so the refusal it gives there is the one those runs earned. Above 7,876 the
#: answer is ``unknown``.
OPENDDE_BF16_PEAK = PeakLaw(
    model="opendde",
    profile=(
        "bf16 trunk, 5 samples, released schedule, structural tokens"
    ),
    coeffs=(("bf16 trunk", (("1", 4015.90712), ("n2", 0.0048308474))),),
    domain_tokens=(1902, 7876),
    allowance_bytes=1171 * _MIB,
    calibration_id=f"opendde-bf16-{_CALIBRATION_STRUCTURAL}",
    # Checked, not fitted: 144 completed serial jctc-v3 runs at 246-1,820
    # structural tokens all sit under `upper`, the closest 3,128 MiB under it.
    admits_from=246,
    # 4,040 structural tokens (L2000_5dei, 2,096 residues) has been measured
    # completing on the shipped program: 78,588-78,616 MiB across two jobs
    # (docs/archive/opendde-2k-single-card-2026-09-24.md) and 77,112.8 on the jctc-v3
    # snapshot. The law reads 80.9 GiB there and refused it. The bound is the
    # highest measurement plus the same-snapshot repeat spread (28 MiB):
    # 78,644 MiB, 131 MiB under the 78,775 MiB threshold of the 0.9 pool on the
    # 95.6 GiB card it ran on. Only this exact size: 4,034 and 4,041 still read
    # the law and are refused there.
    measured_uppers=((4040, (78616 + 28) * _MIB),),
)

#: OpenDDE with the trunk pinned to float32, which since 2026-09-11 is a
#: request rather than the default. One fitted size, measured twice --
#: 43,090 MiB and 42,291.2 at 1,902 structural tokens -- so this arm is a
#: square term with no intercept: one measurement determines one parameter.
#: The arguments are absorbed into the coefficient, which makes it read high
#: below its domain, and its domain does not reach down there. The coefficient
#: lands at 1.092 of the measured fp32 arena coefficient, i.e. an arena that
#: is 91.6% of the peak -- the share the bf16 arm shows too.
#:
#: Its own censored confirmation: at 2,978 structural tokens (1,531 residues)
#: this arm asked for 93.40 GiB and died, and the documented escape hatch
#: (`PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla`) asked 80.72 GiB and died
#: too. Above the wall no backend choice helps and only bfloat16 does, which
#: is why the bf16 trunk is the lever this law's refusal names.
OPENDDE_FP32_PEAK = PeakLaw(
    model="opendde",
    profile=(
        "fp32 trunk, 5 samples, released schedule, structural tokens"
    ),
    coeffs=(("fp32 trunk", (("n2", 0.0118007941),)),),
    domain_tokens=(1902, 7876),
    allowance_bytes=1199 * _MIB,
    calibration_id=f"opendde-fp32-{_CALIBRATION_STRUCTURAL}",
)

#: ESMFold2, refitted 2026-09-23 on the rolled block-loop program (main
#: ``30ce707``): 35,023.9 MiB at 2,096 tokens (5DEI, measured pass) and
#: 69,350.8 at 3,012 (6ZTX, measured pass with the pool preallocated). The
#: released program read 14,733.3 / 46,041.8 at 1,003 / 2,096 and could not
#: run 3,012 on one card; the blocked, rolled and lent-buffer trunk of
#: `models/esmfold2/models/trunk.py` completes it, so the domain moves up to
#: the two sizes that were measured on this code. 1,003 tokens is below the
#: domain now and reads through the fit as an extrapolation.
#:
#: **There is no sample term, and that is measured rather than assumed.**
#: The confidence head's own ``num_samples * L^2 * 4*c_z`` is divided away
#: by `confidence_sample_sequential` (on by default); what remains is the
#: folding trunk, which has no sample axis: on the released program the
#: 2,096-token peak moved 0.53% between 5 and 32 samples. The law is fitted
#: at 5, declared valid to 32 through :func:`off_profile_reason`'s
#: ``samples_validated``, and the allowance covers that row.
#:
#: The 3,012-token point needs ``XLA_PYTHON_CLIENT_PREALLOCATE`` at its JAX
#: default (true): with the pool grown on demand the same program fails on
#: allocator fragmentation asking a 66.5 GiB contiguous arena while less
#: than a tenth of the pool is in use. The law describes the program, not
#: the allocator; the CLI leaves the JAX default in place.
#:
#: The fit predates the expected PAE/PDE matrices being returned by default
#: (``return_expected_errors``). Re-measured on that default program at
#: 3,012 tokens: 69,508.4 MiB against the law's upper 69,645.8 MiB, so the
#: head's matrices sit inside the allowance and the fit stands.
ESMFOLD2_PEAK = PeakLaw(
    model="esmfold2",
    profile="released schedule, 5-32 samples, sequential confidence head",
    coeffs=(("whole run", (("1", 2793.12271), ("n2", 0.00733648819))),),
    domain_tokens=(2096, 3012),
    allowance_bytes=295 * _MIB,
    calibration_id=f"esmfold2-{_CALIBRATION_ESMFOLD2}",
)

#: The sample count every law above was fitted at.
CALIBRATED_NUM_SAMPLES = 5

#: Per port, the highest sample count it measured its law still describes:
#: :func:`off_profile_reason`'s ``samples_validated``, read by the port's own
#: admission and by `foldjax plan`, so the two agree. Only ESMFold2 measured
#: one (46,041.8 MiB at 5 samples against 46,284.8 at its released 32, at
#: 2,096 tokens).
SAMPLES_VALIDATED: dict[str, int] = {"esmfold2": 32}

#: Ports whose admission treats any nucleic-acid or ligand token as outside
#: the fitted composition, where the estimate is only a lower bound: every
#: OpenFold3 point was protein-only, and 5NPK (DNA gyrase with DNA and
#: ligands, 3,061 tokens) peaked at 41,260 MiB against the chunked law's
#: 29,876 MiB upper estimate.
PROTEIN_ONLY_LAWS = frozenset({"openfold3"})


def non_protein_reason(tokens: int | str) -> str:
    """Why a `PROTEIN_ONLY_LAWS` estimate is a lower bound for this run."""
    return (
        f"{tokens} nucleic-acid or ligand tokens "
        "(the law was fitted on protein-only inputs)"
    )


def off_profile_reason(
    *,
    num_samples: int,
    extras: Sequence[str] = (),
    samples_validated: int | None = None,
) -> tuple[str, ...]:
    """What about this run the laws were not fitted at, if anything.

    An estimate fitted at five samples over-predicts a one-sample run, and an
    over-prediction under a refusing policy is a run that never starts. So a
    run outside the calibrated configuration keeps its estimate -- it is still
    the best number available -- and loses the refusal.

    ``samples_validated`` raises the top of the sample range this is silent
    about, and only a port that *measured* a second count may pass it.
    ESMFold2 is the one that did: at 2,096 tokens its peak is 46,041.8 MiB at
    5 samples and 46,284.8 at 32, a 0.53% move, because its peak is a folding
    trunk with no sample axis -- and 32 is its released count, so treating it
    as off profile would mean a refusal that never fires on a default run. The
    law's allowance carries the difference. A count *below* five is still off
    profile everywhere, including there: that is the direction in which the
    estimate reads high.
    """
    high = CALIBRATED_NUM_SAMPLES if samples_validated is None else samples_validated
    if high < CALIBRATED_NUM_SAMPLES:
        raise ValueError(
            "samples_validated cannot be below the count the laws were fitted "
            f"at ({CALIBRATED_NUM_SAMPLES}); got {samples_validated!r}"
        )
    reasons = []
    if not CALIBRATED_NUM_SAMPLES <= num_samples <= high:
        fitted = (
            str(CALIBRATED_NUM_SAMPLES)
            if high == CALIBRATED_NUM_SAMPLES
            else f"{CALIBRATED_NUM_SAMPLES}-{high}"
        )
        reasons.append(
            f"{num_samples} sample{'' if num_samples == 1 else 's'} rather "
            f"than the {fitted} the law was fitted at"
        )
    reasons.extend(extras)
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    """One configuration's estimated peak, and whether it is admitted."""

    name: str
    estimate_bytes: int
    upper_bytes: int
    #: ``None`` when there is no budget to compare against.
    fits: bool | None


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """What the policy concluded, and enough of the working to report it."""

    state: State
    #: The admitted candidate's name, or ``None`` unless ``state`` is "fits".
    selected: str | None
    estimates: tuple[CandidateEstimate, ...]
    #: Bytes a candidate's upper estimate must not exceed, or ``None``.
    threshold: int | None
    reason: str


def exceeding_profile(
    decision: MemoryDecision, exceeds_profile: Sequence[str] = ()
) -> MemoryDecision:
    """The decision as it stands for a run that needs more than the law says.

    The mirror of :func:`off_profile_reason`, and deliberately not the same
    mechanism. A run configured *below* the fitted profile -- fewer samples,
    a split arena -- keeps its estimate and loses the refusal, because the
    estimate reads high. A run configured *above* it -- serving padding, a
    float32 trunk or pair stream, float32 matmuls -- has an estimate that
    reads low: measured, those runs exceeded the upper estimate by up to
    1.69x. So its refusal still binds, but a ``fits`` is an underestimate
    clearing the threshold, which guarantees nothing, and becomes
    ``unknown``. Idempotent, so every layer that reports the decision can
    apply it.
    """
    if decision.state != "fits" or not exceeds_profile:
        return decision
    return replace(
        decision,
        state="unknown",
        selected=None,
        reason=(
            f"{decision.reason}, but that estimate is a lower bound for this "
            "run, which uses " + "; ".join(exceeds_profile) + " -- the law was "
            "fitted without them, so clearing the threshold is not a guarantee"
        ),
    )


def resolve_memory_policy(
    *,
    model: str,
    n_token: int,
    msa_rows: int | None,
    budget_bytes: int | None,
    candidates: Sequence[tuple[str, PeakLaw]],
) -> MemoryDecision:
    """Say whether a configuration's upper estimate fits the budget.

    A candidate is admitted when its upper estimate is at most
    :data:`ADMISSION_FRACTION` of ``budget_bytes``; ``candidates`` is walked in
    order and the first admitted one is selected. With none admitted the state
    is ``over_budget``. Every port here passes exactly one candidate -- the
    configuration it is about to run -- so the result is a verdict on that
    configuration rather than a choice between several, and the sequence is
    what carries that one estimate into the manifest under its own name.

    The state is ``unknown``, and the caller must proceed, whenever the
    comparison cannot be made: no budget, a token count outside any
    candidate's fitted range, or a law that needs an MSA row count the caller
    does not have. Estimates are still reported when they can be computed, so
    a run with no readable budget still records what it expected to need.
    """
    if not candidates:
        raise ValueError("resolve_memory_policy needs at least one candidate")
    for name, law in candidates:
        if law.model != model:
            raise ValueError(
                f"candidate {name!r} carries a {law.model} law, not {model}"
            )

    outside = [name for name, law in candidates if not law.bounds(n_token)]
    missing = [
        name for name, law in candidates if law.needs_msa_rows and msa_rows is None
    ]
    laws = dict(candidates)
    if outside or missing:
        detail = (
            f"{n_token} tokens is outside the fitted range of "
            + ", ".join(
                f"{name} ({_token_span(laws[name], admitted=True)})"
                for name in outside
            )
            if outside
            else f"no processed MSA row count for {', '.join(missing)}"
        )
        return MemoryDecision(
            state="unknown",
            selected=None,
            estimates=(),
            threshold=None if budget_bytes is None else _threshold(budget_bytes),
            reason=f"no {model} peak estimate: {detail}",
        )

    threshold = None if budget_bytes is None else _threshold(budget_bytes)
    estimates = tuple(
        _candidate(name, law, n_token, msa_rows, threshold) for name, law in candidates
    )
    if threshold is None:
        return MemoryDecision(
            state="unknown",
            selected=None,
            estimates=estimates,
            threshold=None,
            reason=(
                "no device memory budget is readable, so nothing was compared; "
                f"{_render(estimates[0])} estimated for {estimates[0].name}"
            ),
        )
    for candidate in estimates:
        if candidate.fits:
            return MemoryDecision(
                state="fits",
                selected=candidate.name,
                estimates=estimates,
                threshold=threshold,
                reason=(
                    f"{candidate.name} fits: {_render(candidate)} against a "
                    f"{_gib(threshold)} threshold"
                ),
            )
    below = [name for name, law in candidates if not law.covers(n_token)]
    if below:
        # Below the fitted domain the law was checked against completed runs
        # and only ever read high there: trusted to admit, not to refuse.
        return MemoryDecision(
            state="unknown",
            selected=None,
            estimates=estimates,
            threshold=threshold,
            reason=(
                f"{n_token} tokens is below the fitted range of "
                + ", ".join(
                    f"{name} ({_token_span(laws[name], admitted=False)})"
                    for name in below
                )
                + ", where its estimate bounds the measured runs but is not "
                "used to refuse one"
            ),
        )
    return MemoryDecision(
        state="over_budget",
        selected=None,
        estimates=estimates,
        threshold=threshold,
        reason=(
            f"no {model} configuration fits: "
            + ", ".join(f"{c.name} {_render(c)}" for c in estimates)
            + f" against a {_gib(threshold)} threshold"
        ),
    )


def _threshold(budget_bytes: int) -> int:
    return int(budget_bytes * ADMISSION_FRACTION)


def _token_span(law: PeakLaw, *, admitted: bool) -> str:
    """The token range ``law`` covers: fitted, or also the band below it
    where admission still uses it (:meth:`MemoryLaw.bounds`)."""
    low, high = law.domain_tokens
    if admitted and law.admits_from is not None:
        low = min(low, law.admits_from)
    return f"{low:,}-{high:,} tokens"


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f} GiB"


def _candidate(
    name: str,
    law: PeakLaw,
    n_token: int,
    msa_rows: int | None,
    threshold: int | None,
) -> CandidateEstimate:
    upper = law.upper(n_token, msa_rows)
    return CandidateEstimate(
        name=name,
        estimate_bytes=law.estimate(n_token, msa_rows),
        upper_bytes=upper,
        # At exactly the threshold a candidate fits: the threshold is the
        # largest peak this policy admits, not the smallest it refuses.
        fits=None if threshold is None else upper <= threshold,
    )


def _render(candidate: CandidateEstimate) -> str:
    allowance = candidate.upper_bytes - candidate.estimate_bytes
    if allowance < 0:
        # A `measured_uppers` size, where the measured bound replaced the law's.
        return (
            f"{_gib(candidate.upper_bytes)} measured at this size (the law "
            f"reads {_gib(candidate.estimate_bytes)})"
        )
    return f"{_gib(candidate.estimate_bytes)} + {_gib(allowance)} allowance"


# ---------------------------------------------------------------------------
# The budget side: what the allocator will give, and what the caller planned for
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """The ceiling admission is measured against, and where it came from."""

    #: The allocator's ceiling, or ``None`` when it cannot be read.
    pool_bytes: int | None
    #: The device's capacity, recovered from the pool fraction.
    card_bytes: int | None
    #: A caller's planning budget, or ``None``.
    override_bytes: int | None
    budget_bytes: int | None
    source: BudgetSource


def resolve_budget(
    *,
    pool_bytes: int | None,
    card_bytes: int | None,
    override_gib: float | None,
) -> MemoryBudget:
    """Combine the allocator's ceiling with a caller's planning budget.

    The smaller of the two wins when both are known: an override above the pool
    would admit a job the allocator will refuse, and one below it is a caller
    planning for a device they are not on -- or sharing this one.
    """
    override = None if override_gib is None else int(override_gib * _GIB)
    if override is not None and override <= 0:
        raise ValueError("a memory budget must be positive")
    if pool_bytes is None:
        budget, source = override, ("override" if override is not None else "none")
    elif override is None or override >= pool_bytes:
        budget, source = pool_bytes, "pool"
    else:
        budget, source = override, "override"
    return MemoryBudget(
        pool_bytes=pool_bytes,
        card_bytes=card_bytes,
        override_bytes=override,
        budget_bytes=budget,
        source=source,
    )


def device_memory_budget(*, override_gib: float | None = None) -> MemoryBudget:
    """Read the allocator's ceiling and fold in a planning override.

    The only function here that touches the device. It initializes the JAX
    backend, so a caller that is not about to run a model should not call it.
    """
    pool, card = oom.device_budget()
    return resolve_budget(pool_bytes=pool, card_bytes=card, override_gib=override_gib)


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

#: `--memory-check` values. "warn" prints what "refuse" would have raised.
CHECK_MODES = ("refuse", "warn")

#: What `--memory-check` does when the caller has not said.
DEFAULT_CHECK_MODE = "refuse"


def parse_check_mode(value: object | None) -> str:
    """Validate a ``memory_check`` option value, defaulting when absent."""
    if value is None:
        return DEFAULT_CHECK_MODE
    if not isinstance(value, str) or value not in CHECK_MODES:
        raise ValueError(f"memory_check must be one of {CHECK_MODES}; got {value!r}")
    return value


def parse_budget_gib(value: object | None) -> float | None:
    """Validate a ``memory_budget_gib`` option value, or pass ``None`` on."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"memory_budget_gib must be a number; got {value!r}")
    if value <= 0:
        raise ValueError(f"memory_budget_gib must be positive; got {value!r}")
    return float(value)


#: The lever every port has, named in every over-budget message.
_FRACTION_LEVER = (
    "raise the pool with --mem-fraction (it is a fraction of the device, and "
    "the device itself is the hard limit)"
)


def check_message(
    decision: MemoryDecision,
    *,
    model: str,
    n_token: int | None,
    msa_rows: int | None,
    budget: MemoryBudget,
    mode: str = DEFAULT_CHECK_MODE,
    levers: Sequence[str] = (),
    off_profile: Sequence[str] = (),
    token_label: str = "tokens",
    exceeds_profile: Sequence[str] = (),
) -> str:
    """Say what was estimated, what it was compared with, and what to change.

    ``token_label`` names the unit ``n_token`` is counted in. Every law but
    OpenDDE's is keyed on tokens; OpenDDE's is keyed on structural tokens,
    which at these sizes is about 1.9x the residue count the caller asked for,
    so a message that called them "tokens" would name a number the user never
    typed. ``n_token`` of ``None`` is a port with no law at all -- see
    :func:`admit_unmeasured` -- where there is no shape to report.
    ``exceeds_profile`` is applied as :func:`exceeding_profile` describes.
    """
    raw_state = decision.state
    decision = exceeding_profile(decision, exceeds_profile)
    shape = None if n_token is None else f"{n_token} {token_label}"
    if shape is not None and msa_rows is not None:
        shape += f", {msa_rows} processed MSA rows"
    lines = [
        f"{model} at {shape}: {decision.reason}."
        if shape is not None
        else f"{model}: {decision.reason}."
    ]
    if budget.pool_bytes is not None:
        lines.append(
            f"The allocator's pool is {_gib(budget.pool_bytes)}"
            + (
                f" of a {_gib(budget.card_bytes)} device"
                if budget.card_bytes is not None
                else ""
            )
            + f", at {oom.mem_fraction():g}."
        )
    if budget.source == "override" and budget.override_bytes is not None:
        lines.append(
            f"--memory-budget-gib held the budget to {_gib(budget.override_bytes)}."
        )
    # With no estimate there is nothing for the off-profile note to qualify.
    if off_profile and decision.estimates:
        lines.append(
            "This estimate is advisory rather than binding: the run uses "
            + "; ".join(off_profile)
            + "."
        )
    if decision.state == "unknown":
        lines.append(
            "The run proceeds; the allocator will answer."
            if raw_state == "fits"
            else "The run proceeds without the check."
        )
        return " ".join(lines)
    lines.append("Levers: " + "; ".join((_FRACTION_LEVER, *levers)) + ".")
    if mode == "refuse" and not off_profile:
        lines.append(
            "--memory-check=warn runs it anyway and lets the allocator answer."
        )
    else:
        lines.append("Running anyway; the allocator will answer.")
    return " ".join(lines)


def enforce(
    decision: MemoryDecision,
    *,
    model: str,
    n_token: int | None,
    msa_rows: int | None,
    budget: MemoryBudget,
    mode: str = DEFAULT_CHECK_MODE,
    levers: Sequence[str] = (),
    off_profile: Sequence[str] = (),
    token_label: str = "tokens",
    exceeds_profile: Sequence[str] = (),
) -> None:
    """Raise, warn, or say nothing, according to the state and the mode.

    ``fits`` is silent: a policy that reported every admitted run would be
    ignored by the time it mattered. ``unknown`` warns once per process, so a
    run with no readable budget still says that nothing was checked.
    ``over_budget`` raises under "refuse" and warns under "warn"; an
    off-calibration run warns whatever the mode, because the estimate that
    would justify the refusal does not describe it. A ``fits`` under a
    non-empty ``exceeds_profile`` is ``unknown`` (:func:`exceeding_profile`)
    and warns once per model on its own key, so an unreadable ceiling earlier
    in the process does not silence it.
    """
    if mode not in CHECK_MODES:
        raise ValueError(f"memory_check must be one of {CHECK_MODES}; got {mode!r}")
    demoted = decision.state == "fits" and bool(exceeds_profile)
    state = exceeding_profile(decision, exceeds_profile).state
    if state == "fits":
        return
    message = check_message(
        decision,
        model=model,
        n_token=n_token,
        msa_rows=msa_rows,
        budget=budget,
        mode=mode,
        levers=levers,
        off_profile=off_profile,
        token_label=token_label,
        exceeds_profile=exceeds_profile,
    )
    if state == "unknown":
        # Once per model per process. A run with no readable ceiling says so,
        # and a multi-seed request does not say it five times.
        key = f"exceeds_profile:{model}" if demoted else f"unknown:{model}"
        if not _first_time(key):
            return
    elif mode == "refuse" and not off_profile:
        raise MemoryError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


_WARNED: set[str] = set()


def _first_time(key: str) -> bool:
    """True the first time this process reaches ``key``, False after."""
    if key in _WARNED:
        return False
    _WARNED.add(key)
    return True


def reset_warnings() -> None:
    """Forget which one-time warnings have fired. For tests."""
    _WARNED.clear()


# ---------------------------------------------------------------------------
# Provenance: the decision reaches the manifest from wherever it was taken
# ---------------------------------------------------------------------------

#: The last decision this prediction took. The port that takes it is several
#: layers below the manifest writer -- inside a native CLI, in one case -- and
#: threading a field up through six backends to record one provenance block is
#: more surface than the block is worth. `foldjax.api` clears this before each
#: prediction, so a stale decision cannot be attributed to the next run; the
#: same pattern, for the same reason, as `models/_capture.py`.
_RECORDED: ContextVar[dict[str, Any] | None] = ContextVar(
    "foldjax_memory_decision", default=None
)


def record(
    decision: MemoryDecision,
    *,
    model: str,
    n_token: int | None,
    msa_rows: int | None,
    budget: MemoryBudget,
    calibration_id: str | None,
    mode: str,
    off_profile: Sequence[str] = (),
    exceeds_profile: Sequence[str] = (),
) -> None:
    """Keep this decision for the run manifest. JSON-native values only.

    The state recorded is the one :func:`exceeding_profile` leaves, so a
    manifest never says ``fits`` for a run whose estimate was a lower bound.
    """
    decision = exceeding_profile(decision, exceeds_profile)
    _RECORDED.set(
        {
            "model": model,
            "n_token": None if n_token is None else int(n_token),
            "msa_rows": None if msa_rows is None else int(msa_rows),
            "state": decision.state,
            "selected": decision.selected,
            "estimates": [
                {
                    "name": candidate.name,
                    "estimate_bytes": int(candidate.estimate_bytes),
                    "upper_bytes": int(candidate.upper_bytes),
                    "fits": candidate.fits,
                }
                for candidate in decision.estimates
            ],
            "threshold_bytes": decision.threshold,
            "pool_bytes": budget.pool_bytes,
            "card_bytes": budget.card_bytes,
            "budget_bytes": budget.budget_bytes,
            "budget_source": budget.source,
            "calibration_id": calibration_id,
            "mem_fraction": oom.mem_fraction(),
            "memory_check": mode,
            "off_profile": list(off_profile),
            "exceeds_profile": list(exceeds_profile),
        }
    )


def recorded() -> dict[str, Any] | None:
    """The decision this prediction took, or ``None`` if none was taken."""
    return _RECORDED.get()


def clear_record() -> None:
    """Forget the last decision, so the next run reports only its own."""
    _RECORDED.set(None)


def admit(
    *,
    model: str,
    n_token: int,
    msa_rows: int | None,
    candidates: Sequence[tuple[str, PeakLaw]],
    budget: MemoryBudget,
    mode: str = "refuse",
    levers: Sequence[str] = (),
    off_profile: Sequence[str] = (),
    token_label: str = "tokens",
    exceeds_profile: Sequence[str] = (),
) -> MemoryDecision:
    """Resolve, record and enforce in one call: what a port's wiring needs.

    ``off_profile`` names what about this run needs *less* than the law
    describes; ``exceeds_profile`` names what needs *more*. See
    :func:`off_profile_reason` and :func:`exceeding_profile`. The decision
    returned is the one recorded and enforced.
    """
    decision = resolve_memory_policy(
        model=model,
        n_token=n_token,
        msa_rows=msa_rows,
        budget_bytes=budget.budget_bytes,
        candidates=candidates,
    )
    selected = dict(candidates).get(decision.selected or "")
    record(
        decision,
        model=model,
        n_token=n_token,
        msa_rows=msa_rows,
        budget=budget,
        calibration_id=(
            selected.calibration_id
            if selected is not None
            else candidates[0][1].calibration_id
        ),
        mode=mode,
        off_profile=off_profile,
        exceeds_profile=exceeds_profile,
    )
    enforce(
        decision,
        model=model,
        n_token=n_token,
        msa_rows=msa_rows,
        budget=budget,
        mode=mode,
        levers=levers,
        off_profile=off_profile,
        token_label=token_label,
        exceeds_profile=exceeds_profile,
    )
    return exceeding_profile(decision, exceeds_profile)


def admit_unmeasured(
    *,
    model: str,
    budget: MemoryBudget,
    mode: str = DEFAULT_CHECK_MODE,
    detail: str,
    n_token: int | None = None,
) -> MemoryDecision:
    """Answer the memory flags on a port whose peak nobody has fitted.

    ``--memory-check`` and ``--memory-budget-gib`` are one vocabulary across
    every port, and a port with no law cannot admit or refuse. Before this
    existed the two flags were not accepted at all on such a port, so asking
    for them ended the run with "unsupported <port> options" -- which reads
    like a spelling mistake rather than like a missing measurement.

    So the answer is the third state this module already has. ``unknown`` is
    reached deliberately here rather than by an uncovered size: the run
    proceeds, with one warning per process that names the port and says the
    check did not happen. The decision reaches the manifest like any other, so
    a finished run records that nothing was compared and why.
    """
    decision = MemoryDecision(
        state="unknown",
        selected=None,
        estimates=(),
        threshold=(
            None
            if budget.budget_bytes is None
            else _threshold(budget.budget_bytes)
        ),
        reason=f"no {model} peak law is fitted, so nothing was compared: {detail}",
    )
    record(
        decision,
        model=model,
        n_token=n_token,
        msa_rows=None,
        budget=budget,
        calibration_id=None,
        mode=mode,
    )
    enforce(
        decision,
        model=model,
        n_token=n_token,
        msa_rows=None,
        budget=budget,
        mode=mode,
    )
    return decision
