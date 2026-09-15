"""Re-derive the peak-law literals in ``foldjax.memory_policy`` from measurements.

Not a test -- ``pytest`` does not collect it. It lives here because the numbers
it prints are frozen as literals in the shipped module, and a frozen number
whose derivation is not runnable is a number nobody can check.
``tests/test_memory_policy.py::test_the_frozen_laws_are_what_this_script_fits``
imports :func:`fit_all` and compares.

Run it with the repository's interpreter::

    python tests/calibrate_memory_policy.py

Every peak below is ``peak_bytes_in_use`` read from XLA on one 95.6 GiB device,
in MiB, at the released defaults of its port. Laws are fitted by nonnegative
least squares: a negative coefficient fits these points slightly better and
then bends the wrong way just outside them, and an admission policy is asked
about sizes larger than the ones it was fitted on.

A law is a **maximum over phase laws**, not a sum, because a peak is one
moment: a program whose trunk and whose diffusion each need 20 GiB, at
different times, peaks at 20 and not 40. Most of these have one phase.
Protenix has two, and the measurements say so -- see :data:`PROTENIX_PAIR` .

The allowance is the largest *underestimate* the law makes on the points it was
fitted on, plus the spread the same measurement shows when it is repeated.

It was leave-one-size-out first, and that rule refused a run that had
completed: Boltz-2 at 4,888 tokens peaked at 77,312 MiB inside an 87.5 GiB
pool, and a 3,451 MiB fold allowance put its upper estimate at 80,320 against a
78,775 MiB threshold. A policy whose job is to say "this will not fit" must not
say it about a run that did, so the rule is now the in-sample one and the first
thing :func:`main` prints is every fit point's verdict.

The folds are still computed and still printed, because they answer the other
question: how far wrong the law is likely to be at a size nobody measured.
Dropping the smallest or the largest size makes that fold an extrapolation,
which is exactly why its error is not a margin to carry at a measured size. A
fold the remaining points cannot determine is reported and skipped.

The repeat spread is the floor under any residual: re-running the identical
configuration does not always give the identical peak. Measured at 2,096
tokens -- OpenFold3 repeats to the MiB, Boltz-2 gave 18,511 twice, and Protenix
gave 21,225.3 / 21,225.4 / 21,253.7 across snapshots -- so only Protenix
carries a nonzero one.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import nnls

#: Basis term name -> value, given a token count and a processed MSA row count.
_TERMS = {
    "1": lambda n, m: 1.0,
    "n": lambda n, m: float(n),
    "n2": lambda n, m: float(n) ** 2,
    "n3": lambda n, m: float(n) ** 3,
    "mn": lambda n, m: float(m) * float(n),
}

#: Boltz-2, bf16 trunk, fp32 diffusion, 5 samples, released schedule.
BOLTZ2_POINTS = (
    # (n_token, processed msa rows, peak MiB)
    (1003, None, 8453),
    (2096, None, 18511),
    (3012, None, 29416),
    (4100, None, 51712),
    (4888, None, 77312),
)

#: OpenFold3 with the pair-stack row loop blocked at the resolved width of 128
#: rows (`inference.RESOLVED_PAIR_CHUNK_SIZE`), released schedule. This is the
#: automatic configuration from 1,003 tokens up, so this is the law admission
#: is given.
OF3_CHUNKED_POINTS = (
    (1003, None, 4316.7),
    (2096, None, 13882.0),
    (3012, None, 23774.0),
    (4100, None, 42468.6),
    (4888, None, 59214.8),
)

#: The same runs with the row loop unblocked (``pair_chunk_size=0``). Three
#: sizes, which is what makes this arm fittable on its own: a two-parameter law
#: over three token counts still leaves a leave-one-size-out fold determined.
#: The quadratic is the form: adding a linear term fits all three points
#: exactly and then leaves every fold underdetermined, which is a fit with no
#: allowance rather than a better one.
#:
#: Nothing was measured above 3,012 tokens on this arm, so
#: :data:`domain_tokens` stops there. This arm is not an automatic candidate:
#: it costs 44% more peak at 1,003 tokens for equal wall time and 65% more at
#: 2,096 for 3.4% less, so the blocked width is the automatic answer at every
#: size in the validated domain and this law is the estimate for a run that
#: asks for the unblocked loop by name.
OF3_UNCHUNKED_POINTS = (
    (1003, None, 6194.6),
    (2096, None, 22967.0),
    (3012, None, 45363.0),
)

#: Protenix's pair phase: runs whose MSA cap is low enough that the peak stops
#: moving with it. Measured at 2,096 tokens, 4,096 rows cost 13,754 MiB and
#: 2,048 cost 13,727 -- 27 MiB apart over half the alignment -- so what sets
#: the peak there is not the alignment.
PROTENIX_PAIR = (
    (2096, 4096, 13754),
    (2096, 2048, 13727),
    (3012, 4096, 27059),
)

#: Protenix's MSA phase: runs where the peak does move with the row count.
#: At 2,096 tokens, 8,192 rows cost 14,730 MiB against the pair phase's 13,741,
#: and full depth (13,267 rows under the released 16,384 cap) costs 21,225.
#: ``c * rows * tokens`` with no intercept: least squares puts a +1,632 MiB
#: intercept there and then reads 20% high at 1,003 tokens, where the whole
#: peak is 6,404 MiB.
#:
#: An older 4,100-token point exists and is excluded from both phases: it was
#: taken on a different snapshot, and a law fitted across two configurations
#: describes neither.
PROTENIX_MSA = (
    (1003, 8808, 6404),
    (2096, 8192, 14730),
    (2096, 13267, 21225),
    (3012, 17542, 37537),
)


def _design(terms, points):
    return np.array(
        [[_TERMS[name](n, m) for name in terms] for n, m, _ in points], float
    )


def _fit(terms, points):
    """NNLS, or ``None`` when these points cannot determine these terms."""
    if len(points) < len(terms):
        return None
    design = _design(terms, points)
    if np.linalg.matrix_rank(design) < len(terms):
        return None
    coeffs, _ = nnls(design, np.array([p for _, _, p in points], float))
    return coeffs


def _evaluate(phases, coeffs, n_token, msa_rows):
    return max(
        float(_design(terms, [(n_token, msa_rows, 0)])[0] @ fitted)
        for (_name, terms), fitted in zip(phases, coeffs)
    )


def _calibrate(phases, phase_points, domain, repeat_spread=0.0):
    """Fit every phase, then take the allowance from the in-sample residuals.

    ``phases`` is ``((name, terms), ...)`` and ``phase_points`` holds the
    points belonging to each, in the same order. ``repeat_spread`` is how far
    apart the same measurement came out when it was repeated, in MiB.

    The folds are computed here too, and reported by :func:`main`, but they do
    not reach the allowance: see this module's docstring for the run they
    refused.
    """
    coeffs = [_fit(terms, points) for (_n, terms), points in zip(phases, phase_points)]
    if any(fitted is None for fitted in coeffs):
        raise ValueError("the measured points do not determine every phase")
    points = [row for group in phase_points for row in group]

    residuals = {}
    for n, m, peak in points:
        residuals[(n, m)] = peak - _evaluate(phases, coeffs, n, m)

    folds, skipped = {}, []
    for size in sorted({n for n, _, _ in points}):
        kept = [[row for row in group if row[0] != size] for group in phase_points]
        refit = [_fit(terms, group) for (_n, terms), group in zip(phases, kept)]
        if any(fitted is None for fitted in refit):
            skipped.append(size)
            continue
        for n, m, peak in [row for row in points if row[0] == size]:
            folds[(n, m)] = peak - _evaluate(phases, refit, n, m)

    under = [value for value in residuals.values() if value > 0]
    return {
        "phases": tuple(
            (name, tuple(terms), tuple(float(v) for v in fitted))
            for (name, terms), fitted in zip(phases, coeffs)
        ),
        "allowance_mib": math.ceil(max(under, default=0.0) + repeat_spread),
        "domain_tokens": domain,
        "folds": folds,
        "skipped_folds": tuple(skipped),
        "residuals": residuals,
        "repeat_spread_mib": repeat_spread,
        "worst_residual": max(abs(residuals[(n, m)]) / peak for n, m, peak in points),
    }


def fit_all() -> dict[str, dict]:
    """Every law's fitted phases, allowance and domain."""
    fits = {
        # The local exponent of the five Boltz-2 points rises from 1.06 to
        # 2.29, so one square term cannot hold both ends: fitted as
        # `a + c*n^2` the same points come out 12% high at 1,003 tokens and the
        # allowance triples. NNLS drives the square term of a full cubic to
        # zero, which is why it is absent.
        "boltz2": _calibrate(
            (("whole run", ("1", "n", "n3")),), (BOLTZ2_POINTS,), (1003, 4888)
        ),
        # The one law with a nonzero repeat spread: 2,096 tokens at full depth
        # came out 21,225.3 / 21,225.4 / 21,253.7 MiB across snapshots, so a
        # residual smaller than 28.4 MiB would be measuring the harness.
        "protenix": _calibrate(
            (("pair", ("1", "n2")), ("msa", ("mn",))),
            (PROTENIX_PAIR, PROTENIX_MSA),
            (1003, 3012),
            repeat_spread=21_253.7 - 21_225.3,
        ),
        # A linear term as well as the square: fitted as `a + c*n^2` the same
        # five points read 14% high at 1,003 tokens, where the whole peak is
        # 4.3 GiB, and the allowance grows rather than shrinks.
        "openfold3_chunked": _calibrate(
            (("chunked", ("1", "n", "n2")),), (OF3_CHUNKED_POINTS,), (1003, 4888)
        ),
        "openfold3_unchunked": _calibrate(
            (("unchunked", ("1", "n2")),), (OF3_UNCHUNKED_POINTS,), (1003, 3012)
        ),
    }
    return fits


#: The ceiling this host reports at the CLI's own 0.9 fraction, for the
#: admission table below: ``bytes_limit`` = 91,779,760,128 B.
_HOST_POOL_MIB = 91_779_760_128 / 2**20

_POINTS = {
    "boltz2": BOLTZ2_POINTS,
    "protenix": PROTENIX_PAIR + PROTENIX_MSA,
    "openfold3_chunked": OF3_CHUNKED_POINTS,
    "openfold3_unchunked": OF3_UNCHUNKED_POINTS,
}


def main() -> None:
    from foldjax.memory_policy import ADMISSION_FRACTION

    threshold = ADMISSION_FRACTION * _HOST_POOL_MIB
    for name, fit in fit_all().items():
        print(f"=== {name}")
        for phase, terms, coeffs in fit["phases"]:
            rendered = ", ".join(
                f"{term}={value:.9g}" for term, value in zip(terms, coeffs)
            )
            print(f"  phase {phase}: {rendered}")
        print(f"  allowance  {fit['allowance_mib']} MiB")
        print(f"  domain     {fit['domain_tokens']}")
        print(f"  worst in-sample residual {100 * fit['worst_residual']:.2f}%")
        print(f"  repeat spread {fit['repeat_spread_mib']:.1f} MiB")
        print("  in-sample residuals (actual - predicted, MiB), the allowance:")
        for key, value in sorted(fit["residuals"].items(), key=lambda kv: str(kv[0])):
            print(f"    {key}: {value:+.0f}")
        print("  leave-one-size-out errors -- an extrapolation diagnostic, not")
        print("  the allowance; see this module's docstring:")
        for key, value in sorted(fit["folds"].items(), key=lambda kv: str(kv[0])):
            print(f"    {key}: {value:+.0f}")
        for size in fit["skipped_folds"]:
            print(f"    {size}: skipped, the remaining points underdetermine a phase")
        # An admission policy that refuses a run which demonstrably completed
        # is a defect of its own, so say which of the fit points this law
        # admits against the ceiling the CLI asks for by default.
        print(f"  admission against {threshold:.0f} MiB (0.9 of this host's pool):")
        for n, m, peak in _POINTS[name]:
            estimate = _evaluate(
                tuple((phase, terms) for phase, terms, _c in fit["phases"]),
                [np.array(c) for _p, _t, c in fit["phases"]],
                n,
                m,
            )
            upper = estimate + fit["allowance_mib"]
            verdict = (
                f"fits, {threshold - upper:.0f} MiB spare"
                if upper <= threshold
                else "REFUSED"
            )
            print(
                f"    n={n} m={m}: measured {peak} est {estimate:.0f} "
                f"upper {upper:.0f} -> {verdict}"
            )


if __name__ == "__main__":
    main()
