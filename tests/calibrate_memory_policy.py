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


#: OpenDDE, bf16 trunk (the released default since 2026-09-11), 5 samples,
#: released 200-step/10-cycle schedule, serial. The token count here is the
#: **structural** token count, not the residue count: OpenDDE folds in
#: structural-token space and the ratio to residues drifts with composition
#: (1.896 at 1,003 residues, 1.945 at 1,531 and at 4,100), so a law keyed on
#: residues would import that drift squared.
#:
#: Two completed rows:
#:
#: * N_st 1,902 (1,003 residues, 3OG2) -> 21,492 MiB. The serial row of
#:   `docs/scale-rows-master-2026-09-10.md` (the 1k comparison table and the
#:   context-parallel table, which quote the same serial peak).
#: * N_st 2,978 (1,531 residues) -> 45.76 GiB = 46,858.2 MiB. The blocked-arm
#:   side of the 2026-08-23 clocked A/B recorded in the `opendde-arena-law-and-
#:   fp32-ceiling` note; N_st = 2,978 was recovered there from the buffer dump.
#:
#: `a + c*n^2` over two points is an exact fit, which normally means a law
#: with no residual to take an allowance from. It is kept here because the two
#: fitted parameters are separately corroborated rather than free:
#:
#: * `a` comes out 4,016 MiB = 3.92 GiB, inside the 3.0-4.6 GiB of arguments
#:   (weights plus features) this port was measured to carry across these
#:   sizes -- the term that is *not* the quadratic arena.
#: * `c` comes out 0.883 of the fused-arm arena coefficient 5.4724e-3 MiB per
#:   structural-token pair, against 0.866 measured independently: the bf16
#:   trunk routes its triangle multiplication to the blocked XLA path, which
#:   took 2,650 MiB off the 19,797 MiB fused arena at N_st 1,902.
#:
#: So the allowance is the snapshot spread alone: the same 1,003-residue bf16
#: configuration read 19.85 GiB = 20,326.4 MiB on 2026-08-23 and 21,492 MiB on
#: 2026-09-10, 1,165.6 MiB apart, and the 45.76 GiB row is quoted to 0.01 GiB
#: (5.12 MiB).
OPENDDE_BF16_POINTS = (
    # (structural tokens, processed msa rows, peak MiB)
    (1902, None, 21492.0),
    (2978, None, 46858.2),
)

#: The same configuration one snapshot earlier, kept as a validation row
#: rather than a fit point: a law fitted across two snapshots describes
#: neither, and this is where the 1,165.6 MiB spread above comes from.
OPENDDE_BF16_VALIDATION = ((1902, None, 20326.4),)

#: OpenDDE with the trunk pinned to float32 (`--trunk-dtype fp32`), 5 samples,
#: released schedule, serial. One fitted size, measured twice:
#:
#: * N_st 1,902 -> 43,090 MiB, 2026-08-23, 5 samples (the figure the retired
#:   arena preflight carried in its own docstring, beside 42,877 at 1 sample).
#: * N_st 1,902 -> 41.3 GiB = 42,291.2 MiB, the 2026-09-10 serial row, taken
#:   when fp32 was still this port's released default.
#:
#: One fitted size determines one parameter, so this arm is `c*n^2` with no
#: intercept: the arguments are absorbed into the coefficient, which makes the
#: law read high below its domain and is why its domain does not reach down
#: there. The coefficient lands at 1.092 of the measured fp32 *arena*
#: coefficient 1.0807e-2, i.e. the arena is 91.6% of the peak -- the same
#: arena-to-peak share the bf16 arm shows, so the shape is not this arm's own
#: invention.
#:
#: The allowance is the in-sample underestimate (half the 799 MiB the two
#: measurements are apart, so the higher of them is covered by construction)
#: plus the 798.8 MiB they are apart, which is what a repeat of this
#: configuration was measured to move.
OPENDDE_FP32_POINTS = (
    (1902, None, 43090.0),
    (1902, None, 42291.2),
)

#: ESMFold2, released defaults, 5 samples, 200 steps, 10 cycles, serial, in
#: token counts (this port has no structural-token space):
#:
#: * 1,003 tokens -> 14,733.3 MiB
#: * 2,096 tokens -> 46,041.8 MiB
#:
#: Both from the control arm of GPU rows 1111/1112, transcribed at
#: `models/esmfold2/models/model.py`'s `confidence_dtype` note, which is where
#: this port's peaks are recorded to the tenth of a MiB; the ledger's 14.4 and
#: 45.0 GiB are the same two runs rounded.
#:
#: **No sample term, and that is measured rather than assumed.** Three places
#: in this repository still describe this peak as `num_samples * L^2 * 4*c_z`;
#: that is the confidence head's own term, and `confidence_sample_sequential`
#: (on by default) divides it away. What is left is the folding trunk, which
#: has no sample axis at all: at 2,096 tokens the peak is 46,041.8 MiB at 5
#: samples and 45.2 GiB = 46,284.8 at 32 -- the released count -- a 0.53%
#: move. So the law is fitted at 5, declared valid to 32, and its allowance
#: covers the 32-sample row (243.0 MiB above the fit, plus the 51.2 MiB that
#: row is quoted to). The true repeat is 0.5 MiB: the `confidence_dtype` arm
#: read 46,042.3 against the control's 46,041.8 at the same size.
ESMFOLD2_POINTS = (
    # Refitted 2026-09-23 on the rolled block-loop program (main 30ce707):
    # 5DEI at 2,096 (measured pass) and 6ZTX at 3,012 (measured pass, pool
    # preallocated). The released program's 14,733.3 / 46,041.8 at
    # 1,003 / 2,096 are superseded; 1,003 is not re-measured on this code.
    (2096, None, 35023.9),
    (3012, None, 69350.8),
)

#: The released 32-sample row at 2,096 tokens. A completed run, so the
#: allowance has to admit it, but not a fit point: it is a different sample
#: count, and the point of quoting it is that the peak barely noticed.
ESMFOLD2_VALIDATION = ((2096, None, 35023.9 + (46284.8 - 46041.8)),)


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

    # Keyed by the point's position as well as its shape: OpenDDE's float32
    # arm is the same configuration measured twice at one size, and a
    # shape-keyed mapping would keep only the second of them -- which is the
    # one whose residual is negative, so the allowance would have been taken
    # from a law that underestimates a completed run by 399 MiB.
    residuals = {}
    for index, (n, m, peak) in enumerate(points):
        residuals[(index, n, m)] = peak - _evaluate(phases, coeffs, n, m)

    folds, skipped = {}, []
    for size in sorted({n for n, _, _ in points}):
        kept = [[row for row in group if row[0] != size] for group in phase_points]
        refit = [_fit(terms, group) for (_n, terms), group in zip(phases, kept)]
        if any(fitted is None for fitted in refit):
            skipped.append(size)
            continue
        for index, (n, m, peak) in enumerate(points):
            if n == size:
                folds[(index, n, m)] = peak - _evaluate(phases, refit, n, m)

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
        "worst_residual": max(
            abs(residuals[(index, n, m)]) / peak
            for index, (n, m, peak) in enumerate(points)
        ),
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
        # Both OpenDDE arms are keyed on the *structural* token count, and
        # both domains end at 7,876 -- the structural token count of a
        # 4,100-residue job. Nothing above 2,978 was fitted, and no failure
        # was: what the top of the domain rests on is that the law's
        # *verdict* has been checked against an outcome there. At N_st ~4,034
        # (2,096 residues, 5DEI) this port asked for an 87 GiB arena and died
        # on a 95.6 GiB pool, serial and on both 1-D layouts; at 7,876 it
        # asked for 331.5 GiB and died. `a + c*n^2` with both coefficients
        # nonnegative bends nowhere in between, so the refusal it gives there
        # is the refusal those runs earned. Above 7,876 the answer is
        # `unknown` and the run proceeds.
        "opendde_bf16": _calibrate(
            (("bf16 trunk", ("1", "n2")),),
            (OPENDDE_BF16_POINTS,),
            (1902, 7876),
            repeat_spread=(21_492.0 - 20_326.4) + 5.12,
        ),
        # The fp32 arm's censored confirmation is its own: at N_st 2,978
        # (1,531 residues) it asked for 93.40 GiB and died, and the documented
        # escape hatch (`PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla`) asked
        # 80.72 GiB and died too.
        "opendde_fp32": _calibrate(
            (("fp32 trunk", ("n2",)),),
            (OPENDDE_FP32_POINTS,),
            (1902, 7876),
            repeat_spread=43_090.0 - 42_291.2,
        ),
        # Fitted at 5 samples over 2,096-3,012 tokens on the rolled program
        # (2026-09-23); the 32-sample validation row keeps the released
        # program's 243.0 MiB sample spread on top of the new 2,096 point.
        "esmfold2": _calibrate(
            (("whole run", ("1", "n2")),),
            (ESMFOLD2_POINTS,),
            (2096, 3012),
            repeat_spread=(46_284.8 - 46_041.8) + 51.2,
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
    "opendde_bf16": OPENDDE_BF16_POINTS + OPENDDE_BF16_VALIDATION,
    "opendde_fp32": OPENDDE_FP32_POINTS,
    "esmfold2": ESMFOLD2_POINTS + ESMFOLD2_VALIDATION,
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
