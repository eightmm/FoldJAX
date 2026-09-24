"""Model-level rows for the Pallas pair kernels (x43), against the x42 campaign rows.

Each row is the JCTC campaign's own body (`jctc/submit.py:script`, one cold and
one warm pass). Its code, results, work, log and cache paths are moved into this
directory and pointed at the whole-tree snapshot `code-<sha>`. The row id carries
a suffix naming the arm:

* `-pallas`  the port's Pallas kernels on the multiplication and the transition,
             with cuEquivariance attention unchanged:
             Boltz-2   BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND=pallas, glu_backend=pallas
             Protenix  PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=pallas, glu_backend=pallas
             OpenFold3 OPENFOLD3_TRIANGLE_BACKEND=cueq-pallas, glu_backend=pallas
* `-nocueq`  Protenix without cuEquivariance at all: the `-pallas` arm plus
             tokamax triangle attention in the trunk and the confidence head. It
             is the one port whose native options already reach a
             non-cuEq triangle attention, so it is the one prediction that
             measures what dropping the wheel costs.
* `-ctl`     the same snapshot with no option: the released path on this code,
             so a difference against x42 can be split into code drift and arm.

    python make_rows.py SHA            # writes rows/*.sbatch for code-SHA
    python compare.py ROW [ROW ...]    # against x42 (and the -ctl row if any)
"""

# ruff: noqa: E501, N806 -- bench tooling: long table f-strings, A/B array names

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = HERE.parent / "jctc"
sys.path.insert(0, str(J))
import submit  # noqa: E402

SHA = sys.argv[1]
BASE = {
    "boltz2": ["L1000_3og2", "L3000_6ztx"],
    "protenix": ["L1000_3og2", "L3000_6ztx"],
    "openfold3": ["L1000_3og2", "L3000_6ztx"],
}
ENV = {
    "boltz2": "export BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND=pallas",
    "protenix": "export PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=pallas",
    "openfold3": "export OPENFOLD3_TRIANGLE_BACKEND=cueq-pallas",
}
#: The `-nocueq` tripwire. Protenix's remaining triangle-attention call sites
#: (the template stack) read PROTENIX_TRIANGLE_BACKEND. A shadowing
#: `cuequivariance_jax` that refuses to import makes any call that still reaches
#: the wheel fail the row loudly (`_cueq.load_cueq` raises), instead of being
#: timed under a cuEq-free label. It is appended to PYTHONPATH so the snapshot
#: check still reads the code path first; every PYTHONPATH entry precedes
#: site-packages.
NOCUEQ_ENV = r"""export PROTENIX_TRIANGLE_BACKEND=tokamax
mkdir -p "$CACHE/nocueq-shim/cuequivariance_jax"
echo 'raise ImportError("x43 -nocueq row: cuEquivariance is blocked on purpose")' \
  > "$CACHE/nocueq-shim/cuequivariance_jax/__init__.py"
export PYTHONPATH=$PYTHONPATH:$CACHE/nocueq-shim"""
ARMS = {
    "pallas": lambda model: (ENV[model], "--option glu_backend=pallas"),
    "nocueq": lambda model: (
        ENV[model] + "\n" + NOCUEQ_ENV,
        "--option glu_backend=pallas --option trunk_triangle_attention_backend=tokamax "
        "--option confidence_triangle_attention_backend=tokamax",
    ),
    "ctl": lambda model: ("", ""),
}
PLAN = [(m, c, "pallas") for m, cases in BASE.items() for c in cases]
PLAN += [("protenix", "L3000_6ztx", "nocueq"), ("protenix", "L1000_3og2", "nocueq")]
PLAN += [(m, "L3000_6ztx", "ctl") for m in BASE]

matrix = {r["row_id"]: r for r in csv.DictReader(open(J / "matrix.csv"))}
(HERE / "rows").mkdir(exist_ok=True)
for model, case, arm in PLAN:
    base_id = f"{case}-{model}-fj-1c-def-s101"
    row = dict(matrix[base_id])
    row["row_id"] = f"{base_id}-{arm}"
    row["warm_repeats"] = "1"
    env, options = ARMS[arm](model)
    row["options"] = options
    # headroom: a first-compile Triton kernel set and an unmeasured arm
    row["time_limit_min"] = str(int(int(row["time_limit_min"]) * 1.5))
    text = submit.script(row)
    text = (
        text.replace("CODE=$J/code-d57e1bb", f"CODE={HERE}/code-{SHA}")
        .replace("OUT=$J/work/$ROW", f"OUT={HERE}/work/$ROW")
        .replace("RES=$J/results", f"RES={HERE}/results")
        .replace("LOG=$J/logs", f"LOG={HERE}/logs")
        .replace("CACHE=$J/cache/$ROW", f"CACHE={HERE}/cache/$ROW")
    )
    assert "code-d57e1bb" not in text, row["row_id"]
    if env:
        assert text.count("unset XLA_FLAGS\n") == 1, row["row_id"]
        text = text.replace("unset XLA_FLAGS\n", f"unset XLA_FLAGS\n{env}\n")
    (HERE / "rows" / f"{row['row_id']}.sbatch").write_text(text)
    print(row["row_id"])
