import json

from bench.openbind_acceptance import classify, evaluate


def _case(snapshot, native, case, entities, *, calls, bitwise=True, backend="cueq"):
    (native / case / "native-cueq").mkdir(parents=True)
    (native / case / "native-cueq" / "kernel-calls.json").write_text(
        json.dumps({"calls": calls})
    )
    (snapshot / f"{case}-{backend}-vs-native-cueq-comparison.json").write_text(
        json.dumps({
            "coordinates": {"entity_max_rmsd": entities},
            "public_confidence": {"plddt": {"max_absolute_error": 1.0}},
        })
    )
    if bitwise is not None:
        (snapshot / f"{case}-{backend}-cross-process.json").write_text(
            json.dumps({"fields": {"coordinates": {"bitwise_equal": bitwise}}})
        )


CUEQ = {"attention.cueq": 500, "attention.cueq_fallback_false": 500, "trimul.cueq": 428}
SMALL = {"attention.cueq_fallback_true": 500, "trimul.cueq": 428}


def test_bands():
    assert classify(0.049) == "pass"
    assert classify(0.05) == "deferred"
    assert classify(0.1) == "investigate"


def test_pass_and_deferred_accept_but_investigate_fails(tmp_path):
    snap, nat = tmp_path / "snap", tmp_path / "nat"
    snap.mkdir()
    _case(snap, nat, "a", {"A": 0.04}, calls=CUEQ)
    _case(snap, nat, "b", {"A": 0.07, "L": 0.02}, calls=CUEQ)
    result = evaluate(snap, nat, ["a", "b"])
    assert result["accepted"]
    assert [r["status"] for r in result["rows"]] == ["pass", "deferred"]
    _case(snap, nat, "c", {"A": 0.2}, calls=CUEQ)
    assert not evaluate(snap, nat, ["a", "b", "c"])["accepted"]


def test_small_token_cases_are_excluded_not_counted(tmp_path):
    snap, nat = tmp_path / "snap", tmp_path / "nat"
    snap.mkdir()
    _case(snap, nat, "tiny", {"A": 0.5}, calls=SMALL)
    result = evaluate(snap, nat, ["tiny"])
    assert result["rows"][0]["status"] == "excluded"
    assert not result["accepted"]  # nothing counted, nothing accepted
    _case(snap, nat, "big", {"A": 0.01}, calls=CUEQ)
    assert evaluate(snap, nat, ["tiny", "big"])["accepted"]


def test_cross_process_drift_or_missing_repeat_fails(tmp_path):
    snap, nat = tmp_path / "snap", tmp_path / "nat"
    snap.mkdir()
    _case(snap, nat, "drift", {"A": 0.01}, calls=CUEQ, bitwise=False)
    _case(snap, nat, "norepeat", {"A": 0.01}, calls=CUEQ, bitwise=None)
    result = evaluate(snap, nat, ["drift", "norepeat"])
    assert [r["status"] for r in result["rows"]] == ["fail", "fail"]
    assert not result["accepted"]


def test_missing_reports_are_reported_and_block(tmp_path):
    snap, nat = tmp_path / "snap", tmp_path / "nat"
    snap.mkdir()
    nat.mkdir()
    result = evaluate(snap, nat, ["x"])
    assert result["rows"][0]["status"] == "missing" and not result["accepted"]
