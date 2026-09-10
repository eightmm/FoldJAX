"""The table builder reads a work directory, and the directory is the hard part.

Every fixture here is a shape the 2026-09-10 scale directory actually contains:
an upstream result whose `label` field is null while its filename carries the
label, a job that left only a log, a label with a hyphen in it, a case whose
token count is recorded by the upstream run because every FoldJAX run at that
size died, and a monomer whose ipTM is a real 0.0 rather than a missing key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import scale_table


def scores(**values: float) -> list[dict]:
    return [{"seed": 101, "scores": dict(values)}]


def write_result(
    work: Path, case: str, model: str, label: str | None, **fields: object
) -> Path:
    document: dict[str, object] = {
        "case": case,
        "model": model,
        "label": label,
        "impl": "foldjax",
    }
    document.update(fields)
    name = label if label is not None else "upstream"
    path = work / "results" / f"{case}-{model}-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def write_log(work: Path, stem: str, text: str) -> Path:
    path = work / "logs" / f"{stem}.slurm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def row_for(rendered: str, case: str) -> list[str]:
    for line in rendered.splitlines():
        if line.startswith("|") and f"| {case} |" in line:
            return cells(line)
    raise AssertionError(f"no row for {case} in\n{rendered}")


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """Two cases, three models, one of every cell the scale table can print."""
    root = tmp_path / "work"
    write_result(
        root,
        "L1000_x",
        "protenix",
        "scale",
        length=1003,
        wall_s=65.13,
        peak_mib=6863.5,
        samples=scores(iptm=0.0, ptm=0.96),
    )
    write_result(
        root,
        "L1000_x",
        "protenix",
        None,
        impl="upstream",
        length=1003,
        wall_s=96.4,
        peak_mib=12902.4,
        samples=scores(iptm=0.0, ptm=0.95),
    )
    write_log(root, "L1000_x-boltz2-scale", "jax.errors: RESOURCE_EXHAUSTED: 85 GiB\n")
    write_log(root, "L1000_x-openfold3-scale", "job 945 master openfold3\n")
    write_result(
        root,
        "L2000_y",
        "protenix",
        "scale",
        length=2096,
        wall_s=210.4,
        peak_mib=23449.6,
        samples=scores(iptm=0.0),
    )
    return root


def test_a_log_without_a_result_says_which_kind_of_absence(work: Path) -> None:
    rendered = scale_table.render_measurements(
        scale_table.measurement_rows(
            scale_table.load_work(work), label="scale", upstream_label="upstream"
        )
    )
    header = cells(rendered.splitlines()[0])
    assert header[:3] == ["band", "case", "tokens"]
    assert header[3:] == [
        "boltz2 FoldJAX",
        "boltz2 upstream",
        "openfold3 FoldJAX",
        "openfold3 upstream",
        "protenix FoldJAX",
        "protenix upstream",
    ]
    assert row_for(rendered, "L1000_x") == [
        "1k",
        "L1000_x",
        "1003",
        "OOM",
        "-",
        "running",
        "-",
        "65 / 6.7",
        "96 / 12.6",
    ]
    # The second case has no artifact at all for two of the three models, which
    # is a different cell from a job that ran and died.
    assert row_for(rendered, "L2000_y")[3:] == [
        "-",
        "-",
        "-",
        "-",
        "210 / 22.9",
        "-",
    ]


def test_rows_are_ordered_by_length(work: Path) -> None:
    rows = scale_table.measurement_rows(
        scale_table.load_work(work), label="scale", upstream_label="upstream"
    )
    assert [row["case"] for row in rows] == ["L1000_x", "L2000_y"]


def test_the_upstream_label_is_the_filename_not_the_null_field(work: Path) -> None:
    artifacts = scale_table.load_work(work)
    cell = artifacts[("L1000_x", "protenix", "upstream")]
    assert cell["impl"] == "upstream"
    assert cell["wall_s"] == 96.4


def test_the_band_is_the_name_and_the_token_count_is_the_measurement(
    tmp_path: Path,
) -> None:
    """`L5000_8e2f` is 4,888 tokens, and every FoldJAX run at that size OOMed.

    The length therefore has to come from the upstream result, and the band from
    the case name, or the row shows no size at all.
    """
    root = tmp_path / "work"
    write_result(
        root,
        "L5000_z",
        "boltz2",
        None,
        impl="upstream",
        length=4888,
        wall_s=103.23,
        peak_mib=93949.4,
        samples=[],
    )
    write_log(root, "L5000_z-boltz2-scale", "RESOURCE_EXHAUSTED: Out of memory\n")
    rendered = scale_table.render_measurements(
        scale_table.measurement_rows(
            scale_table.load_work(root), label="scale", upstream_label="upstream"
        )
    )
    assert row_for(rendered, "L5000_z") == [
        "5k",
        "L5000_z",
        "4888",
        "OOM",
        "103 / 91.7",
    ]


def test_mixed_bands_come_from_the_named_prefix(tmp_path: Path) -> None:
    root = tmp_path / "work"
    write_result(
        root,
        "mixed_2k_q",
        "boltz2",
        "mixed",
        length=2097,
        wall_s=348.0,
        peak_mib=18841.6,
        samples=scores(iptm=0.865),
    )
    write_result(
        root,
        "L2000_y",
        "boltz2",
        "mixed",
        length=2096,
        wall_s=1.0,
        peak_mib=1024.0,
        samples=scores(iptm=0.5),
    )
    rows = scale_table.measurement_rows(
        scale_table.load_work(root),
        label="mixed",
        upstream_label="upstream",
        case_prefix="mixed_",
    )
    assert [(row["case"], row["band"], row["length"]) for row in rows] == [
        ("mixed_2k_q", "2k", 2097)
    ]
    rendered = scale_table.render_measurements(rows, iptm=True)
    assert row_for(rendered, "mixed_2k_q") == [
        "2k",
        "mixed_2k_q",
        "2097",
        "348 / 18.4",
        "-",
        "0.865",
        "",
    ]


def test_iptm_falls_back_on_a_missing_key_and_not_on_a_zero() -> None:
    # A monomer scores ipTM 0.0. Reading that as "no ipTM" and reporting the
    # model's ptm instead would put a 0.96 in the column.
    assert scale_table.best_iptm(scores(iptm=0.0, ptm=0.96)) == 0.0
    assert scale_table.best_iptm(scores(protein_iptm=0.94, ptm=0.91)) == 0.94
    assert scale_table.best_iptm(scores(ptm=0.91)) == 0.91
    assert scale_table.best_iptm(scores(plddt=88.0)) is None
    assert scale_table.best_iptm([]) is None
    assert scale_table.best_iptm(None) is None
    # Best of the samples, not the first or the last.
    many = scores(iptm=0.5) + scores(iptm=0.9) + scores(iptm=0.7)
    assert scale_table.best_iptm(many) == 0.9


def test_a_run_with_no_iptm_at_all_leaves_the_cell_blank(tmp_path: Path) -> None:
    root = tmp_path / "work"
    write_result(
        root,
        "mixed_1k_a",
        "esmfold2",
        "mixed",
        length=1138,
        wall_s=186.0,
        peak_mib=15052.8,
        samples=scores(plddt=0.89),
    )
    rendered = scale_table.render_measurements(
        scale_table.measurement_rows(
            scale_table.load_work(root), label="mixed", upstream_label="upstream"
        ),
        iptm=True,
    )
    assert row_for(rendered, "mixed_1k_a")[3:] == ["186 / 14.7", "-", "", ""]


def test_alternatives_ratio_is_against_the_baseline_label(tmp_path: Path) -> None:
    root = tmp_path / "work"
    write_result(
        root,
        "L3000_w",
        "protenix",
        "scale",
        length=3012,
        wall_s=600.0,
        peak_mib=40960.0,
        samples=scores(iptm=0.0),
    )
    write_result(
        root,
        "L3000_w",
        "protenix",
        "bf16",
        length=3012,
        wall_s=300.0,
        peak_mib=20480.0,
        samples=scores(iptm=0.0),
    )
    write_result(
        root,
        "L3000_w",
        "protenix",
        "det-notriton",
        length=3012,
        wall_s=660.0,
        peak_mib=40140.8,
        samples=scores(iptm=0.0),
    )
    write_log(root, "L3000_w-protenix-excl", "job 945 master protenix\n")
    # A single-label group is the scale table again, so it is left out.
    write_result(
        root,
        "L3000_w",
        "boltz2",
        "scale",
        length=3012,
        wall_s=806.0,
        peak_mib=40857.6,
        samples=scores(iptm=0.0),
    )
    rows = scale_table.alternative_rows(
        scale_table.load_work(root), label="scale", upstream_label="upstream"
    )
    assert [row["model"] for row in rows] == ["protenix"] * 4
    # The baseline sorts first, the rest alphabetically.
    assert [row["label"] for row in rows] == ["scale", "bf16", "det-notriton", "excl"]
    rendered = scale_table.render_alternatives(rows)
    assert row_for(rendered, "protenix")[2:] == [
        "scale",
        "600 / 40.0",
        "1.00x / 1.00x",
    ]
    body = [cells(line) for line in rendered.splitlines() if line.startswith("| L3000")]
    assert [row[2:] for row in body] == [
        ["scale", "600 / 40.0", "1.00x / 1.00x"],
        ["bf16", "300 / 20.0", "0.50x / 0.50x"],
        ["det-notriton", "660 / 39.2", "1.10x / 0.98x"],
        ["excl", "running", "-"],
    ]


def test_alternatives_without_a_baseline_result_report_no_ratio(
    tmp_path: Path,
) -> None:
    """OpenFold3 at 3,012 tokens has `cueq` and `cueq-full` and no `scale`."""
    root = tmp_path / "work"
    for label, wall in (("cueq", 951.0), ("cueq-full", 863.0)):
        write_result(
            root,
            "L3000_w",
            "openfold3",
            label,
            length=3012,
            wall_s=wall,
            peak_mib=50380.8,
            samples=scores(iptm=0.0),
        )
    rows = scale_table.alternative_rows(
        scale_table.load_work(root), label="scale", upstream_label="upstream"
    )
    assert [row["label"] for row in rows] == ["cueq", "cueq-full"]
    assert all(row["wall_ratio"] is None for row in rows)
    rendered = scale_table.render_alternatives(rows)
    body = [cells(line) for line in rendered.splitlines() if line.startswith("| L3000")]
    assert [row[4] for row in body] == ["-", "-"]


def test_a_label_may_contain_a_hyphen_and_so_may_a_model() -> None:
    models = ("protenix", "protenix-v2", "openfold3")
    assert scale_table.parse_stem("L3000_6ztx-protenix-det-notriton", models) == (
        "L3000_6ztx",
        "protenix",
        "det-notriton",
    )
    # The longer of two names sharing a prefix wins, or every `protenix-v2` run
    # is filed under `protenix` with a label of `v2`.
    assert scale_table.parse_stem("L1000_3og2-protenix-v2-scale", models) == (
        "L1000_3og2",
        "protenix-v2",
        "scale",
    )
    # A model that never appears in a result still parses, on the first two
    # hyphens, which is all that is knowable from the name alone.
    assert scale_table.parse_stem("L1000_3og2-newmodel-cueq-full", ()) == (
        "L1000_3og2",
        "newmodel",
        "cueq-full",
    )
    assert scale_table.parse_stem("nothyphenated", ()) is None


def test_the_oom_marker_is_the_most_specific_one_present() -> None:
    assert scale_table.oom_marker("RESOURCE_EXHAUSTED: Out of memory") == (
        "RESOURCE_EXHAUSTED"
    )
    assert scale_table.oom_marker("torch.OutOfMemoryError") == "OutOfMemoryError"
    assert scale_table.oom_marker("slurmstepd: oom-kill event") == "oom-kill"
    assert scale_table.oom_marker("/bin/sh: line 1: Killed") == "Killed"
    assert scale_table.oom_marker("job 945 master protenix") is None


def test_a_result_wins_over_the_log_of_the_same_run(tmp_path: Path) -> None:
    """A finished job's log may still hold the word `Killed` from another line."""
    root = tmp_path / "work"
    write_result(
        root,
        "L1000_x",
        "protenix",
        "scale",
        length=1003,
        wall_s=65.13,
        peak_mib=6863.5,
        samples=scores(iptm=0.0),
    )
    write_log(root, "L1000_x-protenix-scale", "Killed a background helper\n")
    cell = scale_table.load_work(root)[("L1000_x", "protenix", "scale")]
    assert cell["status"] == "ok"
    assert cell["samples"] == 1


COMPARISON = [
    {
        "model": "openfold3",
        "case": "L1000_3og2",
        "foldjax_samples": 5,
        "upstream_samples": 5,
        "within_foldjax": {
            "tm": {"n": 10, "min": 0.98, "median": 0.99, "max": 0.994},
            "rmsd": {"n": 10, "min": 1.0, "median": 1.34, "max": 2.895},
            "dropped": 0,
            "permuted": 0,
        },
        "within_upstream": {
            "tm": {"n": 10, "min": 0.991, "median": 0.993, "max": 0.996},
            "rmsd": {"n": 10, "min": 0.637, "median": 1.402, "max": 1.589},
            "dropped": 0,
            "permuted": 3,
        },
        "cross": {
            "tm": {"n": 25, "min": 0.988, "median": 0.992, "max": 0.996},
            "rmsd": {"n": 25, "min": 0.656, "median": 1.539, "max": 2.857},
            "dropped": 0,
            "permuted": 7,
        },
    },
    {
        "model": "boltz2",
        "case": "L4000_1gte",
        "foldjax_samples": 5,
        "upstream_samples": 0,
        "within_foldjax": {
            "tm": {"n": 10, "min": 0.5, "median": 0.6, "max": 0.7},
            "rmsd": {"n": 10, "min": 3.0, "median": 4.0, "max": 5.0},
            "dropped": 0,
            "permuted": 0,
        },
        "within_upstream": None,
        "cross": None,
    },
]


def test_spread_passes_the_comparison_through(tmp_path: Path) -> None:
    path = tmp_path / "compare-structures.json"
    path.write_text(json.dumps(COMPARISON))
    rows = scale_table.load_comparison(path)
    assert rows == COMPARISON

    rendered = scale_table.render_spread(rows)
    assert cells(rendered.splitlines()[0]) == [
        "model",
        "case",
        "cross TM",
        "within FoldJAX TM",
        "within upstream TM",
        "cross RMSD A",
        "within FoldJAX RMSD A",
        "within upstream RMSD A",
        "chain perm",
    ]
    # The five columns this shares with `bench.structures --markdown` are
    # rendered the way that command renders them, so the two can be diffed.
    assert row_for(rendered, "L1000_3og2") == [
        "openfold3",
        "L1000_3og2",
        "0.992 (0.988-0.996)",
        "0.990 (0.980-0.994)",
        "0.993 (0.991-0.996)",
        "1.539 (0.656-2.857)",
        "1.340 (1.000-2.895)",
        "1.402 (0.637-1.589)",
        "cross 7/25; up 3/10",
    ]
    # A comparison the upstream side of which never ran keeps its row.
    assert row_for(rendered, "L4000_1gte") == [
        "boltz2",
        "L4000_1gte",
        "-",
        "0.600 (0.500-0.700)",
        "-",
        "-",
        "4.000 (3.000-5.000)",
        "-",
        "-",
    ]


def test_a_missing_comparison_file_is_an_empty_table(tmp_path: Path) -> None:
    assert scale_table.load_comparison(tmp_path / "absent.json") == []
    assert "| model | case |" in scale_table.render_spread([])


def test_the_cli_selects_a_table_and_emits_its_rows(
    work: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert scale_table.main(["--work", str(work), "--table", "scale", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["case"] for row in rows] == ["L1000_x", "L2000_y"]
    boltz2 = rows[0]["cells"]["boltz2"]
    assert boltz2["foldjax"]["status"] == "oom"
    assert boltz2["foldjax"]["marker"] == "RESOURCE_EXHAUSTED"
    assert boltz2["upstream"] is None
    assert rows[0]["cells"]["protenix"]["upstream"]["samples"] == 1

    assert scale_table.main(["--work", str(work), "--table", "scale"]) == 0
    assert "| 1k | L1000_x | 1003 |" in capsys.readouterr().out


def test_the_cli_defaults_a_label_per_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--table all` cannot take one label, so each table brings its own."""
    root = tmp_path / "work"
    write_result(
        root,
        "L1000_x",
        "protenix",
        "scale",
        length=1003,
        wall_s=65.13,
        peak_mib=6863.5,
        samples=scores(iptm=0.0),
    )
    write_result(
        root,
        "mixed_1k_a",
        "protenix",
        "mixed",
        length=1138,
        wall_s=80.0,
        peak_mib=12083.2,
        samples=scores(iptm=0.941),
    )
    assert scale_table.main(["--work", str(root), "--table", "all", "--json"]) == 0
    built = json.loads(capsys.readouterr().out)
    assert [row["case"] for row in built["scale"]] == ["L1000_x"]
    assert [row["case"] for row in built["mixed"]] == ["mixed_1k_a"]
    assert built["spread"] == []
    assert built["alternatives"] == []
