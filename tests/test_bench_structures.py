"""Chain-permutation handling in the structure comparison.

`bench.structures` pairs residues by (chain id, residue id). For a homomeric
complex that key is not stable: two implementations -- and two samples of one
implementation -- may label the identical chains in a different order, and the
comparison then superposes chain A of one prediction onto a different physical
copy in the other. On the 2026-09-10 scale rows that reported OpenFold3's
homotetramer at 37 A / TM 0.569 while each side was internally identical to
TM 1.000.

The fixtures here are deliberately not rigid copies of one shape. If both
chains had the same conformation, a label swap would be undone by a rotation of
the whole assembly and the naive comparison would already score ~0, so the test
would pass against the code it is meant to fail against.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import structures

_HEADER = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_asym_id",
    "pdbx_PDB_model_num",
)

_CYCLE = ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU")


def sequence(length: int, offset: int = 0) -> list[str]:
    """A reproducible residue-name sequence; `offset` makes a distinct entity."""
    return [_CYCLE[(i + offset) % len(_CYCLE)] for i in range(length)]


def helix(length: int, shift: np.ndarray) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    return np.stack(
        [1.5 * t, 4.0 * np.cos(t * 1.7), 4.0 * np.sin(t * 1.7)], axis=-1
    ) + shift


def zigzag(length: int, shift: np.ndarray) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    return np.stack(
        [2.0 * t, 5.0 * (t % 2), 0.35 * t * t / max(length, 1)], axis=-1
    ) + shift


def bend(length: int, shift: np.ndarray) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    return np.stack([t, 0.02 * t * t, 3.0 * np.sin(t * 0.4)], axis=-1) + shift


def write_cif(path, chains: dict[str, tuple[list[str], np.ndarray]]) -> None:
    """One mmCIF whose atom_site loop carries an N and a CA per residue."""
    lines = ["data_test", "#", "loop_"]
    lines += [f"_atom_site.{name}" for name in _HEADER]
    serial = 0
    for chain, (comps, coords) in chains.items():
        for index, (comp, xyz) in enumerate(zip(comps, coords, strict=True), start=1):
            for atom, element, point in (
                ("N", "N", xyz + np.array([-1.2, 0.0, 0.0])),
                ("CA", "C", xyz),
            ):
                serial += 1
                lines.append(
                    " ".join(
                        (
                            "ATOM",
                            str(serial),
                            element,
                            atom,
                            ".",
                            comp,
                            chain,
                            "1",
                            str(index),
                            "?",
                            f"{point[0]:.5f}",
                            f"{point[1]:.5f}",
                            f"{point[2]:.5f}",
                            "1.00",
                            "90.00",
                            "?",
                            str(index),
                            chain,
                            "1",
                        )
                    )
                )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def load(tmp_path, name, chains) -> structures.CAStructure:
    path = tmp_path / f"{name}.cif"
    write_cif(path, chains)
    return structures.ca_coords(path)


def naive_tm_and_rmsd(left, right) -> tuple[float, float]:
    """What the comparison did before it knew about interchangeable chains."""
    a, b = structures.common_residues(left, right)
    assert len(a) >= 4
    return structures.tm_and_rmsd(a, b)


# --- the parser now carries what the grouping needs -------------------------


def test_ca_coords_reports_chains_and_residue_names(tmp_path):
    comps = sequence(6)
    left = load(
        tmp_path,
        "one",
        {"A": (comps, helix(6, np.zeros(3))), "B": (comps, zigzag(6, np.zeros(3)))},
    )
    assert left.coords.shape == (12, 3)
    assert left.keys[:2] == ["A:1", "A:2"]
    assert left.chains == ["A"] * 6 + ["B"] * 6
    assert left.comps[:3] == comps[:3]
    assert structures.chain_sequences(left) == {
        "A": "ACDEFG",
        "B": "ACDEFG",
    }


# --- the case the scale rows hit --------------------------------------------


def test_swapped_homodimer_realigns(tmp_path):
    comps = sequence(24)
    first = helix(24, np.zeros(3))
    second = zigzag(24, np.array([40.0, 5.0, -3.0]))
    left = load(tmp_path, "left", {"A": (comps, first), "B": (comps, second)})
    right = load(tmp_path, "right", {"A": (comps, second), "B": (comps, first)})

    # Pairing by chain id compares a helix against a zigzag.
    naive_tm, naive_rmsd = naive_tm_and_rmsd(left, right)
    assert naive_rmsd > 5.0
    assert naive_tm < 0.7

    block = structures.compare([left], [right])
    assert block["permuted"] == 1
    assert block["rmsd"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert block["tm"]["median"] == pytest.approx(1.0, abs=1e-6)
    assert structures.best_assignment(left, right)[0] == {"A": "B", "B": "A"}


def test_single_chain_numbers_do_not_move(tmp_path):
    comps = sequence(40)
    left = load(tmp_path, "left", {"A": (comps, helix(40, np.zeros(3)))})
    right = load(
        tmp_path,
        "right",
        {"A": (comps, helix(40, np.zeros(3)) + np.array([0.3, -0.2, 0.15]))},
    )

    expected_tm, expected_rmsd = naive_tm_and_rmsd(left, right)
    block = structures.compare([left], [right])
    assert block["permuted"] == 0
    assert block["tm"]["median"] == expected_tm
    assert block["rmsd"]["median"] == expected_rmsd
    assert structures.best_assignment(left, right) == ({"A": "A"}, False)


def test_identical_chain_order_keeps_the_identity_assignment(tmp_path):
    """Chains that already correspond keep their labels and report no move."""
    comps = sequence(20)
    chains = {
        "A": (comps, helix(20, np.zeros(3))),
        "B": (comps, helix(20, np.array([30.0, 0.0, 0.0]))),
    }
    left = load(tmp_path, "left", chains)
    right = load(tmp_path, "right", chains)
    mapping, permuted = structures.best_assignment(left, right)
    assert (mapping, permuted) == ({"A": "A", "B": "B"}, False)


def test_heteromer_swaps_only_the_interchangeable_pair(tmp_path):
    comps = sequence(22)
    other = sequence(18, offset=3)
    first = helix(22, np.zeros(3))
    second = zigzag(22, np.array([45.0, 0.0, 0.0]))
    third = bend(18, np.array([0.0, 45.0, 10.0]))

    left = load(
        tmp_path,
        "left",
        {"A": (comps, first), "B": (comps, second), "C": (other, third)},
    )
    right = load(
        tmp_path,
        "right",
        {"A": (comps, second), "B": (comps, first), "C": (other, third)},
    )

    mapping, permuted = structures.best_assignment(left, right)
    assert permuted is True
    assert mapping == {"A": "B", "B": "A", "C": "C"}

    block = structures.compare([left], [right])
    assert block["permuted"] == 1
    assert block["rmsd"]["median"] == pytest.approx(0.0, abs=1e-6)


def test_within_set_pairs_are_permuted_too(tmp_path):
    """The Boltz-2 4k row: one implementation's own samples come out permuted."""
    comps = sequence(20)
    first = helix(20, np.zeros(3))
    second = zigzag(20, np.array([35.0, 0.0, 0.0]))
    samples = [
        load(tmp_path, "s0", {"A": (comps, first), "B": (comps, second)}),
        load(tmp_path, "s1", {"A": (comps, second), "B": (comps, first)}),
    ]
    block = structures.compare(samples, None)
    assert block["permuted"] == 1
    assert block["rmsd"]["median"] == pytest.approx(0.0, abs=1e-6)


# --- the branch real data never reaches -------------------------------------


def test_hungarian_fallback_agrees_with_the_exhaustive_search(tmp_path, monkeypatch):
    comps = sequence(16)
    shapes = {
        "A": helix(16, np.zeros(3)),
        "B": zigzag(16, np.array([60.0, 0.0, 0.0])),
        "C": bend(16, np.array([0.0, 60.0, 0.0])),
        "D": helix(16, np.array([0.0, 0.0, 60.0])) * np.array([1.0, -1.0, 1.0]),
    }
    left = load(tmp_path, "left", {c: (comps, shapes[c]) for c in "ABCD"})
    right = load(
        tmp_path,
        "right",
        {
            "A": (comps, shapes["A"]),
            "B": (comps, shapes["B"]),
            "C": (comps, shapes["D"]),
            "D": (comps, shapes["C"]),
        },
    )

    exhaustive, _ = structures.best_assignment(left, right)
    assert exhaustive == {"A": "A", "B": "B", "C": "D", "D": "C"}

    monkeypatch.setattr(structures, "MAX_ASSIGNMENTS", 4)
    fallback, permuted = structures.best_assignment(left, right)
    assert permuted is True
    assert fallback == exhaustive


def test_no_shared_sequence_leaves_the_keys_alone(tmp_path):
    left = load(tmp_path, "left", {"A": (sequence(12), helix(12, np.zeros(3)))})
    right = load(
        tmp_path, "right", {"A": (sequence(12, offset=4), bend(12, np.zeros(3)))}
    )
    mapping, permuted = structures.best_assignment(left, right)
    assert mapping == {}
    assert permuted is False
