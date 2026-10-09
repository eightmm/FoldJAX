"""Independent upstream-input feature comparison for the common job path.

Every GPU validation so far checked that an upstream arm read the native input
FoldJAX's own writer produced for the same job. That proves the two runs saw
one document; it cannot see a translation error, because both arms inherit it.

This harness closes that hole on CPU. For each representative job it holds two
documents:

* the **common** job (``docs/input.md``), translated by
  :func:`foldjax.input.materialize_native_input` exactly as ``foldjax predict``
  translates it; and
* a **native** document per backend, written by hand from that upstream's own
  format reference and examples (Boltz-2 ``docs/prediction.md``, Protenix and
  OpenDDE ``docs/infer_json_format.md``, OpenFold3
  ``docs/source/input_format_reference.md``) -- never from the writer. Where the
  reference marks a field optional (Boltz-2 ``version`` and a restraint's
  ``max_distance``, Protenix ``modifications``, ``count`` defaults) the native
  omits it, so the writer's explicit defaults are compared against upstream's.

Both are featurized by the backend's torch-free featurizer -- the same code,
the same seed, the same assets -- and the arrays are compared bitwise. With one
featurizer on both sides, any difference is a difference in what the two
documents mean. The harness also records, per (job, backend), whether the
writer materialized, refused or dropped-with-record the job, and checks that
against what :func:`foldjax.input.common_schema_features` promises.

What it cannot do here: AlphaFold 3's featurizer needs its runtime, which does
not build on this host; ESMFold2 has no second dialect (``_esmfold2`` returns
the common document) and its upstream Biohub ``prepare_input.py`` is torch-only,
so its row is a round-trip of the common document and the featurizer parity is
a torch-host question. OpenFold3's mapped-template form is the port's own
``entries_json`` cache (upstream's is a pickled NPZ the port refuses), so only
the bare-file (CIF-direct) template is independently expressible for it.

Run::

    FOLDJAX_HOME=<scratch> JAX_PLATFORMS=cpu python -m bench.independent_inputs \\
        --store /path/to/a/fetched/.foldjax --out results/independent-inputs

``--store`` is read only: CCD files are reached through file symlinks in the
scratch ``FOLDJAX_HOME`` and every other asset is passed by explicit path.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("boltz2", "protenix", "opendde", "openfold3", "esmfold2")

#: One seed for every featurizer call of both arms. Boltz-2, Protenix, OpenDDE,
#: OpenFold3 and ESMFold2 all draw their reference-conformer augmentation from
#: it, so two semantically equal documents give bitwise equal arrays.
SEED = 7

#: Backends whose writer drops (records) rather than refuses these features at
#: the released default, per ``foldjax.input`` (``_IGNORED_CONSTRAINT_MODELS``,
#: ``_USE_TEMPLATE_MODELS``).
_DROPPED_BY_DEFAULT = {
    "opendde": frozenset({"pocket_constraints", "contact_constraints", "templates"}),
    "protenix": frozenset({"templates"}),
}

# --------------------------------------------------------------------------
# Fixtures: sequences, alignments, a template structure
# --------------------------------------------------------------------------

P1 = "MKTAYIAKQRQISFVKSHFSRQ"  # 22 aa; S13 carries the phospho-modification
P2 = "GSWTEHKSPDGRTYYYNTETKQSTWEKP"  # 28 aa
P3 = "MKNGSTLEYANVTQK"  # 15 aa; N3 carries the N-glycan
DNA = "GATTACAGC"
RNA = "GGCUAGCC"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
GLYCAN = ["NAG", "NAG", "BMA"]
GLYCAN_BONDS = [
    [["A", 3, "ND2"], ["G", 1, "C1"]],
    [["G", 1, "O4"], ["G", 2, "C1"]],
    [["G", 2, "O4"], ["G", 3, "C1"]],
]

# UniProt-style headers: Protenix pairs rows of two chains' paired alignments by
# the species mnemonic (``tr|ACC|ENTRY_SPECIES``); the other readers ignore it.
P1_A3M = """>query
MKTAYIAKQRQISFVKSHFSRQ
>tr|Q1AAA1|Q1AAA1_HUMAN
MKSAYIAKQRQLSFVKSHFSRQ
>tr|Q1AAA2|Q1AAA2_MOUSE
MKTAYLAKQRQIS-VKSHFSRQ
>tr|Q1AAA3|Q1AAA3_YEAST
MKTAYIAKaQRQISFVKTHFSRQ
"""
P2_A3M = """>query
GSWTEHKSPDGRTYYYNTETKQSTWEKP
>tr|Q2BBB1|Q2BBB1_MOUSE
GSWTEHRSPDGRTYYYNTETKQSTWEKP
>tr|Q2BBB2|Q2BBB2_HUMAN
GSWSEHKSPDGRTYFYNTETKQS-WEKP
>tr|Q2BBB3|Q2BBB3_ECOLI
GSWTEHKSPDGRTYYYNtTETKQSTWERP
"""
P3_A3M = """>query
MKNGSTLEYANVTQK
>tr|Q3CCC1|Q3CCC1_HUMAN
MKNGSSLEYANVTQK
>tr|Q3CCC2|Q3CCC2_MOUSE
MRNGSTLEYANITQK
"""
# Paired alignments: three species rows per chain, in different orders, so the
# species-keyed pairing (Protenix/OpenDDE) has to look them up. OpenFold3 reads
# a precomputed paired file as row-aligned (``colabfold_paired``).
P1_PAIRED = """>query
MKTAYIAKQRQISFVKSHFSRQ
>tr|Q1AAA1|Q1AAA1_HUMAN
MKSAYIAKQRQLSFVKSHFSRQ
>tr|Q1AAA2|Q1AAA2_MOUSE
MKTAYLAKQRQIS-VKSHFSRQ
>tr|Q1AAA4|Q1AAA4_ECOLI
MKTAYIAKQRQISFVRSHFSRQ
"""
P2_PAIRED = """>query
GSWTEHKSPDGRTYYYNTETKQSTWEKP
>tr|Q2BBB1|Q2BBB1_MOUSE
GSWTEHRSPDGRTYYYNTETKQSTWEKP
>tr|Q2BBB3|Q2BBB3_ECOLI
GSWTEHKSPDGRTYYYNTETKQSTWERP
>tr|Q2BBB2|Q2BBB2_HUMAN
GSWSEHKSPDGRTYFYNTETKQS-WEKP
"""


def _atom_site_rows(
    rows: list[tuple[str, str, str, int, str, int, float]],
) -> list[str]:
    """``(label_asym, auth_asym, comp, label_seq, atom, auth_seq, x)`` lines."""
    out = []
    for n, (label, auth, comp, seq, atom, auth_seq, x) in enumerate(rows, 1):
        element = atom[0]
        out.append(
            f"ATOM {n} {element} {atom} . {comp} {label} "
            f"{'1' if label == 'A' else '2'} {seq} ? {x:.3f} {0.5 * n:.3f} "
            f"{0.25 * n:.3f} 1.00 20.00 ? {auth_seq} {comp} {auth} {atom} 1"
        )
    return out


def _backbone(
    label: str, auth: str, comp: str, seq: int, x0: float
) -> list[tuple[str, str, str, int, str, int, float]]:
    atoms = [
        (label, auth, comp, seq, "N", seq, x0),
        (label, auth, comp, seq, "CA", seq, x0 + 1.4),
        (label, auth, comp, seq, "C", seq, x0 + 2.9),
        (label, auth, comp, seq, "O", seq, x0 + 4.1),
    ]
    if comp != "GLY":
        atoms.append((label, auth, comp, seq, "CB", seq, x0 + 1.9))
    return atoms


_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}  # fmt: skip

#: A two-chain template. Entity 1 (label A / author A, ``GAAG``) is a decoy;
#: entity 2 (label B / author X) is P1 residues 2-21 with two substitutions
#: (L for I at template 5, N for S at template 12), and template residue 9 (Q)
#: declared in ``_entity_poly_seq`` but unresolved. Long enough for
#: OpenFold3's CIF-direct score (identity x coverage >= 0.1). Label and author
#: ids differ on purpose: the common ``chain_id`` is the author id, Boltz-2 and
#: OpenFold3 address chains by label id.
TEMPLATE_X_ONE = "KTAYLAKQRQINFVKSHFSR"
_TEMPLATE_X_SEQ = [_THREE[letter] for letter in TEMPLATE_X_ONE]
_TEMPLATE_X_UNRESOLVED = 9
_TEMPLATE_ATOMS: list[tuple[str, str, str, int, str, int, float]] = []
for _i, _comp in enumerate(["GLY", "ALA", "ALA", "GLY"], 1):
    _TEMPLATE_ATOMS += _backbone("A", "A", _comp, _i, 10.0 * _i)
for _i, _comp in enumerate(_TEMPLATE_X_SEQ, 1):
    if _i == _TEMPLATE_X_UNRESOLVED:
        continue  # unresolved
    _TEMPLATE_ATOMS += _backbone("B", "X", _comp, _i, 100.0 + 3.8 * _i)

TEMPLATE_CIF = "\n".join(
    [
        "data_1tpl",
        "_entry.id 1tpl",
        "loop_",
        "_entity.id",
        "_entity.type",
        "1 polymer",
        "2 polymer",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_seq_one_letter_code",
        "_entity_poly.pdbx_seq_one_letter_code_can",
        "1 polypeptide(L) A GAAG GAAG",
        f"2 polypeptide(L) X {TEMPLATE_X_ONE} {TEMPLATE_X_ONE}",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
        *(f"1 {i} {c} n" for i, c in enumerate(["GLY", "ALA", "ALA", "GLY"], 1)),
        *(f"2 {i} {c} n" for i, c in enumerate(_TEMPLATE_X_SEQ, 1)),
        "loop_",
        "_chem_comp.id",
        "_chem_comp.type",
        *(
            f"{comp} 'L-peptide linking'"
            for comp in sorted({"GLY", "ALA", *_TEMPLATE_X_SEQ})
        ),
        "loop_",
        "_pdbx_poly_seq_scheme.asym_id",
        "_pdbx_poly_seq_scheme.entity_id",
        "_pdbx_poly_seq_scheme.seq_id",
        "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.pdb_seq_num",
        "_pdbx_poly_seq_scheme.auth_seq_num",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
        "_pdbx_poly_seq_scheme.pdb_ins_code",
        *(
            f"A 1 {i} {c} {i} {i} A ."
            for i, c in enumerate(["GLY", "ALA", "ALA", "GLY"], 1)
        ),
        *(
            f"B 2 {i} {c} {i} {'?' if i == _TEMPLATE_X_UNRESOLVED else i} X ."
            for i, c in enumerate(_TEMPLATE_X_SEQ, 1)
        ),
        "loop_",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "A 1",
        "B 2",
        "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.data_content_type",
        "_pdbx_audit_revision_history.major_revision",
        "_pdbx_audit_revision_history.minor_revision",
        "_pdbx_audit_revision_history.revision_date",
        "1 'Structure model' 1 0 2001-01-01",
        "loop_",
        *(
            f"_atom_site.{key}"
            for key in (
                "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
                "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code "
                "Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv "
                "pdbx_formal_charge auth_seq_id auth_comp_id auth_asym_id "
                "auth_atom_id pdbx_PDB_model_num"
            ).split()
        ),
        *_atom_site_rows(_TEMPLATE_ATOMS),
        "",
    ]
)

#: The same template chain as a Protenix/OpenDDE user hands it in: one chain,
#: observed residues only, which is what their ``parse_simple_cif`` reads
#: (``docs/infer_json_format.md`` names only ``.a3m``/``.hhr``; the JSON form is
#: ``examples/example_with_json_template``, whose entries carry the mmCIF text
#: and 0-based ``queryIndices``/``templateIndices`` over observed residues).
#: Query residue i+1 (0-based i = 1..20) sits on template residue i; the
#: unresolved residue has no observed ordinal, so its pair is absent and every
#: later template index is one lower.
TEMPLATE_MAPPING_COMMON = {
    "query_indices": list(range(1, 21)),
    "template_indices": list(range(0, 20)),
}
TEMPLATE_MAPPING_OBSERVED = {
    "queryIndices": [q for q in range(1, 21) if q != _TEMPLATE_X_UNRESOLVED],
    "templateIndices": list(range(0, 19)),
}
_TEMPLATE_X_OBSERVED: list[tuple[str, str, str, int, str, int, float]] = []
for _ordinal, _seq in enumerate(
    sorted({row[3] for row in _TEMPLATE_ATOMS if row[0] == "B"}), 1
):
    _TEMPLATE_X_OBSERVED += [
        ("X", "X", row[2], _ordinal, row[4], row[5], row[6])
        for row in _TEMPLATE_ATOMS
        if row[0] == "B" and row[3] == _seq
    ]
TEMPLATE_X_ONLY_CIF = "\n".join(
    [
        "data_1tplX",
        "_entry.id 1tplX",
        "loop_",
        *(
            f"_atom_site.{key}"
            for key in (
                "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
                "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code "
                "Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv "
                "pdbx_formal_charge auth_seq_id auth_comp_id auth_asym_id "
                "auth_atom_id pdbx_PDB_model_num"
            ).split()
        ),
        *_atom_site_rows(_TEMPLATE_X_OBSERVED),
        "",
    ]
)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One job, its common document and the hand-written natives."""

    name: str
    features: frozenset[str]
    common: dict[str, Any]
    #: fixture files written into the case directory (relative name -> text)
    files: dict[str, str]
    #: model -> callable(case_dir) -> native document (dict, or YAML text)
    natives: dict[str, Callable[[Path], Any]]
    #: model -> options handed to both the writer and the featurizer
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: model -> (outcome, reason) that overrides the capability-table rule
    expect: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: the missing-alignment policy for the writer
    msa: str = "none"
    #: a case this one must featurize differently from (the vacuity tripwire
    #: for features no array name identifies: a modification, a paired MSA)
    baseline: str | None = None
    note: str = ""


def _p(d: Path, name: str) -> str:
    return str(d / name)


def _common_protein(chain, seq, msa=None, paired=None, **extra):
    body: dict[str, Any] = {"type": "protein", "id": chain, "sequence": seq}
    if msa:
        body["unpaired_msa"] = msa
    if paired:
        body["paired_msa"] = paired
    body.update(extra)
    return body


# -- Boltz-2 natives (docs/prediction.md "Input format") --------------------


def _bz_protein(ids, seq, msa=None, mods=None):
    lines = [
        "  - protein:",
        f"      id: {ids}",
        f"      sequence: {seq}",
        f"      msa: {msa if msa else 'empty'}",
    ]
    if mods:
        lines.append("      modifications:")
        for position, ccd in mods:
            lines += [f"        - position: {position}", f"          ccd: {ccd}"]
    return lines


def _bz_doc(sequences: list[str], tail: list[str] | None = None) -> str:
    return "\n".join(["sequences:", *sequences, *(tail or []), ""])


# -- Protenix / OpenDDE natives (docs/infer_json_format.md) -----------------


def _px_protein(seq, ids, unpaired=None, paired=None, mods=None, templates=None):
    body: dict[str, Any] = {"sequence": seq, "count": len(ids), "id": list(ids)}
    if unpaired:
        body["unpairedMsaPath"] = unpaired
    if paired:
        body["pairedMsaPath"] = paired
    if mods:
        body["modifications"] = [
            {"ptmType": f"CCD_{ccd}", "ptmPosition": position} for position, ccd in mods
        ]
    if templates:
        body["templatesPath"] = templates
    return {"proteinChain": body}


def _px_job(name, sequences, **extra):
    return [{"name": name, "sequences": sequences, **extra}]


# -- OpenFold3 natives (docs/source/input_format_reference.md) --------------


def _of3_chain(kind, ids, **fields):
    return {"molecule_type": kind, "chain_ids": list(ids), **fields}


def _of3_query(name, chains, **extra):
    return {"queries": {name: {"chains": chains, **extra}}}


# -- ESMFold2: the common document is the native one ------------------------


def _esm_native(common: dict[str, Any]) -> Callable[[Path], Any]:
    return lambda d: json.loads(json.dumps(common))


def build_cases() -> list[Case]:
    cases: list[Case] = []

    # 1. protein monomer with an alignment
    common = {
        "name": "monomer_msa",
        "entities": [_common_protein("A", P1, "p1.a3m")],
    }
    cases.append(
        Case(
            "monomer_msa",
            frozenset({"unpaired_msa"}),
            common,
            {"p1.a3m": P1_A3M, "uniref90_hits.a3m": P1_A3M},
            {
                "boltz2": lambda d: _bz_doc(_bz_protein("[A]", P1, _p(d, "p1.a3m"))),
                "protenix": lambda d: _px_job(
                    "monomer_msa", [_px_protein(P1, "A", _p(d, "p1.a3m"))]
                ),
                "opendde": lambda d: _px_job(
                    "monomer_msa",
                    [_px_protein(P1, "A", _p(d, "p1.a3m"))],
                    modelSeeds=[SEED],
                ),
                # A user names the file after its database; the writer links it
                # as ``colabfold_main``. Both stems are far above 4 rows.
                "openfold3": lambda d: _of3_query(
                    "monomer_msa",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                        )
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 2. single-sequence monomer (no alignment at all)
    common = {"name": "monomer_single", "entities": [_common_protein("A", P2)]}
    cases.append(
        Case(
            "monomer_single",
            frozenset(),
            common,
            {},
            {
                "boltz2": lambda d: _bz_doc(_bz_protein("[A]", P2)),
                "protenix": lambda d: _px_job(
                    "monomer_single", [_px_protein(P2, "A")]
                ),
                "opendde": lambda d: _px_job(
                    "monomer_single", [_px_protein(P2, "A")], modelSeeds=[SEED]
                ),
                "openfold3": lambda d: _of3_query(
                    "monomer_single", [_of3_chain("protein", "A", sequence=P2)]
                ),
                "esmfold2": _esm_native(common),
            },
            msa="single",
        )
    )

    # 3. heteromer, unpaired alignments only
    common = {
        "name": "heteromer_msa",
        "entities": [
            _common_protein("A", P1, "p1.a3m"),
            _common_protein("B", P2, "p2.a3m"),
        ],
    }
    cases.append(
        Case(
            "heteromer_msa",
            frozenset({"unpaired_msa"}),
            common,
            {
                "p1.a3m": P1_A3M,
                "p2.a3m": P2_A3M,
                "a/uniref90_hits.a3m": P1_A3M,
                "b/uniref90_hits.a3m": P2_A3M,
            },
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P1, _p(d, "p1.a3m"))
                    + _bz_protein("[B]", P2, _p(d, "p2.a3m"))
                ),
                "protenix": lambda d: _px_job(
                    "heteromer_msa",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m")),
                        _px_protein(P2, "B", _p(d, "p2.a3m")),
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "heteromer_msa",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m")),
                        _px_protein(P2, "B", _p(d, "p2.a3m")),
                    ],
                    modelSeeds=[SEED],
                ),
                "openfold3": lambda d: _of3_query(
                    "heteromer_msa",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "a/uniref90_hits.a3m")],
                        ),
                        _of3_chain(
                            "protein",
                            "B",
                            sequence=P2,
                            main_msa_file_paths=[_p(d, "b/uniref90_hits.a3m")],
                        ),
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 4. heteromer with explicit paired alignments
    common = {
        "name": "heteromer_paired",
        "entities": [
            _common_protein("A", P1, "p1.a3m", "p1_paired.a3m"),
            _common_protein("B", P2, "p2.a3m", "p2_paired.a3m"),
        ],
    }
    cases.append(
        Case(
            "heteromer_paired",
            frozenset({"unpaired_msa", "paired_msa"}),
            common,
            {
                "p1.a3m": P1_A3M,
                "p2.a3m": P2_A3M,
                "p1_paired.a3m": P1_PAIRED,
                "p2_paired.a3m": P2_PAIRED,
                "a/uniref90_hits.a3m": P1_A3M,
                "b/uniref90_hits.a3m": P2_A3M,
                "a/colabfold_paired.a3m": P1_PAIRED,
                "b/colabfold_paired.a3m": P2_PAIRED,
            },
            {
                "protenix": lambda d: _px_job(
                    "heteromer_paired",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m"), _p(d, "p1_paired.a3m")),
                        _px_protein(P2, "B", _p(d, "p2.a3m"), _p(d, "p2_paired.a3m")),
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "heteromer_paired",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m"), _p(d, "p1_paired.a3m")),
                        _px_protein(P2, "B", _p(d, "p2.a3m"), _p(d, "p2_paired.a3m")),
                    ],
                    modelSeeds=[SEED],
                ),
                # ``paired_msa_order`` of the released config reads one paired
                # stem, ``colabfold_paired`` (dataset_config_components.py).
                "openfold3": lambda d: _of3_query(
                    "heteromer_paired",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "a/uniref90_hits.a3m")],
                            paired_msa_file_paths=[_p(d, "a/colabfold_paired.a3m")],
                        ),
                        _of3_chain(
                            "protein",
                            "B",
                            sequence=P2,
                            main_msa_file_paths=[_p(d, "b/uniref90_hits.a3m")],
                            paired_msa_file_paths=[_p(d, "b/colabfold_paired.a3m")],
                        ),
                    ],
                ),
            },
            baseline="heteromer_msa",
        )
    )

    # 5. homodimer with a CCD ligand
    common = {
        "name": "homomer_ligand_ccd",
        "entities": [
            _common_protein(["A", "B"], P1, "p1.a3m"),
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ],
    }
    cases.append(
        Case(
            "homomer_ligand_ccd",
            frozenset({"unpaired_msa", "ligand_ccd"}),
            common,
            {"p1.a3m": P1_A3M, "uniref90_hits.a3m": P1_A3M},
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A, B]", P1, _p(d, "p1.a3m"))
                    + ["  - ligand:", "      id: [L]", "      ccd: ATP"]
                ),
                "protenix": lambda d: _px_job(
                    "homomer_ligand_ccd",
                    [
                        _px_protein(P1, ["A", "B"], _p(d, "p1.a3m")),
                        {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["L"]}},
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "homomer_ligand_ccd",
                    [
                        _px_protein(P1, ["A", "B"], _p(d, "p1.a3m")),
                        {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["L"]}},
                    ],
                    modelSeeds=[SEED],
                ),
                "openfold3": lambda d: _of3_query(
                    "homomer_ligand_ccd",
                    [
                        _of3_chain(
                            "protein",
                            ["A", "B"],
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                        ),
                        _of3_chain("ligand", "L", ccd_codes="ATP"),
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 6. SMILES ligand
    common = {
        "name": "ligand_smiles",
        "entities": [
            _common_protein("A", P2, "p2.a3m"),
            {"type": "ligand", "id": "L", "smiles": ASPIRIN},
        ],
    }
    cases.append(
        Case(
            "ligand_smiles",
            frozenset({"unpaired_msa", "ligand_smiles"}),
            common,
            {"p2.a3m": P2_A3M, "uniref90_hits.a3m": P2_A3M},
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P2, _p(d, "p2.a3m"))
                    + ["  - ligand:", "      id: [L]", f"      smiles: '{ASPIRIN}'"]
                ),
                "protenix": lambda d: _px_job(
                    "ligand_smiles",
                    [
                        _px_protein(P2, "A", _p(d, "p2.a3m")),
                        {"ligand": {"ligand": ASPIRIN, "count": 1, "id": ["L"]}},
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "ligand_smiles",
                    [
                        _px_protein(P2, "A", _p(d, "p2.a3m")),
                        {"ligand": {"ligand": ASPIRIN, "count": 1, "id": ["L"]}},
                    ],
                    modelSeeds=[SEED],
                ),
                "openfold3": lambda d: _of3_query(
                    "ligand_smiles",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P2,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                        ),
                        _of3_chain("ligand", "L", smiles=ASPIRIN),
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 7. modified residue (phosphoserine at 13)
    common = {
        "name": "modified_residue",
        "entities": [
            _common_protein(
                "A", P1, "p1.a3m", modifications=[{"ccd": "SEP", "position": 13}]
            )
        ],
    }
    cases.append(
        Case(
            "modified_residue",
            frozenset({"unpaired_msa", "modifications"}),
            common,
            {"p1.a3m": P1_A3M, "uniref90_hits.a3m": P1_A3M},
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P1, _p(d, "p1.a3m"), mods=[(13, "SEP")])
                ),
                "protenix": lambda d: _px_job(
                    "modified_residue",
                    [_px_protein(P1, "A", _p(d, "p1.a3m"), mods=[(13, "SEP")])],
                ),
                "opendde": lambda d: _px_job(
                    "modified_residue",
                    [_px_protein(P1, "A", _p(d, "p1.a3m"), mods=[(13, "SEP")])],
                    modelSeeds=[SEED],
                ),
                "openfold3": lambda d: _of3_query(
                    "modified_residue",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                            non_canonical_residues={"13": "SEP"},
                        )
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
            baseline="monomer_msa",
        )
    )

    # 8. N-glycan: a three-component ligand chain bonded to ASN 3
    common = {
        "name": "glycan_bonds",
        "entities": [
            _common_protein("A", P3, "p3.a3m"),
            {"type": "ligand", "id": "G", "ccd": GLYCAN},
        ],
        "bonds": GLYCAN_BONDS,
    }
    px_bonds = [
        {
            "entity1": 1 if left[0] == "A" else 2,
            "copy1": 1,
            "position1": left[1],
            "atom1": left[2],
            "entity2": 2,
            "copy2": 1,
            "position2": right[1],
            "atom2": right[2],
        }
        for left, right in GLYCAN_BONDS
    ]
    cases.append(
        Case(
            "glycan_bonds",
            frozenset({"unpaired_msa", "ligand_ccd", "multi_residue_ligand", "bonds"}),
            common,
            {"p3.a3m": P3_A3M},
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P3, _p(d, "p3.a3m"))
                    + ["  - ligand:", "      id: [G]", f"      ccd: {GLYCAN}"],
                    tail=[
                        "constraints:",
                        *(
                            line
                            for left, right in GLYCAN_BONDS
                            for line in (
                                "  - bond:",
                                f"      atom1: {left}",
                                f"      atom2: {right}",
                            )
                        ),
                    ],
                ),
                "protenix": lambda d: _px_job(
                    "glycan_bonds",
                    [
                        _px_protein(P3, "A", _p(d, "p3.a3m")),
                        {
                            "ligand": {
                                "ligand": "CCD_" + "_".join(GLYCAN),
                                "count": 1,
                                "id": ["G"],
                            }
                        },
                    ],
                    covalent_bonds=px_bonds,
                ),
                "opendde": lambda d: _px_job(
                    "glycan_bonds",
                    [
                        _px_protein(P3, "A", _p(d, "p3.a3m")),
                        {
                            "ligand": {
                                "ligand": "CCD_" + "_".join(GLYCAN),
                                "count": 1,
                                "id": ["G"],
                            }
                        },
                    ],
                    covalent_bonds=px_bonds,
                    modelSeeds=[SEED],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 9. protein + DNA + RNA
    common = {
        "name": "nucleic",
        "entities": [
            _common_protein("A", P1, "p1.a3m"),
            {"type": "dna", "id": "D", "sequence": DNA},
            {"type": "rna", "id": "R", "sequence": RNA},
        ],
    }
    cases.append(
        Case(
            "nucleic",
            frozenset({"unpaired_msa"}),
            common,
            {"p1.a3m": P1_A3M, "uniref90_hits.a3m": P1_A3M},
            {
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P1, _p(d, "p1.a3m"))
                    + [
                        "  - dna:",
                        "      id: [D]",
                        f"      sequence: {DNA}",
                        "  - rna:",
                        "      id: [R]",
                        f"      sequence: {RNA}",
                    ]
                ),
                "protenix": lambda d: _px_job(
                    "nucleic",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m")),
                        {"dnaSequence": {"sequence": DNA, "count": 1, "id": ["D"]}},
                        {"rnaSequence": {"sequence": RNA, "count": 1, "id": ["R"]}},
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "nucleic",
                    [
                        _px_protein(P1, "A", _p(d, "p1.a3m")),
                        {"dnaSequence": {"sequence": DNA, "count": 1, "id": ["D"]}},
                        {"rnaSequence": {"sequence": RNA, "count": 1, "id": ["R"]}},
                    ],
                    modelSeeds=[SEED],
                ),
                "openfold3": lambda d: _of3_query(
                    "nucleic",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                        ),
                        _of3_chain("dna", "D", sequence=DNA),
                        _of3_chain("rna", "R", sequence=RNA),
                    ],
                ),
                "esmfold2": _esm_native(common),
            },
        )
    )

    # 10. mapped template on author chain X (label B), one unresolved residue
    mapping = TEMPLATE_MAPPING_COMMON
    common = {
        "name": "template_mapped",
        "entities": [
            _common_protein(
                "A",
                P1,
                "p1.a3m",
                templates=[{"mmcif": "template.cif", "chain_id": "X", **mapping}],
            )
        ],
    }
    use_template = {"use_template": True}
    cases.append(
        Case(
            "template_mapped",
            frozenset({"unpaired_msa", "templates"}),
            common,
            {
                "p1.a3m": P1_A3M,
                "template.cif": TEMPLATE_CIF,
                # The Protenix/OpenDDE form: the observed residues of one chain,
                # indices over them; the unresolved ILE (template residue 3)
                # has no observed ordinal and so no pair.
                "template_x.json": json.dumps(
                    [{"mmcif": TEMPLATE_X_ONLY_CIF, **TEMPLATE_MAPPING_OBSERVED}]
                ),
            },
            {
                "protenix": lambda d: _px_job(
                    "template_mapped",
                    [
                        _px_protein(
                            P1, "A", _p(d, "p1.a3m"), templates=_p(d, "template_x.json")
                        )
                    ],
                ),
                "opendde": lambda d: _px_job(
                    "template_mapped",
                    [
                        _px_protein(
                            P1, "A", _p(d, "p1.a3m"), templates=_p(d, "template_x.json")
                        )
                    ],
                    modelSeeds=[SEED],
                ),
            },
            options={"protenix": use_template, "opendde": use_template},
            expect={
                "openfold3": (
                    "n/a",
                    "the mapped form is the port's own entries_json cache "
                    "(upstream's preprocessor NPZ is pickled and the port "
                    "refuses it), so no independent document exists",
                ),
            },
        )
    )

    # 11. bare template file (the model aligns it): Boltz-2 and OpenFold3
    common = {
        "name": "template_bare",
        "entities": [
            _common_protein(
                "A",
                P1,
                "p1.a3m",
                templates=[{"mmcif": "template.cif", "chain_id": "X"}],
            )
        ],
    }
    cases.append(
        Case(
            "template_bare",
            frozenset({"unpaired_msa", "templates_unmapped"}),
            common,
            {
                "p1.a3m": P1_A3M,
                "uniref90_hits.a3m": P1_A3M,
                "template.cif": TEMPLATE_CIF,
            },
            {
                # ``template_id`` is the structure's chain by label id (B).
                "boltz2": lambda d: _bz_doc(
                    _bz_protein("[A]", P1, _p(d, "p1.a3m")),
                    tail=[
                        "templates:",
                        f"  - cif: {_p(d, 'template.cif')}",
                        "    chain_id: [A]",
                        "    template_id: [B]",
                    ],
                ),
                "openfold3": lambda d: _of3_query(
                    "template_bare",
                    [
                        _of3_chain(
                            "protein",
                            "A",
                            sequence=P1,
                            main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                            template_cif_paths=[_p(d, "template.cif")],
                            template_cif_chain_ids=["B"],
                        )
                    ],
                ),
            },
        )
    )

    # 12/13. pocket restraint, with and without max_distance
    def pocket_case(name: str, max_distance: float | None) -> Case:
        pocket: dict[str, Any] = {"binder": "L", "contacts": [["A", 3], ["A", 7]]}
        if max_distance is not None:
            pocket["max_distance"] = max_distance
        common = {
            "name": name,
            "entities": [
                _common_protein("A", P1, "p1.a3m"),
                {"type": "ligand", "id": "L", "ccd": "ATP"},
            ],
            "constraints": [{"pocket": pocket}],
        }
        bz_pocket = [
            "constraints:",
            "  - pocket:",
            "      binder: L",
            "      contacts: [[A, 3], [A, 7]]",
        ]
        of3_pocket: dict[str, Any] = {
            "ligand_chain_id": "L",
            "pocket_residues": [["A", 3], ["A", 7]],
        }
        px_pocket: dict[str, Any] = {
            "binder_chain": {"entity": 2, "copy": 1},
            "contact_residues": [
                {"entity": 1, "copy": 1, "position": 3},
                {"entity": 1, "copy": 1, "position": 7},
            ],
        }
        if max_distance is not None:
            bz_pocket.append(f"      max_distance: {max_distance}")
            of3_pocket["max_distance"] = max_distance
            px_pocket["max_distance"] = max_distance
        natives: dict[str, Callable[[Path], Any]] = {
            "boltz2": lambda d: _bz_doc(
                _bz_protein("[A]", P1, _p(d, "p1.a3m"))
                + ["  - ligand:", "      id: [L]", "      ccd: ATP"],
                tail=bz_pocket,
            ),
            "openfold3": lambda d: _of3_query(
                name,
                [
                    _of3_chain(
                        "protein",
                        "A",
                        sequence=P1,
                        main_msa_file_paths=[_p(d, "uniref90_hits.a3m")],
                    ),
                    _of3_chain("ligand", "L", ccd_codes="ATP"),
                ],
                pocket_constraint=of3_pocket,
            ),
        }
        expect: dict[str, tuple[str, str]] = {}
        # OpenDDE reads no constraint: its port drops the field as upstream's
        # inference build does, so the native arm carries the Protenix form and
        # both arms must end without it.
        natives["opendde"] = lambda d: _px_job(
            name,
            [
                _px_protein(P1, "A", _p(d, "p1.a3m")),
                {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["L"]}},
            ],
            modelSeeds=[SEED],
            constraint={"pocket": {**px_pocket, "max_distance": max_distance or 6.0}},
        )
        if max_distance is not None:
            natives["protenix"] = lambda d: _px_job(
                name,
                [
                    _px_protein(P1, "A", _p(d, "p1.a3m")),
                    {"ligand": {"ligand": "CCD_ATP", "count": 1, "id": ["L"]}},
                ],
                constraint={"pocket": px_pocket},
            )
        else:
            expect["protenix"] = (
                "refused",
                "upstream Protenix requires pocket max_distance (input.py "
                "_validate_pockets)",
            )
        return Case(
            name,
            frozenset({"unpaired_msa", "ligand_ccd", "pocket_constraints"}),
            common,
            {"p1.a3m": P1_A3M, "uniref90_hits.a3m": P1_A3M},
            natives,
            expect=expect,
        )

    cases.append(pocket_case("pocket_default", None))
    cases.append(pocket_case("pocket_explicit", 7.5))

    # 14/15. contact restraint between two chains, with and without max_distance
    def contact_case(name: str, max_distance: float | None) -> Case:
        contact: dict[str, Any] = {"token1": ["A", 2], "token2": ["B", 3]}
        if max_distance is not None:
            contact["max_distance"] = max_distance
        common = {
            "name": name,
            "entities": [
                _common_protein("A", P1, "p1.a3m"),
                _common_protein("B", P2, "p2.a3m"),
            ],
            "constraints": [{"contact": contact}],
        }
        bz_contact = [
            "constraints:",
            "  - contact:",
            "      token1: [A, 2]",
            "      token2: [B, 3]",
        ]
        px_contact: dict[str, Any] = {
            "entity1": 1,
            "copy1": 1,
            "position1": 2,
            "entity2": 2,
            "copy2": 1,
            "position2": 3,
        }
        if max_distance is not None:
            bz_contact.append(f"      max_distance: {max_distance}")
            px_contact["max_distance"] = max_distance
        natives: dict[str, Callable[[Path], Any]] = {
            "boltz2": lambda d: _bz_doc(
                _bz_protein("[A]", P1, _p(d, "p1.a3m"))
                + _bz_protein("[B]", P2, _p(d, "p2.a3m")),
                tail=bz_contact,
            ),
        }
        expect: dict[str, tuple[str, str]] = {}
        natives["opendde"] = lambda d: _px_job(
            name,
            [
                _px_protein(P1, "A", _p(d, "p1.a3m")),
                _px_protein(P2, "B", _p(d, "p2.a3m")),
            ],
            modelSeeds=[SEED],
            constraint={
                "contact": [{**px_contact, "max_distance": max_distance or 6.0}]
            },
        )
        if max_distance is not None:
            natives["protenix"] = lambda d: _px_job(
                name,
                [
                    _px_protein(P1, "A", _p(d, "p1.a3m")),
                    _px_protein(P2, "B", _p(d, "p2.a3m")),
                ],
                constraint={"contact": [px_contact]},
            )
        else:
            expect["protenix"] = (
                "refused",
                "upstream Protenix requires contact max_distance (input.py "
                "_validate_contacts)",
            )
        return Case(
            name,
            frozenset({"unpaired_msa", "contact_constraints"}),
            common,
            {"p1.a3m": P1_A3M, "p2.a3m": P2_A3M},
            natives,
            expect=expect,
        )

    cases.append(contact_case("contact_default", None))
    cases.append(contact_case("contact_explicit", 8.0))
    return cases


# --------------------------------------------------------------------------
# Assets and featurizers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Assets:
    """Read-only inputs the featurizers need, each by explicit path."""

    components_cif: Path
    ccd_rdkit_cache: Path
    boltz_mols: Path
    esmfold2_ccd: Path

    @classmethod
    def from_store(cls, store: Path) -> Assets:
        assets = cls(
            components_cif=store / "assets" / "components.cif",
            ccd_rdkit_cache=store / "assets" / "components.cif.rdkit_mol.pkl",
            boltz_mols=store / "weights" / "boltz2" / "mols",
            esmfold2_ccd=store / "weights" / "esmfold2" / "ccd.pkl",
        )
        missing = [
            str(path)
            for path in (
                assets.components_cif,
                assets.ccd_rdkit_cache,
                assets.boltz_mols,
                assets.esmfold2_ccd,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"store {store} lacks: {missing}")
        return assets


def prepare_home(assets: Assets, home: Path) -> Path:
    """A scratch ``FOLDJAX_HOME`` whose CCD files are symlinks into the store.

    The writer's CCD checks (``input._require_component``) and the Protenix
    managed-asset fallback resolve through ``foldjax.paths.assets_dir()``; file
    symlinks make them read the real dictionary while every write (lock files,
    caches) lands in the scratch directory.
    """
    directory = home / "assets"
    directory.mkdir(parents=True, exist_ok=True)
    for source in (assets.components_cif, assets.ccd_rdkit_cache):
        link = directory / source.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
    os.environ["FOLDJAX_HOME"] = str(home)
    return home


def featurize_native(
    model: str,
    path: Path,
    *,
    assets: Assets,
    work: Path,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one backend's torch-free featurizer on a native document."""
    work.mkdir(parents=True, exist_ok=True)
    if model == "boltz2":
        from foldjax.models.boltz2.data.featurize import featurize_yaml

        feats, _manifest, _struct_dir = featurize_yaml(
            path, work, assets.boltz_mols, seed=SEED
        )
        return dict(feats)
    if model in ("protenix", "opendde"):
        from foldjax.models.protenix.data.featurize_json import (
            FeaturizerAssets,
            load_first_job,
        )

        feature_assets = FeaturizerAssets(
            components_cif=assets.components_cif,
            ccd_rdkit_cache=assets.ccd_rdkit_cache,
        )
        job = load_first_job(path)
        kwargs = dict(
            base_dir=path.parent,
            seed=SEED,
            use_template=bool(options.get("use_template", False)),
            use_rna_msa=bool(options.get("use_rna_msa", False)),
            assets=feature_assets,
        )
        if model == "protenix":
            from foldjax.models.protenix.data.featurize_json import (
                featurize_protein_json,
            )

            return featurize_protein_json(job, **kwargs)
        from foldjax.models.opendde.data.featurize_json import featurize_opendde_json

        return featurize_opendde_json(job, **kwargs)
    if model == "openfold3":
        from foldjax.models.openfold3.data.featurize import featurize_query

        return dict(featurize_query(path, seed=SEED))
    if model == "esmfold2":
        from foldjax.models.esmfold2.data.all_atom import build_job_features

        document = json.loads(path.read_text(encoding="utf-8"))
        return dict(
            build_job_features(
                document,
                base_dir=path.parent,
                ccd_path=assets.esmfold2_ccd,
                seed=SEED,
            )
        )
    raise ValueError(f"no featurizer for {model}")


def _featurize_in_child(
    model: str, path: Path, assets: Assets, work: Path, options: dict[str, Any]
) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return featurize_native(model, path, assets=assets, work=work, options=options)


def featurize_isolated(
    model: str,
    path: Path,
    *,
    assets: Assets,
    work: Path,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """:func:`featurize_native` in a fresh interpreter.

    The SMILES conformer of Boltz-2 (``parse/schema.py`` ``compute_3d_conformer``)
    and of Protenix/OpenDDE (``featurize_json.py`` ``_external_ccd_molecule``)
    is embedded with RDKit's unseeded, process-global stream, as upstream embeds
    it: two calls in one interpreter draw different conformers, two fresh
    interpreters the same one. Featurizing each arm in its own process is what
    makes a difference between the arms attributable to the documents.
    """
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(1)
    try:
        return pool.apply(
            _featurize_in_child, (model, path, assets, work, dict(options))
        )
    finally:
        pool.close()
        pool.join()


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _leaves(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _leaves(f"{prefix}.{key}" if prefix else str(key), value[key])
    else:
        yield prefix, value


def _as_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], str):
        try:
            return np.asarray(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, (int, float, bool, np.generic)):
        return np.asarray(value)
    return None


def compare_features(
    native: Mapping[str, Any], foldjax: Mapping[str, Any]
) -> dict[str, Any]:
    """Bitwise comparison of two feature dicts, leaf by leaf."""
    left = dict(_leaves("", native))
    right = dict(_leaves("", foldjax))
    report: dict[str, Any] = {
        "only_native": sorted(set(left) - set(right)),
        "only_foldjax": sorted(set(right) - set(left)),
        "mismatches": {},
        "compared": 0,
    }
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        array_a, array_b = _as_array(a), _as_array(b)
        report["compared"] += 1
        if array_a is None or array_b is None:
            if a != b:
                report["mismatches"][key] = {
                    "kind": "value",
                    "native": repr(a)[:200],
                    "foldjax": repr(b)[:200],
                }
            continue
        if array_a.shape != array_b.shape:
            report["mismatches"][key] = {
                "kind": "shape",
                "native": list(array_a.shape),
                "foldjax": list(array_b.shape),
            }
            continue
        if array_a.dtype != array_b.dtype:
            report["mismatches"][key] = {
                "kind": "dtype",
                "native": str(array_a.dtype),
                "foldjax": str(array_b.dtype),
            }
            continue
        if array_a.dtype.kind in "OUS":
            if not np.array_equal(array_a, array_b):
                report["mismatches"][key] = {"kind": "value"}
            continue
        if array_a.dtype.kind in "fc":
            differ = ~(
                (array_a == array_b) | (np.isnan(array_a) & np.isnan(array_b))
            )
        else:
            differ = array_a != array_b
        count = int(np.count_nonzero(differ))
        if count:
            entry: dict[str, Any] = {
                "kind": "values",
                "count": count,
                "size": int(differ.size),
            }
            if array_a.dtype.kind in "fciub":
                diff = np.abs(
                    array_a.astype(np.float64) - array_b.astype(np.float64)
                )
                entry["max_abs"] = float(np.nanmax(np.where(differ, diff, 0.0)))
            report["mismatches"][key] = entry
    return report


#: Per feature, the array name patterns that must be non-trivial in BOTH arms
#: for a match to mean anything (the vacuity tripwire): two documents that each
#: lost the feature compare equal for the wrong reason.
_PRESENCE: dict[str, dict[str, tuple[str, ...]]] = {
    "unpaired_msa": {model: ("msa",) for model in MODELS},
    "paired_msa": {model: ("msa",) for model in MODELS},
    "templates": {
        "protenix": ("template_backbone_frame_mask",),
        "opendde": ("template_backbone_frame_mask",),
        "openfold3": ("template_backbone_frame_mask",),
        "boltz2": ("template_mask",),
    },
    "templates_unmapped": {
        "boltz2": ("template_mask",),
        "openfold3": ("template_backbone_frame_mask",),
    },
    "pocket_constraints": {
        "boltz2": ("contact_conditioning",),
        "protenix": ("constraint_feature*",),
        "openfold3": ("pocket_*",),
    },
    "contact_constraints": {
        "boltz2": ("contact_conditioning",),
        "protenix": ("constraint_feature*",),
    },
}

#: Arrays that are non-zero without the feature (Boltz-2's
#: ``contact_conditioning`` is a class matrix initialised to ``UNSELECTED``);
#: present means more than one class.
_CLASS_MATRICES = frozenset({"contact_conditioning"})


def presence(
    model: str, features: frozenset[str], arrays: Mapping[str, Any]
) -> dict[str, bool]:
    """Whether each tripwired feature left a visible trace in ``arrays``."""
    leaves = dict(_leaves("", arrays))
    out: dict[str, bool] = {}
    for feature in sorted(features):
        patterns = _PRESENCE.get(feature, {}).get(model)
        if patterns is None:
            continue
        found = False
        for key, value in leaves.items():
            if not any(fnmatch.fnmatch(key, pattern) for pattern in patterns):
                continue
            array = _as_array(value)
            if array is None:
                continue
            if feature in ("unpaired_msa", "paired_msa"):
                # A real alignment has more than the query row; the leading
                # batch axis (when present) is not the row axis.
                rows = array.shape[1] if array.ndim >= 3 else array.shape[0]
                found = found or rows > 1
            elif key.rsplit(".", 1)[-1] in _CLASS_MATRICES:
                found = found or len(np.unique(array)) > 1
            elif array.dtype.kind in "biufc":
                found = found or bool(np.any(array != 0))
            else:
                found = True
        out[feature] = found
    return out


# --------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------


def expected_outcome(case: Case, model: str) -> tuple[str, str]:
    """What the capability table says the writer does with this job."""
    if model in case.expect:
        return case.expect[model]
    from foldjax.input import common_schema_features

    supported = set(common_schema_features(model))
    missing = case.features - supported
    options = case.options.get(model, {})
    dropped = _DROPPED_BY_DEFAULT.get(model, frozenset())
    if options.get("use_template"):
        dropped = dropped - {"templates"}
    if missing and missing <= dropped:
        return "dropped", f"{sorted(missing)} dropped with a manifest record"
    if missing:
        return "refused", f"not in common_schema_features({model}): {sorted(missing)}"
    return "materialized", "every feature is in the capability table"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _write_case(case: Case, root: Path) -> Path:
    directory = root / case.name
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in case.files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (directory / "common.json").write_text(
        json.dumps(case.common, indent=2), encoding="utf-8"
    )
    return directory


def _write_native(model: str, document: Any, directory: Path) -> Path:
    suffix = ".yaml" if model == "boltz2" else ".json"
    path = directory / f"native_{model}{suffix}"
    text = document if isinstance(document, str) else json.dumps(document, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


def _materialize(
    case: Case, model: str, directory: Path
) -> tuple[str, Path | None, dict[str, Any]]:
    """Run the writer; return (outcome, native path, records)."""
    from foldjax.input import materialize_native_input
    from foldjax.registry import capabilities

    records: dict[str, list[dict[str, Any]]] = {
        "ignored": [],
        "ignored_templates": [],
        "ignored_constraints": [],
        "constraints": [],
    }
    out = directory / f"foldjax_{model}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = materialize_native_input(
                directory / "common.json",
                capabilities(model),
                out,
                seed=SEED,
                msa=case.msa,
                options=case.options.get(model) or None,
                ignored=records["ignored"],
                ignored_templates=records["ignored_templates"],
                ignored_constraints=records["ignored_constraints"],
                constraints=records["constraints"],
            )
    except ValueError as error:
        return "refused", None, {"error": str(error), **records}
    dropped = any(
        records[key] for key in ("ignored", "ignored_templates", "ignored_constraints")
    )
    return ("dropped" if dropped else "materialized"), path, records


def run_case(
    case: Case,
    model: str,
    *,
    assets: Assets,
    root: Path,
    baselines: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One (job, backend) cell.

    ``baselines`` collects the translated arm's arrays by ``(case, model)`` so
    a later case can assert it featurized differently from an earlier one.
    """
    directory = _write_case(case, root)
    expected, why = expected_outcome(case, model)
    result: dict[str, Any] = {
        "case": case.name,
        "model": model,
        "expected": expected,
        "expected_reason": why,
    }
    if expected == "n/a":
        result["status"] = "n/a"
        return result
    started = time.monotonic()
    outcome, path, records = _materialize(case, model, directory)
    result["writer"] = outcome
    result["records"] = records
    if outcome != expected:
        result["status"] = f"unexpected-{outcome}"
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
    if outcome == "refused":
        result["status"] = "refused"
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
    native_builder = case.natives.get(model)
    if native_builder is None:
        result["status"] = "no-native"
        return result
    assert path is not None
    native_path = _write_native(model, native_builder(directory), directory)
    options = case.options.get(model, {})
    try:
        native = featurize_isolated(
            model,
            native_path,
            assets=assets,
            work=directory / f"work_native_{model}",
            options=options,
        )
        translated = featurize_isolated(
            model,
            path,
            assets=assets,
            work=directory / f"work_foldjax_{model}",
            options=options,
        )
        # The control: the native document again, in another fresh process.
        # Anything that differs here is the featurizer's own floor and is
        # reported as such rather than charged to the writer.
        control = featurize_isolated(
            model,
            native_path,
            assets=assets,
            work=directory / f"work_control_{model}",
            options=options,
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()[-4000:]
        result["seconds"] = round(time.monotonic() - started, 2)
        return result
    comparison = compare_features(native, translated)
    floor = compare_features(native, control)
    unstable = sorted(
        set(floor["mismatches"])
        | set(floor["only_native"])
        | set(floor["only_foldjax"])
    )
    result["comparison"] = comparison
    result["control"] = {"unstable": unstable}
    if unstable:
        # A key the featurizer cannot reproduce on its own input says nothing
        # about the writer; keep it out of the verdict but on the record.
        comparison["mismatches"] = {
            key: value
            for key, value in comparison["mismatches"].items()
            if key not in unstable
        }
        comparison["unstable"] = unstable
    seen = presence(model, case.features, translated)
    seen_native = presence(model, case.features, native)
    result["presence"] = {"foldjax": seen, "native": seen_native}
    vacuous = [name for name, ok in seen.items() if not ok] + [
        name for name, ok in seen_native.items() if not ok
    ]
    if baselines is not None:
        baselines[(case.name, model)] = translated
        reference = baselines.get((case.baseline, model)) if case.baseline else None
        if reference is not None:
            against = compare_features(reference, translated)
            result["baseline"] = {
                "case": case.baseline,
                "changed": sorted(
                    set(against["mismatches"])
                    | set(against["only_native"])
                    | set(against["only_foldjax"])
                ),
            }
            if not result["baseline"]["changed"]:
                vacuous.append(f"no change against {case.baseline}")
    if any(
        comparison[key] for key in ("mismatches", "only_native", "only_foldjax")
    ):
        result["status"] = "mismatch"
    elif vacuous:
        result["status"] = "vacuous"
        result["vacuous"] = sorted(set(vacuous))
    elif unstable:
        result["status"] = "match*"
    else:
        result["status"] = "match"
    result["seconds"] = round(time.monotonic() - started, 2)
    return result


def run(
    cases: list[Case],
    models: tuple[str, ...],
    *,
    assets: Assets,
    root: Path,
    log: Callable[[str], None] = lambda text: None,
) -> list[dict[str, Any]]:
    results = []
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        for model in models:
            result = run_case(
                case, model, assets=assets, root=root, baselines=baselines
            )
            results.append(result)
            detail = ""
            if result["status"] == "mismatch":
                keys = sorted(result["comparison"]["mismatches"])
                detail = " " + ", ".join(keys)[:200]
            elif result["status"] == "error":
                detail = " " + result["error"][:200]
            log(f"{case.name:20s} {model:10s} {result['status']}{detail}")
    return results


def summary_table(results: list[dict[str, Any]]) -> str:
    """A job x model Markdown table of outcomes."""
    cases = list(dict.fromkeys(result["case"] for result in results))
    models = list(dict.fromkeys(result["model"] for result in results))
    by = {(result["case"], result["model"]): result for result in results}
    lines = ["| job | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for case in cases:
        cells = []
        for model in models:
            result = by.get((case, model))
            if result is None:
                cells.append("")
                continue
            status = result["status"]
            if status == "mismatch":
                keys = sorted(result["comparison"]["mismatches"])
                extra = result["comparison"]["only_native"] + result["comparison"][
                    "only_foldjax"
                ]
                status = f"MISMATCH ({', '.join((keys + extra)[:4])})"
            elif status.startswith("unexpected"):
                status = status.upper()
            elif status == "vacuous":
                status = f"vacuous ({', '.join(result['vacuous'])})"
            cells.append(status)
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="a fetched FoldJAX store: CCD, Boltz-2 mols, ESMFold2 ccd.pkl (read only)",
    )
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument(
        "--models", nargs="*", default=list(MODELS), choices=MODELS, metavar="MODEL"
    )
    parser.add_argument("--cases", nargs="*", default=None, metavar="CASE")
    parser.add_argument(
        "--keep", action="store_true", help="keep the generated documents and work dirs"
    )
    args = parser.parse_args(argv)

    home = Path(os.environ.get("FOLDJAX_HOME") or (args.out / "home")).resolve()
    assets = Assets.from_store(args.store.resolve())
    prepare_home(assets, home)

    cases = build_cases()
    if args.cases:
        unknown = set(args.cases) - {case.name for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case.name in args.cases]
    root = args.out / "cases"
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    results = run(
        cases,
        tuple(args.models),
        assets=assets,
        root=root,
        log=lambda text: print(text, file=sys.stderr, flush=True),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    table = summary_table(results)
    (args.out / "TABLE.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    bad = [r for r in results if r["status"] not in ("match", "refused", "n/a")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
