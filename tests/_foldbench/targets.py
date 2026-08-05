"""Real PDB entries to benchmark and compare against.

Every entry here is a real deposition, fetched from RCSB. Synthetic inputs are
excluded on purpose: a hand-built batch tends to be uniform -- one chain, one
molecule type, sequential residue numbering, no gaps -- and uniformity is exactly
what hides tokenization, masking and pairing bugs. It also cannot show how cost
scales, because the shapes are whatever the author picked.

The set spans two axes deliberately:

* **length**, 24 to ~2030 tokens, so time and memory can be fitted against size
  rather than measured at one point. The upper end matters more than it looks:
  pair representations and their attention are quadratic in token count, so a
  port that is comfortable at 600 tokens can be unrunnable at 2000, and nothing
  below ~1500 reveals that.
* **composition**, so protein-only code paths cannot pass for complete: nucleic
  acids, organic ligands, metal ions, homomers with repeated chains, and a
  protein-DNA complex.

Entries were selected from RCSB's own ``rcsb_assembly_info.polymer_monomer_count``
rather than by recalling which structures are "big", so the sizes are the
deposition's and not an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpec:
    """One benchmark entry and why it is in the set."""

    pdb_id: str
    approx_tokens: int
    rationale: str


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("1BNA", 24, "DNA duplex; two identical strands, no protein at all"),
    TargetSpec("1UBQ", 76, "protein monomer; the smallest useful protein case"),
    TargetSpec("1EHZ", 76, "tRNA; RNA plus Mg/Mn, so metal ions are exercised"),
    TargetSpec("1CBS", 137, "protein with one organic ligand (retinoic acid)"),
    TargetSpec("3HTB", 164, "protein with three distinct ligands in one entry"),
    TargetSpec(
        "4HHB",
        574,
        "haemoglobin; two entities at two copies each, plus haem and phosphate -- "
        "repeated chains are what break symmetry and pairing handling",
    ),
    TargetSpec("7KDT", 597, "protein heterodimer with no ligands"),
    TargetSpec(
        "1TUP",
        699,
        "p53 core bound to DNA; three protein copies, two DNA strands and zinc, "
        "so protein and nucleic acid are tokenized in the same target",
    ),
    TargetSpec(
        "6M0J",
        832,
        "SARS-CoV-2 RBD with ACE2, with glycan and metal",
    ),
    TargetSpec(
        "3QEX",
        934,
        "RB69 DNA polymerase ternary complex; one 903-residue chain with primer "
        "and template DNA strands of different lengths plus an incoming dGTP, so "
        "a single target covers protein, both DNA strands and a nucleotide ligand",
    ),
    TargetSpec(
        "5WLH",
        1494,
        "LbaCas13a bound to pre-crRNA; a 1440-residue protein with a 54-nucleotide "
        "RNA, which is the only way to reach this size with RNA still present",
    ),
    TargetSpec(
        "8RB1",
        1900,
        "DNA-bound human MutS-beta; a genuine heterodimer (934 + 918) with a "
        "24-mer DNA duplex and ADP/Mg -- four chains, three molecule types",
    ),
    TargetSpec(
        "3U7Q",
        2030,
        "nitrogenase MoFe protein at 1.0 A; two entities at two copies each and "
        "the FeMo cofactor, so the largest case is also a repeated-chain case "
        "rather than one long chain, which is the harder tokenization",
    ),
)

#: Components no port can currently take as a CCD code, with the reason. These are
#: real parts of their depositions; the limitation is in the CCD-to-molecule step,
#: which sanitizes with rdkit and rejects entries whose deposited bond orders imply
#: an impossible valence. Recorded here so a target stays reproducible instead of
#: needing the offending code removed by hand each time.
UNSUPPORTED_COMPONENTS = {
    "ICS": "FeMo cofactor with homocitrate; its CCD bonds give one carbon valence 5, "
    "which rdkit's sanitizer rejects",
}

#: Components that are crystallization additives rather than part of the
#: biological assembly: cryoprotectants, buffers and precipitants. They are real
#: parts of the deposition, but asking a folding model to place ethylene glycol is
#: not a meaningful prediction task, and including them makes a target's token
#: count depend on how the crystal was grown.
CRYSTALLIZATION_ADDITIVES = frozenset(
    {
        "GOL",  # glycerol
        "EDO",  # ethylene glycol
        "PEG",
        "PGE",
        "MPD",
        "MRD",
        "TRS",  # tris buffer
        "MES",
        "IMD",  # imidazole
        "ACT",  # acetate
        "FMT",  # formate
        "SO4",
        "PO4",
        "CL",
        "NA",
        "K",
        "BCT",
        "IOD",
        "NO3",
        "IPA",
        "EOH",
        "DMS",
    }
)

BY_ID = {target.pdb_id: target for target in TARGETS}


def by_size(limit: int | None = None) -> tuple[TargetSpec, ...]:
    """Targets in ascending size, optionally capped at ``limit`` tokens."""
    ordered = tuple(sorted(TARGETS, key=lambda t: t.approx_tokens))
    if limit is None:
        return ordered
    return tuple(t for t in ordered if t.approx_tokens <= limit)
