"""Translate a neutral :class:`~tests._foldbench.rcsb.Target` into each port's input.

The four models agree on what a target *is* -- sequences, molecule types, chain
identity, ligand components -- and disagree on essentially everything downstream of
that. So this module stops at input specifications: it writes the file each port's
own pipeline reads, and lets that pipeline do the featurization. Trying to share
features instead would be building on the one thing the models do not share.
"""

from __future__ import annotations

import string
from collections.abc import Iterator

from tests._foldbench.rcsb import Target

# Chain identifiers for ligand entries, which have none in the deposition: the
# nonpolymer entities are named by component, not by chain.
_LIGAND_IDS = tuple(
    f"{first}{second}"
    for first in string.ascii_uppercase
    for second in string.ascii_uppercase
)


def _fresh_ids(taken: set[str]) -> Iterator[str]:
    for candidate in _LIGAND_IDS:
        if candidate not in taken:
            taken.add(candidate)
            yield candidate


def foldjax_document(
    target: Target,
    *,
    include_ligands: bool = False,
    drop_additives: bool = True,
    name: str | None = None,
) -> dict:
    """Build FoldJAX's model-neutral input document for one target.

    FoldJAX translates this into each backend's native input, which is what makes a
    cross-model comparison possible from a single real deposition rather than four
    hand-written files that might not describe the same structure.

    ``include_ligands`` defaults to *off*, which is the opposite of
    :func:`openfold3_query`, and the asymmetry is deliberate. The ports do not
    express ligands the same way -- Chai takes SMILES only, the others take CCD
    codes -- and they do not tokenize them the same way either, so a
    ligand-bearing target has a different token count in each model. That is fine
    when measuring one model against its own torch upstream, and it destroys a
    comparison *between* models. Polymers are expressed identically everywhere, so
    a polymer-only target has one token count that holds for all of them.

    Args:
        target: the parsed structure.
        include_ligands: add ligands as CCD entities. Backends that take SMILES
            only will reject the resulting document, which is the intended
            behaviour: FoldJAX refuses rather than silently dropping them.
        drop_additives: keep only biologically meaningful ligands; only consulted
            when ``include_ligands`` is set.
        name: job name; defaults to the PDB id.
    """
    entities: list[dict] = []
    for polymer in target.polymers:
        entities.append(
            {
                "type": polymer.molecule_type,
                "id": list(polymer.chain_ids),
                "sequence": polymer.sequence,
            }
        )
    if include_ligands:
        taken = {
            chain_id for polymer in target.polymers for chain_id in polymer.chain_ids
        }
        ids = _fresh_ids(taken)
        codes = target.biological_ligands if drop_additives else target.ligands
        for code in codes:
            entities.append({"type": "ligand", "id": [next(ids)], "ccd": code})
    return {"name": name or target.pdb_id, "entities": entities}


def openfold3_query(
    target: Target,
    *,
    include_ligands: bool = True,
    drop_additives: bool = False,
    query_id: str | None = None,
) -> dict:
    """Build OpenFold3's inference query JSON for one target.

    Ligands become their own chains with ``molecule_type: "ligand"`` and a CCD
    code, which is how OpenFold3 addresses them; each gets a synthetic chain id
    because nonpolymer entities are identified by component rather than by chain.

    Args:
        target: the parsed structure.
        include_ligands: drop the ligands, which is useful for isolating whether a
            failure comes from the polymer path or the ligand path.
        drop_additives: keep only the biologically meaningful ligands. Use this for
            cost measurements, where buffer components would otherwise make the
            token count depend on the crystallization condition.
        query_id: defaults to the PDB id.
    """
    chains: list[dict] = []
    taken = {chain_id for polymer in target.polymers for chain_id in polymer.chain_ids}

    for polymer in target.polymers:
        chains.append(
            {
                "molecule_type": polymer.molecule_type,
                "chain_ids": list(polymer.chain_ids),
                "sequence": polymer.sequence,
            }
        )

    if include_ligands:
        ids = _fresh_ids(taken)
        codes = target.biological_ligands if drop_additives else target.ligands
        for code in codes:
            chains.append(
                {
                    "molecule_type": "ligand",
                    "chain_ids": [next(ids)],
                    "ccd_codes": [code],
                }
            )

    return {"queries": {query_id or target.pdb_id: {"chains": chains}}}
