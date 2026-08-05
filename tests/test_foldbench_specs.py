

def test_foldjax_document_is_polymer_only_by_default() -> None:
    """A cross-model document must not carry ligands unless asked.

    The ports disagree on how ligands are expressed and how they are tokenized, so
    including them gives each model a different token count for the same entry --
    which is exactly what a scaling comparison must not have.
    """
    from tests._foldbench.rcsb import Polymer, Target
    from tests._foldbench.specs import foldjax_document

    target = Target(
        pdb_id="TEST",
        polymers=(Polymer("protein", "GSHM", ("A", "B")),),
        ligands=("HEM", "GOL"),
    )
    document = foldjax_document(target)
    assert document["name"] == "TEST"
    assert [entity["type"] for entity in document["entities"]] == ["protein"]
    assert document["entities"][0]["id"] == ["A", "B"]
    assert document["entities"][0]["sequence"] == "GSHM"


def test_foldjax_document_adds_ligands_on_request_without_additives() -> None:
    from tests._foldbench.rcsb import Polymer, Target
    from tests._foldbench.specs import foldjax_document

    target = Target(
        pdb_id="TEST",
        polymers=(Polymer("protein", "GSHM", ("A",)),),
        ligands=("HEM", "GOL"),
    )
    document = foldjax_document(target, include_ligands=True)
    ligands = [e for e in document["entities"] if e["type"] == "ligand"]
    # GOL is a cryoprotectant, so drop_additives (on by default here) removes it.
    assert [entity["ccd"] for entity in ligands] == ["HEM"]
    # Ligand chain ids must not collide with the polymers'.
    assert ligands[0]["id"] != ["A"]


def test_foldjax_document_ligand_ids_do_not_collide() -> None:
    """Synthetic ligand ids have to avoid every deposited chain id."""
    from tests._foldbench.rcsb import Polymer, Target
    from tests._foldbench.specs import foldjax_document

    target = Target(
        pdb_id="TEST",
        polymers=(Polymer("protein", "GS", ("AA", "AB")),),
        ligands=("HEM", "NAG"),
    )
    document = foldjax_document(target, include_ligands=True)
    polymer_ids = {"AA", "AB"}
    ligand_ids = [
        entity["id"][0] for entity in document["entities"] if entity["type"] == "ligand"
    ]
    assert not polymer_ids & set(ligand_ids)
    assert len(set(ligand_ids)) == len(ligand_ids)
