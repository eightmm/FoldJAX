from dataclasses import replace

import numpy as np

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.crop.cropper import Cropper
from foldjax.models.boltz2.data.types import Tokenized


class AffinityCropper(Cropper):
    """Crop a predicted complex to the ligand and its nearest protein tokens."""

    def __init__(
        self,
        neighborhood_size: int = 10,
        max_tokens_protein: int = 200,
    ) -> None:
        self.neighborhood_size = neighborhood_size
        self.max_tokens_protein = max_tokens_protein

    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: int | None = None,
    ) -> Tokenized:
        token_data = data.tokens
        token_bonds = data.bonds
        valid_tokens = token_data[token_data["resolved_mask"]]
        if not valid_tokens.size:
            raise ValueError("No valid tokens in structure")

        ligand_coords = valid_tokens[valid_tokens["affinity_mask"]]["center_coords"]
        if not ligand_coords.size:
            raise ValueError("No resolved affinity ligand tokens in structure")
        dists = np.min(
            np.sqrt(
                np.sum(
                    (
                        valid_tokens["center_coords"][:, None]
                        - ligand_coords[None]
                    )
                    ** 2,
                    axis=-1,
                )
            ),
            axis=1,
        )
        indices = np.argsort(dists)

        cropped: set[int] = set()
        cropped_protein: set[int] = set()
        total_atoms = 0
        ligand_ids = set(
            valid_tokens[
                valid_tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]["token_idx"]
        )

        for idx in indices:
            token = valid_tokens[idx]
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]
            if len(chain_tokens) <= self.neighborhood_size:
                new_tokens = chain_tokens
            else:
                min_idx = token["res_idx"] - self.neighborhood_size
                max_idx = token["res_idx"] + self.neighborhood_size
                max_token_set = chain_tokens[
                    (chain_tokens["res_idx"] >= min_idx)
                    & (chain_tokens["res_idx"] <= max_idx)
                ]
                min_idx = max_idx = token["res_idx"]
                new_tokens = max_token_set[
                    max_token_set["res_idx"] == token["res_idx"]
                ]
                while new_tokens.size < self.neighborhood_size:
                    min_idx -= 1
                    max_idx += 1
                    new_tokens = max_token_set[
                        (max_token_set["res_idx"] >= min_idx)
                        & (max_token_set["res_idx"] <= max_idx)
                    ]

            new_indices = set(new_tokens["token_idx"]) - cropped
            new_tokens = token_data[list(new_indices)]
            new_atoms = int(np.sum(new_tokens["atom_num"]))
            new_protein = new_indices - ligand_ids
            if (
                len(new_indices) > max_tokens - len(cropped)
                or (max_atoms is not None and total_atoms + new_atoms > max_atoms)
                or len(cropped_protein | new_protein) > self.max_tokens_protein
            ):
                break
            cropped.update(new_indices)
            cropped_protein.update(new_protein)
            total_atoms += new_atoms

        token_data = token_data[sorted(cropped)]
        kept = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], kept)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], kept)]
        return replace(data, tokens=token_data, bonds=token_bonds)
