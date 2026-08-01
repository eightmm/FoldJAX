"""Torch-free Chai public result and prediction-output contracts."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from foldjax.models.chai.ranking.rank import SampleRanking, get_scores

if TYPE_CHECKING:
    from foldjax.models.chai.data.structure import StructureContext

CifTextWriter = Callable[[np.ndarray, np.ndarray | None, int], str]


def write_msa_coverage_pdf(
    input_tokens: np.ndarray,
    msa_tokens: np.ndarray,
    msa_mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a compact, dependency-free MSA identity and coverage plot.

    The plot follows Chai's public visualization semantics: rows are sorted by
    sequence identity, colored cells are non-gap/non-padding residues scaled by
    identity, and the black line is per-position sequence coverage.
    """

    query = np.asarray(input_tokens).reshape(-1)
    tokens = np.asarray(msa_tokens)
    mask = np.asarray(msa_mask, dtype=bool)
    if tokens.ndim != 2 or mask.shape != tokens.shape:
        raise ValueError("msa_tokens and msa_mask must have shape (depth, token)")
    if tokens.shape[1] != query.size:
        raise ValueError("MSA token count must match input_tokens")
    active_rows = mask.any(axis=1)
    tokens = tokens[active_rows]
    mask = mask[active_rows]
    if tokens.shape[0] == 0:
        raise ValueError("cannot plot an empty MSA")

    valid = mask & (tokens != 31) & (tokens != 32)
    identities = (tokens == query[None, :]).mean(axis=1)
    order = np.argsort(-identities, kind="stable")
    tokens, valid, identities = tokens[order], valid[order], identities[order]
    coverage = valid.mean(axis=0)

    # Bound vector-PDF size for production MSAs while retaining the full span.
    row_indices = np.linspace(0, tokens.shape[0] - 1, min(tokens.shape[0], 200)).astype(
        np.int64
    )
    column_groups = np.array_split(np.arange(query.size), min(query.size, 400))
    width, height = 612.0, 420.0
    left, bottom, plot_width, plot_height = 48.0, 70.0, 520.0, 290.0
    cell_width = plot_width / len(column_groups)
    cell_height = plot_height / len(row_indices)
    commands = [
        "BT /F1 12 Tf 48 390 Td (MSA sequence coverage) Tj ET",
        "BT /F1 9 Tf 270 42 Td (Positions) Tj ET",
        "BT /F1 9 Tf 12 205 Td 90 Tz (Sequences) Tj ET",
        "0.8 w 0 0 0 RG",
    ]
    for display_row, row in enumerate(row_indices):
        identity = float(identities[row])
        y = bottom + plot_height - (display_row + 1) * cell_height
        red, green, blue = 1.0 - identity, 0.25 + 0.55 * identity, identity
        commands.append(f"{red:.3f} {green:.3f} {blue:.3f} rg")
        for display_column, columns in enumerate(column_groups):
            valid_fraction = float(valid[row, columns].mean())
            if valid_fraction == 0.0:
                continue
            x = left + display_column * cell_width
            commands.append(
                f"{x:.2f} {y:.2f} {cell_width + 0.05:.2f} {cell_height + 0.05:.2f} re f"
            )
    coverage_reduced = np.asarray(
        [coverage[columns].mean() for columns in column_groups], dtype=np.float64
    )
    commands.append("0 0 0 RG 1.2 w")
    points = [
        (
            left + (index + 0.5) * cell_width,
            bottom + float(value) * plot_height,
        )
        for index, value in enumerate(coverage_reduced)
    ]
    commands.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
    commands.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
    commands.append("S")

    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.0f} "
            f"{height:.0f}] /Resources << /Font << /F1 5 0 R >> >> "
            "/Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pdf)
    return destination


class OutputDirectoryNotEmptyError(RuntimeError):
    """Raised before inference outputs can overwrite an existing directory."""


@dataclass(frozen=True)
class StructureCandidates:
    """Chai-compatible paths, rankings, and unpadded confidence scores."""

    cif_paths: list[Path]
    ranking_data: list[SampleRanking]
    msa_coverage_plot_path: Path | None
    pae: np.ndarray
    pde: np.ndarray
    plddt: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.cif_paths)
        if len(self.ranking_data) != count:
            raise ValueError("cif_paths and ranking_data must have equal length")
        if self.pae.ndim != 3 or self.pae.shape[0] != count:
            raise ValueError("pae must have shape (candidate, token, token)")
        if self.pde.shape != self.pae.shape:
            raise ValueError("pde must have the same shape as pae")
        if self.pae.shape[1] != self.pae.shape[2]:
            raise ValueError("pae and pde token axes must be square")
        if self.plddt.shape != self.pae.shape[:2]:
            raise ValueError("plddt must have shape (candidate, token)")

    def sorted(self) -> StructureCandidates:
        """Return candidates ordered by descending aggregate score."""

        scores = np.asarray(
            [_aggregate_score(ranking) for ranking in self.ranking_data],
            dtype=np.float64,
        )
        indices = np.argsort(-scores, kind="stable")
        return StructureCandidates(
            cif_paths=[self.cif_paths[int(index)] for index in indices],
            ranking_data=[self.ranking_data[int(index)] for index in indices],
            msa_coverage_plot_path=self.msa_coverage_plot_path,
            pae=self.pae[indices],
            pde=self.pde[indices],
            plddt=self.plddt[indices],
        )

    @classmethod
    def concat(cls, candidates: Sequence[StructureCandidates]) -> StructureCandidates:
        """Concatenate trunk samples in upstream generation order."""

        if not candidates:
            raise ValueError("cannot concatenate an empty candidate sequence")
        token_count = candidates[0].pae.shape[1]
        if any(candidate.pae.shape[1] != token_count for candidate in candidates):
            raise ValueError("all trunk candidates must have the same token count")
        return cls(
            cif_paths=list(
                itertools.chain.from_iterable(
                    candidate.cif_paths for candidate in candidates
                )
            ),
            ranking_data=list(
                itertools.chain.from_iterable(
                    candidate.ranking_data for candidate in candidates
                )
            ),
            msa_coverage_plot_path=candidates[0].msa_coverage_plot_path,
            pae=np.concatenate([candidate.pae for candidate in candidates], axis=0),
            pde=np.concatenate([candidate.pde for candidate in candidates], axis=0),
            plddt=np.concatenate([candidate.plddt for candidate in candidates], axis=0),
        )


@dataclass(frozen=True)
class TrunkPrediction:
    """All model-independent data needed to write one trunk's candidates."""

    atom_coords: Any
    atom_plddt: Any | None
    atom_mask: Any
    token_mask: Any
    ranking_data: Sequence[SampleRanking]
    pae: Any
    pde: Any
    plddt: Any
    msa_coverage_plot_path: Path | None = None
    structure_context: StructureContext | None = None
    asym_entity_names: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        if self.asym_entity_names is not None or self.structure_context is None:
            return
        context = self.structure_context
        valid_tokens = np.asarray(context.token_exists_mask, dtype=bool)
        names: dict[int, str] = {}
        for value in np.unique(context.token_asym_id[valid_tokens]):
            asym_id = int(value)
            if asym_id == 0:
                continue
            token_indices = np.flatnonzero(
                valid_tokens & (context.token_asym_id == asym_id)
            )
            encoded_names = np.unique(context.subchain_id[token_indices], axis=0)
            if encoded_names.shape[0] != 1:
                raise ValueError(f"asym_id {asym_id} has multiple subchain names")
            names[asym_id] = "".join(
                chr(int(code)) for code in encoded_names[0] if int(code) != 255
            ).rstrip("\x00")
        object.__setattr__(self, "asym_entity_names", names)


def write_prediction_outputs(
    output_dir: str | Path,
    trunk_predictions: Sequence[TrunkPrediction],
    *,
    cif_text_writer: CifTextWriter | None = None,
) -> StructureCandidates:
    """Write Chai filenames and return concatenated public candidates.

    By default, ``prediction.structure_context`` drives the native ModelCIF
    writer. An injected ``cif_text_writer`` remains available for lightweight
    tests and custom serialization; it receives unpadded coordinates, optional
    atom pLDDT values on Chai's 0--100 B-factor scale, and the model index.
    """

    if not trunk_predictions:
        raise ValueError("at least one trunk prediction is required")
    normalized = [_normalize_prediction(prediction) for prediction in trunk_predictions]
    if cif_text_writer is None and any(
        prediction.structure_context is None for prediction in normalized
    ):
        raise ValueError(
            "structure_context is required when cif_text_writer is omitted"
        )
    root = _prepare_empty_output_directory(output_dir)
    multiple_trunks = len(normalized) > 1
    results = []
    for trunk_index, prediction in enumerate(normalized):
        prediction_dir = root / f"trunk_{trunk_index}" if multiple_trunks else root
        if multiple_trunks:
            prediction_dir.mkdir(parents=True, exist_ok=False)
        results.append(
            _write_single_trunk(
                prediction_dir,
                prediction,
                cif_text_writer=cif_text_writer,
            )
        )
    return StructureCandidates.concat(results)


@dataclass(frozen=True)
class _NormalizedPrediction:
    atom_coords: np.ndarray
    atom_plddt: np.ndarray | None
    atom_mask: np.ndarray
    token_mask: np.ndarray
    ranking_data: list[SampleRanking]
    pae: np.ndarray
    pde: np.ndarray
    plddt: np.ndarray
    msa_coverage_plot_path: Path | None
    structure_context: StructureContext | None
    asym_entity_names: Mapping[int, str] | None


def _normalize_prediction(prediction: TrunkPrediction) -> _NormalizedPrediction:
    coords = np.asarray(prediction.atom_coords)
    atom_mask = np.asarray(prediction.atom_mask, dtype=bool)
    token_mask = np.asarray(prediction.token_mask, dtype=bool)
    pae = np.asarray(prediction.pae)
    pde = np.asarray(prediction.pde)
    plddt = np.asarray(prediction.plddt)
    rankings = list(prediction.ranking_data)
    count = len(rankings)
    if count == 0:
        raise ValueError("each trunk prediction requires at least one candidate")
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("atom_coords must have shape (candidate, atom, 3)")
    if coords.shape[0] != count:
        raise ValueError("atom_coords candidate count must match ranking_data")
    if atom_mask.shape != coords.shape[1:2]:
        raise ValueError("atom_mask must have shape (atom,)")
    if token_mask.ndim != 1:
        raise ValueError("token_mask must have shape (token,)")
    expected_pair = (count, token_mask.size, token_mask.size)
    if pae.shape != expected_pair or pde.shape != expected_pair:
        raise ValueError("pae and pde must match padded candidate/token dimensions")
    if plddt.shape != (count, token_mask.size):
        raise ValueError("plddt must match padded candidate/token dimensions")
    atom_plddt = None
    if prediction.atom_plddt is not None:
        atom_plddt = np.asarray(prediction.atom_plddt)
        if atom_plddt.shape != coords.shape[:2]:
            raise ValueError("atom_plddt must have shape (candidate, atom)")
    context = prediction.structure_context
    if context is not None:
        if context.num_atoms != coords.shape[1]:
            raise ValueError("structure_context atom count must match atom_coords")
        if context.num_tokens != token_mask.size:
            raise ValueError("structure_context token count must match token_mask")
    return _NormalizedPrediction(
        atom_coords=coords,
        atom_plddt=atom_plddt,
        atom_mask=atom_mask,
        token_mask=token_mask,
        ranking_data=rankings,
        pae=pae,
        pde=pde,
        plddt=plddt,
        msa_coverage_plot_path=prediction.msa_coverage_plot_path,
        structure_context=context,
        asym_entity_names=prediction.asym_entity_names,
    )


def _prepare_empty_output_directory(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(root)
        entries = list(root.iterdir())
        if entries and not (
            len(entries) == 1 and entries[0].name == "msas" and entries[0].is_dir()
        ):
            raise OutputDirectoryNotEmptyError(f"output directory is not empty: {root}")
    else:
        root.mkdir(parents=True)
    return root


def _write_single_trunk(
    output_dir: Path,
    prediction: _NormalizedPrediction,
    *,
    cif_text_writer: CifTextWriter | None,
) -> StructureCandidates:
    atom_indices = np.flatnonzero(prediction.atom_mask)
    token_indices = np.flatnonzero(prediction.token_mask)
    cif_paths = []
    for model_index, ranking in enumerate(prediction.ranking_data):
        coords = prediction.atom_coords[model_index, atom_indices]
        b_factors = None
        if prediction.atom_plddt is not None:
            b_factors = 100.0 * prediction.atom_plddt[model_index, atom_indices]
        if cif_text_writer is None:
            assert prediction.structure_context is not None
            from foldjax.models.chai.io.modelcif import modelcif_text

            cif_text = modelcif_text(
                coords,
                b_factors,
                prediction.structure_context,
                atom_indices=atom_indices,
                asym_entity_names=prediction.asym_entity_names,
            )
        else:
            cif_text = cif_text_writer(coords, b_factors, model_index)
        if not isinstance(cif_text, str):
            raise TypeError("cif_text_writer must return str")
        cif_path = output_dir / f"pred.model_idx_{model_index}.cif"
        cif_path.write_text(cif_text, encoding="utf-8")
        score_path = output_dir / f"scores.model_idx_{model_index}.npz"
        np.savez(score_path, **get_scores(ranking))
        cif_paths.append(cif_path)

    pae = prediction.pae[:, token_indices][:, :, token_indices]
    pde = prediction.pde[:, token_indices][:, :, token_indices]
    plddt = prediction.plddt[:, token_indices]
    return StructureCandidates(
        cif_paths=cif_paths,
        ranking_data=prediction.ranking_data,
        msa_coverage_plot_path=prediction.msa_coverage_plot_path,
        pae=pae,
        pde=pde,
        plddt=plddt,
    )


def _aggregate_score(ranking: SampleRanking) -> float:
    score = np.asarray(ranking.aggregate_score)
    if score.size != 1:
        raise ValueError("ranking aggregate_score must contain exactly one value")
    return float(score.reshape(-1)[0])
