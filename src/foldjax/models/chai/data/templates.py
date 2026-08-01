"""Torch-free Chai template contexts and portable feature archives."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from foldjax.models.chai.data.asset_cache import AssetCache
from foldjax.models.chai.data.input import (
    EntityType,
    Input,
    constituents_of_modified_fasta,
)

_FORMAT = "chai-jax-template-context"
_VERSION = 1
_MANIFEST = "__chai_jax_template_manifest__"
_GAP_RESTYPE = 31
_ARRAY_NAMES = (
    "template_restype",
    "template_pseudo_beta_mask",
    "template_backbone_frame_mask",
    "template_distances",
    "template_unit_vector",
)
_RESTYPES = "ARNDCQEGHILKMFPSTWYV"
_RESTYPE_ORDER = {residue: index for index, residue in enumerate(_RESTYPES)}
_STANDARD_PROTEIN_CODES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
_AA_3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _sha256(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_metadata(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": _sha256(memoryview(contiguous)),
    }


@dataclass(frozen=True)
class TemplateContext:
    """Exact NumPy counterpart of Chai's aligned template context."""

    template_restype: np.ndarray
    template_pseudo_beta_mask: np.ndarray
    template_backbone_frame_mask: np.ndarray
    template_distances: np.ndarray
    template_unit_vector: np.ndarray

    def __post_init__(self) -> None:
        restype = np.asarray(self.template_restype)
        if restype.ndim != 2:
            raise ValueError("template_restype must have shape (templates, tokens)")
        templates, tokens = restype.shape
        expected = {
            "template_restype": ((templates, tokens), np.dtype(np.int32)),
            "template_pseudo_beta_mask": (
                (templates, tokens),
                np.dtype(np.bool_),
            ),
            "template_backbone_frame_mask": (
                (templates, tokens),
                np.dtype(np.bool_),
            ),
            "template_distances": (
                (templates, tokens, tokens),
                np.dtype(np.float32),
            ),
            "template_unit_vector": (
                (templates, tokens, tokens, 3),
                np.dtype(np.float32),
            ),
        }
        for name, (shape, dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"{name} must have shape {shape} and dtype {dtype}, "
                    f"got {value.shape} and {value.dtype}"
                )
        if np.any((restype < 0) | (restype > _GAP_RESTYPE)):
            raise ValueError("template_restype contains an unknown residue class")
        if (
            not np.isfinite(self.template_distances).all()
            or not np.isfinite(self.template_unit_vector).all()
        ):
            raise ValueError("template coordinates contain non-finite values")

    @property
    def num_templates(self) -> int:
        return self.template_restype.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.template_restype.shape[1]

    @property
    def template_mask(self) -> np.ndarray:
        return self.template_restype != _GAP_RESTYPE

    @property
    def num_nonnull_templates(self) -> int:
        return int(np.any(self.template_mask, axis=-1).sum())

    @classmethod
    def empty(cls, n_templates: int, n_tokens: int) -> TemplateContext:
        if n_templates < 1 or n_tokens < 0:
            raise ValueError(
                "template count must be positive and token count non-negative"
            )
        shape = (n_templates, n_tokens)
        return cls(
            np.full(shape, _GAP_RESTYPE, np.int32),
            np.zeros(shape, np.bool_),
            np.zeros(shape, np.bool_),
            np.zeros(shape + (n_tokens,), np.float32),
            np.zeros(shape + (n_tokens, 3), np.float32),
        )

    def index_select(self, indices: np.ndarray) -> TemplateContext:
        selected = np.asarray(indices)
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("template token indices must be an integer vector")
        if np.any((selected < 0) | (selected >= self.num_tokens)):
            raise ValueError("template token index is out of range")
        return TemplateContext(
            self.template_restype[:, selected],
            self.template_pseudo_beta_mask[:, selected],
            self.template_backbone_frame_mask[:, selected],
            self.template_distances[:, selected][:, :, selected],
            self.template_unit_vector[:, selected][:, :, selected],
        )

    def pad(
        self,
        max_templates: int | None = None,
        max_tokens: int | None = None,
    ) -> TemplateContext:
        max_templates = self.num_templates if max_templates is None else max_templates
        max_tokens = self.num_tokens if max_tokens is None else max_tokens
        if self.num_templates > max_templates or self.num_tokens > max_tokens:
            raise ValueError(
                "cannot pad a template context to fewer templates or tokens"
            )
        output = TemplateContext.empty(max_templates, max_tokens)
        template_slice = slice(0, self.num_templates)
        token_slice = slice(0, self.num_tokens)
        restype = output.template_restype.copy()
        pseudo = output.template_pseudo_beta_mask.copy()
        backbone = output.template_backbone_frame_mask.copy()
        distances = output.template_distances.copy()
        vectors = output.template_unit_vector.copy()
        restype[template_slice, token_slice] = self.template_restype
        pseudo[template_slice, token_slice] = self.template_pseudo_beta_mask
        backbone[template_slice, token_slice] = self.template_backbone_frame_mask
        distances[template_slice, token_slice, token_slice] = self.template_distances
        vectors[template_slice, token_slice, token_slice] = self.template_unit_vector
        return TemplateContext(restype, pseudo, backbone, distances, vectors)

    @classmethod
    def merge(cls, contexts: Sequence[TemplateContext]) -> TemplateContext:
        """Merge chain contexts with Chai's block-diagonal pair layout."""
        if not contexts:
            return cls.empty(n_templates=4, n_tokens=1)
        template_count = max(context.num_templates for context in contexts)
        padded = [context.pad(max_templates=template_count) for context in contexts]
        restype = np.concatenate(
            [context.template_restype for context in padded], axis=1
        )
        pseudo = np.concatenate(
            [context.template_pseudo_beta_mask for context in padded], axis=1
        )
        backbone = np.concatenate(
            [context.template_backbone_frame_mask for context in padded], axis=1
        )
        tokens = sum(context.num_tokens for context in padded)
        distances = np.zeros((template_count, tokens, tokens), np.float32)
        vectors = np.zeros((template_count, tokens, tokens, 3), np.float32)
        start = 0
        for context in padded:
            stop = start + context.num_tokens
            distances[:, start:stop, start:stop] = context.template_distances
            vectors[:, start:stop, start:stop] = context.template_unit_vector
            start = stop
        return cls(restype, pseudo, backbone, distances, vectors)

    def to_model_inputs(self, *, batched: bool = False) -> dict[str, np.ndarray]:
        result = {name: np.asarray(getattr(self, name)) for name in _ARRAY_NAMES}
        if batched:
            result = {name: value[None] for name, value in result.items()}
        return result


@dataclass(frozen=True)
class NativeTemplateContext:
    context: TemplateContext
    source_id: str
    source_sha256: str
    query_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RawTemplateContext:
    """Template features plus immutable provenance for raw m8+CIF inputs."""

    context: TemplateContext
    source_sha256: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _M8Row:
    query_id: str
    subject_id: str
    subject_start: int
    subject_end: int
    evalue: float


def materialize_server_template_hits(
    inputs: Sequence[Input],
    search_results: Sequence[Mapping[str, str]],
    *,
    cache_dir: str | Path,
) -> Path:
    """Remap per-sequence ColabFold m8 files into Chai chain-name lookup form."""
    proteins = [
        item for item in inputs if item.entity_type == EntityType.PROTEIN.value
    ]
    entity_names = [item.entity_name for item in proteins]
    if len(set(entity_names)) != len(entity_names):
        raise ValueError("protein entity names must be unique for template lookup")
    if any(not name or any(char in name for char in "\t\r\n") for name in entity_names):
        raise ValueError("protein entity names cannot contain a tab or newline")
    if len(proteins) != len(search_results):
        raise ValueError("template search results do not match protein inputs")
    source_hashes: list[str] = []
    source_cache_keys: list[str] = []
    output_lines: list[str] = []
    for item, result in zip(proteins, search_results, strict=True):
        try:
            hits_path = Path(result["templateHitsPath"])
            provenance_path = Path(result["provenancePath"])
        except KeyError as error:
            raise ValueError("template-enabled search result is incomplete") from error
        if not hits_path.is_file() or not provenance_path.is_file():
            raise ValueError("template-enabled search cache is incomplete")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            raw = hits_path.read_bytes()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("template search cache is unreadable") from error
        metadata = provenance.get("files", {}).get("pdb70.m8", {})
        digest = _sha256(raw)
        if metadata.get("sha256") != digest or metadata.get("bytes") != len(raw):
            raise ValueError("template search cache content hash mismatch")
        cache_key = provenance.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("template search cache provenance is invalid")
        source_hashes.append(digest)
        source_cache_keys.append(cache_key)
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("template search m8 is not UTF-8") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 13:
                raise ValueError(
                    f"template search m8 line {line_number} must have 13 fields"
                )
            if fields[0] != "101":
                raise ValueError(
                    "per-sequence template search m8 must use query ID 101"
                )
            fields[0] = item.entity_name
            output_lines.append("\t".join(fields))

    identity = {
        "format": "chai-jax-server-template-hits",
        "version": 1,
        "queries": [
            {
                "entity_name": item.entity_name,
                "sequence_sha256": _sha256(item.sequence.encode()),
            }
            for item in proteins
        ],
        "source_cache_keys": source_cache_keys,
        "source_sha256": source_hashes,
    }
    cache_key = _sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    root = Path(cache_dir) / "server_hits"
    destination = root / cache_key
    hits = destination / "all_chain_templates.m8"
    provenance_path = destination / "provenance.json"
    content = ("\n".join(output_lines) + ("\n" if output_lines else "")).encode()
    manifest = {
        **identity,
        "cache_key": cache_key,
        "output_sha256": _sha256(content),
        "output_bytes": len(content),
    }
    if destination.exists():
        if not hits.is_file() or not provenance_path.is_file():
            raise ValueError("combined template search cache is incomplete")
        try:
            cached_manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "combined template search provenance is invalid"
            ) from error
        if cached_manifest != manifest or hits.read_bytes() != content:
            raise ValueError("combined template search cache content mismatch")
        return hits.resolve()

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=root))
    try:
        (temporary / "all_chain_templates.m8").write_bytes(content)
        (temporary / "provenance.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.rename(destination)
        except FileExistsError:
            shutil.rmtree(temporary)
            return materialize_server_template_hits(
                inputs, search_results, cache_dir=cache_dir
            )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return hits.resolve()


def _parse_m8(path: str | Path) -> list[_M8Row]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"template m8 input is not a file: {source}")
    rows: list[_M8Row] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) != 13:
            raise ValueError(
                f"template m8 line {line_number} must contain exactly 13 fields"
            )
        try:
            subject_start = int(fields[8]) - 1
            subject_end = int(fields[9])
            evalue = float(fields[10])
            for index in (6, 7):
                int(fields[index])
        except ValueError as error:
            raise ValueError(
                f"template m8 line {line_number} has invalid numeric fields"
            ) from error
        if subject_start < 0 or subject_end <= subject_start:
            raise ValueError(
                f"template m8 line {line_number} has an invalid subject interval"
            )
        subject_parts = fields[1].rsplit("_", maxsplit=1)
        if len(subject_parts) != 2 or not all(subject_parts):
            raise ValueError(
                f"template m8 line {line_number} subject must be identifier_chain"
            )
        rows.append(_M8Row(fields[0], fields[1], subject_start, subject_end, evalue))
    return sorted(rows, key=lambda row: (row.query_id, row.evalue))


def _align_with_kalign(
    reference: str,
    query: str,
    executable: str | Path = "kalign",
) -> tuple[str, str]:
    """Run the same external Kalign alignment required by official Chai-1."""
    executable_text = str(executable)
    resolved = (
        shutil.which(executable_text)
        if not Path(executable_text).is_file()
        else executable_text
    )
    if resolved is None:
        raise RuntimeError(
            "kalign>=3.3 is required for raw Chai template preprocessing"
        )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.fasta"
        output_path = root / "output.fasta"
        input_path.write_text(
            f">ref\n{reference}\n>query\n{query.upper().replace('-', '')}\n",
            encoding="ascii",
        )
        try:
            subprocess.run(
                [resolved, "-i", str(input_path), "-o", str(output_path)],
                check=True,
                close_fds=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError("kalign failed while aligning a template hit") from error
        records: list[str] = []
        sequence: list[str] = []
        for line in output_path.read_text(encoding="ascii").splitlines():
            if line.startswith(">"):
                if sequence:
                    records.append("".join(sequence))
                    sequence = []
            else:
                sequence.append(line.strip())
        if sequence:
            records.append("".join(sequence))
        if len(records) != 2 or len(records[0]) != len(records[1]):
            raise RuntimeError("kalign produced an invalid two-sequence alignment")
        return records[0], records[1]


def _read_cif_bytes(data: bytes) -> gemmi.Structure:
    if data.startswith(b"\x1f\x8b"):
        try:
            data = gzip.decompress(data)
        except gzip.BadGzipFile as error:
            raise ValueError("template CIF gzip stream is invalid") from error
    try:
        document = gemmi.cif.read_string(data.decode("utf-8"))
        return gemmi.make_structure_from_block(document.sole_block())
    except (UnicodeDecodeError, RuntimeError) as error:
        raise ValueError("template CIF/mmCIF content is invalid") from error


def _resolve_cif(
    identifier: str,
    *,
    cif_dir: Path | None,
    cache: AssetCache,
) -> tuple[gemmi.Structure, str, Mapping[str, object]]:
    candidates = (
        []
        if cif_dir is None
        else [
            cif_dir / f"{identifier}.cif.gz",
            cif_dir / f"{identifier}.cif",
            cif_dir / f"{identifier.lower()}.cif.gz",
            cif_dir / f"{identifier.lower()}.cif",
            cif_dir / f"{identifier.upper()}.cif.gz",
            cif_dir / f"{identifier.upper()}.cif",
        ]
    )
    source: str | Path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        f"https://files.rcsb.org/download/{urllib.parse.quote(identifier.upper())}.cif.gz",
    )
    asset = cache.resolve(source)
    return _read_cif_bytes(asset.path.read_bytes()), asset.sha256, asset.provenance


def _atom(residue: gemmi.Residue, name: str) -> np.ndarray | None:
    atoms = list(residue[name])
    if not atoms:
        return None
    _, _, selected = max(
        (float(atom.occ), -index, atom) for index, atom in enumerate(atoms)
    )
    return np.asarray(selected.pos.tolist(), np.float32)


def _protein_residues(
    structure: gemmi.Structure, chain_id: str
) -> tuple[str, list[gemmi.Residue]]:
    if len(structure) == 0:
        raise ValueError("template CIF contains no models")
    try:
        chain = structure[0][chain_id]
    except KeyError as error:
        raise ValueError(
            f"template CIF does not contain auth chain {chain_id}"
        ) from error
    polymer = list(chain.get_polymer())
    if not polymer:
        raise ValueError(f"template chain {chain_id} is not a polymer")
    if any(residue.name not in _STANDARD_PROTEIN_CODES for residue in polymer):
        raise ValueError(
            f"template chain {chain_id} contains modified/non-protein residues"
        )
    sequence = "".join(
        gemmi.find_tabulated_residue(residue.name).one_letter_code
        for residue in polymer
    )
    return sequence, polymer


def _protein_query_sequence(item: Input) -> str:
    parts = constituents_of_modified_fasta(item.sequence)
    if parts is None:
        raise ValueError(
            f"invalid protein FASTA for template lookup: {item.sequence!r}"
        )
    return "".join(
        part if len(part) == 1 else _AA_3_TO_1.get(part, "X") for part in parts
    )


def _alignment_indices(
    reference_aligned: str,
    hit_aligned: str,
    *,
    hit_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(reference_aligned) != len(hit_aligned) or not reference_aligned:
        raise ValueError("template alignment is empty or length-mismatched")
    query_index = -1
    hit_index = hit_start
    query_indices: list[int] = []
    hit_indices: list[int] = []
    valid: list[bool] = []
    previous_hit = hit_start
    for reference_residue, hit_residue in zip(
        reference_aligned, hit_aligned, strict=True
    ):
        if reference_residue != "-":
            query_index += 1
            query_indices.append(query_index)
            is_valid = hit_residue != "-"
            hit_indices.append(hit_index if is_valid else previous_hit)
            valid.append(is_valid)
        if hit_residue != "-":
            previous_hit = hit_index
            hit_index += 1
    non_gap = np.flatnonzero(valid)
    if non_gap.size == 0:
        raise ValueError("template alignment contains no matched hit residues")
    selection = slice(int(non_gap[0]), int(non_gap[-1]) + 1)
    return (
        np.asarray(query_indices[selection], np.int32),
        np.asarray(hit_indices[selection], np.int32),
        np.asarray(valid[selection], np.bool_),
    )


def _frame_rotations(
    n_position: np.ndarray,
    ca_position: np.ndarray,
    c_position: np.ndarray,
) -> np.ndarray:
    eps = np.float32(1e-12)
    n_xyz = n_position - ca_position
    c_xyz = c_position - ca_position
    cx, cy, cz = np.moveaxis(c_xyz, -1, 0)
    norm_xy = np.sqrt(eps + cx**2 + cy**2)
    sin_c1 = -cy / norm_xy
    cos_c1 = cx / norm_xy
    c1 = np.zeros((len(cx), 3, 3), np.float32)
    c1[:, 0, 0] = cos_c1
    c1[:, 0, 1] = -sin_c1
    c1[:, 1, 0] = sin_c1
    c1[:, 1, 1] = cos_c1
    c1[:, 2, 2] = 1
    norm = np.sqrt(eps + np.sum(c_xyz**2, axis=-1))
    sin_c2 = cz / norm
    cos_c2 = np.sqrt(cx**2 + cy**2) / norm
    c2 = np.zeros_like(c1)
    c2[:, 0, 0] = cos_c2
    c2[:, 0, 2] = sin_c2
    c2[:, 1, 1] = 1
    c2[:, 2, 0] = -sin_c2
    c2[:, 2, 2] = cos_c2
    c_rotation = c2 @ c1
    rotated_n = np.einsum("nij,nj->ni", c_rotation, n_xyz)
    ny, nz = rotated_n[:, 1], rotated_n[:, 2]
    n_norm = np.sqrt(eps + ny**2 + nz**2)
    sin_n = -nz / n_norm
    cos_n = ny / n_norm
    n_rotation = np.zeros_like(c1)
    n_rotation[:, 0, 0] = 1
    n_rotation[:, 1, 1] = cos_n
    n_rotation[:, 1, 2] = -sin_n
    n_rotation[:, 2, 1] = sin_n
    n_rotation[:, 2, 2] = cos_n
    return np.swapaxes(n_rotation @ c_rotation, -1, -2)


def _featurize_hit(
    residues: list[gemmi.Residue],
    query_indices: np.ndarray,
    hit_indices: np.ndarray,
    valid: np.ndarray,
    query_length: int,
) -> TemplateContext:
    if np.any(hit_indices >= len(residues)) or np.any(query_indices >= query_length):
        raise ValueError("template alignment indices exceed query or CIF chain length")
    count = len(query_indices)
    restype = np.full(count, _GAP_RESTYPE, np.int32)
    pseudo = np.zeros(count, np.bool_)
    backbone = np.zeros(count, np.bool_)
    reference_positions = np.zeros((count, 3), np.float32)
    n_positions = np.zeros((count, 3), np.float32)
    ca_positions = np.zeros((count, 3), np.float32)
    c_positions = np.zeros((count, 3), np.float32)
    for index, (hit_index, is_valid) in enumerate(zip(hit_indices, valid, strict=True)):
        if not is_valid:
            continue
        residue = residues[int(hit_index)]
        one_letter = gemmi.find_tabulated_residue(residue.name).one_letter_code
        restype[index] = _RESTYPE_ORDER.get(one_letter, 20)
        ca = _atom(residue, "CA")
        cb = ca if residue.name == "GLY" else _atom(residue, "CB")
        n = _atom(residue, "N")
        c = _atom(residue, "C")
        if cb is not None:
            pseudo[index] = True
            reference_positions[index] = cb
        if n is not None and ca is not None and c is not None:
            backbone[index] = True
            n_positions[index], ca_positions[index], c_positions[index] = n, ca, c
    distance = np.linalg.norm(
        reference_positions[:, None] - reference_positions[None, :], axis=-1
    ).astype(np.float32)
    distance[~(pseudo[:, None] & pseudo[None, :])] = 100.0
    rotations = _frame_rotations(n_positions, ca_positions, c_positions)
    delta = ca_positions[None, :, :] - ca_positions[:, None, :]
    rigid_vector = np.einsum("ilk,ikj->ilj", delta, rotations)
    inverse_distance = 1.0 / np.sqrt(
        np.float32(1e-12) + np.sum(rigid_vector**2, axis=-1)
    )
    frame_pair = backbone[:, None] & backbone[None, :]
    unit_vector = rigid_vector * (inverse_distance * frame_pair)[..., None]
    aligned = TemplateContext.empty(1, query_length)
    aligned_restype = aligned.template_restype.copy()
    aligned_pseudo = aligned.template_pseudo_beta_mask.copy()
    aligned_backbone = aligned.template_backbone_frame_mask.copy()
    aligned_distance = aligned.template_distances.copy()
    aligned_vector = aligned.template_unit_vector.copy()
    aligned_restype[0, query_indices] = restype
    aligned_pseudo[0, query_indices] = pseudo
    aligned_backbone[0, query_indices] = backbone
    aligned_distance[0][np.ix_(query_indices, query_indices)] = distance
    aligned_vector[0][np.ix_(query_indices, query_indices)] = unit_vector
    return TemplateContext(
        aligned_restype,
        aligned_pseudo,
        aligned_backbone,
        aligned_distance,
        aligned_vector,
    )


def build_template_context_from_m8(
    inputs: Sequence[Input],
    *,
    token_asym_id: np.ndarray,
    token_residue_index: np.ndarray,
    m8_path: str | Path,
    template_cif_dir: str | Path | None,
    cache_dir: str | Path,
    kalign_executable: str | Path = "kalign",
) -> RawTemplateContext:
    """Build Chai template features from raw m8 hits and CIF/mmCIF structures."""
    asym_id = np.asarray(token_asym_id)
    residue_index = np.asarray(token_residue_index)
    if asym_id.ndim != 1 or residue_index.shape != asym_id.shape:
        raise ValueError("template token asym/residue indices must be matching vectors")
    rows = _parse_m8(m8_path)
    cache = AssetCache(cache_dir)
    cif_dir = None if template_cif_dir is None else Path(template_cif_dir)
    contexts: list[TemplateContext] = []
    cif_hashes: dict[str, str] = {}
    cif_provenance: dict[str, Mapping[str, object]] = {}
    used_template_ids: list[str] = []
    for chain_index, item in enumerate(inputs, start=1):
        token_selection = np.flatnonzero(asym_id == chain_index)
        if token_selection.size == 0:
            raise ValueError(f"input chain {chain_index} has no structure tokens")
        _, blowout = np.unique(residue_index[token_selection], return_inverse=True)
        query_length = int(blowout.max()) + 1
        loaded: list[TemplateContext] = []
        if item.entity_type == EntityType.PROTEIN.value:
            query_sequence = _protein_query_sequence(item)
            if len(query_sequence) != query_length:
                raise ValueError(
                    "template query residue count does not match "
                    "structure token indices"
                )
            for row in (row for row in rows if row.query_id == item.entity_name):
                if len(loaded) == 4:
                    break
                identifier, chain_id = row.subject_id.rsplit("_", maxsplit=1)
                structure, cif_hash, provenance = _resolve_cif(
                    identifier, cif_dir=cif_dir, cache=cache
                )
                cif_hashes[row.subject_id] = cif_hash
                cif_provenance[row.subject_id] = provenance
                sequence, residues = _protein_residues(structure, chain_id)
                if row.subject_end > len(sequence):
                    raise ValueError(
                        f"template m8 interval exceeds {row.subject_id} chain length"
                    )
                reference_aligned, hit_aligned = _align_with_kalign(
                    query_sequence,
                    sequence[row.subject_start : row.subject_end],
                    kalign_executable,
                )
                query_indices, hit_indices, valid = _alignment_indices(
                    reference_aligned,
                    hit_aligned,
                    hit_start=row.subject_start,
                )
                loaded.append(
                    _featurize_hit(
                        residues, query_indices, hit_indices, valid, query_length
                    )
                )
                used_template_ids.append(row.subject_id)
        chain_context = (
            TemplateContext.empty(1, query_length)
            if not loaded
            else TemplateContext(
                *(
                    np.concatenate([getattr(hit, name) for hit in loaded], axis=0)
                    for name in _ARRAY_NAMES
                )
            )
        )
        contexts.append(chain_context.index_select(blowout.astype(np.int32)))
    context = TemplateContext.merge(contexts)
    m8_bytes = Path(m8_path).read_bytes()
    source_manifest = {
        "m8_sha256": _sha256(m8_bytes),
        "cif_sha256": dict(sorted(cif_hashes.items())),
        "queries": [
            {
                "entity_name": item.entity_name,
                "entity_type": item.entity_type,
                "sequence_sha256": _sha256(item.sequence.encode("utf-8")),
            }
            for item in inputs
        ],
        "preprocessor": "chai-jax-raw-template-v1",
    }
    source_sha256 = _sha256(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    return RawTemplateContext(
        context=context,
        source_sha256=source_sha256,
        provenance={
            **source_manifest,
            "source_sha256": source_sha256,
            "template_ids": used_template_ids,
            "cif_provenance": cif_provenance,
        },
    )


def save_native_template_context(
    context: TemplateContext,
    path: str | Path,
    *,
    source_id: str,
    source_sha256: str,
    query_identity: Mapping[str, Any],
) -> None:
    """Atomically write fully featurized templates for Torch-free inference."""
    if not source_id:
        raise ValueError("template source_id is required")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    try:
        json.dumps(query_identity, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("template query identity must be JSON-serializable") from error
    arrays = context.to_model_inputs()
    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "query_identity": dict(query_identity),
        "arrays": {name: _array_metadata(value) for name, value in arrays.items()},
    }
    manifest_array = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), np.uint8
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        np.savez(output, **{_MANIFEST: manifest_array, **arrays})
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_native_template_context(
    path: str | Path,
    *,
    expected_query_identity: Mapping[str, Any] | None = None,
) -> NativeTemplateContext:
    """Load and checksum-validate precomputed Chai template model inputs."""
    with np.load(path, allow_pickle=False) as archive:
        if _MANIFEST not in archive.files:
            raise ValueError("native template manifest is missing")
        try:
            manifest = json.loads(archive[_MANIFEST].tobytes().decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native template manifest is invalid") from error
        if manifest.get("format") != _FORMAT or manifest.get("version") != _VERSION:
            raise ValueError("unsupported native template archive format")
        if expected_query_identity is not None and manifest.get(
            "query_identity"
        ) != dict(expected_query_identity):
            raise ValueError("native template query identity mismatch")
        if set(archive.files) - {_MANIFEST} != set(_ARRAY_NAMES):
            raise ValueError("native template array names mismatch")
        arrays: dict[str, np.ndarray] = {}
        metadata = manifest.get("arrays")
        if not isinstance(metadata, dict) or set(metadata) != set(_ARRAY_NAMES):
            raise ValueError("native template array manifest is invalid")
        for name in _ARRAY_NAMES:
            value = np.array(archive[name], copy=True)
            if _array_metadata(value) != metadata[name]:
                raise ValueError(f"native template array metadata mismatch: {name}")
            arrays[name] = value
    return NativeTemplateContext(
        context=TemplateContext(**arrays),
        source_id=manifest["source_id"],
        source_sha256=manifest["source_sha256"],
        query_identity=manifest["query_identity"],
    )


def resolve_native_template_context(
    source: str | Path,
    *,
    cache_dir: str | Path,
    expected_sha256: str | None = None,
    expected_query_identity: Mapping[str, Any] | None = None,
) -> NativeTemplateContext:
    """Resolve local/server template features with content verification."""
    asset = AssetCache(cache_dir).resolve(source, expected_sha256=expected_sha256)
    return load_native_template_context(
        asset.path, expected_query_identity=expected_query_identity
    )


__all__ = [
    "NativeTemplateContext",
    "RawTemplateContext",
    "TemplateContext",
    "build_template_context_from_m8",
    "load_native_template_context",
    "materialize_server_template_hits",
    "resolve_native_template_context",
    "save_native_template_context",
]
