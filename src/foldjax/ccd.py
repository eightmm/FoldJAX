"""Model-neutral lookups in the wwPDB Chemical Component Dictionary.

The common input layer asks two questions of the CCD before any model runs:
does a ligand or modification code name a component at all, and is a bond's
atom name an atom of the residue it points at. Every backend answers both,
but only after its weights and featurizer have loaded, and some answered with
a ``KeyError`` instead of a sentence.

The dictionary is the shared ``components.cif`` that `foldjax weights fetch`
puts in the asset store for Protenix and OpenDDE (or the file
``PROTENIX_CCD_COMPONENTS_FILE`` names). When it is absent every lookup returns
``None`` and the caller skips the check: a store without those assets must not
start refusing jobs a backend would have run.

Nothing here imports a port, gemmi or RDKit; one data block is read through
``mmap`` without parsing the 490 MB file.
"""

from __future__ import annotations

import mmap
import os
import re
from functools import lru_cache
from pathlib import Path

#: Protein one-letter codes and the component each one is.
PROTEIN_RESIDUES = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "U": "SEC",
    "O": "PYL",
}
DNA_RESIDUES = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
RNA_RESIDUES = {"A": "A", "C": "C", "G": "G", "U": "U"}

_TOKEN = re.compile(r"'(?:[^']|'(?=\S))*'|\"(?:[^\"]|\"(?=\S))*\"|\S+")


def components_file() -> Path | None:
    """The ``components.cif`` this process would read, or None when absent."""
    configured = os.environ.get("PROTENIX_CCD_COMPONENTS_FILE")
    candidates = [Path(configured)] if configured else []
    from foldjax.paths import assets_dir

    candidates.append(assets_dir() / "components.cif")
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def _sorted_bounds(mapped: mmap.mmap, code: bytes) -> tuple[int, int] | None:
    """Binary-search a code-sorted dictionary for ``data_<code>``."""
    low, high = 0, len(mapped)
    while low < high:
        midpoint = (low + high) // 2
        marker = mapped.rfind(b"\ndata_", 0, midpoint + 1)
        start = 0 if marker < 0 else marker + 1
        if not mapped[start : start + 5] == b"data_":
            return None
        line_end = mapped.find(b"\n", start)
        if line_end < 0:
            line_end = len(mapped)
        current = mapped[start + 5 : line_end].rstrip(b"\r")
        if current == code:
            end = mapped.find(b"\ndata_", line_end)
            return start, len(mapped) if end < 0 else end + 1
        if current < code:
            following = mapped.find(b"\ndata_", line_end)
            if following < 0 or following + 1 <= low:
                return None
            low = following + 1
        else:
            if start >= high:
                return None
            high = start
    return None


def _linear_bounds(mapped: mmap.mmap, code: bytes) -> tuple[int, int] | None:
    marker = b"data_" + code + b"\n"
    start = mapped.find(marker)
    while start > 0 and mapped[start - 1 : start] != b"\n":
        start = mapped.find(marker, start + 1)
    if start < 0:
        return None
    end = mapped.find(b"\ndata_", start + len(marker))
    return start, len(mapped) if end < 0 else end + 1


def _block(path: Path, code: str) -> str | None:
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            if mapped.find(b"data_", 0, 1 << 16) < 0:
                # Not a CIF dictionary at all; no answer rather than "absent".
                raise ValueError(f"{path} holds no CIF data block")
            raw = code.encode()
            # The released file is sorted by code; a custom one may not be.
            bounds = _sorted_bounds(mapped, raw) or _linear_bounds(mapped, raw)
            if bounds is None:
                return None
            return mapped[bounds[0] : bounds[1]].decode("utf-8", errors="replace")


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _atom_names(block: str) -> tuple[str, ...]:
    """``_chem_comp_atom.atom_id`` of one block, looped or single-valued."""
    lines = block.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("_chem_comp_atom.atom_id"):
            # Key-value form: a one-atom component (an ion) is not looped.
            parts = _TOKEN.findall(stripped)
            if len(parts) >= 2:
                return (_unquote(parts[1]),)
        if stripped != "loop_":
            continue
        headers: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip().startswith("_"):
            headers.append(lines[cursor].strip())
            cursor += 1
        if not headers or not headers[0].startswith("_chem_comp_atom."):
            continue
        column = headers.index("_chem_comp_atom.atom_id")
        names: list[str] = []
        values: list[str] = []
        while cursor < len(lines):
            text = lines[cursor].strip()
            if not text or text.startswith("#") or text == "loop_":
                break
            if text.startswith("_") or text.startswith("data_"):
                break
            values.extend(_TOKEN.findall(text))
            while len(values) >= len(headers):
                names.append(_unquote(values[column]))
                values = values[len(headers) :]
            cursor += 1
        return tuple(names)
    return ()


@lru_cache(maxsize=512)
def _lookup(
    path: str, stamp: tuple[int, int, int], code: str
) -> tuple[bool, tuple[str, ...]]:
    block = _block(Path(path), code)
    return (False, ()) if block is None else (True, _atom_names(block))


def lookup(code: str) -> tuple[bool, tuple[str, ...]] | None:
    """Whether ``code`` is a CCD component, and its atom names.

    None means the question could not be asked: no dictionary is installed,
    or it could not be read. An empty name tuple for a known component means
    its atom table could not be parsed, so atom checks are skipped for it.
    """
    path = components_file()
    if path is None:
        return None
    try:
        info = path.stat()
        return _lookup(
            str(path.resolve()), (info.st_ino, info.st_size, info.st_mtime_ns), code
        )
    except (OSError, ValueError):
        return None
