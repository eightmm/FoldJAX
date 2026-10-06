"""`foldjax report DIR`: one static, self-contained HTML page per output directory.

Every input gets a section with one card per model side by side: the run's
metadata (model, weights, seeds, options, sampling, MSA and template policy,
what the model never read, warnings), a per-sample table of the model's own
scores, a per-residue pLDDT plot and, where the model wrote one, a PAE
heatmap. The page carries no script, stylesheet or font from anywhere else:
plots are inline SVG and the heatmap is an inline PNG, so the file can be
mailed or archived on its own.

Like everything that reads a finished run, it reads only the canonical files
(`foldjax_run.json`, each sample's ``confidence.json``, ``confidence_full.npz``
and mmCIF) and keeps each model's numbers under that model's own names and
scales; cards sit side by side so a reader can look at each model's output,
not so their scores can be ranked against each other.
"""

from __future__ import annotations

import base64
import html
import json
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from foldjax import confidence_arrays

_HEATMAP_PIXELS = 360
#: Sequential blue ramp (light -> dark); low PAE (confident) is dark.
_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip

_CSS = """
:root {
  color-scheme: light;
  --surface-0: #f5f4f0; --surface-1: #fcfcfb; --border: #dddcd6;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --series-1: #2a78d6; --other: #b9b8b0; --grid: #e7e6e1; --warn-bg: #fff4dc;
  --warn-ink: #6b4a00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --other: #5d5c56; --grid: #2c2c29; --warn-bg: #3a2f12;
    --warn-ink: #f3d38a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #383835;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --other: #5d5c56; --grid: #2c2c29; --warn-bg: #3a2f12;
  --warn-ink: #f3d38a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 12px; overflow-wrap: anywhere; }
h3 { font-size: 15px; margin: 0 0 8px; }
p.note, .muted { color: var(--text-secondary); }
.grid { display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }
.card { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 3px 8px 3px 0; vertical-align: top;
  border-bottom: 1px solid var(--grid); overflow-wrap: anywhere; }
th { color: var(--text-secondary); font-weight: 600; }
td.num, th.num { font-variant-numeric: tabular-nums; text-align: right; }
td.num { padding-right: 12px; } th.num { padding-right: 12px; }
.scroll { overflow-x: auto; }
.scroll th, .scroll td { overflow-wrap: normal; white-space: nowrap; }
p, h2 { overflow-wrap: anywhere; }
.warn { background: var(--warn-bg); color: var(--warn-ink); border-radius: 6px;
  padding: 6px 10px; margin: 8px 0; font-size: 12.5px; }
figure { margin: 12px 0 0; }
figure svg { display: block; max-width: 760px; }
figcaption { color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; }
svg text { fill: var(--text-secondary); font-size: 11px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px;
  color: var(--text-secondary); margin-top: 4px; }
.swatch { display: inline-block; width: 14px; height: 2px; vertical-align: middle;
  margin-right: 4px; }
img.heat { width: 100%; max-width: 360px; image-rendering: pixelated;
  border: 1px solid var(--border); display: block; }
.ramp { height: 8px; max-width: 360px; border-radius: 2px;
  background: linear-gradient(90deg, %RAMP%); }
.ramp-labels { display: flex; justify-content: space-between; max-width: 360px;
  font-size: 11px; color: var(--text-secondary); }
details summary { cursor: pointer; color: var(--text-secondary); margin-top: 8px; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _png(rgb: np.ndarray) -> str:
    """A data URI for an [h, w, 3] uint8 image (no imaging library needed)."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _ramp_rgb() -> np.ndarray:
    return np.asarray(
        [[int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in _RAMP],
        dtype=float,
    )


def pae_image(pae: np.ndarray, *, vmax: float = 32.0) -> str:
    """PAE as a PNG data URI, block-averaged down to at most 360 pixels."""
    matrix = np.asarray(pae, dtype=float)
    n = matrix.shape[0]
    if n > _HEATMAP_PIXELS:
        edges = np.linspace(0, n, _HEATMAP_PIXELS + 1).astype(int)
        rows = np.add.reduceat(matrix, edges[:-1], axis=0) / np.diff(edges)[:, None]
        matrix = np.add.reduceat(rows, edges[:-1], axis=1) / np.diff(edges)[None, :]
    ramp = _ramp_rgb()
    # 0 A is the darkest step: confident is strong.
    position = (1.0 - np.clip(matrix / vmax, 0.0, 1.0)) * (len(ramp) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(ramp) - 1)
    frac = (position - low)[..., None]
    rgb = ramp[low] * (1 - frac) + ramp[high] * frac
    return _png(np.clip(rgb, 0, 255).astype(np.uint8))


def residue_plddt(
    sample_dir: Path, structure: Path | None
) -> tuple[list[tuple[str, int]], np.ndarray, str] | None:
    """Per-residue pLDDT on 0-100: (chain, residue) labels, values, source."""
    try:
        arrays = confidence_arrays.load_confidence_arrays(sample_dir)
    except FileNotFoundError:
        arrays = None
    if arrays is not None:
        for name, chains_key, residues_key in (
            ("token_plddt", "token_chain_id", "token_residue_index"),
            ("atom_plddt", "atom_chain_id", "atom_residue_index"),
        ):
            values = arrays.get(name)
            chains = arrays.get(chains_key)
            residues = arrays.get(residues_key)
            if values is None or chains is None or residues is None:
                continue
            if not len(values) == len(chains) == len(residues):
                continue
            factor = 100.0 if arrays.describe(name).get("scale") == "0-1" else 1.0
            order: dict[tuple[str, int], list[float]] = {}
            for chain, residue, value in zip(
                chains.astype(str), residues.astype(int), values, strict=True
            ):
                order.setdefault((str(chain), int(residue)), []).append(float(value))
            labels = list(order)
            means = np.asarray([np.mean(order[key]) for key in labels]) * factor
            return labels, means, f"{name} (confidence_full.npz), residue mean"
    if structure is None or not structure.is_file():
        return None
    import gemmi

    model = gemmi.read_structure(str(structure))[0]
    labels, means = [], []
    for chain in model:
        for residue in chain:
            values = [atom.b_iso for atom in residue]
            if values:
                labels.append((chain.name, int(residue.seqid.num)))
                means.append(float(np.mean(values)))
    if not labels:
        return None
    return labels, np.asarray(means), "mmCIF B-factor column as written, residue mean"


def _plddt_svg(curves: Sequence[tuple[np.ndarray, bool, str]], labels) -> str:
    width, height = 640, 190
    left, right, top, bottom = 34, 8, 10, 22
    n = max(len(curve) for curve, _, _ in curves)
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x(i: float) -> float:
        return left + (plot_w * i / max(n - 1, 1))

    def y(v: float) -> float:
        return top + plot_h * (1 - min(max(v, 0.0), 100.0) / 100.0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        'aria-label="per-residue pLDDT">'
    ]
    for level in (0, 50, 70, 90):
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y(level):.1f}" '
            f'y2="{y(level):.1f}" stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{left - 4}" y="{y(level) + 3.5:.1f}" text-anchor="end">'
            f"{level}</text>"
        )
    previous = None
    for index, (chain, _residue) in enumerate(labels):
        if chain != previous:
            if index:
                parts.append(
                    f'<line x1="{x(index - 0.5):.1f}" x2="{x(index - 0.5):.1f}" '
                    f'y1="{top}" y2="{top + plot_h}" stroke="var(--text-muted)" '
                    'stroke-dasharray="3 3" stroke-width="1"/>'
                )
            parts.append(
                f'<text x="{x(index) + 2:.1f}" y="{height - 6}">{_e(chain)}</text>'
            )
            previous = chain
    ordered = sorted(curves, key=lambda item: item[1])
    for curve, best, name in ordered:
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(curve))
        color = "var(--series-1)" if best else "var(--other)"
        stroke = 2 if best else 1
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linejoin="round" points="{points}"><title>{_e(name)}</title>'
            "</polyline>"
        )
    best_curve = next((curve for curve, best, _ in curves if best), curves[0][0])
    step = max(1, len(best_curve) // 400)
    for i in range(0, len(best_curve), step):
        chain, residue = labels[i] if i < len(labels) else ("?", i)
        parts.append(
            f'<rect x="{x(i) - plot_w / max(n, 1) / 2 * step:.1f}" y="{top}" '
            f'width="{max(plot_w / max(n, 1) * step, 1):.1f}" height="{plot_h}" '
            f'fill="transparent"><title>{_e(chain)} {residue}: '
            f"{best_curve[i]:.1f}</title></rect>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _e(value)
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _metadata_rows(run: Any) -> list[tuple[str, str]]:
    manifest = run.manifest
    weights = (
        manifest.get("weights") if isinstance(manifest.get("weights"), Mapping) else {}
    )
    rows = [
        ("model", run.model),
        (
            "weights",
            weights.get("label") or weights.get("profile") or weights.get("identity"),
        ),
        ("profile", weights.get("profile")),
        ("seeds", ", ".join(str(s) for s in run.seeds) + f"  ({run.seed_source})"),
        ("sampling", json.dumps(manifest.get("sampling") or {}, sort_keys=True)),
        ("options", json.dumps(manifest.get("options") or {}, sort_keys=True)),
        ("msa", manifest.get("msa")),
        ("templates", manifest.get("templates")),
        (
            "padding",
            json.dumps(manifest.get("padding")) if manifest.get("padding") else None,
        ),
        ("configuration", run.configuration),
        ("foldjax", manifest.get("foldjax")),
        ("finished", manifest.get("finished")),
    ]
    cost = manifest.get("cost") if isinstance(manifest.get("cost"), Mapping) else {}
    if cost.get("seconds") is not None:
        rows.append(("wall", f"{cost['seconds']:.1f} s"))
    return [(key, str(value)) for key, value in rows if value not in (None, "", "{}")]


def _warnings(run: Any, samples: Sequence[Any]) -> list[str]:
    found = []
    manifest = run.manifest
    for item in manifest.get("warnings") or []:
        found.append(str(item))
    memory = manifest.get("memory")
    if isinstance(memory, Mapping) and memory.get("state") not in (None, "fits"):
        found.append(f"memory admission {memory.get('state')}: {memory.get('reason')}")
    for label, value in (
        ("ignored MSAs", run.ignored_msas),
        ("ignored templates", run.ignored_templates),
        ("ignored constraints", run.ignored_constraints),
    ):
        if value:
            found.append(
                f"{label}: {json.dumps([dict(v) for v in value], sort_keys=True)}"
            )
    for sample in samples:
        if not sample.structure_verified:
            found.append(
                f"seed {sample.seed} sample {sample.sample}: structure missing or "
                "changed since the run"
            )
    return found


def _card(run: Any) -> str:
    samples = list(run.samples)
    parts = ['<section class="card">', f"<h3>{_e(run.model)}</h3>"]
    for message in _warnings(run, samples):
        parts.append(f'<div class="warn">{_e(message)}</div>')
    parts.append('<table class="meta">')
    for key, value in _metadata_rows(run):
        parts.append(f"<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>")
    parts.append("</table>")

    names: list[str] = []
    for sample in samples:
        for name in sample.scores:
            if name not in names:
                names.append(name)
    ranking = (run.best or {}).get("score")
    if ranking in names:
        names.remove(ranking)
        names.insert(0, ranking)
    parts.append(
        "<figure><figcaption>Scores as the model reports them (its own names "
        'and scales)</figcaption><div class="scroll"><table><tr><th>seed</th>'
        "<th>sample</th>"
        + "".join(f'<th class="num">{_e(name)}</th>' for name in names)
        + "<th>best</th></tr>"
    )
    for sample in samples:
        parts.append(
            f"<tr><td>{sample.seed}</td><td>{sample.sample}</td>"
            + "".join(
                f'<td class="num">{_number(sample.scores.get(name))}</td>'
                for name in names
            )
            + f"<td>{'yes' if sample.is_best else ''}</td></tr>"
        )
    parts.append("</table></div></figure>")

    curves = []
    labels = None
    source = None
    for sample in samples:
        if sample.structure_path is None:
            continue
        found = residue_plddt(sample.structure_path.parent, sample.structure_path)
        if found is None:
            continue
        sample_labels, values, sample_source = found
        if labels is None or len(sample_labels) > len(labels):
            labels = sample_labels
        source = sample_source
        curves.append(
            (values, bool(sample.is_best), f"seed {sample.seed} sample {sample.sample}")
        )
    if curves and not any(best for _, best, _ in curves):
        values, _, name = curves[0]
        curves[0] = (values, True, name)
    if curves and labels is not None:
        parts.append(
            f"<figure><figcaption>pLDDT per residue (0-100; {_e(source)})"
            f"</figcaption>{_plddt_svg(curves, labels)}"
            '<div class="legend"><span><span class="swatch" '
            'style="background:var(--series-1)"></span>best sample (or first)</span>'
            '<span><span class="swatch" style="background:var(--other)"></span>'
            "other samples</span></div></figure>"
        )
    else:
        parts.append('<p class="muted">No per-residue pLDDT in this run.</p>')

    shown = next((s for s in samples if s.is_best), samples[0] if samples else None)
    pae_note = "No PAE: "
    if shown is not None and shown.structure_path is not None:
        try:
            arrays = confidence_arrays.load_confidence_arrays(
                shown.structure_path.parent
            )
        except FileNotFoundError:
            arrays = None
            pae_note += "the run wrote no confidence_full.npz."
        if arrays is not None and "pae" in arrays:
            chain_ids = arrays.get("token_chain_id")
            chains = (
                " / ".join(dict.fromkeys(str(c) for c in chain_ids))
                if chain_ids is not None
                else ""
            )
            pae = np.asarray(arrays["pae"], dtype=float)
            parts.append(
                f"<figure><figcaption>PAE, seed {shown.seed} sample {shown.sample} "
                f"({pae.shape[0]} tokens; chains {_e(chains)}); aligned on row i, "
                f'error of column j</figcaption><img class="heat" alt="PAE heatmap" '
                f'src="{pae_image(pae)}"><div class="ramp"></div>'
                '<div class="ramp-labels"><span>0 A</span><span>16 A</span>'
                "<span>32 A or more</span></div></figure>"
            )
        elif arrays is not None:
            pae_note += _e(arrays.unavailable.get("pae", "the model wrote no PAE."))
            parts.append(f'<p class="muted">{pae_note}</p>')
        else:
            parts.append(f'<p class="muted">{_e(pae_note)}</p>')
    parts.append("</section>")
    return "".join(parts)


def render(root: str | os.PathLike[str]) -> str:
    """The report for every run under ``root``, as one HTML string."""
    from foldjax.results import load_results

    report = load_results(root)
    groups: dict[tuple[str, str | None], list[Any]] = {}
    for run in report.runs:
        jobs = {sample.job for sample in run.samples} or {None}
        for job in sorted(jobs, key=lambda item: item or ""):
            groups.setdefault((run.input, job), []).append(run)
    sections = []
    for (input_path, job), runs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        title = Path(input_path).stem if input_path else "input"
        if job and job != title:
            title = f"{title} / {job}"
        cards = "".join(_card(run) for run in sorted(runs, key=lambda r: r.model))
        sections.append(
            f'<h2>{_e(title)}</h2><p class="note">{_e(input_path)}</p>'
            f'<div class="grid">{cards}</div>'
        )
    failures = ""
    if report.failures:
        rows = "".join(
            f"<tr><td>{_e(f.model)}</td><td>{_e(f.input)}</td><td>{_e(f.seed)}</td>"
            f"<td>{_e(f.error_type)}: {_e(f.error)}</td></tr>"
            for f in report.failures
        )
        failures = (
            '<h2>Failed runs</h2><div class="card scroll"><table><tr><th>model</th>'
            f"<th>input</th><th>seed</th><th>error</th></tr>{rows}</table></div>"
        )
    css = _CSS.replace("%RAMP%", ", ".join(reversed(_RAMP)))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>FoldJAX run report</title><style>{css}</style></head><body><main>"
        f'<h1>FoldJAX run report</h1><p class="note">{_e(report.root)} -- '
        f"{len(report.runs)} run(s). Each model's scores keep that model's own "
        "names, definitions and calibration; cards sit side by side for reading, "
        "not for ranking models against each other.</p>"
        + "".join(sections)
        + failures
        + "</main></body></html>"
    )


def write_report(
    root: str | os.PathLike[str], out: str | os.PathLike[str] | None = None
) -> Path:
    """Write the report (default ``<root>/foldjax_report.html``) and return its path."""
    from foldjax.results import load_results

    target = (
        Path(out)
        if out is not None
        else load_results(root).root / "foldjax_report.html"
    )
    text = render(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.tmp")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, target)
    return target
