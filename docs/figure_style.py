"""Shared chart tokens and marks for the figures in this directory.

One place for the palette, the chrome, and the two mark helpers, so the
benchmark, scaling and capacity figures cannot drift into three different looks.

The palette is a validated instance rather than a taste: the accent is the
categorical blue and everything that is context is the de-emphasis gray, which
is the *emphasis* form -- one series is the subject, the rest is background.
Both modes are selected for their own surface rather than flipped.

The two spacers do the separating: a surface-coloured gap between touching
bars, and a surface ring on markers that overlap. Nothing gets a stroke drawn
around it to hold it apart, because a stroke is ink that is not data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Every colour a figure here is allowed to use."""

    name: str
    surface: str
    accent: str          # FoldJAX: the series the figures are about
    context: str         # upstream: present, deliberately recessive
    ink: str
    ink_secondary: str
    muted: str           # axis and tick labels
    grid: str
    baseline: str
    limit: str           # the card-capacity rule

    @property
    def dark(self) -> bool:
        return self.name == "dark"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    accent="#2a78d6",
    context="#898781",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    limit="#c3c2b7",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    accent="#3987e5",
    context="#898781",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    limit="#383835",
)

THEMES = (LIGHT, DARK)

#: Bars are capped rather than filled to the band: the leftover is air, and air
#: is what keeps a dense figure readable.
BAR_THICKNESS_PT = 9.0
#: The data end is rounded; the baseline end stays square, so the bars all start
#: from one edge and the eye reads length rather than shape.
BAR_ROUND_PX = 4.0
#: White doing the separating, at one consistent width.
SURFACE_GAP_PX = 2.0


def apply_theme(figure, axes, theme: Theme) -> None:
    """Chrome: recessive grid, no box, ticks that do not compete with data."""
    figure.patch.set_facecolor(theme.surface)
    for axis in axes:
        axis.set_facecolor(theme.surface)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(theme.baseline)
        axis.spines["bottom"].set_linewidth(1.0)
        axis.tick_params(colors=theme.muted, labelsize=8.5, length=0)
        axis.set_axisbelow(True)


def rounded_bar(axis, x0, x1, y, thickness_pt, color, *, theme, zorder=3):
    """One horizontal bar: square at the baseline, 4px round at the data end.

    The radius is in pixels rather than data units on purpose. These panels
    include a log axis, where a data-space radius would make every bar a
    different shape and the rounding would read as an encoding.
    """
    import numpy as np
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    figure = axis.figure
    scale = figure.dpi / 72.0
    half = thickness_pt * scale / 2.0

    (px0, py), (px1, _) = axis.transData.transform([(x0, y), (x1, y)])
    if px1 < px0:
        px0, px1 = px1, px0
    radius = min(BAR_ROUND_PX * scale, (px1 - px0), half)
    if radius <= 0.5:
        radius = 0.0
    top, bottom = py + half, py - half

    # Anticlockwise from the baseline corner, rounding only the far end.
    vertices = [
        (px0, bottom),
        (px1 - radius, bottom),
        (px1, bottom),
        (px1, bottom + radius),
        (px1, top - radius),
        (px1, top),
        (px1 - radius, top),
        (px0, top),
        (px0, bottom),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(np.asarray(vertices), codes),
        facecolor=color,
        edgecolor="none",
        transform=None,
        zorder=zorder,
        clip_on=False,
    )
    axis.add_patch(patch)
    return patch


def did_not_run(axis, y, text, theme, *, zorder=4):
    """A size that produced no structure, marked without pretending to be one.

    Drawn as a label at the axis edge rather than a hatched bar spanning the
    panel: a failure has no magnitude, and a full-width fill claims one. It is
    also texture, which belongs to accessibility and print rather than to
    decoration.
    """
    axis.text(
        0.995,
        y,
        text,
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=7.5,
        color=theme.muted,
        style="italic",
        zorder=zorder,
    )
