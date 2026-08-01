"""Render the FoldJAX repository banner.

Two ideas, stacked. The wordmark sets each letter in its own rounded tile, the
way JAX writes its own name — this project is a JAX project before it is
anything else, and the tiles carry the gradient that runs through the graphic
below them. Under it, a flat residue chain genuinely coils into an alpha helix:
the curve is computed, not drawn — radius ramps from zero, the turn angle
accumulates only while folding, and the x advance compresses as the coil
tightens, so the shape is a real helix seen from the side rather than a
decorative squiggle.

    uv run python docs/make_banner.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 400
SCALE = 3  # supersample, then downsample for clean edges

# The wordmark: one rounded tile per letter, centred, with the gradient running
# across them. Everything else in the layout is placed relative to this block.
WORDMARK = "FOLDJAX"
TILE_SIDE = 104
TILE_GAP = 14
TILE_TOP = 84
# The chain runs as a band under the wordmark rather than through it.
AXIS_Y = 328.0
AXIS_RADIUS = 56.0

FONTS = {
    "display": "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
    "display_book": "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "sans": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}


class Theme:
    def __init__(self, name, ground, deep, paper, muted, hairline, stops, amber):
        self.name = name
        self.ground = ground
        self.deep = deep
        self.paper = paper
        self.muted = muted
        self.hairline = hairline
        self.stops = stops
        self.amber = amber


DARK = Theme(
    name="dark",
    ground=(8, 13, 28),
    deep=(5, 8, 19),
    paper=(238, 242, 255),
    muted=(139, 152, 189),
    hairline=(30, 41, 74),
    stops=((139, 92, 246), (59, 130, 246), (45, 212, 191)),
    amber=(245, 158, 11),
)
LIGHT = Theme(
    name="light",
    ground=(247, 248, 252),
    deep=(237, 240, 249),
    paper=(15, 23, 42),
    muted=(88, 100, 130),
    hairline=(214, 220, 236),
    stops=((124, 58, 237), (37, 99, 235), (13, 148, 136)),
    amber=(217, 119, 6),
)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def ramp(theme, t):
    """Three-stop gradient: unfolded violet -> azure -> folded teal."""
    a, b, c = theme.stops
    return mix(a, b, t / 0.5) if t < 0.5 else mix(b, c, (t - 0.5) / 0.5)


def smoothstep(edge0, edge1, x):
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)


def toward(colour, target, t):
    return mix(colour, target, t)


def fold_curve(n=4200, turns=9.0, tilt=0.62):
    """Return the folding chain as (x, y, z, theta, t) samples.

    A helix drawn straight from the side is just a sine wave: every x has one
    y, so nothing ever overlaps and the eye reads it flat. Tilting the axis
    puts ``cos(theta)`` into x as well, which makes x non-monotonic once the
    pitch drops below the coil width — that self-overlap is what reads as a
    three-dimensional coil.
    """
    # The coil has to finish inside the frame: a turn clipped by the right edge
    # reads as a rendering accident rather than the end of a helix.
    x0, x1 = 74.0, 1198.0
    axis_y = AXIS_Y
    radius = AXIS_RADIUS

    folded = [smoothstep(0.26, 0.60, i / (n - 1)) for i in range(n)]
    total = sum(folded) or 1.0

    points = []
    turned = 0.0
    for i, f in enumerate(folded):
        t = i / (n - 1)
        turned += f
        theta = turns * 2 * math.pi * (turned / total)
        r = radius * f
        # The straight run spends x quickly; once the coil forms, x advances
        # only by the helix pitch, which is what tightens it.
        advance = (i / (n - 1)) * 0.34 + (turned / total) * 0.66
        x = lerp(x0, x1, advance) + r * math.cos(theta) * tilt
        y = axis_y + r * math.sin(theta)
        points.append((x, y, math.cos(theta) * f, theta, t))
    return points


def ribbon_quads(points, width=13.0):
    """Turn the folded part of the chain into a twisting ribbon.

    The ribbon is widest where it faces the viewer and narrows to an edge at
    the sides, which is what gives a cartoon helix its twist.
    """
    quads = []
    for i in range(len(points) - 1):
        x, y, z, theta, t = points[i]
        nx, ny, _, _, _ = points[i + 1]
        dx, dy = nx - x, ny - y
        length = math.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length
        # Overlap neighbours by half a segment; abutting quads leave hairline
        # seams once the ribbon is downsampled.
        ux, uy = dx / length * 0.6, dy / length * 0.6
        x, y, nx, ny = x - ux, y - uy, nx + ux, ny + uy
        half = width * (0.22 + 0.78 * abs(math.cos(theta))) * min(1.0, t / 0.34 + 0.12)
        quads.append(((x, y, px, py, half), (nx, ny, px, py, half), z, t))
    return quads


def draw_banner(theme: Theme) -> Image.Image:
    w, h = WIDTH * SCALE, HEIGHT * SCALE
    image = Image.new("RGB", (w, h), theme.ground)
    draw = ImageDraw.Draw(image, "RGBA")

    def s(value):
        return value * SCALE

    # Ground: a soft vertical settle so the wordmark sits on something.
    for row in range(h):
        t = row / h
        draw.line(
            [(0, row), (w, row)],
            fill=mix(theme.ground, theme.deep, smoothstep(0.35, 1.0, t)),
        )

    # A faint lattice, spaced on the type scale: this is a compiler's grid, so
    # it stays quiet and square rather than decorative.
    step = 32
    for gx in range(0, WIDTH + 1, step):
        draw.line([(s(gx), 0), (s(gx), h)], fill=(*theme.hairline, 46), width=SCALE)
    for gy in range(0, HEIGHT + 1, step):
        draw.line([(0, s(gy)), (w, s(gy))], fill=(*theme.hairline, 46), width=SCALE)

    points = fold_curve()
    fold_start = 0.24

    # Unfolded run: discrete residues, because that is what a sequence is.
    straight = [p for p in points if p[4] <= fold_start + 0.02]
    for i in range(len(straight) - 1):
        x, y, _, _, tt = straight[i]
        nx, ny = straight[i + 1][0], straight[i + 1][1]
        draw.line(
            [(s(x), s(y)), (s(nx), s(ny))],
            fill=(*ramp(theme, tt), 255),
            width=max(1, int(round(s(4.0)))),
        )
    for i in range(0, len(straight), 72):
        x, y, _, _, tt = straight[i]
        r = s(5.4)
        draw.ellipse(
            [s(x) - r, s(y) - r, s(x) + r, s(y) + r], fill=(*ramp(theme, tt), 255)
        )

    # Folded run: a twisting ribbon, painted back to front so near turns
    # occlude far ones.
    quads = [q for q in ribbon_quads(points) if q[3] > fold_start]
    for (x, y, px, py, hw), (nx, ny, npx, npy, nhw), z, tt in sorted(
        quads, key=lambda q: q[2]
    ):
        near = (z + 1) / 2
        face = toward(ramp(theme, tt), theme.ground, 0.62 * (1 - near))
        draw.polygon(
            [
                (s(x + px * hw), s(y + py * hw)),
                (s(nx + npx * nhw), s(ny + npy * nhw)),
                (s(nx - npx * nhw), s(ny - npy * nhw)),
                (s(x - px * hw), s(y - py * hw)),
            ],
            fill=(*face, 255),
        )

    # One amber residue where the chain starts to fold — the single loud note,
    # and the only place the low-confidence colour appears.
    hinge = next(p for p in points if p[4] >= fold_start)
    hx, hy = hinge[0], hinge[1]
    for radius, alpha in ((18, 40), (11, 90)):
        rr = s(radius)
        draw.ellipse(
            [s(hx) - rr, s(hy) - rr, s(hx) + rr, s(hy) + rr],
            fill=(*theme.amber, alpha),
        )
    rr = s(5.6)
    draw.ellipse(
        [s(hx) - rr, s(hy) - rr, s(hx) + rr, s(hy) + rr], fill=(*theme.amber, 255)
    )

    # --- type ------------------------------------------------------------
    tagline = ImageFont.truetype(FONTS["sans"], int(s(26)))
    label = ImageFont.truetype(FONTS["mono"], int(s(15)))
    centre = s(WIDTH / 2)

    # An eyebrow naming what actually runs, in the vernacular of the field
    # rather than the codebase.
    draw.text(
        (centre, s(56)),
        "A L P H A F O L D  3   ·   B O L T Z - 2   ·   C H A I - 1"
        "   ·   O P E N D D E   ·   P R O T E N I X",
        font=label,
        fill=theme.muted,
        anchor="ms",
    )

    draw_wordmark(draw, theme, s)

    draw.text(
        (centre, s(TILE_TOP + TILE_SIDE + 44)),
        "biomolecular structure prediction, compiled",
        font=tagline,
        fill=theme.muted,
        anchor="ms",
    )

    return image.resize((WIDTH, HEIGHT), Image.LANCZOS)


def draw_wordmark(draw, theme, s) -> None:
    """Set the name one letter per tile, the way JAX writes its own.

    The tiles carry the same violet -> azure -> teal ramp as the chain below,
    sampled at each letter's position, so the wordmark and the graphic are
    visibly the same object. Letters are knocked out to the page colour rather
    than painted, which is what makes a tile read as a tile.
    """
    count = len(WORDMARK)
    span = count * TILE_SIDE + (count - 1) * TILE_GAP
    left = (WIDTH - span) / 2
    font = ImageFont.truetype(FONTS["display"], int(s(TILE_SIDE * 0.58)))

    for index, character in enumerate(WORDMARK):
        x0 = left + index * (TILE_SIDE + TILE_GAP)
        fill = ramp(theme, index / (count - 1))
        draw.rounded_rectangle(
            [s(x0), s(TILE_TOP), s(x0 + TILE_SIDE), s(TILE_TOP + TILE_SIDE)],
            radius=s(TILE_SIDE * 0.26),
            fill=(*fill, 255),
        )
        # Centre on the glyph's own ink, not on its advance width: the letters
        # have different side bearings and centring on the box makes O and X
        # sit visibly off-axis from each other.
        box = draw.textbbox((0, 0), character, font=font)
        draw.text(
            (
                s(x0 + TILE_SIDE / 2) - (box[0] + box[2]) / 2,
                s(TILE_TOP + TILE_SIDE / 2) - (box[1] + box[3]) / 2,
            ),
            character,
            font=font,
            fill=(*theme.ground, 255),
        )


def main() -> None:
    out = Path(__file__).resolve().parent
    for theme in (DARK, LIGHT):
        path = out / f"banner-{theme.name}.png"
        draw_banner(theme).save(path, optimize=True)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
