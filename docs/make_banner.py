"""Render the FoldJAX repository banner.

The graphic is the product's one-line pitch: a flat residue chain on the left
genuinely coils into an alpha helix on the right. The curve is computed, not
drawn — radius ramps from zero, the turn angle accumulates only while folding,
and the x advance compresses as the coil tightens, so the shape is a real helix
seen from the side rather than a decorative squiggle.

    uv run python docs/make_banner.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 400
SCALE = 3  # supersample, then downsample for clean edges

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
    x0, x1 = 62.0, 1268.0
    axis_y = 312.0
    radius = 66.0

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
        advance = (i / (n - 1)) * 0.26 + (turned / total) * 0.74
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
    display = ImageFont.truetype(FONTS["display"], int(s(92)))
    book = ImageFont.truetype(FONTS["display_book"], int(s(92)))
    tagline = ImageFont.truetype(FONTS["sans"], int(s(25)))
    mono = ImageFont.truetype(FONTS["mono"], int(s(19)))
    label = ImageFont.truetype(FONTS["mono"], int(s(16)))

    left, baseline = s(88), s(132)

    # "Fold" in the book weight, "JAX" in demi: the fold is the subject, JAX is
    # the substrate, and the weight change says so without a second colour.
    fold_w = draw.textlength("Fold", font=book)
    draw.text((left, baseline), "Fold", font=book, fill=theme.paper, anchor="ls")
    draw.text(
        (left + fold_w, baseline),
        "JAX",
        font=display,
        fill=theme.stops[2],
        anchor="ls",
    )

    draw.text(
        (left + s(4), baseline + s(38)),
        "protein folding, compiled",
        font=tagline,
        fill=theme.muted,
        anchor="ls",
    )

    # An eyebrow that says what the thing actually is, in the vernacular of the
    # field rather than the codebase.
    draw.text(
        (left + s(5), s(58)),
        "B O L T Z - 2   ·   C H A I - 1   ·   O P E N D D E   ·   P R O T E N I X",
        font=label,
        fill=theme.muted,
        anchor="ls",
    )

    # The real command, because it is the shortest true description of the API.
    command = "$ foldjax predict --model boltz2 --input job.yaml"
    pad_x = s(18)
    box_w = draw.textlength(command, font=mono) + pad_x * 2
    box_h = s(44)
    box_y = s(186)
    draw.rounded_rectangle(
        [left, box_y, left + box_w, box_y + box_h],
        radius=s(8),
        fill=(*theme.hairline, 130),
        outline=(*theme.hairline, 255),
        width=max(1, int(s(1))),
    )
    draw.text(
        (left + pad_x, box_y + box_h / 2),
        command,
        font=mono,
        fill=theme.muted,
        anchor="lm",
    )

    return image.resize((WIDTH, HEIGHT), Image.LANCZOS)


def main() -> None:
    out = Path(__file__).resolve().parent
    for theme in (DARK, LIGHT):
        path = out / f"banner-{theme.name}.png"
        draw_banner(theme).save(path, optimize=True)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
