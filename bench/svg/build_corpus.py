# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.23.5",
#   "pillow>=10",
#   "resvg-py>=0.2",
#   "scikit-image==0.25.2",
# ]
# ///
"""
Rebuild the SVG benchmark corpus in ``bench/svg/corpus/``.

The corpus is checked in, so you only need this script to audit where an image
came from or to add new ones. Every image comes from one of three sources:

* ``photo-*``  CC0 / public-domain photographs bundled with scikit-image.
* ``illus-*``  Twemoji graphics (CC-BY-4.0), rasterized with resvg.
* ``synth-*``  Logos, flat illustrations and stress cases drawn by this script
               (CC0-1.0). They are deterministic: same script, same pixels.

Usage (from the repository root)::

    uv run bench/svg/build_corpus.py

The script also regenerates ``corpus/SOURCES.md``.
"""

from __future__ import annotations

import io
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

TWEMOJI_TAG = "v16.0.1"
TWEMOJI_URL = (
    "https://raw.githubusercontent.com/jdecked/twemoji/{tag}/assets/svg/{code}.svg"
)
SKIMAGE_URL = "https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data"

# Supersampling factor for synthetic images, so edges are anti-aliased the way
# a real exported logo would be.
SS = 4


@dataclass
class Entry:
    name: str
    category: str
    license: str
    source: str
    notes: str


ENTRIES: list[Entry] = []


def save(img: Image.Image, name: str, category: str, lic: str, src: str, notes: str):
    path = CORPUS_DIR / name
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    if path.suffix == ".jpg":
        img.save(path, quality=92, optimize=True)
    else:
        img.save(path, optimize=True)
    w, h = img.size
    ENTRIES.append(Entry(name, category, lic, src, f"{w}x{h}. {notes}"))
    print(f"  {name:32s} {w}x{h}  {path.stat().st_size / 1024:7.1f} KiB")


# ---------------------------------------------------------------------------
# Photographs (scikit-image sample data)
# ---------------------------------------------------------------------------

# (function name, output name, license, provenance from the skimage docstring)
PHOTOS = [
    ("astronaut", "photo-astronaut.jpg", "LicenseRef-PublicDomain", "NASA"),
    ("chelsea", "photo-cat.jpg", "CC0-1.0", "Stefan van der Walt"),
    ("coffee", "photo-coffee.jpg", "CC0-1.0", "Rachel Michetti"),
    ("rocket", "photo-rocket.jpg", "LicenseRef-PublicDomain", "SpaceX"),
    ("hubble_deep_field", "photo-hubble.jpg", "LicenseRef-PublicDomain", "NASA"),
    ("retina", "photo-retina.jpg", "CC0-1.0", "Mikael Häggström"),
    ("camera", "photo-camera-gray.png", "CC0-1.0", "Lav Varshney"),
    ("clock", "photo-clock-gray.png", "LicenseRef-PublicDomain", "Stefan van der Walt"),
    ("cell", "photo-cell-gray.png", "CC0-1.0", "Paul Müller et al."),
    ("brick", "photo-brick-texture.png", "CC0-1.0", "CC0Textures"),
    ("grass", "photo-grass-texture.png", "CC0-1.0", "linolafett (DeviantArt)"),
    ("gravel", "photo-gravel-texture.png", "CC0-1.0", "CC0Textures"),
    ("horse", "illus-horse-silhouette.png", "CC0-1.0", "Andreas Preuss (openclipart)"),
]


def build_photos():
    import skimage.data

    for fn, name, lic, author in PHOTOS:
        arr = getattr(skimage.data, fn)()
        if arr.dtype == bool:
            arr = np.where(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        category = "illustration" if name.startswith("illus-") else "photo"
        save(
            img,
            name,
            category,
            lic,
            f"[scikit-image `data.{fn}()`]({SKIMAGE_URL})",
            f"By {author}.",
        )


# ---------------------------------------------------------------------------
# Illustrations (Twemoji)
# ---------------------------------------------------------------------------

TWEMOJI = [
    ("1f994", "illus-hedgehog.png", 512),
    ("1f33b", "illus-sunflower.png", 384),
    ("1f30d", "illus-globe.png", 640),
    ("1f3a8", "illus-palette.png", 256),
    ("1f680", "illus-rocket.png", 768),
]


def build_twemoji():
    import resvg_py

    for code, name, size in TWEMOJI:
        url = TWEMOJI_URL.format(tag=TWEMOJI_TAG, code=code)
        with urllib.request.urlopen(url, timeout=30) as resp:
            svg = resp.read().decode("utf-8")
        png = resvg_py.svg_to_bytes(
            svg_string=svg, width=size, height=size, background="#ffffff"
        )
        img = Image.open(io.BytesIO(bytes(png)))
        save(
            img,
            name,
            "illustration",
            "CC-BY-4.0",
            f"[Twemoji {TWEMOJI_TAG} `{code}.svg`]({url})",
            "Copyright Twitter, Inc and other contributors. Rasterized with resvg.",
        )


# ---------------------------------------------------------------------------
# Synthetic logos / illustrations / stress cases
# ---------------------------------------------------------------------------


def canvas(w: int, h: int, bg) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w * SS, h * SS), bg)
    return img, ImageDraw.Draw(img)


def finish(img: Image.Image) -> Image.Image:
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def star(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def synth_rings():
    """Concentric rings: nested holes, the classic even-odd stress case."""
    w = h = 512
    img, d = canvas(w, h, "#ffffff")
    colors = ["#1b3a6b", "#ffffff", "#e94f37", "#ffffff", "#f6c90e", "#ffffff"]
    c = w * SS / 2
    for i, col in enumerate(colors * 2):
        r = (230 - i * 19) * SS
        if r <= 0:
            break
        d.ellipse([c - r, c - r, c + r, c + r], fill=col)
    return finish(img)


def synth_badge():
    """Flat logo: shield, star and a 'wordmark' made of bars."""
    w = h = 640
    img, d = canvas(w, h, "#f4f1ea")
    s = SS
    shield = [(320, 60), (560, 140), (530, 420), (320, 590), (110, 420), (80, 140)]
    d.polygon([(x * s, y * s) for x, y in shield], fill="#2d6a4f")
    inner = [(320, 95), (520, 162), (495, 405), (320, 548), (145, 405), (120, 162)]
    d.polygon([(x * s, y * s) for x, y in inner], fill="#40916c")
    d.polygon(star(320 * s, 250 * s, 120 * s, 50 * s), fill="#ffd166")
    for i, bw in enumerate([200, 150, 230]):
        y0 = (395 + i * 38) * s
        d.rounded_rectangle(
            [(320 - bw / 2) * s, y0, (320 + bw / 2) * s, y0 + 22 * s],
            radius=11 * s,
            fill="#f4f1ea",
        )
    return finish(img)


def synth_flat_shapes():
    """Random overlapping flat shapes from a small palette (seeded)."""
    w, h = 800, 600
    rng = np.random.default_rng(637)
    palette = [
        "#264653", "#2a9d8f", "#e9c46a", "#f4a261",
        "#e76f51", "#8ab17d", "#ffffff", "#3d348b",
    ]  # fmt: skip
    img, d = canvas(w, h, "#fefae0")
    for _ in range(60):
        col = palette[rng.integers(len(palette))]
        x, y = rng.uniform(0, w) * SS, rng.uniform(0, h) * SS
        r = rng.uniform(15, 110) * SS
        kind = rng.integers(3)
        if kind == 0:
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        elif kind == 1:
            d.rectangle([x - r, y - r * 0.6, x + r, y + r * 0.6], fill=col)
        else:
            a = rng.uniform(0, 2 * math.pi)
            pts = [
                (x + r * math.cos(a + k * 2 * math.pi / 3),
                 y + r * math.sin(a + k * 2 * math.pi / 3))
                for k in range(3)
            ]  # fmt: skip
            d.polygon(pts, fill=col)
    return finish(img)


def synth_repeated_icons():
    """A grid of identical icons: the best case for <use> deduplication."""
    w, h = 768, 512
    img, d = canvas(w, h, "#ffffff")
    s = SS
    for row in range(4):
        for col in range(6):
            cx, cy = (64 + col * 128) * s, (64 + row * 128) * s
            d.ellipse([cx - 48 * s, cy - 48 * s, cx + 48 * s, cy + 48 * s], "#0077b6")
            d.polygon(star(cx, cy, 36 * s, 15 * s), fill="#ffffff")
            d.ellipse([cx - 8 * s, cy - 8 * s, cx + 8 * s, cy + 8 * s], "#ef476f")
    return finish(img)


def synth_gradient():
    """Smooth two-axis gradient: tests banding into many thin regions."""
    w, h = 512, 256
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    r = 255 * x * np.ones_like(y)
    g = 255 * y * np.ones_like(x)
    b = 255 * (1 - x) * (1 - y) + 64 * x * y
    arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


PIXEL_SPRITE = """
................
.....kkkkkk.....
...kkooooookk...
..kooowwoooook..
.koowwkwooooook.
.koowwwwooooook.
kooooooooooooook
kooorrooooorrook
koooooooooooooo.
.kooookkkkoooook
.koooooooooooook
..kooooooooooko.
...kkooooookk...
....gkkkkkkg....
...gg......gg...
................
"""


def synth_pixel_art():
    """Upscaled 16x16 sprite: hard axis-aligned edges, no anti-aliasing."""
    colors = {
        ".": (135, 206, 235),
        "k": (40, 30, 30),
        "o": (230, 126, 34),
        "w": (255, 255, 255),
        "r": (231, 76, 60),
        "g": (39, 174, 96),
    }
    rows = [r for r in PIXEL_SPRITE.strip().splitlines()]
    arr = np.array([[colors[c] for c in r] for r in rows], dtype=np.uint8)
    return Image.fromarray(arr).resize((256, 256), Image.NEAREST)


def synth_tiny_icon():
    """32x32 icon: smallest input, where fixed SVG overhead dominates."""
    w = h = 32
    img, d = canvas(w, h, "#ffffff")
    s = SS
    d.rounded_rectangle([2 * s, 2 * s, 30 * s, 30 * s], radius=7 * s, fill="#6a4c93")
    d.polygon(star(16 * s, 16.5 * s, 11 * s, 4.5 * s), fill="#ffca3a")
    return finish(img)


def synth_wide_banner():
    """Very wide, short banner with sine-wave bands."""
    w, h = 1600, 200
    img, d = canvas(w, h, "#023047")
    bands = ["#219ebc", "#8ecae6", "#ffb703", "#fb8500"]
    for i, col in enumerate(bands):
        pts = [(0, h * SS)]
        for x in range(0, w * SS + 1, 8):
            y = (60 + i * 30 + 20 * math.sin(x / SS / 90 + i)) * SS
            pts.append((x, y))
        pts.append((w * SS, h * SS))
        d.polygon(pts, fill=col)
    return finish(img)


SYNTH = [
    (synth_rings, "synth-logo-rings.png", "logo"),
    (synth_badge, "synth-logo-badge.png", "logo"),
    (synth_tiny_icon, "synth-logo-tiny-icon.png", "logo"),
    (synth_flat_shapes, "synth-flat-shapes.png", "illustration"),
    (synth_repeated_icons, "synth-repeated-icons.png", "illustration"),
    (synth_pixel_art, "synth-pixel-art.png", "illustration"),
    (synth_wide_banner, "synth-wide-banner.png", "illustration"),
    (synth_gradient, "synth-gradient.png", "stress"),
]


def build_synthetic():
    for fn, name, category in SYNTH:
        save(
            fn(),
            name,
            category,
            "CC0-1.0",
            "`bench/svg/build_corpus.py`",
            (fn.__doc__ or "").strip(),
        )


# ---------------------------------------------------------------------------


def write_sources():
    lines = [
        "# SVG benchmark corpus sources",
        "",
        "<!-- Generated by bench/svg/build_corpus.py. Do not edit by hand. -->",
        "",
        "Every image in this directory is permissively licensed. Licenses are",
        "also declared per file in the repository's `REUSE.toml`; license texts",
        "live in `LICENSES/`.",
        "",
    ]
    rows = [["File", "Category", "License", "Source", "Notes"]]
    for e in sorted(ENTRIES, key=lambda e: e.name):
        rows.append([f"`{e.name}`", e.category, e.license, e.source, e.notes])
    # Pad columns the way Prettier does, so `pnpm format:check` passes.
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt(cells):
        padded = (c.ljust(w) for c, w in zip(cells, widths, strict=True))
        return "| " + " | ".join(padded) + " |"

    lines.append(fmt(rows[0]))
    lines.append(fmt(["-" * w for w in widths]))
    lines += [fmt(r) for r in rows[1:]]
    lines.append("")
    (CORPUS_DIR / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print("photos (scikit-image):")
    build_photos()
    print(f"illustrations (Twemoji {TWEMOJI_TAG}):")
    build_twemoji()
    print("synthetic:")
    build_synthetic()
    write_sources()
    total = sum(p.stat().st_size for p in CORPUS_DIR.iterdir() if p.suffix != ".md")
    print(f"{len(ENTRIES)} images, {total / 1024 / 1024:.2f} MiB total")


if __name__ == "__main__":
    main()
