"""Draw the app icons: a rounded square with a recorded line rising into a dashed forecast.

  assets/app.ico    blue, Antibiotic Resistance Forecast
  assets/entry.ico  lavender, Antibiotic Resistance Forecast Studio

Run once (needs Pillow); the icons are kept in the repository.
  python packaging/make_icon.py          both icons
  python packaging/make_icon.py entry    only the Studio's
"""

import sys

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
WHITE = (255, 255, 255)
ICONS = {  # name: (face, shadow)
    "app": ((42, 120, 214), (31, 99, 184)),
    "entry": ((110, 86, 207), (91, 69, 184)),
}


def draw(size: int, face: tuple, shadow: tuple) -> Image.Image:
    scale = 4  # draw large, then downsample for smooth edges
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = round(s * 0.04)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=round(s * 0.22), fill=shadow)
    d.rounded_rectangle([pad, pad, s - pad, s - pad - round(s * 0.05)], radius=round(s * 0.22), fill=face)

    pts = [(0.18, 0.70), (0.36, 0.62), (0.52, 0.66), (0.66, 0.42)]  # recorded
    ahead = [(0.66, 0.42), (0.84, 0.21)]  # forecast
    xy = [(x * s, y * s) for x, y in pts]
    width = max(2, round(s * (0.075 if size <= 24 else 0.06)))
    d.line(xy, fill=WHITE, width=width, joint="curve")
    # dashed forecast segment
    (x0, y0), (x1, y1) = [(x * s, y * s) for x, y in ahead]
    for a, b in ([(0.25, 0.75)] if size <= 24 else [(0.2, 0.42), (0.58, 0.8)]):  # clear of the end dots
        d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
               fill=WHITE, width=width)
    r = width * 0.95
    for x, y in xy:
        d.ellipse([x - r, y - r, x + r, y + r], fill=WHITE)
    r2 = width * 1.15
    d.ellipse([x1 - r2, y1 - r2, x1 + r2, y1 + r2], fill=face, outline=WHITE, width=max(2, round(width * 0.7)))
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    for name in sys.argv[1:] or ICONS:
        out = ROOT / "assets" / f"{name}.ico"
        out.parent.mkdir(exist_ok=True)
        images = [draw(n, *ICONS[name]) for n in SIZES]
        images[-1].save(out, format="ICO", sizes=[(n, n) for n in SIZES], append_images=images[:-1])
        images[-1].save(ROOT / "assets" / f"{name}-256.png")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
