"""Draw the app icon: a cluster map of droplets on a macOS-style rounded square.
Writes resources/droplet_map.png (1024) and droplet_map.icns."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4096                      # supersampled canvas
OUT = Path(__file__).resolve().parents[1] / "src" / "droplet_map" / "resources"


def rounded_mask(size, inset, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([inset, inset, size - inset, size - inset], radius=radius, fill=255)
    return m


def gradient(size, top, bottom):
    t = np.linspace(0, 1, size)[:, None, None]
    g = (1 - t) * np.array(top, float) + t * np.array(bottom, float)
    return Image.fromarray(np.repeat(g, size, axis=1).astype(np.uint8), "RGB")


def droplet(layer, cx, cy, r, rgb):
    d = ImageDraw.Draw(layer)
    # soft glow
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([cx - r * 1.6, cy - r * 1.6, cx + r * 1.6, cy + r * 1.6], fill=rgb + (70,))
    layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(r * 0.5)))
    # body with radial shading
    n = 24
    for i in range(n):
        f = 1 - i / n
        rr = r * f
        k = 0.55 + 0.45 * (i / n)
        col = tuple(int(min(255, c * k + 255 * (1 - k) * (i / n) * 0.6)) for c in rgb)
        ox, oy = -r * 0.25 * (i / n), -r * 0.25 * (i / n)
        d.ellipse([cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr], fill=col + (255,))
    # specular highlight
    hl = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse([cx - r * 0.55, cy - r * 0.62, cx - r * 0.12, cy - r * 0.25], fill=(255, 255, 255, 190))
    layer.alpha_composite(hl.filter(ImageFilter.GaussianBlur(r * 0.08)))


def main():
    inset, radius = int(S * 0.098), int(S * 0.18)
    bg = gradient(S, (20, 34, 66), (12, 88, 112)).convert("RGBA")
    # faint map grid
    grid = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid)
    for x in range(inset + S // 20, S - inset, S // 10):
        gd.line([(x, inset), (x, S - inset)], fill=(255, 255, 255, 22), width=8)
        gd.line([(inset, x), (S - inset, x)], fill=(255, 255, 255, 22), width=8)
    bg.alpha_composite(grid)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    rng = np.random.default_rng(7)
    clusters = [((0.36, 0.40), (88, 200, 255), 0.075, 9),   # cyan
                ((0.66, 0.36), (255, 176, 72), 0.060, 7),   # amber
                ((0.52, 0.68), (130, 232, 160), 0.068, 8)]  # green
    drops = []
    for (fx, fy), rgb, spread, n in clusters:
        for _ in range(n):
            r = S * rng.uniform(0.022, 0.058)
            x = S * (fx + rng.normal(0, spread)); y = S * (fy + rng.normal(0, spread))
            drops.append((y, x, r, rgb))
    for y, x, r, rgb in sorted(drops):
        droplet(layer, x, y, r, rgb)
    # one large "reference" droplet with a ring
    cx, cy, r = S * 0.36, S * 0.40, S * 0.085
    droplet(layer, cx, cy, r, (88, 200, 255))
    ImageDraw.Draw(layer).ellipse([cx - r * 1.45, cy - r * 1.45, cx + r * 1.45, cy + r * 1.45], outline=(255, 255, 255, 220), width=int(S * 0.012))
    bg.alpha_composite(layer)
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([inset, inset + S * 0.012, S - inset, S - inset + S * 0.012], radius=radius, fill=(0, 0, 0, 110))
    icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(S * 0.012)))
    icon.paste(bg, (0, 0), rounded_mask(S, inset, radius))
    big = icon.resize((1024, 1024), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    big.save(OUT / "droplet_map.png")
    big.save(OUT / "droplet_map.icns", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
