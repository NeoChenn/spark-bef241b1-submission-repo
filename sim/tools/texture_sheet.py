"""Make one labeled preview image of every texture.

    C:\\isaacsim\\python.bat sim\\tools\\texture_sheet.py

Writes sim/output/texture_sheet.png.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIM_DIR = Path(__file__).resolve().parents[1]
TEXTURES = SIM_DIR / "assets" / "third_party" / "textures"
GENERATED = SIM_DIR / "assets" / "generated"
OUT = SIM_DIR / "output" / "texture_sheet.png"

TILE = 300
LABEL = 62
GAP = 14
COLUMNS = 4
USED_BY = {
    "metal_plate": "Full rig: floor relief",
    "box_profile_metal_sheet": "Full rig: wall relief",
    "concrete_floor_damaged_01": "Full rig: leg relief",
    "metal_grate_rusty": "Deck scene",
    "rusty_metal_02": "Deck scene",
    "rusty_painted_metal": "Deck scene",
    "green_metal_rust": "Deck scene",
    "blue_metal_plate": "Deck scene",
    "rust_coarse_01": "Deck scene",
}

entries = [(folder.name, folder / "diff.jpg", USED_BY.get(folder.name, "Not used yet")) for folder in sorted(TEXTURES.iterdir())]
entries.append(("sea_normal", GENERATED / "sea_normal.png", "Full rig: sea ripples"))
entries.append(("gauge_dial", GENERATED / "gauge_dial.png", "Full rig: gauges"))

rows = (len(entries) + COLUMNS - 1) // COLUMNS
sheet = Image.new(
    "RGB",
    (COLUMNS * (TILE + GAP) + GAP, rows * (TILE + LABEL + GAP) + GAP),
    (24, 26, 29),
)
draw = ImageDraw.Draw(sheet)
try:
    title = ImageFont.truetype("arialbd.ttf", 18)
    detail = ImageFont.truetype("arial.ttf", 15)
except OSError:
    title = detail = ImageFont.load_default()

for index, (name, path, used) in enumerate(entries):
    x = GAP + (index % COLUMNS) * (TILE + GAP)
    y = GAP + (index // COLUMNS) * (TILE + LABEL + GAP)
    sheet.paste(Image.open(path).convert("RGB").resize((TILE, TILE)), (x, y))
    draw.text((x, y + TILE + 8), name, fill=(240, 240, 240), font=title)
    draw.text((x, y + TILE + 33), used, fill=(170, 176, 182), font=detail)

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT)
print(f"SHEET_WRITTEN {OUT} entries={len(entries)}")
