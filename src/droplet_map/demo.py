"""Synthetic batch files: five purification batches, each with its own formation conditions (not
always the same sets), batch values (purification, zeta potential), particles, an in vitro sheet
(FRAP, turbidity) and a cell delivery sheet. Used by the self-test, the tests and
파일 › 가상 예제 열기. Made-up numbers, not experiment data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core.batchfile import (COND_ID, INSIDE_PREFIX, CYTO_PREFIX, ITEM_NAME, RATIO_PREFIX, SHEET_DELIVERY,
                             SHEET_INVITRO, TEMPLATE_NAME, template, write_batch_file)

# batch -> (quality 0..1, purification date, protein mg/mL, column, conditions run)
FULL = [(ph, salt) for ph in (5, 6, 7, 8) for salt in (0, 150, 300)]
BATCHES = {
    "B01": (0.9, "2026-03-02", 4.1, "HisTrap", FULL),
    "B02": (0.8, "2026-04-15", 3.8, "HisTrap", FULL),
    "B03": (0.5, "2026-05-20", 2.9, "HisTrap", [(ph, s) for ph, s in FULL if s != 300]),
    "B04": (0.3, "2026-06-11", 2.2, "IMAC-Co", [(ph, 150) for ph in (5, 6, 7, 8)]),
    "B05": (0.1, "2026-07-08", 1.6, "IMAC-Co", [(ph, s) for ph, s in FULL if ph in (6, 7)]),
}


def demo_particles(q, conds, rng, per_image=80) -> pd.DataFrame:
    """Higher quality = rounder particles that gain more circularity with pH and partition more."""
    rows = []
    for i, (ph, salt) in enumerate(conds):
        for im in range(2):
            n = per_image + int(rng.integers(-15, 15))
            d = rng.lognormal(np.log(0.5 + 0.15 * (ph - 4) + 0.4 * q + salt / 1500), 0.45, n)
            circ = np.clip(rng.normal(0.70 + 0.04 * q * (ph - 5) + 0.15 * q + 0.01 / d, 0.06), 0.1, 1)
            rows.append(pd.DataFrame({
                COND_ID: f"C{i + 1:02d}", "이미지": f"img{im + 1}",
                "particle": np.arange(n), "area_px": np.maximum(1, (d / 0.2) ** 2 * 0.8).astype(int),
                "area_um2": np.pi * (d / 2) ** 2, "eq_diameter_um": d, "circularity": circ,
                "solidity": np.clip(circ + 0.05, 0, 1),
                "aspect_ratio": 1 + rng.exponential(0.35 - 0.2 * q, n),
                "nnd_um": rng.lognormal(1 + salt / 600, 0.3, n),
                "CH1_partition": rng.lognormal(np.log(8 + 25 * q + 3 * (ph - 5)), 0.3, n),
                "x_um": rng.uniform(0, 100, n), "y_um": rng.uniform(0, 100, n)}))
    return pd.concat(rows, ignore_index=True)


def demo_batch_file(name: str, per_image=80, seed=0):
    q, date, conc, col, conds = BATCHES[name]
    rng = np.random.default_rng(seed + list(BATCHES).index(name))
    bf = template()
    bf.set_item(ITEM_NAME, name)
    bf.set_item("정제일", date)
    bf.set_item("컬럼", col)
    bf.set_item("단백질 농도", conc)
    bf.set_item("제타 포텐셜", round(-8 - 14 * q + rng.normal(0, 1), 1))
    cs = bf.conditions
    cs.add_rows(len(conds))
    for i, (ph, salt) in enumerate(conds):
        cs.set_value(i, COND_ID, f"C{i + 1:02d}")
        cs.set_value(i, "pH", ph)
        cs.set_value(i, "NaCl", salt)
        cs.set_value(i, "PEG", 10)
    bf.particles.append_frame(demo_particles(q, conds, rng, per_image))
    # in vitro: FRAP and turbidity, three repeats, on the 150 mM conditions only
    iv = bf.sheet(SHEET_INVITRO)
    rows = []
    for i, (ph, salt) in enumerate(conds):
        if salt != 150:
            continue
        for rep in range(3):
            rows.append({COND_ID: f"C{i + 1:02d}", "반복": rep + 1,
                         "FRAP 회복률": round(30 + 50 * q + 3 * (ph - 5) + rng.normal(0, 3), 1),
                         "FRAP t½": round(40 - 25 * q + rng.normal(0, 2), 1),
                         "탁도": round(0.2 + 0.05 * (ph - 5) + 0.3 * q + rng.normal(0, 0.02), 3)})
    iv.append_frame(pd.DataFrame(rows))
    # cell delivery: particles of the pH 7 / 150 mM condition, 0 and 30 min
    dl = bf.sheet(SHEET_DELIVERY)
    k = next(i for i, c in enumerate(conds) if c[0] == 7 and c[1] in (150, 0))
    rows = []
    for t in (0, 30):
        for rep in range(3):
            inside = 600 * (1 - 0.3 * q * t / 30) * (1 + rng.normal(0, 0.05))
            cyto = 600 * (0.05 + 0.5 * q * t / 30) * (1 + rng.normal(0, 0.05))
            rows.append({COND_ID: f"C{k + 1:02d}", "반복": rep + 1, "시간": t, "항체": 1,
                         CYTO_PREFIX + "항체 신호": round(cyto, 1), INSIDE_PREFIX + "항체 신호": round(inside, 1),
                         "이미지 수": 6})
    dl.append_frame(pd.DataFrame(rows))
    bf.fill_ratios()
    return bf


def write_demo_folder(folder, per_image=80) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in BATCHES:
        write_batch_file(demo_batch_file(name, per_image), folder / f"{name}.xlsx")
    write_batch_file(template(), folder / TEMPLATE_NAME)
    return folder


def demo_project(per_image=80, folder=None):
    """A ready Project read from a folder of synthetic batch files (a temporary folder unless given)."""
    import tempfile
    from .core.project import Project
    folder = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="droplet_demo_")) / "가상 예제"
    write_demo_folder(folder, per_image)
    pr = Project.open_folder(folder)
    pr.batches.references = {"B01"}
    return pr
