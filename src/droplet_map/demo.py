"""Synthetic multi-batch data: several purification batches, each with particle formation groups
(not always the same condition sets), batch-level values (purification conditions, zeta
potential) and cell delivery summaries. Used by the self-test, the tests and
파일 › 가상 예제 프로젝트."""
from __future__ import annotations

import numpy as np
import pandas as pd

# batch -> (quality 0..1, purification date, protein mg/mL, column, conditions run)
FULL = [(ph, salt) for ph in (5, 6, 7, 8) for salt in (0, 150, 300)]
BATCHES = {
    "B01": (0.9, "2026-03-02", 4.1, "HisTrap", FULL),
    "B02": (0.8, "2026-04-15", 3.8, "HisTrap", FULL),
    "B03": (0.5, "2026-05-20", 2.9, "HisTrap", [(ph, s) for ph, s in FULL if s != 300]),
    "B04": (0.3, "2026-06-11", 2.2, "IMAC-Co", [(ph, 150) for ph in (5, 6, 7, 8)]),
    "B05": (0.1, "2026-07-08", 1.6, "IMAC-Co", [(ph, s) for ph, s in FULL if ph in (6, 7)]),
}


def demo_particles(per_image=80, seed=0) -> pd.DataFrame:
    """Higher quality = rounder particles that gain more circularity with pH and partition more."""
    rng = np.random.default_rng(seed)
    rows = []
    for b, (q, _date, _conc, _col, conds) in BATCHES.items():
        for ph, salt in conds:
            for im in range(2):
                n = per_image + int(rng.integers(-15, 15))
                d = rng.lognormal(np.log(0.5 + 0.15 * (ph - 4) + 0.4 * q + salt / 1500), 0.45, n)
                circ = np.clip(rng.normal(0.70 + 0.04 * q * (ph - 5) + 0.15 * q + 0.01 / d, 0.06), 0.1, 1)
                rows.append(pd.DataFrame({
                    "group": f"{b}, pH{ph}, {salt}mM NaCl, 10% PEG", "image": f"img{im}",
                    "particle": np.arange(n), "area_px": np.maximum(1, (d / 0.2) ** 2 * 0.8).astype(int),
                    "area_um2": np.pi * (d / 2) ** 2, "eq_diameter_um": d, "circularity": circ,
                    "solidity": np.clip(circ + 0.05, 0, 1),
                    "aspect_ratio": 1 + rng.exponential(0.35 - 0.2 * q, n),
                    "nnd_um": rng.lognormal(1 + salt / 600, 0.3, n),
                    "CH1_partition": rng.lognormal(np.log(8 + 25 * q + 3 * (ph - 5)), 0.3, n),
                    "x_um": rng.uniform(0, 100, n), "y_um": rng.uniform(0, 100, n)}))
    return pd.concat(rows, ignore_index=True)


def demo_batch_table() -> pd.DataFrame:
    """One row per batch: purification conditions and zeta potential (in vitro)."""
    rng = np.random.default_rng(1)
    rows = []
    for b, (q, date, conc, col, _c) in BATCHES.items():
        rows.append({"batch": b, "정제일": date, "단백질 농도 (mg/mL)": conc, "컬럼": col,
                     "제타 포텐셜 (mV)": round(-8 - 14 * q + rng.normal(0, 1), 1)})
    return pd.DataFrame(rows)


def demo_delivery() -> pd.DataFrame:
    """Delivery summary in the image-analysis format: antibody (CH2) and particle (CH1) channels,
    inside particles and in the cytoplasm, at 0 and 30 min. Better batches release more antibody
    into the cytoplasm."""
    rng = np.random.default_rng(2)
    rows = []
    for b, (q, *_rest) in BATCHES.items():
        for t in (0, 30):
            g = f"{b}, 1uM antibody, 5uM particle, {t}min"
            for region, base in (("particle", 1.0), ("cyto - particle", 0.0)):
                for ch, lvl in (("CH1_Mean", 900), ("CH2_Mean", 600)):
                    if region == "particle":
                        m = lvl * (1 - 0.3 * q * t / 30)
                    else:
                        m = lvl * (0.05 + (0.5 * q if ch == "CH2_Mean" else 0.1) * t / 30)
                    m *= 1 + rng.normal(0, 0.05)
                    rows.append({"group": g, "region": region, "n_images": 6, "parameter": ch, "n": 6,
                                 "mean": m, "sd": 0.15 * m, "min": 0.6 * m, "max": 1.5 * m})
    return pd.DataFrame(rows)


def demo_project(per_image=80):
    """A ready Project: batches assigned, batch table filled, conditions filled from group names,
    delivery groups with conditions."""
    import tempfile
    from pathlib import Path
    from .core.conditions import ConditionDef, fill_from_group_names
    from .core.delivery import fill_delivery_from_names
    from .core.project import Project

    tmp = Path(tempfile.mkdtemp(prefix="droplet_demo_"))
    parts = demo_particles(per_image)
    pr = None
    for b in BATCHES:
        path = tmp / f"{b}_particles.csv"
        parts[parts["group"].str.startswith(b + ",")].to_csv(path, index=False)
        if pr is None:
            pr = Project.from_csv([path], batch=b)
        else:
            pr.add_csv([path], batch=b)
    fill_from_group_names(pr.conditions)
    bt = demo_batch_table().set_index("batch")
    cats = {"정제일": "정제 조건", "단백질 농도 (mg/mL)": "정제 조건", "컬럼": "정제 조건", "제타 포텐셜 (mV)": "in vitro 관측"}
    for col in bt.columns:
        name, unit = (col.split(" (")[0], col.split(" (")[1].rstrip(")")) if " (" in col else (col, "")
        kind = "categorical" if col in ("정제일", "컬럼") else "numeric"
        pr.batches.table.add_condition(ConditionDef(name, kind, unit))
        pr.batches.set_category(name, cats[col])
        for b, v in bt[col].items():
            pr.batches.table.set_value(b, name, v)
    dpath = tmp / "delivery_summary.csv"
    demo_delivery().to_csv(dpath, index=False)
    pr.add_delivery([dpath])
    for g in pr.delivery.groups():
        pr.batches.assign([g], g.split(",")[0])
    fill_delivery_from_names(pr.delivery.conditions)
    pr.batches.references = {"B01"}
    return pr
