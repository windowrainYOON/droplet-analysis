import numpy as np
import pandas as pd
import pytest

from droplet_map.core.analysis import Settings, reliability_weight, run_all
from droplet_map.core.conditions import ConditionDef, ConditionTable
from droplet_map.core.project import Project


def synthetic(n_groups=10, per_image=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ph = 5 + g % 4
        for im in range(2):
            d = rng.lognormal(np.log(0.6 + 0.15 * ph), 0.4, per_image)
            circ = np.clip(rng.normal(0.95 - 0.04 * ph + 0.02 / d, 0.05), 0.1, 1)
            rows.append(pd.DataFrame({
                "group": f"G{g:02d} pH{ph}", "image": f"img{im}", "particle_id": np.arange(per_image),
                "area_px": np.maximum(1, (d / 0.1) ** 2).astype(int), "area_um2": np.pi * (d / 2) ** 2,
                "eq_diameter_um": d, "circularity": circ, "solidity": np.clip(circ + 0.05, 0, 1),
                "aspect_ratio": 1 + rng.exponential(0.2, per_image), "CH1_partition": rng.lognormal(np.log(10 * ph), 0.3, per_image),
                "centroid_x": rng.uniform(0, 100, per_image)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "data.csv"
    synthetic().to_csv(p, index=False)
    pr = Project.from_csv([p])
    t = pr.conditions
    t.add_condition(ConditionDef("pH", "numeric"))
    t.add_condition(ConditionDef("메모", "categorical"))
    for g in pr.groups:
        t.set_value(g, "pH", float(g.split("pH")[1]))
    t.set_value(pr.groups[0], "메모", "A")   # sparse: mostly blank
    t.references = {pr.groups[1]}
    return pr


def test_weight_curve():
    w = reliability_weight(np.array([0.2, 1.2, 10]))
    assert w[0] < 0.01 and abs(w[1] - 0.5) < 1e-9 and w[2] > 0.99


def test_roles_guess(project):
    feats = project.roles.features()
    assert "circularity" in feats and "centroid_x" not in feats and "particle_id" not in feats


def test_blank_conditions_stay_blank(project):
    t = project.conditions
    assert t.recorded_count("메모") == 1
    assert t.numeric("pH").notna().all()
    assert pd.isna(t.value(project.groups[2], "메모"))


def test_run_all_and_roundtrip(project, tmp_path):
    s = Settings(min_groups_assoc=6)
    r = run_all(project.particles, project.roles, project.conditions, s, with_map=False)
    assert r.score is not None and r.score.loc[project.groups[1], "score"] == pytest.approx(100)
    assert r.assoc.strength.loc["pH", "CH1_partition"] > 0.8
    assert bool(r.assoc.hidden["메모"])
    path = tmp_path / "p.dmap"
    project.save(path)
    back = Project.load(path)
    assert back.conditions.to_dict() == project.conditions.to_dict()
    assert len(back.particles) == len(project.particles)


def test_fill_from_group_names():
    from droplet_map.core.conditions import fill_from_group_names, parse_group_name
    assert parse_group_name("FITC-PRD, pH5, 150mM NaCl, 10% PEG") == {"pH": "5", "NaCl": "150", "PEG": "10"}
    t = ConditionTable(["FITC-PRD, pH6, 0mM NaCl, 10% PEG", "FITC-PRD, pH7.5, 450mM NaCl, 10% PEG", "control"])
    assert fill_from_group_names(t, "B260430-1hsU-200-4") == ["pH", "NaCl", "PEG", "Batch"]
    assert t.get("NaCl").unit == "mM" and t.get("PEG").kind == "numeric"
    assert list(t.numeric("pH")[:2]) == [6, 7.5] and t.numeric("NaCl").iloc[0] == 0
    assert t.value("control", "pH") is None and t.value("control", "Batch") == "B260430-1hsU-200-4"
    # typed values are kept; a second run has nothing left to fill
    t.set_value("FITC-PRD, pH6, 0mM NaCl, 10% PEG", "pH", "6.2")
    assert fill_from_group_names(t, "other") == []
    assert t.value("FITC-PRD, pH6, 0mM NaCl, 10% PEG", "pH") == "6.2" and t.value("control", "Batch") == "B260430-1hsU-200-4"
