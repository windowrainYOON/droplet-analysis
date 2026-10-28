import numpy as np
import pandas as pd
import pytest

from droplet_map.core.analysis import Settings, prepare, reliability_weight, run_all
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


def test_feature_weight_changes_map_input_and_cohesion(project):
    from droplet_map.core.analysis import cohesion, map_matrix, prepare
    s = Settings()
    p = prepare(project.particles, project.roles, s)
    j = p.features.index("CH1_partition")
    base = map_matrix(p, s)
    s.feature_weights["CH1_partition"] = 3.0
    assert np.allclose(map_matrix(p, s)[:, j], 3 * base[:, j])
    s.feature_weights["CH1_partition"] = 0.0
    assert np.allclose(map_matrix(p, s)[:, j], 0)
    rng = np.random.default_rng(0)
    lab = np.repeat(["a", "b"], 200)
    tight = np.c_[np.where(lab == "a", 0, 5) + rng.normal(0, .3, 400), rng.normal(0, .3, 400)]
    loose = np.c_[np.where(lab == "a", 0, 1) + rng.normal(0, 1, 400), rng.normal(0, 1, 400)]
    assert cohesion(tight, lab) > 0.8 > cohesion(loose, lab)
    assert cohesion(tight, np.where(lab == "a", "a", None)) is None


def test_sub_map(project):
    from droplet_map.core.analysis import particle_map, prepare
    s = Settings(n_neighbors=10)
    p = prepare(project.particles, project.roles, s)
    idx = np.flatnonzero(p.data["group"].str.endswith("pH5").to_numpy())
    pm = particle_map(p, s, idx=idx, name="pH5")
    assert pm.umap.shape == (len(idx), 2) and len(pm.clusters) == len(idx) and pm.name == "pH5"


def _write(df, path):
    df.to_csv(path, index=False)
    return path


def test_multi_csv_missing_column_duplicates_and_image_names(tmp_path):
    df = synthetic(n_groups=4)
    a = _write(df[df["group"].str.contains("pH5|pH6")], tmp_path / "a.csv")
    b = _write(df[~df["group"].str.contains("pH5|pH6")].drop(columns=["CH1_partition"]), tmp_path / "b.csv")
    pr = Project.from_csv([a, b])
    assert "CH1_partition" not in pr.roles.features() and any("CH1_partition" in n for n in pr.notes)
    r = run_all(pr.particles, pr.roles, pr.conditions, Settings(), with_map=False)
    assert set(r.prepared.data["source_file"]) == {"a.csv", "b.csv"}      # no file silently dropped
    n = len(pr.particles)
    notes = pr.add_csv([a])
    assert len(pr.particles) == n and notes                             # same file not added twice
    (tmp_path / "copy").mkdir()
    import shutil
    shutil.copy(a, tmp_path / "copy" / "renamed.csv")
    assert pr.add_csv([tmp_path / "copy" / "renamed.csv"]) and len(pr.particles) == n   # same content, new name
    other = df[df["group"].str.contains("pH5")].copy(); other["eq_diameter_um"] *= 1.1
    (tmp_path / "exp2").mkdir()
    o = _write(other, tmp_path / "exp2" / "a.csv")                     # same file name, other data
    pr.add_csv([o])
    assert len(pr.particles) == n + len(other) and "exp2/a.csv" in set(pr.particles["source_file"])
    n = len(pr.particles)
    c = _write(df[df["group"].str.contains("pH5")], tmp_path / "c.csv")  # same group + image names
    pr.add_csv([c])
    imgs = pr.particles[pr.particles["source_file"] == "c.csv"]["image"]
    assert imgs.str.startswith("c:").all()
    pr.save(tmp_path / "m.dmap")
    back = Project.load(tmp_path / "m.dmap")
    assert back.add_csv([a]) and len(back.particles) == len(pr.particles)   # duplicate check after reopen


def test_odd_format_file_does_not_switch_off_measurements(tmp_path):
    df = synthetic(n_groups=4)
    a = _write(df, tmp_path / "a.csv")
    ij = pd.DataFrame({"group": "IJ", "image": "x", "Area": np.random.rand(60), "Circ.": np.random.rand(60),
                       "Solidity": np.random.rand(60), "AR": 1 + np.random.rand(60)})
    b = _write(ij, tmp_path / "ij.csv")
    pr = Project.from_csv([a, b])
    assert "circularity" in pr.roles.features() and len(pr.notes) == 1 and "ij.csv" in pr.notes[0]



def test_add_missing_column_file_and_odd_file_together(tmp_path):
    df = synthetic(n_groups=4)
    a = _write(df, tmp_path / "a.csv")
    no_part = df.drop(columns=[c for c in df.columns if "partition" in c]).copy()
    no_part["group"] = "B_" + no_part["group"]
    no_part["circularity"] = no_part["circularity"] * 0.9
    b = _write(no_part, tmp_path / "b.csv")
    ij = pd.DataFrame({"group": "IJ", "image": "x", "Area": np.random.rand(60), "Circ.": np.random.rand(60),
                       "Solidity": np.random.rand(60), "AR": 1 + np.random.rand(60)})
    c = _write(ij, tmp_path / "ij.csv")
    pr = Project.from_csv([a])
    notes = pr.add_csv([b, c])
    feats = pr.roles.features()
    assert not any("partition" in f for f in feats) and "circularity" in feats
    assert not {"Area", "Circ."} & set(feats)
    text = "\n".join(notes)
    assert "b.csv" in text and "ij.csv: 표준 측정값" in text
    prep = prepare(pr.particles, pr.roles, pr.settings)
    assert (prep.data["source_file"] == "b.csv").any() and not (prep.data["source_file"] == "ij.csv").any()

def test_numeric_looking_names_survive_save(tmp_path):
    df = synthetic(n_groups=2)
    df["group"] = np.where(df["group"].str.contains("pH5"), "01", "002")
    p = _write(df, tmp_path / "n.csv")
    pr = Project.from_csv([p])
    assert set(pr.groups) == {"01", "002"}
    pr.conditions.add_condition(ConditionDef("pH", "numeric"))
    pr.conditions.set_value("01", "pH", "5")
    pr.save(tmp_path / "n.dmap")
    back = Project.load(tmp_path / "n.dmap")
    assert set(back.groups) == {"01", "002"} and back.conditions.value("01", "pH") == "5"


def test_failed_save_keeps_previous_file(tmp_path, project):
    path = tmp_path / "keep.dmap"
    project.save(path)
    before = path.read_bytes()
    project.ui = {"bad": object()}        # serialisable via the fallback, so force a real failure:
    project.particles = None
    with pytest.raises(Exception):
        project.save(path)
    assert path.read_bytes() == before and not (tmp_path / "keep.dmap.saving").exists()


def test_classify_imagej_results(tmp_path):
    from droplet_map.core.io import classify_file
    df = pd.DataFrame({" ": range(1, 51), "Area": np.random.rand(50), "Circ.": np.random.rand(50),
                       "Solidity": np.random.rand(50)})
    assert classify_file(_write(df, tmp_path / "Results.csv")) == "particles"
    cond = pd.DataFrame({"group": ["a", "b"], "pH": [5, 6]})
    assert classify_file(_write(cond, tmp_path / "cond.csv")) == "conditions"


def test_map_can_leave_out_measurements_and_condition_values(tmp_path):
    from droplet_map.core.analysis import map_matrix, map_rows, particle_map
    df = synthetic(n_groups=4)
    pr = Project.from_csv([_write(df, tmp_path / "a.csv")])
    pr.conditions.add_condition(ConditionDef("pH", "numeric"))
    for g in pr.groups:
        if "pH5" in g:
            pr.conditions.set_value(g, "pH", "5")
    s = pr.settings
    s.map_skip_features = ["circularity"]
    s.map_skip_values = {"pH": [""]}            # groups without a recorded pH
    p = prepare(pr.particles, pr.roles, s)
    rows = map_rows(p, pr.conditions, s)
    assert set(p.data["group"].iloc[rows]) == {g for g in pr.groups if "pH5" in g}
    X = map_matrix(p, s)
    assert np.all(X[:, p.features.index("circularity")] == 0)
    pm = particle_map(p, s, idx=rows, restandardize=False)
    assert len(pm.clusters) == len(rows)
    pr.save(tmp_path / "m.dmap")
    back = Project.load(tmp_path / "m.dmap")
    assert back.settings.map_skip_features == ["circularity"] and back.settings.map_skip_values == {"pH": [""]}


def _two_batches(tmp_path):
    """Batch A and batch B with the same pH conditions; batch B's particles partition 1.5x more."""
    a = synthetic(n_groups=4)
    b = synthetic(n_groups=4, seed=1)
    b["group"] = "B " + b["group"]
    b["CH1_partition"] *= 1.5
    pa, pb = _write(a, tmp_path / "a.csv"), _write(b, tmp_path / "b.csv")
    pr = Project.from_csv([pa], batch="A")
    pr.add_csv([pb], batch="B")
    pr.conditions.add_condition(ConditionDef("pH", "numeric"))
    for g in pr.groups:
        pr.conditions.set_value(g, "pH", g.split("pH")[1])
    return pr


def test_batches_membership_view_and_save(tmp_path):
    pr = _two_batches(tmp_path)
    assert pr.batches.names() == ["A", "B"]
    assert {pr.batch_of(g) for g in pr.groups if g.startswith("B ")} == {"B"}
    pr.batches.table.add_condition(ConditionDef("정제일", "categorical"))
    pr.batches.table.set_value("B", "정제일", "2026-10-01")
    v = pr.view
    assert v.names()[:2] == ["배치", "배치: 정제일"] and "pH" in v.names()
    g = next(g for g in pr.groups if g.startswith("B "))
    assert v.value(g, "배치: 정제일") == "2026-10-01" and v.value(g, "배치") == "B"
    pr.save(tmp_path / "b.dmap")
    back = Project.load(tmp_path / "b.dmap")
    assert back.batches.of == pr.batches.of and back.batches.table.value("B", "정제일") == "2026-10-01"


def test_old_batch_condition_becomes_batch(tmp_path, project):
    project.conditions.add_condition(ConditionDef("정제 배치", "categorical"))
    for g in project.groups:
        project.conditions.set_value(g, "정제 배치", "X1")
    project.save(tmp_path / "old.dmap")
    import zipfile
    with zipfile.ZipFile(tmp_path / "old.dmap") as z, zipfile.ZipFile(tmp_path / "v1.dmap", "w") as out:
        for n in z.namelist():
            if n != "batches.json":
                out.writestr(n, z.read(n))
    back = Project.load(tmp_path / "v1.dmap")
    assert back.batches.names() == ["X1"] and "정제 배치" not in back.conditions.names() and back.notes


def test_condition_effect_is_judged_within_batches(tmp_path):
    from droplet_map.core.analysis import association
    pr = _two_batches(tmp_path)
    p = prepare(pr.particles, pr.roles, pr.settings)
    med = p.data.groupby("group")["CH1_partition"].median().to_frame()
    view = pr.view
    a = association(view, med, min_groups=4, strata=pr.batches.series(pr.groups))
    assert a.level["pH"] == "group" and a.strength.at["pH", "CH1_partition"] > 0.9
    assert a.level["배치"] == "batch" and a.hidden["배치"]          # only 2 batches


def test_batch_comparison_and_reference_batch_score(tmp_path):
    from droplet_map.core.analysis import group_distance
    from droplet_map.core.batch_compare import (batch_distance, batch_feature_delta, batch_reference_score,
                                                 batch_summary, matched_groups)
    pr = _two_batches(tmp_path)
    p = prepare(pr.particles, pr.roles, pr.settings)
    m = matched_groups(pr.groups, pr.conditions, pr.batches)
    assert list(m.columns) == ["A", "B"] and len(m) == 4
    D, N = batch_distance(group_distance(p, pr.settings), m)
    assert N.at["A", "B"] == 4 and D.at["A", "B"] > 0
    delta = batch_feature_delta(p, m, "A", "B")
    assert delta.at["평균", "CH1_partition"] > 0.2
    sc = batch_reference_score(p, pr.settings, pr.groups, pr.conditions, pr.batches, "A")
    summ = batch_summary(sc)
    assert summ.at["A", "평균 score"] == pytest.approx(100) and summ.at["B", "평균 score"] < 100


def test_delivery_summary(tmp_path):
    from droplet_map.core.delivery import fill_delivery_from_names
    from droplet_map.core.io import classify_file
    rows = []
    for t in (0, 10):
        g = f"1uM AHP-r568, 5uM FITC-PRD, 10% PEG, {t}min"
        for region, base in (("particle", 500), ("cyto - particle", 50)):
            for par in ("CH1_Mean", "CH2_Mean", "area_um2", "particle_threshold"):
                rows.append({"group": g, "region": region, "n_images": 3, "parameter": par, "n": 10,
                             "mean": base * (1 + t / 10), "sd": 1.0, "min": 0.0, "max": 1.0})
    f = tmp_path / "deliv.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    assert classify_file(f) == "delivery"
    pr = Project.from_csv([_write(synthetic(n_groups=2), tmp_path / "a.csv")], batch="A")
    pr.add_delivery([f], batch="A")
    d = pr.delivery
    assert len(d.groups()) == 2 and d.channels() == ["CH1_Mean", "CH2_Mean"]
    assert d.ratio()["CH1_Mean"].round(3).tolist() == [0.1, 0.1]
    assert fill_delivery_from_names(d.conditions) == ["AHP-r568", "FITC-PRD", "PEG", "시간"]
    d.uses[d.groups()[0]] = pr.groups[0]
    pr.save(tmp_path / "d.dmap")
    back = Project.load(tmp_path / "d.dmap")
    assert back.delivery.uses == d.uses and back.delivery.conditions.value(d.groups()[1], "시간") == "10"
    assert all(back.batch_of(g) == "A" for g in back.all_groups())
