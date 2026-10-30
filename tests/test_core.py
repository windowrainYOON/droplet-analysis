import numpy as np
import pandas as pd
import pytest

from droplet_map.core.analysis import Settings, prepare, reliability_weight, run_all
from droplet_map.core.conditions import ConditionDef, ConditionTable
from droplet_map.core.batchfile import (COND_ID, ITEM_NAME, read_batch_file, template, write_batch_file)
from droplet_map.core.importing import add_particles
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


def make_folder(tmp_path, batches: dict, name="data", ph=True, items=None) -> "Path":
    """One batch file per batch; 조건 ID = the CSV group name; pH from the name when asked."""
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    for b, df in batches.items():
        bf = template()
        bf.set_item(ITEM_NAME, b)
        for k, v in (items or {}).get(b, {}).items():
            bf.set_item(k, v)
        df = df.copy()
        df["source_file"] = f"{b}.csv"
        groups = list(pd.unique(df["group"]))
        add_particles(bf, df, {g: g for g in groups})
        cs = bf.conditions
        if ph:
            for r in range(len(cs.data)):
                cid = cs.data.at[r, COND_ID]
                if "pH" in cid:
                    cs.set_value(r, "pH", cid.split("pH")[1])
        write_batch_file(bf, folder / f"{b}.xlsx")
    return folder


@pytest.fixture
def project(tmp_path):
    pr = Project.open_folder(make_folder(tmp_path, {"A": synthetic()}))
    t = pr.conditions
    t.add_condition(ConditionDef("메모", "categorical"))
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
    assert back.conditions.references == project.conditions.references and back.folder == project.folder
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


def test_batch_file_roundtrip_check_and_copy(tmp_path):
    bf = template()
    bf.set_item(ITEM_NAME, "B1")
    bf.set_item("제타 포텐셜", "-21.5")
    cs = bf.conditions
    cs.add_rows(3)
    for i, ph in enumerate(("5", "6", "7")):
        cs.set_value(i, "pH", ph)
    assert bf.fill_condition_ids() == 3 and bf.condition_ids() == ["C01", "C02", "C03"]
    cs.set_value(0, "NaCl", 150)
    assert bf.condition_label("C01") == "pH 5 · NaCl 150 mM"
    iv = bf.sheet("in vitro 실험")
    iv.append_frame(pd.DataFrame({COND_ID: ["C01", "C09", None], "반복": [1, 1, 2], "FRAP 회복률": [50, "많음", 40]}))
    msgs = [i.message for i in bf.check()]
    assert any("C09" in m for m in msgs) and any("많음" in m for m in msgs) and any("조건 ID가 빈" in m for m in msgs)
    iv.add_column(__import__("droplet_map.core.batchfile", fromlist=["Column"]).Column("점도", "cP"))
    bf.add_sheet("DLS")
    path = write_batch_file(bf, tmp_path / "B1")
    back, notes = read_batch_file(path)
    assert not notes and back.name == "B1" and [s.name for s in back.sheets][-1] == "DLS"
    assert back.sheet("in vitro 실험").column("점도").unit == "cP" and back.items()[-1]["value"] == -21.5
    assert back.sheet("DLS").kind == "조건별 측정" and back.conditions.column("NaCl").unit == "mM"
    new = back.copy_for_new_batch("B2")
    assert new.name == "B2" and len(new.sheet("in vitro 실험").data) == 0 and len(new.conditions.data) == 3


def test_import_duplicates_odd_files_and_missing_columns(tmp_path):
    from droplet_map.core.importing import read_particle_files, suggest_conditions, conditions_from_names
    df = synthetic(n_groups=4)
    a = _write(df[df["group"].str.contains("pH5|pH6")], tmp_path / "a.csv")
    frame, notes = read_particle_files([a, a])
    assert notes and set(frame["source_file"]) == {"a.csv"}
    bf = template()
    bf.set_item(ITEM_NAME, "A")
    groups = list(pd.unique(frame["group"]))
    assert suggest_conditions(bf, groups) == {g: None for g in groups}
    ids = conditions_from_names(bf, groups)              # only when asked: values from the names
    assert list(bf.conditions.text("pH")) == ["5", "6"] and suggest_conditions(bf, groups) == ids
    n0 = add_particles(bf, frame, ids)
    assert "입자" in n0[0] and len(bf.particles.data) == len(frame)
    again = add_particles(bf, frame, ids)                 # the same data a second time is skipped
    assert len(bf.particles.data) == len(frame) and "건너뛰" in again[0]
    ij = pd.DataFrame({"group": "IJ", "image": "x", "Area": np.random.rand(60), "Circ.": np.random.rand(60),
                       "Solidity": np.random.rand(60), "AR": 1 + np.random.rand(60), "source_file": "ij.csv"})
    with pytest.raises(ValueError):
        add_particles(bf, ij, {"IJ": "C01"})
    # a batch whose particle sheet lacks a measurement switches it off for the analysis (reported)
    other = synthetic(n_groups=4, seed=2).drop(columns=["CH1_partition"])
    folder = make_folder(tmp_path, {"A": df, "B": other})
    pr = Project.open_folder(folder)
    assert "CH1_partition" not in pr.roles.features() and any("CH1_partition" in n for n in pr.notes)


def test_condition_ids_stay_text_and_reload_picks_up_changes(tmp_path):
    df = synthetic(n_groups=2)
    df["group"] = np.where(df["group"].str.contains("pH5"), "01", "002")
    folder = make_folder(tmp_path, {"A": df}, ph=False)
    pr = Project.open_folder(folder)
    assert set(pr.groups) == {"A · 01", "A · 002"}
    bf, _ = read_batch_file(folder / "A.xlsx")
    r = list(bf.conditions.text(COND_ID)).index("01")
    bf.conditions.set_value(r, "pH", "5")
    bf.set_item("정제일", "2026-10-01")
    import os, time
    write_batch_file(bf, folder / "A.xlsx")
    os.utime(folder / "A.xlsx", (time.time() + 5, time.time() + 5))
    pr.reload()
    assert pr.conditions.value("A · 01", "pH") == "5" and pr.batches.table.value("A", "정제일") == "2026-10-01"


def test_failed_save_keeps_previous_file(tmp_path, project):
    path = tmp_path / "keep.dmap"
    project.save(path)
    before = path.read_bytes()
    project.roles = None                   # force a real failure
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
    df["group"] = df["group"].str.replace(r" pH([678])", r" x\1", regex=True)   # pH recorded for pH5 groups only
    pr = Project.open_folder(make_folder(tmp_path, {"A": df}))
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
    b["CH1_partition"] *= 1.5
    return Project.open_folder(make_folder(tmp_path, {"A": a, "B": b}, items={"B": {"정제일": "2026-10-01"}}))


def test_batches_membership_view_and_save(tmp_path):
    pr = _two_batches(tmp_path)
    assert pr.batches.names() == ["A", "B"]
    assert {pr.batch_of(g) for g in pr.groups if g.startswith("B ")} == {"B"}
    v = pr.view
    assert v.names()[:2] == ["배치", "배치: 정제일"] and "pH" in v.names()
    g = next(g for g in pr.groups if g.startswith("B "))
    assert v.value(g, "배치: 정제일") == "2026-10-01" and v.value(g, "배치") == "B"
    pr.batches.references = {"A"}
    pr.save(tmp_path / "b.dmap")
    back = Project.load(tmp_path / "b.dmap")
    assert back.batches.of == pr.batches.of and back.batches.references == {"A"}


def test_old_projects_open_as_batch_files(tmp_path):
    import json, zipfile
    df = synthetic(n_groups=4)
    groups = list(pd.unique(df["group"]))
    cond = ConditionTable(groups, [ConditionDef("정제 배치", "categorical"), ConditionDef("NaCl (mM)", "numeric")])
    for g in groups:
        cond.set_value(g, "정제 배치", "X1")
        cond.set_value(g, "NaCl (mM)", "150")
    from droplet_map.core.io import guess_roles
    with zipfile.ZipFile(tmp_path / "v1.dmap", "w") as z:          # format 1: batch was a group condition
        z.writestr("particles.csv", df.assign(source_file="a.csv").to_csv(index=False))
        z.writestr("roles.json", json.dumps(guess_roles(df).to_dict()))
        z.writestr("conditions.json", cond.to_json())
        z.writestr("settings.json", json.dumps(Settings().to_dict()))
    back = Project.load(tmp_path / "v1.dmap")
    assert back.batches.names() == ["X1"] and "정제 배치" not in back.conditions.names() and back.notes
    assert back.conditions.get("NaCl").unit == "mM" and len(back.particles) == len(df) and back.folder is None
    with pytest.raises(ValueError):
        back.save(tmp_path / "x.dmap")                          # data must go to a folder first
    files = back.export_batch_files(tmp_path / "out")
    back.save(tmp_path / "x.dmap")
    again = Project.load(tmp_path / "x.dmap")
    assert len(files) == 1 and len(again.particles) == len(df) and (tmp_path / "out" / "_템플릿.xlsx").exists()


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


def test_delivery_summary_goes_to_a_sheet(tmp_path):
    from droplet_map.core.importing import add_delivery, read_delivery_files
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
    d, _ = read_delivery_files([f])
    assert d.channels() == ["CH1_Mean", "CH2_Mean"]
    folder = make_folder(tmp_path, {"A": synthetic(n_groups=2)})
    bf, _ = read_batch_file(folder / "A.xlsx")
    cid = bf.condition_ids()[0]
    notes = add_delivery(bf, "세포 전달", d, {g: cid for g in d.groups()}, sub_from_names=True)
    s = bf.sheet("세포 전달")
    assert "2행" in notes[0] and s.column("시간").role == "하위 조건"
    assert list(s.numeric("세포질/입자 CH1_Mean").round(3)) == [0.1, 0.1]
    write_batch_file(bf)
    pr = Project.open_folder(folder)
    ms = pr.measure_set("세포 전달")
    assert ms is not None and "시간" in ms.subs and len(ms.measures) == 2
    assert all(ms.formation_of[g] == f"A · {cid}" for g in ms.measures.index)
    desc = pr.batch_descriptors()
    assert "세포 전달" in desc.categories()


# ---------------------------------------------------------------- batch map
def test_batch_descriptors_trends_matched_and_direct():
    from droplet_map.demo import demo_project
    from droplet_map.core.batch_map import KIND_TREND, KIND_MATCHED, KIND_DIRECT, batch_distance_map, batch_scores
    pr = demo_project(per_image=40)
    d = pr.batch_descriptors()
    kinds = set(d.meta["kind"])
    assert {KIND_TREND, KIND_MATCHED, KIND_DIRECT} <= kinds
    # B04 ran pH only (NaCl fixed): it still gets a pH trend, but no NaCl trend
    assert pd.notna(d.values.at["B04", "circularity / pH"]) and pd.isna(d.values.at["B04", "circularity / NaCl"])
    # better batches gain more circularity per pH unit
    assert d.values.at["B01", "circularity / pH"] > d.values.at["B05", "circularity / pH"]
    # dates typed as text count as days
    assert d.meta.at["정제일", "type"] == "numeric" and d.values.at["B05", "정제일"] > 100
    assert d.meta.at["제타 포텐셜", "category"] == "배치 성질"
    # every per-condition sheet is its own category; FRAP gets a pH trend, delivery a time trend
    assert {"입자 형태", "in vitro 실험", "세포 전달"} <= set(d.categories())
    assert pd.notna(d.values.at["B01", "FRAP 회복률 / pH"]) and pd.notna(d.values.at["B01", "세포질/입자 항체 신호 / 시간"])
    bm = batch_distance_map(d)
    D = bm.distance
    assert D.at["B01", "B02"] < D.at["B01", "B05"]
    assert bm.coords is not None and bm.explained[0] > 50
    ax = bm.coords["축 1"]
    assert abs(pd.Series(ax).rank().corr(pd.Series(range(5), index=ax.index), method="spearman")) > 0.8
    s = batch_scores(bm, "B01")
    assert s["B01"] == 100 and s["B02"] > s["B05"]
    # category weight 0 and skipped parts change the map
    only_purif = batch_distance_map(d, weights={c: 0 for c in d.categories() if c != "정제 조건"})
    assert set(only_purif.by_category) == {"정제 조건"}
    no_trend = batch_distance_map(d, skip={"입자 형태 · 경향"})
    assert not any(it.startswith("circularity / ") for it in no_trend.used)


def test_new_category_from_a_batch_item_and_one_batch(tmp_path):
    from droplet_map.core.batch_map import batch_distance_map
    folder = make_folder(tmp_path, {"A": synthetic(n_groups=4)})
    bf, _ = read_batch_file(folder / "A.xlsx")
    bf.set_item("응집 시작", 12, "일", "안정성")
    write_batch_file(bf)
    q = Project.open_folder(folder)
    assert "안정성" in q.batches.categories and q.batches.category("응집 시작") == "안정성"
    bm = batch_distance_map(q.batch_descriptors())
    assert bm.coords is None and list(bm.distance.index) == ["A"]
