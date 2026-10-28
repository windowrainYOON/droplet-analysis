"""Headless self-test: drives the whole app on synthetic data. Run `python -m droplet_map --selftest`
(or `"/Applications/Droplet Map.app/Contents/MacOS/Droplet Map" --selftest`). Exit code 0 = passed."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic(n_groups=8, per_image=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ph = 5 + g % 4
        salt = 150 * (g // 4)
        for im in range(2):
            d = rng.lognormal(np.log(0.5 + 0.2 * ph), 0.45, per_image)
            circ = np.clip(rng.normal(0.98 - 0.05 * (ph - 5) + 0.01 / d, 0.05), 0.1, 1)
            rows.append(pd.DataFrame({
                "group": f"TEST, pH{ph}, {salt}mM NaCl, 10% PEG", "image": f"img{im}",
                "particle": np.arange(per_image), "area_px": np.maximum(1, (d / 0.2) ** 2 * 0.8).astype(int),
                "area_um2": np.pi * (d / 2) ** 2, "eq_diameter_um": d, "circularity": circ,
                "solidity": np.clip(circ + 0.03, 0, 1), "aspect_ratio": 1 + rng.exponential(0.1 + 0.05 * (ph - 5), per_image),
                "nnd_um": rng.lognormal(1, 0.3, per_image), "CH1_partition": rng.lognormal(np.log(10 * ph), 0.3, per_image),
                "x_um": rng.uniform(0, 100, per_image), "y_um": rng.uniform(0, 100, per_image)}))
    return pd.concat(rows, ignore_index=True)


def run() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from .gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"[{'OK' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)

    def wait(w, timeout=300):
        t0 = time.time()
        while w._thread is not None and time.time() - t0 < timeout:
            app.processEvents(); time.sleep(0.02)
        app.processEvents()
        return w._thread is None

    tmp = Path(tempfile.mkdtemp(prefix="droplet_selftest_"))
    csv = tmp / "particles.csv"
    synthetic().to_csv(csv, index=False)
    w = MainWindow(); w.dirty = False
    w.ask_batch = lambda what, n, project=None: "A"
    w.show()
    w.import_files([str(csv)])
    check("입자 CSV 자동 인식", w.project is not None and len(w.project.groups) == 8)
    cond = tmp / "conditions.csv"
    t = pd.DataFrame({"group": w.project.groups})
    t["pH"] = t["group"].str.extract(r"pH(\d+)")[0]
    t["NaCl (mM)"] = t["group"].str.extract(r"(\d+)mM")[0]
    t["메모"] = ["a"] + [""] * (len(t) - 1)
    t.to_csv(cond, index=False, encoding="utf-8-sig")
    w.import_files([str(cond)])
    tbl = w.project.conditions
    check("조건 CSV 자동 인식", tbl.names()[:2] == ["pH", "NaCl (mM)"] and tbl.recorded_count("메모") == 1)
    tbl.references = {w.project.groups[2]}
    w.project.settings.min_groups_assoc = 6

    w.run_analysis(); ok = wait(w)
    r = w.result
    check("분석 실행", ok and r is not None and r.pmap is not None, f"군집 {len(set(r.pmap.clusters)) if r else 0}개")
    check("기준 score", r.score is not None and abs(r.score.loc[w.project.groups[2], "score"] - 100) < 1e-6)
    check("조건 연관 (pH ↔ partition)", r.assoc is not None and r.assoc.strength.loc["pH", "CH1_partition"] > 0.7)

    mp = w.map_page
    for key in ("particle", "image", "group", "cond"):
        mp.unit.setCurrentIndex(mp.unit.findData(key))
        app.processEvents()
    check("점 단위 4가지 그리기", True)
    mp.color_by.setCurrentIndex(mp.color_by.findText("조건: pH")); app.processEvents()
    check("응집도 표시", "응집도" in mp.cohesion.text(), mp.cohesion.text().split("<b>")[-1].split("</b>")[0])
    mp.color_by.setCurrentIndex(mp.color_by.findText("조건: pH (단계별)"))
    mp.shape_by.setCurrentIndex(mp.shape_by.findText("조건: NaCl (mM)")); app.processEvents()
    check("색 = 조건 단계, 모양 = 두 번째 조건", mp.color_by.currentText().endswith("(단계별)") and mp.shape_by.currentIndex() > 0)
    mp.shape_by.setCurrentIndex(0)
    mp.unit.setCurrentIndex(0)

    w.groups_panel.cond.setCurrentText("pH"); w.groups_panel.value.setCurrentText("5")
    w.groups_panel._only_value(); app.processEvents()
    check("조건으로 그룹 거르기", len(w.groups_panel.visible_groups()) == 2)
    w.groups_panel._check_all(True)

    mp.cluster_list.item(0).setSelected(True)
    mp._submap_clusters(); ok = wait(w)
    check("계층 지도 (군집 → 하위 지도)", ok and len(mp.levels) == 2, mp.current().name if mp.levels else "")
    mp.go_up()
    check("상위 지도로", len(mp.levels) == 1)

    before = r.score["score"].copy()
    w.weights.rows["CH1_partition"].setValue(0.0)
    w.weights._timer.stop(); w._weights_changed(); ok = wait(w)
    after = w.result.score["score"]
    check("가중치 → score 반영", ok and not np.allclose(before.reindex(after.index), after))
    mp.auto_rerun.setChecked(False)
    w.rerun_map(); ok = wait(w)
    check("가중치 → 지도 다시 계산", ok and w.result.pmap is not None)

    # leave a measurement and a condition value out of the map only
    n_score = len(w.result.score)
    for i in range(mp.map_feats.count()):
        if mp.map_feats.item(i).data(QtCore.Qt.UserRole) == "circularity":
            mp.map_feats.item(i).setCheckState(QtCore.Qt.Unchecked)
    mp.skip_cond.setCurrentText("pH"); app.processEvents()
    for i in range(mp.skip_vals.count()):
        if mp.skip_vals.item(i).data(QtCore.Qt.UserRole) == "5":
            mp.skip_vals.item(i).setCheckState(QtCore.Qt.Unchecked)
    w.rerun_map(); ok = wait(w)
    pm = w.result.pmap
    on_map = set(w.result.prepared.data["group"].iloc[pm.idx]) if pm is not None and pm.idx is not None else set()
    ph5 = {g for g in w.project.groups if "pH5" in g}
    check("지도에서 측정값·조건 값 빼기", ok and on_map and not on_map & ph5 and len(w.result.score) == n_score
          and w.project.settings.map_skip_features == ["circularity"], f"지도 그룹 {len(on_map)}개")
    mp._include_all(); w.rerun_map(); ok = wait(w)
    check("모두 넣기", ok and w.result.pmap.idx is None)

    out = tmp / "export"
    files = w.export_to(out, {k: True for k in ("particles", "groups", "score", "assoc", "distance",
                                                "composition", "conditions", "settings", "figures")})
    check("통합 저장", (out / "particles_with_clusters.csv").exists() and len(files) >= 10, f"{len(files)}개 파일")
    proj = tmp / "test.dmap"
    w.project.path = proj; w.save()
    from .core.project import Project
    back = Project.load(proj)
    check("프로젝트 저장·열기", back.conditions.to_dict() == w.project.conditions.to_dict() and bool(back.ui) and bool(back.log))
    # several CSVs: a file without one measurement, the same file twice; failed save
    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    QtWidgets.QMessageBox.exec = lambda self: 0
    extra = synthetic(n_groups=2, seed=1).drop(columns=["CH1_partition"])
    extra["group"] = "EXTRA, " + extra["group"]
    b = tmp / "extra.csv"; extra.to_csv(b, index=False)
    n0 = len(w.project.particles)
    w.import_files([str(b), str(csv)])
    n1 = len(w.project.particles)
    check("여러 CSV 추가 (같은 파일은 건너뜀)", n1 == n0 + len(extra), f"{n0:,} → {n1:,}")
    w.run_analysis(); ok = wait(w)
    kept = set(w.result.prepared.data["source_file"]) if ok and w.result else set()
    check("측정값이 빠진 파일의 입자도 분석에 포함", {"particles.csv", "extra.csv"} <= kept)
    # batches: a second batch with the same formation conditions, its particles partition more
    bb = synthetic(seed=2)
    bb["group"] = bb["group"].str.replace("TEST,", "TEST-B,", regex=False)
    bb["CH1_partition"] *= 1.6
    bcsv = tmp / "batch_b.csv"; bb.to_csv(bcsv, index=False)
    w.ask_batch = lambda what, n, project=None: "B"
    w.import_files([str(bcsv)])
    bgroups = [g for g in w.project.groups if g.startswith("TEST-B")]
    tb = pd.DataFrame({"group": bgroups})
    tb["pH"] = tb["group"].str.extract(r"pH(\d+)")[0]
    tb["NaCl (mM)"] = tb["group"].str.extract(r"(\d+)mM")[0]
    bc = tmp / "conditions_b.csv"; tb.to_csv(bc, index=False, encoding="utf-8-sig")
    w.import_files([str(bc)])
    pr = w.project
    check("배치 지정 (불러올 때)", pr.batches.names()[:2] == ["A", "B"] and {pr.batch_of(g) for g in bgroups} == {"B"}
          and pr.view.names()[0] == "배치", ", ".join(pr.batches.names()))
    w.cond_page.refresh_all()
    check("조건 탭: 배치 표 + 그룹 표", w.cond_page.models["batch"].rowCount() == 2
          and w.cond_page.models["formation"].rowCount() == len(pr.groups))
    tree_batches = w.groups_panel.tree.topLevelItemCount()
    nvis = len(w.groups_panel.visible_groups())
    check("왼쪽 목록: 배치 › 그룹", tree_batches == 2 and nvis == len(pr.groups), f"배치 {tree_batches}개 · 표시 {nvis}/{len(pr.groups)}")
    w.run_analysis(); ok = wait(w)
    a = w.result.assoc if ok else None
    check("조건 효과는 배치 안에서 계산", a is not None and a.level["pH"] == "group" and a.level["배치"] == "batch"
          and a.strength.loc["pH"].max() > 0.7)
    bp = w.batch_page
    check("배치 비교: 같은 조건 짝", len(bp.matched) == 7,   # one A group also has a 메모 value
          f"{len(bp.matched)}가지")
    bp.ref.setCurrentText("A"); bp.compute_score(); app.processEvents()
    from .core.batch_compare import batch_summary
    summ = batch_summary(bp.score) if bp.score is not None else None
    check("기준 배치 score", summ is not None and summ.at["A", "평균 score"] > 99 and summ.at["B", "평균 score"] < summ.at["A", "평균 score"],
          "" if summ is None else f"A {summ.at['A', '평균 score']:.0f} · B {summ.at['B', '평균 score']:.0f}")
    mp.batch_pick.setCurrentText("B"); mp._submap_batch(); ok = wait(w)
    on = set(w.result.prepared.data["group"].iloc[mp.current().idx]) if ok and mp.current().idx is not None else set()
    check("배치 안 지도", ok and len(mp.levels) == 2 and on == set(bgroups), mp.current().name)
    mp.go_to(0)
    mp.batch_mode.setCurrentIndex(0); mp._show_batch_maps(); app.processEvents()
    check("배치별 지도 (같은 좌표)", len(mp.batch_canvas.fig.axes) == 2)
    mp.batch_mode.setCurrentIndex(1); mp._show_batch_maps(); ok = wait(w)
    check("배치별 지도 (따로 계산)", ok and mp.batch_maps is not None and set(mp.batch_maps) == {"A", "B"}
          and len(mp.batch_canvas.fig.axes) == 2)
    # cell delivery: summary CSV, linked to the formation group whose particles were used
    rows = []
    for t in (0, 10):
        g = f"1uM AHP-r568, 5uM FITC-PRD, 10% PEG, {t}min"
        for region, base in (("particle", 500.0), ("cyto - particle", 50.0)):
            for par in ("area_um2", "particle_threshold", "CH1_Mean", "CH2_Mean"):
                rows.append({"group": g, "region": region, "n_images": 3, "parameter": par, "n": 10,
                             "mean": base * (1 + t / 10), "sd": base / 10, "min": 0.0, "max": base * 3})
    dcsv = tmp / "particle_intensity_summary_group.csv"; pd.DataFrame(rows).to_csv(dcsv, index=False)
    w.ask_batch = lambda what, n, project=None: "A"
    w.import_files([str(dcsv)])
    d = w.project.delivery
    check("세포 전달 CSV 자동 인식", len(d.groups()) == 2 and all(w.project.batch_of(g) == "A" for g in d.groups()))
    w.cond_page.group_tabs.setCurrentIndex(1)
    w.cond_page.fill_from_names()
    check("세포 전달 조건 (이름에서)", d.conditions.value(d.groups()[1], "시간") == "10")
    d.uses[d.groups()[0]] = w.project.groups[2]
    w._conditions_changed(); app.processEvents()
    dp = w.delivery_page
    dp.refresh_link()
    linked = dp.link.rowCount() == 2 and any("입자 기준 score" == dp.link.horizontalHeaderItem(j).text()
                                             for j in range(dp.link.columnCount()))
    check("세포 전달 그림 + 사용한 입자와 잇기", bool(dp.plot.fig.axes) and linked)
    proj2 = tmp / "batches.dmap"
    w.project.path = proj2; w.save()
    back = Project.load(proj2)
    check("배치·세포 전달 저장·열기", back.batches.of == w.project.batches.of and back.delivery.uses == d.uses
          and back.delivery.conditions.names() == d.conditions.names())
    w._mark_dirty()
    w.project.path = tmp / "no_such_dir" / "x.dmap"
    check("저장 실패 시 오류 표시 (앱 계속 동작)", w.save() is False and w.dirty)
    w.dirty = False
    w.close()
    n_fail = checks.count(False)
    print(f"selftest: {len(checks) - n_fail}/{len(checks)} 통과", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run())
