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
    from PySide6 import QtWidgets
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
    w.project.path = tmp / "no_such_dir" / "x.dmap"
    check("저장 실패 시 오류 표시 (앱 계속 동작)", w.save() is False and w.dirty)
    w.dirty = False
    w.close()
    n_fail = checks.count(False)
    print(f"selftest: {len(checks) - n_fail}/{len(checks)} 통과", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run())
