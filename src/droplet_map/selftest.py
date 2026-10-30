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
    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    QtWidgets.QMessageBox.exec = lambda self: 0
    from .gui import batch_note
    from .core.project import Project
    from .core.batchfile import TEMPLATE_NAME, read_batch_file, template, write_batch_file

    def auto_import(dlg):
        """The CSV import dialog, answered: delivery rows -> the first condition, particle groups ->
        an existing condition with the same values, else a new one from the name."""
        if dlg.sheet is not None:
            first = dlg.bf.condition_ids()[0]
            for cb in dlg.combos:
                cb.setEditText(first)
            dlg.subs.setChecked(True)
        else:
            dlg.suggest()
            dlg.make_from_names()
        return 1
    batch_note.ImportDialog.exec = auto_import

    def settle(w, ms=3000):
        """Let the folder watcher and the reload timer run."""
        t0 = time.time()
        while time.time() - t0 < ms / 1000:
            app.processEvents(); time.sleep(0.02)
            if not w.reload_timer.isActive() and time.time() - t0 > 0.4:
                break
        app.processEvents()

    csv = tmp / "particles.csv"
    synthetic().to_csv(csv, index=False)
    folder = tmp / "배치 노트"
    folder.mkdir()
    write_batch_file(template(), folder / TEMPLATE_NAME)
    w = MainWindow(); w.dirty = False
    w.show()
    w.open_folder(folder)                       # no batch files yet: the batch note opens
    note = w.note_window
    check("빈 배치 폴더 → 배치 노트 열림", note is not None and note.isVisible() and w.project is not None)
    a = note.create_batch("A")
    check("새 배치 (템플릿에서)", a is not None and (folder / "A.xlsx").exists()
          and [s.name for s in a.bf.sheets][:3] == ["배치 정보", "형성 조건", "입자 형태"])
    note.import_csvs([str(csv)])
    check("입자 CSV → 입자 시트 (그룹 이름에서 조건 만들기, 누를 때만)", len(a.bf.particles.data) == len(synthetic())
          and len(a.bf.condition_ids()) == 8 and a.dirty)
    note.save(); settle(w)
    pr = w.project
    check("저장 → Droplet Map이 다시 읽음", len(pr.groups) == 8 and len(pr.particles) == len(synthetic()),
          f"조건 {len(pr.groups)}개 · 입자 {len(pr.particles):,}개")
    g2 = pr.groups[2]
    pr.conditions.references = {g2}
    pr.settings.min_groups_assoc = 6

    w.run_analysis(); ok = wait(w)
    r = w.result
    check("분석 실행", ok and r is not None and r.pmap is not None, f"군집 {len(set(r.pmap.clusters)) if r else 0}개")
    check("기준 score", r.score is not None and abs(r.score.loc[g2, "score"] - 100) < 1e-6)
    check("조건 연관 (pH ↔ partition)", r.assoc is not None and r.assoc.strength.loc["pH", "CH1_partition"] > 0.7)

    mp = w.map_page
    for key in ("particle", "image", "group", "cond"):
        mp.unit.setCurrentIndex(mp.unit.findData(key))
        app.processEvents()
    check("점 단위 4가지 그리기", True)
    mp.color_by.setCurrentIndex(mp.color_by.findText("조건: pH")); app.processEvents()
    check("응집도 표시", "응집도" in mp.cohesion.text(), mp.cohesion.text().split("<b>")[-1].split("</b>")[0])
    mp.color_by.setCurrentIndex(mp.color_by.findText("조건: pH (단계별)"))
    mp.shape_by.setCurrentIndex(mp.shape_by.findText("조건: NaCl")); app.processEvents()
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
    ph5 = {g for g in w.project.groups if w.project.conditions.value(g, "pH") in ("5", 5)}
    check("지도에서 측정값·조건 값 빼기", ok and on_map and len(ph5) == 2 and not on_map & ph5 and len(w.result.score) == n_score
          and w.project.settings.map_skip_features == ["circularity"], f"지도 그룹 {len(on_map)}개")
    mp._include_all(); w.rerun_map(); ok = wait(w)
    check("모두 넣기", ok and w.result.pmap.idx is None)

    out = tmp / "export"
    files = w.export_to(out, {k: True for k in ("particles", "groups", "score", "assoc", "distance",
                                                "composition", "conditions", "settings", "figures")})
    check("통합 저장", (out / "particles_with_clusters.csv").exists() and len(files) >= 10, f"{len(files)}개 파일")
    proj = tmp / "test.dmap"
    w.project.path = proj; w.save()
    back = Project.load(proj)
    check("프로젝트 저장·열기 (폴더 위치 + 설정)", back.folder == folder and back.conditions.references == {g2}
          and back.conditions.to_dict() == w.project.conditions.to_dict() and bool(back.ui) and bool(back.log))

    # a second CSV without one measurement, plus the same file again
    extra = synthetic(n_groups=2, seed=1).drop(columns=["CH1_partition"])
    extra["group"] = "EXTRA, " + extra["group"]
    b = tmp / "extra.csv"; extra.to_csv(b, index=False)
    n0 = len(a.bf.particles.data)
    note.import_csvs([str(b), str(csv)])
    n1 = len(a.bf.particles.data)
    check("여러 CSV 넣기 (같은 데이터는 건너뜀)", n1 == n0 + len(extra), f"{n0:,} → {n1:,}")
    note.save(); settle(w)
    check("입자 데이터가 바뀌면 분석 결과는 버림", w.result is None and len(w.project.particles) == n1)
    w.run_analysis(); ok = wait(w)
    kept = {s.split(": ")[-1] for s in w.result.prepared.data["source_file"]} if ok and w.result else set()
    check("측정값이 빠진 CSV의 입자도 분석에 포함", {"particles.csv", "extra.csv"} <= kept, ", ".join(sorted(kept)))

    # batch B: a copy of A's layout and conditions; its particles partition more
    bb = synthetic(seed=2)
    bb["group"] = bb["group"].str.replace("TEST,", "TEST-B,", regex=False)
    bb["CH1_partition"] *= 1.6
    bcsv = tmp / "batch_b.csv"; bb.to_csv(bcsv, index=False)
    n_cond = len(a.bf.condition_ids())        # the EXTRA groups have the same values as existing conditions
    bn = note.create_batch("B", folder / "A.xlsx")
    check("새 배치 (이전 배치 복사: 형성 조건 유지, 측정값 비움)", bn is not None and n_cond == 8 and len(bn.bf.condition_ids()) == n_cond
          and len(bn.bf.particles.data) == 0)
    note.import_csvs([str(bcsv)])
    check("CSV 그룹 → 같은 값의 조건 찾기", len(bn.bf.condition_ids()) == n_cond and len(bn.bf.particles.data) == len(bb))
    note.save(); settle(w)
    pr = w.project
    bgroups = [g for g in pr.groups if pr.batch_of(g) == "B" and g in set(pr.particles["group"])]
    check("배치 = 배치 파일", pr.batches.names()[:2] == ["A", "B"] and len(bgroups) == 8 and pr.view.names()[0] == "배치",
          ", ".join(pr.batches.names()))
    check("배치 표 (보기 전용)", w.batch_table_page.table.rowCount() == 2)
    tree_batches = w.groups_panel.tree.topLevelItemCount()
    nvis = len(w.groups_panel.visible_groups())
    check("왼쪽 목록: 배치 › 형성 조건", tree_batches == 2 and nvis == len(pr.groups), f"배치 {tree_batches}개 · 표시 {nvis}/{len(pr.groups)}")
    w.run_analysis(); ok = wait(w)
    asc = w.result.assoc if ok else None
    check("조건 효과는 배치 안에서 계산", asc is not None and asc.level["pH"] == "group" and asc.level["배치"] == "batch"
          and asc.strength.loc["pH"].max() > 0.7)
    bp = w.batch_page
    check("배치 비교: 같은 조건 짝", len(bp.matched) == 8, f"{len(bp.matched)}가지")
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

    # cell delivery summary -> the 세포 전달 sheet of batch A, linked to the condition whose particles were used
    rows = []
    for t in (0, 10):
        g = f"1uM AHP-r568, 5uM FITC-PRD, 10% PEG, {t}min"
        for region, base in (("particle", 500.0), ("cyto - particle", 50.0)):
            for par in ("area_um2", "particle_threshold", "CH1_Mean", "CH2_Mean"):
                rows.append({"group": g, "region": region, "n_images": 3, "parameter": par, "n": 10,
                             "mean": base * (1 + t / 10), "sd": base / 10, "min": 0.0, "max": base * 3})
    dcsv = tmp / "particle_intensity_summary_group.csv"; pd.DataFrame(rows).to_csv(dcsv, index=False)
    note.files.setCurrentRow(0); app.processEvents()
    a = note.cur
    note.import_csvs([str(dcsv)])
    dl = a.bf.sheet("세포 전달")
    check("세포 전달 CSV → 세포 전달 시트 (시간은 하위 조건)", len(dl.data) == 2 and dl.has("시간")
          and sorted(dl.numeric("시간").dropna()) == [0, 10], f"{len(dl.data)}행")
    note.save(); settle(w)
    keep = w.result
    check("입자 데이터가 같으면 분석 결과 유지", keep is not None and w.result is keep)
    w._set_tab_open(w.measure_page, True)
    dp = w.measure_page
    dp.sheet.setCurrentText("세포 전달"); app.processEvents()
    dp.xaxis.setCurrentText("시간"); app.processEvents()
    dp.refresh_link()
    linked = dp.link.rowCount() == 2 and any(dp.link.horizontalHeaderItem(j).text() == "입자 기준 score"
                                             for j in range(dp.link.columnCount()))
    check("측정 시트 그림 + 그 조건의 입자와 나란히", bool(dp.plot.fig.axes) and linked)

    # batch note editing: a cell, undo, a new column; then a change made outside (Excel)
    note.tabs.setCurrentIndex(0)
    m = note.model()
    ix = m.index(0, 1)
    m.setData(ix, "A-정제-1")
    edited = a.bf.item("배치 이름") == "A-정제-1"
    note.do_undo()
    check("배치 노트: 칸 고치기 · 실행 취소", edited and a.bf.item("배치 이름") == "A")
    note.tabs.setCurrentIndex(1)
    from .core.batchfile import Column
    note.before_edit(); note.sheet().add_column(Column("완충액", "", "조건")); note.after_edit()
    note.tabs.setCurrentIndex(1)
    note.view().setCurrentIndex(note.model().index(0, note.sheet().names().index("완충액")))
    note.paste_cells("HEPES\nHEPES\nTris")
    check("배치 노트: 열 추가 · 붙여넣기", list(note.sheet().data["완충액"].iloc[:3]) == ["HEPES", "HEPES", "Tris"])
    note.save(); settle(w)
    check("형성 조건 추가 → 분석 조건에 반영", "완충액" in w.project.conditions.names())
    bf, _ = read_batch_file(folder / "B.xlsx")
    bf.set_item("제타 포텐셜", -21.5, "mV", "배치 성질")
    write_batch_file(bf, folder / "B.xlsx")
    settle(w, 6000)
    v = w.project.batches.table.value("B", "제타 포텐셜") if "제타 포텐셜" in w.project.batches.table.names() else None
    check("엑셀에서 고친 파일 자동으로 다시 읽기", v is not None and float(v) == -21.5, f"B 제타 포텐셜 = {v}")

    proj2 = tmp / "batches.dmap"
    w.project.path = proj2; w.save()
    back = Project.load(proj2)
    check("배치 폴더 프로젝트 저장·열기", back.batches.names() == w.project.batches.names()
          and [m.name for m in back.measures] == [m.name for m in w.project.measures])
    moved = tmp / "옮긴 폴더"
    folder.rename(moved)
    try:
        Project.load(proj2)
        missing = False
    except FileNotFoundError:
        missing = True
    back = Project.load(proj2, folder=moved)
    check("폴더를 옮기면 찾아서 열기", missing and len(back.groups) == len(w.project.groups))
    w._mark_dirty()
    w.project.path = tmp / "no_such_dir" / "x.dmap"
    check("저장 실패 시 오류 표시 (앱 계속 동작)", w.save() is False and w.dirty)
    w.dirty = False
    note.close()
    batch_level(w, check, wait, app, tmp, settle)
    w.dirty = False
    w.close()
    n_fail = checks.count(False)
    print(f"selftest: {len(checks) - n_fail}/{len(checks)} 통과", flush=True)
    return 1 if n_fail else 0


def batch_level(w, check, wait, app, tmp, settle):
    """The batch map is the main view; the particle map opens for one batch."""
    from .core.project import Project
    from .core.batchfile import read_batch_file, write_batch_file
    w.open_demo(); app.processEvents()
    bmp = w.batch_map_page
    bm = bmp.bm
    ok = bm is not None and bm.coords is not None and len(bm.batches) == 5
    check("가상 예제: 배치 지도 (배치 5개, PCoA)", ok and bm.distance.at["B01", "B02"] < bm.distance.at["B01", "B05"],
          "" if not ok else f"B01–B02 {bm.distance.at['B01', 'B02']:.2f} · B01–B05 {bm.distance.at['B01', 'B05']:.2f}")
    vis = [w.center.isTabVisible(w.pages.index(p)) for p in (bmp, w.batch_table_page, w.batch_page, w.map_page)]
    check("메인 흐름 탭 열림, 입자 수준 탭 닫힘", vis == [True, True, True, False])
    d = bmp.desc
    kinds = set(d.meta["kind"]) if d is not None else set()
    check("배치 항목: 경향 · 같은 조건 · 직접 입력", {"경향", "같은 조건", "직접 입력"} <= kinds
          and w.batch_table_page.derived.rowCount() > 50, f"항목 {0 if d is None else len(d.meta)}개")
    check("경향: 배치마다 다른 조건이어도 계산 (B04는 pH만)", d is not None and pd.notna(d.values.at["B04", "circularity / pH"])
          and d.values.at["B01", "circularity / pH"] > d.values.at["B05", "circularity / pH"])
    cats = set(d.meta["category"]) if d is not None else set()
    check("범주 = 배치 정보의 구분 + 시트", {"정제 조건", "배치 성질", "입자 형태", "in vitro 실험", "세포 전달"} <= cats,
          ", ".join(sorted(cats)))
    check("in vitro 시트 (FRAP)에서 계산한 항목", any("FRAP 회복률" in str(i) for i in d.meta.index))
    for c in ("입자 형태", "in vitro 실험", "세포 전달"):
        bmp.weight_spins[c].setValue(0)
    app.processEvents()
    only = set(bmp.bm.by_category)
    for c in ("입자 형태", "in vitro 실험", "세포 전달"):
        bmp.weight_spins[c].setValue(1)
    app.processEvents()
    check("범주 무게 0 = 범주 빼기", only == {"정제 조건", "배치 성질"} and len(bmp.bm.by_category) == 5)
    n_used = len(bmp.bm.used)
    bmp.part_checks["입자 형태 · 같은 조건"].setChecked(False); app.processEvents()
    fewer = len(bmp.bm.used)
    bmp.part_checks["입자 형태 · 같은 조건"].setChecked(True); app.processEvents()
    check("항목 묶음 끄기", fewer < n_used and len(bmp.bm.used) == n_used, f"{n_used} → {fewer}")
    kc = w.batch_table_page.kind_checks
    check("계산 항목 켜기/끄기: 시트마다", {"입자 형태:경향", "in vitro 실험:같은 조건", "세포 전달:경향"} <= set(kc)
          and not kc["입자 형태:대표값"].isChecked() and kc["in vitro 실험:대표값"].isChecked())
    folder = w.project.folder
    for i, name in enumerate(["B01", "B02", "B03", "B04", "B05"]):
        bf, _ = read_batch_file(folder / f"{name}.xlsx")
        bf.set_item("세포 생존율", 95 - 8 * i, "%", "세포 독성")
        write_batch_file(bf, folder / f"{name}.xlsx")
    settle(w, 6000)
    check("배치 파일에 범주·항목 추가 → 배치 지도에 반영", "세포 독성" in bmp.bm.by_category and "세포 생존율" in bmp.desc.values.columns)
    bmp.ref.setCurrentText("B01"); app.processEvents()
    sc = bmp.score_frame()
    check("기준 배치 score (배치 지도)", sc is not None and sc.at["B01", "score"] == 100 and sc.at["B02", "score"] > sc.at["B05", "score"],
          "" if sc is None else f"B02 {sc.at['B02', 'score']:.0f} · B05 {sc.at['B05', 'score']:.0f}")
    w.open_batch_compare("B01", "B05"); app.processEvents()
    t = w.batch_page.items
    check("배치 비교: 항목 차이", w.center.currentWidget() is w.batch_page.view and len(t) > 20 and bool(w.batch_page.item_canvas.fig.axes))
    w.open_batch_particles("B03")       # runs the particle analysis first, then the batch's sub-map
    t0 = time.time()
    while (w._thread is not None or len(w.map_page.levels) < 2) and time.time() - t0 < 300:
        wait(w); app.processEvents(); time.sleep(0.05)
    ok = w._thread is None
    mp = w.map_page
    cur = mp.current() if ok else None
    on = set(w.result.prepared.data["group"].iloc[cur.idx]) if cur is not None and cur.idx is not None else set()
    check("배치 → 입자 지도 (입자 수준으로)", cur is not None and cur.name == "배치 B03" and on and all(g.startswith("B03") for g in on)
          and w.center.isTabVisible(w.pages.index(mp)), "" if cur is None else cur.name)
    path = tmp / "demo.dmap"
    w.project.path = path; w.save()
    back = Project.load(path)
    check("기준 배치·지도 설정 저장·열기", back.batches.references == {"B01"} and "closed_tabs" in back.ui
          and back.settings.bm_weights == w.project.settings.bm_weights)
    out = tmp / "export_demo"
    files = w.export_to(out, {"batch_map": True, "figures": True, "measures": True})
    names = {p.name for p in files}
    check("통합 저장: 배치 지도 표·그림 · 측정 시트 요약", {"batch_items.csv", "batch_distance.csv", "batch_map_coords.csv"} <= names
          and any(n.startswith("batchmap_canvas") for n in names) and any(n.startswith("sheet_") for n in names),
          f"{len(files)}개 파일")


if __name__ == "__main__":
    sys.exit(run())
