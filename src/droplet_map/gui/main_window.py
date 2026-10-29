"""Main window: left batch › group list · center views · right settings tabs, one menu per feature,
auto-detecting import, work log and background workers.

Main flow: 배치 지도 · 배치 표 · 배치 비교 (batch level). Data entry: 데이터 · 그룹 조건. Particle level
(입자 지도, 조건 비교, 기준 score, 세포 전달) opens from the 입자 수준 menu or from a batch on the map."""
from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.analysis import map_rows, particle_map, reference_score, run_all
from ..core.batches import UNASSIGNED
from ..core.io import classify_file
from ..core.project import Project
from .common import app_icon, save_figure_dialog, save_frame_dialog, set_group_names, warn
from .batch_map_page import BatchMapPage
from .batch_page import BatchPage
from .batch_table_page import BatchTablePage
from .compare_page import ComparePage
from .conditions_page import ConditionsPage
from .data_page import DataPage
from .delivery_page import DeliveryPage
from .export_page import ExportDialog, export_folder_name, write_exports
from .map_page import MapPage
from .score_page import ScorePage
from .side_panels import GroupPanel, WeightsPanel

MAX_RECENT = 8
# numba (UMAP) compiles on first use and overflows QThread's 512 KB default stack on macOS,
# which kills the app without a Python traceback.
WORKER_STACK = 64 * 1024 * 1024


class Task(QtCore.QObject):
    """Runs fn(progress) in a worker thread."""
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(self.progress.emit))
        except Exception as e:  # shown to the user, with the traceback for bug reports
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.result = None
        self.dirty = False
        self._thread = None
        self._used = None              # (d50, k, min_px, roles) the current result was computed with
        self.settings = QtCore.QSettings("droplet-map", "Droplet Map")
        self.setWindowIcon(app_icon())
        self.resize(1500, 920)
        self.setAcceptDrops(True)

        self.batch_map_page = BatchMapPage(self)
        self.batch_table_page = BatchTablePage(self)
        self.batch_page = BatchPage(self)
        self.data_page = DataPage(self)
        self.cond_page = ConditionsPage(self)
        self.map_page = MapPage(self)
        self.compare_page = ComparePage(self)
        self.score_page = ScorePage(self)
        self.delivery_page = DeliveryPage(self)
        self.pages = [self.batch_map_page, self.batch_table_page, self.batch_page, self.cond_page, self.data_page,
                      self.map_page, self.compare_page, self.score_page, self.delivery_page]
        # particle level: closed until asked for (입자 수준 menu, or a batch on the batch map)
        self.particle_pages = [self.map_page, self.compare_page, self.score_page, self.delivery_page]

        self.groups_panel = GroupPanel()
        self.weights = WeightsPanel()
        self.center = QtWidgets.QTabWidget()
        self.center.setDocumentMode(True)
        self.center.setTabsClosable(True)
        self.center.tabCloseRequested.connect(lambda i: self._set_tab_open(self.pages[i], False))
        self.right_stack = QtWidgets.QStackedWidget()
        for p in self.pages:
            self.center.addTab(p.view, p.title)
            self.right_stack.addWidget(p.panel)
        self.right = QtWidgets.QTabWidget()
        self.right.addTab(self.right_stack, "설정")
        self.right.addTab(self.weights, "측정값 가중치")
        self.right.setMinimumWidth(320)
        self.groups_panel.setMinimumWidth(180)

        self.split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.split.addWidget(self.groups_panel)
        self.split.addWidget(self.center)
        self.split.addWidget(self.right)
        self.split.setStretchFactor(0, 0); self.split.setStretchFactor(1, 1); self.split.setStretchFactor(2, 0)
        self.split.setCollapsible(1, False)
        sizes = self.settings.value("split_sizes")
        self.split.setSizes([int(x) for x in sizes] if sizes else [250, 820, 430])
        self.split.splitterMoved.connect(lambda *_: self.settings.setValue("split_sizes", self.split.sizes()))
        self.setCentralWidget(self.split)
        self.center.currentChanged.connect(self._center_changed)

        # signals between panes
        self.data_page.changed.connect(self._data_changed)
        self.cond_page.changed.connect(self._conditions_changed)
        self.groups_panel.visible_changed.connect(self._visible_changed)
        self.groups_panel.references_changed.connect(self._references_changed)
        self.map_page.run_requested.connect(self.rerun_map)
        self.map_page.submap_requested.connect(self.run_submap)
        self.map_page.inputs_changed.connect(self._map_inputs_changed)
        self.map_page.batch_maps_requested.connect(self.run_batch_maps)
        self.score_page.recompute_requested.connect(self.recompute_score)
        self.weights.changed.connect(self._weights_changed)
        self.batch_map_page.descriptors_ready.connect(self._descriptors_ready)
        self.batch_map_page.settings_changed.connect(self._batch_map_settings_changed)
        self.batch_map_page.particle_map_requested.connect(self.open_batch_particles)
        self.batch_map_page.compare_requested.connect(self.open_batch_compare)
        self.batch_table_page.changed.connect(self._batch_table_changed)
        self.batch_table_page.kinds_changed.connect(self._batch_kinds_changed)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.setMaximumWidth(160); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self._build_menus()
        self._set_enabled(False)
        self._update_title()
        self.statusBar().showMessage("파일 › 불러오기로 입자 CSV나 프로젝트(.dmap)를 여세요. 창에 끌어다 놓아도 됩니다.")

    # ---- menus -----------------------------------------------------------
    def _act(self, menu, text, fn, shortcut=None, checkable=False):
        a = QtGui.QAction(text, self)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("파일")
        self._act(f, "새 프로젝트 (입자 CSV)…", self.new_from_csv, QtGui.QKeySequence.New)
        self._act(f, "불러오기 (자동 인식)…", self.import_files, QtGui.QKeySequence.Open)
        self._act(f, "가상 예제 프로젝트 열기 (배치 5개)", self.open_demo)
        self.recent_menu = f.addMenu("최근 프로젝트")
        self._fill_recent()
        f.addSeparator()
        self.a_save = self._act(f, "프로젝트 저장", self.save, QtGui.QKeySequence.Save)
        self.a_saveas = self._act(f, "다른 이름으로 저장…", self.save_as, QtGui.QKeySequence.SaveAs)
        self.a_export = self._act(f, "모든 결과 통합 저장…", self.export_all, "Ctrl+Shift+E")
        f.addSeparator()
        self._act(f, "작업 기록 보기…", self.show_log, "Ctrl+Shift+L")
        self.a_logcsv = self._act(f, "작업 기록 저장 (CSV)…", self.save_log)
        f.addSeparator()
        q = self._act(f, "종료", self.close, QtGui.QKeySequence.Quit)
        q.setMenuRole(QtGui.QAction.QuitRole)

        self.tab_actions = {}
        bmm = mb.addMenu("배치 지도")
        self.tab_actions[self.batch_map_page] = self._act(bmm, "배치 지도 탭 열기", lambda on: self._set_tab_open(self.batch_map_page, on), checkable=True)
        bmm.addSeparator()
        self.a_bmap = self._act(bmm, "배치 지도 다시 계산", self.batch_map_page.compute, "Ctrl+B")
        bmm.addSeparator()
        self._act(bmm, "배치 지도 그림 저장…", lambda: self._save_fig(self.batch_map_page.canvas, "batch_map"))
        self._act(bmm, "범주별 지도 그림 저장…", lambda: self._save_fig(self.batch_map_page.cat_canvas, "batch_map_by_category"))
        self._act(bmm, "거리 표 그림 저장…", lambda: self._save_fig(self.batch_map_page.dist_canvas, "batch_distance_matrix"))
        self._act(bmm, "배치 군집 그림 저장…", lambda: self._save_fig(self.batch_map_page.tree_canvas, "batch_dendrogram"))
        self._act(bmm, "지도 좌표 · 거리 표 저장 (CSV)…", self._save_batch_map_tables)
        self._act(bmm, "기준 배치 score 표 저장 (CSV)…", self._save_batch_map_score)

        btm = mb.addMenu("배치 표")
        self.tab_actions[self.batch_table_page] = self._act(btm, "배치 표 탭 열기", lambda on: self._set_tab_open(self.batch_table_page, on), checkable=True)
        btm.addSeparator()
        self._act(btm, "배치 값 CSV 불러오기…", lambda: self._need_project() and self.cond_page.import_csv())
        self._act(btm, "배치 표 저장 (직접 입력 + 계산한 항목, CSV)…", self._save_batch_items)

        bm = mb.addMenu("배치 비교")
        self.tab_actions[self.batch_page] = self._act(bm, "배치 비교 탭 열기", lambda on: self._set_tab_open(self.batch_page, on), checkable=True)
        bm.addSeparator()
        self._act(bm, "항목 차이 그림 저장…", lambda: self._save_fig(self.batch_page.item_canvas, "batch_item_difference"))
        self._act(bm, "항목 차이 표 저장 (CSV)…", self._save_batch_items_diff)
        bm.addSeparator()
        self._act(bm, "같은 조건 기준 배치 score 계산 (입자 수준)", lambda: self._need_result() and self.batch_page.compute_score())
        self._act(bm, "같은 조건 배치 거리 그림 저장…", lambda: self._save_fig(self.batch_page.dist, "batch_distance"))
        self._act(bm, "같은 조건 측정값 차이 그림 저장…", lambda: self._save_fig(self.batch_page.delta, "batch_difference"))
        self._act(bm, "같은 조건 기준 배치 score 표 저장 (CSV)…", self._save_batch_score)

        dm = mb.addMenu("데이터")
        self.tab_actions[self.data_page] = self._act(dm, "데이터 탭 열기", lambda on: self._set_tab_open(self.data_page, on), checkable=True)
        self.tab_actions[self.cond_page] = self._act(dm, "그룹 조건 탭 열기", lambda on: self._set_tab_open(self.cond_page, on), checkable=True)
        dm.addSeparator()
        self.a_add = self._act(dm, "입자 CSV 추가…", self.add_csv)
        self._act(dm, "세포 전달 요약 CSV 추가…", self.add_delivery)
        dm.addSeparator()
        self._act(dm, "조건 CSV 불러오기 (그룹 또는 배치)…", lambda: self._need_project() and self.cond_page.import_csv())
        self._act(dm, "조건 CSV 저장…", lambda: self._need_project() and self.cond_page.export_csv())
        self._act(dm, "조건 목록 템플릿 불러오기…", lambda: self._need_project() and self.cond_page.load_template())
        self._act(dm, "조건 목록 템플릿 저장…", lambda: self._need_project() and self.cond_page.save_template())
        dm.addSeparator()
        self.a_save_particles = self._act(dm, "입자 표 저장 (QC 통과 + 지도 좌표·군집, CSV)…", self._save_particles)

        pl = mb.addMenu("입자 수준")
        self.a_run = self._act(pl, "입자 분석 실행", self.run_analysis, "Ctrl+R")
        pl.addSeparator()
        mm = pl.addMenu("입자 지도")
        self.tab_actions[self.map_page] = self._act(mm, "입자 지도 탭 열기", lambda on: self._set_tab_open(self.map_page, on), checkable=True)
        mm.addSeparator()
        self._act(mm, "상위 지도로", self.map_page.go_up, "Ctrl+Up")
        mm.addSeparator()
        self._act(mm, "지도 그림 저장 (PNG)…", self._save_map_png)
        self._act(mm, "군집 해석 그림 저장…", lambda: self._save_fig(self.map_page.cluster_canvas, "cluster_profile"))
        self._act(mm, "군집 구성 그림 저장…", lambda: self._save_fig(self.map_page.comp_canvas, "cluster_composition"))
        self._act(mm, "선택 영역 요약 그림 저장…", lambda: self._save_fig(self.map_page.sel_canvas, "selection_summary"))
        self._act(mm, "입자 + 군집 표 저장 (CSV)…", self._save_particles)

        pm = pl.addMenu("조건 비교")
        self.tab_actions[self.compare_page] = self._act(pm, "조건 비교 탭 열기", lambda on: self._set_tab_open(self.compare_page, on), checkable=True)
        pm.addSeparator()
        self._act(pm, "추세 그림 저장…", lambda: self._save_fig(self.compare_page.trend, "condition_trend"))
        self._act(pm, "연관 그림 저장…", lambda: self._save_fig(self.compare_page.assoc, "condition_association"))
        self._act(pm, "유사도 그림 저장…", lambda: self._save_fig(self.compare_page.sim, "group_similarity"))
        self._act(pm, "그룹 요약 표 저장 (CSV)…", self._save_group_summary)

        sm = pl.addMenu("기준 score")
        self.tab_actions[self.score_page] = self._act(sm, "기준 score 탭 열기", lambda on: self._set_tab_open(self.score_page, on), checkable=True)
        sm.addSeparator()
        self._act(sm, "score 다시 계산", self.recompute_score)
        self._act(sm, "순위 그림 저장…", lambda: self._save_fig(self.score_page.rank, "reference_score_rank"))
        self._act(sm, "분포 비교 그림 저장…", lambda: self._save_fig(self.score_page.dist, "reference_distributions"))
        self._act(sm, "score 표 저장 (CSV)…", self._save_score)

        vm = pl.addMenu("세포 전달")
        self.tab_actions[self.delivery_page] = self._act(vm, "세포 전달 탭 열기", lambda on: self._set_tab_open(self.delivery_page, on), checkable=True)
        vm.addSeparator()
        self._act(vm, "세포 전달 요약 CSV 추가…", self.add_delivery)
        self._act(vm, "강도 그림 저장…", lambda: self._save_fig(self.delivery_page.plot, "cell_delivery"))
        self._act(vm, "세포 전달 요약 표 저장 (CSV)…", self._save_delivery)

        for a in self.tab_actions.values():
            a.setChecked(True)

        tb = self.addToolBar("main"); tb.setMovable(False); tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        tb.addAction(self.a_bmap)
        tb.addAction(self.a_run)
        self.elapsed = QtWidgets.QLabel(""); self.elapsed.setStyleSheet("color:#555;padding-left:10px;")
        tb.addWidget(self.elapsed)

    def _need_project(self):
        if self.project is None:
            warn(self, "먼저 입자 CSV나 프로젝트를 여세요.")
            return False
        return True

    def _need_result(self):
        if self.result is None:
            warn(self, "먼저 '분석 실행'을 누르세요.")
            return False
        return True

    def _set_enabled(self, on):
        for a in (self.a_save, self.a_saveas, self.a_export, self.a_add, self.a_run, self.a_save_particles, self.a_bmap):
            a.setEnabled(on)
        self.weights.setEnabled(on)

    def _set_tab_open(self, page, on):
        i = self.pages.index(page)
        self.center.setTabVisible(i, on)
        self.tab_actions[page].setChecked(on)
        if on:
            self.center.setCurrentIndex(i)

    def _center_changed(self, i):
        if 0 <= i < len(self.pages):
            p = self.pages[i]
            self.right_stack.setCurrentWidget(p.panel)
            self.right.setTabText(0, f"{p.title} 설정")

    def _update_title(self):
        name = self.project.path.name if self.project and self.project.path else ("제목 없음" if self.project else "")
        star = " *" if self.dirty else ""
        self.setWindowTitle(f"Droplet Map {__version__}" + (f" — {name}{star}" if name else ""))

    # ---- work log --------------------------------------------------------
    def log(self, msg):
        self.statusBar().showMessage(msg)
        if self.project is not None:
            self.project.log.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg])

    def show_log(self):
        rows = self.project.log if self.project else []
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("작업 기록"); dlg.resize(760, 480)
        lay = QtWidgets.QVBoxLayout(dlg)
        t = QtWidgets.QTableWidget(len(rows), 2); t.setHorizontalHeaderLabels(["시간", "작업"])
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setWordWrap(True)
        for i, (ts, m) in enumerate(rows):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(ts)); t.setItem(i, 1, QtWidgets.QTableWidgetItem(m))
        t.resizeColumnToContents(0); t.resizeRowsToContents(); t.scrollToBottom()
        lay.addWidget(t)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    def save_log(self):
        if self._need_project():
            save_frame_dialog(self, pd.DataFrame(self.project.log, columns=["time", "action"]), "work_log.csv", index=False)

    # ---- dirty / stale state ---------------------------------------------
    def _mark_dirty(self):
        self.dirty = True
        self._update_title()

    def _visible_changed(self):
        self.map_page.set_visible_groups(self.groups_panel.visible_groups())
        self.map_page.visible_changed_for_map()
        self.batch_map_page.set_visible_batches(self.groups_panel.visible_batches())

    # ---- batch level ------------------------------------------------------
    def _descriptors_ready(self, desc):
        self.batch_table_page.set_descriptors(desc)
        self._batch_map_settings_changed(dirty=False)

    def _batch_map_settings_changed(self, dirty=True):
        bm = self.batch_map_page.bm
        self.batch_page.set_descriptors(self.batch_map_page.desc, bm.used if bm is not None else None)
        if dirty:
            self._mark_dirty()

    def _batch_table_changed(self):
        self._conditions_changed()
        self.cond_page.refresh_all()
        self.batch_map_page.compute()

    def _batch_kinds_changed(self):
        self._mark_dirty()
        self.batch_map_page.compute()
        self.log("배치 지도에 쓸 계산 항목: " + ", ".join(k for k, v in self.project.settings.bm_kinds.items() if v))

    def open_batch_compare(self, a, b):
        self._set_tab_open(self.batch_page, True)
        self.batch_page.show_pair(a, b)

    def open_batch_particles(self, batch):
        """Particle level for one batch: the particle map of that batch's particles only."""
        if not self._need_project():
            return
        self._set_tab_open(self.map_page, True)
        self.log(f"입자 수준으로: 배치 {batch}")
        if self.result is None or self._needs_full_run():
            self._after_analysis = lambda: self.open_batch_particles(batch)
            self.run_analysis()
            return
        rows = self.map_page.batch_rows().get(batch)
        if rows is None or not len(rows):
            warn(self, f"'{batch}' 배치에는 분석된 입자가 없습니다."); return
        while len(self.map_page.levels) > 1:
            self.map_page.go_up()
        self.run_submap(rows, f"배치 {batch}")

    def _map_inputs_changed(self):
        self._mark_dirty()
        self.statusBar().showMessage(self._map_inputs_text())

    def _map_inputs_text(self):
        s = self.project.settings
        vals = "; ".join(f"{k}={'/'.join(v or '기록 안 함' for v in vs)}" for k, vs in s.map_skip_values.items())
        return ("지도에서 뺀 값 — 측정값: " + (", ".join(s.map_skip_features) or "없음")
                + " · 조건 값: " + (vals or "없음")
                + (f" · 체크 안 한 그룹 {len(s.map_skip_groups)}개" if s.map_skip_groups else ""))

    def _data_changed(self):
        self._mark_dirty()
        self.weights.rebuild()
        self.map_page.fill_inputs()
        if self.result is not None:
            self.map_page.set_stale("측정값·QC 설정이 바뀌었습니다. '분석 실행'으로 다시 계산하세요.")

    def _conditions_changed(self):
        self._mark_dirty()
        self.groups_panel.refresh()
        for p in (self.map_page, self.compare_page, self.score_page, self.batch_page, self.delivery_page,
                  self.batch_table_page, self.batch_map_page):
            p.conditions_changed()
        if self.result is not None:
            self.statusBar().showMessage("조건이 바뀌었습니다. 연관 분석은 '분석 실행' 후에 갱신됩니다.")

    def _references_changed(self):
        self._mark_dirty()
        self.cond_page.model.refresh()
        self.score_page.conditions_changed()
        self.batch_page.conditions_changed()
        self.batch_table_page.refresh()
        self.batch_map_page._fill_batch_combos()
        self.batch_map_page.draw_score(); self.batch_map_page.draw()
        self.log("기준 그룹: " + (", ".join(sorted(self.project.conditions.references)) or "없음"))
        self.recompute_score()

    def _weights_changed(self):
        self._mark_dirty()
        w = self.project.settings.feature_weights
        self.log("측정값 가중치: " + ", ".join(f"{k}={v:g}" for k, v in w.items()))
        if self.result is None:
            return
        self.recompute_score()
        if self.map_page.auto_rerun.isChecked():
            self._pending_map = True
        else:
            self.map_page.set_stale("가중치가 바뀌었습니다. 지도에 반영하려면 오른쪽 '지도 다시 계산'을 누르세요 (score는 이미 반영됨).")

    # ---- project / import --------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QtWidgets.QMessageBox.question(self, "Droplet Map", "저장하지 않은 변경이 있습니다. 저장할까요?",
                                           QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if r == QtWidgets.QMessageBox.Save:
            return self.save()
        return r == QtWidgets.QMessageBox.Discard

    def set_project(self, project: Project, dirty=False):
        self.project = project
        self.result = None
        self._used = None
        self.dirty = dirty
        set_group_names(project.groups)
        for p in self.pages:
            p.set_project(project)
        self.groups_panel.set_project(project)
        self.weights.set_project(project)
        self._set_enabled(True)
        self._restore_ui(project.ui)
        self._update_title()
        self.batch_map_page.visible = self.groups_panel.visible_batches()
        self.batch_map_page.compute()
        n_b = len([b for b in project.batches.names() if b != UNASSIGNED])
        self.log(f"불러옴: 배치 {n_b}개 · 그룹 {len(project.groups)}개 · 입자 {len(project.particles):,}개")
        self.map_page.set_stale("아직 분석 전입니다. 툴바의 '입자 분석 실행'(⌘R)을 누르세요.")

    def new_from_csv(self):
        if not self._confirm_discard():
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "입자 CSV 열기", str(Path.home()), "CSV (*.csv *.txt)")
        if paths:
            self._open_particles(paths)

    def ask_batch(self, what: str, n: int, project=None):
        """Which batch the files' groups belong to. Returns the batch name, '' (decide later) or None
        (cancel). Replaced in the self-test."""
        names = [b for b in (project.batches.names() if project else []) if b != UNASSIGNED]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("배치 지정")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(f"{what} {n}개의 그룹은 어느 정제 배치의 단백질로 만들었나요?"))
        cb = QtWidgets.QComboBox(); cb.setEditable(True); cb.addItems(names)
        cb.lineEdit().setPlaceholderText("배치 이름 (예: 정제 로트 번호)")
        if len(names) == 1:
            cb.setCurrentIndex(0)
        else:
            cb.setEditText("")
        lay.addWidget(cb)
        lay.addWidget(QtWidgets.QLabel("<span style='color:#666'>비워 두면 '배치 미지정'으로 두고, 조건 탭에서 나중에 정할 수 있습니다.</span>"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if not dlg.exec():
            return None
        return cb.currentText().strip()

    def _open_particles(self, paths):
        batch = self.ask_batch("입자 CSV", len(paths))
        if batch is None:
            return
        try:
            pr = Project.from_csv(paths, batch=batch or None)
        except Exception as e:
            warn(self, f"CSV를 읽지 못했습니다.\n{e}"); return
        self.set_project(pr, dirty=True)
        self.log(f"입자 CSV {len(paths)}개 열기: " + ", ".join(Path(p).name for p in paths))
        self._show_notes(pr.notes)
        self._set_tab_open(self.cond_page, True)

    def open_demo(self):
        """Synthetic five-batch project, to try the batch level before real batches exist."""
        if not self._confirm_discard():
            return
        from ..demo import demo_project
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pr = demo_project()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.set_project(pr, dirty=True)
        self.log("가상 예제 프로젝트: 배치 5개 (B01 품질 높음 → B05 낮음), 배치마다 실험한 조건이 조금씩 다름, 제타 포텐셜·세포 전달 포함")
        self._set_tab_open(self.batch_map_page, True)

    def _show_notes(self, notes):
        for n in notes:
            self.log(n)
        if notes:
            QtWidgets.QMessageBox.information(self, "불러오기 확인", "\n\n".join(notes))

    def add_csv(self, paths=None):
        if not self._need_project():
            return
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "입자 CSV 추가", str(Path.home()), "CSV (*.csv *.txt)")
            if not paths:
                return
        batch = self.ask_batch("입자 CSV", len(paths), self.project)
        if batch is None:
            return
        n0 = len(self.project.particles)
        ui = self._ui_state()           # before the new groups exist, so they come in shown
        try:
            notes = self.project.add_csv(paths, batch=batch or None)
        except Exception as e:
            warn(self, f"CSV를 추가하지 못했습니다.\n{e}"); return
        added = len(self.project.particles) - n0
        self.project.ui = ui
        self.set_project(self.project, dirty=self.dirty or added > 0)
        self.log(f"입자 CSV 추가: 입자 {added:,}개 (" + ", ".join(Path(p).name for p in paths) + ")")
        self._show_notes(notes)

    def add_delivery(self, paths=None):
        if not self._need_project():
            return
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "세포 전달 요약 CSV 추가", str(Path.home()), "CSV (*.csv *.txt)")
            if not paths:
                return
        batch = self.ask_batch("세포 전달 CSV", len(paths), self.project)
        if batch is None:
            return
        n0 = len(self.project.delivery.groups())
        try:
            notes = self.project.add_delivery(paths, batch=batch or None)
        except Exception as e:
            warn(self, f"세포 전달 CSV를 추가하지 못했습니다.\n{e}"); return
        self._mark_dirty()
        self.cond_page.set_project(self.project)
        self.groups_panel.refresh()
        self.delivery_page.refresh()
        self.batch_page.conditions_changed()
        self.batch_table_page.refresh()
        self.batch_map_page.compute()
        n = len(self.project.delivery.groups()) - n0
        self.log(f"세포 전달 CSV 추가: 그룹 {n}개 (" + ", ".join(Path(p).name for p in paths) + ")")
        self._show_notes(notes)
        self._set_tab_open(self.delivery_page, True)

    def import_files(self, paths=None):
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "불러오기 (자동 인식)", str(Path.home()),
                "Droplet Map 파일 (*.dmap *.csv *.txt *.json);;모든 파일 (*)")
            if not paths:
                return
        kinds = {p: classify_file(p) for p in paths}
        projects = [p for p, k in kinds.items() if k == "project"]
        particles = [p for p, k in kinds.items() if k == "particles"]
        conds = [p for p, k in kinds.items() if k == "conditions"]
        templates = [p for p, k in kinds.items() if k == "template"]
        deliveries = [p for p, k in kinds.items() if k == "delivery"]
        unknown = [p for p, k in kinds.items() if k == "unknown"]
        if projects:
            self.open_project(projects[0])
        if particles:
            if self.project is None:
                self._open_particles(particles)
            else:
                r = QtWidgets.QMessageBox.question(
                    self, "입자 CSV", f"입자 CSV {len(particles)}개를 지금 프로젝트에 추가할까요?\n"
                                     "'아니요'를 누르면 새 프로젝트로 엽니다.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
                if r == QtWidgets.QMessageBox.Yes:
                    self.add_csv(particles)
                elif r == QtWidgets.QMessageBox.No and self._confirm_discard():
                    self._open_particles(particles)
        if deliveries:
            if self.project is None:
                warn(self, "세포 전달 요약은 입자 데이터를 먼저 연 다음에 불러올 수 있습니다.")
            else:
                self.add_delivery(deliveries)
        if self.project is not None:
            for p in templates:
                self.cond_page.load_template_path(p)
                self.log(f"조건 템플릿 불러옴: {Path(p).name}")
            for p in conds:
                self.cond_page.import_csv_path(p)
                self.log(f"조건 CSV 불러옴: {Path(p).name}")
        elif templates or conds:
            warn(self, "조건 파일은 입자 데이터를 먼저 연 다음에 불러올 수 있습니다.")
        if unknown:
            warn(self, "무슨 파일인지 알 수 없어 건너뛰었습니다:\n" + "\n".join(Path(p).name for p in unknown))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.import_files(paths)

    def open_project(self, path=None):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "프로젝트 열기", str(Path.home()), "Droplet Map 프로젝트 (*.dmap)")
            if not path:
                return
        try:
            pr = Project.load(path)
            self.set_project(pr)
        except Exception as e:
            warn(self, f"프로젝트를 열지 못했습니다.\n{e}"); return
        self._add_recent(path)
        if pr.notes:
            self._mark_dirty()
            self._show_notes(pr.notes)

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = self.settings.value("recent", []) or []
        if isinstance(recent, str):
            recent = [recent]
        for p in recent:
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda _c=False, path=p: self.open_project(path))
        self.recent_menu.setEnabled(bool(recent))

    def _add_recent(self, path):
        recent = self.settings.value("recent", []) or []
        if isinstance(recent, str):
            recent = [recent]
        path = str(path)
        recent = [path] + [p for p in recent if p != path]
        self.settings.setValue("recent", recent[:MAX_RECENT])
        self._fill_recent()

    def save(self) -> bool:
        if self.project is None:
            return False
        if self.project.path is None:
            return self.save_as()
        self._collect_settings()
        self.project.ui = self._ui_state()
        path = self.project.path
        try:
            self.project.save(path)
        except Exception as e:
            self.statusBar().showMessage("프로젝트를 저장하지 못했습니다.")
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "저장 실패",
                                        f"프로젝트를 저장하지 못했습니다. 이전에 저장한 파일은 그대로 있습니다.\n\n{path}\n\n{e}",
                                        parent=self)
            box.setDetailedText(traceback.format_exc())
            box.exec()
            return False
        self.log(f"프로젝트 저장: {path}")
        self.dirty = False
        self._update_title()
        self._add_recent(path)
        return True

    def save_as(self) -> bool:
        if self.project is None:
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "프로젝트 저장", str(Path.home() / "Documents" / "droplet_project.dmap"),
                                                        "Droplet Map 프로젝트 (*.dmap)")
        if not path:
            return False
        if not path.endswith(".dmap"):
            path += ".dmap"
        self.project.path = Path(path)
        return self.save()

    def _collect_settings(self):
        self.map_page.apply_settings()
        self.score_page.apply_settings()

    def _ui_state(self) -> dict:
        return {"open_tabs": [p.title for p in self.pages if self.center.isTabVisible(self.pages.index(p))],
                "closed_tabs": [p.title for p in self.pages if not self.center.isTabVisible(self.pages.index(p))],
                "current_tab": self.center.currentIndex(),
                "right_tab": self.right.currentIndex(),
                "visible_groups": sorted(self.groups_panel.visible_groups()),
                "known_groups": sorted(self.project.groups) if self.project else [],
                "map": self.map_page.ui_state(), "compare": self.compare_page.ui_state(), "score": self.score_page.ui_state(),
                "batch": self.batch_page.ui_state(), "delivery": self.delivery_page.ui_state(),
                "batch_map": self.batch_map_page.ui_state(), "batch_table": self.batch_table_page.ui_state()}

    def _restore_ui(self, st):
        st = st or {}
        closed = st.get("closed_tabs")
        if closed is None:          # new project, or saved before 0.4: particle level closed, the rest open
            closed = [p.title for p in self.particle_pages]
        for i, p in enumerate(self.pages):
            on = p.title not in closed
            self.center.setTabVisible(i, on)
            self.tab_actions[p].setChecked(on)
        cur = int(st.get("current_tab", 0)) if "closed_tabs" in st else 0
        self.center.setCurrentIndex(cur if self.center.isTabVisible(cur) else 0)
        self._center_changed(self.center.currentIndex())
        self.right.setCurrentIndex(int(st.get("right_tab", 0)))
        if "visible_groups" in st:
            vis = set(st["visible_groups"])
            if "known_groups" in st and self.project is not None:     # groups added since: shown
                vis |= set(self.project.groups) - set(st["known_groups"])
            self.groups_panel.set_visible_groups(vis)
            self.map_page.visible = vis
        self.map_page.restore_ui(st.get("map"))
        self.compare_page.restore_ui(st.get("compare"))
        self.score_page.restore_ui(st.get("score"))
        self.batch_page.restore_ui(st.get("batch"))
        self.delivery_page.restore_ui(st.get("delivery"))
        self.batch_map_page.restore_ui(st.get("batch_map"))
        self.batch_table_page.restore_ui(st.get("batch_table"))

    # ---- background work -------------------------------------------------
    def _start(self, label, fn, on_done):
        if self._thread is not None:
            self.statusBar().showMessage("다른 계산이 진행 중입니다. 끝난 뒤 다시 시도하세요.")
            return False
        self.progress.show()
        self.a_run.setEnabled(False)
        self._t0 = time.time()
        self._label = label
        self._thread = QtCore.QThread(self)
        self._thread.setStackSize(WORKER_STACK)
        self._task = Task(fn)
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._on_done = on_done
        # bound methods (not lambdas) so the slots run in the GUI thread
        self._task.progress.connect(self._progress)
        self._task.finished.connect(self._finish)
        self._task.failed.connect(self._failed)
        self._thread.start()
        return True

    def _cleanup_thread(self):
        self._thread.quit(); self._thread.wait()
        self._thread = None
        self.progress.hide()
        self.a_run.setEnabled(True)
        dt = time.time() - self._t0
        self.elapsed.setText(f"{self._label} {dt:.1f}초")
        return dt

    @QtCore.Slot(str)
    def _progress(self, msg):
        self.statusBar().showMessage(msg)

    @QtCore.Slot(object)
    def _finish(self, r):
        dt = self._cleanup_thread()
        self._on_done(r, dt)
        if getattr(self, "_pending_map", False):
            self._pending_map = False
            self.rerun_map()

    @QtCore.Slot(str)
    def _failed(self, msg):
        self._cleanup_thread()
        self.log(f"{self._label} 실패: {msg.splitlines()[0]}")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, f"{self._label} 실패", msg.split("\n\n")[0], parent=self)
        box.setDetailedText(msg)
        box.exec()

    def run_analysis(self):
        if not self._need_project():
            return
        self._collect_settings()
        p = self.project
        s = p.settings
        used = (s.d50, s.k, s.min_px, tuple(p.roles.features()))
        if s.map_skip_features or s.map_skip_values or s.map_skip_groups:
            self.log(self._map_inputs_text())
        view, strata = p.view, p.batches.series(p.groups)
        self._start("분석", lambda prog: run_all(p.particles, p.roles, view, s, progress=prog, strata=strata),
                    lambda r, dt: self._analysis_done(r, dt, used))

    def _analysis_done(self, result, dt, used):
        self.result = result
        self._used = used
        for p in (self.map_page, self.compare_page, self.score_page, self.batch_page, self.delivery_page):
            p.set_result(result)
        self.map_page.set_visible_groups(self.groups_panel.visible_groups())
        kept = result.prepared.data["group"].value_counts()
        lost = [g for g in self.project.groups if kept.get(g, 0) == 0]
        if lost:
            msg = (f"QC와 빈 측정값 때문에 입자가 하나도 남지 않은 그룹이 {len(lost)}개 있습니다:\n"
                   + "\n".join(lost[:15]) + ("\n…" if len(lost) > 15 else "")
                   + "\n\n데이터 탭에서 최소 픽셀 수나 측정값 선택을 확인하세요.")
            self.log(msg.splitlines()[0])
            QtWidgets.QMessageBox.warning(self, "분석 확인", msg)
        n_cl = len(set(result.pmap.clusters)) if result.pmap is not None else 0
        self.log(f"분석 완료 ({dt:.1f}초) — 입자 {len(result.prepared.data):,}개 (QC 제외 {result.prepared.n_dropped:,}개), 군집 {n_cl}개")
        self._mark_dirty()
        then = getattr(self, "_after_analysis", None)
        if then is not None:
            self._after_analysis = None
            QtCore.QTimer.singleShot(0, then)

    def _needs_full_run(self):
        s = self.project.settings
        return self.result is None or self._used != (s.d50, s.k, s.min_px, tuple(self.project.roles.features()))

    def rerun_map(self):
        """'지도 다시 계산': the current map level with the current weights and map settings."""
        if not self._need_project():
            return
        self._collect_settings()
        if self._needs_full_run() or len(self.map_page.levels) <= 1:
            return self.run_analysis()
        pm = self.map_page.current()
        prep, s = self.result.prepared, self.project.settings
        idx = map_rows(prep, self.project.view, s, pm.idx)
        self._start("하위 지도 다시 계산",
                    lambda prog: particle_map(prep, s, prog, idx=idx, name=pm.name),
                    lambda r, dt: (self.map_page.replace_level(r), self.log(f"지도 다시 계산: {pm.name} ({dt:.1f}초)")))

    def run_submap(self, idx, name):
        if not self._need_result():
            return
        self._collect_settings()
        prep, s = self.result.prepared, self.project.settings
        idx = map_rows(prep, self.project.view, s, idx)
        self._start("하위 지도",
                    lambda prog: particle_map(prep, s, prog, idx=idx, name=name),
                    lambda r, dt: (self.map_page.push_level(r),
                                   self.log(f"하위 지도: {name} — 입자 {len(idx):,}개, 군집 {len(set(r.clusters))}개 ({dt:.1f}초)")))

    def run_batch_maps(self):
        """One map per batch, each computed on that batch's particles only."""
        if not self._need_result():
            return
        self._collect_settings()
        prep, s = self.result.prepared, self.project.settings
        rows = {b: map_rows(prep, self.project.view, s, r) for b, r in self.map_page.batch_rows().items()}
        rows = {b: r for b, r in rows.items() if r is not None and len(r) >= 10}

        def work(prog):
            out = {}
            for i, (b, r) in enumerate(rows.items()):
                prog(f"배치별 지도 {i + 1}/{len(rows)}: {b} ({len(r):,}개 입자)")
                out[b] = particle_map(prep, s, None, idx=r, name=f"배치 {b}")
            return out
        self._start("배치별 지도", work,
                    lambda r, dt: (self.map_page.set_batch_maps(r), self.log(f"배치별 지도: {len(r)}개 ({dt:.1f}초)")))

    def recompute_score(self):
        if self.result is None:
            return
        self._collect_settings()
        refs = sorted(self.project.conditions.references)
        if not refs:
            self.result.score = None
            self.score_page.draw()
            return
        prep, s = self.result.prepared, self.project.settings
        self._start("score 계산", lambda prog: reference_score(prep, s, refs), self._score_done)

    def _score_done(self, score, dt):
        self.result.score = score
        self.score_page.set_result(self.result)
        self.compare_page.set_result(self.result)
        self.log(f"기준 score 다시 계산 ({dt:.1f}초)")

    # ---- per-feature saves -------------------------------------------------
    def _save_fig(self, canvas, name):
        if not canvas.fig.axes:
            warn(self, "저장할 그림이 아직 없습니다."); return
        p = save_figure_dialog(self, canvas.fig, f"{name}.png")
        if p:
            self.log(f"그림 저장: {p}")

    def _save_map_png(self):
        if not self._need_result():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "지도 그림 저장", "particle_map.png", "PNG (*.png)")
        if path:
            self.map_page.save_map_image(path)
            self.log(f"지도 그림 저장: {path}")

    def _save_particles(self):
        if self._need_result():
            p = save_frame_dialog(self, self.map_page.particles_frame(), "particles_with_clusters.csv", index=False)
            if p:
                self.log(f"입자 표 저장: {p}")

    def _save_group_summary(self):
        if self._need_result() and self.compare_page.summary_frame is not None:
            p = save_frame_dialog(self, self.compare_page.summary_frame, "group_summary.csv")
            if p:
                self.log(f"그룹 요약 저장: {p}")

    def _save_score(self):
        if self._need_result():
            if self.result.score is None:
                warn(self, "기준 그룹을 먼저 고르세요."); return
            p = save_frame_dialog(self, self.result.score, "reference_score.csv")
            if p:
                self.log(f"score 표 저장: {p}")

    def _save_batch_map_tables(self):
        bm = self.batch_map_page.bm
        if bm is None:
            warn(self, "배치 지도를 먼저 계산하세요."); return
        t = bm.distance.copy()
        if bm.coords is not None:
            t = bm.coords.join(t)
        p = save_frame_dialog(self, t, "batch_map_coords_distance.csv")
        if p:
            self.log(f"배치 지도 좌표·거리 저장: {p}")

    def _save_batch_map_score(self):
        f = self.batch_map_page.score_frame()
        if f is None:
            warn(self, "배치 지도 탭에서 기준 배치를 고르세요."); return
        p = save_frame_dialog(self, f, "batch_map_score.csv")
        if p:
            self.log(f"기준 배치 score 저장: {p}")

    def _save_batch_items(self):
        d = self.batch_map_page.desc
        if d is None:
            warn(self, "배치 지도를 먼저 계산하세요."); return
        out = d.meta.T.reindex(columns=d.values.columns)
        out = pd.concat([out, d.values.astype(object)])
        p = save_frame_dialog(self, out.T, "batch_items.csv")
        if p:
            self.log(f"배치 표 저장: {p}")

    def _save_batch_items_diff(self):
        t = self.batch_page.items
        if t is None or t.empty:
            warn(self, "배치 비교 탭에서 두 배치를 고르세요."); return
        p = save_frame_dialog(self, t, "batch_item_difference.csv", index=False)
        if p:
            self.log(f"항목 차이 표 저장: {p}")

    def _save_batch_score(self):
        s = self.batch_page.score_frame()
        if s is None:
            warn(self, "배치 비교 탭에서 기준 배치 score를 먼저 계산하세요."); return
        p = save_frame_dialog(self, s, "batch_reference_score.csv")
        if p:
            self.log(f"기준 배치 score 표 저장: {p}")

    def _save_delivery(self):
        if not self._need_project():
            return
        if self.project.delivery.empty():
            warn(self, "세포 전달 데이터가 없습니다."); return
        p = save_frame_dialog(self, self.project.delivery.measures(), "cell_delivery_summary.csv")
        if p:
            self.log(f"세포 전달 요약 표 저장: {p}")

    def figures(self) -> dict:
        out = {}
        if self.result is not None and self.map_page.current() is not None:
            out["map_particle_map"] = self.map_page.save_map_image
        for prefix, page, names in (("batchmap", self.batch_map_page, ("canvas", "cat_canvas", "dist_canvas", "tree_canvas", "score_canvas")),
                                    ("batchcompare", self.batch_page, ("item_canvas",)),
                                    ("map", self.map_page, ("cluster_canvas", "comp_canvas", "sel_canvas")),
                                    ("compare", self.compare_page, ("trend", "assoc", "sim")),
                                    ("score", self.score_page, ("rank", "dist")),
                                    ("batch", self.batch_page, ("dist", "delta", "score_canvas")),
                                    ("delivery", self.delivery_page, ("plot",))):
            for n in names:
                c = getattr(page, n, None)
                if c is not None and c.fig.axes:
                    out[f"{prefix}_{n}"] = c.fig
        return out

    def export_all(self):
        if not self._need_project():
            return
        last = self.settings.value("export_checks")
        dlg = ExportDialog(self, last if isinstance(last, dict) else None)
        if not dlg.exec():
            return
        which = dlg.selection()
        self.settings.setValue("export_checks", which)
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "저장할 폴더", str(Path.home() / "Desktop"))
        if not base:
            return
        out = Path(base) / export_folder_name(self.project)
        self.export_to(out, which)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out)))

    def export_to(self, out: Path, which: dict) -> list:
        self._collect_settings()
        parts = self.map_page.particles_frame() if self.result is not None else None
        files = write_exports(out, self.project, self.result, which,
                              self.figures() if which.get("figures") else {}, particles=parts,
                              batch_score=self.batch_page.score_frame(), batch_map=self.batch_map_page)
        keep = self.project.path
        self.project.ui = self._ui_state()
        self.project.save(out / f"{(keep.stem if keep else 'droplet_project')}.dmap")
        self.project.path = keep
        self.log(f"통합 저장: {out} ({len(files) + 1}개 파일)")
        return files

    def closeEvent(self, e):
        if self._confirm_discard():
            self.settings.setValue("split_sizes", self.split.sizes())
            e.accept()
        else:
            e.ignore()
