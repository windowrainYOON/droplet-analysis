"""Main window: left batch › formation condition list · center views · right settings tabs, one menu
per feature, work log and background workers.

Data come from a folder of batch files (one Excel workbook per batch), edited in the batch note
window or in Excel; the folder is watched and read again when a file changes. Main flow: 배치 지도 ·
배치 표 · 배치 비교 (batch level). Particle level (입자 지도, 조건 비교, 기준 score, 측정 시트) opens
from the 입자 수준 menu or from a batch on the map."""
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
from ..core.batchfile import batch_paths
from ..core.batchset import forget_cache
from ..core.project import Project
from .common import app_icon, save_figure_dialog, save_frame_dialog, set_group_names, warn
from .batch_map_page import BatchMapPage
from .batch_page import BatchPage
from .batch_table_page import BatchTablePage
from .compare_page import ComparePage
from .data_page import DataPage
from .export_page import ExportDialog, export_folder_name, write_exports
from .map_page import MapPage
from .measure_page import MeasurePage
from .score_page import ScorePage
from .side_panels import GroupPanel, WeightsPanel

MAX_RECENT = 8
RELOAD_DELAY_MS = 1500     # Excel writes a file in several steps: wait until it settles
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
        self.map_page = MapPage(self)
        self.compare_page = ComparePage(self)
        self.score_page = ScorePage(self)
        self.measure_page = MeasurePage(self)
        self.pages = [self.batch_map_page, self.batch_table_page, self.batch_page, self.data_page,
                      self.map_page, self.compare_page, self.score_page, self.measure_page]
        # particle level: closed until asked for (입자 수준 menu, or a batch on the batch map)
        self.particle_pages = [self.map_page, self.compare_page, self.score_page, self.measure_page]
        self.note_window = None

        self.watcher = QtCore.QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(self._folder_touched)
        self.watcher.fileChanged.connect(self._folder_touched)
        self.reload_timer = QtCore.QTimer(self)
        self.reload_timer.setSingleShot(True)
        self.reload_timer.setInterval(RELOAD_DELAY_MS)
        self.reload_timer.timeout.connect(self.reload_folder)
        self._stamps = {}

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
        self.batch_table_page.edit_requested.connect(lambda: self.open_batch_note())
        self.batch_table_page.kinds_changed.connect(self._batch_kinds_changed)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.setMaximumWidth(160); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self._build_menus()
        self._set_enabled(False)
        self._update_title()
        self.statusBar().showMessage("파일 › 배치 폴더 열기(⌘O)로 배치 파일 폴더를 여세요. 폴더나 .dmap을 창에 끌어다 놓아도 됩니다.")

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
        self._act(f, "배치 폴더 열기…", self.open_folder_dialog, QtGui.QKeySequence.Open)
        self._act(f, "새 배치 폴더 만들기…", self.new_folder, QtGui.QKeySequence.New)
        self._act(f, "프로젝트 열기 (.dmap)…", self.open_project, "Ctrl+Shift+O")
        self._act(f, "가상 예제 열기 (배치 5개)", self.open_demo)
        self.recent_menu = f.addMenu("최근 연 폴더 · 프로젝트")
        self._fill_recent()
        f.addSeparator()
        self.a_note = self._act(f, "배치 노트 열기 (배치 파일 편집)…", lambda: self.open_batch_note(), "Ctrl+E")
        self.a_reload = self._act(f, "배치 폴더 다시 읽기", self.reload_folder, "Ctrl+Shift+R")
        self.a_reveal = self._act(f, "배치 폴더 보기 (Finder)", self.reveal_folder)
        self.a_export_files = self._act(f, "배치 파일로 꺼내기 (예전 프로젝트)…", self.export_batch_files)
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
        self._act(btm, "배치 노트에서 편집…", lambda: self.open_batch_note())
        self._act(btm, "배치 표 저장 (배치 값 + 계산한 항목, CSV)…", self._save_batch_items)

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
        dm.addSeparator()
        self._act(dm, "입자 CSV · 세포 전달 CSV 넣기 (배치 노트)…", lambda: self.open_batch_note())
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

        vm = pl.addMenu("측정 시트 (in vitro · 세포 전달 …)")
        self.tab_actions[self.measure_page] = self._act(vm, "측정 시트 탭 열기", lambda on: self._set_tab_open(self.measure_page, on), checkable=True)
        vm.addSeparator()
        self._act(vm, "그림 저장…", lambda: self._save_fig(self.measure_page.plot, "measurement_sheet"))
        self._act(vm, "조건별 요약 표 저장 (CSV)…", self._save_measure)

        for a in self.tab_actions.values():
            a.setChecked(True)

        tb = self.addToolBar("main"); tb.setMovable(False); tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        tb.addAction(self.a_bmap)
        tb.addAction(self.a_run)
        self.elapsed = QtWidgets.QLabel(""); self.elapsed.setStyleSheet("color:#555;padding-left:10px;")
        tb.addWidget(self.elapsed)

    def _need_project(self):
        if self.project is None:
            warn(self, "먼저 배치 폴더나 프로젝트를 여세요.")
            return False
        return True

    def _need_result(self):
        if self.result is None:
            warn(self, "먼저 '분석 실행'을 누르세요.")
            return False
        return True

    def _set_enabled(self, on):
        for a in (self.a_save, self.a_saveas, self.a_export, self.a_run, self.a_save_particles, self.a_bmap,
                  self.a_reload, self.a_reveal, self.a_export_files):
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
        p = self.project
        name = p.path.name if p and p.path else (p.folder.name if p and p.folder else ("제목 없음" if p else ""))
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
        for p in (self.map_page, self.compare_page, self.score_page, self.batch_page, self.measure_page,
                  self.batch_table_page, self.batch_map_page):
            p.conditions_changed()
        if self.result is not None:
            self.statusBar().showMessage("조건이 바뀌었습니다. 연관 분석은 '분석 실행' 후에 갱신됩니다.")

    def _references_changed(self):
        self._mark_dirty()
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

    def set_project(self, project: Project, dirty=False, keep_result=None):
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
        self._watch()
        if self.note_window is not None:
            self.note_window.set_folder(project.folder)
        self.batch_map_page.visible = self.groups_panel.visible_batches()
        self.batch_map_page.compute()
        if keep_result is not None:
            self.result, self._used = keep_result
            for p in (self.map_page, self.compare_page, self.score_page, self.batch_page, self.measure_page):
                p.set_result(self.result)
            self.map_page.set_visible_groups(self.groups_panel.visible_groups())
            return
        n_b = len([b for b in project.batches.names() if b != UNASSIGNED])
        n_p = int(project.particles["group"].nunique()) if len(project.particles) else 0
        self.log(f"불러옴: 배치 {n_b}개 · 형성 조건 {len(project.groups)}개 (입자 있는 조건 {n_p}개) · "
                 f"입자 {len(project.particles):,}개 · 조건별 측정 시트 {len(project.measures)}개")
        self.map_page.set_stale("아직 분석 전입니다. 툴바의 '입자 분석 실행'(⌘R)을 누르세요.")

    def _show_notes(self, notes):
        for n in notes:
            self.log(n)
        if notes:
            QtWidgets.QMessageBox.information(self, "불러오기 확인", "\n\n".join(notes))

    # ---- batch folder ----------------------------------------------------------
    def open_folder_dialog(self):
        start = str(self.project.folder) if self.project and self.project.folder else str(Path.home() / "Documents")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "배치 폴더 열기 (배치 파일 .xlsx가 있는 폴더)", start)
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder):
        if not self._confirm_discard():
            return
        folder = Path(folder)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pr = Project.open_folder(folder)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            warn(self, f"배치 폴더를 읽지 못했습니다.\n{e}"); return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.set_project(pr)
        self._add_recent(folder)
        if not pr.source.files:
            self.log(f"배치 폴더 {folder.name}: 배치 파일이 아직 없습니다.")
            if QtWidgets.QMessageBox.question(
                    self, "배치 폴더", f"'{folder.name}' 폴더에 배치 파일(.xlsx)이 없습니다.\n배치 노트를 열어 첫 배치를 만들까요?") \
                    == QtWidgets.QMessageBox.Yes:
                self.open_batch_note()
            return
        self._set_tab_open(self.batch_map_page, True)
        if pr.notes:
            self._show_notes(pr.notes)

    def new_folder(self):
        """A new (empty) batch folder with the default template, then the batch note to fill it."""
        if not self._confirm_discard():
            return
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "새 배치 폴더를 만들 곳", str(Path.home() / "Documents"))
        if not base:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "새 배치 폴더", "폴더 이름 (예: 단백질 이름):", text="배치 노트")
        if not ok or not name.strip():
            return
        from ..core.batchfile import TEMPLATE_NAME, template, write_batch_file
        folder = Path(base) / name.strip()
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / TEMPLATE_NAME).exists():
            write_batch_file(template(), folder / TEMPLATE_NAME)
        self.dirty = False
        self.set_project(Project.open_folder(folder))
        self._add_recent(folder)
        self.log(f"새 배치 폴더: {folder}")
        self.open_batch_note(new_batch=True)

    def reveal_folder(self):
        if self._need_project() and self.project.folder is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.project.folder)))

    def open_batch_note(self, new_batch=False, csv_paths=None):
        """The batch note window: edits the batch files of the open folder."""
        from .batch_note import BatchNoteWindow
        if self.project is None or self.project.folder is None:
            if self.project is not None and self.project.folder is None:
                warn(self, "이 프로젝트는 아직 배치 파일이 없습니다. 먼저 '파일 › 배치 파일로 꺼내기'로 폴더에 쓰세요.")
            else:
                warn(self, "먼저 배치 폴더를 여세요 (파일 › 배치 폴더 열기, 또는 새 배치 폴더 만들기).")
            return None
        if self.note_window is None:
            self.note_window = BatchNoteWindow()
            self.note_window.saved.connect(self._note_saved)
        self.note_window.set_folder(self.project.folder)
        self.note_window.show(); self.note_window.raise_(); self.note_window.activateWindow()
        if new_batch:
            self.note_window.new_batch()
        if csv_paths:
            self.note_window.import_csvs(csv_paths)
        return self.note_window

    def _note_saved(self, path):
        forget_cache(path)
        self.reload_timer.start(200)

    def export_batch_files(self) -> bool:
        """An old project (data inside the .dmap): write its batches as batch files into a folder."""
        if not self._need_project():
            return False
        p = self.project
        start = (p.path.parent if p.path else Path.home() / "Documents")
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "배치 파일을 쓸 폴더를 만들 곳", str(start))
        if not base:
            return False
        name = (p.path.stem if p.path else "droplet") + " 배치 파일"
        folder = Path(base) / name
        if folder.exists() and any(batch_paths(folder)):
            warn(self, f"'{folder}' 폴더에 이미 배치 파일이 있습니다. 다른 곳을 고르세요."); return False
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            files = p.export_batch_files(folder)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            warn(self, f"배치 파일을 쓰지 못했습니다.\n{e}"); return False
        QtWidgets.QApplication.restoreOverrideCursor()
        self._watch()
        self._mark_dirty()
        self.log(f"배치 파일로 꺼냄: {folder} (배치 {len(files)}개). 이제 이 폴더가 데이터입니다.")
        QtWidgets.QMessageBox.information(
            self, "배치 파일로 꺼내기",
            f"배치 {len(files)}개를 '{folder}'에 배치 파일로 썼습니다.\n\n이제 데이터는 이 폴더에서 읽습니다. "
            "프로젝트를 저장하면 .dmap에는 분석 설정과 폴더 위치만 들어갑니다.")
        return True

    # ---- watching the folder ------------------------------------------------------
    def _stamp(self) -> dict:
        f = self.project.folder if self.project else None
        if f is None or not f.is_dir():
            return {}
        out = {}
        for p in batch_paths(f):
            try:
                st = p.stat()
            except OSError:
                continue
            out[p.name] = (st.st_mtime, st.st_size)
        return out

    def _watch(self):
        w = self.watcher
        if w.files():
            w.removePaths(w.files())
        if w.directories():
            w.removePaths(w.directories())
        f = self.project.folder if self.project else None
        if f is None or not f.is_dir():
            self._stamps = {}
            return
        w.addPath(str(f))
        files = [str(p) for p in batch_paths(f)]
        if files:
            w.addPaths(files)
        self._stamps = self._stamp()

    def _folder_touched(self, *_):
        self.reload_timer.start()

    def reload_folder(self):
        """Read the batch folder again (only changed files are parsed); keep settings and view."""
        p = self.project
        if p is None or p.folder is None:
            return
        if self._thread is not None:          # finish the running calculation first
            self.reload_timer.start()
            return
        new = self._stamp()
        if new == self._stamps and self.sender() is not self.a_reload:
            self._watch()                     # a temp file came and went; files may have been replaced
            return
        old = self._stamps
        changed = sorted(n for n in new if n in old and new[n] != old[n])
        added = sorted(set(new) - set(old))
        gone = sorted(set(old) - set(new))
        before = (len(p.particles), pd.util.hash_pandas_object(p.particles, index=False).sum() if len(p.particles) else 0)
        self._collect_settings()
        p.ui = self._ui_state()
        try:
            notes = p.reload()
        except Exception as e:
            self.log(f"배치 폴더를 다시 읽지 못했습니다: {e}")
            self._watch()
            return
        after = (len(p.particles), pd.util.hash_pandas_object(p.particles, index=False).sum() if len(p.particles) else 0)
        keep = (self.result, self._used) if self.result is not None and before == after else None
        self.set_project(p, dirty=self.dirty, keep_result=keep)
        what = ", ".join([f"{n} 바뀜" for n in changed] + [f"{n} 추가" for n in added] + [f"{n} 없어짐" for n in gone])
        self.log("배치 폴더 다시 읽음" + (f": {what}" if what else "")
                 + ("" if keep or self.result is None else " · 입자 데이터가 바뀌어 입자 분석은 다시 실행해야 합니다"))
        if self.result is None and before != after and before[0]:
            self.map_page.set_stale("배치 파일의 입자 데이터가 바뀌었습니다. '입자 분석 실행'(⌘R)으로 다시 계산하세요.")
        for n in notes:
            self.log(n)

    # ---- opening ----------------------------------------------------------------
    def open_demo(self):
        """Synthetic five-batch folder, to try the batch level before real batches exist."""
        if not self._confirm_discard():
            return
        from ..demo import demo_project
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            pr = demo_project()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.set_project(pr)
        self.log(f"가상 예제: 배치 5개 (B01 품질 높음 → B05 낮음), 배치마다 실험한 조건이 조금씩 다름, "
                 f"제타 포텐셜 · FRAP · 세포 전달 포함 — 폴더 {pr.folder}")
        self._set_tab_open(self.batch_map_page, True)

    def import_files(self, paths=None):
        """Dropped or chosen files: a folder or .xlsx opens its batch folder, a .dmap opens the project,
        CSVs go into a batch file through the batch note."""
        paths = [Path(p) for p in (paths or [])]
        dirs = [p for p in paths if p.is_dir()]
        dmaps = [p for p in paths if p.suffix.lower() == ".dmap"]
        books = [p for p in paths if p.suffix.lower() in (".xlsx", ".xlsm")]
        csvs = [p for p in paths if p.suffix.lower() in (".csv", ".txt")]
        other = [p for p in paths if p not in dirs + dmaps + books + csvs]
        if dmaps:
            self.open_project(str(dmaps[0]))
        elif dirs:
            self.open_folder(dirs[0])
        elif books:
            folder = books[0].parent
            if self.project is None or self.project.folder != folder:
                self.open_folder(folder)
        if csvs:
            if self.project is None or self.project.folder is None:
                warn(self, "CSV는 배치 파일에 넣어 씁니다. 먼저 배치 폴더를 열거나 만드세요 (파일 › 새 배치 폴더 만들기).")
            else:
                self.open_batch_note(csv_paths=[str(p) for p in csvs])
        if other:
            warn(self, "무슨 파일인지 알 수 없어 건너뛰었습니다:\n" + "\n".join(p.name for p in other))

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
            try:
                pr = Project.load(path)
            except FileNotFoundError as e:
                warn(self, f"이 프로젝트가 읽던 배치 폴더를 찾지 못했습니다:\n{e}\n\n옮긴 폴더를 골라 주세요.")
                folder = QtWidgets.QFileDialog.getExistingDirectory(self, "배치 폴더 찾기", str(Path(path).parent))
                if not folder:
                    return
                pr = Project.load(path, folder=folder)
                pr.notes.append(f"배치 폴더 위치가 바뀌었습니다: {folder} (저장하면 새 위치를 기억합니다).")
            self.set_project(pr)
        except Exception as e:
            warn(self, f"프로젝트를 열지 못했습니다.\n{e}"); return
        self._add_recent(path)
        if pr.folder is None:
            pr.notes.append("예전 형식 프로젝트입니다 (데이터가 .dmap 안에 있음). 분석은 그대로 할 수 있고, "
                            "'파일 › 배치 파일로 꺼내기'로 배치 파일 폴더를 만들면 배치 노트에서 고칠 수 있습니다.")
        if pr.notes:
            self._mark_dirty()
            self._show_notes(pr.notes)

    def _open_recent(self, path):
        if Path(path).is_dir():
            self.open_folder(path)
        elif Path(path).exists():
            self.open_project(path)
        else:
            warn(self, f"찾을 수 없습니다:\n{path}")

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = self.settings.value("recent", []) or []
        if isinstance(recent, str):
            recent = [recent]
        for p in recent:
            a = self.recent_menu.addAction(("📁 " if not p.endswith(".dmap") else "") + p)
            a.triggered.connect(lambda _c=False, path=p: self._open_recent(path))
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
        if self.project.folder is None and not self.export_batch_files():
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
        start = (self.project.folder.parent / f"{self.project.folder.name}.dmap") if self.project.folder else (Path.home() / "Documents" / "droplet_project.dmap")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "프로젝트 저장 (분석 설정 · 화면 상태)", str(start),
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
                "batch": self.batch_page.ui_state(), "measure": self.measure_page.ui_state(),
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
        self.measure_page.restore_ui(st.get("measure"))
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
        for p in (self.map_page, self.compare_page, self.score_page, self.batch_page, self.measure_page):
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

    def _save_measure(self):
        if not self._need_project():
            return
        f = self.measure_page.summary_frame()
        if f is None:
            warn(self, "조건별 측정 시트에 값이 없습니다."); return
        p = save_frame_dialog(self, f, f"{self.measure_page.sheet.currentText()}_summary.csv")
        if p:
            self.log(f"측정 시트 요약 표 저장: {p}")

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
                                    ("measure", self.measure_page, ("plot",))):
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
        if self.project.folder is not None:
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
