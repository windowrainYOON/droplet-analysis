"""Main window: left group list · center views · right settings tabs, one menu per feature,
auto-detecting import, work log and background workers."""
from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.analysis import particle_map, reference_score, run_all
from ..core.io import classify_file
from ..core.project import Project
from .common import app_icon, save_figure_dialog, save_frame_dialog, set_group_names, warn
from .compare_page import ComparePage
from .conditions_page import ConditionsPage
from .data_page import DataPage
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

        self.data_page = DataPage(self)
        self.cond_page = ConditionsPage(self)
        self.map_page = MapPage(self)
        self.compare_page = ComparePage(self)
        self.score_page = ScorePage(self)
        self.pages = [self.data_page, self.cond_page, self.map_page, self.compare_page, self.score_page]

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
        self.groups_panel.visible_changed.connect(lambda: self.map_page.set_visible_groups(self.groups_panel.visible_groups()))
        self.groups_panel.references_changed.connect(self._references_changed)
        self.map_page.run_requested.connect(self.rerun_map)
        self.map_page.submap_requested.connect(self.run_submap)
        self.score_page.recompute_requested.connect(self.recompute_score)
        self.weights.changed.connect(self._weights_changed)

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
        dm = mb.addMenu("데이터")
        self.tab_actions[self.data_page] = self._act(dm, "데이터 탭 열기", lambda on: self._set_tab_open(self.data_page, on), checkable=True)
        dm.addSeparator()
        self.a_add = self._act(dm, "입자 CSV 추가…", self.add_csv)
        self.a_save_particles = self._act(dm, "입자 표 저장 (QC 통과 + 지도 좌표·군집, CSV)…", self._save_particles)

        cm = mb.addMenu("조건")
        self.tab_actions[self.cond_page] = self._act(cm, "조건 탭 열기", lambda on: self._set_tab_open(self.cond_page, on), checkable=True)
        cm.addSeparator()
        self._act(cm, "조건 CSV 불러오기…", lambda: self._need_project() and self.cond_page.import_csv())
        self._act(cm, "조건 CSV 저장…", lambda: self._need_project() and self.cond_page.export_csv())
        cm.addSeparator()
        self._act(cm, "조건 목록 템플릿 불러오기…", lambda: self._need_project() and self.cond_page.load_template())
        self._act(cm, "조건 목록 템플릿 저장…", lambda: self._need_project() and self.cond_page.save_template())

        mm = mb.addMenu("입자 지도")
        self.tab_actions[self.map_page] = self._act(mm, "입자 지도 탭 열기", lambda on: self._set_tab_open(self.map_page, on), checkable=True)
        mm.addSeparator()
        self.a_run = self._act(mm, "분석 실행 (전체)", self.run_analysis, "Ctrl+R")
        self._act(mm, "상위 지도로", self.map_page.go_up, "Ctrl+Up")
        mm.addSeparator()
        self._act(mm, "지도 그림 저장 (PNG)…", self._save_map_png)
        self._act(mm, "군집 해석 그림 저장…", lambda: self._save_fig(self.map_page.cluster_canvas, "cluster_profile"))
        self._act(mm, "군집 구성 그림 저장…", lambda: self._save_fig(self.map_page.comp_canvas, "cluster_composition"))
        self._act(mm, "선택 영역 요약 그림 저장…", lambda: self._save_fig(self.map_page.sel_canvas, "selection_summary"))
        self._act(mm, "입자 + 군집 표 저장 (CSV)…", self._save_particles)

        pm = mb.addMenu("조건 비교")
        self.tab_actions[self.compare_page] = self._act(pm, "조건 비교 탭 열기", lambda on: self._set_tab_open(self.compare_page, on), checkable=True)
        pm.addSeparator()
        self._act(pm, "추세 그림 저장…", lambda: self._save_fig(self.compare_page.trend, "condition_trend"))
        self._act(pm, "연관 그림 저장…", lambda: self._save_fig(self.compare_page.assoc, "condition_association"))
        self._act(pm, "유사도 그림 저장…", lambda: self._save_fig(self.compare_page.sim, "group_similarity"))
        self._act(pm, "그룹 요약 표 저장 (CSV)…", self._save_group_summary)

        sm = mb.addMenu("기준 score")
        self.tab_actions[self.score_page] = self._act(sm, "기준 score 탭 열기", lambda on: self._set_tab_open(self.score_page, on), checkable=True)
        sm.addSeparator()
        self._act(sm, "score 다시 계산", self.recompute_score)
        self._act(sm, "순위 그림 저장…", lambda: self._save_fig(self.score_page.rank, "reference_score_rank"))
        self._act(sm, "분포 비교 그림 저장…", lambda: self._save_fig(self.score_page.dist, "reference_distributions"))
        self._act(sm, "score 표 저장 (CSV)…", self._save_score)

        for a in self.tab_actions.values():
            a.setChecked(True)

        tb = self.addToolBar("main"); tb.setMovable(False); tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
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
        for a in (self.a_save, self.a_saveas, self.a_export, self.a_add, self.a_run, self.a_save_particles):
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

    def _data_changed(self):
        self._mark_dirty()
        self.weights.rebuild()
        if self.result is not None:
            self.map_page.set_stale("측정값·QC 설정이 바뀌었습니다. '분석 실행'으로 다시 계산하세요.")

    def _conditions_changed(self):
        self._mark_dirty()
        self.groups_panel.refresh()
        for p in (self.map_page, self.compare_page, self.score_page):
            p.conditions_changed()
        if self.result is not None:
            self.statusBar().showMessage("조건이 바뀌었습니다. 연관 분석은 '분석 실행' 후에 갱신됩니다.")

    def _references_changed(self):
        self._mark_dirty()
        self.cond_page.model.refresh()
        self.score_page.conditions_changed()
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
        self.log(f"불러옴: 그룹 {len(project.groups)}개 · 입자 {len(project.particles):,}개")
        self.map_page.set_stale("아직 분석 전입니다. 툴바의 '분석 실행'(⌘R)을 누르세요.")

    def new_from_csv(self):
        if not self._confirm_discard():
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "입자 CSV 열기", str(Path.home()), "CSV (*.csv *.txt)")
        if paths:
            self._open_particles(paths)

    def _open_particles(self, paths):
        try:
            pr = Project.from_csv(paths)
        except Exception as e:
            warn(self, f"CSV를 읽지 못했습니다.\n{e}"); return
        self.set_project(pr, dirty=True)
        self.log(f"입자 CSV {len(paths)}개 열기: " + ", ".join(Path(p).name for p in paths))
        self._show_notes(pr.notes)
        self._set_tab_open(self.cond_page, True)

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
        n0 = len(self.project.particles)
        try:
            notes = self.project.add_csv(paths)
        except Exception as e:
            warn(self, f"CSV를 추가하지 못했습니다.\n{e}"); return
        added = len(self.project.particles) - n0
        self.project.ui = self._ui_state()
        self.set_project(self.project, dirty=self.dirty or added > 0)
        self.log(f"입자 CSV 추가: 입자 {added:,}개 (" + ", ".join(Path(p).name for p in paths) + ")")
        self._show_notes(notes)

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
            self.set_project(Project.load(path))
        except Exception as e:
            warn(self, f"프로젝트를 열지 못했습니다.\n{e}"); return
        self._add_recent(path)

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
                "current_tab": self.center.currentIndex(),
                "right_tab": self.right.currentIndex(),
                "visible_groups": sorted(self.groups_panel.visible_groups()),
                "map": self.map_page.ui_state(), "compare": self.compare_page.ui_state(), "score": self.score_page.ui_state()}

    def _restore_ui(self, st):
        st = st or {}
        open_tabs = st.get("open_tabs")
        for i, p in enumerate(self.pages):
            on = open_tabs is None or p.title in open_tabs
            self.center.setTabVisible(i, on)
            self.tab_actions[p].setChecked(on)
        self.center.setCurrentIndex(int(st.get("current_tab", 1 if open_tabs is None else 0)))
        self._center_changed(self.center.currentIndex())
        self.right.setCurrentIndex(int(st.get("right_tab", 0)))
        if "visible_groups" in st:
            self.groups_panel.set_visible_groups(st["visible_groups"])
            self.map_page.visible = set(st["visible_groups"])
        self.map_page.restore_ui(st.get("map"))
        self.compare_page.restore_ui(st.get("compare"))
        self.score_page.restore_ui(st.get("score"))

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
        self._start("분석", lambda prog: run_all(p.particles, p.roles, p.conditions, s, progress=prog),
                    lambda r, dt: self._analysis_done(r, dt, used))

    def _analysis_done(self, result, dt, used):
        self.result = result
        self._used = used
        for p in (self.map_page, self.compare_page, self.score_page):
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
        self._start("하위 지도 다시 계산",
                    lambda prog: particle_map(prep, s, prog, idx=pm.idx, name=pm.name),
                    lambda r, dt: (self.map_page.replace_level(r), self.log(f"지도 다시 계산: {pm.name} ({dt:.1f}초)")))

    def run_submap(self, idx, name):
        if not self._need_result():
            return
        self._collect_settings()
        prep, s = self.result.prepared, self.project.settings
        self._start("하위 지도",
                    lambda prog: particle_map(prep, s, prog, idx=idx, name=name),
                    lambda r, dt: (self.map_page.push_level(r),
                                   self.log(f"하위 지도: {name} — 입자 {len(idx):,}개, 군집 {len(set(r.clusters))}개 ({dt:.1f}초)")))

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

    def figures(self) -> dict:
        out = {}
        if self.result is not None and self.map_page.current() is not None:
            out["map_particle_map"] = self.map_page.save_map_image
        for prefix, page, names in (("map", self.map_page, ("cluster_canvas", "comp_canvas", "sel_canvas")),
                                    ("compare", self.compare_page, ("trend", "assoc", "sim")),
                                    ("score", self.score_page, ("rank", "dist"))):
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
                              self.figures() if which.get("figures") else {}, particles=parts)
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
