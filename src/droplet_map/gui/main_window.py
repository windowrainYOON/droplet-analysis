"""Main window: sidebar navigation, project actions and the background analysis worker."""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis import run_all
from ..core.project import Project
from .common import app_icon, set_group_names, warn, worker_thread
from .compare_page import ComparePage
from .conditions_page import ConditionsPage
from .data_page import DataPage
from .export_page import ExportPage
from .map_page import MapPage
from .score_page import ScorePage

PAGES = ("데이터", "조건 편집기", "입자 지도", "조건 비교", "기준 score", "내보내기")


class Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, project, with_map=True):
        super().__init__()
        self.project = project
        self.with_map = with_map

    @QtCore.Slot()
    def run(self):
        try:
            p = self.project
            r = run_all(p.particles, p.roles, p.conditions, p.settings, progress=self.progress.emit, with_map=self.with_map)
            self.finished.emit(r)
        except Exception as e:  # shown to the user, with the traceback for bug reports
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.result = None
        self.dirty = False
        self._thread = None
        self.setWindowTitle("Droplet Map")
        self.setWindowIcon(app_icon())
        self.resize(1400, 900)

        central = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(0, 0, 0, 0)
        self.nav = QtWidgets.QListWidget()
        self.nav.setFixedWidth(160)
        self.nav.setStyleSheet("QListWidget{border:none;background:#f3f4f6;font-size:14px;padding-top:8px}"
                               "QListWidget::item{padding:10px 14px}"
                               "QListWidget::item:selected{background:#dbe6fb;color:#123;}")
        for p in PAGES:
            self.nav.addItem(p)
        h.addWidget(self.nav)
        self.stack = QtWidgets.QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.data_page = DataPage()
        self.cond_page = ConditionsPage()
        self.map_page = MapPage()
        self.compare_page = ComparePage()
        self.score_page = ScorePage()
        self.export_page = ExportPage(figure_sources=self.figures)
        self.pages = [self.data_page, self.cond_page, self.map_page, self.compare_page, self.score_page, self.export_page]
        for w in self.pages:
            wrap = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(wrap); l.setContentsMargins(16, 12, 16, 12); l.addWidget(w)
            self.stack.addWidget(wrap)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.data_page.open_requested.connect(self.new_from_csv)
        self.data_page.add_requested.connect(self.add_csv)
        self.data_page.changed.connect(self._mark_stale)
        self.cond_page.changed.connect(self._conditions_changed)
        self.map_page.run_requested.connect(self.run_analysis)
        self.score_page.score_changed.connect(self._score_changed)
        # one set of measurement weights, shown on both pages
        self.map_page.weights.changed.connect(self._weights_changed)
        self.score_page.weights.changed.connect(self._weights_changed)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.setMaximumWidth(160); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self._build_menu()
        self._set_enabled(False)
        self.statusBar().showMessage("CSV를 열어 새 프로젝트를 만들거나, 저장된 프로젝트(.dmap)를 여세요.")

    # ---- menu ------------------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("파일")
        tb = self.addToolBar("main"); tb.setMovable(False); tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.actions = {}
        for key, text, sc, fn, on_tb in (
                ("new", "CSV로 새 프로젝트…", QtGui.QKeySequence.New, self.new_from_csv, True),
                ("open", "프로젝트 열기…", QtGui.QKeySequence.Open, self.open_project, True),
                ("add", "CSV 추가…", None, self.add_csv, False),
                ("save", "저장", QtGui.QKeySequence.Save, self.save, True),
                ("saveas", "다른 이름으로 저장…", QtGui.QKeySequence.SaveAs, self.save_as, False),
                ("run", "분석 실행", QtGui.QKeySequence("Ctrl+R"), self.run_analysis, True)):
            a = QtGui.QAction(text, self)
            if sc is not None:
                a.setShortcut(sc)
            a.triggered.connect(fn)
            m.addAction(a)
            if on_tb:
                tb.addAction(a)
            self.actions[key] = a
        m.addSeparator()
        q = QtGui.QAction("종료", self); q.setShortcut(QtGui.QKeySequence.Quit); q.triggered.connect(self.close); m.addAction(q)
        self.stale_label = QtWidgets.QLabel("")
        self.stale_label.setStyleSheet("color:#b05a00;padding-left:12px;")
        tb.addWidget(self.stale_label)

    def _set_enabled(self, on):
        for k in ("add", "save", "saveas", "run"):
            self.actions[k].setEnabled(on)
        for i in range(1, len(PAGES)):
            it = self.nav.item(i)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEnabled if on else it.flags() & ~QtCore.Qt.ItemIsEnabled)

    def _update_title(self):
        name = self.project.path.name if self.project and self.project.path else "제목 없음"
        self.setWindowTitle(f"Droplet Map — {name}{' •' if self.dirty else ''}")

    def _mark_dirty(self):
        self.dirty = True
        self._update_title()

    def _weights_changed(self):
        self.map_page.weights.refresh()
        self.score_page.weights.refresh()
        self._mark_dirty()
        if self.result is not None:
            self.stale_label.setText("가중치가 바뀌었습니다. '분석 실행'(또는 지도 페이지의 '가중치 적용')으로 다시 계산하세요.")

    def _mark_stale(self):
        self._mark_dirty()
        if self.project is not None:
            self.map_page.features_changed()
            self.score_page.features_changed()
        if self.result is not None:
            self.stale_label.setText("설정이 바뀌었습니다. '분석 실행'으로 다시 계산하세요.")

    def _conditions_changed(self):
        self._mark_dirty()
        for p in (self.map_page, self.compare_page, self.score_page):
            p.conditions_changed()
        if self.result is not None:
            self.stale_label.setText("조건이 바뀌었습니다. 연관·score는 '분석 실행' 후에 갱신됩니다.")

    def _score_changed(self):
        self._mark_dirty()
        self.compare_page.set_result(self.result)

    # ---- project ---------------------------------------------------------
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
        self.dirty = dirty
        set_group_names(project.groups)
        for p in self.pages:
            p.set_project(project)
        self._set_enabled(True)
        self.stale_label.setText("아직 분석 전입니다. '분석 실행'을 누르세요.")
        self._update_title()
        g = len(project.groups)
        self.statusBar().showMessage(f"그룹 {g}개 · 입자 {len(project.particles):,}개를 불러왔습니다.")

    def new_from_csv(self):
        if not self._confirm_discard():
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "입자 CSV 열기", str(Path.home()), "CSV (*.csv *.txt)")
        if not paths:
            return
        try:
            self.set_project(Project.from_csv(paths), dirty=True)
        except Exception as e:
            warn(self, f"CSV를 읽지 못했습니다.\n{e}"); return
        self.nav.setCurrentRow(1)

    def add_csv(self):
        if self.project is None:
            return self.new_from_csv()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "CSV 추가", str(Path.home()), "CSV (*.csv *.txt)")
        if not paths:
            return
        try:
            self.project.add_csv(paths)
        except Exception as e:
            warn(self, f"CSV를 추가하지 못했습니다.\n{e}"); return
        self.set_project(self.project, dirty=True)

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

    def save(self) -> bool:
        if self.project is None:
            return False
        if self.project.path is None:
            return self.save_as()
        self._collect_settings()
        self.project.save(self.project.path)
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"저장했습니다: {self.project.path}", 5000)
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
        if self.result is not None:
            self.score_page.apply_settings()

    # ---- analysis --------------------------------------------------------
    def run_analysis(self):
        if self.project is None or self._thread is not None:
            return
        self._collect_settings()
        self.progress.show()
        self.actions["run"].setEnabled(False)
        self._thread = worker_thread(self)
        self._worker = Worker(self.project)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda s: self.statusBar().showMessage(s))
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._thread.start()

    def _cleanup_thread(self):
        self._thread.quit(); self._thread.wait()
        self._thread = None
        self.progress.hide()
        self.actions["run"].setEnabled(True)

    def _done(self, result):
        self._cleanup_thread()
        self.result = result
        for p in (self.map_page, self.compare_page, self.score_page, self.export_page):
            p.set_result(result)
        self.stale_label.setText("")
        n_cl = len(set(result.pmap.clusters)) if result.pmap is not None else 0
        self.statusBar().showMessage(f"분석 완료 — 입자 {len(result.prepared.data):,}개 (QC 제외 {result.prepared.n_dropped:,}개), 군집 {n_cl}개")
        if self.nav.currentRow() < 2:
            self.nav.setCurrentRow(2)

    def _failed(self, msg):
        self._cleanup_thread()
        self.statusBar().showMessage("분석 실패")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "분석 실패", msg.split("\n\n")[0], parent=self)
        box.setDetailedText(msg)
        box.exec()

    def figures(self) -> dict:
        out = {}
        for prefix, page, names in (("map", self.map_page, ("cluster_canvas", "comp_canvas", "sel_canvas")),
                                    ("compare", self.compare_page, ("trend", "assoc", "sim")),
                                    ("score", self.score_page, ("rank", "dist"))):
            for n in names:
                c = getattr(page, n, None)
                if c is not None and c.fig.axes:
                    out[f"{prefix}_{n}"] = c.fig
        return out

    def closeEvent(self, e):
        if self._confirm_discard():
            e.accept()
        else:
            e.ignore()
