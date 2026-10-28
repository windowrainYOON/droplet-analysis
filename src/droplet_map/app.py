"""Application entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _run_as_interpreter() -> bool:
    """multiprocessing / joblib helpers (e.g. the resource tracker) re-launch sys.executable as a
    Python interpreter: `<exe> -B -I -c "<code>"`. In the app bundle sys.executable is the app's
    own launcher, so without this a second, hidden copy of the GUI starts and never exits."""
    if "-c" not in sys.argv[1:]:
        return False
    i = sys.argv.index("-c")
    code = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    sys.argv = ["-c"] + sys.argv[i + 2:]
    exec(compile(code, "<string>", "exec"), {"__name__": "__main__"})
    return True


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    if _run_as_interpreter():
        return
    # numba (used by UMAP) needs a writable cache; the app bundle is read-only
    cache = Path.home() / "Library" / "Caches" / "DropletMap"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))

    if "--selftest" in sys.argv:
        from .selftest import run
        sys.exit(run())

    from PySide6 import QtWidgets
    from .gui.common import app_icon
    from .gui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Droplet Map")
    app.setOrganizationName("droplet-map")
    app.setWindowIcon(app_icon())
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if a.endswith(".dmap")]
    if args:
        w.open_project(args[0])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
