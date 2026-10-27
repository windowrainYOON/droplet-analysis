"""Application entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main():
    # numba (used by UMAP) needs a writable cache; the app bundle is read-only
    cache = Path.home() / "Library" / "Caches" / "DropletMap"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))

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
