# Droplet Map — notes for Claude

Desktop app (PySide6, macOS Apple Silicon) that maps and compares phase-separated protein droplets
from per-particle CSVs produced by an external image analysis. Single user (Chang Woo Yoon).
UI text and replies to the user are Korean; code, commits and docs in English are fine.

## Structure

| Module | Role |
|---|---|
| `core/io.py` | read particle CSVs, guess column roles, `classify_file` for auto-detect import |
| `core/conditions.py` | condition schema + per-group values (blank = not recorded, never 0), CSV merge, fill-from-names helper |
| `core/analysis.py` | size reliability weight, robust z, UMAP + Leiden (`particle_map`, also on a subset = hierarchical map), group profiles, Wasserstein distance, reference score, association |
| `core/project.py` | `.dmap` zip: particles, roles, conditions, settings, ui state, work log |
| `gui/main_window.py` | three panes, menus, auto-detect import / drag & drop, work log, background `Task`, export |
| `gui/side_panels.py` | left group list (visibility, ★ reference, condition filter); shared measurement-weight panel |
| `gui/map_page.py` | particle map: dot unit, colouring, cohesion readout, region selection, hierarchical sub-maps |
| `gui/compare_page.py`, `score_page.py`, `data_page.py`, `conditions_page.py` | one feature each; each exposes `.view` (center) and `.panel` (right) |
| `gui/export_page.py` | export dialog + `write_exports` |
| `selftest.py` | headless end-to-end test on synthetic data (`--selftest`) |

## User rules (keep)

- Layout: left list · center view (only pane that grows) · right settings tabs; pane widths remembered.
- One menu per feature (`탭 열기` + that feature's save/load). No save/load buttons inside panels. `파일` = common things only.
- Conditions are entered by the user; group names are parsed only by the explicit "그룹 이름에서 조건 채우기" button (fills blanks only).
- Shape values are trusted less for small particles (w(d)), never silently.
- Commit and push every change; rebuild with `scripts/build_mac.sh` (installs to /Applications, DMG `Droplet Map-<ver>-<commit>_apple silicon.dmg` on the Desktop).
- Bump `src/droplet_map/__init__.py` `__version__` (and pyproject) for each release; keep `docs/ko`, `docs/en` and this file current.
- Experiment data (`*.csv`, `*.dmap`) never goes into git: the repository is public.
- Add a selftest check for each new feature.
- Worker QThreads need a 64 MB stack (numba); the app binary must handle `-c` (multiprocessing helpers re-launch sys.executable).

## History

| Version | Change |
|---|---|
| 0.1.0 | First app: data, conditions, particle map, comparison, reference score, export |
| 0.2.0 | Three-pane layout and per-feature menus; dot unit (particle/image/group/condition); hierarchical sub-maps; shared weights applied to map and score with cohesion readout; auto-detect import, work log, recent projects, selftest |
| 0.2.1 | Bundle fix: when multiprocessing/joblib helpers re-launch the app binary with `-c`, run the code instead of a hidden second GUI; selftest in build script has a 5 min timeout |
| 0.2.2 | Multi-CSV and save fixes: measurements missing from some files are turned off (with a note) instead of dropping those files' particles; same file not added twice; clashing image names get the file name; ImageJ-style results recognised as particles; group/image names stay text; save writes a temp file then replaces, and shows an error dialog on failure; warning when a group loses all particles |
