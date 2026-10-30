# Droplet Map — notes for Claude

Desktop app (PySide6, macOS Apple Silicon) that maps and compares phase-separated protein droplets
from per-particle CSVs produced by an external image analysis. Since 0.5 the data live in a folder of
batch files (one Excel workbook per purification batch), edited in Excel (the app only creates new
batch files from the template and opens them in Excel); Droplet Map only reads that folder (auto-reload) and the .dmap keeps only the analysis side. Single user (Chang Woo Yoon).
UI text and replies to the user are Korean; code, commits and docs in English are fine.

## Structure

| Module | Role |
|---|---|
| `core/batchfile.py` | batch file model: sheets (batch values / condition table / per particle / per condition), columns with unit + role, hidden `_droplet` layout sheet, visible `_사용법` guide sheet, `[role]` tags in headers, read (calamine, openpyxl fallback) / write (temp + replace), checks, template, copy for a new batch |
| `core/batchset.py` | a folder of batch files → particles, formation conditions (`<batch> · <조건 ID>`), batch table, `MeasureSet` per per-condition sheet; mtime cache so a reload parses only changed files |
| `core/importing.py` | image-analysis CSVs into a batch file: particle CSVs (duplicates by content, odd-format refusal), delivery summaries (one row per group); CSV group → 조건 ID; group names read only on request |
| `core/io.py` | read particle CSVs, guess column roles, `classify_file` (particle vs delivery CSV) |
| `core/conditions.py` | condition schema + per-group values (blank = not recorded, never 0), CSV merge, fill-from-names helper |
| `core/analysis.py` | size reliability weight, robust z, UMAP + Leiden (`particle_map`, also on a subset = hierarchical map), group profiles, Wasserstein distance, reference score, association |
| `core/batches.py` | purification batches above groups: batch table + group → batch membership, migration from the old '정제 배치' condition, merged read-only condition view (`Project.view`) |
| `core/delivery.py` | cell delivery summaries (group × region × parameter), ratios, delivery conditions from names, link to the formation group used |
| `core/batch_map.py` | batch descriptors (typed batch values + trend / matched / level items from particle and delivery groups, in categories), Gower-like batch distance with category weights, PCoA map, axis items, dendrogram, pair item differences, batch score |
| `core/batch_compare.py` | groups paired across batches by identical formation conditions: batch distance, per-measurement delta, reference-batch score |
| `demo.py` | synthetic five-batch folder of batch files (File menu, selftest, tests) |
| `core/project.py` | `Project` built from a `BatchSet`; `.dmap` zip format 4 = folder location (absolute + relative), roles, references, settings, ui state, work log; formats 1–3 converted to batch files in memory (`export_batch_files`) |
| `gui/main_window.py` | three panes, menus, open batch folder / project / demo, folder watcher + debounced reload (keeps the particle result when particle data are unchanged), drag & drop, work log, background `Task`, export |
| `gui/batch_files.py` | app side of batch files: new batch (template or copy) opened in Excel, CSV import dialog (group → 조건 ID) writing into a batch file, check report |
| `gui/side_panels.py` | left batch › group tree (visibility, ★ reference group/batch, condition filter); shared measurement-weight panel |
| `gui/map_page.py` | particle map: dot unit, colouring, cohesion readout, region selection, hierarchical sub-maps, map inputs, within-batch map, per-batch maps |
| `gui/batch_map_page.py` | main view: batch map, per-category maps, distance table, batch dendrogram, reference-batch score; double-click opens the batch's particle map |
| `gui/batch_table_page.py` | read-only batch table (batch-sheet values by category) + computed items table; which kinds are computed, per sheet (`settings.bm_kinds` "sheet:kind") |
| `gui/compare_page.py`, `score_page.py`, `data_page.py`, `batch_page.py` (item differences + matched-condition particle comparison), `measure_page.py` (any per-condition sheet vs formation / sub-conditions, beside the particles of that condition) | one feature each; each exposes `.view` (center) and `.panel` (right) |
| `gui/export_page.py` | export dialog + `write_exports` |
| `selftest.py` | headless end-to-end test on synthetic data (`--selftest`) |

## User rules (keep)

- Layout: left list · center view (only pane that grows) · right settings tabs; pane widths remembered.
- One menu per feature (`탭 열기` + that feature's save/load). No save/load buttons inside panels. `파일` = common things only.
- Batch (purification) is the level above groups; formation and delivery conditions depend on it. Formation-condition effects are judged within batches.
- Batch map is the main feature; particle-level tabs (particle map, condition comparison, reference score, cell delivery) are sub-features under the `입자 수준` menu, closed at start.
- Batch descriptors: particle data enters as within-batch trends (change per condition unit) and matched-condition values, not raw sub-conditions; zeta potential, antibody delivery efficiency etc. are one value per batch. Categories default to 정제 조건 / in vitro 관측 / 세포 전달, user can add more.
- Data entry is separate from analysis and done in Excel (user found in-app editing poor; no editing window): one .xlsx per batch (default sheets 배치 정보 / 형성 조건 / 입자 형태 / in vitro 실험 / 세포 전달, all sheets, columns, conditions and parameters user-editable); "조건" = particle formation condition, linked by 조건 ID; particle sheet holds raw per-particle rows. Droplet Map has no data-entry tabs.
- Conditions are entered by the user; group names are parsed only by explicit buttons in the CSV import dialog (fills blanks only).
- Shape values are trusted less for small particles (w(d)), never silently.
- Commit and push every change; rebuild with `scripts/build_mac.sh` (installs to /Applications, DMG `Droplet Map-<ver>-<commit>_apple silicon.dmg` on the Desktop).
- Bump `src/droplet_map/__init__.py` `__version__` (and pyproject) for each release; keep `docs/ko`, `docs/en` and this file current.
- Experiment data (`*.csv`, `*.dmap`, batch `*.xlsx`) never goes into git: the repository is public.
- Add a selftest check for each new feature.
- Worker QThreads need a 64 MB stack (numba); the app binary must handle `-c` (multiprocessing helpers re-launch sys.executable).

## History

| Version | Change |
|---|---|
| 0.1.0 | First app: data, conditions, particle map, comparison, reference score, export |
| 0.2.0 | Three-pane layout and per-feature menus; dot unit (particle/image/group/condition); hierarchical sub-maps; shared weights applied to map and score with cohesion readout; auto-detect import, work log, recent projects, selftest |
| 0.2.1 | Bundle fix: when multiprocessing/joblib helpers re-launch the app binary with `-c`, run the code instead of a hidden second GUI; selftest in build script has a 5 min timeout |
| 0.2.2 | Multi-CSV and save fixes: measurements missing from some files are turned off (with a note) instead of dropping those files' particles; same file not added twice; clashing image names get the file name; ImageJ-style results recognised as particles; group/image names stay text; save writes a temp file then replaces, and shows an error dialog on failure; warning when a group loses all particles |
| 0.2.3 | Duplicate files detected by content, not name; each file gets a unique source label (`folder/particle_shape.csv`); a file lacking most measurements is left out on its own instead of switching measurements off for every file; one merged note |
| 0.2.4 | Adding files to an open project: a file of another format (e.g. ImageJ) no longer brings its columns in as measurements, so a missing-measurement file is still handled; "different format" is judged against the measurements most particles have; clearer note |
| 0.2.5 | Particle map can leave out measurements and groups (by condition value or the left-list checks) without touching the score/comparison; settings saved in the project |
| 0.3.0 | Batch hierarchy: batch table with batch conditions, every group belongs to a batch (asked on import), left tree batch › groups; formation-condition effects judged within batches; batch comparison tab (paired by identical conditions, reference-batch score); within-batch and per-batch maps; cell delivery summaries (inside particle / cytoplasm intensity, ratio) linked to the formation group used; project format 2 with migration |
| 0.4.0 | Batch map as the main feature: batch descriptors in categories (typed values, within-batch trends, matched-condition values, medians), batch distance with category weights, PCoA map with axis items, batch dendrogram, reference-batch score, item differences between two batches; batch table tab; particle level moved to its own menu; synthetic five-batch demo project; project format 3 |
| 0.5.0 | Data entry split from analysis: one Excel batch file per batch in a batch folder, batch note window (sheets / columns / roles / units editable, Excel paste, undo, CSV import with group → condition mapping, checks, template, new batch from template or copy), Droplet Map reads the folder and reloads on change; measurement-sheet tab for any per-condition sheet; batch table read-only with per-sheet computed kinds; group-condition and delivery tabs removed; project format 4 (old projects convert) |
| 0.5.1 | Editing moved to Excel: batch note window removed; 새 배치 writes a file with the default sheets and parameters and opens it in Excel; open-in-Excel menu, batch file check report, CSV import into a batch file; `_사용법` guide sheet and `[role]` header tags; duplicate CSV detection survives the Excel round trip |
