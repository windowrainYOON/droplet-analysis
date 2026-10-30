# Droplet Map usage (0.5)

From 0.5 **data entry and analysis are separate.** Batch information and experiment data live in **one Excel workbook per purification batch** (a batch file); Droplet Map reads the **batch folder** holding those files and only analyses. Batch files are edited in the in-app **batch note** window or in Excel; on save Droplet Map reads them again.

## Batch files (one batch = one .xlsx)
Five default sheets; sheets, columns and items can all be added, changed and removed. A "condition" is a particle formation condition, and every sheet links to the condition table by **조건 ID** (condition ID).

| Sheet | Shape | Content |
|---|---|---|
| 배치 정보 | batch values | item · value · unit · category (구분) · note. One value per batch (name, purification date, column, protein concentration, zeta potential …). The category is the batch-map category. |
| 형성 조건 | condition table | one row per condition ID (C01 …): pH, NaCl (mM), PEG (%), protein (µM) … |
| 입자 형태 | per particle | one row per particle from the image analysis |
| in vitro 실험 | per condition | repeated measurements per condition (FRAP recovery, FRAP t½, turbidity …) |
| 세포 전달 | per condition | one row per delivery run: time and antibody dose (sub-conditions), cytoplasm / inside-particle signal, ratio (filled when blank) |

- Headers are `name (unit)`. Column roles: condition · measurement · sub-condition (time, dose) · replicate · note.
- Sheet shapes, categories and roles are kept in a hidden `_droplet` sheet; without it (a file made in Excel) they are guessed from names.
- Blanks are "not recorded", never zero.
- `_템플릿.xlsx` in the folder is the starting layout for new batches. Files starting with `_` or `~$` are not batches.

## Getting started
- `File › New batch folder` (⌘N), `File › Open batch folder` (⌘O); folders, .xlsx and .dmap files can be dropped on the window.
- `File › Open demo (5 batches)`: a synthetic batch folder (made-up data).
- Projects from before 0.5 (data inside the .dmap) still open; `File › Export as batch files` writes a batch folder to edit from then on.

## Batch note (⌘E)
- Left: batch files and the 📐 template. Center: one tab per sheet (edit like Excel, ⌘C / ⌘V tables from Excel, Delete clears). Right: sheet, column and row tools, CSV import, checks.
- New batch from the template or as a copy of another batch (layout, conditions and batch values kept, measurements emptied).
- CSV import: particle CSVs → the per-particle sheet, cell delivery summaries → the delivery sheet; each CSV group is assigned a condition ID. Group names are read only on the buttons "find from group names" (blanks only) and "make new conditions from names". Repeated data and files with mostly different measurements are refused.
- Checks: duplicate or unknown condition IDs, non-numbers in measurement columns; cells are shown in red.
- ⌘S saves, ⌘Z undoes. Changes saved from Excel are picked up automatically (changed files only).

## Droplet Map
- **Left – batches › formation conditions**; **center**: Batch map · Batch table (read-only) · Batch comparison · Data; **particle level** (closed at start): Particle map · Condition comparison · Reference score · Measurement sheets; **right**: settings and measurement weights.
- **Batch table**: batch values from the batch sheet plus items computed per sheet — trend (change per condition unit within a batch, also per sub-condition), matched (value at a condition set run by two or more batches), batch median (off for particles by default); each sheet is a category, and the three kinds can be switched per sheet.
- **Batch map**: Gower-like distance over shared items, averaged per part, per category and with category weights; PCoA map with axis items, per-category maps, distance table, dendrogram, reference-batch score.
- **Batch comparison**: item differences, and at particle level groups paired by identical conditions.
- **Measurement sheets**: any per-condition sheet against a formation condition or a sub-condition, one line per batch, a summary table, and the particles of the same condition side by side.
- The particle analysis (⌘R) must be re-run when particle data change; changes to other sheets keep the result.

## Saving
The data are in the batch files. The project file (.dmap) holds only the analysis side: the folder location, column roles, references, map settings, weights, open tabs and the work log; if the folder moved you are asked for it. Per-feature saves are in each menu; `File › Save all results` (⇧⌘E) writes tables, measurement-sheet summaries, figures and the project into one folder.
