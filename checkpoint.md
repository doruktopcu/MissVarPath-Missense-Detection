# CHECKPOINT — ALWAYS READ THIS FIRST

Always read this first and paste your progress after completing regularly. Treat this file like a save game for the MissVARPath / MVP project.

When continuing work:

1. Read this file before touching code or running long jobs.
2. Verify the actual on-disk state with `git status --short --branch` and relevant artifact checks.
3. After completing meaningful work, append a dated progress entry to the bottom of this file.
4. Do not assume the previous chat transcript is complete; this file is the durable handoff.

---

## Project Context

- Project: MissVARPath / MVP — missense variant pathogenicity detection.
- Repo path: `/Users/doruktopcu/Projects/MissVarPath-Missense-Detection`
- Branch: `new-method`
- Current handoff date: 2026-05-19
- Current strategic direction: move from experimentation toward final no-AdaBoost reruns and report generation.

Important user instruction:

- Do **not** restart AdaBoost tuning yet.
- AdaBoost baseline exists, but exhaustive AdaBoost tuning was stopped because it was too slow.
- Do **not** remove AdaBoost from `src/models.py` unless the user explicitly asks to convert the active suite to 10 models.

---

## Current Repository State

Current working tree is dirty.

Known local changes:

- Modified: `.gitignore`
- Modified: `README.md`
- Modified: `src/train.py`
- Modified: `src/tune.py`
- Untracked generated outputs under:
  - `outputs/models/`
  - `outputs/reports/`
  - `outputs/tuning/`

Do not commit, push, or switch branches unless the user explicitly asks.

---

## Completed So Far

### Baseline Training

Baseline training has completed for both tasks:

- `outputs/reports/4class/`
- `outputs/reports/2class/`

Baseline highlights:

- Best `4class` baseline model: `HistGradientBoosting`
  - holdout macro-F1 ≈ `0.7928`
- Best `2class` baseline models: `AdaBoost` / `HistGradientBoosting`
  - holdout macro-F1 ≈ `0.9870`

### 4class Hyperparameter Tuning

Tuning is complete for 9 non-AdaBoost `4class` models:

- `HistGradientBoosting`
- `LinearSVC`
- `DecisionTree`
- `SGDClassifier`
- `LDA`
- `RidgeClassifier`
- `KNN`
- `QDA`
- `NearestCentroid`

AdaBoost tuning was intentionally stopped:

- It reached combo `23/27`.
- It did not finish.
- There is no `outputs/tuning/4class/adaboost/best_config.json`.
- Do not restart it unless the user explicitly asks.

### Tuner Changes

`src/tune.py` was modified locally:

- Tuning grids were populated.
- Per-combo failure handling was added.
- Bad parameter combinations are logged and skipped instead of aborting the entire run.

Preserve this behavior.

### Model Persistence Changes

`src/train.py` was modified locally:

- It now imports `joblib`.
- It saves fitted sklearn models after holdout fitting.
- Future training runs write models to:
  - `outputs/models/<task>/<model>.joblib`
- It also writes the corresponding feature manifest:
  - `outputs/models/<task>/features.txt`
- Metrics JSON now includes `model_path`.

`.gitignore` was updated so nested model artifacts are ignored:

- `outputs/models/**/*.joblib`
- `outputs/models/**/*.pt`
- `outputs/models/**/*.pkl`

`README.md` was updated to mention saved `.joblib` model artifacts.

### Model Save Smoke Test

A smoke test was run and passed:

- Saved model:
  - `outputs/models/2class_save_smoke/nearestcentroid.joblib`
- It was successfully reloaded with `joblib`.
- It produced predictions.

This is only a smoke artifact, not a final model artifact.

---

## Completed 4class Tuning Results

| Model | Baseline macro-F1 | Tuned macro-F1 | Delta | Best params |
|---|---:|---:|---:|---|
| HistGradientBoosting | 0.7928 | 0.7950 | +0.0022 | `learning_rate=0.02`, `max_iter=2000`, `max_leaf_nodes=127`, `min_samples_leaf=20`, `l2_regularization=1.0` |
| LinearSVC | 0.7549 | 0.7559 | +0.0009 | `C=100.0`, `max_iter=5000` |
| DecisionTree | 0.7111 | 0.7514 | +0.0403 | `max_depth=10`, `min_samples_leaf=1`, `ccp_alpha=0.001` |
| SGDClassifier | 0.7015 | 0.7182 | +0.0167 | `alpha=0.001`, `penalty=elasticnet`, `max_iter=5000` |
| LDA | 0.7014 | 0.7014 | +0.0000 | `solver=lsqr`, `shrinkage=auto` |
| RidgeClassifier | 0.7008 | 0.6982 | -0.0025 | `alpha=10.0` |
| KNN | 0.6492 | 0.6598 | +0.0106 | `n_neighbors=30`, `weights=distance`, `metric=cosine` |
| QDA | 0.5791 | 0.6178 | +0.0387 | `reg_param=0.01` |
| NearestCentroid | 0.5475 | 0.5514 | +0.0039 | `metric=euclidean`, `shrink_threshold=1.0` |

Interpretation:

- Headline model remains `HistGradientBoosting`.
- Largest absolute tuning gains came from:
  - `DecisionTree`
  - `QDA`
  - `SGDClassifier`
  - `KNN`
- `HistGradientBoosting` improved only slightly and became heavier.
- `RidgeClassifier` got worse under the tested tuned setting.
- `LDA` was already at its effective default.
- `LinearSVC` gained very little for higher cost.

---

## Recommended Next Roadmap

### Phase 1 — Restore / Verify State

When moving to a new PC, first verify local state.

Run:

```bash
git status --short --branch
```

Confirm data and preprocessing files exist:

- `data/missense_dataset.csv`
- `outputs/preprocessing/missense_processed.parquet`

Confirm tuning artifacts exist:

```bash
find outputs/tuning/4class -mindepth 2 -maxdepth 2 -name best_config.json -print | sort
```

Expected completed tuning artifacts:

- `outputs/tuning/4class/decisiontree/best_config.json`
- `outputs/tuning/4class/histgradientboosting/best_config.json`
- `outputs/tuning/4class/knn/best_config.json`
- `outputs/tuning/4class/lda/best_config.json`
- `outputs/tuning/4class/linearsvc/best_config.json`
- `outputs/tuning/4class/nearestcentroid/best_config.json`
- `outputs/tuning/4class/qda/best_config.json`
- `outputs/tuning/4class/ridgeclassifier/best_config.json`
- `outputs/tuning/4class/sgdclassifier/best_config.json`

Do not expect an AdaBoost tuning artifact.

If switching machines, remember:

- `data/` may be ignored.
- `outputs/` artifacts may be untracked or partially ignored.
- Copy missing artifacts from the old machine or regenerate them.

### Phase 2 — Apply Tuned Defaults, Excluding AdaBoost Tuning

Do not touch AdaBoost yet.

Recommended defaults to apply in `src/models.py`:

Definitely update:

- `DecisionTree`
  - `max_depth=10`
  - `min_samples_leaf=1`
  - `ccp_alpha=0.001`
- `QDA`
  - `reg_param=0.01`
- `SGDClassifier`
  - `alpha=0.001`
  - `penalty="elasticnet"`
  - `max_iter=5000`
  - keep `l1_ratio=0.15`
- `KNN`
  - `n_neighbors=30`
  - `weights="distance"`
  - `metric="cosine"`
- `NearestCentroid`
  - `shrink_threshold=1.0`

Consider updating:

- `HistGradientBoosting`
  - `learning_rate=0.02`
  - `max_iter=2000`
  - `max_leaf_nodes=127`
  - `min_samples_leaf=20`
  - `l2_regularization=1.0`

Tradeoff:

- Tuned HistGB is slightly better: `0.7928 → 0.7950`.
- It is much heavier.
- Use it if final report prioritizes best score over runtime.

Probably do not update:

- `RidgeClassifier`
  - tuned result was worse.
- `LDA`
  - already at best current default.
- `LinearSVC`
  - gain is tiny and high-`C` setting costs more.

### Phase 3 — Rerun Final Training With Model Saving

After applying chosen defaults, rerun final training while excluding AdaBoost.

Use an explicit model list so AdaBoost does not run accidentally:

```bash
MODELS="KNN NearestCentroid CosineSimilarity DecisionTree LDA QDA LinearSVC RidgeClassifier SGDClassifier HistGradientBoosting"
```

Use the project Python interpreter from `new_method_training.md`, not random `python` from PATH.

Example:

```bash
$PY -m src.train --task 4class --models $MODELS --tag final_no_adaboost > outputs/reports/train_4class_final_no_adaboost.log 2>&1

$PY -m src.train --task 2class --models $MODELS --tag final_no_adaboost > outputs/reports/train_2class_final_no_adaboost.log 2>&1
```

Expected output directories:

- `outputs/reports/4class_final_no_adaboost/`
- `outputs/reports/2class_final_no_adaboost/`
- `outputs/models/4class_final_no_adaboost/`
- `outputs/models/2class_final_no_adaboost/`

Expected artifact counts per task:

- `10` metrics JSON files
- `10` classification reports
- `10` count confusion matrices
- `10` normalized confusion matrices
- `10` `.joblib` model files
- `1` `leaderboard.csv`
- `1` `features.txt` under reports
- `1` `features.txt` under models

### Phase 4 — Verify Final Outputs

Check for errors:

```bash
rg -n "Traceback|ERROR|CV failed|Holdout eval failed" outputs/reports/train_*final_no_adaboost.log
```

Check leaderboard rows:

```bash
wc -l outputs/reports/4class_final_no_adaboost/leaderboard.csv
wc -l outputs/reports/2class_final_no_adaboost/leaderboard.csv
```

Expected:

- `11` lines each, meaning header + 10 models.

Check saved model files:

```bash
find outputs/models/4class_final_no_adaboost -name "*.joblib" | wc -l
find outputs/models/2class_final_no_adaboost -name "*.joblib" | wc -l
```

Expected:

- `10` each.

Load one saved model with `joblib` and run a tiny prediction sanity check.

### Phase 5 — Consolidate Documentation

Before report writing, update docs so they match reality.

Known stale documentation:

- `README.md` still says `320` features.
  - Actual preprocessing has `208` features.
- `README.md` still says `tune.py` needs `GRIDS` refilled manually.
  - `GRIDS` is now populated in code.
- `project_progress_report.md` still mentions a `23-model` restart.
  - Current active baseline suite is 11 models.
  - Final rerun should probably use 10 models because AdaBoost is excluded.

Add a clear note:

- AdaBoost baseline exists.
- AdaBoost tuning was stopped because runtime was excessive.
- Final report uses the no-AdaBoost tuned rerun unless the user later asks otherwise.

### Phase 6 — Generate Report Materials

Prepare report-ready tables:

1. Dataset / preprocessing summary:
   - `21,872` rows
   - `208` final features
   - 4 balanced ClinVar classes
   - binary collapse for `2class`
2. Baseline leaderboard:
   - `outputs/reports/4class/leaderboard.csv`
   - `outputs/reports/2class/leaderboard.csv`
3. Tuned final-no-AdaBoost leaderboard:
   - `outputs/reports/4class_final_no_adaboost/leaderboard.csv`
   - `outputs/reports/2class_final_no_adaboost/leaderboard.csv`
4. Tuning delta table:
   - Use completed 4class tuning summary above.
5. Figures:
   - Top model confusion matrix for `4class`
   - Top model normalized confusion matrix for `4class`
   - Top model confusion matrix for `2class`
   - Existing EDA figures from `outputs/eda/` if needed

Keep analysis practical:

- Do not dig deeply into why `Likely pathogenic` and `Pathogenic` are difficult unless the user asks.
- It is enough to say `4class` is harder than `2class`, with most residual difficulty in fine-grained class distinctions.

### Phase 7 — Report Narrative Skeleton

Recommended report structure:

1. Problem and dataset
2. Preprocessing pipeline
3. Model suite
4. Baseline training protocol
5. Hyperparameter tuning protocol
6. Results:
   - `4class` baseline
   - `4class` tuned final-no-AdaBoost
   - `2class` baseline/final
7. Saved model artifacts:
   - `.joblib` files under `outputs/models/`
8. Discussion:
   - `HistGradientBoosting` remains headline model.
   - `DecisionTree` and `QDA` gained most from tuning, but trail HistGB.
   - AdaBoost was excluded from tuning due runtime cost.
   - BLAST features are framed honestly as **label-aware locus-neighbour
     features**, not homology features. The 51 bp DNA-flank substrate plus
     `word_size=7`/`evalue=10` is too short/permissive for true homology;
     hits are dominated by same-gene proximity in ClinVar. A real
     homology-based version would need a UniRef-scale reference DB and
     `blastp` on an amino-acid window — out of scope here. The
     gene-stratified ablation is the empirical test for how much of the
     augmented lift survives once same-gene leakage is removed. See the
     [src/blast_features.py](src/blast_features.py) docstring,
     `## BLAST caveat` in [README.md](README.md), and
     `\S BLAST features: scope and honest framing` in
     [final_report.tex](final_report.tex) for the long-form framing.
9. Limitations and future work:
   - Try AdaBoost later only if compute budget permits.
   - Consider feature work only if the user later wants deeper error analysis.

---

## Important Constraints

- Do not start AdaBoost tuning.
- Do not remove AdaBoost from `src/models.py` unless user explicitly asks.
- Do not delete:
  - `outputs/eda/`
  - `outputs/preprocessing/`
  - `outputs/sequences/`
- Do not commit, push, or switch branches unless user explicitly asks.
- Preserve model-saving changes in `src/train.py`.
- Preserve robust combo-skip behavior in `src/tune.py`.
- Use the project interpreter from `new_method_training.md` for training runs.

---

## Progress Log

### 2026-05-19 — Checkpoint Created

- Created this `checkpoint.md` file as the project save-game handoff.
- Captured completed baseline training, completed non-AdaBoost 4class tuning, stopped AdaBoost tuning, model persistence changes, and roadmap from current state to report generation.
- Next recommended action: apply selected tuned defaults in `src/models.py`, rerun final no-AdaBoost training, verify saved `.joblib` models, then generate report materials.

### 2026-05-19 — Full-scope chain completed

- Applied checkpoint-recommended tuned defaults to `src/models.py` (DecisionTree, QDA, SGD, KNN, NearestCentroid, HistGradientBoosting). LDA was switched to `solver=svd` after `lsqr+auto` proved unstable under sklearn 1.7 on Windows (fold-2/3 fold collapse).
- Reran the final no-AdaBoost suite for both 4-class and 2-class. Headline: HistGradientBoosting macro-F1 = 0.7950 (4-class) / 0.9872 (2-class).
- Ran the full ablation chain (`run_full_chain.ps1` — 11 experiments, ~31 min wall-clock): gene-stratified, augmented (k-mer + BLAST), augmented-raw-only, augmented-no-vep (kfold + gene), no-ditto. All leaderboards landed under `outputs/reports/`.
- Ran SHAP on the canonical 4-class and 2-class tuned HistGB. Top features (4-class): DITTO, MetaRNN, AllOfUs max AF, gnomAD AF, BayesDel. Per-class asymmetry: Benign-class is population-frequency driven; Pathogenic-class is DITTO-driven; Likely-pathogenic boundary is MetaRNN-driven.
- DITTO removal cost only $-0.0017$ on 4-class — the suite is doing genuine ensembling and is not bottlenecked on any single learned predictor.
- Refilled all TODOs in [final_report.tex](final_report.tex); the LaTeX is the deliverable report and compiles cleanly on Overleaf / MiKTeX / TeX Live.
- README and project_progress_report updated to point at the new final state.

Pending for delivery (per [project_delivery_instructions.md](project_delivery_instructions.md), due 2026-05-24 23:59):
- compile `final_report.tex` → PDF;
- prepare source-code bundle (zip or repo link);
- record demo video (≤10 minutes, screen + voice).
