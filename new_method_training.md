# New-Method Training Playbook

A step-by-step playbook for running the **MissVARPath / MVP** 11-model fast-baseline suite on both classification tasks (4-class and 2-class) and saving all leaderboard artifacts. Follow the steps in order. Do not deviate.

---

## 1. Goal

Train and evaluate every model in `src/models.py` → `MODEL_SPECS` on the `data/missense_dataset.csv` ClinVar / OpenCRAVAT dataset under the standard 80/20 stratified split + 5-fold cross-validation protocol, for both the 4-class and 2-class targets. Save:

- per-model classification report (markdown)
- per-model confusion matrix (counts + row-normalized PNGs)
- per-model metrics JSON (per-fold CV + holdout)
- aggregated `leaderboard.csv` per task, sorted by held-out macro-F1

All outputs land under `outputs/reports/<task>/`. Logs go to `outputs/reports/train_<task>.log`.

---

## 2. Working directory & Python interpreter

- **Project root:** `/Users/doruktopcu/Projects/MissVarPath-Missense-Detection`
- **Git branch:** `new-method` (do not switch off this branch)
- **Python interpreter — use this exact path, not `python` from `$PATH`:**

  ```
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
  ```

  The pyenv `python` at `/Users/doruktopcu/.pyenv/shims/python` is **Python 3.11.9 and does NOT have the project dependencies installed**. If you use that one everything will fail with `ModuleNotFoundError`.

  Convenience: `export PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` at the top of your shell and use `$PY ...` thereafter.

---

## 3. Active model suite (11 models — do not modify)

| # | Model | Family | Notes |
|---|---|---|---|
| 1 | KNN | classical | k=15, distance-weighted, standard-scaled |
| 2 | NearestCentroid | classical | Euclidean, standard-scaled |
| 3 | CosineSimilarity | classical | Custom class-centroid classifier using cosine distance (defined in `src/models.py` as `CosineSimilarityClassifier`) |
| 4 | DecisionTree | classical | Full-depth, class-balanced |
| 5 | LDA | classical | solver=lsqr, shrinkage=auto |
| 6 | QDA | classical | reg_param=0.1 |
| 7 | LinearSVC | classical | C=1.0, class-balanced, max_iter=5000 |
| 8 | RidgeClassifier | classical | alpha=1.0, class-balanced |
| 9 | SGDClassifier | classical | log-loss, early stopping |
| 10 | AdaBoost | ensemble | 200 stumps, lr=0.5 |
| 11 | HistGradientBoosting | ensemble | max_iter=600, leaves=63, lr=0.05 |

Verify the registry before running:

```bash
$PY -c "from src.models import MODEL_SPECS; print(len(MODEL_SPECS)); [print(s.name) for s in MODEL_SPECS]"
```

**Expected output: `11` followed by the 11 names above in that order.** If it does not match, STOP and report to the user — do not proceed.

---

## 4. Prerequisites

### 4.1 Data must exist

```bash
ls -la data/missense_dataset.csv
ls -la outputs/preprocessing/missense_processed.parquet
```

Both must exist. If `data/missense_dataset.csv` is missing, STOP and tell the user — the raw dataset cannot be regenerated automatically.

If `outputs/preprocessing/missense_processed.parquet` is missing, regenerate it:

```bash
$PY -m src.preprocessing
```

### 4.2 Dependencies must import

```bash
$PY -c "import numpy, pandas, sklearn, matplotlib, joblib, tqdm, pyarrow; print('deps ok')"
```

Must print `deps ok`. If anything is missing, run `$PY -m pip install -r requirements.txt` and re-check.

### 4.3 No stale outputs

If `outputs/reports/` already contains a `4class/` or `2class/` directory from a previous run, **delete them first** to avoid mixing old and new artifacts:

```bash
rm -rf outputs/reports/4class outputs/reports/2class outputs/reports/train_4class.log outputs/reports/train_2class.log
mkdir -p outputs/reports
```

The other `outputs/` subdirectories (`eda/`, `preprocessing/`, `sequences/`) are **dataset-derived and must NOT be deleted**.

---

## 5. Run the training (two sequential commands)

### 5.1 4-class task

```bash
$PY -m src.train --task 4class > outputs/reports/train_4class.log 2>&1
```

- **Expected wall-clock:** 8–12 minutes total.
- **Expected per-model timings (approximate, /fold):**
  - KNN, NearestCentroid, CosineSimilarity, LDA, RidgeClassifier, SGDClassifier: ≤0.5s
  - QDA: ~1s
  - DecisionTree: ~3s
  - HistGradientBoosting: ~15s
  - LinearSVC: ~20s
  - AdaBoost: ~30s

Anything more than ~60s/fold for any of the listed models is a red flag — capture the log line and report it.

### 5.2 2-class task

Only run this **after 4class has completed successfully** (see verification in §6):

```bash
$PY -m src.train --task 2class > outputs/reports/train_2class.log 2>&1
```

- **Expected wall-clock:** similar or slightly faster than 4class.

### 5.3 (Optional) Chain both in one shell invocation

If you prefer not to babysit:

```bash
$PY -m src.train --task 4class > outputs/reports/train_4class.log 2>&1 \
  && $PY -m src.train --task 2class > outputs/reports/train_2class.log 2>&1 \
  && echo "ALL TRAINING DONE"
```

The trailing `&& echo` is what you grep for to confirm both completed.

---

## 6. Verification (run after each task)

### 6.1 Log shows leaderboard

```bash
tail -25 outputs/reports/train_4class.log     # repeat for train_2class.log
```

Look for a line that begins `INFO train: Leaderboard:` followed by a pandas-style table covering all 11 models. If any model is missing, that means it errored — search the log:

```bash
grep -nE "Traceback|ERROR|CV failed|Holdout eval failed" outputs/reports/train_4class.log
```

If you find errors, capture the offending model name and the traceback, and continue with the surviving models — do NOT abort the whole run unless every model failed.

### 6.2 Leaderboard CSV exists

```bash
ls outputs/reports/4class/leaderboard.csv
ls outputs/reports/2class/leaderboard.csv
head -1 outputs/reports/4class/leaderboard.csv     # column header
wc -l outputs/reports/4class/leaderboard.csv       # should be 12: header + 11 rows
```

### 6.3 Per-model artifacts present

For each task, every model should produce four files. Quick sanity:

```bash
for task in 4class 2class; do
  echo "== $task =="
  ls outputs/reports/$task/*_metrics.json | wc -l            # expect 11
  ls outputs/reports/$task/*_classification_report.md | wc -l # expect 11
  ls outputs/reports/$task/*_confusion_matrix.png | wc -l    # expect 11
  ls outputs/reports/$task/*_confusion_matrix_normalized.png | wc -l # expect 11
done
```

Each count must be 11. Anything less means a model errored and didn't write artifacts.

### 6.4 Spot-check a metric

The holdout macro-F1 for HistGradientBoosting on 4class should be roughly in the 0.78–0.82 band (this is the empirical band from prior runs on the same data with similar tree models). Outside that band, flag it.

```bash
$PY -c "
import pandas as pd
df = pd.read_csv('outputs/reports/4class/leaderboard.csv')
print(df[['model','holdout_macro_f1','fit_seconds']].to_string(index=False))
"
```

---

## 7. Hand-back to the user

Once both leaderboards exist, post:

1. The full leaderboard for **4class** (cv_macro_f1_mean ± std, holdout macro-F1, holdout MCC, fit_seconds), sorted by holdout macro-F1 descending.
2. The full leaderboard for **2class**, same columns.
3. Total wall-clock for each task (compute from the first and last timestamps in each log).
4. Any model that errored or fell outside the expected timing band (§5).

Markdown table format is fine. Pull the data directly from `outputs/reports/<task>/leaderboard.csv`.

Example one-shot to generate the markdown:

```bash
$PY -c "
import pandas as pd
for t in ['4class','2class']:
    df = pd.read_csv(f'outputs/reports/{t}/leaderboard.csv').sort_values('holdout_macro_f1', ascending=False)
    cols = ['model','cv_macro_f1_mean','cv_macro_f1_std','holdout_acc','holdout_macro_f1','holdout_mcc','fit_seconds']
    print(f'### {t}')
    print(df[cols].to_markdown(index=False))
    print()
"
```

---

## 8. Failure handling

| Symptom | Likely cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Wrong Python interpreter | Use the explicit `/Library/Frameworks/...` path from §2 |
| `FileNotFoundError: missense_processed.parquet` | Preprocessing not run | `$PY -m src.preprocessing` then retry §5 |
| One model traceback in the log | Single-model issue (e.g. convergence warning escalated) | Continue. Report the model name in your hand-back; do not retry |
| All models fail with the same traceback | Likely a data/import problem | STOP. Report the traceback verbatim |
| Run exceeds 30 minutes for one task | Something is wrong (probably swap or interpreter pinned to wrong env) | Kill (`pkill -f "src.train"`), report state, do not auto-retry |
| `leaderboard.csv` written but empty | Every model errored before holdout | STOP. Report the log |

Do not `git commit` anything. Do not modify `src/models.py` or any other source file. Do not change `RANDOM_STATE`. Do not delete `outputs/eda/`, `outputs/preprocessing/`, or `outputs/sequences/`.

---

## 9. Reproducibility constants (do not change)

These live in `src/config.py` and must be preserved:

- `RANDOM_STATE = 42`
- `N_SPLITS = 5`
- `TEST_SIZE = 0.2`

If you find these changed in `src/config.py`, STOP and report.
