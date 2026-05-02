# MissVARPath — Missense Variant Pathogenicity Detection

A meta-classifier for missense variant pathogenicity built on **ClinVar** labels and **OpenCRAVAT** annotations. Ten model families (classical ML, gradient boosting, and deep learning) are benchmarked under a unified stratified 80/20 + 5-fold CV protocol on both the original 4-class task and a collapsed binary task.

CMP682 course project — Doruk Topçu & Alihan. See [project proposal](CMP682%20-%20Project%20Proposal%20-%20Doruk%20%26%20Alihan%20-%20MVP%20%281%29.pdf) for the full plan and [project_progress_report.md](project_progress_report.md) for the running lab notebook.

---

## Dataset

- **Source:** ~63k ClinVar entries → annotated with OpenCRAVAT (158 annotator groups: `clinvar`, `gnomad4`, `alphamissense`, `cadd`, `revel`, `sift`, `phylop`, …) → down-sampled to a class-balanced subset.
- **Shape:** `data/missense_dataset.csv` — **21,872 rows × 777 columns**.
- **Label:** `clinvar__sig` ∈ {Benign, Likely benign, Likely pathogenic, Pathogenic}, 5,468 each (perfectly balanced).
- **After preprocessing:** **320 features** (identifier columns dropped, missing values imputed; see [src/preprocessing.py](src/preprocessing.py)).

Two task formulations are supported:

| Task | Classes |
| --- | --- |
| `4class` | Benign / Likely benign / Likely pathogenic / Pathogenic |
| `2class` | Benign+Likely benign vs. Pathogenic+Likely pathogenic |

---

## Repo layout

```
src/
  config.py          paths, label encoding, RNG seeds
  data_loader.py     CSV → DataFrame
  eda.py             EDA driver (figures + summary tables)
  preprocessing.py   column drops, NA handling, imputation, label encoding
  models.py          model zoo (sklearn + PyTorch)
  torch_models.py    CNN1D / LSTM / RNN definitions + TorchClassifier wrapper
  train.py           80/20 stratified split + 5-fold CV runner
  utils.py           plotting, metric reporting

data/                missense_dataset.csv
outputs/
  eda/               EDA figures + summary CSVs/JSON
  preprocessing/     missense_processed.parquet, final_features.txt, dropped_columns.txt
  reports/<task>/    per-model classification report (md), confusion matrices (png),
                     metrics (json), and the leaderboard.csv
```

---

## Models

Ten model families, run end-to-end by [src/train.py](src/train.py):

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. XGBoost
5. LightGBM
6. CatBoost
7. Shallow MLP (1 hidden layer)
8. CNN 1D (PyTorch)
9. LSTM (PyTorch)
10. RNN (PyTorch)

All exposed through a single `ModelSpec` interface in [src/models.py](src/models.py) so the training loop is family-agnostic.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the curated dataset at `data/missense_dataset.csv`.

---

## Usage

```bash
# 1. Exploratory data analysis  → outputs/eda/
python -m src.eda

# 2. Preprocessing (writes processed parquet + feature list)
python -m src.preprocessing

# 3. Train + evaluate all 10 models
python -m src.train --task 4class
python -m src.train --task 2class

# Optional: subset of models
python -m src.train --task 4class --models RandomForest LightGBM XGBoost
```

Each training run produces, per model and per task:

- `outputs/reports/<task>/<model>_classification_report.md`
- `outputs/reports/<task>/<model>_confusion_matrix.png` (counts + row-normalized)
- `outputs/reports/<task>/<model>_metrics.json` (per-fold CV + holdout)
- `outputs/reports/<task>/leaderboard.csv` (aggregated, sorted by holdout macro-F1)

Reproducibility: stratified split with `random_state=42`, `N_SPLITS=5`, `TEST_SIZE=0.2` ([src/config.py](src/config.py)).

---

## Results

### 4-class — held-out test set (sorted by macro-F1)

| Model | Holdout Acc | Macro-F1 | MCC | ROC-AUC (OvR macro) | Fit (s) |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 0.8011 | 0.8013 | 0.7353 | 0.9481 | 47.5 |
| XGBoost | 0.7977 | 0.7977 | 0.7308 | 0.9492 | 18.4 |
| RandomForest | 0.7904 | 0.7904 | 0.7210 | 0.9455 | 11.0 |
| ExtraTrees | 0.7870 | 0.7867 | 0.7165 | 0.9434 | 2.4 |
| CatBoost | 0.7870 | 0.7866 | 0.7166 | 0.9453 | 15.6 |
| LogisticRegression | 0.7568 | 0.7572 | 0.6766 | 0.9346 | 13.7 |
| ShallowNN_MLP | 0.7474 | 0.7470 | 0.6649 | 0.9347 | 4.4 |
| CNN1D | 0.7179 | 0.7160 | 0.6254 | 0.9194 | 15.1 |
| RNN | 0.6667 | 0.6234 | 0.5860 | 0.8968 | 242.5 |
| LSTM | 0.6213 | 0.6131 | 0.4995 | 0.8841 | 64.9 |

### 2-class — held-out test set (sorted by macro-F1)

| Model | Holdout Acc | Macro-F1 | MCC | Fit (s) |
| --- | --- | --- | --- | --- |
| CatBoost | 0.9895 | 0.9895 | 0.9790 | 7.5 |
| XGBoost | 0.9881 | 0.9881 | 0.9763 | 4.7 |
| LightGBM | 0.9881 | 0.9881 | 0.9763 | 11.4 |
| LogisticRegression | 0.9867 | 0.9867 | 0.9736 | 2.6 |
| RandomForest | 0.9861 | 0.9861 | 0.9721 | 9.9 |
| ExtraTrees | 0.9856 | 0.9856 | 0.9712 | 1.1 |
| ShallowNN_MLP | 0.9851 | 0.9851 | 0.9703 | 7.9 |
| CNN1D | 0.9819 | 0.9819 | 0.9639 | 14.0 |
| RNN | 0.9429 | 0.9429 | 0.8859 | 206.2 |
| LSTM | 0.9131 | 0.9131 | 0.8263 | 63.3 |

**Takeaways:** Gradient-boosted trees (LightGBM / XGBoost / CatBoost) lead on both tasks. The 2-class collapse is near-saturated (>98% across most families), while the 4-class task — distinguishing *likely* from definitive calls — is the harder, more informative benchmark. Sequence-style PyTorch models (LSTM/RNN) underperform on this tabular feature set, as expected.

Full per-model classification reports and confusion matrices live under [outputs/reports/4class/](outputs/reports/4class) and [outputs/reports/2class/](outputs/reports/2class).

---

## Reproducing the leaderboard

```bash
python -m src.preprocessing
python -m src.train --task 4class
python -m src.train --task 2class
```

Logs are written to `outputs/reports/train_<task>.log`.
