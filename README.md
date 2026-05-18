# MissVARPath — Missense Variant Pathogenicity Detection

A meta-classifier for missense variant pathogenicity built on **ClinVar** labels and **OpenCRAVAT** annotations. An 11-model fast-baseline suite (classical ML + two fast ensembles) is benchmarked under a unified stratified 80/20 + 5-fold CV protocol on both the original 4-class task and a collapsed binary task.

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
  models.py          model zoo — MODEL_SPECS registry
  train.py           80/20 stratified split + 5-fold CV runner
  tune.py            grid-search hyperparameter tuner (refill GRIDS per run)
  shap_explain.py    SHAP attributions per model
  utils.py           plotting, metric reporting

data/                missense_dataset.csv
outputs/
  eda/               EDA figures + summary CSVs/JSON
  preprocessing/     missense_processed.parquet, final_features.txt, dropped_columns.txt
  models/<task>/     fitted sklearn estimators (.joblib) + feature manifests
  reports/<task>/    per-model classification report (md), confusion matrices (png),
                     metrics (json), and the leaderboard.csv
```

---

## Models

The active model registry lives in [src/models.py](src/models.py) (`MODEL_SPECS`).
11-model fast-baseline suite:

- Classical / linear: KNN, NearestCentroid, CosineSimilarity, DecisionTree, LDA, QDA, LinearSVC, RidgeClassifier, SGDClassifier
- Ensembles: AdaBoost, HistGradientBoosting

All models are exposed through a single `ModelSpec` interface so the training loop is family-agnostic.

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

# 3. Train + evaluate all models
python -m src.train --task 4class
python -m src.train --task 2class

# Optional: subset of models
python -m src.train --task 4class --models KNN HistGradientBoosting LDA
```

Each training run produces, per model and per task:

- `outputs/models/<task>/<model>.joblib` (fitted held-out estimator)
- `outputs/reports/<task>/<model>_classification_report.md`
- `outputs/reports/<task>/<model>_confusion_matrix.png` (counts + row-normalized)
- `outputs/reports/<task>/<model>_metrics.json` (per-fold CV + holdout)
- `outputs/reports/<task>/leaderboard.csv` (aggregated, sorted by holdout macro-F1)

Reproducibility: stratified split with `random_state=42`, `N_SPLITS=5`, `TEST_SIZE=0.2` ([src/config.py](src/config.py)).

---

## Results

Results to be regenerated against the current `MODEL_SPECS` suite. Per-model classification reports, confusion matrices, and the aggregated leaderboard land under `outputs/reports/<task>/` after each run.

---

## Reproducing the leaderboard

```bash
python -m src.preprocessing
python -m src.train --task 4class
python -m src.train --task 2class
```

Logs are written to `outputs/reports/train_<task>.log`.
