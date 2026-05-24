# MissVARPath — Missense Variant Pathogenicity Detection

A meta-classifier for missense variant pathogenicity built on **ClinVar** labels and **OpenCRAVAT** annotations. An 11-model fast-baseline suite (classical ML + two fast ensembles) is benchmarked under a unified stratified 80/20 + 5-fold CV protocol on both the original 4-class task and a collapsed binary task. The pipeline optionally augments the tabular VEP-score features with sequence-derived **k-mer** counts and **BLAST**-based nearest-neighbour features.

CMP682 course project — Doruk Topçu & Alihan. See [project proposal](CMP682%20-%20Project%20Proposal%20-%20Doruk%20%26%20Alihan%20-%20MVP%20%281%29.pdf) for the full plan, [project_progress_report.md](project_progress_report.md) for the running lab notebook, and [checkpoint.md](checkpoint.md) for the current save-game / roadmap.

---

## Dataset

- **Source:** ~63k ClinVar entries → annotated with OpenCRAVAT (158 annotator groups: `clinvar`, `gnomad4`, `alphamissense`, `cadd`, `revel`, `sift`, `phylop`, …) → down-sampled to a class-balanced subset.
- **Shape:** `data/missense_dataset.csv` — **21,872 rows × 777 columns**.
- **Label:** `clinvar__sig` ∈ {Benign, Likely benign, Likely pathogenic, Pathogenic}, 5,468 each (perfectly balanced).
- **After tabular preprocessing:** **208 features** (label-leakage and identifier columns dropped, columns with >50% missingness dropped, numeric medians imputed, zero-variance columns removed — see [src/preprocessing.py](src/preprocessing.py)). The empirical figures (rows, kept/dropped column counts) are written to `outputs/preprocessing/summary.json`.
- **After sequence augmentation** (`--variant augmented`): **+196 k-mer features** (64 ref + 64 alt + 64 diff over the 4³ k-mer alphabet, plus `gc_ref`, `gc_alt`, `entropy_ref`, `entropy_alt`) **and 10 BLAST-derived neighbour-label features** (see [BLAST caveat](#blast-caveat-not-true-homology-features) below).

Two task formulations are supported:

| Task | Classes |
| --- | --- |
| `4class` | Benign / Likely benign / Likely pathogenic / Pathogenic |
| `2class` | Benign+Likely benign vs. Pathogenic+Likely pathogenic |

---

## Repo layout

```
src/
  config.py                paths, label encoding, RNG seeds, FLANK_SIZE, Ensembl URL
  data_loader.py           CSV → DataFrame
  eda.py                   EDA driver (figures + summary tables)
  preprocessing.py         column drops, NA handling, imputation, label encoding
  sequence_fetch.py        ±25 bp flank fetch from Ensembl REST (GRCh38)
  sequence_features.py     k-mer counts + GC% + Shannon entropy
  blast_features.py        BLAST nearest-neighbour features (train-only DB)
  build_augmented_dataset.py  joins tabular + k-mer + BLAST → augmented parquet
  models.py                model zoo — MODEL_SPECS registry, CosineSimilarityClassifier
  train.py                 80/20 stratified split + 5-fold CV runner
  tune.py                  grid-search hyperparameter tuner (per-combo failure handling)
  shap_explain.py          SHAP attributions per model
  shap_per_class.py        per-class SHAP summaries
  utils.py                 logging, plotting, metric reporting

data/                missense_dataset.csv
outputs/
  eda/                EDA figures + summary CSVs/JSON
  preprocessing/      missense_processed.parquet, final_features.txt,
                       dropped_columns.txt, summary.json,
                       missense_augmented.parquet (optional)
  sequences/          flanks.parquet, kmer_features.parquet, blast_features.parquet
  models/<task>[_…]/  fitted sklearn estimators (.joblib) + feature manifest
  reports/<task>[_…]/ per-model classification report (md), confusion matrices (png),
                       metrics (json), and the leaderboard.csv
  tuning/<task>/<model>/ grid_results.csv, grid_summary.csv, best_config.json
  shap/               SHAP artifacts
```

---

## Models

The active model registry lives in [src/models.py](src/models.py) (`MODEL_SPECS`).
11-model fast-baseline suite:

- Classical / linear: `KNN`, `NearestCentroid`, `CosineSimilarity`, `DecisionTree`, `LDA`, `QDA`, `LinearSVC`, `RidgeClassifier`, `SGDClassifier`
- Ensembles: `AdaBoost`, `HistGradientBoosting`

All models are exposed through a single `ModelSpec` interface so the training loop is family-agnostic. `CosineSimilarityClassifier` is a small custom estimator that classifies by cosine similarity to L2-normalised class centroids (defined in [src/models.py](src/models.py)). Models that need scaling are wrapped in a `Pipeline([StandardScaler, …])`; tree/boosting models are not.

---

## Setup

### Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python ≥3.10. Place the curated dataset at `data/missense_dataset.csv`.

### External dependencies (only for sequence augmentation)

The k-mer / BLAST augmentation path needs two extra things — skip them if you only run the tabular baseline.

- **NCBI BLAST+** (`makeblastdb`, `blastn`) on `PATH`. Install via the [NCBI binaries](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) or a package manager (`brew install blast`, `apt install ncbi-blast+`, conda `bioconda::blast`).
- **Internet access** to the [Ensembl REST API](https://rest.ensembl.org) for the GRCh38 flank fetch (`src/sequence_fetch.py`).

---

## Usage

### Tabular baseline (no sequence augmentation)

```bash
# 1. Exploratory data analysis  → outputs/eda/
python -m src.eda

# 2. Preprocessing (writes outputs/preprocessing/missense_processed.parquet)
python -m src.preprocessing

# 3. Train + evaluate all models on each task
python -m src.train --task 4class
python -m src.train --task 2class

# Subset of models
python -m src.train --task 4class --models KNN HistGradientBoosting LDA
```

Each training run produces, per model and per task:

- `outputs/models/<task>/<model>.joblib` (fitted held-out estimator) + `features.txt` manifest
- `outputs/reports/<task>/<model>_classification_report.md`
- `outputs/reports/<task>/<model>_confusion_matrix.png` (counts) and `..._normalized.png` (row-normalized)
- `outputs/reports/<task>/<model>_metrics.json` (per-fold CV + holdout, includes the `.joblib` path)
- `outputs/reports/<task>/leaderboard.csv` (aggregated, sorted by holdout macro-F1)
- `outputs/reports/<task>/features.txt` (feature manifest used for that run)

### Sequence-augmented variant

```bash
# Fetch ±25 bp GRCh38 flanks (cached at outputs/sequences/flanks.parquet)
python -m src.sequence_fetch

# Compute 3-mer features (overlapping, frame-shift-1 over the {A,C,G,T} alphabet)
python -m src.sequence_features

# BLAST features — DB is built from the 80/20 train portion only
python -m src.blast_features

# Join tabular + k-mer + BLAST → outputs/preprocessing/missense_augmented.parquet
python -m src.build_augmented_dataset

# Train on the augmented matrix
python -m src.train --task 4class --variant augmented --tag augmented
```

### Hyperparameter tuning

```bash
python -m src.tune --task 4class --models HistGradientBoosting QDA DecisionTree
```

Writes `outputs/tuning/<task>/<model>/{grid_results.csv, grid_summary.csv, best_config.json}`. Grids are populated for the 10 tunable models in `src/tune.py::GRIDS`; per-combo failures are logged and skipped without aborting the run.

### Training CLI reference

```text
--task {4class,2class}        Target encoding.
--variant {base,augmented}    base = tabular only; augmented = + k-mer + BLAST.
--cv-mode {kfold,gene}        kfold = StratifiedKFold(5);
                              gene = StratifiedGroupKFold by gene_symbol with
                                     GroupShuffleSplit holdout (group-leakage-free).
--drop-prefixes …             Column-name prefixes to drop (e.g. VEP-score ablation).
--keep-prefixes …             Column-name prefixes to keep — drops everything else.
--tag <name>                  Tag appended to the report/model directory.
--models …                    Subset of model names to run (default: all 11).
```

Report directories are named `<task>[_<variant>][_<cv_mode>][_<tag>]`.

---

## Reproducibility

Pinned constants live in [src/config.py](src/config.py) and must be preserved across runs:

| Constant | Value | Used by |
| --- | --- | --- |
| `RANDOM_STATE` | `42` | All splits (train/test, CV, BLAST DB build) |
| `N_SPLITS` | `5` | StratifiedKFold / StratifiedGroupKFold |
| `TEST_SIZE` | `0.2` | Holdout fraction |
| `FLANK_SIZE` | `25` bp | Flank window (`2·FLANK_SIZE+1` = 51 bp) |
| `ENSEMBL_ASSEMBLY` | `GRCh38` | Coordinate system for the REST flank fetch |

Sequence-feature parameters:

- **k-mer k:** `3` (alphabet `{A, C, G, T}`, so the materialised k-mer space is 4³ = 64 columns each for ref, alt, and `alt − ref` diff).
- **BLAST:** `blastn` with `-evalue 10`, `-word_size 7`, `-max_target_seqs 11`; top-K = 10 hits per query; DB built only from train-portion ref-flanks (matching the 80/20 split). See the caveat below — these are *not* homology features in the usual sense; CV-fold use would leak.

### BLAST caveat — not true homology features

The `--variant augmented` path runs `blastn` against a DB of training-set
ref-flanks and aggregates the labels of the top-K hits into 10 features
(`blast_n_pathogenic`, `blast_top_pident`, `blast_target_4_top1`, …). It is
important to be explicit about what these features actually encode, because
the naming is misleading without context:

- **51 bp DNA windows are well below the regime where BLAST produces
  meaningful homology calls.** With `word_size=7` and `evalue=10`, a "hit"
  is essentially local 7-mer matching, not orthology or paralogy.
- **The hits are dominated by same-locus / same-gene proximity.** ClinVar has
  many variants per gene; two variants 10 bp apart share ~80% of their flank
  by construction and will trivially be each other's top BLAST hits. The
  features therefore behave mostly as a *neighbour-label proxy for the
  surrounding locus* — useful signal, but not "homology."
- **Empirical test.** Under gene-stratified CV (`--cv-mode gene`) this
  signal should largely collapse, since whole genes are held out and the
  same-locus neighbours become unavailable. That ablation is what
  quantifies how much of the augmented lift is real generalisation versus
  same-gene leakage.

A *properly* homology-driven version of this idea would need infrastructure
that is out of scope for this course project:

- a **large reference database** (UniRef50/90 or a curated pathogenic/benign
  protein corpus — millions of sequences, not the ~17.5k train flanks we
  index here),
- **protein-level BLAST** (`blastp`) on an amino-acid window with the
  variant substitution applied, so cross-gene paralog hits can contribute,
- and **tighter cutoffs** (`evalue ≤ 1e-3`, `pident ≥ 90`, length filters)
  so only meaningful hits feed the aggregator.

We kept the DNA-flank / train-DB formulation because it works as a label-aware
locus-neighbour signal under the 80/20 holdout, but the report frames these
features accordingly. See the docstring of [src/blast_features.py](src/blast_features.py)
for the long-form discussion.

Software:

- Python ≥3.10, scikit-learn ≥1.5 (see [requirements.txt](requirements.txt) for the full pin set).
- NCBI BLAST+ ≥2.13 (any modern release works; only `makeblastdb` and `blastn` are invoked).

### Reproducing the leaderboard

```bash
python -m src.preprocessing
python -m src.train --task 4class > outputs/reports/train_4class.log 2>&1
python -m src.train --task 2class > outputs/reports/train_2class.log 2>&1
```

Logs are written to `outputs/reports/train_<task>.log`.

---

## Results

Final tuned suite (10 models, AdaBoost excluded; see [checkpoint.md](checkpoint.md) for the tuning + chain history):

| Regime | Top model | 4-class macro-F1 | 2-class macro-F1 |
| --- | --- | ---: | ---: |
| Canonical kfold (untuned baseline) | HistGradientBoosting / AdaBoost | 0.7928 | 0.9870 |
| Final tuned no-AdaBoost (canonical) | HistGradientBoosting | **0.7950** | **0.9872** |
| Augmented kfold (k-mer + BLAST) | HistGradientBoosting | 0.7948 | 0.9904 |
| Gene-stratified CV (base) | HistGradientBoosting | 0.7658 | 0.9874 |
| No-VEP / VUS, augmented kfold | HistGradientBoosting | 0.7727 | 0.9764 |
| No-VEP / VUS, augmented gene-CV | HistGradientBoosting | 0.7611 | — |
| Raw-only, augmented kfold | HistGradientBoosting | 0.7212 | 0.9339 |
| DITTO removed, canonical kfold | HistGradientBoosting | 0.7911 | — |

Full per-model leaderboards live under `outputs/reports/<task>[_<variant>][_<cv_mode>][_<tag>]/leaderboard.csv`; the narrative reading of these numbers is in [final_report.tex](final_report.tex) (compile with any LaTeX engine; the figures it pulls in live under `final_report/figures/`).
