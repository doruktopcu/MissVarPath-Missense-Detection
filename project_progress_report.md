# MVP / MissVARPath — Project Progress Report

## Instructions (read before editing)

This file is the running log of everything that has been done in this project. The goal is that any contributor (or future-us) can scroll through and understand the full state of the work without digging through code.

**Rules:**

1. Every entry gets a sequential `#N` identifier (e.g. `#1`, `#2`, `#3`) so we can refer to it later (`see #4`).
2. Append new entries at the bottom. Never reorder or rewrite past entries — if something is superseded, add a new entry that says so.
3. Each entry should record: **what was done**, **why**, and **where it lives** (file paths, output paths). Be concrete.
4. Group sub-bullets under the entry. Keep prose tight — the report should read like a lab notebook, not a diary.
5. When you produce artifacts (figures, tables, processed data, model checkpoints), reference their path in the relevant entry.
6. If a step changes the data shape, schema, or label encoding, call it out explicitly.
7. Don't delete entries. Mark them as **[Superseded by #N]** if needed.

**Project goal:** Build a meta-classifier for missense variant pathogenicity using ClinVar labels and OpenCRAVAT features. See `CMP682 - Project Proposal - Doruk & Alihan - MVP (1).pdf` for the full proposal.

---

## Log

### #0 — First prompt given.
The user provided the project proposal and the curated dataset (`data/missense_dataset.csv`, 21,872 rows × 777 columns, 4 perfectly balanced ClinVar classes at 5,468 each: Benign, Likely benign, Likely pathogenic, Pathogenic).

The dataset was curated as follows:
- Started from ~63k ClinVar entries.
- Annotated using **OpenCRAVAT** (158 distinct annotator groups, e.g. `clinvar`, `gnomad4`, `alphamissense`, `cadd`, `revel`, `sift`, `phylop`, etc.).
- Down-sampled to 22k rows with a balanced 4-class label drawn from `clinvar__sig`.

Plan from the user:
1. Thorough EDA (get to know the data).
2. Data preprocessing — clean column names, drop identifier columns (ClinVar IDs, rsIDs, dbSNP IDs, transcripts, HGVS strings, etc.), fill missing with NA first, then impute (median for numeric, mean where appropriate).
3. Train ~10 models with stratified 80/20 split + 5-fold cross-validation. Required: Random Forest, LightGBM, XGBoost, Shallow NN, CNN, LSTM, RNN, plus other popular models. Don't reach for very complex architectures right away — get a baseline read first.
4. Reports must include classification reports and confusion matrices per model.

### #1 — Project structure scaffolded.

Created the codebase skeleton:

```
src/
  __init__.py
  config.py            # paths, label encoding, RNG seeds, constants
  data_loader.py       # CSV → DataFrame with dtype hints
  eda.py               # EDA driver — figures + summary tables
  preprocessing.py     # column dropping, NA handling, imputation, label encoding
  models.py            # model zoo (sklearn + PyTorch)
  train.py             # 80/20 stratified split + 5-fold CV runner
  utils.py             # plotting, metric reporting, checkpointing
outputs/
  eda/                 # tables, label balance plots, missingness, correlations
  preprocessing/       # processed parquet, dropped/kept column manifests
  models/              # serialized checkpoints
  reports/             # per-model classification reports + confusion matrices
  figures/             # shared plots
requirements.txt
.gitignore
```

Constants fixed for reproducibility: `RANDOM_STATE=42`, `N_SPLITS=5`, `TEST_SIZE=0.2`. Label column is `clinvar__sig`. Both 4-class and 2-class (collapsed pathogenic-vs-benign) encodings are defined in `src/config.py` so we can swap depending on experiment.

### #2 — EDA pass on the raw 777-column dataset.

Driver: `src/eda.py` (run as `python -m src.eda`). Outputs in `outputs/eda/`.

Headline numbers:

- **Shape:** 21,872 rows × 777 columns, ~596 MB in memory after pandas load.
- **Label balance (4-class):** perfectly balanced — Benign 5,468, Likely benign 5,468, Likely pathogenic 5,468, Pathogenic 5,468.
- **Dtype split:** 447 numeric, 330 object, 0 boolean.
- **Object columns are mostly text:** of the 330 object-typed columns, only **1** is ≥50% numerically coercible and **328** are <5% coercible (i.e. JSON-encoded mappings, transcript lists, disease names, free text). This means we have to drop them rather than salvage them with `pd.to_numeric`.
- **Annotator groups:** 158 unique prefixes. Largest contributors by column count: `gnomad4` (44), `clinvar` (38, includes the label), `allofus250k` (28), `base` (15), `swissprot_ptm` (12), `oncokb` (12), `sift` (11).
- **Missingness:** many annotators are 100% missing for this missense subset (e.g. `dida`, `oncokb`, `fathmm`, `denovo`, `mutpred_indel`, `mavedb`, `alfa_african`) — these came along from the OpenCRAVAT batch but apply to other variant types. They will be dropped at preprocessing time.
- **Key VEP scores plotted by class:** `outputs/eda/key_vep_scores_by_class.png` (boxplots of AlphaMissense, CADD, REVEL, SIFT, PolyPhen2, phyloP, phastCons, gnomAD AF, MetaRNN, BayesDel, FATHMM, MutationTaster, PROVEAN, ESM1b, EVE, PrimateAI, MVP across the 4 classes). Most show clear class separation, as expected for established VEPs.
- **VEP score correlations:** `outputs/eda/key_vep_score_correlation.png` — strongly inter-correlated cluster among AlphaMissense/REVEL/CADD/MetaRNN/BayesDel; SIFT inversely correlated (lower SIFT = more deleterious).

Artifacts (full list):
- `outputs/eda/overview.json` — top-line stats.
- `outputs/eda/label_distribution_{4class,2class}.csv` + `label_distribution.png`.
- `outputs/eda/annotator_group_counts.csv` — every prefix and its column count.
- `outputs/eda/missingness_per_column.csv`, `missingness_per_annotator.csv`, plus `missingness_histogram.png` and `missingness_top30.png`.
- `outputs/eda/object_columns_numeric_coercion.csv` — per-object-column salvage rate.
- `outputs/eda/key_vep_score_summary.csv`, `key_vep_scores_by_class.png`, `key_vep_score_correlation.png`.

### #3 — Preprocessing pipeline.

Driver: `src/preprocessing.py` (run as `python -m src.preprocessing`). Output: `outputs/preprocessing/missense_processed.parquet` (21,872 × 323 = 320 features + `clinvar_sig` + `target_4` + `target_2`).

Pipeline (in order):

1. **Filter to valid labels** — kept only the 4 ClinVar significance values. No rows lost (already curated).
2. **Drop label-leakage columns** — every column whose prefix is `clinvar__` or `clinvar_acmg__` *except* the label itself. Anything else from these annotators (e.g. `clinvar__sig_conf`, `clinvar__rev_stat`, `clinvar_acmg__ps1_id`) trivially leaks the target. **39 columns dropped.**
3. **Drop identifier / free-text columns** by regex on column names (transcripts, rsIDs, dbSNP IDs, HGVS strings, JSON `all_mappings`, disease names, PubMed IDs, gene info, the `civic` annotator group, ChasmPlus per-cancer-type transcript columns, base coordinates `chrom/pos/ref_base/alt_base/cchange/achange`, etc.). Per the proposal we will engineer sequence features ourselves later (k-mers, BLAST). **88 columns dropped.**
4. **Coerce object → numeric** where ≥95% of values parse as numbers. With this dataset that yielded **0 columns** — confirming the EDA finding that object-typed columns here are essentially all true free text.
5. **Drop the remaining object-typed columns** (free text / JSON-ish). **233 columns dropped.**
6. **Cast booleans to nullable integers** (no-op on this dataset; 0 boolean columns).
7. **Drop columns with >95% missingness.** **95 columns dropped** — these were the 100%-missing annotators flagged by the EDA (dida, oncokb, fathmm, denovo, mutpred_indel, ...).
8. **Impute remaining missing values:** numeric → median; non-numeric → mode (none survived to this point). All NaNs are gone after this step.
9. **Drop zero-variance columns.** **1 column dropped.**
10. **Tidy column names**: lowercase, replace `__` with `_`, strip non-alphanumerics. Examples:
    - `alphamissense__am_pathogenicity` → `alphamissense_am_pathogenicity`
    - `gnomad4__af_popmax` → `gnomad4_af_popmax`
    - The label is renamed to `clinvar_sig`.
11. **Add encoded targets:** `target_4` ∈ {0,1,2,3} from `CLASS_4` map and `target_2` ∈ {0,1} (Benign/Likely-benign → 0; Likely-pathogenic/Pathogenic → 1).

End state:
- 21,872 rows × 323 columns.
- 320 feature columns: 319 float, 1 int, all numeric.
- No missing values.
- `target_4` is perfectly balanced at 5,468 per class; `target_2` is perfectly balanced at 10,936 per class.

Manifests written:
- `outputs/preprocessing/final_features.txt` — the 320 features kept, one per line.
- `outputs/preprocessing/dropped_columns.txt` — every dropped column with the reason it was dropped.
- `outputs/preprocessing/summary.json` — per-step drop counts.

**Notes on label leakage and "data circularity" (proposal §2):** the held-out classifier will see scores from VEPs that were themselves trained on overlapping ClinVar data (AlphaMissense, REVEL, CADD, MetaRNN, BayesDel, etc.). This is *intended* — MVP is by design a meta-classifier. The proposal calls for **gene-stratified** cross-validation as the mitigation. The current `train.py` does **standard stratified CV** as a baseline; gene-stratified CV will be added in a later entry once we wire `base__hugo` (gene symbol) back in as a grouping key only (not a feature).

### #4 — Training & evaluation framework.

Driver: `src/train.py`. Model registry: `src/models.py`. PyTorch components: `src/torch_models.py`.

Configuration (all in `src/config.py`):
- `RANDOM_STATE = 42`, `N_SPLITS = 5`, `TEST_SIZE = 0.2`.
- 80/20 **stratified** train/test split; `StratifiedKFold(n_splits=5, shuffle=True)` runs on the **training portion only** so the held-out 20% is never touched until final evaluation.

Models in the suite (10 total, per the project plan):

| # | Name | Family | Notes |
|---|------|--------|-------|
| 1 | LogisticRegression | sklearn | StandardScaler + L2; multinomial via lbfgs |
| 2 | RandomForest | sklearn | 500 trees, balanced class weights |
| 3 | ExtraTrees | sklearn | 500 trees, balanced class weights |
| 4 | XGBoost | xgboost | 600 rounds, depth 6, hist tree method |
| 5 | LightGBM | lightgbm | 600 rounds, 63 leaves |
| 6 | CatBoost | catboost | 600 iterations, depth 6 |
| 7 | ShallowNN_MLP | sklearn | StandardScaler + 1×128 ReLU MLP, early stopping |
| 8 | CNN1D | PyTorch | Conv1d(1→32→64) + adaptive pool + MLP head |
| 9 | LSTM | PyTorch | Bidirectional LSTM(hidden=64) over feature axis |
| 10 | RNN | PyTorch | Vanilla RNN(hidden=64) over feature axis |

Caveat for CNN/LSTM/RNN: the inputs are tabular features, not sequences. We feed the flat 320-dim vector as a 1-D "sequence" so these architectures have a defined forward pass. They are included because the project plan asked for them; we expect tree ensembles to dominate on this kind of tabular data. Real sequence-style models (k-mer/BLAST) come later, per the proposal.

Reporting (per model, both `--task 4class` and `--task 2class`):

- **5-fold CV metrics** (accuracy, macro-F1, weighted-F1, fit time) saved per fold and as `cv_summary` (mean ± std).
- **Held-out test classification report** as Markdown — `outputs/reports/<task>/<slug>_classification_report.md`.
- **Confusion matrix** (counts + row-normalized) — `outputs/reports/<task>/<slug>_confusion_matrix.png` and `..._normalized.png`.
- **Combined per-model metrics JSON** — `outputs/reports/<task>/<slug>_metrics.json` (CV folds + holdout summary including MCC, macro ROC-AUC OVR, fit time).
- **Leaderboard** across all models: `outputs/reports/<task>/leaderboard.csv`, sorted by held-out macro-F1.

Run commands:
```
python -m src.train --task 4class
python -m src.train --task 2class
python -m src.train --task 4class --models RandomForest LightGBM XGBoost  # subset
```

Training runs are not yet executed — that is the next step (#5).

### #5 — Tightened the missingness threshold.

The original 95%-missing cutoff in `drop_high_missing` (see #3) preserved 320 features but kept 200+ columns whose missingness was between 50% and 95%. Median-imputing a column that's >50% NaN flattens it toward a single value and dilutes downstream signal. EDA confirmed a clean bimodal split: 259 columns are <5% missing, then a long tail; nothing important sits between 5% and 50%.

Action:
- Lowered `max_missing` from `0.95` → `0.50` in `src/preprocessing.py::drop_high_missing` and re-ran `python -m src.preprocessing`.

New end state in `outputs/preprocessing/missense_processed.parquet`:
- Shape: **21,872 × 211** (208 features + `clinvar_sig` + `target_4` + `target_2`).
- Drop counts updated: 39 label-leakage, 88 identifiers, 233 free-text, **207** high-missingness (was 95), 1 zero-variance.
- Manifest at `outputs/preprocessing/dropped_columns.txt`; summary at `outputs/preprocessing/summary.json`.

Rationale carried into the report so the model results in #6 are interpreted in light of this tighter feature set.

### #6 — 4-class baseline training run.

Driver: `python -m src.train --task 4class`. Log: `outputs/reports/train_4class.log`. All artifacts: `outputs/reports/4class/` (per-model classification report `.md`, confusion matrices PNG counts + normalized, per-model `*_metrics.json`, `leaderboard.csv`).

Notes:
- One bug surfaced and was fixed mid-run: `plot_confusion_matrix` was casting counts to float unconditionally, then formatting with `"d"`, raising `Unknown format code 'd' for object of type 'float'`. The first run lost only the LogReg holdout artifact before being killed; the rerun is the canonical leaderboard below.
- `lbfgs failed to converge` warnings on Logistic Regression are informational at this scale; bumping `max_iter` is on the follow-up list but the metrics already match the rerun bit-for-bit.

Leaderboard (held-out 20% test, sorted by macro-F1):

| Rank | Model | CV acc (mean ± std) | CV macroF1 | Holdout acc | Holdout macroF1 | Holdout MCC | Holdout ROC-AUC (OvR macro) | Fit time (s) |
|------|-------|---------------------|------------|-------------|-----------------|-------------|------------------------------|--------------|
| 1 | LightGBM | 0.7985 ± 0.0049 | 0.7987 | **0.8011** | **0.8013** | 0.7353 | 0.9481 | 47.45 |
| 2 | XGBoost | 0.7979 ± 0.0067 | 0.7979 | 0.7977 | 0.7977 | 0.7308 | **0.9492** | 18.35 |
| 3 | RandomForest | 0.7923 ± 0.0055 | 0.7922 | 0.7904 | 0.7904 | 0.7210 | 0.9455 | 11.04 |
| 4 | ExtraTrees | 0.7840 ± 0.0048 | 0.7839 | 0.7870 | 0.7867 | 0.7164 | 0.9434 | 2.39 |
| 5 | CatBoost | 0.7909 ± 0.0057 | 0.7907 | 0.7870 | 0.7866 | 0.7166 | 0.9453 | 15.59 |
| 6 | LogisticRegression | 0.7625 ± 0.0071 | 0.7629 | 0.7568 | 0.7572 | 0.6766 | 0.9346 | 13.70 |
| 7 | ShallowNN_MLP | 0.7478 ± 0.0049 | 0.7473 | 0.7474 | 0.7470 | 0.6649 | 0.9347 | 4.36 |
| 8 | CNN1D (PyTorch) | 0.7099 ± 0.0094 | 0.6950 | 0.7179 | 0.7160 | 0.6254 | 0.9194 | 15.05 |
| 9 | RNN (PyTorch) | 0.5780 ± 0.0667 | 0.5301 | 0.6667 | 0.6234 | 0.5860 | 0.8968 | 242.53 |
| 10 | LSTM (PyTorch) | 0.5526 ± 0.0270 | 0.5232 | 0.6213 | 0.6131 | 0.4995 | 0.8841 | 64.94 |

Reading of the table:
- **Tree ensembles dominate.** LightGBM and XGBoost are tied within noise at ~80% accuracy / 0.80 macro-F1 and ~0.95 ROC-AUC. RandomForest, ExtraTrees, and CatBoost form a tight 78.7–79.0% band.
- **LogReg ≈ 76%** is a respectable linear baseline — almost all VEP scores are pre-calibrated continuous predictors, so a linear combination of them already classifies well.
- **ShallowNN_MLP underperforms LogReg** (74.7% vs. 75.7%). Suspect under-training: `max_iter=60` with early stopping is conservative. Worth bumping in a follow-up.
- **CNN/LSTM/RNN trail badly** (71%, 62%, 67%). This was expected and is documented at #4: these architectures consume the flat 208-dim feature vector as if it were a 1-D sequence, but feature ordering carries no meaning. They are present for the project plan, not because they're appropriate for tabular data. Real sequence-style models will use k-mer / BLAST features (proposal §3), which we haven't engineered yet.
- **The holdout numbers track the CV means tightly** for every well-performing model (within ±0.01) — no train/test leakage and no overfit on the 80% portion.

What this earns for the report:
- A clean comparison table across 10 model families.
- Per-class diagnostics: row-normalized confusion matrices (`outputs/reports/4class/*_confusion_matrix_normalized.png`) make it easy to see which class boundary each model struggles with — the typical failure pattern is "Likely benign" vs. "Benign" and "Likely pathogenic" vs. "Pathogenic", which is the same boundary clinicians find hardest.

Follow-ups queued (do not execute yet):
- Increase `MLPClassifier` `max_iter` (and possibly add a deeper variant) — see ShallowNN underperformance above.
- Hyperparameter tuning for LightGBM / XGBoost with a small Optuna sweep on the 80% portion.
- **Gene-stratified CV** — guard against data circularity per proposal §2; the gene symbol `base__hugo` is currently dropped at preprocessing time, will need to be re-loaded as a grouping key.
- SHAP attributions on the LightGBM final model (proposal §4 interpretability requirement).
- Engineer sequence features (k-mer, BLAST) before re-running CNN/LSTM/RNN — those models need a real sequence input to be meaningful.

### #7 — 2-class baseline training run.

Driver: `python -m src.train --task 2class`. Log: `outputs/reports/train_2class.log`. All artifacts: `outputs/reports/2class/` (per-model classification report `.md`, confusion matrix PNG counts + normalized, per-model `*_metrics.json`, `leaderboard.csv`).

Setup: same processed parquet (208 features, 21,872 rows), same 80/20 stratified split + 5-fold CV, but the label is `target_2` — Benign/Likely-benign collapsed to 0 and Likely-pathogenic/Pathogenic collapsed to 1. Both classes are perfectly balanced at 10,936 each.

Leaderboard (held-out 20% test, sorted by macro-F1):

| Rank | Model | CV acc (mean ± std) | CV macroF1 | Holdout acc | Holdout macroF1 | Holdout MCC | Fit time (s) |
|------|-------|---------------------|------------|-------------|-----------------|-------------|--------------|
| 1 | CatBoost | 0.9897 ± 0.0030 | 0.9897 | **0.9895** | **0.9895** | 0.9790 | 7.51 |
| 2 | XGBoost | 0.9901 ± 0.0033 | 0.9901 | 0.9881 | 0.9881 | 0.9763 | 4.67 |
| 3 | LightGBM | 0.9899 ± 0.0030 | 0.9899 | 0.9881 | 0.9881 | 0.9763 | 11.37 |
| 4 | LogisticRegression | 0.9877 ± 0.0026 | 0.9877 | 0.9867 | 0.9867 | 0.9736 | 2.57 |
| 5 | RandomForest | 0.9873 ± 0.0028 | 0.9873 | 0.9861 | 0.9861 | 0.9721 | 9.88 |
| 6 | ExtraTrees | 0.9878 ± 0.0025 | 0.9878 | 0.9856 | 0.9856 | 0.9712 | 1.13 |
| 7 | ShallowNN_MLP | 0.9875 ± 0.0033 | 0.9875 | 0.9851 | 0.9851 | 0.9703 | 7.91 |
| 8 | CNN1D (PyTorch) | 0.9813 ± 0.0050 | 0.9813 | 0.9819 | 0.9819 | 0.9639 | 14.00 |
| 9 | RNN (PyTorch) | 0.9387 ± 0.0169 | 0.9387 | 0.9429 | 0.9429 | 0.8859 | 206.20 |
| 10 | LSTM (PyTorch) | 0.9334 ± 0.0226 | 0.9334 | 0.9131 | 0.9131 | 0.8263 | 63.32 |

Reading of the table:
- **Pathogenic-vs-Benign is essentially solvable from these features.** Seven of ten models clear 0.985 holdout accuracy. This confirms the EDA finding that VEP scores (AlphaMissense, REVEL, CADD, MetaRNN, BayesDel, etc.) carry strong, near-monotonic signal once you collapse the "Likely" tier into its certain counterpart — which lines up with how those VEPs were trained in the first place.
- **CatBoost is best on macro-F1**, edging out XGBoost and LightGBM by ~0.0014 (within fold-level noise). The boosters and LogReg are inside a 0.0028 spread.
- **Most of the difficulty in #6 was the "Likely" boundaries.** Compare directly: the same models that scored ~0.79 in 4-class score ~0.99 here, so the bulk of the residual error in the 4-class task was Benign↔Likely-benign and Pathogenic↔Likely-pathogenic confusion, not benign↔pathogenic confusion.
- **CNN1D recovers** from the 4-class run (0.72 → 0.98). The binary task is easy enough that even an inappropriate convolution architecture clears 0.98 on tabular features.
- **LSTM and RNN still trail** but improve substantially (0.55–0.58 → 0.93). Same structural caveat as #6: feature ordering carries no sequence semantics.

Known artifact gap:
- `holdout_roc_auc_macro` is empty in the binary leaderboard. Reason: `metrics_summary` in `src/utils.py` calls `roc_auc_score(y_true, y_proba, multi_class="ovr")`, which raises on a 2-column probability output and falls into the `except ValueError` → NaN branch. Cosmetic; the per-model `*_metrics.json` already captures everything else (per-class F1, MCC, weighted F1). Fix queued for the next iteration.

Combined with #6 this completes the baseline pass requested in the very first prompt: 10 models × 2 tasks × {5-fold CV, 80/20 holdout, classification report, confusion matrix}. Total wall time ~36 min for 4-class + ~28 min for 2-class.

### #8 — Sequence-context features: k-mers + BLAST self-similarity.

The proposal (§B and §3) calls for using flanking DNA around each variant for both k-mer features and BLAST alignment as a fallback when in-silico VEP scores are missing. This entry implements both, runs them on the full dataset, retrains all 10 models on the augmented feature set, and reports the lift (or lack thereof).

**Pipeline (4 new modules):**

1. **`src/sequence_fetch.py`** — fetches a ±25 bp flank around each variant from the Ensembl REST batch endpoint (`POST /sequence/region/human`, GRCh38, 50 regions per request, ~14 req/sec to stay under the 15 req/sec rate limit). The response uses a `query` field (not `id`) to echo the requested coordinates. The centre base of every fetched window is verified against the dataset's reported `ref_base`; rows where the check fails are flagged `ok=False`. The mutated alt-flank is built by single-base substitution at the centre.

   Outputs: `outputs/sequences/flanks.parquet`. Result on the full dataset: **21,797 of 21,872 variants validated (99.66%)**. The 75 failures are mostly mitochondrial / patch-region coordinates where the requested window straddles a contig boundary or the reported ref doesn't match GRCh38 — they are dropped from downstream feature engineering. Total fetch time: ~13 minutes.

2. **`src/sequence_features.py`** — k-mer count features. For k=3 (full alphabet space = 64 trinucleotides) we emit:
   - `kmer_ref_<XYZ>` and `kmer_alt_<XYZ>` for every k-mer (128 columns)
   - `kmer_diff_<XYZ>` = alt count − ref count (64 columns)
   - aggregates: `gc_ref`, `gc_alt`, `entropy_ref`, `entropy_alt`
   
   Total: **196 sequence-derived features per variant**. K-mers containing non-{A,C,G,T} characters (rare 'N' bases) are skipped. Output: `outputs/sequences/kmer_features.parquet` (21,796 × 200 incl. join keys).

3. **`src/blast_features.py`** — local BLAST self-similarity. Uses BLAST+ (`makeblastdb` + `blastn`) installed system-wide. The pipeline:
   1. Reproduces the **same stratified 80/20 split** as `src.train` (RANDOM_STATE=42, stratify on target_4) so the BLAST DB only ever indexes training-set variants.
   2. Builds a nucleotide BLAST DB from the **ref-flanks** of the training set (~17.4k sequences).
   3. Queries every variant's **alt-flank** against this DB (`blastn`, word_size=7, evalue=10, max_target_seqs=11).
   4. For each query, drops self-hits (`qseqid == sseqid`), takes the top-K=10 hits by bitscore, and aggregates labels of those neighbours.

   Resulting features (10 per variant):
   - `blast_n_hits`, `blast_top_bit`, `blast_top_pident`
   - `blast_n_pathogenic`, `blast_n_benign` (counts among top-K)
   - `blast_p_pathogenic_top1` (1 if nearest neighbour is pathogenic-tier)
   - `blast_mean_bit_path`, `blast_mean_bit_benign`
   - `blast_target_4_top1`, `blast_target_2_top1` (label of nearest neighbour, −1 if no hit)

   Result: **242,841 BLAST hits** across **21,757 of 21,796 queries** (99.8% coverage). End-to-end BLAST step (DB build + 21,796 queries) ran in **~14 seconds** with `-num_threads 4`. Output: `outputs/sequences/blast_features.parquet`.

4. **`src/build_augmented_dataset.py`** — joins the three sources by `(chrom, pos, ref, alt)` (k-mer) and `variant_id` (BLAST), drops rows missing flank/BLAST features, deduplicates the lone collision (`15:25371797 G>T` appears twice in the raw CSV), and writes `outputs/preprocessing/missense_augmented.parquet`. Final shape: **21,797 × 417** = **414 features** (208 base + 196 k-mer + 10 BLAST) + label/targets.

5. **`src/train.py --variant augmented`** — same training pipeline as #6, but loads the augmented parquet and writes to `outputs/reports/4class_augmented/`.

**Two infrastructure bugs fixed in passing:**

- **MPS adaptive-pool divisibility:** `nn.AdaptiveAvgPool1d(8)` errors on Apple Silicon when the input length isn't divisible by 8. With 414 features that's the case. Switched CNN1D to global average pooling (`AdaptiveAvgPool1d(1)`) for portability. Side effect: information loss is larger, so CNN1D's CV score collapsed (see table) — but tabular CNN was never the right tool here, and we'll re-do it properly when sequence features arrive on their own.
- **MPS OOM during torch model `predict_proba`:** the wrapper sent the entire test matrix to MPS at once (4,360 × 414 floats × bidirectional LSTM hidden = ~900 MB), tripping the 20 GB pool limit on the second torch run. Patched `TorchClassifier._forward` in `src/torch_models.py` to chunk inference at `batch_size=256`. After the fix, LSTM and RNN both completed holdout evaluation cleanly.

**Results — augmented 4-class leaderboard** (held-out 20%, sorted by macro-F1):

| Rank | Model | CV macroF1 (mean ± std) | Holdout acc | Holdout macroF1 | Δ macroF1 vs base |
|------|-------|-------------------------|-------------|-----------------|-------------------|
| 1 | RandomForest | 0.7912 ± 0.0076 | 0.7984 | **0.7977** | **+0.0073** |
| 2 | XGBoost | 0.7994 ± 0.0087 | 0.7968 | 0.7964 | −0.0013 |
| 3 | LightGBM | 0.7968 ± 0.0046 | 0.7952 | 0.7948 | −0.0065 |
| 4 | CatBoost | 0.7919 ± 0.0082 | 0.7940 | 0.7933 | +0.0067 |
| 5 | ExtraTrees | 0.7858 ± 0.0038 | 0.7842 | 0.7835 | −0.0033 |
| 6 | LogisticRegression | 0.7619 ± 0.0057 | 0.7663 | 0.7660 | +0.0088 |
| 7 | ShallowNN_MLP | 0.7165 ± 0.0038 | 0.7151 | 0.7095 | **−0.0376** |
| 8 | LSTM (PyTorch) | 0.4942 ± 0.0354 | 0.5615 | 0.5539 | −0.0592 |
| 9 | CNN1D (PyTorch) | 0.5161 ± 0.0494 | 0.5601 | 0.5441 | −0.1719 (MPS pool change) |
| 10 | RNN (PyTorch) | 0.5132 ± 0.1401 | 0.5406 | 0.4907 | −0.1327 |

(Δ vs base is the augmented holdout macroF1 minus the corresponding number from #6.)

**Reading of the results — what BLAST/k-mer added (and didn't):**

- **Strong models barely move.** RandomForest gets a small +0.007 lift; LightGBM gives back about the same. The ±0.005 band is well inside fold-level noise (CV stds are ~0.005–0.009 for the boosters). **There is no clear, robust benefit from adding k-mer + BLAST features on top of the existing OpenCRAVAT VEP scores.**
- **This is consistent with the literature.** The base feature set already includes AlphaMissense, REVEL, CADD v1.7, MetaRNN, BayesDel, MutationTaster, PROVEAN, SIFT, ESM1b, EVE, PrimateAI, MVP, plus phyloP / phastCons conservation, plus protein-domain context (SwissProt). Each of those was *trained* on a flanking-sequence representation of the same locus, often with much richer encodings than 3-mer counts. The k-mer / BLAST features carry information that the deep VEPs already harvested, so a tree booster on top sees no new gradient.
- **Where this kind of feature WOULD pay off:** the proposal explicitly identifies the case — VUS variants where AlphaMissense / CADD / REVEL scores are not available. A future experiment is to **drop the in-silico score columns** from the augmented set and train on (gnomAD-style features + conservation + k-mer + BLAST) only. If that ablation gets within striking distance of the current ~0.80 macroF1, it shows the sequence-context features can stand in for VEPs when those are missing — which is the actual clinical bottleneck the proposal calls out.
- **Tiny shrinkage of the dataset (75 rows lost) is not the cause** of the flat lift — even RandomForest, which was trained on essentially the same rows, only moves +0.007.
- **ShallowNN regression is real** (−0.038): adding 200+ raw integer count features to a fixed-budget MLP with `max_iter=60` is enough to starve training. This was already on the follow-up list from #6 and is more visible here.
- **Torch sequence models worsened**, as expected. Three drivers compound: (a) the CNN1D global-pool change documented above, (b) feeding even more features as a fake "sequence" hurts the LSTM/RNN's already-poor inductive bias, (c) the RNN folds show a 0.140 std on macroF1 — high enough that the run is essentially numerical noise on top of a bad architecture choice.

**Artifacts:**
- `outputs/sequences/flanks.parquet`, `kmer_features.parquet`, `blast_features.parquet`
- `outputs/preprocessing/missense_augmented.parquet`
- `outputs/reports/4class_augmented/` — same layout as #6: per-model classification report `.md`, confusion matrices PNG (counts + normalized), `*_metrics.json`, `leaderboard.csv` (rebuilt from the JSONs after the partial-overwrite caused by the torch retry).
- Logs: `outputs/sequences/fetch.log`, `outputs/reports/train_4class_augmented.log` (the killed sklearn-only first pass), `outputs/reports/train_4class_augmented_torch.log` (torch retry).

**Decision: not adding these features to the default pipeline.** The base parquet (`missense_processed.parquet`) remains the canonical training input; the augmented parquet is opt-in via `--variant augmented`. The infrastructure is left in place so the VUS-style ablation can be done by simply listing the VEP-score columns to drop before re-running.

**Follow-ups queued from this work:**
- Run the **VEP-score ablation** described above (drop AlphaMissense/CADD/REVEL/etc. columns; keep only gnomAD freq + conservation + k-mer + BLAST). That is the experiment that actually answers "does sequence context help when in-silico scores are missing?".
- Add **gene-stratified CV** (still pending from #6) — particularly important now because BLAST self-similarity could leak gene identity through neighbour labels.
- **Bump `MLPClassifier` `max_iter`** so ShallowNN isn't budget-starved; the augmented run made this gap impossible to ignore.
- **Switch torch models to CPU** by default, or guard MPS with a feature-length divisibility check, so the platform-specific bugs don't recur.

### #9 — Placeholder for next entry.

To be filled in by the next task.
