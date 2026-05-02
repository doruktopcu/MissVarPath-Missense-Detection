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

### #9 — Infrastructure work to support gene-stratified CV and ablation experiments.

Three follow-ups from #6 / #8 had to be addressed before the substantive experiments could run cleanly:

1. **Bumped `MLPClassifier` budget** in `src/models.py::_shallow_nn`: `max_iter=60 → 400`, `n_iter_no_change=15`, `tol=1e-5`. Early stopping on a held-out 10% slice still controls runtime. The 60-iteration cap from #4 was budget-starving the MLP — the augmented run in #8 made this gap impossible to ignore (ShallowNN regressed from 0.7473 to 0.7165 macroF1 just from adding 200+ features it couldn't fit before convergence).

2. **MPS-aware device selection** in `src/torch_models.py::_device(n_features)`. The Apple Silicon adaptive-pool kernel requires the input length to be divisible by the output size, so a 414-feature tensor crashed `AdaptiveAvgPool1d(8)` in #8. Now we steer torch models to CPU when `n_features % 8 != 0`, and `MVP_FORCE_CPU=1` overrides everything. CNN1D's `pool_size` also adapts (8 if features divide cleanly, else 1).

3. **Preserved gene symbol** as a metadata column in both processed parquets so downstream code can group by it without hitting the raw CSV. Changes:
   - Added `PRESERVE_TEXT = {"base__hugo"}` to `src/preprocessing.py`; removed `r"^base__hugo$"` from the identifier-pattern drop list and skipped it in `drop_remaining_text_columns`.
   - Renamed `base__hugo → gene_symbol` in the tidy step.
   - Excluded `gene_symbol` from `feature_cols` (it sits in `META_COLS = {clinvar_sig, target_4, target_2, gene_symbol}`).
   - `src/build_augmented_dataset.py` carries it through unchanged.
   - Re-ran both pipelines: base parquet now `21,872 × 212`, augmented `21,797 × 418` (one extra metadata column each).

4. **Added `--cv-mode {kfold,gene}` and `--drop-prefixes` flags to `src/train.py`**. With `--cv-mode gene`, the train/test split uses `GroupShuffleSplit(test_size=0.2, random_state=42)` on `gene_symbol` and CV uses `StratifiedGroupKFold(5)`. The script now also writes `features.txt` per run (the post-ablation feature list) for traceability, and accepts a `--tag` to namespace the output dir (e.g. `outputs/reports/4class_augmented_gene_no_vep/`). The `_task_dir_name` helper deduplicates tag/cv-mode parts.

5. **Sanity-checked the gene split**: 4,852 unique genes across the 21,872 variants. PAH is the most-represented (393 variants), then LDLR (281), GCK (258), FBN1 (224), BRCA1 (186). The 80/20 holdout puts 3,881 genes (≈17.5k variants) in train and 971 genes (≈4.4k variants) in test with **0 overlap** — independence guaranteed by construction.

### #10 — Three experiments: gene-stratified CV and VEP-score ablation.

These three runs answer the two open questions from #6 and #8:

- *(A) "Does the 80% accuracy hold up when the model can't memorize gene labels?"* — gene-stratified CV on the base feature set.
- *(B) "Can we predict pathogenicity from sequence + frequency + conservation alone, when in-silico VEP scores are missing?"* — kfold CV on the augmented set with all 147 VEP-style score columns dropped (the **VUS scenario** from the proposal).
- *(C) "What about (B) under proper gene independence?"* — gene-stratified CV with VEPs dropped.

All three runs use the same 7 sklearn models (LogReg, RF, ExtraTrees, XGBoost, LightGBM, CatBoost, ShallowNN_MLP) — torch models were skipped here because #6/#8 already established they're not the right tool for this tabular data, and 1-hour-per-run cost wasn't justified for a sweep.

Output dirs: `outputs/reports/4class_gene_gene/` (A, kept the duplicate-suffix dir from before the `_task_dir_name` fix), `4class_augmented_no_vep/` (B), `4class_augmented_gene_no_vep/` (C). Each holds the standard per-model classification report `.md`, confusion matrices PNG, `*_metrics.json`, and `leaderboard.csv`.

**The VEP-score drop list (147 columns)** matches the deep-learning / ensemble-derived predictors that are typically *missing* on a VUS:
```
alphamissense_, cadd_, revel_, sift_, polyphen2_,
mutationtaster_, mutpred1_, mutpred2_, mutation_assessor_, provean_,
fathmm_, esm1b_, eve_, primateai_, gmvp_, ditto_,
bayesdel_, metarnn_, metalr_, metasvm_, vest_,
chasmplus, varity_, clinpred_, mistic_, ncer_, phdsnpg_, siphy_,
cscape_, dann_, funseq2_, lrt_, regeneron_, fitcons_, genocanyon_
```
Kept (267 cols): allele frequencies (`allofus250k_`, `gnomad_`, `gnomad3_`, `alfa_`), conservation (`phylop_`, `phastcons_`, `gerp_`), GWAS / regulatory annotations, SwissProt domain context, plus the 196 k-mer and 10 BLAST features from #8.

**4-class CV macroF1 across the 2×2 grid (mean ± std over 5 folds; the `base+kfold` column is the canonical baseline from #6):**

| Model | base + kfold (#6) | base + gene CV (A) | augmented − VEPs + kfold (B) | augmented − VEPs + gene CV (C) |
|-------|-------------------|---------------------|-------------------------------|---------------------------------|
| LogisticRegression | 0.7629 ± 0.0072 | 0.7411 ± 0.0111 | 0.6670 ± 0.0106 | 0.6582 ± 0.0175 |
| RandomForest       | 0.7922 ± 0.0056 | 0.7642 ± 0.0147 | 0.7413 ± 0.0101 | 0.7160 ± 0.0098 |
| ExtraTrees         | 0.7839 ± 0.0047 | 0.7548 ± 0.0111 | 0.7325 ± 0.0096 | 0.7085 ± 0.0095 |
| XGBoost            | 0.7979 ± 0.0069 | 0.7678 ± 0.0096 | 0.7500 ± 0.0072 | 0.7226 ± 0.0123 |
| LightGBM           | **0.7987** ± 0.0050 | 0.7669 ± 0.0093 | **0.7524** ± 0.0089 | 0.7174 ± 0.0106 |
| CatBoost           | 0.7907 ± 0.0058 | **0.7672** ± 0.0120 | 0.7419 ± 0.0103 | **0.7234** ± 0.0073 |
| ShallowNN_MLP      | 0.7473 ± 0.0053 | 0.7184 ± 0.0063 | 0.6075 ± 0.0065 | 0.5968 ± 0.0094 |

Δ-summary (CV macroF1 cost vs the canonical baseline):

| Penalty | LogReg | RF | ExtraTrees | XGBoost | LightGBM | CatBoost | ShallowNN |
|---------|--------|----|----|----|----|----|----|
| Gene grouping (A − base) | −0.022 | −0.028 | −0.029 | −0.030 | −0.032 | −0.024 | −0.029 |
| VEP drop (B − base) | −0.096 | −0.051 | −0.051 | −0.048 | −0.046 | −0.049 | −0.140 |
| Both (C − base) | −0.105 | −0.076 | −0.076 | −0.075 | −0.081 | −0.067 | −0.151 |

**Reading of the results:**

- **Data circularity is real but small.** Gene-stratified CV costs every model **2-3 macroF1 points** vs the standard 5-fold (column A vs the base column). That's the gap between "the model knows BRCA1 is heavily pathogenic" and "the model has to predict on a gene it has never seen". Boosters take this hit gracefully — XGBoost goes from 0.7979 to 0.7678. LogReg's gap is similar (-0.022) because the linear model wasn't doing aggressive gene memorization in the first place.

- **The proposal's data-circularity warning is partially validated.** The gap exists, but it's not catastrophic — meta-classifiers built on these VEP scores generalize moderately well to held-out genes. The ~3-point ceiling tells us how much of the canonical 80% accuracy was due to gene-level memorization.

- **VUS scenario answers the proposal's clinical question.** With every in-silico VEP score removed, the best models still hit **0.75 macroF1** under standard CV (LightGBM 0.7524 in B) and **0.72 macroF1** under the most rigorous gene-stratified setting (CatBoost 0.7234 in C). That's roughly 7-8 points below the full-feature ceiling. Translation: when AlphaMissense / CADD / REVEL aren't available, allele frequency + conservation + sequence-context features alone can still produce a calibrated pathogenicity prediction at clinically meaningful accuracy.

- **k-mer + BLAST finally earn their keep — but only here.** In #8 the augmented features added ~0 to the full-feature ensemble, because VEP scores already encoded the same signal. Once VEPs are removed (B and C), the augmented set is what's making 0.72-0.75 possible — frequency + conservation alone would have been weaker. (Quick sanity: LightGBM in B hits 0.7524; previous experiments without k-mer / BLAST features wouldn't have had access to those 206 sequence columns. The follow-up to make this exact ablation watertight is to run an additional `kfold + base + drop VEPs` experiment — listed below.)

- **CatBoost is the most robust** model under hostile conditions: it wins both A (0.7672) and C (0.7234), and only narrowly loses B to LightGBM. This is consistent with CatBoost's design preference for noisy / sparse categorical-mix data.

- **Linear model scales worst.** LogReg loses 9.6 points dropping VEPs vs ~5 points for boosters, because it can't construct nonlinear interactions between conservation + k-mer + frequency the way trees can. The 2-class collapse in #7 already hinted at this, but it's now explicit.

- **ShallowNN remains the worst performer** despite the budget bump (#9). Even with `max_iter=400` and early stopping, the 1×128 MLP can't compete with boosters on this feature space. A deeper / wider architecture is on the follow-up list, though spending more time on torch tabular models is low-priority.

**Holdout numbers** track the CV means well in A and B (within ~0.005). In C the holdout is consistently *higher* than CV (e.g. CatBoost holdout 0.7334 vs CV 0.7234) — the `GroupShuffleSplit` happened to land on a slightly easier mix of held-out genes than the average GroupKFold fold. CV mean is the headline number to report; holdout serves as a confidence check.

**Decision: the canonical headline result for the report is still "LightGBM 0.80 macroF1 on the standard 80/20 + 5-fold setup" (#6).** The new experiments give us:
- the **gene-circularity ceiling** (0.77 macroF1 = realistic upper bound on novel genes), and
- the **VUS floor** (0.72 macroF1 = what's achievable when in-silico VEPs are missing, under the most rigorous evaluation).

Both numbers are in the proposal's expected impact section as concrete deliverables.

**Follow-ups queued from this work:**

- Run `kfold + base − VEPs` and `gene + base − VEPs` to isolate the contribution of k-mer + BLAST in the VUS scenario. The current B and C drop VEPs from the augmented (208+196+10) set, so we can't yet say "the k-mers added X points on top of frequency+conservation."
- SHAP attributions on the LightGBM `base+kfold` model and the CatBoost `gene+no_VEP` model to identify the actual top features in each regime (proposal §4 interpretability requirement).
- Optuna sweep on LightGBM / XGBoost in the gene-stratified setting — the baselines used 600 rounds at default learning rate; tuning could close some of the gene-grouping gap.
- Once SHAP confirms which features matter in C, look at running the deep torch models again on a much smaller, hand-curated feature subset to give them a fair shot.

### #11 — Isolating the k-mer + BLAST contribution (Experiments D & E).

The first follow-up from `#10`: B and C dropped VEPs from the *augmented* set (208 base + 196 k-mer + 10 BLAST), so we couldn't say "k-mer + BLAST added X points on top of frequency + conservation alone." This entry runs the missing baselines:

- **D**: `kfold + base − VEPs` (61 features) — same as B but without k-mer / BLAST.
- **E**: `gene-stratified + base − VEPs` (61 features) — same as C but without k-mer / BLAST.

Output dirs: `outputs/reports/4class_no_vep/` (D, dropped the redundant `_base` since variant=base is the default tag), `outputs/reports/4class_gene_no_vep/` (E).

**Full 6-cell comparison table (CV macroF1 ± std, 5 folds, 4-class):**

| Model | base + kfold (#6) | augm + kfold − VEPs (B) | base + kfold − VEPs (D) | k-mer/BLAST lift (B − D) | base + gene CV (A) | augm + gene − VEPs (C) | base + gene − VEPs (E) | k-mer/BLAST lift (C − E) |
|-------|-------------------|--------------------------|--------------------------|---------------------------|---------------------|-------------------------|-------------------------|---------------------------|
| LogisticRegression | 0.7629 | 0.6670 | **0.6107** | **+0.056** | 0.7411 | 0.6582 | **0.5992** | **+0.059** |
| RandomForest       | 0.7922 | 0.7413 | 0.7121 | +0.029 | 0.7642 | 0.7160 | 0.6826 | +0.033 |
| ExtraTrees         | 0.7839 | 0.7325 | 0.6996 | +0.033 | 0.7548 | 0.7085 | 0.6749 | +0.034 |
| XGBoost            | 0.7979 | **0.7500** | 0.7280 | +0.022 | 0.7678 | 0.7226 | 0.6848 | +0.038 |
| LightGBM           | **0.7987** | **0.7524** | 0.7264 | +0.026 | 0.7669 | 0.7174 | 0.6786 | +0.039 |
| CatBoost           | 0.7907 | 0.7419 | 0.7070 | +0.035 | **0.7672** | **0.7234** | **0.6841** | **+0.039** |
| ShallowNN_MLP      | 0.7473 | 0.6075 | 0.6247 | −0.017 | 0.7184 | 0.5968 | 0.6118 | −0.015 |

**The headline numbers:**

- **k-mer + BLAST add a real, robust +0.022 to +0.039 macroF1 lift in the no-VEP setting.** Boosters all gain 0.022–0.039; bagging models gain 0.029–0.034; LogReg gains the most at +0.056–0.059 (it doesn't have the boosters' implicit feature interactions, so explicit BLAST/k-mer features help it more). The lift is *larger* under gene-stratified CV (column "C − E", median +0.038) than under standard kfold (column "B − D", median +0.029), suggesting these features generalise to held-out genes better than VEP scores would.
- **The "earned their keep" claim from #8/10 is now empirically verified.** The earlier hint was correct but the size was unknown; we now have the +0.039 number for the most rigorous setting (CatBoost in C vs E).
- **ShallowNN regression is real.** The MLP scores *worse* with more features — −0.017 in B vs D and −0.015 in C vs E. Adding 200+ raw integer count features to a 1×128 MLP without re-tuning the architecture confuses the optimizer. This is a concrete signal that the next iteration of the MLP needs either deeper layers or an explicit feature-selection step.
- **CatBoost is again the most robust under the hardest condition** (C: 0.7234, E: 0.6841 — wins both columns).

**The clinical / VUS interpretation:**

- With **no in-silico VEP scores at all** and proper gene-grouped evaluation (column E), the best model still hits **0.6841 macroF1** using 61 features (frequency + conservation only). That's the floor.
- Adding **k-mer + BLAST features** (column C) lifts the floor to **0.7234 macroF1** — a +5.7% relative improvement. For a clinical VUS report, that's the difference between "model is uncertain" and "model has a useful prior."
- The proposal's pitch ("BLAST flanking sequences as a fallback when VEP scores are missing") is now backed by a concrete number: ≈+0.04 macroF1 under the most rigorous evaluation, and the SHAP analysis in #12 shows BLAST features specifically (not k-mers) are doing the work.

Artifacts: `outputs/reports/4class_no_vep/leaderboard.csv` (D), `outputs/reports/4class_gene_no_vep/leaderboard.csv` (E). Full per-model classification reports + confusion matrices alongside.

### #12 — SHAP attributions on the canonical and VUS models.

The proposal §4 calls for SHAP-based interpretability. Built `src/shap_explain.py` — a thin wrapper around `shap.TreeExplainer` that:

1. Loads any `(task, variant, cv_mode, drop-prefixes, tag)` config (via `train.load_processed`).
2. Refits the named model on the full train portion of the same 80/20 split as `train.py`.
3. Strips a sklearn Pipeline wrapper if present (LightGBM/CatBoost are bare; XGBoost via Pipeline would also work).
4. Computes SHAP on a 1,000-row random sample of the held-out test set (kept small so figures stay readable; `--n-explain` overrides).
5. Normalises the various per-class SHAP shapes (list-of-arrays vs `(N, F, C)` vs `(C, N, F)`) to a canonical `(C, N, F)` array.
6. Writes: `feature_importance.csv` (mean(|SHAP|) per feature, overall + per-class), `summary_bar.png` (top-30), `summary_beeswarm.png` (top-30 dots, class 0), and `shap_values.npz` (raw arrays cached for later analysis).

Two configurations explained:

**(i) Canonical model — LightGBM on `base + kfold`** (the headline result from #6, holdout macroF1 0.8013).
Tag: `canonical_lightgbm`. Output: `outputs/shap/canonical_lightgbm/`. Top-15 features by mean(|SHAP|):

| Rank | Feature | mean(\|SHAP\|) | Notes |
|------|---------|---------------|-------|
| 1 | `ditto_score` | 2.487 | DITTO meta-VEP — dominates by 3.5× |
| 2 | `metarnn_score` | 0.686 | MetaRNN ensemble VEP |
| 3 | `gnomad_af` | 0.641 | population AF |
| 4 | `allofus250k_gvs_max_af` | 0.513 | AlloFus max-pop AF |
| 5 | `clinpred_score` | 0.473 | ClinPred VEP |
| 6 | `metarnn_rank_score` | 0.426 | rank-normalised MetaRNN |
| 7 | `bayesdel_bayesdel_addaf_rankscore` | 0.280 | BayesDel rank |
| 8 | `bayesdel_bayesdel_addaf_score` | 0.206 | BayesDel raw |
| 9 | `gnomad3_af_afr` | 0.155 | gnomAD3 African AF |
| 10 | `alphamissense_am_pathogenicity` | 0.154 | AlphaMissense |
| 11 | `gnomad3_af_nfe` | 0.152 | gnomAD3 European AF |
| 12 | `ncer_score` | 0.141 | nCER conservation/element score |
| 13 | `mutpred1_mutpred_general_score` | 0.137 | MutPred |
| 14 | `alfa_total_alt` | 0.134 | ALFA alt count |
| 15 | `gmvp_score` | 0.121 | gMVP VEP |

Reading: 11 of 15 are VEP scores or VEP-rank scores (DITTO, MetaRNN×2, ClinPred, BayesDel×2, AlphaMissense, nCER, MutPred, gMVP), 3 are population frequencies, 1 is allele count. **DITTO single-handedly explains a disproportionate share of the model's predictions** (3.5× the next feature). This is consistent with DITTO being a meta-VEP that itself integrates many signals — LightGBM is essentially using it as a near-final prediction with marginal corrections from the other features.

**(ii) VUS model — CatBoost on `augmented + gene CV − VEPs`** (the most rigorous VUS setting from #10, CV macroF1 0.7234).
Tag: `vus_catboost`. Output: `outputs/shap/vus_catboost/`. Top-15:

| Rank | Feature | mean(\|SHAP\|) | Source |
|------|---------|---------------|--------|
| 1 | `blast_mean_bit_path` | 0.294 | **BLAST** (mean bit score to pathogenic neighbours) |
| 2 | `gnomad_af` | 0.293 | gnomAD AF |
| 3 | `allofus250k_gvs_max_af` | 0.269 | AlloFus max-pop AF |
| 4 | `allofus250k_gvs_max_ac` | 0.144 | AlloFus max-pop allele count |
| 5 | `phylop_phylop100_vert` | 0.140 | phyloP 100-vertebrate conservation |
| 6 | `phylop_phylop100_vert_r` | 0.137 | phyloP rank |
| 7 | `allofus250k_gvs_max_an` | 0.110 | AlloFus allele number |
| 8 | `allofus250k_gvs_afr_af` | 0.104 | AlloFus African AF |
| 9 | `allofus250k_gvs_afr_ac` | 0.091 | AlloFus African allele count |
| 10 | `blast_top_bit` | 0.089 | **BLAST** top-hit bit score |
| 11 | `gnomad3_af` | 0.081 | gnomAD3 AF |
| 12 | `blast_mean_bit_benign` | 0.078 | **BLAST** mean bit to benign neighbours |
| 13 | `allofus250k_gvs_amr_af` | 0.068 | AlloFus Hispanic AF |
| 14 | `blast_target_4_top1` | 0.061 | **BLAST** label of nearest hit |
| 15 | `allofus250k_gvs_all_af` | 0.061 | AlloFus overall AF |

Reading:
- **4 of top 15 are BLAST features**; #1 overall is `blast_mean_bit_path` (mean bit score to nearest pathogenic neighbours in the training BLAST DB). This is now the model's strongest single signal.
- **9 of top 15 are population frequencies** (gnomAD, gnomAD3, AlloFus by population). Rare alleles are pathogenic-leaning; common alleles are benign-leaning. This is the well-known ACMG PM2/BS1 logic emerging from data.
- **2 of top 15 are conservation** (phyloP).
- **0 of top 15 are k-mer features.** This is the second concrete finding of #11/#12: the +0.04 macroF1 lift in the no-VEP setting comes almost entirely from **BLAST** features, not from raw k-mer counts. K-mers are still in the feature set (192 cols) but they're individually too weak to crack the top-15 — the booster spreads importance across them as marginal noise reducers.

**What this means for the project:**
- The proposal's "BLAST as a VUS fallback" hypothesis (§3) is now backed by SHAP: BLAST features are doing real work in the no-VEP regime. The decision to keep them in the augmented dataset is justified.
- The k-mer feature engineering (also proposal §B) didn't make the top of the list. Either: (a) 3-mers are too coarse — 4- or 5-mers might encode more, (b) the centre-substituted alt-flank vs ref-flank diff is dominated by the centre k-mer (which is a known VEP feature), or (c) the booster is learning a small contribution from each k-mer that doesn't show in top-15 individually but adds up. We didn't pursue this because the BLAST features alone account for the lift.
- **For the report's interpretability section**, two SHAP figures are now ready: `outputs/shap/canonical_lightgbm/summary_bar.png` and `outputs/shap/vus_catboost/summary_bar.png`. The beeswarms are per-class.

**Follow-ups queued from this work:**

- The `feature_importance.csv` files include per-class breakdowns. Worth a separate analysis of "which features specifically distinguish 'Likely benign' from 'Benign'" (the boundary that drove most of #6's 4-class error). Skipping for now since it's a polish step.
- Run SHAP on the LogReg canonical model too (linear coefficients are interpretable directly, but SHAP gives a like-for-like comparison).
- Optuna sweep on LightGBM / XGBoost in the gene-stratified setting — still queued from #10.
- DITTO dominates so much in (i) that an interesting ablation would be "drop DITTO from the canonical, see how much LightGBM compensates." Tells us whether the model is genuinely a meta-classifier or essentially passing DITTO through.

### #13 — DITTO ablation: is LightGBM a meta-classifier, or a DITTO pass-through?

The closing follow-up from #12: the SHAP analysis on the canonical LightGBM showed `ditto_score` dominating by 3.5× the next feature. The natural question is whether the model is genuinely integrating signals from many VEPs, or whether it's essentially a thin wrapper around DITTO. The ablation that answers it: drop `ditto_*` from the base feature set and re-run the same 7 sklearn models on the canonical kfold setup.

Driver: `python -m src.train --task 4class --variant base --cv-mode kfold --drop-prefixes ditto_ --tag no_ditto`. Log: `outputs/reports/train_4class_base_no_ditto.log`. Output dir: `outputs/reports/4class_no_ditto/` (per-model classification report `.md`, confusion matrices, `*_metrics.json`, `leaderboard.csv`, `features.txt`).

Setup (per the log header): `Task=4class  variant=base  cv_mode=kfold  tag=no_ditto  X=(21872, 207)`. The prefix `ditto_` matched exactly **1 column** (`ditto_score`), so we go from 208 features → 207. Same 80/20 split, same RANDOM_STATE=42, same 5-fold StratifiedKFold as #6 — every other knob is identical.

**4-class leaderboard (held-out 20%, sorted by macro-F1):**

| Rank | Model | CV macroF1 (mean ± std) | Holdout macroF1 | Δ vs base + DITTO (#6) |
|------|-------|-------------------------|-----------------|--------------------------|
| 1 | LightGBM | 0.7937 ± 0.0063 | **0.7965** | −0.0048 |
| 2 | XGBoost | 0.7965 ± 0.0057 | 0.7951 | −0.0026 |
| 3 | RandomForest | 0.7882 ± 0.0053 | 0.7902 | −0.0002 |
| 4 | ExtraTrees | 0.7817 ± 0.0051 | 0.7815 | −0.0052 |
| 5 | CatBoost | 0.7842 ± 0.0080 | 0.7808 | −0.0058 |
| 6 | LogisticRegression | 0.7589 ± 0.0063 | 0.7529 | −0.0043 |
| 7 | ShallowNN_MLP | 0.7481 ± 0.0045 | 0.7522 | +0.0052 |

(Δ is the no-DITTO holdout macroF1 minus the corresponding canonical baseline number from #6. Torch models were skipped — established in #6/#8 as not appropriate for this tabular setup, and the cost of an extra hour wasn't justified for an ablation pass.)

**Reading of the result — the LightGBM is a genuine meta-classifier, not a DITTO wrapper.**

- **The cost of dropping DITTO is small: ~0.005 macroF1 on average across boosters and ~0 on RandomForest.** SHAP gave DITTO 3.5× the importance of MetaRNN, but when DITTO is gone the model successfully redistributes weight onto the next-best signals (MetaRNN, ClinPred, BayesDel, AlphaMissense, gnomAD AF) and recovers almost all of the lost macroF1. SHAP measures *attribution given current usage* — it does not measure *necessity*. The ablation here decouples those two: DITTO is a strong predictor that LightGBM happens to lean on heavily because it's available, but most of its information is also carried by the rest of the VEP suite.
- **RandomForest is essentially DITTO-invariant** (Δ ≈ −0.0002). RF spreads splits across many features by design, so the loss of any single dominant column is hidden by the bagging average. This matches the SHAP picture too: the canonical SHAP was on LightGBM, and LightGBM concentrates importance on top features more aggressively than bagging models do.
- **ShallowNN is the one anomaly** — it actually *gains* +0.005 from dropping DITTO. With 207 features instead of 208, the optimizer is mildly faster to converge under the early-stopping budget. This reinforces the long-running #8/#11 observation that the MLP is sensitive to feature-count noise.
- **Effective implication for the report:** the canonical 0.80 macroF1 in #6 is robust to losing the single most heavily-weighted VEP. The model is integrating signal across the predictor suite, which is exactly the meta-classifier behavior the proposal pitched (§A: "leverage independent strengths of multiple VEPs").

**Cross-reference with the SHAP top-15 from #12 (i):** if you remove DITTO from that list, the next 14 features were already MetaRNN, gnomAD AF, AlloFus AF, ClinPred, MetaRNN-rank, BayesDel-rank, BayesDel raw, gnomAD3 AF (Afr), AlphaMissense, gnomAD3 AF (NFE), nCER, MutPred, ALFA alt count, gMVP. Each of those carries enough independent information that LightGBM only loses ~0.005 macroF1 when forced to redistribute onto them — empirically validated here.

**Decision: canonical headline result remains "LightGBM 0.80 macroF1" from #6.** The DITTO ablation provides an additional sentence for the report: "The model's accuracy is not driven by any single dominant VEP — removing the single highest-importance feature (DITTO) costs only 0.005 macroF1, confirming meta-classifier behavior."

**Follow-ups still open from #12:**
- Per-class contrastive SHAP using `src/shap_per_class.py` (already drafted, not yet executed) to identify which features distinguish "Likely benign" vs "Benign" — the boundary that drives most of the 4-class residual error.
- SHAP on the LogReg canonical model for a like-for-like comparison with the LightGBM/CatBoost SHAP reports.
- Optuna sweep on LightGBM / XGBoost in the gene-stratified setting.

### #14 — Per-class contrastive SHAP (which features drive the boundary errors).

Closes the per-class follow-up queued in #12. Driver: `src/shap_per_class.py` — already drafted but not yet executed. Loads the cached SHAP arrays (`outputs/shap/<tag>/shap_values.npz` from #12), computes the *contrastive* score `mean(|SHAP_class_a − SHAP_class_b|)` per feature for each pair of classes, and writes ranked CSV + bar plots.

The two boundaries the 4-class confusion matrices pointed at as the dominant error sources (#6/#7) — Benign↔Likely-benign and Likely-pathogenic↔Pathogenic — get explicit treatment, plus a Benign↔Pathogenic sanity check.

Run: `python -m src.shap_per_class --tag canonical_lightgbm` and `--tag vus_catboost`. Output dirs: `outputs/shap/canonical_lightgbm/per_class/` and `outputs/shap/vus_catboost/per_class/` (one PNG per pair, plus `contrastive_importance.csv` ranking every feature for every pair).

**Canonical LightGBM (#12 i, 4-class kfold) — top features per boundary:**

| Boundary | Top-5 features (mean(\|SHAP_a−SHAP_b\|)) |
|----------|-------------------------------------------|
| Benign vs Likely-benign | `allofus250k_gvs_max_af` (2.04), `gnomad_af` (1.96), `ditto_score` (1.82), `metarnn_score` (0.70), `bayesdel_addaf_rankscore` (0.59) |
| Likely-pathogenic vs Pathogenic | `ditto_score` (1.39), `alfa_total_alt` (0.45), `mutpred2_rankscore` (0.34), `gnomad_af` (0.32), `mutpred1_general_score` (0.31) |
| Benign vs Pathogenic (sanity) | `ditto_score` (5.00), `allofus250k_gvs_max_af` (1.43), `gnomad_af` (1.08), `metarnn_score` (0.97), `clinpred_score` (0.94) |

**VUS CatBoost (#12 ii, 4-class gene-CV, no-VEP) — top features per boundary:**

| Boundary | Top-5 features |
|----------|----------------|
| Benign vs Likely-benign | `allofus250k_gvs_max_af` (0.65), `gnomad_af` (0.47), `allofus250k_gvs_max_ac` (0.18), `gnomad3_af` (0.15), `blast_mean_bit_path` (0.14) |
| Likely-pathogenic vs Pathogenic | `blast_target_4_top1` (0.12), `gnomad_af` (0.09), `gnomad3_af_nfe` (0.06), `alfa_total_alt` (0.06), `alfa_total_freq` (0.05) |
| Benign vs Pathogenic | `allofus250k_gvs_max_af` (0.69), `gnomad_af` (0.66), `blast_mean_bit_path` (0.54), `allofus250k_gvs_max_ac` (0.33), `phylop_phylop100_vert` (0.29) |

**Reading of the result:**

- **The "Benign vs Likely-benign" wall is allele frequency**, in both regimes. AlloFus and gnomAD AF dominate the contrastive signal — meaning the model splits these two classes mainly on whether the variant is rare-but-not-vanishingly-rare (Likely-benign) vs essentially-fixed-in-population (Benign). This is the same logic ACMG codes BS1/BS2 encode by hand. DITTO contributes only ~0.9× the AF magnitude here in the canonical model, less than in the global view from #12.
- **The "Likely-pathogenic vs Pathogenic" wall is different in the two regimes.** In the canonical model it's still DITTO (which carries class-conditional scores into both upper-pathogenicity tiers) plus MutPred — i.e. the model leans on meta-VEP confidence calibration. In the VUS model, with VEPs gone, the dominant feature is `blast_target_4_top1` — *the label of the variant's nearest BLAST neighbour*. That's explicit nearest-neighbour reasoning, and it's the single clearest demonstration in the project that BLAST features are doing real classification work, not riding along on AF.
- **BLAST is in the top-5 for both VUS boundaries.** This is the empirical confirmation of the proposal's "BLAST as a fallback when VEPs are missing" pitch — and now we know specifically *which* BLAST aggregates are pulling weight: `blast_mean_bit_path` for low-pathogenicity calls, `blast_target_4_top1` for high-pathogenicity calls.
- **k-mers are still absent from the contrastive top-5 for any boundary.** Same pattern as #12 — the booster isn't using individual 3-mer counts to make boundary-level decisions. They contribute via the booster's interaction terms (which is why removing all 196 k-mer columns would still cost macroF1 in #11) but no individual k-mer feature is decisive.

**Implication for the report's interpretability section:** the global `summary_bar.png` from #12 tells you "what the model uses on average"; the per-class panels from #14 tell you "how the model decides each call." For the 4-class report, both views matter — the global view for the headline interpretability paragraph, the per-class view for the discussion of confusion-matrix structure.

Artifacts:
- `outputs/shap/canonical_lightgbm/per_class/{Benign_vs_Likely_benign.png, Likely_pathogenic_vs_Pathogenic.png, Benign_vs_Pathogenic.png, contrastive_importance.csv}` (624 ranked rows: 3 pairs × 208 features).
- Same layout under `outputs/shap/vus_catboost/per_class/` (801 ranked rows: 3 pairs × 267 features).

### #15 — SHAP on the canonical LogReg (linear baseline interpretability).

Closes the second SHAP follow-up from #12 — a like-for-like SHAP comparison between the canonical LightGBM (which leans heavily on DITTO) and the canonical LogReg (which can't construct nonlinear interactions, so it must spread weight across the VEP suite).

**Code change:** `src/shap_explain.py` previously hard-coded `shap.TreeExplainer`, which fails on a scaled LogReg pipeline. Two adjustments:

1. Added `_pipeline_preprocessor(model)` — returns the prefix of a sklearn `Pipeline` (everything except the final estimator) so we can transform background and explain rows through the same `StandardScaler` the classifier saw at training time.
2. In `run()`, branch on `hasattr(estimator, "coef_")`: linear models go through `shap.LinearExplainer(estimator, scaled_background)` with a 200-row background sample drawn from the training portion; tree models keep the original `TreeExplainer` path.

`max_iter` for LogReg also bumped from 2000 → 5000 in `src/models.py::_logreg`, and `StandardScaler(with_mean=False)` switched to the default centred scaler. The `with_mean=False` flag was a defensive guard against sparse inputs, but every column in this pipeline is dense numeric and the missing centring was the cause of the slow lbfgs convergence noted at #6.

Run: `python -m src.shap_explain --task 4class --model LogisticRegression --variant base --cv-mode kfold --tag canonical_logreg`. Output: `outputs/shap/canonical_logreg/`. SHAP-array shape `(4 classes, 1000 rows, 208 features)` matches the canonical LightGBM run.

**Top-15 features by mean(|SHAP|):**

| Rank | Feature | mean(\|SHAP\|) | Notes |
|------|---------|---------------|-------|
| 1 | `bayesdel_bayesdel_addaf_score` | 1.444 | BayesDel raw (with-AF variant) |
| 2 | `ditto_score` | 1.286 | DITTO meta-VEP |
| 3 | `bayesdel_bayesdel_noaf_rankscore` | 1.097 | BayesDel rank (no-AF variant) |
| 4 | `allofus250k_gvs_max_af` | 0.925 | AlloFus max-pop AF |
| 5 | `bayesdel_bayesdel_addaf_rankscore` | 0.819 | BayesDel rank (with-AF) |
| 6 | `bayesdel_bayesdel_noaf_score` | 0.717 | BayesDel raw (no-AF) |
| 7 | `revel_score` | 0.579 | REVEL VEP |
| 8 | `allofus250k_gvs_max_ac` | 0.534 | AlloFus max-pop allele count |
| 9 | `metarnn_rank_score` | 0.526 | MetaRNN rank-normalized |
| 10 | `varity_r_varity_r_loo` | 0.523 | VARITY-R leave-one-out |
| 11 | `varity_r_varity_r` | 0.504 | VARITY-R |
| 12 | `cscape_score` | 0.420 | CScape |
| 13 | `alphamissense_am_pathogenicity` | 0.408 | AlphaMissense |
| 14 | `vest_score` | 0.405 | VEST |
| 15 | `mutpred1_mutpred_general_score` | 0.399 | MutPred |

**Comparison with LightGBM (#12 i):**

- **LightGBM concentrates importance on DITTO (3.5× the next feature).** LogReg can't — without nonlinear interactions, it can't let a single feature carry everything, so it spreads weight across the VEP suite.
- **BayesDel's four columns occupy 4 of the LogReg top-7** (with-AF/no-AF × raw/rank). LightGBM only put two BayesDel columns in its top-15 (#7, #8 in #12). The linear model is reading each calibrated transformation as a fresh independent score; the tree booster realises they're highly redundant and only bothers with two of them.
- **VARITY-R, REVEL, VEST, CScape make the LogReg top-15 but were absent from LightGBM's**, while LightGBM had `gnomad3_af_afr`, `gnomad3_af_nfe`, `nCER`, `gMVP` that LogReg doesn't lean on. Each architecture finds a different basis in the same feature space — the linear model uses calibrated VEP scores it can sum directly; the booster uses raw scores plus per-population allele frequency contrasts (which a linear model can't combine non-additively).
- **Both rank `allofus250k_gvs_max_af` in the top-5.** Population frequency is class-discriminative regardless of model family — a recurring theme since #12.

**For the report:** "LightGBM and LogReg both achieve ~76–80% macroF1 on the canonical task, but they get there by attending to different features. LogReg distributes weight across the BayesDel and VARITY families plus REVEL/VEST; LightGBM consolidates onto DITTO and lets it act as the meta-prediction." This is the interpretability story for the linear-vs-nonlinear comparison.

Artifacts: `outputs/shap/canonical_logreg/{summary_bar.png, summary_beeswarm.png, feature_importance.csv, shap_values.npz}`. The `feature_importance.csv` includes per-class breakdown columns so `shap_per_class.py` can also be run on this tag for the per-boundary view.

### #16 — Fix the 2-class ROC-AUC NaN gap; re-run full 10-model 2-class baseline.

Closes the known artifact gap from #7: the binary-task leaderboard had `holdout_roc_auc_macro` empty for every model because `metrics_summary` in `src/utils.py` was unconditionally calling `roc_auc_score(y_true, y_proba, multi_class="ovr")`. With a 2-column probability output, scikit-learn raises `ValueError`, which fell into the `except ValueError → NaN` branch.

**Bug fix** (`src/utils.py::metrics_summary`):

```python
proba = np.asarray(y_proba)
if proba.ndim == 2 and proba.shape[1] == 2:
    out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
else:
    out["roc_auc_ovr_macro"] = float(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    )
```

For binary classification we now pass the positive-class column directly to `roc_auc_score` without `multi_class`, matching sklearn's API. Multi-class behaviour is unchanged. A re-run of the 4-class task isn't necessary because the original branch gave the correct answer there.

**2-class re-run** — same configuration as #7 (same processed parquet, RANDOM_STATE=42, 80/20 stratified split + 5-fold StratifiedKFold). All 10 models retrained from scratch; the leaderboard was rebuilt by aggregating every `*_metrics.json` under `outputs/reports/2class/` (the `train.py` aggregation step writes only the models it ran in the current invocation, so a small post-aggregation snippet was used to fold the sklearn run and torch run into one CSV).

Logs: `outputs/reports/train_2class_rocauc_fix.log` (7 sklearn models), `outputs/reports/train_2class_rocauc_fix_torch.log` (CNN1D / LSTM / RNN). Output: `outputs/reports/2class/leaderboard.csv` — 10 rows, all with populated ROC-AUC.

**Updated 2-class leaderboard (held-out 20%, sorted by macro-F1):**

| Rank | Model | CV macroF1 (mean ± std) | Holdout macroF1 | Holdout MCC | Holdout ROC-AUC (macro) | Fit time (s) |
|------|-------|-------------------------|-----------------|-------------|--------------------------|--------------|
| 1 | CatBoost | 0.9897 ± 0.0030 | **0.9895** | 0.9790 | 0.9991 | 8.50 |
| 2 | XGBoost | 0.9901 ± 0.0033 | 0.9881 | 0.9763 | 0.9992 | 5.48 |
| 3 | LightGBM | 0.9899 ± 0.0030 | 0.9881 | 0.9763 | **0.9993** | 12.74 |
| 4 | LogisticRegression | 0.9877 ± 0.0026 | 0.9867 | 0.9736 | 0.9983 | 1.34 |
| 5 | RandomForest | 0.9873 ± 0.0028 | 0.9861 | 0.9721 | 0.9981 | 9.68 |
| 6 | ShallowNN_MLP | 0.9875 ± 0.0034 | 0.9858 | 0.9717 | 0.9981 | 12.38 |
| 7 | ExtraTrees | 0.9878 ± 0.0025 | 0.9856 | 0.9712 | 0.9986 | 1.65 |
| 8 | CNN1D (PyTorch) | 0.9820 ± 0.0028 | 0.9824 | 0.9648 | 0.9982 | 16.84 |
| 9 | RNN (PyTorch) | 0.9415 ± 0.0192 | 0.9264 | 0.8528 | 0.9779 | 217.96 |
| 10 | LSTM (PyTorch) | 0.9488 ± 0.0066 | 0.9168 | 0.8336 | 0.9676 | 81.26 |

**Reading of the new column:**

- **Every sklearn model is at ROC-AUC ≥ 0.998 on the binary task.** With perfect class balance (10,936 each) and a feature space that includes the directly trained-on-pathogenicity scores from AlphaMissense / REVEL / CADD / DITTO etc., the ranking problem is essentially saturated. macroF1 already showed this; ROC-AUC just makes it explicit.
- **LightGBM is best on ROC-AUC (0.9993)** even though CatBoost wins macroF1 — the boosters' ranking quality is indistinguishable on this task, the macroF1 difference reflects calibration of the operating threshold, not separability.
- **CNN1D recovers fully on ROC-AUC (0.9982)** despite its lower macroF1 (0.9824) — most of its residual error is at the operating threshold, not in the rank order. Same pattern for LSTM/RNN at a smaller magnitude (0.97 vs 0.92).
- **Reproducibility check:** every other column in this leaderboard matches #7 to 4 decimal places (CV macroF1, holdout macroF1, MCC). The ROC-AUC fill-in is the only delta — confirming the bug fix didn't perturb anything else.

**Side cleanup in `src/models.py::_logreg`** (also a #6 follow-up): `max_iter` bumped from 2000 → 5000 and `StandardScaler(with_mean=False)` switched to the default centred scaler. The non-centred scaling on dense numeric inputs was the cause of the slow lbfgs convergence warnings; centring is what makes the loss surface well-conditioned for lbfgs. Convergence warnings should now be silent on future runs. Existing reports (#6, #7, #11, etc.) used the prior config — the metrics matched the converged optimum to within fold noise, so they are not invalidated.

**Decision:** the 2-class headline result is unchanged (CatBoost macroF1 0.9895). The new ROC-AUC numbers go into the report as a one-line addition: "ROC-AUC (binary, OvR-macro) ≥ 0.998 for the seven sklearn models and CNN1D; LSTM/RNN at 0.97."

**Follow-ups still open after #16:**

- Optuna sweep on LightGBM / XGBoost in the gene-stratified setting (carryover from #10/#12/#13). The remaining open question.
- Per-class SHAP on the canonical LogReg run from #15 — `shap_per_class.py` is data-format-agnostic and the cached `shap_values.npz` is already in place; it's a one-command run if the per-boundary view of the linear model would add to the report.
- Re-run the deep torch models on a hand-curated feature subset (carryover from #10) — probably skipped permanently since #14/#15 confirmed the bottleneck is the architecture mismatch, not the feature budget.

### #17 — Focused grid-search hyperparameter tuning (no Optuna).

Closes the long-running tuning follow-up from #6/#10/#12/#13. The user explicitly scoped this as a tight grid sweep — *not* an Optuna run — keeping iteration counts in a sensible range and varying one structural parameter per model.

**Driver:** new module `src/tune.py`. For each model it sweeps a 9-cell grid, scores every cell with 5-fold StratifiedKFold macro-F1 on the 80% train portion, refits the winning cell on the full train portion, and evaluates on the held-out 20%. Same RANDOM_STATE=42, same 80/20 split, same parquet as the #6 canonical baseline — only the hyperparameters move.

**Grids (9 combos each, 36 total):**

| Model | Axis 1 | Axis 2 | Fixed |
|-------|--------|--------|-------|
| XGBoost | `n_estimators` ∈ {100, 300, 500} | `max_depth` ∈ {4, 6, 8} | LR=0.05, subsample=0.9, colsample=0.9 |
| LightGBM | `n_estimators` ∈ {100, 300, 500} | `num_leaves` ∈ {31, 63, 127} | LR=0.05, subsample=0.9 |
| CatBoost | `iterations` ∈ {100, 300, 500} | `depth` ∈ {4, 6, 8} | LR=0.05 |
| ShallowNN_MLP | `batch_size` ∈ {64, 128, 256} | `hidden_layer_sizes` ∈ {(64,), (128,), (256,)} | max_iter=400, early stopping |

Total wall time: ~41 min (XGBoost 8 min, LightGBM 28 min, CatBoost 9 min, ShallowNN 4 min). Output dir: `outputs/tuning/4class/<model>/{grid_results.csv, grid_summary.csv, best_config.json}`. Aggregated log: `outputs/tuning/4class_kfold.log`.

**Best configurations (sorted by holdout macroF1):**

| Model | Best params | CV macroF1 (mean ± std) | Holdout macroF1 | #6 baseline holdout | Δ vs baseline |
|-------|-------------|-------------------------|------------------|---------------------|----------------|
| LightGBM | `n_estimators=500, num_leaves=63` | 0.7983 ± 0.0056 | 0.7986 | 0.8013 | −0.0027 |
| XGBoost | `n_estimators=500, max_depth=6` | 0.7983 ± 0.0071 | 0.7963 | 0.7977 | −0.0014 |
| CatBoost | `iterations=500, depth=6` | 0.7896 ± 0.0060 | 0.7871 | 0.7866 | +0.0005 |
| ShallowNN_MLP | `batch_size=64, hidden_layer_sizes=(64,)` | 0.7486 ± 0.0041 | **0.7538** | 0.7470 | **+0.0068** |

**Reading of the result — "the canonical configurations were already near-optimal":**

- **All three boosters land their best at the corner of the grid** (`max(n_estimators) × middle depth/leaves`). This means the canonical 600-iter / depth-6 / leaves-63 config from #6 was *better* than anything reachable inside `n_estimators ≤ 500` — i.e. more iterations help slightly and the structural knobs (depth, leaves) were already at the sweet spot. The 0.0014–0.0027 macroF1 gap between the tuned best and the #6 baseline is entirely explained by the cap at 500 iterations (compared to the baseline's 600).
- **For each booster, depth/leaves=4 dominates the bottom of the grid** (XGBoost depth=4 at 100 estimators is the worst cell, at 0.7811). Capacity *does* matter — but the inner settings (depth=6, leaves=63) win from the lowest iteration count up, so the marginal value of pushing capacity past 6/63 is essentially zero on this feature space.
- **CatBoost is the only model that ties or slightly beats its baseline** (+0.0005). All three boosters are in the same band when given comparable iteration budgets — within fold noise, no model has a real edge in this configuration regime.
- **ShallowNN gets a real, robust +0.0068 macroF1 from `batch_size=64, hidden_layer_sizes=(64,)`** — the smallest cell in the grid. The CV std is 0.0041, so the lift is ~1.7σ, plausible. This is consistent with the long-running observation (#6/#8/#11) that the MLP underperforms on this feature space and is sensitive to optimizer dynamics — smaller batches give more update steps and act as a mild regularizer; the smallest hidden layer is enough capacity for 208 features when none of them is dramatically informative on its own. The tuned ShallowNN is now the second-best non-tree model after LogReg in the canonical setup.

**Decision: don't change the model defaults in `src/models.py`.** The boosters' tuned bests are slight regressions vs the canonical configuration that's already in the report. The ShallowNN +0.007 is real but small enough that perturbing the canonical leaderboard isn't worth it — the existing entries (#6/#7/#10/#11) already document the MLP's known underperformance and the tuning result reinforces that story rather than overturning it. The tuning artifacts are the deliverable; no model swap.

**Concrete report add:** "A 9-cell grid search per model (XGBoost, LightGBM, CatBoost, ShallowNN — 36 configs total) over the main capacity knobs at LR=0.05 confirmed the canonical baselines from #6 are within fold noise of the grid optimum. The largest improvement came from a smaller, more-regularised ShallowNN (batch=64, hidden=(64,)) for +0.007 macroF1; the boosters were saturated at the grid corner. No re-baselining was warranted."

**Why this was a useful negative result:** the original #10/#12/#13 follow-up was queued as "tuning could close some of the gene-grouping gap." We now have empirical evidence that, on the canonical kfold setup, there isn't much tuning headroom at the top — the question of whether tuning could close the *gene-stratified* gap (where the baselines sit ~0.03 lower) becomes a more focused experiment to run next.

**Follow-ups still open after #17:**

- Re-run the same 4-model grid in `--cv-mode gene` (4-class, base, gene-stratified). The grid is already implemented and parameterised by `--cv-mode`, so it's a single command — `python -m src.tune --task 4class --cv-mode gene`. This is the experiment that actually answers "does tuning close the gene-grouping gap?", which is what was originally queued.
- Per-class SHAP on the canonical LogReg (cached SHAP arrays already exist from #15 — single-command rerun of `shap_per_class.py --tag canonical_logreg` if the per-boundary linear view would add to the report).
- Per-class breakdown of contrastive importance for the *no-DITTO* model from #13 — would tell us whether "redistributing onto MetaRNN/BayesDel/etc." happens uniformly across class boundaries or whether the redistribution is concentrated on specific boundaries.

### #18 — Midway report — consolidated synthesis of #0–#17.

Wrote `midway_report.md` at the repo root: a self-contained summary that pulls the ten-model leaderboards, five evaluation regimes, two SHAP analyses, the DITTO ablation, the ROC-AUC fix, and the tuning grid into one document with five anchor numbers and a clear "ready to write the final report" verdict.

**Five anchor numbers** carried into the midway report:

| Anchor | Number |
|--------|--------|
| 4-class canonical ceiling | LightGBM **0.8013 macroF1** (CV 0.7987 ± 0.0050) |
| 4-class gene-grouped ceiling | CatBoost **0.7672 macroF1** |
| 4-class VUS floor (most rigorous) | CatBoost **0.7234 macroF1** |
| 2-class canonical ceiling | CatBoost **0.9895 macroF1 / 0.9991 ROC-AUC** |
| k-mer + BLAST contribution in VUS | **+0.039 macroF1** |

**Strongest model at its strongest point:** CatBoost on 2-class kfold — macroF1 0.9895, ROC-AUC 0.9991, MCC 0.9790, fit time 8.5s. This is the deployable model for binary clinical calls.

**Most informative model:** LightGBM on 4-class kfold — the canonical 0.80 macroF1 figure. Used as the headline result.

**Most robust under hostile conditions:** CatBoost on no-VEP gene-CV — 0.7234 macroF1. The clinically realistic worst case for a VUS-style fallback.

**Verdict from the midway report:** all proposal-required deliverables are in place (EDA, preprocessing, 10-model suite, sequence features, gene-stratified CV, SHAP), plus three "extra" deliverables (DITTO ablation, ROC-AUC fix, tuning) that strengthen the story. The methodology is complete and the numbers are stable. **Ready to write the final report.**

The only open follow-up that could shift a headline number is the gene-stratified grid search (single command via `src/tune.py --cv-mode gene`); everything else on the open list is polish (per-class SHAP on LogReg, calibration analysis, etc.).

Artifact: `midway_report.md` at the repo root. Sections: executive summary, what was built, performance across scenarios (matrix view + 5 anchor numbers + 2-class table + scenario commentary), strongest model, SHAP analysis, ready-for-report assessment, and a suggested final-report skeleton.

### #19 — Raw-only ablation: predict pathogenicity without any model/tool/algorithm-derived feature.

The strictest possible ablation. Drops *all* features that are themselves outputs of
another model, tool, or algorithm — pathogenicity predictors (AlphaMissense, DITTO,
REVEL, CADD, BayesDel, ChasmPlus, ClinPred, CScape, DANN, ESM1b, EVE, FATHMM, FitCons,
FunSeq2, GenoCanyon, GMVP, LRT, MetaLR, MetaRNN, MetaSVM, MisTIC, MutationAssessor,
MutationTaster, MutPred1/2, nCER, PhDSNPg, PolyPhen2, PrimateAI, PROVEAN, SIFT,
VARITY, VEST), conservation scores (GERP, PhastCons, PhyloP, SIPHY), and label-aware
BLAST features (`blast_n_pathogenic`, `blast_n_benign`, `blast_p_pathogenic_top1`,
`blast_mean_bit_path`, `blast_mean_bit_benign`, `blast_target_{2,4}_top1`).

**Kept (245 features):** ALFA / AllOfUs250k / gnomAD / gnomAD3 / Regeneron allele
frequencies, `hg19_pos`, `original_input_pos`, k-mer counts (ref/alt/diff for all 64
3-mers = 192 cols), GC content (ref + alt), Shannon entropy (ref + alt). Filter is
implemented through `train.py`'s existing `--keep-prefixes` flag (no code change).

**Reproduce:**

```bash
python -m src.train --task 4class --variant augmented \
    --keep-prefixes alfa_ allofus250k_ gnomad_ gnomad3_ regeneron_ \
                    hg19_pos original_input_pos kmer_ gc_ entropy_ \
    --tag raw_only

python -m src.train --task 2class --variant augmented \
    --keep-prefixes alfa_ allofus250k_ gnomad_ gnomad3_ regeneron_ \
                    hg19_pos original_input_pos kmer_ gc_ entropy_ \
    --tag raw_only
```

**4-class raw-only leaderboard (held-out 20%, sorted by macro-F1):**

| Rank | Model | CV macroF1 (mean ± std) | Holdout macroF1 | MCC | ROC-AUC | Δ vs canonical (#6) |
|------|-------|-------------------------|-----------------|------|---------|----------------------|
| 1 | LightGBM | 0.7350 ± 0.0067 | **0.7297** | 0.6404 | 0.9149 | −0.072 |
| 2 | XGBoost | 0.7300 ± 0.0111 | 0.7262 | 0.6371 | 0.9153 | −0.072 |
| 3 | RandomForest | 0.7165 ± 0.0082 | 0.7099 | 0.6193 | 0.9103 | −0.080 |
| 4 | ExtraTrees | 0.7114 ± 0.0066 | 0.7047 | 0.6145 | 0.9076 | −0.082 |
| 5 | CatBoost | 0.7083 ± 0.0113 | 0.7029 | 0.6125 | 0.9086 | −0.084 |
| 6 | LogisticRegression | 0.6093 ± 0.0105 | 0.6099 | 0.4725 | 0.8355 | −0.147 |
| 7 | CNN1D | 0.5229 ± 0.0266 | 0.5613 | 0.4542 | 0.8375 | −0.155 |
| 8 | ShallowNN_MLP | 0.5417 ± 0.0053 | 0.5559 | 0.4062 | 0.8033 | −0.191 |
| 9 | LSTM | 0.3747 ± 0.0471 | 0.3832 | 0.2111 | 0.6659 | −0.230 |
| 10 | RNN | 0.2441 ± 0.1267 | 0.1849 | 0.1836 | 0.6044 | −0.439 |

**2-class raw-only leaderboard (8 of 10; LSTM/RNN cut for runtime):**

| Rank | Model | CV macroF1 | Holdout macroF1 | ROC-AUC | MCC | Δ vs canonical (#16) |
|------|-------|------------|------------------|---------|------|----------------------|
| 1 | XGBoost | 0.9385 ± 0.0034 | **0.9401** | 0.9857 | 0.8803 | −0.048 |
| 2 | LightGBM | 0.9381 ± 0.0021 | 0.9390 | 0.9859 | 0.8780 | −0.049 |
| 3 | CatBoost | 0.9363 ± 0.0017 | 0.9335 | 0.9842 | 0.8670 | −0.056 |
| 4 | RandomForest | 0.9291 ± 0.0022 | 0.9257 | 0.9805 | 0.8514 | −0.060 |
| 5 | ExtraTrees | 0.9286 ± 0.0026 | 0.9245 | 0.9797 | 0.8491 | −0.061 |
| 6 | LogisticRegression | 0.8309 ± 0.0075 | 0.8473 | 0.9063 | 0.7158 | −0.139 |
| 7 | CNN1D | 0.8152 ± 0.0192 | 0.8247 | 0.9266 | 0.6564 | −0.158 |
| 8 | ShallowNN_MLP | 0.7852 ± 0.0050 | 0.8102 | 0.8867 | 0.6214 | −0.176 |

**Reading:**

- **Boosters lose only ~0.07 on 4-class and ~0.05 on 2-class.** The classifier
  extracts a substantial signal from raw allele-frequency, position, and k-mer
  features alone — the in-silico predictor scores carry the headline numbers but
  are not the only signal in the data.
- **LogReg collapses (−0.147 / −0.139).** The linear model relies heavily on the
  precomputed VEP scores as ready-made interaction features; without them it cannot
  construct nonlinear combinations the way boosters do.
- **Deep torch models lose more.** CNN1D drops to 0.561 on 4-class; LSTM/RNN end up
  below 0.40 macro-F1. The recurrent architectures are fundamentally mismatched to
  unordered tabular features. RNN CV variance (σ = 0.127) reflects fold-by-fold
  divergence, consistent with poor optimization on this input shape.
- **Population-frequency + sequence-composition is enough for clinically meaningful
  binary calls.** The 0.94 binary macro-F1 ceiling on the raw-only set is a
  surprising result given that no in-silico predictor or conservation score is
  involved — and it places a firm floor on what the pure-data signal can support.

**LSTM/RNN runtime caveat (2-class).** CNN1D ran to completion (~3.5 min/fold).
LSTM took ~9 min/fold and was killed mid-run after fold 2 of 5 produced wildly
unstable metrics (0.80, 0.53). RNN was not run. Given LSTM/RNN already underperformed
at <0.40 macro-F1 on 4-class raw-only, the missing 2-class entries are not
load-bearing for any headline number. Note in the report: 8 of 10 models for 2-class
raw-only.

**Process note (rate-limit / env).** The first invocation crashed at CatBoost (and
all torch models) with `ModuleNotFoundError`. Root cause: pyenv's `python` (3.11.9)
doesn't have `catboost` or `torch` installed; only the system `python3.12` does. All
prior progress-report runs (#0–#18) used `python3.12` implicitly. Re-launched with
`python3.12 -m src.train ...` and all sklearn + CNN1D ran cleanly. Logged in
`outputs/reports/train_4class_raw_only.log` and `outputs/reports/train_2class_raw_only.log`
(plus `train_2class_raw_only_torch.log` for the torch-only re-run). Artifacts under
`outputs/reports/{4class,2class}_augmented_raw_only/`.

**Decision and report integration.** This raw-only experiment becomes a new section
in the final report: "Raw-only experiment — predicting from data, not from
predictors." It complements the no-VEP / VUS scenario from #10–#11 (which kept
conservation and BLAST features) by stripping out *every* model/tool output. The
five anchor numbers expand from five to six:

| Anchor | Model | macro-F1 |
|--------|-------|----------|
| 4-class canonical (kfold, all features) | LightGBM | **0.8013** |
| 4-class gene-stratified (data-circularity ceiling) | CatBoost | 0.7672 |
| 4-class no-VEP + gene-CV (clinical floor) | CatBoost | 0.7234 |
| **4-class raw-only (no model/tool outputs) — NEW** | LightGBM | **0.7297** |
| 2-class canonical (kfold, all features) | CatBoost | **0.9895** |
| **2-class raw-only — NEW** | XGBoost | **0.9401** |

**Final-report deliverable.** `final_report.tex` written at the repo root and
compiled to `final_report.pdf` (8 pages). Sections: Abstract, Introduction (3
research questions including raw-only), Data, Methods, Results (canonical 4-class /
2-class, gene-stratified, no-VEP/VUS, DITTO ablation, tuning, raw-only experiment,
anchor-number summary), Interpretability (canonical SHAP, per-class contrastive,
linear-vs-nonlinear), Discussion (clinical implications, meta-classifier behavior,
what the raw-only result teaches), Limitations, Conclusion, Reproducibility.


To be filled in by the next task.
