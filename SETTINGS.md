# Settings and ablations — supplemental documentation

Companion file for [outputs/reports/grand_summary.csv](outputs/reports/grand_summary.csv).
The grand-summary CSV is a wide table with one row per model and 96 columns
named `<setting>.<metric>`. This document explains every setting and metric.

---

## Metrics (per setting × model cell)

Each `(setting, model)` cell reports four held-out-test metrics:

| Column suffix | Definition |
|---|---|
| `.macro_f1`        | unweighted mean of per-class F1 — equal weight per class regardless of class size; the headline metric used throughout the report |
| `.accuracy`        | raw fraction of correctly classified held-out samples |
| `.macro_precision` | unweighted mean of per-class precision — *of the variants predicted as class c, how many were truly class c?* averaged across classes |
| `.macro_recall`    | unweighted mean of per-class recall — *of the variants truly of class c, how many did we recover?* averaged across classes |

All four metrics come from the **20% held-out test split** that no model sees during training or CV (the same split is reused across every setting that derives from the same task). Macro-precision and macro-recall are recomputed from the saved `.joblib` models against this held-out test set; macro-F1 and accuracy mirror the leaderboard.csv values.

`RANDOM_STATE = 42` everywhere; the 80/20 split + 5-fold CV are deterministic.

---

## Tasks

The four classification tasks differ in label space:

| Task    | Classes | Class counts | Total rows |
|---------|---------|---|---|
| 4-class | Benign / Likely-benign / Likely-pathogenic / Pathogenic                      | 5,468 each                       | 21,872 |
| 2-class | Benign-side / Pathogenic-side (Likely-* collapsed into definitive labels)    | 10,936 / 10,936                  | 21,872 |
| 3-class | Benign-side / Pathogenic-side / VUS (strict-VUS subset of the curated pro-set)| 10,936 / 10,936 / 5,428          | 27,300 |
| 5-class | All four canonical labels + VUS                                              | 5,468 / 5,468 / 5,468 / 5,468 / 5,428 | 27,300 |

VUS data: `data/missense_VUS_pro_set_annotated.csv`, filtered to
`clinvar__sig == "Uncertain significance"` (5,428 of 5,468 rows survive).
Built by [scripts/build_vus_train_parquets.py](scripts/build_vus_train_parquets.py).

---

## Settings included in the grand summary

Setting labels in `grand_summary.csv` follow the convention `<task>_<modifier>`. Each maps to an on-disk directory under `outputs/reports/<dirname>/`.

### Canonical baselines (no ablation, no special split)

| Setting label                | Dirname                            | Notes |
|---|---|---|
| `4class_canonical_kfold`     | `4class/`                          | Untuned 11-model suite, base parquet (208 features) |
| `4class_tuned`               | `4class_final_no_adaboost/`        | Tuned 10-model suite (AdaBoost excluded), **headline** |
| `2class_canonical_kfold`     | `2class/`                          | Untuned 11-model suite, base parquet |
| `2class_tuned`               | `2class_final_no_adaboost/`        | Tuned 10-model suite, **headline** |

### Strict-evaluation regimes (stress-test the headline)

| Setting label             | Dirname                  | What is being tested |
|---|---|---|
| `4class_gene_cv`          | `4class_gene/`           | Gene-stratified CV (no gene appears in both train and test); measures generalisation across genes |
| `2class_gene_cv`          | `2class_gene/`           | Same for the binary task |
| `4class_augmented`        | `4class_augmented/`      | Base 208 features + 196 k-mer + 10 BLAST locus-neighbour features = 414 features |
| `2class_augmented`        | `2class_augmented/`      | Same for the binary task |

### "VUS scenario" — what survives without learned-predictor scores

| Setting label             | Dirname                              | What it simulates |
|---|---|---|
| `4class_no_vep_kfold`     | `4class_augmented_no_vep/`           | Drops every learned-predictor + multi-species conservation score (130 cols) on the augmented parquet — simulates scoring an actual VUS where the in-silico predictor pipeline hasn't run |
| `2class_no_vep_kfold`     | `2class_augmented_no_vep/`           | Same on binary task |
| `4class_no_vep_gene`      | `4class_augmented_gene_no_vep/`      | No-VEP + gene-stratified CV — the strictest "real-world VUS" estimate |
| `4class_raw_only`         | `4class_augmented_raw_only/`         | Strict ablation: keep only population AFs + position + k-mer/GC/entropy + BLAST (no learned predictor *or* multi-species conservation score) |
| `2class_raw_only`         | `2class_augmented_raw_only/`         | Same on binary task |
| `4class_no_ditto`         | `4class_no_ditto/`                   | Sanity check — drop only DITTO (the top-SHAP feature). Quantifies single-predictor dependency |

### VUS-as-class study

| Setting label             | Dirname           | What's new |
|---|---|---|
| `3class_with_vus`         | `3class_vus/`     | 10-model suite on Benign-side / Pathogenic-side / VUS (5,428 strict-VUS rows added as a third class) |
| `5class_with_vus`         | `5class_vus/`     | 10-model suite on Benign / Likely-benign / Likely-pathogenic / Pathogenic / VUS |

### Ablation-optimal lean models

Built by training the full 10-model suite on the canonical parquet with the indicated `--drop-prefixes`:

| Setting label   | Dirname              | Feature set | Features kept | Drops |
|---|---|---|---|---|
| `4class_lean_A` | `4class_lean/`       | drops 8 group-LOO waste groups including `ditto_` and `metarnn_`        | 110 | 97 |
| `4class_lean_B` | `4class_leanB/`      | as Lean A but keeps `ditto_` and `metarnn_` prefixes (per-feature winners) | 113 | 94 |
| `2class_lean_A` | `2class_lean/`       | same drop list, binary task | 110 | 97 |
| `2class_lean_B` | `2class_leanB/`      | same drop list, binary task | 113 | 94 |
| `3class_lean_A` | `3class_lean/`       | same drop list, 3-class with VUS | 110 | 97 |
| `3class_lean_B` | `3class_leanB/`      | same drop list, 3-class with VUS | 113 | 94 |
| `5class_lean_A` | `5class_lean/`       | same drop list, 5-class with VUS | 110 | 97 |
| `5class_lean_B` | `5class_leanB/`      | same drop list, 5-class with VUS | 113 | 94 |

**Lean B is the headline "ablation-optimal" model**: drops 45% of the 208-feature base schema and stays within 0.012 macro-F1 of the full-feature baseline on every task. Lean A is the more aggressive variant (also drops ditto_score / metarnn_score) and shows ~0.005 more loss per task.

---

## Feature groups (defined in [scripts/run_feature_ablation.py](scripts/run_feature_ablation.py))

The 208 base features partition into 12 functional groups:

| Group           | Members (prefixes)                                                                                                              | n features | Phase A verdict |
|---|---|---|---|
| `population_af` | `allofus250k_`, `gnomad_`, `gnomad3_`, `alfa_`, `regeneron_`                                                                    | 47 | **Indispensable** (HistGB delta +0.047 on 4-class) |
| `functional`    | `fitcons_`, `ncer_`                                                                                                              | 4  | **Indispensable for VUS tasks** (+0.027 on 5-class, +0.047 on 3-class) |
| `other_vep`     | `clinpred_`, `mistic_`, `mutpred1_`, `mutpred2_`, `vest_`, `sift_`, `polyphen2_`, `fathmm_`, `lrt_`, `mutation_assessor_`, `mutationtaster_`, `dann_`, `cscape_`, `eve_`, `esm1b_`, `primateai_`, `gmvp_`, `varity_`, `provean_`, `metalr_`, `metasvm_`, `phdsnpg_`, `genocanyon_`, `funseq2_` | 56 | Useful (+0.013 on 4-class) |
| `cadd`          | `cadd_`, `cadd_exome_`                                                                                                          | 4  | Borderline (-0.005 on 4-class, edge of noise floor) |
| `ditto`         | `ditto_`                                                                                                                         | 1  | Waste at group level (+0.004) but **per-feature top driver** (perm. importance 0.13 on 4-class) — kept in Lean B |
| `metarnn`       | `metarnn_`                                                                                                                       | 2  | Waste at group level (+0.001) but per-feature winner (perm. importance 0.05 on 4-class) — kept in Lean B |
| `bayesdel`      | `bayesdel_`                                                                                                                      | 4  | Waste (max \|delta\|=0.005 across tasks) — dropped by Lean B |
| `position`      | `hg19_pos`, `original_input_pos`                                                                                                | 2  | Waste at group level — dropped by Lean B |
| `revel`         | `revel_`                                                                                                                         | 2  | Waste (max \|delta\|=0.004) — dropped by Lean B |
| `chasmplus`     | `chasmplus`                                                                                                                      | 68 | **Confirmed waste** at both group and per-feature levels — dropped by Lean B |
| `alphamissense` | `alphamissense_`                                                                                                                 | 1  | Waste — dropped by Lean B |
| `conservation`  | `phastcons_`, `phylop_`, `gerp_`, `siphy_`                                                                                       | 17 | **Confirmed waste** at both levels — dropped by Lean B |

The 48 per-group leave-one-out runs (4 tasks × 12 groups) populate `outputs/reports/<task>_ablate_<group>/`. They are NOT in `grand_summary.csv` (too many columns); the master table for them is at `outputs/reports/feature_ablation/master.csv` (504 rows, long format).

---

## How to reproduce any cell

Every setting maps to one `src.train` invocation. Examples:

```bash
PY=C:/Users/Doruk-Topcu/anaconda3/python.exe

# Canonical 4-class tuned
$PY -m src.train --task 4class --tag final_no_adaboost \
    --models KNN NearestCentroid CosineSimilarity DecisionTree LDA QDA \
             LinearSVC RidgeClassifier SGDClassifier HistGradientBoosting

# 3-class with VUS
$PY -m scripts.build_vus_train_parquets    # one-time parquet build
$PY -m src.train --task 3class --tag vus

# Lean B (the ablation-optimal headline)
$PY -m src.train --task 4class \
    --drop-prefixes bayesdel_ hg19_pos original_input_pos revel_ \
                    chasmplus alphamissense_ phastcons_ phylop_ gerp_ siphy_ \
    --tag leanB

# All 48 per-group ablations + summary tables
$PY -m scripts.run_feature_ablation --max-workers 6
$PY -m scripts.summarize_feature_ablation
$PY -m scripts.waste_group_per_feature

# Rebuild the grand summary + table CSVs
$PY -m scripts.build_grand_summary
$PY -m scripts.build_report_table_csvs
```

---

## File index

- **Wide grand table:** [outputs/reports/grand_summary.csv](outputs/reports/grand_summary.csv) — 11 models × 96 columns
- **Long grand table:** [outputs/reports/grand_summary_long.csv](outputs/reports/grand_summary_long.csv) — 257 rows of (model, setting, 4 metrics)
- **Per-table CSVs (one per report table):** [outputs/reports/tables_csv/](outputs/reports/tables_csv/) — 13 files
- **Ablation master:** [outputs/reports/feature_ablation/master.csv](outputs/reports/feature_ablation/master.csv) — every (task, model, group, ablated) cell
- **Headline ablation table:** [outputs/reports/feature_ablation/headline_histgb.csv](outputs/reports/feature_ablation/headline_histgb.csv) — HistGB delta × task × group
- **Per-feature drill-in:** [outputs/reports/feature_ablation/per_feature_waste.csv](outputs/reports/feature_ablation/per_feature_waste.csv) — permutation importance for every feature in every waste group, per task
- **Per-task pivot:** [outputs/reports/feature_ablation/{task}_pivot.csv](outputs/reports/feature_ablation/) — per-task wide pivot of (model × group) → delta
- **Lean models on disk:** [outputs/models/{task}_lean/](outputs/models/) and [outputs/models/{task}_leanB/](outputs/models/) — full saved joblibs per task
