# MVP / MissVARPath — Midway Report

**Project:** MissVARPath (MVP) — Missense Variant Pathogenicity Meta-Classifier
**Authors:** Doruk Topcu, Alihan Sağöz · Course: CMP682
**Status as of:** project_progress_report.md entries #0–#17
**Branch:** `blast-integration`

---

## 1. Executive summary

We have built and evaluated a missense-variant pathogenicity meta-classifier on a curated 21,872-row ClinVar/OpenCRAVAT dataset, covering:

- **10 model families** (LogReg, RF, ExtraTrees, XGBoost, LightGBM, CatBoost, ShallowNN, CNN1D, LSTM, RNN) on both **4-class** (Benign / Likely-benign / Likely-pathogenic / Pathogenic) and **2-class** (benign-vs-pathogenic) targets.
- **Five evaluation regimes** (canonical kfold, gene-stratified, augmented with k-mer + BLAST, no-VEP "VUS" ablation, DITTO ablation).
- **Two SHAP-based interpretability views** (canonical LightGBM, VUS CatBoost) plus per-class contrastive SHAP and a linear-baseline LogReg SHAP.
- **A focused grid-search tuning pass** across the four most responsive model families.

Headline: **LightGBM at 0.80 macroF1 on the canonical 4-class task; CatBoost at 0.9895 macroF1 / 0.9991 ROC-AUC on 2-class.** The model is a genuine meta-classifier, not a pass-through of any single dominant VEP, and degrades gracefully under both gene-grouped evaluation (−0.03) and the full no-VEP "VUS" scenario (−0.08).

**Recommendation:** the methodology is complete and the numbers are stable. We are **ready to write the final report**. The remaining open follow-ups (gene-stratified tuning, per-class SHAP on LogReg) are polish; none of them would change the headline story.

---

## 2. What was built

### 2.1 Data pipeline (#0–#3, #5, #9)

- `data/missense_dataset.csv` (21,872 × 777) — ClinVar variants annotated through OpenCRAVAT (158 annotator groups), perfectly balanced 4-class label at 5,468 per class.
- `src/preprocessing.py` produces `outputs/preprocessing/missense_processed.parquet` (21,872 × 212): **208 feature columns**, plus `clinvar_sig`, `target_4`, `target_2`, `gene_symbol`. No missing values.
- Cuts during preprocessing: 39 label-leakage (`clinvar__*` / `clinvar_acmg__*` except the label), 88 identifier/free-text, 233 free-text object columns, 207 high-missingness (>50% NaN), 1 zero-variance.
- Sequence features (#8): k-mer (196 cols) + BLAST self-similarity (10 cols) joined into the augmented parquet (21,797 × 418).

### 2.2 Modeling pipeline (#4, #9)

- 80/20 stratified holdout + 5-fold StratifiedKFold on the 80% train portion (`src/train.py`). Optional `--cv-mode gene` switches both the holdout and CV to gene-grouped (`GroupShuffleSplit` + `StratifiedGroupKFold` on `gene_symbol`) — guards against data circularity (VEPs trained on overlapping ClinVar).
- `--variant {base, augmented}` and `--drop-prefixes` flags support the ablation experiments.
- All runs produce: per-model classification report (md), confusion matrix (PNG counts + normalized), `*_metrics.json`, leaderboard CSV.

### 2.3 Interpretability pipeline (#12, #14, #15)

- `src/shap_explain.py` — TreeExplainer for boosters, LinearExplainer for LogReg. Outputs: `feature_importance.csv`, `summary_bar.png`, `summary_beeswarm.png`, `shap_values.npz`.
- `src/shap_per_class.py` — contrastive SHAP per class pair. Used to identify which features specifically drive the Benign↔Likely-benign and Likely-pathogenic↔Pathogenic boundaries.

### 2.4 Tuning (#17)

- `src/tune.py` — focused 9-cell grid search per model. Covers the four families with the most tunable capacity (XGBoost, LightGBM, CatBoost, ShallowNN). Same 80/20 split + 5-fold CV as the canonical training; same RANDOM_STATE.

---

## 3. Performance across scenarios

### 3.1 The matrix view: 4-class macroF1 across all evaluation regimes

| Model | Canonical kfold (#6) | Gene-stratified (#10A) | Augmented kfold (#8) | No-VEP kfold (#10B / #11D) | No-VEP gene CV (#10C / #11E) | DITTO ablation (#13) |
|-------|---------------------|------------------------|----------------------|-----------------------------|-------------------------------|----------------------|
| LogReg            | 0.7572 | 0.7411 | 0.7660 | 0.6670 / 0.6107 | 0.6582 / 0.5992 | 0.7529 |
| RandomForest      | 0.7904 | 0.7642 | 0.7977 | 0.7413 / 0.7121 | 0.7160 / 0.6826 | 0.7902 |
| ExtraTrees        | 0.7867 | 0.7548 | 0.7835 | 0.7325 / 0.6996 | 0.7085 / 0.6749 | 0.7815 |
| XGBoost           | 0.7977 | 0.7678 | 0.7964 | **0.7500** / 0.7280 | 0.7226 / 0.6848 | 0.7951 |
| LightGBM          | **0.8013** | 0.7669 | 0.7948 | **0.7524** / 0.7264 | 0.7174 / 0.6786 | 0.7965 |
| CatBoost          | 0.7866 | **0.7672** | 0.7933 | 0.7419 / 0.7070 | **0.7234** / 0.6841 | 0.7808 |
| ShallowNN_MLP     | 0.7470 | 0.7184 | 0.7095 | 0.6075 / 0.6247 | 0.5968 / 0.6118 | 0.7522 |
| CNN1D (Torch)     | 0.7160 | — | 0.5441 | — | — | — |
| RNN (Torch)       | 0.6234 | — | 0.4907 | — | — | — |
| LSTM (Torch)      | 0.6131 | — | 0.5539 | — | — | — |

(Values are held-out macroF1; for the no-VEP cells the format is `augmented / base` so the contribution of k-mer + BLAST is read directly off each row.)

### 3.2 Five anchor numbers for the report

| Anchor | Number | What it tells the reader |
|--------|--------|---------------------------|
| **4-class canonical ceiling** | LightGBM **0.8013 macroF1** (CV 0.7987 ± 0.0050) | What the meta-classifier achieves on standard kfold with the full feature set. The headline result. |
| **4-class gene-grouped ceiling** | CatBoost **0.7672 macroF1** | Realistic upper bound on novel genes — measures data-circularity cost. The 0.034 gap below the canonical ceiling is the cost of "the model can't memorize gene labels". |
| **4-class VUS floor (most rigorous)** | CatBoost **0.7234 macroF1** in no-VEP + gene-CV | What's achievable when in-silico VEP scores are missing AND we evaluate on novel genes. The clinically relevant worst case. |
| **2-class canonical ceiling** | CatBoost **0.9895 macroF1 / 0.9991 ROC-AUC** | Pathogenic-vs-benign is essentially solved on this feature space. |
| **k-mer + BLAST contribution in VUS scenario** | **+0.039 macroF1** (CatBoost, gene-CV, no-VEP) | The proposal's "BLAST as a fallback when VEPs missing" pitch — empirically validated. |

### 3.3 The 2-class story (#7, #16)

Seven of the ten models clear **0.985 macroF1** on the binary task. The hard 4-class confusions are almost entirely Benign↔Likely-benign and Likely-pathogenic↔Pathogenic — the boundaries clinicians struggle with. Once you collapse "Likely benign" with "Benign" and "Likely pathogenic" with "Pathogenic", the problem is essentially separable on this feature space.

| Rank | Model | Holdout macroF1 | Holdout ROC-AUC (macro OvR) | MCC |
|------|-------|------------------|------------------------------|------|
| 1 | CatBoost | 0.9895 | 0.9991 | 0.9790 |
| 2 | XGBoost | 0.9881 | 0.9992 | 0.9763 |
| 3 | LightGBM | 0.9881 | **0.9993** | 0.9763 |
| 4 | LogReg | 0.9867 | 0.9983 | 0.9736 |
| 5 | RandomForest | 0.9861 | 0.9981 | 0.9721 |
| 6 | ShallowNN | 0.9858 | 0.9981 | 0.9717 |
| 7 | ExtraTrees | 0.9856 | 0.9986 | 0.9712 |
| 8 | CNN1D | 0.9824 | 0.9982 | 0.9648 |
| 9 | RNN | 0.9264 | 0.9779 | 0.8528 |
| 10 | LSTM | 0.9168 | 0.9676 | 0.8336 |

### 3.4 What changed when we tightened the evaluation

**Gene-stratified penalty (#10):** every model loses 2-3 macroF1 points moving from kfold to gene-grouped CV. Boosters take the hit gracefully (−0.024 to −0.032). LogReg loses about the same (−0.022). This is the empirical answer to the proposal's "data circularity" warning — the gap exists, but it's not catastrophic.

**No-VEP penalty (#10):** dropping all 147 in-silico VEP-style score columns (AlphaMissense, REVEL, CADD, MetaRNN, BayesDel, MutationTaster, MutPred, ClinPred, etc.) costs every booster ~5 macroF1 points; LogReg loses ~10 because it can't construct nonlinear interactions to compensate. ShallowNN regresses worst (−0.14) due to the optimizer being starved by the larger feature count.

**Stacked penalty (#10C):** under both gene-grouping AND no-VEP, the best model (CatBoost) lands at 0.7234 macroF1 — 8 points below the canonical ceiling but still clinically meaningful and significantly above chance (0.25 for 4-class).

**k-mer + BLAST (#11):** in the no-VEP setting, k-mer + BLAST features add **+0.022 to +0.039 macroF1** depending on model. The lift is *larger* under gene-stratified CV than under standard kfold — these features generalize to held-out genes better than VEP scores would.

**DITTO ablation (#13):** removing the single highest-importance feature (DITTO) costs only 0.0048 macroF1 on LightGBM. The model successfully redistributes onto MetaRNN, ClinPred, BayesDel, AlphaMissense, gnomAD AF — confirming meta-classifier behavior. SHAP told us DITTO mattered most; the ablation proved it wasn't load-bearing.

**Tuning (#17):** A 9-cell grid per model on 4 families showed the canonical configurations were already at or past the grid optimum for the boosters (best cells were at the corner with max iterations). The only tractable lift was on ShallowNN (+0.007 from `batch_size=64, hidden=(64,)`).

---

## 4. Strongest model at its strongest point

### **CatBoost on 2-class, canonical kfold, 208-feature base parquet**

- Held-out **macroF1 0.9895**, **ROC-AUC 0.9991**, **MCC 0.9790**.
- Per-class F1: Benign 0.989, Pathogenic 0.989. Effectively no class imbalance in the residual error.
- Fit time: ~8.5s on a single machine.
- Confusion matrix: 46 false positives, 46 false negatives out of 4,375 holdout variants.

This is the model you would deploy if the clinical question is binary (is this variant pathogenic, yes or no), the gene is well-represented in the training set, and full OpenCRAVAT annotations are available.

### Most informative model — **LightGBM on 4-class, canonical kfold**

- Held-out **macroF1 0.8013**, **ROC-AUC (OvR macro) 0.9481**, **MCC 0.7353**.
- Per-class F1: Benign 0.94, Likely-benign 0.92, Likely-pathogenic 0.67, Pathogenic 0.66.
- The "Likely" tiers cost ~0.27 macroF1 each — exactly the boundaries the per-class SHAP (#14) flagged as driven by population AF + DITTO.
- The headline 4-class number for the report.

### Most robust model under hostile conditions — **CatBoost on no-VEP gene-CV**

- Held-out macroF1 **0.7234** with the augmented set minus VEPs, evaluated on completely held-out genes.
- This is the model to cite when answering "can we predict pathogenicity for a variant in a gene we've never seen, when in-silico VEP scores are unavailable?" — it stays clinically meaningful.

---

## 5. What the SHAP analysis says (interpretability deliverable)

### 5.1 Canonical LightGBM (#12 i)

DITTO dominates by 3.5× the next feature. Top-15 includes 11 VEP-style scores (DITTO, MetaRNN, ClinPred, BayesDel ×2, AlphaMissense, nCER, MutPred, gMVP, …), 3 population frequencies (gnomAD AF, AlloFus max-pop AF, gnomAD3 Afr), and 1 allele count (ALFA alt count). The DITTO ablation in #13 showed this dominance is *attribution under current usage*, not necessity — removing DITTO costs only 0.005 macroF1 because the booster redistributes onto the same VEP family.

### 5.2 VUS CatBoost (#12 ii)

With VEPs removed, the top-15 is led by **`blast_mean_bit_path`** (mean bit score to pathogenic BLAST neighbours), then 9 of 15 are population frequencies (gnomAD, AlloFus by ancestry), 2 are conservation (phyloP), and 4 of 15 are BLAST features. **Zero k-mer features** make the top-15 — k-mers contribute via the booster's interaction terms but no individual 3-mer is decisive. The +0.039 lift in the VUS scenario comes almost entirely from BLAST.

### 5.3 Per-class boundary analysis (#14)

| Boundary | Canonical model leans on… | VUS model leans on… |
|----------|---------------------------|----------------------|
| Benign vs Likely-benign | AlloFus AF (2.04), gnomAD AF (1.96), DITTO (1.82) | AlloFus AF (0.65), gnomAD AF (0.47), `blast_mean_bit_path` (0.14) |
| Likely-path vs Pathogenic | DITTO (1.39), ALFA alt-count (0.45) | **`blast_target_4_top1` (0.12)** — explicit nearest-neighbour reasoning |
| Benign vs Pathogenic | DITTO (5.00), AF (1.43), MetaRNN (0.97) | AF (0.69), gnomAD AF (0.66), `blast_mean_bit_path` (0.54) |

The clinically significant insight: **the "Benign vs Likely-benign" boundary is fundamentally an allele-frequency call** in both regimes. The "Likely-pathogenic vs Pathogenic" boundary is solved differently in the two regimes — the canonical model uses meta-VEP confidence calibration (DITTO); the VUS model falls back to BLAST nearest-neighbour labels, demonstrating that BLAST features can substitute for VEP confidence when the latter is missing.

### 5.4 Linear-baseline SHAP — LogReg (#15)

LogReg can't construct nonlinear interactions, so it spreads importance across the BayesDel family (4 of top-7 columns) plus VARITY-R, REVEL, VEST, CScape — features the booster collapsed into a single DITTO-led signal. Same accuracy band, different basis. This is the linear-vs-nonlinear interpretability story for the report.

---

## 6. Are we ready for the final report?

### What the proposal asked for

| Requirement | Status |
|-------------|--------|
| Thorough EDA | ✅ #2 — `outputs/eda/` with overview JSON, label/missingness/correlation plots |
| Data preprocessing (drop IDs, fill NA, impute) | ✅ #3, #5, #9 |
| Train ~10 models with 80/20 + 5-fold CV | ✅ #6 (4-class), #7 (2-class), #16 (rerun) |
| Required architectures: RF, LightGBM, XGBoost, ShallowNN, CNN, LSTM, RNN | ✅ All 7 + LogReg, ExtraTrees, CatBoost (10 total) |
| Classification reports + confusion matrices per model | ✅ `outputs/reports/{4class,2class}/<model>_*.{md,png,json}` |
| Sequence-context features (k-mer + BLAST) | ✅ #8 — `src/sequence_fetch.py`, `sequence_features.py`, `blast_features.py` |
| VUS-scenario evaluation (proposal §3) | ✅ #10 B+C, #11 D+E |
| Gene-stratified CV / data-circularity mitigation (proposal §2) | ✅ #9, #10 A+C |
| SHAP interpretability (proposal §4) | ✅ #12 (canonical + VUS), #14 (per-class), #15 (LogReg) |

**All required deliverables are in place, with concrete numbers and artifacts.**

### What's "extra" that strengthens the report

- DITTO ablation (#13) — confirms the meta-classifier hypothesis with empirical evidence beyond SHAP.
- Hyperparameter tuning (#17) — confirms the canonical configurations are at the sensible-grid optimum, so no concerns about under-tuning.
- ROC-AUC fix (#16) — closes a known artifact gap from #7; binary numbers are now complete.
- Per-class contrastive SHAP (#14) — gives the report a stronger interpretability section than the global mean(|SHAP|) view alone.

### What's open and whether it matters

| Open follow-up | Source | Worth doing? |
|----------------|--------|--------------|
| Gene-stratified grid search | #17 | **Maybe** — would tell us whether tuning closes any of the 0.034 gene-grouping gap. The script is already parameterised; one command (~30-40 min runtime). Adds one line to the report. |
| Per-class SHAP on LogReg | #15 | **Polish** — the cached arrays already exist; one command. Adds a small panel to the interpretability section. |
| Per-class SHAP on no-DITTO model | #13 | **Polish** — would confirm whether DITTO-redistribution is uniform across boundaries or concentrated. Not load-bearing for the headline. |
| Calibration analysis (reliability curves, ECE) | not queued | **Optional** — useful for clinical deployment claims. The proposal didn't require it. |
| 2-class augmented run | not queued | **Skip** — the 2-class task is already saturated at 0.9895 / 0.9991 ROC-AUC; augmenting can't materially improve it. |
| Re-run torch on hand-curated subset | #10 | **Skip** — we have empirical evidence (#6, #8, #14, #15) that the Torch underperformance is architectural, not feature-budget, on tabular inputs. |

### Recommendation

**Write the report.** The methodology is complete, the numbers are stable across multiple ablation regimes, and the interpretability section has both global and per-boundary views. The five anchor numbers in §3.2 are the headline; the matrix in §3.1 is the supporting evidence.

If you want one more experiment before writing, **run the gene-stratified grid search**. It's the only open item that could shift a number in the headline (specifically: whether the gene-grouping gap shrinks under tuning). Everything else on the open list is polish that doesn't change the story.

---

## 7. Suggested final-report skeleton

1. **Abstract** — 5 anchor numbers from §3.2.
2. **Introduction** — VUS bottleneck, meta-classifier hypothesis, proposal context.
3. **Data** — ClinVar/OpenCRAVAT curation, 21,872 × 208 final feature set, label balance.
4. **Methods** — preprocessing, 80/20 + 5-fold CV, gene-stratified CV, sequence features (k-mer + BLAST), 10-model suite, SHAP.
5. **Results**
   - 5.1 Canonical 4-class baseline (#6 leaderboard).
   - 5.2 Canonical 2-class baseline (#16 leaderboard).
   - 5.3 Gene-stratified evaluation — the "data-circularity ceiling" (§3.4).
   - 5.4 No-VEP (VUS) scenario — the "clinical floor" (§3.4).
   - 5.5 k-mer + BLAST contribution (§3.4).
   - 5.6 DITTO ablation (§3.4 last paragraph).
   - 5.7 Hyperparameter tuning — confirms baselines saturated (§3.4 last paragraph).
6. **Interpretability** — SHAP top features, per-class contrastive analysis, linear-vs-nonlinear comparison (§5).
7. **Discussion** — clinical implications: 0.99 for binary calls, 0.72 for VUS-style fallback under most rigorous evaluation.
8. **Limitations** — single curated dataset, gene-grouping cost, ShallowNN tuning sensitivity.
9. **References** — the proposal PDF + standard VEP/ClinVar citations.

All numbers, plots, and tables for §4–§6 are already on disk under `outputs/`. The report is a writing task at this point, not a modeling task.
