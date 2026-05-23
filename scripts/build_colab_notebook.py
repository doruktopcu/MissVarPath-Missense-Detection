"""Build an end-to-end Colab notebook from the project source.

Reads every module under src/ and scripts/ and embeds them as %%writefile
cells so the resulting notebook is self-contained: on Run-All in Colab,
once the user has uploaded the data + parquets, every step in the report
reproduces from scratch with no external dependencies (no git clone, no
Drive mount). All artifacts land in /content/outputs/ for the user to
inspect or download.

Run: python -m scripts.build_colab_notebook
Output: MissVarPath_Colab.ipynb in the project root.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "MissVarPath_Colab.ipynb"


# ---------- nbformat helpers --------------------------------------------------

def _split(text: str) -> list[str]:
    """nbformat wants `source` as a list of strings with embedded \\n.
    Last line may or may not have a trailing \\n."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


_cell_counter = {"n": 0}
def _next_id() -> str:
    _cell_counter["n"] += 1
    return f"cell-{_cell_counter['n']:04d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(),
            "metadata": {}, "source": _split(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(text),
    }


def writefile(rel_path: str, content: str) -> dict:
    """%%writefile cell that materialises a file under /content/."""
    body = f"%%writefile {rel_path}\n{content}"
    if not body.endswith("\n"):
        body += "\n"
    return code(body)


def file_writefile(disk_path: Path, target_rel_path: str) -> dict:
    return writefile(target_rel_path, disk_path.read_text(encoding="utf-8"))


# ---------- Cells -------------------------------------------------------------

cells: list[dict] = []

# ---- 0. Title + overview ----
cells.append(md("""\
# MissVARPath — End-to-end Colab reproduction

This notebook reproduces every result in the report
*Missense variant pathogenicity prediction — a multi-model benchmark
with VUS deployment, VUS-as-class extension, and a systematic feature
ablation* from raw data to the final headline tables.

It is self-contained: the project source code is materialised by
`%%writefile` cells in the **Source code installation** section, so on
Run-All in Colab there is no `git clone` step, no Drive mount, and no
external dependency beyond the data files you upload (described in the
next section).

## What this notebook produces

By the time Run-All finishes you will have, under `/content/outputs/`:

- `outputs/preprocessing/missense_processed.parquet` — 208-feature base parquet
- `outputs/reports/4class_final_no_adaboost/`, `outputs/reports/2class_final_no_adaboost/` — canonical tuned leaderboards (10 models each)
- `outputs/reports/{4class,2class}_augmented/` — augmented (k-mer + BLAST) variant
- `outputs/reports/{4class,2class}_gene/` — strict gene-stratified CV
- `outputs/reports/{4class,2class}_augmented_no_vep/`, `outputs/reports/4class_augmented_gene_no_vep/` — no-VEP / VUS scenario
- `outputs/reports/{4class,2class}_augmented_raw_only/` — raw-only ablation
- `outputs/reports/4class_no_ditto/` — DITTO ablation
- `outputs/predictions/vus_{pro_,}2class.csv`, `vus_{pro_,}4class.csv` — VUS deployment predictions
- `outputs/figures/vus_per_gene_predictions.png` — per-gene VUS prediction figure
- `outputs/reports/{3class,5class}_vus/` — VUS-as-class study
- `outputs/reports/feature_ablation/` — Phase A group-LOO master + Phase B per-feature drill-in
- `outputs/reports/{2class,3class,4class,5class}_lean/`, `*_leanB/` — Lean A/B ablation-optimal models
- `outputs/shap/canonical_4class_histgb/`, `canonical_2class_histgb/` — SHAP attributions
- `outputs/reports/grand_summary.csv` and `outputs/reports/tables_csv/*.csv` — the at-a-glance tables

## Realistic compute budget

A full Run-All in Colab on a CPU runtime takes **roughly 2 hours** with the
recommended uploads (precomputed parquets + ablation master.csv) because
that skips the slow legs: re-running preprocessing, fetching DNA flanks
from Ensembl REST, regenerating the BLAST database, and the 48-cell
ablation sweep. If you want a full from-scratch reproduction (no
parquets uploaded), expect 6–10 hours on Colab CPU.
"""))

# ---- 1. Uploads & runtime ----
cells.append(md("""\
## 1. Setup

### 1.1 What to upload to Colab before pressing Run-All

Upload the following to the Colab session filesystem (drag-drop into
the **Files** panel, or use `files.upload()` interactively):

**Required at `/content/data/`:**

| File | Size | Purpose |
|---|---|---|
| `missense_dataset.csv`              | 288 MB | Raw 21,872-variant ClinVar / OpenCRAVAT training corpus |
| `missense_VUS_pro_set_annotated.csv`| 77 MB  | 5,468-row annotated VUS pro-set (used for deployment + 3-class/5-class with VUS) |

**Highly recommended at `/content/outputs/preprocessing/`** (skips slow steps):

| File | Size | Purpose |
|---|---|---|
| `missense_processed.parquet`  | ~13 MB | Output of preprocessing (208 features). Without this we re-run preprocessing (~1 min). |
| `missense_augmented.parquet`  | ~15 MB | Augmented parquet (414 features = base + k-mer + BLAST). **Without this we cannot run the augmented / no-VEP / raw-only sections** unless you also have ncbi-blast+ installed and ~30 min for Ensembl REST flank fetching. |

**Highly recommended at `/content/outputs/reports/feature_ablation/`** (skips 1.5+ hours):

| File | Size | Purpose |
|---|---|---|
| `master.csv` | ~50 KB | Precomputed 48-cell ablation master table. Without this the ablation section runs 48 fresh suite runs. |

The notebook detects which files are present and prints what it will
regenerate vs. skip in the **Data location check** section.
"""))

cells.append(md("""\
### 1.2 Install dependencies"""))

cells.append(code("""\
# Pin the same scientific stack used to produce the report.
!pip install -q numpy pandas pyarrow scikit-learn==1.5.* matplotlib seaborn shap joblib tqdm
# requests is for the optional Ensembl REST flank fetch (skip-able)
!pip install -q requests
print("deps installed")
"""))

cells.append(md("""\
### 1.3 Optional: install NCBI BLAST+ (only needed if you don't upload the augmented parquet)

If `missense_augmented.parquet` is uploaded, **skip this cell** — it
saves a few minutes. If you want to regenerate BLAST features from
scratch (or run the BLAST/k-mer pipeline live), uncomment and run.
"""))

cells.append(code("""\
# !apt-get update -qq && apt-get install -y -qq ncbi-blast+
# !which makeblastdb blastn
"""))

# ---- 2. Project directory + source code installation ----
cells.append(md("""\
## 2. Project directory setup

We materialise the project structure under `/content/`. The src/ and
scripts/ modules are written by `%%writefile` cells in the next
sub-section, so the notebook is fully self-contained.
"""))

cells.append(code("""\
import os, sys
from pathlib import Path
os.chdir("/content")
for d in ["src", "scripts", "data",
          "outputs/preprocessing", "outputs/models", "outputs/reports",
          "outputs/predictions", "outputs/figures", "outputs/sequences",
          "outputs/shap", "outputs/eda", "outputs/tuning",
          "outputs/reports/feature_ablation"]:
    Path(d).mkdir(parents=True, exist_ok=True)
# Ensure /content is on sys.path so `import src.train` works.
if "/content" not in sys.path:
    sys.path.insert(0, "/content")
print("project tree ready under /content/")
"""))

cells.append(md("""\
### 2.1 Source code (auto-written by `%%writefile`)

Every cell below materialises one project module. **You do not need to
edit anything here** — the cells are an exact copy of the project's
`src/` and `scripts/` files at the time this notebook was built. The
modules import from each other normally.
"""))

# Write src/ modules
SRC = ROOT / "src"
SCR = ROOT / "scripts"

# Order matters for clarity but Python imports are lazy so any order works.
src_files = [
    "__init__.py",
    "config.py",
    "utils.py",
    "data_loader.py",
    "preprocessing.py",
    "models.py",
    "train.py",
    "tune.py",
    "sequence_fetch.py",
    "sequence_features.py",
    "blast_features.py",
    "build_augmented_dataset.py",
    "shap_explain.py",
    "shap_per_class.py",
]
for fname in src_files:
    p = SRC / fname
    if not p.exists():
        continue
    cells.append(md(f"`src/{fname}`"))
    cells.append(file_writefile(p, f"/content/src/{fname}"))

# Need a scripts/__init__.py too so 'import scripts.xxx' resolves
cells.append(md("`scripts/__init__.py` (empty marker so `python -m scripts.*` works)"))
cells.append(writefile("/content/scripts/__init__.py", ""))

script_files = [
    "predict_vus.py",
    "build_vus_pro_set.py",
    "build_vus_train_parquets.py",
    "run_feature_ablation.py",
    "summarize_feature_ablation.py",
    "waste_group_per_feature.py",
    "build_grand_summary.py",
    "build_report_table_csvs.py",
    "summarize_runs.py",
]
for fname in script_files:
    p = SCR / fname
    if not p.exists():
        continue
    cells.append(md(f"`scripts/{fname}`"))
    cells.append(file_writefile(p, f"/content/scripts/{fname}"))

# ---- 3. Data location check ----
cells.append(md("""\
## 3. Data location check

Verify which inputs are present and decide which steps will be
regenerated vs. skipped. Prints a clear status table.
"""))

cells.append(code("""\
from pathlib import Path
import os, sys
os.chdir("/content")

CHECKS = [
    ("data/missense_dataset.csv",                                 "raw training CSV",   True),
    ("data/missense_VUS_pro_set_annotated.csv",                   "annotated VUS CSV",  True),
    ("outputs/preprocessing/missense_processed.parquet",          "208-feature parquet (skips preprocessing if present)", False),
    ("outputs/preprocessing/missense_augmented.parquet",          "augmented parquet (k-mer+BLAST) — required for augmented/no-VEP/raw-only", False),
    ("outputs/reports/feature_ablation/master.csv",               "precomputed ablation master (skips 1.5+ hours)", False),
]

print(f"{'status':>8}  {'path':<60} purpose")
print("-" * 110)
state = {}
for rel, label, required in CHECKS:
    exists = Path(rel).exists()
    state[rel] = exists
    flag = "OK" if exists else ("MISSING (required)" if required else "absent")
    sz = f"{Path(rel).stat().st_size/1e6:.1f} MB" if exists else "-"
    print(f"{flag:>8}  {rel:<60} {label}  [{sz}]")

# Hard-fail only on required artifacts.
missing_required = [rel for rel, lbl, req in CHECKS if req and not state[rel]]
if missing_required:
    raise SystemExit(f"Missing required uploads: {missing_required}. Upload them, then re-run from this cell.")
"""))

# ---- 4. Preprocessing ----
cells.append(md("""\
## 4. Data preprocessing

If `outputs/preprocessing/missense_processed.parquet` was uploaded we
skip this section. Otherwise we run `src.preprocessing.run()`, which:

1. Tidies column names (lowercase, single underscores).
2. Filters to valid 4-class labels (drops VUS / conflicting / etc.).
3. Drops label-leakage columns (everything in `clinvar*` except the label).
4. Drops identifier / free-text columns (~98 columns of HGVS / dbSNP / etc.).
5. Coerces ≥95% numerically-parseable object columns to numeric.
6. Drops the remaining text columns.
7. Drops columns with >50% missingness.
8. Imputes numeric NaNs with the per-column median; mode for categoricals.
9. Drops near-zero-variance columns.
10. Saves the parquet + a feature manifest.
"""))

cells.append(code("""\
from pathlib import Path
PROC = Path("outputs/preprocessing/missense_processed.parquet")
if PROC.exists():
    print(f"[preprocessing] {PROC} already present — skipping.")
else:
    from src.preprocessing import run as preprocess_run
    res = preprocess_run()
    print(f"[preprocessing] wrote {PROC}  shape={res.df.shape}")
"""))

# ---- 5. Canonical tuned baselines ----
cells.append(md("""\
## 5. Canonical tuned baselines (4-class + 2-class)

Trains the full 10-model tuned suite on both the original 4-class
target (Benign / Likely-benign / Likely-pathogenic / Pathogenic) and
the collapsed 2-class target (benign-side / pathogenic-side). The
tuned hyperparameters from §3.5 of the report are already hard-coded
into `src/models.py::MODEL_SPECS`, so this just runs the suite under
those defaults — no re-tuning needed.

**Headline expected numbers (HistGradientBoosting):**
- 4-class macro-F1 ≈ 0.795
- 2-class macro-F1 ≈ 0.987

Wall-clock on a Colab CPU runtime: ~5–8 min per task.
"""))

cells.append(code("""\
from src.train import run as train_run
MODELS_10 = ["KNN", "NearestCentroid", "CosineSimilarity",
             "DecisionTree", "LDA", "QDA", "LinearSVC",
             "RidgeClassifier", "SGDClassifier", "HistGradientBoosting"]
print(">>> 4-class tuned, AdaBoost excluded")
train_run("4class", MODELS_10, tag="final_no_adaboost")
"""))

cells.append(code("""\
print(">>> 2-class tuned, AdaBoost excluded")
train_run("2class", MODELS_10, tag="final_no_adaboost")
"""))

cells.append(md("""\
### 5.1 Headline leaderboards"""))
cells.append(code("""\
import pandas as pd
for task in ["4class", "2class"]:
    df = pd.read_csv(f"outputs/reports/{task}_final_no_adaboost/leaderboard.csv")
    print(f"\\n=== {task} tuned ({len(df)} models) ===")
    print(df[["model","cv_macro_f1_mean","holdout_macro_f1",
              "holdout_acc","holdout_mcc"]].round(4).to_string(index=False))
"""))

# ---- 6. Augmented variant ----
cells.append(md("""\
## 6. Augmented variant (k-mer + BLAST locus-neighbour features)

The augmented variant adds **196 k-mer features** (every 3-mer count
in the ±25 bp flanks for ref + alt + diff, plus GC content + Shannon
entropy) and **10 BLAST-derived features** (counts and bit-scores of
top-10 train-set neighbours by alt-flank similarity) on top of the
208-feature base parquet.

**Two paths:**

- **Fast path:** `outputs/preprocessing/missense_augmented.parquet`
  was uploaded → we use it directly.
- **Slow path:** regenerate. Requires (a) `ncbi-blast+` installed (see
  §1.3) and (b) ~30 min of Ensembl REST traffic to fetch flanks. This
  notebook implements the slow path defensively; if either prerequisite
  is missing it skips the augmented section with a clear message.

> **Important methodological caveat** (kept from §3.4 of the report):
> these BLAST features are **label-aware locus-neighbour features**, not
> biologically meaningful homology features. They are dominated by
> same-locus / same-gene proximity in ClinVar rather than orthology or
> paralogy. Under gene-stratified CV (§7) they largely collapse — the
> empirical test for what they actually encode.
"""))

cells.append(code("""\
from pathlib import Path
AUG = Path("outputs/preprocessing/missense_augmented.parquet")
AUG_AVAILABLE = AUG.exists()
print(f"[augmented] parquet present = {AUG_AVAILABLE}")
if not AUG_AVAILABLE:
    # Try the slow path if BLAST is available; otherwise warn and skip.
    import shutil
    has_blast = shutil.which("makeblastdb") is not None and shutil.which("blastn") is not None
    if has_blast:
        print("[augmented] No parquet, but BLAST tools detected — regenerating from scratch.")
        # 1) Fetch flanks (slow: ~30 min on Ensembl REST)
        from src import sequence_fetch
        sequence_fetch.run()
        # 2) Build k-mer features
        from src import sequence_features
        sequence_features.run()
        # 3) Build BLAST features (re-runs the 80/20 split)
        from src import blast_features
        blast_features.run()
        # 4) Join into augmented parquet
        from src import build_augmented_dataset
        build_augmented_dataset.run()
        AUG_AVAILABLE = AUG.exists()
    else:
        print("[augmented] Neither parquet uploaded nor BLAST tools installed; sections 6-7 will skip.")
"""))

cells.append(md("""\
### 6.1 Train the suite on the augmented parquet"""))
cells.append(code("""\
if AUG_AVAILABLE:
    print(">>> 4-class augmented")
    train_run("4class", MODELS_10, variant="augmented", tag="augmented")
    print(">>> 2-class augmented")
    train_run("2class", MODELS_10, variant="augmented", tag="augmented")
else:
    print("[skipped] augmented parquet not available")
"""))

# ---- 7. Strict regimes ----
cells.append(md("""\
## 7. Strict-evaluation regimes

Five evaluation regimes that stress-test the canonical headline. Each
is one `src.train` invocation with different flags:

- **Gene-stratified CV** (`--cv-mode gene`): no gene appears in both
  train and test folds — strictest test of generalisation across genes.
- **No-VEP kfold / gene** (`--drop-prefixes <all-predictor-prefixes>`):
  drops every learned-predictor score + multi-species conservation score
  (130 columns). Simulates the deployment case where the predictor
  pipeline has not run on the variant.
- **Raw-only** (`--keep-prefixes <raw-prefixes>`): the strictest ablation
  — keep only population allele frequencies, genomic position, and
  direct sequence-composition features. Predicts from data, not from
  any pre-existing model's output.
- **No-DITTO**: drop only the single `ditto_` column. Sanity check on
  whether the suite is bottlenecked on its top-SHAP feature.

Wall-clock: roughly 5–10 min per task on Colab CPU.
"""))

cells.append(md("""\
### 7.1 Gene-stratified CV"""))
cells.append(code("""\
print(">>> 4-class gene-CV")
train_run("4class", MODELS_10, cv_mode="gene", tag="gene")
print(">>> 2-class gene-CV")
train_run("2class", MODELS_10, cv_mode="gene", tag="gene")
"""))

cells.append(md("""\
### 7.2 No-VEP scenario (augmented variant; skipped if augmented parquet absent)"""))
cells.append(code("""\
VEP_PREFIXES = [
    "alphamissense_", "cadd_", "cadd_exome_", "revel_", "sift_",
    "metarnn_", "bayesdel_", "fathmm_", "mutationtaster_", "provean_",
    "esm1b_", "eve_", "primateai_", "mvp_", "ditto_",
    "mutation_assessor_", "vest_", "chasmplus", "mutpred1_", "aloft_",
    "gmvp_", "polyphen2_", "gerp_", "phastcons_", "phylop_", "siphy_",
]
if AUG_AVAILABLE:
    print(">>> 4-class no-VEP kfold (augmented)")
    train_run("4class", MODELS_10, variant="augmented",
              drop_prefixes=VEP_PREFIXES, tag="no_vep")
    print(">>> 2-class no-VEP kfold (augmented)")
    train_run("2class", MODELS_10, variant="augmented",
              drop_prefixes=VEP_PREFIXES, tag="no_vep")
    print(">>> 4-class no-VEP gene-CV (augmented) — operational VUS scenario")
    train_run("4class", MODELS_10, variant="augmented", cv_mode="gene",
              drop_prefixes=VEP_PREFIXES, tag="no_vep")
else:
    print("[skipped] augmented parquet not available")
"""))

cells.append(md("""\
### 7.3 Raw-only (predicting from data, not from predictors)"""))
cells.append(code("""\
RAW_PREFIXES = [
    "alfa_", "allofus250k_", "gnomad_", "gnomad3_", "regeneron_",
    "hg19_pos", "original_input_pos", "kmer_", "gc_", "entropy_",
]
if AUG_AVAILABLE:
    print(">>> 4-class raw-only (augmented)")
    train_run("4class", MODELS_10, variant="augmented",
              keep_prefixes=RAW_PREFIXES, tag="raw_only")
    print(">>> 2-class raw-only (augmented)")
    train_run("2class", MODELS_10, variant="augmented",
              keep_prefixes=RAW_PREFIXES, tag="raw_only")
else:
    print("[skipped] augmented parquet not available")
"""))

cells.append(md("""\
### 7.4 No-DITTO (single-feature sanity check)"""))
cells.append(code("""\
print(">>> 4-class no-DITTO")
train_run("4class", MODELS_10, drop_prefixes=["ditto_"], tag="no_ditto")
"""))

# ---- 8. VUS deployment ----
cells.append(md("""\
## 8. VUS deployment

Apply the tuned 4-class and 2-class HistGB classifiers to the
5,468-row **annotated VUS pro-set** assembled from ClinVar 2-star +
3-star review-status filters and processed through the same
OpenCRAVAT pipeline as the training data.

The deployment yields two interpretable signals:
1. the **distribution** of predicted classes — how the trained
   classifier partitions the uncertain pool;
2. **cross-task consistency** between the 4-class and 2-class heads
   (both trained independently).

The script also writes per-variant probabilities with identifier
columns (`base__chrom`, `base__pos`, `base__ref_base`, `base__alt_base`,
`base__hugo`) so the output joins back to the input.
"""))

cells.append(code("""\
from scripts.predict_vus import predict
from pathlib import Path
VUS_CSV = Path("data/missense_VUS_pro_set_annotated.csv")
assert VUS_CSV.exists(), VUS_CSV

for task in ["4class", "2class"]:
    model_p = Path(f"outputs/models/{task}_final_no_adaboost/histgradientboosting.joblib")
    feats_p = model_p.parent / "features.txt"
    out_p = Path(f"outputs/predictions/vus_pro_{task}.csv")
    predict(VUS_CSV, model_p, feats_p, task, out_p)
"""))

cells.append(md("""\
### 8.1 Per-gene VUS prediction figure

Stacked-bar 4-class breakdown for the top-20 most-frequent genes in
the pro-set, sorted by P+LP rate. The figure that appears in §4.5 of
the report.
"""))

cells.append(code("""\
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/predictions/vus_pro_4class.csv")
gene_col = "base__hugo" if "base__hugo" in df.columns else "gene_symbol"
top20 = df[gene_col].value_counts().head(20).index.tolist()
sub = df[df[gene_col].isin(top20)]
class_cols = [c for c in sub.columns if c.startswith("proba_")]
order_classes = ["Benign", "Likely_benign", "Likely_pathogenic", "Pathogenic"]
class_to_col = {c: next((cc for cc in class_cols if c in cc), None) for c in order_classes}

# Compute prediction-class counts per gene
counts = pd.crosstab(sub[gene_col], sub["pred_label"])
counts = counts.reindex(top20)
# Stack normalisation: fractions
frac = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
# Sort by pathogenic-side rate
path_rate = frac.get("Pathogenic", 0) + frac.get("Likely pathogenic", 0)
order = path_rate.sort_values(ascending=False).index
frac = frac.loc[order]
counts = counts.loc[order]

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"Pathogenic":"#b71c1c","Likely pathogenic":"#ef6c00",
          "Likely benign":"#1976d2","Benign":"#43a047"}
bottoms = pd.Series([0.0]*len(frac), index=frac.index)
for label in ["Pathogenic","Likely pathogenic","Likely benign","Benign"]:
    if label not in frac.columns:
        continue
    ax.barh(frac.index, frac[label], left=bottoms, color=colors.get(label,"#777"),
            label=label, edgecolor="white", linewidth=0.5)
    bottoms = bottoms + frac[label]
for i, gene in enumerate(frac.index):
    ax.text(1.01, i, f"n={int(counts.loc[gene].sum())}", va="center", fontsize=9)
ax.set_xlim(0, 1.10)
ax.invert_yaxis()
ax.set_xlabel("Predicted-class fraction (top-20 genes by P+LP rate)")
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Per-gene VUS prediction breakdown (4-class HistGB)")
Path("outputs/figures").mkdir(exist_ok=True, parents=True)
fig.tight_layout()
fig.savefig("outputs/figures/vus_per_gene_predictions.png", dpi=150)
plt.show()
"""))

cells.append(md("""\
### 8.2 Cross-task consistency check"""))
cells.append(code("""\
d2 = pd.read_csv("outputs/predictions/vus_pro_2class.csv")
d4 = pd.read_csv("outputs/predictions/vus_pro_4class.csv")
join_keys = [c for c in ("base__chrom","base__pos","base__ref_base","base__alt_base") if c in d2.columns and c in d4.columns]
merged = d2.merge(d4, on=join_keys, suffixes=("_2c","_4c"))
# Collapse 4c to pathogenic-side / benign-side
collapsed = merged["pred_label_4c"].isin(["Pathogenic","Likely pathogenic"]).astype(int)
two_class_path = (merged["pred_class_2c"] == 1).astype(int)
agree = (collapsed == two_class_path).mean()
print(f"Cross-task agreement (4c collapsed vs 2c): {agree:.4f}")
print("\\n2-class prediction distribution:")
print(d2["pred_label"].value_counts(normalize=True).round(3))
print("\\n4-class prediction distribution:")
print(d4["pred_label"].value_counts(normalize=True).round(3))
"""))

# ---- 9. VUS as class ----
cells.append(md("""\
## 9. VUS-as-class study (3-class + 5-class)

Re-train the full suite with the **strict-VUS** subset of the
pro-set (rows with `clinvar__sig == "Uncertain significance"` only,
5,428 of 5,468 after filtering NaN / drifted labels) added as a
labelled class.

Two task definitions:

| Task    | Classes | Counts |
|---------|---------|---|
| 3-class | Benign-side / Pathogenic-side / VUS | 10,936 / 10,936 / 5,428 |
| 5-class | Benign / Likely-benign / Likely-pathogenic / Pathogenic / VUS | 5,468 × 4 + 5,428 |

The 3-class collapses Likely-* into definitive labels (same rule as
the canonical 2-class task). The 5-class keeps the original four
labels and adds VUS as a fifth class.

**Why this matters:** the VUS deployment in §8 *scores* unclassified
variants with classifiers trained on labelled data. The VUS-as-class
study asks the complementary question — can the model learn to
*recognise* uncertain variants as a distinct class? The expected
result (and the headline finding): **yes**, VUS is one of the cleanest
classes to recognise, with F1 ≈ 0.90 in the 5-class study and the
4-class metrics essentially unchanged when VUS is added.
"""))

cells.append(code("""\
import subprocess, sys
print(">>> Building 3-class and 5-class parquets")
out = subprocess.run([sys.executable, "-m", "scripts.build_vus_train_parquets"],
                     capture_output=True, text=True)
print(out.stdout)
if out.returncode != 0:
    print(out.stderr); raise SystemExit(out.returncode)
"""))

cells.append(code("""\
MODELS_11 = MODELS_10 + ["AdaBoost"]
print(">>> 3-class with VUS")
train_run("3class", MODELS_11, tag="vus")
print(">>> 5-class with VUS")
train_run("5class", MODELS_11, tag="vus")
"""))

cells.append(md("""\
### 9.1 Per-class breakdown"""))
cells.append(code("""\
import json, numpy as np
for task, names in [
    ("3class", ["Benign-side","Pathogenic-side","VUS"]),
    ("5class", ["Benign","Likely benign","Likely pathogenic","Pathogenic","VUS"])
]:
    j = json.load(open(f"outputs/reports/{task}_vus/histgradientboosting_metrics.json"))
    h = j["holdout"]
    print(f"\\n=== {task} HistGB: acc={h['accuracy']:.4f}  macro-F1={h['macro_f1']:.4f}  MCC={h['mcc']:.4f}")
    print("per-class F1:")
    for k, v in h["f1_per_class"].items():
        print(f"  {k:24s}  {v:.4f}")
"""))

# ---- 10. Feature ablation ----
cells.append(md("""\
## 10. Feature ablation study

Two-phase ablation across all four tasks × the full 10-model suite:

- **Phase A — group LOO:** 12 functional feature groups (population_af,
  conservation, other_vep, chasmplus, ditto, metarnn, ...). For each
  (task × group) we re-run the full suite under `--drop-prefixes <group>`
  and compare HistGB delta against the baseline. Identifies which groups
  are useful and which are waste at the group level.
- **Phase B — per-feature permutation:** for every feature inside the
  group-LOO-confirmed waste groups, we permute the column on the
  held-out test set (n_repeats=10) and measure the macro-F1 drop on
  the trained baseline HistGB. Surfaces individual features hiding
  inside ``waste'' groups whose marginal contribution is large
  (the key example: `ditto_score` alone has permutation importance
  ≈ 0.13, but the `ditto` group at LOO costs only ≈ 0.004 because
  metarnn / bayesdel / etc. compensate).

If `outputs/reports/feature_ablation/master.csv` was uploaded we skip
Phase A's 48 fresh runs (~1.5 hours on Colab CPU) and use the
precomputed table. Otherwise the cell launches the full sweep.
"""))

cells.append(code("""\
import subprocess, sys
from pathlib import Path
MASTER = Path("outputs/reports/feature_ablation/master.csv")
if MASTER.exists():
    print(f"[ablation] {MASTER} uploaded — skipping the 48-run sweep, will summarise the existing table.")
else:
    print("[ablation] No master.csv uploaded; launching the 48-run sweep "
          "(~1.5 hours on Colab CPU).")
    # max_workers=2 is realistic for Colab CPU's 2 vCPU; locally we used 6.
    out = subprocess.run([sys.executable, "-m", "scripts.run_feature_ablation",
                          "--max-workers", "2"], capture_output=True, text=True)
    print(out.stdout[-2000:])
    if out.returncode != 0:
        print(out.stderr); raise SystemExit(out.returncode)
"""))

cells.append(md("""\
### 10.1 Phase A summary — group LOO deltas

Builds the per-task pivots, the HistGB headline table, the waste-group list,
and the per-feature drill-in input. **If you uploaded `master.csv`** the
script uses it directly instead of trying to rebuild from the 48 per-group
leaderboards (which don't exist when the sweep was skipped). The fallback
path requires the per-group leaderboards on disk.
"""))
cells.append(code("""\
import subprocess, sys
out = subprocess.run([sys.executable, "-m", "scripts.summarize_feature_ablation"],
                     capture_output=True, text=True)
print(out.stdout)
if out.returncode != 0:
    print(f"!!! summarize_feature_ablation failed (rc={out.returncode})")
    print(out.stderr)
"""))

cells.append(md("""\
### 10.2 Phase B — per-feature permutation inside the waste groups

Requires `waste_groups.txt` (produced by Phase A above). If Phase A wasn't
able to run (for example you uploaded an outdated `master.csv` and have no
per-group leaderboards on disk), Phase B prints a skip message and exits 0.
"""))
cells.append(code("""\
import subprocess, sys
out = subprocess.run([sys.executable, "-m", "scripts.waste_group_per_feature"],
                     capture_output=True, text=True)
print(out.stdout)
if out.returncode != 0:
    print(f"!!! waste_group_per_feature failed (rc={out.returncode})")
    print(out.stderr)
"""))

# ---- 11. Lean model ----
cells.append(md("""\
## 11. Ablation-optimal Lean model

Two candidate lean feature subsets:

- **Lean A** — drop **all 8** group-LOO-confirmed waste groups
  (ditto, metarnn, bayesdel, position, revel, chasmplus,
  alphamissense, conservation). 110/207 features kept; 47% reduction.
- **Lean B** — same drop list but **keep** `ditto_score` and
  `metarnn_score` (the per-feature Phase-B winners). 113/207 features;
  45% reduction.

Run both for every task and look at HistGB head-to-head against
baseline. The expected finding: Lean B loses at most 0.012 macro-F1
on any task, several models actually improve (KNN, Cosine,
NearestCentroid — the curse-of-dimensionality story).
"""))

cells.append(code("""\
DROP_A = ["ditto_","metarnn_","bayesdel_","hg19_pos","original_input_pos",
          "revel_","chasmplus","alphamissense_","phastcons_","phylop_",
          "gerp_","siphy_"]
DROP_B = ["bayesdel_","hg19_pos","original_input_pos","revel_","chasmplus",
          "alphamissense_","phastcons_","phylop_","gerp_","siphy_"]
MODELS_11 = MODELS_10 + ["AdaBoost"]

for task in ["4class","2class","3class","5class"]:
    print(f">>> {task} Lean A")
    train_run(task, MODELS_11, drop_prefixes=DROP_A, tag="lean")
    print(f">>> {task} Lean B")
    train_run(task, MODELS_11, drop_prefixes=DROP_B, tag="leanB")
"""))

cells.append(md("""\
### 11.1 Head-to-head HistGB summary"""))
cells.append(code("""\
import pandas as pd
baselines = {"4class":"4class_final_no_adaboost","2class":"2class_final_no_adaboost",
             "3class":"3class_vus","5class":"5class_vus"}
rows = []
for task, bd in baselines.items():
    b = pd.read_csv(f"outputs/reports/{bd}/leaderboard.csv")
    a = pd.read_csv(f"outputs/reports/{task}_lean/leaderboard.csv")
    bb = pd.read_csv(f"outputs/reports/{task}_leanB/leaderboard.csv")
    def f1(df): return float(df[df["model"]=="HistGradientBoosting"]["holdout_macro_f1"].iloc[0])
    rows.append({"task":task, "baseline":f1(b), "leanA":f1(a), "leanB":f1(bb)})
df = pd.DataFrame(rows)
df["leanA_delta"] = df["baseline"] - df["leanA"]
df["leanB_delta"] = df["baseline"] - df["leanB"]
print(df.round(4).to_string(index=False))
"""))

# ---- 12. SHAP ----
cells.append(md("""\
## 12. SHAP attributions

Tree-explainer SHAP on a 1,000-row sample of the held-out test set
(`shap.TreeExplainer` is exact and fast for HistGB). Produces:

- `outputs/shap/<tag>/feature_importance.csv` — per-feature mean(|SHAP|)
  overall and per-class.
- `outputs/shap/<tag>/summary_bar.png` — top-30 features bar chart.
- `outputs/shap/<tag>/summary_beeswarm.png` — beeswarm for class 0.

Plus per-class contrastive SHAP (the per-class difference) for the
key diagnostic boundaries: Benign↔Likely-benign,
Likely-pathogenic↔Pathogenic.
"""))

cells.append(code("""\
from src.shap_explain import run as shap_run
from src.shap_per_class import run as shap_perclass_run
shap_run("4class", "HistGradientBoosting", tag="canonical_4class_histgb", n_explain=1000)
shap_run("2class", "HistGradientBoosting", tag="canonical_2class_histgb", n_explain=1000)
shap_perclass_run("canonical_4class_histgb")
shap_perclass_run("canonical_2class_histgb")
"""))

cells.append(md("""\
### 12.1 Top-15 features (canonical 4-class)"""))
cells.append(code("""\
import pandas as pd
imp = pd.read_csv("outputs/shap/canonical_4class_histgb/feature_importance.csv")
print(imp.head(15).round(3).to_string(index=False))
"""))

# ---- 13. Grand summary ----
cells.append(md("""\
## 13. Grand summary CSVs

Walks every setting on disk and produces:

- `outputs/reports/grand_summary.csv` — wide: 11 models × ~96 columns
  (each setting contributes 4 metric columns: macro_f1, accuracy,
  macro_precision, macro_recall). The at-a-glance table.
- `outputs/reports/grand_summary_long.csv` — long format, easier for
  filtering / pivoting.
- `outputs/reports/tables_csv/*.csv` — one CSV per aggregated table in
  the report (gene_strat, no_vep, raw_only, vus_class, ablation_headline,
  per_feature, lean_headline, lean_full, anchors, leaderboards).
"""))

cells.append(code("""\
import subprocess, sys
out = subprocess.run([sys.executable, "-m", "scripts.build_grand_summary"],
                     capture_output=True, text=True)
print(out.stdout[-2000:])
if out.returncode != 0:
    print(f"!!! build_grand_summary failed (rc={out.returncode})")
    print(out.stderr)
out = subprocess.run([sys.executable, "-m", "scripts.build_report_table_csvs"],
                     capture_output=True, text=True)
print(out.stdout)
if out.returncode != 0:
    print(f"!!! build_report_table_csvs failed (rc={out.returncode})")
    print(out.stderr)
"""))

cells.append(md("""\
### 13.1 Headline (HistGradientBoosting macro-F1 across every setting)"""))
cells.append(code("""\
import pandas as pd
g = pd.read_csv("outputs/reports/grand_summary.csv", index_col=0)
hgb = g.loc["HistGradientBoosting"].filter(regex=r"\\.macro_f1$").rename(lambda c: c.replace(".macro_f1",""))
print(hgb.round(4).to_string())
"""))

# ---- 14. Wrap-up ----
cells.append(md("""\
## 14. Artifact tree (everything we produced)

Walks `/content/outputs/` and prints a summary of every directory.
You can download any CSV / PNG / joblib via the Colab Files panel
(right-click → Download), or zip the whole tree:

```python
!zip -qr /content/outputs.zip /content/outputs
from google.colab import files
files.download("/content/outputs.zip")
```
"""))

cells.append(code("""\
from pathlib import Path
ROOT = Path("/content/outputs")
total_files = 0
for d in sorted(ROOT.rglob("*")):
    if d.is_dir():
        files_here = sum(1 for _ in d.iterdir() if _.is_file())
        if files_here:
            print(f"  {d.relative_to(ROOT)}  -- {files_here} files")
            total_files += files_here
print(f"\\nTotal files under outputs/: {total_files}")
"""))

cells.append(md("""\
## 15. Reproducibility checks

- `RANDOM_STATE = 42` everywhere; the 80/20 + 5-fold CV split is
  deterministic across runs.
- `HistGradientBoosting` headline macro-F1 should land at:
  4-class ≈ 0.795, 2-class ≈ 0.987, 3-class ≈ 0.946, 5-class ≈ 0.798.
- Lean B HistGB delta vs. baseline: at most 0.012 macro-F1 on any task.
- All numbers cross-reference with `final_report.pdf` in the GitHub
  repository.

If any of the headline numbers drift by more than the noise floor
(±0.005 macro-F1), check that:
1. `outputs/preprocessing/missense_processed.parquet` matches the
   project parquet (208 features, 21,872 rows).
2. The 80/20 train/test split was computed with `RANDOM_STATE = 42`
   (see `src/config.py`).
3. The model registry in `src/models.py::MODEL_SPECS` has not been
   modified — the tuned defaults are baked in there.
"""))


# ---------- write notebook ----------------------------------------------------

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH}  ({len(cells)} cells)")
