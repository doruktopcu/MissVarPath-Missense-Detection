"""Project paths and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

RAW_CSV = DATA_DIR / "missense_dataset.csv"

EDA_DIR = OUTPUTS_DIR / "eda"
PREPROC_DIR = OUTPUTS_DIR / "preprocessing"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in (EDA_DIR, PREPROC_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset
LABEL_COL = "clinvar__sig"

# Class mapping: 4-class (original task) and 2-class (binary collapse)
CLASS_4 = {"Benign": 0, "Likely benign": 1, "Likely pathogenic": 2, "Pathogenic": 3}
CLASS_4_NAMES = ["Benign", "Likely benign", "Likely pathogenic", "Pathogenic"]
CLASS_2 = {"Benign": 0, "Likely benign": 0, "Likely pathogenic": 1, "Pathogenic": 1}
CLASS_2_NAMES = ["Benign/Likely benign", "Pathogenic/Likely pathogenic"]

# Reproducibility
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

# Processed data outputs
PROCESSED_PARQUET = PREPROC_DIR / "missense_processed.parquet"
FEATURE_LIST_TXT = PREPROC_DIR / "final_features.txt"
DROPPED_COLS_TXT = PREPROC_DIR / "dropped_columns.txt"
