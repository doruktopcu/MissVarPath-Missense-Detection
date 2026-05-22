"""VUS inference — apply a trained MissVARPath classifier to unlabeled variants.

Expected input
--------------
A CSV/TSV/parquet of VUS variants annotated through the *same* OpenCRAVAT
pipeline as the training dataset, so the column names match the raw
777-column schema. The script does NOT require labels; rows whose
``clinvar__sig`` is "Uncertain significance" (or missing) are fine.

Output
------
A CSV with one row per input variant containing:
  - variant identifier columns (chrom, pos, ref, alt, gene if present)
  - ``pred_class``           — integer predicted class (0/1 for 2class; 0..3 for 4class)
  - ``pred_label``           — human-readable label
  - ``proba_<class_name>``   — per-class probability (if the model supports
                               ``predict_proba``)

Usage
-----
    python -m scripts.predict_vus \\
        --input  data/vus_missense.csv \\
        --task   2class \\
        --model  outputs/models/2class_final_no_adaboost/histgradientboosting.joblib \\
        --output outputs/predictions/vus_2class.csv

Defaults pick the headline tuned HistGradientBoosting on the 2-class task.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASS_2_NAMES,
    CLASS_4_NAMES,
    MODELS_DIR,
    PROCESSED_PARQUET,
)
from src.preprocessing import (
    coerce_numeric_objects,
    drop_identifier_columns,
    drop_label_leakage,
    drop_remaining_text_columns,
    tidy_column,
)


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Same column-tidy step the training pipeline uses, applied uniformly."""
    df = df.copy()
    rename_map = {c: tidy_column(c) for c in df.columns}
    # Mirror the training preprocessing's special-cases
    if "clinvar__sig" in df.columns:
        rename_map["clinvar__sig"] = "clinvar_sig"
    if "base__hugo" in df.columns:
        rename_map["base__hugo"] = "gene_symbol"
    return df.rename(columns=rename_map)


def _load_training_medians(features: list[str]) -> pd.Series:
    """Pull per-feature medians from the training parquet so VUS imputation is
    aligned with what the model saw at fit time. Falls back to NaN if a feature
    is missing (caller will then fill with 0)."""
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PARQUET} not found — run `python -m src.preprocessing` "
            "first so we can read training-set medians for imputation."
        )
    train = pd.read_parquet(PROCESSED_PARQUET)
    available = [f for f in features if f in train.columns]
    medians = train[available].median(numeric_only=True)
    return medians.reindex(features)


def predict(input_path: Path, model_path: Path, features_path: Path,
            task: str, output_path: Path) -> pd.DataFrame:
    class_names = CLASS_2_NAMES if task == "2class" else CLASS_4_NAMES

    raw = _read_any(input_path)
    print(f"[predict_vus] Loaded {len(raw)} variants from {input_path}")

    # Same column-pruning passes as src/preprocessing.run(), but WITHOUT the
    # label filter — VUS rows by definition won't have a valid 4-class label.
    df, _ = drop_label_leakage(raw)
    df, _ = drop_identifier_columns(df)
    df, _ = coerce_numeric_objects(df)
    df, _ = drop_remaining_text_columns(df)
    df = _tidy_columns(df)

    # Align to the trained feature manifest. Any feature the model expects but
    # the input lacks gets created as NaN, then imputed by training-set median.
    features = features_path.read_text().splitlines()
    features = [f for f in features if f.strip()]
    missing_in_input = [f for f in features if f not in df.columns]
    if missing_in_input:
        print(f"[predict_vus] {len(missing_in_input)} expected features absent "
              f"from input; will impute with training medians.")
        for f in missing_in_input:
            df[f] = np.nan

    feat_df = df[features].copy()
    medians = _load_training_medians(features)
    feat_df = feat_df.fillna(medians)
    # Anything still NaN (feature not in training parquet either) → 0.
    feat_df = feat_df.fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)

    model = joblib.load(model_path)
    print(f"[predict_vus] Loaded model {model_path}")
    print(f"[predict_vus] Inference matrix: {X.shape}")

    y_pred = model.predict(X)
    out = pd.DataFrame({
        "pred_class": y_pred.astype(int),
        "pred_label": [class_names[i] for i in y_pred.astype(int)],
    })
    # Carry identifier columns over if they exist (so the user can join back)
    for keep in ("base__chrom", "base__pos", "base__ref_base", "base__alt_base",
                 "base__hugo", "gene_symbol", "clinvar__sig", "clinvar_sig"):
        if keep in raw.columns:
            out[keep] = raw[keep].values

    try:
        proba = model.predict_proba(X)
        for i, name in enumerate(class_names):
            out[f"proba_{name.replace(' ', '_').replace('/', '_')}"] = proba[:, i]
    except (AttributeError, NotImplementedError):
        print("[predict_vus] Model has no predict_proba; skipping probability columns.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[predict_vus] Wrote {len(out)} predictions to {output_path}")

    # Quick summary so the user sees something useful in the terminal
    print("\n[predict_vus] Prediction class distribution:")
    print(out["pred_label"].value_counts().to_string())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="VUS variant file (CSV/TSV/parquet, OpenCRAVAT 777-col schema).")
    p.add_argument("--task", choices=["2class", "4class"], default="2class")
    p.add_argument("--model", type=Path, default=None,
                   help="Path to a .joblib model. Default: tuned HistGB for the chosen task.")
    p.add_argument("--features", type=Path, default=None,
                   help="features.txt manifest. Default: alongside the chosen model.")
    p.add_argument("--output", type=Path, default=Path("outputs/predictions/vus_predictions.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or (
        MODELS_DIR / f"{args.task}_final_no_adaboost" / "histgradientboosting.joblib"
    )
    features_path = args.features or (model_path.parent / "features.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {features_path}")
    predict(args.input, model_path, features_path, args.task, args.output)


if __name__ == "__main__":
    main()
