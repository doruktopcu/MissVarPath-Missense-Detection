"""Print headline numbers from every leaderboard.csv under outputs/reports/."""
import pandas as pd
from pathlib import Path

REP = Path('outputs/reports')

RUNS = [
    ('canonical 4class',    '4class'),
    ('canonical 2class',    '2class'),
    ('tuned 4class',        '4class_final_no_adaboost'),
    ('tuned 2class',        '2class_final_no_adaboost'),
    ('gene-CV 4class',      '4class_gene'),
    ('gene-CV 2class',      '2class_gene'),
    ('augmented 4class',    '4class_augmented'),
    ('augmented 2class',    '2class_augmented'),
    ('raw-only 4class',     '4class_augmented_raw_only'),
    ('raw-only 2class',     '2class_augmented_raw_only'),
    ('no-VEP 4class',       '4class_augmented_no_vep'),
    ('no-VEP 2class',       '2class_augmented_no_vep'),
    ('no-VEP gene 4class',  '4class_augmented_gene_no_vep'),
    ('no-DITTO 4class',     '4class_no_ditto'),
]

for label, d in RUNS:
    p = REP / d / 'leaderboard.csv'
    if not p.exists():
        print(f'{label:22s}  MISSING ({p})')
        continue
    df = pd.read_csv(p).sort_values('holdout_macro_f1', ascending=False)
    top = df.iloc[0]
    n = len(df)
    name = str(top['model'])
    f1 = top['holdout_macro_f1']
    acc = top['holdout_acc']
    mcc = top['holdout_mcc']
    print(f"{label:22s}  n={n:2d}  top={name:22s}  F1={f1:.4f}  acc={acc:.4f}  MCC={mcc:.4f}")
