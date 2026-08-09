"""Why does target encoding of a 9,000-category column produce 3 values?

Verifies the smoothing behaviour of the configured TargetEncoder and checks
whether the identical min/max seen for `industry` and `work_experience` is a
coincidence or a column-alignment bug in categorical_encoding.transform.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import processing, utils  # noqa: E402

warnings.filterwarnings("ignore")

ID_COLS = ['User_id', 'emi_1_dpd', 'emi_2_dpd', 'emi_3_dpd', 'emi_4_dpd',
           'emi_5_dpd', 'emi_6_dpd', 'max_dpd', 'yearmo', 'label']

df = utils.process_data("input/credit_risk_data.csv", ['gender'])
train, val, hold_out = utils.data_split(df)
processing.create_label(train, dpd=60, months=3)
processing.derived_features(train)

cat_cols = list(train.drop(columns=ID_COLS).select_dtypes(include=['category', 'object']).columns)
params = {"verbose": 0, "cols": None, "drop_invariant": False, "return_df": True,
          "handle_missing": 'value', "handle_unknown": 'value',
          "min_samples_leaf": 5000, "smoothing": 1}
enc = processing.categorical_encoding(params)
enc.fit(train, cat_cols, 'label')

print("encoder fitted column order :", enc.feature_names)
print("prior                       :", train.label.mean())

print("\n--- category frequency: how many categories clear min_samples_leaf=5000? ---")
for c in ['industry', 'work_experience', 'pincode', 'role']:
    vc = train[c].value_counts(dropna=False)
    print(f"{c:<18} categories={len(vc):>6,}  "
          f">=5000 rows: {(vc >= 5000).sum():>3}  "
          f"top counts: {list(vc.head(4).values)}")

print("\n--- encoder internal mapping (distinct encoded values per column) ---")
for c in ['industry', 'work_experience', 'pincode', 'role']:
    m = enc.te.mapping[c]
    print(f"{c:<18} mapping rows={len(m):>6,}  distinct values={m.round(6).nunique():>3}  "
          f"min={m.min():.6f} max={m.max():.6f}")

print("\n--- are industry and work_experience genuinely sharing extreme values? ---")
mi, mw = enc.te.mapping['industry'], enc.te.mapping['work_experience']
print(f"industry        extremes: {mi.min():.8f} / {mi.max():.8f}")
print(f"work_experience extremes: {mw.min():.8f} / {mw.max():.8f}")
print(f"identical min? {np.isclose(mi.min(), mw.min())}   identical max? {np.isclose(mi.max(), mw.max())}")

# Which raw categories produce those extremes, and how many rows do they cover?
for name, col, m in [("industry", 'industry', mi), ("work_experience", 'work_experience', mw)]:
    inv = enc.te.ordinal_encoder.mapping
    lut = [d for d in inv if d['col'] == col][0]['mapping']
    rev = {v: k for k, v in lut.items()}
    print(f"\n{name}: categories at the extremes")
    for idx in [m.idxmin(), m.idxmax()]:
        raw = rev.get(idx, "<unmapped>")
        n = (train[col] == raw).sum()
        label_mean = train.loc[train[col] == raw, 'label'].mean() if n else np.nan
        shown = str(raw)
        shown = shown[:34] + "..." if len(shown) > 37 else shown
        print(f"   encoded={m[idx]:.6f}  rows={n:>7,}  observed_default_rate={label_mean:.6f}  value={shown}")

print("\n--- column alignment sanity check on transform() ---")
t = enc.transform(train.copy())
for c in ['industry', 'work_experience']:
    # recompute expected encoding by hand from the mapping and compare
    lut = [d for d in enc.te.ordinal_encoder.mapping if d['col'] == c][0]['mapping']
    expected = train[c].map(lut).map(enc.te.mapping[c])
    match = np.allclose(expected.fillna(-1), t[c].fillna(-1))
    print(f"{c:<18} transform matches manual mapping: {match}")
