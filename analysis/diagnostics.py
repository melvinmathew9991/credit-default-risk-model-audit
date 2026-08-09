"""Independent diagnostics on the credit default dataset and modelling design.

This is deliberately separate from the training pipeline. Its job is to try to
break the result: to ask whether a 0.96 AUC default model is believable, and if
not, to locate exactly where the number comes from.

    python analysis/diagnostics.py        (run from the repository root)

Checks
------
1. Split hygiene      - do the same users appear in train and hold-out?
2. Feature timing     - are any features unavailable at application time?
3. Univariate power   - which single columns already carry the signal?
4. Encoder behaviour  - is target encoding doing anything, or overflowing?
5. Derived features   - are the engineered ratios actually defined?
6. Counterfactual     - what does the model score without outcome features?
"""

import os
import sys
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import processing, utils  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

DATA = "data/credit_risk_data.csv"
ID_COLS = [
    "User_id",
    "emi_1_dpd",
    "emi_2_dpd",
    "emi_3_dpd",
    "emi_4_dpd",
    "emi_5_dpd",
    "emi_6_dpd",
    "max_dpd",
    "yearmo",
    "label",
]

# Fields that describe repayment behaviour on the loan itself. The data
# dictionary calls them "last 2 years" history, but see check 2.
OUTCOME_FEATURES = [
    "total_payement",
    "received_principal",
    "interest_received",
    "interest_received_ratio",
    "total_payement_per_loan",
]

results = {}


def header(n, title):
    print("\n" + "=" * 78)
    print(f"CHECK {n}: {title}")
    print("=" * 78)


# ---------------------------------------------------------------------------
df = utils.process_data(DATA, ["gender"])
train, val, hold_out = utils.data_split(df)
for d in (train, val, hold_out):
    processing.create_label(d, dpd=60, months=3)
    processing.derived_features(d)


# ---------------------------------------------------------------------- 1
header(1, "Split hygiene - user overlap across time-based splits")

tr_u, va_u, ho_u = set(train.User_id), set(val.User_id), set(hold_out.User_id)
print(f"train users     : {len(tr_u):>7,}  (rows {len(train):,})")
print(f"val users       : {len(va_u):>7,}  (rows {len(val):,})")
print(f"hold_out users  : {len(ho_u):>7,}  (rows {len(hold_out):,})")
print(
    f"\ntrain & val      overlap : {len(tr_u & va_u):>6,} users "
    f"({100*len(tr_u & va_u)/len(va_u):.2f}% of val users)"
)
print(
    f"train & hold_out overlap : {len(tr_u & ho_u):>6,} users "
    f"({100*len(tr_u & ho_u)/len(ho_u):.2f}% of hold_out users)"
)
print(f"duplicated User_id rows in full data : {df.User_id.duplicated().sum():,}")
results["user_overlap_val"] = len(tr_u & va_u)
results["user_overlap_hold_out"] = len(tr_u & ho_u)


# ---------------------------------------------------------------------- 2
header(2, "Feature timing - is repayment history available at application?")

print("The data dictionary describes total_payement / received_principal /")
print("interest_received / number_of_loans as activity over the LAST 2 YEARS,")
print("i.e. prior loans. If that is true, a borrower with zero prior loans must")
print("have zero prior repayment.\n")

zero_loans = df[df.number_of_loans == 0]
print(
    f"rows with number_of_loans == 0 : {len(zero_loans):,} "
    f"({100*len(zero_loans)/len(df):.2f}% of data)"
)
print(
    f"  of those, rows with total_payement > 0     : " f"{(zero_loans.total_payement > 0).sum():,}"
)
print(
    f"  of those, rows with received_principal > 0 : "
    f"{(zero_loans.received_principal > 0).sum():,}"
)
print(f"  their mean total_payement                 : " f"{zero_loans.total_payement.mean():,.2f}")
print(f"\nreceived_principal max = {df.received_principal.max():,.2f}")
print("=> repayment amounts exist for borrowers with no prior loans, so these")
print("   columns describe the CURRENT loan, not prior history.")
results["zero_loan_rows_with_payment"] = int((zero_loans.total_payement > 0).sum())


# ---------------------------------------------------------------------- 3
header(3, "Univariate discrimination - where does the signal live?")

num_cols = [
    "total_income",
    "dependents",
    "delinq_2yrs",
    "total_payement",
    "received_principal",
    "interest_received",
    "number_of_loans",
    "interest_received_ratio",
    "total_payement_per_loan",
    "delinq_2yrs_ratio",
]
rows = []
for c in num_cols:
    x = train[c].fillna(train[c].median())
    a = roc_auc_score(train.label, x)
    rows.append({"feature": c, "auc": a, "gini_abs": abs(2 * a - 1)})
uni = pd.DataFrame(rows).sort_values("gini_abs", ascending=False)
print(uni.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n(AUC < 0.5 simply means the feature is inversely related to default.)")
results["top_univariate"] = uni.iloc[0].to_dict()


# ---------------------------------------------------------------------- 4
header(4, "Target encoder behaviour")

cat_cols = list(train.drop(columns=ID_COLS).select_dtypes(include=["category", "object"]).columns)
print("categorical columns and cardinality (train):")
for c in cat_cols:
    print(
        f"  {c:<22} {train[c].nunique():>7,} distinct   "
        f"{100*train[c].isna().mean():>5.1f}% null"
    )

params = {
    "verbose": 0,
    "cols": None,
    "drop_invariant": False,
    "return_df": True,
    "handle_missing": "value",
    "handle_unknown": "value",
    "min_samples_leaf": 5000,
    "smoothing": 1,
}
enc = processing.categorical_encoding(params)
enc.fit(train, cat_cols, "label")
enc_train = enc.transform(train.copy())

print(f"\nprior (train default rate) = {train.label.mean():.6f}")
print("\nencoded value spread per column (if std ~ 0 the encoder collapsed to prior):")
for c in cat_cols:
    v = enc_train[c]
    print(
        f"  {c:<22} min={v.min():.6f}  max={v.max():.6f}  std={v.std():.6f}  "
        f"nunique={v.nunique()}"
    )
print("\nNote: with min_samples_leaf=5000 and smoothing=1 the blending weight is")
print("1/(1+exp(-(count-5000)/1)), which overflows to 0 for any category with")
print("fewer than ~5000 rows - so most categories are pinned to the prior.")


# ---------------------------------------------------------------------- 5
header(5, "Derived feature validity")

for c in ["interest_received_ratio", "total_payement_per_loan", "delinq_2yrs_ratio"]:
    v = train[c]
    print(
        f"{c:<26} zeros={100*(v == 0).mean():>6.2f}%  std={v.std():.4f}  "
        f"nunique={v.nunique():,}"
    )
print("\ntotal_payement_per_loan and delinq_2yrs_ratio divide by number_of_loans,")
print(f"which is 0 for {100*(train.number_of_loans == 0).mean():.2f}% of training rows;")
print("those become 0 via the inf/NaN fill, so the features are near-constant.")


# ---------------------------------------------------------------------- 6
header(6, "Counterfactual - model without current-loan outcome features")


def fit_and_score(feature_cols, tag):
    p = {
        "objective": "binary",
        "metric": "auc",
        "boosting": "gbdt",
        "num_leaves": 12,
        "max_depth": 9,
        "learning_rate": 0.01,
        "feature_fraction": 0.669,
        "max_bin": 100,
        "min_data_in_leaf": 625,
        "min_data_in_bin": 30,
        "bagging_freq": 20,
        "random_seed": 2019,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
    }
    dtr = lgb.Dataset(tr[feature_cols], label=tr.label)
    dva = lgb.Dataset(va[feature_cols], label=va.label)
    m = lgb.train(
        p,
        dtr,
        20000,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    out = {"model": tag, "n_features": len(feature_cols), "n_trees": m.best_iteration}
    for nm, d in (("train", tr), ("val", va), ("hold_out", ho)):
        out[f"{nm}_auc"] = roc_auc_score(d.label, m.predict(d[feature_cols]))
        out[f"{nm}_gini"] = 2 * out[f"{nm}_auc"] - 1
    return out


tr, va, ho = (enc.transform(d.copy()) for d in (train, val, hold_out))
all_feats = [c for c in tr.columns if c not in ID_COLS and c != "pincode"]
clean_feats = [c for c in all_feats if c not in OUTCOME_FEATURES]

print(f"full feature set  ({len(all_feats)}): {all_feats}")
print(f"\nremoved as outcome-contaminated: " f"{[c for c in all_feats if c in OUTCOME_FEATURES]}")
print(f"application-time feature set ({len(clean_feats)}): {clean_feats}\n")

comp = pd.DataFrame(
    [
        fit_and_score(all_feats, "A: as-built (all features)"),
        fit_and_score(clean_feats, "B: application-time features only"),
    ]
)
print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

os.makedirs("output", exist_ok=True)
comp.to_csv("output/leakage_counterfactual.csv", index=False)
uni.to_csv("output/univariate_power.csv", index=False)

print("\n" + "=" * 78)
print("Written: output/leakage_counterfactual.csv, output/univariate_power.csv")
print("=" * 78)
