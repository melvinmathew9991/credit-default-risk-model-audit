"""What did each governance decision cost in predictive power?

A credit committee asked to accept a Gini of 0.29 instead of the 0.93 originally
reported is entitled to see where the difference went, decision by decision.

Each step removes exactly one thing and is otherwise identical: same LightGBM
parameters, same encoder, same train (202201-202203) and validation (202204)
periods. Only the last step changes the evaluation population.

Step 1 deliberately includes leakage and protected attributes. That is the
sanctioned comparison baseline documented in MODEL_REVIEW.md - it is never a
candidate model, and `enforce_policy` is called with the explicit escape hatch.

    python analysis/governance_cost.py
"""

import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import governance, processing, utils  # noqa: E402
from ml_pipeline.config import Config  # noqa: E402
from ml_pipeline.logging_utils import setup_logging  # noqa: E402
from ml_pipeline.validation import drop_leaked_users  # noqa: E402

warnings.filterwarnings("ignore")
setup_logging("ERROR")
cfg = Config()

LGB_PARAMS = {'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt',
              'num_leaves': 16, 'max_depth': 6, 'learning_rate': 0.02,
              'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
              'min_data_in_leaf': 200, 'lambda_l1': 1.0, 'lambda_l2': 1.0,
              'random_seed': 2019, 'verbose': -1}
ENC = dict(cfg.encoder_params, min_samples_leaf=1000, smoothing=1)

LEAKAGE = ['total_payement', 'received_principal', 'interest_received',
           'interest_received_ratio', 'total_payement_per_loan']
PROTECTED = ['married', 'pincode']            # gender is already gone
RESTRICTED = ['dependents', 'has_social_profile', 'is_verified']


def load(collapse_placeholders):
    df = utils.process_data(cfg.data_path, ['gender'])
    processing.create_label(df, cfg.label_dpd, cfg.label_months)
    if collapse_placeholders:
        processing.clean_placeholders(df)
    processing.derived_features(df)
    train = df[df.yearmo <= 202203].reset_index(drop=True)
    val = df[df.yearmo == 202204].reset_index(drop=True)
    return train, val


def evaluate(train, val, features, label):
    cat = [c for c in features if train[c].dtype == object]
    enc = processing.categorical_encoding(ENC)
    enc.fit(train, cat, 'label')
    tr, va = enc.transform(train.copy()), enc.transform(val.copy())

    m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[features], label=tr.label), 3000,
                  valid_sets=[lgb.Dataset(va[features], label=va.label)],
                  valid_names=['val'],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
    auc = roc_auc_score(va.label, m.predict(va[features]))
    return {"step": label, "n_features": len(features), "val_auc": auc,
            "val_gini": 2 * auc - 1, "n_val_rows": len(va)}


def main():
    rows = []
    train, val = load(collapse_placeholders=False)
    base = governance.permitted_features(train.columns, allow_restricted=True,
                                         allow_leakage=True)

    # 1. everything the original model could see
    f1 = sorted(set(base) | set(PROTECTED))
    governance.enforce_policy(
        [c for c in f1 if c not in PROTECTED], allow_leakage=True)  # sanctioned baseline
    rows.append(evaluate(train, val, f1, "1. as originally built (leakage + protected)"))

    # 2. remove the post-origination fields
    f2 = [c for c in f1 if c not in LEAKAGE]
    rows.append(evaluate(train, val, f2, "2. - leakage (finding C1)"))

    # 3. remove protected attributes
    f3 = [c for c in f2 if c not in PROTECTED]
    rows.append(evaluate(train, val, f3, "3. - protected: married, pincode (H4)"))

    # 4. remove restricted alternative data
    f4 = [c for c in f3 if c not in RESTRICTED]
    rows.append(evaluate(train, val, f4, "4. - restricted alternative data"))

    # 5. collapse the placeholder artifact
    train_c, val_c = load(collapse_placeholders=True)
    rows.append(evaluate(train_c, val_c, f4, "5. - placeholder artifact (H1)"))

    # 6. remove repeat customers from the evaluation period
    val_d = drop_leaked_users(train_c, val_c, name="val")
    rows.append(evaluate(train_c, val_d, f4, "6. - repeat customers in val (M7)"))

    res = pd.DataFrame(rows)
    res["gini_change"] = res["val_gini"].diff()

    print("=" * 96)
    print("COST OF EACH GOVERNANCE DECISION  (train 202201-202203, validate 202204)")
    print("=" * 96)
    print(f"{'step':<46}{'feats':>6}{'val_auc':>10}{'val_gini':>10}{'change':>10}{'val rows':>10}")
    print("-" * 96)
    for _, r in res.iterrows():
        chg = "" if pd.isna(r.gini_change) else f"{r.gini_change:+.4f}"
        print(f"{r.step:<46}{r.n_features:>6}{r.val_auc:>10.4f}"
              f"{r.val_gini:>10.4f}{chg:>10}{r.n_val_rows:>10,}")
    print("=" * 96)
    total = res.val_gini.iloc[-1] - res.val_gini.iloc[0]
    print(f"Total: {res.val_gini.iloc[0]:.4f} -> {res.val_gini.iloc[-1]:.4f} Gini ({total:+.4f})")
    kept = res.val_gini.iloc[-1] / res.val_gini.iloc[0]
    print(f"Of the original apparent power, {100*kept:.1f}% is real and permissible.")
    print("\nNote: these use fixed, untuned LightGBM parameters so that only the")
    print("feature set varies. Absolute values sit below a properly tuned model;")
    print("the differences between steps are what this table is for.")

    os.makedirs("output", exist_ok=True)
    res.to_csv("output/governance_cost.csv", index=False)
    print("\nWritten: output/governance_cost.csv")


if __name__ == "__main__":
    main()
