"""How much categorical signal is recoverable under governance constraints?

Two questions, answered on the governance-compliant feature set only:

1. The project encodes categoricals with `min_samples_leaf=5000`, which pins any
   category smaller than that to the global prior - so `pincode` became a
   constant and `industry` collapsed to 3 distinct values. Does a sane smoothing
   level recover real signal, or was the aggressive setting protecting against
   overfitting?

2. `industry` and `work_experience` are 84% placeholder zeros written two ways,
   and the spelling predicts default. What does collapsing that artifact cost?

Every configuration is trained with identical LightGBM parameters, so only the
encoding differs. Selection is on validation (202204); the hold-out (202205) is
not touched here.

    python analysis/encoder_sweep.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline import processing, utils, governance  # noqa: E402
from ml_pipeline.config import Config  # noqa: E402
from ml_pipeline.logging_utils import setup_logging  # noqa: E402

warnings.filterwarnings("ignore")
setup_logging("WARNING")

cfg = Config()

# Fixed model parameters - only the encoder varies across configurations.
LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt',
    'num_leaves': 16, 'max_depth': 6, 'learning_rate': 0.02,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_data_in_leaf': 200, 'lambda_l1': 1.0, 'lambda_l2': 1.0,
    'random_seed': 2019, 'verbose': -1,
}


def prepare(clean_placeholders):
    """Load, split, label and derive - optionally collapsing the placeholders."""
    df = utils.process_data(cfg.data_path, [])          # keep gender out later via policy
    train, val, hold_out = utils.data_split(df)
    for part in (train, val, hold_out):
        processing.create_label(part, cfg.label_dpd, cfg.label_months)
        if clean_placeholders:
            processing.clean_placeholders(part)
        processing.derived_features(part)
    return train, val, hold_out


def run(train, val, min_samples_leaf, smoothing, allow_restricted=True, tag=""):
    """Encode with the given smoothing, fit, and score."""
    params = dict(cfg.encoder_params)
    params["min_samples_leaf"] = min_samples_leaf
    params["smoothing"] = smoothing

    feats = governance.permitted_features(train.columns, allow_restricted=allow_restricted)
    cat_cols = [c for c in feats if train[c].dtype == object]

    enc = processing.categorical_encoding(params)
    enc.fit(train, cat_cols, "label")
    tr, va = enc.transform(train.copy()), enc.transform(val.copy())

    governance.enforce_policy(feats)

    dtr = lgb.Dataset(tr[feats], label=tr.label)
    dva = lgb.Dataset(va[feats], label=va.label)
    m = lgb.train(LGB_PARAMS, dtr, 5000, valid_sets=[dva], valid_names=["val"],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    tr_auc = roc_auc_score(tr.label, m.predict(tr[feats]))
    va_auc = roc_auc_score(va.label, m.predict(va[feats]))

    # how much variety survived the encoder
    distinct = {c: int(tr[c].nunique()) for c in cat_cols}
    return {
        "config": tag,
        "min_samples_leaf": min_samples_leaf,
        "smoothing": smoothing,
        "n_features": len(feats),
        "trees": m.best_iteration,
        "train_auc": tr_auc,
        "val_auc": va_auc,
        "val_gini": 2 * va_auc - 1,
        "gap": tr_auc - va_auc,
        "distinct_encoded": distinct,
    }


def main():
    print("=" * 100)
    print("Governance-permitted feature set")
    print("=" * 100)
    df = utils.process_data(cfg.data_path, [])
    train, _, _ = utils.data_split(df)
    processing.create_label(train, cfg.label_dpd, cfg.label_months)
    processing.derived_features(train)

    allowed = governance.permitted_features(train.columns)
    print(f"permitted ({len(allowed)}): {sorted(allowed)}")
    strict = governance.permitted_features(train.columns, allow_restricted=False)
    print(f"excluding RESTRICTED ({len(strict)}): {sorted(strict)}")
    excluded = sorted(set(train.columns) - set(allowed))
    print("\nexcluded by policy:")
    for c in excluded:
        cls = governance.classify(c)
        if cls and cls.value != "non_feature":
            print(f"  {c:<26} {cls.value.upper()}")

    rows = []
    grid = [(5000, 1), (1000, 1), (500, 10), (200, 10), (100, 10),
            (50, 10), (20, 5), (10, 2)]

    for clean in (False, True):
        tr, va, _ = prepare(clean_placeholders=clean)
        tag = "placeholders collapsed" if clean else "placeholders as-is"
        print(f"\n{'=' * 100}\n{tag}\n{'=' * 100}")
        print(f"{'min_samples_leaf':>17}{'smoothing':>11}{'trees':>8}"
              f"{'train_auc':>11}{'val_auc':>10}{'val_gini':>10}{'gap':>9}")
        for msl, sm in grid:
            r = run(tr, va, msl, sm, tag=tag)
            rows.append(r)
            print(f"{msl:>17}{sm:>11}{r['trees']:>8}{r['train_auc']:>11.4f}"
                  f"{r['val_auc']:>10.4f}{r['val_gini']:>10.4f}{r['gap']:>9.4f}")

    res = pd.DataFrame(rows)

    print(f"\n{'=' * 100}\nBest per configuration\n{'=' * 100}")
    for tag, g in res.groupby("config", sort=False):
        b = g.loc[g.val_auc.idxmax()]
        print(f"{tag:<26} best val_auc={b.val_auc:.4f} (gini {b.val_gini:.4f}) "
              f"at min_samples_leaf={int(b.min_samples_leaf)}, smoothing={b.smoothing}")

    print(f"\n{'=' * 100}\nCategory variety surviving the encoder (placeholders collapsed)\n{'=' * 100}")
    sub = res[res.config == "placeholders collapsed"]
    cols = sorted(sub.iloc[0]["distinct_encoded"].keys())
    print(f"{'min_samples_leaf':>17}" + "".join(f"{c[:13]:>15}" for c in cols))
    for _, r in sub.iterrows():
        print(f"{int(r.min_samples_leaf):>17}" +
              "".join(f"{r['distinct_encoded'][c]:>15}" for c in cols))

    # RESTRICTED fields: what do they actually contribute?
    print(f"\n{'=' * 100}\nCost of dropping RESTRICTED fields (dependents, has_social_profile, is_verified)\n{'=' * 100}")
    tr, va, _ = prepare(clean_placeholders=True)
    best = res[res.config == "placeholders collapsed"].loc[
        res[res.config == "placeholders collapsed"].val_auc.idxmax()]
    msl, sm = int(best.min_samples_leaf), best.smoothing
    with_r = run(tr, va, msl, sm, allow_restricted=True, tag="with restricted")
    without_r = run(tr, va, msl, sm, allow_restricted=False, tag="without restricted")
    print(f"  with RESTRICTED    ({with_r['n_features']:>2} features): val_auc={with_r['val_auc']:.4f}  gini={with_r['val_gini']:.4f}")
    print(f"  without RESTRICTED ({without_r['n_features']:>2} features): val_auc={without_r['val_auc']:.4f}  gini={without_r['val_gini']:.4f}")
    print(f"  cost of exclusion: {with_r['val_gini'] - without_r['val_gini']:+.4f} Gini")

    os.makedirs("output", exist_ok=True)
    res.drop(columns=["distinct_encoded"]).to_csv("output/encoder_sweep.csv", index=False)
    print("\nWritten: output/encoder_sweep.csv")


if __name__ == "__main__":
    main()
