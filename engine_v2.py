"""Honest model: application-time features only, under the governance policy.

This is the remediated successor to `engine.py`. What changed and why:

  * features are filtered through `ml_pipeline.governance` - post-origination
    repayment fields (finding C1) and protected attributes (H4) cannot enter,
    and the policy is enforced with an exception, not a comment
  * the `industry` / `work_experience` placeholder artifact (H1) is collapsed
  * a single validation month is replaced by walk-forward folds (M2), and the
    hold-out month is never scored until the final evaluation (M3)
  * repeat customers are removed from later periods (M7)
  * the output is calibrated (M5)
  * a WOE + logistic scorecard is fitted as a challenger

    python engine_v2.py                      # full run
    python engine_v2.py --max-evals 10       # quicker
    python engine_v2.py --no-restricted      # drop RESTRICTED attributes too

Artefacts land in `output_v2/` so the original run is left untouched.
"""

import argparse
import datetime
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, tpe
from sklearn.metrics import roc_auc_score

from ml_pipeline import evaluation as ev
from ml_pipeline import governance, processing, training, utils, woe
from ml_pipeline.calibration import Calibrator, calibration_report
from ml_pipeline.config import Config
from ml_pipeline.data_contract import validate_raw_data
from ml_pipeline.logging_utils import get_logger, setup_logging
from ml_pipeline.monitoring import build_baseline_from_raw, save_baseline
from ml_pipeline.validation import drop_leaked_users, expanding_window_folds, summarise_folds

logger = get_logger("engine_v2")

# From analysis/encoder_sweep.py: loosening the encoder below ~500 buys no
# validation AUC and triples the train/validation gap. 1000 was the best of the
# grid, marginally ahead of the original 5000.
ENCODER_PARAMS = {
    "verbose": 0,
    "cols": None,
    "drop_invariant": False,
    "return_df": True,
    "handle_missing": "value",
    "handle_unknown": "value",
    "min_samples_leaf": 1000,
    "smoothing": 1,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-path", dest="data_path")
    p.add_argument("--output-dir", dest="output_dir", default="output_v2")
    p.add_argument("--max-evals", dest="max_evals", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--no-restricted",
        action="store_true",
        help="exclude RESTRICTED attributes as well as PROHIBITED",
    )
    p.add_argument(
        "--keep-placeholders",
        action="store_true",
        help="do not collapse the industry/work_experience placeholder",
    )
    p.add_argument("--calibration", choices=["isotonic", "platt"], default="isotonic")
    p.add_argument(
        "--strict-validation",
        action="store_true",
        help="fail the run if the input breaches the data contract",
    )
    p.add_argument("--log-level", dest="log_level", default="INFO")
    return p.parse_args(argv)


def build_datasets(cfg, args):
    """Load, validate, label, clean, and split into tuning months and hold-out."""
    df = utils.process_data(cfg.data_path, [])

    # The contract runs on the raw frame, before clean_placeholders. It will
    # report the placeholder-spelling defect that this pipeline goes on to
    # remediate - that is intended: the run record should show the state of the
    # data as received, not as repaired. --strict-validation turns it into a
    # gate, for once the source export is fixed.
    args._validation = validate_raw_data(df, strict=args.strict_validation)

    processing.create_label(df, cfg.label_dpd, cfg.label_months)
    if not args.keep_placeholders:
        processing.clean_placeholders(df)
    processing.derived_features(df)

    tuning = df[df.yearmo <= cfg.val_yearmo].reset_index(drop=True)
    hold_out = df[df.yearmo == cfg.hold_out_yearmo].reset_index(drop=True)

    # customers seen during tuning must not be scored again in the hold-out
    hold_out = drop_leaked_users(tuning, hold_out, name="hold_out")
    return df, tuning, hold_out


def encode_fold(train, valid, features, params=ENCODER_PARAMS):
    """Fit the target encoder on this fold's training window only."""
    cat_cols = [c for c in features if train[c].dtype == object]
    enc = processing.categorical_encoding(params)
    enc.fit(train, cat_cols, "label")
    return enc, enc.transform(train.copy()), enc.transform(valid.copy())


def cv_score(cfg, tuning, features, lgb_params, num_boost_round, early_stopping):
    """Walk-forward CV. Returns per-fold val AUC, train AUC and best iterations."""
    val_aucs, train_aucs, iters = [], [], []

    for _, _train_months, val_month, tr_idx, va_idx in expanding_window_folds(
        tuning, min_train_months=1
    ):
        tr = tuning.loc[tr_idx].reset_index(drop=True)
        va = tuning.loc[va_idx].reset_index(drop=True)
        va = drop_leaked_users(tr, va, name=f"fold val {val_month}")
        if va.empty or va.label.nunique() < 2:
            continue

        _, tr_e, va_e = encode_fold(tr, va, features)
        m = lgb.train(
            lgb_params,
            lgb.Dataset(tr_e[features], label=tr_e.label),
            num_boost_round=num_boost_round,
            valid_sets=[lgb.Dataset(va_e[features], label=va_e.label)],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
        )
        train_aucs.append(roc_auc_score(tr_e.label, m.predict(tr_e[features])))
        val_aucs.append(roc_auc_score(va_e.label, m.predict(va_e[features])))
        iters.append(m.best_iteration)
        del m

    if not val_aucs:
        raise RuntimeError("no usable folds")
    return val_aucs, train_aucs, iters


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(
        data_path=args.data_path or Config().data_path,
        output_dir=args.output_dir,
        max_evals=args.max_evals,
        seed=args.seed,
        log_level=args.log_level,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logging(cfg.log_level, cfg.resolved_log_file)

    lineage = governance.data_fingerprint(cfg.data_path)
    logger.info(
        "input fingerprint sha256=%s (%d bytes)", lineage["sha256"][:16] + "...", lineage["bytes"]
    )

    df, tuning, hold_out = build_datasets(cfg, args)

    # ---- feature policy -------------------------------------------------
    features = governance.permitted_features(df.columns, allow_restricted=not args.no_restricted)
    governance.enforce_policy(features)
    logger.info("governance-permitted features (%d): %s", len(features), sorted(features))
    excluded = {
        c: governance.classify(c).value
        for c in df.columns
        if governance.classify(c)
        and c not in features
        and governance.classify(c).value != "non_feature"
    }
    logger.info("excluded by policy: %s", excluded)

    # ---- tuning on walk-forward folds -----------------------------------
    trial_rows, state = [], {"i": 0}

    def objective(space):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting": "gbdt",
            "num_leaves": int(space["num_leaves"]),
            "max_depth": int(space["max_depth"]),
            "learning_rate": space["learning_rate"],
            "feature_fraction": space["feature_fraction"],
            "max_bin": int(space["max_bin"]),
            "min_data_in_leaf": int(space["min_data_in_leaf"]),
            "min_data_in_bin": int(space["min_data_in_bin"]),
            "lambda_l1": space["lambda_l1"],
            "lambda_l2": space["lambda_l2"],
            "bagging_freq": 20,
            "pos_bagging_fraction": space["pos_bagging_fraction"],
            "neg_bagging_fraction": space["neg_bagging_fraction"],
            "random_seed": 2019,
            "verbose": -1,
        }
        val_aucs, train_aucs, iters = cv_score(
            cfg, tuning, features, params, cfg.num_boost_round, cfg.early_stopping_rounds
        )

        v, t = summarise_folds(val_aucs), summarise_folds(train_aucs)
        gap = t["mean"] - v["mean"]
        # maximise mean fold AUC, penalise both overfitting and fold-to-fold
        # instability - a model that only works in one month is not deployable
        score = -(v["mean"] - 0.5 * max(gap, 0) - 0.5 * v["std"])

        state["i"] += 1
        row = {
            **{
                f"p_{k}": val
                for k, val in params.items()
                if k not in ("objective", "metric", "boosting", "verbose")
            },
            "cv_val_auc_mean": v["mean"],
            "cv_val_auc_std": v["std"],
            "cv_val_auc_min": v["min"],
            "cv_train_auc_mean": t["mean"],
            "gap": gap,
            "mean_best_iter": float(np.mean(iters)),
            "score": score,
        }
        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(cfg.path("hyperopt_results_v2.csv"), index=False)
        logger.info(
            "trial %2d/%d  cv_val_auc=%.4f (+/-%.4f, min %.4f)  gap=%.4f  score=%.5f",
            state["i"],
            cfg.max_evals,
            v["mean"],
            v["std"],
            v["min"],
            gap,
            score,
        )
        return score

    logger.info("tuning: %d trials over walk-forward folds", cfg.max_evals)
    fmin(
        fn=objective,
        space=training.space,
        algo=tpe.suggest,
        max_evals=cfg.max_evals,
        trials=Trials(),
        rstate=np.random.default_rng(cfg.seed),
        show_progressbar=False,
    )

    results = pd.DataFrame(trial_rows)
    best = results.loc[results.score.idxmin()]
    best_params = {k[2:]: v for k, v in best.items() if k.startswith("p_")}
    for k in (
        "num_leaves",
        "max_depth",
        "max_bin",
        "min_data_in_leaf",
        "min_data_in_bin",
        "bagging_freq",
        "random_seed",
    ):
        best_params[k] = int(best_params[k])
    best_params.update({"objective": "binary", "metric": "auc", "boosting": "gbdt", "verbose": -1})
    n_rounds = max(50, int(round(best["mean_best_iter"])))
    logger.info(
        "best trial: cv_val_auc=%.4f (+/-%.4f), %d rounds",
        best["cv_val_auc_mean"],
        best["cv_val_auc_std"],
        n_rounds,
    )

    # ---- final fit -------------------------------------------------------
    # Refit on every tuning month at the CV-chosen number of rounds. The last
    # month is held back only to fit the calibrator, never to select the model.
    calib_month = cfg.val_yearmo
    fit_part = tuning[tuning.yearmo < calib_month].reset_index(drop=True)
    calib_part = tuning[tuning.yearmo == calib_month].reset_index(drop=True)
    calib_part = drop_leaked_users(fit_part, calib_part, name="calibration")

    encoder, fit_e, calib_e = encode_fold(fit_part, calib_part, features)
    model = lgb.train(
        best_params, lgb.Dataset(fit_e[features], label=fit_e.label), num_boost_round=n_rounds
    )

    calibrator = Calibrator(args.calibration).fit(model.predict(calib_e[features]), calib_e.label)

    # ---- WOE + logistic challenger --------------------------------------
    woe_enc = woe.WOEEncoder(n_bins=10, min_bin_frac=0.02)
    woe_train = woe_enc.fit_transform(fit_part[features], fit_part.label)
    iv = woe_enc.iv_table()
    kept = woe_enc.select(min_iv=0.02) or features
    challenger = woe.fit_logistic_scorecard(woe_train[kept], fit_part.label)

    # ---- hold-out: scored once, at the end ------------------------------
    hold_e = encoder.transform(hold_out.copy())
    raw_ho = model.predict(hold_e[features])
    cal_ho = calibrator.transform(raw_ho)
    chal_ho = challenger.predict_proba(woe_enc.transform(hold_out[features])[kept])[:, 1]

    champion = ev.discrimination_metrics(hold_out.label, cal_ho)
    chal_metrics = ev.discrimination_metrics(hold_out.label, chal_ho)
    calib = calibration_report(hold_out.label, raw_ho, cal_ho)

    # ---- persist ---------------------------------------------------------
    model.save_model(cfg.path("model_v2.txt"))
    for name, obj in [
        ("target_encoder_v2.pkl", encoder),
        ("calibrator_v2.pkl", calibrator),
        ("woe_encoder_v2.pkl", woe_enc),
        ("challenger_v2.pkl", challenger),
    ]:
        with open(cfg.path(name), "wb") as f:
            pickle.dump(obj, f)
    with open(cfg.path("feature_columns_v2.json"), "w") as f:
        json.dump(features, f, indent=2)
    iv.to_csv(cfg.path("information_value.csv"), index=False)
    woe.coefficient_table(challenger, kept, woe_matrix=woe_train[kept]).to_csv(
        cfg.path("challenger_coefficients.csv"), index=False
    )
    ev.decile_table(hold_out.label, cal_ho).to_csv(
        cfg.path("decile_table_hold_out.csv"), index=False
    )
    ev.approval_curve(hold_out.label, cal_ho).to_csv(
        cfg.path("approval_curve_hold_out.csv"), index=False
    )

    # ---- monitoring baseline --------------------------------------------
    # Produced with the model so the two cannot drift apart. Built from the raw
    # tuning frame, before clean_placeholders, because placeholder rates must be
    # measured on data as received - see build_baseline_from_raw.
    raw_tuning = utils.process_data(cfg.data_path, [])
    processing.create_label(raw_tuning, cfg.label_dpd, cfg.label_months)
    raw_tuning = raw_tuning[raw_tuning.yearmo <= cfg.val_yearmo].reset_index(drop=True)
    baseline = build_baseline_from_raw(
        raw_tuning,
        model,
        encoder,
        calibrator,
        features,
        labels=raw_tuning["label"],
        model_version=os.path.basename(os.path.abspath(cfg.output_dir)),
    )
    baseline["window"] = {
        "up_to": int(cfg.val_yearmo),
        "months": sorted(int(m) for m in raw_tuning.yearmo.unique()),
    }
    save_baseline(baseline, cfg.path("monitoring_baseline.json"))

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "model": "v2 - application-time features under governance policy",
        "python": sys.version.split()[0],
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "data_lineage": lineage,
        "data_contract": {
            "strict": args.strict_validation,
            "errors": [str(i) for i in args._validation.errors],
            "warnings": [str(i) for i in args._validation.warnings],
        },
        "governance": {
            "policy": governance.policy_summary(),
            "features_used": features,
            "excluded": excluded,
            "restricted_included": not args.no_restricted,
            "placeholders_collapsed": not args.keep_placeholders,
        },
        "label": {"dpd": cfg.label_dpd, "months": cfg.label_months},
        "validation": {
            "scheme": "expanding-window walk-forward over yearmo",
            "tuning_months": sorted(int(m) for m in tuning.yearmo.unique()),
            "hold_out_month": int(cfg.hold_out_yearmo),
            "hold_out_rows_after_dedup": int(len(hold_out)),
            "cv_val_auc_mean": float(best["cv_val_auc_mean"]),
            "cv_val_auc_std": float(best["cv_val_auc_std"]),
        },
        "best_params": best_params,
        "num_boost_round": n_rounds,
        "calibration": {"method": args.calibration, **calib},
        "hold_out_champion": champion,
        "hold_out_challenger": chal_metrics,
        "information_value_top": iv.head(10).to_dict("records"),
    }
    with open(cfg.path("run_manifest_v2.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 86)
    print("HOLD-OUT (202205) - scored once, after all selection was complete")
    print("=" * 86)
    print(f"{'model':<34}{'n':>8}{'bad%':>8}{'AUC':>9}{'Gini':>9}{'KS':>8}{'PR-AUC':>9}")
    print("-" * 86)
    print(
        f"{'champion: LightGBM (calibrated)':<34}{champion['n']:>8}"
        f"{100*champion['default_rate']:>7.2f}%{champion['roc_auc']:>9.4f}"
        f"{champion['gini']:>9.4f}{champion['ks']:>8.4f}{champion['pr_auc_class1']:>9.4f}"
    )
    print(
        f"{'challenger: WOE + logistic':<34}{chal_metrics['n']:>8}"
        f"{100*chal_metrics['default_rate']:>7.2f}%{chal_metrics['roc_auc']:>9.4f}"
        f"{chal_metrics['gini']:>9.4f}{chal_metrics['ks']:>8.4f}"
        f"{chal_metrics['pr_auc_class1']:>9.4f}"
    )
    print("=" * 86)
    print(
        f"CV val AUC across folds : {best['cv_val_auc_mean']:.4f} "
        f"+/- {best['cv_val_auc_std']:.4f} (min {best['cv_val_auc_min']:.4f})"
    )
    print(f"\nCalibration ({args.calibration}):")
    b, a = calib["before"], calib["after"]
    print(
        f"  Brier {b['brier']:.5f} -> {a['brier']:.5f}   "
        f"ECE {b['expected_calibration_error']:.5f} -> {a['expected_calibration_error']:.5f}"
    )
    print(
        f"  mean predicted {b['mean_predicted']:.4f} -> {a['mean_predicted']:.4f} "
        f"vs observed {a['observed_rate']:.4f}"
    )
    print(
        f"  AUC {b['auc']:.4f} -> {a['auc']:.4f} ({calib['auc_delta']:+.4f}; "
        f"isotonic ties cost a little discrimination - material: "
        f"{calib['auc_materially_changed']})"
    )
    print("\nInformation value (train):")
    print(iv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArtefacts written to {cfg.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
