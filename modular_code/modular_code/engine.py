"""Training entry point for the credit default risk model.

    python engine.py                          # defaults
    python engine.py --max-evals 5            # quick smoke run
    python engine.py --config my_run.json     # everything from a file
    CRD_MAX_EVALS=5 python engine.py          # environment override

Writes to the output directory:
    model.txt              trained LightGBM booster
    target_encoder.pkl     fitted target encoder (required to score raw data)
    feature_columns.json   model feature names, in order
    processed_splits.pkl   the encoded train/val/hold_out frames
    hyperopt_results.csv   one row per tuning trial
    run_manifest.json      full configuration, versions and results of the run

Then run `python evaluate.py` for the evaluation pack.
"""

# import libraries
import argparse
import datetime
import gc
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from hyperopt import fmin, tpe, Trials
from sklearn.metrics import roc_auc_score

from ml_pipeline import processing, utils, training
from ml_pipeline.config import Config
from ml_pipeline.logging_utils import setup_logging, get_logger

logger = get_logger("engine")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML/JSON file with configuration overrides")
    p.add_argument("--data-path", dest="data_path", help="input csv")
    p.add_argument("--output-dir", dest="output_dir", help="where artefacts are written")
    p.add_argument("--max-evals", dest="max_evals", type=int, help="hyperopt trials")
    p.add_argument("--seed", type=int, help="random seed for the search")
    p.add_argument("--label-dpd", dest="label_dpd", type=int, choices=[30, 60, 90])
    p.add_argument("--label-months", dest="label_months", type=int, choices=range(1, 7))
    p.add_argument("--log-level", dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def build_objective(cfg, train, val, hold_out, state):
    """Create the hyperopt objective closed over the prepared datasets.

    The objective minimises
        (|train_auc - val_auc| + 1) / (1 + val_auc)^2
    trading validation AUC against the train/validation gap.
    """
    feature_cols = [c for c in train.columns if c not in cfg.id_cols]

    def objective(space):
        lgb_train = lgb.Dataset(train[feature_cols], label=train.label)
        lgb_val = lgb.Dataset(val[feature_cols], label=val.label)

        params = {
            'num_leaves': int(space['num_leaves']),
            'max_depth': int(space['max_depth']),
            'learning_rate': space['learning_rate'],
            'objective': 'binary',
            'metric': 'auc',
            "boosting": "gbdt",
            'feature_fraction': space['feature_fraction'],
            'max_bin': int(space['max_bin']),
            'min_data_in_leaf': int(space['min_data_in_leaf']),
            "min_data_in_bin": int(space['min_data_in_bin']),
            "bagging_freq": 20,
            "random_seed": 2019,
            "lambda_l1": space['lambda_l1'],
            "lambda_l2": space['lambda_l2'],
            'pos_bagging_fraction': space['pos_bagging_fraction'],
            'neg_bagging_fraction': space['neg_bagging_fraction'],
            'verbose': -1,
        }

        evals_result = {}
        # LightGBM 4.x: early stopping / logging / eval recording are callbacks now.
        clf = lgb.train(params, lgb_train,
                        num_boost_round=cfg.num_boost_round,
                        valid_sets=[lgb_val],
                        valid_names=['val'],
                        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                                   lgb.log_evaluation(0),
                                   lgb.record_evaluation(evals_result)])
        gc.collect()

        result = pd.DataFrame(clf.params, index=[0])

        pred_train = clf.predict(train[clf.feature_name()])
        pred_val = clf.predict(val[clf.feature_name()])
        pred_hold_out = clf.predict(hold_out[clf.feature_name()])
        gc.collect()

        train_auc = roc_auc_score(train.label, pred_train)
        val_auc = roc_auc_score(val.label, pred_val)
        hold_out_auc = roc_auc_score(hold_out.label, pred_hold_out)
        gc.collect()

        score = (abs(train_auc - val_auc) + 1) / ((1 + val_auc) * (1 + val_auc))

        result["train_auc"] = train_auc
        result["val_auc"] = val_auc
        result["hold_out_auc"] = hold_out_auc
        result["train_test_diff"] = train_auc - val_auc
        result["n_estimators"] = clf.best_iteration
        result["score"] = score

        n_estimators = int(clf.best_iteration)
        del clf

        # DataFrame.append was removed in pandas 2.0
        state["results"] = pd.concat([state["results"], result], ignore_index=True)
        state["results"].to_csv(cfg.hyperopt_results_path, index=False)
        state["i"] += 1
        logger.info("trial %2d/%d  train_auc=%.4f val_auc=%.4f hold_out_auc=%.4f "
                    "n_est=%d score=%.5f",
                    state["i"], cfg.max_evals, train_auc, val_auc, hold_out_auc,
                    n_estimators, score)
        return score

    return objective


def main(argv=None):
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = Config.load(path=args.config, overrides=overrides)

    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logging(cfg.log_level, cfg.resolved_log_file)
    logger.info("configuration: %s", json.dumps(cfg.to_dict(), default=str))

    # 1. read data and drop columns
    df = utils.process_data(cfg.data_path, cfg.drop_columns)

    # 2. split data
    train, val, hold_out = utils.data_split(
        df, cfg.train_max_yearmo, cfg.val_yearmo, cfg.hold_out_yearmo)

    # 3. label creation
    for part in (train, val, hold_out):
        processing.create_label(part, dpd=cfg.label_dpd, months=cfg.label_months)

    # 4. derived features
    ## - % Amount Paid as interest in past Loan Repayment
    ## - % of Loans defaulted in last 2 years
    for part in (train, val, hold_out):
        processing.derived_features(part)

    # 5. categorical encoding (fitted on train only)
    train, val, hold_out, target_encoder = processing.categorical_transform(
        train, val, hold_out, cfg.id_cols, cfg.encoder_params)

    # 6. feature selection
    train, val, hold_out, dropped_features = processing.select_features(
        train, val, hold_out, cfg.id_cols, cfg.rf_params, cfg.dt_params)

    # 7. hyperparameter tuning
    state = {"results": pd.DataFrame(), "i": 0}
    objective = build_objective(cfg, train, val, hold_out, state)
    trials = Trials()
    logger.info("starting hyperopt: %d trials, seed %d", cfg.max_evals, cfg.seed)
    fmin(fn=objective,
         space=training.space,
         algo=tpe.suggest,
         max_evals=cfg.max_evals,
         trials=trials,
         rstate=np.random.default_rng(cfg.seed),
         show_progressbar=False)
    logger.info("tuning results saved to %s", cfg.hyperopt_results_path)

    # 8. take the best set of parameters
    hyperopt_results = pd.read_csv(cfg.hyperopt_results_path)
    best_param_index = hyperopt_results['score'].idxmin()

    # Select LightGBM params by name rather than by positional slice `iloc[:, :19]`:
    # the column order of clf.params is not part of the LightGBM API.
    param_cols = ['num_leaves', 'max_depth', 'learning_rate', 'objective', 'metric',
                  'boosting', 'feature_fraction', 'max_bin', 'min_data_in_leaf',
                  'min_data_in_bin', 'bagging_freq', 'random_seed', 'lambda_l1',
                  'lambda_l2', 'pos_bagging_fraction', 'neg_bagging_fraction', 'verbose']
    int_params = ['num_leaves', 'max_depth', 'max_bin', 'min_data_in_leaf',
                  'min_data_in_bin', 'bagging_freq', 'random_seed', 'verbose']
    lgbm_params = dict(hyperopt_results.loc[best_param_index, param_cols])
    for p in int_params:
        lgbm_params[p] = int(lgbm_params[p])
    logger.info("best trial %d (score %.5f): %s", best_param_index,
                hyperopt_results.loc[best_param_index, 'score'], lgbm_params)

    # 9. train the final model
    clf = training.train_lgb(train, val, lgbm_params,
                             num_boost_round=cfg.num_boost_round,
                             early_stopping_rounds=cfg.early_stopping_rounds,
                             non_feature_cols=cfg.id_cols)

    # 10. save the model and everything needed to reproduce or serve it
    # A booster on its own cannot score raw data: the fitted target encoder, the
    # selected feature list and the label definition are all part of the model.
    clf.save_model(cfg.path('model.txt'), num_iteration=clf.best_iteration)

    with open(cfg.path('target_encoder.pkl'), 'wb') as f:
        pickle.dump(target_encoder, f)
    with open(cfg.path('feature_columns.json'), 'w') as f:
        json.dump(clf.feature_name(), f, indent=2)
    with open(cfg.path('processed_splits.pkl'), 'wb') as f:
        pickle.dump({'train': train, 'val': val, 'hold_out': hold_out}, f)

    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc)
                               .replace(microsecond=0).isoformat(),
        'python': sys.version.split()[0],
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__,
                     'lightgbm': lgb.__version__},
        'config': cfg.to_dict(),
        'data_rows': int(len(df)),
        'split_rows': {'train': int(len(train)), 'val': int(len(val)),
                       'hold_out': int(len(hold_out))},
        'default_rate': {'train': float(train.label.mean()),
                         'val': float(val.label.mean()),
                         'hold_out': float(hold_out.label.mean())},
        'dropped_zero_importance_features': list(dropped_features),
        'best_trial_index': int(best_param_index),
        'best_params': {k: (int(v) if isinstance(v, np.integer)
                            else float(v) if isinstance(v, np.floating) else v)
                        for k, v in lgbm_params.items()},
        'best_iteration': int(clf.best_iteration),
        'features': clf.feature_name(),
    }
    with open(cfg.path('run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info("artefacts written to %s/: model.txt, target_encoder.pkl, "
                "feature_columns.json, processed_splits.pkl, run_manifest.json",
                cfg.output_dir)
    logger.info("next: run `python evaluate.py` for the evaluation pack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
