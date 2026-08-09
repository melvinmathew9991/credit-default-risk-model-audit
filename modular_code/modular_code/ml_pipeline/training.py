# import
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
from lightgbm import LGBMClassifier
import gc
from hyperopt import fmin, tpe, hp, anneal, Trials

from .logging_utils import get_logger

logger = get_logger(__name__)



#Non feature Columns (not to be used as features in model)
# NOTE: 'label' MUST be listed here. It is the target - if it stays in the feature
# matrix LightGBM trains on the answer and every AUC becomes 1.0.
id_cols = ['User_id','emi_1_dpd', 'emi_2_dpd', 'emi_3_dpd', 'emi_4_dpd', 'emi_5_dpd', 'emi_6_dpd', 'max_dpd', 'yearmo', 'label']


## Hyperparameter space
## Space is selection of data point from the given distribution
## Distribution is defined for every hyperparameter seperately
space = {
    'num_leaves': hp.quniform('num_leaves', 2, 24, 1), # Uniform integer between 2 and 24
    'max_depth': hp.quniform('max_depth', 2, 12, 1), # Uniform integer between 2 and 12
    'learning_rate': hp.uniform('learning_rate', 0.005, 0.015), # Values between 0.005 to 0.015
    'feature_fraction' : hp.uniform('feature_fraction', 0.1, 1), # Values between 0.1 to 1
    'max_bin' : hp.quniform('max_bin', 10, 100, 10),
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 25, 1000, 25),
    'lambda_l1' : hp.uniform('lambda_l1', 0, 50),
    'lambda_l2' : hp.uniform('lambda_l2', 0, 50),
    'min_data_in_bin' : hp.quniform('min_data_in_bin', 5, 100, 5),
    'pos_bagging_fraction' : hp.uniform('pos_bagging_fraction', 0.1, 1),
    'neg_bagging_fraction' : hp.uniform('neg_bagging_fraction', 0.1, 1)
    }




# Train lgb
def train_lgb(train, val, lgbm_params, num_boost_round=20000,
              early_stopping_rounds=50, non_feature_cols=None):
    '''
    train the model on the best set of parameters
    -------
    train: DataFrame
    val: DataFrame
    lgbm_params: parameter dictionary
    num_boost_round: int
    early_stopping_rounds: int
    non_feature_cols: List, optional
        Columns to exclude from the feature matrix; defaults to `id_cols`.

    Returns
    -------
    clf : lgb.Booster

    Raises
    ------
    ValueError
        If the target would end up inside the feature matrix.
    '''
    non_feature_cols = list(id_cols if non_feature_cols is None else non_feature_cols)

    if 'label' not in non_feature_cols:
        raise ValueError(
            "'label' is missing from the non-feature columns, so the target "
            "would be used as a predictor and every metric would be invalid")

    feature_cols = [c for c in train.columns if c not in non_feature_cols]
    if not feature_cols:
        raise ValueError("no feature columns remain after excluding non-feature columns")

    logger.info("training on %d features, %d rows (val %d rows)",
                len(feature_cols), len(train), len(val))

    lgb_train = lgb.Dataset(train[feature_cols], label = train['label'])
    lgb_val = lgb.Dataset(val[feature_cols], label = val['label'])
    evals_result = {}
    # LightGBM 4.x moved early_stopping_rounds / verbose_eval / evals_result
    # out of lgb.train() and into the callbacks API.
    clf = lgb.train(lgbm_params, lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_val],
                    valid_names=['val'],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                               lgb.log_evaluation(0),
                               lgb.record_evaluation(evals_result)])

    logger.info("stopped at iteration %d of %d", clf.best_iteration, num_boost_round)
    if clf.best_iteration >= num_boost_round:
        logger.warning("early stopping never triggered - the model may be "
                       "under-trained; consider raising num_boost_round")
    return clf







