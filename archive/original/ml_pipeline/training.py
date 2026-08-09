# import
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
from lightgbm import LGBMClassifier
import gc
from hyperopt import fmin, tpe, hp, anneal, Trials



#Non feature Columns (not to be used as features in model)
id_cols = ['User_id','emi_1_dpd', 'emi_2_dpd', 'emi_3_dpd', 'emi_4_dpd', 'emi_5_dpd', 'emi_6_dpd', 'max_dpd', 'yearmo']


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
def train_lgb(train, val, lgbm_params):
    '''
    train the model on the best set of parameters
    -------
    train: DataFrame
    val: DataFrame
    lgbm_params: parameter dictionary

    '''
    try:
        lgb_train = lgb.Dataset(train.drop(columns = id_cols), label = train['label'])
        lgb_val = lgb.Dataset(val.drop(columns = id_cols), label = val['label'])
        evals_result = {}
        clf = lgb.train(lgbm_params, lgb_train, 20000, valid_sets=lgb_val,
                        valid_names='val',
                        early_stopping_rounds=50,
                        verbose_eval=False, evals_result=evals_result)

    except Exception as e:
        print(e)

    else:
        return clf







