import pandas as pd
import numpy as np

import category_encoders as ce
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score

import shap

from hyperopt import fmin, tpe, hp, anneal, Trials

import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def numeric_columns(df):
    """Numeric columns of a DataFrame.

    Uses the public select_dtypes API rather than the private
    DataFrame._get_numeric_data(), which carries no stability guarantee.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    Index
    """
    return df.select_dtypes(include=[np.number]).columns


def categorical_columns(df):
    """Categorical / object columns of a DataFrame.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    Index
    """
    return df.select_dtypes(include=['category', 'object']).columns


def process_data(path, drop_columns):
    """Read data, drop columns and do processing
    Parameters
    ----------
    path : String
    drop_columns : List

    Returns
    -------
    df :  DataFrame
    """
    # low_memory=False forces single-pass type inference. With the default
    # chunked inference pandas assigns different python types to identical text
    # in `industry` and `work_experience` depending on which 128k-row chunk it
    # lands in (str '0' vs int 0 vs float 0.0), which makes the encoded feature
    # values depend on the row's position in the file.
    df = pd.read_csv(path, low_memory=False).drop(columns = drop_columns)
    return df

def understand_data(df, id_cols):
    """Print Numeric & Categorical columns separately,
    Print columnwise null counts,
    Print Columns with 0 Variance
    Parameters
    ----------
    df : DataFrame
    id_cols : List

    Returns
    -------
    """
    features = df.drop(columns = id_cols)
    numeric = features[numeric_columns(features)]
    print(f"Numeric Columns : {list(numeric.columns)}")
    print("")
    print(f"Categorical Columns : {list(categorical_columns(features))}")
    print("")
    print(f"Null Counts")
    print(features.isna().sum())
    print("")
    print(f"Zero Variance Columns: {list(numeric.loc[:, numeric.std() == 0].columns)}")

def data_split(df):
    """Split data in train, val, hold_out
    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    train :  DataFrame,
    val :  DataFrame,
    hold_out :  DataFrame
    """
    train = df[df.yearmo<=202203]
    val = df[df.yearmo==202204]
    hold_out = df[df.yearmo==202205]

    return train.reset_index(drop = True), val.reset_index(drop = True), hold_out.reset_index(drop = True)

def univariate_stats(df, id_cols):
    """Returns Univatiate stats for numeric and categorical variables
    Parameters
    ----------
    df : DataFrame
    id_cols : List

    Returns
    -------
    numeric summary :  DataFrame
    categorical summary :  DataFrame
    """
    features = df.drop(columns = id_cols)
    return (features[numeric_columns(features)].describe(),
            features[categorical_columns(features)].describe())


def dpd_roll_rate(df):
    """DPD Roll Rate Analysis,
    number and % of customer passed the particular dpd

    `user_percent` is the share of ALL customers that reached each DPD bucket.
    `roll_rate` is the share of the previous bucket that rolled forward into
    this one, and `recovery_rate` its complement - those are the numbers that
    justify a label cutoff, so they are reported explicitly rather than left to
    be inferred from `user_percent`.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    dpd_flow : DataFrame
    """

    dpd_flow = pd.DataFrame(columns = ["dpd","user_count"])
    for dpd in [0,30,60,90]:
        user_count = len(df[df.max_dpd>=dpd])
        dpd_flow.loc[len(dpd_flow.index)] = [dpd, user_count]
    dpd_flow['user_count']  = dpd_flow['user_count'].astype(int)
    dpd_flow['user_percent'] = (dpd_flow['user_count']*100/max(dpd_flow['user_count'])).round(2).astype(str)+' %'

    previous = dpd_flow['user_count'].shift(1)
    roll = dpd_flow['user_count']*100/previous
    dpd_flow['roll_rate'] = roll.round(2).astype(str).replace('nan', '-')+' %'
    dpd_flow['recovery_rate'] = (100-roll).round(2).astype(str).replace('nan', '-')+' %'
    return dpd_flow


def window_roll_rate(df, dpd):
    """Window Roll Rate Analysis,
    First EMI of reaching dpd >= X, count and % by First EMI
    Parameters
    ----------
    df : DataFrame
    dpd : Int (30, 60, 90)

    Returns
    -------
    window_roll : DataFrame
    """
    # .copy() - assigning into a slice of `df` raises SettingWithCopyWarning and
    # is a silent no-op under copy-on-write.
    df2 = df[df.max_dpd>=dpd].copy()
    df2['first_default'] = np.where(df2.emi_1_dpd>=dpd, 1,
                                   np.where(df2.emi_2_dpd>=dpd, 2,
                                           np.where(df2.emi_3_dpd>=dpd, 3,
                                                   np.where(df2.emi_4_dpd>=dpd, 4,
                                                           np.where(df2.emi_5_dpd>=dpd, 5,
                                                                   np.where(df2.emi_6_dpd>=dpd, 6, 0))))))

    window_roll = df2.groupby('first_default')['User_id'].count().reset_index()
    window_roll['user_percent'] = (window_roll['User_id']*100/sum(window_roll['User_id'])).round(2).astype(str)+' %'
    window_roll.columns = ['first_default_emi','users_count', '% of Users']
    return window_roll

def create_label(df, dpd, months):
    """Genrate label according to dpd in months,
    returns dataframe with label columns
    Parameters
    ----------
    df : DataFrame
    dpd : Int (30, 60, 90)
    months : Int (1,2,3,4,5,6)

    Returns
    -------
    df : DataFrame
    """
    months = ["emi_"+str(x)+"_dpd" for x in range(1, months+1)]
    df['label'] = np.where(df[months].max(axis = 1)>=dpd, 1, 0)
    print("label columns added to dataframe")
    return df

def label_distribution(data_list, data_list_name, label_name):
    """Print label distribution of list of dataframes
    ----------
    data_list : List of DataFrames
    data_list_name : List of Data Type (like Training, Validation)
    label_name : String (Column Name of Label)

    Returns
    -------
    """
    i = 0
    for d in data_list:
        label_distribution = pd.DataFrame(d[label_name].value_counts()).reset_index()
        label_distribution.columns = [label_name, 'user_count']
        label_distribution['% users'] = label_distribution['user_count']*100/sum(label_distribution['user_count'])
        print("")
        print(f"label distribution of {data_list_name[i]}")
        print(label_distribution)
        i = i+1

def derived_features(df):
    """Create Some Features
    ----------
    df : DataFrame

    Returns
    -------
    df : DataFrame
    """
    df['interest_received_ratio'] = (df['interest_received']/df['total_payement']).replace([np.inf, -np.inf], 0).fillna(0)
    df['total_payement_per_loan'] = (df['total_payement']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)
    df['delinq_2yrs_ratio'] = (df['delinq_2yrs']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

class eda:
    """EDA Class
    - Univarite EDA
      - Numeric Features Summary
      - Categorical Features Summary
    - Bivariate EDA
      - Correlation Plot
      - Box Plot"""
    def __init__(self, df, id_cols):
        """Initialize Class
        Parameters
        ----------
        df : DataFrame
        id_cols : List
        """
        self.df = df
        self.id_cols = id_cols
        features = df.drop(columns = id_cols)
        self.num_cols = numeric_columns(features)
        self.cat_cols = categorical_columns(features)

    def numeric_summary(self):
        """Summary of Numeric Features"""
        return self.df[self.num_cols].describe()

    def categorical_summary(self):
        """Summary of Categorical Features"""
        return self.df[self.cat_cols].describe()

    def correlation_plot(self):
        """Correlation Plot of Numerical Features"""
        corr = self.df[self.num_cols].corr()

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        fig = plt.figure(figsize=(len(corr.columns), len(corr.columns)))
        with sns.axes_style("white"):
            ax = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                             square=True, annot=True)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_autoscalex_on(True)
        ax.set_autoscaley_on(True)

        plt.show()
        plt.close(fig)

    def box_plot(self, group):
        """Box Plot of Numerical Features vs Group Features
        Parameters
        ----------
        group : List of Features (against with box plot of numerica features to be done)
        """
        for g in group:
            for col in self.num_cols:
                fig = plt.figure()
                sns.boxplot(x = g, y = col, data = self.df)
                plt.ylabel('Values')
                plt.title(col)
                plt.show()
                # close explicitly: this loop opens one figure per numeric
                # feature and matplotlib keeps every one of them in memory.
                plt.close(fig)

class categorical_encoding:
    """Target Encoding of categorical variables
    input dataframe, categorical columns, label name, parameters of target_encoder
    """
    def __init__(self,params):
        """
        Parameters
        ----------
        params : Dict
        """
        self.params = params

    def fit(self, df, cat_cols, label):
        """Fitting Encoder
        Parameters
        ----------
        df : DataFrame
        cat_cols : List (Categorical columns)
        label : String
        """
        self.te = ce.target_encoder.TargetEncoder(**self.params)
        self.te.fit(df[cat_cols], df[label])
        # category_encoders renamed `feature_names` -> `get_feature_names_in()`
        # in 2.6; cache the fitted column order so transform works on either.
        self.feature_names = list(getattr(self.te, "feature_names", None)
                                  or self.te.get_feature_names_in())

    def transform(self, d):
        """Transforming Data Encode and inplace transform categorical features
        Parameters
        ----------
        d : DataFrame

        Returns
        -------
        d : DataFrame
        """
        d = pd.concat([d.drop(columns = self.feature_names), self.te.transform(d[self.feature_names])], axis = 1)
        return d

def random_forest_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using random forest
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    rf = RandomForestClassifier(**params)
    rf.fit(df.drop(columns = id_cols).fillna(0), df['label'])
    fi = pd.DataFrame({"features":df.drop(columns = id_cols).columns, "importance":rf.feature_importances_})
    zero_fi = fi[fi.importance==0]['features']
    return zero_fi

def decision_tree_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using decision tree
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    dt = DecisionTreeClassifier(**params)
    dt.fit(df.drop(columns = id_cols).fillna(0), df['label'])
    fi = pd.DataFrame({"features":df.drop(columns = id_cols).columns, "importance":dt.feature_importances_})
    zero_fi = fi[fi.importance==0]['features']
    return zero_fi


def _check_lists(target_list, pred_list, data_type_list=None):
    """Validate that the parallel metric inputs line up.

    The plotting and metric helpers below take any number of datasets. They used
    to index positions 0, 1 and 2 directly, which silently produced an
    IndexError - or worse, quietly ignored extra datasets.
    """
    if len(target_list) != len(pred_list):
        raise ValueError(f"target_list has {len(target_list)} entries but "
                         f"pred_list has {len(pred_list)}")
    if data_type_list is not None and len(data_type_list) != len(target_list):
        raise ValueError(f"data_type_list has {len(data_type_list)} entries but "
                         f"target_list has {len(target_list)}")
    if not target_list:
        raise ValueError("no datasets supplied")
    return [f"dataset_{i}" for i in range(len(target_list))] if data_type_list is None else list(data_type_list)


def roc_auc(target_list, pred_list, data_type_list=None):
    """Print ROC AUC for Target and Predictions
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list, optional
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)
    for name, y, p in zip(names, target_list, pred_list):
        print(f"{name}: {roc_auc_score(y, p)}")

def roc_auc_curve(target_list, pred_list, data_type_list=None):
    """Print ROC AUC Curve
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list, optional
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)

    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title('ROC Curve')
    for i, (name, y, p) in enumerate(zip(names, target_list, pred_list)):
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f'{name} AUC = %0.3f' % roc_auc_score(y, p), color=f'C{i}')

    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1],'r--', color = 'black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close(fig)


## PR AUC
def pr_auc(target_list, pred_list, data_type_list=None):
    """Print PR AUC Values
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list, optional
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)
    for name, y, p in zip(names, target_list, pred_list):
        pr, re, _ = precision_recall_curve(y, p)
        print(f"{name}: {auc(re, pr)}")

def pr_auc_curve(target_list, pred_list, data_type_list=None):
    """Print PR AUC Curve
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list, optional
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)

    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.title('Precision Recall Curve')
    for i, (name, y, p) in enumerate(zip(names, target_list, pred_list)):
        pr, re, _ = precision_recall_curve(y, p)
        plt.plot(re, pr, label=f'{name} Precision = %0.3f' % average_precision_score(y, p), color=f'C{i}')

    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.close(fig)

def score_distribution(target_list, pred_list, data_type_list):
    """Print Score Distribution Plots
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)

    for name, y_actual, y_predicted in zip(names, target_list, pred_list):
        sub_df = pd.DataFrame({"y_actual": np.asarray(y_actual), "y_predicted": np.asarray(y_predicted)})

        f, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=(16, 6))
        # sns.distplot was deprecated in seaborn 0.11 and removed in 0.14.
        sns.histplot(sub_df[sub_df['y_actual']==1].y_predicted.values, kde=True, stat="density",
                     label="Defaulter", ax=ax, color="C1")
        sns.histplot(sub_df[sub_df['y_actual']==0].y_predicted.values, kde=True, stat="density",
                     label="Non-Defaulter", ax=ax, color="C0")
        plt.xlabel('Predicted positive class score')
        plt.ylabel('Density')
        plt.title(str(name) +' Distribution of predicted score')
        plt.legend(loc="upper right")
        plt.show()
        plt.close(f)

def shap_importance(model, data_list, data_type_list):
    """Plot SHAP for top 20 features
    Parameters
    ----------
    model : object
        Model Object (Classifier).
    data_list : list
        List of Multiple DataFrames
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """
    if len(data_list) != len(data_type_list):
        raise ValueError(f"data_list has {len(data_list)} entries but "
                         f"data_type_list has {len(data_type_list)}")

    # A booster reloaded from model.txt has empty .params, which makes
    # shap.TreeExplainer raise KeyError('objective').
    if hasattr(model, "params") and "objective" not in model.params:
        model.params["objective"] = "binary"

    explainer = shap.TreeExplainer(model)
    for d, name in zip(data_list, data_type_list):
        tmp_shap_values = explainer.shap_values(d[model.feature_name()])
        # For a binary booster shap returns [class_0, class_1] where
        # class_0 == -class_1. Index 1 is the defaulter class; taking index 0
        # would invert the sign of every explanation, so a feature that reduces
        # default risk would appear to increase it.
        if isinstance(tmp_shap_values, list):
            tmp_shap_values = tmp_shap_values[1]
        shap.summary_plot(tmp_shap_values, d[model.feature_name()], plot_type="dot", max_display = 20, show=False)
        fig = plt.gcf()
        fig.set_size_inches(14.5, 10.5)
        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.set_title(name+" Shap Values (class 1 = defaulter)")
        ax.set(ylabel="Features", xlabel="SHAP value")
        plt.show()
        plt.close(fig)

def class_rate(target_list, pred_list, data_type_list):

    """Print Class Rate Curves
    Parameters
    ----------
    target_list : list
        List of Multiple Target Arrays.
    pred_list : list
        List of Multiple Predicted Array Arrays
    data_type_list : list
        Data Tagging like Train, Val, Hold Out
    """
    names = _check_lists(target_list, pred_list, data_type_list)

    def buckets(y_actual, y_predicted, bins):
        df = pd.DataFrame({"y_actual": np.asarray(y_actual), "y_predicted": np.asarray(y_predicted)})
        if bins is None:
            _, bins = pd.qcut(y_predicted, 30, retbins=True, duplicates="drop")
            df['score_bucket'] = pd.cut(df["y_predicted"], bins=bins)
            return df, bins
        else:
            df['score_bucket'] = pd.cut(df["y_predicted"], bins=bins)
            return df

    def slope_df(actual, predicted, data_type):
        slope = pd.DataFrame(columns=['score_bucket', 'score_bins','count', 'sum', 'positive_class_rate', 'volume_percentage','Data'])
        # Bins are always taken from the first dataset. The original keyed this
        # on the literal name "Train", so any run whose first dataset was not
        # called "Train" raised UnboundLocalError on `bins`.
        bins = None
        for i in range(len(data_type)):
            y_actual = actual[i]
            y_predicted = predicted[i]
            df_type = data_type[i]
            if bins is None:
                df_bucket, bins = buckets(y_actual, y_predicted, None)
            else:
                df_bucket = buckets(y_actual, y_predicted, bins)
            df_slope = df_bucket.groupby(['score_bucket'])["y_actual"].agg(['count', 'sum']).sort_index(ascending=False).reset_index()
            df_slope['positive_class_rate'] = (df_slope['sum'] / df_slope['count'])
            df_slope['volume_percentage'] = df_slope['count'] / df_slope['count'].sum()
            df_slope['Data'] = df_type
            slope = pd.concat([df_slope, slope], ignore_index=True)
        slope = slope.reset_index(drop = True)
        return slope, bins

    def slope_plot(df):
        fig = plt.figure(figsize=(12, 8))
        plt.grid(True)

        ax1 = sns.pointplot(x="score_bucket", y="positive_class_rate", data=df, hue="Data")
        ax1.set(ylabel="% Default", xlabel="Score Buckets")
        ax1.legend(loc='center right')
        ax1.set_xticklabels(df["score_bucket"].unique().tolist(), rotation=90)
        ax1.set_title("Bucket wise % Default")

        ax2 = ax1.twinx()
        ax2 = sns.barplot(x="score_bucket", y="volume_percentage", hue="Data", data=df, **{'alpha': 0.3})
        ax2.set(ylabel="Percentage of Volume", xlabel="")
        ax2.legend(loc='upper left')

        plt.show()
        plt.close(fig)

    slope, slope_bins = slope_df(target_list, pred_list, names)
    slope_plot(slope)

def cutoff_score(label, prediction, default_rate):
    """Cutoff Score at a particular default rate
    Parameters
    ----------
    label : Array
        Labels according to which cutoff need to be decided.
    prediction : Array
        Model Scores
    default_rate : Float
        Desired Cummulative default rate

    Returns
    -------
    cutoff : Float
        Highest score that can be accepted while keeping the cumulative default
        rate of the accepted population at or below `default_rate`.
    """
    pred = pd.DataFrame({"label":label, "score":prediction}).sort_values(by = 'score').reset_index(drop = True)
    pred['cummulative_defaulters'] = pred['label'].cumsum(axis = 0)
    # denominator is the number of accounts accepted so far, which is index+1 -
    # dividing by the 0-based index understates it by one account.
    pred['cummulative_default'] = pred['cummulative_defaulters']/(pred.index + 1)
    cutoff = pred[pred.cummulative_default<=default_rate]['score'].max()

    return cutoff
