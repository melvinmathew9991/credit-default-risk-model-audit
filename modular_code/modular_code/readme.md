### Introduction

Credit Risk is the possibility of a loss resulting from a borrower's failure to repay a
loan or meet a contractual obligation. The primary goal of a credit risk assessment is to find out whether potential borrowers are creditworthy and have the means to repay their debts so that credit risk or loss can be minimized and the loan is granted to only creditworthy applicants.

If the borrower shows an acceptable level of default risk, then their loan application can be approved upon agreed terms.

This project involves understanding financial terminologies attached to credit risk and building a classification model for default prediction with LightGBM. Hyperparameter Optimization is done using the Hyperopt library and SHAP is used for model explainability.

> **Read `MODEL_REVIEW.md` before using any output of this pipeline.** The model
> reaches ~0.96 AUC, but a validation review found that most of that comes from
> features that are not available when a loan application is scored. The pipeline
> is technically sound; the feature set is not fit for the stated purpose.

#### Environment

Python **3.10** (3.10.11 verified). Two pins in the original `requirements.txt`
had no Python 3.10 wheel and were moved to the nearest release that does
(`pandas` 1.3.0 -> 1.3.5, `shap` 0.40.0 -> 0.41.0).

```
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "setuptools<81"      # hyperopt 0.2.7 imports pkg_resources
```

#### Folder Structure

```
input/
  credit_risk_data.csv
documents/
  project_document.pdf
  lightgbm_explanation.pdf
lib/                      exploratory notebook (reference only, not the pipeline)
  model.ipynb
  utils.py
  hyperopt_results.csv
ml_pipeline/              importable pipeline stages
  utils.py                data loading and time-based splitting
  processing.py           labelling, feature engineering, encoding, selection
  training.py             hyperparameter space and LightGBM training
  evaluation.py           discrimination, calibration, stability, decile views
analysis/                 independent validation, not part of training
  diagnostics.py          leakage / encoder / feature-validity checks
  encoder_check.py        target-encoder smoothing behaviour
output/                   all generated artefacts
engine.py                 training entry point
evaluate.py               evaluation entry point
predict.py                batch scoring entry point
requirements.txt
readme.md
MODEL_REVIEW.md           validation findings
```

#### Steps

1. Install dependencies as above.
2. `python engine.py` — trains and writes to `output/`:
   `model.txt`, `target_encoder.pkl`, `feature_columns.json`,
   `processed_splits.pkl`, `run_manifest.json`, `hyperopt_results.csv`.
   Override trial count and seed with the `MAX_EVALS` / `SEED` environment variables.
3. `python evaluate.py` — writes `metrics.json`, decile tables, calibration
   tables, approval curve, feature importance and plots to `output/`.
4. `python analysis/diagnostics.py` — runs the validation checks.
5. `python predict.py --input <raw.csv> --output output/scores.csv` — batch scoring.

#### Pipeline

| Step | What happens | Where |
|------|--------------|-------|
| 1 | Read csv, drop `gender` (not permissible as a credit factor) | `ml_pipeline/utils.process_data` |
| 2 | Time-based split: train `<=202203`, val `202204`, hold-out `202205` | `ml_pipeline/utils.data_split` |
| 3 | Label = DPD 60+ within first 3 EMIs (from roll-rate analysis) | `ml_pipeline/processing.create_label` |
| 4 | Derived ratio features | `ml_pipeline/processing.derived_features` |
| 5 | Target encoding of categoricals, fitted on train only | `ml_pipeline/processing.categorical_transform` |
| 6 | Drop features with zero importance in both RF and a decision tree | `ml_pipeline/processing.select_features` |
| 7 | 50-trial Hyperopt TPE search over the LightGBM space | `engine.py` + `ml_pipeline/training.space` |
| 8-10 | Refit best configuration, persist model and artefacts | `engine.py` |

The tuning objective minimises `(|train_auc - val_auc| + 1) / (1 + val_auc)^2`,
which trades validation AUC against the train/validation gap.
