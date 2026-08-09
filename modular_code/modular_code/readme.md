### Introduction

Credit Risk is the possibility of a loss resulting from a borrower's failure to repay a
loan or meet a contractual obligation. The primary goal of a credit risk assessment is to find out whether potential borrowers are creditworthy and have the means to repay their debts so that credit risk or loss can be minimized and the loan is granted to only creditworthy applicants.

If the borrower shows an acceptable level of default risk, then their loan application can be approved upon agreed terms.

This project involves understanding financial terminologies attached to credit risk and building a classification model for default prediction with LightGBM. Hyperparameter Optimization is done using the Hyperopt library and SHAP is used for model explainability.

> **Read `MODEL_REVIEW.md` before using any output of this pipeline.** The
> original model reaches ~0.96 AUC, but a validation review found that most of
> that comes from features unavailable when a loan application is scored.
>
> **`engine.py` reproduces the original model. `engine_v2.py` is the remediated
> one** — application-time features only, under an enforced data governance
> policy, walk-forward validation, calibrated output and a WOE scorecard
> challenger. It scores **Gini 0.2914** on an untouched hold-out. See
> `MODEL_CARD.md`. Neither model is approved for production use.

#### Two pipelines

| | `engine.py` | `engine_v2.py` |
|---|---|---|
| Purpose | reproduce the original result | the honest, governed model |
| Features | 19, incl. post-origination and protected attributes | 13, governance-filtered |
| Validation | single month | expanding-window walk-forward |
| Hold-out | scored every trial | scored once, at the end |
| Output | uncalibrated | isotonic-calibrated |
| Challenger | none | WOE + logistic scorecard |
| Hold-out Gini | 0.9272 (inflated by leakage) | **0.2914** |
| Artefacts | `output/` | `output_v2/` |

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
  config.py               configuration with validation
  logging_utils.py        logging setup
  governance.py           field classification and enforced feature policy
  data_contract.py        input schema, quality tripwires, leakage screen
  utils.py                data loading and time-based splitting
  processing.py           labelling, feature engineering, encoding, selection
  training.py             hyperparameter space and LightGBM training
  validation.py           walk-forward folds and customer dedup
  calibration.py          isotonic / Platt calibration
  woe.py                  weight of evidence, information value, scorecard
  scorecard.py            score points, adverse action reason codes, cutoffs
  monitoring.py           baseline snapshot and monthly drift checks
  evaluation.py           discrimination, calibration, stability, decile views
analysis/                 independent validation, not part of training
  diagnostics.py          leakage / encoder / feature-validity checks
  encoder_check.py        target-encoder smoothing behaviour
  encoder_sweep.py        how much categorical signal is recoverable
  governance_cost.py      what each governance decision cost, step by step
  cutoff_policy.py        approval rate vs book bad rate, on the hold-out
tests/                    205 regression tests
validate_data.py          data contract gate (exit 1 on failure)
monitor.py                monitoring baseline and monthly checks (exit 1 on alert)
output/                   artefacts from engine.py
output_v2/                artefacts from engine_v2.py
engine.py                 training entry point (original model)
engine_v2.py              training entry point (remediated model)
evaluate.py               evaluation entry point
predict.py                batch scoring entry point (v1)
predict_v2.py             batch scoring entry point (v2, calibrated + reasons)
requirements.txt
readme.md
MODEL_REVIEW.md           validation findings
MODEL_CARD.md             model card for v2
```

#### Steps

1. Install dependencies as above.
2. `python engine.py` — trains and writes to `output/`:
   `model.txt`, `target_encoder.pkl`, `feature_columns.json`,
   `processed_splits.pkl`, `run_manifest.json`, `hyperopt_results.csv`.
3. `python evaluate.py` — writes `metrics.json`, decile tables, calibration
   tables, approval curve, feature importance and plots to `output/`.
4. `python analysis/diagnostics.py` — runs the validation checks.
5. `python predict.py --input <raw.csv> --output output/scores.csv` — batch scoring.
6. `python -m pytest tests -q` — 205 regression tests.

#### Validating the data first

```bash
python validate_data.py                       # exits 1 if the contract fails
python validate_data.py --no-strict           # report without failing
```

`ml_pipeline/data_contract.py` declares what good input looks like: schema, null
rates, cardinality, ranges, and four tripwires drawn from the model review —
type stability, placeholder spellings, degenerate columns, and a single-feature
leakage screen. Both training entry points run it and record the result; use
`--strict-validation` to make it a gate.

On the shipped dataset it reports 2 errors and 7 warnings, and independently
rediscovers findings H1, H2, M7, M10 and C1.

#### Monitoring

```bash
python monitor.py baseline --up-to 202204        # once, with the model
python monitor.py check --period 202205          # every month
python monitor.py check --period 202206 --no-labels   # cohort not yet matured
```

`check` exits 1 on any ALERT so it can be scheduled. Input and score checks run
immediately; outcome checks need three EMIs to mature and report `SKIPPED` until
then rather than being silently omitted. See `MODEL_CARD.md` §8.

#### Scoring with the governed model

```bash
python engine_v2.py                                  # train
python analysis/cutoff_policy.py --target-bad-rate 0.06   # choose a cutoff
python predict_v2.py --input applications.csv --cutoff-score 555
```

`predict_v2.py` returns a calibrated probability of default, a score in points,
an approve/decline flag, and — for declines — up to four principal reasons.
Calibration is applied unconditionally and cannot be switched off: the raw
booster understates risk by 41%, so its output is not a probability.

The scores file is keyed by `User_id` and therefore carries personal data; see
`DATA.md`.

#### Configuration

All settings live in `ml_pipeline/config.py`. Precedence, lowest to highest:

```
dataclass defaults  <  --config file  <  CRD_* environment  <  CLI flags
```

```bash
python engine.py --help
python engine.py --max-evals 5 --output-dir output_smoke   # quick smoke run
python engine.py --config runs/experiment.json
CRD_MAX_EVALS=5 python engine.py
```

The config validates itself on construction and refuses to run a configuration
that would put the target into the feature matrix — the defect that made the
original `engine.py` train on its own label.

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
