# Credit Default Risk Model

Credit risk is the possibility of a loss resulting from a borrower's failure to
repay a loan. This project builds a classification model for default prediction
with LightGBM, tunes it with Hyperopt, and explains it with SHAP.

> **Read `docs/MODEL_REVIEW.md` before using any output of this pipeline.** The
> original model reaches ~0.96 AUC, but a validation review found that most of
> that comes from features unavailable when a loan application is scored.
>
> **`engine.py` reproduces the original model. `engine_v2.py` is the remediated
> one** — application-time features only, under an enforced data governance
> policy, with walk-forward validation, calibrated output and a WOE scorecard
> challenger. It scores **Gini 0.2914** on an untouched hold-out. See
> `docs/MODEL_CARD.md`. **Neither model is approved for production use.**

## Two pipelines

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

## The data

**This repository ships a sample, not the full dataset.** The full 22.8 MB file
came bundled with a paid course carrying no licence statement, so it is not
redistributed. A stratified 11,439-row sample is committed in its place.

Everything works against either: tests, CI and all entry points use
`data/credit_risk_data.csv` when it is present and fall back to
`data/credit_risk_data_sample.csv` when it is not. If you have the full file,
drop it at that path — nothing needs configuring.

The sample preserves the properties the findings rest on and reproduces the same
data-contract errors. It does **not** reproduce the headline numbers: a run
against it scores roughly Gini 0.20, against the 0.2914 in `docs/MODEL_CARD.md`.
See `docs/DATA.md` for the fidelity comparison and what cannot survive sampling.

## Setup

Python **3.10** (3.10.11 verified).

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, ruff, pre-commit
```

Two pins in the original `requirements.txt` had no Python 3.10 wheel and were
moved to the nearest release that does (`pandas` 1.3.0 → 1.3.5, `shap` 0.40.0 →
0.41.0). `hyperopt 0.2.7` additionally needs `setuptools<81`, which
`requirements-dev.txt` pins.

## Layout

```
data/credit_risk_data_sample.csv   11,439-row sample (see docs/DATA.md)
ml_pipeline/                  importable pipeline stages
  config.py                   configuration with validation
  governance.py               field classification and enforced feature policy
  data_contract.py            input schema, quality tripwires, leakage screen
  utils.py                    data loading and time-based splitting
  processing.py               labelling, feature engineering, encoding, selection
  training.py                 hyperparameter space and LightGBM training
  validation.py               walk-forward folds and customer dedup
  calibration.py              isotonic / Platt calibration
  woe.py                      weight of evidence, information value, scorecard
  scorecard.py                score points, adverse action reason codes, cutoffs
  monitoring.py               baseline snapshot and monthly drift checks
  evaluation.py               discrimination, calibration, stability, deciles
  logging_utils.py            logging setup
analysis/                     independent validation experiments, re-runnable
tests/                        243 regression tests
notebooks/                    exploratory notebook and its analysis library
docs/                         review, model card, data register, project log
archive/original/             the sources as delivered, kept as review evidence
output/ output_v2/            generated artefacts
```

Entry points live at the repository root and are run from there.

## Running it

```bash
python validate_data.py                      # data contract gate
python engine.py                             # original model
python evaluate.py                           # evaluation pack for engine.py
python engine_v2.py                          # governed model
python analysis/diagnostics.py               # the validation checks
python -m pytest tests -q                    # 243 tests
```

### Scoring with the governed model

```bash
python engine_v2.py
python analysis/cutoff_policy.py --target-bad-rate 0.06   # choose a cutoff
python predict_v2.py --input applications.csv --cutoff-score 555
```

`predict_v2.py` returns a calibrated probability of default, a score in points,
an approve/decline flag, and — for declines — up to four principal reasons.
Calibration is applied unconditionally and cannot be switched off: the raw
booster understates risk by 41%, so its output is not a probability.

The scores file is keyed by `User_id` and therefore carries personal data; see
`docs/DATA.md`. It is git-ignored, and both pre-commit and CI fail if one is
ever staged.

### Monitoring

```bash
python monitor.py baseline --up-to 202204        # once, with the model
python monitor.py check --period 202205          # every month
python monitor.py check --period 202206 --no-labels   # cohort not yet matured
```

`check` exits 1 on any ALERT so it can be scheduled. Input and score checks run
immediately; outcome checks need three EMIs to mature and report `SKIPPED` until
then rather than being silently omitted. See `docs/MODEL_CARD.md` §8.

## Configuration

All settings live in `ml_pipeline/config.py`. Precedence, lowest to highest:

```
dataclass defaults  <  --config file  <  CRD_* environment  <  CLI flags
```

```bash
python engine.py --help
python engine.py --max-evals 5 --output-dir output_smoke
CRD_MAX_EVALS=5 python engine.py
```

The config validates itself on construction and refuses to run a configuration
that would put the target into the feature matrix — the defect that made the
original `engine.py` train on its own label.

## Pipeline

| Step | What happens | Where |
|------|--------------|-------|
| 0 | Validate the input against the data contract | `ml_pipeline/data_contract.py` |
| 1 | Read csv, drop `gender` (not permissible as a credit factor) | `ml_pipeline/utils.process_data` |
| 2 | Time-based split: train `<=202203`, val `202204`, hold-out `202205` | `ml_pipeline/utils.data_split` |
| 3 | Label = DPD 60+ within the first 3 EMIs (from roll-rate analysis) | `ml_pipeline/processing.create_label` |
| 4 | Derived ratio features | `ml_pipeline/processing.derived_features` |
| 5 | Target encoding of categoricals, fitted on train only | `ml_pipeline/processing.categorical_transform` |
| 6 | Drop features with zero importance in both RF and a decision tree | `ml_pipeline/processing.select_features` |
| 7 | Hyperopt TPE search over the LightGBM space | `engine.py` |
| 8–10 | Refit the best configuration, persist model and artefacts | `engine.py` |

`engine_v2.py` replaces steps 5–7 with a governance-filtered feature set,
placeholder cleaning, walk-forward folds and calibration.

## Documentation

| File | Purpose |
|---|---|
| `docs/MODEL_REVIEW.md` | Validation findings and remediation status |
| `docs/MODEL_CARD.md` | Intended use, limitations, monitoring plan, approval block |
| `docs/DATA.md` | Canonical dataset, SHA-256, known quality issues |
| `docs/PROJECT_LOG.md` | Complete record of the engagement |
