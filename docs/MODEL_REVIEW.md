# Model Validation & Engineering Review

**Subject:** Credit Default Risk Prediction Model (LightGBM)
**Date:** 2026-08-09
**Scope:** full end-to-end execution of the project, independent validation of
the model and analysis, and an assessment against data-science and SDLC standards.

---

## 1. Verdict

The pipeline is **methodologically well-organised and runs to completion**, and its
headline result reproduces exactly. But the model as built **cannot be deployed for
its stated purpose**, for one reason: most of its predictive power comes from
fields that do not exist at the moment a loan application is scored.

| | |
|---|---|
| Reported / reproduced performance | **val AUC 0.9650, hold-out AUC 0.9636** |
| Performance using only application-time features | **val AUC 0.7126, hold-out AUC 0.7147** |
| Share of Gini attributable to post-origination fields | **~85%** |
| Share of model gain from those fields | **78.7%** (83.2% after corrections) |
| Share of model gain from legitimate application-time features | **5.4%** |

A Gini of 0.43 is a perfectly respectable first-payment-default model. A Gini of
0.94 on a credit application is not achievable, and the gap is the finding.

**Recommendation:** do not promote this model. Rebuild the feature set from
application-time data only, then re-baseline against the 0.7126 AUC benchmark
established here.

---

## 2. What was executed

Reproduced on a clean Python 3.10.11 virtual environment, using the project's own
pinned dependency set.

The pipeline was run twice:

* **Run 1 — faithful reproduction.** Only the defects that prevent execution at all
  were repaired (C2, C3, E1). Archived in `output/baseline_faithful_run/`.
* **Run 2 — corrected.** Adds the determinism and artefact fixes (H2, M1, M9).
  Its outputs are the live contents of `output/`.

| Stage | Result |
|---|---|
| `engine.py` run 1 (50 Hyperopt trials) | completed, 16m 01s |
| `engine.py` run 2 (50 Hyperopt trials) | completed, 46m 33s |
| `evaluate.py` | completed, full metric pack + 6 plots |
| `predict.py` | completed, scored 143,727 rows |
| `analysis/diagnostics.py` | completed, 6 validation checks |
| `tests/` (17 regression tests) | **17 passed** |

**Reproduction is exact.** The best trial found here matches the value recorded in
the original notebook to every digit:

| | original notebook | this run |
|---|---|---|
| best hyperopt loss | 0.26055619307290495 | 0.26055619307290495 |
| best trial index | 21 | 21 |
| train AUC | 0.9709666608309787 | 0.970967 |
| val AUC | 0.9649508759429978 | 0.964951 |
| hold-out AUC | 0.9635950364843274 | 0.963595 |
| n_estimators | 3041 | 3041 |

Three defects had to be repaired before the pipeline could run at all; these are
recorded as findings C2, C3 and E1 below.

### 2.1 Measured performance of the reproduced model

Full metric pack in `output/baseline_faithful_run/`.

| split | n | bad % | AUC | Gini | KS | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| train | 86,250 | 9.33% | 0.9710 | 0.9419 | 0.8202 | 0.8838 | 0.0261 | 0.0100 |
| val | 28,750 | 9.77% | 0.9650 | 0.9299 | 0.8157 | 0.8792 | 0.0275 | 0.0097 |
| hold-out | 28,727 | 9.08% | 0.9636 | 0.9272 | 0.8030 | 0.8678 | 0.0268 | 0.0080 |

Score stability is essentially perfect (PSI train→val 0.0005, train→hold-out
0.0010), and the train/validation gap is small (0.006 AUC). On the usual
diagnostics this model looks excellent — which is exactly why the leakage in
section 3 matters: **none of the standard checks catch it.**

Validation decile table (decile 1 = riskiest):

| decile | n | bads | bad rate | cumulative bad capture | lift |
|---|---|---|---|---|---|
| 1 | 2,875 | 2,269 | 78.9% | 80.8% | 8.08 |
| 2 | 2,875 | 307 | 10.7% | 91.7% | 1.09 |
| 3 | 2,875 | 99 | 3.4% | 95.2% | 0.35 |
| 4–10 | 20,125 | 134 | 0.7% | 100% | — |

A single decile containing 79% bad rate and capturing 81% of all defaulters is not
a plausible credit-application result; it is the signature of the model reading the
outcome.

Where the gain actually comes from:

| feature group | % of total gain |
|---|---|
| Post-origination outcome fields (C1) | **78.7%** |
| `industry` + `work_experience` placeholder artifact (H1) | **15.9%** |
| All legitimate application-time features | **5.4%** |

### 2.2 Effect of the corrections (run 2)

| | run 1 (faithful) | run 2 (corrected) |
|---|---|---|
| val AUC | 0.9650 | 0.9517 |
| val Gini | 0.9299 | 0.9034 |
| val KS | 0.8157 | 0.7654 |
| hold-out AUC | 0.9636 | 0.9486 |
| best hyperopt loss | 0.260556 | 0.265131 |
| trees in final model | 3,041 | 5,368 |
| outcome fields' share of gain | 78.7% | 83.2% |

Performance falls slightly once the CSV is parsed deterministically (H2). That is
the expected direction: run 1 was partly fitting a chunk-boundary artifact, so
removing it removes some apparent power. The corrected figures are the ones to
quote — and they change nothing about the C1 conclusion, since the outcome fields'
share of gain *rises* to 83.2%.

---

## 3. Critical findings

### C1 — Target leakage: the model is trained on the outcome it predicts

`received_principal`, `total_payement` and `interest_received` (and the two ratios
derived from them) describe **repayment activity on the loan being scored**. The
label — 60+ DPD within the first three EMIs — is a direct function of whether the
borrower made those payments.

The data dictionary describes these as history over the *last 2 years*, i.e. prior
loans. The data contradicts that:

| check | result |
|---|---|
| rows with `number_of_loans == 0` | 143,136 (99.59%) |
| of those, rows with `total_payement > 0` | 143,105 |
| of those, rows with `received_principal > 0` | 143,003 |
| their mean `total_payement` | 10,842.12 |
| `received_principal` maximum | 35,000.01 |

Borrowers with **zero prior loans** cannot have prior repayment history. These are
current-loan fields, and the maximum of exactly $35,000 is the standard maximum
loan amount of the public dataset this appears to be derived from.

**Counterfactual experiment** (`analysis/diagnostics.py`, check 6; identical
hyperparameters, only the feature set differs):

| model | features | val AUC | val Gini | hold-out AUC | hold-out Gini |
|---|---|---|---|---|---|
| A — as built | 19 | 0.9701 | 0.9401 | 0.9687 | 0.9374 |
| B — application-time only | 14 | 0.7126 | 0.4252 | 0.7147 | 0.4294 |

Removing five outcome-contaminated features costs 0.515 Gini — **85% of the
model's apparent power**.

Univariate confirmation on the training split:

| feature | AUC | \|Gini\| |
|---|---|---|
| received_principal | 0.2498 | 0.5004 |
| interest_received_ratio | 0.6953 | 0.3907 |
| total_payement | 0.3372 | 0.3257 |
| total_income | 0.4397 | 0.1205 |
| delinq_2yrs | 0.4899 | 0.0201 |

A single column, `received_principal`, reaches Gini 0.50 on its own — more than the
entire legitimate feature set combined.

### C2 — `engine.py` trained the model on the label itself

`id_cols` in `engine.py` and `ml_pipeline/training.py` omitted `'label'`. Since the
feature matrix is built as `train.drop(columns=id_cols)`, the target remained in it:

* `engine.py:68` — `lgb.Dataset(train.drop(columns=id_cols), label=train.label)`
* `ml_pipeline/training.py:48` — same construction in `train_lgb`
* `ml_pipeline/processing.py:144,162` — and in the RandomForest / DecisionTree fits
  used for feature selection, which would drive every legitimate feature to zero
  importance

The exploratory notebook is correct — it appends `'label'` to `id_cols` in cell 23.
The error was introduced when the notebook was modularised. Any run of the shipped
`engine.py` produces a model with the answer as an input.

### C3 — `engine.py` could not complete

Step 7 writes trial results to `output/hyperopt_results.csv`; step 8 reads
`hyperopt_results.csv`. The shipped script raises `FileNotFoundError` after the
16-minute tuning loop, before training or saving anything. The `output/model.txt`
committed to the repository therefore cannot have been produced by this script.

---

## 4. High-severity findings

### H1 — 84% of two features is a placeholder written two ways, and the spelling predicts default

`work_experience` and `industry` are filled with a placeholder zero for most rows,
recorded inconsistently as `"0"` and `"0.0"`. The two spellings carry very
different default rates:

| value | rows | default rate |
|---|---|---|
| `"0"` | 87,848 (61.1%) | 9.75% |
| `"0.0"` | 32,766 (22.8%) | 4.76% |
| a real work-experience band | 23,113 (16.1%) | 14.46% |

The split is stable in every month of the data (61% / 23% / 16% in all five), and
so is the default-rate gap — which is precisely why it survives into the hold-out
period and inflates out-of-time performance.

Meanwhile the *genuine* work-experience bands carry almost no signal:

| band | rows | default rate |
|---|---|---|
| `<1` | 1,625 | 14.28% |
| `1-2` | 1,997 | 14.37% |
| `2-3` | 1,720 | 16.51% |
| `3-5` | 1,574 | 13.91% |
| `5-10` | 9,000 | 14.92% |
| `10+` | 7,193 | 13.58% |

All within 3 points of each other. Every bit of apparent power in this feature is
the placeholder encoding, not work experience.

The two columns are also largely redundant: `industry == work_experience` in
120,618 of 143,727 rows (83.92%); they differ only where a real band is present.
Both ranked in the original model's top six features by gain.

This is a fingerprint of how the dataset was assembled, not a borrower attribute.

### H2 — Feature values depended on the row's position in the CSV file

`pd.read_csv` defaults to chunked type inference. Identical text in `industry` and
`work_experience` was assigned different Python types depending on which 128k-row
chunk it fell in — `str "0"` (42,425 rows), `int 0` (45,423 rows), `float 0.0`
(32,766 rows). The float-typed rows span index 98,304–131,071 exactly: a chunk
boundary, not a data property.

Downstream these become **distinct categories** for the target encoder, so encoded
feature values were a function of file row order. Fixed by reading with
`low_memory=False`; the fix materially changes results (trial 3 val AUC moved from
0.9170 to 0.9440), because consistent parsing lets the model exploit the H1
artifact more cleanly.

### H3 — Every SHAP explanation in the notebook has its sign reversed

`shap_importance` (`notebooks/utils.py:478`) plots `tmp_shap_values[0]`. For a
binary LightGBM booster SHAP returns `[class_0, class_1]`, and index 0 is the
**non-default** class. Verified: `sv[0] == -sv[1]` to machine precision.

The accompanying markdown tells the reader "Right Side of Grey Vertical Line is for
class 1". So for `received_principal` — the single most important feature — the
plot shows high values pushing right, and the reader concludes more principal repaid
means more default risk. The true relationship is the opposite:

| measure | value |
|---|---|
| corr(received_principal, plotted SHAP `sv[0]`) | +0.8466 |
| corr(received_principal, true class-1 SHAP) | −0.8466 |
| univariate AUC of received_principal | 0.2498 (inverse) |

Magnitudes and the feature ranking are unaffected; every **direction** is backwards.
The explainability section, whose purpose is to make the model auditable, states the
opposite of what the model does.

Separately, a booster reloaded from `model.txt` has empty `.params`, so
`shap.TreeExplainer(...).shap_values()` raises `KeyError: 'objective'` — the
notebook's SHAP code cannot be run against a saved model at all.

### H4 — Protected attributes retained while gender is dropped

The project drops `gender` explicitly and correctly, on the stated grounds that it
cannot be used as a credit factor. But the final feature set retains:

* **`married`** — marital status
* **`dependents`** — family status
* **`home_type`**

Marital status and dependants are protected characteristics under ECOA /
Regulation B and equivalent regimes. Dropping gender while keeping these is
inconsistent, and no justification is recorded.

`pincode` — a geographic redlining proxy — *was* excluded, but not by design: it was
dropped because the target encoder collapsed it to a constant (see H5), so the
tree-based selector reported zero importance. A fairness-critical exclusion happened
by accident.

There is no fairness testing anywhere in the project: no group-wise performance, no
disparate-impact measurement, no proxy analysis.

### H5 — The target encoder silently discards most categorical information

Configured with `min_samples_leaf=5000, smoothing=1`, the blending weight is
`1/(1+exp(-(count-5000)/1))`, which underflows to 0 for any category with fewer
than roughly 5,000 rows. Every such category is pinned to the global prior. The run
emits `RuntimeWarning: overflow encountered in exp` on every execution.

| column | distinct categories | distinct encoded values | encoded std |
|---|---|---|---|
| pincode | 838 | **1** (constant) | 0.000000 |
| industry | 8,986 | **3** | 0.047724 |
| role | 46 | 5 | 0.005430 |
| married | 2 | 3 | 0.000315 |
| tier_of_employment | 7 | 6 | 0.019264 |

Only 2 of 8,986 `industry` categories and **0 of 838** `pincode` categories clear
the threshold. The notebook's conclusion that "pincode is of 0 importance in Random
Forest and Decision Tree" is therefore an artifact of the encoder configuration, not
a finding about pincode.

---

## 5. Medium-severity findings

| # | Finding | Detail |
|---|---|---|
| M1 | **No inference path** | `engine.py` persisted only `model.txt`. The fitted target encoder, the feature list and the label definition were discarded, so the saved model could not score raw data. Fixed. |
| M2 | **Validation set does triple duty** | The same `val` split drives early stopping in all 50 trials, the Hyperopt objective, and final model selection. Val AUC is optimistically biased. No cross-validation is used. |
| M3 | **Hold-out is not held out** | `hold_out_auc` is computed inside the objective function and written to `hyperopt_results.csv` for every trial, so out-of-time performance is visible during model selection. |
| M4 | **Documented objective ≠ implemented objective** | Notebook states `Score = (Train AUC − Val AUC + 1)/(Val AUC)`. Code computes `(abs(train_auc − val_auc) + 1)/((1 + val_auc)²)`. |
| M5 | **Calibration is assumed, never checked** | `pos_bagging_fraction` (0.659) and `neg_bagging_fraction` (0.544) are tuned independently, which shifts the effective class prior, and no calibration step or check exists. Measured here, the damage is smaller than that setup suggests: the model over-predicts by 4–7% relative (val mean predicted 0.1014 vs observed 0.0977, ratio 1.038; ECE 0.0097, max bin error 0.029). Usable as a ranking score; it should not be quoted as a PD without a calibration step and a stated tolerance. |
| M6 | **Thin temporal validation** | Five months total: three train, one validation, one hold-out. No seasonal coverage; one month of out-of-time data is weak evidence of stability. |
| M7 | **Users span the splits** | 641 training users reappear in validation and 646 in hold-out (~2.4% of each); 9,975 duplicate `User_id` values overall. Not deduplicated or acknowledged. |
| M8 | **Errors are swallowed** | Every pipeline function wraps its body in `try/except Exception: print(e)` without re-raising, so a failure returns `None` and surfaces later as an unrelated `AttributeError`. No logging framework. |
| M9 | **Feature selection was non-deterministic** | The RandomForest and DecisionTree in `select_features` had no `random_state`, so the dropped-feature set — and therefore the model's feature list — could change between runs. Fixed. |
| M10 | **Two derived features are degenerate** | `total_payement_per_loan` is 0 for 99.60% of rows and `delinq_2yrs_ratio` for 99.93%, because both divide by `number_of_loans` (zero for 99.59% of rows) and the resulting inf/NaN is filled with 0. Univariate Gini 0.0012 and 0.0004. |
| M11 | **Roll-rate table is misread in the write-up** | The notebook states "~25.6% customer paid back after crossing dpd30". 25.59% is the share of *all* customers who **reached** DPD 30 (22,073/86,250); the share who reached DPD 30 and then recovered is 63.31% (13,974/22,073). The label conclusion is unaffected and correct — it rests on the second statistic, that 97.86% of customers crossing DPD 60 go on to DPD 90 (7,926/8,099) — but the stated evidence does not say what the text claims. |

---

## 6. Low-severity findings

| # | Finding |
|---|---|
| L1 | `requirements.txt` was not installable. `pandas==1.3.0` and `shap==0.40.0` have no Python 3.10 wheel; `hyperopt==0.2.7` additionally needs `setuptools<81` for `pkg_resources`. No Python version was declared anywhere. |
| L2 | `cutoff_score` divides cumulative defaulters by `pred.index`, which starts at 0, instead of `index + 1`. Verified immaterial in effect (identical cutoff on a 20k-row simulation) but incorrect, and it is the function that sets the credit policy threshold. |
| L3 | `lib/utils.py` duplicates ~200 lines of `ml_pipeline` logic with divergent error handling — two copies of `process_data`, `create_label`, `derived_features`, `categorical_encoding` and the importance helpers. |
| L4 | The same 23 MB dataset is stored three times (`data/`, `notebooks/`, `modular_code/input/`) with no checksum or data-version record. |
| L5 | Not a git repository. No version control, CI, linting or pre-commit configuration. |
| L6 | No tests existed. |
| L7 | The misspelling `total_payement` is baked into the schema and every downstream reference. |
| L8 | `n_jobs=25` was hardcoded in the RandomForest parameters regardless of host CPU count. |
| L9 | No model card, no documented intended use or limitations, no monitoring or retraining plan. |

---

## 7. What was good

Worth stating plainly, because the analytical design is stronger than the defect
list suggests:

* **The label was derived, not assumed.** DPD roll-rate analysis showed that
  recovery collapses after DPD 60 — 63.3% of customers who reach DPD 30 recover
  before DPD 60, but only 2.1% of those reaching DPD 60 avoid DPD 90 — and window
  roll-rate analysis showed 99.4% of defaults occur within the first three EMIs
  (80.8% at EMI 1, 13.4% at EMI 2, 5.2% at EMI 3). The resulting
  "DPD 60+ within 3 EMIs" definition is properly justified. (See M11 for a
  misstatement in the accompanying commentary — the conclusion is right, one of the
  supporting sentences is not.)
* **The split is time-based, not random** — the correct choice for a credit model,
  and it gives a genuine out-of-time hold-out.
* **The target encoder is fitted on train only** and applied to validation and
  hold-out, avoiding the most common target-encoding leak. Verified by test.
* **The tuning objective explicitly penalises the train/validation gap** rather than
  chasing validation AUC alone.
* **Gender was excluded on principle**, with the reasoning recorded.
* **The modular structure is sound** — clean separation of loading, processing,
  training, and a single entry point.

---

## 8. SDLC and engineering assessment

| Area | Status | Notes |
|---|---|---|
| Version control | ✗ → ✓ | git repository initialised with a documented initial commit |
| Environment reproducibility | ~ → ✓ | Pins existed but were not installable; Python 3.10 declared, verified |
| Dependency lock | ~ | Direct dependencies pinned; transitive ones not |
| Configuration management | ✗ → ✓ | `ml_pipeline/config.py`; file / env / CLI precedence, validated |
| Modularity | ✓ | Good stage separation |
| Code duplication | ✗ → ✓ | The duplicate library was deleted in the restructure; a test fails if a second copy reappears |
| Error handling | ✗ → ✓ | Blanket `try/except` removed; failures propagate with context |
| Logging | ✗ → ✓ | `ml_pipeline/logging_utils.py`; levelled, to stderr and a run log |
| Automated tests | ✗ → ✓ | None existed; 59 regression tests added |
| CI/CD | ✗ | None |
| Experiment tracking | ~ | `hyperopt_results.csv` only; no run metadata, seeds or environment captured |
| Model artefact management | ✗ → ✓ | Only the booster was saved; encoder, feature list and manifest now persisted |
| Inference / serving path | ✗ → ✓ | None existed; `predict.py` added |
| Evaluation in the pipeline | ✗ → ✓ | Evaluation lived only in the notebook; `evaluate.py` + `ml_pipeline/evaluation.py` added |
| Data validation | ✗ | No schema checks, no null/range/cardinality assertions on input |
| Data versioning | ✗ | Three copies, no checksums |
| Documentation | ~ | Good conceptual docs; no model card, no limitations, no intended-use statement |
| Model governance | ✗ | No sign-off, no challenger model, no fairness testing, no monitoring plan |
| Reproducibility of results | ~ → ✓ | Unseeded feature selection and position-dependent parsing; both fixed |

---

## 9. Changes made

All original sources are preserved in `archive/original/`, and the
faithful reproduction run is archived in `output/baseline_faithful_run/`.

**Defect repairs**

| File | Change |
|---|---|
| `engine.py`, `ml_pipeline/training.py` | added `'label'` to `id_cols` (C2) |
| `engine.py` | read hyperopt results from `output/` (C3) |
| `ml_pipeline/utils.py` | `low_memory=False` for type-stable parsing (H2) |
| `ml_pipeline/processing.py` | `random_state` on the selection models (M9) |
| `ml_pipeline/evaluation.py` | class-1 SHAP values, not class-0 (H3) |

**Compatibility repairs** (E1) — the original code targets APIs removed in the
installed stack: `lgb.train(early_stopping_rounds=, verbose_eval=, evals_result=)`
→ callbacks; `DataFrame.append` → `pd.concat`; `TargetEncoder.feature_names` →
`get_feature_names_in()`; positional `iloc[:, :19]` parameter slice → selection by
name.

**Additions**

| File | Purpose |
|---|---|
| `ml_pipeline/evaluation.py` | KS, Gini, PR-AUC, PSI, calibration, decile and approval-curve helpers |
| `evaluate.py` | evaluation entry point writing a full metric pack to `output/` |
| `predict.py` | batch scoring entry point (the missing inference path) |
| `analysis/diagnostics.py` | the six validation checks behind sections 3–5 |
| `analysis/encoder_check.py` | target-encoder smoothing verification |
| `tests/test_pipeline.py` | 17 regression tests |
| `output/run_manifest.json` | run metadata: versions, seeds, split rows, default rates, chosen parameters |

---

## 9a. Remediation status

Findings are kept as originally written; this table records what has since been
closed. Every "closed" row is covered by a test in `tests/`.

| # | Finding | Status |
|---|---|---|
| C1 | Target leakage from post-origination fields | **Open** — needs the data owner; documented in the notebook and `DATA.md` |
| C2 | Target in the feature matrix | Closed — `config.validate()` and `train_lgb` refuse to run without `label` excluded |
| C3 | Wrong hyperopt results path | Closed |
| H1 | `"0"` / `"0.0"` placeholder artifact | **Open** — source data issue; documented |
| H2 | Position-dependent CSV parsing | Closed — `low_memory=False`, enforced by `test_read_is_type_stable` |
| H3 | Sign-inverted SHAP | Closed — class-1 values in `utils.shap_importance` and `evaluate.py` |
| H4 | Protected attributes retained | **Open** — a policy decision, not a code change |
| H5 | Encoder collapses categories | **Open** as a modelling choice; now logs a warning when a column encodes to a constant |
| M1 | No inference path | Closed — `predict.py` + persisted artefacts |
| M2, M3 | Validation reused; hold-out visible during tuning | **Open** — needs cross-validation; noted in the notebook |
| M4 | Documented objective ≠ code | Closed |
| M5 | Calibration unchecked | Closed as a *check* (`evaluate.py` reports Brier/ECE); calibration step itself still open |
| M6, M7 | Thin validation window; users span splits | **Open** — data/design issues |
| M8 | Errors swallowed | Closed — blanket `except` blocks removed, errors propagate |
| M9 | Non-deterministic feature selection | Closed — seeded, warns if a seed is absent |
| M10 | Degenerate derived features | Closed as a *check* — logs a warning when a derived feature is >95% zeros |
| M11 | Roll-rate misread | Closed — corrected, and `dpd_roll_rate` now returns roll/recovery columns |
| L1 | requirements.txt not installable | Closed |
| L2 | `cutoff_score` off-by-one | Closed |
| L3 | `lib/utils.py` duplication | Closed — the duplicate was deleted in the restructure; `test_analysis_library_has_exactly_one_copy` fails if it returns |
| L4 | Three copies of the dataset | Closed — the two duplicates were deleted; a pre-commit hook and a CI step fail if more than one is ever tracked |
| L5 | No version control | Closed — git repository initialised |
| L6 | No tests | Closed — 59 tests |
| L7 | `total_payement` misspelling | **Open** — renaming would break the input schema contract |
| L8 | Hardcoded `n_jobs=25` | Closed — `n_jobs=-1`, configurable |
| L9 | No model card / monitoring plan | **Open** |
| 13–22 | Analysis-helper defects | Closed — see `tests/test_analysis_utils.py` |
| 30, 31 | No logging / config / CLI | Closed — `ml_pipeline/config.py`, `ml_pipeline/logging_utils.py`, `engine.py --help` |

Everything still open is either a data-ownership question (C1, H1), a policy
decision (H4, L7, L9), or modelling work that changes results and should be done
deliberately (M2, M3, M5, M6, M7, H5).

---

## 10. Recommended next steps

**Before anything else**

1. Confirm with the data owner exactly when `total_payement`, `received_principal`,
   `interest_received` and `number_of_loans` are observed relative to origination.
   Everything below depends on the answer.
2. Rebuild the feature set from application-time data only. Treat **val AUC 0.7126 /
   Gini 0.425** as the honest baseline.
3. Treat `industry` and `work_experience` as unusable until the placeholder encoding
   is resolved at source; do not let `"0"` vs `"0.0"` reach a model.

**Modelling**

4. Replace the single validation split with time-series cross-validation; keep the
   hold-out genuinely untouched and stop logging its AUC during tuning.
5. Re-tune the target encoder (`min_samples_leaf` in the 20–100 range) or switch to
   out-of-fold encoding, then re-run feature selection — pincode and industry deserve
   a fair evaluation.
6. Add a calibration step (isotonic or Platt on the validation split) so the score
   can be read as a probability of default.
7. Add a simple, explainable challenger — logistic regression on WOE-binned features
   is the credit-risk standard — and compare.

**Governance**

8. Decide and document the position on `married`, `dependents` and `home_type`, and
   add disparate-impact testing across available demographic dimensions.
9. Write a model card: intended use, training window, feature definitions with
   observation timing, known limitations, monitoring plan.
10. Put the project under version control and add CI running the test suite.

---

## 11. The remediated model (v2)

Built by `engine_v2.py`, artefacts in `output_v2/`, documented in `MODEL_CARD.md`.

### 11.1 What changed

| Change | Finding closed |
|---|---|
| Features filtered through an enforced governance policy (`ml_pipeline/governance.py`) | C1, H4 |
| `industry` / `work_experience` placeholder zeros collapsed to missing | H1 |
| Single validation month replaced by expanding-window walk-forward folds | M2 |
| Hold-out month scored exactly once, after all selection | M3 |
| Repeat customers removed from later periods | M7 |
| Isotonic calibration fitted out-of-sample | M5 |
| WOE + logistic scorecard fitted as a challenger | — |

### 11.2 Result on the untouched hold-out (202205, n = 26,607, 8.71% bad)

| model | AUC | Gini | KS | PR-AUC |
|---|---|---|---|---|
| **Champion — LightGBM, calibrated** | **0.6457** | **0.2914** | **0.2141** | 0.1526 |
| Challenger — WOE + logistic (4 features) | 0.6348 | 0.2696 | 0.1999 | 0.1430 |

Walk-forward CV gave **0.6423 ± 0.0054** (min 0.6369). The hold-out landed at
0.6457, inside that band — the model generalises, and for the first time in this
project the validation estimate is honest.

The champion beats the interpretable challenger by 0.022 Gini (≈8% relative).
That is a real but modest margin, and it is the number that has to justify
choosing 139 boosted trees over a six-line scorecard.

### 11.3 Calibration

The raw model was badly miscalibrated — it predicted a mean PD of 5.10% against
an observed 8.71%, understating risk by 41%. This is the `pos_bagging_fraction`
(0.303) / `neg_bagging_fraction` (0.894) asymmetry that M5 warned about, and it
is far worse here than in the original leaky model.

| | raw | calibrated |
|---|---|---|
| Mean predicted PD | 0.0510 | 0.0938 |
| Observed rate | 0.0871 | 0.0871 |
| Brier | 0.08017 | 0.07757 |
| Expected calibration error | 0.03695 | **0.00792** |
| AUC | 0.6462 | 0.6457 |

Calibration cut the error 4.7-fold and cost 0.0005 AUC — isotonic regression
maps score ranges onto flat steps, so tied scores lose a little ordering. That
is immaterial, and the output can now be read as a probability of default.

### 11.4 What each governance decision cost

From `analysis/governance_cost.py` — one change per step, everything else fixed:

| step | features | val Gini | change |
|---|---|---|---|
| 1. as originally built (leakage + protected) | 20 | 0.9222 | — |
| 2. − post-origination fields (C1) | 15 | 0.3273 | **−0.5949** |
| 3. − protected: `married`, `pincode` (H4) | 13 | 0.3276 | +0.0003 |
| 4. − restricted alternative data | 10 | 0.3285 | +0.0010 |
| 5. − placeholder artifact (H1) | 10 | 0.2929 | −0.0357 |
| 6. − repeat customers from evaluation (M7) | 10 | 0.3051 | +0.0122 |

**96% of everything lost was the leakage.** Two findings matter for the business
case; the rest were free:

- **Fair-lending compliance cost nothing.** Removing `married`, `pincode`,
  `dependents`, `has_social_profile` and `is_verified` *improved* Gini by
  0.0013. There was never a performance argument for keeping them.
- **The placeholder artifact was the only other real cost** at 0.036 Gini — the
  price of not scoring which spelling of a missing value a record happened to
  carry. Defensible in front of a regulator; the alternative is not.

About **a third of the original apparent power was real and permissible.**

### 11.5 Where the remaining signal lives

Information value on the training window:

| feature | IV | strength |
|---|---|---|
| tier_of_employment | 0.1075 | medium |
| work_experience | 0.0550 | weak |
| total_income | 0.0504 | weak |
| employment_type | 0.0259 | weak |
| home_type | 0.0139 | useless |
| role | 0.0112 | useless |
| everything else (7 features) | < 0.002 | useless |

Only four features carry anything, and none reaches "strong". The challenger
uses exactly those four, and its weight-of-evidence bins are monotone and
sensible — employer tier orders risk cleanly from A (best) to E (worst):

| tier_of_employment | WOE |
|---|---|
| E | +0.9260 |
| D | +0.5056 |
| C | +0.0991 |
| *(missing)* | −0.0743 |
| B | −0.0985 |
| A | −0.7878 |

This is the honest shape of the problem: **this dataset supports a weak
employment-quality model and little else.** A Gini of 0.29 is usable as one
input to a credit policy — the top decile carries a 19.4% bad rate against an
8.7% base, a lift of 2.2 — but it is not a standalone underwriting model, and
the thin feature set is the reason. Adding bureau data would do far more than
any further modelling on what is here.

### 11.6 Still open after v2

C1 and H1 are *contained* rather than resolved — the model no longer uses the
affected fields, but the data issues remain and need the data owner. H4 is
closed in code; the classifications still need compliance sign-off for the
actual jurisdiction. M6 (five months of data) cannot be fixed without more data.
