# Project Log — Credit Default Risk Model

A complete record of the work: what was found, what was built, what was decided,
and what is still open.

**Date:** 2026-08-09 · **Commits:** 6 · **Tests:** 237 · **Status:** v2 model is decision-capable, not approved

---

## 1. Summary

The project arrived as a LightGBM credit default model reporting **0.96 AUC**. It
was run end to end, reproduced exactly, and reviewed.

The headline finding: **most of that 0.96 came from data that does not exist when
a loan application is scored.** A model rebuilt on application-time features
only, under an enforced data governance policy, scores **Gini 0.2914** on an
untouched hold-out. That is the real number.

Around that, the project gained the things a lending model needs before anyone
can use it: a governance policy enforced in code, walk-forward validation,
calibrated probabilities, an interpretable challenger, adverse action reason
codes, a cutoff policy, a data contract, CI, and production monitoring.

| | Original | Remediated (v2) |
|---|---|---|
| Hold-out AUC | 0.9636 | 0.6457 |
| Hold-out Gini | 0.9272 | **0.2914** |
| Features | 19 | 13 (governance-filtered) |
| Validation | single month | walk-forward, hold-out scored once |
| Output | uncalibrated | isotonic-calibrated |
| Can it be served? | no | yes |
| Can a decline be explained? | no | yes |
| Can it be monitored? | no | yes |
| Tests | 0 | 237 |

---

## 2. Starting point

```
data/data/credit_risk_data.csv            23 MB, 143,727 applications
notebooks/notebooks/model.ipynb           84-cell exploratory notebook
notebooks/notebooks/utils.py              572-line analysis library
modular_code/modular_code/                engine.py + ml_pipeline/ + lib/ (a copy of notebooks/)
Solution Methodology.pdf
```

Not a git repository. No tests, no config, no logging, no CI, no inference path,
no model documentation. Every top-level folder was doubled (`data/data/`,
`notebooks/notebooks/`, `modular_code/modular_code/`) and the same 23 MB dataset
was stored three times.

---

## 3. Phases of work

### Phase 1 — Environment and end-to-end reproduction

Built an isolated **Python 3.10.11** virtual environment using the project's own
pinned dependencies. Two pins could not be installed on 3.10 and were moved to
the nearest release publishing a cp310 wheel:

| Pin | Problem | Resolution |
|---|---|---|
| `pandas==1.3.0` | no cp310 wheel (1.3.4 is the first) | `1.3.5` |
| `shap==0.40.0` | no cp310 wheel | `0.41.0` |
| `hyperopt==0.2.7` | imports `pkg_resources`, removed in setuptools 81 | pinned `setuptools<81` |

Three defects had to be repaired before the pipeline would run at all (C2, C3,
E1 below). After that, the 50-trial run reproduced the original notebook **to
every digit**:

| | Original notebook | This run |
|---|---|---|
| Best hyperopt loss | 0.26055619307290495 | 0.26055619307290495 |
| Best trial index | 21 | 21 |
| Train AUC | 0.9709666608309787 | 0.970967 |
| Val AUC | 0.9649508759429978 | 0.964951 |
| Hold-out AUC | 0.9635950364843274 | 0.963595 |
| n_estimators | 3,041 | 3,041 |

That exactness matters: it means every finding below is about the project, not
about the port.

### Phase 2 — Independent validation

31 findings, from 4 critical to 9 low. The full register is in section 6; the
decisive one was proven by experiment rather than argued:

**The counterfactual.** Identical hyperparameters, only the feature set differs:

| Model | Features | Val AUC | Val Gini |
|---|---|---|---|
| As built | 19 | 0.9701 | 0.9401 |
| Application-time only | 14 | 0.7126 | 0.4252 |

Removing five post-origination fields costs 0.515 Gini — **85% of the model's
apparent power**.

**The proof they are post-origination.** The data dictionary describes
`total_payement` / `received_principal` / `interest_received` as activity over
the last 2 years, i.e. prior loans. But 143,136 rows (99.59%) have
`number_of_loans == 0` while 143,105 of those show `total_payement > 0`. A
borrower with zero prior loans cannot have prior repayment history.

### Phase 3 — Bug remediation

All 20 remaining code and documentation defects fixed, each with a regression
test. Notable ones:

- **SHAP sign inversion** — every explanation in the notebook pointed the wrong
  way. `shap_values[0]` is the non-default class; verified `sv[0] == -sv[1]`.
  For `received_principal`, the plot said "more principal repaid → more risk"
  when the truth is the opposite.
- **`class_rate` crashed** unless the first dataset was literally named "Train".
- **Metric helpers hardcoded exactly three datasets**, silently.
- Plus `sns.distplot` removal, private pandas API use, a slice assignment, a
  figure leak, an off-by-one in the cutoff function, and three documentation
  errors including a misread of the project's own roll-rate table.

Also added: config management with validation, structured logging, removal of
the blanket `try/except: print(e)` blocks, git initialisation, and a data
checksum register.

### Phase 4 — The governed v2 model

Following the instruction that **banking data must follow data governance**,
`ml_pipeline/governance.py` classifies every column with a written rationale and
enforces it — `enforce_policy()` raises rather than warns, and an *unclassified*
column also blocks the run.

| Excluded | Class | Reason |
|---|---|---|
| `gender` | PROHIBITED | ECOA/Reg B 1002.6(b)(9) |
| `married` | PROHIBITED | Marital status — Reg B 1002.6(b)(8) |
| `pincode` | PROHIBITED | Geographic redlining proxy |
| 5 repayment fields | LEAKAGE | Not observable at application |

Plus: placeholder artifact collapsed, walk-forward CV replacing the single
validation month, hold-out scored exactly once, repeat customers removed,
isotonic calibration, and a WOE + logistic challenger.

**Result on the untouched hold-out (202205, n = 26,607, 8.71% bad):**

| Model | AUC | Gini | KS |
|---|---|---|---|
| Champion — LightGBM, calibrated | **0.6457** | **0.2914** | 0.2141 |
| Challenger — WOE + logistic (4 features) | 0.6348 | 0.2696 | 0.1999 |

Walk-forward CV gave 0.6423 ± 0.0054; the hold-out landed inside that band.

**What each governance decision cost** (`analysis/governance_cost.py`, one
change per step):

| Step | Val Gini | Change |
|---|---|---|
| 1. As originally built | 0.9222 | — |
| 2. − post-origination fields | 0.3273 | **−0.5949** |
| 3. − `married`, `pincode` | 0.3276 | **+0.0003** |
| 4. − restricted alternative data | 0.3285 | **+0.0010** |
| 5. − placeholder artifact | 0.2929 | −0.0357 |
| 6. − repeat customers | 0.3051 | +0.0122 |

**96% of the loss was the leakage. Fair-lending compliance cost nothing** —
removing every protected and restricted attribute *improved* Gini by 0.0013.

### Phase 5 — The decisioning layer

A calibrated PD is not yet a credit decision.

- **Score points** — `score = offset + factor·ln((1−PD)/PD)`, PDO 20, 600 at 50:1 odds
- **Adverse action reason codes** from class-1 SHAP, in applicant-facing language,
  referring only to the applicant's own record. Legally required for declines and
  previously absent. A test enforces that every model feature has mapped text.
- **Cutoff policy** on the hold-out
- **`predict_v2.py`** — closes the serving gap; **calibration is unconditional**,
  because the raw booster understates risk by 41%

**At a 6% book bad-rate appetite:**

| | |
|---|---|
| Cutoff | 555 points (PD 0.0861) |
| Approval rate | 55% |
| Book bad rate | 5.66% vs 8.71% unscreened — a **35% reduction** |
| Defaults avoided | 1,489 of 2,318 (**64.2%**) |
| Good customers declined | 10,484 |

Across 56,181 declines, reason citations spread sensibly — the most frequent
appears in 19%, so no single feature drives the cutoff.

### Phase 6 — Data contract and CI

Every defect found in Phase 2 was in the file from day one. None needed a model
to detect — they needed something to look.

`ml_pipeline/data_contract.py` is declarative and dependency-free (pandera would
break the pinned 2022-era stack). Beyond schema, nulls, cardinality and ranges,
it carries four tripwires drawn from the findings.

**On a cold read of the shipped file it independently rediscovers:**

| Finding | Originally found by | Contract verdict |
|---|---|---|
| H1 placeholder spelling | manual investigation | **ERROR** |
| H2 mixed dtypes by chunk | manual investigation | **ERROR** |
| M7 repeat customers | manual investigation | WARNING |
| M10 degenerate column | manual investigation | WARNING |
| C1 leakage | counterfactual experiment | flagged by the screen |

It also found one the manual review missed: `home_type` holds 42 rows valued
literally `'none'`, colliding with the missing-value vocabulary.

CI runs ruff, the test suite, a dataset checksum verification, smoke trains of
both pipelines, a scoring run, monitoring, and a guard that fails the build if
any scored output is tracked in git.

### Phase 7 — Production monitoring

`MODEL_CARD.md` §8 specified a monitoring plan; this implements it.

**Checks are split by when the evidence exists.** A first-payment-default label
needs three EMIs, so a freshly scored cohort has no outcome for ~3 months.

| Immediate | After 3 EMIs |
|---|---|
| Score PSI, per-feature PSI, band volumes, mean PD shift, placeholder-rate change | Calibration ratio, discrimination Gini, decile bad-rate drift |

Outcome checks report `SKIPPED` **with the reason** rather than being omitted, so
a report can never look green because half of it silently did not run.

The placeholder check watches the **raw** field, not the encoded one: if the
upstream export changes how it writes "not captured", the encoder maps the
unfamiliar category to the prior and PSI on the encoded value barely moves.

**Latest run — 202205, n = 28,727: all checks OK.** Score PSI 0.0001,
calibration ratio 1.042, Gini 0.2745, largest decile drift 0.0097.

---

## 4. Model performance journey

| Stage | Val/Hold-out Gini | What changed |
|---|---|---|
| As reported | 0.9272 (hold-out) | — |
| Reproduced exactly | 0.9272 | environment rebuilt, 3 blocking defects fixed |
| − leakage | 0.3273 | post-origination fields removed |
| − protected attributes | 0.3276 | `married`, `pincode` |
| − restricted data | 0.3285 | alternative data |
| − placeholder artifact | 0.2929 | provenance signal removed |
| **v2, tuned + calibrated** | **0.2914 (hold-out)** | walk-forward CV, dedup, calibration |

About **a third of the original apparent power was real and permissible.**

Where the remaining signal lives (information value, training window):

| Feature | IV | Strength |
|---|---|---|
| tier_of_employment | 0.1075 | medium |
| work_experience | 0.0550 | weak |
| total_income | 0.0504 | weak |
| employment_type | 0.0259 | weak |
| everything else (9 features) | < 0.02 | useless |

**This dataset supports a weak employment-quality model and little else.**
Bureau data would do more than any further modelling on what is here.

---

## 5. What was built

### New modules — `ml_pipeline/`

| File | Lines | Purpose |
|---|---|---|
| `data_contract.py` | 503 | Input schema, quality tripwires, leakage screen |
| `monitoring.py` | 441 | Baseline snapshot, drift checks, report |
| `governance.py` | 276 | Field classification, enforced feature policy |
| `scorecard.py` | 237 | Score points, reason codes, cutoff policy |
| `woe.py` | 189 | Weight of evidence, information value, challenger |
| `config.py` | 167 | Configuration with validation |
| `evaluation.py` | 160 | KS, Gini, PR-AUC, PSI, calibration, deciles |
| `calibration.py` | 116 | Isotonic / Platt calibration |
| `validation.py` | 106 | Walk-forward folds, customer dedup |
| `logging_utils.py` | 43 | Logging setup |

### Entry points

| File | Purpose |
|---|---|
| `engine.py` | Original model (reproduction) |
| `engine_v2.py` | Governed model |
| `evaluate.py` | Evaluation pack |
| `predict.py` / `predict_v2.py` | Batch scoring |
| `validate_data.py` | Data contract gate (exit 1 on failure) |
| `monitor.py` | Baseline and monthly checks (exit 1 on alert) |

### Analysis — `analysis/`

`diagnostics.py`, `encoder_check.py`, `encoder_sweep.py`, `governance_cost.py`,
`cutoff_policy.py` — the experiments behind the findings, re-runnable.

### Documentation

| File | Purpose |
|---|---|
| `MODEL_REVIEW.md` | Full validation findings and remediation status |
| `MODEL_CARD.md` | Intended use, limitations, monitoring plan, approval block |
| `DATA.md` | Canonical dataset, SHA-256, known quality issues |
| `readme.md` | Rewritten: two pipelines, config, scoring, monitoring |
| `PROJECT_LOG.md` | This document |

### Infrastructure

`.github/workflows/ci.yml`, `pyproject.toml` (ruff + pytest),
`.pre-commit-config.yaml`, `requirements-dev.txt`, `.gitignore`

### Final structure

The delivered tree had every top-level folder doubled and the dataset stored
three times. It was flattened to a conventional layout:

```
data/            the dataset, tracked exactly once
ml_pipeline/     importable pipeline stages
analysis/        independent validation experiments
tests/           237 regression tests
notebooks/       exploratory notebook and its analysis library
docs/            review, model card, data register, project log
archive/original/  the sources as delivered, kept as review evidence
output/ output_v2/  generated artefacts
```

Entry points sit at the repository root. The working tree went from **141.5 MB
to 41.5 MB** — the removals were two duplicate dataset copies (45.6 MB), a
regenerable intermediate (31.3 MB), orphaned model binaries (12.6 MB), a
duplicated notebook and library, caches, and two scored output files totalling
21.5 MB that were keyed by `User_id` and should never have been sitting on disk.

---

## 6. Findings register

**Critical**

| # | Finding | Status |
|---|---|---|
| C1 | Target leakage — model trained on post-origination repayment | **Open** (contained: excluded by policy; needs data owner) |
| C2 | `engine.py` trained on the label itself | Closed |
| C3 | `engine.py` crashed before completing | Closed |

**High**

| # | Finding | Status |
|---|---|---|
| H1 | Placeholder written two ways, spelling predicts default | **Open at source** (collapsed in pipeline; contract errors on it) |
| H2 | Feature values depended on CSV chunk position | Closed |
| H3 | SHAP explanations sign-inverted | Closed |
| H4 | Protected attributes retained while gender dropped | Closed in code; **needs compliance sign-off** |
| H5 | Encoder collapses most categorical information | Investigated — aggressive smoothing was correct; now warns |

**Medium**

| # | Finding | Status |
|---|---|---|
| M1 | No inference path | Closed |
| M2 | Validation set did triple duty | Closed (walk-forward CV) |
| M3 | Hold-out visible during tuning | Closed |
| M4 | Documented objective ≠ implemented | Closed |
| M5 | Calibration assumed, never checked | Closed |
| M6 | Thin temporal validation (5 months) | **Open** — needs more data |
| M7 | Customers span the splits | Closed |
| M8 | Errors swallowed by blanket `except` | Closed |
| M9 | Feature selection non-deterministic | Closed |
| M10 | Two derived features degenerate | Closed (warns) |
| M11 | Roll-rate table misread in the write-up | Closed |

**Low** — L1 requirements, L2 cutoff off-by-one, L3 duplication, L4 data copies,
L5 version control, L6 tests, L8 hardcoded `n_jobs`: all closed.
L7 (`total_payement` misspelling) and L9 (model card) — L9 closed, L7 open by choice.

**Analysis-helper defects** (#12–#22): all closed, test-covered.
**Repo hygiene** (#26–#31): closed.

---

## 7. Test coverage

| File | Tests | Covers |
|---|---|---|
| `test_data_contract.py` | 41 | Schema, tripwires, leakage screen, integration with the real file |
| `test_governance.py` | 38 | Classification, policy enforcement, PII, lineage |
| `test_scorecard.py` | 35 | Points scaling, reason codes, cutoff policy |
| `test_monitoring.py` | 32 | Baseline, drift injection, maturity handling |
| `test_validation_calibration_woe.py` | 32 | Walk-forward folds, calibration, WOE/IV |
| `test_config_and_errors.py` | 25 | Config precedence, validation, error propagation |
| `test_analysis_utils.py` | 17 | The notebook library's 11 fixed defects |
| `test_pipeline.py` | 17 | Labelling, splitting, encoding, leakage guards |
| **Total** | **237** | |

Most monitoring and contract tests **inject a specific failure** and assert the
right check fires while the others stay quiet. A test suite that only proves
things work on clean data proves very little.

---

## 8. Commits

| Hash | Change | Files | Insertions |
|---|---|---|---|
| `076f5d5` | Initial commit: model with validation review and fixes | 56 | 158,537 |
| `6c42f89` | Remediation status and config/CLI interface | 2 | 69 |
| `b99d108` | Governed v2 model — Gini 0.2914 | 23 | 2,574 |
| `8c309c0` | Decisioning layer — points, reason codes, cutoffs, serving | 8 | 910 |
| `1934f9f` | Data contract and CI | 36 | 2,826 |
| `959b6d8` | Production monitoring | 9 | 1,670 |

93 files tracked. The initial commit's line count is dominated by the 23 MB
dataset; roughly **8,000 lines of code and documentation** were written after it.

---

## 9. Mistakes made and corrected

Recorded because a validation exercise that hides its own errors is not credible.

| Mistake | How it surfaced | Resolution |
|---|---|---|
| First `id_cols` notebook fix would have re-created the C2 leak downstream | Verification pass before committing | Replaced with an idempotent rebind |
| Committed `scores_v2.csv` — 143,727 rows keyed by `User_id` | Reviewing the staged file list | Removed from history, glob widened |
| Predicted the encoder was "the most likely source of lift" | The sweep disproved it | Reported the negative result |
| Told the user 0.7126 AUC was "the real starting line" | Cross-check against governance | It wasn't compliant either; corrected to 0.2914 |
| Baseline serialised both infinities as `null`, breaking bin monotonicity | First real monitoring run | Edges stored finite, opened at use time |
| `reason_codes()` raised `IndexError` when `top_n` > feature count | New test | Slots capped |
| Ranked the WOE scorecard by raw coefficient — put a noise feature first | New test | Ranked by `|coef| × std(WOE)` instead |
| Reported both pipelines had "failed" | Investigation | My own command: PowerShell `Select-Object -First` killed the process |
| Did not back up the two `utils.py` files before rewriting | Noticed at commit time | Disclosed; recoverable from the transcript |

---

## 10. What remains

**Blocked on the data owner**
1. Exact observation timing of the four repayment fields relative to origination (C1)
2. Placeholder encoding fixed at source, so `"0"` and `"0.0"` stop meaning the same thing (H1)

**Blocked on compliance / legal**
3. Sign-off on the `governance.py` classifications for the actual jurisdiction
4. Legal review of the adverse action reason wording before use in a live notice

**Available, unblocked**
5. MLflow experiment tracking and Optuna in place of Hyperopt (tooling)
6. Out-of-fold target encoding — the one untested alternative
7. PyYAML — `config.py` advertises YAML support that is not installed

**Needs a decision**
8. Physical dataset dedup — would break the notebooks unless they are repointed
9. `pincode` fair evaluation — only if compliance permits geography with disparate-impact testing, which this dataset cannot support

**Recommendation: do not deploy.** The model is decision-capable but the
limitations in `MODEL_CARD.md` §6 stand — Gini 0.29 on a thin feature set, five
months of data, no bureau data, no fairness testing possible without demographic
data, and no reject inference. Re-baseline any future work against **Gini 0.2914
on the 202205 hold-out**.
