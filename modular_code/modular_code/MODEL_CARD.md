# Model Card — Credit Default Risk Model v2

| | |
|---|---|
| **Model name** | Credit Default Risk (First Payment Default) v2 |
| **Version** | 2.0 |
| **Date** | 2026-08-09 |
| **Status** | **Candidate — not approved for production use** |
| **Owner** | *to be assigned* |
| **Reviewer / validator** | Independent review recorded in `MODEL_REVIEW.md` |
| **Artefacts** | `output_v2/` |
| **Reproduce with** | `python engine_v2.py --max-evals 40 --output-dir output_v2` |

> This card records what the model is and what it is not. It is a technical
> document produced alongside the model; it is **not** a compliance sign-off.
> The classifications in `ml_pipeline/governance.py` and the limitations below
> require review by the institution's compliance and model risk functions before
> any lending decision is made with this model.

---

## 1. Intended use

**Purpose.** Rank-order loan applicants by the probability of *first payment
default* — reaching 60+ days past due within the first three EMIs.

**Intended users.** Credit risk analysts and the credit policy function.

**Intended decision.** One input among several to an application scoring policy,
alongside bureau data, affordability checks and policy rules.

### Out of scope

| Not for | Why |
|---|---|
| Standalone automated approve/decline | Gini 0.29 on a thin feature set. It cannot carry a decision alone. |
| Pricing or expected-loss calculation | It estimates PD only. EL needs LGD and EAD, which this project does not model. |
| Collections, provisioning, or IFRS 9 / ECL | Different target, different observation window, different regulatory basis. |
| Behavioural scoring of existing customers | Trained on application-time data for new originations. |
| Any population outside the training geography or product | See §6. |

---

## 2. Training data

| | |
|---|---|
| Source | `input/credit_risk_data.csv` |
| SHA-256 | `2f816622a796f383cd48667607438ebbbfea4b07c919073e03497f3245b00688` |
| Rows | 143,727 applications |
| Period | 202201 – 202205 (five months) |
| Training window | 202201 – 202203 |
| Calibration window | 202204 |
| Hold-out | 202205 (26,607 rows after removing repeat customers) |
| Base default rate | 8.7% – 9.8% by month |

Known data-quality issues are catalogued in `DATA.md`.

---

## 3. Target definition

`label = 1` if maximum DPD across EMIs 1–3 is ≥ 60, else 0.

Justified by roll-rate analysis on the training window:

- 63.3% of customers reaching DPD 30 recover before DPD 60, but only **2.1%** of
  those reaching DPD 60 avoid DPD 90 — DPD 60 is where recovery collapses.
- **99.4%** of defaults occur within the first three EMIs (80.8% / 13.4% / 5.2%).

This is a **first-payment-default** target, which is narrower than the 90+ DPD
at 12 months used for most application scorecards. It behaves partly as an
early-fraud and affordability signal.

---

## 4. Features

13 features, all filtered through the enforced policy in
`ml_pipeline/governance.py`. Every column in the dataset carries a
classification and a written rationale.

**Used:** `employment_type`, `tier_of_employment`, `industry`, `role`,
`work_experience`, `total_income`, `home_type`, `delinq_2yrs`,
`number_of_loans`, `delinq_2yrs_ratio`, and — as RESTRICTED, pending sign-off —
`dependents`, `has_social_profile`, `is_verified`.

**Excluded, and why:**

| Field | Class | Reason |
|---|---|---|
| `gender` | PROHIBITED | Sex — prohibited basis, ECOA/Reg B 1002.6(b)(9) |
| `married` | PROHIBITED | Marital status — prohibited basis, Reg B 1002.6(b)(8) |
| `pincode` | PROHIBITED | Geographic redlining proxy; cannot be disparate-impact tested without demographic data |
| `total_payement` | LEAKAGE | Repayment on the loan being scored — unavailable at application |
| `received_principal` | LEAKAGE | as above |
| `interest_received` | LEAKAGE | as above |
| `interest_received_ratio` | LEAKAGE | derived from the above |
| `total_payement_per_loan` | LEAKAGE | derived from the above |

Removing the protected and restricted attributes **improved** Gini by 0.0013 —
there was no performance cost to compliance.

**Data quality treatment.** `industry` and `work_experience` are 83.9%
placeholder zeros recorded as both `"0"` and `"0.0"`, with materially different
default rates (9.75% vs 4.76%). Both are mapped to missing, because the
distinction reflects how the file was assembled, not the borrower. Cost: 0.036
Gini.

---

## 5. Performance

Hold-out 202205, scored once after all selection was complete.

| model | AUC | Gini | KS | PR-AUC |
|---|---|---|---|---|
| **Champion — LightGBM (139 trees), calibrated** | 0.6457 | 0.2914 | 0.2141 | 0.1526 |
| Challenger — WOE + logistic (4 features) | 0.6348 | 0.2696 | 0.1999 | 0.1430 |

Walk-forward CV: **0.6423 ± 0.0054** (min 0.6369). The hold-out sits inside that
band.

**Calibration** (isotonic, fitted on 202204):

| | raw | calibrated |
|---|---|---|
| Mean predicted PD | 0.0510 | 0.0938 |
| Observed | 0.0871 | 0.0871 |
| Expected calibration error | 0.03695 | 0.00792 |
| Brier | 0.08017 | 0.07757 |

Scores are calibrated probabilities. **The raw booster output is not** — it
understates risk by 41% — so `predict` must always apply the calibrator.

**Risk separation.** Top decile: 19.4% bad rate vs 8.7% base (lift 2.2),
capturing 22.3% of all defaults.

### Decisioning

Scores are expressed in points: `score = offset + factor · ln((1−PD)/PD)`, with
PDO 20 and 600 points at 50:1 good:bad odds. Higher score = lower risk.

Cutoff policy on the hold-out (`analysis/cutoff_policy.py`). At a 6% book bad
rate appetite:

| | |
|---|---|
| Cutoff | **555 points** (PD 0.0861) |
| Approval rate | 55% |
| Book bad rate | 5.66%, against 8.71% unscreened — a **35% reduction** |
| Defaults avoided | 1,489 of 2,318 (**64.2%**) |
| Good customers declined | 10,484 |

That last row is the cost of the cutoff and belongs in the decision: at Gini
0.29 the model turns away a large number of customers who would have paid. The
score range is also compressed (466–587 across the hold-out), which is what a
weak model looks like in points — cutoffs are sensitive to small score moves.

---

## 6. Limitations

1. **Weak discrimination.** Gini 0.29. Only four features have any information
   value, and none reaches "strong". This dataset supports an
   employment-quality model and little else.
2. **No bureau data.** The single biggest improvement available is external
   credit history, not further modelling of these columns.
3. **Five months of data.** No seasonal coverage; one month of out-of-time
   validation. Stability beyond this window is unknown.
4. **Unresolved data provenance.** The leakage (C1) and placeholder (H1) issues
   are *contained* by exclusion, not *resolved* at source.
5. **No fairness testing performed.** Prohibited attributes are excluded by
   construction, but no disparate-impact measurement was possible — the dataset
   holds no demographic data to test against. **Exclusion is not proof of
   fairness.**
6. **Ambiguous jurisdiction.** The data mixes Indian conventions (pincode, EMI)
   with US ones (mortgage/rent/own, a 35,000 loan cap). The governance defaults
   reference both regimes and need to be settled for the real one.
7. **No reject inference.** Trained only on approved and booked loans, so it is
   biased relative to the full applicant population.
8. **First-payment-default target**, not a conventional 90+/12-month definition.

---

## 7. Ethical and regulatory considerations

- **Adverse action reason codes are implemented** (`ml_pipeline/scorecard.py`,
  surfaced by `predict_v2.py`). For each declined applicant the model returns up
  to four principal reasons, derived from the class-1 SHAP decomposition of its
  log-odds output and expressed in applicant-facing language. Every model
  feature has mapped reason text — a test enforces this, because a decline
  explained as "other information" is not an acceptable reason. The wording
  refers only to the applicant's own record, never to a group.
  **The reason text still requires legal review before use in a live notice.**
- **Alternative data.** `has_social_profile` and `is_verified` are RESTRICTED:
  social-media presence can proxy for age and national origin. They contribute
  essentially nothing (IV 0.0003 and 0.0002) and should simply be dropped.
- **Human review.** Given Gini 0.29, no decline should rest on this score alone.
- **Personal data.** `User_id` is a direct identifier. Any scored output file
  inherits the confidentiality classification of the source data, is subject to
  the applicable retention policy, and must not be circulated without control.

---

## 8. Monitoring plan (required before deployment)

| Check | Frequency | Trigger |
|---|---|---|
| Score PSI vs the training distribution | Monthly | > 0.10 investigate, > 0.25 escalate |
| Feature-level PSI | Monthly | > 0.25 on any feature in use |
| Observed vs predicted default rate | Monthly, once the 3-EMI window matures | Ratio outside 0.8–1.25 |
| Discrimination (AUC/Gini/KS) on matured cohorts | Quarterly | Gini below 0.20 |
| Placeholder rate in `industry` / `work_experience` | Monthly | Any change in the encoding convention upstream |
| Approval-rate and bad-rate by decile | Monthly | Material drift from the hold-out decile table |

Helpers exist in `ml_pipeline/evaluation.py`
(`population_stability_index`, `decile_table`, `approval_curve`); the scheduled
job that runs them does not.

**Retraining.** No cadence set. With five months of data and an unresolved
provenance question, retraining should not be automated yet.

---

## 9. Approval

| Role | Name | Date | Decision |
|---|---|---|---|
| Model developer | | | |
| Independent validator | | | |
| Compliance / fair lending | | | |
| Model risk committee | | | |

**Recommendation: do not deploy.** Resolve the C1 observation-timing question
and the H1 placeholder encoding at source, obtain bureau data, get the reason
code wording through legal review, and stand up the monitoring in §8.
Re-baseline against **Gini 0.2914** on the 202205 hold-out.

The model is now *decision-capable* — it produces calibrated PDs, score points,
a defensible cutoff and explainable declines. It is not *approved*, and the
limitations in §6 are the reason.
