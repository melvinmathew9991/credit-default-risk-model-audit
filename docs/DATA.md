# Dataset

**This repository ships a sample, not the full dataset.** The full file is 22.8 MB —
79% of the repository — and arrived bundled with a paid course carrying no
licence statement, so it is not redistributed here.

## What is in the repository

```
data/credit_risk_data_sample.csv
```

| | |
|---|---|
| SHA-256 | `a4d1128f162a1456a49ccc9d1a94acfe44d839efe90ab602f61c49a80998420a` |
| Size | 1,927,638 bytes (1.8 MB) |
| Rows | 11,439 (plus header) — 8.0% of the full file |
| Columns | 27 — the complete schema |
| Application months | 202201 – 202205 |

Built by `analysis/make_sample.py`, which also prints a fidelity report
comparing it against the full file. CI verifies this checksum on every run.

## What is not

```
data/credit_risk_data.csv        (git-ignored; place it here if you have it)
```

| | |
|---|---|
| SHA-256 | `2f816622a796f383cd48667607438ebbbfea4b07c919073e03497f3245b00688` |
| Size | 23,918,876 bytes (22.8 MB) |
| Rows | 143,727 (plus header) |

Everything — tests, CI, all entry points — uses the full file when it is present
at that path and falls back to the sample when it is not. Drop it in and nothing
needs configuring.

```powershell
Get-FileHash data\credit_risk_data.csv -Algorithm SHA256
```

```bash
sha256sum data/credit_risk_data.csv
```

## What the sample preserves, and what it cannot

Stratified on **yearmo × label × placeholder spelling**, so the structure the
tests assert on survives. Measured against the full file:

| Property | Full | Sample |
|---|---|---|
| Default rate | 0.0937 | 0.0989 |
| Placeholder `"0"` share | 0.6112 | 0.6209 |
| Placeholder `"0.0"` share | 0.2280 | 0.2091 |
| Real work-experience share | 0.1608 | 0.1699 |
| `number_of_loans == 0` | 0.9959 | 0.9959 |
| Univariate AUC `received_principal` | 0.2493 | 0.2434 |
| Duplicate `User_id` rows | 9,975 | 1,682 |

So the sample still reproduces findings **H1** (the `"0"` / `"0.0"` split),
**M7** (repeat customers), **M10** (degenerate column) and **C1** (the leakage
relationship). Running `validate_data.py` against it produces the same
2 errors and 7 warnings as the full file.

**One property cannot survive sampling.** Finding **H2** — identical text parsed
as different Python types depending on which csv chunk it lands in — needs more
than 128k rows to cross a pandas chunk boundary. The two tests covering it are
marked `@pytest.mark.requires_full_dataset` and skip on the sample rather than
pass vacuously.

> ### ⚠️ Results from the sample are not the project's results
>
> The sample exists so the code can be exercised, not so the findings can be
> reproduced. A 2-trial run against it scores roughly **Gini 0.20**, against the
> **0.2914** in `MODEL_CARD.md`, which came from 40 trials on the full file.
> Monitoring will also raise alerts on it — 2,268 hold-out rows are too few for
> thresholds calibrated on 28,727. Both are expected; neither is drift.

## One copy, deliberately

The project as delivered stored the dataset three times — in `data/data/`,
`notebooks/notebooks/` and `modular_code/modular_code/input/` — 69 MB for 23 MB
of data, with no checksum and no way to tell which was authoritative. All three
were verified byte-identical and the duplicates removed.

Two guards keep it that way: a pre-commit hook and a CI step both fail if more
than one copy is ever tracked.

If you regenerate the sample, update its checksum here and in
`.github/workflows/ci.yml` — CI verifies it and will fail on a mismatch.

## Reading the file

Always read with `low_memory=False`:

```python
pd.read_csv(path, low_memory=False)
```

With pandas' default chunked type inference, identical text in `industry` and
`work_experience` is assigned different Python types depending on which
128k-row chunk it falls in — `str "0"` for rows outside index 98,304–131,071 and
`float 0.0` for rows inside it. Those become distinct categories downstream, so
encoded feature values end up depending on a row's position in the file.
`ml_pipeline.utils.process_data` already does this; the test
`test_read_is_type_stable` enforces it.

## Known data-quality issues

These are documented in full in `docs/MODEL_REVIEW.md`.

| Issue | Detail |
|---|---|
| **Target leakage** | `total_payement`, `received_principal`, `interest_received` describe repayment on the loan being scored, not prior loans, despite the data dictionary. 99.59% of rows have `number_of_loans == 0` yet show non-zero repayment. |
| **Placeholder encoding** | `industry` and `work_experience` are 83.92% placeholder zeros written two ways — `"0"` (87,848 rows, 9.75% default) and `"0.0"` (32,766 rows, 4.76% default). The spelling predicts default better than the real values do. |
| **Redundant columns** | `industry` and `work_experience` are identical in 120,618 of 143,727 rows. |
| **Degenerate columns** | `number_of_loans` is 0 for 99.59% of rows, which makes two derived ratio features near-constant. |
| **Repeated users** | 9,975 duplicate `User_id` values; 641 training users reappear in validation and 646 in hold-out. |
| **High null rates** | `employment_type` / `tier_of_employment` 59%, `married` 33%, `has_social_profile` 33%, `is_verified` 25%. |
