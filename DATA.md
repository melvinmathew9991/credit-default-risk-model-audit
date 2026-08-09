# Dataset

## Canonical copy

```
modular_code/modular_code/input/credit_risk_data.csv
```

| | |
|---|---|
| SHA-256 | `2f816622a796f383cd48667607438ebbbfea4b07c919073e03497f3245b00688` |
| Size | 23,918,876 bytes (23 MB) |
| Rows | 143,727 (plus header) |
| Columns | 27 |
| Application months | 202201 – 202205 |

Verify before any run:

```powershell
Get-FileHash modular_code\modular_code\input\credit_risk_data.csv -Algorithm SHA256
```

```bash
sha256sum modular_code/modular_code/input/credit_risk_data.csv
```

## Duplicate copies

Two byte-identical duplicates exist so the notebooks can be opened and run from
their own folders:

| Path | Purpose |
|---|---|
| `notebooks/notebooks/credit_risk_data.csv` | the exploratory notebook reads `credit_risk_data.csv` relative to itself |
| `data/data/credit_risk_data.csv` | original delivery folder |

All three were verified identical (same SHA-256, above). **Only the canonical
copy is tracked in git** — the other two are listed in `.gitignore`, so the
repository carries 23 MB of data rather than 69 MB.

If you change the dataset, update all three copies and the checksum in this file,
or delete the duplicates and repoint the notebooks at the canonical path.

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

These are documented in full in `modular_code/modular_code/MODEL_REVIEW.md`.

| Issue | Detail |
|---|---|
| **Target leakage** | `total_payement`, `received_principal`, `interest_received` describe repayment on the loan being scored, not prior loans, despite the data dictionary. 99.59% of rows have `number_of_loans == 0` yet show non-zero repayment. |
| **Placeholder encoding** | `industry` and `work_experience` are 83.92% placeholder zeros written two ways — `"0"` (87,848 rows, 9.75% default) and `"0.0"` (32,766 rows, 4.76% default). The spelling predicts default better than the real values do. |
| **Redundant columns** | `industry` and `work_experience` are identical in 120,618 of 143,727 rows. |
| **Degenerate columns** | `number_of_loans` is 0 for 99.59% of rows, which makes two derived ratio features near-constant. |
| **Repeated users** | 9,975 duplicate `User_id` values; 641 training users reappear in validation and 646 in hold-out. |
| **High null rates** | `employment_type` / `tier_of_employment` 59%, `married` 33%, `has_social_profile` 33%, `is_verified` 25%. |
