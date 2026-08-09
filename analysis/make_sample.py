"""Build the committed sample of the dataset.

The full file is 22.8 MB - 79% of the repository - and is not distributed here.
This produces a small sample that is committed instead, so that the test suite
and CI exercise real data rather than fixtures.

A sample is only worth committing if it preserves the properties the tests
assert on. This one is stratified on the three dimensions that matter and is
checked against the full file afterwards:

    yearmo                  all five months, so the time-based split works
    label                   the default rate per month
    placeholder spelling    the "0" / "0.0" / real split in work_experience,
                            which is finding H1 and the thing the data contract
                            errors on

It also deliberately carries forward repeat User_id values (finding M7) and the
`received_principal` leakage relationship (finding C1), because tests assert on
both.

One property cannot survive sampling: the mixed-dtype defect (H2) needs more
than 128k rows to cross a pandas chunk boundary. The test for it is marked as
requiring the full file.

    python analysis/make_sample.py --rows 10000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FULL = "data/credit_risk_data.csv"
SAMPLE = "data/credit_risk_data_sample.csv"
EMIS = [f"emi_{i}_dpd" for i in range(1, 4)]


def placeholder_bucket(s):
    """The H1 split: which spelling of the placeholder a row carries."""
    return s.astype(str).where(s.astype(str).isin(["0", "0.0"]), "real")


def label_of(df):
    return (df[EMIS].max(axis=1) >= 60).astype(int)


def build(full, n_rows, seed=7, n_repeat_customers=100):
    """Proportional stratified sample over yearmo x label x placeholder spelling."""
    rng = np.random.default_rng(seed)
    strata = pd.DataFrame(
        {
            "yearmo": full["yearmo"],
            "label": label_of(full),
            "ph": placeholder_bucket(full["work_experience"]),
        }
    )
    key = strata["yearmo"].astype(str) + "|" + strata["label"].astype(str) + "|" + strata["ph"]

    frac = n_rows / len(full)
    picked = []
    for _k, idx in full.groupby(key.values).groups.items():
        idx = np.asarray(idx)
        # one row minimum so no combination disappears; anything more distorts
        # the shares, because the rare strata get inflated relative to the
        # common ones.
        take = min(len(idx), max(1, int(round(len(idx) * frac))))
        picked.append(rng.choice(idx, size=take, replace=False))
    sample = full.loc[np.concatenate(picked)].copy()

    # Finding M7 asserts repeat customers exist. Anything added here is by
    # definition unstratified, so it is capped hard: the tests need duplicates
    # to be present, not to be present at their original rate. Pulling siblings
    # for every selected repeat customer added 6,690 rows and moved every share
    # by 2-6 points.
    dup_ids = set(full.loc[full["User_id"].duplicated(keep=False), "User_id"])
    in_sample = sorted(dup_ids & set(sample["User_id"]))
    if in_sample:
        chosen = rng.choice(in_sample, size=min(n_repeat_customers, len(in_sample)), replace=False)
        siblings = full[full["User_id"].isin(chosen) & ~full.index.isin(sample.index)]
        sample = pd.concat([sample, siblings])

    return sample.sort_index()


def fidelity(full, sample):
    """Compare the sample against the full file on everything tests rely on."""
    rows = []

    def add(metric, f, s, tol=None):
        ok = "" if tol is None else ("ok" if abs(f - s) <= tol else "DRIFT")
        rows.append({"metric": metric, "full": f, "sample": s, "check": ok})

    add("rows", len(full), len(sample))
    add("columns", full.shape[1], sample.shape[1])

    lf, ls = label_of(full), label_of(sample)
    add("default rate", lf.mean(), ls.mean(), 0.01)

    for m in sorted(full["yearmo"].unique()):
        add(
            f"default rate {m}",
            lf[full.yearmo == m].mean(),
            ls[sample.yearmo == m].mean(),
            0.02,
        )

    pf, ps = (
        placeholder_bucket(full["work_experience"]),
        placeholder_bucket(sample["work_experience"]),
    )
    for b in ["0", "0.0", "real"]:
        add(f"placeholder share '{b}'", (pf == b).mean(), (ps == b).mean(), 0.02)

    add(
        "number_of_loans == 0 share",
        (full.number_of_loans == 0).mean(),
        (sample.number_of_loans == 0).mean(),
        0.02,
    )
    add(
        "duplicate User_id rows",
        int(full.User_id.duplicated().sum()),
        int(sample.User_id.duplicated().sum()),
    )

    for c in ["received_principal", "total_payement", "total_income"]:
        add(
            f"univariate AUC {c}",
            roc_auc_score(lf, full[c].fillna(full[c].median())),
            roc_auc_score(ls, sample[c].fillna(sample[c].median())),
            0.03,
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--out", default=SAMPLE)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.exists(args.full):
        print(f"the full dataset is not present at {args.full}; nothing to sample from")
        return 1

    # dtype=str + na_filter=False reads the file exactly as written, so the
    # "0" / "0.0" distinction survives into the sample byte for byte.
    raw = pd.read_csv(args.full, dtype=str, keep_default_na=False, na_filter=False)
    typed = pd.read_csv(args.full, low_memory=False)
    for c in ["yearmo", "number_of_loans", "User_id", *EMIS]:
        raw[c] = typed[c]
    for c in ["received_principal", "total_payement", "total_income"]:
        raw[c] = pd.to_numeric(typed[c], errors="coerce")

    sample = build(raw, args.rows, args.seed)

    report = fidelity(raw, sample)
    print("=" * 78)
    print(f"SAMPLE FIDELITY  ({len(sample):,} of {len(raw):,} rows)")
    print("=" * 78)
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    drift = report[report["check"] == "DRIFT"]
    if len(drift):
        print(f"\n{len(drift)} metric(s) drifted beyond tolerance:")
        print(drift.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Write the original text back verbatim with LF endings. Opening the handle
    # with newline="\n" makes the line terminator explicit without depending on
    # to_csv's keyword, which pandas renamed from line_terminator to
    # lineterminator in 2.0 - this project is pinned to 1.3.5.
    out = pd.read_csv(args.full, dtype=str, keep_default_na=False, na_filter=False).loc[
        sample.index
    ]
    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        out.to_csv(f, index=False)

    size = os.path.getsize(args.out)
    print(
        f"\nWritten {args.out}  ({size / 1024:.0f} KB, {size / os.path.getsize(args.full):.1%} of full)"
    )
    return 1 if len(drift) else 0


if __name__ == "__main__":
    sys.exit(main())
