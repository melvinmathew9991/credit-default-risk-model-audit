"""Evaluation entry point.

Run after engine.py. Loads the artefacts engine.py persisted (model, encoder,
feature list, processed splits) and writes a full evaluation pack to output/.

    python evaluate.py

Outputs
-------
output/metrics.json               headline metrics for train / val / hold_out
output/decile_table_<split>.csv   risk-ranked decile view
output/approval_curve_val.csv     bad-rate of accepted book vs approval rate
output/calibration_<split>.csv    predicted vs observed default rate by bin
output/feature_importance.csv     LightGBM split and gain importance
output/plots/*.png                ROC, PR, score distribution, SHAP, class rate
"""

import json
import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ml_pipeline import evaluation as ev

OUT = "output"
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)


# ---------------------------------------------------------------- load
clf = lgb.Booster(model_file=os.path.join(OUT, "model.txt"))
with open(os.path.join(OUT, "processed_splits.pkl"), "rb") as f:
    splits = pickle.load(f)
with open(os.path.join(OUT, "feature_columns.json")) as f:
    feature_cols = json.load(f)

train, val, hold_out = splits["train"], splits["val"], splits["hold_out"]
datasets = {"train": train, "val": val, "hold_out": hold_out}

preds = {name: clf.predict(d[clf.feature_name()]) for name, d in datasets.items()}
labels = {name: d["label"].values for name, d in datasets.items()}

print(f"Model features ({len(clf.feature_name())}): {clf.feature_name()}")
print(f"Best iteration: {clf.current_iteration()}")


# ---------------------------------------------------------------- metrics
report = {
    "n_features": len(clf.feature_name()),
    "features": clf.feature_name(),
    "n_trees": clf.current_iteration(),
    "splits": {},
}

for name in datasets:
    disc = ev.discrimination_metrics(labels[name], preds[name])
    cal, cal_tab = ev.calibration_metrics(labels[name], preds[name])
    report["splits"][name] = {**disc, **cal}
    cal_tab.to_csv(os.path.join(OUT, f"calibration_{name}.csv"), index=False)
    ev.decile_table(labels[name], preds[name]).to_csv(
        os.path.join(OUT, f"decile_table_{name}.csv"), index=False
    )

report["stability"] = {
    "psi_train_vs_val": ev.population_stability_index(preds["train"], preds["val"]),
    "psi_train_vs_hold_out": ev.population_stability_index(preds["train"], preds["hold_out"]),
    "psi_val_vs_hold_out": ev.population_stability_index(preds["val"], preds["hold_out"]),
}

ev.approval_curve(labels["val"], preds["val"]).to_csv(
    os.path.join(OUT, "approval_curve_val.csv"), index=False
)

fi = pd.DataFrame(
    {
        "feature": clf.feature_name(),
        "split": clf.feature_importance("split"),
        "gain": clf.feature_importance("gain"),
    }
).sort_values("gain", ascending=False)
fi["gain_pct"] = 100 * fi["gain"] / fi["gain"].sum()
fi.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
report["top_features_by_gain"] = fi.head(10).to_dict("records")

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(report, f, indent=2)


# ---------------------------------------------------------------- plots
COLORS = {"train": "#4269d0", "val": "#efb118", "hold_out": "#ff725c"}


def save(filename):
    """Write the current figure to the plots directory and close it."""
    plt.savefig(os.path.join(PLOTS, filename), dpi=120, bbox_inches="tight")
    plt.close()


plt.figure(figsize=(11, 7))
for name in datasets:
    fpr, tpr, _ = roc_curve(labels[name], preds[name])
    plt.plot(
        fpr,
        tpr,
        color=COLORS[name],
        label=f"{name} AUC = {roc_auc_score(labels[name], preds[name]):.4f}",
    )
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
save("roc_curve.png")

plt.figure(figsize=(11, 7))
for name in datasets:
    pr, re, _ = precision_recall_curve(labels[name], preds[name])
    plt.plot(
        re,
        pr,
        color=COLORS[name],
        label=f"{name} AP = {average_precision_score(labels[name], preds[name]):.4f}",
    )
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (defaulter class)")
plt.legend()
plt.grid(alpha=0.3)
save("pr_curve.png")

fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharex=True)
for ax, name in zip(axes, datasets, strict=False):
    sub = pd.DataFrame({"y": labels[name], "p": preds[name]})
    sns.kdeplot(sub[sub.y == 1].p, ax=ax, fill=True, label="Defaulter", color="#ff725c")
    sns.kdeplot(sub[sub.y == 0].p, ax=ax, fill=True, label="Non-Defaulter", color="#4269d0")
    ax.set_title(f"{name} - predicted score")
    ax.set_xlabel("P(default)")
    ax.legend()
save("score_distribution.png")

plt.figure(figsize=(11, 7))
for name in datasets:
    d = ev.decile_table(labels[name], preds[name])
    plt.plot(d["decile"], d["bad_rate"], marker="o", color=COLORS[name], label=name)
plt.xlabel("Risk decile (1 = riskiest)")
plt.ylabel("Observed default rate")
plt.title("Class rate by score decile")
plt.legend()
plt.grid(alpha=0.3)
save("class_rate.png")

plt.figure(figsize=(8, 7))
for name in datasets:
    c = pd.read_csv(os.path.join(OUT, f"calibration_{name}.csv"))
    plt.plot(c["predicted"], c["actual"], marker="o", color=COLORS[name], label=name)
plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
plt.xlabel("Mean predicted P(default)")
plt.ylabel("Observed default rate")
plt.title("Calibration")
plt.legend()
plt.grid(alpha=0.3)
save("calibration.png")

plt.figure(figsize=(10, 6))
top = fi.head(15).iloc[::-1]
plt.barh(top["feature"], top["gain_pct"], color="#4269d0")
plt.xlabel("% of total gain")
plt.title("LightGBM feature importance (gain)")
save("feature_importance.png")

try:
    import shap

    # A booster reloaded from model.txt has empty .params, which makes
    # shap.TreeExplainer raise KeyError('objective'). Restore it.
    if "objective" not in clf.params:
        clf.params["objective"] = "binary"
    explainer = shap.TreeExplainer(clf)
    sample = val.sample(min(5000, len(val)), random_state=7)
    sv = explainer.shap_values(sample[clf.feature_name()])
    # For a binary booster shap returns [class_0, class_1] with class_0 == -class_1.
    # Index 1 is the defaulter class - index 0 would invert every sign and so
    # reverse the direction of every explanation.
    sv = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv, sample[clf.feature_name()], max_display=20, show=False)
    plt.gcf().set_size_inches(12, 8)
    plt.title("SHAP summary - validation")
    save("shap_summary.png")
    print("SHAP summary written")
except Exception as e:  # SHAP is explanatory only - never fail the run on it
    print(f"SHAP skipped: {e}")


# ---------------------------------------------------------------- console
print("\n" + "=" * 78)
print(f"{'split':<10}{'n':>8}{'bad%':>8}{'AUC':>9}{'Gini':>8}{'KS':>8}{'PR-AUC':>9}{'Brier':>9}")
print("=" * 78)
for name in datasets:
    s = report["splits"][name]
    print(
        f"{name:<10}{s['n']:>8}{100*s['default_rate']:>7.2f}%{s['roc_auc']:>9.4f}"
        f"{s['gini']:>8.4f}{s['ks']:>8.4f}{s['pr_auc_class1']:>9.4f}{s['brier']:>9.4f}"
    )
print("=" * 78)
print(f"PSI  train->val      : {report['stability']['psi_train_vs_val']:.4f}")
print(f"PSI  train->hold_out : {report['stability']['psi_train_vs_hold_out']:.4f}")
print("\nTop features by gain:")
print(fi.head(10).to_string(index=False))
print(f"\nEvaluation pack written to {OUT}/")
