"""Figures 2 to 9 of the manuscript (Figure 1 is drawn by preprocessing/build_analytic_data.R).

Run model_comparison.py and run_analysis.py first; this script only reads their output.

Usage:  python analysis/make_figures.py
Writes: results/figures/figure-02 ... figure-09 (PNG, 300 dpi)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from common import FIGURES, RESULTS, get_logger, sweep

log = get_logger("make_figures")
DPI = 300


def pretty(feature: str) -> str:
    """Readable label for a model column such as 'region_2' or 'num__age'."""
    from common import LABELS, LEVELS
    stem = feature.split("__", 1)[-1]
    if stem in LABELS:
        return LABELS[stem]
    var, _, code = stem.rpartition("_")
    if var in LABELS and code.isdigit():
        level = LEVELS.get(var, {}).get(int(code), code)
        return f"{LABELS[var]}: {level}"
    return stem


def importance_plot(csv: str, out: str) -> None:
    d = pd.read_csv(RESULTS / csv).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([pretty(f) for f in d["feature"]], d["importance"], color="#4c72b0")
    ax.set_xlabel("Permutation importance (decrease in ROC AUC)")
    fig.tight_layout()
    fig.savefig(FIGURES / out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = json.load(open(RESULTS / "results.json"))
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    y, p = oof["y_true"].to_numpy(), oof["oof_proba"].to_numpy()
    thr = res["thresholds"]

    importance_plot("importance_random_forest.csv", "figure-02-importance-random-forest.png")
    importance_plot("importance_lightgbm.csv", "figure-03-importance-lightgbm.png")

    coef = pd.read_csv(RESULTS / "lasso_coefficients.csv")
    coef = coef[coef["coef"] != 0].reindex(coef["coef"].abs().sort_values().index).dropna()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(coef) + 1.2))
    ax.barh([pretty(f) for f in coef["feature"]], coef["coef"],
            color=np.where(coef["coef"] > 0, "#c44e52", "#4c72b0"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LASSO coefficient (log-odds per SD, or versus reference level)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-04-lasso-coefficients.png", dpi=DPI)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {res['oof']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-05-roc-oof.png", dpi=DPI)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec, color="tab:green", label=f"Average precision = {res['oof']['pr_auc']:.3f}")
    ax.axhline(y.mean(), color="grey", ls=":", lw=0.8, label=f"Prevalence = {y.mean():.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-06-precision-recall-oof.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, (p >= thr["balanced"]).astype(int),
                                            display_labels=["No hypertension", "Hypertension"],
                                            cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_xlabel(f"Predicted (threshold {thr['balanced']:.3f})")
    ax.set_ylabel("Observed")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-07-confusion-matrix-oof.png", dpi=DPI)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(mean_pred, frac_pos, "o-", label=f"Brier = {res['oof']['brier']:.3f}, "
                                            f"slope = {res['oof']['slope']:.2f}")
    ax.set_xlabel("Mean predicted probability (deciles)")
    ax.set_ylabel("Observed proportion with hypertension")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-08-calibration-oof.png", dpi=DPI)
    plt.close(fig)

    s = sweep(y, p)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, lab in [("sn", "Sensitivity"), ("sp", "Specificity"), ("ppv", "PPV"), ("npv", "NPV")]:
        ax.plot(s["t"], s[key], label=lab)
    for key, lab, st in [("high_sens", "High sensitivity", ":"), ("balanced", "Balanced", "--"),
                         ("high_ppv", "High PPV", "-.")]:
        ax.axvline(thr[key], color="grey", ls=st, lw=1, label=f"{lab} ({thr[key]:.3f})")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure-09-threshold-sweep-oof.png", dpi=DPI)
    plt.close(fig)
    log.info("wrote figures 2 to 9")


if __name__ == "__main__":
    main()
