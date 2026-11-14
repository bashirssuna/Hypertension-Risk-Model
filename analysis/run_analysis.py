"""Nested cross-validated evaluation of the calibrated LASSO hypertension model.

The model is built in two steps: LASSO selects variables, then a calibrated LASSO
(isotonic, 3-fold) is refit on the selected variables. Every reported performance number
comes from out-of-fold (OOF) predictions: in each of 5 outer folds, both steps (selection,
penalty tuning by inner 5-fold CV, and calibration) are fit on the training folds only,
then the held-out fold is predicted. The deployed model is the same two steps fit on all
participants.

The three operating thresholds are chosen on the pooled OOF predictions. Sensitivity,
specificity, PPV and NPV are reported at those fixed cut-offs, with 95% intervals from
2,000 bootstrap resamples of the OOF predictions. Because the cut-offs are chosen on the
same predictions, a cross-fitted check chooses them on four folds and applies them to the
fifth.

Usage:  python analysis/run_analysis.py
Reads:  data/htn_analytic.csv, data/htn_analytic_imputations.csv
Writes: results/results.json, results/oof_predictions.csv,
        results/selection_by_fold.csv, results/lasso_coefficients.csv,
        results/lasso_coefficients_nob8.csv,
        app/lasso_htn_full.joblib, app/lasso_htn_nob8.joblib
"""
from __future__ import annotations

import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

from common import (APP, CAT_VARS, DATA, NUM_VARS, RESULTS, SEED, TARGET, calibration,
                    get_logger, lasso_pipeline, load_data, metrics_at, select_thresholds,
                    selected_variables, sha256)

N_BOOT = 2000
log = get_logger("run_analysis")


def fit_two_step(X: pd.DataFrame, y: np.ndarray, cat_vars: list[str],
                 num_vars: list[str]) -> tuple[CalibratedClassifierCV, list[str], pd.DataFrame]:
    """Step 1: LASSO selects variables. Step 2: calibrated LASSO on the selected ones.

    Returns the calibrated model, the selected variables and the step-1 coefficients.
    """
    step1 = lasso_pipeline(cat_vars, num_vars).fit(X[cat_vars + num_vars], y)
    keep = selected_variables(step1, cat_vars, num_vars)
    names = step1.named_steps["pre"].get_feature_names_out()
    coef = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names],
                         "coef": step1.named_steps["lasso"].coef_.ravel()})
    cats = [v for v in cat_vars if v in keep]
    nums = [v for v in num_vars if v in keep]
    model = CalibratedClassifierCV(lasso_pipeline(cats, nums), method="isotonic", cv=3)
    model.fit(X[cats + nums], y)
    return model, cats + nums, coef


def nested_oof(df: pd.DataFrame, cat_vars: list[str], num_vars: list[str]
               ) -> tuple[np.ndarray, np.ndarray, list[float], list[list[str]]]:
    """Return OOF probabilities, the outer fold of each row, per-fold AUCs and the
    variables selected in each training fold (from the same fits that predict)."""
    X, y = df[cat_vars + num_vars], df[TARGET].to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof, fold = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    aucs, selections = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        model, feats, _ = fit_two_step(X.iloc[tr], y[tr], cat_vars, num_vars)
        oof[te] = model.predict_proba(X.iloc[te][feats])[:, 1]
        fold[te] = k
        aucs.append(float(roc_auc_score(y[te], oof[te])))
        selections.append(feats)
    return oof, fold, aucs, selections


def bootstrap(y: np.ndarray, p: np.ndarray, thr: dict[str, float]) -> dict:
    """Percentile CIs. Classification metrics are taken at the fixed cut-offs in `thr`;
    the thresholds themselves are also re-chosen in each resample to show how much
    they move."""
    stats: dict[str, list[float]] = {}
    n = len(y)
    for i in range(N_BOOT):
        idx = resample(np.arange(n), replace=True, random_state=i)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        row = {"roc_auc": roc_auc_score(yb, pb), "pr_auc": average_precision_score(yb, pb),
               "brier": brier_score_loss(yb, pb)}
        row.update(calibration(yb, pb))
        for name, t in thr.items():
            for k, v in metrics_at(yb, pb, t).items():
                if k in ("sn", "sp", "ppv", "npv"):
                    row[f"{name}_{k}"] = v
        for name, t in select_thresholds(yb, pb).items():
            row[f"{name}_threshold"] = t
        for k, v in row.items():
            stats.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in stats.items()}


def crossfit_thresholds(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> dict:
    """Choose thresholds on four folds' OOF predictions, apply them to the fifth, pool."""
    pred: dict[str, np.ndarray] = {}
    chosen: dict[str, list[float]] = {}
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        for name, t in select_thresholds(y[tr], p[tr]).items():
            pred.setdefault(name, np.zeros(len(y), dtype=bool))[te] = p[te] >= t
            chosen.setdefault(name, []).append(t)
    out = {}
    for name, yhat in pred.items():
        tp = int((yhat & (y == 1)).sum())
        fp = int((yhat & (y == 0)).sum())
        fn = int((~yhat & (y == 1)).sum())
        tn = int((~yhat & (y == 0)).sum())
        out[name] = {"thresholds_by_fold": chosen[name], "sn": tp / (tp + fn),
                     "sp": tn / (tn + fp), "ppv": tp / (tp + fp), "npv": tn / (tn + fn)}
    return out


def risk_bands(y: np.ndarray, p: np.ndarray, low: float, high: float) -> dict:
    bands = {"low": p < low, "moderate": (p >= low) & (p < high), "high": p >= high}
    return {name: {"n": int(m.sum()), "pct": float(100 * m.mean()), "cases": int(y[m].sum()),
                   "case_rate": float(100 * y[m].mean()) if m.any() else None,
                   "share_of_all_cases": float(100 * y[m].sum() / y.sum())}
            for name, m in bands.items()}


def final_model(df: pd.DataFrame, cat_vars: list[str], num_vars: list[str]) -> dict:
    """The deployed model: the same two steps, fit on all participants."""
    model, feats, coef = fit_two_step(df, df[TARGET].to_numpy(), cat_vars, num_vars)
    return {"selected": feats, "coef": coef, "model": model, "features": feats}


def main() -> None:
    df = load_data()
    y = df[TARGET].to_numpy()
    log.info("N=%d, cases=%d (%.1f%%)", len(y), y.sum(), 100 * y.mean())

    oof, fold, fold_aucs, selections = nested_oof(df, CAT_VARS, NUM_VARS)
    thr = select_thresholds(y, oof)
    ci = bootstrap(y, oof, thr)
    log.info("OOF ROC AUC %.3f; thresholds %s", roc_auc_score(y, oof), thr)

    res: dict = {
        "data": {"file": "data/htn_analytic.csv", "sha256": sha256(DATA / "htn_analytic.csv"),
                 "n": int(len(y)), "cases": int(y.sum()), "prevalence": float(y.mean())},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "fold_aucs": fold_aucs,
        "oof": {"roc_auc": float(roc_auc_score(y, oof)),
                "pr_auc": float(average_precision_score(y, oof)),
                "brier": float(brier_score_loss(y, oof)), **calibration(y, oof)},
        "ci95": ci, "thresholds": thr,
        "at_threshold": {k: metrics_at(y, oof, t) for k, t in thr.items()},
        "risk_bands": risk_bands(y, oof, thr["high_sens"], thr["high_ppv"]),
        "crossfit_thresholds": crossfit_thresholds(y, oof, fold),
    }

    sel = pd.DataFrame([{v: int(v in s) for v in NUM_VARS + CAT_VARS} for s in selections])
    sel.index = pd.Index([f"fold_{i + 1}" for i in range(len(sel))])
    sel.T.assign(folds_selected=sel.sum()).to_csv(RESULTS / "selection_by_fold.csv")

    # Cholesterol-free variant, evaluated the same way.
    num_nob8 = [v for v in NUM_VARS if v != "b8"]
    oof_nob8, _, aucs_nob8, _ = nested_oof(df, CAT_VARS, num_nob8)
    thr_nob8 = select_thresholds(y, oof_nob8)
    res["nob8"] = {"roc_auc": float(roc_auc_score(y, oof_nob8)), "fold_aucs": aucs_nob8,
                   "pr_auc": float(average_precision_score(y, oof_nob8)),
                   "brier": float(brier_score_loss(y, oof_nob8)), **calibration(y, oof_nob8),
                   "thresholds": thr_nob8,
                   "at_threshold": {k: metrics_at(y, oof_nob8, t) for k, t in thr_nob8.items()}}

    # Sensitivity: the same evaluation on each of the five imputed datasets.
    imps = pd.read_csv(DATA / "htn_analytic_imputations.csv")
    res["imputation_sensitivity_auc"] = {}
    for _, d in imps.groupby(".imp"):
        k = int(d[".imp"].iloc[0])
        p_k, _, _, _ = nested_oof(d.reset_index(drop=True), CAT_VARS, NUM_VARS)
        res["imputation_sensitivity_auc"][k] = float(roc_auc_score(d[TARGET], p_k))
    log.info("AUC by imputation: %s", res["imputation_sensitivity_auc"])

    full = final_model(df, CAT_VARS, NUM_VARS)
    nob8 = final_model(df, CAT_VARS, num_nob8)
    res["final_selected"] = full["selected"]
    res["final_selected_nob8"] = nob8["selected"]
    full["coef"].to_csv(RESULTS / "lasso_coefficients.csv", index=False)
    nob8["coef"].to_csv(RESULTS / "lasso_coefficients_nob8.csv", index=False)
    APP.mkdir(exist_ok=True)
    for name, m, t in [("lasso_htn_full", full, thr), ("lasso_htn_nob8", nob8, thr_nob8)]:
        joblib.dump({"model": m["model"], "features": m["features"], "thresholds": t,
                     "sklearn": sklearn.__version__}, APP / f"{name}.joblib")

    pd.DataFrame({"y_true": y, "oof_proba": oof, "oof_proba_nob8": oof_nob8}).to_csv(
        RESULTS / "oof_predictions.csv", index=False)
    with open(RESULTS / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    log.info("wrote results/results.json")


if __name__ == "__main__":
    main()
