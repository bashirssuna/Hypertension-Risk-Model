"""Compare nine classifiers (Table 2) and compute tree-model permutation importance.

Same procedure as the authors' original model-discovery notebook: repeated stratified
2-fold cross-validation (3 repeats); in each validation fold the threshold is the one
with the highest specificity among thresholds giving sensitivity 0.70 to 0.75 (or the
maximum geometric mean if none qualifies). Logistic regression, SVM and MLP use an
expanded feature set with pairwise interaction terms; the other six use the base set.

Usage:  python analysis/model_comparison.py
Writes: results/model_comparison.csv, results/importance_random_forest.csv,
        results/importance_lightgbm.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import CAT_VARS, NUM_VARS, RESULTS, SEED, TARGET, get_logger, lasso, load_data, sweep

log = get_logger("model_comparison")
# LightGBM is fit inside a pipeline on unnamed columns; permutation importance then
# warns about missing feature names. The warning does not affect the result.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def preprocessors(df: pd.DataFrame) -> tuple[ColumnTransformer, Pipeline]:
    cats = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[CAT_VARS]).categories_
    base = ColumnTransformer([
        ("num", StandardScaler(), NUM_VARS),
        ("cat", OneHotEncoder(handle_unknown="ignore", categories=cats, sparse_output=False), CAT_VARS),
    ])
    poly = Pipeline([("base", base),
                     ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))])
    return base, poly


def models(df: pd.DataFrame) -> dict[str, Pipeline]:
    base, poly = preprocessors(df)
    y = df[TARGET]
    spw = float((y == 0).sum() / (y == 1).sum())
    return {
        "LASSO": Pipeline([("pre", base), ("clf", lasso())]),
        "Logistic regression": Pipeline([("pre", poly), ("clf", LogisticRegression(
            random_state=SEED, class_weight="balanced", max_iter=2000, solver="liblinear"))]),
        "SVM": Pipeline([("pre", poly), ("clf", SVC(
            random_state=SEED, class_weight="balanced", probability=True))]),
        "MLP": Pipeline([("pre", poly), ("clf", MLPClassifier(
            random_state=SEED, max_iter=1000, hidden_layer_sizes=(100, 50),
            early_stopping=True, n_iter_no_change=15))]),
        "Random forest": Pipeline([("pre", base), ("clf", RandomForestClassifier(
            random_state=SEED, class_weight="balanced", n_estimators=150, max_depth=10))]),
        "Balanced random forest": Pipeline([("pre", base), ("clf", BalancedRandomForestClassifier(
            random_state=SEED))]),
        "XGBoost": Pipeline([("pre", base), ("clf", XGBClassifier(
            random_state=SEED, scale_pos_weight=spw, eval_metric="logloss"))]),
        "LightGBM": Pipeline([("pre", base), ("clf", LGBMClassifier(
            random_state=SEED, scale_pos_weight=spw, verbose=-1))]),
        "Gaussian naive Bayes": Pipeline([("pre", base), ("clf", GaussianNB())]),
    }


def fold_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.round(np.arange(0.001, 1.0, 0.001), 3)
    s = sweep(y, p, grid)
    ok = np.where((s["sn"] >= 0.70) & (s["sn"] <= 0.75))[0]
    if len(ok):
        return float(grid[ok[np.argmax(s["sp"][ok])]])
    return float(grid[np.nanargmax(np.sqrt(s["sn"] * s["sp"]))])


def main() -> None:
    df = load_data()
    X, y = df[CAT_VARS + NUM_VARS], df[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=SEED)
    rows = []
    for name, pipe in models(df).items():
        folds = []
        for tr, va in cv.split(X, y):
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            t = fold_threshold(y[va], p)
            s = sweep(y[va], p, np.array([t]))
            folds.append({"sn": s["sn"][0], "sp": s["sp"][0], "ppv": s["ppv"][0],
                          "f1": f1_score(y[va], (p >= t).astype(int)),
                          "roc_auc": roc_auc_score(y[va], p)})
        f = pd.DataFrame(folds)
        row: dict[str, object] = {"model": name}
        for k in f.columns:
            row[f"{k}_mean"], row[f"{k}_sd"] = f[k].mean(), f[k].std(ddof=1)
        rows.append(row)
        log.info("%-24s ROC AUC %.3f (SD %.3f)", name, row["roc_auc_mean"], row["roc_auc_sd"])
    out = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    out.to_csv(RESULTS / "model_comparison.csv", index=False)

    for name in ["Random forest", "LightGBM"]:
        pipe = models(df)[name].fit(X, y)
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        imp = permutation_importance(clf, pre.transform(X), y, n_repeats=10,
                                     random_state=SEED, scoring="roc_auc")
        pd.DataFrame({"feature": pre.get_feature_names_out(),
                      "importance": imp.importances_mean}).sort_values(
            "importance", ascending=False).to_csv(
            RESULTS / f"importance_{name.lower().replace(' ', '_')}.csv", index=False)
    log.info("wrote results/model_comparison.csv and importance files")


if __name__ == "__main__":
    main()
