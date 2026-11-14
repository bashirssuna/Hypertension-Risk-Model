"""Shared settings and helpers for the hypertension risk model analysis."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
APP = ROOT / "app"

SEED = 42
TARGET = "hypertension"

# 24 candidate predictors. SBP and DBP define the outcome and are never candidates.
CAT_VARS = ["c1", "region", "status", "edu", "smoke", "t12",
            "p1", "p4", "p7", "p10", "p13", "fbg61"]
NUM_VARS = ["age", "d1", "d3", "d4", "sit_hours", "m11", "m12", "m14", "m15",
            "b8", "bmi", "HR"]

LABELS = {
    "c1": "Sex", "region": "Region", "status": "Residence", "edu": "Education",
    "smoke": "Smoking status", "t12": "Uses smokeless tobacco", "p1": "Vigorous-intensity work",
    "p4": "Moderate-intensity work", "p7": "Walks or cycles for travel",
    "p10": "Vigorous-intensity sport or recreation",
    "p13": "Moderate-intensity sport or recreation",
    "fbg61": "Fasting glucose >= 6.1 mmol/L", "age": "Age (years)",
    "d1": "Days per week eating fruit", "d3": "Days per week eating vegetables",
    "d4": "Vegetable servings on days eaten", "sit_hours": "Hours sitting per day",
    "m11": "Height (cm)", "m12": "Weight (kg)", "m14": "Waist circumference (cm)",
    "m15": "Hip circumference (cm)", "b8": "Total cholesterol (mmol/L)",
    "bmi": "Body mass index (kg/m2)", "HR": "Heart rate (beats per minute)",
}
LEVELS = {
    "c1": {1: "Male", 2: "Female"},
    "region": {1: "Central", 2: "Eastern", 3: "Northern", 4: "Western"},
    "status": {1: "Urban", 2: "Rural"},
    # c5 1 = no formal schooling; 2 = primary; 3 to 5 = O-level, junior secondary or
    # A-level; 6 and 7 = university or postgraduate
    "edu": {1: "No formal schooling", 2: "Primary", 3: "Secondary", 4: "Tertiary"},
    "smoke": {1: "Never", 2: "Former", 3: "Current"},
    "fbg61": {0: "No", 1: "Yes"},
    "bp_meds": {0: "No", 1: "Yes"},
}
for _v in ["t12", "p1", "p4", "p7", "p10", "p13"]:
    LEVELS[_v] = {1: "Yes", 2: "No"}

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 3)
SN_MIN = 0.70   # sensitivity floor for the high-PPV threshold
PPV_MIN = 0.30  # PPV floor for the high-sensitivity threshold


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the analysis dataset; stop with a clear message if it is absent."""
    path = path or DATA / "htn_analytic.csv"
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}. See data/README.md.")
    return pd.read_csv(path)


def lasso(seed: int = SEED) -> LogisticRegressionCV:
    """L1-penalised logistic regression with the penalty strength tuned by 5-fold CV.

    scikit-learn 1.8 deprecated `penalty`; there, l1_ratios=(1.0,) is the L1 penalty.
    Leaving both out would silently give an L2 (ridge) model, so the version is checked.
    """
    common = dict(Cs=10, cv=5, solver="saga", class_weight="balanced",
                  scoring="roc_auc", max_iter=5000, random_state=seed)
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        # use_legacy_attributes only changes the stored CV score arrays, not coef_
        return LogisticRegressionCV(l1_ratios=(1.0,), use_legacy_attributes=False, **common)
    return LogisticRegressionCV(penalty="l1", **common)


def lasso_pipeline(cat_vars: list[str], num_vars: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_vars),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_vars),
    ])
    return Pipeline([("pre", pre), ("lasso", lasso())])


def selected_variables(fitted: Pipeline, cat_vars: list[str], num_vars: list[str]) -> list[str]:
    """Return candidate variables with at least one non-zero LASSO coefficient."""
    names = fitted.named_steps["pre"].get_feature_names_out()
    coef = fitted.named_steps["lasso"].coef_.ravel()
    keep = set()
    for name, c in zip(names, coef):
        if c == 0:
            continue
        stem = name.split("__", 1)[1]
        for v in num_vars + cat_vars:
            if stem == v or stem.startswith(v + "_"):
                keep.add(v)
    return [v for v in num_vars + cat_vars if v in keep]


def sweep(y: np.ndarray, p: np.ndarray, grid: np.ndarray = GRID) -> dict[str, np.ndarray]:
    """Sensitivity, specificity, PPV and NPV at every threshold in `grid` (p >= t)."""
    pred = p[:, None] >= grid[None, :]
    pos = (y == 1)[:, None]
    tp = (pred & pos).sum(0)
    fp = (pred & ~pos).sum(0)
    fn = pos.sum() - tp
    tn = (~pos).sum() - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "t": grid, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sn": tp / (tp + fn), "sp": tn / (tn + fp),
            "ppv": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan),
            "npv": np.where(tn + fn > 0, tn / np.maximum(tn + fn, 1), np.nan),
        }


def select_thresholds(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Pick the three operating thresholds from predictions.

    balanced:  maximum geometric mean of sensitivity and specificity
    high_sens: maximum sensitivity with PPV >= 0.30 (ties: higher PPV)
    high_ppv:  maximum PPV with sensitivity >= 0.70 (ties: higher sensitivity)
    """
    s = sweep(y, p)
    gmean = np.sqrt(s["sn"] * s["sp"])
    out = {"balanced": float(s["t"][np.nanargmax(gmean)])}
    ok = np.where(s["ppv"] >= PPV_MIN)[0]
    i = ok[np.lexsort((s["ppv"][ok], s["sn"][ok]))[-1]]
    out["high_sens"] = float(s["t"][i])
    ok = np.where(s["sn"] >= SN_MIN)[0]
    i = ok[np.lexsort((s["sn"][ok], s["ppv"][ok]))[-1]]
    out["high_ppv"] = float(s["t"][i])
    return out


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    s = sweep(y, p, np.array([t]))
    return {k: float(s[k][0]) for k in ["sn", "sp", "ppv", "npv", "tp", "fp", "fn", "tn"]}


def calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Observed/expected ratio and calibration slope (logistic recalibration, IRLS)."""
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    x = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-x @ beta))
        w = mu * (1 - mu)
        step = np.linalg.solve(x.T @ (x * w[:, None]), x.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return {"oe_ratio": float(y.mean() / p.mean()), "slope": float(beta[1])}
