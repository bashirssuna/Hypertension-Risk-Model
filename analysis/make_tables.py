"""Participant characteristics by hypertension status (manuscript Tables 3 and 4).

Usage:  python analysis/make_tables.py
Writes: results/table3_numeric.csv, results/table4_categorical.csv
"""
from __future__ import annotations

import pandas as pd

from common import LABELS, LEVELS, RESULTS, TARGET, get_logger, load_data

log = get_logger("make_tables")

NUMERIC = ["age", "SBP", "DBP", "bmi", "m11", "m12", "m14", "m15", "HR", "b8",
           "sit_hours", "d1", "d3", "d4"]
CATEGORICAL = ["c1", "region", "status", "edu", "bp_meds", "fbg61", "smoke", "t12",
               "p1", "p4", "p7", "p10", "p13"]
EXTRA = {"SBP": "Systolic blood pressure (mmHg)", "DBP": "Diastolic blood pressure (mmHg)",
         "bp_meds": "Currently treated for raised blood pressure"}


def main() -> None:
    df = load_data()
    htn, normal = df[df[TARGET] == 1], df[df[TARGET] == 0]
    rows = []
    for v in NUMERIC:
        a, b = htn[v], normal[v]
        rows.append({"variable": v, "label": EXTRA.get(v, LABELS.get(v, v)),
                     "htn_mean": round(a.mean(), 1), "htn_sd": round(a.std(), 1),
                     "htn_min": round(a.min(), 1), "htn_max": round(a.max(), 1),
                     "normal_mean": round(b.mean(), 1), "normal_sd": round(b.std(), 1),
                     "normal_min": round(b.min(), 1), "normal_max": round(b.max(), 1)})
    pd.DataFrame(rows).to_csv(RESULTS / "table3_numeric.csv", index=False)

    rows = []
    for v in CATEGORICAL:
        for code, level in LEVELS[v].items():
            n1, n0 = int((htn[v] == code).sum()), int((normal[v] == code).sum())
            rows.append({"variable": v, "label": EXTRA.get(v, LABELS.get(v, v)), "level": level,
                         "htn_n": n1, "htn_pct": round(100 * n1 / len(htn), 1),
                         "normal_n": n0, "normal_pct": round(100 * n0 / len(normal), 1)})
    pd.DataFrame(rows).to_csv(RESULTS / "table4_categorical.csv", index=False)
    log.info("N=%d (hypertension %d, normal %d); wrote tables 3 and 4", len(df), len(htn), len(normal))


if __name__ == "__main__":
    main()
