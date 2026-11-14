# Results

All numbers below come from `results.json`, `model_comparison.csv` and
`selection_by_fold.csv`, produced by the scripts in `../analysis/` from
`../data/htn_analytic.csv` (SHA-256 `57f677cb...97afd3`). Methods and caveats are in
`../docs/EVALUATION_NOTES.md`.

## Sample

3,554 adults; 825 (23.2%) had hypertension. Of those, 763 had a mean blood pressure of
at least 140/90 mmHg and 128 reported taking prescribed blood pressure drugs; 62 of the
treated had a measured pressure below 140/90. Mean age was 38.3 years (SD 13.8) and
2,189 (61.6%) were women. Participant characteristics by hypertension status are in
`table3_numeric.csv` and `table4_categorical.csv`.

## Final model: out-of-fold performance

Nested 5-fold cross-validation; 95% intervals from 2,000 bootstrap resamples.

| Measure | Estimate | 95% CI |
| --- | --- | --- |
| ROC AUC | 0.730 | 0.710 to 0.749 |
| Average precision (PR AUC) | 0.432 | 0.397 to 0.466 |
| Brier score | 0.157 | 0.150 to 0.164 |
| Observed / expected | 0.997 | 0.940 to 1.052 |
| Calibration slope | 0.875 | 0.763 to 0.984 |

ROC AUC in the five outer folds ranged from 0.717 to 0.751. Repeating the whole
evaluation on each of the five imputed datasets gave ROC AUCs of 0.729 to 0.731. A
calibration slope below 1 means the predicted risks are a little too spread out: slightly
too high for the highest-risk people and too low for the lowest.

Sensitivity, specificity, PPV and NPV at the three cut-offs the app uses, with bootstrap
intervals at those fixed cut-offs:

| Threshold | Cut-off | Sensitivity | Specificity | PPV | NPV |
| --- | --- | --- | --- | --- | --- |
| High sensitivity | 0.115 | 0.887 (0.865 to 0.908) | 0.376 (0.358 to 0.394) | 0.300 (0.281 to 0.318) | 0.917 (0.901 to 0.933) |
| Balanced | 0.230 | 0.728 (0.696 to 0.758) | 0.623 (0.604 to 0.641) | 0.368 (0.345 to 0.392) | 0.884 (0.869 to 0.898) |
| High PPV | 0.235 | 0.714 (0.682 to 0.744) | 0.634 (0.616 to 0.652) | 0.371 (0.347 to 0.394) | 0.880 (0.866 to 0.895) |

At 0.230 the model finds 601 of 825 cases and raises 1,030 false alarms among
2,729 people without hypertension.

The cut-offs were chosen on the same out-of-fold predictions they are scored on. In the
cross-fitted check below, each rule picks its cut-off on four folds and is scored on the
fifth; the pooled results are close to those above.

| Rule | Cut-off chosen in each fold | Sensitivity | Specificity | PPV | NPV |
| --- | --- | --- | --- | --- | --- |
| High sensitivity | 0.115, 0.115, 0.115, 0.115, 0.120 | 0.884 | 0.382 | 0.302 | 0.916 |
| Balanced | 0.230, 0.185, 0.220, 0.230, 0.230 | 0.743 | 0.594 | 0.356 | 0.884 |
| High PPV | 0.235, 0.235, 0.235, 0.235, 0.235 | 0.714 | 0.634 | 0.371 | 0.880 |

## Risk bands

| Band | Rule | Participants | Cases | Case rate | Share of all cases |
| --- | --- | --- | --- | --- | --- |
| Low | p < 0.115 | 1,118 (31.5%) | 93 | 8.3% | 11.3% |
| Moderate | 0.115 to < 0.235 | 848 (23.9%) | 143 | 16.9% | 17.3% |
| High | p >= 0.235 | 1,588 (44.7%) | 589 | 37.1% | 71.4% |

## Selected predictors

Fit on all 3,554 participants, LASSO kept 11 of the 24 candidates: age, total
cholesterol, waist circumference, sex, residence, weight, heart rate, region, fasting
glucose of at least 6.1 mmol/L, hip circumference and vegetable servings. Coefficients are
in `lasso_coefficients.csv` (log-odds per standard deviation for numeric variables,
against the reference level for categorical ones).

Across the five training folds, nine variables were selected every time: age, weight,
waist, hip, cholesterol, heart rate, sex, region and residence. Vegetable servings were
selected in four folds. BMI and moderate-intensity work were never selected
(`selection_by_fold.csv`).

## Model without cholesterol

| Measure | Estimate |
| --- | --- |
| ROC AUC | 0.727 (folds 0.707 to 0.743) |
| Average precision | 0.426 |
| Brier score | 0.158 |
| Observed / expected | 1.000 |
| Calibration slope | 0.894 |

Thresholds: high sensitivity 0.130, balanced 0.220, high PPV 0.245. Without
cholesterol, LASSO kept 19 variables (`lasso_coefficients_nob8.csv`).

## Model comparison (Table 2)

Repeated stratified 2-fold cross-validation, 3 repeats; mean (SD) ROC AUC across the
six validation folds. Sensitivity, specificity, PPV and F1 use a threshold picked on the
same fold (sensitivity 0.70 to 0.75), so they are optimistic and only useful for ranking.

| Model | ROC AUC | Sensitivity | Specificity | PPV | F1 |
| --- | --- | --- | --- | --- | --- |
| LASSO | 0.729 (0.012) | 0.70 | 0.65 | 0.38 | 0.49 |
| Random forest | 0.709 (0.012) | 0.70 | 0.62 | 0.36 | 0.48 |
| Balanced random forest | 0.707 (0.008) | 0.71 | 0.60 | 0.35 | 0.47 |
| SVM | 0.706 (0.012) | 0.70 | 0.63 | 0.36 | 0.48 |
| LightGBM | 0.683 (0.009) | 0.70 | 0.58 | 0.34 | 0.45 |
| MLP | 0.676 (0.035) | 0.70 | 0.56 | 0.33 | 0.45 |
| XGBoost | 0.666 (0.011) | 0.70 | 0.54 | 0.32 | 0.44 |
| Gaussian naive Bayes | 0.663 (0.015) | 0.70 | 0.54 | 0.32 | 0.44 |
| Logistic regression | 0.628 (0.018) | 0.70 | 0.49 | 0.29 | 0.41 |

Logistic regression, SVM and MLP use pairwise interaction terms; the others use the base
predictors.

## Figures

| File | Content |
| --- | --- |
| `figures/figure-01-imputation-density.png` | Observed against imputed values |
| `figures/figure-02-importance-random-forest.png` | Permutation importance, random forest |
| `figures/figure-03-importance-lightgbm.png` | Permutation importance, LightGBM |
| `figures/figure-04-lasso-coefficients.png` | Non-zero LASSO coefficients, final model |
| `figures/figure-05-roc-oof.png` | Out-of-fold ROC curve |
| `figures/figure-06-precision-recall-oof.png` | Out-of-fold precision-recall curve |
| `figures/figure-07-confusion-matrix-oof.png` | Confusion matrix at 0.230 |
| `figures/figure-08-calibration-oof.png` | Calibration by decile of predicted risk |
| `figures/figure-09-threshold-sweep-oof.png` | Sensitivity, specificity, PPV and NPV by threshold |
