# Evaluation notes

This file records how the model was built and tested, the choices behind each step, and
what the evaluation cannot tell you. The numbers themselves are in
`../results/RESULTS.md` and `../results/results.json`.

## Outcome and predictors

Hypertension follows the WHO STEPS definition: mean systolic pressure of at least
140 mmHg or mean diastolic pressure of at least 90 mmHg over readings 2 and 3, or
current use of prescribed blood pressure drugs. Treated participants count as cases even
when their measured pressure is below 140/90.

Blood pressure is never a predictor. The tool is meant for settings where pressure has
not yet been measured, so the 24 candidate predictors are things a health worker can ask
or measure without a cuff: age, sex, region, residence, education, smoking status,
smokeless tobacco, five physical activity items, sitting time, fruit and vegetable
intake, height, weight, BMI, waist, hip, resting heart rate, total cholesterol and
fasting glucose of at least 6.1 mmol/L. Heart rate comes from the same device and
session as blood pressure; a user without that device needs another way to measure it.

`fbg61` is fasting blood glucose of at least 6.1 mmol/L. It includes people with
diabetes-range values (7.0 mmol/L or more), so it is not impaired fasting glucose in the
strict sense, and the code and tables label it by its cut-off instead.

## Model

The model is L1-penalised (LASSO) logistic regression with balanced class weights.
Numeric predictors are standardised; categorical predictors are one-hot coded with the
first level as reference. The penalty strength is chosen from 10 values by 5-fold
cross-validation on ROC AUC. Variables with a non-zero coefficient are kept. The model is
then wrapped in isotonic calibration (3-fold), because class weighting pushes the raw
probabilities away from the observed risk.

The deployed model (`app/lasso_htn_full.joblib`) is built in two steps on all 3,554
participants: LASSO selects the variables, then the calibrated model is refit on those
variables only. A second model (`app/lasso_htn_nob8.joblib`) does the same without
cholesterol, for settings with no lipid test.

## How performance was estimated

All headline performance numbers are out-of-fold (OOF). The same two-step procedure
that builds the deployed model (selection, then the calibrated refit, with penalty tuning
in both) runs inside each of 5 outer cross-validation folds, and only the held-out fold is
predicted. Pooling the five
held-out folds gives one honest prediction per participant (`results/oof_predictions.csv`).
An earlier version of this work selected variables on the full dataset before
cross-validation, which leaks information and inflates the estimates; that is fixed here.

Which variables LASSO keeps changes from fold to fold. `results/selection_by_fold.csv`
shows which variables were selected in each training fold, taken from the same fits that
produced the OOF predictions. Variables kept in all five folds are the stable core
of the model; the rest are kept or dropped depending on the training sample.

Three thresholds are chosen on the OOF predictions over a grid of 0.005:

- **High sensitivity:** the highest sensitivity with a PPV of at least 0.30.
- **Balanced:** the highest geometric mean of sensitivity and specificity.
- **High PPV:** the highest PPV with a sensitivity of at least 0.70.

The app reports low risk below the high-sensitivity threshold, high risk at or above the
high-PPV threshold, and moderate risk in between.

The 95% confidence intervals come from 2,000 bootstrap resamples of the OOF predictions.
Sensitivity, specificity, PPV and NPV are computed at the fixed cut-offs the app uses, so
their intervals show sampling variability at those cut-offs. Because the cut-offs were
chosen on the same OOF predictions, the point estimates at them are slightly optimistic.
`results.json` therefore also reports a cross-fitted check (`crossfit_thresholds`): the
thresholds are chosen on four folds' OOF predictions and applied to the fifth, and the
pooled result shows what the threshold rules deliver on data they were not chosen on. The
thresholds are also re-chosen in each bootstrap resample to show how far they move
(`*_threshold` entries in `ci95`). None of the intervals include the variability of
refitting the model, which would need the nested cross-validation repeated inside each
resample.

Calibration is summarised by the Brier score, the ratio of observed to expected cases
(O/E) and the calibration slope (logistic regression of the outcome on the logit of the
predicted probability).

Two sensitivity analyses repeat the full OOF evaluation: without cholesterol, and on
each of the five imputed datasets.

## Model comparison (Table 2)

`analysis/model_comparison.py` compares nine classifiers with the protocol of the
original model-discovery work: repeated stratified 2-fold cross-validation (3 repeats).
In each validation fold the threshold is the one with the highest specificity among
thresholds that give a sensitivity of 0.70 to 0.75. Because that threshold is picked on
the same fold it is scored on, the sensitivity, specificity, PPV and F1 in Table 2 are
optimistic and only useful for ranking the models. The ROC AUC does not depend on a
threshold and is not affected.

Permutation importance for the random forest and LightGBM (Figures 2 and 3) is the drop
in ROC AUC when one column is shuffled, computed on the data the model was fit on. It
ranks variables by how much each model leans on them. It is not a measure of causal
effect, and correlated variables (height, weight and BMI; waist and hip) share
importance in ways that make any single one look weaker than the group.

## What this evaluation does not cover

- **No external validation.** All estimates come from internal cross-validation in one
  survey. Performance in clinics, in other years, or in other countries is unknown.
- **Survey design ignored.** STEPS uses multistage cluster sampling with weights. The
  model is fit and tested on the unweighted sample, so prevalence, PPV and NPV describe
  this sample, not the Ugandan adult population. Cross-validation folds split
  individuals, not clusters, so people from the same village can be in both training and
  test folds, which may make the estimates slightly optimistic.
- **Sampling frame.** The extract holds the 3,554 participants who had a fasting glucose
  measurement, which is a subset of the 3,694 records in the public file. People who
  refused or missed the blood test are not represented.
- **Imputation.** The main results use imputation 1; the other four imputations give
  the sensitivity range reported in `results.json`. Imputation was run once on all
  participants before cross-validation. The outcome and the blood pressure values were
  not used in the imputation models, so no outcome information reaches the test folds,
  but the test folds did contribute to the imputation models for predictors.
- **Cross-sectional data.** The model identifies people who have hypertension now. It
  does not predict who will develop it.
- **Cholesterol floor.** The survey meter recorded total cholesterol only from 2.59 to
  10.36 mmol/L, and 43.5% of raw values sit at 2.59. The model cannot tell a value of
  2.59 from anything lower, which weakens the cholesterol signal at the low end.
- **Measurement.** Blood pressure was measured on one day. A single-visit definition
  overcounts hypertension compared with a diagnosis confirmed over several visits, and
  a model trained on it inherits that.
