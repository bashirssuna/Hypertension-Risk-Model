# Hypertension risk model for Ugandan adults

Code, data and model files for a hypertension screening model built on the Uganda 2023
WHO STEPS survey. The model estimates the probability that an adult aged 18 to 69 years
has hypertension from age, sex, body measurements, heart rate, cholesterol, fasting
glucose and a few lifestyle questions. It does not use a blood pressure reading, because
its job is to decide who should get one first.

The model is L1-penalised (LASSO) logistic regression with isotonic calibration. All
performance figures are out-of-fold estimates from nested 5-fold cross-validation, with
variable selection, tuning and calibration repeated inside every fold.

## Main results

3,554 participants, 825 (23.2%) with hypertension (WHO STEPS definition: mean of
readings 2 and 3 at least 140/90 mmHg, or on prescribed blood pressure drugs).

| Measure | Estimate (95% CI) |
| --- | --- |
| ROC AUC | 0.730 (0.710 to 0.749) |
| Average precision | 0.432 (0.397 to 0.466) |
| Brier score | 0.157 (0.150 to 0.164) |
| Calibration slope | 0.88 (0.76 to 0.98) |
| Sensitivity / specificity at 0.230 | 0.73 / 0.62 |
| PPV / NPV at 0.230 | 0.37 / 0.88 |

Predicted risk below 0.115 is low (31% of participants, 8.3% of them with
hypertension); 0.235 or above is high (45% of participants, holding 71% of all cases).
A version without cholesterol reaches a ROC AUC of 0.727. Full results, including the
nine-model comparison, are in [results/RESULTS.md](results/RESULTS.md); how they were
produced and what they cannot tell you is in
[docs/EVALUATION_NOTES.md](docs/EVALUATION_NOTES.md).

This is an internally validated research model. It has not been tested outside the
survey it was built on and is not a diagnostic test.

## Repository layout

```
data/            analysis datasets (de-identified) and data dictionary
preprocessing/   R script that builds the datasets from the survey extract
analysis/        Python scripts: model comparison, evaluation, tables, figures
results/         results.json, tables, out-of-fold predictions, figures
app/             Streamlit screening app and the two trained model files
docs/            evaluation notes
```

## Reproduce the results

Python 3.14 with the versions pinned in `requirements.txt`:

```
python -m venv .venv
.venv\Scripts\activate            (on macOS or Linux: source .venv/bin/activate)
pip install -r requirements.txt
python analysis/model_comparison.py
python analysis/run_analysis.py
python analysis/make_tables.py
python analysis/make_figures.py
```

`run_analysis.py` takes about 10 minutes and `model_comparison.py` about 10 minutes on a
laptop. Seeds are fixed in `analysis/common.py`, so a rerun with the pinned versions
should reproduce `results/results.json`; multithreaded libraries can shift the last
decimal of the Table 2 numbers. Rebuilding the datasets themselves needs the survey extract,
which is not public (see [preprocessing/README.md](preprocessing/README.md)).

## Run the app

```
pip install -r app/requirements.txt
streamlit run app/HTNtool.py
```

The app asks whether a cholesterol result is available and loads the matching model. It
shows the predicted probability and a low, moderate or high risk band. The model files
were saved with scikit-learn 1.8.0 and need that version to load.

## Data

The analysis files in `data/` hold the 3,554 participants with only the variables the
model uses. IDs, village, district and survey weights are removed. Every row matches a
record in the public survey file released with Kusolo et al. (2025), PLOS Global Public
Health 5(4):e0003755, doi:10.1371/journal.pgph.0003755. Twelve survey items (past
smoking, physical activity, waist, hip, heart rate and cholesterol) are not in that public
file; they come from the study extract and are shared with the survey custodians'
permission. Column definitions and sources are in [data/README.md](data/README.md).

## Citation

If you use this code, data or model, cite the repository (see `CITATION.cff`) and the
Kusolo et al. (2025) survey paper.

## Licence

Code: MIT (see `LICENSE`). Data: the columns that also appear in the public survey file
follow its CC BY 4.0 licence; the other columns are shared with the survey custodians'
permission (see [data/README.md](data/README.md)).

## Contact

Bashir Ssuna, Makerere University Hospital, Department of Research, Kampala, Uganda.
