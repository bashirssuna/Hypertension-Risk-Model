# Preprocessing

`build_analytic_data.R` turns the survey extract into the two analysis files in
`../data/`. You only need it to rebuild those files; the Python analysis reads the CSVs
directly.

## Run

```
Rscript preprocessing/build_analytic_data.R data/raw/steps_HTN.xlsx data/raw/pgph.0003755.s002.dta <out_dir>
```

Inputs:

- `steps_HTN.xlsx`, the study extract of the Uganda 2023 WHO STEPS survey (3,554
  participants who had a fasting blood glucose measurement). It carries participant IDs
  and village and district names, so it is not in this repository. It is available from
  the corresponding author.
- `pgph.0003755.s002.dta`, the public STEPS file (S2 Data of Kusolo et al., PLOS Global
  Public Health 2025, doi:10.1371/journal.pgph.0003755). The script takes one variable
  from it, `m7` (currently taking drugs prescribed for raised blood pressure), which the
  extract does not contain.

Tested with R 4.5.2, readxl 1.4.5, haven 2.5.5, dplyr 1.1.4, mice 3.19.0 and
lattice 0.22.7. One run takes about 8 minutes on a laptop, almost all of it in `mice`.

## What the script does

1. **Adds treatment status.** The extract and the public file use different participant
   IDs, so the script links records on 18 measured values (age, sex, region, residence,
   education, two tobacco items, two diet items, sitting time, the four blood pressure
   values of readings 2 and 3, height, weight and fasting glucose). Every extract row
   links to exactly one public row; the script stops if any row fails to link or links
   twice.
2. **Recodes missing-value codes.** The value 77 means "don't know" only in `d1`, `d3`,
   `d4`, `p16a`, `p16b`, `b8` and `b5`, where it is set to missing. In blood pressure,
   heart rate, weight, waist and hip, 77 is a real measurement and is kept. (An earlier
   version of this pipeline treated 77 as missing everywhere and dropped 375 people whose
   diastolic pressure was 77 mmHg.) Education code 88 (refused) is set to missing.
3. **Removes implausible body size.** Height below 130 cm or above 210 cm, or a BMI
   below 12 or above 60 kg/m2, is treated as an entry error: height and weight are set to
   missing and imputed. Waist or hip below 50 cm is set to missing in the same way.
4. **Applies two skip patterns.** `d4` (vegetable servings on a day vegetables are
   eaten) is skipped when `d3` is 0 days, so it is set to 0 rather than imputed;
   imputed `d4` values for people who do eat vegetables are kept between 1 and 20. `t8`
   ("In the past, did you ever smoke?") is asked only of people who do not smoke now, so
   `t1` and `t8` are combined into one smoking status: never, former or current.
5. **Groups education** (`c5`) into four levels: no formal schooling (1), primary (2),
   secondary (O-level, junior secondary or A-level, 3 to 5) and tertiary (6 and 7).
6. **Excludes** anyone missing blood pressure readings 2 or 3 or the treatment question.
   In this extract nobody is excluded.
7. **Imputes** the remaining missing predictor values with `mice` (5 imputations, 50
   iterations, seed 123; predictive mean matching for numeric variables, logistic or
   polytomous regression for categorical ones). The ID and the blood pressure and
   treatment variables that define the outcome are not used as predictors in the
   imputation models.
8. **Derives** the analysis variables in each imputed dataset:
   - `SBP`, `DBP`: mean of readings 2 and 3 (WHO STEPS method)
   - `bp_meds`: 1 if `m7` = 1
   - `hypertension`: 1 if SBP is at least 140 mmHg, DBP is at least 90 mmHg, or
     `bp_meds` = 1
   - `bmi`: weight / height squared (kg/m2)
   - `HR`: mean of the three heart rate readings
   - `sit_hours`: `p16a` + `p16b` / 60 (hours plus minutes sitting on a typical day)
   - `fbg61`: 1 if fasting blood glucose is at least 6.1 mmol/L
   - `d4` is reset to 0 wherever an imputed `d3` is 0

Every count from these steps is written to `build_log.txt` (a copy is in
`../results/build_log.txt`). The density plot of observed against imputed values is
`../results/figures/figure-01-imputation-density.png`.

## Outputs

| File | Contents |
| --- | --- |
| `htn_analytic.csv` | Imputation 1; the dataset every main analysis uses |
| `htn_analytic_imputations.csv` | All five imputations stacked, with `.imp` = 1 to 5 |
| `build_log.txt` | Counts at each step |
| `figure-01-imputation-density.png` | Observed and imputed distributions |

The output files keep only the variables the analysis uses. Participant ID, village,
district and the survey weights are dropped; `pid` is a new row number with no link to
the survey ID.
