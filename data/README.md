# Data

Two files, built by `../preprocessing/build_analytic_data.R`:

| File | Rows | SHA-256 |
| --- | --- | --- |
| `htn_analytic.csv` | 3,554 (imputation 1) | `57f677cba262e9821b838f9d5baec427182a35b47e96bee8e8a6becbd097afd3` |
| `htn_analytic_imputations.csv` | 17,770 (5 x 3,554) | `18d9452bb54a237b55dcbbe7ea3653a3ec4abbc1b0e63902e93a7a20af95e705` |

`results/results.json` records the SHA-256 of the file each run used, so you can check
that your copy matches the published results.

## Source and licence

The participants are the 3,554 adults aged 18 to 69 years in the Uganda 2023 WHO STEPS
survey who had a fasting blood glucose measurement. Every row corresponds to a record in
the public survey file released with Kusolo et al. (2025):

> Kusolo R, Mutungi GN, Mbuliro M, Kajjura R, Wesonga R, Bahendeka SK, Guwatudde D.
> Changes in the prevalence of the common risk factors for non-communicable diseases in
> Uganda between 2014 and 2023: informed by nationally representative cross-sectional
> surveys. PLOS Global Public Health. 2025;5(4):e0003755.
> doi:10.1371/journal.pgph.0003755 (survey data: S2 Data,
> doi:10.1371/journal.pgph.0003755.s002)

The values come from two sources. Most columns are also in that public file, which PLOS
releases under the Creative Commons Attribution 4.0 licence. Twelve survey items are not
in the public file and come from the study's own extract of the survey: past smoking
(`t8`, used in `smoke`), the five physical activity items (`p1`, `p4`, `p7`, `p10`,
`p13`), waist and hip circumference (`m14`, `m15`), the three heart rate readings (used in
`HR`) and total cholesterol (`b8`). These are shared here with the survey custodians'
permission, not under the PLOS licence. If you use these data, cite Kusolo et al. (2025)
and this repository.

The files hold only the variables the analysis uses. Participant ID, village, district
and the survey weights are removed. `pid` is a row number created by the script
and cannot be traced back to a survey ID.

Height, weight, waist, hip, cholesterol, fasting glucose, education, diet and sitting
time contain imputed values (see `../preprocessing/README.md` for the counts and the
method). Blood pressure, treatment status and the outcome are never imputed.

## Columns

| Column | Description | Coding |
| --- | --- | --- |
| `pid` | Row number | 1 to 3,554 |
| `hypertension` | Outcome (WHO STEPS) | 1 = mean SBP at least 140 or mean DBP at least 90 mmHg (readings 2 and 3), or on prescribed blood pressure drugs; 0 = no |
| `SBP`, `DBP` | Mean of readings 2 and 3 (mmHg) | Define the outcome; never used as predictors |
| `bp_meds` | Currently taking prescribed blood pressure drugs (survey item `m7`) | 1 = yes, 0 = no |
| `age` | Age (years) | 18 to 69 |
| `c1` | Sex | 1 = male, 2 = female |
| `region` | Region | 1 = Central, 2 = Eastern, 3 = Northern, 4 = Western |
| `status` | Residence | 1 = urban, 2 = rural |
| `edu` | Highest level of education completed (`c5`, grouped) | 1 = no formal schooling, 2 = primary, 3 = secondary (O-level, junior secondary or A-level), 4 = university or postgraduate |
| `smoke` | Smoking status (from `t1` and `t8`) | 1 = never, 2 = former, 3 = current |
| `t12` | Currently uses smokeless tobacco | 1 = yes, 2 = no |
| `p1` | Work involves vigorous-intensity activity | 1 = yes, 2 = no |
| `p4` | Work involves moderate-intensity activity | 1 = yes, 2 = no |
| `p7` | Walks or cycles at least 10 minutes to get to and from places | 1 = yes, 2 = no |
| `p10` | Vigorous-intensity sport, fitness or recreation | 1 = yes, 2 = no |
| `p13` | Moderate-intensity sport, fitness or recreation | 1 = yes, 2 = no |
| `fbg61` | Fasting blood glucose at least 6.1 mmol/L (`b5`) | 1 = yes, 0 = no |
| `d1` | Days per week eating fruit | 0 to 7 |
| `d3` | Days per week eating vegetables | 0 to 7 |
| `d4` | Servings of vegetables on a day vegetables are eaten | 0 when `d3` = 0 |
| `sit_hours` | Time sitting or reclining on a typical day (hours) | `p16a` + `p16b` / 60 |
| `m11` | Height (cm) | |
| `m12` | Weight (kg) | |
| `m14` | Waist circumference (cm) | |
| `m15` | Hip circumference (cm) | |
| `b8` | Total cholesterol (mmol/L) | Recorded range 2.59 to 10.36; 44% of values sit at the 2.59 floor (see note below) |
| `bmi` | Body mass index (kg/m2) | `m12` / (`m11` / 100) squared |
| `HR` | Resting heart rate (beats per minute) | Mean of three readings |

`htn_analytic_imputations.csv` has the same columns plus `.imp` (1 to 5).

The education groups follow the survey's answer options. The question asks for the
highest level completed and has no option for primary school started but not finished,
so group 1 probably also holds adults who left primary school early.

Total cholesterol was recorded only between 2.59 and 10.36 mmol/L (100 to 400 mg/dL).
In the raw extract 1,546 of 3,554 values (43.5%) are exactly 2.59, so a value of 2.59 means
"2.59 or lower". Eight values sit at the 10.36 ceiling.
