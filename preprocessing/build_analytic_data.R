#!/usr/bin/env Rscript
# Build the hypertension analysis dataset from the Uganda 2023 WHO STEPS extract.
#
# Usage:
#   Rscript preprocessing/build_analytic_data.R <steps_HTN.xlsx> <pgph.0003755.s002.dta> <out_dir>
#
# Inputs
#   steps_HTN.xlsx           survey extract (3,554 participants with a fasting glucose
#                            measurement); not public, available from the authors
#   pgph.0003755.s002.dta    public 2023 STEPS file (S2 Data, Kusolo et al. 2025,
#                            doi:10.1371/journal.pgph.0003755). Used only to add m7,
#                            current treatment for raised blood pressure, which the
#                            extract did not carry.
# Outputs (in out_dir)
#   htn_analytic.csv               imputation 1, the analysis dataset
#   htn_analytic_imputations.csv   all five imputations, long format (.imp = 1..5)
#   build_log.txt                  counts at each step
#   figure-01-imputation-density.png

suppressPackageStartupMessages({
  library(readxl)
  library(haven)
  library(dplyr)
  library(mice)
  library(lattice)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript build_analytic_data.R <steps_HTN.xlsx> <s002.dta> <out_dir>")
}
extract_path <- args[1]
s2_path <- args[2]
out_dir <- args[3]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

log_lines <- character(0)
note <- function(...) {
  msg <- paste0(...)
  log_lines <<- c(log_lines, msg)
  message(msg)
}

raw <- read_excel(extract_path)
s2 <- zap_labels(read_dta(s2_path))
note("extract rows: ", nrow(raw), "; public file rows: ", nrow(s2))

# 1. Add m7 from the public file. The two files use different participant IDs, so
#    records are linked on 18 measured values, which identify every extract row
#    uniquely. The script stops if any row fails to link or links twice.
keys <- c("age", "c1", "region", "status", "c5", "t1", "t12", "d1", "d3",
          "p16a", "p16b", "m5a", "m5b", "m6a", "m6b", "m11", "m12", "b5")
key_frame <- function(d) {
  k <- as.data.frame(lapply(d[keys], function(x) round(as.numeric(x), 2)))
  k[is.na(k)] <- -999
  k
}
a <- key_frame(raw)
a$ID <- raw$ID
b <- key_frame(s2)
b$m7 <- as.numeric(s2$m7)
b$.linked <- TRUE
stopifnot(!anyDuplicated(a[keys]), !anyDuplicated(b[keys]))
linked <- merge(a, b, by = keys, all.x = TRUE)
stopifnot(nrow(linked) == nrow(raw), !anyDuplicated(linked$ID), all(linked$.linked %in% TRUE))
note("extract rows linked to the public file: ", sum(linked$.linked %in% TRUE), " of ", nrow(raw))
raw <- merge(raw, linked[c("ID", "m7")], by = "ID", all.x = TRUE)

# 2. Recode missing-value codes. 77 means "don't know" or "not done" only in these
#    questionnaire and laboratory items. In blood pressure, heart rate, weight,
#    waist and hip, 77 is a real measurement and is kept.
code77 <- c("d1", "d3", "d4", "p16a", "p16b", "b8", "b5")
for (v in code77) {
  note("  ", v, ": ", sum(raw[[v]] == 77, na.rm = TRUE), " values of 77 set to missing")
  raw[[v]][raw[[v]] == 77] <- NA
}
note("  c5 = 88 (refused) set to missing: ", sum(raw$c5 == 88, na.rm = TRUE))
raw$c5[raw$c5 == 88] <- NA

# Implausible body size. Height outside 130 to 210 cm, or a BMI outside 12 to 60 kg/m2,
# is treated as an entry error: height and weight are set to missing and imputed.
bmi_raw <- raw$m12 / (raw$m11 / 100)^2
bad_size <- with(raw, !is.na(m11) & !is.na(m12) &
                   (m11 < 130 | m11 > 210 | bmi_raw < 12 | bmi_raw > 60))
note("  implausible height or BMI, height and weight set to missing: ", sum(bad_size))
raw$m11[bad_size] <- NA
raw$m12[bad_size] <- NA
# Waist or hip below 50 cm is not possible in an adult; set to missing and impute.
for (v in c("m14", "m15")) {
  note("  ", v, " below 50 cm set to missing: ", sum(raw[[v]] < 50, na.rm = TRUE))
  raw[[v]][!is.na(raw[[v]]) & raw[[v]] < 50] <- NA
}

# Skip patterns. d4 (vegetable servings on a vegetable day) is not asked when d3 = 0
# days; those participants eat no vegetables, so d4 is 0, not missing.
note("  d4 set to 0 where d3 = 0 (question skipped): ", sum(raw$d3 == 0 & is.na(raw$d4), na.rm = TRUE))
raw$d4[!is.na(raw$d3) & raw$d3 == 0] <- 0

# t8 ("In the past, did you ever smoke any tobacco products?") is asked only when
# t1 = 2 (not a current smoker). t1 and t8 together give smoking status:
# 1 never, 2 former, 3 current.
raw$smoke <- with(raw, ifelse(t1 == 1, 3L, ifelse(t8 == 1, 2L, ifelse(t8 == 2, 1L, NA_integer_))))
note("  smoking status from t1 and t8: never ", sum(raw$smoke == 1, na.rm = TRUE),
     ", former ", sum(raw$smoke == 2, na.rm = TRUE), ", current ", sum(raw$smoke == 3, na.rm = TRUE),
     ", missing ", sum(is.na(raw$smoke)))

# Education grouped to four levels: 1 no formal schooling (c5 = 1); 2 primary (c5 = 2);
# 3 secondary (O-level, junior secondary or A-level, codes 3 to 5); 4 tertiary (6 to 7).
raw$edu <- cut(raw$c5, breaks = c(0, 1, 2, 5, 7), labels = FALSE)

# 3. Exclusions: missing either of the blood pressure readings used for the outcome
#    (readings 2 and 3) or missing the treatment question.
keep <- with(raw, !is.na(m5a) & !is.na(m5b) & !is.na(m6a) & !is.na(m6b) & !is.na(m7))
note("excluded for missing BP reading 2 or 3 or treatment status: ", sum(!keep))
dat <- raw[keep, ]

cat_vars <- c("c1", "region", "status", "edu", "smoke", "t12",
              "p1", "p4", "p7", "p10", "p13")
num_vars <- c("age", "d1", "d3", "d4", "p16a", "p16b", "m16a", "m16b", "m16c",
              "m11", "m12", "m14", "m15", "b8", "b5")
outcome_parts <- c("m5a", "m5b", "m6a", "m6b", "m7")
dat <- dat[c("ID", cat_vars, num_vars, outcome_parts)]
dat[cat_vars] <- lapply(dat[cat_vars], factor)
note("analysis sample before imputation: ", nrow(dat))
note("missing values by variable: ",
     paste(names(which(colSums(is.na(dat)) > 0)),
           colSums(is.na(dat))[colSums(is.na(dat)) > 0], sep = "=", collapse = ", "))

# 4. Multiple imputation by chained equations: predictive mean matching for numeric
#    variables, logistic or polytomous regression for categorical ones. ID and the
#    outcome components are not used to impute anything.
meth <- make.method(dat)
pred <- make.predictorMatrix(dat)
pred[, c("ID", outcome_parts)] <- 0
# d4 is only missing for people who eat vegetables on some days (the skip-pattern zeros
# are filled above), so imputed servings are kept between 1 and 20. Without this,
# predictive mean matching can borrow a skip-pattern 0.
post <- make.post(dat)
post["d4"] <- "imp[[j]][, i] <- squeeze(imp[[j]][, i], c(1, 20))"
imp <- mice(dat, method = meth, predictorMatrix = pred, post = post, m = 5, maxit = 50,
            seed = 123, printFlag = FALSE)
note("mice ", as.character(packageVersion("mice")), "; methods: ",
     paste(names(meth[meth != ""]), meth[meth != ""], sep = "=", collapse = ", "))

if (!is.null(imp$loggedEvents)) {
  ev <- imp$loggedEvents
  note("mice logged events: ", nrow(ev), "; by action and variable: ",
       paste(names(table(paste(ev$meth, ev$out))), table(paste(ev$meth, ev$out)),
             sep = "=", collapse = "; "))
}

# Density plot only for variables with enough imputed values to draw a curve.
imputed_num <- names(which(colSums(is.na(dat[num_vars])) >= 10))
png(file.path(out_dir, "figure-01-imputation-density.png"), width = 1800, height = 1100, res = 200)
print(densityplot(imp, as.formula(paste("~", paste(imputed_num, collapse = " + ")))))
invisible(dev.off())

# 5. Derive analysis variables. The outcome follows the WHO STEPS definition:
#    mean of readings 2 and 3 of SBP >= 140 or DBP >= 90 mmHg, or current
#    treatment with prescribed drugs for raised blood pressure.
derive <- function(d) {
  d %>%
    mutate(
      across(all_of(cat_vars), ~ as.integer(as.character(.x))),
      SBP = (m5a + m6a) / 2,
      DBP = (m5b + m6b) / 2,
      bp_meds = as.integer(m7 == 1),
      hypertension = as.integer(SBP >= 140 | DBP >= 90 | m7 == 1),
      # keep the skip rule when d3 itself was imputed as 0
      d4 = ifelse(d3 == 0, 0, d4),
      bmi = m12 / (m11 / 100)^2,
      HR = (m16a + m16b + m16c) / 3,
      sit_hours = p16a + p16b / 60,
      fbg61 = as.integer(b5 >= 6.1)
    )
}
keep_cols <- c("pid", "hypertension", "SBP", "DBP", "bp_meds",
               "age", "c1", "region", "status", "edu", "smoke", "t12",
               "p1", "p4", "p7", "p10", "p13", "fbg61",
               "d1", "d3", "d4", "sit_hours", "m11", "m12", "m14", "m15", "b8", "bmi", "HR")

all_imps <- lapply(1:5, function(k) {
  d <- derive(complete(imp, k))
  d$pid <- seq_len(nrow(d))
  d$.imp <- k
  d
})
first <- all_imps[[1]]
write.csv(first[keep_cols], file.path(out_dir, "htn_analytic.csv"), row.names = FALSE)
long <- do.call(rbind, lapply(all_imps, function(d) d[c(".imp", keep_cols)]))
write.csv(long, file.path(out_dir, "htn_analytic_imputations.csv"), row.names = FALSE)

by_bp <- with(first, SBP >= 140 | DBP >= 90)
note("final N: ", nrow(first), "; hypertension: ", sum(first$hypertension),
     " (", sprintf("%.1f", 100 * mean(first$hypertension)), "%)")
note("  raised measured BP: ", sum(by_bp), "; on treatment: ", sum(first$bp_meds),
     "; on treatment with measured BP below 140/90: ", sum(first$bp_meds == 1 & !by_bp))
note("  fasting glucose >= 6.1 mmol/L (imputation 1): ", sum(first$fbg61))
writeLines(log_lines, file.path(out_dir, "build_log.txt"))
