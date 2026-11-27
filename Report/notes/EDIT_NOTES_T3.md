# Task 3 report edits

## EDA changes

- Retained the four existing EDA figures. They cover the same questions as the EDA notebook and already add useful detail, including the zero-spending rate by segment and the log-residual distribution by segment.
- Kept the positive-spending mean at **20,235 THB**. The recomputed mean is **20,234.88 THB**. The figure text now describes **300,000 THB** as the observed maximum.
- Distinguished the share of total spending in segment 3 from its share of OOF absolute error.
- Clarified that the algebraic checks measure agreement **within 0.01** in the respective units on complete rows. This is separate from the missing-rate reduction after filling. The identity plot uses log-scaled axes.
- Gave panels (a) and (b) separate titles in each of the four EDA captions. The report text explains each panel, including the previous-month subset on the left of the offset figure and the full offset rule on the right.
- Added the offset-source table immediately after that figure and linked it to the EDA-to-preprocessing summary.
- Clarified that the segment imputer is fitted before the outer folds and does not use spending as a target. The `p_whale` feature is a 0/1 indicator for an observed segment and a predicted probability when the segment is missing.

## Offset coverage

### Why add this table?

The left panel of the offset figure shows how previous-month spending relates to the target when that earlier value is available. It leaves out the training payers whose previous-month spending is missing. The table shows what the amount model uses for those rows and how often each fallback is needed.

Historical spending supplies the offset for **3,112 of the 3,125** training payers with no previous-month value. Only **13** use the constant. This connects the offset choice in EDA to the actual preprocessing and quantifies coverage. Evaluating prediction accuracy for each offset source would require a separate error breakdown.

The counts use the final notebook's `exact_fill` and the order implemented by `prev_offset`. For each row, use the previous-month value if present after filling. Otherwise use the historical-spending fallback if available, or the constant if both values are missing.

| Offset source | Train payers | All train | All test |
|---|---:|---:|---:|
| `log1p(prev_month_spending)` | 50,826 | 97,985 | 24,357 |
| `log1p(historical_spending) - 1.05` | 3,112 | 5,988 | 1,528 |
| Constant `6.5` | 13 | 27 | 4 |
| Total | 53,951 | 104,000 | 25,889 |

The payer column filters on `spending_30d > 0`. The test target is unavailable, so the test column includes all rows. Training payers are the population used to fit the amount model.

## Where the numbers come from

- [Final Task 3 notebook](../../task3/task3.ipynb) · the preprocessing cell defines `exact_fill`, `add_feats`, `prev_offset` and the segment imputer.
- [Task 3 EDA notebook](../../task3/task3_eda.ipynb) · the four analyses used to check coverage against the report.
- [Report figure generator](../scripts/gen_figs_t3.py) · `fig_target`, `fig_whales`, `fig_identities` and `fig_offset` implement the existing EDA plots.
- Offset coverage and the positive-spending mean were recomputed from the local competition splits under `Kaggle/cpe342-kaggle-main/03_regression/` using the final notebook's fill rule. No model fitting is needed to reproduce these counts.

## Data Preprocessing

- Clarified that each algebraic fill pass evaluates all six rules before writing any values. Later passes can use values recovered in earlier passes.
- Added `prev_x_freq = prev_month_spending * purchase_frequency` and named the spending proxy `exp_monthly_spend`. These are separate inputs derived from purchase frequency.
- Specified the calculation order for `n_missing`. It counts missing entries after algebraic fill and base feature engineering, before dropping `player_id` from the predictors and before imputing `segment`.
- Documented the segment imputer's stratified 90% fit and 10% early-stopping split. The notebook reports 92.92% accuracy on the early-stopping split. It uses the fitted classifier for missing segments in both train and test and records the results in `seg_imputed` and `p_whale`.
- Recorded the feature counts for Stage 1 (45) and Stage 2 (49). Named the four additional log-derived features and linked the Stage 2 target to the existing offset table.
- Clarified that the five shuffled folds are stratified on zero spending and four quartiles of ranked positive spending, using seed 42.

These edits follow the preprocessing code and saved feature counts in the [final Task 3 notebook](../../task3/task3.ipynb).

## Model Design

- Gave Figure 5 separate titles for the MAE-optimal payer quantile and the calibration plot. Moved the calibration bin counts and marker-size explanation into the report text.

## Figure captions

- Standardised all 13 Task 3 captions as short figure titles, with separate titles for panels labelled (a) and (b). Moved reading details and results into the surrounding text where needed.
- Corrected the gain across the final three steps of the milestone chart to 0.00144, using the plotted endpoints of 0.79391 and 0.79535.
