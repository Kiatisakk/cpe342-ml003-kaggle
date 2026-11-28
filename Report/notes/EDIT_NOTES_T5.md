# Task 5 report edits

## EDA additions

- Added a comparison table for the `acckda` box. The existing figure plots four-period means, which are useful for seeing the cluster. The detector checks each observed period value in all three metrics. This explains why an average inside an interval can hide an observed value outside it.
- Added a low-stability recovery table using the counts already in the report. It starts with 1,257 accounts missing `behavior_stability`, narrows to 129 candidates outside Stage 1 and inside the companion box, and identifies 53 posterior flags. The final union gains 52 accounts because one already matches `acckda`.

## Alignments with the notebooks

- Updated the pipeline diagram to show that recovery candidates are drawn from outside Stage 1. The output then combines Stage 1, `acckda`, and posterior recovery.
- Used the same missing-value convention in the EDA, detection rules, and diagram. The box checks every observed value and treats missing cells as compatible.
- Clarified the overlap counts and the breakdown caption. The five Stage 1 detectors are disjoint. Within the shock rule, 479 accounts meet both geographic and behavioural conditions. The box and recovery share one account.
- Described 14 as the assumed number of hidden low-stability members. The existing 12.6 estimate supports the approximate scale and the sensitivity figure shows the effect of changing the assumption.
- Described the box precision as conditional on the chance model and attributed the estimated shock subtypes to the research notes.

The new tables retain the counts and intervals already reported in `task5.tex`. The [final notebook](../../task5/task5.ipynb) and [EDA notebook](../../task5/task5_eda.ipynb) were used to verify the order of the steps and the behaviour of the detector.

## Figure and table titles

- Standardised all 21 Task 5 figure captions as descriptive titles. Figures with lettered panels have a separate title for each panel.
- Moved reading details into the report text, including the login-jump sample size, the denominator for security flag rates, and the marker convention in the gap scan. The figure titles use the existing plots and report values.
- Shortened the title of Table 3 and moved the data release and grouping details into the preceding explanation of the chance model.
- Shortened the Table 4 title and its seven scan names. Moved the scan settings and citations into the preceding text and kept the results in the table.
- Standardised all eight table captions as short titles. The text surrounding the results and baseline tables explains the reported metrics and comparison sets.
- Aligned all Task 5 table captions with the left text margin.

## EDA group layout

- Added an introductory sentence before the figure for Groups 3 through 6 so each run-in heading appears before its plot. Kept the introductory text and figure together for all seven groups.

## Report cleanup

- Removed the first-release identifier paragraph and its inline TODO from Common Mistakes and Failed Experiments.

## Pre-PR accuracy review

- Confirmed that the final and EDA notebooks identify the same accounts and reproduce the archived 2,104-account prediction file.
- Distinguished Stage 1 disjointness from overlap in the final union and described what the shuffled controls and box sensitivity tests support.
- Corrected the cross-release comparison to distinguish recurring groups from changing group sizes. Marked the hidden-member and F3 figures as research estimates.
- Clarified the leaderboard probes, the small gain from one threshold change, and the agreement measured by the baseline detectors.
- Kept the gap-scan plot and its interpretation on the same page.
