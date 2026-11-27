# Decision Record: Task 2 Report Refinement

- **Date:** 2026-11-27
- **Author:** Siriwan (Team ML003)
- **Target File:** `Report/task2/task2.tex`
- **Branch:** `improve-task2-report`
- **Pull Request Target:** `Kiatisakk/cpe342-ml003-kaggle:main`

---

## 1. Context & Review Goals
Following our internal audit of Task 2 of the final report (`Report/task2/task2.tex`, 12 pages), we evaluated the document against academic grading criteria, content conciseness, visual necessity, and strict alignment with `SPEC.md`.

Task 2 is a 4-class player segmentation problem evaluated on Macro $F_1$. Our team's core finding was that features exhibit near-zero conditional correlation given the class ($|r| \approx 0.007$), directly matching the Naive Bayes conditional independence assumption and allowing a non-parametric histogram Naive Bayes model with EM label noise correction to outperform standard gradient boosted trees (LightGBM/XGBoost).

During review, we identified the following points for refinement:
1. **Redundant Visual (Too Much):** Figure 6 (`fig_t2_progression.png`) duplicated the exact numerical progression already presented in Table 3 (`tab:t2-results`). The text itself stated *"Table 3 lists the out-of-fold macro F1... Figure 6 draws the same numbers."* Removing Figure 6 eliminates unnecessary visual clutter and streamlines the narrative flow.
2. **Missing Baseline in Headline Table (Too Little):** Table 3 presented model components (MLP, Spline LR, EM-NB, Blend, Final Rule) but omitted the tuned LightGBM baseline (`0.75240`, $\Delta = -0.00526$). Adding the baseline directly into Table 3 provides immediate comparative context on why Naive Bayes was chosen over Tree-based methods.
3. **Discrete/Categorical Feature Handling (Too Little):** Section 3 (Data Preprocessing) explained continuous histogram binning well, but was terse regarding discrete/categorical columns. We clarified that discrete features use empirical probability mass functions (PMFs) with Laplace smoothing $\alpha=1$, whereas discriminative models (Spline LR and MLP) require one-hot encoding after median imputation and quantile transformation.
4. **Fold Stability Metric (Too Little):** Section 5 presented the final OOF Macro $F_1$ score (0.75826) and fold ranges for NB, but lacked the overall fold-level stability metric ($\text{Mean} \pm \text{Std} = 0.7583 \pm 0.0026$). Adding this metric confirms the robustness of the final blend + affine rule across folds.
5. **Decision Log Float Page Budget:** Figure 7 (`decision_log_t2.pdf`) previously used `\begin{figure}[p]` with `height=0.85\textheight`, occupying a dedicated float page and leaving loose vertical gaps. Adjusting it to `\begin{figure}[H]` and `height=0.68\textheight` integrates the chart seamlessly into the text flow.
6. **Task 3 Reference Policy:** We verified that the brief mention of Task 3 in Section 7 is contextually essential (explaining that `player_id` was tested across tasks but cross-task label agreement was only 28%, proving independent generator draws), so it was intentionally retained.

---

## 2. Changes Made (Summary of Modifications)

### A. Results Streamlining & Baseline Integration (Section 5)
- **Table 3 (`tab:t2-results`):** Added the baseline benchmark row:
  - `Baseline (Tuned LightGBM) & 0.75240 & $-0.00526$ \\`
- **Removed Redundant Figure 6 (`fig_t2_progression.png`):** Deleted the redundant bar chart of model progression and updated the introductory sentence:
  - *From:* `"Table~\ref{tab:t2-results} lists the out-of-fold macro F1 of each component and of the final prediction; Figure~\ref{fig:t2-progress} draws the same numbers."*
  - *To:* `"Table~\ref{tab:t2-results} lists the out-of-fold macro F1 of the baseline, each component, and the final prediction."*
- **Cross-Validation Stability:** Added the exact 5-fold stability metric to the narrative:
  - `"The final pipeline reaches an OOF macro F1 of 0.75826 with high fold-to-fold stability (fold mean $0.7583 \pm 0.0026$)."`

### B. Clarification of Feature Preprocessing (Section 3)
- Explicitly contrasted categorical and discrete feature representations between the generative and discriminative models:
  - **Naive Bayes:** Discrete features utilize empirical categorical PMFs with Laplace smoothing $\alpha=1$ (plus explicit `NA` tokens), avoiding arbitrary ordinality assumptions.
  - **Spline LR & MLP:** Categorical columns (\feat{engagement_level}, \feat{loyalty_tier}) are converted to dense indicators via one-hot encoding alongside median imputation and quantile normal transformation.

### C. Layout & Figure 7 Optimization
- Adjusted Figure 7 (`decision_log_t2.pdf`) from float specifier `[p]` with `height=0.85\textheight` to in-flow specifier `[H]` with `height=0.68\textheight`, improving page utilization and reading continuity.

---

## 3. Verification & Build Results
- Compiled locally using **XeLaTeX (MiKTeX 25.12)**:
  ```bash
  cd Report/task2 && xelatex -interaction=nonstopmode task2.tex && xelatex -interaction=nonstopmode task2.tex
  ```
- **Result:** Successfully compiled with 0 errors (`Output written on task2.pdf (12 pages)`).
- Document pagination remained precisely within the 12-page target, with clean float placement and no dangling headings or broken cross-references.

---

## 4. Teammate Review Checklist
- [ ] Confirm that Table 3 clearly highlights the performance delta between Tuned LightGBM baseline and EM Naive Bayes blend.
- [ ] Verify that removing Figure 6 improves text flow without losing any unique information.
- [ ] Inspect `task2.pdf` around Section 3, Section 5, and the Decision Log figure.
- [ ] Ensure smooth merge into `main`.
