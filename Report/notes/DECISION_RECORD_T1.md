# Decision Record: Task 1 Report Refinement

- **Date:** 2026-11-27
- **Author:** Siriwan (Team ML003)
- **Target File:** `Report/task1/task1.tex`
- **Branch:** `improve-task1-report`
- **Pull Request Target:** `Kiatisakk/cpe342-ml003-kaggle:main`

---

## 1. Context & Review Goals
Following our internal audit of the compiled Task 1 technical report (`task1.pdf`, 12 pages), we evaluated the document against academic grading standards, aesthetic layout balance, and the rules in `SPEC.md`. 

While the quantitative analysis, feature engineering bounds, and decision log were exceptionally strong, two key structural imbalances were identified:
1. **Section 8 (Domain Interpretation & Business Impact) was disproportionately brief (129 words):** It did not sufficiently explain how the aggressive $F_2$-optimal threshold (flagging 59.81% of test accounts) can be realistically deployed in live game operations without causing disastrous false-positive bans.
2. **Layout & Figure Scaling Inefficiencies:** Certain visual elements, specifically Figure 1 (a simple binary class balance chart), consumed an excessive vertical portion of the page (`height=0.35\textheight`), adding unnecessary page bulk.
3. **Metric & Error Analysis Integrity:** We verified that:
   - Section 1 does **not** need a boilerplate mathematical definition of $F_2$ because the conceptual rationale (recall weighting to suppress missed cheaters) is already lucid and concise.
   - Section 6 already possesses rigorous mathematical error analysis (`Where the errors concentrate` and `How far can the score go?`). We intentionally refrained from fabricating qualitative personas (e.g., "pro-player false positives"), maintaining strict compliance with `SPEC.md` against synthetic data hallucination.

---

## 2. Changes Made (Summary of Modifications)

### A. Substantial Expansion of Section 8 (Business Impact)
Expanded Section 8 into three structured, domain-grounded engineering paragraphs:
1. **Tiered Anti-Cheat Operational Architecture:**
   - Clarified that the $F_2$-optimized threshold operates as an aggressive **Tier-1 Statistical Pre-Filter** rather than an automated ban executioner.
   - Formulated a 3-tier graduated enforcement pipeline:
     - *Soft containment:* Silent routing to low-trust matchmaking pools (shadow quarantine).
     - *Dynamic friction gates:* Progressive challenges (e.g., behavioral CAPTCHA or out-of-band mobile authentication).
     - *Forensic triage:* High-confidence detections ($> 0.70$) dispatched directly to server-side telemetry inspection and human review queues.
2. **Asymmetric Cost-Benefit Trade-off:**
   - Provided the economic justification for binary $F_2$ ($\beta = 2$): an undetected cheater degrades the match experience for up to 9 legitimate players simultaneously, compounding into matchmaking distrust and long-term player churn (LTV destruction).
   - Showed that within a tiered pre-filter architecture, false positives incur only minor verification latency, making high recall economically optimal.
3. **Operational Drift & Telemetry Health Monitoring:**
   - Linked real-world operational reliability back to the empirical missingness findings in Figure 10 (`fig:t1-missing-auc`), where missing key features drops AUC from 0.922 to 0.831.
   - Recommended real-time telemetry health monitoring to detect covariate drift caused by client balance patches.

### B. Visual Layout & Figure Scaling Optimizations
- **Figure 1 (`class_balance.png`):** Reduced maximum vertical allocation from `0.35\textheight` to `0.22\textheight` with proportional width constraint (`width=0.78\linewidth`). This significantly tightens vertical spacing and improves typography density.
- **Figure 9 (`fig_t1_masking_gain.png`):** Decoupled the duplicate panel (a) (which previously reprinted the uniform distribution of ratio $r$ identical to Figure 5(a)). Cropped to display solely panel (b) (`fig_t1_masking_gain.png`, `width=0.62\linewidth`), focusing exclusively on the missingness-dependent OOF ROC-AUC masking gain and eliminating redundancy.

---

## 3. Verification & Build Results
- Compiled locally using **XeLaTeX (MiKTeX 25.12)**:
  ```bash
  cd Report/task1 && xelatex -interaction=nonstopmode task1.tex
  ```
- **Result:** Successfully compiled with 0 errors (`Output written on task1.pdf`).
- All cross-references (`\ref`, `\label`) and document structure were preserved without breaking standalone compilation or `main.tex` compilation.

---

## 4. Teammate Review Checklist
- [ ] Verify that Section 8 accurately reflects our team's operational strategy for the Kaggle submission.
- [ ] Confirm visual appearance in `task1.pdf` and check layout flow around Figure 1 and Section 8.
- [ ] Ensure no merge conflicts against `main`.
