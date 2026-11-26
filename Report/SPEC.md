# SPEC for report authors (wave 2)

Reference submission: `sub_v90f_entropy_dualhead_md.csv`, public LB 0.87351 (1st). Report language: English.

## File ownership
| Agent | Owns (may edit) |
|---|---|
| A (wave 1, done) | `main.tex`, `preamble.tex`, `macros.tex`, `task1/`, `scripts/plot_style.py`, `data/decision_log_t1.csv` (initial), `README.md`, `SPEC.md` |
| B | `task2/`, `task3/`, `scripts/gen_figs_t2.py`, `scripts/gen_figs_t3.py`, `data/decision_log_t2.csv`, `data/decision_log_t3.csv` |
| C | `task4/`, `scripts/gen_figs_t4.py`, `data/decision_log_t4.csv` |
| D | `task5/`, `front/*`, `scripts/gen_decision_logs.py`, `data/decision_log_t1.csv`, `data/decision_log_t5.csv` |

**Do not edit `preamble.tex`, `macros.tex`, `main.tex`.** If you need a macro or package, report it; the coordinator adds it.
Numbers come only from the notebooks (`taskN/taskN.ipynb`), the repo README, or the CLAUDE.md log of the pipeline repo.
Unknown number: write `\todo{...}`, never guess.

## The 8 headings (identical text, this order, in every `taskN.tex`)
Each file starts with `\subsection{Task N: <name>}`, then exactly these `\subsubsection`s:
1 Problem and Data, 2 EDA Findings, 3 Data Preprocessing, 4 Model Design, 5 Evaluation and Results,
6 Insights from Model Behaviour, 7 Common Mistakes and Failed Experiments, 8 Domain Interpretation and Business Impact.
Use `\paragraph{...}` for finer structure (paragraphs are not numbered). Replace each `\todo{}` and `\placeholderfigure` skeleton with real content.
Skeleton file header: `\documentclass[../main.tex]{subfiles}`, `\begin{document}`, `\graphicspath{{figures/}{taskN/figures/}}`, ..., `\end{document}`; keep it so the file builds alone and in `main`.

## Macros (macros.tex)
| Macro | Example |
|---|---|
| `\feat{name}` monospace feature, breaks after `_` | `\feat{kill_death_ratio}` |
| `\todo{text}` visible orange marker | `\todo{add OOF F1}` |
| `\placeholderfig{shows}{decision}` fixed 4.6 cm frame | `\placeholderfig{class balance}{use stratified folds}` |
| `\placeholderfigure{label}{caption}{shows}{decision}` floated version | see skeletons |
| `evidencetable{caption}{label}` 2 columns, EDA evidence / Pipeline decision | `\begin{evidencetable}{Cap.}{tab:t2-evidence} Fig.~\ref{fig:t2-eda1} finding & decision \\ \end{evidencetable}` |
| TikZ `stage`, `model`, `decision`, `flow`, `note` | `\node[stage]{Raw data}; \draw[flow] (a)--(b); \node[note]{annotation};` see `task1/task1.tex` Figure "pipeline" |
| `\taskcolor{N}` colour name of task N | `fill=\taskcolor{3}!12` or `\textcolor{\taskcolor{2}}{...}` |
Colours also defined by name: `task1..task5`, `kept`, `dropped`, `ruleex`, `ink`, `inkmuted`, `gridline`.
Allowed pgf/tikz libraries: positioning, arrows.meta, fit, calc, shapes.geometric, shapes.misc, backgrounds. pgfplots (compat 1.18), siunitx, subcaption, float are loaded.
The TikZ styles use text width 4.3 cm; place three columns at x = 0, 5.4, 10.8 cm to fit the text width (see task1).

## Palette (validated with the dataviz skill; print-safe, one accent per task)
T1 blue `#257DE0`, T2 orange `#F47230`, T3 aqua `#00B894`, T4 violet `#6A4CE2`, T5 magenta `#EE7FAC`.
Decision logs: kept green `#008A17`, dropped grey `#8A8A8A`, rule-excluded red `#DC3C3D` drawn with hatch `///`.
Ink `#191919`, muted ink `#666666`, grid `#E7E7E7`, axis `#C2C2C2`. Validator result (adjacent pairs, light surface): CVD and normal-vision separation pass; contrast
below 3:1 for orange, aqua, magenta, so always add direct labels or a table and never put text in these colours on white.
The same values are in `macros.tex` and `scripts/plot_style.py` (`TASK_COLORS`, `STATUS_COLORS`, `apply_style()`); matplotlib figures must use them.

## Figure conventions
- PNG at 300 dpi or PDF, in `taskN/figures/`, include with `\includegraphics[width=\linewidth]{file}` (or 0.9) inside `figure[H]` with a `\label{fig:tN-...}`.
- Caption = what it shows + one-line takeaway. Every figure is cited with `\ref` in the text and followed by "what we see, so what we decided".
- Labels: `fig:tN-...`, `tab:tN-...`, `sec:tN`. Do not use `\hypertarget`.
- Tables: `table[H]` with the caption above; avoid `longtable` inside `\small` groups (caused a glue error).
- Pipeline diagram in section 4 (TikZ), decision-log chart in section 5 (`fig:tN-decision-log`, file `figures/decision_log_tN.pdf`, made by `scripts/gen_decision_logs.py` from the CSV).

## Decision-log CSV schema (`data/decision_log_tN.csv`)
Columns `experiment,metric,delta,status,source,note`. `status` in `{kept, dropped, rule_excluded}`. `metric` states honestly what `delta` measures
(for example `LB public`, `nested CV F2`, `OOF ROC-AUC (seed 42)`, `estimate LB`); never mix them silently. `delta` is relative to the immediately preceding best
(empty when not applicable). `source` names the CLAUDE.md log entry or notebook. Quote fields that contain commas.

## Build
`latexmk -xelatex main.tex` in `Report/`; `latexmk -xelatex taskN.tex` in `Report/taskN/` (both must work, no undefined references).
Compile output files are git-ignored.

## Open questions
1. Resolved 26 Nov: the T1 operating-point figure is redrawn by `scripts/gen_figs_t1.py` (curve from a re-run of the same pipeline, marker = the notebook threshold 0.37264). The team confirmed that the small threshold difference comes from running on different CPUs.
2. The T1 masking ablation (OOF AUC 0.89403 to 0.89446, seed 42) is from a re-run on other hardware; the text says so.
3. Resolved 26 Nov: the team confirmed the T1 test-rank-cut wording and the T2 v10c/v10d leak attribution.
4. The teammate's PR branch lives on a fork; it was fetched through `refs/pull/1/head` (local branch `task1-report`), and `report` is based on it.
5. Pre-existing T1 numbers are the teammate's (notebook-derived); the 56 Model A features and the other counts were not re-verified in wave 1.
