# CPE342 ML003 technical report (LaTeX)

Requirements: XeLaTeX (MiKTeX or TeX Live), `latexmk`. Packages install on first build with MiKTeX.

```
cd Report
latexmk -xelatex main.tex            # full report -> main.pdf
cd task1 && latexmk -xelatex task1.tex   # one task alone (works for task1..task5)
```

Layout: `main.tex` (title, TOC, front matter, Methodology = `\subfile{taskN/taskN}`), `preamble.tex`, `macros.tex`,
`front/` (summary, collaboration, conclusion, appendix), `taskN/taskN.tex` + `taskN/figures/`, `scripts/` (figure generators,
`plot_style.py`), `data/decision_log_tN.csv`. Read `SPEC.md` before editing anything.

Bibliography: `references.bib` (natbib, `unsrtnat`). `latexmk -xelatex main.tex` runs BibTeX automatically (needs the `bibtex` binary,
included in MiKTeX and TeX Live). Standalone task builds show `[?]` for citations.
