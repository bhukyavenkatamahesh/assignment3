# Report build instructions

`main.tex` is a plain `article`-class single-column LaTeX document
(matches the style of my Assignment 1 report). No special class files
needed. To compile:

```bash
pdflatex main
pdflatex main     # second pass for table/figure references
```

The figures referenced in the `.tex`
(`figs/curves_fold0.pdf`, `figs/perclass_bar.pdf`, `figs/confusion.pdf`)
are produced by:

```bash
python scripts/plot_results.py    # curves + per-class bar
python scripts/plot_confusion.py  # confusion matrices
```
