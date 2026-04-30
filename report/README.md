# Report build instructions

`main.tex` follows the **CVPR 2024 author kit** template. To compile,
drop `cvpr.sty` and `ieeenat_fullname.bst` (from the kit) into this
folder, then:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Easiest way: Overleaf

1. Go to [overleaf.com](https://overleaf.com) → **New Project →
   Templates** → search "CVPR" and pick the official 2024 template.
2. Replace the example `main.tex` with our `main.tex`.
3. Upload `refs.bib` and the `figs/` folder (the three PDFs).
4. Click **Recompile** → download the PDF.

## Files referenced

- `figs/perclass_bar.pdf` -- per-class bar chart
- `figs/confusion.pdf` -- confusion matrices (backbone vs AdvSoli)
- `figs/curves_fold0.pdf` -- training curves on fold 0
- `refs.bib` -- bibliography

The figures are produced by:
```bash
python scripts/plot_results.py    # curves + per-class bar
python scripts/plot_confusion.py  # confusion matrices
```
