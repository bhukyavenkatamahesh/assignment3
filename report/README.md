# Report build instructions

The `main.tex` follows the official **CVPR 2024 author kit**.
To compile, drop `cvpr.sty` and `ieeenat_fullname.bst` from the kit
into this folder, then:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The figures referenced in the `.tex` (`figs/curves_fold0.pdf`,
`figs/perclass_bar.pdf`, `figs/arch.pdf`) are filled by:

```bash
python scripts/plot_results.py
```

The architecture diagram (`figs/arch.pdf`) is hand-drawn in
draw.io / OmniGraffle / similar, then exported.
