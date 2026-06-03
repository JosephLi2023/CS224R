# CS224R Final Report — Hybrid Advantage Shaping (HAS) + TurnRD

Build:
```
cd reports/final_paper
latexmk -pdf main.tex      # pdflatex + bibtex
```
Output: `main.pdf` (page 1 = one-page extended abstract; then ~8-page main paper).

Figures reused from the finalized poster (`reports/poster_paloalto/figures/`):
- `architecture.png` (Fig 1), `F2_webshop_5method_bar.pdf` (Fig 3),
  `F4_alfworld_trajectory.pdf` (Fig 2), `F7_film_mechanism.pdf` (Fig 4).

Regenerate data figures:
- `python scripts/analysis/plot_eval_trajectory_3method.py --flatgrpo-dir /tmp/flat_canon`
- `python scripts/analysis/plot_film_mechanism.py`
- `python scripts/analysis/plot_webshop_bars.py`
