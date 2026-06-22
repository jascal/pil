# Paper: *Projective Incidence Calculus for Transformer Decision Logic*

ACM `sigconf` submission for the `pil` project. Every quantitative claim is
transcribed from a checked-in result file under [`../results/`](../results) — see
the provenance table below.

## Files

| file | what |
|---|---|
| `pic.tex` | the paper (acmart / sigconf) |
| `references.bib` | bibliography (external entries flagged for verification) |
| `make_figures.py` | regenerates the four **data** figures from `../results/` numbers |
| `figures/*.pdf` | the four data figures (PDF for LaTeX, PNG for preview) |

(Figure 1, the Cell-Capacity-vs-TRC schematic, is drawn inline with TikZ in `pic.tex`
— it is conceptual, not data-driven, so it is not produced by `make_figures.py`.)

## Build

Needs a TeX distribution with `acmart.cls` (TeX Live `texlive-publishers`, or
MiKTeX). **Not installed in this environment** — build elsewhere:

```bash
python paper/make_figures.py          # (re)generate figures/*.pdf  [needs ../.venv]
cd paper
pdflatex pic && bibtex pic && pdflatex pic && pdflatex pic
```

For double-blind review, change the class line to
`\documentclass[sigconf,anonymous,review]{acmart}` and drop `nonacm`. For
camera-ready, drop `nonacm` and add the venue's `\acmConference`, `\setcopyright`,
and `\acmDOI`.

## Claim → result-file provenance

| paper element | source |
|---|---|
| Tab. 1 (frame_reg null) | `results/hard_sweep.txt`, `results/frame_reg_surface.txt` |
| Fig. 1 / §5.2 (generation lever) | `results/comp_sweep.txt` |
| Tab. 2 (selection vs training 2×2) | `results/scored2_sweep.txt` |
| §5.3 capacity slack (~1e59) | `results/capacity_diagnostic.txt` |
| Fig. 3 / Tab. 3 (M-dominance) | `results/routing_complexity_sweep.txt` |
| Tab. 4 (interference in ℝ^M) | `results/interference_probe.txt`, `results/coherence_reg.txt` |
| Fig. 2 / Tab. 6 (rank vs scale) | `results/real_dla_sweep.txt` |
| Tab. 7 (consistent late-MLP circuit) | `results/decode_circuit.txt` |
| Fig. 4 (coherence slack) | `results/real_dla_sweep.txt` |
| Tab. 8 (difficulty), Tab. 9 (multilingual) | `results/real_dla_sweep.txt` |
| Theorems (softmax / separation / rank / Welch) | machine-checked in `i-orca` (cited) |

## Pending

A **Pythia** scaling-suite replication is in progress. When the numbers land, add
rows to the scale table (`tab:scale`), the multilingual table (`tab:lang`) if
applicable, and a `pythia` group/series in `make_figures.py`
(`fig_rank_concentration`, `fig_coherence_slack`); the surrounding prose in
§\ref{sec:real} already anticipates the second model family.
