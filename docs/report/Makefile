# RSLAB technical report
#
#   make        figures + PDF
#   make pdf    compile rslab.tex (needs figures present)
#   make figs   regenerate all report figures as PDF (paper mode, black axes)
#   make data   re-run the same-run corpus + ll/mf + estimate benches (heavy)
#
# Figures are the README bench plots rendered in paper mode (RSLAB_REPORT=1):
# black axes, serif font, PDF into figures/. The PDF needs tectonic (or latexmk).

PY ?= python
ROOT := ../..

.PHONY: all pdf figs data clean
all: figs pdf

figs:
	cd $(ROOT) && bash benches/report_figs.sh

data:
	cd $(ROOT) && bash benches/run_corpus_auto.sh && bash benches/run_corpus_llmf.sh

pdf:
	latexmk -pdf -interaction=nonstopmode rslab.tex

clean:
	rm -f rslab.pdf
