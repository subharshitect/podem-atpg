PDF=main.pdf
TEX=main.tex

.PHONY: all open clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error $(TEX)

open: all
	code $(PDF)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk $(PDF)
