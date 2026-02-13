# Makefile for EDISGA Report Generation

# Configuration
TEX_DIR = tex
BUILD_DIR = build_EDISGA
MAIN_TEX = main.tex
PDF_TARGET = main.pdf

# Tools
LATEX = pdflatex
LATEX_FLAGS = -halt-on-error -interaction=nonstopmode

.PHONY: all clean report help

PY ?= python3
TEX ?= pdflatex

SEED ?= 7
DURATION_S ?= 240

# Unique build directory every run
OUTDIR := build_EDISGA

all: report

sim:
	@mkdir -p "$(OUTDIR)"
	@$(PY) simulate_edisga.py --outdir "$(OUTDIR)" --seed "$(SEED)" --duration_s "$(DURATION_S)"

all: report

# Rule to generate the PDF
# We run pdflatex twice to ensure TOC and references are resolved correctly.
report:
	@echo "Building LaTeX report..."
	cd $(TEX_DIR) && $(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	cd $(TEX_DIR) && $(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@echo "Build complete. PDF is located at $(TEX_DIR)/$(PDF_TARGET)"

# Clean up build artifacts in the tex directory
clean:
	@echo "Cleaning LaTeX artifacts..."
	cd $(TEX_DIR) && rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk $(PDF_TARGET)

# Help command
help:
	@echo "Available targets:"
	@echo "  all     : Build the PDF report (default)"
	@echo "  report  : Build the PDF report"
	@echo "  clean   : Remove generated PDF and intermediate LaTeX files"