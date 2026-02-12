.RECIPEPREFIX := >
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

VENV := .podem
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help venv install check-venv test lint run examples report clean

help:
>@echo "Targets:"
>@echo "  venv      - create virtual environment"
>@echo "  install   - install dependencies"
>@echo "  test      - run pytest"
>@echo "  lint      - run ruff"
>@echo "  run       - run default example"
>@echo "  examples  - run example faults"
>@echo "  report    - build report/report.pdf"
>@echo "  clean     - remove build artifacts"

venv:
>python3 -m venv $(VENV)

install: venv
>$(PIP) install -r requirements.txt

check-venv:
>@test -x "$(PY)" || (echo "Virtualenv missing. Run: make install"; exit 1)

test: check-venv
>$(PY) -m pytest -v

lint: check-venv
>$(PY) -m ruff check .

run: check-venv
>mkdir -p artifacts
>$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA0 > artifacts/run.txt

examples: check-venv
>mkdir -p artifacts
>$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA0 --json > artifacts/toy1_n1_SA0.json
>$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA1 --json > artifacts/toy1_n1_SA1.json
>$(PY) -m atpg run --netlist examples/toy2.bench --fault n1/SA0 --json > artifacts/toy2_n1_SA0.json
>$(PY) -m atpg run --netlist examples/toy3.bench --fault n1/SA0 --json > artifacts/toy3_n1_SA0.json
>PYTHONPATH=. $(PY) -m scripts.run_faults --netlists examples/toy1.bench examples/toy2.bench examples/toy3.bench
>$(PY) scripts/summarize_results.py


report: examples
>test -d report || (echo "report/ directory missing."; exit 1)
>test -f report/main.tex || (echo "report/main.tex not found."; exit 1)
>mkdir -p report/artifacts
>$(PY) scripts/render_results_tex.py
>$(PY) scripts/render_trace_tex.py
>$(PY) scripts/render_fault_sweep_tex.py
>command -v pdflatex >/dev/null 2>&1 || (echo "pdflatex not found; cannot build report."; exit 1)
>cd report
>pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
>pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null


clean:
>rm -rf $(VENV) .pytest_cache __pycache__ artifacts
>rm -f report/*.aux report/*.log report/*.out report/*.pdf
>rm -rf report/artifacts

# cd report
# pdflatex -interaction=nonstopmode -halt-on-error main2.tex
# pdflatex -interaction=nonstopmode -halt-on-error main2.tex
