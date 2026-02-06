VENV := .podem
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help venv install test lint run examples report clean

help:
	@echo "Targets:"
	@echo "  venv      - create virtual environment"
	@echo "  install   - install dependencies"
	@echo "  test      - run pytest"
	@echo "  lint      - run ruff"
	@echo "  run       - run default example"
	@echo "  examples  - run example faults"
	@echo "  report    - build report/report.pdf"
	@echo "  clean     - remove build artifacts"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -v

lint:
	$(PY) -m ruff check .

run:
	@mkdir -p artifacts
	$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA0 > artifacts/run.txt

examples:
	@mkdir -p artifacts
	@rm -f artifacts/results.json
	$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA0 --json > artifacts/toy1_n1_SA0.json
	$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA1 --json > artifacts/toy1_n1_SA1.json
	$(PY) -m atpg run --netlist examples/toy2.bench --fault n1/SA0 --json > artifacts/toy2_n1_SA0.json
	$(PY) -m atpg run --netlist examples/toy3.bench --fault n1/SA0 --json > artifacts/toy3_n1_SA0.json
	$(PY) scripts/summarize_results.py

report: examples
	@mkdir -p report/artifacts
	$(PY) scripts/render_results_tex.py
	@command -v pdflatex >/dev/null 2>&1 || (echo "pdflatex not found; cannot build report."; exit 1)
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

clean:
	@rm -rf $(VENV) .pytest_cache __pycache__ artifacts
	@rm -f report/*.aux report/*.log report/*.out report/*.pdf
	@rm -rf report/artifacts
