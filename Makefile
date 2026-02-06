PY := .venv/bin/python
PIP := .venv/bin/pip

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
	python3 -m venv .venv

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
	$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA0 --json > artifacts/toy1_n1_SA0.json
	$(PY) -m atpg run --netlist examples/toy1.bench --fault n1/SA1 --json > artifacts/toy1_n1_SA1.json
	$(PY) -m atpg run --netlist examples/toy2.bench --fault n1/SA0 --json > artifacts/toy2_n1_SA0.json
	$(PY) -m atpg run --netlist examples/toy3.bench --fault n1/SA0 --json > artifacts/toy3_n1_SA0.json
	$(PY) - <<'PY'
import json
from pathlib import Path

outputs = []
for path in sorted(Path("artifacts").glob("*.json")):
    data = json.loads(path.read_text())
    outputs.append({
        "file": path.name,
        "status": data["status"],
        "test_vector": ", ".join(f"{k}={v}" for k, v in data["test_vector"].items()),
    })
Path("artifacts/results.json").write_text(json.dumps(outputs, indent=2))
PY

report: examples
	@mkdir -p report/artifacts
	$(PY) - <<'PY'
import json
from pathlib import Path

results_path = Path("artifacts/results.json")
if not results_path.exists():
    raise SystemExit("Run 'make examples' first to generate results.json")

rows = json.loads(results_path.read_text())
lines = [
    "\\begin{table}[h]",
    "\\centering",
    "\\begin{tabular}{llll}",
    "\\toprule",
    "Netlist & Fault & Status & Test Vector \\\\",
    "\\midrule",
]
for row in rows:
    parts = row["file"].replace(".json", "").split("_")
    netlist = parts[0]
    fault = f\"{parts[1]}/{parts[2]}\"
    status = row["status"]
    vector = row["test_vector"]
    lines.append(f"{netlist} & {fault} & {status} & {vector} \\\\")
lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "\\caption{Example ATPG results.}",
    "\\label{tab:results}",
    "\\end{table}",
])
Path("report/artifacts/results.tex").write_text("\n".join(lines))
PY
	@command -v pdflatex >/dev/null 2>&1 || (echo "pdflatex not found; cannot build report."; exit 1)
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

clean:
	@rm -rf .pytest_cache __pycache__ artifacts report/*.aux report/*.log report/*.out report/*.pdf report/artifacts/results.tex
