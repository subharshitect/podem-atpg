# PODEM ATPG (Python)

A complete Python implementation of ATPG using the PODEM algorithm for combinational circuits with gate-output stuck-at faults.

## Quickstart
```sh
make venv
make install
make test
make run
make examples
make report
```

## CLI Usage
```sh
python -m atpg run --netlist examples/toy1.bench --fault n1/SA0
python -m atpg run --netlist examples/toy1.bench --fault n1/SA0 --json
```

### Output
- **Status**: DETECTED / UNTESTABLE / ABORTED
- **Test vector**: values for each primary input
- **PO observations**: any D/DBAR values observed at primary outputs

## Netlist Format
Use `.bench` style lines:
```
INPUT(a)
INPUT(b)
n1 = AND(a, b)
out = NOT(n1)
OUTPUT(out)
```

Supported gate types: `AND OR NAND NOR NOT BUF XOR XNOR`.

Sequential elements or cyclic dependencies are rejected.

## Fault Format
Fault string format: `<net>/SA0` or `<net>/SA1`.

Constraints:
- `<net>` must be a **gate-output net** present in the circuit.
- Primary input faults are rejected.

## Project Structure
```
 atpg/            Core package (parser, logic, PODEM, CLI)
 tests/           Pytest tests
 examples/        Small example circuits
 report/          LaTeX report
 artifacts/       Generated outputs from runs
```

## Makefile Targets
- `make help` – list targets
- `make venv` – create `.venv`
- `make install` – install dependencies
- `make test` – run pytest
- `make lint` – run ruff (if installed)
- `make run` – run a default example and save output in `artifacts/run.txt`
- `make examples` – run example faults and store outputs in `artifacts/`
- `make report` – build `report/report.pdf` (requires `pdflatex`)
- `make clean` – remove build/test artifacts
