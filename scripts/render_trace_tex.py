"""Render verbose trace logs into LaTeX-friendly verbatim blocks."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    src = Path("artifacts")
    dst = Path("report/artifacts")
    dst.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.glob("trace_*.txt")):
        body = path.read_text(errors="replace")
        out = dst / f"{path.stem}.tex"
        out.write_text("\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n")


if __name__ == "__main__":
    main()
