"""Build EchoMind's evidence-backed unified benchmark report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark_summary import build_summary, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/eval/results"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/benchmark_summary_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/EchoMind统一评测总报告.md"))
    args = parser.parse_args()

    summary = build_summary(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(args.output)
    print(args.markdown)


if __name__ == "__main__":
    main()
