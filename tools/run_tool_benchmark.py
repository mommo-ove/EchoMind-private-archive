"""Run and persist the deterministic campus Tool benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.tool_benchmark import run_tool_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/tool_benchmark_latest.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="echomind-tool-eval-") as directory:
        report = asyncio.run(run_tool_benchmark(Path(directory) / "campus.db"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"case_count": report["case_count"], "metrics": report["metrics"], "latency_ms": report["latency_ms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
