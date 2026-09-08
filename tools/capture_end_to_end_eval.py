"""Run the real HTTP evaluation endpoint and persist a traceable report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]


def capture_report(
    *,
    base_url: str,
    dataset_path: Path,
    output_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    payload_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = (
        dict(payload_override)
        if payload_override is not None
        else json.loads(dataset_path.read_text(encoding="utf-8"))
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/eval/run",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with opener(request, timeout=1800) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict) or not {"total", "metrics", "results"}.issubset(result):
        raise ValueError("endpoint did not return a complete evaluation report")
    if not isinstance(result["results"], list) or result["total"] != len(result["results"]):
        raise ValueError("evaluation report total does not match result rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/campus_golden.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/campus_end_to_end_latest.json"))
    parser.add_argument("--no-refresh-summary", action="store_true")
    args = parser.parse_args()

    report = capture_report(
        base_url=args.base_url,
        dataset_path=args.dataset,
        output_path=args.output,
    )
    print(args.output)
    print(json.dumps({"total": report["total"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    if not args.no_refresh_summary:
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "build_benchmark_summary.py")],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
