"""Run a fresh ProcureBuddy evaluation into an isolated testing folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from summarize_eval import build_summary, render_summary, write_summary_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "eval_framework" / "test_cases_smoke_20.json"
RUNS_DIR = ROOT / "testing" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fresh eval into testing/runs/<timestamp>.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset JSON path.")
    parser.add_argument("--limit", type=int, default=20, help="Number of cases to run.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base API URL.")
    parser.add_argument("--label", default="smoke", help="Short label for the run folder.")
    parser.add_argument("--duration", default=None, help="Optional duration cap, e.g. 15m.")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request API timeout in seconds.")
    parser.add_argument("--delay", type=float, default=1.0, help="Fixed delay between cases in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for API timeouts/errors.")
    parser.add_argument("--strict-eval", action="store_true", help="Enable stricter evaluator rules.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_file = run_dir / "eval_report.csv"
    failed_file = run_dir / "eval_failed_cases.json"
    summary_file = run_dir / "summary.json"

    command = [
        sys.executable,
        "-m",
        "tests.eval_framework.test_runner",
        "--dataset",
        str(Path(args.dataset).resolve()),
        "--limit",
        str(args.limit),
        "--api-url",
        args.api_url,
        "--report-file",
        str(report_file),
        "--failed-file",
        str(failed_file),
        "--batch-size",
        str(args.batch_size),
        "--timeout",
        str(args.timeout),
        "--delay",
        str(args.delay),
        "--retries",
        str(args.retries),
        "--avoid-recent-minutes",
        "0",
        "--no-resume",
    ]
    if args.duration:
        command.extend(["--duration", args.duration])
    if args.strict_eval:
        command.append("--strict-eval")

    print(f"Starting fresh eval in: {run_dir}")
    print("Command:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        print(f"Eval runner failed with exit code {completed.returncode}")
        return completed.returncode

    summary = build_summary(report_file)
    write_summary_json(summary, summary_file)
    print()
    print(render_summary(summary))
    print()
    print(f"Run directory: {run_dir}")
    print(f"CSV report: {report_file}")
    print(f"Summary JSON: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
