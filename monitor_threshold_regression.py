"""Live monitor for threshold regression CSV output."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "testing" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the latest threshold regression CSV live.")
    parser.add_argument("--run-dir", default="", help="Specific run directory to watch.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--tail", type=int, default=5, help="How many recent rows to show.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    return parser.parse_args()


def resolve_csv_path(run_dir_arg: str) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
        csv_path = run_dir / "threshold_regression_results.csv"
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    candidates = sorted(RUNS_DIR.glob("*/threshold_regression_results.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No threshold regression CSV found under testing/runs.")
    return candidates[0]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def summarize(rows: list[dict[str, str]]) -> tuple[int, int, int, float]:
    processed = len(rows)
    passed = sum(1 for row in rows if str(row.get("passed", "")).strip().lower() == "true")
    failed = processed - passed
    accuracy = (passed / processed * 100.0) if processed else 0.0
    return processed, passed, failed, accuracy


def print_snapshot(csv_path: Path, rows: list[dict[str, str]], tail: int) -> None:
    processed, passed, failed, accuracy = summarize(rows)
    print("=" * 72)
    print(f"CSV: {csv_path}")
    print(f"Processed: {processed}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {failed}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print("-" * 72)
    for row in rows[-tail:]:
        print(
            f"#{row.get('id', '?')} "
            f"passed={row.get('passed', '')} "
            f"expected={row.get('expected_mode', '')} "
            f"predicted={row.get('predicted_mode', '')} "
            f"bug={row.get('bug_type', '')} "
            f"question={row.get('question', '')[:90]}"
        )


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv_path(args.run_dir)
    last_processed = -1

    while True:
        rows = load_rows(csv_path)
        processed, _, _, _ = summarize(rows)
        if processed != last_processed:
            print_snapshot(csv_path, rows, args.tail)
            last_processed = processed
        if args.once:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
