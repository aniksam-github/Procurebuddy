"""Summarize a ProcureBuddy eval CSV with category-level breakdowns."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_summary(report_file: str | Path) -> dict[str, Any]:
    path = Path(report_file).resolve()
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    total = len(rows)
    passed_rows = [row for row in rows if str(row.get("Passed", "")).strip().upper() == "PASS"]
    failed_rows = [row for row in rows if str(row.get("Passed", "")).strip().upper() != "PASS"]
    average_score = round(sum(float(row["Score"]) for row in rows) / total, 3) if total else 0.0
    accuracy = round((len(passed_rows) / total) * 100, 1) if total else 0.0

    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("Type", "unknown")).strip().lower()].append(row)

    category_rows: list[dict[str, Any]] = []
    for category, items in sorted(by_type.items()):
        count = len(items)
        passes = sum(1 for item in items if str(item.get("Passed", "")).strip().upper() == "PASS")
        avg_score = round(sum(float(item["Score"]) for item in items) / count, 3) if count else 0.0
        category_rows.append(
            {
                "category": category,
                "count": count,
                "passes": passes,
                "accuracy": round((passes / count) * 100, 1) if count else 0.0,
                "average_score": avg_score,
            }
        )

    return {
        "report_file": str(path),
        "total": total,
        "passed": len(passed_rows),
        "failed": len(failed_rows),
        "accuracy": accuracy,
        "average_score": average_score,
        "categories": category_rows,
    }


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "ProcureBuddy Fresh Eval Summary",
        "=" * 36,
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Accuracy: {summary['accuracy']}%",
        f"Average Score: {summary['average_score']}",
        "",
        "By Category:",
    ]
    for item in summary["categories"]:
        lines.append(
            f"- {item['category']}: {item['passes']}/{item['count']} pass "
            f"({item['accuracy']}%), avg={item['average_score']}"
        )
    return "\n".join(lines)


def write_summary_json(summary: dict[str, Any], target_file: str | Path) -> None:
    path = Path(target_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an eval CSV.")
    parser.add_argument("report_file", help="Path to eval_report.csv")
    parser.add_argument("--json-out", default=None, help="Optional JSON summary output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args.report_file)
    print(render_summary(summary))
    if args.json_out:
        write_summary_json(summary, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
