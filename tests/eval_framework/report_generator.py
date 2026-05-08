"""Report writers for evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluator import EvaluationResult


def write_csv_report(results: list[EvaluationResult], path: str | Path) -> None:
    """Write the main evaluation report as CSV."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ID",
                "Question",
                "Type",
                "Difficulty",
                "Answer",
                "Score",
                "Passed",
                "Semantic Score",
                "Relevance Score",
                "Completeness Score",
                "Source Score",
                "Error Reason",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.question,
                    result.question_type,
                    result.difficulty,
                    result.answer,
                    result.score,
                    "PASS" if result.passed else "FAIL",
                    result.semantic_score,
                    result.relevance_score,
                    result.completeness_score,
                    result.source_score,
                    result.error_reason,
                ]
            )


def append_csv_report(results: list[EvaluationResult], path: str | Path) -> None:
    """Append evaluation results to the CSV report, writing the header if needed."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "ID",
                    "Question",
                    "Type",
                    "Difficulty",
                    "Answer",
                    "Score",
                    "Passed",
                    "Semantic Score",
                    "Relevance Score",
                    "Completeness Score",
                    "Source Score",
                    "Error Reason",
                ]
            )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.question,
                    result.question_type,
                    result.difficulty,
                    result.answer,
                    result.score,
                    "PASS" if result.passed else "FAIL",
                    result.semantic_score,
                    result.relevance_score,
                    result.completeness_score,
                    result.source_score,
                    result.error_reason,
                ]
            )


def write_failed_cases(results: list[EvaluationResult], path: str | Path) -> None:
    """Write only failed cases as JSON for quick debugging."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    failed = [result.to_dict() for result in results if not result.passed]

    with output.open("w", encoding="utf-8") as handle:
        json.dump(failed, handle, indent=2, ensure_ascii=False)


def append_failed_cases(results: list[EvaluationResult], path: str | Path) -> None:
    """Append failed cases to a JSON list without losing earlier failures."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if output.exists():
        try:
            loaded = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [item for item in loaded if isinstance(item, dict)]
        except Exception:
            existing = []

    failed = [result.to_dict() for result in results if not result.passed]
    output.write_text(json.dumps([*existing, *failed], indent=2, ensure_ascii=False), encoding="utf-8")


def load_processed_case_ids(path: str | Path) -> set[str]:
    """Read processed case IDs from an existing CSV report."""

    output = Path(path).resolve()
    if not output.exists():
        return set()

    processed_ids: set[str] = set()
    with output.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = str(row.get("ID") or row.get("id") or row.get("case_id") or "").strip()
            if not raw_id:
                continue
            processed_ids.add(raw_id)
    return processed_ids
