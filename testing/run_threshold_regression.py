"""Run an overnight threshold regression against ProcureBuddy.

This runner is purpose-built for the repeated failures we saw in:
- amount parsing
- route mapping
- source/version leakage
- weak structured process output

It can run in two modes:
- api: call the live /chat endpoint
- local: import the Python AI service directly and bypass HTTP
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TESTING_DIR = ROOT / "testing"
RUNS_DIR = TESTING_DIR / "runs"
DEFAULT_API_URL = "http://127.0.0.1:8000"
LATEST_RESULTS_FILE = ROOT / "threshold_regression_latest_results.json"
LATEST_SUMMARY_FILE = ROOT / "threshold_regression_latest_summary.json"

if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

from generate_threshold_regression_dataset import (  # noqa: E402
    DEFAULT_OUTPUT,
    expected_mode,
    generate_dataset,
)


VALID_SECTIONS = [
    "## Quick Answer",
    "## Rule Priority Applied",
    "## Why This Applies",
    "## Detailed Process",
    "## Key Documents / Outputs",
    "## FLOWCHART",
    "## Source Basis",
    "## TL;DR",
]

MODE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Direct Purchase", ("applicable mode: direct purchase", "direct purchase")),
    ("LPC", ("applicable mode: lpc", "local purchase committee", " lpc ")),
    ("LTE", ("applicable mode: lte", "limited tender enquiry", "limited tender", " lte ")),
    ("OTE", ("applicable mode: ote", "open tender enquiry", "open tender", " ote ")),
)

UNCERTAINTY_PATTERNS = (
    "committee ?",
    "t&pc ?",
    "to be confirmed",
    "tbc",
    "not sure",
    "uncertain",
    "maybe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the threshold regression dataset overnight.")
    parser.add_argument("--dataset", default="", help="Existing dataset JSON path. If omitted, generate one.")
    parser.add_argument("--size", type=int, default=3000, help="Dataset size when generating a fresh dataset.")
    parser.add_argument("--seed", type=int, default=20260505, help="Dataset seed when generating a fresh dataset.")
    parser.add_argument("--mode", choices=("api", "local"), default="api", help="Execution mode.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL for /chat when mode=api.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-question timeout in seconds.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between questions in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for request failures.")
    parser.add_argument("--label", default="threshold_regression", help="Run-label suffix for the output folder.")
    parser.add_argument("--max-failures", type=int, default=100, help="How many failure examples to save in JSON.")
    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print progress every N questions.",
    )
    return parser.parse_args()


def load_or_generate_dataset(args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
        records = json.loads(dataset_path.read_text(encoding="utf-8"))
        return records, dataset_path

    dataset_path = run_dir / f"threshold_regression_dataset_{args.size}.json"
    summary_path = run_dir / f"threshold_regression_dataset_{args.size}_summary.json"
    records, summary = generate_dataset(size=args.size, seed=args.seed)
    dataset_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return records, dataset_path


def build_caller(args: argparse.Namespace) -> Callable[[str], tuple[str, str, float]]:
    if args.mode == "local":
        service_root = ROOT / "python-ai-service"
        if str(service_root) not in sys.path:
            sys.path.insert(0, str(service_root))
        from app.services.response_service import generate_response  # type: ignore

        def call_local(question: str) -> tuple[str, str, float]:
            started = time.perf_counter()
            try:
                result = generate_response(
                    query=question,
                    user="threshold-regression@procurebuddy.local",
                    bypass_cache=True,
                )
                latency = time.perf_counter() - started
                answer = str(result.get("answer", "")).strip()
                return answer, "", latency
            except Exception as exc:  # pragma: no cover - defensive harness path
                latency = time.perf_counter() - started
                return "", f"LOCAL_ERROR: {exc}", latency

        return call_local

    import requests

    session = requests.Session()
    endpoint = f"{args.api_url.rstrip('/')}/chat"

    def call_api(question: str) -> tuple[str, str, float]:
        payload = {
            "message": question,
            "user": "threshold-regression@procurebuddy.local",
            "displayName": "Threshold Regression Runner",
            "bypass_cache": True,
        }
        for attempt in range(args.retries + 1):
            started = time.perf_counter()
            try:
                response = session.post(endpoint, json=payload, timeout=args.timeout)
                latency = time.perf_counter() - started
                response.raise_for_status()
                body = response.json()
                answer = str(body.get("answer") or body.get("response") or "").strip()
                if answer:
                    return answer, "", latency
                if attempt >= args.retries:
                    return "", "EMPTY_RESPONSE", latency
            except Exception as exc:  # pragma: no cover - network path
                latency = time.perf_counter() - started
                if attempt >= args.retries:
                    return "", f"API_ERROR: {exc}", latency
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
        return "", "UNKNOWN_API_ERROR", 0.0

    return call_api


def extract_predicted_mode(response: str) -> str:
    lowered = f" {response.lower()} "
    quick_answer_match = re.search(r"applicable mode:\s*([^\n]+)", lowered)
    if quick_answer_match:
        mode_line = quick_answer_match.group(1)
        if "direct purchase" in mode_line:
            return "Direct Purchase"
        if "lpc" in mode_line or "local purchase committee" in mode_line:
            return "LPC"
        if "lte" in mode_line or "limited tender" in mode_line:
            return "LTE"
        if "ote" in mode_line or "open tender" in mode_line:
            return "OTE"

    for mode, patterns in MODE_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return mode
    return "UNKNOWN"


def strip_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def has_uncertain_output(text: str) -> bool:
    prose = strip_fenced_code_blocks(text).lower()
    if any(pattern in prose for pattern in UNCERTAINTY_PATTERNS):
        return True
    return bool(re.search(r"\b(?:committee|t&pc|lpc|lte|ote)\s*\?\b", prose))


def validation_failures(response: str) -> list[str]:
    failures: list[str] = []
    if "GFR 2025" in response:
        failures.append("WRONG_SOURCE")
    if has_uncertain_output(response):
        failures.append("UNCERTAIN_OUTPUT")
    if "Total steps: 1" in response:
        failures.append("WEAK_PROCESS")
    if "Rs.\n" in response or "Rs \n" in response:
        failures.append("BROKEN_SENTENCE")
    if "FINAL DECISION:" not in response:
        failures.append("MISSING_FINAL_DECISION")
    missing_sections = [section for section in VALID_SECTIONS if section not in response]
    if missing_sections:
        failures.append("MISSING_SECTIONS")
    return failures


def score_response(response: str) -> float:
    score = 0
    if "## Quick Answer" in response:
        score += 1
    if "## Detailed Process" in response:
        score += 1
    if "Total steps: 1" not in response:
        score += 1
    if "GFR 2025" not in response:
        score += 1
    if "?" not in response:
        score += 1
    return score / 5.0


def classify_bug(response: str, error: str, validations: list[str], predicted_mode: str, expected: str) -> str:
    if error:
        return "REQUEST_ERROR"
    if validations:
        return validations[0]
    if predicted_mode == "UNKNOWN":
        return "MODE_NOT_FOUND"
    if predicted_mode != expected:
        return f"MODE_MISMATCH:{expected}->{predicted_mode}"
    return "OTHER"


def print_progress(index: int, total: int, passed: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    accuracy = (passed / index * 100.0) if index else 0.0
    print(
        f"[{index}/{total}] accuracy={accuracy:.2f}% "
        f"passed={passed} elapsed={elapsed/60.0:.1f}m",
        flush=True,
    )


def write_csv_header(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "question",
                "amount_text",
                "amount_rupees",
                "expected_mode",
                "expected_rule",
                "predicted_mode",
                "passed",
                "latency_seconds",
                "quality_score",
                "bug_type",
                "validation_failures",
                "error",
                "response",
            ]
        )


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                row["id"],
                row["question"],
                row["amount_text"],
                row["amount_rupees"],
                row["expected_mode"],
                row["expected_rule"],
                row["predicted_mode"],
                row["passed"],
                f"{row['latency_seconds']:.3f}",
                f"{row['quality_score']:.3f}",
                row["bug_type"],
                ",".join(row["validation_failures"]),
                row["error"],
                row["response"],
            ]
        )


def build_summary(results: list[dict[str, Any]], dataset_path: Path, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row["passed"])
    failed = total - passed
    accuracy = round((passed / total * 100.0), 2) if total else 0.0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_rule: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    bug_distribution = Counter()
    validation_distribution = Counter()

    for row in results:
        expected = str(row["expected_mode"])
        predicted = str(row["predicted_mode"])
        confusion[expected][predicted] += 1
        per_rule[expected]["total"] += 1
        if row["passed"]:
            per_rule[expected]["correct"] += 1
        if row["bug_type"] != "OTHER":
            bug_distribution[row["bug_type"]] += 1
        for failure in row["validation_failures"]:
            validation_distribution[failure] += 1

    rule_accuracy = {
        rule: {
            "total": values["total"],
            "correct": values["correct"],
            "accuracy": round((values["correct"] / values["total"] * 100.0), 2) if values["total"] else 0.0,
        }
        for rule, values in sorted(per_rule.items())
    }

    average_quality = round(
        sum(float(row["quality_score"]) for row in results) / total,
        4,
    ) if total else 0.0
    average_latency = round(
        sum(float(row["latency_seconds"]) for row in results) / total,
        4,
    ) if total else 0.0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "mode": args.mode,
        "api_url": args.api_url if args.mode == "api" else "",
        "size": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "average_quality_score": average_quality,
        "average_latency_seconds": average_latency,
        "confusion_matrix": {key: dict(value) for key, value in sorted(confusion.items())},
        "rule_accuracy": rule_accuracy,
        "bug_distribution": dict(bug_distribution),
        "validation_distribution": dict(validation_distribution),
        "top_failures": [
            {
                "id": row["id"],
                "question": row["question"],
                "amount_text": row["amount_text"],
                "expected_mode": row["expected_mode"],
                "predicted_mode": row["predicted_mode"],
                "bug_type": row["bug_type"],
                "validation_failures": row["validation_failures"],
                "error": row["error"],
            }
            for row in results
            if not row["passed"]
        ][: args.max_failures],
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(flush=True)
    print(f"Total: {summary['size']}", flush=True)
    print(f"Passed: {summary['passed']}", flush=True)
    print(f"Failed: {summary['failed']}", flush=True)
    print(f"Accuracy: {summary['accuracy']:.2f}%", flush=True)
    print(f"Avg quality score: {summary['average_quality_score']:.3f}", flush=True)
    print(f"Avg latency: {summary['average_latency_seconds']:.3f}s", flush=True)
    print(flush=True)
    print("CONFUSION MATRIX:", flush=True)
    for expected, predicted_counts in summary["confusion_matrix"].items():
        print(f"{expected}: {predicted_counts}", flush=True)
    print(flush=True)
    print("RULE ACCURACY:", flush=True)
    for rule, stats in summary["rule_accuracy"].items():
        print(f"{rule}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})", flush=True)
    print(flush=True)
    print("BUG DISTRIBUTION:", flush=True)
    for bug, count in sorted(summary["bug_distribution"].items()):
        print(f"{bug}: {count}", flush=True)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records, dataset_path = load_or_generate_dataset(args, run_dir)
    call_question = build_caller(args)

    csv_path = run_dir / "threshold_regression_results.csv"
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    failures_path = run_dir / "failed_cases.json"
    write_csv_header(csv_path)

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Questions: {len(records)}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Progress updates every {args.report_every} questions.", flush=True)

    results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    passed_count = 0

    for index, record in enumerate(records, start=1):
        question = str(record["question"])
        expected = str(record["expected_mode"])
        answer, error, latency = call_question(question)
        predicted_mode = extract_predicted_mode(answer) if answer else "UNKNOWN"
        validations = validation_failures(answer) if answer else ["NO_RESPONSE"]
        quality_score = score_response(answer) if answer else 0.0
        passed = bool(answer) and predicted_mode == expected and not validations
        if passed:
            passed_count += 1
        bug_type = classify_bug(answer, error, validations, predicted_mode, expected)

        row = {
            "id": record["id"],
            "question": question,
            "amount_text": record["amount_text"],
            "amount_rupees": record["amount_rupees"],
            "expected_mode": expected,
            "expected_rule": record["expected_rule"],
            "predicted_mode": predicted_mode,
            "passed": passed,
            "latency_seconds": latency,
            "quality_score": quality_score,
            "bug_type": bug_type,
            "validation_failures": validations,
            "error": error,
            "response": answer,
        }
        results.append(row)
        append_csv_row(csv_path, row)

        if args.report_every > 0 and (index % args.report_every == 0 or index == len(records)):
            print_progress(index, len(records), passed_count, started_at)
        if args.delay > 0 and index < len(records):
            time.sleep(args.delay)

    summary = build_summary(results, dataset_path, args, run_dir)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failures_path.write_text(json.dumps(summary["top_failures"], indent=2), encoding="utf-8")
    LATEST_RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LATEST_SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary)
    print(flush=True)
    print(f"CSV report: {csv_path}", flush=True)
    print(f"Results JSON: {results_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)
    print(f"Top failures JSON: {failures_path}", flush=True)
    print(f"Latest results JSON: {LATEST_RESULTS_FILE}", flush=True)
    print(f"Latest summary JSON: {LATEST_SUMMARY_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
