"""Sequential evaluation runner for ProcureBuddy RAG.

Design goals:
- single-threaded only
- resumable CSV execution
- predictable retries
- easy to debug and control
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from .config import (
    API_URL,
    BATCH_SIZE,
    CSV_REPORT_FILE,
    DEFAULT_DATASET_FILE,
    DELAY,
    FAILED_CASES_FILE,
    LOW_CONFIDENCE_THRESHOLD,
    RECENT_CASE_COOLDOWN_MINUTES,
    RECENT_CASE_HISTORY_FILE,
    RETRIES,
    TIMEOUT,
    USE_RESUME,
    WORKERS,
    retry_backoff_seconds,
)
from .dataset_loader import (
    DatasetSelection,
    TestCase,
    append_case_selection_history,
    count_fresh_cases,
    load_dataset,
    recent_question_ids,
)
from .evaluator import EvaluationResult, evaluate_response
from .report_generator import append_csv_report, append_failed_cases, load_processed_case_ids


logger = logging.getLogger("procurebuddy-eval")


def get_processed_ids(csv_file: str | Path) -> set[int]:
    """Return case IDs already present in the CSV report for resume runs."""

    return load_processed_case_ids(csv_file)


def _timestamp() -> str:
    """Return a simple local timestamp for request timing logs."""

    return datetime.now().isoformat(timespec="seconds")


def _parse_duration(value: str) -> float:
    """Parse a duration string into seconds.

    Supported formats:
    - 900
    - 900s
    - 15m
    - 1.5h
    """

    normalized = str(value).strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", normalized)
    if not match:
        raise argparse.ArgumentTypeError("Duration must look like 900, 900s, 15m, or 1.5h.")

    amount = float(match.group(1))
    if amount <= 0:
        raise argparse.ArgumentTypeError("Duration must be greater than 0.")

    unit = match.group(2) or "s"
    multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return amount * multiplier


def _format_duration(seconds: float | None) -> str:
    """Render seconds as a compact human-readable duration."""

    if seconds is None:
        return "unknown"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _estimate_remaining_seconds(
    total_cases: int,
    completed_cases: int,
    elapsed_seconds: float,
    timeout_seconds: float = TIMEOUT,
    delay_seconds: float = DELAY,
    duration_limit_seconds: float | None = None,
) -> float:
    """Estimate remaining runtime using a conservative start and then observed throughput."""

    remaining_cases = max(total_cases - completed_cases, 0)
    if remaining_cases == 0:
        return 0.0

    if completed_cases > 0 and elapsed_seconds > 0:
        average_seconds_per_case = elapsed_seconds / completed_cases
    else:
        average_seconds_per_case = timeout_seconds + delay_seconds

    remaining_seconds = remaining_cases * average_seconds_per_case

    if duration_limit_seconds is not None:
        remaining_seconds = min(remaining_seconds, max(duration_limit_seconds - elapsed_seconds, 0.0))

    return max(remaining_seconds, 0.0)


def get_batches(test_cases: list[TestCase], batch_size: int = BATCH_SIZE) -> list[list[TestCase]]:
    """Split cases into manageable chunks."""

    return [test_cases[index:index + batch_size] for index in range(0, len(test_cases), batch_size)]


def validate_response(result: EvaluationResult) -> str:
    """Force-fail low-confidence results."""

    if result.score < LOW_CONFIDENCE_THRESHOLD:
        result.passed = False
        if "FAIL_LOW_CONFIDENCE" not in result.error_reason:
            result.error_reason = (
                f"{result.error_reason}, FAIL_LOW_CONFIDENCE".strip(", ")
                if result.error_reason
                else "FAIL_LOW_CONFIDENCE"
            )
        return "FAIL_LOW_CONFIDENCE"
    return "PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProcureBuddy RAG evaluation sequentially.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_FILE),
        help=f"Path to dataset JSON (default: {DEFAULT_DATASET_FILE.name})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of cases to run. Defaults to 50 for controlled debugging.",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"Chatbot API base URL (default: {API_URL})",
    )
    parser.add_argument(
        "--report-file",
        default=str(CSV_REPORT_FILE),
        help=f"CSV report path (default: {CSV_REPORT_FILE})",
    )
    parser.add_argument(
        "--failed-file",
        default=str(FAILED_CASES_FILE),
        help=f"Failed-cases JSON path (default: {FAILED_CASES_FILE})",
    )
    parser.add_argument(
        "--duration",
        type=_parse_duration,
        default=None,
        help="Optional wall-clock cap for the run. Examples: 900, 900s, 15m, 1h.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT,
        help=f"Per-request API timeout in seconds (default: {TIMEOUT}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY,
        help=f"Fixed delay between cases in seconds (default: {DELAY}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=RETRIES,
        help=f"Retry count for API errors/timeouts (default: {RETRIES}).",
    )
    parser.add_argument(
        "--avoid-recent-minutes",
        type=int,
        default=RECENT_CASE_COOLDOWN_MINUTES,
        help=f"Avoid questions used in the last N minutes (default: {RECENT_CASE_COOLDOWN_MINUTES}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of cases to process per batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=USE_RESUME,
        help="Resume from the existing CSV report by skipping already processed case IDs.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore the existing CSV report and allow intentional reruns of already processed cases.",
    )
    parser.add_argument(
        "--strict-eval",
        action="store_true",
        help="Apply balanced contract checks: semantic>=0.6, relevance>=0.6, and FINAL DECISION required.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)


def call_chat_api(
    session: requests.Session,
    api_url: str,
    question: str,
    timeout_seconds: float,
    retries: int,
) -> tuple[str | None, str]:
    """Call the chatbot API with retries and rate-limit-aware backoff."""

    endpoint = f"{api_url.rstrip('/')}/chat"
    payload = {
        "message": question,
        "user": "eval-runner@procurebuddy.local",
        "displayName": "Eval Runner",
        "bypass_cache": True,
    }

    max_attempts = retries + 1
    for attempt in range(max_attempts):
        attempt_label = f"{attempt + 1}/{max_attempts}"
        print(f"Request start: {_timestamp()} | attempt={attempt_label} | question={question[:80]}")
        try:
            response = session.post(endpoint, json=payload, timeout=timeout_seconds)

            if response.status_code == 429:
                error_text = f"HTTP 429: {response.text[:200]}"
                logger.warning("ERROR case=rate_limit attempt=%s question=%s", attempt_label, question[:80])
                if attempt < retries:
                    time.sleep(retry_backoff_seconds(attempt))
                    continue
                print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=429")
                return None, error_text

            response.raise_for_status()
            body = response.json()
            answer = str(body.get("answer") or body.get("response") or "").strip()
            if not answer:
                if attempt < retries:
                    logger.warning("ERROR case=empty_answer attempt=%s question=%s", attempt_label, question[:80])
                    time.sleep(retry_backoff_seconds(attempt))
                    continue
                print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=empty_answer")
                return None, "Empty answer"
            print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=success")
            return answer, ""

        except requests.exceptions.Timeout:
            logger.warning("ERROR case=timeout attempt=%s question=%s", attempt_label, question[:80])
            if attempt < retries:
                time.sleep(retry_backoff_seconds(attempt))
                continue
            print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=timeout")
            return None, f"Timeout after {timeout_seconds}s"

        except requests.exceptions.ConnectionError as exc:
            logger.warning("ERROR case=connection attempt=%s question=%s", attempt_label, question[:80])
            if attempt < retries:
                time.sleep(retry_backoff_seconds(attempt))
                continue
            print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=connection_error")
            return None, f"Connection error: {exc}"

        except requests.exceptions.HTTPError as exc:
            body_text = exc.response.text[:200] if exc.response is not None else ""
            logger.warning("ERROR case=http attempt=%s question=%s", attempt_label, question[:80])
            if attempt < retries and exc.response is not None and exc.response.status_code >= 500:
                time.sleep(retry_backoff_seconds(attempt))
                continue
            print(
                f"Request end: {_timestamp()} | attempt={attempt_label} | "
                f"status=http_{exc.response.status_code if exc.response else '?'}"
            )
            return None, f"HTTP {exc.response.status_code if exc.response else '?'}: {body_text}"

        except Exception as exc:  # pragma: no cover - defensive runner safety
            logger.exception("ERROR case=unexpected attempt=%s question=%s", attempt_label, question[:80])
            print(f"Request end: {_timestamp()} | attempt={attempt_label} | status=unexpected_error")
            return None, f"Unexpected error: {exc}"

    return None, "Unknown API failure"


def process_case(
    session: requests.Session,
    api_url: str,
    case: TestCase,
    strict_eval: bool = False,
    timeout_seconds: float = TIMEOUT,
    retries: int = RETRIES,
) -> EvaluationResult:
    """Run one case end-to-end and always return a result object."""

    answer, error_reason = call_chat_api(session, api_url, case.question, timeout_seconds, retries)
    result = evaluate_response(case, answer or "", error_reason=error_reason, strict=strict_eval)
    validation_status = validate_response(result)

    if error_reason:
        logger.warning("FAIL id=%s score=%.3f reason=%s", case.id, result.score, result.error_reason)
    elif result.passed:
        logger.info("PASS id=%s score=%.3f", case.id, result.score)
    else:
        logger.info("FAIL id=%s score=%.3f reason=%s", case.id, result.score, result.error_reason)
    if validation_status == "FAIL_LOW_CONFIDENCE":
        logger.warning("FORCE_FAIL id=%s score=%.3f reason=%s", case.id, result.score, validation_status)

    return result


def print_summary(results: list[EvaluationResult], resume_enabled: bool) -> None:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    failed = total - passed
    average_score = (sum(item.score for item in results) / total) if total else 0.0
    accuracy = (passed / total * 100.0) if total else 0.0

    print("\n" + "=" * 72)
    print("ProcureBuddy Evaluation Summary")
    print("=" * 72)
    print(f"Total Cases:     {total}")
    print(f"Passed:          {passed}")
    print(f"Failed:          {failed}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Average Score:   {average_score:.3f}")
    print(f"Workers:         {WORKERS} (sequential only)")
    print(f"Resume Enabled:  {resume_enabled}")
    print("=" * 72 + "\n")


def main() -> int:
    configure_logging()
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    report_file = Path(args.report_file).resolve()
    failed_file = Path(args.failed_file).resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    processed_ids = get_processed_ids(report_file) if args.resume else set()
    recent_ids = recent_question_ids(
        history_file=RECENT_CASE_HISTORY_FILE,
        cooldown_minutes=args.avoid_recent_minutes,
    )
    fresh_case_count = count_fresh_cases(
        dataset_path,
        excluded_case_ids=recent_ids,
        hard_excluded_case_ids=processed_ids,
    )

    # Random sample with recent-repeat avoidance plus hard resume exclusion.
    selection: DatasetSelection = load_dataset(
        dataset_path,
        limit=args.limit,
        excluded_case_ids=recent_ids,
        hard_excluded_case_ids=processed_ids,
    )
    cases = selection.cases
    if not cases:
        print("No pending fresh cases available for this run.")
        print(f"Already processed in report: {len(processed_ids)}")
        print(f"Recently avoided by history window: {len(recent_ids)}")
        if args.resume and processed_ids:
            print(f"Resume source: {report_file}")
            print("Tip: rerun intentionally with `--no-resume`, or write to a fresh file with `--report-file`.")
        return 0

    append_case_selection_history(
        history_file=RECENT_CASE_HISTORY_FILE,
        dataset_file=dataset_path,
        cases=cases,
    )
    initial_estimate = _estimate_remaining_seconds(
        total_cases=len(cases),
        completed_cases=0,
        elapsed_seconds=0.0,
        timeout_seconds=args.timeout,
        delay_seconds=args.delay,
        duration_limit_seconds=args.duration,
    )
    print(f"\nLoaded {len(cases)} test cases from {dataset_path.name}")
    print(f"Target: {args.api_url}")
    print(f"Workers: {WORKERS} | Timeout: {args.timeout}s | Retries: {args.retries} | Delay: {args.delay:.1f}s")
    print(f"Batch Size: {args.batch_size}")
    print(f"Strict Eval: {args.strict_eval}")
    if args.duration is not None:
        print(f"Duration Cap: {_format_duration(args.duration)}")
    print(f"Recent Repeat Window: {args.avoid_recent_minutes} minutes")
    print(f"Fresh Questions Available: {fresh_case_count}")
    print(f"Already Processed In CSV: {len(processed_ids)}")
    logger.info("Fresh questions selected: %s", selection.fresh_selected_count)
    logger.info("Repeated questions avoided: %s", selection.repeated_questions_avoided)
    print(f"Resume: {args.resume}")
    print(f"CSV Report Path: {report_file}")
    print(f"Failed JSON Path: {failed_file}\n")
    print(f"Estimated time left at start: {_format_duration(initial_estimate)}\n")
    if selection.reused_questions_count > 0:
        print(
            f"Note: {selection.reused_questions_count} repeated questions were reused because only "
            f"{selection.fresh_selected_count} fresh questions were available in this cooldown window.\n"
        )

    results: list[EvaluationResult] = []
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    run_started_at = time.perf_counter()
    stopped_by_duration = False

    try:
        processed_in_run = 0
        for batch_index, batch in enumerate(get_batches(cases, args.batch_size), start=1):
            print(f"Batch {batch_index}: processing {len(batch)} cases")
            batch_results: list[EvaluationResult] = []
            for case in batch:
                case_index = processed_in_run + 1
                elapsed_before_case = time.perf_counter() - run_started_at
                if args.duration is not None and elapsed_before_case >= args.duration:
                    stopped_by_duration = True
                    print(
                        f"Duration cap reached after {_format_duration(elapsed_before_case)}. "
                        f"Stopping before case {case_index}."
                    )
                    break

                estimated_left = _estimate_remaining_seconds(
                    total_cases=len(cases),
                    completed_cases=len(results),
                    elapsed_seconds=elapsed_before_case,
                    timeout_seconds=args.timeout,
                    delay_seconds=args.delay,
                    duration_limit_seconds=args.duration,
                )
                print(
                    f"Progress: case {case_index}/{len(cases)} | "
                    f"elapsed={_format_duration(elapsed_before_case)} | "
                    f"estimated time left={_format_duration(estimated_left)}"
                )
                result = process_case(
                    session,
                    args.api_url,
                    case,
                    strict_eval=args.strict_eval,
                    timeout_seconds=args.timeout,
                    retries=args.retries,
                )
                results.append(result)
                batch_results.append(result)
                processed_in_run += 1

                if processed_in_run < len(cases):
                    elapsed_after_case = time.perf_counter() - run_started_at
                    remaining_cap = None if args.duration is None else max(args.duration - elapsed_after_case, 0.0)
                    sleep_seconds = args.delay if remaining_cap is None else min(args.delay, remaining_cap)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    elif args.duration is not None:
                        stopped_by_duration = True
                        print(
                            f"Duration cap reached after {_format_duration(elapsed_after_case)}. "
                            "Stopping before the next case."
                        )
                        break

            if batch_results:
                append_csv_report(batch_results, report_file)
                append_failed_cases(batch_results, failed_file)
            if stopped_by_duration:
                break
    except KeyboardInterrupt:
        logger.warning("ERROR case=runner message=Interrupted by user, writing partial reports")
    except Exception as exc:  # pragma: no cover - defensive runner safety
        logger.exception("ERROR case=runner message=Unhandled runner failure")
        print(f"\nRunner error: {exc}")
    finally:
        total_elapsed = time.perf_counter() - run_started_at
        session.close()
        print_summary(results, resume_enabled=args.resume)
        print(f"Elapsed Time: {_format_duration(total_elapsed)}")
        if stopped_by_duration:
            print("Stopped Early: duration cap reached")
        print(f"CSV Report:   {report_file}")
        print(f"Failed JSON:  {failed_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
