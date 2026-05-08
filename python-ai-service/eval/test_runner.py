"""ProcureBuddy RAG Evaluation Runner.

Runs random batches of questions against the /chat endpoint and measures:
  1. Compliance Precision — keyword coverage from expected_keywords
  2. Hallucination Rate   — invented rule numbers not in the GFR truth set
  3. Latency              — per-question and aggregate timing

Usage:
    python -m eval.test_runner                    # 3 batches × 50 = 150 questions
    python -m eval.test_runner --batch-size 10    # quick smoke test
    python -m eval.test_runner --batches 1        # single batch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# ── Configuration ───────────────────────────────────────────────────────────
BASE_URL = os.getenv("PROCUREBUDDY_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BASE_URL}/chat"
QUESTIONS_FILE = Path(__file__).parent / "all_questions.json"
REPORT_FILE = Path(__file__).parent / "eval_report.csv"
SUMMARY_FILE = Path(__file__).parent / "eval_summary.json"
HISTORY_WINDOW_HOURS = 1

# Known valid GFR rule numbers — anything outside this set is a hallucination
VALID_RULE_NUMBERS = {
    "21", "135", "136", "137", "138", "139", "140", "141", "142", "143",
    "144", "145", "146", "147", "148", "149", "150", "151", "152", "153",
    "154", "155", "156", "157", "158", "159", "160", "161", "162", "163",
    "164", "165", "166", "167", "168", "169", "170", "171", "172", "173",
    "174", "175", "176", "177", "178", "196", "197", "198",
}


# ── Question Loader ────────────────────────────────────────────────────────

def load_all_questions(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── History Tracking ────────────────────────────────────────────────────────

def get_recent_history(csv_file: Path, window_hours: int = HISTORY_WINDOW_HOURS) -> set[str]:
    """Return question IDs run within the last `window_hours`."""
    history: set[str] = set()
    if not csv_file.exists():
        return history
    cutoff = datetime.now() - timedelta(hours=window_hours)
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts > cutoff:
                    history.add(row["id"])
            except (KeyError, ValueError):
                continue
    return history


# ── Metric Calculators ──────────────────────────────────────────────────────

from eval.eval_v2 import extract_mode

def measure_compliance_precision(answer: str, expected_mode: str) -> float:
    """Return 1.0 if predicted mode matches expected mode, else 0.0."""
    predicted = extract_mode(answer)
    return 1.0 if predicted == expected_mode else 0.0


def infer_expected_mode_from_keywords(expected_keywords: list[str]) -> str:
    """Infer expected mode from rule numbers / common tokens in expected_keywords."""
    blob = " ".join(expected_keywords or []).lower()
    # Prefer explicit route tokens if present.
    if "direct purchase" in blob:
        return "Direct Purchase"
    if "local purchase committee" in blob or re.search(r"\blpc\b", blob):
        return "LPC"
    if "limited tender" in blob or re.search(r"\blte\b", blob):
        return "LTE"
    if "open tender" in blob or re.search(r"\bote\b", blob):
        return "OTE"

    # Rule-based inference (covers TH-017 style cases).
    if re.search(r"\brule\s*154\b", blob):
        return "Direct Purchase"
    if re.search(r"\brule\s*155\b", blob):
        return "LPC"
    if re.search(r"\brule\s*162\b", blob):
        return "LTE"
    if re.search(r"\brule\s*161\b", blob):
        return "OTE"
    return "UNKNOWN"


def detect_hallucinated_rules(answer: str) -> list[str]:
    """Return rule numbers cited in the answer that are NOT in the valid set."""
    cited_rules = set(re.findall(r"\brule\s+(\d{2,3})\b", answer, re.IGNORECASE))
    return sorted(cited_rules - VALID_RULE_NUMBERS)


def measure_hallucination_rate(answer: str) -> tuple[float, list[str]]:
    """Return (hallucination_rate, list_of_bad_rules). Rate is 0.0–1.0."""
    all_cited = set(re.findall(r"\brule\s+(\d{2,3})\b", answer, re.IGNORECASE))
    if not all_cited:
        return 0.0, []
    bad = sorted(all_cited - VALID_RULE_NUMBERS)
    return round(len(bad) / len(all_cited), 4), bad


# ── API Caller ──────────────────────────────────────────────────────────────

from app.services.response_service import generate_response

def call_procurebuddy(question_text: str, user: str = "eval_runner") -> dict:
    """Call the generate_response locally and return the response + latency."""
    start = time.perf_counter()
    try:
        data = generate_response(question_text, user=user)
        latency = round(time.perf_counter() - start, 3)
        return {
            "answer": data.get("answer", ""),
            "generation_mode": data.get("generation_mode", "unknown"),
            "metadata": {},
            "latency_seconds": latency,
            "error": None,
        }
    except Exception as exc:
        latency = round(time.perf_counter() - start, 3)
        return {
            "answer": "",
            "generation_mode": "error",
            "metadata": {},
            "latency_seconds": latency,
            "error": str(exc),
        }


# ── CSV Writer ──────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "id", "category", "question",
    "compliance_precision", "accuracy_applicable", "accuracy_pass", "hallucination_rate", "quality_score", "hallucinated_rules",
    "latency_seconds", "generation_mode", "answer_length", "error",
]


def is_mode_accuracy_applicable(category: str, expected_mode: str) -> bool:
    """Mode-accuracy should be computed only for threshold/routing tasks."""
    return category.strip().upper() == "THRESHOLD"


def init_csv(csv_file: Path) -> None:
    if not csv_file.exists():
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row(csv_file: Path, row: dict) -> None:
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ── ETA Tracker ─────────────────────────────────────────────────────────────

class ETATracker:
    """Tracks per-question latency and estimates remaining time."""

    def __init__(self, total_questions: int):
        self.total_questions = total_questions
        self.completed = 0
        self.total_time = 0.0

    def update(self, latency: float) -> str:
        """Record a question's latency and return formatted ETA string."""
        self.completed += 1
        self.total_time += latency
        avg_per_q = self.total_time / self.completed
        remaining = self.total_questions - self.completed
        eta_seconds = avg_per_q * remaining

        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        secs = int(eta_seconds % 60)

        if hours > 0:
            return f"ETA: {hours}h {minutes}m  (avg {avg_per_q:.1f}s/q)"
        elif minutes > 0:
            return f"ETA: {minutes}m {secs}s  (avg {avg_per_q:.1f}s/q)"
        else:
            return f"ETA: {secs}s  (avg {avg_per_q:.1f}s/q)"


# ── Batch Runner ────────────────────────────────────────────────────────────

def run_evaluation_batch(
    all_questions: list[dict],
    batch_size: int = 50,
    batch_number: int = 1,
    eta: ETATracker | None = None,
) -> dict:
    """Run one batch of randomly sampled, non-repeated questions."""
    history = get_recent_history(REPORT_FILE)
    available = [q for q in all_questions if q["id"] not in history]

    if len(available) < batch_size:
        print(f"  [!] Only {len(available)} un-tested questions remain (need {batch_size})")
        batch_size = len(available)
        if batch_size == 0:
            print("  [X]  No questions left -- reset eval_report.csv or wait for history window")
            return {}

    selected = random.sample(available, batch_size)
    init_csv(REPORT_FILE)

    # Batch-level accumulators
    total_compliance = 0.0
    compliance_applicable_count = 0
    total_halluc_rate = 0.0
    total_latency = 0.0
    halluc_count = 0
    error_count = 0

    print(f"\n{'='*70}")
    print(f"  BATCH {batch_number}  --  {batch_size} questions")
    print(f"{'='*70}")

    for i, question in enumerate(selected, start=1):
        qid = question["id"]
        qtext = question["question"]
        expected_kws = question.get("expected_keywords", [])
        expected = infer_expected_mode_from_keywords(expected_kws)
        category = question.get("category", "UNKNOWN")

        print(f"  [{i:3d}/{batch_size}]  {qid:<8s} {qtext[:55]}...", end="", flush=True)

        result = call_procurebuddy(qtext)
        answer = result["answer"]

        # ── Metrics ──
        accuracy_applicable = is_mode_accuracy_applicable(category, expected) and expected != "UNKNOWN" and not bool(result["error"])
        compliance = measure_compliance_precision(answer, expected) if accuracy_applicable else 0.0
        halluc_rate, bad_rules = measure_hallucination_rate(answer)
        latency = result["latency_seconds"]

        if accuracy_applicable:
            total_compliance += compliance
            compliance_applicable_count += 1
        total_halluc_rate += halluc_rate
        total_latency += latency
        if bad_rules:
            halluc_count += 1
        if result["error"]:
            error_count += 1

        # Status indicator + ETA
        # Status: do not "skip" execution; always show some result.
        if accuracy_applicable:
            status = "OK" if compliance >= 1.0 and not bad_rules else "FAIL"
        else:
            # Non-threshold (or missing expected_mode) gets a quality-only status.
            status = "OK" if (not result["error"] and not bad_rules) else "WARN" if not bad_rules else "FAIL"
        eta_str = ""
        if eta:
            eta_str = f"  | {eta.update(latency)}"
        acc_label = f"{compliance:.2f}" if accuracy_applicable else "NA"
        print(f"  {status}  acc={acc_label}  halluc={halluc_rate:.2f}  {latency:.1f}s{eta_str}")

        append_row(REPORT_FILE, {
            "timestamp": datetime.now().isoformat(),
            "id": qid,
            "category": category,
            "question": qtext,
            "compliance_precision": round(compliance, 4) if accuracy_applicable else "",
            "accuracy_applicable": int(accuracy_applicable),
            "accuracy_pass": int(compliance == 1.0) if accuracy_applicable else "",
            "hallucination_rate": halluc_rate,
            "quality_score": round(1.0 - halluc_rate, 4),
            "hallucinated_rules": ";".join(bad_rules) if bad_rules else "",
            "latency_seconds": latency,
            "generation_mode": result["generation_mode"],
            "answer_length": len(answer),
            "error": result["error"] or "",
        })

    # ── Batch Summary ──
    n = batch_size
    summary = {
        "batch": batch_number,
        "questions_run": n,
        "accuracy_applicable_count": compliance_applicable_count,
        "avg_compliance_precision": round(total_compliance / max(1, compliance_applicable_count), 4),
        "avg_hallucination_rate": round(total_halluc_rate / max(1, n), 4),
        "questions_with_hallucinations": halluc_count,
        "avg_latency_seconds": round(total_latency / max(1, n), 3),
        "max_latency_seconds": round(max(total_latency, 0), 3),
        "error_count": error_count,
    }

    print(f"  -- Batch {batch_number} Summary --")
    print(f"     Compliance Precision : {summary['avg_compliance_precision']:.2%}")
    print(f"     Accuracy Scope       : {summary['accuracy_applicable_count']}/{n} threshold questions")
    print(f"     Hallucination Rate   : {summary['avg_hallucination_rate']:.2%}")
    print(f"     Hallucinated Answers : {halluc_count}/{n}")
    print(f"     Avg Latency          : {summary['avg_latency_seconds']:.2f}s")
    print(f"     Errors               : {error_count}/{n}")
    if eta:
        elapsed_h = eta.total_time / 3600
        print(f"     Progress             : {eta.completed}/{eta.total_questions} done  ({elapsed_h:.1f}h elapsed)")
    return summary


# ── Final Report ────────────────────────────────────────────────────────────

def generate_final_report(batch_summaries: list[dict]) -> None:
    """Aggregate all batch summaries and write a JSON report."""
    if not batch_summaries:
        return

    total_q = sum(b["questions_run"] for b in batch_summaries)
    total_accuracy_scope = sum(int(b.get("accuracy_applicable_count", 0)) for b in batch_summaries)
    avg_cp = (
        sum(b["avg_compliance_precision"] * int(b.get("accuracy_applicable_count", 0)) for b in batch_summaries)
        / max(1, total_accuracy_scope)
    )
    avg_hr = sum(b["avg_hallucination_rate"] * b["questions_run"] for b in batch_summaries) / max(1, total_q)
    avg_lat = sum(b["avg_latency_seconds"] * b["questions_run"] for b in batch_summaries) / max(1, total_q)
    total_halluc = sum(b["questions_with_hallucinations"] for b in batch_summaries)
    total_errors = sum(b["error_count"] for b in batch_summaries)

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "total_questions": total_q,
        "accuracy_scope_questions": total_accuracy_scope,
        "batches": len(batch_summaries),
        "overall_compliance_precision": round(avg_cp, 4),
        "overall_hallucination_rate": round(avg_hr, 4),
        "total_hallucinated_answers": total_halluc,
        "overall_avg_latency_seconds": round(avg_lat, 3),
        "total_errors": total_errors,
        "batch_details": batch_summaries,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  FINAL REPORT  —  {total_q} questions across {len(batch_summaries)} batches")
    print(f"{'='*70}")
    print(f"  Compliance Precision  : {avg_cp:.2%}")
    print(f"  Accuracy Scope        : {total_accuracy_scope}/{total_q} threshold questions")
    print(f"  Hallucination Rate    : {avg_hr:.2%} ({total_halluc} answers with fake rules)")
    print(f"  Avg Latency           : {avg_lat:.2f}s")
    print(f"  Errors                : {total_errors}")
    print(f"\n  Reports saved to:")
    print(f"    CSV  -> {REPORT_FILE}")
    print(f"    JSON -> {SUMMARY_FILE}")
    print(f"{'='*70}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ProcureBuddy RAG Evaluation Runner")
    parser.add_argument("--batch-size", type=int, default=50, help="Questions per batch (default: 50)")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches (default: 3)")
    parser.add_argument("--gap", type=int, default=5, help="Seconds between batches (default: 5)")
    parser.add_argument("--url", type=str, default=None, help="Override base URL")
    parser.add_argument("--questions", type=str, default=None, help="Path to questions JSON file (default: eval/all_questions.json)")
    args = parser.parse_args()

    if args.url:
        global CHAT_ENDPOINT
        CHAT_ENDPOINT = f"{args.url.rstrip('/')}/chat"

    questions_file = Path(args.questions) if args.questions else QUESTIONS_FILE

    print(f"\n>> ProcureBuddy Evaluation Runner")
    print(f"   Endpoint : {CHAT_ENDPOINT}")
    print(f"   Batches  : {args.batches} x {args.batch_size} = {args.batches * args.batch_size} questions")
    print(f"   Questions: {questions_file}")

    if not questions_file.exists():
        print(f"\n[X] Questions file not found: {questions_file}")
        sys.exit(1)

    all_questions = load_all_questions(questions_file)
    print(f"   Loaded   : {len(all_questions)} questions from bank")


    # Health check
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        health.raise_for_status()
        info = health.json()
        print(f"   Model    : {info.get('embedding_model', 'unknown')}")
        print(f"   Chunks   : {info.get('chunk_count', '?')}")
        print(f"   LLM      : {info.get('model_name', 'unknown')}")
    except Exception as exc:
        print(f"\n[!] Health check failed: {exc}")
        print(f"   Make sure the server is running at {BASE_URL}")
        sys.exit(1)

    total_planned = args.batches * args.batch_size
    eta = ETATracker(total_questions=min(total_planned, len(all_questions)))
    print(f"\n   Estimated : ~{total_planned} questions")
    print(f"   ETA will update after the first question completes.\n")

    batch_summaries: list[dict] = []
    for i in range(1, args.batches + 1):
        summary = run_evaluation_batch(all_questions, args.batch_size, batch_number=i, eta=eta)
        if summary:
            batch_summaries.append(summary)
        if i < args.batches:
            print(f"\n  ... Waiting {args.gap}s before next batch...")
            time.sleep(args.gap)

    generate_final_report(batch_summaries)


if __name__ == "__main__":
    main()
