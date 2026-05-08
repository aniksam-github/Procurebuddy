"""Generate a threshold-focused procurement regression dataset.

This dataset is intentionally narrow and deterministic:
- it stresses amount parsing
- it stresses route-boundary selection
- it produces ready-to-run ground truth for overnight regression tests
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "testing" / "generated"
DEFAULT_OUTPUT = OUTPUT_DIR / "threshold_regression_dataset_3000.json"

AMOUNT_TEXTS = [
    "Rs. 50",
    "Rs. 500",
    "Rs. 5,000",
    "Rs. 50,000",
    "Rs. 1,50,000",
    "Rs. 2,00,000",
    "Rs. 2,00,001",
    "Rs. 3,00,000",
    "Rs. 4,99,999",
    "Rs. 5,00,000",
    "Rs. 5,00,001",
    "Rs. 8,00,000",
    "Rs. 25,00,000",
    "Rs. 49,00,000",
    "Rs. 50,00,000",
    "Rs. 50,00,001",
    "Rs. 56,00,000",
    "Rs. 1,00,00,000",
    "1 lakh",
    "2 lakh",
    "2.00001 lakh",
    "3 lakh",
    "5 lakh",
    "5.00001 lakh",
    "8 lakh",
    "25 lakh",
    "49 lakh",
    "50 lakh",
    "50.00001 lakh",
    "56 lakh",
    "0.5 crore",
    "1 crore",
]

QUESTION_TEMPLATES = [
    "how to procure an item of {amount}?",
    "process for purchase of {amount}",
    "what is the procurement method for {amount}?",
    "how can I buy equipment of {amount}?",
    "which rule applies for a purchase of {amount}?",
    "show the process for procurement of {amount}",
    "what committee is needed for {amount}?",
    "procurement workflow for {amount}",
    "how to purchase a machine costing {amount}?",
    "which tender type applies for {amount}?",
    "tell me the route for procurement of {amount}",
    "for a case of {amount}, which mode should apply?",
    "what should be the buying process for {amount}?",
    "which committee handles procurement worth {amount}?",
    "what is the correct procurement route for {amount}?",
    "for purchase value {amount}, what is the applicable method?",
]

ITEM_HINTS = [
    "",
    " for a laboratory instrument",
    " for a microscope",
    " for an equipment purchase",
    " for a machine",
    " for a procurement file",
]


def _parse_numeric(value: str) -> float:
    return float(value.replace(",", "").strip())


def normalize_amount(amount_text: str) -> int:
    """Normalize human amount text into integer rupees."""
    normalized = amount_text.strip().lower().replace(",", "")

    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(crore|crores|cr)\b", normalized)
    if crore_match:
        return int(round(_parse_numeric(crore_match.group(1)) * 10_000_000))

    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs)\b", normalized)
    if lakh_match:
        return int(round(_parse_numeric(lakh_match.group(1)) * 100_000))

    thousand_match = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|k)\b", normalized)
    if thousand_match:
        return int(round(_parse_numeric(thousand_match.group(1)) * 1_000))

    rs_match = re.search(r"rs\.?\s*(\d+(?:\.\d+)?)", normalized)
    if rs_match:
        return int(round(_parse_numeric(rs_match.group(1))))

    numeric_match = re.search(r"\d+(?:\.\d+)?", normalized)
    if numeric_match:
        return int(round(_parse_numeric(numeric_match.group(0))))

    raise ValueError(f"Could not normalize amount: {amount_text}")


def expected_mode(amount_text: str) -> tuple[str, str]:
    rupees = normalize_amount(amount_text)
    if rupees <= 200_000:
        return "Direct Purchase", "Rule 154"
    if rupees <= 500_000:
        return "LPC", "Rule 155"
    if rupees <= 5_000_000:
        return "LTE", "Rule 162"
    return "OTE", "Rule 161"


def generate_dataset(size: int = 3000, seed: int = 20260505) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []

    for case_id in range(1, size + 1):
        amount_text = rng.choice(AMOUNT_TEXTS)
        template = rng.choice(QUESTION_TEMPLATES)
        item_hint = rng.choice(ITEM_HINTS)
        question = f"{template.format(amount=amount_text).rstrip('?')}{item_hint}?"
        mode, rule = expected_mode(amount_text)
        rupees = normalize_amount(amount_text)
        records.append(
            {
                "id": case_id,
                "question": question,
                "amount_text": amount_text,
                "amount_rupees": rupees,
                "expected_mode": mode,
                "expected_rule": rule,
                "test_family": "threshold_regression",
            }
        )

    mode_distribution = Counter(str(record["expected_mode"]) for record in records)
    amount_distribution = Counter(str(record["amount_text"]) for record in records)
    summary = {
        "seed": seed,
        "size": size,
        "mode_distribution": dict(mode_distribution),
        "unique_amounts": len(amount_distribution),
        "amount_distribution": dict(amount_distribution),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a threshold-focused overnight regression dataset.")
    parser.add_argument("--size", type=int, default=3000, help="Number of questions to generate.")
    parser.add_argument("--seed", type=int, default=20260505, help="Random seed for deterministic generation.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional summary JSON path. Defaults to <output>_summary.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    summary_path = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records, summary = generate_dataset(size=args.size, seed=args.seed)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote dataset: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Size: {summary['size']}")
    print(f"Mode distribution: {summary['mode_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
