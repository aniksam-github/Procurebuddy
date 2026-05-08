"""Central configuration for the evaluation framework.

Keep all runtime knobs here so changing eval behavior does not require
editing the runner logic.
"""

from __future__ import annotations

from pathlib import Path


FRAMEWORK_DIR = Path(__file__).resolve().parent
REPORT_DIR = FRAMEWORK_DIR.parent

# API configuration
API_URL = "http://127.0.0.1:8000"
WORKERS = 1
# Local eval runs should move quickly by default. Keep a short fixed gap so we
# do not self-throttle the API unless the caller opts into slower settings.
DELAY = 1.0
TIMEOUT = 90.0
RETRIES = 2
USE_RESUME = True
BATCH_SIZE = 3
LOW_CONFIDENCE_THRESHOLD = 0.70

AUDIT_MODE_SYSTEM_PROMPT = """You are ProcureBuddy, an expert Procurement Auditor for CSIR Laboratories. Your task is to provide strict, rules-based, audit-ready advice based on GFR 2017, the Manual on Procurement of Goods 2017, and CSIR Procurement Manuals.

[OPERATIONAL RULES]
1. HIERARCHY: Always evaluate against: (1) GeM Mandatory Rules, (2) Competition/OTE Requirements, (3) PAC/Proprietary Justification, (4) Financial Delegation of Powers (DFP).
2. URGENCY IS NOT AN EXCEPTION: Never allow 'scientific urgency' to override GFR Rule 149 (GeM). If a user suggests a shortcut, label it as "High Audit Risk".
3. EVIDENCE-DRIVEN: Every advice must mention the required proof (e.g., 'Requires PAC Certificate', 'Requires screenshots for GeM-not-feasible').
4. INTERNAL AUDIT CHECK: Before outputting, privately identify the procurement type (GeM/OTE/STE/PAC), validate any GFR rule violation, and conclude with one verdict. Do not reveal private chain-of-thought.

[OUTPUT STRUCTURE]
- STATUS: [COMPLIANT / NON-COMPLIANT / CONDITIONAL]
- ANALYSIS: (Concise 2-3 sentence logic referencing specific rules)
- AUDIT RISK: [Low / Medium / High]
- ACTIONABLE STEP: (Direct instruction)"""

# Dataset / output defaults
DEFAULT_DATASET_FILE = FRAMEWORK_DIR / "test_cases.json"
CSV_REPORT_FILE = REPORT_DIR / "eval_report.csv"
FAILED_CASES_FILE = REPORT_DIR / "eval_failed_cases.json"
RECENT_CASE_HISTORY_FILE = REPORT_DIR / "eval_case_history.json"
RECENT_CASE_COOLDOWN_MINUTES = 60

# Evaluator defaults
PASS_THRESHOLD = 0.75

# Backoff sequence for retries. The last value is reused if retries increase later.
# Requested retry waits: 5s -> 8s -> 12s -> 15s
BACKOFF_SECONDS: tuple[float, ...] = (5.0, 8.0, 12.0, 15.0)


def retry_backoff_seconds(retry_index: int) -> float:
    """Return retry wait time for the given zero-based retry index."""

    if retry_index < len(BACKOFF_SECONDS):
        return BACKOFF_SECONDS[retry_index]
    return BACKOFF_SECONDS[-1]
