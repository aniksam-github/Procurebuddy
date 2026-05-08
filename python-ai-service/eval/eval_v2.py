import json
from pathlib import Path

def _extract_quick_answer_block(resp: str) -> str:
    if not resp:
        return ""
    # Grab content between "## Quick Answer" and next "## " heading.
    import re

    m = re.search(r"##\s*Quick Answer\s*(.*?)\n##\s+", resp, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: try end-of-string.
    m2 = re.search(r"##\s*Quick Answer\s*(.*)$", resp, flags=re.DOTALL | re.IGNORECASE)
    return m2.group(1) if m2 else ""


def extract_mode(resp: str) -> str:
    """Extract predicted procurement mode from the Quick Answer section."""
    if not resp:
        return "UNKNOWN"

    import re
    # Strict Hard-Coded Regex matching only [A-Z_]+ as requested
    m = re.search(r"##\s*Quick Answer.*?Applicable mode:\s*([A-Z_]+)", resp, flags=re.DOTALL | re.IGNORECASE)
    
    if m:
        t = m.group(1).upper()
        if t in ["LPC", "LTE", "OTE"]:
            return t
        if t == "DIRECT_PURCHASE" or t == "DIRECT":
            return "Direct Purchase"
    
    # Check if rule based output applies
    m2 = re.search(r"##\s*Quick Answer\s*- Applicable mode:\s*([A-Za-z\s]+)", resp, flags=re.IGNORECASE)
    if m2:
        t = m2.group(1).lower().strip()
        if "direct" in t: return "Direct Purchase"
        if "lpc" in t or "local" in t: return "LPC"
        if "lte" in t or "limited" in t: return "LTE"
        if "ote" in t or "open" in t: return "OTE"
        
    return "UNKNOWN"



def get_procurement_mode(amount: float) -> str:
    if amount <= 200000:
        return "Direct Purchase"
    elif amount <= 500000:
        return "LPC"
    elif amount <= 5000000:
        return "LTE"
    else:
        return "OTE"

def classify_bug(resp: str) -> str:
    if not resp or resp.strip() == "":
        return "LLM_TIMEOUT"
    if "UNKNOWN" in extract_mode(resp):
        return "MODE_EXTRACTION_FAIL"
    if "GFR 2025" in resp:
        return "HALLUCINATION_GFR2025"
    if "## Quick Answer" not in resp:
        return "FORMAT_FAIL"
    return "NONE"

def evaluate_response(response: str, expected_mode: str) -> dict:
    predicted = extract_mode(response)
    bug = classify_bug(response)
    passed = predicted == expected_mode
    
    return {
        "expected": expected_mode,
        "predicted": predicted,
        "passed": passed,
        "bug": bug
    }
