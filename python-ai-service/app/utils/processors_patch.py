"""Scratch: validates new extract_amount_lakhs logic before patching."""
import re

def extract_amount_lakhs(message: str) -> float | None:
    cleaned = message.lower().strip()
    normalized = cleaned.replace(",", "")

    # 1. Crore — explicit unit
    m = re.search(r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", normalized)
    if m:
        return float(m.group(1)) * 100.0

    # 2. Lakh — explicit keyword (NOT bare 'l')
    m = re.search(r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)\b", normalized)
    if m:
        return float(m.group(1))

    # 3. Thousand / k
    m = re.search(r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:thousand|k)\b", normalized)
    if m:
        return float(m.group(1)) / 100.0

    # 4. Bare currency prefix without explicit unit (Rs. 56 = 56 rupees)
    m = re.search(r"(?:rs\.?|inr|₹)\s*([\d]+(?:\.\d+)?)\b", normalized)
    if m:
        val = float(m.group(1))
        return val / 100000.0   # 56 rupees = 0.00056 lakh (correct)

    # 5. Compact 5–8 digit number ONLY if lakh context word present
    lakh_context = any(kw in normalized for kw in ("lakh", "lac", "crore", "threshold", "limit", "value band"))
    m = re.search(r"\b(\d{5,8})\b", normalized)
    if m and lakh_context:
        return float(m.group(1)) / 100000.0

    # 6. Shorthand: Nl / Ncr
    m = re.search(r"\b([\d]+(?:\.[\d]+)?)(l|cr)\b", normalized)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val * 100.0 if unit == "cr" else val

    return None


# Test cases
tests = [
    ("Rs. 8 lakh procurement", 8.0),
    ("Rs. 56 lakh case", 56.0),
    ("Rs 56 rupees worth", 56 / 100000),
    ("5 crore project", 500.0),
    ("Rs. 25000 purchase", 0.25),
    ("56 lakh tender", 56.0),
    ("2.5l value", 2.5),
    ("50l threshold", 50.0),
    ("1cr limit", 100.0),
    ("purchase of Rs. 1,50,000", 1.5),
]

print("Amount Extraction Tests")
print("-" * 50)
all_pass = True
for msg, expected in tests:
    result = extract_amount_lakhs(msg)
    status = "PASS" if result is not None and abs(result - expected) < 0.001 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{status} '{msg}' -> {result} (expected {expected})")

print()
print("ALL PASS" if all_pass else "SOME TESTS FAILED")
