"""Generate 3000 procurement evaluation questions from templates.

Expands the base 180 questions into 3000 by creating systematic variations
across amounts, items, phrasings, and scenarios.

Usage:
    python eval/generate_3000.py
    # Creates eval/all_questions_3000.json
"""

import json
import random
from pathlib import Path

OUTPUT = Path(__file__).parent / "all_questions_3000.json"
BASE_FILE = Path(__file__).parent / "all_questions.json"

# ── Amount Templates ────────────────────────────────────────────────────────
AMOUNTS = [
    ("Rs. 5,000", "DIRECT_PURCHASE"), ("Rs. 10,000", "DIRECT_PURCHASE"),
    ("Rs. 15,000", "DIRECT_PURCHASE"), ("Rs. 20,000", "DIRECT_PURCHASE"),
    ("Rs. 25,000", "DIRECT_PURCHASE"), ("Rs. 30,000", "DIRECT_PURCHASE"),
    ("Rs. 40,000", "DIRECT_PURCHASE"), ("Rs. 45,000", "DIRECT_PURCHASE"),
    ("Rs. 50,000", "DIRECT_PURCHASE"),
    ("Rs. 60,000", "LPC"), ("Rs. 75,000", "LPC"), ("Rs. 80,000", "LPC"),
    ("Rs. 90,000", "LPC"), ("Rs. 1 lakh", "LPC"), ("Rs. 1.2 lakhs", "LPC"),
    ("Rs. 1.5 lakhs", "LPC"), ("Rs. 2 lakhs", "LPC"), ("Rs. 2.5 lakhs", "LPC"),
    ("Rs. 3 lakhs", "LPC"), ("Rs. 3.5 lakhs", "LPC"), ("Rs. 4 lakhs", "LPC"),
    ("Rs. 4.5 lakhs", "LPC"), ("Rs. 5 lakhs", "LPC"),
    ("Rs. 5.5 lakhs", "LTE"), ("Rs. 6 lakhs", "LTE"), ("Rs. 7 lakhs", "LTE"),
    ("Rs. 8 lakhs", "LTE"), ("Rs. 10 lakhs", "LTE"), ("Rs. 12 lakhs", "LTE"),
    ("Rs. 15 lakhs", "LTE"), ("Rs. 18 lakhs", "LTE"), ("Rs. 20 lakhs", "LTE"),
    ("Rs. 22 lakhs", "LTE"), ("Rs. 25 lakhs", "LTE"), ("Rs. 26 lakhs", "LTE"),
    ("Rs. 30 lakhs", "LTE"), ("Rs. 35 lakhs", "LTE"), ("Rs. 40 lakhs", "LTE"),
    ("Rs. 45 lakhs", "LTE"), ("Rs. 50 lakhs", "LTE"),
    ("Rs. 60 lakhs", "OTE"), ("Rs. 75 lakhs", "OTE"), ("Rs. 1 crore", "OTE"),
    ("Rs. 1.5 crore", "OTE"), ("Rs. 2 crore", "OTE"), ("Rs. 2.5 crore", "OTE"),
    ("Rs. 3 crore", "OTE"), ("Rs. 5 crore", "OTE"),
]

SLAB_KEYWORDS = {
    "DIRECT_PURCHASE": ["direct purchase", "rule 154", "market rate"],
    "LPC": ["local purchase committee", "lpc", "rule 155", "quotation"],
    "LTE": ["limited tender", "lte", "rule 162"],
    "OTE": ["open tender", "ote", "rule 161"],
}

ITEMS = [
    "lab chemicals", "computer peripherals", "office furniture", "scientific instruments",
    "safety equipment", "printer cartridges", "networking equipment", "HVAC spare parts",
    "reagents", "glassware", "microscope accessories", "centrifuge tubes",
    "UPS batteries", "stationery supplies", "cleaning materials", "PPE kits",
    "electrical fittings", "plumbing materials", "paint supplies", "tools and hardware",
    "projector", "whiteboard", "air conditioner", "water purifier", "generator",
    "spectrophotometer", "oscilloscope", "fume hood", "autoclave", "incubator",
    "deep freezer", "analytical balance", "pH meter", "conductivity meter",
    "vacuum pump", "rotary evaporator", "thermal cycler", "gel electrophoresis unit",
    "server rack", "firewall appliance", "CCTV cameras", "biometric device",
]

QUESTION_TEMPLATES_THRESHOLD = [
    "What procurement method should I use for {item} worth {amount}?",
    "Which procurement route applies for {item} costing {amount}?",
    "What is the correct tender method for a {amount} purchase of {item}?",
    "How should I procure {item} if the estimated cost is {amount}?",
    "For {item} worth {amount}, which GFR rule governs the procurement?",
    "Can I use direct purchase for {item} worth {amount}?",
    "Is LPC applicable for {item} costing {amount}?",
    "Do I need open tender for {item} worth {amount}?",
    "What approvals are needed to buy {item} for {amount}?",
    "What is the step-by-step process to procure {item} worth {amount}?",
]

# ── Scenario Templates ──────────────────────────────────────────────────────
SCENARIO_TEMPLATES = [
    {"q": "A scientist urgently needs {item} worth {amount}. What is the fastest compliant route?",
     "kw": ["urgent", "procurement", "approval"], "cat": "SCENARIO"},
    {"q": "We received only 1 quotation for {item} worth {amount} under the chosen tender route. Is this valid?",
     "kw": ["quotation", "competition", "single offer", "committee"], "cat": "SCENARIO"},
    {"q": "The vendor claims {item} worth {amount} is proprietary. How to verify?",
     "kw": ["proprietary", "pac", "verification"], "cat": "SCENARIO"},
    {"q": "Our lab needs {item} worth {amount} but it is not available on GeM. What process?",
     "kw": ["gem", "not available", "procurement"], "cat": "SCENARIO"},
    {"q": "Can we split a {amount} purchase of {item} into smaller orders?",
     "kw": ["split", "not allowed", "procurement"], "cat": "SCENARIO"},
    {"q": "The L1 bidder for {item} worth {amount} did not meet technical specs. Can we go to L2?",
     "kw": ["l1", "l2", "technical", "committee"], "cat": "SCENARIO"},
    {"q": "We need {item} worth {amount} for a time-bound project. Can urgency justify a faster route?",
     "kw": ["urgency", "procurement", "approval", "justification"], "cat": "SCENARIO"},
    {"q": "A foreign manufacturer is the sole source for {item} worth {amount}. What route?",
     "kw": ["single tender", "proprietary", "foreign"], "cat": "SCENARIO"},
    {"q": "The previous year's rate contract for {item} has expired. Can we still use the old rates for a {amount} order?",
     "kw": ["rate contract", "validity", "procurement"], "cat": "SCENARIO"},
    {"q": "All bids for {item} worth {amount} exceed the estimate. What should we do?",
     "kw": ["exceed", "estimate", "re-tender", "committee"], "cat": "SCENARIO"},
]

# ── Analytical / Comparison Templates ───────────────────────────────────────
METHODS = ["Direct Purchase", "LPC", "LTE", "OTE", "STE"]
COMPARISON_ASPECTS = [
    "approval process", "documentation requirements", "timeline",
    "committee involvement", "GeM applicability", "audit risk",
    "threshold limits", "number of quotations required",
]

# ── General Knowledge Templates ─────────────────────────────────────────────
GENERAL_TOPICS = [
    ("EMD", ["earnest money", "deposit", "bid", "security"]),
    ("bid security", ["bid", "security", "guarantee", "tender"]),
    ("performance guarantee", ["performance", "guarantee", "bank", "contract"]),
    ("warranty clause", ["warranty", "defect", "replacement", "period"]),
    ("rate contract", ["rate", "contract", "schedule", "validity"]),
    ("purchase order", ["purchase", "order", "supply", "vendor"]),
    ("two-bid system", ["two", "bid", "technical", "financial"]),
    ("pre-bid conference", ["pre-bid", "conference", "clarification"]),
    ("vendor registration", ["vendor", "registration", "empanelment"]),
    ("debarment", ["debarment", "supplier", "blacklist", "penalty"]),
    ("comparative statement", ["comparative", "statement", "quotation"]),
    ("tender evaluation", ["tender", "evaluation", "committee", "bid"]),
    ("L1 bidder", ["l1", "lowest", "bidder", "evaluation"]),
    ("buyback", ["buyback", "old", "replacement", "procurement"]),
    ("advance payment", ["advance", "payment", "rule", "procurement"]),
    ("inspection", ["inspection", "quality", "goods", "receipt"]),
    ("acceptance of stores", ["acceptance", "stores", "inspection"]),
    ("penalty clause", ["penalty", "delay", "delivery", "contract"]),
    ("risk purchase", ["risk", "purchase", "default", "supplier"]),
    ("liquidated damages", ["liquidated", "damages", "delay", "contract"]),
]

GENERAL_TEMPLATES = [
    "What is {topic} in government procurement?",
    "Explain the concept of {topic} under GFR.",
    "What are the rules regarding {topic} in CSIR procurement?",
    "How does {topic} work in the procurement process?",
]

# ── Rule-specific Templates ─────────────────────────────────────────────────
RULES = [
    (144, ["procurement", "principle", "fundamental"]),
    (145, ["integrity", "transparency", "procurement"]),
    (149, ["gem", "mandatory", "government e-marketplace"]),
    (154, ["direct purchase", "market rate"]),
    (155, ["local purchase committee", "quotation"]),
    (161, ["open tender", "wide publicity", "portal"]),
    (162, ["limited tender", "firms"]),
    (166, ["single tender", "proprietary", "pac"]),
    (170, ["bid", "evaluation", "tender"]),
    (171, ["acceptance", "tender", "authority"]),
    (172, ["purchase", "preference", "domestic"]),
    (173, ["integrity", "pact", "procurement"]),
    (174, ["advance", "payment"]),
    (175, ["inspection", "quality"]),
    (176, ["payment", "supplier", "terms"]),
]

RULE_TEMPLATES = [
    "What does Rule {num} of GFR say?",
    "Explain GFR Rule {num}.",
    "What is the significance of Rule {num} in procurement?",
    "What are the key provisions of Rule {num}?",
    "How does Rule {num} apply to CSIR procurement?",
]


def generate_questions():
    questions = []
    qid = 1

    # Load base questions first
    with open(BASE_FILE, "r", encoding="utf-8") as f:
        base = json.load(f)
    for q in base:
        q["id"] = f"BASE-{qid:04d}"
        questions.append(q)
        qid += 1

    # ── Threshold + Amount variations ──
    for amount, slab in AMOUNTS:
        selected_items = random.sample(ITEMS, min(5, len(ITEMS)))
        for item in selected_items:
            tmpl = random.choice(QUESTION_TEMPLATES_THRESHOLD)
            questions.append({
                "id": f"GEN-TH-{qid:04d}",
                "category": "THRESHOLD",
                "question": tmpl.format(item=item, amount=amount),
                "expected_keywords": SLAB_KEYWORDS[slab],
            })
            qid += 1

    # ── Scenario variations ──
    for _ in range(1600):
        amount, slab = random.choice(AMOUNTS)
        item = random.choice(ITEMS)
        tmpl = random.choice(SCENARIO_TEMPLATES)
        kw = list(tmpl["kw"]) + SLAB_KEYWORDS[slab][:2]
        questions.append({
            "id": f"GEN-SC-{qid:04d}",
            "category": "SCENARIO",
            "question": tmpl["q"].format(item=item, amount=amount),
            "expected_keywords": kw,
        })
        qid += 1

    # ── Comparison / Analytical variations ──
    for i in range(len(METHODS)):
        for j in range(i + 1, len(METHODS)):
            m1, m2 = METHODS[i], METHODS[j]
            for aspect in COMPARISON_ASPECTS:
                questions.append({
                    "id": f"GEN-AN-{qid:04d}",
                    "category": "ANALYTICAL",
                    "question": f"Compare the {aspect} of {m1} and {m2}.",
                    "expected_keywords": [m1.lower(), m2.lower(), aspect.split()[0]],
                })
                qid += 1

    # ── General knowledge variations ──
    for topic, kw in GENERAL_TOPICS:
        for tmpl in GENERAL_TEMPLATES:
            questions.append({
                "id": f"GEN-GN-{qid:04d}",
                "category": "GENERAL",
                "question": tmpl.format(topic=topic),
                "expected_keywords": kw,
            })
            qid += 1

    # ── Rule-specific variations ──
    for rule_num, kw in RULES:
        for tmpl in RULE_TEMPLATES:
            questions.append({
                "id": f"GEN-RL-{qid:04d}",
                "category": "RULE",
                "question": tmpl.format(num=rule_num),
                "expected_keywords": kw + [f"rule {rule_num}"],
            })
            qid += 1

    # ── GeM-specific variations ──
    gem_questions = [
        ("Is GeM mandatory for {item}?", ["gem", "mandatory", "rule 149"]),
        ("Can I skip GeM for purchasing {item}?", ["gem", "exemption", "not available"]),
        ("How to procure {item} through GeM?", ["gem", "portal", "procurement"]),
        ("{item} is not available on GeM. What should I do?", ["gem", "not available", "certificate"]),
        ("The GeM price for {item} is higher than local market. Can I buy locally?", ["gem", "mandatory", "price"]),
    ]
    for item in ITEMS:
        for tmpl_q, kw in gem_questions:
            questions.append({
                "id": f"GEN-GM-{qid:04d}",
                "category": "GEM",
                "question": tmpl_q.format(item=item),
                "expected_keywords": kw,
            })
            qid += 1

    # ── PAC variations ──
    pac_items = [
        "specialized microscope", "gas chromatograph", "NMR spectrometer",
        "mass spectrometer", "electron microscope", "DNA sequencer",
        "proprietary software license", "OEM spare parts", "patented reagent",
        "calibration service from OEM", "licensed analytical method kit",
    ]
    pac_templates = [
        ("We need {item} from the sole manufacturer. Is PAC required?",
         ["pac", "proprietary", "single tender", "sole"]),
        ("Can we use single tender for {item} without PAC?",
         ["single tender", "pac", "justification"]),
        ("Who can issue PAC for purchasing {item}?",
         ["pac", "authority", "competent", "technical"]),
        ("What is the audit risk of buying {item} via PAC?",
         ["pac", "audit", "risk", "justification"]),
    ]
    for item in pac_items:
        for tmpl_q, kw in pac_templates:
            questions.append({
                "id": f"GEN-PA-{qid:04d}",
                "category": "PAC",
                "question": tmpl_q.format(item=item),
                "expected_keywords": kw,
            })
            qid += 1

    # ── Committee variations ──
    comm_questions = [
        "Who should be part of the purchase committee for {item} worth {amount}?",
        "Is finance concurrence needed for {item} worth {amount}?",
        "What is the role of the indenting officer when procuring {item} worth {amount}?",
        "How many quotations are needed for {item} worth {amount}?",
    ]
    for _ in range(150):
        amount, slab = random.choice(AMOUNTS)
        item = random.choice(ITEMS)
        q = random.choice(comm_questions).format(item=item, amount=amount)
        questions.append({
            "id": f"GEN-CM-{qid:04d}",
            "category": "COMMITTEE",
            "question": q,
            "expected_keywords": ["committee", "procurement", "approval"],
        })
        qid += 1

    # ── Make in India variations ──
    mi_templates = [
        "Does Make in India preference apply for {item} worth {amount}?",
        "Are only local suppliers eligible to bid for {item} worth {amount}?",
        "What is the local content requirement for {item} procurement?",
        "If L1 for {item} is not a local supplier, can we give preference to local L2?",
    ]
    for _ in range(120):
        amount, slab = random.choice(AMOUNTS)
        item = random.choice(ITEMS)
        q = random.choice(mi_templates).format(item=item, amount=amount)
        questions.append({
            "id": f"GEN-MI-{qid:04d}",
            "category": "MAKE_IN_INDIA",
            "question": q,
            "expected_keywords": ["local", "supplier", "make in india"],
        })
        qid += 1

    # ── Edge case fillers ──
    edge_templates = [
        "Can I combine purchases of {item1} and {item2} into a single tender?",
        "What if the delivery of {item} is delayed beyond the contract period?",
        "The vendor for {item} has been debarred. Can we still use the pending order?",
        "Can we procure {item} from an unregistered vendor?",
        "Is it mandatory to have a technical specification committee for {item}?",
        "What happens if the LPC cannot reach a consensus on {item} procurement?",
        "Can we extend the bid validity period for {item} procurement?",
        "Is e-tendering mandatory for {item} above Rs. 50 lakhs?",
    ]
    for _ in range(200):
        item = random.choice(ITEMS)
        item2 = random.choice(ITEMS)
        q = random.choice(edge_templates).format(item=item, item1=item, item2=item2)
        questions.append({
            "id": f"GEN-EG-{qid:04d}",
            "category": "EDGE_CASE",
            "question": q,
            "expected_keywords": ["procurement", "rule", "approval"],
        })
        qid += 1

    # Top up if the template expansions undershoot the target size.
    while len(questions) < 3000:
        amount, slab = random.choice(AMOUNTS)
        item = random.choice(ITEMS)
        tmpl = random.choice(QUESTION_TEMPLATES_THRESHOLD)
        questions.append({
            "id": f"GEN-FL-{qid:04d}",
            "category": "THRESHOLD",
            "question": tmpl.format(item=item, amount=amount),
            "expected_keywords": SLAB_KEYWORDS[slab],
        })
        qid += 1

    # Shuffle and trim to exactly 3000
    random.seed(42)
    random.shuffle(questions)
    questions = questions[:3000]

    # Re-number
    for i, q in enumerate(questions, 1):
        q["id"] = f"Q-{i:04d}"

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(questions)} questions -> {OUTPUT}")

    # Category breakdown
    from collections import Counter
    cats = Counter(q["category"] for q in questions)
    for cat, count in cats.most_common():
        print(f"  {cat:20s} : {count}")


if __name__ == "__main__":
    generate_questions()
