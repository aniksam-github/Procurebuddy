"""Generate a balanced 10,000-question procurement evaluation dataset.

The output is written as:
- testing/generated/procurement_eval_dataset_10000.json
- testing/generated/procurement_eval_dataset_10000_batch_01.json ... _10.json
- testing/generated/procurement_eval_dataset_10000_summary.json

The generator is deterministic and enforces:
- 10 batches x 1,000 questions
- strict category balance per batch
- strict difficulty balance per batch
- exact tricky / rule-conflict / decision-making counts per batch
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 20260430
BATCHES = 10
QUESTIONS_PER_BATCH = 1000

OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
COMBINED_FILE = OUTPUT_DIR / "procurement_eval_dataset_10000.json"
SUMMARY_FILE = OUTPUT_DIR / "procurement_eval_dataset_10000_summary.json"

CATEGORY_COUNTS = {
    "scenario": 300,
    "edge_case": 150,
    "analytical": 150,
    "rule": 100,
    "process": 80,
    "threshold": 70,
    "pac": 70,
    "gem": 40,
    "make_in_india": 20,
    "others": 20,
}

CATEGORY_DIFFICULTY_COUNTS = {
    "scenario": {"easy": 70, "medium": 170, "hard": 60},
    "edge_case": {"easy": 0, "medium": 80, "hard": 70},
    "analytical": {"easy": 40, "medium": 90, "hard": 20},
    "rule": {"easy": 55, "medium": 40, "hard": 5},
    "process": {"easy": 20, "medium": 50, "hard": 10},
    "threshold": {"easy": 55, "medium": 10, "hard": 5},
    "pac": {"easy": 20, "medium": 30, "hard": 20},
    "gem": {"easy": 15, "medium": 20, "hard": 5},
    "make_in_india": {"easy": 5, "medium": 10, "hard": 5},
    "others": {"easy": 20, "medium": 0, "hard": 0},
}

CATEGORY_FLAG_COUNTS = {
    "scenario": {"tricky": 120, "conflict": 70, "decision": 90},
    "edge_case": {"tricky": 70, "conflict": 40, "decision": 20},
    "analytical": {"tricky": 20, "conflict": 30, "decision": 15},
    "rule": {"tricky": 20, "conflict": 25, "decision": 5},
    "process": {"tricky": 10, "conflict": 5, "decision": 25},
    "threshold": {"tricky": 25, "conflict": 10, "decision": 10},
    "pac": {"tricky": 15, "conflict": 10, "decision": 15},
    "gem": {"tricky": 8, "conflict": 5, "decision": 10},
    "make_in_india": {"tricky": 7, "conflict": 3, "decision": 8},
    "others": {"tricky": 5, "conflict": 2, "decision": 2},
}

RULES = {
    "gem": "Rule 149",
    "direct": "Rule 154",
    "lpc": "Rule 155",
    "ote": "Rule 161",
    "lte": "Rule 162",
    "pac": "Rule 166",
}

ITEMS = [
    "LC-MS solvent bottles",
    "PCR reagents",
    "analytical balance",
    "autoclave gasket set",
    "biological safety cabinet",
    "biometric attendance device",
    "calibration gases",
    "centrifuge rotor",
    "cleanroom gloves",
    "conductivity meter",
    "cryovials",
    "CCTV storage server",
    "deep freezer compressor",
    "DNA extraction kit",
    "electrical safety tester",
    "electrophoresis power pack",
    "ELISA reader",
    "fume hood blower motor",
    "gas chromatograph column",
    "glass reaction vessel",
    "HPLC pump seal kit",
    "HVAC control panel",
    "ICP-MS torch assembly",
    "incubator controller",
    "industrial RO membrane",
    "inverter batteries",
    "laboratory chairs",
    "laboratory dishwasher",
    "laser particle counter",
    "liquid nitrogen hose",
    "mass spectrometer software license",
    "microscope camera",
    "microscope immersion oil",
    "modular UPS",
    "network firewall appliance",
    "nitrogen generator service kit",
    "office workstation",
    "oscilloscope probes",
    "oxygen sensors",
    "pH meter electrode",
    "pipeline leak detector",
    "PLC module",
    "portable gas detector",
    "precision weighing pans",
    "printer cartridges",
    "protein assay kit",
    "qPCR plates",
    "refrigerated centrifuge",
    "RFID access controller",
    "RNA stabilization reagent",
    "rotary evaporator chiller",
    "safety shower valve set",
    "sample storage racks",
    "seismic data logger",
    "server rack",
    "spectrophotometer lamp",
    "stainless steel worktable",
    "temperature data logger",
    "thermal cycler",
    "tissue culture flasks",
    "ultra-low freezer",
    "UPS bypass switch",
    "vacuum pump",
    "vibration meter",
    "water purification cartridges",
    "weatherproof field laptop",
    "XRD sample holder",
]

ROLES = [
    "stores officer",
    "indenting scientist",
    "purchase committee convener",
    "internal finance officer",
    "section head",
    "laboratory director",
    "technical evaluation member",
    "pre-audit officer",
    "project coordinator",
    "materials manager",
]

CONTEXTS = [
    "for a time-bound field trial",
    "for a grant closing this quarter",
    "for a safety-critical replacement",
    "for a NABL-linked calibration cycle",
    "for a student training programme starting next week",
    "for a pilot plant shutdown window",
    "for a sponsored project milestone",
    "for a cleanroom restart",
    "for a monsoon preparedness drive",
    "for a plant maintenance outage",
]

VENDORS = [
    "the OEM",
    "a sole authorized dealer",
    "the previous year's supplier",
    "a GeM-listed reseller",
    "an unregistered vendor",
    "a foreign manufacturer",
    "a local integrator",
    "a Class I local supplier",
    "a Class II local supplier",
    "the incumbent AMC vendor",
]

TRICKY_CLAUSES = [
    "The file note treats separate invoices as proof that the demand is separate.",
    "Two quotations came from sister concerns using nearly identical commercial terms.",
    "The user section says the item is unique, but gives no technical comparison note.",
    "The proposal relies on last year's practice rather than the current routing logic.",
    "The price looks attractive, so the team wants to overlook a control weakness.",
    "The note assumes that urgency alone cures gaps in competition.",
]

CONFLICT_CLAUSES = [
    "A stale manual extract in the file still mentions the older Rs. 2.5 lakh threshold.",
    "The user wing cites speed, while finance cites the normal competition route.",
    "GeM shows similar listings, but the scientist says the required configuration is not visible.",
    "The technical team treats brand continuity as proprietary, while stores treats it as a competition issue.",
    "The draft note cites PAC logic even though the facts look closer to standardization than exclusivity.",
    "The bid file mixes route-selection logic with approval-power logic as if they were the same test.",
]

DECISION_ENDINGS = [
    "Which route should control?",
    "Would you clear the note as drafted, or send it back for correction?",
    "What is the safest defensible decision on file?",
    "Should the case move forward, be re-tendered, or be re-framed first?",
    "Is approval safer than modification here, or not?",
]

INSUFFICIENT_INFO_ENDINGS = [
    "Is there enough information to approve any route at all?",
    "Can the buyer safely choose a method yet, or is the file incomplete?",
]

PROCESS_ENDINGS = [
    "What sequence of actions should the officer follow?",
    "What is the correct workflow from this point?",
    "How should the file move, step by step?",
]

RULE_STYLE_OPENERS = [
    "Under",
    "When applying",
    "In practical file work under",
    "For an approval note relying on",
]

ANALYTICAL_OPENERS = [
    "Why does",
    "How does",
    "From an audit perspective, why does",
    "What makes",
]


def format_rs(value: int) -> str:
    return f"Rs. {value:,}"


def route_for_amount(value: int) -> tuple[str, str]:
    if value <= 50_000:
        return "direct purchase", RULES["direct"]
    if value <= 500_000:
        return "LPC", RULES["lpc"]
    if value <= 5_000_000:
        return "LTE", RULES["lte"]
    return "OTE", RULES["ote"]


def route_label_for_value(value: int) -> str:
    route, rule = route_for_amount(value)
    return f"{route} under {rule}"


def choose(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def normalized_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def line_answer(line1: str, line2: str, line3: str | None = None) -> str:
    parts = [line1.strip(), line2.strip()]
    if line3:
        parts.append(line3.strip())
    return "\n".join(parts)


def procurement_threshold_table() -> str:
    rows = [
        (
            f"Up to {format_rs(50_000)}",
            "direct purchase",
            "No",
            "-",
            RULES["direct"],
        ),
        (
            f"Above {format_rs(50_000)} to {format_rs(500_000)}",
            "LPC",
            "Yes",
            "Local Purchase Committee (LPC)",
            RULES["lpc"],
        ),
        (
            f"Above {format_rs(500_000)} to {format_rs(5_000_000)}",
            "LTE",
            "Yes",
            "Technical & Purchase Committee (T&PC)",
            RULES["lte"],
        ),
        (
            f"Above {format_rs(5_000_000)}",
            "OTE",
            "Yes",
            "Technical & Purchase Committee (T&PC)",
            RULES["ote"],
        ),
    ]
    table = (
        "| Cost Category | Procurement Mode | Committee Required | Which Committee | Rule |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    for row in rows:
        table += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
    return table.rstrip()


def table_answer(summary_line: str, detail_line: str | None = None) -> str:
    parts = [summary_line.strip()]
    if detail_line:
        parts.append(detail_line.strip())
    parts.append(procurement_threshold_table())
    return "\n".join(parts)


def make_key_points(*points: str) -> list[str]:
    return [point for point in points if point]


def make_schedule(category: str, rng: random.Random) -> list[dict[str, object]]:
    count = CATEGORY_COUNTS[category]
    diffs = []
    for difficulty, qty in CATEGORY_DIFFICULTY_COUNTS[category].items():
        diffs.extend([difficulty] * qty)
    rng.shuffle(diffs)

    tricky = [True] * CATEGORY_FLAG_COUNTS[category]["tricky"] + [False] * (
        count - CATEGORY_FLAG_COUNTS[category]["tricky"]
    )
    conflict = [True] * CATEGORY_FLAG_COUNTS[category]["conflict"] + [False] * (
        count - CATEGORY_FLAG_COUNTS[category]["conflict"]
    )
    decision = [True] * CATEGORY_FLAG_COUNTS[category]["decision"] + [False] * (
        count - CATEGORY_FLAG_COUNTS[category]["decision"]
    )
    rng.shuffle(tricky)
    rng.shuffle(conflict)
    rng.shuffle(decision)

    return [
        {
            "difficulty": diffs[i],
            "tricky": tricky[i],
            "conflict": conflict[i],
            "decision": decision[i],
        }
        for i in range(count)
    ]


def amount_pool(category: str, difficulty: str) -> list[int]:
    base = {
        "easy": [
            28_000, 39_500, 50_000, 50_001, 88_000, 125_000, 275_000, 500_000,
            500_001, 820_000, 1_450_000, 2_500_000, 4_800_000, 5_000_000,
            5_000_001, 6_200_000,
        ],
        "medium": [
            47_800, 49_900, 52_000, 198_000, 310_000, 499_500, 505_000, 760_000,
            1_180_000, 1_980_000, 2_490_000, 2_610_000, 3_750_000, 4_950_000,
            5_050_000, 6_200_000,
        ],
        "hard": [
            50_000, 50_001, 499_999, 500_000, 500_001, 4_999_999, 5_000_000,
            5_000_001, 5_250_000, 7_800_000,
        ],
    }
    values = list(base[difficulty])
    if category in {"scenario", "edge_case", "pac", "gem", "make_in_india"}:
        values.extend([640_000, 915_000, 1_560_000, 3_250_000])
    return values


def maybe_add_clause(question: str, flags: dict[str, object], rng: random.Random, category: str) -> str:
    pieces: list[str] = [question.rstrip()]
    if bool(flags["tricky"]):
        pieces.append(choose(rng, TRICKY_CLAUSES))
    if bool(flags["conflict"]):
        pieces.append(choose(rng, CONFLICT_CLAUSES))
    if bool(flags["decision"]):
        if category == "process":
            pieces.append(choose(rng, PROCESS_ENDINGS))
        else:
            pieces.append(choose(rng, DECISION_ENDINGS))
    return " ".join(pieces)


def scenario_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    role = choose(rng, ROLES)
    context = choose(rng, CONTEXTS)
    total = choose(rng, amount_pool("scenario", difficulty))
    route, rule = route_for_amount(total)
    scenario_type = rng.randrange(10)

    if scenario_type == 0:
        split_count = rng.choice([2, 3])
        smaller = max(total // split_count, 20_000)
        question = (
            f"A {role} proposes to break a {format_rs(total)} requirement for {item} into "
            f"{split_count} indents of about {format_rs(smaller)} each {context} so the case stays below a higher route."
            " What compliance issue arises?"
        )
        answer = line_answer(
            "This points to artificial splitting, which is not a valid way to move a common requirement into a lower method.",
            f"The combined need should control the route, so the case should be tested against {route} logic under {rule} rather than the smaller indents.",
        )
        points = make_key_points(
            "Check whether the demand is one requirement in timing, purpose, and funding",
            "Apply the route on aggregate value, not invoice value",
            "Record splitting risk as a high audit concern",
            "Recast the case under the correct method if needed",
        )
    elif scenario_type == 1:
        question = (
            f"The {role} wants to buy {item} worth {format_rs(total)} from a local vendor because the same item looks cheaper outside GeM."
            " GeM shows a similar listing."
            " Can the buyer leave GeM on price alone?"
        )
        answer = line_answer(
            "No. Price alone is usually not enough to bypass GeM when the item is available there.",
            f"The file should first examine GeM applicability under {RULES['gem']}; only a real specification gap, non-availability, or properly recorded exception can justify moving to {route}.",
        )
        points = make_key_points(
            "Check GeM availability first",
            "Treat lower offline price as insufficient by itself",
            "Record any specification mismatch or non-availability evidence",
            f"After a valid GeM departure, use the normal value route under {rule}",
        )
    elif scenario_type == 2:
        question = (
            f"An {item} case estimated at {format_rs(total)} was opened through {route}, but only one valid bid was received."
            " The user division wants immediate placement because the work is time-bound."
            " What is the defensible next step?"
        )
        answer = line_answer(
            "A single valid offer does not automatically make the award unsafe, but it does require recorded scrutiny of competition and reasonableness.",
            f"The committee should review publicity, specification restrictiveness, urgency, and value for money before deciding whether to re-tender or proceed with competent approval under {rule}.",
        )
        points = make_key_points(
            "Test whether competition was adequate",
            "Examine if specifications were too restrictive",
            "Record price reasonableness and urgency if proceeding",
            "Take competent authority approval on the single-offer decision",
        )
    elif scenario_type == 3:
        question = (
            f"For {item} valued at {format_rs(total)}, the L1 bidder is cheapest but failed a key technical parameter during evaluation."
            " The committee is considering L2."
            " Is that path supportable?"
        )
        answer = line_answer(
            "Yes, but only if L1 is rejected for a clearly recorded technical reason and L2 fully meets the specifications.",
            "The file should show a documented non-responsiveness finding, comparison on the surviving bidders, and fresh approval before award.",
        )
        points = make_key_points(
            "Reject L1 only on recorded, defensible technical grounds",
            "Confirm L2 is technically compliant",
            "Document comparative evaluation after rejection",
            "Obtain approval on the recommended bidder",
        )
    elif scenario_type == 4:
        base = max(total - rng.choice([35_000, 55_000, 140_000]), 25_000)
        full = base + rng.choice([40_000, 70_000, 180_000])
        full_route, full_rule = route_for_amount(full)
        question = (
            f"The base price for {item} was noted as {format_rs(base)}, but mandatory freight, installation, and AMC bring the realistic total to {format_rs(full)}."
            " The draft note still wants to use the lower route chosen on base price."
            " Which valuation should control?"
        )
        answer = line_answer(
            "The controlling value should be the realistic procurement value, not the stripped base price.",
            f"If mandatory charges move the case upward, the route should be re-tested on {format_rs(full)} and may need to shift to {full_route} under {full_rule}.",
        )
        points = make_key_points(
            "Aggregate mandatory cost components",
            "Do not choose route on an artificially reduced base price",
            "Revise the estimate if the total changes the slab",
            "Re-open the route if the controlling band changes",
        )
    elif scenario_type == 5:
        question = (
            f"The user section says {item} must match an existing system and wants to treat the {format_rs(total)} purchase as proprietary from {choose(rng, VENDORS)}."
            " What should be checked before accepting that logic?"
        )
        answer = line_answer(
            "Brand continuity alone is not automatically the same as proprietary exclusivity.",
            f"The file should separate standardization from true sole-source justification, record why alternatives are impractical, and use PAC or another valid single-source basis only if the facts meet {RULES['pac']}.",
        )
        points = make_key_points(
            "Distinguish standardization from pure proprietary claim",
            "Verify whether alternatives can meet compatibility needs",
            "Require technical justification from competent authority",
            "Use single-source logic only when facts actually support it",
        )
    elif scenario_type == 6:
        question = (
            f"A {role} wants to use last year's rate for {item} in a fresh {format_rs(total)} order because the earlier vendor performed well."
            " The old rate contract has just expired."
            " Can the file rely on the expired terms?"
        )
        answer = line_answer(
            "No. Once the rate contract has expired, it cannot be treated as a fresh procurement basis merely for convenience.",
            f"The buyer should start a new process, test GeM first, and then follow the current route for {format_rs(total)} under {rule}.",
        )
        points = make_key_points(
            "Check whether the old contract is still valid",
            "Do not use expired rates as a substitute for fresh competition",
            "Re-test GeM applicability",
            f"Restart procurement under {rule} if needed",
        )
    elif scenario_type == 7:
        question = (
            f"The GeM portal is down while {item} worth {format_rs(total)} is urgently needed {context}."
            " The section wants to buy offline immediately."
            " What documentation makes that choice defensible?"
        )
        answer = line_answer(
            "The case needs evidence, not just a statement that the portal was unavailable.",
            "Screenshots, timestamps, urgency justification, and a note on why waiting was impractical should appear on file before any alternative route is taken.",
        )
        points = make_key_points(
            "Capture GeM downtime evidence",
            "Record urgency and why delay was unacceptable",
            "Explain why the alternative route was necessary",
            "Apply the normal value route once the GeM exception is justified",
        )
    elif scenario_type == 8:
        bid = total + rng.choice([40_000, 95_000, 250_000])
        question = (
            f"All compliant bids for {item} are above the estimate; the best offer is {format_rs(bid)} against a sanctioned estimate of {format_rs(total)}."
            " What should the committee do before award?"
        )
        answer = line_answer(
            "The committee should not jump straight to award on an overshoot without re-examining reasonableness and approval.",
            "It should record market assessment, consider re-tendering if prices look abnormal, and obtain revised sanction if the higher value is still justified.",
        )
        points = make_key_points(
            "Compare the quoted rate with current market conditions",
            "Consider whether re-tendering is warranted",
            "Seek revised estimate or sanction if proceeding",
            "Record competent approval on the revised financial position",
        )
    else:
        question = (
            f"A {role} initiated {item} procurement at {format_rs(total)} {context}, but the urgency exists largely because the old AMC was not renewed in time."
            " Can emergency logic still be used without qualification?"
        )
        answer = line_answer(
            "Not cleanly. A preventable planning lapse weakens the case for treating the matter as a pure emergency.",
            "The file may still need expedited handling, but it should acknowledge the planning failure and avoid using urgency as a blanket excuse to ignore normal controls.",
        )
        points = make_key_points(
            "Distinguish genuine emergency from avoidable delay",
            "Do not let urgency automatically displace normal controls",
            "Record the planning lapse and corrective action",
            "Use only the minimum justified departure from the regular method",
        )

    return maybe_add_clause(question, flags, rng, "scenario"), answer, points


def edge_case_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    role = choose(rng, ROLES)
    edge_type = rng.randrange(10)

    if edge_type == 0:
        amount = rng.choice([50_000, 50_001, 500_000, 500_001, 5_000_000, 5_000_001])
        route, rule = route_for_amount(amount)
        question = (
            f"An approval note is prepared for {item} at exactly {format_rs(amount)}."
            " The section is unsure whether the file should stay in the lower band or move up immediately at that figure."
            f" Which side of the threshold controls for {format_rs(amount)}?"
        )
        answer = line_answer(
            f"The exact figure should be read against the slab boundary itself, so the route is {route} under {rule}.",
            "The safer approach is to decide on the estimated value at initiation and record the boundary logic expressly when the amount sits right on the cut-off.",
        )
        points = make_key_points(
            "Apply the exact slab boundary, not a rough approximation",
            "Use estimated value at the time of route selection",
            "Record why the chosen side of the threshold applies",
            "Revisit the route if the estimate changes materially",
        )
    elif edge_type == 1:
        question = (
            f"A PAC for {item} is signed by a retired scientist who designed the original system."
            " The user division says no serving officer knows the technology as well."
            " Can the certificate still support single-source procurement?"
        )
        answer = line_answer(
            "No. Technical familiarity alone does not cure the authority problem.",
            "A PAC should come from a competent serving technical authority; otherwise the single-source justification remains vulnerable even if the retired expert is knowledgeable.",
        )
        points = make_key_points(
            "Check the signatory's present authority, not only expertise",
            "Treat a retired signatory as non-compliant for PAC issuance",
            "Obtain certification from a competent serving technical officer",
            "Reassess whether single-source logic is still supportable",
        )
    elif edge_type == 2:
        question = (
            f"The file says 'GeM not feasible' for {item}, but no screenshots, search record, or specification-gap note is attached."
            " Can the departure survive audit on that wording alone?"
        )
        answer = line_answer(
            "Usually no. A bare conclusion is weaker than the evidence needed to justify leaving GeM.",
            "The file should show how availability, specification mismatch, or urgency was actually checked before moving to an offline route.",
        )
        points = make_key_points(
            "Do not rely on a conclusory GeM-not-feasible note",
            "Preserve search evidence or specification mismatch proof",
            "Link the GeM departure to the actual facts",
            "Apply the normal value route only after the GeM exception is recorded",
        )
    elif edge_type == 3:
        question = (
            f"A laboratory received only one valid bid in two successive attempts for {item} and now wants to treat the third attempt as automatic STE."
            " Is that a safe legal shortcut?"
        )
        answer = line_answer(
            "No. Repeated low response does not by itself transform the case into an automatic proprietary purchase.",
            "The buyer still has to show why competition failed, whether specifications were restrictive, and whether another justified route exists before using single-source logic.",
        )
        points = make_key_points(
            "Do not equate repeated low response with proprietary status",
            "Revisit specifications and market reach",
            "Record why re-tendering is or is not feasible",
            "Use single-source logic only on an independent valid basis",
        )
    elif edge_type == 4:
        question = (
            f"The OEM gives a PAC for {item}, but the actual requirement is a consumable where aftermarket compatibility is disputed."
            " How should that ambiguity be resolved?"
        )
        answer = line_answer(
            "The buyer should not assume that a PAC for the system automatically settles the consumable question.",
            "It should examine whether compatible alternatives genuinely fail technical needs; if they may work, competition should be tested rather than treating the consumable as automatically proprietary.",
        )
        points = make_key_points(
            "Separate the proprietary status of the system from that of the consumable",
            "Test compatibility claims with technical evidence",
            "Do not extend PAC logic beyond what the facts support",
            "Compete the item if acceptable alternatives exist",
        )
    elif edge_type == 5:
        question = (
            f"A file proposes direct purchase of {item} at {format_rs(rng.choice([18_000, 24_500, 42_000]))} because the value is low, but the item is clearly available on GeM."
            " Does the low amount settle the issue by itself?"
        )
        answer = line_answer(
            "No. Low value affects the route, but it does not erase the need to consider GeM availability first.",
            f"The file should test GeM under {RULES['gem']} and only then rely on direct-purchase logic if a valid departure or low-value justification still remains supportable.",
        )
        points = make_key_points(
            "Route threshold does not outrank GeM availability",
            "Check GeM first even in low-value cases",
            "Record why the offline choice remains justified",
            "Keep market-rate reasonableness on file",
        )
    elif edge_type == 6:
        question = (
            f"The section cites the old Rs. 2.5 lakh slab from a stale manual extract while stores applies the updated Rs. 5 lakh boundary for {item}."
            " Which threshold should prevail?"
        )
        answer = line_answer(
            "The updated GFR 2025 threshold logic should control over stale lower limits.",
            "If the file contains conflicting threshold references, the routing note should expressly correct the outdated text and apply the current slab table.",
        )
        points = make_key_points(
            "Resolve stale and current threshold conflict explicitly",
            "Apply the updated Rs. 5 lakh LPC/LTE boundary",
            "Correct the approval note instead of silently ignoring the conflict",
            "Avoid routing decisions on legacy limits",
        )
    elif edge_type == 7:
        question = (
            f"The same officer is both indenting authority and approving authority for {item} because the lab says staffing is thin."
            " Is that automatically fatal to the case?"
        )
        answer = line_answer(
            "Not automatically fatal, but it is a control weakness that should not be hidden.",
            "The safer course is to escalate awareness to the next higher authority, document the staffing constraint, and preserve as much separation of scrutiny as the organisation can realistically maintain.",
        )
        points = make_key_points(
            "Treat separation-of-duties failure as a real audit issue",
            "Record why the overlap was unavoidable",
            "Inform or involve a higher authority where possible",
            "Keep scrutiny and recommendation independent if approval cannot be",
        )
    elif edge_type == 8:
        question = (
            f"A bidder claims Class I local supplier status for {item}, but the self-certification gives only a percentage and no basis of calculation."
            " Can evaluation safely apply purchase preference on that record?"
        )
        answer = line_answer(
            "Not safely. Preference should rest on a supportable local-content declaration, not a bare label.",
            "The committee should seek clarification or verification under the applicable Make in India framework before giving ranking or split-order benefits.",
        )
        points = make_key_points(
            "Do not accept local-content status blindly",
            "Seek calculation basis or verification evidence",
            "Delay preference until eligibility is supportable",
            "Record how the committee resolved the claim",
        )
    else:
        question = (
            f"A cross-year procurement for {item} was approved in March, but delivery, inspection, and booking will fall in April after the original budget head has closed."
            " Is there any automatically valid option left, or must the file be regularized first?"
        )
        answer = line_answer(
            "The file should not assume automatic continuity across the financial year boundary.",
            "Budget provision, approval validity, and booking year have to be checked afresh; if those controls are missing, there may be no clean route without regularization or renewed sanction.",
        )
        points = make_key_points(
            "Check budget provision in the year of booking",
            "Review approval validity across year-end",
            "Do not assume March approval solves April expenditure",
            "Regularize or renew sanction where required",
        )

    if bool(flags["decision"]) and rng.random() < 0.35:
        question = f"{question} {choose(rng, INSUFFICIENT_INFO_ENDINGS)}"
    return maybe_add_clause(question, flags, rng, "edge_case"), answer, points


def analytical_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    opener = choose(rng, ANALYTICAL_OPENERS)
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("analytical", difficulty))
    role = choose(rng, ROLES)
    context = choose(rng, CONTEXTS)
    analytical_type = rng.randrange(10)

    if analytical_type == 0:
        question = (
            f"{opener} the aggregate value matter more than individual line items when several small "
            f"{item}-related purchases serve one project objective {context}?"
        )
        answer = line_answer(
            "Because procurement control targets the real requirement, not the arithmetic of how it is split on paper.",
            "If common timing, funding, and purpose show one requirement, the aggregate better reflects the competition route and audit risk.",
        )
        points = make_key_points(
            "Aggregate value tests the real requirement",
            "Line-item treatment can hide route circumvention",
            "Purpose, timing, and funding help decide clubbing",
            "Audit looks at substance over invoice structure",
        )
    elif analytical_type == 1:
        question = (
            f"Why is a PAC stronger than a vendor's claim of being the only convenient source for {item}, "
            "and why does that distinction matter in procurement reasoning?"
        )
        answer = line_answer(
            "A PAC is meant to establish justified exclusivity through competent technical certification, not mere commercial convenience.",
            "That distinction matters because single-source procurement should rest on a defensible technical basis, not on habit, loyalty, or ease of purchase.",
        )
        points = make_key_points(
            "PAC is about defensible exclusivity",
            "Convenience is weaker than proprietary necessity",
            "Technical authority matters",
            "Single-source risk rises if the file blurs this distinction",
        )
    elif analytical_type == 2:
        question = (
            f"{opener} checking GeM logically come before debating whether {item} at about {format_rs(amount)} "
            "falls in direct purchase, LPC, LTE, or OTE?"
        )
        answer = line_answer(
            "Because the route-by-value question only becomes meaningful after the sourcing constraint is checked.",
            f"If the item is available on GeM, {RULES['gem']} can control the sourcing path first; only then does the threshold route determine the compliant offline or downstream method.",
        )
        points = make_key_points(
            "GeM is a sourcing gate, not just another route detail",
            "Value slabs apply after GeM logic is tested",
            "Do not treat route and source platform as separate worlds",
            "Record why GeM does or does not control",
        )
    elif analytical_type == 3:
        question = (
            f"Why should urgency caused by internal delay carry less weight than urgency caused by an external "
            f"breakdown or safety risk in a {item} procurement?"
        )
        answer = line_answer(
            "Because self-created urgency is weaker as a justification for relaxing competition or documentation.",
            "The file still has to solve the present need, but audit will ask whether the departure was unavoidable or created by poor planning.",
        )
        points = make_key_points(
            "Separate genuine urgency from avoidable delay",
            "Internal planning failure weakens exception logic",
            "Expediency does not erase control obligations",
            "Corrective planning should be recorded",
        )
    elif analytical_type == 4:
        question = (
            f"{opener} the audit risk change when a {role}'s note on {item} says 'L1 is lowest' but says almost "
            "nothing about responsiveness or technical acceptability?"
        )
        answer = line_answer(
            "The risk rises because price alone does not establish that the award is legally or technically sound.",
            "A thin note suggests the evaluation may have skipped the question of whether the bidder actually met the tender conditions.",
        )
        points = make_key_points(
            "Lowest price is not enough by itself",
            "Responsiveness and qualification still matter",
            "Thin evaluation notes create challenge risk",
            "The award rationale should show more than price ranking",
        )
    elif analytical_type == 5:
        question = (
            f"{opener} separation of duties matter in a {format_rs(amount)} procurement file for {item} even "
            "when everyone involved appears honest and technically competent?"
        )
        answer = line_answer(
            "Because procurement controls are designed to survive personnel changes, pressure, and hindsight, not just to reflect present trust.",
            "Independent scrutiny reduces the chance that one person's preference silently becomes the institution's untested decision.",
        )
        points = make_key_points(
            "Separation protects institutional integrity",
            "Honesty does not remove structural risk",
            "Independent scrutiny improves defensibility",
            "Audit tests process robustness, not only intent",
        )
    elif analytical_type == 6:
        question = (
            f"{opener} treating standardization and proprietary purchase as interchangeable lead to weak reasoning "
            f"for {item} {context}?"
        )
        answer = line_answer(
            "Because the two ideas answer different questions: compatibility versus exclusivity.",
            "If the file does not separate them, it may skip competition where alternatives could still meet the technical need.",
        )
        points = make_key_points(
            "Standardization is not automatically exclusivity",
            "Compatibility should be proved, not assumed",
            "Mislabeling can overstate single-source justification",
            "Competition may still be possible",
        )
    elif analytical_type == 7:
        question = (
            f"Why are outdated threshold references dangerous even when the final route chosen for a "
            f"{format_rs(amount)} {item} case happens to look reasonable on the facts?"
        )
        answer = line_answer(
            "Because a file that relies on stale thresholds is vulnerable even if the outcome accidentally lands near the right route.",
            "The concern is not only the result but whether the decision was made through the current controlling framework.",
        )
        points = make_key_points(
            "Correct result reached through wrong logic is still risky",
            "Current threshold table should be cited explicitly",
            "Outdated references weaken defensibility",
            "Audit checks reasoning chain, not outcome alone",
        )
    elif analytical_type == 8:
        question = (
            f"{opener} verifying local-content claims before using purchase preference matter more in a "
            f"higher-value tender for {item} than in casual low-value buying?"
        )
        answer = line_answer(
            "Because preference affects eligibility, ranking, and award outcome more seriously as procurement value and competition intensity increase.",
            "If local-content status is wrong, the tender may be distorted at the evaluation stage rather than merely at a minor filing stage.",
        )
        points = make_key_points(
            "Local-content status can change eligibility and preference",
            "Higher-value tenders magnify evaluation consequences",
            "Verification should precede award benefit",
            "Misclassification can taint the final ranking",
        )
    else:
        question = (
            f"{opener} sparse file documentation make post-award complaints on {item} harder to defend even "
            "if the committee's judgment was probably correct?"
        )
        answer = line_answer(
            "Because undocumented reasoning cannot be reconstructed safely after the dispute begins.",
            "A sound decision that is weakly recorded may fail audit or complaint review because the institution cannot show how it reached the conclusion.",
        )
        points = make_key_points(
            "Documentation preserves the reasoning trail",
            "Post-award defence depends on the file, not memory",
            "Sparse notes increase complaint vulnerability",
            "Correct decisions still need recorded support",
        )

    return maybe_add_clause(question, flags, rng, "analytical"), answer, points


def rule_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    opener = choose(rng, RULE_STYLE_OPENERS)
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("rule", difficulty))
    role = choose(rng, ROLES)
    context = choose(rng, CONTEXTS)
    rule_type = rng.randrange(10)

    if rule_type == 0:
        question = (
            f"{opener} {RULES['gem']}, what factual showing normally justifies stepping away from GeM in a "
            f"{format_rs(amount)} purchase of {item} rather than merely preferring an offline seller?"
        )
        answer = line_answer(
            "The file should show non-availability, material specification mismatch, or another recorded exception, not just convenience or a lower informal quote.",
            "A defensible departure depends on evidence of why GeM could not meet the requirement on the facts.",
        )
        points = make_key_points(
            "Price preference alone is weak",
            "Need evidence of non-availability or mismatch",
            "Record the departure basis explicitly",
            "Only then move to the normal procurement route",
        )
    elif rule_type == 1:
        question = (
            f"{opener} {RULES['direct']}, what is the most commonly missed requirement when a {role} assumes "
            f"that low value for {item} means no reasoning is needed?"
        )
        answer = line_answer(
            "Low value reduces procedure, but it does not eliminate the need to show market-rate reasonableness and basic sourcing discipline.",
            "A direct-purchase file still needs a rational record of why the price and route were acceptable.",
        )
        points = make_key_points(
            "Direct purchase is not zero-document procurement",
            "Market reasonableness should still appear on file",
            "GeM relevance should not be ignored",
            "Competent approval still matters",
        )
    elif rule_type == 2:
        question = (
            f"{opener} {RULES['lpc']}, at what point does the committee route stop being available if quotations "
            f"for {item} have already started but the value picture shifts {context}?"
        )
        answer = line_answer(
            "It stops being safe when the controlling estimated value moves beyond the LPC band.",
            "If the realistic procurement value crosses that ceiling, the file should be re-tested under the next route instead of clinging to the earlier committee process.",
        )
        points = make_key_points(
            "LPC availability depends on the controlling value band",
            "Started process does not justify staying in the wrong slab",
            "Revise the estimate if needed",
            "Shift route when the band changes",
        )
    elif rule_type == 3:
        question = (
            f"{opener} {RULES['lte']}, what is the real control purpose behind inviting multiple firms for "
            f"{item} rather than treating one known source as enough?"
        )
        answer = line_answer(
            "The point is to preserve a meaningful test of competition and reasonableness within a limited market.",
            "If the file jumps straight to one source without proving why others are unavailable, it loses the protection that limited competition was meant to provide.",
        )
        points = make_key_points(
            "Multiple invitations protect competition",
            "One known source is not the same as only possible source",
            "Reasonableness is easier to defend with comparative response",
            "Explain low-response situations separately",
        )
    elif rule_type == 4:
        question = (
            f"{opener} {RULES['ote']}, what practical obligation follows from wide publicity for a "
            f"{format_rs(amount)} {item} tender beyond simply uploading a notice somewhere?"
        )
        answer = line_answer(
            "Wide publicity means the tender should be genuinely discoverable to the relevant market, not merely technically posted.",
            "If the notice is obscure, too short-lived, or paired with restrictive specifications, the spirit of open competition is not really met.",
        )
        points = make_key_points(
            "Publicity should be meaningful, not token",
            "Visibility and market reach matter",
            "Restrictive specs can defeat open notice",
            "Competition quality affects defensibility",
        )
    elif rule_type == 5:
        question = (
            f"{opener} {RULES['pac']}, what must exist beyond the phrase 'single source' before a proprietary "
            f"purchase of {item} becomes supportable?"
        )
        answer = line_answer(
            "The file needs a real basis for exclusivity, usually backed by competent technical certification and reasons why alternatives are not workable.",
            "Without that, 'single source' is just a label rather than a justified departure from competition.",
        )
        points = make_key_points(
            "Single-source label is not enough",
            "Need technical basis for exclusivity",
            "PAC should come from competent authority",
            "Alternative availability must be examined",
        )
    elif rule_type == 6:
        question = (
            f"{opener} {RULES['gem']} and the value-slab rules together, which question should a {role} answer first "
            f"on file for {item} and why?"
        )
        answer = line_answer(
            "The file should answer the GeM question first because sourcing availability can control whether the buyer may move outside the platform at all.",
            "Once that is settled, the value slab tells the buyer which downstream method governs the case.",
        )
        points = make_key_points(
            "Check GeM before route selection",
            "Platform logic and value logic should be read together",
            "Do not treat them as competing silos",
            "Record the sequence clearly in the note",
        )
    elif rule_type == 7:
        question = (
            f"{opener} the PAC framework, what defect makes a certificate weak even if {item} may in fact be unique?"
        )
        answer = line_answer(
            "A weak signatory, vague reasoning, or missing explanation of alternatives can all undermine the certificate.",
            "Even a genuinely special item needs a PAC that is competent, specific, and traceable to the actual requirement.",
        )
        points = make_key_points(
            "Authority, specificity, and reasoning all matter",
            "Vague PAC language is risky",
            "Actual requirement should match the PAC claim",
            "Alternative-source analysis should not be absent",
        )
    elif rule_type == 8:
        question = (
            f"{opener} direct purchase and LPC logic, why is the exact boundary note important when a "
            f"{item} case sits at {format_rs(rng.choice([50_000, 50_001, 500_000, 500_001]))}?"
        )
        answer = line_answer(
            "Boundary cases attract avoidable disputes if the note does not explain why one side of the slab was chosen.",
            "A short explanation on the estimated value and the precise threshold can prevent later confusion about whether the wrong route was adopted.",
        )
        points = make_key_points(
            "Boundary reasoning should be explicit",
            "Use precise estimated value",
            "Avoid casual rounding near thresholds",
            "Prevent retrospective route disputes",
        )
    else:
        question = (
            f"Which procurement issue does competent authority approval fail to cure if the route for {item} "
            f"or the factual basis behind it is wrong?"
        )
        answer = line_answer(
            "Approval does not cure a fundamentally defective route choice or an unsupported exception.",
            "Authority can approve only within a lawful and factually supported framework; it cannot turn a weak basis into a strong one by signature alone.",
        )
        points = make_key_points(
            "Approval is not a substitute for legal basis",
            "Unsupported exception remains weak despite signature",
            "Route logic and facts must be correct first",
            "Competence and compliance are separate checks",
        )

    return maybe_add_clause(question, flags, rng, "rule"), answer, points


def process_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("process", difficulty))
    route, rule = route_for_amount(amount)
    process_type = rng.randrange(10)

    if process_type == 0:
        question = (
            f"{item} worth {format_rs(amount)} is not traceable on GeM in the required configuration."
            " Before buying offline, what sequence of actions should the officer follow?"
        )
        answer = line_answer(
            "First document the GeM search and the mismatch or non-availability, then move to the normal value-based route.",
            f"After the GeM record is complete, the file should proceed through {route} under {rule} with the usual approvals and comparative documentation.",
        )
        points = make_key_points(
            "Document GeM search results first",
            "Record mismatch or non-availability clearly",
            f"Then apply {rule} on value",
            "Keep approval and comparison records in sequence",
        )
    elif process_type == 1:
        question = (
            f"An LTE for {item} produced only one responsive offer."
            f" The estimated value remains {format_rs(amount)}."
            " What is the correct workflow from this point?"
        )
        answer = line_answer(
            "The committee should first test whether competition was adequate, then decide whether re-tendering or acceptance is justified.",
            "If proceeding, the file should record reasonableness, urgency if any, and competent approval on the single-offer decision.",
        )
        points = make_key_points(
            "Review adequacy of market reach and specifications",
            "Consider re-tendering where appropriate",
            "Document value for money if proceeding",
            "Obtain approval on the final course",
        )
    elif process_type == 2:
        question = (
            f"A user division wants a PAC-backed purchase of {item}."
            " How should the file move, step by step, before single-source approval is considered?"
        )
        answer = line_answer(
            "The file should first define the exact requirement, then test whether exclusivity is real, and only then seek competent PAC certification.",
            "After that, it should record why competition is infeasible, attach approvals, and preserve the supporting technical note.",
        )
        points = make_key_points(
            "Define the requirement precisely",
            "Test whether alternatives exist",
            "Obtain competent technical PAC certification",
            "Record why competition is not feasible",
        )
    elif process_type == 3:
        question = (
            f"Quotations for {item} were sought on an estimate below a threshold, but the realistic total is now higher because mandatory charges were omitted."
            " What is the correct workflow from this point?"
        )
        answer = line_answer(
            "The estimate should be corrected first, then the route should be re-tested on the revised total.",
            "If the revised value changes the governing slab, the buyer should not continue the old process as though the valuation mistake never happened.",
        )
        points = make_key_points(
            "Revise the estimate before continuing",
            "Re-test the route on total value",
            "Change method if the slab changes",
            "Record why the original estimate was incomplete",
        )
    elif process_type == 4:
        question = (
            f"A March-approved purchase of {item} will spill into the next financial year."
            " What sequence of checks should the dealing hand complete before issue of supply order or booking of expenditure?"
        )
        answer = line_answer(
            "The officer should check budget provision, approval validity, and booking year before assuming continuity.",
            "If any of those controls fail, the file needs renewal, regularization, or revised sanction before moving further.",
        )
        points = make_key_points(
            "Check budget availability in the relevant year",
            "Review validity of prior approval",
            "Confirm how expenditure will be booked",
            "Regularize or renew sanction if needed",
        )
    elif process_type == 5:
        question = (
            f"The evaluation committee wants to reject L1 for {item} and consider L2."
            " What file sequence keeps that decision defensible?"
        )
        answer = line_answer(
            "The file should record the technical or responsiveness failure first, then compare the surviving compliant bidders, then place a reasoned recommendation for approval.",
            "Skipping directly from L1 rejection to award recommendation is weaker than showing the full evaluation chain.",
        )
        points = make_key_points(
            "Record L1 rejection grounds first",
            "Evaluate remaining compliant bidders afresh",
            "Maintain comparative reasoning on file",
            "Seek approval on the final recommendation",
        )
    elif process_type == 6:
        question = (
            f"During bid evaluation for {item}, a bidder claims local supplier preference."
            " What sequence should the committee follow before applying that benefit?"
        )
        answer = line_answer(
            "The committee should first verify the bidder's declared class and local-content basis, then test value-band eligibility, and only then apply ranking or split-order preference.",
            "Preference should follow verified status, not precede it.",
        )
        points = make_key_points(
            "Check local-content declaration first",
            "Confirm class and eligibility for the value band",
            "Apply preference only after verification",
            "Record the evaluation basis clearly",
        )
    elif process_type == 7:
        question = (
            f"The GeM portal remained inaccessible while {item} was needed urgently."
            " How should the file move, step by step, if an offline purchase is contemplated?"
        )
        answer = line_answer(
            "Capture downtime evidence first, then record urgency, then choose the offline route that matches the value band.",
            "The file should also explain why waiting for portal restoration would have harmed the requirement.",
        )
        points = make_key_points(
            "Collect downtime proof",
            "Record urgency and impact of delay",
            "Apply the correct offline value route",
            "Preserve approval and supporting justification",
        )
    elif process_type == 8:
        question = (
            f"A repeat order to the same vendor for {item} is being considered."
            " What checks should be completed in sequence before the proposal is cleared?"
        )
        answer = line_answer(
            "The file should check whether the original order was competitive, whether the price is unchanged or lower, and whether the validity conditions still support repetition.",
            "Only after those tests should the approval note treat repeat ordering as a serious option.",
        )
        points = make_key_points(
            "Verify competitive origin of the earlier order",
            "Check price continuity or reduction",
            "Review validity period and continuing need",
            "Take approval on the repeat-order rationale",
        )
    else:
        question = (
            f"Several departments want related {item} purchases from the same budget head within the same quarter."
            " What sequence should be followed to decide whether the demands must be clubbed?"
        )
        answer = line_answer(
            "First test purpose, timing, funding, and interchangeability of the demands; then estimate the aggregate if they are really one requirement.",
            "Only after that should the buyer choose the route and document why clubbing was or was not required.",
        )
        points = make_key_points(
            "Test whether the demands are truly related",
            "Use aggregate value if they form one requirement",
            "Choose route after clubbing analysis",
            "Record reasons for clubbing or separation",
        )

    return maybe_add_clause(question, flags, rng, "process"), answer, points


def threshold_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    threshold_type = rng.randrange(9)

    if threshold_type == 0:
        amount = choose(rng, amount_pool("threshold", difficulty))
        route, rule = route_for_amount(amount)
        question = f"For {item} estimated at {format_rs(amount)}, which method controls if no exception facts are present?"
        answer = line_answer(
            f"The controlling method is {route} under {rule}.",
            "That assumes the estimate is realistic and that no GeM, PAC, or other exception changes the sourcing logic first.",
        )
        points = make_key_points(
            "Map the estimate to the correct slab",
            "Use realistic total value",
            "Check whether any exception facts change the position",
            "Record the chosen rule in the note",
        )
    elif threshold_type == 1:
        amount = rng.choice([50_000, 50_001, 500_000, 500_001, 5_000_000, 5_000_001])
        route, rule = route_for_amount(amount)
        question = f"At exactly {format_rs(amount)} for {item}, does the case stay in the lower band or move to the next one?"
        answer = line_answer(
            f"At {format_rs(amount)}, the file should apply {route} under {rule}.",
            "Boundary notes are worth making explicit because rounding language often creates avoidable confusion near thresholds.",
        )
        points = make_key_points(
            "Apply exact boundary value",
            "Do not round away the cut-off",
            "State the chosen band expressly",
            "Revise if the estimated total later changes",
        )
    elif threshold_type == 2:
        base = rng.choice([485_000, 495_000, 2_480_000])
        extra = rng.choice([21_000, 38_000, 55_000])
        total = base + extra
        route, rule = route_for_amount(total)
        question = (
            f"The quoted base price for {item} is {format_rs(base)}, but mandatory freight and installation add {format_rs(extra)}."
            " Which figure should drive route selection?"
        )
        answer = line_answer(
            f"The route should be chosen on the realistic total of {format_rs(total)}, which places the case in {route} under {rule}.",
            "Mandatory charges should not be ignored simply because the base price sits below a more convenient boundary.",
        )
        points = make_key_points(
            "Use total landed or mandatory value",
            "Ignore artificial understatements of cost",
            "Shift route if the total crosses a slab",
            "Document the calculation basis",
        )
    elif threshold_type == 3:
        amount1 = rng.choice([500_000, 500_001, 5_000_000, 5_000_001])
        route1, rule1 = route_for_amount(amount1)
        amount2 = amount1 + 1
        route2, rule2 = route_for_amount(amount2)
        question = (
            f"How does the route change for {item} when the estimate moves from {format_rs(amount1)} to {format_rs(amount2)} after a realistic revision?"
        )
        answer = line_answer(
            f"At {format_rs(amount1)} the route is {route1} under {rule1}, while at {format_rs(amount2)} it becomes {route2} under {rule2} if that crosses the boundary.",
            "The note should show that the revised estimate, not the original hope, controls the later route decision.",
        )
        points = make_key_points(
            "Compare route before and after revision",
            "Use revised realistic estimate",
            "A one-rupee crossing can still change the slab",
            "Update the route note when the estimate changes",
        )
    elif threshold_type == 4:
        line_value = rng.choice([45_000, 50_000, 75_000])
        qty = rng.choice([8, 10, 12])
        total = line_value * qty
        route, rule = route_for_amount(total)
        question = (
            f"If {qty} similar units of {item} each cost about {format_rs(line_value)} for one project, should the route follow the unit price or the aggregate value of {format_rs(total)}?"
        )
        answer = line_answer(
            f"The buyer should test whether the demand is one requirement; if it is, the aggregate value of {format_rs(total)} should control, leading to {route} under {rule}.",
            "Unit-wise treatment is weaker when the items share one purpose, timing, and budget.",
        )
        points = make_key_points(
            "Assess whether the units form one requirement",
            "Use aggregate value if they do",
            "Do not let unit price hide the true slab",
            "Record clubbing logic on file",
        )
    elif threshold_type == 5:
        amount = rng.choice([250_000, 400_000, 900_000, 3_000_000])
        route, rule = route_for_amount(amount)
        question = (
            f"The stores note for {item} cites the updated threshold table, but the user note still mentions the old Rs. 2.5 lakh cut-off."
            f" For a case of {format_rs(amount)}, which threshold note should be followed?"
        )
        answer = line_answer(
            f"The file should follow the updated threshold table, which places {format_rs(amount)} in {route} under {rule}.",
            "The stale lower cut-off should be corrected explicitly rather than silently carried forward into the approval note.",
        )
        points = make_key_points(
            "Use current threshold table",
            "Correct stale threshold references",
            "Apply current route for the actual amount",
            "Keep the reasoning explicit on file",
        )
    elif threshold_type == 6:
        amount = rng.choice([24_999, 25_000, 25_001, 49_999, 50_000, 50_001])
        route, rule = route_for_amount(amount)
        question = f"For a small-value purchase of {item} at {format_rs(amount)}, does the threshold answer alone settle the case if the item is available on GeM?"
        answer = line_answer(
            f"No. The value suggests {route} under {rule}, but GeM still has to be tested first.",
            "Threshold and source-platform logic must be read together rather than treating the value slab as the only question.",
        )
        points = make_key_points(
            "Threshold answer is not the whole answer",
            "Check GeM availability first",
            "Then apply the value band",
            "Record both parts of the reasoning",
        )
    elif threshold_type == 7:
        amount = rng.choice([5_000_000, 5_000_001])
        route, rule = route_for_amount(amount)
        question = f"A note says {item} is 'around 50 lakh' and proposes LTE without pinning down whether the estimate is exactly or above that figure. Why is the exact number critical?"
        answer = line_answer(
            f"Because the route changes at the boundary: the exact figure determines whether the case remains in LTE or moves into {route} under {rule}.",
            "Rounding language is risky near slab transitions, so the estimate should be recorded precisely.",
        )
        points = make_key_points(
            "Exact estimate matters at boundary values",
            "Avoid vague rounded expressions near cut-offs",
            "Route can change immediately after the threshold",
            "Record the precise figure on file",
        )
    else:
        amount = choose(rng, amount_pool("threshold", difficulty))
        route, rule = route_for_amount(amount)
        question = (
            f"For {item} estimated at {format_rs(amount)}, show the applicable threshold table as well and indicate which slab controls"
            " if no exception facts are present."
        )
        answer = table_answer(
            f"At {format_rs(amount)}, the controlling method is {route} under {rule}; the slab table is below.",
            "This assumes the estimate is realistic and that no GeM, PAC, or other exception changes the sourcing logic first.",
        )
        points = make_key_points(
            "If the user asks for a table, include the slab table itself",
            "Identify the controlling slab explicitly",
            "Keep the route answer aligned with the table",
            "Treat exceptions separately from the base threshold table",
        )

    return maybe_add_clause(question, flags, rng, "threshold"), answer, points


def pac_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("pac", difficulty))
    pac_type = rng.randrange(10)

    if pac_type == 0:
        question = f"For {item} worth {format_rs(amount)}, when does a PAC mean a Proprietary Article Certificate rather than just a convenient single-source note?"
        answer = line_answer(
            "It means a Proprietary Article Certificate only when the file is using genuine proprietary or otherwise justified single-source logic under the PAC framework.",
            "A convenience-based vendor preference is weaker and should not be labelled as PAC procurement unless exclusivity is actually demonstrated.",
        )
        points = make_key_points(
            "PAC here means Proprietary Article Certificate",
            "PAC is tied to justified single-source procurement",
            "Convenience is not enough",
            "Exclusivity must be shown on file",
        )
    elif pac_type == 1:
        question = f"A vendor says only it can supply {item}, but no competent technical officer has issued a PAC. Can the buyer safely proceed as proprietary?"
        answer = line_answer(
            "Not safely. A vendor's self-claim is not the same as a competent Proprietary Article Certificate.",
            "The buyer should first test whether alternatives exist and obtain proper technical certification before treating the case as proprietary.",
        )
        points = make_key_points(
            "Vendor claim is not PAC",
            "Need competent technical certification",
            "Check alternative sources",
            "Do not jump to proprietary route on supplier assertion alone",
        )
    elif pac_type == 2:
        question = f"Who should issue the PAC for {item} if the lab wants a single-source purchase to stand up in audit?"
        answer = line_answer(
            "The PAC should come from a competent serving technical authority who can explain why the item is proprietary or otherwise requires that source.",
            "It should not be issued only by the procurement desk or by someone without current technical authority over the requirement.",
        )
        points = make_key_points(
            "Require competent serving technical authority",
            "Technical reasoning should support the PAC",
            "Administrative staff alone should not issue it",
            "Authority and expertise both matter",
        )
    elif pac_type == 3:
        question = f"The OEM confirms {item} is genuine, but several authorized dealers can supply it. Does that fact by itself support PAC procurement?"
        answer = line_answer(
            "No. Multiple capable supply channels weaken a pure proprietary argument unless another single-source basis is proved.",
            "The file should distinguish product authenticity from source exclusivity before deciding on PAC logic.",
        )
        points = make_key_points(
            "Authenticity is not the same as source exclusivity",
            "Multiple dealers undercut a pure PAC claim",
            "Separate OEM genuineness from procurement route",
            "Use competition if alternatives are available",
        )
    elif pac_type == 4:
        question = f"A PAC has been drafted for {item}, but the actual reason is compatibility with an existing line rather than true market exclusivity. How should that be treated?"
        answer = line_answer(
            "The file should state the compatibility or standardization reason honestly instead of disguising it as pure exclusivity.",
            "Single-source approval may still be possible on facts, but the technical basis should match the real ground being used.",
        )
        points = make_key_points(
            "Describe the true basis for single-source choice",
            "Do not mislabel standardization as pure exclusivity",
            "Require technical reasoning on compatibility",
            "Keep the route aligned with the actual facts",
        )
    elif pac_type == 5:
        question = f"The PAC for {item} is signed by an administrative officer because the scientist is on leave. Is the certificate still safe to rely on?"
        answer = line_answer(
            "Usually no. The problem is not just signature formality; it is the lack of competent technical certification.",
            "The file should wait for or obtain a technically competent PAC signatory if proprietary logic is to be defended.",
        )
        points = make_key_points(
            "Administrative signature is not enough",
            "PAC needs technical competence",
            "Authority defect affects audit defensibility",
            "Re-issue through proper signatory",
        )
    elif pac_type == 6:
        question = f"For {item} at {format_rs(amount)}, what additional record should accompany a PAC if the buyer wants the single-source route to look credible in audit?"
        answer = line_answer(
            "The PAC should be supported by a clear note on why competition is not feasible and, where possible, by market or compatibility evidence.",
            "The certificate is central, but the file should also show the factual trail behind it.",
        )
        points = make_key_points(
            "Attach reasoned justification behind the PAC",
            "Show why competition is infeasible",
            "Preserve technical or market evidence",
            "Keep approval trail with the PAC",
        )
    elif pac_type == 7:
        question = f"A failed competitive tender is followed by a proposal to buy {item} from one source. Does the failed tender automatically eliminate the need for PAC reasoning?"
        answer = line_answer(
            "No. Failed competition may support a single-source decision in some cases, but it does not magically convert the case into a proprietary purchase.",
            "The file should explain the market outcome, why re-tendering is not useful, and what legal basis now supports the one-source route.",
        )
        points = make_key_points(
            "Failed tender is not automatic PAC",
            "Record why competition failed",
            "Explain why re-tendering is not preferable",
            "State the actual single-source basis now being used",
        )
    elif pac_type == 8:
        question = f"If {item} exceeds {format_rs(5_000_000)}, what extra caution should the file apply before treating the case as PAC-based?"
        answer = line_answer(
            "Higher value does not change the meaning of PAC, but it increases the need for precise justification, approval discipline, and scrutiny of alternatives.",
            "The file should be especially clear on why open competition is not feasible and who approved the single-source path.",
        )
        points = make_key_points(
            "Higher value raises scrutiny",
            "Justification should be specific and robust",
            "Approval discipline becomes more important",
            "Alternative-source analysis should be visible",
        )
    else:
        question = f"A buyer says a PAC is unnecessary because everyone in the lab already knows that {item} normally comes only from the OEM. Is that enough?"
        answer = line_answer(
            "No. Shared internal belief is weaker than a documented Proprietary Article Certificate backed by technical reasons.",
            "The file should rely on recorded evidence rather than institutional memory or habit.",
        )
        points = make_key_points(
            "Institutional habit is not PAC evidence",
            "Need documented proprietary reasoning",
            "Technical authority should certify the claim",
            "File record matters more than oral understanding",
        )

    return maybe_add_clause(question, flags, rng, "pac"), answer, points


def gem_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("gem", difficulty))
    route, rule = route_for_amount(amount)
    gem_type = rng.randrange(8)

    if gem_type == 0:
        question = f"If {item} is available on GeM, does a value of {format_rs(amount)} let the buyer ignore the platform and proceed straight to {route}?"
        answer = line_answer(
            "No. Value decides the method, but GeM availability still has to be addressed first.",
            f"The buyer should test {RULES['gem']} before relying on the threshold route under {rule}.",
        )
        points = make_key_points(
            "GeM comes before offline route choice",
            "Value and platform logic both matter",
            "Do not skip GeM because the amount looks simple",
            "Record the sourcing decision separately from the slab decision",
        )
    elif gem_type == 1:
        question = f"The GeM listing for {item} is costlier than an informal market quote. Can the buyer leave GeM on that ground alone?"
        answer = line_answer(
            "Usually no. A cheaper informal quote by itself is weaker than the mandatory GeM check where the item is available.",
            "The file should leave GeM only on a real specification or availability problem, not merely because an outside seller looks cheaper.",
        )
        points = make_key_points(
            "Price difference alone is weak",
            "Check whether the GeM item matches the true requirement",
            "Record any real mismatch or non-availability",
            "Do not treat informal offline quotes as automatic override",
        )
    elif gem_type == 2:
        question = f"For {item}, GeM shows broadly similar products but not the exact specification needed. What is the compliant response?"
        answer = line_answer(
            "The buyer should document the specification gap rather than making a bare assertion that GeM is unsuitable.",
            f"Once the mismatch is recorded, the case can move to the normal value route for {format_rs(amount)} under {rule}.",
        )
        points = make_key_points(
            "Record the exact specification gap",
            "Keep evidence of the GeM search",
            "Do not rely on vague non-feasibility wording",
            "Then apply the normal offline route",
        )
    elif gem_type == 3:
        question = f"The GeM portal was down during a need for {item}. What minimum evidence should exist before an offline purchase is defended?"
        answer = line_answer(
            "At minimum, the file should show downtime evidence, attempted access, timing, and why waiting was not practical.",
            "Without that, the offline move can look like convenience rather than necessity.",
        )
        points = make_key_points(
            "Capture downtime proof",
            "Record timestamps and attempts",
            "Explain urgency or impracticality of delay",
            "Link the exception to the actual procurement need",
        )
    elif gem_type == 4:
        question = f"How should GeM be treated for a low-value purchase of {item} below {format_rs(50_000)} when the item is plainly listed there?"
        answer = line_answer(
            "Low value does not erase GeM relevance.",
            "Even where direct purchase logic may otherwise look available, the file should still explain why GeM was used or why departure was justified.",
        )
        points = make_key_points(
            "Low value is not a free pass around GeM",
            "Check listing availability",
            "Record the sourcing logic",
            "Keep basic price reasonableness on file",
        )
    elif gem_type == 5:
        question = f"A department wants to use an old offline rate contract for {item} even though equivalent options are visible on GeM. Which source should be examined first?"
        answer = line_answer(
            "GeM should be examined first because current availability there can control sourcing before the old offline arrangement is treated as relevant.",
            "An old convenient channel does not automatically outrank a live mandatory platform path.",
        )
        points = make_key_points(
            "Prioritize current GeM availability check",
            "Do not assume old offline arrangements override platform logic",
            "Record why any non-GeM source remains justified",
            "Then apply route and approval logic",
        )
    elif gem_type == 6:
        question = f"The file says GeM cannot meet the delivery deadline for {item}, but there is no vendor lead-time comparison attached. Is the exception ready?"
        answer = line_answer(
            "Not fully. Delivery urgency may justify departure, but the file should show how the timing gap was actually assessed.",
            "A conclusion without comparison evidence is weaker than a note that traces the deadline problem concretely.",
        )
        points = make_key_points(
            "Support delivery-urgency claim with facts",
            "Compare available timelines where possible",
            "Do not rely on bare assertion",
            "Link urgency to the procurement need",
        )
    else:
        question = f"An officer says GeM was checked for {item}, but the screenshots are missing from the file. How much audit comfort should that statement provide?"
        answer = line_answer(
            "Limited comfort. A missing evidence trail makes the sourcing decision harder to defend later.",
            "The file is stronger when the search record, listing review, and basis for departure are preserved rather than recalled after the fact.",
        )
        points = make_key_points(
            "Missing screenshots weaken audit trail",
            "Preserve evidence at the time of decision",
            "Statement alone is weaker than documentary proof",
            "Document the basis for using or leaving GeM",
        )

    return maybe_add_clause(question, flags, rng, "gem"), answer, points


def mii_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    amount = choose(rng, amount_pool("make_in_india", difficulty))
    mii_type = rng.randrange(8)

    if mii_type == 0:
        pct = rng.choice([18, 20, 35, 49, 50, 67])
        question = f"A bidder for {item} declares {pct}% local content. How should that declaration be classified before any preference is applied?"
        if pct >= 50:
            label = "Class I local supplier"
        elif pct >= 20:
            label = "Class II local supplier"
        else:
            label = "non-local supplier"
        answer = line_answer(
            f"On the face of the percentage, the bidder falls in the {label} bucket.",
            "But preference should still wait until the declaration is accepted as adequately supported under the applicable Make in India framework.",
        )
        points = make_key_points(
            "Use the declared percentage to classify provisionally",
            "Class I is at least 50%, Class II is at least 20% but below 50%",
            "Below 20% is non-local",
            "Preference follows verification, not just assertion",
        )
    elif mii_type == 1:
        question = f"For {item} above {format_rs(5_000_000)}, can a non-local supplier be treated as ordinarily eligible just because its price is lowest?"
        answer = line_answer(
            "Not ordinarily under the repo's Make in India logic for higher-value tenders.",
            "Eligibility and preference should be examined before price ranking is treated as decisive; a low price does not cure an ineligible status.",
        )
        points = make_key_points(
            "Check eligibility before ranking",
            "Do not let lowest price override class-based restrictions",
            "Apply the higher-value Make in India rule first",
            "Record how eligibility was determined",
        )
    elif mii_type == 2:
        question = f"L1 for {item} is non-local, but a local supplier is within the preference margin and agrees to match L1. What decision logic should the committee apply?"
        answer = line_answer(
            "The committee should test whether the local supplier qualifies for the relevant purchase preference and whether the match condition is actually met.",
            "If those conditions hold, the preference or split-order consequence can be applied; if not, L1 remains stronger.",
        )
        points = make_key_points(
            "Check local-supplier class and eligibility",
            "Verify that the margin condition is satisfied",
            "Confirm willingness to match L1 price",
            "Apply preference only after those checks",
        )
    elif mii_type == 3:
        question = f"A supplier's local-content self-certification for {item} looks exaggerated. Why is that not just a minor paperwork issue?"
        answer = line_answer(
            "Because local-content status can affect eligibility, ranking, and award outcome.",
            "A doubtful declaration can distort the tender itself, not merely the filing record, so it should be checked before any preference is granted.",
        )
        points = make_key_points(
            "Local-content status can change the result",
            "False declaration is more than clerical error",
            "Verification should happen before award benefit",
            "Record how the doubt was resolved",
        )
    elif mii_type == 4:
        question = f"When does the 50-50 split-order idea become available in a Make in India evaluation for {item}, and what condition still has to be met?"
        answer = line_answer(
            "It becomes relevant only when the applicable preference framework allows that consequence and the eligible local supplier is willing to match the L1 price.",
            "Without the match condition, a split-order outcome is usually not defensible.",
        )
        points = make_key_points(
            "Split-order is conditional, not automatic",
            "Eligible local supplier must fit the preference rule",
            "Matching L1 price is essential",
            "Record the committee's basis clearly",
        )
    elif mii_type == 5:
        question = f"For {item} around {format_rs(amount)}, should Make in India status be checked before or after the normal route under the GFR threshold table is identified?"
        answer = line_answer(
            "The normal route and the Make in India test should be read together, but the threshold route is usually identified first and then the bidder-eligibility or preference issue is applied within that tender context.",
            "What matters is that local-supplier preference is not treated as a substitute for choosing the base procurement method.",
        )
        points = make_key_points(
            "Do not let Make in India replace route selection",
            "Identify base procurement method first",
            "Apply eligibility or preference within the chosen tender context",
            "Record both layers of reasoning",
        )
    elif mii_type == 6:
        question = f"If too few local suppliers appear capable for {item}, can the buyer assume the Make in India restriction disappears automatically?"
        answer = line_answer(
            "No. The file should record why the local field is insufficient rather than assuming the restriction vanishes by instinct.",
            "Exception handling should be evidence-based, not just a reaction to market discomfort.",
        )
        points = make_key_points(
            "Do not assume automatic relaxation",
            "Record why local capability is insufficient",
            "Use evidence rather than intuition",
            "Keep the exception logic explicit",
        )
    else:
        question = f"The bill of materials for {item} is too vague to tell whether the claimed local content is credible. Can the committee still award preference safely?"
        answer = line_answer(
            "Not safely. Preference should follow a supportable basis for local-content classification.",
            "If the build-up is too vague, the committee should seek clarification or verification before adjusting ranking or eligibility.",
        )
        points = make_key_points(
            "Vague bill of materials weakens local-content claim",
            "Seek clarification before giving benefit",
            "Do not grant preference on unsupported classification",
            "Document the verification path",
        )

    return maybe_add_clause(question, flags, rng, "make_in_india"), answer, points


def others_question(rng: random.Random, difficulty: str, flags: dict[str, object]) -> tuple[str, str, list[str]]:
    item = choose(rng, ITEMS)
    other_type = rng.randrange(8)

    if other_type == 0:
        question = f"A supplier of {item} is alleged to have submitted forged documents in a previous procurement. What should happen before the buyer treats the vendor as debarred?"
        answer = line_answer(
            "Debarment should follow a fair inquiry and competent approval rather than an informal assumption.",
            "The buyer may protect the current case, but a formal debarment consequence should rest on due process and recorded findings.",
        )
        points = make_key_points(
            "Debarment should follow inquiry and approval",
            "Allegation alone is not final debarment",
            "Protect the live procurement while facts are examined",
            "Record due-process steps",
        )
    elif other_type == 1:
        question = f"What is the practical difference between bid security and performance security when a complaint arises after award of {item}?"
        answer = line_answer(
            "Bid security addresses tender-stage participation risk, while performance security addresses contract-execution risk after award.",
            "Mixing the two can lead to the wrong remedy when the problem occurs at a different stage of the procurement lifecycle.",
        )
        points = make_key_points(
            "Bid security belongs to tender stage",
            "Performance security belongs to post-award execution",
            "Stage of default determines which protection matters",
            "Do not treat them as interchangeable",
        )
    elif other_type == 2:
        question = f"Why does a comparative statement for {item} matter even when the committee already feels sure which offer is best?"
        answer = line_answer(
            "Because confidence without a comparative record is harder to defend than a reasoned comparison that shows how offers were judged.",
            "The statement converts preference into traceable evaluation.",
        )
        points = make_key_points(
            "Comparative statement preserves evaluation logic",
            "Committee confidence alone is weaker than record",
            "Helps defend price and technical choices",
            "Supports audit and complaint review",
        )
    elif other_type == 3:
        question = f"When is finance concurrence more than a routine signature in a {item} procurement file?"
        answer = line_answer(
            "It becomes more than routine when delegated financial powers, sanction limits, or exceptional routing issues are materially engaged.",
            "In those cases, concurrence is part of control review, not just clerical circulation.",
        )
        points = make_key_points(
            "Finance concurrence depends on DFP and risk",
            "Higher-value or exceptional cases need real scrutiny",
            "It is not merely clerical movement",
            "Record why concurrence was or was not required",
        )
    elif other_type == 4:
        question = f"If delivery of {item} is delayed after award, why is it risky to jump straight to payment withholding without checking the contract remedies carefully?"
        answer = line_answer(
            "Because the contract may distinguish between inspection, liquidated damages, replacement, risk purchase, and payment terms.",
            "Using the wrong remedy can create a second dispute on top of the delivery problem.",
        )
        points = make_key_points(
            "Match remedy to the contract clause and breach type",
            "Delay does not always justify the same response",
            "Check inspection and acceptance status",
            "Keep payment actions contract-linked",
        )
    elif other_type == 5:
        question = f"Should an unregistered vendor offering {item} be rejected automatically if the specifications and price otherwise look acceptable?"
        answer = line_answer(
            "Not automatically in every case, because registration is often a control preference rather than a universal absolute bar.",
            "The file should check whether registration is mandatory for that procurement and whether other risk controls can address the concern.",
        )
        points = make_key_points(
            "Do not assume registration is always mandatory",
            "Check procurement-specific eligibility conditions",
            "Consider other risk controls if registration is absent",
            "Record why the vendor was accepted or refused",
        )
    elif other_type == 6:
        question = f"Why is advance payment for {item} treated more cautiously than routine post-supply payment?"
        answer = line_answer(
            "Because the buyer parts with money before receiving the full contractual assurance of supply or performance.",
            "That risk usually calls for clearer justification, safeguards, and approval discipline than ordinary payment after delivery.",
        )
        points = make_key_points(
            "Advance payment shifts risk to the buyer",
            "Needs stronger safeguards than normal payment",
            "Justification and approval should be explicit",
            "Security or contractual protection may be relevant",
        )
    else:
        question = f"What makes inspection and acceptance of {item} more than a postman-like receipt formalism in procurement control?"
        answer = line_answer(
            "Inspection and acceptance are where the institution tests whether the awarded promise was actually delivered.",
            "If that step is weak, the procurement can look compliant on paper while failing in substance.",
        )
        points = make_key_points(
            "Inspection confirms contractual compliance",
            "Acceptance is not just receipt acknowledgement",
            "Weak inspection undermines real value for money",
            "Record defects, conformity, and acceptance basis",
        )

    return maybe_add_clause(question, flags, rng, "others"), answer, points


GENERATORS = {
    "scenario": scenario_question,
    "edge_case": edge_case_question,
    "analytical": analytical_question,
    "rule": rule_question,
    "process": process_question,
    "threshold": threshold_question,
    "pac": pac_question,
    "gem": gem_question,
    "make_in_india": mii_question,
    "others": others_question,
}


def _contains_markdown_table(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return False
    return any(set(line.replace("|", "").strip()) <= {"-", ":", " "} for line in table_lines)


def validate_expected_answer(answer: str) -> None:
    lines = [line for line in answer.splitlines() if line.strip()]
    if _contains_markdown_table(answer):
        if len(lines) < 4:
            raise ValueError(f"table expected_answer is too short: {answer!r}")
        return
    if not (2 <= len(lines) <= 4):
        raise ValueError(f"expected_answer must be 2-4 lines, got {len(lines)}: {answer!r}")


def generate_dataset() -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, object]] = []
    batches_summary: list[dict[str, object]] = []
    seen_questions: set[str] = set()
    global_id = 1

    for batch_number in range(1, BATCHES + 1):
        batch_rng = random.Random(SEED + batch_number * 1009)
        batch_records: list[dict[str, object]] = []
        batch_meta: list[dict[str, object]] = []

        category_order = list(CATEGORY_COUNTS.keys())
        batch_rng.shuffle(category_order)

        for category in category_order:
            schedule = make_schedule(category, batch_rng)
            generator = GENERATORS[category]

            for flags in schedule:
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > 250:
                        raise RuntimeError(f"Could not generate unique question for {category} in batch {batch_number}")

                    question, expected_answer, key_points = generator(
                        batch_rng,
                        str(flags["difficulty"]),
                        flags,
                    )
                    normalized = normalized_question(question)
                    if normalized in seen_questions:
                        continue

                    validate_expected_answer(expected_answer)
                    if len(key_points) < 3:
                        raise ValueError(f"Too few key points: {question}")

                    record = {
                        "id": global_id,
                        "question": question,
                        "category": category,
                        "expected_answer": expected_answer,
                        "key_points": key_points,
                        "difficulty": flags["difficulty"],
                    }
                    batch_records.append(record)
                    batch_meta.append(
                        {
                            "category": category,
                            "difficulty": flags["difficulty"],
                            "tricky": bool(flags["tricky"]),
                            "conflict": bool(flags["conflict"]),
                            "decision": bool(flags["decision"]),
                        }
                    )
                    seen_questions.add(normalized)
                    global_id += 1
                    break

        batch_records.sort(key=lambda item: int(item["id"]))
        all_records.extend(batch_records)

        batch_path = OUTPUT_DIR / f"procurement_eval_dataset_10000_batch_{batch_number:02d}.json"
        batch_path.write_text(json.dumps(batch_records, indent=2, ensure_ascii=False), encoding="utf-8")

        batch_summary = {
            "batch": batch_number,
            "path": str(batch_path.name),
            "count": len(batch_records),
            "category_counts": dict(Counter(meta["category"] for meta in batch_meta)),
            "difficulty_counts": dict(Counter(meta["difficulty"] for meta in batch_meta)),
            "tricky_count": sum(1 for meta in batch_meta if meta["tricky"]),
            "conflict_count": sum(1 for meta in batch_meta if meta["conflict"]),
            "decision_count": sum(1 for meta in batch_meta if meta["decision"]),
        }
        batches_summary.append(batch_summary)

    COMBINED_FILE.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")

    overall_meta = {
        "batches": batches_summary,
        "total_count": len(all_records),
        "category_counts": dict(Counter(record["category"] for record in all_records)),
        "difficulty_counts": dict(Counter(record["difficulty"] for record in all_records)),
    }

    summary = {
        "seed": SEED,
        "batches": BATCHES,
        "per_batch": QUESTIONS_PER_BATCH,
        "output": COMBINED_FILE.name,
        "overall": overall_meta,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_records, summary


def main() -> None:
    records, summary = generate_dataset()
    if len(records) != 10_000:
        raise SystemExit(f"Expected 10000 records, got {len(records)}")

    for batch in summary["overall"]["batches"]:
        if batch["count"] != QUESTIONS_PER_BATCH:
            raise SystemExit(f"Batch size mismatch: {batch}")

    expected_categories = {category: count * BATCHES for category, count in CATEGORY_COUNTS.items()}
    if summary["overall"]["category_counts"] != expected_categories:
        raise SystemExit(
            f"Category mismatch: expected {expected_categories}, got {summary['overall']['category_counts']}"
        )

    expected_difficulty = {"easy": 3000, "medium": 5000, "hard": 2000}
    if summary["overall"]["difficulty_counts"] != expected_difficulty:
        raise SystemExit(
            f"Difficulty mismatch: expected {expected_difficulty}, got {summary['overall']['difficulty_counts']}"
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
