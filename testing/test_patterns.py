"""Quick test of new pattern matching."""
from app.core.orchestrator import build_contextual_family_answer, ToolExecutionResult, PlannerDecision

planner = PlannerDecision(problem_type="SCENARIO", confidence=0.8, risk_level="MEDIUM",
    needs_rag=True, needs_threshold_logic=True, needs_mii_logic=False, needs_rule_lookup=False, tool_hints=[])
ts = ToolExecutionResult(planner=planner, tools_used=[], documents=[], weak_match=False,
    threshold=None, mii=None, rule_lookup={}, structured_context={"relevant_rules": []},
    source_quality="medium", retrieval_quality=0.5)

tests = [
    ("How many quotations are needed for air conditioner worth Rs. 75,000?", "quotations"),
    ("Who should be part of the purchase committee for deep freezer worth Rs. 2.5 lakhs?", "committee"),
    ("Can I combine purchases of reagents and vacuum pump into a single tender?", "combine"),
    ("What if the delivery of networking equipment is delayed beyond the contract period?", "delivery"),
    ("Can we extend the bid validity period for generator procurement?", "bid_validity"),
    ("Can we procure electrical fittings from an unregistered vendor?", "unreg_vendor"),
    ("Is finance concurrence needed for UPS batteries worth Rs. 3 lakhs?", "finance"),
    ("Is GeM mandatory for analytical balance?", "gem_mandatory"),
    ("Can I skip GeM for purchasing conductivity meter?", "skip_gem"),
    ("How to procure air conditioner through GeM?", "gem_process"),
    ("The GeM price for lab chemicals is higher than local market. Can I buy locally?", "gem_price"),
    ("Who can issue PAC for purchasing mass spectrometer?", "pac"),
    ("Is e-tendering mandatory for autoclave above Rs. 25 lakhs?", "e_tender"),
    ("What is the role of the indenting officer when procuring networking equipment?", "indenting"),
]

passed = 0
for q, label in tests:
    result = build_contextual_family_answer(q, ts)
    matched = "MATCH" if result else "NO MATCH"
    if result:
        passed += 1
    print(f"  [{matched}] {label}: {q[:70]}")
    if result:
        print(f"    -> {result[:120]}...")

print(f"\nMatched: {passed}/{len(tests)}")
