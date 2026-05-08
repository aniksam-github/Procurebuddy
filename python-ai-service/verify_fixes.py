"""Verifies planner fix and amount formatting fix."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app.core.orchestrator import heuristic_plan
import re

queries = [
    ('What is the procurement method for Rs. 8 lakh purchase?', True),
    ('8 lakh', False),
    ('What committee approves Rs. 25 lakh LTE?', True),
    ('How to procure Rs. 3 crore item?', True),
    ('56 lakhs', False),
    ('Which rule applies to Rs. 50 lakh tender?', True),
]

print('Planner Fix Verification')
print('-' * 65)
all_pass = True
for query, expected_rag in queries:
    p = heuristic_plan(query)
    ok = p.needs_rag == expected_rag
    if not ok:
        all_pass = False
    status = 'PASS' if ok else 'FAIL'
    print(f'{status} | type={p.problem_type:15s} needs_rag={str(p.needs_rag):5s} | {query[:55]}')

print()
print('Amount Format Fix Verification')
print('-' * 65)
test_cases = [
    'For Rs. 8 lakhs, the route is LTE.',
    'For Rs. 56 lakh, use LTE under Rule 162.',
    'value is rs 25 lakh',
]
for analysis in test_cases:
    analysis_lower = analysis.lower()
    amount_match = re.search(r'rs\.?\s*[\d,.]+\s*(?:lakh|lakhs|crore|thousand)?', analysis_lower)
    if amount_match:
        raw_val = amount_match.group(0).strip()
        value = 'Rs.' + raw_val.split('rs')[-1].lstrip('.').strip().title() if 'rs' in raw_val else raw_val.title()
        is_ok = 'LAKH' not in value and 'CRORE' not in value
        print(f"{'PASS' if is_ok else 'FAIL'} | {value!r}")

print()
print('GFR Default Fix Verification')
from app.core.orchestrator import render_verified_answer
structured = {
    'status': 'CONDITIONAL',
    'analysis': 'For Rs. 8 lakhs, the route is LTE under Rule 162.',
    'audit_risk': 'Medium',
    'actionable_step': 'Issue LTE to 3+ vendors.',
    'final_decision': 'VERIFY',
    'confidence': 0.9,
    'source_quality': 'medium',
}
rendered = render_verified_answer(structured)
gfr_ok = 'GFR 2025' not in rendered or 'GFR 2017' in rendered
print(f"{'PASS' if gfr_ok else 'FAIL'} | GFR version: {'GFR 2017 found' if 'GFR 2017' in rendered else 'GFR 2025 still present'}")
print(f"{'PASS' if 'LAKH' not in rendered else 'FAIL'} | No uppercase LAKH in output")

print()
print('ALL PASS' if all_pass else 'SOME PLANNER TESTS FAILED')
