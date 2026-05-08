"""Quick analysis of eval failure patterns."""
import json
import collections
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\projects\bot\testing\runs\20260429_142648_sample50_llm_restored\eval_failed_cases.json"

with open(path, encoding="utf-8") as f:
    fails = json.load(f)

reason_counts = collections.Counter()
cat_reasons = collections.defaultdict(lambda: collections.Counter())
cat_scores = collections.defaultdict(list)

for item in fails:
    reasons = item["error_reason"].split(", ")
    cat = item["question_type"]
    cat_scores[cat].append(item["score"])
    for r in reasons:
        reason_counts[r.strip()] += 1
        cat_reasons[cat][r.strip()] += 1

print("=== TOP FAILURE REASONS ===")
for reason, count in reason_counts.most_common(10):
    print(f"  {reason}: {count}")

print("\n=== CATEGORY FAILURE BREAKDOWN ===")
for cat in sorted(cat_reasons.keys(), key=lambda x: -len(cat_scores[x])):
    n = len(cat_scores[cat])
    avg = sum(cat_scores[cat]) / n
    print(f"\n  {cat} ({n} fails, avg={avg:.3f}):")
    for reason, cnt in cat_reasons[cat].most_common(5):
        print(f"    {reason}: {cnt}")

near_miss = [item for item in fails if item["score"] >= 0.6]
print(f"\n=== NEAR-MISSES (score >= 0.6): {len(near_miss)} ===")
nm_reasons = collections.Counter()
for item in near_miss:
    for r in item["error_reason"].split(", "):
        nm_reasons[r.strip()] += 1
for reason, count in nm_reasons.most_common(10):
    print(f"  {reason}: {count}")

# Source-only failures
source_only = [item for item in fails if "missing source" in item["error_reason"] and item["semantic_score"] >= 0.7]
print(f"\n=== HIGH-QUALITY BUT MISSING SOURCE: {len(source_only)} ===")
for item in source_only[:8]:
    cid = item["case_id"]
    cat = item["question_type"]
    sc = item["score"]
    q = item["question"][:80]
    print(f"  {cid} ({cat}) score={sc} q={q}")

# Scenario failures (since scenario is the biggest category)
scenario_fails = [item for item in fails if item["question_type"] == "scenario"]
print(f"\n=== SCENARIO FAILURES: {len(scenario_fails)} ===")
sc_reasons = collections.Counter()
for item in scenario_fails:
    for r in item["error_reason"].split(", "):
        sc_reasons[r.strip()] += 1
for reason, count in sc_reasons.most_common(8):
    print(f"  {reason}: {count}")

# Show some scenario failures with their questions
print("\n  Sample scenario failures:")
for item in scenario_fails[:10]:
    cid = item["case_id"]
    sc = item["score"]
    q = item["question"][:90]
    err = item["error_reason"][:60]
    print(f"    {cid} score={sc} err={err}")
    print(f"      Q: {q}")
