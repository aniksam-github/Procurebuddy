"""Analyze eval_report.csv and categorize failures."""
import csv
from collections import Counter

rows = []
with open("eval/eval_report.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

fail = [r for r in rows if float(r["compliance_precision"]) < 0.3]
warn = [r for r in rows if 0.3 <= float(r["compliance_precision"]) < 0.5]
ok = [r for r in rows if float(r["compliance_precision"]) >= 0.5]

print("=" * 70)
print("  FAILURE ANALYSIS  --  %d total questions" % len(rows))
print("=" * 70)

print("\n  FAIL (prec < 0.30): %d questions" % len(fail))
fail_cats = Counter(r["category"] for r in fail)
for cat, count in fail_cats.most_common():
    print("    %-20s : %d" % (cat, count))
    for r in fail:
        if r["category"] == cat:
            q = r["question"][:70]
            print("      %-8s prec=%-6s mode=%-12s Q: %s" % (r["id"], r["compliance_precision"], r["generation_mode"], q))

print("\n  WARN (0.30 <= prec < 0.50): %d questions" % len(warn))
warn_cats = Counter(r["category"] for r in warn)
for cat, count in warn_cats.most_common():
    print("    %-20s : %d" % (cat, count))
    for r in warn:
        if r["category"] == cat:
            q = r["question"][:70]
            print("      %-8s prec=%-6s mode=%-12s Q: %s" % (r["id"], r["compliance_precision"], r["generation_mode"], q))

halluc = [r for r in rows if r["hallucinated_rules"]]
print("\n  HALLUCINATED RULES: %d answers" % len(halluc))
for r in halluc:
    q = r["question"][:65]
    print("    %-8s rules=%-12s Q: %s" % (r["id"], r["hallucinated_rules"], q))

rb = [r for r in rows if r["generation_mode"] == "rule_based"]
llm = [r for r in rows if r["generation_mode"] == "llm"]
err = [r for r in rows if r["generation_mode"] == "error"]
rb_prec = sum(float(r["compliance_precision"]) for r in rb) / max(1, len(rb))
llm_prec = sum(float(r["compliance_precision"]) for r in llm) / max(1, len(llm))

print("\n  GENERATION MODE SPLIT:")
print("    rule_based : %3d questions  avg_prec=%.2f%%" % (len(rb), rb_prec * 100))
print("    llm        : %3d questions  avg_prec=%.2f%%" % (len(llm), llm_prec * 100))
print("    error      : %3d questions" % len(err))

rb_fail = [r for r in rb if float(r["compliance_precision"]) < 0.5]
if rb_fail:
    print("\n  RULE_BASED FALSE NEGATIVES (%d cases - framework may be wrong):" % len(rb_fail))
    for r in rb_fail:
        q = r["question"][:70]
        print("    %-8s prec=%-6s Q: %s" % (r["id"], r["compliance_precision"], q))
