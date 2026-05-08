"""Deep root cause analysis of eval results."""
import csv
from collections import Counter, defaultdict

rows = []
with open("eval/eval_report.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"TOTAL ROWS: {len(rows)}")

# Category breakdown
cat_prec = defaultdict(list)
cat_lat = defaultdict(list)
cat_modes = defaultdict(lambda: Counter())
for r in rows:
    cat_prec[r["category"]].append(float(r["compliance_precision"]))
    cat_lat[r["category"]].append(float(r["latency_seconds"]))
    cat_modes[r["category"]][r["generation_mode"]] += 1

print("\nCATEGORY BREAKDOWN:")
print("%-20s %6s %5s %4s %5s %5s %5s" % ("Cat", "prec", "lat", "n", "fail", "rule", "llm"))
print("-" * 65)
for cat, vals in sorted(cat_prec.items(), key=lambda x: sum(x[1])/len(x[1])):
    avg_p = sum(vals)/len(vals)
    avg_l = sum(cat_lat[cat])/len(cat_lat[cat])
    fail = sum(1 for v in vals if v < 0.5)
    rb = cat_modes[cat].get("rule_based", 0)
    llm = cat_modes[cat].get("llm", 0)
    print("  %-20s %5.0f%% %4.0fs %4d %5d %5d %5d" % (cat, avg_p*100, avg_l, len(vals), fail, rb, llm))

# Mode split
modes = Counter(r["generation_mode"] for r in rows)
print("\nMODE SPLIT:")
for m, c in modes.most_common():
    avg = sum(float(r["compliance_precision"]) for r in rows if r["generation_mode"] == m) / c
    avgL = sum(float(r["latency_seconds"]) for r in rows if r["generation_mode"] == m) / c
    print("  %-15s : %4d  prec=%.0f%%  lat=%.0fs" % (m, c, avg*100, avgL))

# SCENARIO deep dive
print("\n" + "=" * 70)
print("SCENARIO DEEP DIVE")
print("=" * 70)
sc = [r for r in rows if r["category"] == "SCENARIO"]
sc_rb = [r for r in sc if r["generation_mode"] == "rule_based"]
sc_llm = [r for r in sc if r["generation_mode"] == "llm"]
print("  rule_based: %d  llm: %d" % (len(sc_rb), len(sc_llm)))

if sc_rb:
    print("\nSample SCENARIO rule_based (precision, latency, question):")
    for r in sc_rb[:8]:
        print("  prec=%-6s lat=%5.1fs  Q: %s" % (r["compliance_precision"], float(r["latency_seconds"]), r["question"][:65]))

if sc_llm:
    print("\nSample SCENARIO llm (precision, latency, question):")
    for r in sc_llm[:8]:
        print("  prec=%-6s lat=%5.1fs  Q: %s" % (r["compliance_precision"], float(r["latency_seconds"]), r["question"][:65]))

# Latency distribution
latencies = sorted(float(r["latency_seconds"]) for r in rows)
n = len(latencies)
print("\nLATENCY DIST: min=%.0fs  p25=%.0fs  p50=%.0fs  p75=%.0fs  p90=%.0fs  max=%.0fs" % (
    latencies[0], latencies[n//4], latencies[n//2], latencies[3*n//4], latencies[int(n*0.9)], latencies[-1]))

# Check if answers are getting "Context not available" responses
no_context = [r for r in rows if "context" in r.get("error", "").lower() or float(r["compliance_precision"]) == 0.0]
print("\nZERO PRECISION (prec=0.0): %d questions" % len([r for r in rows if float(r["compliance_precision"]) == 0.0]))
