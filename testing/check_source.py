"""Check what the source-missing answers actually look like."""
import json

with open(r"D:\projects\bot\testing\runs\20260429_142648_sample50_llm_restored\eval_failed_cases.json", encoding="utf-8") as f:
    fails = json.load(f)

count = 0
for item in fails:
    if item["question_type"] == "scenario" and "missing source" in item["error_reason"] and item["source_score"] == 0.0:
        print(f"=== {item['case_id']} (score={item['score']}) ===")
        print(f"Q: {item['question'][:90]}")
        print(f"A: {item['answer'][:350]}")
        print(f"Err: {item['error_reason']}")
        print()
        count += 1
        if count >= 5:
            break
