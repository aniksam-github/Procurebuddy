import json
import random

amounts_formats = [
    (150000, "1,50,000", "Direct Purchase"),
    (199999, "1,99,999", "Direct Purchase"),
    (200000, "2 Lakhs", "Direct Purchase"),
    (200001, "2,00,001", "LPC"),
    (450000, "4.5 Lakhs", "LPC"),
    (499999, "4,99,999", "LPC"),
    (500000, "5 Lakhs", "LPC"),
    (500001, "5,00,001", "LTE"),
    (2500000, "25 Lakhs", "LTE"),
    (4999999, "49,99,999", "LTE"),
    (5000000, "50 Lakhs", "LTE"),
    (5000001, "50,00,001", "OTE"),
    (10000000, "1 Crore", "OTE"),
    (35000000, "3.5 Crores", "OTE"),
]

# Expand to 100
generated = []
prefixes = [
    "What is the procurement mode for goods worth Rs. {}",
    "We need to buy scientific equipment for {}",
    "Please advise the process for hiring services estimated at Rs. {}",
    "For a works contract of {}, which tender applies?",
    "If the item costs {}, what committee is needed?"
]

for i in range(100):
    val, text_val, mode = random.choice(amounts_formats)
    # add some randomness to values
    if random.random() > 0.5 and "Lakhs" not in text_val and "Crore" not in text_val:
        offset = random.randint(-100, 100)
        new_val = val + offset
        text_val = f"{new_val:,}"
        if new_val <= 200000: mode = "Direct Purchase"
        elif new_val <= 500000: mode = "LPC"
        elif new_val <= 5000000: mode = "LTE"
        else: mode = "OTE"

    prefix = random.choice(prefixes)
    question = prefix.format(text_val)

    # Output both formats required
    generated.append({
        "id": f"GEN-{i+1:03d}",
        "category": "THRESHOLD",
        "question": question,
        "query": question,
        "expected_mode": mode,
        "expected_keywords": [mode.lower()],
        "amount": val
    })

with open("eval/fresh_dataset_100.json", "w", encoding="utf-8") as f:
    json.dump(generated, f, indent=2)

print("Generated 100 questions.")
