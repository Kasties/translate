import json
import hashlib

INPUT_FILE = "fineweb2_swe_10k.jsonl"
OUTPUT_FILE = "fineweb2_swe_filtered.jsonl"

seen = set()
kept = 0
removed = 0

def is_good(text):
    # Too short
    if len(text) < 300:
        return False

    # Too long
    if len(text) > 8000:
        return False

    # HTML leftovers
    if "<html" in text.lower():
        return False

    # Table-heavy content
    if text.count("|") > 15:
        return False

    # Too many digits
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False

    return True

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)
        text = item["text"].strip()

        fingerprint = hashlib.md5(text.encode("utf-8")).hexdigest()

        if fingerprint in seen:
            removed += 1
            continue

        seen.add(fingerprint)

        if is_good(text):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
        else:
            removed += 1

print(f"Kept: {kept}")
print(f"Removed: {removed}")
print("Saved as fineweb2_swe_filtered.jsonl")