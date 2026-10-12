import json
import hashlib

INPUT_FILE = "fineweb2_swe_filtered.jsonl"
OUTPUT_FILE = "fineweb2_swe_final.jsonl"

seen = set()
kept = 0
removed = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        text = item["text"].strip()

        # Language confidence filter
        if item.get("language_score", 0) < 0.95:
            removed += 1
            continue

        # Deduplication safety check
        fingerprint = hashlib.md5(text.encode("utf-8")).hexdigest()

        if fingerprint in seen:
            removed += 1
            continue

        seen.add(fingerprint)

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1

print(f"Final kept: {kept}")
print(f"Removed: {removed}")
print("Saved as fineweb2_swe_final.jsonl")