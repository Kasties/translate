"""Build DPO preference pairs from an instruction build: native Swedish preferred over a stiff version.

    .venv/bin/python scripts/dpo/build_pairs.py --run artifacts/source2synth/runs/<build> \
        --config artifacts/pipeline/<build>/source2synth.yaml --output artifacts/dpo/<name>

Each pair is one source passage. The prompt is its accepted backtranslation instruction, the chosen
answer is the human passage, and the rejected answer is a flat, literal version of the same passage.
Passages whose rewrite was already accepted reuse that literal version. For the rest, the generator
writes one with the instruction builder's own rewrite prompt, code checks and reviewer, so new
literal versions pass the same bar. Every call is journaled in <output>/calls, so a rerun resumes.

The rewrite prompt is not the DPO prompt: there the literal text is also the input, so preferring
the native answer would mostly teach "do not copy the input". Job ads that start or end mid-text
(dangling headings, URLs) are left out. Needs the generator server from the build's config running.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source2synth.common import write_jsonl  # noqa: E402
from source2synth.instruction_data import (  # noqa: E402
    CompletionClient, Review, atomic_json, draft_request, make_example, review_request,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", type=Path, required=True, help="instruction build with accepted.jsonl")
parser.add_argument("--config", type=Path, required=True, help="the build's resolved source2synth.yaml")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--eval-fraction", type=float, default=0.08)
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
cfg = config["instruction_data"]
accepted = [json.loads(line) for line in (args.run / "accepted.jsonl").open(encoding="utf-8")]
instructions = {a["source"]["id"]: a for a in accepted if a["task"] == "backtranslation"}
rewrites = {a["source"]["id"]: a for a in accepted if a["task"] == "rewrite"}


def clean(text: str) -> bool:
    """Starts like a sentence and ends with sentence punctuation (drops truncated job ads)."""
    text = text.strip()
    return bool(re.match(r"[A-ZÅÄÖ0-9\"”»(]", text)) and bool(re.search(r"[.!?»”\")]$", text))


skipped = Counter()
pairs, missing = [], []
for source_id, row in sorted(instructions.items()):
    if not clean(row["output"]):
        skipped["not_clean_" + row["source"]["register"]] += 1
    elif source_id in rewrites:
        pairs.append((row, rewrites[source_id]["draft"]["literal"], "accepted-rewrite"))
    else:
        missing.append(row)

client = CompletionClient(config, args.output / "calls")
REVIEW_KEYS = ("swedish", "self_contained", "source_faithful", "task_fulfilled")


def literal_for(row: dict):
    """Draft and review a literal version of one passage; returns (literal, None) or (None, reason)."""
    source, key = row["source"], row["source"]["id"]
    try:
        schema, prompt = draft_request("rewrite", source, 0)
        draft = client.call(f"{key}-draft", prompt, schema, cfg["generation_max_tokens"])
        item = make_example("rewrite", source, draft, 0, [])
        review = client.call(f"{key}-review", review_request(item), Review, cfg["review_max_tokens"], 0.1)
        if not all(review[k] is True for k in REVIEW_KEYS):
            return None, "semantic_review_failed"
        return draft["literal"].strip(), None
    except (ValueError, KeyError, TypeError) as error:
        return None, str(error)[:60]


workers = int(config["model"].get("max_parallel_requests") or 1)
with ThreadPoolExecutor(max_workers=workers) as pool:
    for row, (literal, reason) in zip(missing, pool.map(literal_for, missing)):
        if literal:
            pairs.append((row, literal, "new-rewrite"))
        else:
            skipped[reason] += 1

records = []
for row, literal, origin in pairs:
    source = row["source"]
    # A stable hash of the passage decides the split, so reruns keep the same held-out pairs.
    bucket = int(hashlib.sha256(source["id"].encode()).hexdigest(), 16) % 10000 / 10000
    records.append({"id": source["id"], "split": "eval" if bucket < args.eval_fraction else "train",
                    "register": source["register"], "origin": origin, "prompt": row["input"].strip(),
                    "chosen": row["output"], "rejected": literal, "url": source.get("url"),
                    "license": source.get("license")})
records.sort(key=lambda r: r["id"])
write_jsonl(args.output / "pairs.jsonl", records)
report = {"pairs": len(records), "by_split": dict(Counter(r["split"] for r in records)),
          "by_register": dict(Counter(r["register"] for r in records)),
          "by_origin": dict(Counter(r["origin"] for r in records)), "skipped": dict(skipped),
          "source_run": str(args.run), "generator": config["model"]["id"]}
atomic_json(args.output / "report.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
