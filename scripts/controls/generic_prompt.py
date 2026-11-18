"""Response-only control: the same training rows, with every backtranslation instruction replaced by
one fixed generic prompt. The answers (the only supervised tokens) are untouched, so the control
tests whether the generated instruction adds anything over training on the native passages alone.

    python3 scripts/controls/generic_prompt.py --subset artifacts/subsets/X/X-125k.jsonl \
        --accepted artifacts/source2synth/runs/X/accepted.jsonl --output artifacts/subsets/X-control/X-control-125k.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

GENERIC = "Skriv en text på svenska."

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--subset", type=Path, required=True, help="training rows (messages JSONL)")
parser.add_argument("--accepted", type=Path, required=True, help="the build's accepted.jsonl, for each row's task")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

task_of = {}
for line in args.accepted.open(encoding="utf-8"):
    item = json.loads(line)
    task_of[(item["input"], item["output"])] = item["task"]

rows, tasks = [], Counter()
for line in args.subset.open(encoding="utf-8"):
    row = json.loads(line)
    user = next(m for m in row["messages"] if m["role"] == "user")
    answer = next(m for m in row["messages"] if m["role"] == "assistant")
    task = task_of[(user["content"], answer["content"])]  # KeyError = subset not from this build
    if task == "backtranslation":
        user["content"] = GENERIC
    tasks[task] += 1
    rows.append(row)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"{len(rows)} rows, {tasks['backtranslation']} backtranslation instructions replaced; {dict(tasks)} -> {args.output}")
