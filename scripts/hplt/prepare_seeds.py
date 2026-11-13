"""Split an HPLT seed file into one local register source per register for ./pipeline.

    python3 scripts/hplt/prepare_seeds.py hplt-seeds-50k-test.jsonl data/hplt

Writes <output>/<register>.jsonl (news, forum, web, ...) with the fields the register sources read:
id, text, title (the domain), url, date (crawl date) and group_id (the HPLT document id, so the
paragraphs of one page share a split). The HPLT scores are kept for later filtering. Rows are
shuffled with a fixed seed, because the register sources read only the first max_documents rows.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

source, output = Path(sys.argv[1]), Path(sys.argv[2])
by_register = defaultdict(list)
for line in source.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    by_register[row["register"]].append({
        "id": row["seed_id"], "text": row["text"], "title": row["domain"], "url": row["url"],
        "date": row["crawl_date"], "group_id": row["document_id"],
        "hplt": {k: row[k] for k in ("mt_score", "doc_score", "register_scores", "paragraph")},
    })
output.mkdir(parents=True, exist_ok=True)
for register, rows in sorted(by_register.items()):
    random.Random(0).shuffle(rows)
    path = output / f"{register}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{register}: {len(rows):,} paragraphs from {len({r['group_id'] for r in rows}):,} documents -> {path}")
