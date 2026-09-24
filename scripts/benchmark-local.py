"""Small repeatable inference benchmark; run with `uv run python scripts/benchmark-local.py`."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import time

import requests


PROMPTS = [
    'Text: Astrid Lindgren föddes 1907 i Vimmerby och skrev böcker om Pippi Långstrump. '
    'Kopiera texten exakt till både text och evidence och lista alla namngivna entiteter. '
    'Svara med JSON: {"text": "ordagrant citat", "evidence": "samma ordagranna citat", '
    '"entities": [{"text": "...", "type": "per, org, loc eller misc"}]}.',
    'Text: Göta kanal invigdes 1832. Kanalen förbinder Sjötorp vid Vänern med Mem vid Östersjön. '
    'Skapa en fråga som besvaras med ett kort exakt utdrag. Svara med JSON: '
    '{"question": "...", "answer": "...", "evidence": "ordagrant citat"}.',
    'Text: Uppsala universitet grundades 1477. Det är ett universitet i Uppsala. '
    'Skriv en kort anonym uppskattande reaktion grundad i dessa fakta. Svara med JSON: '
    '{"text": "...", "label": "positive", "evidence": "ordagrant citat"}.',
    'Premiss: Biblioteket har hundra böcker. Hypotes: Biblioteket har fler än femtio böcker. '
    'Bedöm entailment, contradiction eller neutral. Svara med JSON: '
    '{"answer": "etikett", "reason": "kort motivering"}.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.getenv("S2S_ENDPOINT", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.getenv("S2S_MODEL", "google/gemma-4-12B-it"))
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.parallel < 1 or args.repeat < 1:
        parser.error("parallel and repeat must be positive")

    def complete(prompt):
        start = time.perf_counter()
        response = requests.post(args.endpoint.rstrip("/") + "/chat/completions", json={
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 42, "max_tokens": 256, "cache_prompt": False,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=180)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        try:
            json.loads(choice["message"]["content"])
            valid = True
        except (ValueError, TypeError):
            valid = False
        return {"seconds": time.perf_counter() - start,
                "completion_tokens": body["usage"]["completion_tokens"],
                "finish_reason": choice["finish_reason"], "valid_json": valid}

    complete(PROMPTS[0])  # Warm up before timing; cache reuse is disabled above.
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        rows = list(executor.map(complete, PROMPTS * args.repeat))
    elapsed = time.perf_counter() - start
    report = {"model": args.model, "endpoint": args.endpoint, "parallel": args.parallel,
              "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest(),
              "requests": len(rows), "seconds": elapsed,
              "output_tokens_per_second": sum(row["completion_tokens"] for row in rows) / elapsed,
              "results": rows,
              "note": "Small fixed Swedish prompt sample; throughput includes prompt processing and HTTP."
              " Record server speculation/slot settings alongside this report."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
