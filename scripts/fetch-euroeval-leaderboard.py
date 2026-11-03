"""Fetch reference scores from the public EuroEval Swedish leaderboard.

Writes one JSON file per reference model to artifacts/experiments/references/,
for the experiment tracker page. By default it picks the official rows for the
fine-tuned base model and its instruction-tuned sibling, plus the best-ranked
open-weight model on the generative leaderboard.

    uv run python scripts/fetch-euroeval-leaderboard.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://euroeval.com"
PAGE = f"{SITE}/leaderboards/swedish"
DATASETS = {"swerec": "swerec", "scala_sv": "scala-sv"}
TYPES = {"🧠": "base", "📝": "instruction", "🤔": "reasoning"}


def get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "TDDE19-experiment-tracker"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def chunk_url(bundle: str, stem: str, ext: str) -> str:
    match = re.search(rf'"/src/frontend/csv/{stem}\.{ext}":\(\)=>\w+\(\(\)=>import\(`\./([^`]+)`', bundle)
    if not match:
        raise SystemExit(f"Could not find {stem}.{ext} in the EuroEval site bundle; the site layout changed.")
    return f"{SITE}/assets/{match.group(1)}"


def template_literal(module: str) -> str:
    body = module[module.index("`") + 1 : module.rindex("`")]
    return body.replace("\\`", "`").replace("\\$", "$").replace("\\\\", "\\")


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def score_pair(cell: str) -> dict | None:
    """Parse 'MCC ± ci / macro-F1 ± ci'."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cell)
    if len(numbers) < 4:
        return None
    mcc, mcc_ci, f1, f1_ci = map(float, numbers[:4])
    return {"mcc": mcc, "mcc_ci": mcc_ci, "macro_f1": f1, "macro_f1_ci": f1_ci}


def number(text: str) -> float | None:
    match = re.match(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def load_leaderboard(stem: str) -> tuple[list[dict], str | None]:
    html = get(PAGE)
    bundle_path = re.search(r'src="(/assets/index-[^"]+\.js)"', html)
    if not bundle_path:
        raise SystemExit("Could not find the EuroEval site bundle.")
    bundle = get(SITE + bundle_path.group(1))
    table = template_literal(get(chunk_url(bundle, stem, "csv")))
    rows = list(csv.reader(io.StringIO(table)))
    header = [strip_html(h) for h in rows[1]]
    records = [dict(zip(header, row)) for row in rows[2:]]
    updated = None
    try:
        meta = get(chunk_url(bundle, stem, "json"))
        found = re.search(r"Last updated: ([^`\"]+)", meta)
        updated = found.group(1).strip() if found else None
    except SystemExit:
        pass
    return records, updated


def reference(row: dict, role: str, label: str, stem: str, updated: str | None) -> dict:
    name = strip_html(row["Model"])
    model = re.sub(r"\s*\(.*\)$", "", name)
    link = re.search(r"href='([^']+)'", row["Model"])
    rank_score = score_pair(row["Rank score"] + " / 0 ± 0") or {}
    params = number(row.get("Parameters", ""))
    scores = {}
    for column, dataset in DATASETS.items():
        pair = score_pair(row.get(column, ""))
        if pair:
            pair["failures"] = int(number(row.get(f"{column}_failures", "")) or 0)
            pair["scored"] = int(number(row.get(f"{column}_scored", "")) or 0)
            pair["euroeval_version"] = row.get(f"{column}_version") or None
            scores[dataset] = pair
    return {
        "id": f"ref-{role}",
        "role": role,
        "label": label,
        "model": model,
        "variant": name[len(model) :].strip(" ()") or None,
        "url": link.group(1) if link else None,
        "type": TYPES.get(row.get("Type", "").strip(), row.get("Type")),
        "open": row.get("Open") == "✓",
        "params_b": round(params / 1e9, 1) if params else None,
        "release": row.get("Release Date") or None,
        "rank": number(row.get("Rank", "")),
        "rank_score": rank_score.get("mcc"),
        "rank_score_ci": rank_score.get("mcc_ci"),
        "scores": scores,
        "leaderboard": stem,
        "leaderboard_updated": updated,
        "source_url": PAGE,
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", default="google/gemma-4-12B")
    parser.add_argument("--leaderboard", default="swedish_generative")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experiments" / "references")
    args = parser.parse_args()

    records, updated = load_leaderboard(args.leaderboard)
    by_name = {strip_html(r["Model"]): r for r in records}
    ranked = [r for r in records if number(r["Rank score"]) is not None]
    ranked.sort(key=lambda r: number(r["Rank score"]))

    picks = []
    for suffix, role, label in (("", "official-base", "Leaderboard base"), ("-it", "official-it", "Leaderboard IT")):
        row = by_name.get(f"{args.base_model}{suffix} (val)") or by_name.get(f"{args.base_model}{suffix}")
        if row:
            picks.append(reference(row, role, label, args.leaderboard, updated))
        else:
            print(f"Not on the leaderboard: {args.base_model}{suffix}")
    best_open = next((r for r in ranked if r.get("Open") == "✓"), None)
    if best_open:
        picks.append(reference(best_open, "best-open", "Best open model", args.leaderboard, updated))

    args.output.mkdir(parents=True, exist_ok=True)
    for pick in picks:
        (args.output / f"{pick['id']}.json").write_text(json.dumps(pick, ensure_ascii=False, indent=2), encoding="utf-8")
        swerec, scala = pick["scores"].get("swerec", {}), pick["scores"].get("scala-sv", {})
        print(f"{pick['role']:14} {pick['model']:32} swerec MCC {swerec.get('mcc')}  scala-sv MCC {scala.get('mcc')}")
    print(f"Leaderboard last updated: {updated}")


if __name__ == "__main__":
    main()
