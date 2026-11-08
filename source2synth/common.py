from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def read_config(path: Path | str = ROOT / "configs/source2synth.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(text: str) -> str:
    # Keep accents and digits; punctuation, case and whitespace are insignificant.
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def contains(text: str, part: str) -> bool:
    text, part = normalize(text), normalize(part)
    return bool(part) and f" {part} " in f" {text} "


def exact_answer(prediction: str, expected: str) -> bool:
    def canonical(text: str) -> str:
        text = normalize(text)
        # Swedish dates conventionally allow an optional introductory "den".
        months = "januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december"
        return re.sub(rf"^den (?=\d{{1,2}} (?:{months})(?: |$))", "", text)
    return bool(canonical(expected)) and canonical(prediction) == canonical(expected)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
