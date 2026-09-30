from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INSTRUCTION_COLUMNS = ("instruction", "prompt", "question", "query", "input", "user")
RESPONSE_COLUMNS = ("output", "response", "answer", "completion", "assistant", "chosen")
CONVERSATION_COLUMNS = ("messages", "conversations", "conversation")
ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def column_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--instruction", help="Instruction column (auto-detected if omitted)")
    parser.add_argument("--response", help="Response column (auto-detected if omitted)")
    return parser.parse_args()


def _ask(kind: str, columns: list[str]) -> str:
    if not sys.stdin.isatty():
        sys.exit(f"Could not detect the {kind} column among {columns}; pass --{kind} <column>.")
    answer = input(f"Which column holds the {kind}? {columns}: ").strip()
    if answer not in columns:
        sys.exit(f"Unknown column: {answer!r}")
    return answer


def detect_columns(columns: list[str], instruction: str | None, response: str | None) -> tuple[str, str]:
    """Pick instruction/response columns: explicit flags first, then known names, then ask."""
    for kind, given in (("instruction", instruction), ("response", response)):
        if given and given not in columns:
            sys.exit(f"--{kind} {given!r} is not a column; available: {columns}")
    instruction = instruction or next((c for c in INSTRUCTION_COLUMNS if c in columns), None)
    response = response or next((c for c in RESPONSE_COLUMNS if c in columns), None)
    instruction = instruction or _ask("instruction", columns)
    response = response or _ask("response", columns)
    print(f"Using instruction column {instruction!r}, response column {response!r}")
    return instruction, response


def conversation_to_messages(turns) -> list[dict]:
    messages = []
    for turn in turns:
        role = ROLES.get(turn.get("from") or turn.get("role"))
        content = turn.get("value") if "value" in turn else turn.get("content")
        if role is None or not isinstance(content, str) or not content.strip():
            return []
        messages.append({"role": role, "content": content.strip()})
    return messages


def to_chat(df: pd.DataFrame, source: str, instruction: str | None = None, response: str | None = None) -> list[dict]:
    """Convert a dataframe to rows with a chat-style `messages` field, dropping empty examples."""
    columns = list(df.columns)
    conversation = next((c for c in CONVERSATION_COLUMNS if c in columns), None)
    if conversation and not (instruction or response):
        print(f"Using conversation column {conversation!r}")
        pairs = (conversation_to_messages(turns) for turns in df[conversation])
    else:
        if not (instruction and response):
            instruction, response = detect_columns(columns, instruction, response)
        # Alpaca-style data keeps optional context in a separate `input` column.
        context = "input" if instruction == "instruction" and "input" in columns else None

        def pair(row) -> list[dict]:
            prompt, answer = row[instruction], row[response]
            if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip():
                return []
            if context and isinstance(row[context], str) and row[context].strip():
                prompt = f"{prompt.strip()}\n\n{row[context].strip()}"
            return [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}]

        pairs = (pair(row) for row in df.to_dict("records"))

    rows = [{"id": f"{source}-{i}", "source": source, "messages": m} for i, m in enumerate(pairs) if m]
    print(f"Kept {len(rows):,} of {len(df):,} rows ({len(df) - len(rows):,} empty or malformed dropped)")
    return rows
