from __future__ import annotations

import re

from .common import contains, exact_answer, normalize
from .pipeline import FirstHop, MergedQuestion, Quality, SecondHop, SolvedAnswer


def quote_in_source(quote: str, source: str) -> bool:
    # Korp separates punctuation with spaces. Compare contiguous tokens, retaining
    # case, punctuation and order, while allowing normal typographic spacing.
    tokens = lambda text: " ".join(re.findall(r"\w+|[^\w\s]", text))
    return bool(quote.strip()) and f" {tokens(quote)} " in f" {tokens(source)} "


def generation_reasons(row: dict) -> list[str]:
    h1, h2, merged = (FirstHop.model_validate(row["hop1"]),
                      SecondHop.model_validate(row["hop2"]),
                      MergedQuestion.model_validate(row["merged"]))
    reasons = []
    if not (h1.possible and h2.possible and merged.possible):
        reasons.append("generation_abstained")
    if not all(x.strip() for x in (h1.question, h2.question, h2.answer, merged.question)):
        reasons.append("empty_field")
    if row["d1_document_id"] == row["d2_document_id"]:
        reasons.append("same_article")
    entity = row["bridge_entity"]
    if not all(contains(row[f"d{i}_text"], entity) for i in (1, 2)):
        reasons.append("bridge_missing_from_sources")
    if not quote_in_source(h1.evidence, row["d1_text"]) or not contains(h1.evidence, entity):
        reasons.append("invalid_d1_evidence")
    if not quote_in_source(h2.evidence, row["d2_text"]) or not contains(h2.evidence, h2.answer):
        reasons.append("invalid_d2_evidence")
    if exact_answer(entity, h2.answer):
        reasons.append("answer_is_bridge")
    if contains(h1.question, entity) or contains(merged.question, entity):
        reasons.append("bridge_leaked_in_question")
    if not contains(h2.question, entity):
        reasons.append("q2_not_about_bridge")
    if contains(merged.question, h2.answer):
        reasons.append("answer_leaked_in_question")
    return reasons


def flatten_generation(row: dict) -> dict:
    return {**row, "q1": row["hop1"]["question"], "q2": row["hop2"]["question"],
            "question": row["merged"]["question"], "answer": row["hop2"]["answer"],
            "evidence1": row["hop1"]["evidence"], "evidence2": row["hop2"]["evidence"]}


def curation_reasons(row: dict, attempts: int) -> list[str]:
    h1 = FirstHop.model_validate(row["imputed_hop1"])
    merged = MergedQuestion.model_validate(row["recomposed"])
    quality = Quality.model_validate(row["quality"])
    reasons = []
    if not h1.possible or not merged.possible or not h1.question.strip() or not merged.question.strip():
        reasons.append("imputation_abstained")
    if not quote_in_source(h1.evidence, row["d1_text"]) or not contains(h1.evidence, row["bridge_entity"]):
        reasons.append("invalid_imputed_evidence")
    if contains(h1.question, row["bridge_entity"]) or contains(merged.question, row["bridge_entity"]):
        reasons.append("bridge_leaked_in_question")
    if contains(merged.question, row["answer"]):
        reasons.append("answer_leaked_in_question")
    predictions = [SolvedAnswer.model_validate(row[f"solve_{i}"]) for i in range(1, attempts + 1)]
    if not any(p.answerable and exact_answer(p.answer, row["answer"]) for p in predictions):
        reasons.append("unanswerable_in_k_attempts")
    for key in ("grounded", "requires_both_documents", "subquestions_consistent", "swedish"):
        if not getattr(quality, key):
            reasons.append(f"judge_failed_{key}")
    return reasons


def training_record(row: dict, curated: bool = False) -> dict:
    question = row["recomposed"]["question"] if curated else row["question"]
    q1 = row["imputed_hop1"]["question"] if curated else row["q1"]
    evidence1 = row["imputed_hop1"]["evidence"] if curated else row["evidence1"]
    # Short, verifiable subanswers and supporting facts, not hidden model reasoning.
    response = (
        f"Delfråga 1: {q1}\nSvar 1: {row['bridge_entity']}\nBelägg i D1: {evidence1}\n"
        f"Delfråga 2: {row['q2']}\nSvar 2: {row['answer']}\nBelägg i D2: {row['evidence2']}\n"
        f"Svar: {row['answer']}"
    )
    return {"messages": [
        {"role": "system", "content": "Besvara frågan på svenska utifrån källorna. Ge korta delfrågor, källbelägg och ett slutligt svar."},
        {"role": "user", "content": f"D1 ({row['d1_title']}): {row['d1_text']}\n\nD2 ({row['d2_title']}): {row['d2_text']}\n\nFråga: {question}"},
        {"role": "assistant", "content": response},
    ]}


def deduplicate(rows: list[dict], field: str = "question") -> tuple[list[dict], list[dict]]:
    accepted, rejected, seen = [], [], set()
    for row in rows:
        text = row["recomposed"]["question"] if field == "recomposed" else row[field]
        key = normalize(text)
        if key in seen:
            rejected.append({**row, "rejection_reasons": ["duplicate_question"]})
        else:
            accepted.append(row)
            seen.add(key)
    return accepted, rejected
