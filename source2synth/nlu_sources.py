"""Broader Korp sampling and deterministic, article-disjoint NLU seed schedules."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

from .common import fingerprint, read_jsonl, write_json, write_jsonl
from .korp import KorpClient, entity_cqp, passage_from_hit
from .nlu_tasks import source_excerpt

TASKS = (
    "classification", "sentiment-classification", "named-entity-recognition",
    "linguistic-acceptability", "reading-comprehension", "natural-language-inference",
    "grammatical-error-detection",
)
LABELS = {
    "sentiment-classification": ["positive", "neutral", "negative"],
    "linguistic-acceptability": ["correct", "incorrect"],
    "natural-language-inference": ["entailment", "neutral", "contradiction"],
    "grammatical-error-detection": ["correct", "incorrect"],
    "named-entity-recognition": ["per", "org", "loc", "misc", "mixed"],
    "reading-comprehension": ["person", "place", "time", "quantity", "object", "cause"],
}
NER_CLUES = {
    "per": re.compile(r"\b(?:född|föddes|författare|politiker|skådespelare|forskare)\b", re.I),
    "org": re.compile(r"\b(?:universitet|förening|myndighet|företag|bolag|riksdag|klubb|organisation|institut)\b|\b(?:AB|IF|FF|FC)\b", re.I),
    "loc": re.compile(r"\b(?:sjö|stad|ort|kommun|landskap|län|flod|berg|region)\b", re.I),
    "misc": re.compile(r"\b(?:roman|film|festival|bok|olympiska|mästerskap|album)\b", re.I),
}
READING_CLUES = {
    "time": re.compile(r"\b\d{4}\b|\b(?:januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december)\b", re.I),
    "quantity": re.compile(r"\b\d+\s+(?:miljoner|miljarder|invånare|meter|kilometer|procent|personer|år)\b", re.I),
    "cause": re.compile(r"\b(?:eftersom|därför|orsak|på grund av|ledde till|för att)\b", re.I),
}


def prepare_nlu(config: dict, data_dir: Path, offline: bool = False, cache_dir: Path | None = None) -> dict:
    cfg, nlu = config["corpus"], config["nlu"]
    client = KorpClient(cfg["api"], cache_dir or data_dir / "raw", offline)
    info = client.get("corpus_info", {"corpus": cfg["id"]})
    passages = {}
    for topic, queries in nlu["topics"].items():
        for query in queries:
            envelope = client.get("query", {
                "corpus": cfg["id"], "cqp": entity_cqp(query), "start": 0,
                "end": nlu["hits_per_query"] - 1, "default_context": "1 paragraph",
                "default_within": "sentence", "show": "word",
                "show_struct": "text_title,text_url,text_permalink,text__id",
                "sort": "random", "random_seed": config["random_seed"],
            })
            for hit in envelope["response"].get("kwic", []):
                row = passage_from_hit(hit, query, cfg, envelope, config["random_seed"])
                if row:
                    # A search term is a sampling stratum, not a ground-truth topic label.
                    passages.setdefault(row["id"], {**row, "source_topic": topic, "query": query})
            print(f"NLU source {topic}/{query}: {len(passages)} distinct passages", flush=True)
    groups = defaultdict(list)
    for row in passages.values():
        groups[(row["source_topic"], row["split"])].append(row)
    selected = []
    for rows in groups.values():
        rows.sort(key=lambda r: fingerprint([config["random_seed"], r["id"]]))
        # One passage per article per topic limits repeated source material.
        unique = {r["document_id"]: r for r in rows}
        selected.extend(list(unique.values())[:nlu["max_passages_per_topic_per_split"]])
    assert_source_disjoint(selected)
    out = data_dir / "nlu"
    write_jsonl(out / "passages.jsonl", selected)
    manifest = {
        "corpus": cfg, "sampling_config": nlu, "random_seed": config["random_seed"],
        "corpus_info": info["response"], "retrieved_at": info["retrieved_at"],
        "passages": len(selected), "sha256": fingerprint(selected),
        "by_topic_and_split": dict(Counter(f"{r['source_topic']}/{r['split']}" for r in selected)),
        "splitting": "Same 45/45/10 article hash as the original QA pipeline",
        "sampling": "Topic-keyword-conditioned Wikipedia; styles and opinions are synthetic transformations",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def assert_source_disjoint(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        doc = row["document_id"]
        if doc in seen and seen[doc] != row["split"]:
            raise ValueError(f"Source document crosses splits: {doc}")
        seen[doc] = row["split"]


def schedule(config: dict, data_dir: Path, split: str, tasks: list[str], count: int, offset: int) -> list[dict]:
    if count < 1 or offset < 0:
        raise ValueError("Candidate count must be positive and offset nonnegative")
    rows = read_jsonl(data_dir / "nlu/passages.jsonl")
    assert_source_disjoint(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[row["source_topic"]].append(row)
    nlu = config["nlu"]
    topics = list(nlu["topics"])
    for topic in topics:
        groups[topic].sort(key=lambda r: fingerprint([config["random_seed"], r["id"]]))
        if not groups[topic]:
            raise ValueError(f"No {split} passages for {topic}; increase nlu.hits_per_query and prepare-nlu")
    all_sources = sorted((row for group in groups.values() for row in group),
                         key=lambda r: fingerprint([config["random_seed"], r["id"]]))
    result = []
    for task in tasks:
        task_index = TASKS.index(task)
        labels = topics if task == "classification" else LABELS[task]
        # NER always uses a fixed excerpt, so revisiting a source creates the
        # same training input. Replay its allocation to keep chunked schedules
        # equivalent while using unseen eligible excerpts before repeating one.
        used_sources = set()
        indices = range(offset + count) if task == "named-entity-recognition" else range(offset, offset + count)
        for index in indices:
            target = labels[index % len(labels)]
            # Label varies faster than topic: every topic sees every label over a full block.
            topic = target if task == "classification" else topics[(index // len(labels) + task_index) % len(topics)]
            pool = groups[topic]
            if task == "named-entity-recognition" and target in NER_CLUES:
                clue = NER_CLUES[target]
                pool = ([row for row in pool if clue.search(source_excerpt(row["text"]))]
                        or [row for row in all_sources if clue.search(source_excerpt(row["text"]))]
                        or pool)
            elif task == "reading-comprehension" and target in READING_CLUES:
                clue = READING_CLUES[target]
                pool = ([row for row in pool if clue.search(row["text"])]
                        or [row for row in all_sources if clue.search(row["text"])] or pool)
            cycle = index // (len(topics) if task == "classification" else len(labels) * len(topics))
            if task == "named-entity-recognition":
                pool = [row for row in pool if row["id"] not in used_sources] or pool
            source = pool[(cycle * len(labels) + index % len(labels) + task_index) % len(pool)]
            used_sources.add(source["id"])
            if index < offset:
                continue
            style = nlu["styles"][(index // len(labels) + index % len(labels)) % len(nlu["styles"])]
            difficulty = nlu["difficulties"][(index // len(labels) + index % len(labels)) % len(nlu["difficulties"])]
            result.append({"seed_id": fingerprint([source["id"], task, index, style, difficulty])[:24],
                           "task": task, "target": target, "style": style, "difficulty": difficulty,
                           "split": split, "source": source, "sample_index": index})
    return result
