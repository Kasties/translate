"""NLU source passages from a Hugging Face dataset or local files, Korp-compatible.

`prepare-nlu` samples Swedish Wikipedia through Korp. These loaders produce the
same `data_dir/nlu/passages.jsonl` schema from other text, so generation, curation
and provenance work unchanged:

* Text is re-tokenized into Korp's token-spaced form ("ord , ord .") because
  evidence quotes, NER excerpts and answer spans are checked against it.
* Each document is split into paragraphs of `corpus.min_words..max_words` tokens.
* A paragraph gets the topic whose `nlu.topics` keyword it contains as a whole
  word (the Korp query rule), or the value of `topic_field` when the data has
  labels. Unmatched paragraphs are dropped, never given an invented topic.
* The same article-hash 45/45/10 split and per-topic caps as `prepare-nlu` apply.

A local JSONL/Parquet file whose rows already have the passage schema (for
example another experiment's passages.jsonl) is reused as-is.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, fingerprint, read_jsonl, write_json, write_jsonl
from .korp import document_split
from .nlu_sources import assert_source_disjoint

PASSAGE_KEYS = {"id", "document_id", "split", "source_topic", "text"}
TEXT_SUFFIXES = {".txt", ".md"}
RECORD_SUFFIXES = {".jsonl", ".json", ".parquet", ".csv", ".tsv"}
# Numbers with decimal/thousand separators, words with inner hyphen/apostrophe, then punctuation.
# As in Korp: abbreviations and initials ("bl.a.", "V."), colon genitives ("SVT:s") and
# compound ellipsis ("musik- och") stay one token, so "." + space still ends a sentence.
ABBREVIATIONS = ("bl.a", "t.ex", "s.k", "m.fl", "f.d", "f.Kr", "e.Kr", "t.o.m", "fr.o.m", "d.v.s",
                 "o.s.v", "resp", "etc", "osv", "dvs", "ca", "nr", "jfr", "kl", "St", "Dr")
TOKEN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True))
    + r")\.?(?!\w)|(?<!\w)[A-ZÅÄÖ]\.(?=\s)|\.{2,}"
    r"|\d+(?:[.,:]\d+)+|\w+(?:[-'’:&×]\w+)*(?:-(?=\s))?|[^\w\s]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def korp_tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


def paragraphs(text: str, min_words: int, max_words: int) -> Iterator[str]:
    """Yield token-spaced paragraphs; split long ones on sentence boundaries."""
    for block in re.split(r"\n\s*\n|\n", text):
        block = block.strip()
        if not block or block.lstrip().startswith(("#", "|", "*", "•", "-")):
            continue
        words = korp_tokens(block)
        if len(words) <= max_words:
            if len(words) >= min_words:
                yield " ".join(words)
            continue
        window: list[str] = []
        for sentence in SENTENCE_END.split(block):
            tokens = korp_tokens(sentence)
            if window and len(window) + len(tokens) > max_words:
                if len(window) >= min_words:
                    yield " ".join(window)
                window = []
            window.extend(tokens[:max_words])
        if len(window) >= min_words:
            yield " ".join(window)


def topic_matchers(topics: dict[str, list[str]]) -> list[tuple[str, str, re.Pattern]]:
    return [(topic, keyword, re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE))
            for topic, keywords in topics.items() for keyword in keywords]


def iter_local_records(path_spec: str) -> Iterator[tuple[dict, str, str]]:
    """Yield (record, origin, file sha256) from a file, directory or glob."""
    path = Path(path_spec)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        files = sorted(Path(p) for p in glob.glob(str(path), recursive=True))
    files = [p for p in files if p.suffix.lower() in TEXT_SUFFIXES | RECORD_SUFFIXES
             or p.name.endswith(".jsonl.gz")]
    if not files:
        raise ValueError(f"No .txt/.md/.jsonl/.json/.parquet/.csv files at {path_spec}")
    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        suffix = file.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            yield {"text": file.read_text(encoding="utf-8"), "title": file.stem,
                   "id": str(file.relative_to(ROOT) if file.is_relative_to(ROOT) else file)}, str(file), digest
        elif file.name.endswith(".jsonl.gz"):
            with gzip.open(file, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line), str(file), digest
        elif suffix == ".jsonl":
            for row in read_jsonl(file):
                yield row, str(file), digest
        elif suffix == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
            for row in data if isinstance(data, list) else [data]:
                yield row, str(file), digest
        else:
            import pandas as pd
            frame = (pd.read_parquet(file) if suffix == ".parquet"
                     else pd.read_csv(file, sep="\t" if suffix == ".tsv" else ","))
            for row in json.loads(frame.to_json(orient="records", force_ascii=False)):
                yield row, str(file), digest


def iter_hf_records(dataset: str, subset: str | None, split: str,
                    revision: str | None) -> Iterator[tuple[dict, str, str]]:
    """Stream parquet/jsonl shards of a Hub dataset without the `datasets` package."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(dataset, revision=revision)
    files = [f.rfilename for f in info.siblings or []]

    def shards(names: list[str]) -> list[str]:
        data = [n for n in names if n.endswith((".parquet", ".jsonl", ".jsonl.gz"))]
        if subset:
            data = [n for n in data if subset in n.split("/")[:-1]]
        else:
            configs = sorted({n.split("/")[0] for n in data if "/" in n} - {"data", "default"})
            if len(configs) > 1:
                raise ValueError(f"{dataset} has several subsets; set one, e.g. {configs[:8]}")
        by_split = [n for n in data if re.search(rf"(?:^|[/_-]){re.escape(split)}(?:[/_.-]|$)", n)]
        return sorted(by_split or (data if split == "train" else []))

    selected, commit = shards(files), info.sha
    if not selected:
        # The Hub publishes an auto-converted parquet copy of most datasets.
        converted = "refs/convert/parquet"
        selected = shards(api.list_repo_files(dataset, repo_type="dataset", revision=converted))
        commit = converted
    if not selected:
        raise ValueError(f"No parquet/jsonl shards for {dataset} subset={subset} split={split}")
    for name in selected:
        local = hf_hub_download(dataset, name, repo_type="dataset", revision=commit)
        origin = f"https://huggingface.co/datasets/{dataset}/blob/{info.sha}/{name}"
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(local).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    yield row, origin, info.sha
        else:
            opener = gzip.open if name.endswith(".gz") else open
            with opener(local, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line), origin, info.sha


def prepare_sources(config: dict, data_dir: Path, *, source: str, path: str | None = None,
                    dataset: str | None = None, subset: str | None = None, split: str = "train",
                    revision: str | None = None, text_field: str = "text",
                    title_field: str | None = "title", id_field: str | None = "id",
                    url_field: str | None = "url", topic_field: str | None = None,
                    max_documents: int | None = None, license_as_listed: str | None = None) -> dict:
    if source == "hf":
        if not dataset:
            raise ValueError("--dataset is required for --source hf")
        records = iter_hf_records(dataset, subset, split, revision)
        corpus = dataset if not subset else f"{dataset}/{subset}"
    elif source == "local":
        if not path:
            raise ValueError("--path is required for --source local")
        records = iter_local_records(path)
        corpus = "local"
    else:
        raise ValueError(f"Unknown source {source!r}; Korp uses prepare-nlu")
    cfg, nlu, seed = config["corpus"], config["nlu"], config["random_seed"]
    matchers = topic_matchers(nlu["topics"])
    retrieved_at = datetime.now(timezone.utc).isoformat()
    candidates, reused, documents, skipped = [], [], 0, Counter()
    for row, origin, digest in records:
        if max_documents and documents >= max_documents:
            break
        documents += 1
        if PASSAGE_KEYS <= row.keys():
            reused.append(row)
            continue
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            skipped["missing_text"] += 1
            continue
        raw_id = str(row.get(id_field) if id_field and row.get(id_field) is not None else "") or \
            fingerprint([origin, text[:2000]])[:24]
        title = str(row.get(title_field) or "") if title_field else ""
        url = str(row.get(url_field) or "") if url_field else ""
        document_id = fingerprint([corpus, raw_id])[:24]
        label = row.get(topic_field) if topic_field else None
        if topic_field and label not in nlu["topics"]:
            skipped["topic_not_in_nlu_topics"] += 1
            continue
        for text_block in paragraphs(text, cfg["min_words"], cfg["max_words"]):
            for topic, keyword, pattern in matchers:
                if label is not None and topic != label:
                    continue
                if label is None and not pattern.search(text_block):
                    continue
                candidates.append({
                    "id": fingerprint([document_id, text_block])[:24], "document_id": document_id,
                    "title": title or raw_id, "url": url or origin, "permalink": "",
                    "corpus_text_id": raw_id, "text": text_block, "corpus_position": 0,
                    "context_start": 0, "corpus": corpus, "entity": keyword,
                    "split": document_split(document_id, seed), "retrieved_at": retrieved_at,
                    "query_url": origin, "response_sha256": digest,
                    "license_as_listed": license_as_listed or "unknown",
                    "resource": origin if source == "local" else f"https://huggingface.co/datasets/{dataset}",
                    "doi": "", "source_topic": topic, "query": keyword,
                })
                if label is not None:
                    break
    if reused and candidates:
        raise ValueError("Mixing prepared passages with raw documents is not supported")
    if reused:
        selected = reused
    else:
        groups = defaultdict(list)
        for row in candidates:
            groups[(row["source_topic"], row["split"])].append(row)
        selected = []
        for rows in groups.values():
            rows.sort(key=lambda r: fingerprint([seed, r["id"]]))
            unique = {}
            for row in rows:
                unique.setdefault(row["document_id"], row)  # One passage per article and topic.
            selected.extend(list(unique.values())[:nlu["max_passages_per_topic_per_split"]])
    assert_source_disjoint(selected)
    counts = Counter(f"{r['source_topic']}/{r['split']}" for r in selected)
    missing = [f"{topic}/{split}" for topic in nlu["topics"] for split in ("slice0", "slice1", "eval")
               if not counts[f"{topic}/{split}"]]
    out = data_dir / "nlu"
    write_jsonl(out / "passages.jsonl", selected)
    manifest = {
        "source": source, "corpus": corpus, "path": path, "dataset": dataset, "subset": subset,
        "split": split, "revision": revision, "fields": {"text": text_field, "title": title_field,
        "id": id_field, "url": url_field, "topic": topic_field},
        "documents_read": documents, "skipped": dict(skipped), "reused_prepared_passages": bool(reused),
        "sampling_config": {"topics": nlu["topics"], "min_words": cfg["min_words"],
                            "max_words": cfg["max_words"],
                            "max_passages_per_topic_per_split": nlu["max_passages_per_topic_per_split"]},
        "random_seed": seed, "retrieved_at": retrieved_at, "passages": len(selected),
        "sha256": fingerprint(selected), "by_topic_and_split": dict(sorted(counts.items())),
        "empty_topic_splits": missing,
        "splitting": "Same 45/45/10 article hash as prepare-nlu",
        "sampling": "Whole-word nlu.topics keyword match per paragraph" if not topic_field
                    else f"Topic label from field {topic_field!r}",
    }
    write_json(out / "manifest.json", manifest)
    if missing:
        print(f"Warning: no passages for {', '.join(missing)}. Generation for these needs more "
              "documents (max_documents), other nlu.topics keywords, or fewer tasks.",
              file=sys.stderr, flush=True)
    return manifest
